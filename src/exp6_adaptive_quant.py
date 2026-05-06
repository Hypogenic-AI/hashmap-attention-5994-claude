"""Experiment 6: Adaptive KV-cache quantization driven by HMA hit frequency.

Hypothesis (H6): Allocate per-key bit-width based on observed attention
hit frequency in the HMA lookup table. Frequently-attended positions
get FP16; rarely-attended positions get INT4 / INT2. Compare against
uniform low-bit quantization at *matched memory budget*.

Setup (still on GPT-2-small, WikiText-2):
- Forward pass with full attention to capture per-(layer, head, position)
  hit count = number of queries that attended to it with weight >= tau.
- Per-(layer, head, position): assign bit-width based on hit count rank.
  - Top q% by hit count → FP16 (16 bits)
  - Next r% → INT4 (4 bits)
  - Rest → INT2 (2 bits)
- Uniform baseline: same total memory, but assigned *uniformly per token*
  (every position gets same bit-width that achieves the matched memory).
  E.g., uniform INT4 ≈ matched memory if 50% FP16 / 50% INT4 on the
  adaptive side (avg bits = 10).
- Re-quantize K cache (per-channel, KIVI-style) and V cache (per-token).
- Re-run forward, compute NLL on full sequence.

This is a small-scale demonstration of the *adaptive bit allocation
principle*; we don't claim production-grade quantization quality.

Outputs:
  results/exp6_adaptive_quant.json
  figures/exp6_quant.png
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_wikitext_validation_text, save_json, set_seed


def quantize_per_channel_K(K: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-channel symmetric quantization of K (KIVI-style for keys).

    K shape: (n_heads, L, d). Each channel d gets its own scale.
    Returns dequantized K (same shape, fp32).
    """
    if bits >= 16:
        return K.clone()
    qmax = (1 << (bits - 1)) - 1
    # per-channel scale
    abs_max = K.abs().amax(dim=(0, 1), keepdim=True).clamp_min(1e-9)  # (1, 1, d)
    scale = abs_max / qmax
    Kq = torch.round(K / scale).clamp(-qmax, qmax)
    return Kq * scale


def quantize_per_token_V(V: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token symmetric quantization of V (KIVI-style for values).

    V shape: (n_heads, L, d). Each (head, token) gets its own scale.
    """
    if bits >= 16:
        return V.clone()
    qmax = (1 << (bits - 1)) - 1
    abs_max = V.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)  # (n_heads, L, 1)
    scale = abs_max / qmax
    Vq = torch.round(V / scale).clamp(-qmax, qmax)
    return Vq * scale


def quantize_per_token_mixed(
    K: torch.Tensor, V: torch.Tensor, bits_per_token: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply different bit-width per (head, token) for V; for K we use channel-quant
    grouped by tokens with the same bit-width.

    bits_per_token: (n_heads, L) int — bit-width for each (head, token).
    """
    n_heads, L, d_v = V.shape
    Vq = V.clone()
    Kq = K.clone()
    unique_bits = torch.unique(bits_per_token).tolist()
    for b in unique_bits:
        b = int(b)
        if b >= 16:
            continue
        # mask of (head, token) that get this bit-width
        mask = (bits_per_token == b)  # (n_heads, L)
        # V per-token: handle each head separately
        for h in range(n_heads):
            tok_idx = mask[h].nonzero(as_tuple=False).flatten()
            if tok_idx.numel() == 0:
                continue
            qmax = (1 << (b - 1)) - 1
            sub_v = V[h, tok_idx, :]
            scales_v = sub_v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / qmax
            Vq[h, tok_idx, :] = torch.round(sub_v / scales_v).clamp(-qmax, qmax) * scales_v
            sub_k = K[h, tok_idx, :]
            scales_k = sub_k.abs().amax(dim=0, keepdim=True).clamp_min(1e-9) / qmax  # per-channel
            Kq[h, tok_idx, :] = torch.round(sub_k / scales_k).clamp(-qmax, qmax) * scales_k
    return Kq, Vq


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    out_root = Path(__file__).resolve().parents[1]
    results_dir = out_root / "results"
    figures_dir = out_root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    L = 512
    print("Loading GPT-2 small...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(device).eval()
    text = load_wikitext_validation_text(limit_chars=80_000)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=L)
    input_ids = enc.input_ids.to(device)
    L = input_ids.shape[1]
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    head_dim = model.config.n_embd // n_heads

    # Capture K, V from the standard forward
    captured_KV = {}
    handles = []

    def make_hook(layer):
        def hook(module, inputs, outputs):
            attn_out, _ = outputs if isinstance(outputs, tuple) else (outputs, None)
            # We need q, k, v AT THIS POINT. Easier: register a hook that captures
            # via running c_attn ourselves.
        return hook

    # Easier approach: monkeypatch attn forward to capture K, V tensors.
    captured = {}
    original_forwards = {}
    for layer in range(n_layers):
        original_forwards[layer] = model.transformer.h[layer].attn.forward
    def make_capture_forward(layer_id):
        attn_module = model.transformer.h[layer_id].attn
        def capture_forward(hidden_states, *args, **kwargs):
            qkv = attn_module.c_attn(hidden_states)
            q, k, v = qkv.split(attn_module.split_size, dim=2)
            B, T, _ = q.shape
            num_heads = attn_module.num_heads
            hd = attn_module.head_dim
            qh = q.view(B, T, num_heads, hd).transpose(1, 2)
            kh = k.view(B, T, num_heads, hd).transpose(1, 2)
            vh = v.view(B, T, num_heads, hd).transpose(1, 2)
            captured.setdefault("K", []).append(kh.detach().cpu())
            captured.setdefault("V", []).append(vh.detach().cpu())
            scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(hd)
            mask = torch.tril(torch.ones(T, T, device=scores.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
            attn_w = torch.softmax(scores, dim=-1)
            captured.setdefault("A", []).append(attn_w.detach().cpu())
            attn_out = torch.matmul(attn_w, vh)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, num_heads * hd)
            attn_out = attn_module.c_proj(attn_out)
            attn_out = attn_module.resid_dropout(attn_out)
            return (attn_out, None)
        return capture_forward
    try:
        for layer in range(n_layers):
            model.transformer.h[layer].attn.forward = make_capture_forward(layer)
        with torch.no_grad():
            out = model(input_ids)
        baseline_logits = out.logits[0]
        targets = input_ids[0, 1:]
        nll_full = F.cross_entropy(baseline_logits[:-1], targets, reduction="mean").item()
    finally:
        for layer in range(n_layers):
            model.transformer.h[layer].attn.forward = original_forwards[layer]

    print(f"Baseline NLL (no quantization): {nll_full:.4f}  PPL={math.exp(nll_full):.2f}", flush=True)
    Ks_layer = [captured["K"][layer][0] for layer in range(n_layers)]  # (n_heads, L, d)
    Vs_layer = [captured["V"][layer][0] for layer in range(n_layers)]
    Attns_layer = [captured["A"][layer][0] for layer in range(n_layers)]  # (n_heads, L, L)

    # Compute hit_count[layer, head, position] = #queries that attended >= tau
    threshold = 0.02
    hit_counts = []
    for layer in range(n_layers):
        # (n_heads, L, L) >= threshold; sum over query dim → (n_heads, L)
        hc = (Attns_layer[layer] >= threshold).sum(dim=1)  # int counts
        hit_counts.append(hc)

    # Define the eval: build new K, V (quantized), patch forward to use them
    def eval_with_quantized_kv(Ks_q, Vs_q):
        """Patch forward to use these K, V tensors (overriding c_attn output)."""
        original_forwards = {}
        for layer in range(n_layers):
            original_forwards[layer] = model.transformer.h[layer].attn.forward
        def make_quant_forward(layer_id, Kq, Vq):
            attn_module = model.transformer.h[layer_id].attn
            def quant_forward(hidden_states, *args, **kwargs):
                qkv = attn_module.c_attn(hidden_states)
                q, _, _ = qkv.split(attn_module.split_size, dim=2)
                B, T, _ = q.shape
                num_heads = attn_module.num_heads
                hd = attn_module.head_dim
                qh = q.view(B, T, num_heads, hd).transpose(1, 2)
                kh = Kq[None].to(q.device)  # (1, n_heads, L, hd)
                vh = Vq[None].to(q.device)
                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(hd)
                mask = torch.tril(torch.ones(T, T, device=scores.device, dtype=torch.bool))
                scores = scores.masked_fill(~mask, float("-inf"))
                attn_w = torch.softmax(scores, dim=-1)
                attn_out = torch.matmul(attn_w, vh)
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, num_heads * hd)
                attn_out = attn_module.c_proj(attn_out)
                attn_out = attn_module.resid_dropout(attn_out)
                return (attn_out, None)
            return quant_forward
        try:
            for layer in range(n_layers):
                model.transformer.h[layer].attn.forward = make_quant_forward(
                    layer, Ks_q[layer], Vs_q[layer]
                )
            with torch.no_grad():
                o = model(input_ids)
            preds = o.logits[0, :-1, :]
            tg = input_ids[0, 1:]
            return F.cross_entropy(preds, tg, reduction="mean").item()
        finally:
            for layer in range(n_layers):
                model.transformer.h[layer].attn.forward = original_forwards[layer]

    runs = [{"method": "fp32", "avg_bits": 32.0, "nll": nll_full, "ppl": math.exp(nll_full)}]
    print(f"  fp32         avg_bits=32.0  NLL={nll_full:.4f}  PPL={math.exp(nll_full):.2f}", flush=True)

    # Uniform quantization (same bits everywhere)
    for bits in [8, 6, 4, 3, 2]:
        Ks_q = [quantize_per_channel_K(K, bits) for K in Ks_layer]
        Vs_q = [quantize_per_token_V(V, bits) for V in Vs_layer]
        nll = eval_with_quantized_kv(Ks_q, Vs_q)
        runs.append({"method": f"uniform_{bits}bit", "avg_bits": float(bits),
                     "nll": nll, "ppl": math.exp(nll)})
        print(f"  uniform_{bits}bit  avg_bits={bits:.1f}   NLL={nll:.4f}  PPL={math.exp(nll):.2f}", flush=True)

    # Adaptive: top q% by hit count → 8 bits, next r% → 4 bits, rest → 2 bits
    # Sweep target average bits to compare against uniform.
    adaptive_configs = [
        # (top_pct_8b, mid_pct_4b, rest_3b) → expected avg
        (0.50, 0.50, 0.00),  # 50% 8-bit + 50% 4-bit         → avg 6.0
        (0.25, 0.75, 0.00),  # 25% 8 + 75% 4                 → avg 5.0
        (0.10, 0.90, 0.00),  # 10% 8 + 90% 4                 → avg 4.4
        (0.50, 0.00, 0.50),  # 50% 8 + 50% 3                 → avg 5.5
        (0.10, 0.40, 0.50),  # 10% 8 + 40% 4 + 50% 3         → avg 3.9
    ]
    for cfg in adaptive_configs:
        p8, p4, p_floor = cfg
        # rest get 3 bits (or whatever the third tier is for this config)
        floor_bits = 3
        # Per (layer, head), assign bits by hit-count rank
        Ks_q = []
        Vs_q = []
        all_bits_used = []
        for layer in range(n_layers):
            hc = hit_counts[layer]  # (n_heads, L)
            bits_per_token = torch.full((n_heads, L), floor_bits, dtype=torch.int32)
            for h in range(n_heads):
                ranks = torch.argsort(hc[h], descending=True)
                n8 = int(p8 * L)
                n4 = int(p4 * L)
                bits_per_token[h, ranks[:n8]] = 8
                bits_per_token[h, ranks[n8 : n8 + n4]] = 4
            all_bits_used.append(bits_per_token)
            Kq, Vq = quantize_per_token_mixed(Ks_layer[layer], Vs_layer[layer], bits_per_token)
            Ks_q.append(Kq)
            Vs_q.append(Vq)
        avg_bits = float(np.mean([b.float().mean().item() for b in all_bits_used]))
        nll = eval_with_quantized_kv(Ks_q, Vs_q)
        tag = f"adaptive_p8={p8:.2f}_p4={p4:.2f}"
        runs.append({"method": tag, "avg_bits": avg_bits,
                     "nll": nll, "ppl": math.exp(nll),
                     "config": {"p8": p8, "p4": p4, "p_floor": p_floor}})
        print(f"  {tag}  avg_bits={avg_bits:.2f}  NLL={nll:.4f}  PPL={math.exp(nll):.2f}", flush=True)

    # Random-priority adaptive (control): same proportions, but random selection.
    # If adaptive beats this, it's the hit-count signal that matters.
    rng = np.random.default_rng(42)
    for cfg in adaptive_configs:
        p8, p4, p_floor = cfg
        floor_bits = 3
        Ks_q = []
        Vs_q = []
        all_bits_used = []
        for layer in range(n_layers):
            bits_per_token = torch.full((n_heads, L), floor_bits, dtype=torch.int32)
            for h in range(n_heads):
                ranks = torch.tensor(rng.permutation(L))
                n8 = int(p8 * L)
                n4 = int(p4 * L)
                bits_per_token[h, ranks[:n8]] = 8
                bits_per_token[h, ranks[n8 : n8 + n4]] = 4
            all_bits_used.append(bits_per_token)
            Kq, Vq = quantize_per_token_mixed(Ks_layer[layer], Vs_layer[layer], bits_per_token)
            Ks_q.append(Kq)
            Vs_q.append(Vq)
        avg_bits = float(np.mean([b.float().mean().item() for b in all_bits_used]))
        nll = eval_with_quantized_kv(Ks_q, Vs_q)
        tag = f"random_p8={p8:.2f}_p4={p4:.2f}"
        runs.append({"method": tag, "avg_bits": avg_bits,
                     "nll": nll, "ppl": math.exp(nll),
                     "config": {"p8": p8, "p4": p4, "p_floor": p_floor}})
        print(f"  {tag}    avg_bits={avg_bits:.2f}  NLL={nll:.4f}  PPL={math.exp(nll):.2f}", flush=True)

    save_json({"L": L, "nll_full": nll_full, "ppl_full": math.exp(nll_full),
               "runs": runs}, results_dir / "exp6_adaptive_quant.json")

    # Plot: avg_bits vs PPL, separate lines for uniform / adaptive / random
    plt.figure(figsize=(8, 5.5))
    by_kind = {"uniform": [], "adaptive": [], "random": []}
    for r in runs:
        if r["method"] == "fp32":
            continue
        if r["method"].startswith("uniform"):
            by_kind["uniform"].append((r["avg_bits"], r["ppl"]))
        elif r["method"].startswith("adaptive"):
            by_kind["adaptive"].append((r["avg_bits"], r["ppl"]))
        elif r["method"].startswith("random"):
            by_kind["random"].append((r["avg_bits"], r["ppl"]))
    for k, pts in by_kind.items():
        pts.sort()
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        plt.plot(x, y, marker="o", label=k)
    plt.axhline(math.exp(nll_full), color="black", linestyle="--", label="fp32")
    plt.xlabel("Average bits per cached value (matched memory budget)")
    plt.ylabel("Perplexity (lower=better)")
    plt.title("Adaptive (HMA-driven) vs. uniform vs. random KV-cache quantization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "exp6_quant.png", dpi=140)
    print(f"Saved {figures_dir / 'exp6_quant.png'}")


if __name__ == "__main__":
    main()
