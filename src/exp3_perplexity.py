"""Experiment 3+4: End-to-end perplexity vs. KV-budget on GPT-2.

Setup:
- Take a passage of WikiText-2 validation text, length L=512 tokens.
- Treat first L_prefill=L/2 tokens as prefill (full attention; populate
  HMA tables from observed attention).
- For tokens L_prefill .. L (decode region), evaluate next-token loss
  with attention restricted to the candidate set chosen by each method.

Methods compared:
  * Full        -- oracle (no restriction beyond causal mask)
  * Streaming(w) -- sinks=4 + sliding window of size w
  * Random(b)   -- random b positions (uniform across heads)
  * H2O(b)      -- top-b accumulated-attention positions (oracle eviction; per layer/head)
  * HMA-id(b)   -- HashMap with token-identity key, capped at b
  * HMA-2g(b)   -- HashMap with 2-gram context key, capped at b

Implementation: All masks pre-computed as (n_layers, n_heads, L, L) bool
tensors using fully-vectorized torch ops. The forward pass uses custom
attention forwards that add a (-inf, 0) additive mask before softmax.

Outputs:
  results/exp3_perplexity.json
  figures/exp3_ppl_vs_budget.png
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_wikitext_validation_text, save_json, set_seed


def _ngram_keys(token_ids: list[int], n: int) -> list[int]:
    keys = []
    for i in range(len(token_ids)):
        ctx = tuple(token_ids[max(0, i - n + 1) : i + 1])
        keys.append(hash(ctx) & 0xFFFFFFFF)
    return keys


def make_causal(L: int) -> torch.Tensor:
    return torch.tril(torch.ones(L, L, dtype=torch.bool))


def make_sink_window(L: int, sink: int, window: int) -> torch.Tensor:
    """Per-(q,k) bool mask: sinks union sliding window of size `window`."""
    q_idx = torch.arange(L)[:, None]
    k_idx = torch.arange(L)[None, :]
    sinks_ok = k_idx < sink
    window_ok = (k_idx >= q_idx - window + 1) & (k_idx <= q_idx)
    return sinks_ok | window_ok


def make_random_mask(L: int, budget: int, sink: int, seed: int = 123) -> torch.Tensor:
    """(L, L) bool. Per-query, randomly select budget keys among [0, q]."""
    g = torch.Generator().manual_seed(seed)
    causal = make_causal(L)
    sw = make_sink_window(L, sink=sink, window=1)  # sinks + current
    out = sw & causal
    # For each query row, fill remaining slots randomly
    for q in range(L):
        avail = q + 1
        already = int(out[q, :avail].sum())
        n_extra = max(0, budget - already)
        if n_extra > 0 and avail > already:
            available = (~out[q, :avail]).nonzero(as_tuple=False).flatten()
            if available.numel() > 0:
                perm = available[torch.randperm(available.numel(), generator=g)][:n_extra]
                out[q, perm] = True
    return out


def make_h2o_mask(
    accum: torch.Tensor,  # (n_heads, L) accumulated attention per layer
    L: int, budget: int, sink: int = 4,
) -> torch.Tensor:
    """(n_heads, L, L) bool, top-b by accumulated attention per query.

    Uses prefix accumulated attention (oracle eviction).
    """
    n_heads = accum.shape[0]
    causal = make_causal(L)
    sw = make_sink_window(L, sink=sink, window=1)
    base = sw & causal  # (L, L)
    out = base[None].expand(n_heads, L, L).clone()
    # For each (head, q), pick top-(budget - already) by accum score among unset positions
    for h in range(n_heads):
        scores = accum[h]
        for q in range(L):
            avail = q + 1
            already = int(out[h, q, :avail].sum())
            n_extra = max(0, budget - already)
            if n_extra > 0 and avail > already:
                # mask out already-included
                masked = scores[:avail].clone()
                masked[out[h, q, :avail]] = float("-inf")
                topk_v, topk_i = torch.topk(masked, k=min(n_extra, avail - already))
                out[h, q, topk_i] = True
    return out


def make_hma_mask(
    attns_layer: torch.Tensor,  # (n_heads, L, L) full oracle attention for one layer
    keys_by_pos: list[int],
    threshold: float,
    prefill_len: int,
    sink: int,
    window: int,
    cap: int | None,
) -> torch.Tensor:
    """(n_heads, L, L) bool HMA-restricted mask for a single layer."""
    n_heads, L, _ = attns_layer.shape
    causal = make_causal(L)
    sw = make_sink_window(L, sink=sink, window=window) & causal  # (L, L)
    out = sw[None].expand(n_heads, L, L).clone()
    keys_t = torch.tensor(keys_by_pos)
    # For each head, build the H dictionary from prefill, then for each query
    # add positions from H[key(q)]
    for h in range(n_heads):
        H: dict[int, set[int]] = defaultdict(set)
        m = attns_layer[h, :prefill_len, :prefill_len] >= threshold  # (prefill, prefill)
        # Build H
        for q in range(prefill_len):
            pos = m[q].nonzero(as_tuple=False).flatten().tolist()
            if pos:
                H[keys_by_pos[q]].update(pos)
        # Build a (L, L) bool stored-mask
        stored = torch.zeros(L, L, dtype=torch.bool)
        for q in range(L):
            ps = H.get(keys_by_pos[q], None)
            if ps:
                ps_t = torch.tensor(sorted(p for p in ps if p <= q), dtype=torch.long)
                if ps_t.numel() > 0:
                    stored[q, ps_t] = True
        out[h] = (out[h] | stored) & causal
        if cap is not None:
            # cap each query's count at `cap`. Mandatory keep = sw[q] (for q within causal).
            # Drop excess from stored-only positions (those in stored but not sw).
            mandatory = sw  # (L, L)
            # Optional positions = (out & ~mandatory)
            optional = out[h] & ~mandatory
            mandatory_count = mandatory.sum(dim=1)  # (L,)
            n_extra_allowed = (cap - mandatory_count).clamp(min=0)  # per-query cap on optional
            # For each q, keep the first n_extra_allowed[q] optional positions
            for q in range(L):
                opt_ps = optional[q].nonzero(as_tuple=False).flatten()
                n_keep = int(n_extra_allowed[q])
                if opt_ps.numel() > n_keep:
                    drop = opt_ps[n_keep:]
                    out[h, q, drop] = False
    return out


def evaluate_with_layerwise_masks(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    layer_masks: list[torch.Tensor],  # list of (n_heads, L, L) bool, on device
    decode_start: int,
) -> tuple[float, float]:
    """Forward pass with each layer's GPT2Attention forward replaced.

    Returns (NLL on decode tokens, mean decode budget).
    """
    n_layers = model.config.n_layer
    additive = []
    for lm in layer_masks:
        a = torch.where(lm, torch.zeros_like(lm, dtype=torch.float32),
                        torch.full_like(lm, float("-inf"), dtype=torch.float32))
        additive.append(a)

    # Compute mean decode budget
    budgets = []
    for lm in layer_masks:
        # lm: (n_heads, L, L). Decode budgets: rows in [decode_start, L)
        budgets.append(lm[:, decode_start:, :].sum(dim=-1).float().mean().item())
    mean_budget = float(np.mean(budgets))

    original_forwards = {}
    for layer in range(n_layers):
        original_forwards[layer] = model.transformer.h[layer].attn.forward

    def make_forward(layer_id):
        attn_module = model.transformer.h[layer_id].attn
        add = additive[layer_id]

        def custom_forward(hidden_states, *args, **kwargs):
            qkv = attn_module.c_attn(hidden_states)
            q, k, v = qkv.split(attn_module.split_size, dim=2)
            B, T, _ = q.shape
            head_dim = attn_module.head_dim
            num_heads = attn_module.num_heads
            q = q.view(B, T, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, T, num_heads, head_dim).transpose(1, 2)
            v = v.view(B, T, num_heads, head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            scores = scores + add[None]  # broadcast batch
            attn_w = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_w, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, num_heads * head_dim)
            attn_out = attn_module.c_proj(attn_out)
            attn_out = attn_module.resid_dropout(attn_out)
            return (attn_out, None)
        return custom_forward

    try:
        for layer in range(n_layers):
            model.transformer.h[layer].attn.forward = make_forward(layer)
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits[0]
        targets = input_ids[0, decode_start + 1 :]
        preds = logits[decode_start : -1, :]
        nll = F.cross_entropy(preds, targets, reduction="mean").item()
    finally:
        for layer in range(n_layers):
            model.transformer.h[layer].attn.forward = original_forwards[layer]

    return nll, mean_budget


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
    decode_start = L // 2

    print("Loading GPT-2 small...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(device).eval()

    text = load_wikitext_validation_text(limit_chars=80_000)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=L)
    input_ids = enc.input_ids.to(device)
    L = input_ids.shape[1]
    decode_start = L // 2
    print(f"Sequence length L={L}, decode_start={decode_start}", flush=True)

    print("Running oracle forward pass...", flush=True)
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
    attns_cpu = [a[0].cpu() for a in out.attentions]  # list of (n_heads, L, L)
    n_layers = model.config.n_layer
    n_heads = model.config.n_head

    targets = input_ids[0, decode_start + 1 :]
    preds_oracle = out.logits[0, decode_start : -1, :]
    nll_full = F.cross_entropy(preds_oracle, targets, reduction="mean").item()
    print(f"Full-attention NLL on decode region: {nll_full:.4f}  PPL={math.exp(nll_full):.2f}", flush=True)

    accum_cpu = [attns_cpu[layer][:, :decode_start, :].mean(dim=1) for layer in range(n_layers)]
    token_ids_list = input_ids[0].tolist()
    keys_id = _ngram_keys(token_ids_list, n=1)
    keys_2g = _ngram_keys(token_ids_list, n=2)

    budgets_to_test = [16, 32, 64, 128, 256]

    runs = [{"method": "full", "budget_target": L, "nll": nll_full,
             "ppl": math.exp(nll_full),
             "mean_budget_decode": float((decode_start + L + 1) / 2)}]
    print(f"  full         budget≈{runs[0]['mean_budget_decode']:.0f}      "
          f"NLL={nll_full:.4f}  PPL={math.exp(nll_full):.2f}", flush=True)

    for b in budgets_to_test:
        # Streaming (uniform across heads/layers): build once
        window = max(1, b - 4)
        sw_mask = (make_sink_window(L, sink=4, window=window) & make_causal(L))
        layer_masks = [sw_mask[None].expand(n_heads, L, L).clone().to(device) for _ in range(n_layers)]
        nll, mb = evaluate_with_layerwise_masks(model, input_ids, layer_masks, decode_start)
        runs.append({"method": "streaming", "budget_target": b, "nll": nll,
                     "ppl": math.exp(nll), "mean_budget_decode": mb})
        print(f"  streaming    budget={b:4d}  NLL={nll:.4f}  PPL={math.exp(nll):.2f}  mean_b={mb:.1f}", flush=True)

        # Random
        rmask = make_random_mask(L, budget=b, sink=4, seed=123 + b)
        layer_masks = [rmask[None].expand(n_heads, L, L).clone().to(device) for _ in range(n_layers)]
        nll, mb = evaluate_with_layerwise_masks(model, input_ids, layer_masks, decode_start)
        runs.append({"method": "random", "budget_target": b, "nll": nll,
                     "ppl": math.exp(nll), "mean_budget_decode": mb})
        print(f"  random       budget={b:4d}  NLL={nll:.4f}  PPL={math.exp(nll):.2f}  mean_b={mb:.1f}", flush=True)

        # H2O
        layer_masks = []
        for layer in range(n_layers):
            lm = make_h2o_mask(accum_cpu[layer], L=L, budget=b, sink=4)
            layer_masks.append(lm.to(device))
        nll, mb = evaluate_with_layerwise_masks(model, input_ids, layer_masks, decode_start)
        runs.append({"method": "h2o", "budget_target": b, "nll": nll,
                     "ppl": math.exp(nll), "mean_budget_decode": mb})
        print(f"  h2o          budget={b:4d}  NLL={nll:.4f}  PPL={math.exp(nll):.2f}  mean_b={mb:.1f}", flush=True)

        # HMA identity: half budget for local window, rest from stored
        hma_window = max(4, b // 2)
        layer_masks = []
        for layer in range(n_layers):
            lm = make_hma_mask(attns_cpu[layer], keys_id, threshold=0.02,
                                prefill_len=decode_start, sink=4,
                                window=hma_window, cap=b)
            layer_masks.append(lm.to(device))
        nll, mb = evaluate_with_layerwise_masks(model, input_ids, layer_masks, decode_start)
        runs.append({"method": "hma_identity", "budget_target": b, "nll": nll,
                     "ppl": math.exp(nll), "mean_budget_decode": mb})
        print(f"  hma_identity budget={b:4d}  NLL={nll:.4f}  PPL={math.exp(nll):.2f}  mean_b={mb:.1f}", flush=True)

        # HMA 2-gram
        layer_masks = []
        for layer in range(n_layers):
            lm = make_hma_mask(attns_cpu[layer], keys_2g, threshold=0.02,
                                prefill_len=decode_start, sink=4,
                                window=hma_window, cap=b)
            layer_masks.append(lm.to(device))
        nll, mb = evaluate_with_layerwise_masks(model, input_ids, layer_masks, decode_start)
        runs.append({"method": "hma_2gram", "budget_target": b, "nll": nll,
                     "ppl": math.exp(nll), "mean_budget_decode": mb})
        print(f"  hma_2gram    budget={b:4d}  NLL={nll:.4f}  PPL={math.exp(nll):.2f}  mean_b={mb:.1f}", flush=True)

    save_json({"L": L, "decode_start": decode_start, "nll_full": nll_full,
               "ppl_full": math.exp(nll_full), "runs": runs},
              results_dir / "exp3_perplexity.json")

    plt.figure(figsize=(8, 5.5))
    by_method = defaultdict(list)
    for r in runs:
        if r["method"] == "full":
            continue
        by_method[r["method"]].append((r["mean_budget_decode"], r["ppl"]))
    for m, pts in by_method.items():
        pts.sort()
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        plt.plot(x, y, marker="o", label=m)
    plt.axhline(math.exp(nll_full), color="black", linestyle="--", label="full attention")
    plt.xscale("log")
    plt.xlabel("Mean candidate budget per query (decode region)")
    plt.ylabel("Perplexity (lower=better)")
    plt.title(f"GPT-2 perplexity vs. KV budget (WikiText-2, L={L}, decode={L - decode_start})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "exp3_ppl_vs_budget.png", dpi=140)
    print(f"Saved {figures_dir / 'exp3_ppl_vs_budget.png'}")


if __name__ == "__main__":
    main()
