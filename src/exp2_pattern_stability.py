"""Experiment 2: Token-pattern stability in a real LLM.

Hypothesis H1: same token id (or n-gram context) attends to similar key
positions across occurrences (in *similar* contexts).

We run GPT-2-small on WikiText-2 validation text, capture per-(layer,
head) attention maps, group queries by token id and by 2-gram context,
and measure pattern overlap (Jaccard / IoU) between attention top-k sets
across grouped query pairs.

We compare:
  - identity (same token id)
  - 2-gram (same last-2 tokens)
  - distinct (random pairs across distinct token ids)  -- baseline

A genuinely token-conditioned attention would show identity >> distinct,
and 2-gram > identity.

Outputs:
  results/exp2_pattern_stability.json
  figures/exp2_jaccard.png
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_wikitext_validation_text, save_json, set_seed


def topk_set(attn_row: torch.Tensor, k: int) -> set[int]:
    """Top-k indices of a 1D attention vector (already softmaxed)."""
    return set(torch.topk(attn_row, k=min(k, attn_row.numel())).indices.tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(__file__).resolve().parents[1]
    results_dir = out_root / "results"
    figures_dir = out_root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    print("Loading GPT-2 small ...")
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(device).eval()
    model.config.output_attentions = True

    # Get text and tokenize
    text = load_wikitext_validation_text(limit_chars=80_000)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc.input_ids.to(device)
    L = input_ids.shape[1]
    print(f"Sequence length: {L}")

    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
    # attentions: tuple of n_layers tensors, each (1, n_heads, L, L)
    attns = [a[0].cpu() for a in out.attentions]
    n_layers = len(attns)
    n_heads = attns[0].shape[0]
    print(f"Captured attention maps: layers={n_layers}, heads={n_heads}")

    # group query positions by token id
    ids_list = input_ids[0].tolist()
    by_id: dict[int, list[int]] = defaultdict(list)
    for pos, t in enumerate(ids_list):
        if pos < 4:  # skip near-sink positions
            continue
        by_id[t].append(pos)
    # group by 2-gram (last-2 token context)
    by_2gram: dict[tuple, list[int]] = defaultdict(list)
    for pos in range(1, L):
        if pos < 4:
            continue
        ctx = (ids_list[pos - 1], ids_list[pos])
        by_2gram[ctx].append(pos)

    # only consider groups with at least 3 occurrences
    repeated_ids = [t for t, ps in by_id.items() if len(ps) >= 3]
    repeated_2grams = [g for g, ps in by_2gram.items() if len(ps) >= 3]
    print(f"Repeated token ids (>=3 occurrences): {len(repeated_ids)}")
    print(f"Repeated 2-grams (>=3 occurrences):   {len(repeated_2grams)}")

    K_TOP = 16
    n_pairs_per_group = 5
    rng = np.random.default_rng(42)

    # Compute Jaccard overlap distributions per (layer, head) for three settings
    # Use a few representative layer/head combos plus an aggregate
    layers_to_probe = [0, n_layers // 2, n_layers - 1]

    results = {"K_TOP": K_TOP, "n_layers": n_layers, "n_heads": n_heads, "L": L,
               "groups": {}}

    fig, axes = plt.subplots(1, len(layers_to_probe), figsize=(15, 4.5), sharey=True)

    for i, layer in enumerate(layers_to_probe):
        per_setting = {"identity": [], "2gram": [], "distinct": []}
        # identity pairs
        for tid in repeated_ids:
            ps = by_id[tid]
            # sample n_pairs_per_group random pairs
            for _ in range(min(n_pairs_per_group, len(ps) * (len(ps) - 1) // 2)):
                a, b = rng.choice(ps, size=2, replace=False)
                a, b = int(a), int(b)
                if a == b: continue
                # average across heads
                head_jacs = []
                for h in range(n_heads):
                    sa = topk_set(attns[layer][h, a, : a + 1], K_TOP)
                    sb = topk_set(attns[layer][h, b, : b + 1], K_TOP)
                    head_jacs.append(jaccard(sa, sb))
                per_setting["identity"].append(np.mean(head_jacs))
        # 2-gram pairs
        for g in repeated_2grams:
            ps = by_2gram[g]
            for _ in range(min(n_pairs_per_group, len(ps) * (len(ps) - 1) // 2)):
                a, b = rng.choice(ps, size=2, replace=False)
                a, b = int(a), int(b)
                if a == b: continue
                head_jacs = []
                for h in range(n_heads):
                    sa = topk_set(attns[layer][h, a, : a + 1], K_TOP)
                    sb = topk_set(attns[layer][h, b, : b + 1], K_TOP)
                    head_jacs.append(jaccard(sa, sb))
                per_setting["2gram"].append(np.mean(head_jacs))
        # distinct: random pairs across different token ids
        repeated_set = set(repeated_ids)
        for _ in range(len(per_setting["identity"])):
            ids_pair = rng.choice(repeated_ids, size=2, replace=False)
            ps_a = by_id[int(ids_pair[0])]
            ps_b = by_id[int(ids_pair[1])]
            a = int(rng.choice(ps_a))
            b = int(rng.choice(ps_b))
            head_jacs = []
            for h in range(n_heads):
                sa = topk_set(attns[layer][h, a, : a + 1], K_TOP)
                sb = topk_set(attns[layer][h, b, : b + 1], K_TOP)
                head_jacs.append(jaccard(sa, sb))
            per_setting["distinct"].append(np.mean(head_jacs))

        results["groups"][f"layer_{layer}"] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "median": float(np.median(v)), "n": len(v)}
            for k, v in per_setting.items()
        }
        # boxplot
        ax = axes[i]
        ax.boxplot([per_setting["identity"], per_setting["2gram"], per_setting["distinct"]],
                   tick_labels=["identity", "2-gram", "distinct"], showfliers=False)
        ax.set_title(f"GPT-2 layer {layer}")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel(f"Jaccard overlap of top-{K_TOP} attended positions")

    plt.suptitle("Token-pattern stability: do same tokens / 2-grams attend to overlapping positions?")
    plt.tight_layout()
    plt.savefig(figures_dir / "exp2_jaccard.png", dpi=140)
    save_json(results, results_dir / "exp2_pattern_stability.json")
    print(f"Saved {results_dir / 'exp2_pattern_stability.json'}")
    print(f"Saved {figures_dir / 'exp2_jaccard.png'}")
    for layer_key, vals in results["groups"].items():
        print(f"  {layer_key}:")
        for setting, m in vals.items():
            print(f"    {setting:10s}  mean={m['mean']:.3f}  median={m['median']:.3f}  n={m['n']}")


if __name__ == "__main__":
    main()
