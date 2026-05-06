"""Experiment 5: Aggregation failure-mode probe (RULER-cwe-style synthetic).

MagicPIG showed that LSH-Top-K methods collapse on aggregation tasks
because they only attend to a sparse top-k subset, missing the broad
distributional information needed for tasks like Common Word Extraction
(cwe). The same trap plausibly affects HMA: if HMA selects only a small
candidate set per query, broad aggregation tasks should suffer.

We construct a synthetic "common word counting" probe at small scale:
- Sequence consists of N "filler" tokens drawn uniformly + repeated
  occurrences of K target words placed at random positions.
- The task: for the LAST query position, the attention output should
  reflect the average value-vector over occurrences of a target word.

We measure the relative error of HMA-restricted attention against full
attention output for this aggregation. If HMA fails this case, error
will be much higher than its general error in Exp 1, replicating the
documented failure mode.

Outputs:
  results/exp5_aggregation_probe.json
  figures/exp5_aggregation.png
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import dense_attention, relative_error, save_json, set_seed
from hma import HMAConfig, HashMapAttention


def make_probe(
    seq_len: int,
    n_target_words: int,
    target_freq: int,
    vocab_size: int,
    n_heads: int,
    d_head: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a sequence with `n_target_words` target tokens, each appearing
    `target_freq` times in random positions. The last position is set to
    one of the target tokens (the 'query' for which we evaluate aggregation).

    The target tokens are special: their per-vocab key direction is
    distinctive so attention from a target query token sharply selects
    other target token positions. (We bake structure in so both full
    attention and HMA can in principle solve the task.)
    """
    g = torch.Generator().manual_seed(seed)
    targets = torch.arange(1, n_target_words + 1)  # ids 1..n_target
    # Build positions
    token_ids = torch.randint(n_target_words + 1, vocab_size,
                               (seq_len,), generator=g)
    # Sprinkle target tokens
    all_target_positions = []
    for t in targets:
        positions = torch.randperm(seq_len - 1, generator=g)[:target_freq].tolist()
        for p in positions:
            token_ids[p] = t
        all_target_positions.append(positions)
    # last position is one of the target tokens
    final_target = int(targets[0])
    token_ids[-1] = final_target

    # Per-vocab key direction
    vocab_k_dir = torch.randn(n_heads, vocab_size, d_head, generator=g)
    vocab_q_dir = torch.randn(n_heads, vocab_size, d_head, generator=g)
    # Make target tokens query-key aligned: target[i] queries strongly select target[i] keys
    for t in targets:
        vocab_q_dir[:, t, :] = vocab_k_dir[:, t, :] * 4  # strong alignment

    Q = vocab_q_dir[:, token_ids, :] + 0.5 * torch.randn(n_heads, seq_len, d_head, generator=g)
    K = vocab_k_dir[:, token_ids, :] + 0.5 * torch.randn(n_heads, seq_len, d_head, generator=g)
    V = torch.randn(n_heads, seq_len, d_head, generator=g)
    return token_ids, Q, K, V


def hma_aggregation_error(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    token_ids: torch.Tensor, threshold: float,
) -> tuple[float, int, int]:
    """Build HMA from the *same* sequence's full attention (oracle pre-fill),
    then measure the error of HMA-restricted attention output at the LAST
    query position vs. dense oracle.

    Returns (relative_error, candidate_count, total_target_count).
    """
    cfg = HMAConfig(threshold=threshold, n_gram=1, sink_size=4, window_size=8,
                     max_budget=None)
    hma = HashMapAttention(n_layers=1, n_heads=Q.shape[0], config=cfg)
    oracle_out, oracle_w = dense_attention(Q, K, V, causal=True)
    hma.populate(0, oracle_w, token_ids)

    n_heads, q_len, d = Q.shape
    last = q_len - 1
    last_token = int(token_ids[last])
    # how many target positions exist?
    total_target_count = int((token_ids == last_token).sum().item())

    # average candidates over heads
    cand_sizes = []
    head_errors = []
    scale = 1.0 / math.sqrt(d)
    for h in range(n_heads):
        cand = sorted(p for p in hma.candidate_positions(0, h, last) if p <= last)
        cand_sizes.append(len(cand))
        idx = torch.tensor(cand, dtype=torch.long)
        k_sel = K[h].index_select(0, idx)
        v_sel = V[h].index_select(0, idx)
        logits = (Q[h, last] @ k_sel.T) * scale
        weights = torch.softmax(logits, dim=-1)
        approx_v = weights @ v_sel
        err = (approx_v - oracle_out[h, last]).norm() / oracle_out[h, last].norm().clamp_min(1e-9)
        head_errors.append(err.item())
    return float(np.mean(head_errors)), int(np.mean(cand_sizes)), total_target_count


def main():
    set_seed(42)
    out_root = Path(__file__).resolve().parents[1]
    results_dir = out_root / "results"
    figures_dir = out_root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    seq_len = 512
    n_heads = 4
    d_head = 32
    vocab_size = 100
    threshold = 0.005
    target_freq_values = [1, 2, 4, 8, 16, 32, 64]
    n_seeds = 5

    results = []
    for tfreq in target_freq_values:
        errs, cand_sizes, target_counts = [], [], []
        for seed in range(n_seeds):
            ids, Q, K, V = make_probe(
                seq_len=seq_len, n_target_words=4, target_freq=tfreq,
                vocab_size=vocab_size, n_heads=n_heads, d_head=d_head, seed=seed,
            )
            err, cand, tcount = hma_aggregation_error(Q, K, V, ids, threshold)
            errs.append(err)
            cand_sizes.append(cand)
            target_counts.append(tcount)
        results.append({
            "target_freq": tfreq,
            "mean_target_count": float(np.mean(target_counts)),
            "mean_candidate_size": float(np.mean(cand_sizes)),
            "mean_rel_error": float(np.mean(errs)),
            "std_rel_error": float(np.std(errs)),
        })
        print(f"  target_freq={tfreq:3d}  target_count≈{np.mean(target_counts):.1f}  "
              f"candidates={np.mean(cand_sizes):.1f}  err={np.mean(errs):.4f}±{np.std(errs):.4f}")

    save_json({"config": {"seq_len": seq_len, "threshold": threshold,
                          "n_heads": n_heads, "d_head": d_head,
                          "vocab_size": vocab_size}, "results": results},
              results_dir / "exp5_aggregation_probe.json")

    plt.figure(figsize=(7, 5))
    x = [r["target_freq"] for r in results]
    y = [r["mean_rel_error"] for r in results]
    s = [r["std_rel_error"] for r in results]
    plt.errorbar(x, y, yerr=s, marker="o", capsize=3, label="HMA aggregation error")
    plt.xlabel("# occurrences of the target token (aggregation breadth)")
    plt.ylabel("Relative error (HMA vs. full) at final query")
    plt.title("HMA aggregation-task probe (cwe-style)")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "exp5_aggregation.png", dpi=140)
    print(f"Saved {figures_dir / 'exp5_aggregation.png'}")


if __name__ == "__main__":
    main()
