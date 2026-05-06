"""Experiment 1: Synthetic Q/K stress test for HMA.

Creates a synthetic sequence with planted "token-conditioned attention
patterns": each token id has a fixed *attended position pattern* relative
to itself, plus noise. This is the best case for HMA — token identity
fully determines attention. We sweep budget and measure:
  - relative attention output error vs. dense oracle
  - top-k recall
  - average candidate budget per query

We compare HMA against:
  - Streaming (sinks + sliding window)
  - Random budget-k

Outputs:
  results/exp1_synthetic.json
  figures/exp1_budget_vs_error.png
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from hma import HMAConfig, HashMapAttention, baseline_random, baseline_streaming
from utils import dense_attention, relative_error, save_json, set_seed


def make_synthetic_sequence(
    seq_len: int,
    vocab_size: int,
    n_heads: int,
    d_head: int,
    pattern_strength: float = 5.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (token_ids, Q, K, V) where each token id has a 'preferred key
    direction' so attention is partially token-conditioned."""
    g = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(0, vocab_size, (seq_len,), generator=g)
    # per-vocab "query/key directions" (heads, vocab, d)
    vocab_q_dir = torch.randn(n_heads, vocab_size, d_head, generator=g)
    vocab_k_dir = torch.randn(n_heads, vocab_size, d_head, generator=g)
    # base random Q/K per position
    Q_noise = torch.randn(n_heads, seq_len, d_head, generator=g)
    K_noise = torch.randn(n_heads, seq_len, d_head, generator=g)
    # mix in the token-dependent direction
    Q = pattern_strength * vocab_q_dir[:, token_ids, :] + Q_noise
    K = pattern_strength * vocab_k_dir[:, token_ids, :] + K_noise
    V = torch.randn(n_heads, seq_len, d_head, generator=g)
    return token_ids, Q, K, V


def compute_method(
    name: str,
    layer: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    hma: HashMapAttention | None,
    budget: int,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Returns (output, mean_budget). `budget` interpreted per method."""
    n_heads, q_len, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    out = torch.zeros_like(Q)
    budgets = []
    rng = torch.Generator().manual_seed(seed)
    for h in range(n_heads):
        for q in range(q_len):
            if name == "hma":
                cand = sorted(p for p in hma.candidate_positions(layer, h, q) if p <= q)
            elif name == "streaming":
                cand = sorted(baseline_streaming(q, sink=4, window=budget))
            elif name == "random":
                cand = sorted(baseline_random(q, k_len=q_len, budget=budget, rng=rng))
            else:
                raise ValueError(name)
            cand = [p for p in cand if p <= q] or [q]
            budgets.append(len(cand))
            idx = torch.tensor(cand, dtype=torch.long)
            k_sel = K[h].index_select(0, idx)
            v_sel = V[h].index_select(0, idx)
            logits = (Q[h, q] @ k_sel.T) * scale
            weights = torch.softmax(logits, dim=-1)
            out[h, q] = weights @ v_sel
    return out, float(np.mean(budgets))


def main():
    set_seed(42)
    out_root = Path(__file__).resolve().parents[1]
    results_dir = out_root / "results"
    figures_dir = out_root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # Experiment config
    seq_len = 512
    vocab_size = 100  # small → lots of repeated tokens
    n_heads = 4
    d_head = 32
    n_seeds = 3
    budgets = [16, 32, 64, 128, 256]

    # Sweep tau values for HMA
    tau_values = [0.001, 0.005, 0.01, 0.02, 0.05]

    all_results = {"config": {
        "seq_len": seq_len, "vocab_size": vocab_size,
        "n_heads": n_heads, "d_head": d_head, "n_seeds": n_seeds,
        "budgets": budgets, "tau_values": tau_values,
    }, "runs": []}

    for seed in range(n_seeds):
        token_ids, Q, K, V = make_synthetic_sequence(
            seq_len, vocab_size, n_heads, d_head, seed=seed
        )
        # Oracle
        oracle_out, oracle_w = dense_attention(Q, K, V, causal=True)
        # HMA: sweep tau
        for tau in tau_values:
            cfg = HMAConfig(threshold=tau, n_gram=1, sink_size=4, window_size=8,
                             max_budget=None)
            hma = HashMapAttention(n_layers=1, n_heads=n_heads, config=cfg)
            hma.populate(0, oracle_w, token_ids)
            approx_out, mean_budget = compute_method(
                "hma", 0, Q, K, V, hma, budget=0, seed=seed
            )
            err = relative_error(approx_out, oracle_out)
            all_results["runs"].append({
                "seed": seed, "method": "hma", "tau": tau,
                "mean_budget": mean_budget, "rel_error": err,
            })
            print(f"  [seed={seed}] hma tau={tau:.3f}  budget={mean_budget:.1f}  err={err:.4f}")
        # Baselines: budget sweep
        for b in budgets:
            for name in ("streaming", "random"):
                approx_out, mean_budget = compute_method(
                    name, 0, Q, K, V, hma=None, budget=b, seed=seed
                )
                err = relative_error(approx_out, oracle_out)
                all_results["runs"].append({
                    "seed": seed, "method": name, "budget": b,
                    "mean_budget": mean_budget, "rel_error": err,
                })
                print(f"  [seed={seed}] {name} budget={b}  mean_budget={mean_budget:.1f}  err={err:.4f}")

    save_json(all_results, results_dir / "exp1_synthetic.json")

    # ---- plot ----
    plt.figure(figsize=(7, 5))
    # group: method -> list of (mean_budget_avg, err_avg, err_std)
    grouped = {}
    for r in all_results["runs"]:
        m = r["method"]
        key = (m, r.get("tau", r.get("budget")))
        grouped.setdefault(key, []).append((r["mean_budget"], r["rel_error"]))
    by_method = {}
    for (m, _), lst in grouped.items():
        budgets_arr = np.array([x[0] for x in lst])
        errs_arr = np.array([x[1] for x in lst])
        by_method.setdefault(m, []).append((budgets_arr.mean(), errs_arr.mean(), errs_arr.std()))
    for m, pts in by_method.items():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        b = [x[0] for x in pts_sorted]
        e = [x[1] for x in pts_sorted]
        s = [x[2] for x in pts_sorted]
        plt.errorbar(b, e, yerr=s, marker="o", label=m, capsize=3)
    plt.xlabel("Mean candidate budget per query")
    plt.ylabel("Relative attention output error")
    plt.title(f"Synthetic Q/K (vocab={vocab_size}, seq={seq_len}): HMA vs. baselines")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "exp1_budget_vs_error.png", dpi=140)
    print(f"Saved figure to {figures_dir / 'exp1_budget_vs_error.png'}")


if __name__ == "__main__":
    main()
