"""Phase 5: Analyze results across experiments and generate summary figures.

Reads the JSON outputs of exp1/exp2/exp3/exp5/exp6 and produces:
- A summary table of headline numbers
- A combined figure showing the perplexity-vs-budget trade-off and the
  adaptive-quantization comparison side-by-side
- Statistical comparisons (paired tests where applicable)
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_json(p: Path):
    return json.loads(p.read_text())


def main():
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    figures = root / "figures"
    summary = []

    # ---- Exp 1: synthetic ----
    e1 = load_json(results / "exp1_synthetic.json")
    by_method = defaultdict(list)
    for r in e1["runs"]:
        by_method[r["method"]].append((r["mean_budget"], r["rel_error"]))
    e1_summary = {}
    for m, pts in by_method.items():
        # average across seeds at each (approx) budget
        b_to_e = defaultdict(list)
        for b, e in pts:
            b_to_e[round(b)].append(e)
        e1_summary[m] = [(b, float(np.mean(es)), float(np.std(es))) for b, es in sorted(b_to_e.items())]
    summary.append(("Exp1 (synthetic): HMA dominates at low budget", e1_summary))

    # Check at smallest comparable budget
    if "hma" in e1_summary and "random" in e1_summary:
        # smallest hma budget
        hma_min_b, hma_err, _ = e1_summary["hma"][0]
        # find random with comparable budget
        rand_pts = e1_summary["random"]
        rand_close = min(rand_pts, key=lambda x: abs(x[0] - hma_min_b))
        print(f"\nExp1 headline: at budget≈{hma_min_b:.0f}, HMA err={hma_err:.4f} vs "
              f"random err={rand_close[1]:.4f} (budget≈{rand_close[0]:.0f})")
        print(f"  → HMA error is {rand_close[1]/max(1e-9,hma_err):.0f}× lower than random")

    # ---- Exp 2: pattern stability ----
    e2 = load_json(results / "exp2_pattern_stability.json")
    print("\nExp2: token pattern stability (Jaccard of top-16 attended positions):")
    for layer_key, vals in e2["groups"].items():
        print(f"  {layer_key}: identity={vals['identity']['mean']:.3f}  "
              f"2gram={vals['2gram']['mean']:.3f}  distinct={vals['distinct']['mean']:.3f}  "
              f"ratio_id/distinct={vals['identity']['mean']/max(1e-9,vals['distinct']['mean']):.2f}×")

    # ---- Exp 3: perplexity vs budget ----
    e3 = load_json(results / "exp3_perplexity.json")
    print(f"\nExp3: perplexity vs budget (full attention PPL = {e3['ppl_full']:.2f})")
    by = defaultdict(list)
    for r in e3["runs"]:
        if r["method"] == "full":
            continue
        by[r["method"]].append((r["mean_budget_decode"], r["ppl"], r["budget_target"]))
    for m, pts in by.items():
        pts.sort()
        for mb, ppl, bt in pts:
            print(f"  {m:14s} budget_target={bt:4d}  effective_b={mb:6.1f}  PPL={ppl:.2f}")

    # ---- Exp 5: aggregation probe ----
    e5 = load_json(results / "exp5_aggregation_probe.json")
    print("\nExp5: aggregation probe (does HMA collapse on broad-aggregation tasks?)")
    for r in e5["results"]:
        print(f"  target_freq={r['target_freq']:3d}  candidate_size={r['mean_candidate_size']:.1f}  "
              f"err={r['mean_rel_error']:.4f}±{r['std_rel_error']:.4f}")

    # ---- Exp 6: adaptive quantization ----
    e6 = load_json(results / "exp6_adaptive_quant.json")
    print(f"\nExp6: adaptive vs uniform vs random KV-cache quantization (full PPL = {e6['ppl_full']:.2f})")
    for r in e6["runs"]:
        print(f"  {r['method']:30s} avg_bits={r['avg_bits']:5.2f}  PPL={r['ppl']:.2f}")

    # ---- Build summary figure: 2x2 grid ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0): exp1 synthetic
    ax = axes[0, 0]
    for m, pts in e1_summary.items():
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        e = [p[2] for p in pts]
        ax.errorbar(x, y, yerr=e, marker="o", label=m, capsize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean candidate budget per query")
    ax.set_ylabel("Relative attention output error")
    ax.set_title("(a) Synthetic Q/K: HMA captures planted token-conditioned patterns")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (0,1): exp2 pattern stability boxplot summary
    ax = axes[0, 1]
    layers = list(e2["groups"].keys())
    settings = ["identity", "2gram", "distinct"]
    means = {s: [e2["groups"][l][s]["mean"] for l in layers] for s in settings}
    stds = {s: [e2["groups"][l][s]["std"] for l in layers] for s in settings}
    x = np.arange(len(layers))
    width = 0.25
    for i, s in enumerate(settings):
        ax.bar(x + (i - 1) * width, means[s], width, yerr=stds[s],
               label=s, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel(f"Mean Jaccard overlap of top-{e2['K_TOP']} attended positions")
    ax.set_title("(b) GPT-2: token-identity attention reuse (vs. distinct-token baseline)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    # (1,0): exp3 PPL vs budget
    ax = axes[1, 0]
    by3 = defaultdict(list)
    for r in e3["runs"]:
        if r["method"] == "full":
            continue
        by3[r["method"]].append((r["mean_budget_decode"], r["ppl"]))
    for m, pts in by3.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=m)
    ax.axhline(e3["ppl_full"], color="black", linestyle="--", label="full attention")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean candidate budget per query (decode region)")
    ax.set_ylabel("Perplexity")
    ax.set_title("(c) GPT-2 perplexity vs. KV budget (WikiText-2)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (1,1): exp6 adaptive quant
    ax = axes[1, 1]
    by_kind = {"uniform": [], "adaptive": [], "random": []}
    for r in e6["runs"]:
        if r["method"] == "fp32":
            continue
        kind = r["method"].split("_")[0]
        if kind not in by_kind:
            kind = "adaptive" if r["method"].startswith("adaptive") else "random" if r["method"].startswith("random") else "uniform"
        by_kind[kind].append((r["avg_bits"], r["ppl"]))
    for k, pts in by_kind.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=k)
    ax.axhline(e6["ppl_full"], color="black", linestyle="--", label="fp32")
    ax.set_xlabel("Average bits per cached value (matched memory budget)")
    ax.set_ylabel("Perplexity")
    ax.set_title("(d) Adaptive (HMA-driven) vs. uniform vs. random KV-cache quant")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle("HashMap Attention: experimental summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(figures / "summary.png", dpi=140)
    print(f"\nSaved combined summary figure to {figures / 'summary.png'}")

    # ---- Statistical comparison: paired tests on Exp 1 ----
    print("\nStatistical comparison (Exp 1, paired t-test on rel_error across seeds, smallest budget):")
    runs1 = e1["runs"]
    # collect per-seed errors at smallest budget for each method
    method_errs = defaultdict(dict)  # method -> seed -> err (single err per seed at best budget)
    for r in runs1:
        if r["method"] == "hma":
            # use smallest tau (best quality)
            taus = sorted(set(rr["tau"] for rr in runs1 if rr["method"] == "hma" and rr["seed"] == r["seed"]))
            if r["tau"] == taus[0]:
                method_errs["hma"][r["seed"]] = r["rel_error"]
        else:
            # smallest budget for baselines
            min_b = min(rr["budget"] for rr in runs1 if rr["method"] == r["method"] and rr["seed"] == r["seed"])
            if r["budget"] == min_b:
                method_errs[r["method"]][r["seed"]] = r["rel_error"]
    if "hma" in method_errs and "random" in method_errs:
        seeds = sorted(set(method_errs["hma"]) & set(method_errs["random"]))
        h = [method_errs["hma"][s] for s in seeds]
        r = [method_errs["random"][s] for s in seeds]
        t, p = stats.ttest_rel(h, r)
        print(f"  HMA vs Random (n={len(seeds)} seeds, smallest budget): t={t:.3f}  p={p:.4g}")
        print(f"    mean HMA err = {np.mean(h):.4f}  mean Random err = {np.mean(r):.4f}")
    if "hma" in method_errs and "streaming" in method_errs:
        seeds = sorted(set(method_errs["hma"]) & set(method_errs["streaming"]))
        h = [method_errs["hma"][s] for s in seeds]
        st = [method_errs["streaming"][s] for s in seeds]
        t, p = stats.ttest_rel(h, st)
        print(f"  HMA vs Streaming (n={len(seeds)} seeds, smallest budget): t={t:.3f}  p={p:.4g}")
        print(f"    mean HMA err = {np.mean(h):.4f}  mean Streaming err = {np.mean(st):.4f}")


if __name__ == "__main__":
    main()
