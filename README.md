# HashMap Attention

Tests a per-token-identity (or fuzzy-n-gram) lookup table of attention
patterns built during prefill and queried during decode, plus its use
for adaptive KV-cache quantization. The full write-up is in
[REPORT.md](REPORT.md).

## Key findings

- **Synthetic stress test**: when attention is genuinely
  token-conditioned, HMA achieves **98× lower attention output error**
  than random selection at the same budget (paired t-test p < 1e-4).
- **Real LLM (GPT-2-small) on WikiText-2**: same-token Jaccard overlap
  of top-16 attended positions is 0.09–0.18, vs. 0.02–0.07 for
  distinct tokens — a 1.8–5× lift, statistically meaningful but
  modest in absolute terms.
- **End-to-end perplexity**: HMA matches/beats sliding-window
  StreamingLLM at small budgets (PPL 90.6 at effective budget 25 vs.
  streaming PPL 95.7 at budget 32) and dramatically beats H2O / random.
- **Aggregation tasks**: HMA's threshold-based storage does not
  collapse on a cwe-style probe (rel. error ≤ 0.013 even at target
  frequency 64), avoiding the LSH-Top-K failure mode flagged by
  MagicPIG.
- **Adaptive KV quantization** driven by HMA hit count beats both
  uniform and random allocation at matched memory: PPL 26.4 at 3.9
  bits vs. uniform 4-bit's PPL 27.3 — a Pareto improvement.

See `figures/summary.png` for the combined comparison.

## Reproducing

```bash
# Activate the project's isolated venv (created by uv)
source .venv/bin/activate

# All five experiments run in ~4 minutes on a single A6000:
python src/exp1_synthetic.py
python src/exp2_pattern_stability.py
python src/exp3_perplexity.py
python src/exp5_aggregation_probe.py
python src/exp6_adaptive_quant.py

# Generate combined summary + statistics:
python src/analyze.py
```

Dependencies are tracked in `pyproject.toml`; install with:
```bash
uv sync
# (PyTorch built for the host CUDA: torch 2.5.1+cu121)
```

## File structure

```
src/
  hma.py                        # HashMap Attention primitive (dict-based)
  utils.py                      # seeding, dense-attention oracle, metrics
  exp1_synthetic.py             # synthetic Q/K stress test
  exp2_pattern_stability.py     # token-pattern stability in GPT-2
  exp3_perplexity.py            # PPL vs KV-budget on WikiText-2
  exp5_aggregation_probe.py     # cwe-style aggregation failure-mode probe
  exp6_adaptive_quant.py        # adaptive vs uniform KV quantization
  analyze.py                    # cross-experiment analysis + summary figure

results/                        # JSON outputs from each experiment
figures/                        # PNG figures

planning.md                     # research plan (Phase 0 + 1)
literature_review.md            # synthesis of related work (pre-gathered)
resources.md                    # catalog of papers / datasets / code (pre-gathered)
REPORT.md                       # full research report
papers/                         # downloaded PDFs of related work
datasets/                       # locally cached WikiText-2
code/                           # cloned reference implementations
```

## Caveats

- Single-pass evaluation: we measure quality with a mask-based forward
  pass, not wall-clock with a custom CUDA kernel. The user's noted
  systems-engineering challenge for HMA at frontier scale was
  explicitly out of scope.
- GPT-2-small (124M, head_dim=64, L=512–1024) is much smaller than
  frontier LLMs. Long-context behavior likely differs.
- Adaptive quantization gains are modest because GPT-2's KV cache is
  remarkably robust to uniform 6-bit; adaptive only helps in the 3–4
  bit regime.

See `REPORT.md` § 6 for full limitations.
