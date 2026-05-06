# HashMap Attention — Research Plan

## Motivation & Novelty Assessment

### Why This Research Matters
Attention is O(L²) in compute and O(L) in memory at long context. Most production
serving stacks at frontier scale already pay this cost twice: (a) prefill, where
all queries attend to all keys, and (b) decode, where every new token reads the
full KV cache. A method that reuses *prior* attention patterns instead of
recomputing scores could cut both costs subquadratically. The user-proposed
mechanism — store the keys that previous instances of a token attended to, and
fuzzily look those up at decode — also opens an adaptive-quantization axis
("frequently-attended keys deserve more bits"), which no published method
exploits.

### Gap in Existing Work
From `literature_review.md`:
- LSH-based attention (Reformer, MagicPIG, HashEvict, SMYRF) hashes the
  *current Q* against *current K*. None caches a token's historical attention
  pattern keyed by its identity.
- Routing Transformer / k-means cluster Q&K spatially but again do not exploit
  per-token-identity reuse across occurrences.
- KV-cache quantization (KIVI, KVQuant, KVTuner) uses *static* per-channel /
  per-token / per-layer policies. None drives the bit-width from observed
  attention frequency.
- MagicPIG showed LSH-Top-K is a biased estimator; aggregation tasks (RULER
  cwe/fwe) collapse. Sampling fixes this. Any HashMap variant must address
  the bias question.
- RetrievalAttention identified the OOD problem between Q and K — naïve ANN
  on K alone fails. Asymmetric / query-aware indexing is required.

### Our Novel Contribution
We test the user-proposed **HashMap Attention (HMA)** primitive: a
per-(layer, head, token-id) lookup table that stores *which key positions
previous occurrences of that token attended to above a threshold*. At decode,
the query for a new occurrence of token `t` only computes scores against
`keys[positions[t]] ∪ small_safety_set`. A fuzzy variant looks up via hashing
of the local n-gram context so that small edit-distance contexts reuse
patterns. We additionally test **HMA-Quant**: drive per-key bit-width from
hit frequency in the lookup table.

This is genuinely novel because the *index structure is keyed by token
identity (or n-gram hash), not by the query embedding*. It is post-hoc and
training-free, so applicable to any pretrained model.

### Experiment Justification
1. **E1 — Synthetic primitive validation.** Before any LLM, on synthetic
   Q/K matrices with known structure: how well does the per-token-identity
   lookup approximate full attention? What's the recall vs. budget trade-off?
   Why: the user's hypothesis hinges on whether attention patterns *are*
   token-conditioned. Without this, the rest is moot.
2. **E2 — Real-LLM token-pattern stability.** Run a small Llama on real
   text, log per-token-identity attention masks across occurrences, measure
   IoU/Jaccard. Why: validates the central assumption that attention reuse
   across occurrences of the same token is meaningful.
3. **E3 — Fuzzy n-gram lookup.** Extend identity → n-gram-context hash;
   measure lift. Why: tests whether SimHash-style fuzziness improves the
   primitive (the user's "small edit distance token sequences" idea).
4. **E4 — End-to-end perplexity / KV-budget trade-off** vs. baselines
   (full attention, StreamingLLM, H2O-style, random selection). Why: the
   actual user-facing question — does it work in a real LM?
5. **E5 — Aggregation failure-mode probe (RULER-cwe-style synthetic).**
   Build a small "common-word counting" probe to test whether HMA falls
   into the same Top-K bias trap as LSH-Top-K methods. Why: explicitly
   anticipated by literature; we must check.
6. **E6 — Adaptive KV quantization.** Use HMA hit-frequency to assign
   per-key bit-width; measure perplexity at fixed memory budget vs. uniform
   2-bit quantization. Why: the second arm of the user's hypothesis.

---

## Research Question
**Can a per-token-identity / fuzzy-n-gram lookup table of attention scores
built during prefill and updated during decode subquadratically approximate
full attention with acceptable quality, and can the same lookup drive
adaptive KV-cache quantization?**

## Background and Motivation
See `literature_review.md`. Closest baselines:
- **MagicPIG**: LSH-sampling, the SOTA LSH-attention paper.
- **HashEvict**: pre-attention LSH eviction.
- **H2O / StreamingLLM**: simple eviction baselines.
- **KIVI**: 2-bit KV-cache quantization companion for the second arm.

## Hypothesis Decomposition
- H1 (Pattern stability): Different occurrences of the same token in similar
  contexts attend to overlapping sets of key positions.
- H2 (Lookup primitive): A per-(token-id) (or n-gram hash) → key-position-set
  lookup, populated during prefill, can predict useful keys for the next
  occurrence.
- H3 (Subquadratic compute): With the lookup, decode-attention compute drops
  from O(L) per token to O(k) where k = lookup-budget << L.
- H4 (Quality): The approximation degrades gracefully with k; at k that gives
  meaningful subquadratic savings, quality (perplexity) is acceptable
  (within 1.0 PPL of full).
- H5 (Aggregation robustness): Using attention-weight thresholding (not
  just Top-K) avoids the MagicPIG-identified Top-K bias on aggregation.
- H6 (Adaptive quant): Bit-width allocation by hit frequency outperforms
  uniform low-bit quantization at matched memory.

## Proposed Methodology

### Approach
Build the lookup table primitive from scratch in PyTorch. Validate first on
**synthetic Q/K** (no model), then on **GPT-2-small** (CPU/GPU runnable,
gpt-2 has tied embeddings and is small enough to manipulate the attention
manually), then on **Llama-3.2-1B-Instruct** if HF download succeeds. Use
**WikiText-2** (already downloaded) for perplexity. Keep models small enough
to run in an afternoon on a single A6000.

The HashMap Attention primitive (per layer, per head):
- A dict `H : token_id → set[int]` of key positions where attention to this
  token's queries previously exceeded threshold τ.
- During prefill, compute full attention; for every (layer, head, query-token,
  key-position) where attention weight > τ, insert key-position into
  `H[token_id_of_query]`.
- During decode for token `t`, the candidate-key set is `H[t]` ∪
  always-on safety set (sliding window + sinks). Compute scores only on
  candidates. A "fuzzy" variant uses an n-gram (e.g., last 2 tokens) hash
  as the lookup key.
- For HMA-Quant: maintain hit-count per key position; allocate FP16 for
  top-q% hit positions, INT4/INT2 for the rest (KIVI-style symmetric).

### Experimental Steps
1. Implement the HMA primitive (dict-based, no fancy CUDA — correctness first).
2. Synthetic Q/K stress test: ground-truth attention vs. HMA approximation,
   sweep budget k, threshold τ, n-gram length.
3. Hook into GPT-2 via `output_attentions=True` to *observe* token-pattern
   stability on WikiText-2.
4. Manually re-implement masked attention forward pass for GPT-2 with
   HMA and measure the approximate perplexity. Compare to baselines:
   full, sliding-window-only, sinks+sliding, random budget-k.
5. Build a synthetic "common-word counting" probe to test the cwe/fwe
   failure mode at small scale.
6. Implement adaptive quantization layer (FP16 / INT4 / INT2 per key)
   driven by hit count; perplexity at matched memory.

### Baselines
- **Full attention** — accuracy ceiling.
- **StreamingLLM-style** (sinks=4 + sliding-window-w) — minimal sparsity.
- **H2O-style** (top-k accumulated attention) — eviction baseline.
- **Random-k** — sanity-check that the structure helps.
- **Quest-style page bound** — implementation-time-permitting, simplified.
- For the quant arm: **uniform INT4** (KIVI-like) at matched memory.

We deliberately do NOT try to reproduce MagicPIG/HashEvict end-to-end
because their codebases are large and the primitive comparison is more
informative at this scale.

### Evaluation Metrics
- **WikiText-2 perplexity** (validation set).
- **Top-K recall** vs. full attention oracle.
- **Attention output relative error** `‖ô − o‖_2 / ‖o‖_2`.
- **Effective KV budget** = avg # candidate keys / sequence length.
- **Aggregation probe accuracy** — synthetic counting task.
- **Memory footprint** (for quant experiments) — bytes per token in KV cache.

### Statistical Analysis Plan
- Synthetic experiments: report mean ± std over 5 seeds.
- LM perplexity: single-seed (deterministic forward pass) but report across
  multiple sequences; report median + IQR.
- Probe: report mean accuracy ± 95% CI (bootstrap, 1000 resamples).
- Compare HMA vs. each baseline at matched budget with paired t-test where
  applicable.

## Expected Outcomes
- H1/H2: We expect *moderate* token-pattern reuse — same token in similar
  positions attends similarly. Cross-context reuse will be weaker; the
  fuzzy n-gram lookup should help here.
- H3/H4: HMA should approximate full attention with k ≈ 10-20% of L on
  WikiText-2 with <1 PPL degradation, beating random-k handily.
- H5: We expect HMA-with-threshold (not strict Top-K) to handle the
  aggregation probe better than strict Top-K, validating MagicPIG's insight
  with our own setup.
- H6: Adaptive quant should beat uniform quant at matched memory by 0.1–
  0.5 PPL.

## Timeline
- Phase 0/1 planning: 30 min.
- Phase 2 env: 15 min.
- Phase 3 implementation: 90 min (the bulk of the work).
- Phase 4 experiments: 90 min.
- Phase 5 analysis & figures: 45 min.
- Phase 6 documentation: 30 min.
Total ≈ 5 hours.

## Potential Challenges
- **GPT-2's small head dim (64)** may be too noisy to see clean reuse.
  Mitigation: also test on a layer-dim-128 head if Llama-1B downloads.
- **WikiText-2 is short** — hard to show subquadratic savings dramatically.
  Mitigation: report relative improvements, use long passages by
  concatenation.
- **HF gated-model auth** for Llama. Mitigation: GPT-2 (open) is the
  primary model; Llama 3.2 1B is open but needs accept-license.
- **Implementing manual attention forward pass** has bug-surface area.
  Mitigation: validate against `model(...)` outputs first; only diverge
  for the HMA mask.

## Success Criteria
- Working HMA primitive that runs on synthetic data.
- Real-model evaluation (at minimum on GPT-2) showing the
  perplexity-vs-budget trade-off vs. ≥3 baselines.
- Aggregation probe result (positive or negative — both informative).
- Adaptive-quant comparison at matched memory budget.
- REPORT.md with all results, figures, and honest discussion.
