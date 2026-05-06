# Literature Review: HashMap Attention

## 1. Research area overview

The HashMap Attention hypothesis sits at the intersection of three active
research threads:

1. **Approximate / sparse attention** — replacing the O(L²) dot-product
   softmax with a subquadratic shortcut. The earliest line, going back
   to 2019–2021, includes Reformer, Longformer, BigBird, Performer,
   Linear Transformers, Routing Transformer, SMYRF, Scatterbrain.
2. **Dynamic KV-cache selection at inference time** — keep the KV cache
   small or load only a subset per query. The 2023–2025 wave: H2O,
   StreamingLLM, SnapKV, Quest, SparQ, Loki, RetrievalAttention,
   ShadowKV, MagicPIG, HashEvict, NSA.
3. **KV-cache quantization** — orthogonal compression in the bit-width
   axis. KIVI, KVQuant, KVTuner, Q-Hitter (sparse + quantized).

The hypothesis specifically proposes building a **lookup table of
attention scores** keyed by the current token (or a fuzzy match to it)
and only computing attention against the keys where previous instances
attended above threshold. That places it in cluster (1) — an attention
shortcut — but with two distinctive features:

- **Token-conditioned reuse**: the lookup table caches *behavior*, not
  just keys. If a previous instance of the same token attended to
  positions `{i, j, k}`, those positions become the sparse pattern for
  the next instance.
- **Fuzzy lookup**: small edit-distance token sequences should hit the
  same bucket. This is exactly an LSH-style problem.

The closest published systems are **HashEvict** (LSH on K vs. Q,
pre-attention eviction), **MagicPIG** (LSH sampling for unbiased
estimation), **Reformer** (LSH on shared Q=K), and **SMYRF**
(asymmetric LSH for Q≠K). None of them implement *exactly* the
proposed mechanism — caching per-token attention patterns from prior
occurrences — but their primitives are reusable.

## 2. Key papers

### 2.1 Reformer (Kitaev, Kaiser, Levskaya, ICLR 2020)
- **Contribution**: First LSH-based attention, replacing O(L²) with
  O(L log L). Uses random rotations + argmax bucketing as the LSH
  scheme.
- **Method**: Forces Q = K (shared linear projection). Hashes are
  computed on-the-fly per layer. Within a bucket, a chunked attention
  is performed; multiple hash rounds ensemble to reduce collision
  errors.
- **Datasets**: enwik8 (64K tokens), imagenet-64 generation (12K
  tokens), synthetic algorithmic tasks.
- **Result**: On par with full attention; large memory and speed
  savings for long sequences.
- **Limitations**: Q=K constraint forces retraining from scratch; the
  hashing/sorting overhead is non-trivial; performance sensitive to
  hash count. On modern decoder-only LLMs at inference, this scheme
  has not displaced full attention.
- **Code**: bundled in `trax`; PyTorch port at
  `lucidrains/reformer-pytorch` (in `code/reformer-pytorch/`).
- **Relevance**: foundation paper. Any HashMap Attention design must
  understand why Reformer-style LSH didn't win.

### 2.2 SMYRF (Daras, Odena, Kitaev, Dimakis, NeurIPS 2020)
- **Contribution**: Asymmetric LSH attention that does *not* require
  Q=K; works as a drop-in replacement for pretrained dense attention
  layers without retraining.
- **Method**: Defines asymmetric transformations
  `F(q) = [q; 0; √(M_Q² + M_K² − ‖q‖²)]`,
  `G(k) = [k; √(...); 0]`
  so that Euclidean distance in transformed space monotonically tracks
  the inner product. Combines with a balanced-cluster LSH so each
  bucket has equal queries and keys (essential for batched GPU
  execution).
- **Datasets**: GLUE (BERT/RoBERTa), IMDB, GAN training on CelebA-HQ.
- **Result**: SMYRF-BERT slightly *outperforms* BERT on GLUE while
  using 50% less memory; 99% of BERT's GLUE score with 75% less memory.
- **Why it matters here**: First to show that hash-based attention can
  work *post-hoc* on a pretrained model. This is the architectural
  pattern HashMap Attention will likely need (no retraining of frontier
  LLMs).
- **Code**: `code/smyrf/`.

### 2.3 MagicPIG (Chen, Sadhukhan et al., CMU + Meta, 2024)
- **Contribution**: LSH-sampling instead of LSH-TopK. Argues that TopK
  attention is a *biased* estimator and breaks on aggregation tasks
  (e.g. RULER's `cwe`/`fwe`); shows sampling proportional to attention
  weight is unbiased and empirically much better.
- **Method**: SimHash with (K, L) hyperparameters — L hash tables, K
  bits per hash. Hash functions on GPU, hash tables on CPU, sparse
  attention compute on CPU. Critically uses many hash tables (much
  larger than Reformer/Scatterbrain) for accurate sampling.
- **Datasets**: RULER (cwe, fwe, niah, multikey), LongBench, Loogle.
  Tests Llama-3.1-8B-Instruct.
- **Result**: 1.5–5× decode throughput, 54 ms latency on a single
  RTX 4090 with 96K context. **Outperforms TopK on aggregation tasks**.
- **Why it's the central reference**: this is the most rigorous,
  most recent demonstration that LSH-attention can be production-fast.
  It also identifies the trap (TopK bias) any HashMap Attention design
  must avoid.
- **Code**: `code/magicpig/`.

### 2.4 HashEvict (Liu, Rabbani et al., 2024)
- **Contribution**: Pre-attention KV-cache eviction using LSH. At each
  decode step, both the new query and all cached keys are hashed to
  c-bit binary codes; eviction is based on Hamming distance.
- **Method**: SimHash (`h(x) = sgn(Rx)` with random Gaussian R).
  Maintains a constant-size c×k binary array on GPU (c is hash dim,
  k is cache size). Decisions made *before* attention compute, unlike
  H2O which uses post-attention scores.
- **Datasets**: GSM8K (reasoning), MedQA, NIAH, Common Word retrieval,
  MultiNews, GovReport (summarization).
- **Result**: 30–70% KV cache compression with high accuracy; 1.5–2×
  prefill speedup; matches H2O on most tasks; 17× faster prefill than
  FastGen.
- **Why it's central**: closest published implementation to "fuzzy
  lookup of which keys to attend to". The HashEvict binary-array
  eviction structure is exactly the kind of hashmap the hypothesis
  proposes.
- **Code release**: as a fork of `cold-compress` (in `code/cold-compress/`).

### 2.5 RetrievalAttention (Liu, Chen et al., MSR, 2024)
- **Contribution**: Identifies and solves the **OOD problem** between
  query and key vectors when applying ANN search to attention.
- **Key insight**: Mahalanobis distance shows query vectors lie >10×
  farther from keys than queries lie from queries. Off-the-shelf ANN
  indexes (built only over keys) need to scan 30–50% of keys to find
  the true Top-K. Building the index *with awareness of the query
  distribution* drops that to 1–3%.
- **Datasets**: ∞-Bench, RULER, on Llama-3-8B (128K context).
- **Result**: 4.9× speedup over exact KNN, 1.98× over traditional ANN,
  same accuracy as full attention.
- **Why it's critical for HashMap Attention**: any LSH-based fuzzy
  lookup over K with Q as the probe will hit the same OOD problem.
  This paper's solution (asymmetric query-aware index) is essentially
  what SMYRF discovered for LSH but in the modern ANN-graph setting.

### 2.6 Quest (Tang et al., MIT, ICML 2024)
- **Contribution**: Page-granularity Top-K selection using cheap min/max
  bounds per page.
- **Method**: KV cache stored in pages (PagedAttention layout). For
  each page, track min/max of each Key dimension. Compute per-page
  upper-bound score from the current Q, pick Top-K pages, run dense
  attention only inside them.
- **Datasets**: PG19 (perplexity), Passkey retrieval, LongBench.
- **Result**: 7.03× attention speedup, 2.23× end-to-end on 32K
  context. **Better than H2O and post-hoc methods at the same budget**.
- **Why it's central as a baseline**: simpler than hashing, often
  competitive. A HashMap Attention prototype must justify itself
  against Quest specifically.

### 2.7 Native Sparse Attention (DeepSeek, 2025)
- **Contribution**: Three-branch (compressed / selected / sliding)
  attention, *natively trainable* end-to-end, with hardware-aligned
  Triton kernels.
- **Critique relevant to HashMap Attention**: explicitly calls out
  the discrete selection in MagicPIG (SimHash) and ClusterKV
  (k-means) as **non-differentiable**, blocking gradient flow during
  training. This means any HashMap Attention method that uses
  hard hashing will be limited to inference-time deployment unless
  paired with a relaxation.
- **Result**: 11.6× decode, 9.0× forward, 6.0× backward speedup at
  64K context, while *matching or exceeding* Full Attention quality
  when pretrained natively.
- **Implication**: HashMap Attention as a pure inference-time shortcut
  is fine; HashMap Attention as a *trainable* mechanism needs careful
  design (continuous relaxation of the lookup, or straight-through
  estimators).

### 2.8 Quantization companion: KIVI (Liu, Yuan et al., ICML 2024)
- **Contribution**: 2-bit KV quantization with asymmetric scheme:
  per-channel for keys (because outlier channels persist in K),
  per-token for values (because V is mixed by attention output).
- **Result**: 2.6× memory reduction, ~4× larger batches, 2.35–3.47×
  throughput, near-zero accuracy drop.
- **Relevance**: the hypothesis proposes the lookup table itself
  could enable adaptive KV quantization. KIVI's per-channel/per-token
  decomposition is the design pattern to extend: a HashMap Attention
  could store frequently-attended keys at higher precision and
  rarely-attended keys at lower precision.

### 2.9 Other notable papers
- **H2O** (Zhang et al., 2023) — heavy-hitter eviction; canonical
  baseline.
- **StreamingLLM** (Xiao et al., 2023) — discovers attention sinks;
  any approximation must preserve them.
- **Loki** (Singhania et al., 2024) — keys lie in low-rank subspace;
  rank tokens cheaply in that subspace.
- **SparQ** (Ribar et al., 2023) — channel-prune Q for cheap
  approximate scores.
- **ShadowKV** (Sun et al., 2024) — low-rank K + offloaded V;
  reconstruction-based sparse selection.
- **Multipole Attention** (Hooper et al., 2025) — clustering-based
  approximation, online cluster updates.
- **Routing Transformer** (Roy et al., 2021) — k-means clustering of
  Q/K; closest non-LSH analog of HashMap Attention.
- **Performer / Linear Transformer** — kernel-based linear attention;
  fundamentally different mechanism but a baseline for "subquadratic
  attention".
- **FlashAttention 1/2** (Dao et al.) — IO-aware exact attention;
  orthogonal optimization, often combined with sparse methods.

## 3. Common methodologies

| Method family | Used by | Core trick |
|---|---|---|
| **Random-projection LSH (SimHash / angular)** | Reformer, MagicPIG, HashEvict, SMYRF, KDEformer | `h(x) = sgn(Rx)` → Hamming distance ≈ angular distance |
| **Asymmetric LSH for Q≠K** | SMYRF, RetrievalAttention | Augment Q and K with extra dims so Euclidean distance tracks inner product |
| **Page/block min-max bounds** | Quest, NSA | Per-block summary stats let cheap upper bounds drive selection |
| **k-means / clustering** | Routing Transformer, ClusterKV, Multipole | Partition keys by centroid; attend within cluster + use centroid as proxy |
| **Heavy-hitter eviction** | H2O, Scissorhands | Track accumulated attention scores; evict low-score tokens |
| **Sliding window + sinks** | StreamingLLM, Longformer | Static local pattern + retain initial tokens |
| **Channel pruning of Q** | SparQ, Loki | Compute approximate scores using a subset of feature dims |
| **Low-rank approximation** | Performer, Linear Transformer, Scatterbrain | Replace softmax with separable kernel features |
| **Per-channel key quantization** | KIVI, KVQuant | Outlier channels in K demand channel-wise grouping |

## 4. Standard baselines an experiment must beat

For a **post-hoc, training-free** HashMap Attention method (most likely
target given systems-cost concerns):
1. **Full attention** (FlashAttention-2) — accuracy ceiling, latency
   floor at long context.
2. **StreamingLLM** — cheapest "approximation" with attention sinks +
   sliding window.
3. **H2O** — heavy-hitter eviction, the simplest dynamic baseline.
4. **Quest** — page-level Top-K, the strongest non-hashing baseline.
5. **MagicPIG** — strongest LSH-attention baseline. *The* paper to beat.
6. **HashEvict** — strongest pre-attention LSH-eviction baseline.

For a **trained-from-scratch** variant:
1. Full attention pretraining (matched compute).
2. Native Sparse Attention (DeepSeek 2025) — trained sparse attention.
3. Reformer (LSH from scratch).

## 5. Standard datasets / benchmarks

| Benchmark | What it measures | Used by |
|---|---|---|
| **WikiText-103** | Short-context language modeling perplexity | Most efficient-attention papers |
| **PG19** | Long-context (10K+) book LM perplexity | Quest, RetrievalAttention, Compressive Transformer |
| **enwik8** | Byte-level (64K) character LM | Reformer |
| **Needle-in-a-Haystack (NIAH)** | Long-context retrieval | StreamingLLM, MagicPIG, RetrievalAttention |
| **PassKey retrieval** | Synthetic long-context | Quest, KIVI |
| **LongBench** (Bai et al.) | Multi-task long-context (QA, summ., few-shot, code) | KIVI, Quest, MagicPIG, HashEvict |
| **RULER** (Hsieh et al.) | Synthetic long-context with NIAH + aggregation (`cwe`, `fwe`) | MagicPIG, ShadowKV |
| **InfiniteBench / ∞-Bench** | 100K–1M token retrieval & reasoning | RetrievalAttention |
| **GSM8K** | Math reasoning | HashEvict (free-response reasoning eval) |
| **MultiNews / GovReport** | Long summarization | HashEvict, LongBench subtasks |
| **MMLU** | Multi-domain knowledge | KIVI |

**For HashMap Attention specifically**: the experiments must include a
RULER-style aggregation task, because that's where TopK-style LSH is
known to fail (MagicPIG Fig. 1). Passing only NIAH and perplexity is
not enough.

## 6. Standard evaluation metrics

- **Perplexity** (PPL) — gold standard for LM approximation.
- **Top-K recall** — how often the approximation selects the same
  tokens as full attention's true top-K (Quest Fig. 4 style).
- **Attention output relative error** `‖ô − o‖ / ‖o‖` — direct measure
  of approximation quality (MagicPIG Fig. 3 style).
- **Task accuracy** — for downstream benchmarks (GSM8K, LongBench, etc.).
- **Decoding latency** (ms/token) — wall-clock at given context length.
- **Throughput** (tok/s) at given batch size.
- **Peak GPU memory** — especially for long context + KV cache.
- **KV cache compression ratio** — for the storage axis.

## 7. Gaps and opportunities (what's uniquely novel about HashMap Attention)

The hypothesis as stated proposes something **not yet in the
literature**:

1. **Per-token-identity attention reuse**. All current methods
   (HashEvict, MagicPIG, Reformer) hash *the current query embedding*
   against *the current keys*. They don't cache a token's historical
   attention pattern keyed by its identity. The idea is closest to
   *retrieval-augmented LM* (RETRO) but applied at the attention-
   pattern level rather than the document level.

2. **Fuzzy lookup over token n-grams**. The hypothesis mentions "small
   edit distance token sequences." No published work hashes n-gram
   contexts → attention patterns. There's a clear bridge to data-
   structure work (suffix arrays, Bloom filters, MinHash) that could
   be borrowed.

3. **Adaptive quantization driven by attention frequency**. KIVI/
   KVQuant use static per-channel/per-token policies. Using observed
   attention frequency to decide bit-width per key vector is novel.

**Risks the literature surfaces:**

1. **OOD problem** (RetrievalAttention): naive ANN/LSH between Q and K
   doesn't work without query-distribution-aware indexing.
2. **Aggregation tasks fail under TopK** (MagicPIG): if the lookup
   table is interpreted as TopK selection, expect failure on `cwe`/
   `fwe`. Sampling is the fix.
3. **Non-differentiability** (NSA): hash-based selection blocks
   gradient flow, restricting training-time use.
4. **Attention-sink robustness** (StreamingLLM): any selection scheme
   must preserve the first ~4 tokens of the sequence or accuracy
   collapses.
5. **Systems cost on accelerators** (the hypothesis itself): hash
   table operations (lookups, insertions, fuzzy matching) are not
   what GPU/TPU hardware excels at. MagicPIG's solution — host the
   table on CPU — is the most promising precedent.

## 8. Recommendations for our experiment

### Recommended datasets
- **WikiText-2** (already downloaded) for fast iteration on the LSH
  primitive and perplexity sanity-checks.
- **WikiText-103** for stronger LM evaluation.
- **PG19** subset (streaming) for genuine long-context perplexity.
- **RULER** (`niah_single_1`, `niah_multikey`, **`cwe`**, **`fwe`**)
  — the **must-have** benchmark; failing here is the failure mode
  prior LSH-TopK methods exhibit.
- **PassKey** + **NIAH** for long-context retrieval.
- **LongBench** (a few subtasks: `qasper`, `multi_news`, `triviaqa`)
  for downstream task quality.

### Recommended baselines
1. Full attention with FlashAttention-2 (accuracy + latency reference).
2. **StreamingLLM** — minimal sparsity baseline.
3. **H2O** — eviction baseline.
4. **Quest** — page-level Top-K, the strongest non-hashing baseline.
5. **MagicPIG** — LSH attention SOTA; **the paper to beat**.
6. **HashEvict** — pre-attention LSH eviction; closest implementation.

### Recommended metrics
- Perplexity (WikiText-103, PG19).
- Per-task accuracy (RULER, LongBench, PassKey).
- **Attention output relative error** at fixed lookup budget — direct
  measure of the hashing primitive's quality.
- Decoding latency at 32K, 96K, 128K context.
- Peak GPU memory.

### Methodological considerations
1. **Start synthetic**. Before plugging into a real LLM, validate the
   lookup primitive on synthetic Q/K matrices: build the hash
   table from prefill, query during decode, compare to dense
   attention on the same Q/K.
2. **Test the failure mode early**. Run RULER `cwe`/`fwe` (or
   reproductions thereof) before claiming success on NIAH. The
   MagicPIG paper makes this concrete: a method can pass NIAH at 0.1%
   density but collapse on `cwe` even at 10% density.
3. **Address OOD up-front**. Either copy RetrievalAttention's
   query-aware index construction or follow SMYRF's asymmetric
   transformation. Naïve symmetric LSH between Q and K will be
   poor.
4. **Preserve attention sinks**. Keep the first ~4 tokens always
   present in the lookup, regardless of hash decisions.
5. **Build inside `cold-compress`** if a HashEvict-style fork is
   the target. It already implements the eval harness and several
   baselines (H2O, StreamingLLM, sliding-window, FastGen).
6. **For the quantization side of the hypothesis**, layer KIVI or
   KVQuant on top of the sparse attention. The simplest "adaptive"
   policy: keys that are frequently retrieved get FP16, the rest
   INT2.
