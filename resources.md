# Resources Catalog: HashMap Attention

## Summary
This document catalogs every paper, dataset, and code repository
gathered for the HashMap Attention research project. It is the
single index the experiment runner should consult first.

| Resource type | Count |
|---|---|
| Papers (PDFs) | 34 |
| Datasets (locally downloaded) | 1 (WikiText-2) |
| Datasets (documented for download) | 9 |
| Code repositories | 12 |

## Papers

All PDFs in `papers/`. Detailed descriptions and reading order in
`papers/README.md` and full synthesis in `literature_review.md`.

### Most directly relevant (Tier 1)

| Title | Year | File | Why relevant |
|---|---|---|---|
| Reformer: The Efficient Transformer | 2020 | `reformer_lsh_attention.pdf` | Foundational LSH attention. |
| MagicPIG: LSH Sampling for Efficient LLM Generation | 2024 | `magicpig_lsh_sampling.pdf` | **The paper to beat**. LSH on GPU+CPU; identifies failure mode of TopK. |
| HashEvict: Pre-Attention KV Cache Eviction via LSH | 2024 | `hashevict_lsh_kv.pdf` | Closest published implementation; pre-attention eviction by Hamming distance. |
| RetrievalAttention: Long-Context LLM via Vector Retrieval | 2024 | `retrieval_attention_anns.pdf` | Solves Q/K out-of-distribution problem any LSH-attention faces. |
| SMYRF: Asymmetric Clustering Attention | 2020 | `smyrf_asymmetric_clustering.pdf` | Drop-in LSH replacement for dense attention; no Q=K constraint, no retraining. |
| Scatterbrain: Sparse + Low-Rank Attention | 2021 | `scatterbrain_sparse_lowrank.pdf` | Hybrid LSH-sparse + low-rank. |
| KDEformer: Kernel Density Estimation Attention | 2023 | `kdeformer_kde_attention.pdf` | Theoretical view of LSH attention error. |
| LSH-Based Efficient Point Transformer | 2024 | `transformer_lsh_point.pdf` | LSH attention applied in real systems (HEP physics). |

### KV-cache sparsity / selection (Tier 2)

| Title | Year | File | Approach |
|---|---|---|---|
| Quest: Query-Aware Sparsity | 2024 | `quest_query_aware.pdf` | Page min/max → Top-K pages. Strong non-hashing baseline. |
| H2O: Heavy-Hitter Oracle | 2023 | `h2o_heavy_hitter.pdf` | Accumulated-attention eviction. |
| StreamingLLM: Attention Sinks | 2023 | `streamingllm_attention_sink.pdf` | Sliding window + attention sinks. |
| SnapKV | 2024 | `snapkv_kv_compression.pdf` | One-shot prune at decode start. |
| Loki: Low-Rank Keys | 2024 | `loki_low_rank_keys.pdf` | Score in low-d subspace. |
| SparQ Attention | 2023 | `sparq_bandwidth.pdf` | Channel-prune Q. |
| Dynamic Context Pruning | 2023 | `dynamic_context_pruning.pdf` | Learnable token dropping. |
| ShadowKV | 2024 | `shadowkv_kv_shadows.pdf` | Low-rank K + offloaded V. |
| LESS: Recurrence + KV Compression | 2024 | `less_recurrence_kv.pdf` | Constant recurrent state with eviction. |
| Multipole Attention | 2025 | `multipole_attention.pdf` | Cluster-based with Triton kernels. |
| Native Sparse Attention (NSA) | 2025 | `native_sparse_attention.pdf` | End-to-end trainable sparse attention. |

### KV-cache quantization (Tier 3)

| Title | Year | File | Why relevant |
|---|---|---|---|
| KIVI: 2-bit KV Quantization | 2024 | `kivi_kv_quantization.pdf` | Per-channel K, per-token V. Companion for the "adaptive quantization" half of the hypothesis. |
| KVQuant | 2024 | `kvquant_quantization.pdf` | Pre-RoPE key quantization, 10M context. |
| KVTuner: Mixed-Precision Layer-wise | 2025 | `kvtuner_mixed_precision.pdf` | Sensitivity-aware bit allocation. |

### Foundations & alternatives (Tier 4)

`attention_is_all_you_need.pdf`, `flashattention_v1.pdf`,
`flashattention_v2.pdf`, `efficient_transformers_survey.pdf`,
`longformer_sparse.pdf`, `bigbird_sparse.pdf`,
`performer_kernel_attention.pdf`, `linear_transformer.pdf`,
`kmeans_attention_routing.pdf`, `ecoformer_binary_attention.pdf`,
`hyperattention_near_linear.pdf`, `retro_retrieval_lm.pdf`.

See `papers/README.md` for full descriptions.

## Datasets

All downloaded data is in `datasets/` and excluded from git via
`datasets/.gitignore`. Small samples and instructions live in
`datasets/README.md`.

### Downloaded locally
| Name | Source | Size | Task | Location |
|---|---|---|---|---|
| WikiText-2 | HuggingFace | 7.8 MB | LM perplexity | `datasets/wikitext-2/` |

### Documented for on-demand download
| Name | Why | Size | Source |
|---|---|---|---|
| WikiText-103 | Stronger LM benchmark | ~520 MB | HF: `wikitext` `wikitext-103-raw-v1` |
| PG19 | Long-document LM | ~11 GB (stream) | HF: `emozilla/pg19` |
| LongBench | Multi-task long-context | ~1 GB | HF: `THUDM/LongBench` |
| RULER | Synthetic long-context (incl. cwe/fwe) | varies | github.com/NVIDIA/RULER |
| Needle-in-a-Haystack | Long-context retrieval | small | github.com/gkamradt/LLMTest_NeedleInAHaystack |
| InfiniteBench | Extreme long-context | ~5 GB | HF: `xinrongzhang2022/InfiniteBench` |
| GSM8K | Math reasoning | <100 MB | HF: `gsm8k` |
| TriviaQA | Open-domain QA | ~3 GB | HF: `trivia_qa` |
| enwik8 | Byte-level (Reformer eval) | ~100 MB | mattmahoney.net |

## Code repositories

All in `code/`. Detailed descriptions in `code/README.md`.

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| MagicPIG | github.com/Infini-AI-Lab/MagicPIG | LSH-sampling attention | `code/magicpig/` | **Reference impl for the paper to beat** |
| Cold Compress | github.com/AnswerDotAI/cold-compress | KV-cache compression toolkit | `code/cold-compress/` | **Primary integration target**; HashEvict is built as fork |
| Quest | github.com/mit-han-lab/Quest | Query-aware page selection | `code/quest/` | Non-hashing baseline; CUDA kernels |
| RetrievalAttention | github.com/microsoft/RetrievalAttention | ANN-based attention | `code/retrieval-attention/` | OOD-aware index |
| SMYRF | github.com/giannisdaras/smyrf | Asymmetric LSH attention | `code/smyrf/` | Drop-in for pretrained dense attention |
| H2O | github.com/FMInference/H2O | Heavy-hitter eviction | `code/h2o/` | Standard sparse-attention baseline |
| StreamingLLM | github.com/mit-han-lab/streaming-llm | Sinks + sliding window | `code/streaming-llm/` | Tiny, easy to read |
| KIVI | github.com/jy-yuan/KIVI | 2-bit KV quantization | `code/kivi/` | Drop-in HF `LlamaForCausalLM` |
| KVQuant | github.com/SqueezeAILab/KVQuant | Pre-RoPE key quant | `code/kvquant/` | Alternative quant baseline |
| Multipole Attention | github.com/SqueezeAILab/MultipoleAttention | Cluster-based | `code/multipole-attention/` | Triton kernels, 2025 |
| FlashAttention | github.com/Dao-AILab/flash-attention | IO-aware exact attention | `code/flash-attention/` | Dense-attention baseline |
| Reformer-PyTorch | github.com/lucidrains/reformer-pytorch | LSH attention | `code/reformer-pytorch/` | PyTorch port; CPU-runnable |

## Resource gathering notes

### Search strategy
Used the `paper-finder` service in *diligent* mode with five queries:
1. "sparse attention key value cache lookup table efficient transformer"
2. "fuzzy hashing locality sensitive hashing attention transformer"
3. "approximate nearest neighbor attention KV cache retrieval LLM"
4. "adaptive KV cache quantization mixed precision attention"
5. "subquadratic attention linear transformer kernel methods"

Total: 348 unique papers in `paper_search_results/`. Filtered to 58
high-relevance papers (relevance score ≥ 3), then to 34 papers
prioritized for download by closeness to the hypothesis (LSH and
KV-cache focus given priority over kernel/linear-attention).

### Selection criteria
- Direct overlap with the hypothesis (LSH, hash, retrieval, lookup).
- Recent state-of-the-art on long-context LLM inference.
- Foundational for the area (Reformer, FlashAttention, Vaswani 2017).
- Standard baselines an experiment must compare against.

### Challenges encountered
- Three papers (Q-Hitter, "Sparse Attention with Learning to Hash",
  "LSH for Long Context NMT") are not on arXiv and the
  Semantic-Scholar `openAccessPdf` field returned empty — likely
  conference-only releases (MLSys, ICLR, IWSLT). They are documented
  in the literature review by abstract; full text not retrieved.
  Coverage is unaffected because their core ideas appear in better-
  documented neighbors (HashEvict, MagicPIG, Reformer).
- Paper-finder was slow on diligent mode (~3 min per query). Two
  queries timed out and were re-run in fast mode. Pipe truncation
  via `head` killed one save mid-write; re-ran without truncation.
- Repository clones initially landed in workspace root because the
  shell `cd` only applied to the first command of a parallel batch;
  moved them to `code/` after.
- `trax` Reformer reference impl is 226 MB and mostly unrelated to
  Reformer specifically; deleted in favor of `lucidrains/reformer-
  pytorch` (110 MB, includes example notebooks but only the
  `reformer_pytorch/` subdirectory is needed).

### Gaps and workarounds
- **No SOTA frontier-scale model checkpoint** is downloaded. Llama-3.1-
  8B-Instruct, Llama-3-8B, Mistral-7B etc. are needed for actual
  benchmarks. Experiment runner must download via HF and they require
  HF authentication for gated models.
- **No GPU verified yet.** Most of the listed code requires CUDA. If
  the experiment runner is CPU-only, recommend starting with
  `smyrf` and `reformer-pytorch` on a small synthetic test.
- **RULER `cwe`/`fwe`** are synthesized on-the-fly by the NVIDIA
  RULER repo. Documented as a download-instruction rather than
  pre-built dataset.

## Recommendations for experiment design

### Primary dataset(s)
- **WikiText-2** (locally available) for development.
- **WikiText-103 + PG19** for perplexity measurements.
- **RULER (`niah_single`, `niah_multikey`, `cwe`, `fwe`)** is the
  must-have evaluation. Failing `cwe`/`fwe` is the documented
  failure mode of LSH-TopK methods (MagicPIG Fig. 1).
- **LongBench** (`qasper`, `multi_news`, `triviaqa`, `gov_report`)
  for downstream task quality.

### Baseline methods
1. **Full attention (FlashAttention-2)** — accuracy ceiling.
2. **StreamingLLM** — cheapest dynamic baseline.
3. **H2O** — eviction baseline.
4. **Quest** — strongest non-hashing query-aware baseline.
5. **MagicPIG** — LSH-sampling SOTA, the headline comparison.
6. **HashEvict** — closest published implementation to the hypothesis.

### Evaluation metrics
- Perplexity (WikiText-103, PG19).
- Attention output relative error at fixed lookup budget.
- Top-K recall against full-attention oracle.
- Task accuracy (RULER, LongBench).
- Wall-clock decoding latency at 32K / 96K / 128K context.
- Peak GPU memory.
- KV-cache compression ratio.

### Code to adapt/reuse
- **Cold Compress** (`code/cold-compress/`) is the primary
  integration target. Implement HashMap Attention as a new
  cache-strategy class. It already runs the eval harness end-to-end.
- **MagicPIG** (`code/magicpig/`) for the LSH primitive and the
  RULER eval scripts.
- **HashEvict** itself ships inside `cold-compress`'s strategy set.
- **KIVI** (`code/kivi/`) for the quantization layer of the
  hypothesis; provides a drop-in `LlamaForCausalLM` quantized model.

### Order of operations recommended
1. Set up HF token and download a small Llama-3.2-1B or Mistral-7B
   for fast iteration.
2. Validate the LSH/lookup primitive on synthetic Q/K (no model)
   against dense attention.
3. Plug into `cold-compress`; reproduce StreamingLLM and Heavy-Hitter
   numbers as a sanity check.
4. Implement HashMap Attention as a new strategy.
5. Evaluate on RULER `cwe`/`fwe` first — if it passes there, run
   LongBench + PG19 + decoding latency.
6. Layer KIVI quantization on top to test the adaptive-quantization
   half of the hypothesis.
