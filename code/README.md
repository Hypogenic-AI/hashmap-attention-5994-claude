# Cloned Code Repositories

12 repositories that implement directly relevant baselines or supporting
infrastructure for HashMap Attention research.

## Tier 1 — implementations of the closest prior art

### `magicpig/` — LSH-sampling attention (CMU/Meta)
- **URL**: https://github.com/Infini-AI-Lab/MagicPIG
- **Paper**: arXiv:2410.16179 (`papers/magicpig_lsh_sampling.pdf`)
- **What it provides**: GPU-CPU heterogeneous LSH-sampling attention.
  Hash tables stored on CPU, hash functions computed on GPU, sparse
  attention runs on CPU. Targets long-context decode throughput.
- **Why it matters**: Most rigorous, most recent LSH-attention system.
  Empirically demonstrates the failure mode of plain TopK selection
  (paper's Figure 1) — **the experiment plan must replicate this on
  RULER cwe/fwe before claiming success**.
- **Key entry points**:
  - `library/` — CPU sparse attention kernels
  - `models/` — Llama wrappers integrating LSH sampling
  - `evaluations/` — RULER, NIAH, Loogle benchmarks
- **Install needs**: PyTorch, FlashInfer, CMake, custom CPU kernel build.
- **Likely use**: starting point for the LSH primitive; reference
  implementation to validate any from-scratch hash-attention build.

### `cold-compress/` — KV-cache compression toolkit (Answer.AI)
- **URL**: https://github.com/AnswerDotAI/cold-compress
- **What it provides**: Pluggable KV-cache compression strategies on top of
  GPT-Fast. Implements Heavy-Hitters, Recent Tokens, Global Tokens, FastGen,
  Pyramid, Local-Global. Designed to be extended.
- **Why it matters**: HashEvict (Liu, Rabbani et al., 2024) explicitly
  releases as a fork of this library. Best framework to add a HashMap
  Attention variant as a new `cache.py` strategy.
- **Key files**:
  - `cache.py` — base KV-cache class; subclass to add new strategy
  - `attention_utils.py` — attention masking & selection helpers
  - `eval.py`, `eval_multi.py` — LongBench / NIAH / RULER evaluation
- **Likely use**: **primary integration target** for a HashMap Attention
  prototype. Implement as a new cache strategy and reuse the existing eval.

### `quest/` — Query-Aware Sparsity (MIT)
- **URL**: https://github.com/mit-han-lab/Quest
- **Paper**: `papers/quest_query_aware.pdf`
- **What it provides**: Page-granularity Top-K KV selection. Custom
  CUDA kernels in `kernels/` for the page criticality estimation.
- **Why it matters**: Strongest non-hashing baseline for query-aware
  KV selection. HashMap Attention should beat or match it.
- **Install**: CMake-based, requires CUDA toolchain. Pre-built kernels
  in `kernels/3rdparty/`.

### `retrieval-attention/` — ANN-based attention (Microsoft)
- **URL**: https://github.com/microsoft/RetrievalAttention
- **Paper**: `papers/retrieval_attention_anns.pdf`
- **What it provides**: Attention-aware ANN index. Solves the Q-K
  distribution mismatch (OOD) that plain ANN/LSH suffers.
- **Why it matters**: If a HashMap Attention prototype uses fuzzy lookup,
  it will hit the same OOD problem. This repo's approach (training a
  query-distribution-aware index) is the most thorough fix in the
  literature.

### `smyrf/` — Asymmetric LSH attention (UT Austin / Google)
- **URL**: https://github.com/giannisdaras/smyrf
- **Paper**: `papers/smyrf_asymmetric_clustering.pdf`
- **What it provides**: Drop-in replacement for dense attention using
  asymmetric LSH transformations on Q and K (so Q≠K is fine).
- **Why it matters**: Reformer requires Q=K, which is unusable for
  pretrained models. SMYRF's asymmetric trick is the cleanest path to
  retrofitting an existing checkpoint with a hash-based attention.

---

## Tier 2 — KV-cache eviction & compression baselines

### `h2o/`
- **URL**: https://github.com/FMInference/H2O
- **Paper**: `papers/h2o_heavy_hitter.pdf`
- **What it provides**: Heavy-Hitter Oracle eviction, with HuggingFace
  (`h2o_hf/`) and FlexGen (`h2o_flexgen/`) integrations.
- **Use**: standard sparse-attention baseline.

### `streaming-llm/`
- **URL**: https://github.com/mit-han-lab/streaming-llm
- **Paper**: `papers/streamingllm_attention_sink.pdf`
- **What it provides**: Sliding-window + attention-sink streaming
  inference. Tiny code, easy to read.
- **Use**: must-include baseline; demonstrates the attention-sink
  phenomenon any HashMap Attention design must respect.

### `kivi/`
- **URL**: https://github.com/jy-yuan/KIVI
- **Paper**: `papers/kivi_kv_quantization.pdf`
- **What it provides**: 2-bit per-channel-key / per-token-value
  quantization. Drop-in replacement for HF `LlamaForCausalLM`.
- **Use**: orthogonal — combine with sparse attention for the
  "adaptive quantization" half of the hypothesis.

### `kvquant/`
- **URL**: https://github.com/SqueezeAILab/KVQuant
- **What it provides**: Pre-RoPE key quantization with dense outlier
  channels. Source code + pre-quantized model checkpoints.
- **Use**: alternative quantization baseline.

### `multipole-attention/`
- **URL**: https://github.com/SqueezeAILab/MultipoleAttention
- **Paper**: `papers/multipole_attention.pdf`
- **What it provides**: Cluster-based approximate attention with
  Triton kernels. Most recent (2025) clustering-attention work.
- **Use**: alternative to LSH (clustering rather than hashing).

---

## Tier 3 — foundations / orthogonal optimizations

### `flash-attention/`
- **URL**: https://github.com/Dao-AILab/flash-attention
- **What it provides**: IO-aware exact attention; Triton/CUDA kernels.
- **Use**: dense baseline for wall-clock comparisons; building block
  if a HashMap Attention prototype materializes a sparse block.

### `reformer-pytorch/`
- **URL**: https://github.com/lucidrains/reformer-pytorch
- **Paper**: `papers/reformer_lsh_attention.pdf`
- **What it provides**: PyTorch implementation of LSH attention with
  the Q=K shared-projection constraint. Pure-Python, easy to read.
- **Use**: reference for understanding the LSH attention math; not
  suitable for production but excellent for prototyping.

---

## Validation status

I did **not** install or run any of these repos — most require GPU
hardware, CUDA toolchains, large model checkpoints (Llama-3.1-8B+),
and long downloads. Each requirement is documented above so the
experiment runner can pick the right starting point.

If running on CPU or constrained hardware:
- `reformer-pytorch` and `smyrf` are pure PyTorch and run on CPU.
- `cold-compress` runs on CPU with smaller models for testing.
- `kivi` has a CPU example mode in `example.py`.
- The rest essentially require GPU.
