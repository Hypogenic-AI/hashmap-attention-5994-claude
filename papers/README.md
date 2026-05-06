# Downloaded Papers

34 papers covering the literature relevant to **HashMap Attention**: hashing-
based attention, KV-cache sparsity/compression, retrieval-augmented attention,
linear/kernel attention, and KV-cache quantization.

Papers are grouped by relevance tier. Within each tier they are ordered
roughly by importance to the hypothesis.

---

## Tier 1 — directly tests the same idea (hash/retrieval-based attention shortcut)

These papers either build a lookup-style attention shortcut, use LSH for
attention, or replace dot-product with ANN search. The HashMap Attention
hypothesis is closest to this body of work.

| File | Paper | Year | Why it's central |
|---|---|---|---|
| `reformer_lsh_attention.pdf` | Reformer: The Efficient Transformer (Kitaev, Kaiser, Levskaya) | 2020 | First LSH attention. Q=K (shared), random rotations → buckets, attend within bucket. O(L log L). Requires shared-QK and retraining. |
| `magicpig_lsh_sampling.pdf` | MagicPIG: LSH Sampling for Efficient LLM Generation (Chen et al.) | 2024 | **Most directly relevant**: GPU computes hashes, CPU stores hash tables and runs sparse attention. Argues TopK selection is biased; sampling via LSH gives unbiased estimator. 5× decode throughput. |
| `hashevict_lsh_kv.pdf` | HashEvict: Pre-Attention KV Cache Eviction via LSH (Liu, Rabbani et al.) | 2024 | **Most directly relevant**: at every decode step, projects Q and cached K to c-bit binary codes, evicts cached token with highest Hamming distance. 30–70% cache reduction. |
| `retrieval_attention_anns.pdf` | RetrievalAttention: ANN-Search KV (Liu et al., MSR) | 2024 | Builds ANN index over keys on CPU, queries it during decode. Identifies and solves OOD problem between Q and K distributions (key insight for any HashMap-Attention design). |
| `smyrf_asymmetric_clustering.pdf` | SMYRF: Asymmetric Clustering Attention (Daras, Odena, Kitaev, Dimakis) | 2020 | Asymmetric LSH for MIPS over Q≠K, balanced buckets. Drop-in replacement for dense attention without retraining. Closest known prior art for fuzzy lookup over an existing model. |
| `scatterbrain_sparse_lowrank.pdf` | Scatterbrain: Sparse + Low-Rank Approximation (Chen et al.) | 2021 | Combines LSH-based sparse attention with low-rank attention. Useful baseline for hybrid approaches. |
| `kdeformer_kde_attention.pdf` | KDEformer: Kernel Density Estimation Attention (Zandieh et al.) | 2023 | Hashing for KDE, then uses it to estimate softmax denominator. Theoretical perspective on LSH attention error. |
| `transformer_lsh_point.pdf` | LSH-Based Efficient Point Transformer (HEP physics, 2024) | 2024 | LSH attention applied to point clouds. Real-world deployment of the LSH approach. |

---

## Tier 2 — KV-cache sparsity / token selection (the systems context)

These methods don't use hashing per se but solve the same downstream
problem: avoid attending to all keys. They're the baselines a HashMap
Attention method must beat.

| File | Paper | Year | Approach |
|---|---|---|---|
| `quest_query_aware.pdf` | Quest: Query-Aware Sparsity (Tang et al., MIT) | 2024 | Per-page min/max key metadata; estimate page criticality from current Q; load Top-K pages. 7× attention speedup. |
| `h2o_heavy_hitter.pdf` | H2O: Heavy-Hitter Oracle (Zhang et al.) | 2023 | Identifies "heavy hitter" tokens using accumulated attention scores; evicts the rest. Strong eviction baseline. |
| `streamingllm_attention_sink.pdf` | StreamingLLM: Attention Sinks (Xiao et al.) | 2023 | Discovers that initial tokens act as "attention sink"; keeping them + sliding window enables infinite-stream decoding. Critical phenomenon for any attention-shortcut method. |
| `snapkv_kv_compression.pdf` | SnapKV (Li et al.) | 2024 | Pre-fills attention pattern with last query window, prunes once before decode. |
| `loki_low_rank_keys.pdf` | Loki: Low-Rank Keys (Singhania et al.) | 2024 | Keys lie in low-d subspace; rank tokens via attention scores in that low-d space. Cheap selection. |
| `sparq_bandwidth.pdf` | SparQ Attention (Ribar et al.) | 2023 | Channel-pruning Q to compute approximate scores cheaply. |
| `dynamic_context_pruning.pdf` | Dynamic Context Pruning (Anagnostidis et al.) | 2023 | Learnable mechanism prunes uninformative tokens during generation. |
| `shadowkv_kv_shadows.pdf` | ShadowKV: KV in Shadows (Sun et al., ByteDance) | 2024 | Stores low-rank keys, offloads values; reconstructs sparse KV on the fly. 6× larger batch, 3× throughput. |
| `less_recurrence_kv.pdf` | LESS: Recurrence + KV Cache Compression (Dong et al.) | 2024 | Constant-size recurrent state alongside eviction; addresses information loss from pure eviction. |
| `multipole_attention.pdf` | Multipole Attention (Hooper et al.) | 2025 | Cluster keys; cluster centroids identify important keys + approximate the rest. Fast online re-clustering. |
| `native_sparse_attention.pdf` | NSA: Natively Trainable Sparse Attention (DeepSeek) | 2025 | Three-branch (compressed / selected / sliding) end-to-end trainable sparse attention. Critiques MagicPIG/ClusterKV's non-differentiable selection. |

---

## Tier 3 — KV-cache quantization (orthogonal but mentioned in hypothesis)

The hypothesis explicitly mentions adaptive KV quantization as a side effect
of the lookup table. These papers establish the state of the art.

| File | Paper | Year | Approach |
|---|---|---|---|
| `kivi_kv_quantization.pdf` | KIVI: Asymmetric 2-bit KV Quantization (Liu et al.) | 2024 | Per-channel for keys (outlier channels), per-token for values. 2.6× memory reduction with little accuracy loss. |
| `kvquant_quantization.pdf` | KVQuant (Hooper et al.) | 2024 | Pre-RoPE key quantization, dense outlier channels. Extends to 10M context. |
| `kvtuner_mixed_precision.pdf` | KVTuner: Mixed-Precision Layer-wise (Li et al.) | 2025 | Sensitivity-aware bit allocation per layer. |

---

## Tier 4 — foundations & alternatives (must-cite for any attention paper)

| File | Paper | Year | Role |
|---|---|---|---|
| `attention_is_all_you_need.pdf` | Attention Is All You Need (Vaswani et al.) | 2017 | Foundational; defines softmax(QK/√d)V. |
| `flashattention_v1.pdf` | FlashAttention (Dao et al.) | 2022 | IO-aware exact attention. Often cited as orthogonal to approximation. |
| `flashattention_v2.pdf` | FlashAttention-2 (Dao) | 2023 | Improved parallelism, hardware utilization. |
| `efficient_transformers_survey.pdf` | Efficient Transformers: A Survey (Tay et al.) | 2020 | Comprehensive taxonomy of efficient-attention methods. Helpful map. |
| `longformer_sparse.pdf` | Longformer (Beltagy et al.) | 2020 | Static sparse attention (sliding window + global). |
| `bigbird_sparse.pdf` | Big Bird (Zaheer et al.) | 2020 | Random + sliding + global attention; theoretical guarantees. |
| `performer_kernel_attention.pdf` | Performer: Rethinking Attention with Performers (Choromanski et al.) | 2020 | FAVOR+ random feature kernel attention; linear complexity. |
| `linear_transformer.pdf` | Transformers are RNNs (Katharopoulos et al.) | 2020 | Kernel attention as RNN; linear time/memory. |
| `kmeans_attention_routing.pdf` | Routing Transformer (Roy et al.) | 2021 | k-means clustering of queries/keys, attend within cluster. Closest non-LSH analog of HashMap Attention. |
| `ecoformer_binary_attention.pdf` | EcoFormer (Liu et al.) | 2022 | Binary attention via kernel approximation. |
| `hyperattention_near_linear.pdf` | HyperAttention: Near-Linear (Han et al.) | 2023 | Theoretical result on near-linear attention via sortLSH. |
| `retro_retrieval_lm.pdf` | RETRO: Retrieval LM (Borgeaud et al., DeepMind) | 2021 | External KV retrieval at LM scale. Different problem (retrieval-augmented LM, not attention shortcut) but same conceptual territory. |

---

## Reading order for the experiment runner

If short on time, read in this order:

1. **HashEvict** — closest implementation to the hypothesis (LSH on K vs Q,
   pre-attention decisions). Get the Hamming-distance scoring code idea.
2. **MagicPIG** — most rigorous treatment of LSH attention quality.
   Crucially shows TopK is biased and **fails** on aggregation tasks
   (`cwe`, `fwe`) — this is the failure mode the experiment must test for.
3. **Reformer** — classic LSH attention; understand the Q=K constraint.
4. **SMYRF** — asymmetric LSH that *doesn't* require Q=K and is drop-in.
   Likely the closest viable starting architecture.
5. **RetrievalAttention** — the OOD problem between Q and K distributions
   is real and breaks naïve ANN/LSH; this paper's solution is a must-read
   before designing the lookup index.
6. **Quest** — page-granularity selection with min/max key bounds; an
   alternative to hashing that's simpler and may be a strong baseline.
7. **Native Sparse Attention** — calls out the non-differentiability
   problem of LSH/cluster selection. If experiments train from scratch,
   this is the design constraint to beat.
8. **KIVI** — for the quantization side of the hypothesis.

The remaining papers fill in baselines and historical context.
