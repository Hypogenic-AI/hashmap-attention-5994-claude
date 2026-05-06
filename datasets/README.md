# Datasets for HashMap Attention Research

Data files are NOT committed to git (see .gitignore). This document describes
each dataset, how to obtain it, and how to load it.

## Dataset selection rationale

The hypothesis "HashMap Attention" requires evaluating an approximate-attention
method along three axes:

1. **Language modeling perplexity** on long sequences — does the approximation
   keep the model's predictions accurate? Standard for any attention-replacement
   work (Reformer, Performer, SMYRF). Use **WikiText** and **PG19**.
2. **Long-context retrieval & aggregation** — the regime where dense attention
   matters most and where lookup-based shortcuts are most likely to fail.
   Use **Needle-in-a-Haystack**, **PassKey**, **RULER**, **LongBench**.
3. **Reasoning / generation quality** at moderate context — sanity-check that
   the approximation does not silently degrade short-context tasks.
   Use **GSM8K**, **TriviaQA**, **MMLU** (subsets).

The MagicPIG paper (arXiv:2410.16179) showed that TopK-style sparse attention
can pass `niah` retrieval but fail aggregation tasks (`cwe`, `fwe`). Any
HashMap-Attention experiment must therefore evaluate on RULER or comparable
aggregation benchmarks, not only retrieval.

---

## Available locally

### WikiText-2 (small, included for prototyping)
- **Location**: `datasets/wikitext-2/`
- **Size**: 7.8 MB on disk
- **Splits**: train (36,718 lines), validation (3,760), test (4,358)
- **Use**: quick prototyping of perplexity changes from approximation
- **Load**:
  ```python
  from datasets import load_from_disk
  ds = load_from_disk("datasets/wikitext-2")
  ```

---

## Datasets to download as needed

### WikiText-103 (full LM benchmark)
- **Why**: standard perplexity benchmark, larger than WikiText-2, used widely
  in efficient-attention papers.
- **Download**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("wikitext", "wikitext-103-raw-v1")
  ds.save_to_disk("datasets/wikitext-103")
  ```
- **Size**: ~520 MB on disk.

### PG19 (long-document language modeling)
- **Why**: long-form (book) language modeling, sequences of 10K+ tokens.
  Used by Quest, Compressive Transformers, and most long-context papers.
- **Download**:
  ```python
  from datasets import load_dataset
  # Streaming recommended due to size
  ds = load_dataset("emozilla/pg19", split="test", streaming=True)
  ```
- **Size**: ~11 GB if fully materialized. Use streaming for evaluation.

### LongBench (Bai et al., 2023)
- **Why**: multi-task long-context benchmark covering QA, summarization,
  few-shot learning, and code completion. Used by KIVI, Quest, MagicPIG.
- **Download**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("THUDM/LongBench", "qasper")  # or other subtasks
  ds.save_to_disk("datasets/longbench")
  ```
- **Size**: ~1 GB total, varies per subtask.

### RULER (Hsieh et al., 2024)
- **Why**: synthetic long-context evaluation with controllable length and
  diverse task types (NIAH retrieval, multi-key aggregation, common/frequent
  word extraction). MagicPIG specifically uses `cwe` and `fwe` subtasks,
  which are the failure modes of TopK attention.
- **Repo**: https://github.com/NVIDIA/RULER
- **Download** (synthesize on the fly):
  ```bash
  git clone https://github.com/NVIDIA/RULER.git datasets/RULER
  cd datasets/RULER && pip install -r requirements.txt
  bash scripts/data/synthetic.sh  # see repo for details
  ```

### Needle-in-a-Haystack (NIAH)
- **Why**: long-context retrieval; canonical "find this sentence in 100K
  tokens" stress test.
- **Download** (script-based, synthesizes):
  ```bash
  git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git
  ```

### InfiniteBench (∞Bench) (Zhang et al., 2024)
- **Why**: extreme long-context (100K–1M tokens) for retrieval and reasoning.
- **Download**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("xinrongzhang2022/InfiniteBench")
  ds.save_to_disk("datasets/infinitebench")
  ```

### GSM8K (math reasoning)
- **Why**: HashEvict uses this for "free response reasoning" eval; tests
  whether eviction degrades chain-of-thought.
- **Download**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("gsm8k", "main")
  ds.save_to_disk("datasets/gsm8k")
  ```

### TriviaQA / Natural Questions (open-domain QA)
- **Why**: short-context retrieval-style QA. Diagnostic for approximation
  quality on moderate sequences.
- **Download**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("trivia_qa", "rc.nocontext")
  ds.save_to_disk("datasets/triviaqa")
  ```

### enwik8 (byte-level)
- **Why**: used by Reformer (LSH attention) for sequences up to 64K bytes.
  Useful if you want to compare directly with Reformer's published numbers.
- **Download**: http://mattmahoney.net/dc/enwik8.zip

---

## Synthetic data for unit-testing the lookup primitive

A HashMap Attention prototype should first be tested on synthetic
query/key matrices before being plugged into a real LLM. The experiment
runner can generate them directly:

```python
import torch
seq_len = 4096
d_head  = 128
n_heads = 32

# Dense baseline
Q = torch.randn(n_heads, seq_len, d_head)
K = torch.randn(n_heads, seq_len, d_head)
V = torch.randn(n_heads, seq_len, d_head)
ref = torch.softmax(Q @ K.transpose(-1,-2) / d_head**0.5, dim=-1) @ V

# HashMap-Attention prototype computes only sparse subset, returns approx
# Compare ||approx - ref|| under different hash budgets / thresholds
```

This is the cheapest first-pass: it isolates the LSH/hashing primitive
from the model's training dynamics.

---

## License notes
- WikiText, PG19, GSM8K, TriviaQA: research-friendly, see HuggingFace cards.
- LongBench, RULER, InfiniteBench: research/non-commercial.
- enwik8: public domain (Wikipedia dump).
