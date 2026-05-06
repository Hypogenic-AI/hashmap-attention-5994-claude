#!/usr/bin/env python3
"""Download high-priority papers using arxiv API and Semantic Scholar."""
import os
import re
import sys
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path

PAPERS_DIR = Path("papers")
PAPERS_DIR.mkdir(exist_ok=True)

# (short_name, query, optional explicit arxiv_id)
PAPERS = [
    ("reformer_lsh_attention", "Reformer The Efficient Transformer", "2001.04451"),
    ("magicpig_lsh_sampling", "MagicPIG LSH Sampling for Efficient LLM Generation", "2410.16179"),
    ("hashevict_lsh_kv", "HashEvict Pre-Attention KV Cache Eviction Locality-Sensitive Hashing", "2412.16187"),
    ("retrieval_attention_anns", "RetrievalAttention Long-Context LLM Inference Vector Retrieval", "2409.10516"),
    ("quest_query_aware", "Quest Query-Aware Sparsity Long-Context LLM", "2406.10774"),
    ("sparq_bandwidth", "SparQ Attention Bandwidth-Efficient LLM Inference", "2312.04985"),
    ("hyperattention_near_linear", "HyperAttention Long-context Attention in Near-Linear Time", "2310.05869"),
    ("loki_low_rank_keys", "Loki Low-Rank Keys for Efficient Sparse Attention", "2406.02542"),
    ("smyrf_asymmetric_clustering", "SMYRF Efficient Attention using Asymmetric Clustering", "2010.05315"),
    ("scatterbrain_sparse_lowrank", "Scatterbrain Unifying Sparse Low-rank Attention", "2110.15343"),
    ("efficient_transformers_survey", "Efficient Transformers A Survey Tay Dehghani", "2009.06732"),
    ("flashattention_v1", "FlashAttention Fast Memory-Efficient Exact Attention", "2205.14135"),
    ("flashattention_v2", "FlashAttention-2 Faster Attention with Better Parallelism", "2307.08691"),
    ("h2o_heavy_hitter", "H2O Heavy-Hitter Oracle Efficient Generative Inference", "2306.14048"),
    ("streamingllm_attention_sink", "Efficient Streaming Language Models with Attention Sinks", "2309.17453"),
    ("kivi_kv_quantization", "KIVI A Tuning-Free Asymmetric 2bit Quantization KV Cache", "2402.02750"),
    ("kvquant_quantization", "KVQuant Towards 10 Million Context Length LLM Inference KV Quantization", "2401.18079"),
    ("longformer_sparse", "Longformer The Long-Document Transformer", "2004.05150"),
    ("bigbird_sparse", "Big Bird Transformers for Longer Sequences", "2007.14062"),
    ("performer_kernel_attention", "Rethinking Attention with Performers", "2009.14794"),
    ("linear_transformer", "Transformers are RNNs Fast Autoregressive Transformers Linear Attention", "2006.16236"),
    ("snapkv_kv_compression", "SnapKV LLM Knows What You are Looking for Before Generation", "2404.14469"),
    ("native_sparse_attention", "Native Sparse Attention Hardware-Aligned Trainable", "2502.11089"),
    ("multipole_attention", "Multipole Attention for Efficient Long Context Reasoning", "2505.13059"),
    ("kdeformer_kde_attention", "KDEformer Accelerating Transformers Kernel Density Estimation", "2302.02451"),
    ("learning_to_hash_sparse", "Sparse Attention with Learning to Hash", None),  # needs lookup
    ("lsh_long_context_nmt", "Locality-Sensitive Hashing Long Context Neural Machine Translation", None),
    ("shadowkv_kv_shadows", "ShadowKV KV Cache in Shadows High-Throughput Long-Context", "2410.21465"),
    ("less_recurrence_kv", "Get More with LESS Synthesizing Recurrence KV Cache Compression", "2402.09398"),
    ("dynamic_context_pruning", "Dynamic Context Pruning Efficient Interpretable Autoregressive Transformers", "2305.15805"),
    ("kvtuner_mixed_precision", "KVTuner Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization", "2502.04420"),
    ("q_hitter_sparse_quantized", "Q-Hitter Better Token Oracle Efficient LLM Inference", None),
    ("ecoformer_binary_attention", "EcoFormer Energy-Saving Attention with Linear Complexity", "2209.09004"),
    ("transformer_lsh_point", "Locality-Sensitive Hashing Efficient Point Transformer High-Energy", "2402.12535"),
    ("attention_is_all_you_need", "Attention Is All You Need Vaswani transformer", "1706.03762"),
    ("retro_retrieval_lm", "Improving language models by retrieving from trillions of tokens", "2112.04426"),
    ("kmeans_attention_routing", "Routing Transformer Sparse Attention Clustering", "2003.05997"),
]


def download_arxiv(arxiv_id, target):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Research-Bot/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            if len(data) < 1024:
                return False, f"too small ({len(data)} bytes)"
            with open(target, "wb") as f:
                f.write(data)
            return True, f"{len(data)} bytes"
    except Exception as e:
        return False, str(e)


def search_arxiv(query):
    """Use arxiv API to find paper id by title."""
    q = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=ti:{q}&max_results=3"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Research-Bot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode("utf-8")
        # find first arxiv id in entries
        m = re.search(r"http://arxiv\.org/abs/([\w./-]+?)(?:v\d+)?(?=<)", text)
        if m:
            return m.group(1)
    except Exception as e:
        return None
    return None


def main():
    log = []
    for name, query, arxiv_id in PAPERS:
        target = PAPERS_DIR / f"{name}.pdf"
        if target.exists() and target.stat().st_size > 1024:
            print(f"[skip] {name}")
            log.append({"name": name, "status": "skipped", "size": target.stat().st_size})
            continue
        if not arxiv_id:
            arxiv_id = search_arxiv(query)
            time.sleep(0.5)
        if not arxiv_id:
            print(f"[FAIL] {name}: no arxiv id found for {query}")
            log.append({"name": name, "status": "no_id", "query": query})
            continue
        ok, info = download_arxiv(arxiv_id, target)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name} ({arxiv_id}): {info}")
        log.append({"name": name, "status": status, "arxiv_id": arxiv_id, "info": info})
        time.sleep(0.5)

    with open("papers/download_log.json", "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
