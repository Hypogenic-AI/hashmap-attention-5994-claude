"""Common utilities: seeding, attention oracle, metrics."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dense_attention(
    Q: torch.Tensor,  # (heads, q_len, d)
    K: torch.Tensor,  # (heads, k_len, d)
    V: torch.Tensor,  # (heads, k_len, d_v)
    causal: bool = True,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (output, weights). Weights are softmax probabilities."""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    logits = torch.einsum("hqd,hkd->hqk", Q, K) * scale
    if causal:
        q_len, k_len = logits.shape[1], logits.shape[2]
        mask = torch.triu(torch.ones(q_len, k_len, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(mask, float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    out = torch.einsum("hqk,hkd->hqd", weights, V)
    return out, weights


def relative_error(approx: torch.Tensor, oracle: torch.Tensor) -> float:
    """`||approx - oracle||_2 / ||oracle||_2` averaged across queries."""
    diff = (approx - oracle).reshape(-1, oracle.shape[-1]).norm(dim=-1)
    norm = oracle.reshape(-1, oracle.shape[-1]).norm(dim=-1).clamp_min(1e-9)
    return (diff / norm).mean().item()


def topk_recall(
    approx_weights: torch.Tensor,
    oracle_weights: torch.Tensor,
    k: int,
) -> float:
    """Fraction of oracle top-k positions that approximation also assigns nonzero weight."""
    n_heads, q_len, k_len = oracle_weights.shape
    recalls = []
    for h in range(n_heads):
        for q in range(q_len):
            o = oracle_weights[h, q]
            a = approx_weights[h, q]
            if o.sum() < 1e-9:
                continue
            true_top = set(torch.topk(o, k=min(k, k_len)).indices.tolist())
            approx_nz = set((a > 1e-9).nonzero(as_tuple=False).flatten().tolist())
            recalls.append(len(true_top & approx_nz) / max(1, len(true_top)))
    return float(np.mean(recalls)) if recalls else 0.0


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_wikitext_validation_text(limit_chars: int = 200_000) -> str:
    """Stream WikiText-2 validation set from local copy and concatenate."""
    from datasets import load_from_disk

    ds = load_from_disk("datasets/wikitext-2/validation")
    text = "\n".join(s for s in ds["text"] if s.strip())
    return text[:limit_chars]
