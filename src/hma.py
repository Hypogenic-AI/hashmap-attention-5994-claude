"""HashMap Attention (HMA) primitive.

Implements a per-(layer, head, token-id-or-ngram) lookup table from
prior attention behavior. During prefill, populate by recording which
key positions any prior occurrence of token id (or n-gram-context hash)
attended to with weight >= threshold tau. During decode (or scoring),
use the lookup as the candidate-key set; optionally union with
"safety" sets (attention sinks + sliding window).

Designed to be small, dependency-light, and dictionary-based for
correctness validation. Speed comes later; here we measure quality.

Key data structures
-------------------
H[(layer, head, key_id)] -> set of int positions
where key_id is either a token id (identity) or an n-gram hash.

For HMA-Quant, also maintain hit_count[(layer, head, position)] -> int
for adaptive bit-width assignment downstream.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class HMAConfig:
    """Configuration for HashMap Attention."""

    threshold: float = 0.02  # tau: attention weight cutoff for storing position
    n_gram: int = 1          # 1 = identity, 2+ = n-gram context hash
    sink_size: int = 4       # always-on initial tokens (StreamingLLM-style)
    window_size: int = 32    # always-on local window
    max_budget: Optional[int] = None  # if set, cap candidate set per query
    fuzzy_hash_dim: Optional[int] = None  # if set, SimHash-bit width on context (n_gram>=2 only)
    use_topk_only: bool = False  # if True, threshold ignored; pick top-k by stored frequency


def _ngram_keys(token_ids: torch.Tensor, n: int) -> list[int]:
    """Return per-position lookup key.

    n=1: identity (token id at that position)
    n>=2: simple Python hash of the (n-1)-token context + current id.
    Returns list of ints aligned with token_ids.
    """
    ids = token_ids.tolist()
    keys = []
    for i in range(len(ids)):
        ctx = tuple(ids[max(0, i - n + 1) : i + 1])
        keys.append(hash(ctx) & 0xFFFFFFFF)
    return keys


class HashMapAttention:
    """Per-(layer, head) HashMap Attention table.

    Used for *analysis* of approximation quality. We do not produce a
    fast CUDA kernel; we produce a faithful candidate-set predictor and
    a reconstructed attention output, then compare to the dense oracle.
    """

    def __init__(self, n_layers: int, n_heads: int, config: HMAConfig):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.cfg = config
        # H[layer][head][key] -> set of positions
        self.H: list[list[dict[int, set[int]]]] = [
            [defaultdict(set) for _ in range(n_heads)] for _ in range(n_layers)
        ]
        # hit_count[layer][head][position] -> int
        self.hit_count: list[list[dict[int, int]]] = [
            [defaultdict(int) for _ in range(n_heads)] for _ in range(n_layers)
        ]
        # cache of token-id sequence (for context lookup)
        self.token_ids: Optional[torch.Tensor] = None
        self.lookup_keys: Optional[list[int]] = None

    def populate(
        self,
        layer: int,
        attn_weights: torch.Tensor,  # (heads, q_len, k_len) softmax probabilities
        token_ids: torch.Tensor,     # (q_len,) token ids — q and k share positions
    ) -> None:
        """Populate lookup table from a full-attention oracle slice."""
        cfg = self.cfg
        if self.lookup_keys is None or self.token_ids is None or len(self.token_ids) != len(token_ids):
            self.token_ids = token_ids.detach().cpu()
            self.lookup_keys = _ngram_keys(self.token_ids, cfg.n_gram)
        attn = attn_weights.detach().cpu()
        H_layer = self.H[layer]
        cnt_layer = self.hit_count[layer]
        n_heads, q_len, k_len = attn.shape
        for h in range(n_heads):
            H = H_layer[h]
            cnt = cnt_layer[h]
            # vectorized: for each query position, find positions above threshold
            # mask: (q_len, k_len)
            mask = attn[h] >= cfg.threshold
            # iterate queries (small q_len since we're at L<=2048)
            for q in range(q_len):
                key = self.lookup_keys[q]
                positions = mask[q].nonzero(as_tuple=False).flatten().tolist()
                if positions:
                    H[key].update(positions)
                    for p in positions:
                        cnt[p] += 1

    def candidate_positions(
        self,
        layer: int,
        head: int,
        query_pos: int,
    ) -> set[int]:
        """Return the candidate-key set for a query at position query_pos."""
        cfg = self.cfg
        candidates: set[int] = set()
        # Sinks
        sink = min(cfg.sink_size, query_pos + 1)
        candidates.update(range(sink))
        # Local sliding window
        win_lo = max(0, query_pos - cfg.window_size + 1)
        win_hi = query_pos + 1
        candidates.update(range(win_lo, win_hi))
        # Lookup-driven candidates
        if self.lookup_keys is not None and query_pos < len(self.lookup_keys):
            key = self.lookup_keys[query_pos]
            stored = self.H[layer][head].get(key, set())
            # only include positions <= query_pos (causal)
            for p in stored:
                if p <= query_pos:
                    candidates.add(p)
        # Optional cap by global hit-frequency (Top-K of stored hits within candidates)
        if cfg.max_budget is not None and len(candidates) > cfg.max_budget:
            cnt = self.hit_count[layer][head]
            sorted_cands = sorted(candidates, key=lambda p: -cnt.get(p, 0))
            candidates = set(sorted_cands[: cfg.max_budget])
            # always re-add sinks + most-recent token to be safe
            candidates.update(range(sink))
            candidates.add(query_pos)
        return candidates

    # ---------- analysis utilities ----------

    def reconstruct_attention(
        self,
        layer: int,
        Q: torch.Tensor,  # (heads, q_len, d)
        K: torch.Tensor,  # (heads, k_len, d)
        V: torch.Tensor,  # (heads, k_len, d_v)
        scale: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention restricted to the HMA candidate set.

        Returns (output, mean_budget_per_query).
        """
        if scale is None:
            scale = 1.0 / math.sqrt(Q.shape[-1])
        n_heads, q_len, d = Q.shape
        k_len = K.shape[1]
        out = torch.zeros(n_heads, q_len, V.shape[-1], dtype=V.dtype)
        budgets = []
        for h in range(n_heads):
            for q in range(q_len):
                cand = sorted(self.candidate_positions(layer, h, q))
                # causal mask: cand already filtered by query_pos
                cand = [p for p in cand if p <= q]
                if not cand:
                    cand = [q]
                budgets.append(len(cand))
                idx = torch.tensor(cand, dtype=torch.long)
                k_sel = K[h].index_select(0, idx)
                v_sel = V[h].index_select(0, idx)
                logits = (Q[h, q] @ k_sel.T) * scale
                weights = torch.softmax(logits, dim=-1)
                out[h, q] = weights @ v_sel
        return out, torch.tensor(budgets, dtype=torch.float32).mean()


# ------------------------------------------------------------------
# Baselines: candidate-set predictors that we compare HMA against.
# Each takes (layer, head, query_pos, q_len, k_len) and returns a
# set[int] of candidate key positions (causal: <= query_pos).
# ------------------------------------------------------------------


def baseline_full(query_pos: int, **_) -> set[int]:
    return set(range(query_pos + 1))


def baseline_streaming(query_pos: int, sink: int = 4, window: int = 32, **_) -> set[int]:
    cands = set(range(min(sink, query_pos + 1)))
    cands.update(range(max(0, query_pos - window + 1), query_pos + 1))
    return cands


def baseline_random(
    query_pos: int, k_len: int, budget: int, rng: torch.Generator, **_
) -> set[int]:
    """Random budget-k subset of [0, query_pos]."""
    avail = query_pos + 1
    if budget >= avail:
        return set(range(avail))
    perm = torch.randperm(avail, generator=rng)[:budget].tolist()
    return set(perm)


def baseline_h2o_oracle(
    layer: int,
    head: int,
    query_pos: int,
    accumulated_attn: torch.Tensor,  # (heads, k_len)
    budget: int,
    sink: int = 4,
    **_,
) -> set[int]:
    """Heavy-hitter (H2O-style) keep top-k of accumulated attention so far.

    Uses an accumulated-attention vector populated externally. This is an
    oracle in the sense that it knows attention behavior. For comparison
    fairness, HMA also gets prefill-time observation.
    """
    cands = set(range(min(sink, query_pos + 1)))
    cands.add(query_pos)
    avail = query_pos + 1
    scores = accumulated_attn[head, :avail]
    if budget >= avail:
        cands.update(range(avail))
        return cands
    topk = torch.topk(scores, k=budget).indices.tolist()
    cands.update(topk)
    return cands
