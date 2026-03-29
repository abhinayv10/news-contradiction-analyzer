"""
Align claims between two articles using rapidfuzz and optional embedding similarity.
"""

from __future__ import annotations

import math
from typing import Any

from rapidfuzz import fuzz

import config
from llm_client import embed_texts


def _claim_repr(c: dict[str, Any]) -> str:
    """Single string for fuzzy matching (entity + type + value + context)."""
    parts = [
        c.get("entity", ""),
        c.get("claim_type", ""),
        c.get("value", ""),
        c.get("date", ""),
        c.get("context", ""),
        (c.get("source_quote", "") or "")[:300],
    ]
    return " | ".join(str(p) for p in parts)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _pair_score(
    ia: int,
    ib: int,
    repr_a: str,
    repr_b: str,
    emb_a: list[float] | None,
    emb_b: list[float] | None,
) -> tuple[float, float, float | None]:
    fuzzy_part = float(fuzz.token_set_ratio(repr_a, repr_b))
    emb_boost = 0.0
    if emb_a is not None and emb_b is not None:
        emb_boost = max(0.0, _cosine(emb_a, emb_b)) * 100.0
    combined = min(100.0, fuzzy_part + emb_boost * 0.2)
    return combined, fuzzy_part, emb_boost if emb_a is not None else None


def align_claims(
    claims_a: list[dict[str, Any]],
    claims_b: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    One-to-one greedy matching: pairs sorted by combined score descending.
    """
    if not claims_a or not claims_b:
        return []

    reps_a = [_claim_repr(c) for c in claims_a]
    reps_b = [_claim_repr(c) for c in claims_b]

    emb_a = emb_b = None
    if config.LLM_EMBEDDING_MODEL and not getattr(config, "DEMO_MODE", False):
        emb = embed_texts(reps_a + reps_b)
        if emb and len(emb) == len(claims_a) + len(claims_b):
            n = len(claims_a)
            emb_a, emb_b = emb[:n], emb[n:]

    candidates: list[tuple[float, int, int, float, float | None]] = []
    for ia, ra in enumerate(reps_a):
        ea = emb_a[ia] if emb_a else None
        for ib, rb in enumerate(reps_b):
            eb = emb_b[ib] if emb_b else None
            combined, fuzzy_part, emb_boost = _pair_score(ia, ib, ra, rb, ea, eb)
            if combined >= config.ALIGNMENT_MIN_SCORE:
                candidates.append((combined, ia, ib, fuzzy_part, emb_boost))

    candidates.sort(key=lambda x: -x[0])
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[dict[str, Any]] = []

    for combined, ia, ib, fuzzy_part, emb_boost in candidates:
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        pairs.append(
            {
                "index_a": ia,
                "index_b": ib,
                "score": round(combined, 2),
                "fuzzy_score": round(fuzzy_part, 2),
                "embedding_boost": round(emb_boost, 2) if emb_boost is not None else None,
            }
        )

    return sorted(pairs, key=lambda x: x["index_a"])
