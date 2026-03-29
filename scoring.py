"""
Aggregate metrics and alignment score.
"""

from __future__ import annotations

from typing import Any


def compute_scores(
    claims_a: list[dict[str, Any]],
    claims_b: list[dict[str, Any]],
    classified_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    total_claims = len(A) + len(B) (reported separately too)
    matched_claims = number of aligned pairs
    conflicting = count label == conflict
    agreements = count label == agree
    alignment_score = (agreements / matched) * 100 if matched > 0 else 0
    """
    total_a = len(claims_a)
    total_b = len(claims_b)
    matched = len(classified_pairs)
    conflicts = sum(1 for p in classified_pairs if p.get("label") == "conflict")
    agreements = sum(1 for p in classified_pairs if p.get("label") == "agree")
    unverifiable = sum(1 for p in classified_pairs if p.get("label") == "unverifiable")

    if matched > 0:
        alignment_score = round((agreements / matched) * 100.0, 2)
    else:
        alignment_score = 0.0

    return {
        "total_claims_article_a": total_a,
        "total_claims_article_b": total_b,
        "total_claims": total_a + total_b,
        "matched_claims": matched,
        "conflicting_claims": conflicts,
        "agreements": agreements,
        "unverifiable_pairs": unverifiable,
        "alignment_score": alignment_score,
    }
