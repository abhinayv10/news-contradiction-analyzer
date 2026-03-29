"""
LLM-based contradiction classification for aligned claim pairs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
import difflib
from typing import Any

from rapidfuzz import fuzz

from llm_client import get_client
import config
from prompts import CONTRADICTION_SYSTEM, CONTRADICTION_USER


def _format_claim(c: dict[str, Any]) -> str:
    return json.dumps(
        {
            "entity": c.get("entity", ""),
            "claim_type": c.get("claim_type", ""),
            "value": c.get("value", ""),
            "date": c.get("date", ""),
            "context": c.get("context", ""),
            "source_quote": (c.get("source_quote", "") or "")[:500],
        },
        ensure_ascii=False,
    )


@dataclass(frozen=True)
class _ParsedMetric:
    # kind: "percent", "amount", "number"
    kind: str
    # for percent: low/high represent range (if range) or same value for point.
    low: float
    high: float
    # original text used for explanation
    raw: str


_PERCENT_RANGE_RE = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(%|percent)"
)
_PERCENT_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(%|percent)")
_AMOUNT_UNIT_RE = re.compile(
    r"(?i)(\$?\s*\d[\d,]*(?:\.\d+)?)\s*(billion|million|trillion)\b"
)
_DOLLAR_NUMBER_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_MONTH_NAME_RE = re.compile(
    r"(?i)\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
)


def _parse_percent(text: str) -> _ParsedMetric | None:
    s = text or ""
    m = _PERCENT_RANGE_RE.search(s)
    if m:
        low = float(m.group(1))
        high = float(m.group(2))
        if low > high:
            low, high = high, low
        return _ParsedMetric(kind="percent", low=low, high=high, raw=m.group(0))
    m = _PERCENT_RE.search(s)
    if m:
        v = float(m.group(1))
        return _ParsedMetric(kind="percent", low=v, high=v, raw=m.group(0))
    return None


def _parse_amount(text: str) -> _ParsedMetric | None:
    s = text or ""
    m = _AMOUNT_UNIT_RE.search(s)
    if not m:
        return None
    raw_num = m.group(1).replace("$", "").replace(" ", "")
    raw_num = raw_num.replace(",", "")
    num = float(raw_num)
    unit = m.group(2).lower()
    mult = 1.0
    if unit == "million":
        mult = 1e6
    elif unit == "billion":
        mult = 1e9
    elif unit == "trillion":
        mult = 1e12
    return _ParsedMetric(kind="amount", low=num * mult, high=num * mult, raw=m.group(0))


def _parse_number(text: str) -> _ParsedMetric | None:
    s = text or ""
    # Only treat $number as a standalone metric when the text looks like EPS/per-share.
    if not re.search(r"(?i)\b(eps|earnings per share|per share)\b", s):
        return None
    m = _DOLLAR_NUMBER_RE.search(s)
    if not m:
        return None
    v = float(m.group(1))
    return _ParsedMetric(kind="number", low=v, high=v, raw=m.group(0))


def _parse_metric(text: str) -> _ParsedMetric | None:
    s = text or ""
    # Prefer percent if present.
    if "%" in s or re.search(r"(?i)\bpercent\b", s):
        pm = _parse_percent(s)
        if pm:
            return pm

    # Prefer amount with explicit million/billion/trillion units.
    if re.search(r"(?i)\b(million|billion|trillion)\b", s):
        am = _parse_amount(s)
        if am:
            return am

    # EPS/per share numeric value.
    num = _parse_number(s)
    if num:
        return num
    return None


def _parse_date(text: str) -> tuple[str, int | None, int | None, int | None] | None:
    """
    Returns a normalized representation: ("date", year, month, day)
    month/day may be None if not present.
    """
    s = text or ""
    m_iso = _ISO_DATE_RE.search(s)
    if m_iso:
        y = int(m_iso.group(1))
        mo = int(m_iso.group(2))
        d = int(m_iso.group(3))
        return ("date", y, mo, d)

    y_m = _YEAR_RE.search(s)
    if y_m and _MONTH_NAME_RE.search(s):
        y = int(y_m.group(1))
        month_raw = _MONTH_NAME_RE.search(s).group(0).lower()
        # Basic month mapping
        month_map = {
            "jan": 1,
            "january": 1,
            "feb": 2,
            "february": 2,
            "mar": 3,
            "march": 3,
            "apr": 4,
            "april": 4,
            "may": 5,
            "jun": 6,
            "june": 6,
            "jul": 7,
            "july": 7,
            "aug": 8,
            "august": 8,
            "sep": 9,
            "sept": 9,
            "september": 9,
            "oct": 10,
            "october": 10,
            "nov": 11,
            "november": 11,
            "dec": 12,
            "december": 12,
        }
        mo = None
        for k, v in month_map.items():
            if month_raw.startswith(k):
                mo = v
                break
        if mo:
            return ("date", y, mo, None)

    # Quarter-like fallback (Q1..Q4)
    q = re.search(r"(?i)\bQ([1-4])\b", s)
    if q:
        return ("quarter", None, int(q.group(1)), None)

    if y_m:
        return ("year", int(y_m.group(1)), None, None)
    return None


def _try_classify_pair_rules(
    claim_a: dict[str, Any],
    claim_b: dict[str, Any],
) -> dict[str, Any] | None:
    type_a = claim_a.get("claim_type", "")
    type_b = claim_b.get("claim_type", "")
    if not type_a or not type_b:
        return None

    if type_a != type_b:
        return None

    # Build evidence strings for numeric parsing:
    # prefer `value` so we don't accidentally parse a *different* metric
    # that happens to co-exist in the same sentence.
    evidence_a = str(claim_a.get("value") or "").strip() or claim_a.get("source_quote") or claim_a.get("context") or ""
    evidence_b = str(claim_b.get("value") or "").strip() or claim_b.get("source_quote") or claim_b.get("context") or ""

    if type_a == "metric":
        pm_a = _parse_metric(evidence_a)
        pm_b = _parse_metric(evidence_b)
        if not pm_a or not pm_b:
            return None
        if pm_a.kind != pm_b.kind:
            return None

        # Percent comparison (range-aware).
        if pm_a.kind == "percent":
            tol = float(getattr(config, "PERCENT_ABS_TOLERANCE", 1.0))
            # Overlap check for ranges.
            overlap_low = max(pm_a.low, pm_b.low)
            overlap_high = min(pm_a.high, pm_b.high)
            if overlap_low <= overlap_high + tol:
                return {
                    "label": "agree",
                    "confidence": 0.92,
                    "explanation": f"Percent matches within tolerance: {pm_a.raw} vs {pm_b.raw}",
                }
            return {
                "label": "conflict",
                "confidence": 0.86,
                "explanation": f"Percent ranges do not overlap (beyond tolerance): {pm_a.raw} vs {pm_b.raw}",
            }

        # Amount comparison (relative tolerance).
        if pm_a.kind == "amount":
            rel_tol = float(getattr(config, "AMOUNT_REL_TOLERANCE", 0.03))
            a = pm_a.low
            b = pm_b.low
            denom = max(abs(a), abs(b), 1.0)
            rel_diff = abs(a - b) / denom
            if rel_diff <= rel_tol:
                return {
                    "label": "agree",
                    "confidence": 0.95,
                    "explanation": f"Amounts match within {rel_tol*100:.1f}% tolerance: {pm_a.raw} vs {pm_b.raw}",
                }
            return {
                "label": "conflict",
                "confidence": 0.88,
                "explanation": f"Amounts differ beyond tolerance ({rel_tol*100:.1f}%): {pm_a.raw} vs {pm_b.raw}",
            }

        # EPS/per-share numeric value.
        if pm_a.kind == "number":
            rel_tol = float(getattr(config, "AMOUNT_REL_TOLERANCE", 0.03))
            a = pm_a.low
            b = pm_b.low
            denom = max(abs(a), abs(b), 1e-9)
            rel_diff = abs(a - b) / denom
            abs_diff = abs(a - b)
            # For per-share numbers, allow some absolute slack too.
            abs_tol = max(0.05, abs(a) * rel_tol, abs(b) * rel_tol)
            if rel_diff <= rel_tol or abs_diff <= abs_tol:
                return {
                    "label": "agree",
                    "confidence": 0.9,
                    "explanation": f"Per-share numbers align: {pm_a.raw} vs {pm_b.raw}",
                }
            return {
                "label": "conflict",
                "confidence": 0.84,
                "explanation": f"Per-share numbers conflict: {pm_a.raw} vs {pm_b.raw}",
            }

        return None

    if type_a == "date":
        pd_a = _parse_date(evidence_a)
        pd_b = _parse_date(evidence_b)
        if not pd_a or not pd_b:
            return None

        # Compare only the normalized fields we got.
        kind_a, y_a, mo_a, d_a = pd_a
        kind_b, y_b, mo_b, d_b = pd_b
        # If the extraction found different granularities, keep it cautious.
        if kind_a != kind_b:
            return {
                "label": "unverifiable",
                "confidence": 0.35,
                "explanation": "Date granularity differs between sources.",
            }

        if y_a is not None and y_b is not None and y_a != y_b:
            return {
                "label": "conflict",
                "confidence": 0.85,
                "explanation": f"Years conflict: {y_a} vs {y_b}.",
            }

        if mo_a is not None and mo_b is not None and mo_a != mo_b:
            return {
                "label": "conflict",
                "confidence": 0.8,
                "explanation": "Months conflict.",
            }

        if d_a is not None and d_b is not None and d_a != d_b:
            return {
                "label": "conflict",
                "confidence": 0.85,
                "explanation": "Dates conflict.",
            }

        if y_a is not None or mo_a is not None:
            return {
                "label": "agree",
                "confidence": 0.78 if kind_a in ("year", "quarter") else 0.9,
                "explanation": "Dates match at extracted granularity.",
            }
        return None

    if type_a == "quote":
        sq_a = (claim_a.get("source_quote") or "").strip()
        sq_b = (claim_b.get("source_quote") or "").strip()
        if not sq_a or not sq_b:
            return None

        ratio = float(fuzz.token_set_ratio(sq_a, sq_b))
        if ratio >= float(getattr(config, "QUOTE_AGREE_THRESHOLD", 90)):
            return {
                "label": "agree",
                "confidence": 0.93,
                "explanation": f"Quote similarity is high ({ratio:.0f}/100).",
            }
        if ratio <= float(getattr(config, "QUOTE_CONFLICT_THRESHOLD", 60)):
            return {
                "label": "conflict",
                "confidence": 0.84,
                "explanation": f"Quote similarity is low ({ratio:.0f}/100).",
            }

        # For medium similarity, avoid over-claiming.
        seq_ratio = difflib.SequenceMatcher(None, sq_a, sq_b).ratio()
        return {
            "label": "unverifiable",
            "confidence": 0.5,
            "explanation": f"Quote overlap unclear (token similarity {ratio:.0f}/100, sequence {seq_ratio:.2f}).",
        }

    return None


def _classify_pair_llm(claim_a: dict[str, Any], claim_b: dict[str, Any]) -> dict[str, Any]:
    """LLM-only contradiction classification (slow, but handles non-numeric/event logic)."""
    user = CONTRADICTION_USER.format(
        claim_a=_format_claim(claim_a),
        claim_b=_format_claim(claim_b),
    )
    client = get_client()
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": CONTRADICTION_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"label": "unverifiable", "confidence": 0.0, "explanation": "Parse error."}

    label = data.get("label", "unverifiable")
    if label not in ("agree", "conflict", "unverifiable"):
        label = "unverifiable"
    conf = data.get("confidence", 0.5)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    expl = str(data.get("explanation", "")).strip() or "No explanation."
    return {"label": label, "confidence": conf, "explanation": expl}


def classify_pair(claim_a: dict[str, Any], claim_b: dict[str, Any]) -> dict[str, Any]:
    """
    Hybrid contradiction classification:
    1) Try local rule-based numeric/date/quote comparisons.
    2) If rules can't decide, fall back to LLM (unless DEMO_MODE).
    """
    rule_out = _try_classify_pair_rules(claim_a, claim_b)
    if rule_out is not None:
        return rule_out
    if getattr(config, "DEMO_MODE", False):
        return {
            "label": "unverifiable",
            "confidence": 0.4,
            "explanation": "Demo mode: rules could not decide for this claim type.",
        }
    return _classify_pair_llm(claim_a, claim_b)


def classify_pairs(
    claims_a: list[dict[str, Any]],
    claims_b: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run classifier for each aligned pair; attach indices and claims for UI."""
    results: list[dict[str, Any]] = []
    for row in alignment:
        ia, ib = row["index_a"], row["index_b"]
        ca, cb = claims_a[ia], claims_b[ib]
        out = classify_pair(ca, cb)
        results.append(
            {
                **row,
                "claim_a": ca,
                "claim_b": cb,
                "label": out["label"],
                "confidence": out["confidence"],
                "explanation": out["explanation"],
            }
        )
    return results
