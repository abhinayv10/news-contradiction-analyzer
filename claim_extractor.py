"""
LLM-driven structured claim extraction with chunking and deduplication.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rapidfuzz import fuzz

import config
from llm_client import chat_json_array_wrapper
from prompts import CLAIM_EXTRACTION_SYSTEM, CLAIM_EXTRACTION_USER


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split long text at paragraph boundaries without breaking mid-sentence when possible."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    chunks: list[str] = []
    paras = re.split(r"\n\s*\n+", text)
    buf = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_chars:
                # Hard split long paragraph
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i : i + max_chars])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks


def _normalize_claim(c: dict[str, Any]) -> dict[str, Any]:
    keys = ("entity", "claim_type", "value", "date", "context", "source_quote")
    out: dict[str, Any] = {}
    for k in keys:
        v = c.get(k, "")
        out[k] = str(v).strip() if v is not None else ""
    if out["claim_type"] not in ("metric", "event", "quote", "date"):
        out["claim_type"] = "event"
    return out


def _signature(claim: dict[str, Any]) -> str:
    return "|".join(
        [
            claim.get("entity", "").lower(),
            claim.get("claim_type", ""),
            claim.get("value", "").lower(),
            claim.get("context", "").lower()[:200],
        ]
    )


def dedupe_claims(claims: list[dict[str, Any]], threshold: float = 92.0) -> list[dict[str, Any]]:
    """Remove near-duplicate claims using fuzzy ratio on a compact signature."""
    kept: list[dict[str, Any]] = []
    sigs: list[str] = []
    for c in claims:
        s = _signature(c)
        is_dup = False
        for prev in sigs:
            if fuzz.ratio(s, prev) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
            sigs.append(s)
    return kept


def _split_sentences(text: str) -> list[str]:
    # Basic sentence splitter; good enough for a demo showcase.
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


_PERCENT_RANGE_RE = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(%|percent)"
)
_PERCENT_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(%|percent)")
_AMOUNT_UNIT_RE = re.compile(
    r"(?i)(\$?\s*\d[\d,]*(?:\.\d+)?)\s*(billion|million|trillion)\b"
)
_EPS_RE = re.compile(r"(?i)\b(EPS|earnings per share)\b.*?\$?\s*(\d+(?:\.\d+)?)")
_DOLLAR_NUMBER_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")

_MONTH_RE = re.compile(
    r"(?i)\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")


def _guess_entity(sentence: str) -> str:
    s = sentence.strip()
    if not s:
        return ""
    # Small set of common demo entities.
    if re.search(r"\bTim Cook\b", s, flags=re.I):
        return "Tim Cook"
    if re.search(r"\bApple\b", s, flags=re.I):
        return "Apple"

    # Generic fallback: try "Company Inc"/"Company Corp" patterns.
    m = re.search(
        r"\b([A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+)*)\s+(Inc\.|Corporation|Corp\.|Ltd\.|Limited|AG|SA)\b",
        s,
    )
    if m:
        return m.group(1).strip()
    return ""


def _extract_metric_expressions_from_source(source: str) -> list[str]:
    """
    Extract metric expressions from a text snippet.
    Returns raw substrings (unit-preserving) like "$143.8 billion" or "13-16%".
    """
    s = source or ""
    exprs: list[str] = []

    # Percent ranges (e.g. 13-16%).
    for m in _PERCENT_RANGE_RE.finditer(s):
        v = m.group(0).strip()
        if v and v not in exprs:
            exprs.append(v)

    # Single percent.
    for m in _PERCENT_RE.finditer(s):
        v = m.group(0).strip()
        if v and v not in exprs:
            exprs.append(v)

    # Amounts with explicit million/billion/trillion.
    for m in _AMOUNT_UNIT_RE.finditer(s):
        v = m.group(0).replace("  ", " ").strip()
        if v and v not in exprs:
            exprs.append(v)

    # EPS / per share.
    if re.search(r"(?i)\b(EPS|earnings per share|per share)\b", s):
        m = _EPS_RE.search(s)
        if m:
            num = m.group(2)
            exprs.append(f"${num} per share")
        else:
            m = _DOLLAR_NUMBER_RE.search(s)
            if m:
                exprs.append(f"${m.group(1)} per share")

    return exprs


def _extract_metric_value_from_source(source: str) -> str:
    exprs = _extract_metric_expressions_from_source(source)
    return exprs[0].strip() if exprs else ""


def _extract_date_value_from_source(source: str) -> str:
    s = source or ""
    m = _ISO_DATE_RE.search(s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    # Month name + year (common in headlines).
    if _MONTH_RE.search(s) and _YEAR_RE.search(s):
        return _MONTH_RE.search(s).group(0) + " " + _YEAR_RE.search(s).group(0)
    # Quarter patterns
    q = re.search(r"(?i)\bQ([1-4])\b", s)
    if q:
        return f"Q{q.group(1)}"
    # Year-only fallback.
    y = _YEAR_RE.search(s)
    if y:
        return y.group(1)
    return ""


def _repair_claim_values_using_source(claim: dict[str, Any]) -> dict[str, Any]:
    # Make extracted metric/date values unit-preserving by re-extracting
    # the numeric phrase from `source_quote`.
    c = dict(claim)
    if c.get("claim_type") == "metric":
        src = c.get("source_quote") or c.get("context") or c.get("value") or ""
        exprs = _extract_metric_expressions_from_source(src)
        if exprs:
            current = str(c.get("value") or "")
            # If the LLM got the number right but the unit wrong,
            # prefer the expression in `src` that contains the numeric token.
            nums = re.findall(r"\d+(?:\.\d+)?", current)
            chosen = ""
            for n in nums:
                n_norm = n.replace(",", "")
                for e in exprs:
                    if n_norm and n_norm in e.replace(",", ""):
                        chosen = e
                        break
                if chosen:
                    break
            c["value"] = chosen or exprs[0]
    if c.get("claim_type") == "date":
        src = c.get("source_quote") or c.get("context") or c.get("date") or ""
        fixed = _extract_date_value_from_source(src)
        if fixed:
            c["date"] = fixed
    return c


def _extract_claims_heuristic(article_text: str) -> list[dict[str, Any]]:
    sentences = _split_sentences(article_text)
    claims: list[dict[str, Any]] = []

    for sentence in sentences:
        sent = sentence.strip()
        if len(sent) < 25:
            continue

        entity = _guess_entity(sent)
        metric_values = _extract_metric_expressions_from_source(sent)

        # Dates: prefer explicit date/quarter patterns.
        date_value = _extract_date_value_from_source(sent)

        # Quote-like claims (demo heuristic).
        has_quote = bool(re.search(r"[\"“”']", sent))

        if metric_values:
            for mv in metric_values:
                claims.append(
                    {
                        "entity": entity,
                        "claim_type": "metric",
                        "value": mv,
                        "date": "",
                        "context": sent,
                        "source_quote": sent,
                    }
                )
            continue

        if date_value and has_quote is False:
            claims.append(
                {
                    "entity": entity,
                    "claim_type": "date",
                    "value": "",
                    "date": date_value,
                    "context": sent,
                    "source_quote": sent,
                }
            )
            continue

        if has_quote and len(sent) <= 220:
            claims.append(
                {
                    "entity": entity,
                    "claim_type": "quote",
                    "value": "quoted statement",
                    "date": "",
                    "context": sent,
                    "source_quote": sent,
                }
            )
            continue

        # Keep a small amount of event claims to anchor alignment.
        if re.search(r"(?i)\b(revenue|earnings|guidance|profit|quarter|forecast|expects|expects to|raised|raised its|cut|declined)\b", sent):
            claims.append(
                {
                    "entity": entity,
                    "claim_type": "event",
                    "value": sent[:120],
                    "date": "",
                    "context": sent,
                    "source_quote": sent,
                }
            )

        if len(claims) >= 60:
            break

    # Repair metric/date unit phrases (important when sentences contain multiple numbers).
    fixed = [_repair_claim_values_using_source(c) for c in claims]
    return dedupe_claims(fixed)


def extract_claims_from_article(article_text: str, article_label: str = "article") -> list[dict[str, Any]]:
    """
    Extract structured claims from an article.

    - Normal mode: LLM extraction per chunk, then repair metric/date values
      by re-extracting the numeric phrases from `source_quote` (unit-safe).
    - Demo mode: fast heuristic extraction (no LLM).
    """
    text = (article_text or "").strip()
    if not text:
        return []

    if getattr(config, "DEMO_MODE", False):
        return _extract_claims_heuristic(text)

    chunks = _chunk_text(text, config.CLAIM_CHUNK_CHARS)
    all_claims: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        user = CLAIM_EXTRACTION_USER.format(text=chunk)
        raw = chat_json_array_wrapper(
            CLAIM_EXTRACTION_SYSTEM,
            user,
            cache_prefix=f"extract_{article_label}_{i}",
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        arr = data.get("claims")
        if not isinstance(arr, list):
            continue
        for item in arr:
            if isinstance(item, dict):
                all_claims.append(_normalize_claim(item))

    merged = dedupe_claims(all_claims)
    repaired = [_repair_claim_values_using_source(c) for c in merged]
    return dedupe_claims(repaired)
