"""
Streamlit dashboard: Financial News Contradiction Analyzer.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

import scoring
from aligner import align_claims
from claim_extractor import extract_claims_from_article
from classifier import classify_pairs
from diff_utils import ratio as difflib_ratio, unified_diff_snippet
from ingestion import extract_text_from_pdf, fetch_article_from_url, normalize_text
from report import build_pdf_report


def _load_inputs(
    mode: str,
    url_a: str,
    url_b: str,
    pdf_a: Any,
    pdf_b: Any,
) -> tuple[str, str]:
    if mode == "URLs":
        if not url_a.strip() or not url_b.strip():
            raise ValueError("Please provide both URLs.")
        with st.spinner("Fetching article A…"):
            text_a = fetch_article_from_url(url_a.strip())
        with st.spinner("Fetching article B…"):
            text_b = fetch_article_from_url(url_b.strip())
    else:
        if pdf_a is None or pdf_b is None:
            raise ValueError("Please upload two PDF files.")
        with st.spinner("Reading PDF A…"):
            text_a = extract_text_from_pdf(pdf_a.read())
            pdf_a.seek(0)
        with st.spinner("Reading PDF B…"):
            text_b = extract_text_from_pdf(pdf_b.read())
            pdf_b.seek(0)
    text_a = normalize_text(text_a)
    text_b = normalize_text(text_b)
    if len(text_a) < 80 or len(text_b) < 80:
        raise ValueError("Extracted text is too short. Check URL/PDF content.")
    return text_a, text_b


def _card_color(label: str) -> str:
    if label == "agree":
        return "#1b5e20"
    if label == "conflict":
        return "#b71c1c"
    return "#424242"


def main() -> None:
    st.set_page_config(page_title="Financial News Contradiction Analyzer", layout="wide")
    st.title("Financial News Contradiction Analyzer")
    st.caption("Compare two financial news sources (URLs or PDFs); extract claims, align, and detect contradictions.")

    mode = st.radio("Input type", ["URLs", "PDFs"], horizontal=True)

    col1, col2 = st.columns(2)
    url_a = url_b = ""
    pdf_a = pdf_b = None
    with col1:
        if mode == "URLs":
            url_a = st.text_input("Article A URL", placeholder="https://…")
        else:
            pdf_a = st.file_uploader("Article A PDF", type=["pdf"], key="pa")
    with col2:
        if mode == "URLs":
            url_b = st.text_input("Article B URL", placeholder="https://…")
        else:
            pdf_b = st.file_uploader("Article B PDF", type=["pdf"], key="pb")

    run = st.button("Analyze", type="primary")

    if not run:
        import config as _config

        if getattr(_config, "DEMO_MODE", False):
            st.info("Configure inputs and click **Analyze** to run the pipeline (demo mode: rules/heuristics; no LLM API key required).")
        else:
            st.info("Configure inputs and click **Analyze** to run the pipeline (LLM API key required unless DEMO_MODE=true).")
        return

    try:
        text_a, text_b = _load_inputs(mode, url_a, url_b, pdf_a, pdf_b)
    except Exception as e:
        st.error(str(e))
        return

    try:
        with st.spinner("Extracting claims from article A…"):
            claims_a = extract_claims_from_article(text_a, "a")
        with st.spinner("Extracting claims from article B…"):
            claims_b = extract_claims_from_article(text_b, "b")
        with st.spinner("Aligning claims…"):
            alignment = align_claims(claims_a, claims_b)
        with st.spinner("Classifying pairs with LLM…"):
            classified = classify_pairs(claims_a, claims_b, alignment)
        score_bundle = scoring.compute_scores(claims_a, claims_b, classified)
    except RuntimeError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.exception(e)
        return

    # Summary panel
    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alignment score", f"{score_bundle['alignment_score']}%")
    m2.metric("Matched pairs", score_bundle["matched_claims"])
    m3.metric("Conflicts", score_bundle["conflicting_claims"])
    m4.metric("Claims (A / B)", f"{score_bundle['total_claims_article_a']} / {score_bundle['total_claims_article_b']}")

    conflicts = [p for p in classified if p.get("label") == "conflict"]
    if conflicts:
        st.markdown("**Key contradictions**")
        for p in conflicts[:12]:
            st.markdown(
                f"- **{p.get('claim_a', {}).get('entity', '?')}**: "
                f"A=`{p.get('claim_a', {}).get('value', '')[:80]}` vs "
                f"B=`{p.get('claim_b', {}).get('value', '')[:80]}` — _{p.get('explanation', '')}_"
            )

    st.subheader("Side-by-side excerpts")
    c1, c2 = st.columns(2)
    with c1:
        st.text_area("Article A (preview)", text_a[:4000], height=200, disabled=True)
    with c2:
        st.text_area("Article B (preview)", text_b[:4000], height=200, disabled=True)

    st.subheader("Matched claim pairs")
    if not classified:
        st.warning("No aligned pairs above the similarity threshold. Try lowering ALIGNMENT_MIN_SCORE in .env or use more overlapping articles.")
    for i, p in enumerate(classified):
        ca, cb = p.get("claim_a") or {}, p.get("claim_b") or {}
        label = p.get("label", "unverifiable")
        color = _card_color(label)
        sim = difflib_ratio(str(ca.get("value", "")), str(cb.get("value", "")))
        st.markdown(
            f'<div style="border-left:4px solid {color}; padding:12px; margin-bottom:12px; background:#fafafa;">'
            f"<strong>{label.upper()}</strong> · alignment {p.get('score')} · "
            f"value similarity {sim:.2f} · confidence {float(p.get('confidence', 0)):.2f}<br/>"
            f"<strong>Entity:</strong> {ca.get('entity','')}<br/>"
            f"<strong>A:</strong> {ca.get('value','')} ({ca.get('claim_type','')})<br/>"
            f"<strong>B:</strong> {cb.get('value','')} ({cb.get('claim_type','')})<br/>"
            f"<em>{p.get('explanation','')}</em>"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"difflib value diff (pair {i + 1})", expanded=False):
            st.code(unified_diff_snippet(str(ca.get("value", "")), str(cb.get("value", ""))))

    pdf_bytes = build_pdf_report(score_bundle, classified)
    json_payload = json.dumps(
        {
            "scores": score_bundle,
            "pairs": classified,
            "claims_a": claims_a,
            "claims_b": claims_b,
        },
        ensure_ascii=False,
        indent=2,
    )
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download PDF report", data=pdf_bytes, file_name="contradiction_report.pdf", mime="application/pdf")
    with d2:
        st.download_button("Download JSON", data=json_payload, file_name="contradiction_analysis.json", mime="application/json")


if __name__ == "__main__":
    main()
