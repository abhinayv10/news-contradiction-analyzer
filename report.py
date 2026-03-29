"""
Downloadable PDF report via reportlab.
"""

from __future__ import annotations

import io
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf_report(
    scores: dict[str, Any],
    classified_pairs: list[dict[str, Any]],
    title: str = "Financial News Contradiction Report",
) -> bytes:
    """Return PDF bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=inch * 0.75,
        leftMargin=inch * 0.75,
        topMargin=inch * 0.75,
        bottomMargin=inch * 0.75,
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=16, spaceAfter=12)
    h2 = ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, spaceAfter=8)
    body = styles["Normal"]

    story: list[Any] = []
    story.append(Paragraph(title, h1))
    story.append(Spacer(1, 0.15 * inch))

    summary_data = [
        ["Metric", "Value"],
        ["Alignment score (%)", str(scores.get("alignment_score", 0))],
        ["Total claims (A / B)", f"{scores.get('total_claims_article_a', 0)} / {scores.get('total_claims_article_b', 0)}"],
        ["Matched pairs", str(scores.get("matched_claims", 0))],
        ["Agreements", str(scores.get("agreements", 0))],
        ["Conflicts", str(scores.get("conflicting_claims", 0))],
        ["Unverifiable pairs", str(scores.get("unverifiable_pairs", 0))],
    ]
    t = Table(summary_data, colWidths=[3 * inch, 3 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Contradictions &amp; comparisons", h2))

    conflicts = [p for p in classified_pairs if p.get("label") == "conflict"]
    rows = [["Entity", "Article A value", "Article B value", "Confidence", "Explanation"]]
    for p in conflicts[:50]:
        ca = p.get("claim_a") or {}
        cb = p.get("claim_b") or {}
        rows.append(
            [
                str(ca.get("entity", ""))[:40],
                str(ca.get("value", ""))[:45],
                str(cb.get("value", ""))[:45],
                f"{float(p.get('confidence', 0)):.2f}",
                str(p.get("explanation", ""))[:120],
            ]
        )

    if len(rows) == 1:
        story.append(Paragraph("No conflicting pairs detected in matched claims.", body))
    else:
        ct = Table(rows, colWidths=[1.1 * inch, 1.35 * inch, 1.35 * inch, 0.7 * inch, 1.5 * inch])
        ct.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C00000")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.2, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(ct)

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("All matched pairs (summary)", h2))
    all_rows = [["Label", "Entity", "Confidence", "Explanation"]]
    for p in classified_pairs[:80]:
        ca = p.get("claim_a") or {}
        all_rows.append(
            [
                str(p.get("label", "")),
                str(ca.get("entity", ""))[:35],
                f"{float(p.get('confidence', 0)):.2f}",
                str(p.get("explanation", ""))[:100],
            ]
        )
    if len(all_rows) > 1:
        at = Table(all_rows, colWidths=[0.9 * inch, 1.5 * inch, 0.7 * inch, 3 * inch])
        at.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#70AD47")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.15, colors.lightgrey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(at)

    doc.build(story)
    return buf.getvalue()
