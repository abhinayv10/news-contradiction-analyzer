"""
URL and PDF text ingestion with normalization.

Uses newspaper3k for articles and pdfplumber for PDFs.
"""

from __future__ import annotations

import io
import re
from typing import BinaryIO

import pdfplumber


def normalize_text(raw: str) -> str:
    """
    Remove boilerplate-ish noise and normalize whitespace.
    Keeps content; strips excessive newlines and common web cruft patterns.
    """
    if not raw:
        return ""
    # Collapse whitespace
    t = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Remove zero-width and BOM
    t = re.sub(r"[\u200b\ufeff]", "", t)
    # Drop lines that look like lone navigation/footer tokens (heuristic)
    lines = []
    for line in t.split("\n"):
        s = line.strip()
        if len(s) < 2:
            continue
        if re.fullmatch(r"(share|tweet|subscribe|cookie|privacy policy)[\s.]*", s, re.I):
            continue
        lines.append(s)
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def fetch_article_from_url(url: str) -> str:
    """Download and parse article body via newspaper3k."""
    from newspaper import Article, Config

    cfg = Config()
    cfg.browser_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    cfg.request_timeout = 30
    art = Article(url, config=cfg)
    art.download()
    art.parse()
    text = art.text or ""
    return normalize_text(text)


def extract_text_from_pdf(file: BinaryIO | bytes) -> str:
    """Extract text from a PDF.

    Tries `pdfplumber` first (fast when PDFs are well-formed). If that produces
    little text or fails due to malformed fonts, falls back to PyMuPDF.
    """
    buf = file if isinstance(file, bytes) else file.read()
    primary = ""
    try:
        parts: list[str] = []
        with pdfplumber.open(io.BytesIO(buf)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
        primary = normalize_text("\n\n".join(parts))
    except Exception:
        primary = ""

    # Heuristic: if pdfplumber returned almost nothing, use PyMuPDF.
    if len(primary.strip()) >= 200:
        return primary

    fallback = ""
    try:
        import fitz  # PyMuPDF

        parts: list[str] = []
        doc = fitz.open(stream=buf, filetype="pdf")
        try:
            for page in doc:
                t = page.get_text() or ""
                if t.strip():
                    parts.append(t)
        finally:
            doc.close()
        fallback = normalize_text("\n\n".join(parts))
    except Exception:
        fallback = ""

    if len(fallback.strip()) > len(primary.strip()) + 20:
        return fallback
    return primary or fallback
