# Financial News Contradiction Analyzer

A sleek, production-oriented prototype designed to ingest two financial news articles (URLs or PDFs), extract structured claims using an LLM, align claims with **rapidfuzz** (and optional embeddings), classify agreement vs. contradiction, and review the results in a modern, polished **Streamlit** dashboard. Results can be exported as a **reportlab** PDF or JSON.

## Features

- **Polished User Interface:** Modern, premium Streamlit design for an enhanced user experience.
- **Multi-format Support:** Ingest articles via URLs or direct PDF uploads.
- **LLM-Powered Extraction:** Structured claim extraction and advanced agreement/contradiction classification.
- **Robust Alignment:** Rapidfuzz and optional embeddings used for high-accuracy claim matching.
- **Detailed Export:** Downloadable PDF reports and JSON payloads.

## Setup

```bash
cd financial-news-contradiction-analyzer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `LLM_API_KEY` and optionally `LLM_BASE_URL` / `LLM_MODEL` for any OpenAI-compatible API.

## Run the app

```bash
streamlit run app.py
```

## Example run (two URLs)

Use two articles that cover the **same** corporate event (earnings, M&A, guidance) so claims overlap.

1. Start the app and choose **URLs**.
2. Paste two finance URLs (Reuters, Bloomberg, WSJ, etc.) that discuss the same story.
3. Click **Analyze**.

**Sample pair (illustrative — replace with current live URLs):**

- Article A: a Reuters piece on a company’s quarterly results  
- Article B: a second outlet’s recap of the **same quarter** for the **same company**

If outlets use different rounding (e.g. $5B vs $5.1B), the classifier should mark **agree** or **conflict** depending on tolerance; **unverifiable** appears when alignment is weak or facts are missing on one side.

## Configuration

| Variable | Purpose |
|----------|---------|
| `LLM_API_KEY` | API key |
| `LLM_BASE_URL` | OpenAI-compatible base (default OpenAI) |
| `LLM_MODEL` | Chat model for extraction + classification |
| `LLM_TEMPERATURE` | Use `0` for deterministic JSON |
| `LLM_EMBEDDING_MODEL` | Optional; boosts alignment when set |
| `DEMO_MODE` | If `true`, uses fast heuristic extraction + rules-only labeling (no LLM calls). |
| `AMOUNT_REL_TOLERANCE` | Relative numeric tolerance for money/amount agreement. |
| `PERCENT_ABS_TOLERANCE` | Absolute numeric tolerance for percent agreement (in percentage points). |
| `CACHE_DIR` | LLM response cache (`.cache`); delete to invalidate |
| `ALIGNMENT_MIN_SCORE` | Minimum fuzzy+embedding score (0–100) to pair claims |
| `CLAIM_CHUNK_CHARS` | Max characters per LLM chunk |

## Architecture

| Module | Role |
|--------|------|
| `app.py` | Streamlit UI |
| `ingestion.py` | `newspaper3k` URLs, `pdfplumber` PDFs, text normalize |
| `claim_extractor.py` | Chunked LLM extraction, claim deduplication |
| `aligner.py` | `rapidfuzz` + optional embeddings, one-to-one matching |
| `classifier.py` | LLM labels: agree / conflict / unverifiable |
| `scoring.py` | Counts and alignment score |
| `report.py` | PDF download |
| `diff_utils.py` | `difflib` similarity for UI |
| `llm_client.py` | Configurable client + optional disk cache |
| `prompts.py` | Prompt templates |

## Notes

- First run with **newspaper3k** may download NLTK data; ensure network access.
- Some sites block scrapers; if a URL fails, try another source or use PDFs.
