# Financial News Contradiction Analyzer

An end-to-end Streamlit application that detects, explains, and scores factual inconsistencies between two financial news articles.

## Overview

The Financial News Contradiction Analyzer helps analysts, researchers, and compliance teams compare multiple reports of the same financial event and systematically identify where they diverge.

Rather than asking “Which article is correct?”, the system focuses on:

> Where do two credible sources disagree, and by how much?

It extracts structured claims from each article, aligns related claims, and highlights contradictions with confidence scores and concise explanations.

## Key Features

- **Flexible input options**
  - Article URLs via `newspaper3k`
  - PDF uploads via `pdfplumber`
- **LLM-driven claim extraction**
  - Extracts structured claims including:
    - Entity
    - Value
    - Date
    - Claim type (metric, date, quote)
    - Source quote (exact supporting text)
- **Intelligent claim alignment**
  - Uses `rapidfuzz` and optional embeddings to match related claims
- **Hybrid contradiction detection**
  - Rule-based logic for numeric, date, and quote comparisons
  - LLM fallback for ambiguous cases
- **Interactive Streamlit dashboard**
  - Side-by-side comparison of claims
  - Classification: agree, conflict, unverifiable
  - Confidence scores and explanations
  - Overall reporting alignment score
- **Export capabilities**
  - JSON output
  - PDF reports via `reportlab`

## System Architecture

```text
Input Layer
├── URL ingestion (newspaper3k)
└── PDF ingestion (pdfplumber)

↓
Text Processing

↓
Claim Extraction (LLM)
└── {entity, value, date, claim_type, source_quote}

↓
Claim Alignment
└── rapidfuzz similarity matching

↓
Contradiction Detection (Hybrid)
├── Rule-based engine
│   ├── Numeric comparison
│   ├── Date parsing
│   └── Quote similarity
└── LLM fallback

↓
Visualization (Streamlit)

↓
Export (JSON, PDF)
```

## Core Insight

In financial and legal journalism, truth is often distributed across multiple credible accounts rather than contained within a single source.

Different reports may present:
- Slightly different numerical values
- Varying timelines
- Selective or paraphrased quotes

This system quantifies and surfaces these differences to support better decision-making.

## Hybrid Detection Strategy

### Rule-Based Detection (Primary)

Applied when structured comparison is feasible.

1. **Numeric Claims**
   - Extract values from `source_quote`
   - Normalize units (million, billion, percentages, ranges)
   - Compare using tolerance thresholds
2. **Date Claims**
   - Parse and standardize dates
   - Compare exact matches or equivalent periods
3. **Quote Claims**
   - Use fuzzy matching (`rapidfuzz`, `difflib`)
   - Classify as:
     - Matching (agree)
     - Distinct (conflict)
     - Ambiguous (unverifiable)

### LLM Fallback

Used only when:
- Numeric parsing fails
- Claims lack sufficient structure
- The comparison remains ambiguous after rule-based checks

This hybrid approach improves accuracy while reducing unnecessary LLM usage.

## Tech Stack

- **Frontend:** Streamlit
- **Scraping:** `newspaper3k`
- **PDF Parsing:** `pdfplumber`
- **Claim Extraction:** Ollama LLM
- **Matching:** `rapidfuzz`
- **Reporting:** `reportlab`
- **Backend:** Python

## Installation

```bash
git clone https://github.com/your-username/financial-contradiction-analyzer.git
cd financial-contradiction-analyzer
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Steps:
1. Provide two article URLs or upload PDF files
2. Run the analysis
3. Review aligned claims, contradictions, and the overall alignment score

## Project Structure

```text
app.py
ingestion/
  scraper.py
  pdf_parser.py
extraction/
  claim_extractor.py
alignment/
  matcher.py
detection/
  numeric_parser.py
  rule_engine.py
  llm_fallback.py
exports/
  pdf_generator.py
utils/
requirements.txt
```

---
Contributions are welcome. Please open an issue or submit a pull request.
