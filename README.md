# 📊 Financial News Contradiction Analyzer

An end-to-end Streamlit application that detects, explains, and scores factual inconsistencies between two financial news articles.

## 🚀 Overview

The Financial News Contradiction Analyzer is designed to help analysts, researchers, and compliance teams compare multiple reports of the same financial event and uncover where they diverge.

Instead of asking “Which article is correct?”, this tool answers:

> “Where do two credible sources disagree, and by how much?”

It extracts structured claims from each article, aligns similar claims, and highlights contradictions with confidence scores and explanations.

## ✨ Key Features

### 🔗 Multi-input support
- Paste article URLs (via `newspaper3k`)
- Upload PDFs (via `pdfplumber`)

### 🧠 LLM-powered claim extraction
Extracts structured claims:
- Entity
- Value
- Date
- Claim type (metric, date, quote)
- Source quote (exact supporting text)

### 🔍 Smart claim alignment
- Uses `rapidfuzz` (and optional embeddings) to match related claims across articles

### ⚖️ Hybrid contradiction detection
Rule-based logic for:
- Numeric comparisons (handles million/billion/%/ranges)
- Date comparisons
- Quote similarity
- LLM fallback for ambiguous cases

### 📊 Interactive dashboard (Streamlit)
- Side-by-side comparison
- Color-coded claim cards:
  - 🟢 **Agree**
  - 🔴 **Conflict**
  - 🟡 **Unverifiable**
- Confidence scores
- One-line explanations per comparison
- Overall Reporting Alignment Score

### 📄 Exportable outputs
- JSON results
- PDF summary reports (via `reportlab`)

## 🧱 System Architecture

```text
Input Layer
 ├── URL ingestion (newspaper3k)
 └── PDF ingestion (pdfplumber)

↓
Text Processing

↓
Claim Extraction (LLM)
 └── Structured schema:
     {entity, value, date, claim_type, source_quote}

↓
Claim Alignment
 └── rapidfuzz similarity matching

↓
Contradiction Detection (Hybrid)
 ├── Rule-based engine:
 │     ├── Numeric comparison
 │     ├── Date parsing
 │     └── Quote similarity
 └── LLM fallback (edge cases)

↓
Visualization Layer (Streamlit)
 ├── Claim cards
 ├── Conflict explanations
 └── Alignment score

↓
Export Layer
 ├── JSON
 └── PDF report
```

### 🧠 Core Insight

In financial and legal journalism, truth is rarely binary.

Different credible sources often report:
- Slightly different numbers
- Varying timelines
- Selective quotes

This system embraces that reality by focusing on:
**Quantifying disagreement instead of labeling truth**

## ⚙️ Hybrid Detection Strategy

### ✅ Rule-Based (Primary)

Used when structured comparison is possible:

1. **Numeric Claims**
   - Extract numbers from source_quote
   - Normalize: Million / Billion, Percentages, Ranges (e.g., 13–16%)
   - Compare with tolerance thresholds
2. **Date Claims**
   - Parse and normalize dates
   - Compare: Exact match, Same period (e.g., quarter/year)
3. **Quotes**
   - Fuzzy matching (`rapidfuzz`, `difflib`)
   - Determine: Same quote → Agree, Clearly different → Conflict

### 🤖 LLM Fallback

Triggered when:
- Numeric parsing fails
- Claims lack sufficient detail
- Ambiguity remains after rule checks

## 🛠️ Tech Stack

| Layer | Tools Used |
|-------|------------|
| Frontend UI | Streamlit |
| Article Scraping | `newspaper3k` |
| PDF Parsing | `pdfplumber` |
| Claim Extraction | LLM API |
| Matching Engine | `rapidfuzz` |
| Reporting | `reportlab` |
| Backend | Python |

## 📦 Installation

```bash
git clone https://github.com/your-username/financial-contradiction-analyzer.git
cd financial-contradiction-analyzer

pip install -r requirements.txt
```

## ▶️ Usage

```bash
streamlit run app.py
```

Then:
1. Paste two article URLs or upload PDFs
2. Click **Analyze**
3. Explore:
   - Matched claims
   - Highlighted contradictions
   - Alignment score

## 📊 Output Example

- **Alignment Score:** 72%
- **Conflicts Found:** 5
- **Key Differences:**
  - Revenue: $2.84B vs $2.48B
  - Growth Rate: 14% vs 16%
  - Timeline mismatch in earnings report

## 📁 Project Structure

```text
├── app.py                  # Streamlit UI
├── ingestion/
│   ├── scraper.py         # newspaper3k logic
│   └── pdf_parser.py
├── extraction/
│   └── claim_extractor.py
├── alignment/
│   └── matcher.py
├── detection/
│   ├── numeric_parser.py
│   ├── rule_engine.py
│   └── llm_fallback.py
├── utils/
├── exports/
│   └── pdf_generator.py
└── requirements.txt
```

## 🔮 Future Improvements

- Embedding-based semantic alignment
- Multi-article comparison (beyond 2 sources)
- Real-time news monitoring
- Domain-specific financial ontologies
- Improved unit normalization (currencies, inflation adjustments)

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests.
