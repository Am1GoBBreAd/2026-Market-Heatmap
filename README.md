# 📊 Investment Bank Sentiment Heatmap (2026 Outlook)

This project extracts outlook language from investment-bank PDF reports, maps views into a 5-level allocation scale, and renders an interactive Plotly heatmap.

## End-to-end workflow

### 1) Extract text from PDFs
Use `PyMuPDF` (`fitz`) as the primary parser and `pypdf` as fallback.

```python
import fitz

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text
```

In this repo, that logic is implemented in `extract_text()` inside `heatmap_pipeline.py`.

### 2) Convert language to sentiment labels
The script uses a practical, finance-focused keyword classifier to map outlook phrases into:

- `Overweight`
- `Slight Over`
- `Neutral`
- `Slight Under`
- `Underweight`

Examples:

- “overweight”, “bullish” → `Overweight`
- “slightly underweight”, “cautious” → `Slight Under`
- “benchmark”, “market weight” → `Neutral`

> Optional extension: replace classifier with FinBERT for more context-aware classification.

### 3) Build the matrix
The extraction output is transformed into a bank × asset matrix and mapped to numeric scores:

- `Underweight = -2`
- `Slight Under = -1`
- `Neutral = 0`
- `Slight Over = 1`
- `Overweight = 2`

### 4) Visualize interactively
`plotly.express.imshow()` creates an interactive heatmap. Hover tooltips show the bank, asset, normalized label, and source evidence sentence.

---

## Installation ⚙️

```bash
pip install -r requirements.txt
```

## Usage ⌨️

1. Put bank outlook PDFs into a folder (default: `pdfs/`), e.g.:

```text
pdfs/
  Goldman_Sachs.pdf
  JPMorgan.pdf
  Morgan_Stanley.pdf
```

2. Run pipeline:

```bash
python heatmap_pipeline.py \
  --pdf-dir pdfs \
  --assets "US Equities" "Emerging Markets" "Fixed Income" "Commodities" \
  --output-html sentiment_heatmap.html \
  --output-csv sentiment_extraction.csv
```

3. Open `sentiment_heatmap.html` in your browser.

## Output artifacts

- `sentiment_extraction.csv`: extracted label + evidence rows by bank/asset
- `sentiment_heatmap.html`: interactive matrix/heatmap

## Project checklist ✅

- Collect PDF outlooks from target banks.
- Pre-process and extract text.
- Focus on allocation-related sections/sentences.
- Map to 5-point sentiment labels.
- Build consensus heatmap for cross-bank comparison.
