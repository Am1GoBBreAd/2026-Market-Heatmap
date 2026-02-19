# 📊 Investment Bank Sentiment Heatmap (2026 Outlook)

Extract text from investment-bank outlook PDFs, map language to allocation sentiment, and visualize cross-bank views in an interactive heatmap.

## Step-by-step implementation

### Step 1: Extract text from PDFs
The pipeline uses:
- **Primary**: `PyMuPDF` (`fitz`) for robust extraction from complex reports.
- **Fallback**: `pypdf` if PyMuPDF is unavailable.

### Step 2: Extract sentiment labels
Supported sentiment labels:
- `Overweight`
- `Slight Over`
- `Neutral`
- `Slight Under`
- `Underweight`

Two classification modes:
- `keyword` (default): regex over finance terms like overweight/bullish/neutral/underweight.
- `finbert`: uses `ProsusAI/finbert` from Hugging Face and maps positive/neutral/negative into overweight/neutral/underweight.

### Step 3: Build matrix
The script builds a **bank × asset** matrix and maps labels to scores:
- `Underweight = -2`
- `Slight Under = -1`
- `Neutral = 0`
- `Slight Over = 1`
- `Overweight = 2`

### Step 4: Build interactive heatmap
`plotly.express.imshow()` renders the matrix as an interactive heatmap.
Hover on any cell to inspect:
- bank name
- asset class
- normalized sentiment label
- original evidence sentence

---

## Installation

```bash
pip install -r requirements.txt
```

> If you plan to use `--method finbert`, also install:

```bash
pip install transformers torch
```

## Usage

1) Put PDF files in `pdfs/` (or any folder):

```text
pdfs/
  Goldman_Sachs.pdf
  JPMorgan.pdf
  Morgan_Stanley.pdf
```

2) Run keyword-based extraction (default):

```bash
python heatmap_pipeline.py \
  --pdf-dir pdfs \
  --assets "US Equities" "Emerging Markets" "Fixed Income" "Commodities" \
  --output-html sentiment_heatmap.html \
  --output-csv sentiment_extraction.csv
```

3) Run FinBERT-based extraction:

```bash
python heatmap_pipeline.py \
  --pdf-dir pdfs \
  --assets "US Equities" "Emerging Markets" "Fixed Income" "Commodities" \
  --method finbert \
  --output-html sentiment_heatmap.html \
  --output-csv sentiment_extraction.csv
```

## Outputs
- `sentiment_heatmap.html`: interactive heatmap
- `sentiment_extraction.csv`: extracted rows with label, evidence, and method
