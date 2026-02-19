# 📊 Investment Bank Sentiment Heatmap Pipeline

This project automates a yearly workflow for reading investment bank outlook reports (for example, **2026 outlook PDFs**) and turning them into a structured market-sentiment heatmap.

## What this pipeline does

- Reads outlook PDFs from a folder
- Extracts text from each PDF
- Uses OpenAI API to convert unstructured text into structured sentiment signals
- Aggregates cross-bank **consensus** and **dispersion** by market
- Generates an interactive Plotly heatmap
- Saves machine-readable JSON outputs

## Why this is useful

Instead of manually comparing many institutional reports every year, this pipeline gives you:

- A quick view of where banks align (consensus)
- A quick view of where they disagree (dispersion)
- A reusable process you can run for 2026, 2027, and beyond

## Installation

```bash
python -m pip install -r requirements.txt
```

## Input format

Put your bank PDFs in:

```text
data/pdfs/
```

Examples:
- `Goldman_Sachs_2026_outlook.pdf`
- `UBS_2026_strategy.pdf`
- `JPM_2026_house_view.pdf`

## Run (real OpenAI extraction)

Set your API key:

```bash
export OPENAI_API_KEY="your_key_here"
```

Or pass it directly at runtime:

```bash
python sentiment_pipeline.py --input-dir data/pdfs --output-dir outputs --year 2026 --model gpt-4.1-mini --api-key "your_key_here"
```

Then run:

```bash
python sentiment_pipeline.py --input-dir data/pdfs --output-dir outputs --year 2026 --model gpt-4.1-mini
```

If you see `Missing OpenAI API key`, either set `OPENAI_API_KEY`, pass `--api-key`, or run `--mock` for local testing without API calls.


### GitHub Codespaces API key setup (recommended)

1. In GitHub repo page: **Settings → Secrets and variables → Codespaces**
2. Create a new secret named `OPENAI_API_KEY`
3. Rebuild or restart the codespace terminal session
4. Verify key is available:

```bash
echo $OPENAI_API_KEY
```

### Local `.env` fallback

You can also create a `.env` file in the repo root (this file is ignored by git):

```bash
cp .env.example .env
# edit .env and paste your real key
```

The script automatically checks this order: `--api-key` → `OPENAI_API_KEY` env var → `.env` file.

## Run (mock mode, no API calls)

Useful for testing the chart and file outputs quickly:

```bash
python sentiment_pipeline.py --input-dir data/pdfs --output-dir outputs --year 2026 --mock
```

## Outputs

The pipeline writes:

- `outputs/sentiment_by_document.json`
  - Structured extraction per PDF
- `outputs/consensus_summary.json`
  - Market-level average sentiment + dispersion
- `outputs/sentiment_heatmap.html`
  - Interactive Plotly heatmap

## Sentiment score scale

| Score | Label |
|---:|---|
| -2 | Underweight / Bearish |
| -1 | Slight Underweight |
| 0 | Neutral / Benchmark |
| 1 | Slight Overweight |
| 2 | Overweight / Bullish |

## Notes

- The extractor is designed for **explicit** market views (avoid inferring unsupported views).
- If model output quality varies, try a stronger model or restrict PDF pages with `--max-pages`.
