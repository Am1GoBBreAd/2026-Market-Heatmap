from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px

try:
    import fitz  # PyMuPDF
except Exception:  # optional dependency at runtime
    fitz = None

try:
    from pypdf import PdfReader
except Exception:  # optional dependency at runtime
    PdfReader = None

SENTIMENT_ORDER = [
    "Underweight",
    "Slight Under",
    "Neutral",
    "Slight Over",
    "Overweight",
]

SENTIMENT_TO_SCORE = {
    "Underweight": -2,
    "Slight Under": -1,
    "Neutral": 0,
    "Slight Over": 1,
    "Overweight": 2,
}

KEYWORD_PATTERNS = [
    (r"\bstrong\s+underweight\b|\bunderweight\b|\bbearish\b", "Underweight"),
    (r"\bslight(?:ly)?\s+under(?:weight)?\b|\bcautious\b", "Slight Under"),
    (r"\bneutral\b|\bbenchmark\b|\bmarket\s+weight\b", "Neutral"),
    (r"\bslight(?:ly)?\s+over(?:weight)?\b|\bcautiously\s+optimistic\b", "Slight Over"),
    (r"\boverweight\b|\bbullish\b|\bstrong\s+buy\b", "Overweight"),
]


@dataclass
class BankView:
    bank: str
    asset: str
    label: str
    evidence: str


def extract_text(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF first, then pypdf fallback."""
    if fitz is not None:
        doc = fitz.open(pdf_path)
        try:
            return "\n".join(page.get_text("text") for page in doc)
        finally:
            doc.close()

    if PdfReader is not None:
        reader = PdfReader(str(pdf_path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    raise RuntimeError("No PDF parser available. Install pymupdf or pypdf.")


def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def classify_sentence(sentence: str) -> str | None:
    lower = sentence.lower()
    for pattern, label in KEYWORD_PATTERNS:
        if re.search(pattern, lower):
            return label
    return None


def extract_bank_sentiment(bank: str, text: str, assets: Iterable[str]) -> list[BankView]:
    sentences = split_sentences(text)
    rows: list[BankView] = []

    for asset in assets:
        asset_matches = [s for s in sentences if asset.lower() in s.lower()]
        chosen_label = "Neutral"
        evidence = "No explicit sentiment found; defaulted to Neutral."

        for sentence in asset_matches:
            label = classify_sentence(sentence)
            if label is not None:
                chosen_label = label
                evidence = sentence
                break

        rows.append(BankView(bank=bank, asset=asset, label=chosen_label, evidence=evidence))

    return rows


def build_matrix(rows: list[BankView]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame([r.__dict__ for r in rows])
    label_matrix = df.pivot(index="bank", columns="asset", values="label")
    numeric_matrix = label_matrix.replace(SENTIMENT_TO_SCORE)
    return label_matrix, numeric_matrix


def build_hover_text(label_df: pd.DataFrame, evidence_df: pd.DataFrame) -> list[list[str]]:
    hover = []
    for bank in label_df.index:
        row = []
        for asset in label_df.columns:
            row.append(
                f"Bank: {bank}<br>Asset: {asset}<br>Label: {label_df.loc[bank, asset]}"
                f"<br>Evidence: {evidence_df.loc[bank, asset]}"
            )
        hover.append(row)
    return hover


def create_interactive_heatmap(label_df: pd.DataFrame, score_df: pd.DataFrame, evidence_df: pd.DataFrame, output_html: Path) -> None:
    hover_text = build_hover_text(label_df, evidence_df)

    fig = px.imshow(
        score_df,
        labels={"x": "Asset Class", "y": "Investment Bank", "color": "Sentiment"},
        color_continuous_scale="RdYlGn",
        zmin=-2,
        zmax=2,
        title="Global Market Outlook Sentiment Heatmap",
        text_auto=True,
    )
    fig.update_traces(
        customdata=hover_text,
        hovertemplate="%{customdata}<extra></extra>",
    )
    fig.write_html(str(output_html), include_plotlyjs="cdn")


def run(pdf_dir: Path, assets: list[str], output_html: Path, output_csv: Path) -> None:
    rows: list[BankView] = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        bank = pdf_path.stem
        text = clean_text(extract_text(pdf_path))
        rows.extend(extract_bank_sentiment(bank, text, assets))

    if not rows:
        raise RuntimeError(f"No PDF files found in {pdf_dir}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    label_df, score_df = build_matrix(rows)
    evidence_df = df.pivot(index="bank", columns="asset", values="evidence")

    create_interactive_heatmap(label_df, score_df, evidence_df, output_html)
    df.to_csv(output_csv, index=False)

    print(f"Saved detailed extraction CSV: {output_csv}")
    print(f"Saved interactive heatmap: {output_html}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract bank outlook sentiment from PDFs and build interactive heatmap.")
    parser.add_argument("--pdf-dir", type=Path, default=Path("pdfs"), help="Directory containing bank outlook PDFs.")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["US Equities", "Emerging Markets", "Fixed Income", "Commodities"],
        help="Asset classes to track.",
    )
    parser.add_argument("--output-html", type=Path, default=Path("sentiment_heatmap.html"))
    parser.add_argument("--output-csv", type=Path, default=Path("sentiment_extraction.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.pdf_dir, args.assets, args.output_html, args.output_csv)
