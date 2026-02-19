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
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

SENTIMENT_TO_SCORE = {
    "Underweight": -2,
    "Slight Under": -1,
    "Neutral": 0,
    "Slight Over": 1,
    "Overweight": 2,
}

FINBERT_TO_SENTIMENT = {
    "negative": "Underweight",
    "neutral": "Neutral",
    "positive": "Overweight",
}

KEYWORD_PATTERNS = [
    (r"\bstrong\s+underweight\b|\bunderweight\b|\bbearish\b", "Underweight"),
    (r"\bslight(?:ly)?\s+under(?:weight)?\b|\bcautious\b", "Slight Under"),
    (r"\bneutral\b|\bbenchmark\b|\bmarket\s+weight\b", "Neutral"),
    (r"\bslight(?:ly)?\s+over(?:weight)?\b|\bcautiously\s+optimistic\b", "Slight Over"),
    (r"\bstrong\s+overweight\b|\boverweight\b|\bbullish\b|\bstrong\s+buy\b", "Overweight"),
]


@dataclass
class BankView:
    bank: str
    asset: str
    label: str
    evidence: str
    method: str


def extract_text(pdf_path: Path) -> str:
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
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def classify_keyword(sentence: str) -> str | None:
    lower = sentence.lower()
    for pattern, label in KEYWORD_PATTERNS:
        if re.search(pattern, lower):
            return label
    return None


def build_finbert_classifier():
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError(
            "FinBERT requested but transformers is not installed. Install with: pip install transformers torch"
        ) from exc

    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def classify_finbert(sentence: str, clf) -> str:
    result = clf(sentence, truncation=True)[0]["label"].lower()
    return FINBERT_TO_SENTIMENT.get(result, "Neutral")


def find_asset_sentences(sentences: Iterable[str], asset: str) -> list[str]:
    asset_aliases = [asset, asset.replace(" ", ""), asset.split()[0]]
    matched = []
    for s in sentences:
        sl = s.lower()
        if any(alias.lower() in sl for alias in asset_aliases):
            matched.append(s)
    return matched


def extract_bank_sentiment(bank: str, text: str, assets: Iterable[str], method: str, finbert_clf=None) -> list[BankView]:
    sentences = split_sentences(text)
    rows: list[BankView] = []

    for asset in assets:
        asset_matches = find_asset_sentences(sentences, asset)
        chosen_label = "Neutral"
        evidence = "No explicit sentiment found; defaulted to Neutral."

        for sentence in asset_matches:
            if method == "finbert":
                label = classify_finbert(sentence, finbert_clf)
            else:
                label = classify_keyword(sentence)

            if label is not None:
                chosen_label = label
                evidence = sentence
                break

        rows.append(BankView(bank=bank, asset=asset, label=chosen_label, evidence=evidence, method=method))

    return rows


def build_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_matrix = df.pivot(index="bank", columns="asset", values="label")
    numeric_matrix = label_matrix.replace(SENTIMENT_TO_SCORE)
    evidence_matrix = df.pivot(index="bank", columns="asset", values="evidence")
    return label_matrix, numeric_matrix, evidence_matrix


def create_interactive_heatmap(label_df: pd.DataFrame, score_df: pd.DataFrame, evidence_df: pd.DataFrame, output_html: Path) -> None:
    fig = px.imshow(
        score_df,
        labels={"x": "Asset Class", "y": "Investment Bank", "color": "Sentiment Score"},
        color_continuous_scale="RdYlGn",
        zmin=-2,
        zmax=2,
        title="Global Market Outlook Sentiment Heatmap",
    )

    customdata = []
    text = []
    for bank in label_df.index:
        custom_row = []
        text_row = []
        for asset in label_df.columns:
            label = label_df.loc[bank, asset]
            text_row.append(label)
            custom_row.append([bank, asset, label, evidence_df.loc[bank, asset]])
        customdata.append(custom_row)
        text.append(text_row)

    fig.update_traces(
        text=text,
        texttemplate="%{text}",
        customdata=customdata,
        hovertemplate=(
            "Bank: %{customdata[0]}<br>"
            "Asset: %{customdata[1]}<br>"
            "Label: %{customdata[2]}<br>"
            "Evidence: %{customdata[3]}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(xaxis_side="top")
    fig.write_html(str(output_html), include_plotlyjs="cdn")


def run(pdf_dir: Path, assets: list[str], output_html: Path, output_csv: Path, method: str) -> None:
    rows: list[BankView] = []
    finbert_clf = build_finbert_classifier() if method == "finbert" else None

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        bank = pdf_path.stem
        text = clean_text(extract_text(pdf_path))
        rows.extend(extract_bank_sentiment(bank, text, assets, method=method, finbert_clf=finbert_clf))

    if not rows:
        raise RuntimeError(f"No PDF files found in {pdf_dir}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    label_df, score_df, evidence_df = build_matrix(df)

    create_interactive_heatmap(label_df, score_df, evidence_df, output_html)
    df.to_csv(output_csv, index=False)

    print(f"Saved detailed extraction CSV: {output_csv}")
    print(f"Saved interactive heatmap: {output_html}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract bank outlook sentiment from PDFs and build interactive heatmap."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("pdfs"),
        help="Directory containing bank outlook PDFs (default: ./pdfs).",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["US Equities", "Emerging Markets", "Fixed Income", "Commodities"],
        help="Asset classes to track, e.g. --assets 'US Equities' 'EM' 'Bonds'.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("sentiment_heatmap.html"),
        help="Path to output interactive heatmap HTML.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("sentiment_extraction.csv"),
        help="Path to output extracted sentiment rows as CSV.",
    )
    parser.add_argument(
        "--method",
        choices=["keyword", "finbert"],
        default="keyword",
        help="Sentiment classification method: keyword (default) or finbert.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.pdf_dir, args.assets, args.output_html, args.output_csv, method=args.method)
