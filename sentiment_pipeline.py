#!/usr/bin/env python3
"""Investment bank outlook sentiment pipeline.

Features:
- Reads outlook PDFs from a folder
- Extracts text
- Uses OpenAI API for structured sentiment extraction
- Aggregates consensus + dispersion
- Generates an interactive Plotly heatmap
- Saves JSON outputs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from pypdf import PdfReader


SCORE_LABELS = {
    -2: "Underweight / Bearish",
    -1: "Slight Underweight",
    0: "Neutral / Benchmark",
    1: "Slight Overweight",
    2: "Overweight / Bullish",
}

SYSTEM_PROMPT = """You are a financial document extraction assistant.
Extract only explicit asset-class/market outlook views from the provided text.
Return strict JSON with this exact schema:
{
  "institution": "string",
  "year": 2026,
  "views": [
    {
      "market": "string",
      "score": -2|-1|0|1|2,
      "rationale": "short quote or summary"
    }
  ]
}
Rules:
- Infer institution from document if possible, otherwise use filename hint.
- Use only markets explicitly mentioned.
- score mapping: -2 very bearish/underweight, -1 slightly bearish, 0 neutral, 1 slightly bullish, 2 bullish/overweight.
- Return valid JSON only, no markdown.
"""


@dataclass
class DocResult:
    source_file: str
    institution: str
    year: int
    views: list[dict[str, Any]]


def extract_text_from_pdf(pdf_path: Path, max_pages: int | None = None) -> str:
    reader = PdfReader(str(pdf_path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    chunks = []
    for page in pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def parse_score(value: Any) -> int:
    if isinstance(value, (int, float)):
        v = int(value)
    else:
        v = int(str(value).strip())
    if v not in SCORE_LABELS:
        raise ValueError(f"Invalid score {v}")
    return v


def infer_institution_from_filename(path: Path) -> str:
    name = path.stem
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\b(20\d{2}|outlook|report|strategy)\b", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip() or path.stem


def extract_structured_sentiment(
    client: OpenAI,
    raw_text: str,
    filename_hint: str,
    year: int,
    model: str,
) -> DocResult:
    prompt = (
        f"Filename hint: {filename_hint}\n"
        f"Target year: {year}\n"
        "Document text (possibly truncated):\n"
        f"{raw_text[:120_000]}"
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    raw_output = response.output_text.strip()
    data = json.loads(raw_output)
    institution = data.get("institution") or infer_institution_from_filename(Path(filename_hint))
    views: list[dict[str, Any]] = []

    for view in data.get("views", []):
        market = str(view.get("market", "")).strip()
        if not market:
            continue
        score = parse_score(view.get("score", 0))
        rationale = str(view.get("rationale", "")).strip()
        views.append({"market": market, "score": score, "rationale": rationale})

    return DocResult(
        source_file=filename_hint,
        institution=institution,
        year=int(data.get("year", year)),
        views=views,
    )


def compute_consensus(doc_results: list[DocResult]) -> dict[str, dict[str, float]]:
    by_market: dict[str, list[int]] = defaultdict(list)
    for doc in doc_results:
        for view in doc.views:
            by_market[view["market"]].append(view["score"])

    consensus: dict[str, dict[str, float]] = {}
    for market, values in sorted(by_market.items()):
        consensus[market] = {
            "avg_score": round(mean(values), 3),
            "dispersion": round(pstdev(values), 3) if len(values) > 1 else 0.0,
            "n_institutions": len(values),
        }
    return consensus


def build_heatmap_dataframe(doc_results: list[DocResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for doc in doc_results:
        for view in doc.views:
            rows.append(
                {
                    "institution": doc.institution,
                    "market": view["market"],
                    "score": view["score"],
                    "rationale": view["rationale"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["institution", "market", "score", "rationale"])

    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(index="institution", columns="market", values="score", aggfunc="mean")
    return pivot.sort_index()


def create_interactive_heatmap(
    pivot_table: pd.DataFrame,
    output_html: Path,
    year: int,
    consensus: dict[str, dict[str, float]],
) -> None:
    if pivot_table.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No sentiment data extracted.", showarrow=False)
        empty_fig.write_html(str(output_html), include_plotlyjs="cdn")
        return

    z = pivot_table.values
    x = list(pivot_table.columns)
    y = list(pivot_table.index)

    hover_text = []
    for yi, institution in enumerate(y):
        row = []
        for xi, market in enumerate(x):
            score = z[yi][xi]
            if np.isnan(score):
                row.append(f"{institution}<br>{market}<br>No coverage")
                continue
            c = consensus.get(market, {})
            row.append(
                f"<b>{institution}</b><br>Market: {market}<br>Score: {score:.1f} ({SCORE_LABELS[int(round(score))]})"
                f"<br>Consensus Avg: {c.get('avg_score', math.nan)}"
                f"<br>Dispersion σ: {c.get('dispersion', math.nan)}"
            )
        hover_text.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            zmin=-2,
            zmax=2,
            text=np.where(np.isnan(z), "", z.astype("float").round(1).astype(str)),
            texttemplate="%{text}",
            textfont={"size": 13},
            colorscale=[
                [0.00, "#145a32"],
                [0.25, "#7cb342"],
                [0.50, "#d9d9d9"],
                [0.75, "#f5b041"],
                [1.00, "#e65100"],
            ],
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
            colorbar={
                "title": "Sentiment",
                "tickvals": [-2, -1, 0, 1, 2],
                "ticktext": [
                    "Underweight / Bearish",
                    "Slight Underweight",
                    "Neutral / Benchmark",
                    "Slight Overweight",
                    "Overweight / Bullish",
                ],
            },
        )
    )

    fig.update_layout(
        title=f"{year} Institutional Investment Sentiment Matrix",
        xaxis_title="Target Market",
        yaxis_title="Investment Bank",
        template="plotly_white",
        width=1400,
        height=800,
        margin={"l": 140, "r": 40, "t": 90, "b": 80},
    )

    fig.write_html(str(output_html), include_plotlyjs="cdn")


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    year: int,
    model: str,
    max_pages: int | None,
    use_mock: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(input_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    client = OpenAI() if not use_mock else None

    doc_results: list[DocResult] = []

    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path, max_pages=max_pages)
        if use_mock:
            inst = infer_institution_from_filename(pdf_path)
            mock_markets = ["US Equities", "EU Equities", "EM/Asia", "China Market", "Japan Market"]
            views = []
            for i, m in enumerate(mock_markets):
                score = ((len(inst) + i) % 5) - 2
                views.append({"market": m, "score": score, "rationale": "Mock extraction"})
            result = DocResult(pdf_path.name, inst, year, views)
        else:
            result = extract_structured_sentiment(
                client=client, raw_text=text, filename_hint=pdf_path.name, year=year, model=model
            )

        doc_results.append(result)

    consensus = compute_consensus(doc_results)
    pivot = build_heatmap_dataframe(doc_results)

    per_document = [
        {
            "source_file": doc.source_file,
            "institution": doc.institution,
            "year": doc.year,
            "views": doc.views,
        }
        for doc in doc_results
    ]

    save_json(per_document, output_dir / "sentiment_by_document.json")
    save_json(consensus, output_dir / "consensus_summary.json")
    create_interactive_heatmap(pivot, output_dir / "sentiment_heatmap.html", year, consensus)

    print(f"Processed {len(doc_results)} PDF(s).")
    print(f"Saved outputs to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract bank sentiment from PDFs and build a heatmap.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/pdfs"), help="Folder containing outlook PDFs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Where JSON + HTML are written")
    parser.add_argument("--year", type=int, default=2026, help="Target outlook year")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model for extraction")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional page limit per PDF")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run without OpenAI API calls; generates deterministic mock sentiment by filename",
    )

    args = parser.parse_args()
    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        year=args.year,
        model=args.model,
        max_pages=args.max_pages,
        use_mock=args.mock,
    )


if __name__ == "__main__":
    main()
