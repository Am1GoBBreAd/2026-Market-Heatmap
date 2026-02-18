# 📊 Investment Bank Sentiment Heatmap (2026 Outlook)

This project is a Python-based visualization tool that generates an interactive heatmap of global market sentiments using **Plotly**. It is designed to aggregate "House Views" from major investment banks (like BofA, Goldman Sachs, J.P. Morgan, and UBS) to identify market consensus and institutional dispersion for the year 2026.


## Introduction 💡

This project simplifies that process by translating qualitative analyst views—such as "Cautiously Optimistic" or "Relatively Bearish"—into a quantitative 5-point scale. It provides a quick overview of institutional positioning across key markets like the S&P 500, China, and Japan.
By using Plotly, this script converts complex analyst reports into an interactive dashboard where users can identify:
- Market Consensus: Which regions have unanimous "Bullish" or "Bearish" views.
- Institutional Dispersion: Where banks have conflicting outlooks (e.g., China Market).

## Installation ⚙️

The required packages to run this code can be found in the `requirements.txt` file. If you are using **GitHub Codespaces**, run the following command in your terminal:

```bash
pip install plotly pandas
```

## Usage ⌨️
To generate the interactive heatmap and view it in your browser, execute the script:
```bash
python heatmap.py
```

## Sentiment Scale Reference 📊
The heatmap uses a standardized color-coded scale to represent institutional positioning based on the original research:

| Score        | Label                          | Color   |
|----------------|--------------------------------------|----------|
| **2**      | Overweight / Bullish     | 🟧 Dark Orange  |
| **1**     | Slight Overweight                  | 🔸 Light Orange |
| **0**  | Neutral / Benchmark | ⬜ Light Grey |
| **-1**  | Slight Underweight | 🔹 Light Green |
| **-2**  | Underweight / Bearish | 🟩 Dark Green |
