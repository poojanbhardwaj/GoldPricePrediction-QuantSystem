# Multi-Asset Market Research & Risk Analytics Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square\&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square\&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square\&logo=scikitlearn)
![Pytest](https://img.shields.io/badge/Tests-Pytest-green?style=flat-square)
![Research Only](https://img.shields.io/badge/Status-Research%20Only-lightgrey?style=flat-square)

A research-only platform for multi-asset forecasting, signal analysis, risk intelligence, historical replay, and benchmark auditing across financial markets.

The project is designed as an engineering and research system, not as a trading shortcut. It focuses on time-series-safe validation, reproducible outputs, conservative risk controls, and honest benchmark comparison.

> **Disclaimer:** This project is for educational and research purposes only. It is not financial advice, does not guarantee profits, and does not execute real-money trades.

---

## Overview

This project started as a gold price prediction system and was expanded into a multi-asset market research platform. It combines data processing, feature engineering, forecasting, signal research, paper-tracking logic, risk controls, regime analysis, benchmark comparison, and Streamlit-based visualization.

The main goal is to answer a practical research question:

> Can a forecasting and signal pipeline produce evidence that is strong enough to beat simple baselines after costs, risk controls, regime filters, and validation checks?

The system is intentionally conservative. It separates research signals from real-money decisions and blocks real-capital recommendations unless strict evidence gates are satisfied.

---

## Supported Assets

The platform is structured around multiple asset classes:

| Asset     | Category     |
| --------- | ------------ |
| Gold      | Commodity    |
| Silver    | Commodity    |
| Crude Oil | Energy       |
| Bitcoin   | Crypto       |
| S&P 500   | Equity Index |
| Gold ETF  | ETF          |

The system supports multiple forecast horizons, including short-term and longer-horizon research windows such as 1D, 5D, 10D, 20D, and 30D.

---

## Core Features

### 1. Multi-Asset Data Pipeline

* Downloads and processes market data from public financial data sources.
* Supports cross-asset feature construction.
* Handles missing values, date alignment, scaling, outlier checks, and time-series splits.
* Maintains a structured dataset for downstream forecasting and research modules.

### 2. Feature Engineering

* Technical indicators such as moving averages, RSI, MACD, Bollinger-style volatility features, and rolling statistics.
* Lag-based features and return-based features.
* Cross-asset ratios and relative market indicators.
* Calendar-based features for time-series modeling.

### 3. Forecasting and Validation

* Machine learning based forecasting workflow.
* Time-series-aware validation instead of random train-test splitting.
* Walk-forward style testing for more realistic evaluation.
* Prediction range logic to avoid presenting point forecasts as certain outcomes.

### 4. Signal Research

* Converts model outputs into research-only signal candidates.
* Separates signal generation from real-money action.
* Tracks pending and matured paper signals.
* Evaluates whether signals are useful across different assets and horizons.

### 5. Risk Intelligence

* Drawdown-aware risk checks.
* Cost and slippage sensitivity.
* Exposure caps and paper-allocation limits.
* Data-quality warnings.
* Conservative handling of missing or unreliable information.

### 6. Market Regime Analysis

* Detects broad market and asset-level conditions.
* Applies regime-based adjustments to paper exposure.
* Flags unfavorable or dangerous market environments.
* Reduces confidence when market conditions are unstable.

### 7. Benchmark and Replay Engine

* Compares research signals against baseline strategies such as hold-only, no-exposure, momentum-style, moving-average-style, and random baselines.
* Uses historical replay/proxy replay logic to test whether signal rules show evidence of edge.
* Avoids claiming model strength when evidence is insufficient.
* Tracks dominance failures when strategies lose to simpler baselines.

### 8. Artifact Store and Reproducibility

* Saves generated research outputs in a structured artifact directory.
* Keeps summaries, leaderboards, risk tables, benchmark results, and quality gates available after reruns.
* Helps make the research workflow easier to inspect and reproduce.

### 9. Streamlit Dashboard

* Interactive dashboard for viewing forecasts, signals, risk analysis, benchmark audits, historical replay outputs, and research summaries.
* Designed for exploration and review, not for automated trading execution.

### 10. Testing and Quality Checks

* Pytest-based checks for important project modules.
* Validation for leakage checks, benchmark sanity, exposure caps, data-quality handling, and research-output consistency.
* Designed to catch misleading results before they appear in the dashboard.

---

## Tech Stack

| Area              | Tools                                                                              |
| ----------------- | ---------------------------------------------------------------------------------- |
| Language          | Python                                                                             |
| Data Processing   | pandas, NumPy                                                                      |
| Market Data       | yfinance, pandas-datareader, optional FRED data                                    |
| Machine Learning  | scikit-learn, XGBoost, LightGBM, CatBoost                                          |
| Visualization     | Streamlit, Plotly, Matplotlib                                                      |
| Testing           | pytest                                                                             |
| Project Structure | Modular Python package with `src/`, `tests/`, `config/`, and dashboard entry point |

---

## Project Architecture

```text
Market Data Sources
        |
        v
Data Loader and Raw Data Storage
        |
        v
Preprocessing and Date Alignment
        |
        v
Feature Engineering and Technical Indicators
        |
        v
Forecasting and Prediction Range Logic
        |
        v
Signal Research and Paper Tracking
        |
        v
Risk Intelligence and Position Sizing
        |
        v
Regime Analysis and Benchmark Audits
        |
        v
Streamlit Dashboard and Artifact Store
```

---

## Project Structure

```text
GoldPricePrediction/
|
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
|
├── config/
│   └── config.yaml
|
├── data/
│   ├── raw/
│   └── processed/
|
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── indicators.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── prediction.py
│   ├── prediction_ranges.py
│   ├── signals.py
│   ├── backtesting.py
│   ├── asset_config.py
│   └── utils.py
|
├── tests/
│   └── test_phase1_modules.py
|
└── artifacts/
```

Note: Generated artifacts, caches, virtual environments, and secret files should not be committed to the public repository.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/poojanbhardwaj/GoldPricePrediction-QuantSystem.git
cd GoldPricePrediction-QuantSystem
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure optional environment variables

If the project uses external APIs such as FRED, create a local `.env` file from the example file:

```bash
cp .env.example .env
```

Then add your own local keys. Do not commit real API keys.

---

## Usage

### Run the Streamlit dashboard

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

### Run module checks

```bash
python -m src.prediction_ranges
python -m src.signals
python -m src.backtesting
```

### Run tests

```bash
python -m pytest tests -q
```

---

## Research Methodology

The project is built around a conservative research workflow:

1. Collect and align multi-asset market data.
2. Engineer lagged, rolling, technical, calendar, and cross-asset features.
3. Train and evaluate forecasting models using time-series-aware validation.
4. Convert forecasts into research-only signal candidates.
5. Track pending and matured paper signals.
6. Apply risk checks, cost sensitivity, drawdown stress, and exposure limits.
7. Compare strategies against simple baselines.
8. Reject or downgrade signals when benchmark dominance, calibration, or data-quality checks fail.

This workflow is meant to reduce common mistakes in financial ML projects, especially leakage, overfitting, unrealistic backtests, and overconfident trading claims.

---

## Current Research Status

The platform is currently a research and paper-tracking system. It is not a live trading system.

Important design decisions:

* Real-money recommendations are blocked by default.
* Signals are treated as research candidates, not trading advice.
* Probability estimates are treated cautiously unless calibration evidence is strong.
* Benchmark comparison is required before claiming strategy strength.
* Missing data, unreliable outputs, or benchmark underperformance reduce confidence.

---

## What This Project Does Not Do

This project does not:

* Provide financial advice.
* Guarantee profitable trades.
* Execute real-money orders.
* Claim that forecasting accuracy alone is enough for a trading strategy.
* Hide weak results or failed benchmarks.

The goal is to build a transparent research system that can honestly show both strengths and weaknesses.

---

## Testing Philosophy

Financial ML projects can look impressive while still being wrong. This project uses tests and quality gates to reduce that risk.

Examples of checks included in the workflow:

* Time-series leakage checks.
* Return sanity checks.
* Exposure-cap enforcement.
* Benchmark comparison checks.
* Data-quality warnings.
* Research-output consistency checks.

---

## Suggested `.gitignore`

The public repository should avoid secrets, local environments, generated artifacts, and cache files.

```gitignore
venv/
__pycache__/
*.pyc
.env
.streamlit/secrets.toml
artifacts/
*.zip
.ipynb_checkpoints/
.DS_Store
```

---

## Future Improvements

Planned improvements include:

* Cleaner historical replay with true saved model snapshots.
* Stronger out-of-sample evaluation.
* More robust transaction-cost modeling.
* Better regime-aware signal filtering.
* Improved dashboard explanations for non-technical users.
* Additional tests for edge cases and data-quality failures.

---

## Author

**Poojan Bhardwaj**
B.Tech in Mathematics and Computing
National Institute of Technology, Kurukshetra
Academic Year: 2024–2028

GitHub: [poojanbhardwaj](https://github.com/poojanbhardwaj)
LinkedIn: [Poojan Bhardwaj](https://www.linkedin.com/in/poojan-bhardwaj-22b36b38a/)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
