# Multi-Asset Quant Research Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)
[![CI](https://github.com/poojanbhardwaj/GoldPricePrediction-QuantSystem/actions/workflows/ci.yml/badge.svg)](https://github.com/poojanbhardwaj/GoldPricePrediction-QuantSystem/actions)
[![Research Only](https://img.shields.io/badge/use-research--only-lightgrey)](#safety-and-limitations)
[![Status](https://img.shields.io/badge/status-stable--v1.0-green)](#deployment)

A deployed, research-only **Multi-Asset Quantitative Research Platform** for studying market forecasts, evidence quality, model reliability, risk constraints, and paper-research plans across **Gold, Silver, Crude Oil, Bitcoin, S&P 500, and GLD**.

Unlike a notebook-only price predictor, this project is structured as a portfolio-grade ML systems application. It combines market data refresh, feature engineering, model training, model comparison, artifact-based research snapshots, diagnostic forecasting, cost/risk planning, evidence validation, candidate ranking, saved research history, and an auth-gated Streamlit workspace.

> **Important:** This project is for research and education only. It is not financial advice, does not execute trades, does not connect to brokers, and does not make buy/sell recommendations.

---

## Live Demo

[Live Streamlit App](https://multi-asset-quant-system.streamlit.app/)

---

## Project Summary

This project was built as an end-to-end ML systems project for financial time-series research.

It focuses on:

- Multi-asset market data processing.
- Time-series feature engineering.
- Classical ML and optional DL model training.
- Shared model registry for comparison and forecasting.
- Forecast diagnostics and actual-vs-predicted analysis.
- Risk-aware signal interpretation.
- Cost-aware research planning.
- Artifact-based dashboard reliability.
- Streamlit Cloud deployment.
- Research-only safety constraints.

The goal is **not** to claim guaranteed market prediction. The goal is to provide a repeatable research workflow for comparing models, checking evidence quality, surfacing uncertainty, and preventing misleading dashboard states.

---

## Screenshots

Screenshots from the deployed research workspace:

| View | Screenshot |
|---|---|
| Research Dashboard | ![Research Dashboard](docs/screenshots/01_dashboard.png) |
| Refresh / Rebuild Research | ![Refresh Research](docs/screenshots/02_refresh_research.png) |
| Advanced Diagnostics | ![Advanced Diagnostics](docs/screenshots/03_advanced_diagnostics.png) |
| Train Models | ![Train Models](docs/screenshots/04_train_models.png) |
| Compare Models | ![Compare Models](docs/screenshots/05_compare_models.png) |
| 30-Day Diagnostic Forecast | ![30-Day Forecast](docs/screenshots/06_30_day_forecast.png) |
| Cost & Risk Plan | ![Cost & Risk Plan](docs/screenshots/07_cost_risk_plan.png) |
| Candidate Watchlist | ![Candidate Watchlist](docs/screenshots/08_candidate_watchlist.png) |
| Evidence & Validation | ![Evidence & Validation](docs/screenshots/09_evidence_validation.png) |

---

## Key Features

- **Multi-asset research dashboard** for Gold, Silver, Crude Oil, Bitcoin, S&P 500, and GLD.
- **Market data refresh / rebuild workflow** with visible source and freshness labels.
- **Time-series feature engineering** for technical, macro, calendar, and cross-asset signals.
- **Model training workflow** for chronological forecasting experiments.
- **Shared ML/DL model registry** for unified training, comparison, and forecast readiness tracking.
- **Model comparison dashboard** with RMSE, MAE, MAPE, R², and Direction Accuracy.
- **Actual-vs-predicted diagnostics** for visual model inspection.
- **30-day diagnostic forecast view** for model-based scenario inspection.
- **Cost & Risk Plan** showing break-even return, cost drag, active/passive estimates, and missing-dependency explanations.
- **Candidate Watchlist** ranking research ideas while keeping weak evidence and blockers visible.
- **Evidence & Validation** views for checking whether an idea survives validation, benchmark, and risk review.
- **Saved research plans and research history** for user-owned paper-research tracking.
- **Advanced Diagnostics** for source freshness, artifact integrity, model training, prediction snapshots, and forecast debugging.
- **Auth-gated workspace** with public preview, local development auth, and optional Supabase verified-email mode.
- **Automated regression tests** covering product flows, auth, navigation, diagnostics, snapshot handling, and safety wording.

---

## Supported Assets

| Asset | Description |
|---|---|
| Gold | Gold futures / gold market proxy |
| Silver | Silver futures / silver market proxy |
| Crude Oil | Crude oil futures / energy market proxy |
| Bitcoin | BTC-USD crypto market data |
| S&P 500 | Broad US equity index proxy |
| GLD | Gold ETF proxy |

The platform is designed to support multi-asset comparison instead of focusing only on a single Gold prediction workflow.

---

## Supported Models

| Type | Models |
|---|---|
| Classical ML | Linear Regression, Decision Tree, Random Forest, SVR |
| Gradient Boosting | XGBoost, LightGBM, CatBoost |
| Optional Deep Learning | LSTM, BiLSTM, GRU, CNN-LSTM, Transformer |

Deep learning support is optional in deployment because TensorFlow can be heavy on cloud environments. The app is designed to degrade safely: unavailable DL models show clear warnings instead of crashing the dashboard.

---

## Shared ML/DL Model Registry

The platform includes a shared model registry that unifies model training, model comparison, and forecasting across multiple model families.

Tracked registry fields include:

- Asset name
- Forecast horizon
- Model family: ML or DL
- Model name
- RMSE
- MAE
- MAPE
- R²
- Direction Accuracy
- Target transformation mode
- Sequence length
- Forecast readiness flag
- Training timestamp
- Failure/warning reason for unavailable models

The registry allows the app to compare classical ML models and optional deep learning models in one workflow. It also prevents forecast pages from relying on unavailable or incomplete model outputs.

---

## System Architecture

```text
Market Data / Saved Research Snapshots
        |
        v
Data Loading -> Feature Engineering -> Model Training / Forecasting
        |                    |                 |
        v                    v                 v
Source Freshness       Validation Evidence   Model Diagnostics
        |                    |                 |
        +--------- Risk, Cost, Benchmarks -----+
                          |
                          v
             Shared ML/DL Model Registry
                          |
                          v
          Candidate Watchlist + Evidence Review
                          |
                          v
        Personalized Paper-Research Plans / History
                          |
                          v
                 Streamlit Product Workspace
