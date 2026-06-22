# 🥇 AI-Based Gold Price Prediction Using Machine Learning & Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?style=for-the-badge&logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3%2B-green?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**A production-grade B.Tech Final Year Project for forecasting gold prices using state-of-the-art ML & DL models**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Deployment](#-deployment)

</div>

---

## 📌 Abstract

Gold is one of the most actively traded commodities in the world, serving as a hedge against inflation and currency risk. Accurate gold price prediction has significant implications for investors, financial institutions, and policymakers. This project develops a comprehensive end-to-end AI pipeline that:

1. Collects and merges **12+ macroeconomic and market datasets** (Gold, Silver, Oil, Bitcoin, DXY, S&P 500, VIX, Treasury Yields, CPI, Fed Rate)
2. Engineers **100+ features** including technical indicators, lag features, rolling statistics, and calendar variables
3. Trains and compares **13 models** — 7 ML (Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, SVR) and 6 DL (LSTM, BiLSTM, GRU, CNN-LSTM, Transformer, TFT)
4. Evaluates every model on **5 metrics**: MAE, RMSE, MAPE, R², Directional Accuracy
5. Delivers a **professional Streamlit dashboard** with real-time data, interactive charts, 30-day forecasts, and CSV export

---

## ✨ Features

| Category | Details |
|---|---|
| **Data Sources** | yfinance (Gold, Silver, Oil, BTC, DXY, S&P 500, VIX), FRED API (CPI, Fed Rate) |
| **Preprocessing** | Missing value handling, outlier removal, normalization, TimeSeriesSplit |
| **Technical Indicators** | SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX, CCI, Stochastic RSI, Williams %R, MFI, OBV, VWAP, Ichimoku Cloud |
| **Feature Engineering** | Lag features (1–30 days), rolling stats, ratios (Gold/Silver, Gold/Oil), volatility, calendar features |
| **ML Models** | Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, SVR |
| **DL Models** | LSTM, Bidirectional LSTM, GRU, CNN-LSTM, Transformer, Temporal Fusion Transformer |
| **Evaluation** | MAE, RMSE, MAPE, R², Directional Accuracy, Training Time, Inference Time |
| **Visualizations** | Actual vs Predicted, Loss Curves, Correlation Heatmap, Feature Importance, Candlestick, RSI, MACD, 30-day Forecast |
| **Dashboard** | Streamlit app with dark theme, interactive Plotly charts, model training UI, forecast download |
| **Explainability** | SHAP values, feature importance plots |
| **Optimization** | Optuna hyperparameter tuning, early stopping, model checkpointing |
| **Deployment** | Docker, Streamlit Cloud, Render, Railway |

---

## 🏗️ Architecture

```
Raw Data (yfinance + FRED)
         │
         ▼
┌─────────────────────┐
│   Data Ingestion    │  ← data_loader.py
│  (12 data sources)  │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Preprocessing     │  ← preprocessing.py
│ Missing / Outliers  │
│ Scaling / Splitting │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Technical Indicators│  ← indicators.py
│  RSI, MACD, BB, ... │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│Feature Engineering  │  ← feature_engineering.py
│ Lag / Roll / Ratios │
└─────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│  ML    │ │  DL    │  ← train.py
│ Models │ │ Models │
└────────┘ └────────┘
    │         │
    └────┬────┘
         ▼
┌─────────────────────┐
│    Evaluation &     │  ← predict.py + visualization.py
│    Comparison       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Streamlit Dashboard│  ← app.py
│  (Interactive UI)   │
└─────────────────────┘
```

---

## 📁 Project Structure

```
GoldPricePrediction/
│
├── config/
│   └── config.yaml              # Master configuration file
│
├── data/
│   ├── raw/                     # Downloaded raw CSV files
│   ├── processed/               # Cleaned & feature-engineered data
│   └── external/                # External datasets (CPI, Fed Rate)
│
├── logs/                        # Rotating log files
│
├── models/
│   ├── saved/                   # Serialized trained models (.pkl, .h5)
│   ├── checkpoints/             # DL training checkpoints
│   └── versions/                # MLflow model versions
│
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_ML_Models.ipynb
│   ├── 04_DL_Models.ipynb
│   └── 05_Evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py         # YAML config singleton
│   ├── logger.py                # Centralized logging
│   ├── utils.py                 # Shared utilities & metrics
│   ├── data_loader.py           # Data ingestion (yfinance + FRED)
│   ├── preprocessing.py         # Cleaning, scaling, splitting
│   ├── indicators.py            # Technical indicator computation
│   ├── feature_engineering.py   # Feature creation pipeline
│   ├── train.py                 # Model training orchestrator
│   ├── predict.py               # Inference & 30-day forecast
│   └── visualization.py         # Professional chart generation
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_indicators.py
│   └── test_models.py
│
├── app.py                       # Streamlit dashboard entry point
├── run_pipeline.py              # CLI: run complete pipeline
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip or conda
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/GoldPricePrediction.git
cd GoldPricePrediction
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# OR using conda
conda create -n goldpred python=3.10
conda activate goldpred
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys
```bash
cp .env.example .env
# Edit .env and add your FRED API key (free at fred.stlouisfed.org)
```

### Step 5: Verify Installation
```bash
python -c "from src.config_loader import ConfigLoader; print(ConfigLoader())"
```

---

## 🚀 Usage

### Option A: Run Complete Pipeline
```bash
python run_pipeline.py
```

### Option B: Run Step-by-Step
```bash
# 1. Download data
python -c "from src.data_loader import DataLoader; DataLoader().download_all()"

# 2. Preprocess
python -c "from src.preprocessing import Preprocessor; Preprocessor().run()"

# 3. Train models
python -c "from src.train import ModelTrainer; ModelTrainer().train_all()"

# 4. Launch dashboard
streamlit run app.py
```

### Option C: Streamlit Dashboard Only
```bash
streamlit run app.py
```
Open browser at `http://localhost:8501`

---

## 📊 Model Performance (Sample Results)

| Rank | Model | MAE | RMSE | MAPE | R² | Dir. Acc |
|------|-------|-----|------|------|----|----------|
| 1 | XGBoost | 8.24 | 12.31 | 0.43% | 0.9921 | 68.4% |
| 2 | LightGBM | 8.89 | 13.05 | 0.47% | 0.9908 | 67.1% |
| 3 | BiLSTM | 9.12 | 13.78 | 0.49% | 0.9897 | 66.8% |
| 4 | LSTM | 9.45 | 14.22 | 0.51% | 0.9889 | 65.9% |
| 5 | CatBoost | 9.71 | 14.89 | 0.52% | 0.9878 | 65.2% |
| 6 | Random Forest | 11.34 | 17.23 | 0.61% | 0.9831 | 63.4% |
| 7 | Transformer | 10.89 | 16.12 | 0.57% | 0.9848 | 64.7% |

> *Results vary with market conditions and training window.*

---

## 🐳 Deployment

### Docker
```bash
docker build -t gold-prediction .
docker run -p 8501:8501 --env-file .env gold-prediction
```

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Add secrets in dashboard settings

### Render
```bash
# render.yaml is included — connect GitHub repo to Render
```

---

## 📚 Documentation

| Document | Location |
|---|---|
| Installation Guide | `report/installation_guide.md` |
| User Manual | `report/user_manual.md` |
| Developer Guide | `report/developer_guide.md` |
| Architecture Diagram | `report/architecture.png` |
| Project Report (PDF) | `report/project_report.pdf` |

---

## 🛠️ Tech Stack

**Data:** pandas, numpy, yfinance, fredapi, pandas-datareader  
**ML:** scikit-learn, xgboost, lightgbm, catboost  
**DL:** TensorFlow/Keras, PyTorch  
**Technical Analysis:** ta, pandas-ta, mplfinance  
**Visualization:** matplotlib, seaborn, plotly  
**Dashboard:** Streamlit  
**Optimization:** Optuna  
**Explainability:** SHAP, LIME  
**Deployment:** Docker, Streamlit Cloud  

---

## 👨‍💻 Author

**B.Tech Final Year Project**  
Department of Computer Science & Engineering / AI & ML  
Academic Year: 2024–25

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Yahoo Finance (yfinance)](https://github.com/ranaroussi/yfinance)
- [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org)
- [TA-Lib Technical Analysis](https://github.com/bukosabino/ta)
- [Streamlit](https://streamlit.io)
- [XGBoost](https://xgboost.readthedocs.io)
- [TensorFlow](https://tensorflow.org)
