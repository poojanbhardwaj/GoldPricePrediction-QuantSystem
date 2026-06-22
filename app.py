"""
app.py — AI Gold Price Prediction Dashboard
==============================================
Professional Streamlit dashboard tying together every module of the
pipeline into an interactive, presentable web application.

Pages
-----
🏠 Home              — project overview, live snapshot
ℹ️  About             — project description, architecture, tech stack
📊 Dataset Explorer   — raw data browsing, summary stats, downloads
📈 Technical Indicators — interactive indicator charts
🤖 Train Models       — trigger training, watch progress, view leaderboard
🏆 Compare Models     — full metrics comparison across all models
🔮 Prediction         — single-model prediction on most recent data
📅 30-Day Forecast    — rolling forecast with confidence bands + CSV export

Run with
--------
    streamlit run app.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config_loader import ConfigLoader
from src.logger import get_logger
from src.data_loader import DataLoader
from src.indicators import TechnicalIndicators
from src.feature_engineering import FeatureEngineer
from src.feature_intelligence import (
    add_phase5_feature_intelligence,
    build_feature_intelligence_report,
    phase5_feature_columns,
)
from src.preprocessing import Preprocessor
from src.train import ModelTrainer
from src.predict import Predictor
from src.visualization import Visualizer
from src.prediction_ranges import calculate_prediction_range
from src.signals import generate_trading_signal
from src.backtesting import run_backtest_from_predictions
from src.asset_config import get_asset_names, get_target_column, get_asset_config
from src.baselines import price_baseline_leaderboard, model_vs_naive_summary
from src.directional_models import (
    train_directional_models,
    directional_leaderboard,
    directional_baseline_leaderboard,
    run_directional_probability_backtest,
)
from src.research_validation import (
    build_validation_report,
    regime_performance,
    walk_forward_validate_model,
)
from src.multiasset_validation import (
    run_multiasset_validation,
    summarize_asset_status,
)
from src.direct_forecast_models import (
    DIRECT_FORECAST_HORIZONS,
    run_asset_horizon_scan,
    run_direct_forecast_report,
    run_direct_forecast_signal_output,
)
from src.signal_engine import (
    run_candidate_deep_diagnostics,
    run_risk_controlled_candidate_upgrade,
    run_signal_engine,
    run_signal_research_scan,
    run_walk_forward_risk_validation,
    run_validation_locked_signal_engine,
)
from src.meta_signal_engine import (
    run_meta_decision_audit,
    run_meta_score_calibration,
    run_regime_aware_meta_signal,
)

logger = get_logger(__name__)
cfg    = ConfigLoader()

# ════════════════════════════════════════════════════════════════
# Page Config
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=cfg.get("dashboard.title", "🥇 Gold Price Prediction AI"),
    page_icon=cfg.get("dashboard.page_icon", "🥇"),
    layout=cfg.get("dashboard.layout", "wide"),
    initial_sidebar_state=cfg.get("dashboard.sidebar_state", "expanded"),
)

# ── Dark theme CSS polish ──────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background-color: #1a1d24; padding: 12px; border-radius: 10px; border: 1px solid #2d3139; }
    div[data-testid="stMetricValue"] { color: #D4AF37; }
    .main-header { font-size: 2.4rem; font-weight: 800; color: #D4AF37; margin-bottom: 0; }
    .sub-header { font-size: 1.0rem; color: #9099a8; margin-top: 0; }
    section[data-testid="stSidebar"] { background-color: #14161b; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# Cached Pipeline Builders
# ════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600 * 6)  # refresh cache every 6 hours
def load_raw_data(start_date: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Load market data from start_date through TODAY. end_date is intentionally
    not a parameter here — DataLoader defaults to the current date whenever
    end_date isn't explicitly provided, so every call automatically pulls the
    latest available trading data rather than a fixed historical snapshot.
    """
    loader = DataLoader(start_date=start_date, end_date=None)
    return loader.load_all(use_cache=use_cache)


def _target_prefix(target_col: str) -> str:
    """Convert a close column such as Gold_Close or BTC_Close to its OHLCV prefix."""
    return str(target_col).replace("_Close", "")


@st.cache_data(show_spinner=False)
def build_features(df: pd.DataFrame, target_col: str = "Gold_Close") -> pd.DataFrame:
    prefix = _target_prefix(target_col)
    ti = TechnicalIndicators(prefix=prefix)
    df = ti.add_all(df)

    # Forward-fill supporting market/macro columns before feature engineering.
    # Some assets update later than the selected target. Without this,
    # FeatureEngineer.dropna() can remove the latest target rows and make the
    # forecast start from an older date. Forward-fill is time-series safe
    # because it only carries past known values forward.
    df = df.sort_index().ffill()

    fe = FeatureEngineer(target_col=target_col)
    df = fe.build_features(df)

    # Phase 5: add market-aware feature intelligence.
    # These features use only current/past information, so they are valid
    # for the next-day target created in preprocessing.py.
    df = add_phase5_feature_intelligence(df, target_col=target_col)
    return df


def get_preprocessor_and_data(df: pd.DataFrame, target_col: str = "Gold_Close"):
    """Not cached — Preprocessor holds unpicklable scaler state we reuse live."""
    pp = Preprocessor(target_col=target_col)
    data = pp.run(df)
    return pp, data


def _asset_mismatch(selected_asset: str) -> bool:
    trained_asset = st.session_state.get("trained_asset")
    return bool(st.session_state.get("trained", False) and trained_asset and trained_asset != selected_asset)


def _stop_if_asset_mismatch(selected_asset: str) -> None:
    if _asset_mismatch(selected_asset):
        st.warning(
            f"Models in memory were trained for **{st.session_state.trained_asset}**, "
            f"but sidebar asset is **{selected_asset}**. Go to **Train Models** and train again."
        )
        st.stop()


def _safe_test_rmse(model_result, data) -> float:
    """Use held-out test error for uncertainty bands, never training error."""
    try:
        rmse = model_result.metrics_test.get("RMSE")
        if rmse is not None and np.isfinite(float(rmse)) and float(rmse) > 0:
            return float(rmse)
    except Exception:
        pass

    actual = np.asarray(data.prices_test, dtype=float)
    pred = np.asarray(model_result.predictions_test, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() == 0:
        raise ValueError("Cannot calculate RMSE because test predictions are invalid.")
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2)))


def _build_backtest_frame(data, model_result, target_col: str = "Gold_Close") -> pd.DataFrame:
    """
    Build a backtest DataFrame from real held-out test predictions.

    The model predicts the target test price. To convert that to a trading
    signal, each predicted price is compared with the previous known actual
    price anchor.
    """
    actual_prices = np.asarray(data.prices_test, dtype=float)
    predicted_prices = np.asarray(model_result.predictions_test, dtype=float)

    if len(actual_prices) != len(predicted_prices):
        raise ValueError("Actual and predicted test arrays have different lengths.")

    first_anchor = float(data.last_price_before_test)
    anchors = np.concatenate([[first_anchor], actual_prices[:-1]])

    return pd.DataFrame(
        {
            target_col: anchors,
            "Predicted_Price": predicted_prices,
            "Actual_Next_Price": actual_prices,
            "Predicted_Return": predicted_prices / anchors - 1.0,
            "Actual_Next_Return": actual_prices / anchors - 1.0,
        },
        index=pd.to_datetime(data.test_index),
    )


# ════════════════════════════════════════════════════════════════
# Session State Initialization
# ════════════════════════════════════════════════════════════════

if "trained" not in st.session_state:
    st.session_state.trained = False
if "trainer" not in st.session_state:
    st.session_state.trainer = None
if "pp" not in st.session_state:
    st.session_state.pp = None
if "data" not in st.session_state:
    st.session_state.data = None
if "df_features" not in st.session_state:
    st.session_state.df_features = None
if "backtest_df" not in st.session_state:
    st.session_state.backtest_df = None
if "trained_asset" not in st.session_state:
    st.session_state.trained_asset = None
if "selected_asset" not in st.session_state:
    st.session_state.selected_asset = "Gold"
if "directional_results" not in st.session_state:
    st.session_state.directional_results = None
if "directional_asset" not in st.session_state:
    st.session_state.directional_asset = None
if "multiasset_validation_report" not in st.session_state:
    st.session_state.multiasset_validation_report = None
if "multiasset_validation_settings" not in st.session_state:
    st.session_state.multiasset_validation_settings = None
if "direct_forecast_report" not in st.session_state:
    st.session_state.direct_forecast_report = None
if "direct_forecast_settings" not in st.session_state:
    st.session_state.direct_forecast_settings = None
if "direct_horizon_scan_report" not in st.session_state:
    st.session_state.direct_horizon_scan_report = None
if "direct_horizon_scan_settings" not in st.session_state:
    st.session_state.direct_horizon_scan_settings = None
if "signal_engine_output" not in st.session_state:
    st.session_state.signal_engine_output = None
if "signal_engine_result" not in st.session_state:
    st.session_state.signal_engine_result = None
if "signal_engine_settings" not in st.session_state:
    st.session_state.signal_engine_settings = None
if "signal_research_scan_report" not in st.session_state:
    st.session_state.signal_research_scan_report = None
if "signal_research_scan_settings" not in st.session_state:
    st.session_state.signal_research_scan_settings = None
if "candidate_diagnostics_report" not in st.session_state:
    st.session_state.candidate_diagnostics_report = None
if "candidate_diagnostics_settings" not in st.session_state:
    st.session_state.candidate_diagnostics_settings = None
if "risk_control_upgrade_report" not in st.session_state:
    st.session_state.risk_control_upgrade_report = None
if "risk_control_upgrade_settings" not in st.session_state:
    st.session_state.risk_control_upgrade_settings = None
if "walk_forward_validation_report" not in st.session_state:
    st.session_state.walk_forward_validation_report = None
if "walk_forward_validation_settings" not in st.session_state:
    st.session_state.walk_forward_validation_settings = None
if "meta_signal_report" not in st.session_state:
    st.session_state.meta_signal_report = None
if "meta_signal_settings" not in st.session_state:
    st.session_state.meta_signal_settings = None
if "meta_decision_audit_report" not in st.session_state:
    st.session_state.meta_decision_audit_report = None
if "meta_decision_audit_settings" not in st.session_state:
    st.session_state.meta_decision_audit_settings = None
if "meta_reliability_grading_report" not in st.session_state:
    st.session_state.meta_reliability_grading_report = None
if "meta_reliability_grading_settings" not in st.session_state:
    st.session_state.meta_reliability_grading_settings = None
if "phase5_audit" not in st.session_state:
    st.session_state.phase5_audit = None
if "phase5_features_preview" not in st.session_state:
    st.session_state.phase5_features_preview = None


# ════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🥇 Gold Price AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "ℹ️ About Project",
        "📊 Dataset Explorer",
        "📈 Technical Indicators",
        "🤖 Train Models",
        "🏆 Compare Models",
        "🔮 Prediction",
        "📉 Backtesting",
        "🎯 Directional Models",
        "🧠 Feature Intelligence",
        "🧪 Research Validation",
        "🌐 Multi-Asset Matrix",
        "🎯 Direct Forecast Models",
        "🧭 Direct Horizon Scanner",
        "🧪 Signal Research Scanner",
        "🔬 Candidate Deep Diagnostics",
        "🛡️ Risk-Controlled Upgrade",
        "🧭 Walk-Forward Validation",
        "🧠 Regime-Aware Meta Signal",
        "🧾 Meta Decision Audit",
        "🏷️ Meta Reliability Grading",
        "📡 Signal Engine",
        "📅 30-Day Forecast",
    ],
    label_visibility="collapsed",
)

asset_names = get_asset_names()
default_asset_index = asset_names.index(st.session_state.selected_asset) if st.session_state.selected_asset in asset_names else 0
selected_asset = st.sidebar.selectbox("Target Asset", asset_names, index=default_asset_index)
st.session_state.selected_asset = selected_asset
target_col = get_target_column(selected_asset)

if _asset_mismatch(selected_asset):
    st.sidebar.warning(
        f"Current trained models are for {st.session_state.trained_asset}. "
        f"Train again to use {selected_asset}."
    )

st.sidebar.markdown("---")
st.sidebar.caption("B.Tech Final Year Project")
st.sidebar.caption(f"v{cfg.get('project.version', '1.0.0')}")


# ════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown('<p class="main-header">📊 Multi-Asset Quant Forecasting Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning & Deep Learning powered forecasting dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Loading market data..."):
        df_raw = load_raw_data("2015-01-01", use_cache=True)

    if target_col not in df_raw.columns:
        st.error(f"Selected target column `{target_col}` is not available in the loaded dataset.")
        st.stop()

    asset_series = df_raw[target_col].dropna()
    latest_price = float(asset_series.iloc[-1])
    prev_price = float(asset_series.iloc[-2])
    latest_date = asset_series.index[-1]
    change = latest_price - prev_price
    pct_change = (change / prev_price) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Latest {selected_asset} Close", f"${latest_price:,.2f}", f"{pct_change:+.2f}%")
    col2.metric("52-Week High", f"${asset_series.tail(252).max():,.2f}")
    col3.metric("52-Week Low", f"${asset_series.tail(252).min():,.2f}")
    col4.metric("Dataset Size", f"{len(df_raw):,} days")

    st.caption(f"Latest local dataset date for {selected_asset}: {pd.Timestamp(latest_date).date()}")

    st.markdown("### Recent Price Trend")
    viz = Visualizer()
    if selected_asset == "Gold":
        fig = viz.plot_candlestick_plotly(df_raw, n_days=120)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df_raw[[target_col]].tail(120))

    st.markdown("### Quick Stats")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(f"**Date Range**\n\n{df_raw.index.min().date()} → {df_raw.index.max().date()}")
    with c2:
        st.info(f"**Data Sources**\n\nGold, Silver, Oil, Bitcoin, DXY, S&P 500, VIX, 10Y Treasury, Fed Rate, CPI")
    with c3:
        st.info(f"**Models Available**\n\n7 ML algorithms + 5 Deep Learning architectures")


# ════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════════════════════

elif page == "ℹ️ About Project":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ### 📌 Objective
    This project predicts future **gold prices** using a combination of historical market data,
    technical indicators, macroeconomic variables, and both classical Machine Learning and
    modern Deep Learning models.

    ### 🏗️ Architecture
    ```
    Raw Data (yfinance + FRED)
            ↓
    Preprocessing (cleaning, scaling, leakage guard)
            ↓
    Technical Indicators (35 indicators: RSI, MACD, Bollinger Bands...)
            ↓
    Feature Engineering (lags, rolling stats, ratios, calendar features)
            ↓
    ┌──────────┐         ┌──────────┐
    │ ML Models│         │ DL Models│
    └──────────┘         └──────────┘
            ↓                   ↓
            └─────── Evaluation ┘
                      ↓
              Streamlit Dashboard
    ```

    ### 🧠 Modeling Approach
    Models predict the **next-day log return** (percentage price change) rather than the raw
    price level. This is the standard quantitative-finance approach — it keeps the target
    statistically stationary across years of data, which lets tree-based models (Random Forest,
    XGBoost, LightGBM, CatBoost) generalize correctly even as gold's price trends from roughly
    $1,050 to $2,800 over the dataset's history, instead of failing to extrapolate beyond price
    ranges seen during training.

    ### 🛠️ Tech Stack
    | Category | Tools |
    |---|---|
    | Data | yfinance, FRED API, pandas |
    | ML | scikit-learn, XGBoost, LightGBM, CatBoost |
    | DL | TensorFlow / Keras |
    | Visualization | Matplotlib, Seaborn, Plotly |
    | Dashboard | Streamlit |
    | Optimization | Optuna |

    ### 👨‍💻 Author
    B.Tech Final Year Project — Computer Science & Engineering / AI & ML
    """)


# ════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ════════════════════════════════════════════════════════════════

elif page == "📊 Dataset Explorer":
    st.markdown('<p class="main-header">📊 Dataset Explorer</p>', unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Loading data..."):
        df_raw = load_raw_data("2015-01-01", use_cache=True)

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📈 Summary Statistics", "⬇️ Download"])

    with tab1:
        n_rows = st.slider("Rows to display", 10, 200, 50)
        st.dataframe(df_raw.tail(n_rows), use_container_width=True)

    with tab2:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df_raw.describe().T, use_container_width=True)

        st.markdown("#### Missing Values")
        null_counts = df_raw.isnull().sum()
        null_df = null_counts[null_counts > 0].to_frame("Missing Count")
        if null_df.empty:
            st.success("No missing values in the dataset ✓")
        else:
            st.dataframe(null_df, use_container_width=True)

    with tab3:
        csv = df_raw.to_csv().encode("utf-8")
        st.download_button(
            "📥 Download Full Dataset (CSV)", data=csv,
            file_name="gold_master_dataset.csv", mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════
# PAGE: TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════

elif page == "📈 Technical Indicators":
    st.markdown(f'<p class="main-header">📈 Technical Indicator Viewer — {selected_asset}</p>', unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Computing indicators..."):
        df_raw = load_raw_data("2015-01-01", use_cache=True)
        df_ind = build_features(df_raw, target_col=target_col)

    import plotly.graph_objects as go

    n_days = st.slider("Days to display", 30, 500, 250)

    tab1, tab2, tab3 = st.tabs(["📊 Moving Averages", "📉 RSI", "📈 MACD"])

    with tab1:
        recent = df_ind.tail(n_days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent.index, y=recent[target_col], name=f"{selected_asset} Price", line=dict(color="white")))
        for p, color in zip([20, 50, 200], ["#FF6B6B", "#4ECDC4", "#FFD93D"]):
            col = f"SMA_{p}"
            if col in recent.columns:
                fig.add_trace(go.Scatter(x=recent.index, y=recent[col], name=f"SMA-{p}", line=dict(color=color)))
        fig.update_layout(template="plotly_dark", title="Price with Moving Averages")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        recent = df_ind.tail(n_days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent.index, y=recent["RSI"], line=dict(color="#9B59B6")))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        fig.update_layout(template="plotly_dark", title="RSI", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        recent = df_ind.tail(n_days)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=recent.index, y=recent["MACD_Hist"], name="Histogram", marker_color="gray", opacity=0.4))
        fig.add_trace(go.Scatter(x=recent.index, y=recent["MACD"], name="MACD", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=recent.index, y=recent["MACD_Signal"], name="Signal", line=dict(color="#D4AF37")))
        fig.update_layout(template="plotly_dark", title="MACD")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE: TRAIN MODELS
# ════════════════════════════════════════════════════════════════

elif page == "🤖 Train Models":
    st.markdown('<p class="main-header">🤖 Train Models</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("Select which model families to train, then click **Start Training**.")
    st.info(f"Training target asset: **{selected_asset}** (`{target_col}`)")
    col1, col2 = st.columns(2)
    with col1:
        train_ml = st.checkbox("Train ML Models (Linear Regression, Trees, Boosting, SVR)", value=True)
    with col2:
        train_dl = st.checkbox("Train DL Models (LSTM, BiLSTM, GRU, CNN-LSTM, Transformer)", value=False)
        if train_dl:
            st.caption("⚠️ DL training is slower — consider reducing epochs below for a quick demo run.")
            dl_epochs = st.slider("DL Epochs (demo)", 3, 100, 10)

    if st.button("🚀 Start Training", type="primary"):
        progress = st.progress(0, text="Loading data...")

        df_raw = load_raw_data("2015-01-01", use_cache=True)
        progress.progress(20, text="Engineering features...")
        df_features = build_features(df_raw, target_col=target_col)
        progress.progress(40, text="Preprocessing...")
        pp, data = get_preprocessor_and_data(df_features, target_col=target_col)

        st.session_state.pp = pp
        st.session_state.data = data
        st.session_state.df_features = df_features
        st.session_state.trained_asset = selected_asset

        progress.progress(55, text="Training ML models...")
        trainer = ModelTrainer(use_optuna=False, target_scaler=data.target_scaler, preprocessor=pp)

        if train_ml:
            trainer.train_all_ml(data)

        progress.progress(85, text="Finalizing...")
        st.session_state.trainer = trainer
        st.session_state.trained = True
        progress.progress(100, text="Done!")

        st.success(f"✔ Training complete for {selected_asset} — {len(trainer.results)} model(s) trained.")
        board = trainer.get_leaderboard("test")
        st.dataframe(board, use_container_width=True)

    elif st.session_state.trained:
        st.info("Models already trained this session. Go to **Compare Models** to view results, or retrain above.")


# ════════════════════════════════════════════════════════════════
# PAGE: COMPARE MODELS
# ════════════════════════════════════════════════════════════════

elif page == "🏆 Compare Models":
    st.markdown('<p class="main-header">🏆 Model Comparison</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not st.session_state.trained:
        st.warning("⚠️ No models trained yet. Go to **Train Models** first.")
    else:
        _stop_if_asset_mismatch(selected_asset)
        trainer = st.session_state.trainer
        viz = Visualizer()

        metric = st.selectbox("Sort/compare by metric", ["RMSE", "MAE", "MAPE", "R2", "DirectionalAccuracy"])
        board = trainer.get_leaderboard("test")

        st.dataframe(board, use_container_width=True)

        st.markdown("### Baseline Checks")
        data = st.session_state.data
        try:
            baseline_board = price_baseline_leaderboard(data)
            st.caption("Baselines use only known price anchors. Naive baseline means: tomorrow's price = today's price.")
            st.dataframe(baseline_board, use_container_width=True)

            summary = model_vs_naive_summary(board, baseline_board)
            if summary:
                improvement = summary["rmse_improvement_pct"]
                if improvement > 0:
                    st.success(
                        f"Best model **{summary['best_model']}** beats Naive RMSE by **{improvement:.2f}%** "
                        f"({summary['best_model_rmse']} vs {summary['naive_rmse']})."
                    )
                else:
                    st.error(
                        f"Best model **{summary['best_model']}** does NOT beat Naive RMSE "
                        f"({summary['best_model_rmse']} vs {summary['naive_rmse']})."
                    )
        except Exception as exc:
            st.warning(f"Could not calculate baseline checks: {exc}")

        fig = viz.plot_model_comparison_plotly(board, metric=metric)
        st.plotly_chart(fig, use_container_width=True)

        best_name, best_result = trainer.get_best_model("test")
        st.success(f"🏆 Best Model: **{best_name}** — RMSE = ${best_result.metrics_test['RMSE']:.2f}, R² = {best_result.metrics_test['R2']:.4f}")

        st.markdown(f"### Actual vs Predicted — {selected_asset} (Best Model)")
        data = st.session_state.data
        fig2 = viz.plot_actual_vs_predicted_plotly(
            data.prices_test, best_result.predictions_test, data.test_index,
            title=f"{selected_asset} / {best_name} — Actual vs Predicted",
        )
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ════════════════════════════════════════════════════════════════

elif page == "🔮 Prediction":
    st.markdown(f'<p class="main-header">🔮 Next-Day Prediction — {selected_asset}</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This system is for educational and research purposes only. "
        "Predictions are estimates and should not be considered financial advice."
    )

    if not st.session_state.trained:
        st.warning("⚠️ No models trained yet. Go to **Train Models** first.")
    else:
        _stop_if_asset_mismatch(selected_asset)
        trainer = st.session_state.trainer
        pp = st.session_state.pp
        data = st.session_state.data

        model_name = st.selectbox("Select a model", list(trainer.results.keys()))
        result = trainer.results[model_name]

        predictor = Predictor(model=result.model, preprocessor=pp, is_sequence_model=False)

        # Use the latest available feature row for the live next-day forecast.
        # After the next-day target fix, data.X_test[-1] is the last row with a
        # known next-day label, not necessarily the newest feature row.
        target_col = getattr(pp, "target_col", "Gold_Close")
        latest_features = st.session_state.df_features[data.feature_cols].tail(1)
        X_latest = data.feature_scaler.transform(latest_features.values)
        last_price = float(st.session_state.df_features[target_col].dropna().iloc[-1])
        next_price = predictor.predict_next_day(X_latest, last_price)

        try:
            rmse = _safe_test_rmse(result, data)

            prediction_range = calculate_prediction_range(
                last_price=last_price,
                predicted_price=next_price,
                rmse=rmse,
                confidence_level=0.68,
                model_used=model_name,
            )

            trading_signal = generate_trading_signal(
                predicted_return_pct=prediction_range.predicted_return_pct,
                lower_return_pct=prediction_range.lower_return_pct,
                upper_return_pct=prediction_range.upper_return_pct,
            )

            st.markdown("### Prediction Range")

            col1, col2, col3 = st.columns(3)
            col1.metric(f"Last Known {selected_asset} Price", f"${prediction_range.last_price:,.2f}")
            col2.metric(
                f"Predicted Next-Day {selected_asset} Price",
                f"${prediction_range.predicted_price:,.2f}",
                f"{prediction_range.predicted_return_pct:+.2f}%",
            )
            col3.metric("Model Used", prediction_range.model_used)

            col4, col5, col6 = st.columns(3)
            col4.metric("Expected Lower Bound", f"${prediction_range.lower_bound:,.2f}")
            col5.metric("Expected Upper Bound", f"${prediction_range.upper_bound:,.2f}")
            col6.metric("Confidence Level", f"{prediction_range.confidence_level:.0f}%")

            st.caption(
                f"Range is based on held-out test RMSE = ${prediction_range.error_used:,.2f}. "
                "It is an uncertainty estimate, not a guaranteed interval."
            )

            st.markdown("### Trading Signal")

            sig1, sig2, sig3 = st.columns(3)
            sig1.metric("Signal", trading_signal.signal)
            sig2.metric("Signal Confidence", trading_signal.confidence_label)
            sig3.metric("Risk Label", trading_signal.risk_label)

            st.caption(trading_signal.explanation)

        except Exception as exc:
            st.error(f"Could not calculate prediction range/signal: {exc}")

            col1, col2, col3 = st.columns(3)
            col1.metric(f"Last Known {selected_asset} Price", f"${last_price:,.2f}")
            col2.metric(f"Predicted Next-Day {selected_asset} Price", f"${next_price:,.2f}", f"{((next_price / last_price) - 1) * 100:+.2f}%")
            col3.metric("Model Used", model_name)

        st.markdown("### Test-Set Predictions (CSV Export)")

        target_col = getattr(pp, "target_col", "Gold_Close")

        export_df = pd.DataFrame({
            "Date": data.test_index,
            "Actual_Price": data.prices_test,
            "Predicted_Price": result.predictions_test,
        })

        # Store real test-set predictions for the Backtesting page.
        try:
            st.session_state.backtest_df = _build_backtest_frame(
                data=data,
                model_result=result,
                target_col=target_col,
            )
        except Exception as exc:
            st.session_state.backtest_df = None
            st.warning(f"Could not prepare backtesting data: {exc}")

        st.dataframe(export_df.tail(20), use_container_width=True)

        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Predictions (CSV)",
            data=csv,
            file_name=f"{model_name}_predictions.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════
# PAGE: BACKTESTING
# ════════════════════════════════════════════════════════════════

elif page == "📉 Backtesting":
    st.markdown('<p class="main-header">📉 Strategy Backtesting</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This backtest is for educational and research purposes only. "
        "It does not include slippage, liquidity limits, taxes, or real execution constraints."
    )

    if not st.session_state.trained:
        st.warning("⚠️ No models trained yet. Go to **Train Models** first.")
    else:
        _stop_if_asset_mismatch(selected_asset)
        trainer = st.session_state.trainer
        pp = st.session_state.pp
        data = st.session_state.data

        model_name = st.selectbox("Select model for backtest", list(trainer.results.keys()))
        result = trainer.results[model_name]

        threshold_pct = st.slider(
            "Signal threshold (%)",
            min_value=0.0,
            max_value=3.0,
            value=0.2,
            step=0.1,
            help="Long only when predicted return is greater than this threshold.",
        )

        transaction_cost_pct = st.number_input(
            "Transaction cost per trade (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.1,
            step=0.05,
        )

        allow_short = st.checkbox("Allow shorting", value=False)

        target_col = getattr(pp, "target_col", "Gold_Close")

        try:
            backtest_df = _build_backtest_frame(
                data=data,
                model_result=result,
                target_col=target_col,
            )

            bt = run_backtest_from_predictions(
                backtest_df,
                price_col=target_col,
                predicted_price_col="Predicted_Price",
                threshold=threshold_pct / 100.0,
                transaction_cost=transaction_cost_pct / 100.0,
                allow_short=allow_short,
            )

            st.session_state.backtest_df = backtest_df

        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

        metrics = bt.metrics
        equity = bt.equity_curve
        trades = bt.trades

        st.markdown("### Backtest Metrics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
        c2.metric("Annualized Return", f"{metrics['annualized_return_pct']:.2f}%")
        c3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        c4.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Win Rate", f"{metrics['win_rate_pct']:.2f}%")
        c6.metric("Trades", int(metrics["number_of_trades"]))
        c7.metric("Buy & Hold Return", f"{metrics['buy_hold_return_pct']:.2f}%")
        c8.metric("Strategy - BuyHold", f"{metrics['strategy_minus_buy_hold_pct']:.2f}%")

        st.markdown("### Strategy vs Buy-and-Hold")
        st.line_chart(equity[["strategy_equity", "buy_hold_equity"]])

        st.markdown("### Drawdown")
        st.line_chart(equity[["strategy_drawdown", "buy_hold_drawdown"]])

        st.markdown("### Position")
        st.line_chart(equity[["position"]])

        with st.expander("View backtest input data"):
            st.dataframe(backtest_df.tail(30), use_container_width=True)

        st.markdown("### Trade Summary")
        if trades.empty:
            st.info("No trades were generated with this threshold.")
        else:
            st.dataframe(trades, use_container_width=True)



# ════════════════════════════════════════════════════════════════
# PAGE: DIRECTIONAL MODELS
# ════════════════════════════════════════════════════════════════

elif page == "🎯 Directional Models":
    st.markdown(f'<p class="main-header">🎯 Directional Models — {selected_asset}</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.info(
        "This page trains separate Up/Down classifiers. Regression models optimize RMSE; "
        "directional classifiers directly optimize tomorrow's direction."
    )

    if not st.session_state.trained:
        st.warning("⚠️ No preprocessing/training session found. Go to **Train Models** first.")
    else:
        _stop_if_asset_mismatch(selected_asset)
        pp = st.session_state.pp
        data = st.session_state.data

        st.markdown("### Directional Baselines")
        try:
            dir_base = directional_baseline_leaderboard(data, pp)
            st.dataframe(dir_base, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not calculate directional baselines: {exc}")

        c_train, c_note = st.columns([1, 2])
        with c_train:
            train_dir = st.button("🚀 Train Directional Models", type="primary")
        with c_note:
            st.caption("Start here after training the normal ML models. This may take a little time if XGBoost/LightGBM/CatBoost classifiers are available.")

        if train_dir or st.session_state.directional_results is None or st.session_state.directional_asset != selected_asset:
            with st.spinner("Training Up/Down directional classifiers..."):
                dir_results = train_directional_models(data, pp, include_heavy=True)
                st.session_state.directional_results = dir_results
                st.session_state.directional_asset = selected_asset
            st.success(f"Directional models trained for {selected_asset}.")

        dir_results = st.session_state.directional_results

        if dir_results:
            st.markdown("### Directional Model Leaderboard")
            dir_board = directional_leaderboard(dir_results)
            st.dataframe(dir_board, use_container_width=True)

            valid_names = [name for name, res in dir_results.items() if len(res.probabilities_test) == len(data.prices_test)]
            if not valid_names:
                st.error("No valid directional model produced probability outputs.")
                st.stop()

            default_idx = 0
            if not dir_board.empty and "Model" in dir_board.columns:
                best_candidate = str(dir_board.iloc[0]["Model"])
                if best_candidate in valid_names:
                    default_idx = valid_names.index(best_candidate)

            st.markdown("### Probability-Based Backtest")
            model_name = st.selectbox("Select directional model", valid_names, index=default_idx)
            probability_threshold_pct = st.slider(
                "Probability threshold (%)",
                min_value=50,
                max_value=75,
                value=55,
                step=1,
                help="Long when P(up) is above this threshold. With shorting ON, short when P(up) is below 100-threshold.",
            )
            transaction_cost_pct = st.number_input(
                "Transaction cost per trade (%)",
                min_value=0.0,
                max_value=5.0,
                value=0.10,
                step=0.05,
            )
            allow_short = st.checkbox("Allow shorting for directional model", value=False)

            selected_result = dir_results[model_name]
            try:
                metrics, equity = run_directional_probability_backtest(
                    data,
                    selected_result.probabilities_test,
                    probability_threshold=probability_threshold_pct / 100.0,
                    transaction_cost=transaction_cost_pct / 100.0,
                    allow_short=allow_short,
                )
            except Exception as exc:
                st.error(f"Directional backtest failed: {exc}")
                st.stop()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
            c2.metric("Annualized Return", f"{metrics['annualized_return_pct']:.2f}%")
            c3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            c4.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Win Rate", f"{metrics['win_rate_pct']:.2f}%")
            c6.metric("Trades", int(metrics["number_of_trades"]))
            c7.metric("Buy & Hold Return", f"{metrics['buy_hold_return_pct']:.2f}%")
            c8.metric("Strategy - BuyHold", f"{metrics['strategy_minus_buy_hold_pct']:.2f}%")

            st.markdown("### Directional Strategy vs Buy-and-Hold")
            st.line_chart(equity[["strategy_equity", "buy_hold_equity"]])

            st.markdown("### Directional Position")
            st.line_chart(equity[["position"]])

            with st.expander("View directional backtest input data"):
                st.dataframe(equity.tail(50), use_container_width=True)



# ════════════════════════════════════════════════════════════════
# PAGE: FEATURE INTELLIGENCE
# ════════════════════════════════════════════════════════════════

elif page == "🧠 Feature Intelligence":
    st.markdown(f'<p class="main-header">🧠 Phase 5 Feature Intelligence — {selected_asset}</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 5 does not try to hide weak validation. It adds better market features first: "
        "volatility regime, trend strength, DXY/VIX/TNX pressure, rolling correlations, relative strength, "
        "drawdowns, and breakout/breakdown pressure. Then Phase 4A tells us honestly whether they help."
    )

    st.markdown("### Build Feature Audit")
    st.caption(
        "This page checks whether the Phase 5 FI_* features are being created for the selected target asset. "
        "It is a feature-quality page, not a trading-signal page."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        start_for_features = st.text_input("Start date", value="2015-01-01", key="phase5_start_date")
    with c2:
        force_refresh_features = st.checkbox("Force fresh market download", value=False, key="phase5_force_refresh")

    if st.button("🔬 Run Phase 5 Feature Audit", type="primary"):
        with st.spinner("Building Phase 5 features..."):
            try:
                raw = load_raw_data(start_for_features, use_cache=not force_refresh_features)
                features = build_features(raw, target_col=target_col)
                audit = build_feature_intelligence_report(features, target_col=target_col)
                st.session_state.phase5_audit = audit
                st.session_state.phase5_features_preview = features[phase5_feature_columns(features)].tail(30)
            except Exception as exc:
                st.error(f"Phase 5 feature audit failed: {exc}")
                st.stop()

    audit = st.session_state.get("phase5_audit")
    preview = st.session_state.get("phase5_features_preview")

    if audit is None:
        st.info("Run the audit to confirm Phase 5 feature generation.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Target", audit.target_col)
        m2.metric("Rows", audit.total_rows)
        m3.metric("Total Columns", audit.total_columns)
        m4.metric("Phase 5 Features", audit.phase5_columns)

        if audit.phase5_columns < 40:
            st.warning("Few Phase 5 features were created. Check available asset/macro columns in the dataset.")
        else:
            st.success("Phase 5 feature intelligence is active.")

        tab1, tab2, tab3 = st.tabs(["Feature Families", "Missingness", "Preview"])
        with tab1:
            st.markdown("### Feature Family Counts")
            st.dataframe(audit.family_counts, use_container_width=True)
            st.markdown("### Sample Phase 5 Columns")
            st.write(audit.sample_columns)
        with tab2:
            st.markdown("### Missingness After Cleaning")
            st.caption("This should usually be near zero because the feature module forward-fills past values and drops only early warm-up rows.")
            st.dataframe(audit.missing_summary, use_container_width=True)
        with tab3:
            st.markdown("### Latest FI_* Feature Values")
            if preview is not None and not preview.empty:
                st.dataframe(preview, use_container_width=True)
            else:
                st.info("No preview data available.")

    st.markdown("---")
    st.info(
        "After this audit passes, go to **🌐 Multi-Asset Matrix** and run all assets again with "
        "**Use Phase 5 feature intelligence** enabled. The goal is not instant high trust; the goal is to see whether "
        "RMSE-vs-naive, direction accuracy, and trust score improve honestly."
    )


# ════════════════════════════════════════════════════════════════
# PAGE: RESEARCH VALIDATION
# ════════════════════════════════════════════════════════════════

elif page == "🧪 Research Validation":
    st.markdown(f'<p class="main-header">🧪 Research-Grade Validation — {selected_asset}</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This page is the hard-truth layer. It checks baselines, leakage risk, regime weakness, "
        "walk-forward stability, and a conservative model trust score. High R² alone is not treated as proof."
    )

    if not st.session_state.trained:
        st.warning("⚠️ Train ML models first. Go to **Train Models**.")
    else:
        _stop_if_asset_mismatch(selected_asset)
        trainer = st.session_state.trainer
        pp = st.session_state.pp
        data = st.session_state.data
        df_features = st.session_state.df_features

        tab1, tab2, tab3, tab4 = st.tabs([
            "Trust Score",
            "Regime Performance",
            "Walk-Forward",
            "Leakage Audit",
        ])

        with tab1:
            st.markdown("### Conservative Model Trust Score")
            st.caption(
                "Score uses RMSE improvement over Naive, directional accuracy, long-only backtest risk, "
                "drawdown, and overfit gap. It intentionally gives little credit to high R² alone."
            )
            try:
                report = build_validation_report(trainer, data, df_features)
                st.dataframe(report.trust_scores, use_container_width=True)
                if not report.trust_scores.empty:
                    best = report.trust_scores.iloc[0]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Best Trust Score", f"{float(best['TrustScore']):.2f}/100")
                    c2.metric("Best Candidate", str(best["Model"]))
                    c3.metric("Verdict", str(best["Verdict"]))
                    if float(best["TrustScore"]) < 55:
                        st.error("No model is currently strong enough for confident trading signals. This is useful truth, not failure.")
                    elif float(best["TrustScore"]) < 75:
                        st.warning("Best model is a medium-trust research candidate. Use risk controls and further validation.")
                    else:
                        st.success("A high-trust candidate exists, but still requires live paper-trading validation.")

                st.markdown("### Price Baselines")
                st.dataframe(report.baseline_board, use_container_width=True)
            except Exception as exc:
                st.error(f"Validation report failed: {exc}")

        with tab2:
            st.markdown("### Regime-wise Model Weakness")
            st.caption("A serious model should not only look good overall; it should reveal where it fails: bull, bear, sideways, high-volatility, low-volatility.")
            model_name = st.selectbox("Select model for regime analysis", list(trainer.results.keys()), key="regime_model")
            result = trainer.results[model_name]
            try:
                regime_board = regime_performance(data, result.predictions_test)
                st.dataframe(regime_board, use_container_width=True)
                if not regime_board.empty:
                    worst = regime_board.sort_values("DirectionalAccuracy", ascending=True).iloc[0]
                    st.warning(
                        f"Weakest regime for **{model_name}**: {worst['RegimeType']} = **{worst['Regime']}** "
                        f"with directional accuracy **{worst['DirectionalAccuracy']:.2f}%**."
                    )
            except Exception as exc:
                st.error(f"Regime analysis failed: {exc}")

        with tab3:
            st.markdown("### Walk-Forward Validation")
            st.caption(
                "This retrains the selected model across rolling time splits. It is slower but much closer to real research than a single train/test split."
            )
            wf_model_name = st.selectbox("Select model for walk-forward", list(trainer.results.keys()), key="wf_model")
            n_splits = st.slider("Walk-forward folds", min_value=3, max_value=8, value=5, step=1)
            max_train = st.selectbox("Training window", ["Expanding", "Last 1000 rows", "Last 1500 rows"], index=0)
            max_train_size = None if max_train == "Expanding" else int(max_train.split()[1])

            if st.button("🚀 Run Walk-Forward Validation", type="primary"):
                with st.spinner("Running walk-forward validation. This can take time for heavier models..."):
                    try:
                        base_result = trainer.results[wf_model_name]
                        folds, summary = walk_forward_validate_model(
                            base_result.model,
                            wf_model_name,
                            data,
                            pp,
                            n_splits=n_splits,
                            max_train_size=max_train_size,
                        )
                        st.session_state.phase4_wf_folds = folds
                        st.session_state.phase4_wf_summary = summary
                    except Exception as exc:
                        st.error(f"Walk-forward validation failed: {exc}")
                        st.stop()

            if "phase4_wf_summary" in st.session_state:
                summary = st.session_state.phase4_wf_summary
                folds = st.session_state.phase4_wf_folds
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean RMSE", f"{summary['MeanRMSE']:.2f}")
                c2.metric("RMSE Std", f"{summary['StdRMSE']:.2f}")
                c3.metric("Mean Direction", f"{summary['MeanDirectionalAccuracy']:.2f}%")
                c4.metric("Stability Score", f"{summary['StabilityScore']:.2f}/100")
                st.dataframe(folds, use_container_width=True)

        with tab4:
            st.markdown("### Leakage / Alignment Audit")
            st.caption("This checks target alignment, scaler leakage, target exclusion, and extreme feature correlation with next-day target.")
            try:
                report = build_validation_report(trainer, data, df_features)
                st.dataframe(report.leakage_report, use_container_width=True)
                bad = report.leakage_report[report.leakage_report["Status"].isin(["FAIL", "ERROR"])]
                review = report.leakage_report[report.leakage_report["Status"].eq("REVIEW")]
                if not bad.empty:
                    st.error("Critical validation issue found. Do not trust model results until fixed.")
                elif not review.empty:
                    st.warning("Some checks need review. This does not always mean leakage, but it must be inspected.")
                else:
                    st.success("Core leakage/alignment checks passed.")
            except Exception as exc:
                st.error(f"Leakage audit failed: {exc}")



# ════════════════════════════════════════════════════════════════
# PAGE: MULTI-ASSET VALIDATION MATRIX
# ════════════════════════════════════════════════════════════════

elif page == "🌐 Multi-Asset Matrix":
    st.markdown('<p class="main-header">🌐 Multi-Asset Research Validation Matrix</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This is the serious all-asset layer. It validates every configured asset, not only Gold. "
        "The goal is to identify which assets are actually modelable, which are weak, and which must not be trusted for signals."
    )

    st.markdown("### Validation Scope")
    all_assets = get_asset_names()
    selected_assets_for_matrix = st.multiselect(
        "Assets to validate",
        all_assets,
        default=all_assets,
        help="For a serious final run, keep all assets selected. For quick debugging, choose 1-2 assets.",
    )

    model_depth = st.selectbox(
        "Model depth",
        [
            "fast",
            "core",
            "full",
        ],
        index=0,
        help=(
            "fast = Linear Regression + Decision Tree; "
            "core = Linear/RF/XGBoost/LightGBM/CatBoost; "
            "full = all ML models including SVR. Use core/full for serious validation."
        ),
    )

    use_phase5_matrix = st.checkbox(
        "Use Phase 5 feature intelligence",
        value=True,
        help="Adds FI_* features: regimes, cross-asset relationships, macro pressure, volatility, and trend intelligence.",
    )

    include_wf_matrix = st.checkbox(
        "Include walk-forward check for best model per asset (slower)",
        value=False,
    )
    wf_splits_matrix = st.slider("Walk-forward folds", 3, 6, 3, disabled=not include_wf_matrix)

    st.caption(
        "Recommendation: use **fast** first to confirm the pipeline, then use **core** for a serious all-asset run. "
        "Full mode may take longer."
    )

    run_matrix = st.button("🚀 Run Multi-Asset Validation", type="primary")

    if run_matrix:
        if not selected_assets_for_matrix:
            st.error("Select at least one asset.")
            st.stop()

        progress = st.progress(0, text="Loading raw market data...")

        def _matrix_progress(i, total, msg):
            pct = int((i - 1) / max(total, 1) * 100)
            progress.progress(pct, text=msg)

        with st.spinner("Running multi-asset research validation. This may take time..."):
            try:
                df_raw = load_raw_data("2015-01-01", use_cache=True)
                report = run_multiasset_validation(
                    raw_df=df_raw,
                    asset_names=selected_assets_for_matrix,
                    model_set=model_depth,
                    include_walk_forward=include_wf_matrix,
                    walk_forward_splits=wf_splits_matrix,
                    use_phase5_features=use_phase5_matrix,
                    progress_callback=_matrix_progress,
                )
                st.session_state.multiasset_validation_report = report
                st.session_state.multiasset_validation_settings = {
                    "assets": selected_assets_for_matrix,
                    "model_depth": model_depth,
                    "include_walk_forward": include_wf_matrix,
                    "walk_forward_splits": wf_splits_matrix,
                    "use_phase5_features": use_phase5_matrix,
                }
                progress.progress(100, text="Multi-asset validation complete.")
            except Exception as exc:
                st.error(f"Multi-asset validation failed: {exc}")
                st.stop()

    report = st.session_state.multiasset_validation_report

    if report is None:
        st.info("Run the validation to generate the all-asset trust matrix.")
    else:
        settings = st.session_state.get("multiasset_validation_settings") or {}
        st.caption(f"Last run settings: {settings}")

        summary = report.asset_summary
        status = summarize_asset_status(summary)

        st.markdown("### Asset-Level Verdict")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("High Trust", status.get("High", 0))
        c2.metric("Medium Trust", status.get("Medium", 0))
        c3.metric("Low Trust", status.get("Low", 0))
        c4.metric("Do Not Trust", status.get("DoNotTrust", 0))

        if summary is not None and not summary.empty:
            st.dataframe(summary, use_container_width=True)

            do_not = summary[summary["AssetVerdict"].astype(str).str.contains("Do not trust", case=False, na=False)]
            if not do_not.empty:
                st.error(
                    "Some assets are currently marked **Do Not Trust for Signals**. "
                    "That is not failure; it tells us where the model needs better features or a different objective."
                )
            else:
                st.success("No selected asset was marked hard Do-Not-Trust in this run.")
        else:
            st.warning("No asset summary was produced.")

        tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs([
            "Model Leaderboard",
            "Price Baselines",
            "Leakage Matrix",
            "Walk-Forward",
            "Errors",
        ])

        with tab_a:
            st.markdown("### All Models Across All Assets")
            st.caption("Sorts all asset/model pairs by conservative trust score. High R² alone does not dominate this score.")
            if report.model_leaderboard is not None and not report.model_leaderboard.empty:
                st.dataframe(report.model_leaderboard, use_container_width=True)
            else:
                st.info("No model leaderboard available.")

        with tab_b:
            st.markdown("### Baselines Across All Assets")
            st.caption("This shows whether an ML model beats simple logic such as tomorrow=today or moving averages.")
            if report.baseline_leaderboard is not None and not report.baseline_leaderboard.empty:
                st.dataframe(report.baseline_leaderboard, use_container_width=True)
            else:
                st.info("No baseline table available.")

        with tab_c:
            st.markdown("### Leakage / Alignment Checks Across Assets")
            if report.leakage_matrix is not None and not report.leakage_matrix.empty:
                st.dataframe(report.leakage_matrix, use_container_width=True)
                failures = report.leakage_matrix[report.leakage_matrix["Status"].isin(["FAIL", "ERROR"])]
                reviews = report.leakage_matrix[report.leakage_matrix["Status"].eq("REVIEW")]
                if not failures.empty:
                    st.error("Critical leakage/alignment failures exist. Do not trust affected assets.")
                elif not reviews.empty:
                    st.warning("Some checks need manual review. This is normal in serious research.")
                else:
                    st.success("Core leakage/alignment checks passed for selected assets.")
            else:
                st.info("No leakage matrix available.")

        with tab_d:
            st.markdown("### Walk-Forward Summary")
            if report.walk_forward_summary is not None and not report.walk_forward_summary.empty:
                st.dataframe(report.walk_forward_summary, use_container_width=True)
            else:
                st.info("Walk-forward was not run, or no walk-forward summary was produced.")

        with tab_e:
            st.markdown("### Assets / Models That Failed")
            if report.errors is not None and not report.errors.empty:
                st.dataframe(report.errors, use_container_width=True)
            else:
                st.success("No asset-level errors in the last multi-asset validation run.")


# ════════════════════════════════════════════════════════════════
# PAGE: DIRECT FORECAST MODELS
# ════════════════════════════════════════════════════════════════

elif page == "🎯 Direct Forecast Models":
    st.markdown('<p class="main-header">🎯 Direct Forecast Models</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 6 trains direct horizon models for future returns and direction. "
        "It does not recursively roll a 30-day price forecast forward, and it must beat naive baselines before it is useful."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        direct_asset = st.selectbox(
            "Asset",
            get_asset_names(),
            index=get_asset_names().index(selected_asset) if selected_asset in get_asset_names() else 0,
            key="direct_forecast_asset",
        )
    with col_b:
        direct_horizon_label = st.selectbox(
            "Horizon",
            [f"{h}D" for h in DIRECT_FORECAST_HORIZONS],
            index=1,
            key="direct_forecast_horizon",
        )
        direct_horizon = int(str(direct_horizon_label).replace("D", ""))
    with col_c:
        direct_depth = st.selectbox(
            "Model depth",
            ["fast", "core"],
            index=0,
            help="fast = linear/logistic + decision tree. core adds random forest and gradient boosting.",
            key="direct_forecast_depth",
        )
    with col_d:
        direct_use_phase5 = st.checkbox(
            "Use Phase 5 feature intelligence",
            value=True,
            help="Adds FI_* market intelligence features before direct horizon targets are created.",
            key="direct_forecast_phase5",
        )

    run_direct = st.button("🚀 Run Direct Forecast Models", type="primary")

    if run_direct:
        with st.spinner("Training direct horizon return and direction models..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                direct_report = run_direct_forecast_report(
                    raw_df=raw_df,
                    asset_name=direct_asset,
                    horizon=direct_horizon,
                    model_depth=direct_depth,
                    use_phase5_features=direct_use_phase5,
                )
                st.session_state.direct_forecast_report = direct_report
                st.session_state.direct_forecast_settings = {
                    "asset": direct_asset,
                    "horizon": direct_horizon,
                    "model_depth": direct_depth,
                    "use_phase5_features": direct_use_phase5,
                }
            except Exception as exc:
                st.error(f"Direct forecast model run failed: {exc}")
                st.stop()

    direct_report = st.session_state.direct_forecast_report

    if direct_report is None:
        st.info("Run direct forecast models to generate a horizon-specific leaderboard.")
    else:
        settings = st.session_state.get("direct_forecast_settings") or {}
        st.caption(f"Last run settings: {settings}")

        leaderboard = direct_report.leaderboard
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{direct_report.rows:,}")
        c2.metric("Features", f"{direct_report.feature_count:,}")
        c3.metric("Test Rows", f"{direct_report.test_rows:,}")
        c4.metric("Horizon", f"{direct_report.horizon}D")

        if leaderboard is None or leaderboard.empty:
            st.warning("No direct model leaderboard was produced.")
        else:
            best = leaderboard.iloc[0]
            st.markdown("### Direct Horizon Leaderboard")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Best Trust Score", f"{float(best['TrustScore']):.2f}/100")
            b2.metric("Best RMSE vs Naive", f"{float(best['RMSE_vs_Naive_%']):+.2f}%")
            b3.metric("Best Direction vs Baseline", f"{float(best['Direction_vs_Baseline_%']):+.2f}%")
            b4.metric("Best Verdict", str(best["Verdict"]))

            st.dataframe(leaderboard, use_container_width=True)

            failed_return = leaderboard[leaderboard["RMSE_vs_Naive_%"].astype(float) < 0]
            failed_direction = leaderboard[leaderboard["Direction_vs_Baseline_%"].astype(float) <= 0]
            if len(failed_return) == len(leaderboard):
                st.error("All direct models failed the zero-return RMSE baseline. Do not use this horizon for signals.")
            elif not failed_return.empty:
                st.warning("Some direct models failed the zero-return RMSE baseline.")

            if len(failed_direction) == len(leaderboard):
                st.error("All direct models failed the direction baseline. Directional signal is not trustworthy.")
            elif not failed_direction.empty:
                st.warning("Some direct models failed the direction baseline.")

            do_not = leaderboard[leaderboard["Verdict"].astype(str).str.contains("Do not trust", case=False, na=False)]
            if not do_not.empty:
                st.error("One or more direct horizon models are marked Do Not Trust for Signals.")

            csv = leaderboard.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Direct Leaderboard CSV",
                data=csv,
                file_name=f"direct_forecast_{direct_report.asset}_{direct_report.horizon}D.csv".replace(" ", "_"),
                mime="text/csv",
            )

        tab_base, tab_errors, tab_alignment = st.tabs(["Baselines", "Errors", "Alignment"])

        with tab_base:
            st.markdown("### Naive Baselines")
            if direct_report.baseline_board is not None and not direct_report.baseline_board.empty:
                st.dataframe(direct_report.baseline_board, use_container_width=True)
            else:
                st.warning("No baseline board was produced.")

        with tab_errors:
            st.markdown("### Model Failures")
            if direct_report.errors is not None and not direct_report.errors.empty:
                st.dataframe(direct_report.errors, use_container_width=True)
            else:
                st.success("No model-level errors in the last direct forecast run.")

        with tab_alignment:
            dataset = direct_report.dataset
            if dataset is not None:
                st.markdown("### Direct Target Alignment")
                st.write(
                    {
                        "return_target": dataset.return_target_col,
                        "direction_target": dataset.direction_target_col,
                        "volatility_target": dataset.volatility_target_col,
                        "dropped_tail_rows": dataset.dropped_tail_rows,
                        "train_start": str(dataset.train_index.min().date()) if len(dataset.train_index) else "",
                        "train_end": str(dataset.train_index.max().date()) if len(dataset.train_index) else "",
                        "test_start": str(dataset.test_index.min().date()) if len(dataset.test_index) else "",
                        "test_end": str(dataset.test_index.max().date()) if len(dataset.test_index) else "",
                    }
                )
                leaked = [c for c in dataset.feature_cols if c.startswith(("future_return_", "future_direction_", "future_realized_vol_"))]
                if leaked:
                    st.error(f"Future target columns leaked into features: {leaked}")
                else:
                    st.success("No future target columns are present in the model feature set.")
            else:
                st.info("No dataset metadata available.")


# ════════════════════════════════════════════════════════════════
# PAGE: DIRECT HORIZON SCANNER
# ════════════════════════════════════════════════════════════════

elif page == "🧭 Direct Horizon Scanner":
    st.markdown('<p class="main-header">🧭 Direct Horizon Scanner</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 6B scans direct forecast models across assets and horizons. "
        "Do Not Trust results are shown plainly; this page is for finding honest pockets of signal, not decorating weak models."
    )

    all_scan_assets = get_asset_names()
    scan_col_a, scan_col_b, scan_col_c = st.columns([2, 2, 1])
    with scan_col_a:
        scan_assets = st.multiselect(
            "Assets",
            all_scan_assets,
            default=all_scan_assets,
            help="Serious validation should include every configured asset.",
            key="direct_horizon_scan_assets",
        )
    with scan_col_b:
        scan_horizon_labels = st.multiselect(
            "Horizons",
            [f"{h}D" for h in DIRECT_FORECAST_HORIZONS],
            default=[f"{h}D" for h in DIRECT_FORECAST_HORIZONS],
            key="direct_horizon_scan_horizons",
        )
        scan_horizons = [int(str(label).replace("D", "")) for label in scan_horizon_labels]
    with scan_col_c:
        scan_depth = st.selectbox(
            "Model depth",
            ["core", "fast"],
            index=0,
            help="core is the default for scanner runs; fast is useful for smoke tests.",
            key="direct_horizon_scan_depth",
        )
        scan_use_phase5 = st.checkbox(
            "Use Phase 5",
            value=True,
            help="Adds FI_* market intelligence features before direct horizon targets are created.",
            key="direct_horizon_scan_phase5",
        )

    run_scan = st.button("🚀 Run Asset × Horizon Scan", type="primary")

    if run_scan:
        if not scan_assets:
            st.error("Select at least one asset.")
            st.stop()
        if not scan_horizons:
            st.error("Select at least one horizon.")
            st.stop()

        progress = st.progress(0, text="Preparing direct horizon scan...")

        def _scan_progress(done, total, msg):
            pct = int(done / max(total, 1) * 100)
            progress.progress(pct, text=msg)

        with st.spinner("Running asset × horizon direct forecast scanner..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                scan_report = run_asset_horizon_scan(
                    raw_df=raw_df,
                    asset_names=scan_assets,
                    horizons=scan_horizons,
                    model_depth=scan_depth,
                    use_phase5_features=scan_use_phase5,
                    progress_callback=_scan_progress,
                )
                st.session_state.direct_horizon_scan_report = scan_report
                st.session_state.direct_horizon_scan_settings = scan_report.settings
                progress.progress(100, text="Direct horizon scan complete.")
            except Exception as exc:
                st.error(f"Direct horizon scan failed: {exc}")
                st.stop()

    scan_report = st.session_state.direct_horizon_scan_report
    if scan_report is None:
        st.info("Run the scanner to evaluate every selected asset × horizon combination.")
    else:
        settings = st.session_state.get("direct_horizon_scan_settings") or {}
        st.caption(f"Last run settings: {settings}")

        counts = scan_report.status_counts or {}
        count_a, count_b, count_c, count_d = st.columns(4)
        count_a.metric("High", counts.get("High", 0))
        count_b.metric("Medium", counts.get("Medium", 0))
        count_c.metric("Low", counts.get("Low", 0))
        count_d.metric("Do Not Trust", counts.get("DoNotTrust", 0))

        summary = scan_report.asset_horizon_summary
        if summary is None or summary.empty:
            st.warning("No scanner summary was produced.")
        else:
            total_rows = len(summary)
            if counts.get("DoNotTrust", 0) == total_rows:
                st.error("All scanned asset-horizon combinations are Do Not Trust for Signals.")

            st.markdown("### Asset × Horizon Summary")
            st.dataframe(summary, use_container_width=True)

            csv = summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Scanner CSV",
                data=csv,
                file_name="direct_horizon_scanner.csv",
                mime="text/csv",
            )

        tab_top, tab_worst, tab_errors, tab_leakage = st.tabs([
            "Top Promising",
            "Worst Failed",
            "Errors",
            "Leakage Check",
        ])

        with tab_top:
            st.markdown("### Top Promising Asset-Horizon Combinations")
            if scan_report.top_promising is not None and not scan_report.top_promising.empty:
                st.dataframe(scan_report.top_promising, use_container_width=True)
            else:
                st.warning("No non-DoNotTrust combinations were found in this scan.")

        with tab_worst:
            st.markdown("### Worst Failed Combinations")
            if scan_report.worst_failed is not None and not scan_report.worst_failed.empty:
                st.dataframe(scan_report.worst_failed, use_container_width=True)
            else:
                st.info("No failed combinations available.")

        with tab_errors:
            st.markdown("### Scan Errors")
            if scan_report.errors is not None and not scan_report.errors.empty:
                st.dataframe(scan_report.errors, use_container_width=True)
            else:
                st.success("No scanner errors in the last run.")

        with tab_leakage:
            st.markdown("### Future Target Feature Check")
            if summary is not None and not summary.empty and "FeatureLeakageCount" in summary.columns:
                leaks = summary[summary["FeatureLeakageCount"].fillna(0).astype(float) > 0]
                if leaks.empty:
                    st.success("No future_return_*, future_direction_*, or future_realized_vol_* columns were used as features.")
                else:
                    st.error("Future target columns leaked into scanner features. Do not trust this run.")
                    st.dataframe(leaks[["Asset", "Horizon", "FeatureLeakageColumns"]], use_container_width=True)
            else:
                st.info("No leakage metadata available.")


# ════════════════════════════════════════════════════════════════
# PAGE: SIGNAL RESEARCH SCANNER
# ════════════════════════════════════════════════════════════════

elif page == "🧪 Signal Research Scanner":
    st.markdown('<p class="main-header">🧪 Signal Research Scanner</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 7D is a validation-locked research scanner. It uses non-overlapping realistic trades only, "
        "selects thresholds and cooldowns from validation evidence only, and never labels results production-ready."
    )

    all_signal_scan_assets = get_asset_names()
    scan_col_a, scan_col_b, scan_col_c = st.columns([2, 2, 1])
    with scan_col_a:
        signal_scan_assets = st.multiselect(
            "Assets",
            all_signal_scan_assets,
            default=all_signal_scan_assets,
            help="Serious signal research should include every configured asset.",
            key="signal_research_scan_assets",
        )
    with scan_col_b:
        signal_scan_horizon_labels = st.multiselect(
            "Horizons",
            [f"{h}D" for h in DIRECT_FORECAST_HORIZONS],
            default=[f"{h}D" for h in DIRECT_FORECAST_HORIZONS],
            key="signal_research_scan_horizons",
        )
        signal_scan_horizons = [int(str(label).replace("D", "")) for label in signal_scan_horizon_labels]
    with scan_col_c:
        signal_scan_depth = st.selectbox(
            "Model depth",
            ["core", "fast"],
            index=0,
            help="core is the serious default; fast is useful for smoke tests.",
            key="signal_research_scan_depth",
        )
        signal_scan_phase5 = st.checkbox(
            "Use Phase 5",
            value=True,
            help="Adds FI_* features before direct horizon targets are created.",
            key="signal_research_scan_phase5",
        )

    rule_col_a, rule_col_b, rule_col_c = st.columns([2, 2, 1])
    with rule_col_a:
        threshold_candidates = st.multiselect(
            "Threshold candidates",
            [0.50, 0.55, 0.60, 0.65, 0.70],
            default=[0.50, 0.55, 0.60, 0.65, 0.70],
            format_func=lambda x: f"{x:.2f}",
            key="signal_research_thresholds",
        )
    with rule_col_b:
        cooldown_candidates = st.multiselect(
            "Cooldown candidates",
            [0, 2, 5],
            default=[0, 2, 5],
            format_func=lambda x: f"{int(x)} rows",
            key="signal_research_cooldowns",
        )
    with rule_col_c:
        validation_segment_pct = st.slider(
            "Validation segment %",
            min_value=30,
            max_value=70,
            value=50,
            step=5,
            help="Chronological share of Phase 6 out-of-sample rows used for threshold and cooldown selection.",
            key="signal_research_validation_segment",
        )

    st.caption(
        "Signal mode: long_only. Backtest style: non_overlapping_realistic. Threshold policy: validation_locked. "
        "Cooldown selection basis: validation score only."
    )

    run_signal_scan = st.button("🚀 Run Signal Research Scan", type="primary")

    if run_signal_scan:
        if not signal_scan_assets:
            st.error("Select at least one asset.")
            st.stop()
        if not signal_scan_horizons:
            st.error("Select at least one horizon.")
            st.stop()
        if not threshold_candidates:
            st.error("Select at least one threshold candidate.")
            st.stop()
        if not cooldown_candidates:
            st.error("Select at least one cooldown candidate.")
            st.stop()

        progress = st.progress(0, text="Preparing signal research scan...")

        def _signal_scan_progress(done, total, msg):
            pct = int(done / max(total, 1) * 100)
            progress.progress(pct, text=msg)

        with st.spinner("Running validation-locked signal research scanner..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                signal_scan_report = run_signal_research_scan(
                    raw_df=raw_df,
                    asset_names=signal_scan_assets,
                    horizons=signal_scan_horizons,
                    model_depth=signal_scan_depth,
                    use_phase5_features=signal_scan_phase5,
                    signal_mode="long_only",
                    threshold_candidates=threshold_candidates,
                    cooldown_candidates=cooldown_candidates,
                    validation_fraction=float(validation_segment_pct) / 100.0,
                    progress_callback=_signal_scan_progress,
                )
                st.session_state.signal_research_scan_report = signal_scan_report
                st.session_state.signal_research_scan_settings = signal_scan_report.settings
                progress.progress(100, text="Signal research scan complete.")
            except Exception as exc:
                st.error(f"Signal research scan failed: {exc}")
                st.stop()

    signal_scan_report = st.session_state.signal_research_scan_report
    if signal_scan_report is None:
        st.info("Run the scanner to evaluate validation-locked realistic signals across selected assets and horizons.")
    else:
        settings = st.session_state.get("signal_research_scan_settings") or {}
        st.caption(f"Last run settings: {settings}")

        counts = signal_scan_report.verdict_counts or {}
        count_cols = st.columns(max(1, min(4, len(counts) or 1)))
        if counts:
            for idx, (label, value) in enumerate(counts.items()):
                count_cols[idx % len(count_cols)].metric(label, value)
        else:
            count_cols[0].metric("Results", 0)

        full_results = signal_scan_report.full_results
        top_candidates = signal_scan_report.top_robust_candidates
        failed_candidates = signal_scan_report.failed_candidates

        if full_results is None or full_results.empty:
            st.warning("No signal scanner results were produced.")
        else:
            if top_candidates is None or top_candidates.empty:
                st.error("All scanned combinations failed the robust validation-locked signal criteria.")

            st.markdown("### Full Signal Research Results")
            st.dataframe(full_results, use_container_width=True)
            full_csv = full_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Full Signal Scan CSV",
                data=full_csv,
                file_name="signal_research_scanner.csv",
                mime="text/csv",
            )

        tab_top, tab_failed, tab_candidates, tab_errors = st.tabs([
            "Top Robust Candidates",
            "Failed / Weak",
            "Cooldown Candidates",
            "Errors",
        ])

        with tab_top:
            st.markdown("### Top Robust Candidates")
            if top_candidates is not None and not top_candidates.empty:
                st.dataframe(top_candidates, use_container_width=True)
            else:
                st.warning("No robust research candidates survived validation-locked realistic evaluation.")

        with tab_failed:
            st.markdown("### Failed And Weak Combinations")
            if failed_candidates is not None and not failed_candidates.empty:
                st.dataframe(failed_candidates, use_container_width=True)
            else:
                st.success("No failed or weak combinations in this scan.")

        with tab_candidates:
            st.markdown("### Cooldown Candidate Diagnostics")
            candidate_results = signal_scan_report.candidate_results
            if candidate_results is not None and not candidate_results.empty:
                st.caption("Cooldown candidates are validation-only diagnostics. Locked test is evaluated once after the cooldown is selected.")
                st.dataframe(candidate_results, use_container_width=True)
                candidate_csv = candidate_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Cooldown Candidate CSV",
                    data=candidate_csv,
                    file_name="signal_research_cooldown_candidates.csv",
                    mime="text/csv",
                )
            else:
                st.info("No cooldown candidate diagnostics available.")

        with tab_errors:
            st.markdown("### Scan Errors")
            if signal_scan_report.errors is not None and not signal_scan_report.errors.empty:
                st.dataframe(signal_scan_report.errors, use_container_width=True)
            else:
                st.success("No scanner errors in the last run.")


# ════════════════════════════════════════════════════════════════
# PAGE: CANDIDATE DEEP DIAGNOSTICS
# ════════════════════════════════════════════════════════════════

elif page == "🔬 Candidate Deep Diagnostics":
    st.markdown('<p class="main-header">🔬 Candidate Deep Diagnostics</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 7E diagnoses validation-locked signal candidates. Thresholds and cooldowns are selected from validation evidence only; "
        "locked-test data is used only for post-selection diagnostics. Nothing here is production-ready."
    )

    diag_assets = get_asset_names()
    diag_default_asset = diag_assets.index("Silver") if "Silver" in diag_assets else 0
    diag_col_a, diag_col_b, diag_col_c, diag_col_d = st.columns(4)
    with diag_col_a:
        diag_asset = st.selectbox(
            "Asset",
            diag_assets,
            index=diag_default_asset,
            key="candidate_diag_asset",
        )
    with diag_col_b:
        diag_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        diag_horizon_default = diag_horizon_labels.index("5D") if "5D" in diag_horizon_labels else 0
        diag_horizon_label = st.selectbox(
            "Horizon",
            diag_horizon_labels,
            index=diag_horizon_default,
            key="candidate_diag_horizon",
        )
        diag_horizon = int(str(diag_horizon_label).replace("D", ""))
    with diag_col_c:
        diag_depth = st.selectbox(
            "Model depth",
            ["core", "fast"],
            index=0,
            key="candidate_diag_depth",
        )
    with diag_col_d:
        diag_phase5 = st.checkbox(
            "Use Phase 5",
            value=True,
            key="candidate_diag_phase5",
        )

    diag_rule_a, diag_rule_b, diag_rule_c, diag_rule_d = st.columns([1, 2, 2, 1])
    with diag_rule_a:
        diag_mode = st.selectbox(
            "Signal mode",
            ["long_only", "long_short", "avoid_only"],
            index=0,
            key="candidate_diag_mode",
        )
    with diag_rule_b:
        diag_thresholds = st.multiselect(
            "Threshold candidates",
            [0.50, 0.55, 0.60, 0.65, 0.70],
            default=[0.50, 0.55, 0.60, 0.65, 0.70],
            format_func=lambda x: f"{x:.2f}",
            key="candidate_diag_thresholds",
        )
    with diag_rule_c:
        diag_cooldowns = st.multiselect(
            "Cooldown candidates",
            [0, 2, 5],
            default=[0, 2, 5],
            format_func=lambda x: f"{int(x)} rows",
            key="candidate_diag_cooldowns",
        )
    with diag_rule_d:
        diag_validation_pct = st.slider(
            "Validation %",
            min_value=30,
            max_value=70,
            value=50,
            step=5,
            key="candidate_diag_validation_pct",
        )

    run_candidate_diag = st.button("🚀 Run Candidate Diagnostics", type="primary")

    if run_candidate_diag:
        if not diag_thresholds:
            st.error("Select at least one threshold candidate.")
            st.stop()
        if not diag_cooldowns:
            st.error("Select at least one cooldown candidate.")
            st.stop()

        with st.spinner("Running validation-locked candidate diagnostics..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                diag_report = run_candidate_deep_diagnostics(
                    raw_df=raw_df,
                    asset_name=diag_asset,
                    horizon=diag_horizon,
                    model_depth=diag_depth,
                    use_phase5_features=diag_phase5,
                    signal_mode=diag_mode,
                    threshold_candidates=diag_thresholds,
                    cooldown_candidates=diag_cooldowns,
                    validation_fraction=float(diag_validation_pct) / 100.0,
                )
                st.session_state.candidate_diagnostics_report = diag_report
                st.session_state.candidate_diagnostics_settings = diag_report.settings
            except Exception as exc:
                st.error(f"Candidate diagnostics failed: {exc}")
                st.stop()

    diag_report = st.session_state.candidate_diagnostics_report
    if diag_report is None:
        st.info("Run diagnostics for any configured asset and horizon. Silver 5D is only the default candidate view.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('candidate_diagnostics_settings') or {}}")

        if diag_report.warnings:
            for warning in diag_report.warnings:
                if warning.startswith("BenchmarkWeakness") or warning.startswith("DrawdownRisk") or warning.startswith("CostFragile"):
                    st.error(warning)
                else:
                    st.warning(warning)

        st.markdown("### Candidate Summary")
        st.dataframe(diag_report.candidate_summary, use_container_width=True)
        st.download_button(
            "📥 Export Candidate Summary CSV",
            data=diag_report.candidate_summary.to_csv(index=False).encode("utf-8"),
            file_name="candidate_diagnostics_summary.csv",
            mime="text/csv",
        )

        tab_trade, tab_time, tab_drawdown, tab_sensitivity, tab_probability = st.tabs([
            "Trade Diagnostics",
            "Time Returns",
            "Equity / Drawdown",
            "Sensitivity",
            "Probability",
        ])

        with tab_trade:
            st.markdown("### Trade Diagnostics")
            st.dataframe(diag_report.trade_diagnostics, use_container_width=True)
            st.download_button(
                "📥 Export Trade Diagnostics CSV",
                data=diag_report.trade_diagnostics.to_csv(index=False).encode("utf-8"),
                file_name="candidate_trade_diagnostics.csv",
                mime="text/csv",
            )

            st.markdown("### Trade Log")
            st.dataframe(diag_report.trade_log, use_container_width=True)
            st.download_button(
                "📥 Export Trade Log CSV",
                data=diag_report.trade_log.to_csv(index=False).encode("utf-8"),
                file_name="candidate_trade_log.csv",
                mime="text/csv",
            )

        with tab_time:
            st.markdown("### Monthly Returns")
            st.dataframe(diag_report.monthly_returns, use_container_width=True)
            st.download_button(
                "📥 Export Monthly Returns CSV",
                data=diag_report.monthly_returns.to_csv(index=False).encode("utf-8"),
                file_name="candidate_monthly_returns.csv",
                mime="text/csv",
            )

            st.markdown("### Quarterly Returns")
            st.dataframe(diag_report.quarterly_returns, use_container_width=True)
            st.download_button(
                "📥 Export Quarterly Returns CSV",
                data=diag_report.quarterly_returns.to_csv(index=False).encode("utf-8"),
                file_name="candidate_quarterly_returns.csv",
                mime="text/csv",
            )

        with tab_drawdown:
            st.markdown("### Equity Curve")
            if diag_report.equity_curve is not None and not diag_report.equity_curve.empty:
                equity_chart = diag_report.equity_curve.set_index("Date")[["Equity"]]
                st.line_chart(equity_chart)
            st.dataframe(diag_report.equity_curve, use_container_width=True)
            st.download_button(
                "📥 Export Equity Curve CSV",
                data=diag_report.equity_curve.to_csv(index=False).encode("utf-8"),
                file_name="candidate_equity_curve.csv",
                mime="text/csv",
            )

            st.markdown("### Drawdown Curve")
            if diag_report.drawdown_curve is not None and not diag_report.drawdown_curve.empty:
                dd_chart = diag_report.drawdown_curve.set_index("Date")[["Drawdown_%"]]
                st.line_chart(dd_chart)
            st.dataframe(diag_report.drawdown_curve, use_container_width=True)
            st.download_button(
                "📥 Export Drawdown Curve CSV",
                data=diag_report.drawdown_curve.to_csv(index=False).encode("utf-8"),
                file_name="candidate_drawdown_curve.csv",
                mime="text/csv",
            )

        with tab_sensitivity:
            st.markdown("### Cost Sensitivity")
            st.caption("Uses the validation-selected threshold and cooldown, then re-evaluates locked-test economics at each transaction cost.")
            st.dataframe(diag_report.cost_sensitivity, use_container_width=True)
            st.download_button(
                "📥 Export Cost Sensitivity CSV",
                data=diag_report.cost_sensitivity.to_csv(index=False).encode("utf-8"),
                file_name="candidate_cost_sensitivity.csv",
                mime="text/csv",
            )

            st.markdown("### Validation Split Sensitivity")
            st.caption("Each split re-runs validation-locked selection independently. Locked-test metrics are post-selection diagnostics.")
            st.dataframe(diag_report.validation_split_sensitivity, use_container_width=True)
            st.download_button(
                "📥 Export Split Sensitivity CSV",
                data=diag_report.validation_split_sensitivity.to_csv(index=False).encode("utf-8"),
                file_name="candidate_split_sensitivity.csv",
                mime="text/csv",
            )

        with tab_probability:
            st.markdown("### Probability Diagnostics")
            st.caption("These are descriptive P(up) diagnostics. They are not calibration claims.")
            st.dataframe(diag_report.probability_diagnostics, use_container_width=True)
            st.download_button(
                "📥 Export Probability Diagnostics CSV",
                data=diag_report.probability_diagnostics.to_csv(index=False).encode("utf-8"),
                file_name="candidate_probability_diagnostics.csv",
                mime="text/csv",
            )

            st.markdown("### Probability Bins")
            st.dataframe(diag_report.probability_bins, use_container_width=True)
            st.download_button(
                "📥 Export Probability Bins CSV",
                data=diag_report.probability_bins.to_csv(index=False).encode("utf-8"),
                file_name="candidate_probability_bins.csv",
                mime="text/csv",
            )


# ════════════════════════════════════════════════════════════════
# PAGE: RISK-CONTROLLED UPGRADE
# ════════════════════════════════════════════════════════════════

elif page == "🛡️ Risk-Controlled Upgrade":
    st.markdown('<p class="main-header">🛡️ Risk-Controlled Upgrade</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 7F tests risk controls as research variants. Variant selection is validation-only; locked test is used after selection. "
        "If risk control reduces drawdown but destroys return, that weakness stays visible."
    )

    rc_assets = get_asset_names()
    rc_default_asset = rc_assets.index("Silver") if "Silver" in rc_assets else 0
    rc_col_a, rc_col_b, rc_col_c, rc_col_d = st.columns(4)
    with rc_col_a:
        rc_asset = st.selectbox("Asset", rc_assets, index=rc_default_asset, key="risk_control_asset")
    with rc_col_b:
        rc_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        rc_default_horizon = rc_horizon_labels.index("5D") if "5D" in rc_horizon_labels else 0
        rc_horizon_label = st.selectbox("Horizon", rc_horizon_labels, index=rc_default_horizon, key="risk_control_horizon")
        rc_horizon = int(str(rc_horizon_label).replace("D", ""))
    with rc_col_c:
        rc_depth = st.selectbox("Model depth", ["core", "fast"], index=0, key="risk_control_depth")
    with rc_col_d:
        rc_phase5 = st.checkbox("Use Phase 5", value=True, key="risk_control_phase5")

    rc_rule_a, rc_rule_b, rc_rule_c, rc_rule_d = st.columns([1, 2, 2, 1])
    with rc_rule_a:
        rc_mode = st.selectbox("Signal mode", ["long_only", "long_short", "avoid_only"], index=0, key="risk_control_mode")
    with rc_rule_b:
        rc_thresholds = st.multiselect(
            "Threshold candidates",
            [0.50, 0.55, 0.60, 0.65, 0.70],
            default=[0.50, 0.55, 0.60, 0.65, 0.70],
            format_func=lambda x: f"{x:.2f}",
            key="risk_control_thresholds",
        )
    with rc_rule_c:
        rc_cooldowns = st.multiselect(
            "Cooldown candidates",
            [0, 2, 5],
            default=[0, 2, 5],
            format_func=lambda x: f"{int(x)} rows",
            key="risk_control_cooldowns",
        )
    with rc_rule_d:
        rc_validation_pct = st.slider("Validation %", min_value=30, max_value=70, value=50, step=5, key="risk_control_validation_pct")

    rc_variant_labels = {
        "Baseline signal": "baseline",
        "Volatility filter": "volatility_filter",
        "Drawdown stop rule": "drawdown_stop",
        "Loss-streak stop rule": "loss_streak_stop",
        "Probability band filter": "probability_band_filter",
        "Position sizing simulation": "position_sizing",
    }
    rc_selected_variant_labels = st.multiselect(
        "Risk-control variants to test",
        list(rc_variant_labels.keys()),
        default=list(rc_variant_labels.keys()),
        key="risk_control_variants",
    )
    rc_variant_keys = [rc_variant_labels[label] for label in rc_selected_variant_labels]

    cost_col_a, cost_col_b = st.columns([2, 1])
    with cost_col_a:
        rc_cost_pct_values = st.multiselect(
            "Cost stress options",
            [0.00, 0.05, 0.10, 0.20, 0.50],
            default=[0.00, 0.05, 0.10, 0.20, 0.50],
            format_func=lambda x: f"{x:.2f}%",
            key="risk_control_cost_stress",
        )
    with cost_col_b:
        rc_transaction_cost_pct = st.number_input(
            "Base transaction cost %",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.01,
            key="risk_control_transaction_cost",
        )

    run_risk_upgrade = st.button("🚀 Run Risk-Controlled Upgrade", type="primary")

    if run_risk_upgrade:
        if not rc_thresholds:
            st.error("Select at least one threshold candidate.")
            st.stop()
        if not rc_cooldowns:
            st.error("Select at least one cooldown candidate.")
            st.stop()
        if not rc_variant_keys:
            st.error("Select at least one risk-control variant.")
            st.stop()
        if not rc_cost_pct_values:
            st.error("Select at least one cost stress option.")
            st.stop()

        with st.spinner("Running validation-only risk-control upgrade..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                risk_report = run_risk_controlled_candidate_upgrade(
                    raw_df=raw_df,
                    asset_name=rc_asset,
                    horizon=rc_horizon,
                    model_depth=rc_depth,
                    use_phase5_features=rc_phase5,
                    signal_mode=rc_mode,
                    threshold_candidates=rc_thresholds,
                    cooldown_candidates=rc_cooldowns,
                    validation_fraction=float(rc_validation_pct) / 100.0,
                    transaction_cost=float(rc_transaction_cost_pct) / 100.0,
                    risk_variant_names=rc_variant_keys,
                    cost_values=[float(v) / 100.0 for v in rc_cost_pct_values],
                )
                st.session_state.risk_control_upgrade_report = risk_report
                st.session_state.risk_control_upgrade_settings = risk_report.settings
            except Exception as exc:
                st.error(f"Risk-controlled upgrade failed: {exc}")
                st.stop()

    risk_report = st.session_state.risk_control_upgrade_report
    if risk_report is None:
        st.info("Run the upgrade for any configured asset and horizon. Silver 5D is only the default candidate view.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('risk_control_upgrade_settings') or {}}")
        if risk_report.warnings:
            for warning in risk_report.warnings:
                if any(flag in warning for flag in ["CostFragile", "DrawdownRisk", "ReturnDestroyed", "NoImprovement"]):
                    st.warning(warning)
                else:
                    st.info(warning)

        selected_info = risk_report.selected_variant or {}
        st.markdown("### Selected Variant")
        st.json(selected_info)

        st.markdown("### Baseline vs Best Risk-Controlled Variant")
        st.dataframe(risk_report.baseline_vs_best, use_container_width=True)
        st.download_button(
            "📥 Export Baseline vs Best CSV",
            data=risk_report.baseline_vs_best.to_csv(index=False).encode("utf-8"),
            file_name="risk_control_baseline_vs_best.csv",
            mime="text/csv",
        )

        tab_variants, tab_costs = st.tabs(["Full Variant Table", "Cost Stress"])

        with tab_variants:
            st.markdown("### Full Variant Table")
            st.caption("Non-selected variants show validation metrics only. Locked test is evaluated for the baseline reference and validation-selected variant.")
            st.dataframe(risk_report.full_variant_table, use_container_width=True)
            st.download_button(
                "📥 Export Full Variant CSV",
                data=risk_report.full_variant_table.to_csv(index=False).encode("utf-8"),
                file_name="risk_control_full_variants.csv",
                mime="text/csv",
            )

        with tab_costs:
            st.markdown("### Cost / Slippage Stress")
            st.caption("Post-selection stress table. It is not used to choose the risk-control variant.")
            st.dataframe(risk_report.cost_stress_table, use_container_width=True)
            st.download_button(
                "📥 Export Cost Stress CSV",
                data=risk_report.cost_stress_table.to_csv(index=False).encode("utf-8"),
                file_name="risk_control_cost_stress.csv",
                mime="text/csv",
            )


# ════════════════════════════════════════════════════════════════
# PAGE: WALK-FORWARD VALIDATION
# ════════════════════════════════════════════════════════════════

elif page == "🧭 Walk-Forward Validation":
    st.markdown('<p class="main-header">🧭 Walk-Forward Validation</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 7G asks whether candidates survive repeated rolling validation/locked-test windows. "
        "Each window selects threshold and cooldown from validation rows only; locked-test rows are evaluation-only."
    )

    wf_assets_all = get_asset_names()
    wf_default_assets = [asset for asset in ["Silver", "Crude Oil"] if asset in wf_assets_all] or wf_assets_all
    wf_col_a, wf_col_b, wf_col_c = st.columns([2, 2, 1])
    with wf_col_a:
        wf_assets = st.multiselect(
            "Assets",
            wf_assets_all,
            default=wf_default_assets,
            key="walk_forward_assets",
        )
    with wf_col_b:
        wf_horizon_options = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        wf_default_horizons = [label for label in ["5D", "1D"] if label in wf_horizon_options] or wf_horizon_options[:1]
        wf_horizon_labels = st.multiselect(
            "Horizons",
            wf_horizon_options,
            default=wf_default_horizons,
            key="walk_forward_horizons",
        )
        wf_horizons = [int(str(label).replace("D", "")) for label in wf_horizon_labels]
    with wf_col_c:
        wf_depth = st.selectbox("Model depth", ["core", "fast"], index=0, key="walk_forward_depth")
        wf_phase5 = st.checkbox("Use Phase 5", value=True, key="walk_forward_phase5")

    wf_rule_a, wf_rule_b, wf_rule_c, wf_rule_d = st.columns([1, 2, 2, 1])
    with wf_rule_a:
        wf_mode = st.selectbox("Signal mode", ["long_only", "long_short", "avoid_only"], index=0, key="walk_forward_mode")
    with wf_rule_b:
        wf_thresholds = st.multiselect(
            "Threshold candidates",
            [0.50, 0.55, 0.60, 0.65, 0.70],
            default=[0.50, 0.55, 0.60, 0.65, 0.70],
            format_func=lambda x: f"{x:.2f}",
            key="walk_forward_thresholds",
        )
    with wf_rule_c:
        wf_cooldowns = st.multiselect(
            "Cooldown candidates",
            [0, 2, 5],
            default=[0, 2, 5],
            format_func=lambda x: f"{int(x)} rows",
            key="walk_forward_cooldowns",
        )
    with wf_rule_d:
        wf_cost_pct = st.number_input(
            "Transaction cost %",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.01,
            key="walk_forward_transaction_cost",
        )

    wf_win_a, wf_win_b, wf_win_c, wf_win_d, wf_win_e = st.columns(5)
    with wf_win_a:
        wf_validation_rows = st.number_input("Validation rows", min_value=30, max_value=500, value=180, step=10, key="walk_forward_validation_rows")
    with wf_win_b:
        wf_test_rows = st.number_input("Test rows", min_value=20, max_value=250, value=90, step=10, key="walk_forward_test_rows")
    with wf_win_c:
        wf_step_rows = st.number_input("Step rows", min_value=10, max_value=250, value=60, step=10, key="walk_forward_step_rows")
    with wf_win_d:
        wf_min_trades = st.number_input("Min trades/window", min_value=1, max_value=50, value=3, step=1, key="walk_forward_min_trades")
    with wf_win_e:
        wf_window_mode = st.selectbox("Window mode", ["rolling", "expanding"], index=0, key="walk_forward_window_mode")

    run_wf = st.button("🚀 Run Walk-Forward Validation", type="primary")

    if run_wf:
        if not wf_assets:
            st.error("Select at least one asset.")
            st.stop()
        if not wf_horizons:
            st.error("Select at least one horizon.")
            st.stop()
        if not wf_thresholds:
            st.error("Select at least one threshold candidate.")
            st.stop()
        if not wf_cooldowns:
            st.error("Select at least one cooldown candidate.")
            st.stop()

        progress = st.progress(0, text="Preparing walk-forward validation...")

        def _wf_progress(done, total, msg):
            pct = int(done / max(total, 1) * 100)
            progress.progress(pct, text=msg)

        with st.spinner("Running walk-forward validation..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                wf_report = run_walk_forward_risk_validation(
                    raw_df=raw_df,
                    asset_names=wf_assets,
                    horizons=wf_horizons,
                    model_depth=wf_depth,
                    use_phase5_features=wf_phase5,
                    signal_mode=wf_mode,
                    threshold_candidates=wf_thresholds,
                    cooldown_candidates=wf_cooldowns,
                    transaction_cost=float(wf_cost_pct) / 100.0,
                    validation_window=int(wf_validation_rows),
                    test_window=int(wf_test_rows),
                    step_size=int(wf_step_rows),
                    min_trades_per_window=int(wf_min_trades),
                    window_mode=wf_window_mode,
                    progress_callback=_wf_progress,
                )
                st.session_state.walk_forward_validation_report = wf_report
                st.session_state.walk_forward_validation_settings = wf_report.settings
                progress.progress(100, text="Walk-forward validation complete.")
            except Exception as exc:
                st.error(f"Walk-forward validation failed: {exc}")
                st.stop()

    wf_report = st.session_state.walk_forward_validation_report
    if wf_report is None:
        st.info("Run walk-forward validation to test whether candidates repeat across rolling historical windows.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('walk_forward_validation_settings') or {}}")

        counts = wf_report.verdict_counts or {}
        count_cols = st.columns(max(1, min(4, len(counts) or 1)))
        if counts:
            for idx, (label, value) in enumerate(counts.items()):
                count_cols[idx % len(count_cols)].metric(label, value)
        else:
            count_cols[0].metric("Results", 0)

        if wf_report.warnings:
            for warning in wf_report.warnings[:8]:
                st.warning(warning)
            if len(wf_report.warnings) > 8:
                st.caption(f"{len(wf_report.warnings) - 8} additional warnings are included in the tables.")

        st.markdown("### Aggregate Walk-Forward Summary")
        st.dataframe(wf_report.aggregate_summary, use_container_width=True)
        st.download_button(
            "📥 Export Aggregate CSV",
            data=wf_report.aggregate_summary.to_csv(index=False).encode("utf-8"),
            file_name="walk_forward_aggregate_summary.csv",
            mime="text/csv",
        )

        tab_windows, tab_errors = st.tabs(["Per-Window Results", "Errors"])
        with tab_windows:
            st.markdown("### Per-Window Results")
            st.dataframe(wf_report.window_results, use_container_width=True)
            st.download_button(
                "📥 Export Per-Window CSV",
                data=wf_report.window_results.to_csv(index=False).encode("utf-8"),
                file_name="walk_forward_window_results.csv",
                mime="text/csv",
            )

        with tab_errors:
            st.markdown("### Walk-Forward Errors")
            if wf_report.errors is not None and not wf_report.errors.empty:
                st.dataframe(wf_report.errors, use_container_width=True)
            else:
                st.success("No walk-forward errors in the last run.")


# ════════════════════════════════════════════════════════════════
# PAGE: REGIME-AWARE META SIGNAL
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

elif page == "🧠 Regime-Aware Meta Signal":
    st.markdown('<p class="main-header">🧠 Regime-Aware Meta Signal</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 8 is a rule-based meta layer. It consumes Phase 7G walk-forward aggregate reliability, "
        "adds current/historical regime context, and never labels results production-ready."
    )

    meta_assets_all = get_asset_names()
    meta_col_a, meta_col_b, meta_col_c = st.columns([2, 2, 1])
    with meta_col_a:
        meta_assets = st.multiselect(
            "Assets",
            meta_assets_all,
            default=meta_assets_all,
            key="meta_signal_assets",
        )
    with meta_col_b:
        meta_horizon_options = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        meta_horizon_labels = st.multiselect(
            "Horizons",
            meta_horizon_options,
            default=meta_horizon_options,
            key="meta_signal_horizons",
        )
        meta_horizons = [int(str(label).replace("D", "")) for label in meta_horizon_labels]
    with meta_col_c:
        meta_depth = st.selectbox("Model depth", ["core", "fast"], index=0, key="meta_signal_depth")
        meta_phase5 = st.checkbox("Use Phase 5", value=True, key="meta_signal_phase5")

    meta_rule_a, meta_rule_b = st.columns([1, 2])
    with meta_rule_a:
        meta_mode = st.selectbox("Signal mode", ["long_only", "long_short", "avoid_only"], index=0, key="meta_signal_mode")
    with meta_rule_b:
        st.caption(
            "Regime features use only current and historical prices: rolling returns, volatility, drawdown, "
            "moving-average trend, and recent buy-and-hold strength."
        )

    latest_wf_report = st.session_state.get("walk_forward_validation_report")
    latest_wf_summary = None
    if latest_wf_report is not None and getattr(latest_wf_report, "aggregate_summary", None) is not None:
        latest_wf_summary = latest_wf_report.aggregate_summary

    uploaded_wf_csv = st.file_uploader(
        "Upload Phase 7G aggregate CSV",
        type=["csv"],
        key="meta_signal_wf_upload",
        help="Use this when no walk-forward aggregate is available in the current session.",
    )

    source_options = []
    if latest_wf_summary is not None and not latest_wf_summary.empty:
        source_options.append("Latest Phase 7G session result")
    if uploaded_wf_csv is not None:
        source_options.append("Uploaded Phase 7G aggregate CSV")

    if source_options:
        meta_source = st.radio("Phase 7G reliability source", source_options, horizontal=True, key="meta_signal_source")
    else:
        meta_source = None
        st.info("Run Phase 7G Walk-Forward Validation first, or upload a Phase 7G aggregate CSV.")

    run_meta = st.button("🚀 Run Meta Signal Analysis", type="primary")

    if run_meta:
        if not meta_assets:
            st.error("Select at least one asset.")
            st.stop()
        if not meta_horizons:
            st.error("Select at least one horizon.")
            st.stop()
        if meta_source is None:
            st.error("Provide a Phase 7G walk-forward aggregate source.")
            st.stop()

        try:
            if meta_source == "Latest Phase 7G session result":
                wf_summary_for_meta = latest_wf_summary.copy()
            else:
                uploaded_wf_csv.seek(0)
                wf_summary_for_meta = pd.read_csv(uploaded_wf_csv)
        except Exception as exc:
            st.error(f"Could not read Phase 7G aggregate data: {exc}")
            st.stop()

        if wf_summary_for_meta is None or wf_summary_for_meta.empty:
            st.error("Phase 7G aggregate data is empty.")
            st.stop()

        with st.spinner("Running regime-aware meta signal analysis..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                meta_report = run_regime_aware_meta_signal(
                    raw_df=raw_df,
                    walk_forward_summary=wf_summary_for_meta,
                    asset_names=meta_assets,
                    horizons=meta_horizons,
                    model_depth=meta_depth,
                    use_phase5_features=meta_phase5,
                    signal_mode=meta_mode,
                )
                st.session_state.meta_signal_report = meta_report
                st.session_state.meta_signal_settings = meta_report.settings
            except Exception as exc:
                st.error(f"Meta signal analysis failed: {exc}")
                st.stop()

    meta_report = st.session_state.meta_signal_report
    if meta_report is None:
        st.info("Run meta signal analysis to convert Phase 7G reliability into regime-aware decisions.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('meta_signal_settings') or {}}")

        summary = meta_report.decision_summary
        metric_cols = st.columns(5)
        for idx, decision in enumerate(["Trade", "No Trade", "Defensive Only", "Research Only", "Avoid"]):
            match = summary[summary["MetaDecision"].eq(decision)]
            value = int(match.iloc[0]["Count"]) if not match.empty else 0
            metric_cols[idx].metric(decision, value)

        if meta_report.warnings:
            for warning in meta_report.warnings[:10]:
                st.warning(warning)
            if len(meta_report.warnings) > 10:
                st.caption(f"{len(meta_report.warnings) - 10} additional warnings are included in the decision table.")

        st.markdown("### Meta Decision Table")
        st.dataframe(meta_report.decision_table, use_container_width=True)
        st.download_button(
            "📥 Export Meta Decisions CSV",
            data=meta_report.decision_table.to_csv(index=False).encode("utf-8"),
            file_name="regime_aware_meta_signal_decisions.csv",
            mime="text/csv",
            key="meta_signal_decisions_download",
        )

        st.markdown("### Grouped Decisions")
        tabs = st.tabs(["Trade", "No Trade", "Defensive Only", "Research Only", "Avoid"])
        for tab, decision in zip(tabs, ["Trade", "No Trade", "Defensive Only", "Research Only", "Avoid"]):
            with tab:
                subset = meta_report.decision_table[meta_report.decision_table["MetaDecision"].eq(decision)]
                if subset.empty:
                    st.info(f"No {decision} rows in the latest meta analysis.")
                else:
                    st.dataframe(subset, use_container_width=True)

        diag_a, diag_b, diag_c = st.tabs(["Regimes", "Reliability / Risk", "Summary"])
        with diag_a:
            st.markdown("### Current Regime Features")
            st.dataframe(meta_report.regime_features, use_container_width=True)
            st.download_button(
                "📥 Export Regime Features CSV",
                data=meta_report.regime_features.to_csv(index=False).encode("utf-8"),
                file_name="regime_features.csv",
                mime="text/csv",
                key="meta_signal_regime_download",
            )
        with diag_b:
            cols = [
                "Asset",
                "Horizon",
                "MetaDecision",
                "MetaConfidenceScore",
                "MetaRiskScore",
                "SignalReliabilityScore",
                "WalkForwardReliabilityScore",
                "RegimeLabel",
                "BenchmarkRiskFlag",
                "CostFragilityFlag",
                "DrawdownRiskFlag",
                "StabilityFlag",
                "Warnings",
            ]
            available_cols = [col for col in cols if col in meta_report.decision_table.columns]
            st.dataframe(meta_report.decision_table[available_cols], use_container_width=True)
        with diag_c:
            st.markdown("### Decision Summary")
            st.dataframe(summary, use_container_width=True)
            st.download_button(
                "📥 Export Decision Summary CSV",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="regime_aware_meta_signal_summary.csv",
                mime="text/csv",
                key="meta_signal_summary_download",
            )


# PAGE: META DECISION AUDIT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

elif page == "🧾 Meta Decision Audit":
    st.markdown('<p class="main-header">🧾 Meta Decision Audit</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 8A is audit and calibration only. It explains existing Phase 8 meta decisions, "
        "does not rerun models, does not retune thresholds, and never marks anything production-ready."
    )

    latest_meta_report = st.session_state.get("meta_signal_report")
    latest_meta_table = None
    if latest_meta_report is not None and getattr(latest_meta_report, "decision_table", None) is not None:
        latest_meta_table = latest_meta_report.decision_table

    uploaded_meta_csv = st.file_uploader(
        "Upload Phase 8 meta decision CSV",
        type=["csv"],
        key="meta_audit_upload",
        help="Use this when no Phase 8 meta decision table is available in the current session.",
    )

    audit_sources = []
    if latest_meta_table is not None and not latest_meta_table.empty:
        audit_sources.append("Latest Phase 8 session result")
    if uploaded_meta_csv is not None:
        audit_sources.append("Uploaded Phase 8 meta decision CSV")

    audit_col_a, audit_col_b = st.columns([1, 2])
    with audit_col_a:
        audit_mode = st.selectbox(
            "Calibration mode",
            ["Conservative", "Balanced", "Aggressive Research"],
            index=0,
            key="meta_audit_mode",
        )
    with audit_col_b:
        if audit_mode == "Conservative":
            st.caption("Strict audit mode: mirrors Phase 8's conservative posture.")
        elif audit_mode == "Balanced":
            st.caption("Research calibration mode: may surface more Research Only / Defensive Only rows, but Trade remains strict.")
        else:
            st.caption("Experimental research mode: may surface candidates for investigation, never production readiness.")

    if audit_sources:
        audit_source = st.radio("Phase 8 decision source", audit_sources, horizontal=True, key="meta_audit_source")
    else:
        audit_source = None
        st.info("Run Phase 8 Regime-Aware Meta Signal first, or upload a Phase 8 meta decision CSV.")

    run_audit = st.button("🚀 Run Meta Decision Audit", type="primary")

    if run_audit:
        if audit_source is None:
            st.error("Provide a Phase 8 meta decision source.")
            st.stop()

        try:
            if audit_source == "Latest Phase 8 session result":
                meta_table_for_audit = latest_meta_table.copy()
            else:
                uploaded_meta_csv.seek(0)
                meta_table_for_audit = pd.read_csv(uploaded_meta_csv)
        except Exception as exc:
            st.error(f"Could not read Phase 8 meta decision data: {exc}")
            st.stop()

        if meta_table_for_audit is None or meta_table_for_audit.empty:
            st.error("Phase 8 meta decision data is empty.")
            st.stop()

        with st.spinner("Running meta decision audit..."):
            try:
                audit_report = run_meta_decision_audit(meta_table_for_audit, calibration_mode=audit_mode)
                st.session_state.meta_decision_audit_report = audit_report
                st.session_state.meta_decision_audit_settings = audit_report.settings
            except Exception as exc:
                st.error(f"Meta decision audit failed: {exc}")
                st.stop()

    audit_report = st.session_state.meta_decision_audit_report
    if audit_report is None:
        st.info("Run the audit to see blocking rules, passing rules, near misses, and calibration tables.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('meta_decision_audit_settings') or {}}")

        if audit_report.warnings:
            for warning in audit_report.warnings:
                st.warning(warning)

        st.markdown("### Meta Decision Audit Table")
        st.dataframe(audit_report.audit_table, use_container_width=True)
        st.download_button(
            "📥 Export Audit CSV",
            data=audit_report.audit_table.to_csv(index=False).encode("utf-8"),
            file_name="meta_decision_audit.csv",
            mime="text/csv",
            key="meta_audit_table_download",
        )

        tab_blocking, tab_passing, tab_near, tab_rankings, tab_modes, tab_thresholds = st.tabs(
            ["Blocking Rules", "Passing Rules", "Near Misses", "Candidate Rankings", "Mode Comparison", "Thresholds"]
        )

        with tab_blocking:
            st.markdown("### Most Common Blocking Rules")
            st.dataframe(audit_report.common_blocking_rules, use_container_width=True)
            st.download_button(
                "📥 Export Blocking Rules CSV",
                data=audit_report.common_blocking_rules.to_csv(index=False).encode("utf-8"),
                file_name="meta_audit_blocking_rules.csv",
                mime="text/csv",
                key="meta_audit_blocking_download",
            )

            cols = ["Asset", "Horizon", "Current MetaDecision", "Calibrated MetaDecision", "MainBlockingRule", "BlockingRules"]
            available = [col for col in cols if col in audit_report.audit_table.columns]
            st.dataframe(audit_report.audit_table[available], use_container_width=True)

        with tab_passing:
            st.markdown("### Passing Rules")
            cols = ["Asset", "Horizon", "Current MetaDecision", "Calibrated MetaDecision", "PassingRules"]
            available = [col for col in cols if col in audit_report.audit_table.columns]
            passing_table = audit_report.audit_table[available]
            st.dataframe(passing_table, use_container_width=True)
            st.download_button(
                "📥 Export Passing Rules CSV",
                data=passing_table.to_csv(index=False).encode("utf-8"),
                file_name="meta_audit_passing_rules.csv",
                mime="text/csv",
                key="meta_audit_passing_download",
            )

        with tab_near:
            st.markdown("### Top Near-Miss Candidates")
            st.dataframe(audit_report.near_miss_candidates, use_container_width=True)
            st.download_button(
                "📥 Export Near Misses CSV",
                data=audit_report.near_miss_candidates.to_csv(index=False).encode("utf-8"),
                file_name="meta_audit_near_misses.csv",
                mime="text/csv",
                key="meta_audit_near_download",
            )

        with tab_rankings:
            rank_a, rank_b, rank_c = st.tabs(["Top Blocked", "Highest Confidence", "Highest Risk"])
            with rank_a:
                st.dataframe(audit_report.top_blocked_candidates, use_container_width=True)
                st.download_button(
                    "📥 Export Top Blocked CSV",
                    data=audit_report.top_blocked_candidates.to_csv(index=False).encode("utf-8"),
                    file_name="meta_audit_top_blocked.csv",
                    mime="text/csv",
                    key="meta_audit_top_blocked_download",
                )
            with rank_b:
                st.dataframe(audit_report.highest_confidence_candidates, use_container_width=True)
                st.download_button(
                    "📥 Export Highest Confidence CSV",
                    data=audit_report.highest_confidence_candidates.to_csv(index=False).encode("utf-8"),
                    file_name="meta_audit_highest_confidence.csv",
                    mime="text/csv",
                    key="meta_audit_high_conf_download",
                )
            with rank_c:
                st.dataframe(audit_report.highest_risk_candidates, use_container_width=True)
                st.download_button(
                    "📥 Export Highest Risk CSV",
                    data=audit_report.highest_risk_candidates.to_csv(index=False).encode("utf-8"),
                    file_name="meta_audit_highest_risk.csv",
                    mime="text/csv",
                    key="meta_audit_high_risk_download",
                )

        with tab_modes:
            st.markdown("### Decision Counts by Mode")
            st.dataframe(audit_report.mode_comparison, use_container_width=True)
            st.download_button(
                "📥 Export Mode Comparison CSV",
                data=audit_report.mode_comparison.to_csv(index=False).encode("utf-8"),
                file_name="meta_audit_mode_comparison.csv",
                mime="text/csv",
                key="meta_audit_modes_download",
            )

        with tab_thresholds:
            st.markdown("### Threshold Configuration")
            st.dataframe(audit_report.threshold_config, use_container_width=True)
            st.download_button(
                "📥 Export Threshold Config CSV",
                data=audit_report.threshold_config.to_csv(index=False).encode("utf-8"),
                file_name="meta_audit_threshold_config.csv",
                mime="text/csv",
                key="meta_audit_thresholds_download",
            )


# PAGE: META RELIABILITY GRADING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

elif page == "🏷️ Meta Reliability Grading":
    st.markdown('<p class="main-header">🏷️ Meta Reliability Grading</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 8B grades research reliability only. MetaDecision remains the action gate; "
        "A-grade candidates are still not production-ready."
    )

    latest_audit_report = st.session_state.get("meta_decision_audit_report")
    latest_audit_table = None
    if latest_audit_report is not None and getattr(latest_audit_report, "audit_table", None) is not None:
        latest_audit_table = latest_audit_report.audit_table

    latest_meta_report = st.session_state.get("meta_signal_report")
    latest_meta_table = None
    if latest_meta_report is not None and getattr(latest_meta_report, "decision_table", None) is not None:
        latest_meta_table = latest_meta_report.decision_table

    uploaded_grading_csv = st.file_uploader(
        "Upload Phase 8A audit CSV or Phase 8 meta decision CSV",
        type=["csv"],
        key="meta_grading_upload",
    )

    grading_sources = []
    if latest_audit_table is not None and not latest_audit_table.empty:
        grading_sources.append("Latest Phase 8A audit result")
    if latest_meta_table is not None and not latest_meta_table.empty:
        grading_sources.append("Latest Phase 8 meta result")
    if uploaded_grading_csv is not None:
        grading_sources.append("Uploaded CSV")

    grading_col_a, grading_col_b = st.columns([1, 2])
    with grading_col_a:
        grading_mode = st.selectbox(
            "Grading mode",
            ["Conservative", "Balanced", "Aggressive Research"],
            index=0,
            key="meta_grading_mode",
        )
    with grading_col_b:
        if grading_mode == "Conservative":
            st.caption("Strict grading: prioritizes evidence quality and downgrades borderline candidates.")
        elif grading_mode == "Balanced":
            st.caption("Balanced grading: surfaces research quality without loosening action gates.")
        else:
            st.caption("Aggressive Research: raises research attention only; it never implies production readiness.")

    if grading_sources:
        grading_source = st.radio("Input source", grading_sources, horizontal=True, key="meta_grading_source")
    else:
        grading_source = None
        st.info("Run Phase 8A audit, run Phase 8 meta signal, or upload a CSV.")

    run_grading = st.button("🚀 Run Reliability Grading", type="primary")

    if run_grading:
        if grading_source is None:
            st.error("Provide a Phase 8A audit table or Phase 8 meta decision table.")
            st.stop()
        try:
            if grading_source == "Latest Phase 8A audit result":
                grading_input = latest_audit_table.copy()
            elif grading_source == "Latest Phase 8 meta result":
                grading_input = latest_meta_table.copy()
            else:
                uploaded_grading_csv.seek(0)
                grading_input = pd.read_csv(uploaded_grading_csv)
        except Exception as exc:
            st.error(f"Could not read grading input: {exc}")
            st.stop()

        if grading_input is None or grading_input.empty:
            st.error("Reliability grading input is empty.")
            st.stop()

        with st.spinner("Running meta reliability grading..."):
            try:
                grading_report = run_meta_score_calibration(grading_input, grading_mode=grading_mode)
                st.session_state.meta_reliability_grading_report = grading_report
                st.session_state.meta_reliability_grading_settings = grading_report.settings
            except Exception as exc:
                st.error(f"Meta reliability grading failed: {exc}")
                st.stop()

    grading_report = st.session_state.meta_reliability_grading_report
    if grading_report is None:
        st.info("Run reliability grading to rank research quality without changing MetaDecision gates.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('meta_reliability_grading_settings') or {}}")

        if grading_report.warnings:
            for warning in grading_report.warnings:
                st.warning(warning)

        st.markdown("### Reliability Grading Table")
        st.dataframe(grading_report.grading_table, use_container_width=True)
        st.download_button(
            "📥 Export Reliability Grading CSV",
            data=grading_report.grading_table.to_csv(index=False).encode("utf-8"),
            file_name="meta_reliability_grading.csv",
            mime="text/csv",
            key="meta_grading_table_download",
        )

        tab_counts, tab_research, tab_defensive, tab_archive, tab_actions, tab_components = st.tabs(
            ["Grade Counts", "Top A/B/C", "Defensive Watch", "Avoid / Archive", "Next Actions", "Score Components"]
        )

        with tab_counts:
            st.markdown("### Grade Counts")
            st.dataframe(grading_report.grade_counts, use_container_width=True)
            st.download_button(
                "📥 Export Grade Counts CSV",
                data=grading_report.grade_counts.to_csv(index=False).encode("utf-8"),
                file_name="meta_reliability_grade_counts.csv",
                mime="text/csv",
                key="meta_grading_counts_download",
            )

        with tab_research:
            st.markdown("### Top A/B/C Research Candidates")
            st.dataframe(grading_report.top_research_candidates, use_container_width=True)
            st.download_button(
                "📥 Export Top Research CSV",
                data=grading_report.top_research_candidates.to_csv(index=False).encode("utf-8"),
                file_name="meta_reliability_top_research.csv",
                mime="text/csv",
                key="meta_grading_research_download",
            )

        with tab_defensive:
            st.markdown("### Defensive Watch List")
            st.dataframe(grading_report.defensive_watchlist, use_container_width=True)
            st.download_button(
                "📥 Export Defensive Watch CSV",
                data=grading_report.defensive_watchlist.to_csv(index=False).encode("utf-8"),
                file_name="meta_reliability_defensive_watch.csv",
                mime="text/csv",
                key="meta_grading_defensive_download",
            )

        with tab_archive:
            st.markdown("### Avoid / Archive List")
            st.dataframe(grading_report.avoid_archive_list, use_container_width=True)
            st.download_button(
                "📥 Export Avoid Archive CSV",
                data=grading_report.avoid_archive_list.to_csv(index=False).encode("utf-8"),
                file_name="meta_reliability_avoid_archive.csv",
                mime="text/csv",
                key="meta_grading_archive_download",
            )

        with tab_actions:
            st.markdown("### Top Next Actions")
            st.dataframe(grading_report.next_action_summary, use_container_width=True)
            st.download_button(
                "📥 Export Next Actions CSV",
                data=grading_report.next_action_summary.to_csv(index=False).encode("utf-8"),
                file_name="meta_reliability_next_actions.csv",
                mime="text/csv",
                key="meta_grading_actions_download",
            )

        with tab_components:
            st.markdown("### Score Component Breakdown")
            st.dataframe(grading_report.score_components, use_container_width=True)
            st.download_button(
                "📥 Export Score Components CSV",
                data=grading_report.score_components.to_csv(index=False).encode("utf-8"),
                file_name="meta_reliability_score_components.csv",
                mime="text/csv",
                key="meta_grading_components_download",
            )


# PAGE: SIGNAL ENGINE
# ════════════════════════════════════════════════════════════════

elif page == "📡 Signal Engine":
    st.markdown('<p class="main-header">📡 Direction-First Signal Engine</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 7 uses Phase 6 direct direction probabilities, not exact price forecasts. "
        "Threshold sweeps are research/reporting only and must not be treated as production threshold tuning on test data."
    )

    sig_col_a, sig_col_b, sig_col_c, sig_col_d = st.columns(4)
    with sig_col_a:
        signal_asset = st.selectbox(
            "Asset",
            get_asset_names(),
            index=get_asset_names().index("Crude Oil") if "Crude Oil" in get_asset_names() else 0,
            key="signal_engine_asset",
            help="Crude Oil 5D is a smoke path only. The engine supports all configured assets.",
        )
    with sig_col_b:
        signal_horizon_label = st.selectbox(
            "Horizon",
            [f"{h}D" for h in DIRECT_FORECAST_HORIZONS],
            index=1,
            key="signal_engine_horizon",
        )
        signal_horizon = int(str(signal_horizon_label).replace("D", ""))
    with sig_col_c:
        signal_depth = st.selectbox(
            "Model depth",
            ["core", "fast"],
            index=0,
            key="signal_engine_depth",
        )
    with sig_col_d:
        signal_use_phase5 = st.checkbox(
            "Use Phase 5 features",
            value=True,
            key="signal_engine_phase5",
        )

    rule_col_a, rule_col_b, rule_col_c, rule_col_d, rule_col_e = st.columns(5)
    with rule_col_a:
        signal_backtest_style = st.selectbox(
            "Backtest style",
            ["non_overlapping_realistic", "overlapping_research"],
            index=0,
            help="Non-overlapping is the realistic trade mode. Overlapping is an optimistic research comparison.",
            key="signal_engine_backtest_style",
        )
    with rule_col_b:
        signal_mode = st.selectbox(
            "Signal mode",
            ["long_only", "long_short", "avoid_only"],
            index=0,
            key="signal_engine_mode",
        )
    with rule_col_c:
        long_threshold = st.slider(
            "Long threshold",
            min_value=0.50,
            max_value=0.80,
            value=0.55,
            step=0.01,
            key="signal_engine_long_threshold",
        )
    with rule_col_d:
        short_threshold = st.slider(
            "Short threshold",
            min_value=0.20,
            max_value=0.50,
            value=0.45,
            step=0.01,
            disabled=signal_mode != "long_short",
            key="signal_engine_short_threshold",
        )
    with rule_col_e:
        transaction_cost_pct = st.number_input(
            "Transaction cost %",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.01,
            key="signal_engine_transaction_cost",
        )

    cooldown_rows = st.number_input(
        "Cooldown rows after trade exit",
        min_value=0,
        max_value=30,
        value=0,
        step=1,
        disabled=signal_backtest_style != "non_overlapping_realistic",
        help="Only used by non-overlapping realistic mode.",
        key="signal_engine_cooldown_rows",
    )

    policy_col_a, policy_col_b = st.columns(2)
    with policy_col_a:
        threshold_policy = st.selectbox(
            "Threshold policy",
            ["validation_locked", "manual_threshold"],
            index=0,
            help="Validation-locked sweeps thresholds only on the first chronological segment and evaluates the selected threshold once on the locked test segment.",
            key="signal_engine_threshold_policy",
        )
    with policy_col_b:
        validation_fraction_pct = st.slider(
            "Validation segment %",
            min_value=30,
            max_value=70,
            value=50,
            step=5,
            disabled=threshold_policy != "validation_locked",
            help="Chronological share of the available Phase 6 out-of-sample signal rows used for threshold selection.",
            key="signal_engine_validation_fraction",
        )

    if threshold_policy == "validation_locked":
        st.info(
            "Validation-locked mode splits the available Phase 6 out-of-sample signal rows into validation and locked-test segments. "
            "Thresholds are selected on validation only, then evaluated once on locked test."
        )

    if signal_backtest_style == "overlapping_research":
        st.warning(
            "Overlapping research mode can be optimistic for multi-day horizons because daily 5D/10D/20D signals overlap. "
            "Use non-overlapping realistic mode before treating a signal as tradable."
        )

    run_signal = st.button("🚀 Run Signal Backtest", type="primary")

    if run_signal:
        if threshold_policy == "manual_threshold" and signal_mode == "long_short" and short_threshold >= long_threshold:
            st.error("Short threshold must be lower than long threshold.")
            st.stop()

        with st.spinner("Training Phase 6 direct model and running signal backtest..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                signal_output = run_direct_forecast_signal_output(
                    raw_df=raw_df,
                    asset_name=signal_asset,
                    horizon=signal_horizon,
                    model_depth=signal_depth,
                    use_phase5_features=signal_use_phase5,
                )
                if threshold_policy == "validation_locked":
                    signal_result = run_validation_locked_signal_engine(
                        signal_output=signal_output,
                        mode=signal_mode,
                        transaction_cost=float(transaction_cost_pct) / 100.0,
                        backtest_style=signal_backtest_style,
                        cooldown=int(cooldown_rows),
                        validation_fraction=float(validation_fraction_pct) / 100.0,
                    )
                else:
                    signal_result = run_signal_engine(
                        signal_output=signal_output,
                        long_threshold=long_threshold,
                        short_threshold=short_threshold if signal_mode == "long_short" else min(0.45, long_threshold - 0.05),
                        mode=signal_mode,
                        transaction_cost=float(transaction_cost_pct) / 100.0,
                        backtest_style=signal_backtest_style,
                        cooldown=int(cooldown_rows),
                    )
                st.session_state.signal_engine_output = signal_output
                st.session_state.signal_engine_result = signal_result
                st.session_state.signal_engine_settings = {
                    "asset": signal_asset,
                    "horizon": signal_horizon,
                    "model_depth": signal_depth,
                    "use_phase5_features": signal_use_phase5,
                    "threshold_policy": threshold_policy,
                    "validation_fraction_pct": validation_fraction_pct,
                    "backtest_style": signal_backtest_style,
                    "mode": signal_mode,
                    "long_threshold": long_threshold,
                    "short_threshold": short_threshold,
                    "transaction_cost_pct": transaction_cost_pct,
                    "cooldown_rows": int(cooldown_rows),
                }
            except Exception as exc:
                st.error(f"Signal engine failed: {exc}")
                st.stop()

    signal_output = st.session_state.signal_engine_output
    signal_result = st.session_state.signal_engine_result

    if signal_output is None or signal_result is None:
        st.info("Run the signal backtest to evaluate thresholded P(up) signals.")
    else:
        settings = st.session_state.get("signal_engine_settings") or {}
        metrics = signal_result.metrics
        st.caption(f"Last run settings: {settings}")
        st.caption(f"Phase 6 direction model: {signal_output.model_name}")

        if signal_output.feature_leakage_columns:
            st.error(f"Future target columns leaked into features: {signal_output.feature_leakage_columns}")
        else:
            st.success("No future_return_*, future_direction_*, or future_realized_vol_* columns are present in the model features.")

        result_style = str(metrics.get("BacktestStyle", ""))
        if result_style == "overlapping_research":
            st.warning("This result uses overlapping research mode. It is useful for diagnostics but can overstate multi-day signal performance.")

        is_validation_locked = str(metrics.get("ThresholdPolicy", "")) == "validation_locked"
        if is_validation_locked:
            st.info(
                "Validation-locked result: threshold selection used only the validation segment inside the available Phase 6 out-of-sample output. "
                "The metrics below are from the locked test segment."
            )
            lock_a, lock_b, lock_c, lock_d = st.columns(4)
            lock_a.metric("Locked Long", f"{float(metrics.get('SelectedLongThreshold', metrics.get('LongThreshold', 0.0))):.2f}")
            lock_b.metric("Locked Short", f"{float(metrics.get('SelectedShortThreshold', metrics.get('ShortThreshold', 0.0))):.2f}")
            lock_c.metric("Validation Score", f"{float(metrics.get('ValidationSelectionScore', 0.0)):.2f}")
            lock_d.metric("Locked Test Rows", int(metrics.get("LockedTestRows", metrics.get("Rows", 0))))

            locked_warning = str(metrics.get("LockedTestWarning", "") or "")
            validation_warning = str(metrics.get("ValidationWarning", "") or "")
            if validation_warning:
                st.warning(f"Validation warning: {validation_warning}")
            if locked_warning:
                st.warning(f"Locked-test warning: {locked_warning}")

        st.markdown("### Signal Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades / Signals", int(metrics.get("NumberOfTrades", metrics.get("SignalCount", 0))))
        m2.metric("Frequency", f"{float(metrics.get('TradeFrequency_%', metrics.get('SignalFrequency_%', 0.0))):.2f}%")
        m3.metric("Win Rate", f"{float(metrics.get('WinRate_%', metrics.get('WinRateActive_%', 0.0))):.2f}%")
        m4.metric("Verdict", str(metrics.get("ThresholdVerdict", "")))

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Strategy Return", f"{float(metrics.get('TotalCompoundedReturn_%', metrics.get('StrategyTotalReturn_%', 0.0))):+.2f}%")
        r2.metric("Buy & Hold", f"{float(metrics.get('BuyHoldReturn_%', 0.0)):+.2f}%")
        r3.metric("Vs Buy & Hold", f"{float(metrics.get('StrategyMinusBuyHold_%', 0.0)):+.2f}%")
        r4.metric("Exposure", f"{float(metrics.get('Exposure_%', metrics.get('SignalFrequency_%', 0.0))):.2f}%")

        metric_df = pd.DataFrame([metrics]).T.reset_index()
        metric_df.columns = ["Metric", "Value"]
        st.dataframe(metric_df, use_container_width=True)

        warnings_text = str(metrics.get("Warnings", "") or "")
        if warnings_text:
            st.error(f"Signal warning: {warnings_text}")
        if float(metrics.get("NumberOfTrades", metrics.get("SignalCount", 0))) < 5:
            st.warning("Very few trades/signals. Treat this as insufficient evidence.")
        if float(metrics.get("StrategyMinusBuyHold_%", 0.0)) <= 0:
            st.error("Strategy failed buy-and-hold on this test split.")

        tab_lock, tab_signals, tab_sweep, tab_phase6 = st.tabs([
            "Validation Lock",
            "Trade / Signal Log",
            "Threshold Sweep",
            "Phase 6 Model",
        ])

        with tab_lock:
            if is_validation_locked:
                st.markdown("### Validation-Locked Threshold Selection")
                selected_threshold = getattr(signal_result, "selected_threshold", {}) or {}
                if selected_threshold:
                    selected_df = pd.DataFrame([selected_threshold]).T.reset_index()
                    selected_df.columns = ["Field", "Value"]
                    st.dataframe(selected_df, use_container_width=True)

                st.markdown("### Validation vs Locked Test")
                comparison = getattr(signal_result, "validation_test_comparison", pd.DataFrame())
                if comparison is not None and not comparison.empty:
                    st.dataframe(comparison, use_container_width=True)
                    comparison_csv = comparison.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Export Validation vs Test CSV",
                        data=comparison_csv,
                        file_name=f"signal_engine_{signal_output.asset}_{signal_output.horizon}D_validation_locked_comparison.csv".replace(" ", "_"),
                        mime="text/csv",
                    )
                else:
                    st.info("No validation/test comparison table available.")

                st.markdown("### Validation Threshold Table")
                validation_sweep = getattr(signal_result, "validation_sweep", pd.DataFrame())
                if validation_sweep is not None and not validation_sweep.empty:
                    st.dataframe(validation_sweep, use_container_width=True)
                    validation_csv = validation_sweep.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Export Validation Threshold CSV",
                        data=validation_csv,
                        file_name=f"signal_engine_{signal_output.asset}_{signal_output.horizon}D_validation_thresholds.csv".replace(" ", "_"),
                        mime="text/csv",
                    )
                else:
                    st.info("No validation threshold table available.")
            else:
                st.info("Manual threshold mode did not run validation-locked selection.")

        with tab_signals:
            st.markdown("### Realistic Trade Log" if result_style == "non_overlapping_realistic" else "### Overlapping Research Signal Log")
            st.dataframe(signal_result.signal_frame, use_container_width=True)
            signal_csv = signal_result.signal_frame.to_csv(index=True).encode("utf-8")
            st.download_button(
                "📥 Export Trade/Signal Log CSV",
                data=signal_csv,
                file_name=f"signal_engine_{signal_output.asset}_{signal_output.horizon}D_signals.csv".replace(" ", "_"),
                mime="text/csv",
            )

        with tab_sweep:
            if is_validation_locked:
                st.markdown("### Validation Threshold Table")
                st.caption("This table was evaluated only on the validation segment and used for locked threshold selection. Locked-test rows were not swept.")
                sweep_table = getattr(signal_result, "validation_sweep", pd.DataFrame())
            else:
                st.markdown("### Research-Only Threshold Sweep")
                st.caption("This sweep is evaluated on the test split for reporting only. Do not select production thresholds from it without separate validation.")
                sweep_table = signal_result.threshold_sweep

            if sweep_table is not None and not sweep_table.empty:
                st.dataframe(sweep_table, use_container_width=True)
                sweep_csv = sweep_table.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Threshold Sweep CSV",
                    data=sweep_csv,
                    file_name=f"signal_engine_{signal_output.asset}_{signal_output.horizon}D_threshold_sweep.csv".replace(" ", "_"),
                    mime="text/csv",
                )
            else:
                st.info("No threshold sweep table available.")

        with tab_phase6:
            st.markdown("### Phase 6 Direct Model Leaderboard")
            st.dataframe(signal_output.leaderboard, use_container_width=True)
            st.markdown("### Phase 6 Direction Baselines")
            st.dataframe(signal_output.baseline_board, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE: 30-DAY FORECAST
# ════════════════════════════════════════════════════════════════

elif page == "📅 30-Day Forecast":
    st.markdown('<p class="main-header">📅 30-Day Price Forecast</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not st.session_state.trained:
        st.warning("⚠️ No models trained yet. Go to **Train Models** first.")
    else:
        _stop_if_asset_mismatch(selected_asset)
        trainer = st.session_state.trainer
        pp = st.session_state.pp
        data = st.session_state.data
        df_features = st.session_state.df_features

        model_name = st.selectbox("Select a model for forecasting", list(trainer.results.keys()))
        n_days = st.slider("Forecast horizon (days)", 5, 60, 30)

        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner(f"Generating {n_days}-day forecast..."):
                result = trainer.results[model_name]
                predictor = Predictor(model=result.model, preprocessor=pp, is_sequence_model=False)

                active_target_col = getattr(pp, "target_col", target_col)
                ti = TechnicalIndicators(prefix=_target_prefix(active_target_col))
                fe = FeatureEngineer(target_col=active_target_col)

                forecast_df = predictor.forecast(
                    df_features, feature_cols=data.feature_cols, n_days=n_days,
                    indicators_engine=ti, feature_engineer=fe,
                )
                vol = df_features["Daily_Return"].std()
                forecast_df = predictor.add_confidence_bands(forecast_df, historical_volatility=vol)

            st.success(f"✔ {n_days}-day {selected_asset} forecast generated using {model_name}")

            viz = Visualizer()
            fig = viz.plot_forecast_plotly(forecast_df, df_features, n_history_days=90)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Forecast Table")
            st.dataframe(forecast_df, use_container_width=True)

            csv = forecast_df.to_csv().encode("utf-8")
            st.download_button("📥 Download Forecast (CSV)", data=csv, file_name=f"{model_name}_{n_days}day_forecast.csv", mime="text/csv")
