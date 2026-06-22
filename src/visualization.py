"""
visualization.py — Professional Chart Generation Engine
===========================================================
Generates every visualization required for the project:

  Actual vs Predicted   | Loss Curve            | Correlation Heatmap
  Feature Importance    | Candlestick Chart      | Moving Averages
  RSI Plot               | MACD Plot              | 30-Day Forecast
  Residual Plot          | Distribution Plot      | Model Comparison Bars

Two backends are provided for every chart where it makes sense:
  - Matplotlib/Seaborn (static, used for PDF reports / saved PNGs)
  - Plotly                (interactive, used in the Streamlit dashboard)

Usage
-----
    from src.visualization import Visualizer
    viz = Visualizer()
    fig = viz.plot_actual_vs_predicted(y_true, y_pred, dates, title="XGBoost")
    fig.savefig("report/actual_vs_predicted.png")          # matplotlib
    viz.plot_candlestick_plotly(df).show()                  # plotly
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.config_loader import ConfigLoader
from src.logger import get_logger
from src.utils import ensure_dir

logger = get_logger(__name__)
cfg    = ConfigLoader()

# ── Global style ───────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.figsize"]  = (12, 6)
plt.rcParams["figure.dpi"]      = 110
plt.rcParams["font.size"]       = 11
plt.rcParams["axes.titlesize"]  = 14
plt.rcParams["axes.titleweight"] = "bold"

GOLD_COLOR   = "#D4AF37"
DARK_BG      = "#0E1117"
ACCENT_GREEN = "#00C896"
ACCENT_RED   = "#FF4B4B"


class Visualizer:
    """
    Generates all required charts for the gold price prediction project.

    Methods (Matplotlib — static, for reports)
    -------------------------------------------
    plot_actual_vs_predicted(y_true, y_pred, dates, title)
    plot_loss_curve(history, title)
    plot_correlation_heatmap(df, top_n)
    plot_feature_importance(importance_series, top_n, title)
    plot_candlestick(df, n_days)
    plot_moving_averages(df, periods)
    plot_rsi(df)
    plot_macd(df)
    plot_forecast(forecast_df, history_df, n_history_days)
    plot_residuals(y_true, y_pred, title)
    plot_distribution(series, title)
    plot_model_comparison(leaderboard_df, metric)

    Methods (Plotly — interactive, for Streamlit)
    -----------------------------------------------
    plot_actual_vs_predicted_plotly(...)
    plot_candlestick_plotly(...)
    plot_forecast_plotly(...)
    plot_model_comparison_plotly(...)
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = ensure_dir(output_dir or cfg.resolve_path("reports") / "figures")

    # ════════════════════════════════════════════════════════════
    # 1. ACTUAL VS PREDICTED
    # ════════════════════════════════════════════════════════════

    def plot_actual_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Actual vs Predicted Gold Price",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Line chart comparing actual and predicted prices over time."""
        fig, ax = plt.subplots(figsize=(14, 6))
        x = dates if dates is not None else np.arange(len(y_true))

        ax.plot(x, y_true, label="Actual", color="#1f77b4", linewidth=1.8)
        ax.plot(x, y_pred, label="Predicted", color=GOLD_COLOR, linewidth=1.8, linestyle="--")
        ax.fill_between(x, y_true, y_pred, color="gray", alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel("Date" if dates is not None else "Time Step")
        ax.set_ylabel("Gold Price (USD)")
        ax.legend(loc="upper left")

        if dates is not None:
            fig.autofmt_xdate()

        fig.tight_layout()
        self._maybe_save(fig, save_as)
        return fig

    # ════════════════════════════════════════════════════════════
    # 2. LOSS CURVE
    # ════════════════════════════════════════════════════════════

    def plot_loss_curve(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Loss Curve",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Plot training vs validation loss across epochs (for DL models)."""
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(history.get("loss", []), label="Train Loss", color="#1f77b4")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="Validation Loss", color=GOLD_COLOR)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend()
        fig.tight_layout()
        self._maybe_save(fig, save_as)
        return fig

    # ════════════════════════════════════════════════════════════
    # 3. CORRELATION HEATMAP
    # ════════════════════════════════════════════════════════════

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        top_n: int = 20,
        target_col: str = "Gold_Close",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """
        Heatmap of correlations between the target and its top-N most
        correlated numeric features (keeps the chart readable).
        """
        numeric_df = df.select_dtypes(include=np.number)
        if target_col in numeric_df.columns:
            top_features = (
                numeric_df.corrwith(numeric_df[target_col]).abs().sort_values(ascending=False).head(top_n).index
            )
        else:
            top_features = numeric_df.columns[:top_n]

        corr = numeric_df[top_features].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
            annot_kws={"size": 7},
        )
        ax.set_title(f"Correlation Heatmap — Top {top_n} Features vs {target_col}")
        fig.tight_layout()
        self._maybe_save(fig, save_as)
        return fig

    # ════════════════════════════════════════════════════════════
    # 4. FEATURE IMPORTANCE
    # ════════════════════════════════════════════════════════════

    def plot_feature_importance(
        self,
        importance_series: pd.Series,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Horizontal bar chart of top-N most important features."""
        top = importance_series.sort_values(ascending=False).head(top_n).sort_values()

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
        ax.barh(top.index, top.values, color=colors)

        ax.set_title(title)
        ax.set_xlabel("Importance")
        fig.tight_layout()
        self._maybe_save(fig, save_as)
        return fig

    # ════════════════════════════════════════════════════════════
    # 5. CANDLESTICK CHART
    # ════════════════════════════════════════════════════════════

    def plot_candlestick(
        self,
        df: pd.DataFrame,
        n_days: int = 90,
        prefix: str = "Gold",
        save_as: Optional[str] = None,
    ):
        """Candlestick chart of the most recent N days using mplfinance."""
        import mplfinance as mpf

        cols = {
            "Open":  f"{prefix}_Open",
            "High":  f"{prefix}_High",
            "Low":   f"{prefix}_Low",
            "Close": f"{prefix}_Close",
        }
        missing = [c for c in cols.values() if c not in df.columns]
        if missing:
            logger.warning(f"Candlestick chart missing columns: {missing}")
            return None

        ohlc = df[list(cols.values())].tail(n_days).copy()
        ohlc.columns = list(cols.keys())
        if f"{prefix}_Volume" in df.columns:
            ohlc["Volume"] = df[f"{prefix}_Volume"].tail(n_days)

        save_path = self.output_dir / (save_as or "candlestick.png")
        mpf.plot(
            ohlc, type="candle", style="charles", title=f"{prefix} Price — Last {n_days} Days",
            volume=("Volume" in ohlc.columns), savefig=dict(fname=str(save_path), dpi=110),
        )
        logger.info(f"Candlestick chart saved → {save_path}")
        return save_path

    # ════════════════════════════════════════════════════════════
    # 6. MOVING AVERAGES
    # ════════════════════════════════════════════════════════════

    def plot_moving_averages(
        self,
        df: pd.DataFrame,
        periods: Optional[List[int]] = None,
        target_col: str = "Gold_Close",
        n_days: int = 250,
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Plot price with overlaid SMA lines for multiple periods."""
        periods = periods or [20, 50, 200]
        recent = df.tail(n_days)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(recent.index, recent[target_col], label="Price", color="black", linewidth=1.2)

        colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#6A4C93"]
        for i, p in enumerate(periods):
            col = f"SMA_{p}"
            if col in recent.columns:
                ax.plot(recent.index, recent[col], label=f"SMA-{p}", color=colors[i % len(colors)], linewidth=1.4)
            else:
                sma = recent[target_col].rolling(p).mean()
                ax.plot(recent.index, sma, label=f"SMA-{p}", color=colors[i % len(colors)], linewidth=1.4)

        ax.set_title(f"Gold Price with Moving Averages (Last {n_days} Days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        self._maybe_save(fig, save_as)
        return fig

    # ════════════════════════════════════════════════════════════
    # 7. RSI PLOT
    # ════════════════════════════════════════════════════════════

    def plot_rsi(
        self,
        df: pd.DataFrame,
        n_days: int = 250,
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """RSI oscillator with overbought (70) / oversold (30) zones."""
        if "RSI" not in df.columns:
            logger.warning("RSI column not found — skipping plot.")
            return None

        recent = df.tail(n_days)
        fig, ax = plt.subplots(figsize=(14, 4))

        ax.plot(recent.index, recent["RSI"], color="#9B59B6", linewidth=1.4)
        ax.axhline(70, color=ACCENT_RED, linestyle="--", linewidth=1, alpha=0.7, label="Overbought (70)")
        ax.axhline(30, color=ACCENT_GREEN, linestyle="--", linewidth=1, alpha=0.7, label="Oversold (30)")
        ax.fill_between(recent.index, 70, 100, color=ACCENT_RED, alpha=0.05)
        ax.fill_between(recent.index, 0, 30, color=ACCENT_GREEN, alpha=0.05)

        ax.set_title("Relative Strength Index (RSI)")
        ax.set_ylabel("RSI")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        self._maybe_save(fig, save_as)
        return fig

    # ════════════════════════════════════════════════════════════
    # 8. MACD PLOT
    # ════════════════════════════════════════════════════════════

    def plot_macd(
        self,
        df: pd.DataFrame,
        n_days: int = 250,
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """MACD line, signal line, and histogram."""
        required = ["MACD", "MACD_Signal", "MACD_Hist"]
        if not all(c in df.columns for c in required):
            logger.warning("MACD columns not found — skipping plot.")
            return None

        recent = df.tail(n_days)
        fig, ax = plt.subplots(figsize=(14, 5))

        colors = [ACCENT_GREEN if v >= 0 else ACCENT_RED for v in recent["MACD_Hist"]]
        ax.bar(recent.index, recent["MACD_Hist"], color=colors, alpha=0.4, width=1.0, label="Histogram")
        ax.plot(recent.index, recent["MACD"], color="#1f77b4", linewidth=1.4, label="MACD")
        ax.plot(recent.index, recent["MACD_Signal"], color=GOLD_COLOR, linewidth=1.4, label="Signal")

        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title("MACD (Moving Average Convergence Divergence)")
        ax.legend(loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        self._maybe_save(fig, save_as)
        return fig

    # ════════════════════════════════════════════════════════════
    # 9. 30-DAY FORECAST
    # ════════════════════════════════════════════════════════════

    def plot_forecast(
        self,
        forecast_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
        target_col: str = "Gold_Close",
        n_history_days: int = 90,
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Plot historical prices followed by the forecasted trajectory (+ confidence bands if present)."""
        fig, ax = plt.subplots(figsize=(14, 6))

        if history_df is not None:
            recent_hist = history_df[target_col].tail(n_history_days)
            ax.plot(recent_hist.index, recent_hist.values, label="Historical", color="#1f77b4", linewidth=1.6)

        ax.plot(
            forecast_df.index, forecast_df["Predicted_Price"],
            label="Forecast", color=GOLD_COLOR, linewidth=1.8, linestyle="--", marker="o", markersize=3,
        )

        if {"Lower_Bound", "Upper_Bound"}.issubset(forecast_df.columns):
            ax.fill_between(
                forecast_df.index, forecast_df["Lower_Bound"], forecast_df["Upper_Bound"],
                color=GOLD_COLOR, alpha=0.15, label="95% Confidence Band",
            )

        ax.axvline(forecast_df.index[0], color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{len(forecast_df)}-Day Gold Price Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend(loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
        self._maybe_save(fig, save_as)
        return fig

    # ════════════════════════════════════════════════════════════
    # 10. RESIDUAL PLOT
    # ════════════════════════════════════════════════════════════

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Plot",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Scatter plot of residuals (errors) vs predicted values."""
        residuals = np.array(y_true) - np.array(y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(y_pred, residuals, alpha=0.5, s=18, color=GOLD_COLOR, edgecolor="black", linewidth=0.3)
        axes[0].axhline(0, color=ACCENT_RED, linestyle="--", linewidth=1.5)
        axes[0].set_xlabel("Predicted Price")
        axes[0].set_ylabel("Residual (Actual − Predicted)")
        axes[0].set_title("Residuals vs Predicted")

        axes[1].hist(residuals, bins=40, color=GOLD_COLOR, edgecolor="black", alpha=0.7)
        axes[1].axvline(0, color=ACCENT_RED, linestyle="--", linewidth=1.5)
        axes[1].set_xlabel("Residual")
        axes[1].set_title("Residual Distribution")

        fig.suptitle(title, fontweight="bold")
        fig.tight_layout()
        self._maybe_save(fig, save_as)
        return fig

    # ════════════════════════════════════════════════════════════
    # 11. DISTRIBUTION PLOT
    # ════════════════════════════════════════════════════════════

    def plot_distribution(
        self,
        series: pd.Series,
        title: str = "Distribution",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Histogram + KDE of any numeric series (e.g. daily returns, prices)."""
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(series.dropna(), kde=True, color=GOLD_COLOR, ax=ax, bins=50)
        ax.axvline(series.mean(), color=ACCENT_RED, linestyle="--", label=f"Mean: {series.mean():.4f}")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        self._maybe_save(fig, save_as)
        return fig

    # ════════════════════════════════════════════════════════════
    # 12. MODEL COMPARISON
    # ════════════════════════════════════════════════════════════

    def plot_model_comparison(
        self,
        leaderboard_df: pd.DataFrame,
        metric: str = "RMSE",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Bar chart comparing all models on a chosen metric."""
        df_sorted = leaderboard_df.sort_values(metric)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted))) if metric != "R2" else \
                 plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_sorted)))

        bars = ax.barh(df_sorted["Model"], df_sorted[metric], color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3)

        ax.set_title(f"Model Comparison — {metric}")
        ax.set_xlabel(metric)
        fig.tight_layout()
        self._maybe_save(fig, save_as)
        return fig

    # ════════════════════════════════════════════════════════════
    # PLOTLY — Interactive versions for Streamlit
    # ════════════════════════════════════════════════════════════

    def plot_actual_vs_predicted_plotly(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Actual vs Predicted Gold Price",
    ):
        """Interactive Plotly line chart — actual vs predicted."""
        import plotly.graph_objects as go

        x = dates if dates is not None else list(range(len(y_true)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y_true, name="Actual", line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=x, y=y_pred, name="Predicted", line=dict(color=GOLD_COLOR, width=2, dash="dash")))
        fig.update_layout(
            title=title, xaxis_title="Date", yaxis_title="Price (USD)",
            template="plotly_dark", hovermode="x unified",
        )
        return fig

    def plot_candlestick_plotly(
        self,
        df: pd.DataFrame,
        n_days: int = 90,
        prefix: str = "Gold",
    ):
        """Interactive Plotly candlestick chart."""
        import plotly.graph_objects as go

        cols = {k: f"{prefix}_{k}" for k in ("Open", "High", "Low", "Close")}
        if not all(c in df.columns for c in cols.values()):
            logger.warning("Candlestick (plotly) missing OHLC columns.")
            return None

        recent = df.tail(n_days)
        fig = go.Figure(data=[go.Candlestick(
            x=recent.index,
            open=recent[cols["Open"]], high=recent[cols["High"]],
            low=recent[cols["Low"]], close=recent[cols["Close"]],
            increasing_line_color=ACCENT_GREEN, decreasing_line_color=ACCENT_RED,
        )])
        fig.update_layout(
            title=f"{prefix} Candlestick — Last {n_days} Days",
            template="plotly_dark", xaxis_rangeslider_visible=False,
        )
        return fig

    def plot_forecast_plotly(
        self,
        forecast_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
        target_col: str = "Gold_Close",
        n_history_days: int = 90,
    ):
        """Interactive Plotly forecast chart with confidence bands."""
        import plotly.graph_objects as go

        fig = go.Figure()

        if history_df is not None:
            recent_hist = history_df[target_col].tail(n_history_days)
            fig.add_trace(go.Scatter(
                x=recent_hist.index, y=recent_hist.values,
                name="Historical", line=dict(color="#1f77b4", width=2),
            ))

        if {"Lower_Bound", "Upper_Bound"}.issubset(forecast_df.columns):
            fig.add_trace(go.Scatter(
                x=forecast_df.index, y=forecast_df["Upper_Bound"],
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df.index, y=forecast_df["Lower_Bound"],
                fill="tonexty", fillcolor="rgba(212,175,55,0.15)",
                line=dict(width=0), name="95% Confidence Band",
            ))

        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["Predicted_Price"],
            name="Forecast", line=dict(color=GOLD_COLOR, width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=4),
        ))

        fig.update_layout(
            title=f"{len(forecast_df)}-Day Gold Price Forecast",
            xaxis_title="Date", yaxis_title="Price (USD)",
            template="plotly_dark", hovermode="x unified",
        )
        return fig

    def plot_model_comparison_plotly(
        self,
        leaderboard_df: pd.DataFrame,
        metric: str = "RMSE",
    ):
        """Interactive Plotly horizontal bar chart for model comparison."""
        import plotly.express as px

        df_sorted = leaderboard_df.sort_values(metric)
        ascending_is_better = metric != "R2" and metric != "DirectionalAccuracy"
        color_scale = "RdYlGn_r" if ascending_is_better else "RdYlGn"

        fig = px.bar(
            df_sorted, x=metric, y="Model", orientation="h",
            color=metric, color_continuous_scale=color_scale,
            title=f"Model Comparison — {metric}", template="plotly_dark",
        )
        fig.update_layout(showlegend=False)
        return fig

    # ════════════════════════════════════════════════════════════
    # Helper
    # ════════════════════════════════════════════════════════════

    def _maybe_save(self, fig: plt.Figure, save_as: Optional[str]) -> None:
        """Save a matplotlib figure to the reports/figures directory if a filename is given."""
        if save_as:
            path = self.output_dir / save_as
            fig.savefig(path, bbox_inches="tight", dpi=110)
            logger.info(f"Figure saved → {path}")


# ════════════════════════════════════════════════════════════════
# Standalone test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.indicators import TechnicalIndicators
    from src.feature_engineering import FeatureEngineer
    from src.preprocessing import Preprocessor
    from src.train import ModelTrainer

    print("=" * 70)
    print("  Visualizer — Full Pipeline Test")
    print("=" * 70)

    loader = DataLoader(start_date="2015-01-01", end_date=None)  # None = today
    df = loader.load_all(use_cache=True)
    ti = TechnicalIndicators(prefix="Gold")
    df = ti.add_all(df)
    fe = FeatureEngineer()
    df = fe.build_features(df)

    pp = Preprocessor()
    data = pp.run(df)

    trainer = ModelTrainer(use_optuna=False, target_scaler=data.target_scaler, preprocessor=pp)
    lr_result = trainer.train_linear_regression(data)
    rf_result = trainer.train_random_forest(data)
    trainer.results["Linear Regression"] = lr_result
    trainer.results["Random Forest"] = rf_result

    viz = Visualizer()

    print("\n[1] Actual vs Predicted...")
    fig1 = viz.plot_actual_vs_predicted(
        data.prices_test, lr_result.predictions_test, data.test_index,
        title="Linear Regression — Actual vs Predicted", save_as="actual_vs_predicted.png",
    )
    plt.close(fig1)

    print("[2] Correlation Heatmap...")
    fig2 = viz.plot_correlation_heatmap(df, top_n=15, save_as="correlation_heatmap.png")
    plt.close(fig2)

    print("[3] Feature Importance...")
    if rf_result.feature_importance is not None:
        fig3 = viz.plot_feature_importance(rf_result.feature_importance, save_as="feature_importance.png")
        plt.close(fig3)

    print("[4] Moving Averages...")
    fig4 = viz.plot_moving_averages(df, save_as="moving_averages.png")
    plt.close(fig4)

    print("[5] RSI...")
    fig5 = viz.plot_rsi(df, save_as="rsi.png")
    plt.close(fig5)

    print("[6] MACD...")
    fig6 = viz.plot_macd(df, save_as="macd.png")
    plt.close(fig6)

    print("[7] Residuals...")
    fig7 = viz.plot_residuals(data.prices_test, lr_result.predictions_test, save_as="residuals.png")
    plt.close(fig7)

    print("[8] Distribution...")
    fig8 = viz.plot_distribution(df["Daily_Return"], title="Daily Return Distribution", save_as="distribution.png")
    plt.close(fig8)

    print("[9] Model Comparison...")
    board = trainer.get_leaderboard("test")
    fig9 = viz.plot_model_comparison(board, metric="RMSE", save_as="model_comparison.png")
    plt.close(fig9)

    print("[10] Candlestick...")
    cs_path = viz.plot_candlestick(df, n_days=90)

    print(f"\n✔ All figures saved to: {viz.output_dir}")
    import os
    for f in sorted(os.listdir(viz.output_dir)):
        print(f"   - {f}")

    print("\n✔ visualization.py working correctly")
