"""
feature_engineering.py — Advanced Feature Creation Pipeline
=============================================================
Builds the full feature set used for model training:
  • Lag features              (1, 3, 5, 7, 15, 30 days)
  • Rolling statistics         (mean, std, max, min)
  • Returns & volatility
  • Cross-asset ratios         (Gold/Silver, Gold/Oil, Dollar/Gold)
  • Momentum & trend strength
  • Calendar / seasonality features
  • Holiday flag

Combines with TechnicalIndicators (indicators.py) to produce the
final feature-engineered DataFrame consumed by preprocessing.py.

Usage
-----
    from src.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    df_feat = fe.build_features(df)       # df already has indicators
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config_loader import ConfigLoader
from src.logger import get_logger
from src.utils import add_holiday_flag, timer

logger = get_logger(__name__)
cfg    = ConfigLoader()


class FeatureEngineer:
    """
    Generates advanced predictive features for gold price forecasting.

    Parameters
    ----------
    target_col : str
        Column to engineer lag/rolling features for (default "Gold_Close").

    Methods
    -------
    build_features(df)        → pd.DataFrame  (full pipeline)
    add_lag_features(df)      → pd.DataFrame
    add_rolling_features(df)  → pd.DataFrame
    add_returns_volatility(df)→ pd.DataFrame
    add_ratio_features(df)    → pd.DataFrame
    add_trend_strength(df)    → pd.DataFrame
    add_calendar_features(df) → pd.DataFrame
    """

    def __init__(self, target_col: Optional[str] = None):
        self.target_col = target_col or cfg.get("data.target_column", "Gold_Close")

        feat_cfg = cfg.get_section("features")
        self.lag_periods     = feat_cfg.get("lag_periods", [1, 3, 5, 7, 15, 30])
        self.rolling_windows = feat_cfg.get("rolling_windows", [5, 10, 20, 30, 60])
        self.include_time    = feat_cfg.get("include_time_features", True)
        self.include_ratios  = feat_cfg.get("include_ratios", True)
        self.include_vol     = feat_cfg.get("include_volatility", True)
        self.include_season  = feat_cfg.get("include_seasonality", True)

    # ──────────────────────────────────────────────────────────────
    # Master orchestrator
    # ──────────────────────────────────────────────────────────────

    @timer
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Date-indexed DataFrame, ideally already containing technical
            indicators (from indicators.py) and merged macro data.

        Returns
        -------
        pd.DataFrame  with all engineered features appended.
        """
        df = df.copy()
        n_before = df.shape[1]

        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_returns_volatility(df)

        if self.include_ratios:
            df = self.add_ratio_features(df)

        df = self.add_trend_strength(df)

        if self.include_time or self.include_season:
            df = self.add_calendar_features(df)

        # Drop rows with NaN created by lag/rolling windows (start of series)
        n_rows_before = len(df)
        df.dropna(inplace=True)
        n_rows_after = len(df)

        n_after = df.shape[1]
        logger.info(
            f"Feature engineering complete: +{n_after - n_before} columns, "
            f"-{n_rows_before - n_rows_after} rows (NaN warm-up period removed)"
        )
        return df

    # ════════════════════════════════════════════════════════════
    # LAG FEATURES
    # ════════════════════════════════════════════════════════════

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagged versions of the target column.

        Creates columns: Gold_Close_lag1, Gold_Close_lag3, ..., Gold_Close_lag30
        """
        for lag in self.lag_periods:
            df[f"{self.target_col}_lag{lag}"] = df[self.target_col].shift(lag)
        logger.debug(f"Lag features added for periods: {self.lag_periods}")
        return df

    # ════════════════════════════════════════════════════════════
    # ROLLING STATISTICS
    # ════════════════════════════════════════════════════════════

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling mean, std, max, min for the target column
        across multiple window sizes.
        """
        for window in self.rolling_windows:
            roll = df[self.target_col].rolling(window=window, min_periods=1)
            df[f"{self.target_col}_roll_mean_{window}"] = roll.mean()
            df[f"{self.target_col}_roll_std_{window}"]  = roll.std()
            df[f"{self.target_col}_roll_max_{window}"]  = roll.max()
            df[f"{self.target_col}_roll_min_{window}"]  = roll.min()

        logger.debug(f"Rolling features added for windows: {self.rolling_windows}")
        return df

    # ════════════════════════════════════════════════════════════
    # RETURNS & VOLATILITY
    # ════════════════════════════════════════════════════════════

    def add_returns_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add daily returns and rolling volatility (std of returns).
        """
        df["Daily_Return"]     = df[self.target_col].pct_change()
        df["Log_Return"]       = np.log(df[self.target_col] / df[self.target_col].shift(1))
        df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

        if self.include_vol:
            for window in (5, 10, 20, 30):
                df[f"Volatility_{window}d"] = df["Daily_Return"].rolling(window, min_periods=1).std()

        logger.debug("Returns & volatility features added")
        return df

    # ════════════════════════════════════════════════════════════
    # CROSS-ASSET RATIOS
    # ════════════════════════════════════════════════════════════

    def add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-asset ratios that are economically meaningful
        for gold pricing dynamics:
          - Gold/Silver Ratio    (classic precious-metals indicator)
          - Gold/Oil Ratio       (inflation & energy linkage)
          - Dollar/Gold Ratio    (USD strength inverse relationship)
          - Gold/SP500 Ratio     (risk-on vs risk-off proxy)
          - Gold/BTC Ratio       (alternative store-of-value comparison)
        """
        col_map = {
            "Silver": ["Silver_Close", "Silver_Adj Close"],
            "Oil":    ["Oil_Close", "Oil_Adj Close"],
            "DXY":    ["DXY_Close", "DXY_Adj Close"],
            "SP500":  ["SP500_Close", "SP500_Adj Close"],
            "BTC":    ["BTC_Close", "BTC_Adj Close"],
        }

        def _find_col(candidates: List[str]) -> Optional[str]:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        target = self.target_col
        target_name = target.replace("_Close", "")

        def _add_target_ratio(label: str, other_col: Optional[str]) -> None:
            if other_col and other_col != target:
                df[f"{target_name}_{label}_Ratio"] = df[target] / (df[other_col] + 1e-9)

        silver_col = _find_col(col_map["Silver"])
        _add_target_ratio("Silver", silver_col)

        oil_col = _find_col(col_map["Oil"])
        _add_target_ratio("Oil", oil_col)

        dxy_col = _find_col(col_map["DXY"])
        if dxy_col:
            df[f"Dollar_{target_name}_Ratio"] = df[dxy_col] / (df[target] + 1e-9)

        sp500_col = _find_col(col_map["SP500"])
        _add_target_ratio("SP500", sp500_col)

        btc_col = _find_col(col_map["BTC"])
        _add_target_ratio("BTC", btc_col)

        found = [c for c in [silver_col, oil_col, dxy_col, sp500_col, btc_col] if c]
        logger.debug(f"Ratio features added for target {target} using columns: {found}")
        return df

    # ════════════════════════════════════════════════════════════
    # TREND STRENGTH / MOMENTUM
    # ════════════════════════════════════════════════════════════

    def add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a simple trend-strength feature:
          trend_strength = (price - SMA_50) / SMA_50
        and a momentum-acceleration feature (2nd derivative of price).
        """
        if "SMA_50" in df.columns:
            df["Trend_Strength"] = (df[self.target_col] - df["SMA_50"]) / (df["SMA_50"] + 1e-9)
        else:
            sma50 = df[self.target_col].rolling(50, min_periods=1).mean()
            df["Trend_Strength"] = (df[self.target_col] - sma50) / (sma50 + 1e-9)

        # Acceleration = change in momentum
        df["Price_Acceleration"] = df[self.target_col].diff().diff()

        # Simple linear-regression slope over a 10-day rolling window
        df["Trend_Slope_10d"] = (
            df[self.target_col]
            .rolling(10, min_periods=2)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
        )

        logger.debug("Trend strength & momentum features added")
        return df

    # ════════════════════════════════════════════════════════════
    # CALENDAR / SEASONALITY
    # ════════════════════════════════════════════════════════════

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar-based seasonality features:
          - Day of week     (0=Monday ... 6=Sunday)
          - Month            (1–12)
          - Quarter           (1–4)
          - Day of month
          - Week of year
          - Is month start / end
          - Holiday flag (US)
          - Cyclical sine/cosine encodings (so model sees periodicity)
        """
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)
            df.index = idx

        df["DayOfWeek"]   = idx.dayofweek
        df["Month"]       = idx.month
        df["Quarter"]     = idx.quarter
        df["DayOfMonth"]  = idx.day
        df["WeekOfYear"]  = idx.isocalendar().week.values
        df["IsMonthStart"] = idx.is_month_start.astype(int)
        df["IsMonthEnd"]   = idx.is_month_end.astype(int)
        df["IsQuarterEnd"] = idx.is_quarter_end.astype(int)

        # Cyclical encodings — helps tree & DL models learn periodicity
        df["Month_sin"]     = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"]     = np.cos(2 * np.pi * df["Month"] / 12)
        df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

        # Holiday flag
        df = add_holiday_flag(df, country="US")

        logger.debug("Calendar & seasonality features added")
        return df

    # ════════════════════════════════════════════════════════════
    # Feature importance grouping (used by SHAP / dashboard)
    # ════════════════════════════════════════════════════════════

    def get_feature_groups(self, df: pd.DataFrame) -> dict:
        """
        Categorize all feature columns into logical groups for
        dashboard display & SHAP analysis.
        """
        cols = df.columns.tolist()
        groups = {
            "Price":       [c for c in cols if c.endswith(("_Open", "_High", "_Low", "_Close", "_Volume"))],
            "Lag":         [c for c in cols if "_lag" in c],
            "Rolling":     [c for c in cols if "_roll_" in c],
            "Returns":     [c for c in cols if "Return" in c or "Volatility" in c],
            "Ratios":      [c for c in cols if "Ratio" in c],
            "Trend":       [c for c in cols if "Trend" in c or "Acceleration" in c],
            "Calendar":    [c for c in cols if c in (
                "DayOfWeek", "Month", "Quarter", "DayOfMonth", "WeekOfYear",
                "IsMonthStart", "IsMonthEnd", "IsQuarterEnd", "is_holiday",
                "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
            )],
            "Indicators": [c for c in cols if c.startswith((
                "SMA_", "EMA_", "RSI", "ROC", "Momentum", "MACD",
                "BB_", "ATR", "ADX", "CCI", "StochRSI", "WilliamsR",
                "Ichimoku_", "OBV", "VWAP", "MFI",
            ))],
        }
        return groups


# ════════════════════════════════════════════════════════════════
# Standalone test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.data_loader import DataLoader
    from src.indicators import TechnicalIndicators

    print("=" * 60)
    print("  Feature Engineering — Smoke Test")
    print("=" * 60)

    loader = DataLoader(start_date="2018-01-01", end_date=None)  # None = today
    df = loader.load_all(use_cache=True)
    print(f"\nRaw data shape: {df.shape}")

    target_col = cfg.get("data.target_column", "Gold_Close")
    ti = TechnicalIndicators(prefix=target_col.replace("_Close", ""))
    df = ti.add_all(df)
    print(f"After indicators: {df.shape}")

    fe = FeatureEngineer(target_col=target_col)
    df_feat = fe.build_features(df)

    print(f"\n✔  Final feature-engineered shape: {df_feat.shape}")
    print(f"   Rows lost to NaN warm-up: {df.shape[0] - df_feat.shape[0]}")

    groups = fe.get_feature_groups(df_feat)
    print("\n   Feature groups:")
    for name, cols in groups.items():
        print(f"     {name:>12}: {len(cols)} columns")

    print(f"\n   Sample features (last row):")
    sample_cols = ["Gold_Close_lag1", "Gold_Close_roll_mean_20", "Daily_Return",
                    "Volatility_20d", "Trend_Strength", "Month_sin", "DayOfWeek"]
    available = [c for c in sample_cols if c in df_feat.columns]
    print(df_feat[available].iloc[-1])

    print("\n✔ feature_engineering.py working correctly")
