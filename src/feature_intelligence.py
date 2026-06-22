"""
feature_intelligence.py — Phase 5 Better Feature Intelligence
==============================================================

Adds market-aware, multi-asset, leakage-safe features designed to improve
NEXT-DAY prediction quality across Gold, Silver, Crude Oil, Bitcoin, S&P 500,
and Gold ETF.

Important design rule
---------------------
These features use only information available at or before timestamp t.
They are valid for a next-day target such as log(price[t+1] / price[t]).
No shift(-1), future return, or future rolling statistic is used here.

Feature families
----------------
FI_Target_*        : target-specific momentum, volatility, drawdown, breakout
FI_Cross_*         : cross-asset relative strength, rolling correlation, beta
FI_Macro_*         : DXY, VIX, TNX, S&P 500 pressure features
FI_Regime_*        : volatility, trend, and risk-off regime flags/scores
FI_Asset_*         : common return/volatility features for each asset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from src.logger import get_logger

logger = get_logger(__name__)

EPS = 1e-9
TRADING_DAYS = 252

# Canonical close columns used across the whole project.
ASSET_CLOSE_COLUMNS: Dict[str, str] = {
    "Gold": "Gold_Close",
    "Silver": "Silver_Close",
    "Oil": "Oil_Close",
    "BTC": "BTC_Close",
    "SP500": "SP500_Close",
    "GLD": "GLD_Close",
}

MACRO_CLOSE_COLUMNS: Dict[str, str] = {
    "DXY": "DXY_Close",
    "VIX": "VIX_Close",
    "TNX": "TNX_Close",
}


@dataclass
class FeatureIntelligenceAudit:
    """Small audit object for the Streamlit Feature Intelligence page."""

    target_col: str
    total_rows: int
    total_columns: int
    phase5_columns: int
    family_counts: pd.DataFrame
    missing_summary: pd.DataFrame
    sample_columns: List[str]


def _safe_name(name: str) -> str:
    return str(name).replace(" ", "_").replace("&", "and").replace("/", "_").replace("-", "_")


def _as_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def _safe_log_return(price: pd.Series, periods: int = 1) -> pd.Series:
    price = pd.to_numeric(price, errors="coerce").astype(float)
    return np.log(price / price.shift(periods)).replace([np.inf, -np.inf], np.nan)


def _safe_pct_change(price: pd.Series, periods: int = 1) -> pd.Series:
    return pd.to_numeric(price, errors="coerce").astype(float).pct_change(periods).replace([np.inf, -np.inf], np.nan)


def _rolling_zscore(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mp = min_periods if min_periods is not None else max(3, min(window, 20))
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std()
    return ((s - mean) / (std + EPS)).replace([np.inf, -np.inf], np.nan)


def _rolling_beta(y_ret: pd.Series, x_ret: pd.Series, window: int) -> pd.Series:
    cov = y_ret.rolling(window, min_periods=max(5, window // 3)).cov(x_ret)
    var = x_ret.rolling(window, min_periods=max(5, window // 3)).var()
    return (cov / (var + EPS)).replace([np.inf, -np.inf], np.nan)


def _add_if_absent(df: pd.DataFrame, name: str, values: pd.Series | np.ndarray | float) -> None:
    """Avoid accidental overwrite if a future module creates the same feature."""
    if name not in df.columns:
        df[name] = values


def _available_price_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {asset: col for asset, col in ASSET_CLOSE_COLUMNS.items() if col in df.columns}


def add_phase5_feature_intelligence(
    df: pd.DataFrame,
    target_col: str = "Gold_Close",
    *,
    include_cross_asset: bool = True,
    include_macro: bool = True,
    clean_output: bool = True,
) -> pd.DataFrame:
    """
    Add Phase 5 feature intelligence to a date-indexed feature DataFrame.

    Parameters
    ----------
    df:
        Existing market/technical feature DataFrame.
    target_col:
        Current target close column, e.g. Gold_Close, BTC_Close.
    include_cross_asset:
        Whether to add rolling relative strength/correlation/beta features.
    include_macro:
        Whether to add DXY/VIX/TNX risk pressure features.
    clean_output:
        Replace +/-inf, forward-fill created gaps, and drop remaining early NaNs.

    Returns
    -------
    pd.DataFrame
        DataFrame with FI_* columns appended.
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} not found in DataFrame")

    out = df.copy().sort_index()
    n_before_cols = out.shape[1]
    n_before_rows = len(out)

    target = _as_float_series(out, target_col)
    target_prefix = _safe_name(target_col.replace("_Close", ""))
    ret1 = _safe_log_return(target, 1)

    # ── Target asset momentum / volatility / drawdown / breakout intelligence ──
    for w in (3, 5, 10, 20, 60):
        logret_w = _safe_log_return(target, w)
        vol_w = ret1.rolling(w, min_periods=max(2, min(w, 5))).std() * np.sqrt(TRADING_DAYS)
        rolling_high = target.rolling(w, min_periods=max(2, min(w, 5))).max()
        rolling_low = target.rolling(w, min_periods=max(2, min(w, 5))).min()
        prior_high = target.shift(1).rolling(w, min_periods=max(2, min(w, 5))).max()
        prior_low = target.shift(1).rolling(w, min_periods=max(2, min(w, 5))).min()

        _add_if_absent(out, f"FI_Target_LogRet_{w}d", logret_w)
        _add_if_absent(out, f"FI_Target_RealizedVol_{w}d", vol_w)
        _add_if_absent(out, f"FI_Target_ReturnZ_{w}d", _rolling_zscore(ret1, w, min_periods=max(3, min(w, 5))))
        _add_if_absent(out, f"FI_Target_Drawdown_{w}d", (target / (rolling_high + EPS)) - 1.0)
        _add_if_absent(out, f"FI_Target_DistanceFromLow_{w}d", (target / (rolling_low + EPS)) - 1.0)
        _add_if_absent(out, f"FI_Target_BreakoutPressure_{w}d", (target / (prior_high + EPS)) - 1.0)
        _add_if_absent(out, f"FI_Target_BreakdownPressure_{w}d", (target / (prior_low + EPS)) - 1.0)

    sma5 = target.rolling(5, min_periods=3).mean()
    sma20 = target.rolling(20, min_periods=5).mean()
    sma50 = target.rolling(50, min_periods=10).mean()
    sma100 = target.rolling(100, min_periods=20).mean()
    vol5 = ret1.rolling(5, min_periods=3).std() * np.sqrt(TRADING_DAYS)
    vol20 = ret1.rolling(20, min_periods=5).std() * np.sqrt(TRADING_DAYS)
    vol60 = ret1.rolling(60, min_periods=10).std() * np.sqrt(TRADING_DAYS)

    _add_if_absent(out, "FI_Target_DistSMA_5", (target / (sma5 + EPS)) - 1.0)
    _add_if_absent(out, "FI_Target_DistSMA_20", (target / (sma20 + EPS)) - 1.0)
    _add_if_absent(out, "FI_Target_DistSMA_50", (target / (sma50 + EPS)) - 1.0)
    _add_if_absent(out, "FI_Target_Trend_5_20", (sma5 / (sma20 + EPS)) - 1.0)
    _add_if_absent(out, "FI_Target_Trend_20_50", (sma20 / (sma50 + EPS)) - 1.0)
    _add_if_absent(out, "FI_Target_Trend_50_100", (sma50 / (sma100 + EPS)) - 1.0)
    _add_if_absent(out, "FI_Target_VolRatio_5_20", vol5 / (vol20 + EPS))
    _add_if_absent(out, "FI_Target_VolRatio_20_60", vol20 / (vol60 + EPS))
    _add_if_absent(out, "FI_Target_TrendPersistence_5d", np.sign(ret1).rolling(5, min_periods=3).mean())
    _add_if_absent(out, "FI_Target_TrendPersistence_20d", np.sign(ret1).rolling(20, min_periods=5).mean())
    _add_if_absent(out, "FI_Target_MomentumExhaustion_20d", _safe_log_return(target, 20) / (vol20 + EPS))
    _add_if_absent(out, "FI_Target_MomentumExhaustion_60d", _safe_log_return(target, 60) / (vol60 + EPS))

    # ── Common asset features and cross-asset relationships ──
    available_assets = _available_price_columns(out)
    target_ret20 = _safe_log_return(target, 20)
    target_ret60 = _safe_log_return(target, 60)

    for asset, col in available_assets.items():
        price = _as_float_series(out, col)
        asset_name = _safe_name(asset)
        asset_ret1 = _safe_log_return(price, 1)
        asset_ret5 = _safe_log_return(price, 5)
        asset_ret20 = _safe_log_return(price, 20)
        asset_ret60 = _safe_log_return(price, 60)
        asset_vol20 = asset_ret1.rolling(20, min_periods=5).std() * np.sqrt(TRADING_DAYS)

        _add_if_absent(out, f"FI_Asset_{asset_name}_LogRet_1d", asset_ret1)
        _add_if_absent(out, f"FI_Asset_{asset_name}_LogRet_5d", asset_ret5)
        _add_if_absent(out, f"FI_Asset_{asset_name}_LogRet_20d", asset_ret20)
        _add_if_absent(out, f"FI_Asset_{asset_name}_Vol_20d", asset_vol20)

        if include_cross_asset and col != target_col:
            _add_if_absent(out, f"FI_Cross_Target_vs_{asset_name}_RelStrength_20d", target_ret20 - asset_ret20)
            _add_if_absent(out, f"FI_Cross_Target_vs_{asset_name}_RelStrength_60d", target_ret60 - asset_ret60)
            _add_if_absent(out, f"FI_Cross_Target_{asset_name}_Corr_20d", ret1.rolling(20, min_periods=8).corr(asset_ret1))
            _add_if_absent(out, f"FI_Cross_Target_{asset_name}_Corr_60d", ret1.rolling(60, min_periods=15).corr(asset_ret1))
            _add_if_absent(out, f"FI_Cross_Target_{asset_name}_Beta_60d", _rolling_beta(ret1, asset_ret1, 60))

    # ── Macro/risk pressure features ──
    macro_zscores: List[pd.Series] = []
    if include_macro:
        for macro, col in MACRO_CLOSE_COLUMNS.items():
            if col not in out.columns:
                continue
            macro_name = _safe_name(macro)
            px = _as_float_series(out, col)
            change_1d = px.diff() if macro == "TNX" else _safe_log_return(px, 1)
            z60 = _rolling_zscore(px, 60, min_periods=15)
            z252 = _rolling_zscore(px, 252, min_periods=30)
            _add_if_absent(out, f"FI_Macro_{macro_name}_Change_1d", change_1d)
            _add_if_absent(out, f"FI_Macro_{macro_name}_Z_60d", z60)
            _add_if_absent(out, f"FI_Macro_{macro_name}_Z_252d", z252)
            macro_zscores.append(z60.rename(macro_name))

        risk_parts: List[pd.Series] = []
        if "VIX_Close" in out.columns:
            risk_parts.append(_rolling_zscore(_as_float_series(out, "VIX_Close"), 60, min_periods=15))
        if "DXY_Close" in out.columns:
            risk_parts.append(_rolling_zscore(_as_float_series(out, "DXY_Close"), 60, min_periods=15))
        if "TNX_Close" in out.columns:
            risk_parts.append(_rolling_zscore(_as_float_series(out, "TNX_Close"), 60, min_periods=15))
        if "SP500_Close" in out.columns:
            # Negative S&P pressure = risk-off contribution.
            risk_parts.append(-_rolling_zscore(_as_float_series(out, "SP500_Close"), 60, min_periods=15))

        if risk_parts:
            risk_score = pd.concat(risk_parts, axis=1).mean(axis=1)
            _add_if_absent(out, "FI_Macro_RiskOffScore", risk_score)
            _add_if_absent(out, "FI_Macro_RiskOffScore_Z_252d", _rolling_zscore(risk_score, 252, min_periods=30))

    # ── Regime flags/scores. These are numeric so sklearn/tree models can use them. ──
    vol_threshold = vol20.rolling(252, min_periods=30).median()
    risk_score_col = out.get("FI_Macro_RiskOffScore")
    risk_threshold = risk_score_col.rolling(252, min_periods=30).median() if isinstance(risk_score_col, pd.Series) else None

    _add_if_absent(out, "FI_Regime_HighVol", (vol20 > vol_threshold).astype(float))
    _add_if_absent(out, "FI_Regime_TrendUp_20_50", (sma20 > sma50).astype(float))
    _add_if_absent(out, "FI_Regime_AboveSMA20", (target > sma20).astype(float))
    _add_if_absent(out, "FI_Regime_TargetVolZ_252d", _rolling_zscore(vol20, 252, min_periods=30))
    if isinstance(risk_score_col, pd.Series):
        _add_if_absent(out, "FI_Regime_RiskOff", (risk_score_col > risk_threshold).astype(float))

    # Economically interpretable spreads when available.
    if "Gold_Close" in out.columns and "Silver_Close" in out.columns:
        _add_if_absent(
            out,
            "FI_Cross_GoldSilver_SpreadRet_20d",
            _safe_log_return(_as_float_series(out, "Gold_Close"), 20) - _safe_log_return(_as_float_series(out, "Silver_Close"), 20),
        )
    if "Gold_Close" in out.columns and "BTC_Close" in out.columns:
        _add_if_absent(
            out,
            "FI_Cross_GoldBTC_StoreValueSpread_20d",
            _safe_log_return(_as_float_series(out, "Gold_Close"), 20) - _safe_log_return(_as_float_series(out, "BTC_Close"), 20),
        )
    if "Gold_Close" in out.columns and "DXY_Close" in out.columns:
        _add_if_absent(
            out,
            "FI_Macro_GoldVsDollar_20d",
            _safe_log_return(_as_float_series(out, "Gold_Close"), 20) + _safe_log_return(_as_float_series(out, "DXY_Close"), 20),
        )

    if clean_output:
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.sort_index().ffill()
        # Do not backfill. Dropping remaining early rows is safer than leaking future values backward.
        out = out.dropna()

    # De-fragment after adding many feature columns.
    out = out.copy()

    n_after_cols = out.shape[1]
    n_after_rows = len(out)
    added = n_after_cols - n_before_cols
    logger.info(
        "Phase 5 feature intelligence added: +%s columns, -%s rows | target=%s",
        added,
        n_before_rows - n_after_rows,
        target_col,
    )
    return out


def phase5_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all Phase 5 FI_* feature columns."""
    return [c for c in df.columns if str(c).startswith("FI_")]


def _family_for_column(col: str) -> str:
    if col.startswith("FI_Target_"):
        return "Target momentum/volatility/regime input"
    if col.startswith("FI_Cross_"):
        return "Cross-asset relationship"
    if col.startswith("FI_Macro_"):
        return "Macro/risk pressure"
    if col.startswith("FI_Regime_"):
        return "Market regime"
    if col.startswith("FI_Asset_"):
        return "Asset return/volatility"
    return "Other"


def build_feature_intelligence_report(df: pd.DataFrame, target_col: str = "Gold_Close") -> FeatureIntelligenceAudit:
    """Build a compact audit report for dashboard display/tests."""
    fi_cols = phase5_feature_columns(df)
    families = pd.Series([_family_for_column(c) for c in fi_cols], name="FeatureFamily")
    family_counts = (
        families.value_counts()
        .rename_axis("FeatureFamily")
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    if fi_cols:
        missing = df[fi_cols].isna().mean().sort_values(ascending=False).head(20)
        missing_summary = missing.reset_index()
        missing_summary.columns = ["Feature", "MissingFraction"]
    else:
        missing_summary = pd.DataFrame(columns=["Feature", "MissingFraction"])

    return FeatureIntelligenceAudit(
        target_col=target_col,
        total_rows=int(len(df)),
        total_columns=int(df.shape[1]),
        phase5_columns=int(len(fi_cols)),
        family_counts=family_counts,
        missing_summary=missing_summary,
        sample_columns=fi_cols[:40],
    )


if __name__ == "__main__":
    # Lightweight synthetic smoke test. Real project tests live in tests/.
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    base = np.linspace(100, 160, len(idx))
    demo = pd.DataFrame(
        {
            "Gold_Close": base + np.sin(np.arange(len(idx)) / 7) * 2,
            "Silver_Close": base * 0.02 + np.sin(np.arange(len(idx)) / 5) * 0.1,
            "Oil_Close": 50 + np.sin(np.arange(len(idx)) / 10),
            "BTC_Close": 10000 + np.arange(len(idx)) * 25,
            "SP500_Close": 3000 + np.arange(len(idx)) * 2,
            "GLD_Close": base * 0.09,
            "DXY_Close": 100 + np.sin(np.arange(len(idx)) / 13),
            "VIX_Close": 20 + np.sin(np.arange(len(idx)) / 3),
            "TNX_Close": 4 + np.sin(np.arange(len(idx)) / 20) * 0.2,
        },
        index=idx,
    )
    out = add_phase5_feature_intelligence(demo, "Gold_Close")
    report = build_feature_intelligence_report(out, "Gold_Close")
    print(f"Phase 5 columns: {report.phase5_columns}")
    print(report.family_counts)
