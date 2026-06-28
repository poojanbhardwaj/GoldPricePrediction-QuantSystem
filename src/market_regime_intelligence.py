"""Phase 15 market regime and environment intelligence.

The module calculates current and historical market regimes from available
price data, then connects those regimes to Phase 14 simulated paper sizing. It
does not alter model training, targets, or upstream capital gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names, get_target_column
from src.artifact_store import resolve_artifact, save_phase_artifacts


MARKET_REGIME_PHASE_NAME = "phase15_market_regime_intelligence"
REGIME_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"

TREND_SHORT_MA_ROWS = 20
TREND_LONG_MA_ROWS = 60
TREND_SLOPE_LOOKBACK_ROWS = 20
TREND_SIDEWAYS_RETURN_THRESHOLD_PCT = 3.0
TREND_SIDEWAYS_SLOPE_THRESHOLD_PCT = 0.75
TREND_SIDEWAYS_MA_GAP_THRESHOLD_PCT = 1.0
TREND_UP_RETURN_THRESHOLD_PCT = 3.0
TREND_UP_SLOPE_THRESHOLD_PCT = 0.75
TREND_UP_MA_GAP_THRESHOLD_PCT = 1.0
TREND_STRONG_RETURN_THRESHOLD_PCT = 10.0
TREND_STRONG_SLOPE_THRESHOLD_PCT = 1.5
TREND_STRONG_MA_GAP_THRESHOLD_PCT = 2.0
TREND_MOMENTUM_CONFIRMATION_PCT = 55.0

REGIME_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "OverallMarketRegime",
    "MarketStressLevel",
    "AssetRegimeStressLevel",
    "RiskSentimentRegime",
    "VolatilityRegime",
    "TrendBreadth",
    "RegimeConfidence",
    "MainDrivers",
    "MainRisks",
    "RecommendedResearchPosture",
)

ASSET_REGIME_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "AssetSpecificRegime",
    "TrendRegime",
    "VolatilityRegime",
    "DrawdownState",
    "MomentumState",
    "CrossAssetSupport",
    "RegimeScore",
    "RegimeConfidence",
    "MainRegimeReason",
    "MainRegimeRisk",
)

ASSET_HORIZON_REGIME_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "AssetSpecificRegime",
    "HorizonRegimeFit",
    "TrendRegime",
    "VolatilityRegime",
    "RiskSentimentRegime",
    "RegimeScore",
    "RegimeConfidence",
    "Phase14OptimizedPaperWeightPct",
    "RegimeSizingMultiplier",
    "RegimeAdjustedPaperWeightPct",
    "RegimeAction",
    "MainReason",
    "ReviewTrigger",
)

REGIME_FACTOR_COLUMNS: Tuple[str, ...] = (
    "Factor",
    "CurrentValue",
    "RollingPercentile",
    "RegimeImpact",
    "AffectedAssets",
    "AffectedHorizons",
    "Explanation",
)

REGIME_TRANSITION_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "PreviousRegime",
    "CurrentRegime",
    "RegimeChanged",
    "DaysInCurrentRegime",
    "TransitionRisk",
    "Explanation",
)

REGIME_RISK_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "RegimeRiskType",
    "Severity",
    "RiskScore",
    "PaperImpact",
    "CapitalImpact",
    "Explanation",
    "RecommendedAction",
)

REGIME_ADJUSTED_SIZING_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "Phase14OptimizedPaperWeightPct",
    "RegimeSizingMultiplier",
    "RegimeAdjustedPaperWeightPct",
    "SizeChangeFromRegimePct",
    "RegimeReason",
    "FinalRegimeSizingDecision",
)

NEXT_REGIME_ACTION_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "Action",
    "AffectedAssets",
    "AffectedHorizons",
    "WhyItMatters",
    "ExpectedEffect",
    "Urgency",
    "DependsOn",
)

REGIME_INPUT_SOURCE_COLUMNS: Tuple[str, ...] = (
    "SourceName",
    "Available",
    "Rows",
    "Columns",
    "LastDate",
    "MissingCriticalColumns",
    "Notes",
)


@dataclass
class MarketRegimeIntelligenceReport:
    regime_summary_table: pd.DataFrame
    asset_regime_table: pd.DataFrame
    asset_horizon_regime_table: pd.DataFrame
    regime_factor_table: pd.DataFrame
    regime_transition_table: pd.DataFrame
    regime_risk_table: pd.DataFrame
    regime_adjusted_sizing_table: pd.DataFrame
    next_regime_actions_table: pd.DataFrame
    regime_input_sources_table: pd.DataFrame
    input_source_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: Dict[str, Any] = field(default_factory=dict)
    saved_artifacts: Dict[str, Any] = field(default_factory=dict)


INPUT_SPECS: Dict[str, Tuple[str, str, bool]] = {
    "dynamic_sizing_summary_table": ("phase14_dynamic_risk_sizing", "dynamic_sizing_summary_table", False),
    "dynamic_position_sizing_table": ("phase14_dynamic_risk_sizing", "dynamic_position_sizing_table", False),
    "optimized_portfolio_table": ("phase14_dynamic_risk_sizing", "optimized_portfolio_table", False),
    "risk_multiplier_summary_table": ("phase14_dynamic_risk_sizing", "risk_multiplier_summary_table", False),
    "risk_adjusted_scenarios_table": ("phase14_dynamic_risk_sizing", "risk_adjusted_scenarios_table", False),
    "asset_horizon_risk_matrix": ("phase13_risk_warning_intelligence", "asset_horizon_risk_matrix", False),
    "risk_summary_table": ("phase13_risk_warning_intelligence", "risk_summary_table", False),
    "top_risks_table": ("phase13_risk_warning_intelligence", "top_risks_table", False),
    "warning_group_table": ("phase13_risk_warning_intelligence", "warning_group_table", False),
    "allocation_plan_table": ("Phase 12 Portfolio Capital Simulator", "allocation_plan_table", False),
    "paper_portfolio_table": ("Phase 12 Portfolio Capital Simulator", "paper_portfolio_table", False),
    "portfolio_drawdown_stress_table": ("Phase 12 Portfolio Capital Simulator", "portfolio_drawdown_stress_table", False),
    "cost_slippage_stress_table": ("Phase 12 Portfolio Capital Simulator", "cost_slippage_stress_table", False),
    "correlation_concentration_table": ("Phase 12 Portfolio Capital Simulator", "correlation_concentration_table", False),
}


def _empty_frame(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _to_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value)


def _normalise_horizon(df: Any) -> pd.DataFrame:
    out = _to_frame(df)
    if out.empty:
        return out
    if "Horizon" in out.columns:
        out["Horizon"] = out["Horizon"].astype(str).str.replace("D", "", regex=False)
        out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce")
    return out


def _prepare_market_data(market_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = _to_frame(market_data)
    if df.empty:
        return df
    df = df.copy()
    date_col = None
    for col in ["Date", "date", "Datetime", "Timestamp"]:
        if col in df.columns:
            date_col = col
            break
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    else:
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()].sort_index()
        except Exception:
            pass
    return df


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _subset(df: pd.DataFrame, asset: str, horizon: int) -> pd.DataFrame:
    if df.empty or "Asset" not in df.columns or "Horizon" not in df.columns:
        return pd.DataFrame()
    h = pd.to_numeric(df["Horizon"].astype(str).str.replace("D", "", regex=False), errors="coerce")
    return df[df["Asset"].astype(str).eq(str(asset)) & h.eq(int(horizon))].copy()


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _pct_change(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return np.nan
    start = _safe_float(series.iloc[-periods - 1])
    end = _safe_float(series.iloc[-1])
    if not np.isfinite(start) or start == 0 or not np.isfinite(end):
        return np.nan
    return (end / start - 1.0) * 100.0


def _rolling_percentile(series: pd.Series, value: float, window: int = 252) -> float:
    if series.empty or not np.isfinite(value):
        return np.nan
    hist = series.dropna().tail(window)
    if hist.empty:
        return np.nan
    return float((hist <= value).mean() * 100.0)


def _realized_vol(series: pd.Series, window: int = 20) -> Tuple[float, float]:
    returns = series.pct_change().dropna()
    if len(returns) < max(10, window // 2):
        return np.nan, np.nan
    rolling = returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252) * 100.0
    current = _safe_float(rolling.dropna().iloc[-1] if not rolling.dropna().empty else np.nan)
    percentile = _rolling_percentile(rolling.dropna(), current)
    return current, percentile


def _classify_trend(series: pd.Series) -> Tuple[str, str, float]:
    if len(series) < TREND_LONG_MA_ROWS:
        return "Unknown", "Insufficient price history for trend classification.", 0.0
    price = _safe_float(series.iloc[-1])
    ma_short = _safe_float(series.rolling(TREND_SHORT_MA_ROWS).mean().iloc[-1])
    ma_long = _safe_float(series.rolling(TREND_LONG_MA_ROWS).mean().iloc[-1])
    ret_long = _pct_change(series, TREND_LONG_MA_ROWS)
    ma_slope = _pct_change(series.rolling(TREND_LONG_MA_ROWS).mean().dropna(), TREND_SLOPE_LOOKBACK_ROWS)
    ma_gap = ((ma_short / ma_long) - 1.0) * 100.0 if np.isfinite(ma_short) and np.isfinite(ma_long) and ma_long != 0 else np.nan
    recent_returns = series.tail(TREND_LONG_MA_ROWS + 1).pct_change().dropna()
    up_consistency = float((recent_returns > 0).mean() * 100.0) if not recent_returns.empty else np.nan
    down_consistency = float((recent_returns < 0).mean() * 100.0) if not recent_returns.empty else np.nan

    metrics = {
        "return": ret_long,
        "slope": ma_slope,
        "ma_gap": ma_gap,
    }
    finite_metrics = [abs(value) for value in metrics.values() if np.isfinite(value)]
    if not finite_metrics:
        return "Unknown", "Insufficient numeric trend signals for classification.", 0.0

    sideways_return = not np.isfinite(ret_long) or abs(ret_long) < TREND_SIDEWAYS_RETURN_THRESHOLD_PCT
    sideways_slope = not np.isfinite(ma_slope) or abs(ma_slope) < TREND_SIDEWAYS_SLOPE_THRESHOLD_PCT
    sideways_gap = not np.isfinite(ma_gap) or abs(ma_gap) < TREND_SIDEWAYS_MA_GAP_THRESHOLD_PCT
    if sideways_return and sideways_slope and sideways_gap:
        return (
            "Sideways",
            f"Trend signals are small: {TREND_LONG_MA_ROWS}D return {ret_long:.2f}%, MA slope {ma_slope:.2f}%, MA gap {ma_gap:.2f}%.",
            50.0,
        )

    positive_magnitude_signals = [
        np.isfinite(ret_long) and ret_long >= TREND_UP_RETURN_THRESHOLD_PCT,
        np.isfinite(ma_slope) and ma_slope >= TREND_UP_SLOPE_THRESHOLD_PCT,
        np.isfinite(ma_gap) and ma_gap >= TREND_UP_MA_GAP_THRESHOLD_PCT,
    ]
    negative_magnitude_signals = [
        np.isfinite(ret_long) and ret_long <= -TREND_UP_RETURN_THRESHOLD_PCT,
        np.isfinite(ma_slope) and ma_slope <= -TREND_UP_SLOPE_THRESHOLD_PCT,
        np.isfinite(ma_gap) and ma_gap <= -TREND_UP_MA_GAP_THRESHOLD_PCT,
    ]
    positive_signals = [
        *positive_magnitude_signals,
        np.isfinite(price) and np.isfinite(ma_short) and np.isfinite(ma_long) and price > ma_short > ma_long,
        np.isfinite(up_consistency) and up_consistency >= TREND_MOMENTUM_CONFIRMATION_PCT,
    ]
    negative_signals = [
        *negative_magnitude_signals,
        np.isfinite(price) and np.isfinite(ma_short) and np.isfinite(ma_long) and price < ma_short < ma_long,
        np.isfinite(down_consistency) and down_consistency >= TREND_MOMENTUM_CONFIRMATION_PCT,
    ]
    positive_count = int(sum(positive_signals))
    negative_count = int(sum(negative_signals))
    positive_magnitude_count = int(sum(positive_magnitude_signals))
    negative_magnitude_count = int(sum(negative_magnitude_signals))
    reason = (
        f"{TREND_LONG_MA_ROWS}D return {ret_long:.2f}%, MA slope {ma_slope:.2f}%, "
        f"MA gap {ma_gap:.2f}%, up consistency {up_consistency:.1f}%."
    )

    if positive_count >= 3 and positive_magnitude_count >= 2 and (
        (np.isfinite(ret_long) and ret_long >= TREND_STRONG_RETURN_THRESHOLD_PCT)
        or (
            np.isfinite(ma_slope)
            and np.isfinite(ma_gap)
            and ma_slope >= TREND_STRONG_SLOPE_THRESHOLD_PCT
            and ma_gap >= TREND_STRONG_MA_GAP_THRESHOLD_PCT
        )
    ):
        return "StrongUptrend", f"Multiple strong positive trend signals: {reason}", 85.0
    if negative_count >= 3 and negative_magnitude_count >= 2 and (
        (np.isfinite(ret_long) and ret_long <= -TREND_STRONG_RETURN_THRESHOLD_PCT)
        or (
            np.isfinite(ma_slope)
            and np.isfinite(ma_gap)
            and ma_slope <= -TREND_STRONG_SLOPE_THRESHOLD_PCT
            and ma_gap <= -TREND_STRONG_MA_GAP_THRESHOLD_PCT
        )
    ):
        return "StrongDowntrend", f"Multiple strong negative trend signals: {reason}", 15.0
    if positive_magnitude_count >= 1 and positive_count >= 2 and positive_count > negative_count:
        return "Uptrend", f"Meaningful positive trend signals: {reason}", 68.0
    if negative_magnitude_count >= 1 and negative_count >= 2 and negative_count > positive_count:
        return "Downtrend", f"Meaningful negative trend signals: {reason}", 32.0
    return "Sideways", f"Trend evidence is mixed or too small: {reason}", 50.0


def _classify_volatility(series: pd.Series) -> Tuple[str, float, float]:
    current, percentile = _realized_vol(series, 20)
    if not np.isfinite(current) or not np.isfinite(percentile):
        return "Unknown", current, percentile
    if percentile >= 90:
        return "ExtremeVolatility", current, percentile
    if percentile >= 70:
        return "HighVolatility", current, percentile
    if percentile <= 30:
        return "LowVolatility", current, percentile
    return "NormalVolatility", current, percentile


def _drawdown_state(series: pd.Series) -> Tuple[str, float]:
    if len(series) < 20:
        return "Unknown", np.nan
    high = series.rolling(min(len(series), 252), min_periods=1).max()
    current_dd = _safe_float(series.iloc[-1] / high.iloc[-1] - 1.0, np.nan) * 100.0
    if not np.isfinite(current_dd):
        return "Unknown", np.nan
    if current_dd <= -20:
        return "SevereDrawdown", current_dd
    if current_dd <= -10:
        return "Drawdown", current_dd
    if current_dd <= -4:
        return "MildDrawdown", current_dd
    return "NoMaterialDrawdown", current_dd


def _momentum_state(series: pd.Series) -> str:
    ret20 = _pct_change(series, 20)
    if not np.isfinite(ret20):
        return "Unknown"
    if ret20 > 2:
        return "Positive"
    if ret20 < -2:
        return "Negative"
    return "Neutral"


def _risk_sentiment(df: pd.DataFrame) -> Tuple[str, str]:
    vix = _series(df, "VIX_Close")
    spx = _series(df, "SP500_Close")
    drivers: List[str] = []
    vix_value = _safe_float(vix.iloc[-1] if not vix.empty else np.nan)
    spx_ret = _pct_change(spx, 20) if not spx.empty else np.nan
    if np.isfinite(vix_value):
        drivers.append(f"VIX {vix_value:.2f}")
    if np.isfinite(spx_ret):
        drivers.append(f"S&P 500 20D return {spx_ret:.2f}%")
    if np.isfinite(vix_value) and vix_value >= 35:
        return "Stress", "; ".join(drivers) or "VIX stress is elevated."
    if (np.isfinite(vix_value) and vix_value >= 25) or (np.isfinite(spx_ret) and spx_ret <= -5):
        return "RiskOff", "; ".join(drivers) or "Risk sentiment is weak."
    if (np.isfinite(vix_value) and vix_value <= 18) and (not np.isfinite(spx_ret) or spx_ret >= 0):
        return "RiskOn", "; ".join(drivers) or "Risk sentiment is supportive."
    return "Neutral", "; ".join(drivers) or "Risk sentiment data is limited."


def _commodity_regime(df: pd.DataFrame) -> str:
    dxy_ret = _pct_change(_series(df, "DXY_Close"), 20)
    oil_ret = _pct_change(_series(df, "Oil_Close"), 20)
    if np.isfinite(dxy_ret) and dxy_ret < -1 and (not np.isfinite(oil_ret) or oil_ret >= 0):
        return "CommoditySupportive"
    if np.isfinite(dxy_ret) and dxy_ret > 2:
        return "CommodityPressure"
    return "CommodityNeutral" if np.isfinite(dxy_ret) or np.isfinite(oil_ret) else "Unknown"


def _calibrated_regime_confidence(
    series: pd.Series,
    market_data: pd.DataFrame,
    trend: str,
    vol: str,
    drawdown: str,
    risk_sentiment: str,
    cross_support: str,
) -> float:
    """Estimate regime confidence without pretending regime labels are certain."""
    if series.empty or trend == "Unknown" or vol == "Unknown":
        return 20.0 if not series.empty else 0.0

    data_score = min(25.0, len(series) / 252.0 * 25.0)
    optional_cols = ["SP500_Close", "VIX_Close", "DXY_Close", "TNX_Close"]
    available_optional = sum(1 for col in optional_cols if col in market_data.columns and not _series(market_data, col).empty)
    factor_score = available_optional / len(optional_cols) * 15.0
    confidence = 40.0 + data_score + factor_score

    if trend in {"StrongUptrend", "StrongDowntrend"}:
        confidence += 12.0
    elif trend in {"Uptrend", "Downtrend"}:
        confidence += 8.0
    elif trend == "Sideways":
        confidence += 4.0

    if vol == "LowVolatility":
        confidence += 5.0
    elif vol == "NormalVolatility":
        confidence += 3.0
    elif vol == "HighVolatility":
        confidence -= 8.0
    elif vol == "ExtremeVolatility":
        confidence -= 14.0

    if drawdown == "MildDrawdown":
        confidence -= 5.0
    elif drawdown == "Drawdown":
        confidence -= 10.0
    elif drawdown == "SevereDrawdown":
        confidence -= 16.0

    if cross_support == "Supportive":
        confidence += 5.0
    elif cross_support == "Pressure":
        confidence -= 3.0
    else:
        confidence -= 5.0

    conflicting_risk_sentiment = (
        risk_sentiment in {"Stress", "RiskOff"} and trend in {"Uptrend", "StrongUptrend"}
    ) or (risk_sentiment == "RiskOn" and trend in {"Downtrend", "StrongDowntrend"})
    if conflicting_risk_sentiment:
        confidence -= 12.0

    if available_optional < 2:
        confidence = min(confidence, 70.0)
    elif available_optional < len(optional_cols):
        confidence = min(confidence, 82.0)
    if cross_support == "Mixed":
        confidence = min(confidence, 85.0)
    if trend == "Sideways":
        confidence = min(confidence, 75.0)
    if drawdown in {"Drawdown", "SevereDrawdown"} or vol == "ExtremeVolatility":
        confidence = min(confidence, 78.0)

    return round(float(np.clip(confidence, 0.0, 90.0)), 4)


def _asset_score(trend: str, vol: str, drawdown: str, risk_sentiment: str) -> float:
    trend_score = {
        "StrongUptrend": 85,
        "Uptrend": 68,
        "Sideways": 50,
        "Downtrend": 32,
        "StrongDowntrend": 15,
        "Unknown": 35,
    }.get(trend, 35)
    score = float(trend_score)
    if vol == "ExtremeVolatility":
        score -= 25
    elif vol == "HighVolatility":
        score -= 12
    elif vol == "LowVolatility":
        score += 5
    if drawdown == "SevereDrawdown":
        score -= 25
    elif drawdown == "Drawdown":
        score -= 14
    elif drawdown == "MildDrawdown":
        score -= 6
    if risk_sentiment == "Stress":
        score -= 10
    elif risk_sentiment == "RiskOff":
        score -= 5
    return float(np.clip(score, 0, 100))


def _asset_specific_regime(score: float, trend: str, vol: str, drawdown: str) -> str:
    if trend == "Unknown" or vol == "Unknown":
        return "Unknown"
    if vol == "ExtremeVolatility" or drawdown == "SevereDrawdown" or trend == "StrongDowntrend":
        return "Dangerous" if score < 35 else "Unfavorable"
    if score >= 70:
        return "Favorable"
    if score >= 45:
        return "Neutral"
    if score >= 25:
        return "Unfavorable"
    return "Dangerous"


def _horizon_fit(horizon: int, trend: str, vol: str, asset_regime: str) -> str:
    if asset_regime == "Unknown":
        return "Unknown"
    if vol == "ExtremeVolatility" and horizon <= 5:
        return "PoorForHorizon"
    if asset_regime == "Dangerous":
        return "PoorForHorizon"
    if trend in {"StrongUptrend", "Uptrend"} and horizon >= 10 and vol in {"LowVolatility", "NormalVolatility"}:
        return "FavorableForHorizon"
    if trend == "Sideways" and vol in {"HighVolatility", "ExtremeVolatility"}:
        return "PoorForHorizon"
    return "NeutralForHorizon"


def _regime_multiplier(asset_regime: str, horizon_fit: str, vol: str, allow_small_increase: bool) -> float:
    multiplier = {
        "Favorable": 1.0,
        "Neutral": 0.9,
        "Unfavorable": 0.6,
        "Dangerous": 0.0,
        "Unknown": 0.5,
    }.get(asset_regime, 0.5)
    if horizon_fit == "PoorForHorizon":
        multiplier = min(multiplier, 0.5)
    elif horizon_fit == "FavorableForHorizon" and allow_small_increase:
        multiplier = min(1.05, max(multiplier, 1.0))
    if vol == "ExtremeVolatility":
        multiplier = min(multiplier, 0.5)
    return float(np.clip(multiplier, 0.0, 1.05 if allow_small_increase else 1.0))


def _regime_action(phase14_weight: float, adjusted_weight: float, asset_regime: str, fit: str) -> str:
    if phase14_weight <= 0:
        return "NoSignal"
    if adjusted_weight <= 0 and asset_regime == "Dangerous":
        return "BlockDueToRegime"
    if adjusted_weight < phase14_weight:
        return "ReducePaperTracking"
    if fit == "PoorForHorizon":
        return "WatchlistOnly"
    return "KeepPaperTracking"


def _final_sizing_decision(phase14_weight: float, adjusted_weight: float, asset_regime: str) -> str:
    if phase14_weight <= 0:
        return "NoPaperSignal"
    if adjusted_weight <= 0 and asset_regime == "Dangerous":
        return "ZeroDueToDangerousRegime"
    if adjusted_weight < phase14_weight:
        return "ReduceDueToRegime"
    return "KeepPhase14Size"


def _resolve_inputs(
    use_artifact_store: bool,
    prefer_uploaded: bool,
    uploaded_overrides: Optional[Dict[str, Any]],
    direct_tables: Dict[str, Any],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    uploaded_overrides = uploaded_overrides or {}
    tables: Dict[str, pd.DataFrame] = {}
    resolved_rows: List[Dict[str, Any]] = []
    for key, (phase, artifact, required) in INPUT_SPECS.items():
        direct = direct_tables.get(key)
        if direct is not None:
            df = _normalise_horizon(direct)
            tables[key] = df
            resolved_rows.append({"Artifact": artifact, "Phase": phase, "Source": "DirectInput", "RunId": "", "Rows": int(len(df)), "CreatedAt": "", "Status": "Loaded", "Path": ""})
            continue
        if use_artifact_store or uploaded_overrides.get(key) is not None:
            resolved = resolve_artifact(phase, artifact, uploaded_file=uploaded_overrides.get(key), prefer_uploaded=prefer_uploaded, required=required)
            data = resolved.get("Data")
            tables[key] = _normalise_horizon(data) if data is not None else pd.DataFrame()
            resolved_rows.append({k: v for k, v in resolved.items() if k != "Data"})
        else:
            tables[key] = pd.DataFrame()
            resolved_rows.append({"Artifact": artifact, "Phase": phase, "Source": "Missing", "RunId": "", "Rows": 0, "CreatedAt": "", "Status": "MissingOptional", "Path": ""})
    source_columns = ["Artifact", "Source", "RunId", "Rows", "CreatedAt", "Status", "Phase", "Path"]
    return tables, pd.DataFrame(resolved_rows, columns=source_columns)


def _load_project_market_data() -> Optional[pd.DataFrame]:
    if not DEFAULT_MARKET_DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DEFAULT_MARKET_DATA_PATH)
    except Exception:
        return None


def _input_source_table(market_data: pd.DataFrame, assets: Iterable[str], project_data_used: bool) -> pd.DataFrame:
    required_cols = [get_target_column(asset) for asset in assets]
    missing = [col for col in required_cols if col not in market_data.columns]
    rows = [
        {
            "SourceName": "market_data",
            "Available": bool(not market_data.empty),
            "Rows": int(len(market_data)),
            "Columns": int(len(market_data.columns)) if not market_data.empty else 0,
            "LastDate": str(market_data.index.max().date()) if not market_data.empty and hasattr(market_data.index.max(), "date") else "",
            "MissingCriticalColumns": "; ".join(missing),
            "Notes": "Loaded from project master dataset." if project_data_used else "Loaded from direct input or upload.",
        }
    ]
    for col in ["SP500_Close", "BTC_Close", "Gold_Close", "Silver_Close", "Oil_Close", "DXY_Close", "VIX_Close", "TNX_Close"]:
        rows.append(
            {
                "SourceName": col,
                "Available": bool(col in market_data.columns),
                "Rows": int(market_data[col].notna().sum()) if col in market_data.columns else 0,
                "Columns": 1 if col in market_data.columns else 0,
                "LastDate": str(market_data[col].dropna().index.max().date()) if col in market_data.columns and not market_data[col].dropna().empty and hasattr(market_data[col].dropna().index.max(), "date") else "",
                "MissingCriticalColumns": "" if col in market_data.columns else col,
                "Notes": "Optional cross-asset factor." if col in market_data.columns else "Optional factor missing.",
            }
        )
    return pd.DataFrame(rows, columns=list(REGIME_INPUT_SOURCE_COLUMNS))


def _asset_regimes(market_data: pd.DataFrame, assets: Iterable[str], risk_sentiment: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, Any]] = {}
    for asset in assets:
        target = get_target_column(asset)
        series = _series(market_data, target)
        if series.empty or len(series) < 60:
            row = {
                "Asset": asset,
                "AssetSpecificRegime": "Unknown",
                "TrendRegime": "Unknown",
                "VolatilityRegime": "Unknown",
                "DrawdownState": "Unknown",
                "MomentumState": "Unknown",
                "CrossAssetSupport": "Unknown",
                "RegimeScore": 0.0,
                "RegimeConfidence": 0.0,
                "MainRegimeReason": "Insufficient price data for regime classification.",
                "MainRegimeRisk": "InsufficientRegimeData",
            }
            rows.append(row)
            details[asset] = {**row, "SeriesLength": len(series)}
            continue
        trend, trend_reason, _ = _classify_trend(series)
        vol, vol_value, vol_percentile = _classify_volatility(series)
        drawdown, drawdown_value = _drawdown_state(series)
        momentum = _momentum_state(series)
        score = _asset_score(trend, vol, drawdown, risk_sentiment)
        regime = _asset_specific_regime(score, trend, vol, drawdown)
        cross_support = "Supportive" if risk_sentiment == "RiskOn" and trend in {"Uptrend", "StrongUptrend"} else "Pressure" if risk_sentiment in {"Stress", "RiskOff"} and trend in {"Downtrend", "StrongDowntrend"} else "Mixed"
        confidence = _calibrated_regime_confidence(series, market_data, trend, vol, drawdown, risk_sentiment, cross_support)
        main_risk = "ExtremeVolatility" if vol == "ExtremeVolatility" else "StrongDowntrend" if trend == "StrongDowntrend" else "RiskOffStress" if risk_sentiment in {"Stress", "RiskOff"} else "None"
        row = {
            "Asset": asset,
            "AssetSpecificRegime": regime,
            "TrendRegime": trend,
            "VolatilityRegime": vol,
            "DrawdownState": drawdown,
            "MomentumState": momentum,
            "CrossAssetSupport": cross_support,
            "RegimeScore": round(score, 4),
            "RegimeConfidence": round(confidence, 4),
            "MainRegimeReason": trend_reason,
            "MainRegimeRisk": main_risk,
        }
        rows.append(row)
        details[asset] = {**row, "Series": series, "VolatilityValue": vol_value, "VolatilityPercentile": vol_percentile, "DrawdownPct": drawdown_value}
    return pd.DataFrame(rows, columns=list(ASSET_REGIME_COLUMNS)), details


def _phase14_weight(tables: Dict[str, pd.DataFrame], asset: str, horizon: int) -> float:
    dyn = _subset(tables.get("dynamic_position_sizing_table", pd.DataFrame()), asset, horizon)
    return max(0.0, _safe_float(dyn["OptimizedPaperWeightPct"].iloc[0] if not dyn.empty and "OptimizedPaperWeightPct" in dyn.columns else 0.0, 0.0))


def _asset_horizon_regimes(
    asset_regime: pd.DataFrame,
    details: Dict[str, Dict[str, Any]],
    tables: Dict[str, pd.DataFrame],
    assets: Iterable[str],
    horizons: Iterable[int],
    risk_sentiment: str,
    allow_small_increase: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ah_rows: List[Dict[str, Any]] = []
    sizing_rows: List[Dict[str, Any]] = []
    risk_rows: List[Dict[str, Any]] = []
    by_asset = {str(row["Asset"]): row for _, row in asset_regime.iterrows()}
    for asset in assets:
        regime_row = by_asset.get(asset, {})
        detail = details.get(asset, {})
        for horizon in horizons:
            phase14_weight = _phase14_weight(tables, asset, int(horizon))
            fit = _horizon_fit(int(horizon), str(regime_row.get("TrendRegime", "Unknown")), str(regime_row.get("VolatilityRegime", "Unknown")), str(regime_row.get("AssetSpecificRegime", "Unknown")))
            multiplier = _regime_multiplier(str(regime_row.get("AssetSpecificRegime", "Unknown")), fit, str(regime_row.get("VolatilityRegime", "Unknown")), allow_small_increase)
            adjusted = phase14_weight * multiplier
            if phase14_weight <= 0:
                adjusted = 0.0
            action = _regime_action(phase14_weight, adjusted, str(regime_row.get("AssetSpecificRegime", "Unknown")), fit)
            reason = f"{regime_row.get('AssetSpecificRegime', 'Unknown')} regime with {fit}; Phase 14 paper weight {phase14_weight:.2f}%."
            ah_rows.append(
                {
                    "Asset": asset,
                    "Horizon": int(horizon),
                    "AssetSpecificRegime": regime_row.get("AssetSpecificRegime", "Unknown"),
                    "HorizonRegimeFit": fit,
                    "TrendRegime": regime_row.get("TrendRegime", "Unknown"),
                    "VolatilityRegime": regime_row.get("VolatilityRegime", "Unknown"),
                    "RiskSentimentRegime": risk_sentiment,
                    "RegimeScore": regime_row.get("RegimeScore", 0.0),
                    "RegimeConfidence": regime_row.get("RegimeConfidence", 0.0),
                    "Phase14OptimizedPaperWeightPct": round(phase14_weight, 4),
                    "RegimeSizingMultiplier": round(multiplier, 4),
                    "RegimeAdjustedPaperWeightPct": round(adjusted, 4),
                    "RegimeAction": action,
                    "MainReason": reason,
                    "ReviewTrigger": "Recheck after next daily data update or regime transition.",
                }
            )
            sizing_rows.append(
                {
                    "Asset": asset,
                    "Horizon": int(horizon),
                    "Phase14OptimizedPaperWeightPct": round(phase14_weight, 4),
                    "RegimeSizingMultiplier": round(multiplier, 4),
                    "RegimeAdjustedPaperWeightPct": round(adjusted, 4),
                    "SizeChangeFromRegimePct": round(adjusted - phase14_weight, 4),
                    "RegimeReason": reason,
                    "FinalRegimeSizingDecision": _final_sizing_decision(phase14_weight, adjusted, str(regime_row.get("AssetSpecificRegime", "Unknown"))),
                }
            )
            risks = []
            if regime_row.get("VolatilityRegime") == "ExtremeVolatility":
                risks.append(("ExtremeVolatility", "High", 78, "ReducePaperSize", "Extreme volatility can distort short-horizon paper evidence."))
            if regime_row.get("TrendRegime") == "Sideways" and regime_row.get("VolatilityRegime") in {"HighVolatility", "ExtremeVolatility"}:
                risks.append(("ChoppySideways", "Medium", 55, "MonitorOnly", "Sideways high-volatility regime is choppy."))
            if regime_row.get("TrendRegime") == "StrongDowntrend":
                risks.append(("StrongDowntrend", "High", 74, "ReducePaperSize", "Strong downtrend is unfavorable."))
            if risk_sentiment in {"Stress", "RiskOff"}:
                risks.append(("RiskOffStress", "High" if risk_sentiment == "Stress" else "Medium", 70 if risk_sentiment == "Stress" else 55, "MonitorOnly", "Risk sentiment is weak or stressed."))
            if fit == "PoorForHorizon":
                risks.append(("UnfavorableHorizonFit", "Medium", 52, "ReducePaperSize", "Regime fit is poor for the selected horizon."))
            if regime_row.get("AssetSpecificRegime") == "Unknown":
                risks.append(("InsufficientRegimeData", "Medium", 45, "MonitorOnly", "Critical price history is missing or too short."))
            for risk_type, severity, score, paper_impact, explanation in risks:
                risk_rows.append(
                    {
                        "Asset": asset,
                        "Horizon": int(horizon),
                        "RegimeRiskType": risk_type,
                        "Severity": severity,
                        "RiskScore": score,
                        "PaperImpact": paper_impact,
                        "CapitalImpact": "BlocksRealCapital",
                        "Explanation": explanation,
                        "RecommendedAction": "Reduce, watch, or pause simulated paper tracking until regime evidence improves.",
                    }
                )
    return (
        pd.DataFrame(ah_rows, columns=list(ASSET_HORIZON_REGIME_COLUMNS)),
        pd.DataFrame(sizing_rows, columns=list(REGIME_ADJUSTED_SIZING_COLUMNS)),
        pd.DataFrame(risk_rows, columns=list(REGIME_RISK_COLUMNS)),
    )


def _factor_explanation(factor: str, impact: str, value: Any, percentile: float, ret: float = np.nan) -> str:
    value_text = f"value {value:.4f}" if isinstance(value, (int, float, np.floating)) and np.isfinite(value) else f"value {value}"
    percentile_text = f"rolling percentile {percentile:.1f}" if np.isfinite(percentile) else "rolling percentile unavailable"
    ret_text = f"20D return {ret:.2f}%" if np.isfinite(ret) else "20D return unavailable"
    if impact == "Supportive":
        return f"{factor} is supportive: {ret_text} is positive with {percentile_text}."
    if impact == "Pressure":
        return f"{factor} shows pressure: {ret_text} is negative or unfavorable with {percentile_text}."
    if impact == "Stress":
        return f"{factor} shows stress: {value_text} is elevated or in a high-risk range with {percentile_text}."
    if impact == "Neutral":
        return f"{factor} is neutral: {value_text}, {ret_text}, and {percentile_text} do not show a strong supportive or pressure condition."
    if impact in {"RiskOn", "RiskOff"}:
        return f"Composite risk sentiment is {impact}; broad factor evidence is summarized from VIX and S&P 500."
    return f"{factor} impact is {impact}; available evidence is limited."


def _factor_table(market_data: pd.DataFrame, assets: Iterable[str], horizons: Iterable[int], risk_sentiment: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    factor_specs = [
        ("S&P 500 trend", "SP500_Close"),
        ("BTC volatility", "BTC_Close"),
        ("Gold trend", "Gold_Close"),
        ("Silver trend", "Silver_Close"),
        ("Oil trend", "Oil_Close"),
        ("DXY trend", "DXY_Close"),
        ("VIX stress", "VIX_Close"),
        ("TNX/yield movement", "TNX_Close"),
    ]
    for factor, col in factor_specs:
        s = _series(market_data, col)
        if s.empty:
            rows.append({"Factor": factor, "CurrentValue": np.nan, "RollingPercentile": np.nan, "RegimeImpact": "Unknown", "AffectedAssets": "ALL", "AffectedHorizons": "ALL", "Explanation": f"{col} is not available."})
            continue
        value = _safe_float(s.iloc[-1])
        ret = np.nan
        if "volatility" in factor.lower():
            current, percentile = _realized_vol(s, 20)
            impact = "Stress" if np.isfinite(percentile) and percentile >= 85 else "Neutral"
            value = current
        elif "vix" in factor.lower():
            percentile = _rolling_percentile(s, value)
            impact = "Stress" if np.isfinite(value) and value >= 30 else "Neutral"
        else:
            ret = _pct_change(s, 20)
            percentile = _rolling_percentile(s.pct_change().rolling(20).sum().dropna(), ret / 100 if np.isfinite(ret) else np.nan)
            impact = "Supportive" if np.isfinite(ret) and ret > 1 else "Pressure" if np.isfinite(ret) and ret < -1 else "Neutral"
        rows.append({"Factor": factor, "CurrentValue": round(value, 4) if np.isfinite(value) else np.nan, "RollingPercentile": round(percentile, 4) if np.isfinite(percentile) else np.nan, "RegimeImpact": impact, "AffectedAssets": "; ".join(assets), "AffectedHorizons": "; ".join(f"{int(h)}D" for h in horizons), "Explanation": _factor_explanation(factor, impact, value, percentile, ret)})
    rows.append({"Factor": "Risk sentiment", "CurrentValue": risk_sentiment, "RollingPercentile": np.nan, "RegimeImpact": risk_sentiment, "AffectedAssets": "; ".join(assets), "AffectedHorizons": "; ".join(f"{int(h)}D" for h in horizons), "Explanation": _factor_explanation("Risk sentiment", risk_sentiment, risk_sentiment, np.nan)})
    return pd.DataFrame(rows, columns=list(REGIME_FACTOR_COLUMNS))


def _transition_table(market_data: pd.DataFrame, assets: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    risk_sentiment, _ = _risk_sentiment(market_data)
    for asset in assets:
        target = get_target_column(asset)
        series = _series(market_data, target)
        if len(series) < 100:
            rows.append({"Asset": asset, "PreviousRegime": "InsufficientHistory", "CurrentRegime": "InsufficientHistory", "RegimeChanged": False, "DaysInCurrentRegime": 0, "TransitionRisk": "InsufficientHistory", "Explanation": "Not enough history for transition analysis."})
            continue
        current_trend, _, _ = _classify_trend(series)
        previous_trend, _, _ = _classify_trend(series.iloc[:-20])
        changed = previous_trend != current_trend
        days = 20 if changed else min(len(series), 60)
        transition_risk = "RegimeShift" if changed else "Stable"
        if risk_sentiment in {"Stress", "RiskOff"} and changed:
            transition_risk = "HighTransitionRisk"
        rows.append({"Asset": asset, "PreviousRegime": previous_trend, "CurrentRegime": current_trend, "RegimeChanged": bool(changed), "DaysInCurrentRegime": int(days), "TransitionRisk": transition_risk, "Explanation": "Compares current trend regime with regime from 20 rows earlier."})
    return pd.DataFrame(rows, columns=list(REGIME_TRANSITION_COLUMNS))


def _market_stress_level(risk_sentiment: str) -> str:
    if risk_sentiment == "Stress":
        return "High"
    if risk_sentiment == "RiskOff":
        return "Medium"
    return "Low"


def _asset_regime_stress_level(asset_regime: pd.DataFrame, asset_horizon: pd.DataFrame) -> Tuple[str, str]:
    if asset_regime.empty:
        return "Unknown", "No asset regime rows available."
    total_assets = max(len(asset_regime), 1)
    dangerous_pct = asset_regime["AssetSpecificRegime"].eq("Dangerous").mean() * 100.0
    unfavorable_pct = asset_regime["AssetSpecificRegime"].isin(["Dangerous", "Unfavorable"]).mean() * 100.0
    poor_fit_pct = 0.0
    if not asset_horizon.empty and "HorizonRegimeFit" in asset_horizon.columns:
        poor_fit_pct = asset_horizon["HorizonRegimeFit"].eq("PoorForHorizon").mean() * 100.0
    dangerous_count = int(asset_regime["AssetSpecificRegime"].eq("Dangerous").sum())
    if dangerous_pct >= 33.0 or poor_fit_pct >= 50.0:
        level = "High"
    elif unfavorable_pct >= 33.0 or poor_fit_pct >= 25.0:
        level = "Medium"
    else:
        level = "Low"
    reason = (
        f"{dangerous_count}/{total_assets} assets are Dangerous; "
        f"{unfavorable_pct:.1f}% are Dangerous/Unfavorable; {poor_fit_pct:.1f}% asset-horizon rows have poor regime fit."
    )
    return level, reason


def _summary_table(asset_regime: pd.DataFrame, asset_horizon: pd.DataFrame, risk_sentiment: str, risk_reason: str) -> pd.DataFrame:
    if asset_regime.empty:
        return pd.DataFrame([{"OverallMarketRegime": "Unknown", "MarketStressLevel": "Unknown", "AssetRegimeStressLevel": "Unknown", "RiskSentimentRegime": "Unknown", "VolatilityRegime": "Unknown", "TrendBreadth": "Unknown", "RegimeConfidence": 0.0, "MainDrivers": "No regime data.", "MainRisks": "InsufficientRegimeData", "RecommendedResearchPosture": "Wait for sufficient market data."}], columns=list(REGIME_SUMMARY_COLUMNS))
    favorable = asset_regime["TrendRegime"].isin(["StrongUptrend", "Uptrend"]).mean() * 100.0
    overall_vol = "ExtremeVolatility" if asset_regime["VolatilityRegime"].eq("ExtremeVolatility").any() else "HighVolatility" if asset_regime["VolatilityRegime"].eq("HighVolatility").any() else "NormalVolatility"
    market_stress = _market_stress_level(risk_sentiment)
    asset_stress, asset_stress_reason = _asset_regime_stress_level(asset_regime, asset_horizon)
    if market_stress == "High" and asset_stress == "High":
        overall = "MixedStress"
        posture = "Keep simulated paper sizing cautious; both broad market stress and asset-level regime stress are elevated."
        driver = f"Broad market stress is high ({risk_reason}). Asset-level stress is also high: {asset_stress_reason}"
    elif market_stress == "High":
        overall = "MarketWideStress"
        posture = "Keep simulated paper sizing cautious until broad stress normalizes."
        driver = f"Broad market stress is high: {risk_reason}. Asset-level stress is {asset_stress}: {asset_stress_reason}"
    elif asset_stress == "High":
        overall = "AssetRegimeStress"
        posture = "Use reduced or selective simulated paper tracking until asset-level regimes improve."
        driver = f"Asset-level trend or horizon stress is high while broad risk sentiment is {risk_sentiment}. {asset_stress_reason}"
    elif market_stress == "Medium" or asset_stress == "Medium":
        overall = "Mixed"
        posture = "Use selective simulated paper tracking and keep regime warnings visible."
        driver = f"Partial stress: market stress {market_stress} ({risk_reason}); asset-regime stress {asset_stress}. {asset_stress_reason}"
    elif favorable >= 50 and risk_sentiment in {"RiskOn", "Neutral"}:
        overall = "Constructive"
        posture = "Keep regime-aware paper tracking with normal monitoring."
        driver = f"Trend breadth is constructive at {favorable:.1f}% while market stress is {market_stress}. {risk_reason}"
    else:
        overall = "Mixed"
        posture = "Use reduced or selective paper tracking until evidence strengthens."
        driver = f"Mixed regime evidence: trend breadth {favorable:.1f}%, market stress {market_stress}, asset stress {asset_stress}. {risk_reason}"
    confidence = float(pd.to_numeric(asset_regime["RegimeConfidence"], errors="coerce").fillna(0).mean())
    main_risks = "; ".join(asset_regime[asset_regime["MainRegimeRisk"].ne("None")]["MainRegimeRisk"].astype(str).unique()) or "No dominant regime risk"
    return pd.DataFrame([{"OverallMarketRegime": overall, "MarketStressLevel": market_stress, "AssetRegimeStressLevel": asset_stress, "RiskSentimentRegime": risk_sentiment, "VolatilityRegime": overall_vol, "TrendBreadth": round(favorable, 4), "RegimeConfidence": round(min(confidence, 90.0), 4), "MainDrivers": driver, "MainRisks": main_risks, "RecommendedResearchPosture": posture}], columns=list(REGIME_SUMMARY_COLUMNS))


def _next_actions(regime_risk: pd.DataFrame, adjusted: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not regime_risk.empty:
        top = regime_risk.sort_values("RiskScore", ascending=False).head(5)
        rows.append({"Rank": 0, "Action": "Review highest regime-risk paper rows.", "AffectedAssets": "; ".join(top["Asset"].astype(str).unique()), "AffectedHorizons": "; ".join(f"{int(h)}D" for h in pd.to_numeric(top["Horizon"], errors="coerce").dropna().astype(int).unique()), "WhyItMatters": "Regime risks can reduce or pause simulated paper sizing.", "ExpectedEffect": "Better alignment between paper tracking and current market environment.", "Urgency": "High", "DependsOn": "Next daily market data update."})
    reduced = adjusted[adjusted["RegimeAdjustedPaperWeightPct"].astype(float) < adjusted["Phase14OptimizedPaperWeightPct"].astype(float)] if not adjusted.empty else pd.DataFrame()
    if not reduced.empty:
        rows.append({"Rank": 0, "Action": "Monitor regime-reduced sizing rows.", "AffectedAssets": "; ".join(reduced["Asset"].astype(str).unique()), "AffectedHorizons": "; ".join(f"{int(h)}D" for h in pd.to_numeric(reduced["Horizon"], errors="coerce").dropna().astype(int).unique()), "WhyItMatters": "Phase 15 reduced simulated paper exposure from Phase 14.", "ExpectedEffect": "Keeps simulated exposure aligned with trend and volatility.", "Urgency": "Medium", "DependsOn": "Regime transition and volatility normalization."})
    if not rows:
        rows.append({"Rank": 0, "Action": "Continue regime monitoring.", "AffectedAssets": "ALL", "AffectedHorizons": "ALL", "WhyItMatters": "No severe regime conflict was detected.", "ExpectedEffect": "Keeps regime labels current.", "Urgency": "Low", "DependsOn": "Next data refresh."})
    actions = pd.DataFrame(rows, columns=list(NEXT_REGIME_ACTION_COLUMNS))
    actions["Rank"] = range(1, len(actions) + 1)
    return actions


def run_market_regime_intelligence(
    *,
    market_data: Optional[pd.DataFrame] = None,
    use_project_market_data: bool = True,
    use_artifact_store: bool = False,
    prefer_uploaded: bool = False,
    uploaded_overrides: Optional[Dict[str, Any]] = None,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    allow_small_paper_increase: bool = False,
    autosave: bool = False,
    **direct_tables: Any,
) -> MarketRegimeIntelligenceReport:
    asset_list = list(assets or get_asset_names())
    horizon_list = [int(h) for h in (horizons or REGIME_HORIZONS)]
    project_used = False
    if market_data is None and use_project_market_data:
        market_data = _load_project_market_data()
        project_used = market_data is not None
    market = _prepare_market_data(market_data)
    tables, input_source = _resolve_inputs(bool(use_artifact_store), bool(prefer_uploaded), uploaded_overrides, direct_tables)
    risk_sentiment, risk_reason = _risk_sentiment(market)
    asset_regime, details = _asset_regimes(market, asset_list, risk_sentiment)
    asset_horizon, adjusted, regime_risk = _asset_horizon_regimes(asset_regime, details, tables, asset_list, horizon_list, risk_sentiment, bool(allow_small_paper_increase))
    factors = _factor_table(market, asset_list, horizon_list, risk_sentiment)
    transitions = _transition_table(market, asset_list)
    summary = _summary_table(asset_regime, asset_horizon, risk_sentiment, risk_reason)
    next_actions = _next_actions(regime_risk, adjusted)
    source_health = _input_source_table(market, asset_list, project_used)
    settings = {"phase": "15", "purpose": "market_regime_intelligence", "assets": asset_list, "horizons": horizon_list, "allow_small_paper_increase": bool(allow_small_paper_increase), "real_capital_gate_source": "upstream Phase 11/12/14"}
    report = MarketRegimeIntelligenceReport(
        regime_summary_table=summary.reset_index(drop=True),
        asset_regime_table=asset_regime.reset_index(drop=True),
        asset_horizon_regime_table=asset_horizon.reset_index(drop=True),
        regime_factor_table=factors.reset_index(drop=True),
        regime_transition_table=transitions.reset_index(drop=True),
        regime_risk_table=regime_risk.reset_index(drop=True),
        regime_adjusted_sizing_table=adjusted.reset_index(drop=True),
        next_regime_actions_table=next_actions.reset_index(drop=True),
        regime_input_sources_table=source_health.reset_index(drop=True),
        input_source_table=input_source.reset_index(drop=True),
        settings=settings,
    )
    if autosave:
        report.saved_artifacts = save_phase_artifacts(
            MARKET_REGIME_PHASE_NAME,
            {
                "regime_summary_table": report.regime_summary_table,
                "asset_regime_table": report.asset_regime_table,
                "asset_horizon_regime_table": report.asset_horizon_regime_table,
                "regime_factor_table": report.regime_factor_table,
                "regime_transition_table": report.regime_transition_table,
                "regime_risk_table": report.regime_risk_table,
                "regime_adjusted_sizing_table": report.regime_adjusted_sizing_table,
                "next_regime_actions_table": report.next_regime_actions_table,
                "regime_input_sources_table": report.regime_input_sources_table,
                "input_source_table": report.input_source_table,
            },
            inputs={},
            config=report.settings,
            warnings=report.regime_risk_table["RegimeRiskType"].dropna().astype(str).unique().tolist() if not report.regime_risk_table.empty else [],
        )
    return report


__all__ = [
    "MARKET_REGIME_PHASE_NAME",
    "MarketRegimeIntelligenceReport",
    "run_market_regime_intelligence",
]
