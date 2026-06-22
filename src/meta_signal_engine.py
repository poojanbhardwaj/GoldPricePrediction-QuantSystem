"""
Phase 8: Regime-aware meta signal engine.

This module is intentionally rule-based. It consumes Phase 7G walk-forward
aggregate results as the reliability source of truth and combines them with
current/historical market regime features. It does not train a meta-model,
create future targets, or tune anything on locked-test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names, get_target_column


TRADING_DAYS = 252
DEFAULT_HORIZONS = [1, 5, 10, 20, 30]

META_DECISIONS = ["Trade", "No Trade", "Defensive Only", "Research Only", "Avoid"]

META_SIGNAL_COLUMNS = [
    "Asset",
    "Horizon",
    "MetaDecision",
    "MetaConfidenceScore",
    "MetaRiskScore",
    "RegimeLabel",
    "SignalReliabilityScore",
    "WalkForwardReliabilityScore",
    "BenchmarkRiskFlag",
    "CostFragilityFlag",
    "DrawdownRiskFlag",
    "StabilityFlag",
    "MainReason",
    "Warnings",
    "SuggestedUseCase",
]

PHASE7G_RELIABILITY_COLUMNS = [
    "NumberOfWindows",
    "BeatBuyHoldRate_%",
    "PositiveReturnRate_%",
    "AvgLockedStrategyReturn_%",
    "MedianLockedStrategyReturn_%",
    "AvgLockedVsBuyHold_%",
    "MedianLockedVsBuyHold_%",
    "WorstLockedVsBuyHold_%",
    "AvgLockedMaxDrawdown_%",
    "WorstLockedMaxDrawdown_%",
    "AvgLockedSharpe",
    "MedianLockedSharpe",
    "AvgTradesPerWindow",
    "LowTradeWindowCount",
    "ThresholdStability",
    "CooldownStability",
    "WalkForwardStabilityScore",
    "WalkForwardVerdict",
    "FailureReason",
]


@dataclass
class MetaSignalReport:
    decision_table: pd.DataFrame
    decision_summary: pd.DataFrame
    regime_features: pd.DataFrame
    warnings: List[str]
    settings: Dict[str, Any]


@dataclass
class MetaDecisionAuditReport:
    audit_table: pd.DataFrame
    threshold_config: pd.DataFrame
    mode_comparison: pd.DataFrame
    common_blocking_rules: pd.DataFrame
    near_miss_candidates: pd.DataFrame
    top_blocked_candidates: pd.DataFrame
    highest_confidence_candidates: pd.DataFrame
    highest_risk_candidates: pd.DataFrame
    warnings: List[str]
    settings: Dict[str, Any]


@dataclass
class MetaReliabilityGradingReport:
    grading_table: pd.DataFrame
    grade_counts: pd.DataFrame
    top_research_candidates: pd.DataFrame
    defensive_watchlist: pd.DataFrame
    avoid_archive_list: pd.DataFrame
    next_action_summary: pd.DataFrame
    score_components: pd.DataFrame
    warnings: List[str]
    settings: Dict[str, Any]


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    if not np.isfinite(value):
        return float(low)
    return float(np.clip(value, low, high))


def _pct_from_ratio(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(value * 100.0)


def _join_warnings(warnings: Iterable[str]) -> str:
    clean = [str(w).strip() for w in warnings if str(w).strip()]
    return "; ".join(dict.fromkeys(clean))


def _series_value(series: pd.Series, default: float = np.nan) -> float:
    if series.empty:
        return float(default)
    value = series.iloc[-1]
    return _safe_float(value, default=default)


def build_regime_features(
    raw_df: Optional[pd.DataFrame],
    asset_names: Optional[Iterable[str]] = None,
    *,
    as_of: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Build current regime features from historical/current prices only.

    No future targets or negative shifts are used. All columns are rolling or
    point-in-time calculations based on data available at or before ``as_of``.
    """
    assets = list(asset_names) if asset_names is not None else list(get_asset_names())
    rows: List[Dict[str, Any]] = []

    if raw_df is None or raw_df.empty:
        for asset in assets:
            rows.append(
                {
                    "Asset": asset,
                    "AsOfDate": "",
                    "TargetColumn": get_target_column(asset),
                    "RegimeDataWarning": "Missing raw price data",
                }
            )
        return pd.DataFrame(rows)

    df = raw_df.copy().sort_index()
    if as_of is not None:
        cutoff = pd.to_datetime(as_of)
        df = df.loc[pd.to_datetime(df.index) <= cutoff]

    sp500_return_60 = np.nan
    if "SP500_Close" in df.columns:
        sp500 = pd.to_numeric(df["SP500_Close"], errors="coerce").dropna()
        if len(sp500) >= 21:
            sp500_log_ret = np.log(sp500 / sp500.shift(1))
            sp500_return_60 = _pct_from_ratio(np.expm1(_series_value(sp500_log_ret.rolling(60, min_periods=20).sum())))

    for asset in assets:
        target_col = get_target_column(asset)
        row: Dict[str, Any] = {"Asset": asset, "TargetColumn": target_col}

        if target_col not in df.columns:
            row.update({"AsOfDate": "", "RegimeDataWarning": f"Missing price column: {target_col}"})
            rows.append(row)
            continue

        price = pd.to_numeric(df[target_col], errors="coerce").dropna()
        if price.empty:
            row.update({"AsOfDate": "", "RegimeDataWarning": f"No valid prices for {target_col}"})
            rows.append(row)
            continue

        log_ret = np.log(price / price.shift(1))
        ret_20 = np.expm1(log_ret.rolling(20, min_periods=5).sum())
        ret_60 = np.expm1(log_ret.rolling(60, min_periods=20).sum())
        vol_20 = log_ret.rolling(20, min_periods=5).std() * np.sqrt(TRADING_DAYS)
        vol_60 = log_ret.rolling(60, min_periods=20).std() * np.sqrt(TRADING_DAYS)
        high_60 = price.rolling(60, min_periods=5).max()
        high_252 = price.rolling(252, min_periods=20).max()
        drawdown_60 = price / high_60 - 1.0
        drawdown_252 = price / high_252 - 1.0
        sma_50 = price.rolling(50, min_periods=20).mean()
        sma_200 = price.rolling(200, min_periods=60).mean()
        latest_price = _safe_float(price.iloc[-1])
        latest_sma_50 = _series_value(sma_50)
        latest_sma_200 = _series_value(sma_200)

        trend_50 = latest_price / latest_sma_50 - 1.0 if latest_sma_50 and np.isfinite(latest_sma_50) else np.nan
        trend_200 = latest_price / latest_sma_200 - 1.0 if latest_sma_200 and np.isfinite(latest_sma_200) else np.nan
        vol_20_value = _series_value(vol_20)
        vol_60_value = _series_value(vol_60)
        vol_ratio = vol_20_value / vol_60_value if np.isfinite(vol_20_value) and np.isfinite(vol_60_value) and vol_60_value > 0 else np.nan

        row.update(
            {
                "AsOfDate": str(pd.to_datetime(price.index[-1]).date()),
                "LatestPrice": latest_price,
                "Return_20D_%": round(_pct_from_ratio(_series_value(ret_20)), 4),
                "Return_60D_%": round(_pct_from_ratio(_series_value(ret_60)), 4),
                "Volatility_20D_Annualized_%": round(_pct_from_ratio(vol_20_value), 4),
                "Volatility_60D_Annualized_%": round(_pct_from_ratio(vol_60_value), 4),
                "VolatilityRatio_20D_vs_60D": round(float(vol_ratio), 4) if np.isfinite(vol_ratio) else np.nan,
                "Drawdown_60D_%": round(_pct_from_ratio(_series_value(drawdown_60)), 4),
                "Drawdown_252D_%": round(_pct_from_ratio(_series_value(drawdown_252)), 4),
                "SMA_50": round(float(latest_sma_50), 4) if np.isfinite(latest_sma_50) else np.nan,
                "SMA_200": round(float(latest_sma_200), 4) if np.isfinite(latest_sma_200) else np.nan,
                "TrendVsSMA50_%": round(_pct_from_ratio(trend_50), 4),
                "TrendVsSMA200_%": round(_pct_from_ratio(trend_200), 4),
                "RecentBuyHoldStrength_%": round(_pct_from_ratio(_series_value(ret_60)), 4),
                "SP500Return_60D_%": round(float(sp500_return_60), 4) if np.isfinite(sp500_return_60) else np.nan,
                "RegimeDataWarning": "",
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def classify_market_regime(regime_row: Any) -> Dict[str, Any]:
    """Classify the latest market regime from current/historical regime features."""
    row = dict(regime_row) if not isinstance(regime_row, pd.Series) else regime_row.to_dict()
    warnings: List[str] = []

    data_warning = str(row.get("RegimeDataWarning", "") or "").strip()
    if data_warning:
        warnings.append(data_warning)
        return {
            "RegimeLabel": "Insufficient regime data",
            "RegimeRiskScore": 70.0,
            "RegimeReason": data_warning,
            "RegimeWarnings": warnings,
        }

    ret_60 = _safe_float(row.get("Return_60D_%"))
    vol_20 = _safe_float(row.get("Volatility_20D_Annualized_%"))
    vol_ratio = _safe_float(row.get("VolatilityRatio_20D_vs_60D"))
    dd_60 = _safe_float(row.get("Drawdown_60D_%"))
    dd_252 = _safe_float(row.get("Drawdown_252D_%"))
    trend_50 = _safe_float(row.get("TrendVsSMA50_%"))
    trend_200 = _safe_float(row.get("TrendVsSMA200_%"))
    sp500_60 = _safe_float(row.get("SP500Return_60D_%"))

    insufficient = sum(np.isfinite(v) for v in [ret_60, vol_20, dd_60, trend_50]) < 3
    if insufficient:
        warnings.append("Insufficient rolling history for a confident regime label")
        return {
            "RegimeLabel": "Insufficient regime data",
            "RegimeRiskScore": 65.0,
            "RegimeReason": "Not enough rolling history",
            "RegimeWarnings": warnings,
        }

    high_vol = (np.isfinite(vol_ratio) and vol_ratio >= 1.35) or (np.isfinite(vol_20) and vol_20 >= 55.0)
    deep_drawdown = (np.isfinite(dd_252) and dd_252 <= -20.0) or (np.isfinite(dd_60) and dd_60 <= -12.0)
    weak_trend = (np.isfinite(trend_200) and trend_200 < -3.0) or (np.isfinite(trend_50) and trend_50 < -3.0 and ret_60 < 0.0)
    constructive = (
        np.isfinite(ret_60)
        and ret_60 > 0.0
        and np.isfinite(trend_50)
        and trend_50 >= 0.0
        and (not np.isfinite(trend_200) or trend_200 >= -2.0)
    )

    risk_score = 35.0
    if high_vol:
        risk_score += 12.0
        warnings.append("Volatility regime is elevated")
    if deep_drawdown:
        risk_score += 25.0
        warnings.append("Current market is in a material drawdown")
    if weak_trend:
        risk_score += 15.0
        warnings.append("Trend regime is weak")
    if np.isfinite(sp500_60) and sp500_60 < -5.0:
        risk_score += 8.0
        warnings.append("Broad market risk appetite is weak")
    if constructive and not high_vol and not deep_drawdown:
        risk_score -= 10.0

    if deep_drawdown and high_vol:
        label = "High-volatility drawdown"
        reason = "Drawdown and volatility are both elevated"
    elif deep_drawdown or weak_trend:
        label = "Risk-off / weak trend"
        reason = "Recent trend or drawdown is unfavorable"
    elif constructive and high_vol:
        label = "Constructive but volatile"
        reason = "Trend is positive but volatility is elevated"
    elif constructive:
        label = "Constructive uptrend"
        reason = "Recent return and moving-average trend are supportive"
    else:
        label = "Range-bound / neutral"
        reason = "No strong trend or stress regime is dominant"

    return {
        "RegimeLabel": label,
        "RegimeRiskScore": round(_clip(risk_score), 2),
        "RegimeReason": reason,
        "RegimeWarnings": warnings,
    }


def build_signal_reliability_profile(walk_forward_row: Optional[Any]) -> Dict[str, Any]:
    """Convert one Phase 7G aggregate row into conservative reliability flags."""
    if walk_forward_row is None:
        row: Dict[str, Any] = {}
    elif isinstance(walk_forward_row, pd.Series):
        row = walk_forward_row.to_dict()
    else:
        row = dict(walk_forward_row)

    warnings: List[str] = []
    missing_row = bool(row.get("MissingPhase7GRow", False)) or not row
    missing_cols = [col for col in PHASE7G_RELIABILITY_COLUMNS if col not in row]
    if missing_row:
        warnings.append("Missing Phase 7G walk-forward aggregate row")
    elif missing_cols:
        warnings.append(f"MissingPhase7GColumns: {', '.join(missing_cols[:8])}")

    beat_rate = _safe_float(row.get("BeatBuyHoldRate_%"), default=0.0)
    positive_rate = _safe_float(row.get("PositiveReturnRate_%"), default=0.0)
    avg_vs = _safe_float(row.get("AvgLockedVsBuyHold_%"), default=0.0)
    median_vs = _safe_float(row.get("MedianLockedVsBuyHold_%"), default=0.0)
    worst_vs = _safe_float(row.get("WorstLockedVsBuyHold_%"), default=0.0)
    avg_dd = _safe_float(row.get("AvgLockedMaxDrawdown_%"), default=0.0)
    worst_dd = _safe_float(row.get("WorstLockedMaxDrawdown_%"), default=0.0)
    avg_sharpe = _safe_float(row.get("AvgLockedSharpe"), default=0.0)
    avg_trades = _safe_float(row.get("AvgTradesPerWindow"), default=0.0)
    low_trade_count = _safe_int(row.get("LowTradeWindowCount"), default=0)
    wf_score = _safe_float(row.get("WalkForwardStabilityScore"), default=np.nan)
    threshold_stability = str(row.get("ThresholdStability", "Unknown") or "Unknown")
    cooldown_stability = str(row.get("CooldownStability", "Unknown") or "Unknown")
    verdict = str(row.get("WalkForwardVerdict", "Missing Phase 7G") or "Missing Phase 7G")
    failure_reason = str(row.get("FailureReason", "") or "").strip()
    number_of_windows = _safe_int(row.get("NumberOfWindows"), default=0)

    computed_score = (
        beat_rate * 0.30
        + positive_rate * 0.20
        + np.clip(avg_vs, -20.0, 20.0) * 0.9
        + np.clip(median_vs, -20.0, 20.0) * 0.6
        + np.clip(avg_sharpe, -2.0, 2.0) * 6.0
        + min(max(avg_trades, 0.0) * 2.0, 12.0)
    )
    if "Strong walk-forward research candidate" in verdict:
        computed_score += 12.0
    elif verdict == "Research candidate":
        computed_score += 6.0
    elif "Do not trust" in verdict:
        computed_score -= 30.0
    elif "Weak" in verdict:
        computed_score -= 12.0

    if threshold_stability.lower() != "stable":
        computed_score -= 8.0
    if cooldown_stability.lower() != "stable":
        computed_score -= 5.0
    if low_trade_count > 0:
        computed_score -= min(20.0, low_trade_count * 5.0)
    if worst_vs <= -15.0:
        computed_score -= min(20.0, abs(worst_vs) * 0.4)
    if worst_dd <= -25.0:
        computed_score -= min(18.0, abs(worst_dd) * 0.35)
    if missing_cols:
        computed_score -= min(20.0, len(missing_cols) * 1.5)
    if missing_row:
        computed_score = 0.0

    if np.isfinite(wf_score):
        walk_forward_score = _clip(wf_score)
        reliability_score = _clip(computed_score * 0.60 + walk_forward_score * 0.40)
    else:
        walk_forward_score = _clip(computed_score)
        reliability_score = _clip(computed_score)

    benchmark_risk = bool(beat_rate < 50.0 or avg_vs <= 0.0 or median_vs <= 0.0 or worst_vs <= -15.0)
    cost_fragile = bool((0.0 < avg_vs < 2.0) or (0.0 < median_vs < 1.5) or "CostFragile" in failure_reason)
    drawdown_risk = bool(worst_dd <= -25.0 or avg_dd <= -20.0 or "drawdown" in failure_reason.lower())

    if missing_row:
        stability_flag = "MissingPhase7G"
    elif low_trade_count > 0 or avg_trades < 2.0:
        stability_flag = "LowEvidence"
        warnings.append("LowTradeCount: walk-forward windows have too few locked-test trades")
    elif threshold_stability.lower() != "stable" and cooldown_stability.lower() != "stable":
        stability_flag = "ThresholdCooldownUnstable"
        warnings.append("SplitUnstable: selected threshold and cooldown changed across windows")
    elif threshold_stability.lower() != "stable":
        stability_flag = "ThresholdUnstable"
        warnings.append("SplitUnstable: selected threshold changed across windows")
    elif cooldown_stability.lower() != "stable":
        stability_flag = "CooldownUnstable"
        warnings.append("SplitUnstable: selected cooldown changed across windows")
    else:
        stability_flag = "Stable"

    if benchmark_risk:
        warnings.append("BenchmarkRisk: walk-forward edge versus buy-and-hold is weak or inconsistent")
    if cost_fragile:
        warnings.append("CostFragile: edge is small enough that costs/slippage may erase it")
    if drawdown_risk:
        warnings.append("DrawdownRisk: walk-forward drawdown is large")
    if failure_reason:
        warnings.append(f"Phase7GFailureReason: {failure_reason}")

    return {
        "Asset": row.get("Asset", ""),
        "Horizon": _safe_int(row.get("Horizon"), default=0),
        "NumberOfWindows": number_of_windows,
        "BeatBuyHoldRate_%": round(float(beat_rate), 4),
        "PositiveReturnRate_%": round(float(positive_rate), 4),
        "AvgLockedVsBuyHold_%": round(float(avg_vs), 4),
        "MedianLockedVsBuyHold_%": round(float(median_vs), 4),
        "WorstLockedVsBuyHold_%": round(float(worst_vs), 4),
        "WorstLockedMaxDrawdown_%": round(float(worst_dd), 4),
        "AvgTradesPerWindow": round(float(avg_trades), 4),
        "WalkForwardVerdict": verdict,
        "SignalReliabilityScore": round(float(reliability_score), 2),
        "WalkForwardReliabilityScore": round(float(walk_forward_score), 2),
        "BenchmarkRiskFlag": benchmark_risk,
        "CostFragilityFlag": cost_fragile,
        "DrawdownRiskFlag": drawdown_risk,
        "StabilityFlag": stability_flag,
        "MissingPhase7GColumns": missing_cols,
        "Warnings": warnings,
        "FailureReason": failure_reason,
    }


def compute_meta_confidence_score(
    reliability_profile: Dict[str, Any],
    regime_profile: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute a transparent confidence score from reliability plus regime fit."""
    regime_profile = regime_profile or {}
    score = _safe_float(reliability_profile.get("SignalReliabilityScore"), default=0.0)
    regime_label = str(regime_profile.get("RegimeLabel", ""))

    if regime_label == "Constructive uptrend":
        score += 6.0
    elif regime_label == "Constructive but volatile":
        score += 1.0
    elif regime_label in {"High-volatility drawdown", "Risk-off / weak trend"}:
        score -= 12.0
    elif regime_label == "Insufficient regime data":
        score -= 8.0

    if reliability_profile.get("BenchmarkRiskFlag"):
        score -= 10.0
    if reliability_profile.get("CostFragilityFlag"):
        score -= 5.0
    if reliability_profile.get("DrawdownRiskFlag"):
        score -= 12.0
    if str(reliability_profile.get("StabilityFlag", "")).lower() != "stable":
        score -= 8.0

    return round(_clip(score), 2)


def compute_meta_risk_score(
    reliability_profile: Dict[str, Any],
    regime_profile: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute a higher-is-worse meta risk score."""
    regime_profile = regime_profile or {}
    reliability = _safe_float(reliability_profile.get("SignalReliabilityScore"), default=0.0)
    regime_risk = _safe_float(regime_profile.get("RegimeRiskScore"), default=50.0)
    risk = (100.0 - reliability) * 0.45 + regime_risk * 0.35

    if reliability_profile.get("BenchmarkRiskFlag"):
        risk += 12.0
    if reliability_profile.get("CostFragilityFlag"):
        risk += 8.0
    if reliability_profile.get("DrawdownRiskFlag"):
        risk += 15.0
    stability = str(reliability_profile.get("StabilityFlag", "Unknown"))
    if stability != "Stable":
        risk += 10.0
    if stability == "MissingPhase7G":
        risk += 20.0

    return round(_clip(risk), 2)


def meta_signal_decision(
    reliability_profile: Dict[str, Any],
    regime_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Make a conservative rule-based meta decision with an explanation."""
    regime_profile = regime_profile or {}
    confidence = compute_meta_confidence_score(reliability_profile, regime_profile)
    risk = compute_meta_risk_score(reliability_profile, regime_profile)
    reliability = _safe_float(reliability_profile.get("SignalReliabilityScore"), default=0.0)
    beat_rate = _safe_float(reliability_profile.get("BeatBuyHoldRate_%"), default=0.0)
    positive_rate = _safe_float(reliability_profile.get("PositiveReturnRate_%"), default=0.0)
    avg_vs = _safe_float(reliability_profile.get("AvgLockedVsBuyHold_%"), default=0.0)
    median_vs = _safe_float(reliability_profile.get("MedianLockedVsBuyHold_%"), default=0.0)
    avg_trades = _safe_float(reliability_profile.get("AvgTradesPerWindow"), default=0.0)
    verdict = str(reliability_profile.get("WalkForwardVerdict", ""))
    stability = str(reliability_profile.get("StabilityFlag", "Unknown"))
    regime_label = str(regime_profile.get("RegimeLabel", "Unknown"))

    warnings = list(reliability_profile.get("Warnings", [])) + list(regime_profile.get("RegimeWarnings", []))
    hard_fail = "Do not trust" in verdict or stability == "MissingPhase7G" or reliability < 20.0
    severe_risk = risk >= 75.0 or (reliability_profile.get("DrawdownRiskFlag") and reliability_profile.get("BenchmarkRiskFlag"))
    stable = stability == "Stable"
    benchmark_ok = not reliability_profile.get("BenchmarkRiskFlag") and avg_vs > 0.0 and median_vs > 0.0
    regime_blocks_trade = regime_label in {"High-volatility drawdown", "Risk-off / weak trend", "Insufficient regime data"}

    if hard_fail:
        decision = "Avoid"
        reason = "Phase 7G reliability is missing, too weak, or explicitly Do Not Trust"
        suggested = "Do not use for signals; keep as rejection/regime evidence."
    elif confidence >= 72.0 and risk <= 35.0 and stable and benchmark_ok and beat_rate >= 60.0 and positive_rate >= 60.0 and avg_trades >= 3.0 and not regime_blocks_trade:
        decision = "Trade"
        reason = "Walk-forward reliability, benchmark edge, stability, and current regime are aligned"
        suggested = "Trader-assistance research watchlist only; validate live/paper behavior before any capital use."
    elif reliability >= 58.0 and benchmark_ok and regime_blocks_trade:
        decision = "Defensive Only"
        reason = "Signal reliability is usable as evidence, but the current regime is too risky for an active trade label"
        suggested = "Use as defensive/risk-awareness evidence, not as an entry signal."
    elif reliability >= 45.0 and not severe_risk and ("Research candidate" in verdict or "Strong" in verdict):
        decision = "Research Only"
        reason = "Candidate has research evidence but does not clear all risk, stability, or regime gates"
        suggested = "Keep in research review; require more walk-forward and cost robustness."
    elif reliability >= 30.0 and not severe_risk:
        decision = "No Trade"
        reason = "Evidence is not strong enough for active use, but it is not a full rejection"
        suggested = "Stand aside; monitor regime and future walk-forward updates."
    else:
        decision = "Avoid"
        reason = "Risk, benchmark weakness, or instability overwhelms the signal evidence"
        suggested = "Do not use for signals; keep as failure evidence for future meta filters."

    return {
        "MetaDecision": decision,
        "MetaConfidenceScore": confidence,
        "MetaRiskScore": risk,
        "MainReason": reason,
        "Warnings": _join_warnings(warnings),
        "SuggestedUseCase": suggested,
    }


def _normalize_walk_forward_summary(walk_forward_summary: Optional[pd.DataFrame]) -> pd.DataFrame:
    if walk_forward_summary is None:
        return pd.DataFrame(columns=["Asset", "Horizon"])
    if isinstance(walk_forward_summary, pd.DataFrame):
        out = walk_forward_summary.copy()
    else:
        out = pd.DataFrame(walk_forward_summary)
    if "Asset" not in out.columns:
        out["Asset"] = ""
    if "Horizon" not in out.columns:
        out["Horizon"] = 0
    out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce").fillna(0).astype(int)
    return out


def run_regime_aware_meta_signal(
    *,
    raw_df: Optional[pd.DataFrame],
    walk_forward_summary: Optional[pd.DataFrame],
    asset_names: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    model_depth: str = "core",
    use_phase5_features: bool = True,
    signal_mode: str = "long_only",
) -> MetaSignalReport:
    """Run the Phase 8 rule-based meta signal layer."""
    wf = _normalize_walk_forward_summary(walk_forward_summary)
    assets = list(asset_names) if asset_names is not None else [a for a in wf["Asset"].dropna().unique() if str(a)]
    if not assets:
        assets = list(get_asset_names())

    scan_horizons = [int(h) for h in horizons] if horizons is not None else sorted([int(h) for h in wf["Horizon"].dropna().unique() if int(h) > 0])
    if not scan_horizons:
        scan_horizons = list(DEFAULT_HORIZONS)

    regime_features = build_regime_features(raw_df, assets)
    regime_by_asset = {str(row["Asset"]): row for _, row in regime_features.iterrows()} if not regime_features.empty else {}

    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for asset in assets:
        regime_row = regime_by_asset.get(asset, {"Asset": asset, "RegimeDataWarning": "Missing regime feature row"})
        regime_profile = classify_market_regime(regime_row)
        for horizon in scan_horizons:
            matches = wf[(wf["Asset"].astype(str) == str(asset)) & (wf["Horizon"].astype(int) == int(horizon))]
            if matches.empty:
                wf_row: Dict[str, Any] = {
                    "Asset": asset,
                    "Horizon": int(horizon),
                    "MissingPhase7GRow": True,
                    "FailureReason": "No Phase 7G walk-forward aggregate row for this asset-horizon",
                }
            else:
                wf_row = matches.iloc[0].to_dict()

            reliability = build_signal_reliability_profile(wf_row)
            decision = meta_signal_decision(reliability, regime_profile)
            out = {
                "Asset": asset,
                "Horizon": int(horizon),
                "MetaDecision": decision["MetaDecision"],
                "MetaConfidenceScore": decision["MetaConfidenceScore"],
                "MetaRiskScore": decision["MetaRiskScore"],
                "RegimeLabel": regime_profile["RegimeLabel"],
                "SignalReliabilityScore": reliability["SignalReliabilityScore"],
                "WalkForwardReliabilityScore": reliability["WalkForwardReliabilityScore"],
                "BenchmarkRiskFlag": bool(reliability["BenchmarkRiskFlag"]),
                "CostFragilityFlag": bool(reliability["CostFragilityFlag"]),
                "DrawdownRiskFlag": bool(reliability["DrawdownRiskFlag"]),
                "StabilityFlag": reliability["StabilityFlag"],
                "MainReason": decision["MainReason"],
                "Warnings": decision["Warnings"],
                "SuggestedUseCase": decision["SuggestedUseCase"],
                "WalkForwardVerdict": reliability["WalkForwardVerdict"],
                "BeatBuyHoldRate_%": reliability["BeatBuyHoldRate_%"],
                "PositiveReturnRate_%": reliability["PositiveReturnRate_%"],
                "AvgLockedVsBuyHold_%": reliability["AvgLockedVsBuyHold_%"],
                "MedianLockedVsBuyHold_%": reliability["MedianLockedVsBuyHold_%"],
                "WorstLockedVsBuyHold_%": reliability["WorstLockedVsBuyHold_%"],
                "WorstLockedMaxDrawdown_%": reliability["WorstLockedMaxDrawdown_%"],
                "AvgTradesPerWindow": reliability["AvgTradesPerWindow"],
                "RegimeRiskScore": regime_profile["RegimeRiskScore"],
                "RegimeReason": regime_profile["RegimeReason"],
            }
            rows.append(out)
            if decision["Warnings"]:
                warnings.append(f"{asset} {int(horizon)}D: {decision['Warnings']}")

    decision_table = pd.DataFrame(rows)
    if decision_table.empty:
        decision_table = pd.DataFrame(columns=META_SIGNAL_COLUMNS)
    else:
        for col in META_SIGNAL_COLUMNS:
            if col not in decision_table.columns:
                decision_table[col] = np.nan
        leading = META_SIGNAL_COLUMNS
        extra = [c for c in decision_table.columns if c not in leading]
        decision_table = decision_table[leading + extra]

    settings = {
        "assets": assets,
        "horizons": scan_horizons,
        "model_depth": str(model_depth),
        "use_phase5_features": bool(use_phase5_features),
        "signal_mode": str(signal_mode),
        "reliability_source": "Phase 7G walk-forward aggregate",
        "meta_model": "rule_based_no_training",
    }
    return MetaSignalReport(
        decision_table=decision_table,
        decision_summary=summarize_meta_signal_results(decision_table),
        regime_features=regime_features,
        warnings=list(dict.fromkeys([w for w in warnings if w])),
        settings=settings,
    )


def summarize_meta_signal_results(meta_results: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Summarize meta decisions without hiding failed/rejected rows."""
    columns = ["MetaDecision", "Count", "AvgMetaConfidenceScore", "AvgMetaRiskScore"]
    if meta_results is None or meta_results.empty or "MetaDecision" not in meta_results.columns:
        return pd.DataFrame(
            [{"MetaDecision": decision, "Count": 0, "AvgMetaConfidenceScore": np.nan, "AvgMetaRiskScore": np.nan} for decision in META_DECISIONS],
            columns=columns,
        )

    rows: List[Dict[str, Any]] = []
    for decision in META_DECISIONS:
        group = meta_results[meta_results["MetaDecision"].astype(str) == decision]
        rows.append(
            {
                "MetaDecision": decision,
                "Count": int(len(group)),
                "AvgMetaConfidenceScore": round(float(group["MetaConfidenceScore"].astype(float).mean()), 2) if not group.empty else np.nan,
                "AvgMetaRiskScore": round(float(group["MetaRiskScore"].astype(float).mean()), 2) if not group.empty else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


AUDIT_COLUMNS = [
    "Asset",
    "Horizon",
    "Current MetaDecision",
    "Calibrated MetaDecision",
    "MetaConfidenceScore",
    "MetaRiskScore",
    "WalkForwardReliabilityScore",
    "RegimeLabel",
    "BlockingRules",
    "PassingRules",
    "MainBlockingRule",
    "RuleAuditExplanation",
    "WhatWouldNeedToImprove",
    "SuggestedNextResearchAction",
]

CALIBRATION_MODES = ["Conservative", "Balanced", "Aggressive Research"]


def _safe_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    try:
        return bool(value)
    except Exception:
        return False


def _row_value(row: Dict[str, Any], column: str, default: Any = np.nan) -> Any:
    value = row.get(column, default)
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return value


def calibrate_meta_decision_thresholds(mode: str = "Conservative") -> Dict[str, Any]:
    """
    Return visible audit thresholds for a calibration mode.

    These thresholds are for explanation and research calibration only. They do
    not retrain models, retune Phase 7 thresholds, or change locked-test facts.
    Trade gates remain strict in every mode.
    """
    normalized = str(mode or "Conservative").strip()
    if normalized not in CALIBRATION_MODES:
        normalized = "Conservative"

    base = {
        "Mode": normalized,
        "MinimumConfidenceForTrade": 72.0,
        "MaximumRiskForTrade": 35.0,
        "MinimumBeatBuyHoldRate_%": 60.0,
        "MinimumMedianVsBuyHold_%": 0.0,
        "MaximumAllowedDrawdown_%": -25.0,
        "MinimumTradeReliability": 58.0,
        "MinimumAvgTradesPerWindow": 3.0,
        "RequireStableThresholdCooldown": True,
        "BlockTradeInRiskOffRegime": True,
        "AllowTradeWithBenchmarkRisk": False,
        "AllowTradeWithDrawdownRisk": False,
        "ResearchOnlyMinimumReliability": 45.0,
        "DefensiveOnlyMinimumReliability": 58.0,
        "NoTradeMinimumReliability": 30.0,
        "ModeLabel": "Strict audit of current Phase 8 behavior",
        "ProductionReadyLabelAllowed": False,
    }

    if normalized == "Balanced":
        base.update(
            {
                "ResearchOnlyMinimumReliability": 38.0,
                "DefensiveOnlyMinimumReliability": 52.0,
                "NoTradeMinimumReliability": 25.0,
                "ModeLabel": "Research calibration; Trade gates remain strict",
            }
        )
    elif normalized == "Aggressive Research":
        base.update(
            {
                "ResearchOnlyMinimumReliability": 30.0,
                "DefensiveOnlyMinimumReliability": 48.0,
                "NoTradeMinimumReliability": 20.0,
                "ModeLabel": "Experimental research surfacing; never production-ready",
            }
        )

    return base


def _threshold_config_table() -> pd.DataFrame:
    return pd.DataFrame([calibrate_meta_decision_thresholds(mode) for mode in CALIBRATION_MODES])


def _missing_audit_columns(row: Dict[str, Any]) -> List[str]:
    required = [
        "MetaDecision",
        "MetaConfidenceScore",
        "MetaRiskScore",
        "WalkForwardReliabilityScore",
        "RegimeLabel",
        "BenchmarkRiskFlag",
        "DrawdownRiskFlag",
        "StabilityFlag",
    ]
    return [col for col in required if col not in row]


def _trade_rule_status(row: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    confidence = _safe_float(_row_value(row, "MetaConfidenceScore", 0.0), default=0.0)
    risk = _safe_float(_row_value(row, "MetaRiskScore", 100.0), default=100.0)
    reliability = _safe_float(_row_value(row, "WalkForwardReliabilityScore", _row_value(row, "SignalReliabilityScore", 0.0)), default=0.0)
    beat_rate = _safe_float(_row_value(row, "BeatBuyHoldRate_%", 0.0), default=0.0)
    avg_vs = _safe_float(_row_value(row, "AvgLockedVsBuyHold_%", 0.0), default=0.0)
    median_vs = _safe_float(_row_value(row, "MedianLockedVsBuyHold_%", 0.0), default=0.0)
    worst_dd = _safe_float(_row_value(row, "WorstLockedMaxDrawdown_%", 0.0), default=0.0)
    avg_sharpe = _safe_float(_row_value(row, "AvgLockedSharpe", 0.0), default=0.0)
    avg_trades = _safe_float(_row_value(row, "AvgTradesPerWindow", 0.0), default=0.0)
    threshold_stability = str(_row_value(row, "ThresholdStability", "") or "")
    cooldown_stability = str(_row_value(row, "CooldownStability", "") or "")
    stability = str(_row_value(row, "StabilityFlag", "Unknown") or "Unknown")
    regime = str(_row_value(row, "RegimeLabel", "Unknown") or "Unknown")
    verdict = str(_row_value(row, "WalkForwardVerdict", "") or "")
    benchmark_risk = _safe_bool(_row_value(row, "BenchmarkRiskFlag", False))
    drawdown_risk = _safe_bool(_row_value(row, "DrawdownRiskFlag", False))
    cost_fragile = _safe_bool(_row_value(row, "CostFragilityFlag", False))
    missing_cols = _missing_audit_columns(row)

    risky_regimes = {"High-volatility drawdown", "Risk-off / weak trend", "Insufficient regime data"}
    checks = [
        (
            "Phase 7G row present",
            not missing_cols and "Missing" not in stability,
            f"Missing columns: {', '.join(missing_cols)}" if missing_cols else "Phase 7G/meta columns present",
        ),
        (
            "Walk-forward verdict is not Do Not Trust",
            "do not trust" not in verdict.lower(),
            f"Walk-forward verdict is {verdict or 'missing'}",
        ),
        (
            f"Confidence >= {thresholds['MinimumConfidenceForTrade']:.1f}",
            confidence >= float(thresholds["MinimumConfidenceForTrade"]),
            f"Current confidence is {confidence:.2f}",
        ),
        (
            f"Risk <= {thresholds['MaximumRiskForTrade']:.1f}",
            risk <= float(thresholds["MaximumRiskForTrade"]),
            f"Current risk is {risk:.2f}",
        ),
        (
            f"Walk-forward reliability >= {thresholds['MinimumTradeReliability']:.1f}",
            reliability >= float(thresholds["MinimumTradeReliability"]),
            f"Current walk-forward reliability is {reliability:.2f}",
        ),
        (
            f"Beat buy-and-hold rate >= {thresholds['MinimumBeatBuyHoldRate_%']:.1f}%",
            beat_rate >= float(thresholds["MinimumBeatBuyHoldRate_%"]),
            f"Current beat-buy-hold rate is {beat_rate:.2f}%",
        ),
        (
            f"Median vs buy-and-hold > {thresholds['MinimumMedianVsBuyHold_%']:.1f}%",
            median_vs > float(thresholds["MinimumMedianVsBuyHold_%"]),
            f"Current median vs buy-and-hold is {median_vs:.2f}%",
        ),
        (
            f"Worst drawdown >= {thresholds['MaximumAllowedDrawdown_%']:.1f}%",
            worst_dd >= float(thresholds["MaximumAllowedDrawdown_%"]),
            f"Current worst drawdown is {worst_dd:.2f}%",
        ),
        (
            f"Average trades/window >= {thresholds['MinimumAvgTradesPerWindow']:.1f}",
            avg_trades >= float(thresholds["MinimumAvgTradesPerWindow"]),
            f"Current average trades/window is {avg_trades:.2f}",
        ),
        (
            "Threshold/cooldown stability is Stable",
            stability == "Stable" or not bool(thresholds["RequireStableThresholdCooldown"]),
            f"Current stability flag is {stability}",
        ),
        (
            "Benchmark risk flag is clear",
            not benchmark_risk or bool(thresholds["AllowTradeWithBenchmarkRisk"]),
            "Benchmark risk flag is active" if benchmark_risk else "Benchmark risk flag is clear",
        ),
        (
            "Drawdown risk flag is clear",
            not drawdown_risk or bool(thresholds["AllowTradeWithDrawdownRisk"]),
            "Drawdown risk flag is active" if drawdown_risk else "Drawdown risk flag is clear",
        ),
        (
            "Regime does not block Trade",
            regime not in risky_regimes or not bool(thresholds["BlockTradeInRiskOffRegime"]),
            f"Current regime is {regime}",
        ),
        (
            "Cost fragility is not active",
            not cost_fragile,
            "Cost fragility flag is active" if cost_fragile else "Cost fragility flag is clear",
        ),
    ]

    passing = [name for name, passed, _detail in checks if passed]
    blocking = [f"{name} ({detail})" for name, passed, detail in checks if not passed]
    return {
        "confidence": confidence,
        "risk": risk,
        "reliability": reliability,
        "beat_rate": beat_rate,
        "avg_vs": avg_vs,
        "median_vs": median_vs,
        "worst_dd": worst_dd,
        "avg_sharpe": avg_sharpe,
        "avg_trades": avg_trades,
        "threshold_stability": threshold_stability,
        "cooldown_stability": cooldown_stability,
        "stability": stability,
        "regime": regime,
        "verdict": verdict,
        "benchmark_risk": benchmark_risk,
        "drawdown_risk": drawdown_risk,
        "cost_fragile": cost_fragile,
        "missing_cols": missing_cols,
        "passing": passing,
        "blocking": blocking,
        "passed_count": len(passing),
        "total_count": len(checks),
        "strict_trade_pass": len(blocking) == 0,
    }


def _calibrated_decision(current_decision: str, status: Dict[str, Any], thresholds: Dict[str, Any]) -> str:
    mode = str(thresholds["Mode"])
    verdict = str(status["verdict"]).lower()
    reliability = float(status["reliability"])
    risk = float(status["risk"])
    benchmark_ok = not bool(status["benchmark_risk"]) and float(status["median_vs"]) > 0.0
    regime_blocks = status["regime"] in {"High-volatility drawdown", "Risk-off / weak trend", "Insufficient regime data"}
    hard_reject = "do not trust" in verdict or status["stability"] == "MissingPhase7G" or reliability < 20.0

    if status["strict_trade_pass"]:
        return "Trade"
    if mode == "Conservative":
        return current_decision or "Avoid"
    if hard_reject:
        return "Avoid"
    if reliability >= float(thresholds["DefensiveOnlyMinimumReliability"]) and benchmark_ok and regime_blocks and risk < 80.0:
        return "Defensive Only"
    if reliability >= float(thresholds["ResearchOnlyMinimumReliability"]) and risk < 85.0:
        return "Research Only"
    if reliability >= float(thresholds["NoTradeMinimumReliability"]) and risk < 90.0:
        return "No Trade"
    return "Avoid"


def _improvement_text(status: Dict[str, Any], thresholds: Dict[str, Any]) -> str:
    needs: List[str] = []
    if status["confidence"] < float(thresholds["MinimumConfidenceForTrade"]):
        needs.append("higher meta confidence from repeatable walk-forward evidence")
    if status["risk"] > float(thresholds["MaximumRiskForTrade"]):
        needs.append("lower meta risk from reduced drawdown, cost, benchmark, or regime risk")
    if status["beat_rate"] < float(thresholds["MinimumBeatBuyHoldRate_%"]):
        needs.append("more walk-forward windows beating buy-and-hold")
    if status["median_vs"] <= float(thresholds["MinimumMedianVsBuyHold_%"]):
        needs.append("positive median edge versus buy-and-hold")
    if status["worst_dd"] < float(thresholds["MaximumAllowedDrawdown_%"]):
        needs.append("controlled worst drawdown")
    if status["avg_trades"] < float(thresholds["MinimumAvgTradesPerWindow"]):
        needs.append("more trades per walk-forward window")
    if status["stability"] != "Stable":
        needs.append("stable selected threshold/cooldown across windows")
    if status["benchmark_risk"]:
        needs.append("clearer benchmark-relative edge")
    if status["drawdown_risk"]:
        needs.append("lower drawdown risk")
    if status["cost_fragile"]:
        needs.append("edge that survives realistic transaction costs")
    if status["regime"] in {"High-volatility drawdown", "Risk-off / weak trend", "Insufficient regime data"}:
        needs.append("a less hostile or better-understood current regime")
    if "do not trust" in str(status["verdict"]).lower():
        needs.append("Phase 7G verdict must improve beyond Do Not Trust")
    if status["missing_cols"]:
        needs.append("complete Phase 8/Phase 7G columns for audit")
    if not needs:
        return "No major rule blockers; review execution, liquidity, and live/paper evidence before any escalation."
    return "; ".join(dict.fromkeys(needs))


def _next_research_action(calibrated_decision: str, status: Dict[str, Any], mode: str) -> str:
    if calibrated_decision == "Trade":
        return "Strict gates pass; keep as research watchlist only and require live/paper tracking before capital use."
    if "do not trust" in str(status["verdict"]).lower():
        return "Do not use as a signal; inspect feature/model reliability and keep as rejection evidence."
    if status["benchmark_risk"]:
        return "Study benchmark dependency and require repeatable edge over buy-and-hold."
    if status["drawdown_risk"]:
        return "Investigate risk controls and drawdown containment before considering escalation."
    if status["stability"] != "Stable":
        return "Rerun walk-forward stability analysis and inspect threshold/cooldown drift."
    if mode == "Aggressive Research" and calibrated_decision == "Research Only":
        return "Experimental research only; paper-track and do not treat as production-ready."
    if calibrated_decision == "Defensive Only":
        return "Use as risk-awareness evidence; avoid treating it as an entry signal."
    return "Monitor future walk-forward updates and keep all weak evidence visible."


def explain_meta_decision_rules(meta_row: Any, calibration_mode: str = "Conservative") -> Dict[str, Any]:
    """Explain which rules passed or blocked one Phase 8 meta decision."""
    row = meta_row.to_dict() if isinstance(meta_row, pd.Series) else dict(meta_row or {})
    thresholds = calibrate_meta_decision_thresholds(calibration_mode)
    status = _trade_rule_status(row, thresholds)
    current_decision = str(_row_value(row, "MetaDecision", _row_value(row, "Current MetaDecision", "Avoid")) or "Avoid")
    calibrated = _calibrated_decision(current_decision, status, thresholds)
    main_block = status["blocking"][0] if status["blocking"] else "None"
    passing = status["passing"]
    blocking = status["blocking"]
    near_miss_score = round(float(status["passed_count"]) / max(float(status["total_count"]), 1.0) * 100.0, 2)

    explanation = (
        f"{current_decision} audited in {thresholds['Mode']} mode. "
        f"{status['passed_count']} of {status['total_count']} strict Trade rules pass. "
        f"Calibrated decision is {calibrated}; this is an audit label, not a production recommendation."
    )
    if thresholds["Mode"] == "Aggressive Research" and calibrated == "Research Only":
        explanation += " Aggressive Research surfaces this only as experimental research."

    what_if_parts = []
    if calibrated not in {"Trade", "Research Only"} and status["reliability"] >= float(thresholds["ResearchOnlyMinimumReliability"]) - 5.0:
        what_if_parts.append("Could become Research Only if reliability/risk evidence improves modestly")
    if calibrated not in {"Trade", "Defensive Only"} and status["benchmark_risk"] and status["reliability"] >= float(thresholds["DefensiveOnlyMinimumReliability"]) - 5.0:
        what_if_parts.append("Could become Defensive Only if benchmark weakness is resolved and regime risk remains high")
    if not status["strict_trade_pass"]:
        what_if_parts.append(f"Trade requires: {_improvement_text(status, thresholds)}")

    warnings = str(_row_value(row, "Warnings", "") or "")
    if thresholds["Mode"] == "Aggressive Research":
        warnings = _join_warnings([warnings, "AggressiveResearchExperimental: research-only calibration, not production-ready"])

    return {
        "Asset": str(_row_value(row, "Asset", "")),
        "Horizon": _safe_int(_row_value(row, "Horizon", 0), default=0),
        "Current MetaDecision": current_decision,
        "Calibrated MetaDecision": calibrated,
        "MetaConfidenceScore": round(float(status["confidence"]), 2),
        "MetaRiskScore": round(float(status["risk"]), 2),
        "WalkForwardReliabilityScore": round(float(status["reliability"]), 2),
        "BeatBuyHoldRate_%": round(float(status["beat_rate"]), 4),
        "MedianLockedVsBuyHold_%": round(float(status["median_vs"]), 4),
        "AvgLockedVsBuyHold_%": round(float(status["avg_vs"]), 4),
        "WorstLockedMaxDrawdown_%": round(float(status["worst_dd"]), 4),
        "AvgLockedSharpe": round(float(status["avg_sharpe"]), 4),
        "AvgTradesPerWindow": round(float(status["avg_trades"]), 4),
        "ThresholdStability": status["threshold_stability"],
        "CooldownStability": status["cooldown_stability"],
        "RegimeLabel": status["regime"],
        "BenchmarkRiskFlag": bool(status["benchmark_risk"]),
        "CostFragilityFlag": bool(status["cost_fragile"]),
        "DrawdownRiskFlag": bool(status["drawdown_risk"]),
        "StabilityFlag": status["stability"],
        "WalkForwardVerdict": status["verdict"],
        "BlockingRules": _join_warnings(blocking),
        "PassingRules": _join_warnings(passing),
        "MainBlockingRule": main_block,
        "RuleAuditExplanation": explanation,
        "WhatWouldNeedToImprove": _join_warnings(what_if_parts) or "Strict Trade gates already pass.",
        "SuggestedNextResearchAction": _next_research_action(calibrated, status, str(thresholds["Mode"])),
        "Warnings": warnings,
        "NearMissScore": near_miss_score,
        "StrictTradeRulesPassed": int(status["passed_count"]),
        "StrictTradeRulesTotal": int(status["total_count"]),
        "Mode": str(thresholds["Mode"]),
        "ModeLabel": str(thresholds["ModeLabel"]),
    }


def build_meta_decision_audit(meta_results: Optional[pd.DataFrame], calibration_mode: str = "Conservative") -> pd.DataFrame:
    """Build row-level audit output while keeping all candidates visible."""
    if meta_results is None or meta_results.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    rows = [explain_meta_decision_rules(row, calibration_mode) for _, row in meta_results.iterrows()]
    out = pd.DataFrame(rows)
    for col in AUDIT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    leading = AUDIT_COLUMNS
    extra = [c for c in out.columns if c not in leading]
    return out[leading + extra]


def summarize_meta_audit(audit_table: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Create summary tables for a meta decision audit."""
    if audit_table is None or audit_table.empty:
        empty_counts = pd.DataFrame(columns=["Calibrated MetaDecision", "Count"])
        empty_rules = pd.DataFrame(columns=["BlockingRule", "Count"])
        return {
            "decision_counts": empty_counts,
            "common_blocking_rules": empty_rules,
            "near_miss_candidates": pd.DataFrame(columns=AUDIT_COLUMNS),
            "top_blocked_candidates": pd.DataFrame(columns=AUDIT_COLUMNS),
            "highest_confidence_candidates": pd.DataFrame(columns=AUDIT_COLUMNS),
            "highest_risk_candidates": pd.DataFrame(columns=AUDIT_COLUMNS),
        }

    counts = (
        audit_table["Calibrated MetaDecision"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Calibrated MetaDecision")
        .reset_index(name="Count")
    )
    blocking_parts: List[str] = []
    for value in audit_table["BlockingRules"].fillna(""):
        blocking_parts.extend([part.strip() for part in str(value).split(";") if part.strip()])
    common = (
        pd.Series(blocking_parts, dtype="object")
        .value_counts()
        .rename_axis("BlockingRule")
        .reset_index(name="Count")
        if blocking_parts
        else pd.DataFrame(columns=["BlockingRule", "Count"])
    )
    near_miss = audit_table[audit_table["Calibrated MetaDecision"].ne("Trade")].copy()
    if not near_miss.empty and "NearMissScore" in near_miss.columns:
        near_miss = near_miss.sort_values(
            ["NearMissScore", "MetaConfidenceScore", "WalkForwardReliabilityScore"],
            ascending=[False, False, False],
        )
    top_blocked = audit_table[audit_table["BlockingRules"].fillna("").astype(str).ne("")].copy()
    if not top_blocked.empty:
        top_blocked["BlockingRuleCount"] = top_blocked["BlockingRules"].astype(str).apply(
            lambda value: len([part for part in value.split(";") if part.strip()])
        )
        top_blocked = top_blocked.sort_values(
            ["BlockingRuleCount", "MetaRiskScore", "NearMissScore"],
            ascending=[False, False, True],
        )
    highest_confidence = audit_table.sort_values(
        ["MetaConfidenceScore", "WalkForwardReliabilityScore"],
        ascending=[False, False],
    ).copy()
    highest_risk = audit_table.sort_values(
        ["MetaRiskScore", "WalkForwardReliabilityScore"],
        ascending=[False, True],
    ).copy()
    return {
        "decision_counts": counts,
        "common_blocking_rules": common,
        "near_miss_candidates": near_miss,
        "top_blocked_candidates": top_blocked,
        "highest_confidence_candidates": highest_confidence,
        "highest_risk_candidates": highest_risk,
    }


def run_meta_decision_audit(
    meta_results: Optional[pd.DataFrame],
    calibration_mode: str = "Conservative",
) -> MetaDecisionAuditReport:
    """Run Phase 8A audit/calibration over an existing Phase 8 decision table."""
    mode = calibration_mode if calibration_mode in CALIBRATION_MODES else "Conservative"
    audit = build_meta_decision_audit(meta_results, mode)
    summaries = summarize_meta_audit(audit)

    mode_rows: List[Dict[str, Any]] = []
    for candidate_mode in CALIBRATION_MODES:
        candidate_audit = build_meta_decision_audit(meta_results, candidate_mode)
        counts = candidate_audit["Calibrated MetaDecision"].fillna("Unknown").value_counts().to_dict() if not candidate_audit.empty else {}
        mode_rows.append(
            {
                "Mode": candidate_mode,
                "Trade": int(counts.get("Trade", 0)),
                "No Trade": int(counts.get("No Trade", 0)),
                "Defensive Only": int(counts.get("Defensive Only", 0)),
                "Research Only": int(counts.get("Research Only", 0)),
                "Avoid": int(counts.get("Avoid", 0)),
                "Total": int(len(candidate_audit)),
                "ModeLabel": calibrate_meta_decision_thresholds(candidate_mode)["ModeLabel"],
            }
        )

    warnings: List[str] = []
    if meta_results is None or meta_results.empty:
        warnings.append("No Phase 8 meta decision rows were provided.")
    elif "MetaDecision" not in meta_results.columns:
        warnings.append("Missing MetaDecision column; audit used conservative defaults.")
    if mode == "Aggressive Research":
        warnings.append("Aggressive Research is experimental research-only calibration and is not production-ready.")

    return MetaDecisionAuditReport(
        audit_table=audit,
        threshold_config=_threshold_config_table(),
        mode_comparison=pd.DataFrame(mode_rows),
        common_blocking_rules=summaries["common_blocking_rules"],
        near_miss_candidates=summaries["near_miss_candidates"],
        top_blocked_candidates=summaries["top_blocked_candidates"],
        highest_confidence_candidates=summaries["highest_confidence_candidates"],
        highest_risk_candidates=summaries["highest_risk_candidates"],
        warnings=warnings,
        settings={
            "calibration_mode": mode,
            "phase": "8A",
            "purpose": "audit_and_calibration_only",
            "retunes_models_or_thresholds": False,
            "production_ready_label_allowed": False,
        },
    )


RELIABILITY_GRADING_COLUMNS = [
    "Asset",
    "Horizon",
    "MetaDecision",
    "CalibratedMetaDecision",
    "ReliabilityGrade",
    "ReliabilityScore_0_100",
    "PromotionReadiness",
    "ResearchPriority",
    "NextBestAction",
    "MetaConfidenceScore",
    "MetaRiskScore",
    "WalkForwardReliabilityScore",
    "BeatBuyHoldRate_%",
    "MedianLockedVsBuyHold_%",
    "AvgLockedVsBuyHold_%",
    "WorstLockedMaxDrawdown_%",
    "ThresholdStability",
    "CooldownStability",
    "RegimeLabel",
    "BenchmarkRiskFlag",
    "CostFragilityFlag",
    "DrawdownRiskFlag",
    "StabilityFlag",
    "MainBlockingRule",
    "GradeExplanation",
]

RELIABILITY_SCORE_COMPONENTS = [
    "WalkForwardReliabilityComponent",
    "BeatBuyHoldComponent",
    "MedianVsBuyHoldComponent",
    "AvgVsBuyHoldComponent",
    "SharpeComponent",
    "TradeCountComponent",
    "StabilityComponent",
    "RegimeComponent",
    "DrawdownPenalty",
    "BenchmarkPenalty",
    "CostFragilityPenalty",
    "RiskPenalty",
    "MissingDataPenalty",
]


def _normalized_grading_row(row: Any) -> Dict[str, Any]:
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row or {})
    current_decision = str(
        _row_value(data, "MetaDecision", _row_value(data, "Current MetaDecision", "Avoid")) or "Avoid"
    )
    calibrated = str(
        _row_value(data, "CalibratedMetaDecision", _row_value(data, "Calibrated MetaDecision", current_decision))
        or current_decision
    )
    return {
        **data,
        "Asset": str(_row_value(data, "Asset", "")),
        "Horizon": _safe_int(_row_value(data, "Horizon", 0), default=0),
        "MetaDecision": current_decision,
        "CalibratedMetaDecision": calibrated,
        "MetaConfidenceScore": _safe_float(_row_value(data, "MetaConfidenceScore", 0.0), default=0.0),
        "MetaRiskScore": _safe_float(_row_value(data, "MetaRiskScore", 75.0), default=75.0),
        "WalkForwardReliabilityScore": _safe_float(
            _row_value(data, "WalkForwardReliabilityScore", _row_value(data, "SignalReliabilityScore", 0.0)),
            default=0.0,
        ),
        "BeatBuyHoldRate_%": _safe_float(_row_value(data, "BeatBuyHoldRate_%", 0.0), default=0.0),
        "MedianLockedVsBuyHold_%": _safe_float(_row_value(data, "MedianLockedVsBuyHold_%", 0.0), default=0.0),
        "AvgLockedVsBuyHold_%": _safe_float(_row_value(data, "AvgLockedVsBuyHold_%", 0.0), default=0.0),
        "WorstLockedMaxDrawdown_%": _safe_float(_row_value(data, "WorstLockedMaxDrawdown_%", 0.0), default=0.0),
        "AvgLockedSharpe": _safe_float(_row_value(data, "AvgLockedSharpe", 0.0), default=0.0),
        "AvgTradesPerWindow": _safe_float(_row_value(data, "AvgTradesPerWindow", 0.0), default=0.0),
        "ThresholdStability": str(_row_value(data, "ThresholdStability", "") or ""),
        "CooldownStability": str(_row_value(data, "CooldownStability", "") or ""),
        "RegimeLabel": str(_row_value(data, "RegimeLabel", "Unknown") or "Unknown"),
        "BenchmarkRiskFlag": _safe_bool(_row_value(data, "BenchmarkRiskFlag", False)),
        "CostFragilityFlag": _safe_bool(_row_value(data, "CostFragilityFlag", False)),
        "DrawdownRiskFlag": _safe_bool(_row_value(data, "DrawdownRiskFlag", False)),
        "StabilityFlag": str(_row_value(data, "StabilityFlag", "Unknown") or "Unknown"),
        "MainBlockingRule": str(_row_value(data, "MainBlockingRule", "") or ""),
        "WalkForwardVerdict": str(_row_value(data, "WalkForwardVerdict", "") or ""),
        "Warnings": str(_row_value(data, "Warnings", "") or ""),
    }


def compute_reliability_score(meta_row: Any, grading_mode: str = "Conservative") -> Dict[str, Any]:
    """Compute a transparent bounded reliability score from existing meta evidence."""
    row = _normalized_grading_row(meta_row)
    mode = grading_mode if grading_mode in CALIBRATION_MODES else "Conservative"

    wf_component = np.clip(row["WalkForwardReliabilityScore"], 0.0, 100.0) * 0.28
    beat_component = np.clip(row["BeatBuyHoldRate_%"], 0.0, 100.0) * 0.14
    median_component = np.clip(row["MedianLockedVsBuyHold_%"], -10.0, 15.0) * 1.0
    avg_component = np.clip(row["AvgLockedVsBuyHold_%"], -10.0, 15.0) * 0.75
    sharpe_component = np.clip(row["AvgLockedSharpe"], -1.5, 2.0) * 5.0
    trade_count_component = min(max(row["AvgTradesPerWindow"], 0.0) * 2.0, 10.0)

    threshold_stable = row["ThresholdStability"].lower() == "stable" or row["StabilityFlag"] == "Stable"
    cooldown_stable = row["CooldownStability"].lower() == "stable" or row["StabilityFlag"] == "Stable"
    stability_component = (7.0 if threshold_stable else -7.0) + (4.0 if cooldown_stable else -4.0)

    regime = row["RegimeLabel"]
    if regime == "Constructive uptrend":
        regime_component = 8.0
    elif regime == "Constructive but volatile":
        regime_component = 3.0
    elif regime in {"High-volatility drawdown", "Risk-off / weak trend"}:
        regime_component = -12.0
    elif regime == "Insufficient regime data":
        regime_component = -8.0
    else:
        regime_component = 0.0

    drawdown_penalty = max(0.0, abs(min(row["WorstLockedMaxDrawdown_%"], 0.0)) - 12.0) * 0.55
    if row["DrawdownRiskFlag"]:
        drawdown_penalty += 10.0
    benchmark_penalty = 15.0 if row["BenchmarkRiskFlag"] else 0.0
    cost_penalty = 8.0 if row["CostFragilityFlag"] else 0.0
    risk_penalty = max(0.0, row["MetaRiskScore"] - 45.0) * 0.28

    missing_cols = [
        col
        for col in [
            "MetaConfidenceScore",
            "MetaRiskScore",
            "WalkForwardReliabilityScore",
            "RegimeLabel",
            "StabilityFlag",
        ]
        if col not in (meta_row.index if isinstance(meta_row, pd.Series) else dict(meta_row or {}).keys())
    ]
    missing_penalty = min(20.0, len(missing_cols) * 4.0)

    mode_adjustment = 0.0
    if mode == "Balanced":
        mode_adjustment = 2.0
    elif mode == "Aggressive Research":
        mode_adjustment = 4.0

    raw_score = (
        35.0
        + wf_component
        + beat_component
        + median_component
        + avg_component
        + sharpe_component
        + trade_count_component
        + stability_component
        + regime_component
        + mode_adjustment
        - drawdown_penalty
        - benchmark_penalty
        - cost_penalty
        - risk_penalty
        - missing_penalty
    )

    if "do not trust" in row["WalkForwardVerdict"].lower():
        raw_score = min(raw_score, 24.0)
    if row["MetaDecision"] == "Avoid" and row["WalkForwardReliabilityScore"] < 30.0:
        raw_score = min(raw_score, 20.0)

    components = {
        "WalkForwardReliabilityComponent": round(float(wf_component), 4),
        "BeatBuyHoldComponent": round(float(beat_component), 4),
        "MedianVsBuyHoldComponent": round(float(median_component), 4),
        "AvgVsBuyHoldComponent": round(float(avg_component), 4),
        "SharpeComponent": round(float(sharpe_component), 4),
        "TradeCountComponent": round(float(trade_count_component), 4),
        "StabilityComponent": round(float(stability_component), 4),
        "RegimeComponent": round(float(regime_component), 4),
        "DrawdownPenalty": round(float(drawdown_penalty), 4),
        "BenchmarkPenalty": round(float(benchmark_penalty), 4),
        "CostFragilityPenalty": round(float(cost_penalty), 4),
        "RiskPenalty": round(float(risk_penalty), 4),
        "MissingDataPenalty": round(float(missing_penalty), 4),
        "ModeAdjustment": round(float(mode_adjustment), 4),
        "MissingColumns": ", ".join(missing_cols),
    }
    return {
        "ReliabilityScore_0_100": round(_clip(raw_score), 2),
        "ScoreComponents": components,
    }


def compute_reliability_grade(meta_row: Any, grading_mode: str = "Conservative") -> Dict[str, Any]:
    """Assign a research-quality grade. This never overrides MetaDecision as the action gate."""
    row = _normalized_grading_row(meta_row)
    score_info = compute_reliability_score(row, grading_mode)
    score = float(score_info["ReliabilityScore_0_100"])
    mode = grading_mode if grading_mode in CALIBRATION_MODES else "Conservative"

    hard_reject = (
        "do not trust" in row["WalkForwardVerdict"].lower()
        or row["WalkForwardReliabilityScore"] < 20.0
        or (row["MetaDecision"] == "Avoid" and row["MetaRiskScore"] >= 85.0)
    )
    strict_grade_a = (
        score >= 78.0
        and row["MetaRiskScore"] <= 45.0
        and row["WalkForwardReliabilityScore"] >= 68.0
        and row["BeatBuyHoldRate_%"] >= 60.0
        and row["MedianLockedVsBuyHold_%"] > 0.0
        and not row["BenchmarkRiskFlag"]
        and not row["CostFragilityFlag"]
        and not row["DrawdownRiskFlag"]
        and row["StabilityFlag"] == "Stable"
        and row["RegimeLabel"] not in {"High-volatility drawdown", "Risk-off / weak trend", "Insufficient regime data"}
        and row["MetaDecision"] != "Avoid"
    )

    if hard_reject:
        grade = "F: Rejected / Diagnostic Only"
    elif strict_grade_a:
        grade = "A: Near-Trade Research Candidate"
    elif score >= 65.0 and row["WalkForwardReliabilityScore"] >= 55.0 and row["MetaRiskScore"] <= 65.0:
        grade = "B: Strong Research Candidate"
    elif score >= 45.0 and row["WalkForwardReliabilityScore"] >= 35.0:
        grade = "C: Weak Research Candidate"
    elif row["RegimeLabel"] in {"High-volatility drawdown", "Risk-off / weak trend"} or row["CalibratedMetaDecision"] == "Defensive Only":
        grade = "D: Defensive Watch / Regime Evidence"
    elif score >= 25.0:
        grade = "E: Avoid as Standalone"
    else:
        grade = "F: Rejected / Diagnostic Only"

    if mode == "Conservative" and grade.startswith("A:") and row["MetaDecision"] != "Trade":
        grade = "B: Strong Research Candidate"
    if mode == "Aggressive Research" and grade.startswith("E:") and score >= 35.0 and not hard_reject:
        grade = "D: Defensive Watch / Regime Evidence"

    if grade.startswith("A:"):
        readiness = "Near-Trade Research Candidate"
    elif grade.startswith("B:"):
        readiness = "Candidate For Deep Research"
    elif grade.startswith("C:"):
        readiness = "Needs More Evidence"
    elif grade.startswith("D:"):
        readiness = "Watchlist"
    else:
        readiness = "Not Eligible"

    explanation_parts = [
        f"Score {score:.2f}/100 from walk-forward reliability, benchmark edge, drawdown, stability, regime, and risk penalties.",
        f"MetaDecision remains {row['MetaDecision']}; grade is research quality only, not a live trading label.",
    ]
    if row["BenchmarkRiskFlag"]:
        explanation_parts.append("Benchmark weakness is active.")
    if row["DrawdownRiskFlag"]:
        explanation_parts.append("Drawdown risk is active.")
    if row["CostFragilityFlag"]:
        explanation_parts.append("Cost fragility is active.")
    if row["StabilityFlag"] != "Stable":
        explanation_parts.append(f"Stability flag is {row['StabilityFlag']}.")
    if hard_reject:
        explanation_parts.append("Rejected because walk-forward evidence is Do Not Trust or too weak.")
    if grade.startswith("A:"):
        explanation_parts.append("A-grade is still not production-ready; it only merits deeper research.")
    if mode == "Aggressive Research":
        explanation_parts.append("Aggressive Research may raise research attention, never production readiness.")

    return {
        "ReliabilityGrade": grade,
        "ReliabilityScore_0_100": score,
        "PromotionReadiness": readiness,
        "GradeExplanation": " ".join(explanation_parts),
        "ScoreComponents": score_info["ScoreComponents"],
    }


def assign_research_priority(meta_row: Any, grade_info: Optional[Dict[str, Any]] = None, grading_mode: str = "Conservative") -> str:
    """Assign research triage priority from grade, score, and risk context."""
    row = _normalized_grading_row(meta_row)
    info = grade_info or compute_reliability_grade(row, grading_mode)
    grade = str(info["ReliabilityGrade"])
    score = float(info["ReliabilityScore_0_100"])
    mode = grading_mode if grading_mode in CALIBRATION_MODES else "Conservative"

    if grade.startswith("A:"):
        return "Critical"
    if grade.startswith("B:"):
        return "High"
    if grade.startswith("C:"):
        return "High" if mode == "Aggressive Research" and score >= 55.0 else "Medium"
    if grade.startswith("D:"):
        return "Medium" if mode != "Conservative" else "Low"
    if grade.startswith("E:"):
        return "Low"
    return "Archive"


def assign_next_best_action(meta_row: Any, grade_info: Optional[Dict[str, Any]] = None, grading_mode: str = "Conservative") -> str:
    """Recommend the next research action without changing the decision gate."""
    row = _normalized_grading_row(meta_row)
    info = grade_info or compute_reliability_grade(row, grading_mode)
    grade = str(info["ReliabilityGrade"])

    if grade.startswith("A:"):
        return "Run candidate diagnostics and risk-control upgrade; paper-track before any escalation."
    if row["CostFragilityFlag"]:
        return "Run cost/slippage stress and improve probability calibration."
    if row["DrawdownRiskFlag"] or row["WorstLockedMaxDrawdown_%"] <= -25.0:
        return "Run risk-control upgrade and test drawdown filters."
    if row["BenchmarkRiskFlag"]:
        return "Re-run walk-forward with wider windows and require stronger benchmark-relative evidence."
    if row["StabilityFlag"] != "Stable":
        return "Re-run walk-forward with wider windows and inspect threshold/cooldown stability."
    if row["RegimeLabel"] in {"High-volatility drawdown", "Risk-off / weak trend"}:
        return "Test regime filter and keep as defensive indicator."
    if grade.startswith("B:"):
        return "Run candidate diagnostics and probability calibration review."
    if grade.startswith("C:"):
        return "Needs more evidence; monitor future walk-forward updates."
    if grade.startswith("D:"):
        return "Keep as defensive indicator and regime evidence."
    if grade.startswith("E:"):
        return "Archive standalone signal but keep as meta input."
    return "Rejected standalone; keep for diagnostics and failure-mode tracking."


def build_meta_reliability_grading(
    meta_or_audit_results: Optional[pd.DataFrame],
    grading_mode: str = "Conservative",
) -> pd.DataFrame:
    """Build reliability-grade rows from Phase 8 meta or Phase 8A audit output."""
    if meta_or_audit_results is None or meta_or_audit_results.empty:
        return pd.DataFrame(columns=RELIABILITY_GRADING_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for _, source_row in meta_or_audit_results.iterrows():
        row = _normalized_grading_row(source_row)
        grade_info = compute_reliability_grade(row, grading_mode)
        priority = assign_research_priority(row, grade_info, grading_mode)
        next_action = assign_next_best_action(row, grade_info, grading_mode)
        out = {
            "Asset": row["Asset"],
            "Horizon": row["Horizon"],
            "MetaDecision": row["MetaDecision"],
            "CalibratedMetaDecision": row["CalibratedMetaDecision"],
            "ReliabilityGrade": grade_info["ReliabilityGrade"],
            "ReliabilityScore_0_100": grade_info["ReliabilityScore_0_100"],
            "PromotionReadiness": grade_info["PromotionReadiness"],
            "ResearchPriority": priority,
            "NextBestAction": next_action,
            "MetaConfidenceScore": row["MetaConfidenceScore"],
            "MetaRiskScore": row["MetaRiskScore"],
            "WalkForwardReliabilityScore": row["WalkForwardReliabilityScore"],
            "BeatBuyHoldRate_%": row["BeatBuyHoldRate_%"],
            "MedianLockedVsBuyHold_%": row["MedianLockedVsBuyHold_%"],
            "AvgLockedVsBuyHold_%": row["AvgLockedVsBuyHold_%"],
            "WorstLockedMaxDrawdown_%": row["WorstLockedMaxDrawdown_%"],
            "ThresholdStability": row["ThresholdStability"],
            "CooldownStability": row["CooldownStability"],
            "RegimeLabel": row["RegimeLabel"],
            "BenchmarkRiskFlag": row["BenchmarkRiskFlag"],
            "CostFragilityFlag": row["CostFragilityFlag"],
            "DrawdownRiskFlag": row["DrawdownRiskFlag"],
            "StabilityFlag": row["StabilityFlag"],
            "MainBlockingRule": row["MainBlockingRule"],
            "GradeExplanation": grade_info["GradeExplanation"],
        }
        out.update(grade_info["ScoreComponents"])
        rows.append(out)

    out_df = pd.DataFrame(rows)
    for col in RELIABILITY_GRADING_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = np.nan
    leading = RELIABILITY_GRADING_COLUMNS
    extra = [c for c in out_df.columns if c not in leading]
    return out_df[leading + extra]


def summarize_reliability_grades(grading_table: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Summarize grade counts, action counts, and useful research slices."""
    if grading_table is None or grading_table.empty:
        empty = pd.DataFrame()
        return {
            "grade_counts": pd.DataFrame(columns=["ReliabilityGrade", "Count"]),
            "top_research_candidates": empty,
            "defensive_watchlist": empty,
            "avoid_archive_list": empty,
            "next_action_summary": pd.DataFrame(columns=["NextBestAction", "Count"]),
            "score_components": empty,
        }

    grade_counts = (
        grading_table["ReliabilityGrade"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("ReliabilityGrade")
        .reset_index(name="Count")
    )
    top_research = grading_table[
        grading_table["ReliabilityGrade"].apply(lambda value: str(value).startswith(("A:", "B:", "C:")))
    ].copy()
    if not top_research.empty:
        top_research = top_research.sort_values(["ReliabilityScore_0_100", "MetaConfidenceScore"], ascending=[False, False])
    defensive = grading_table[grading_table["ReliabilityGrade"].astype(str).str.startswith("D:")].copy()
    avoid_archive = grading_table[
        grading_table["ReliabilityGrade"].apply(lambda value: str(value).startswith(("E:", "F:")))
    ].copy()
    if not avoid_archive.empty:
        avoid_archive = avoid_archive.sort_values(["ReliabilityScore_0_100", "MetaRiskScore"], ascending=[True, False])
    actions = (
        grading_table["NextBestAction"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("NextBestAction")
        .reset_index(name="Count")
    )
    component_cols = ["Asset", "Horizon", "ReliabilityGrade", "ReliabilityScore_0_100"] + [
        col for col in RELIABILITY_SCORE_COMPONENTS if col in grading_table.columns
    ]
    score_components = grading_table[component_cols].copy()
    return {
        "grade_counts": grade_counts,
        "top_research_candidates": top_research,
        "defensive_watchlist": defensive,
        "avoid_archive_list": avoid_archive,
        "next_action_summary": actions,
        "score_components": score_components,
    }


def run_meta_score_calibration(
    meta_or_audit_results: Optional[pd.DataFrame],
    grading_mode: str = "Conservative",
) -> MetaReliabilityGradingReport:
    """Run Phase 8B reliability grading over existing Phase 8 or 8A outputs."""
    mode = grading_mode if grading_mode in CALIBRATION_MODES else "Conservative"
    grading = build_meta_reliability_grading(meta_or_audit_results, mode)
    summaries = summarize_reliability_grades(grading)
    warnings: List[str] = []
    if meta_or_audit_results is None or meta_or_audit_results.empty:
        warnings.append("No Phase 8/8A rows were provided for reliability grading.")
    if mode == "Aggressive Research":
        warnings.append("Aggressive Research can raise research priority, but never production readiness.")
    if not grading.empty and grading["ReliabilityGrade"].astype(str).str.startswith("A:").any():
        warnings.append("A-grade means near-trade research quality only; it is not production-ready.")

    return MetaReliabilityGradingReport(
        grading_table=grading,
        grade_counts=summaries["grade_counts"],
        top_research_candidates=summaries["top_research_candidates"],
        defensive_watchlist=summaries["defensive_watchlist"],
        avoid_archive_list=summaries["avoid_archive_list"],
        next_action_summary=summaries["next_action_summary"],
        score_components=summaries["score_components"],
        warnings=list(dict.fromkeys(warnings)),
        settings={
            "grading_mode": mode,
            "phase": "8B",
            "purpose": "score_calibration_and_reliability_grading_only",
            "retunes_models_or_thresholds": False,
            "production_ready_label_allowed": False,
            "meta_decision_remains_action_gate": True,
        },
    )
