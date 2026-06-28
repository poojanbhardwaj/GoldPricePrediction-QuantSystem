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


@dataclass
class EvidenceExpansionReport:
    full_evidence_table: pd.DataFrame
    robustness_summary: pd.DataFrame
    configuration_summary: pd.DataFrame
    cost_sensitivity_summary: pd.DataFrame
    promotion_recommendations: pd.DataFrame
    warning_table: pd.DataFrame
    overall_summary: pd.DataFrame
    settings: Dict[str, Any]


@dataclass
class EvidenceQualityDiagnosticsReport:
    overall_summary: pd.DataFrame
    evidence_quality_table: pd.DataFrame
    signal_coverage_table: pd.DataFrame
    threshold_cooldown_sensitivity_table: pd.DataFrame
    horizon_quality_table: pd.DataFrame
    benchmark_dependency_table: pd.DataFrame
    regime_concentration_table: pd.DataFrame
    probability_quality_warning_table: pd.DataFrame
    candidate_failure_reason_table: pd.DataFrame
    next_research_action_table: pd.DataFrame
    settings: Dict[str, Any]


@dataclass
class SignalPolicySensitivityReport:
    overall_summary: pd.DataFrame
    full_policy_sensitivity_table: pd.DataFrame
    coverage_recovery_summary: pd.DataFrame
    threshold_sensitivity_table: pd.DataFrame
    cooldown_sensitivity_table: pd.DataFrame
    probability_band_sensitivity_table: pd.DataFrame
    horizon_sensitivity_table: pd.DataFrame
    coverage_edge_frontier_table: pd.DataFrame
    candidate_recommendation_table: pd.DataFrame
    warning_table: pd.DataFrame
    next_research_action_table: pd.DataFrame
    settings: Dict[str, Any]


@dataclass
class ProbabilityCalibrationReport:
    overall_summary: pd.DataFrame
    calibration_summary_table: pd.DataFrame
    probability_bin_table: pd.DataFrame
    probability_filter_simulation_table: pd.DataFrame
    confidence_usefulness_table: pd.DataFrame
    calibration_error_table: pd.DataFrame
    high_confidence_failure_table: pd.DataFrame
    candidate_recommendation_table: pd.DataFrame
    warning_table: pd.DataFrame
    next_research_action_table: pd.DataFrame
    settings: Dict[str, Any]


@dataclass
class TradeEvidenceLedgerReport:
    ledger_table: pd.DataFrame
    ledger_quality_summary: pd.DataFrame
    asset_horizon_coverage_table: pd.DataFrame
    probability_outcome_availability_table: pd.DataFrame
    trade_outcome_distribution_table: pd.DataFrame
    benchmark_outcome_table: pd.DataFrame
    drawdown_outcome_table: pd.DataFrame
    ledger_warning_table: pd.DataFrame
    next_research_action_table: pd.DataFrame
    settings: Dict[str, Any]


@dataclass
class RawTradeLogExporterReport:
    raw_signal_trade_log_table: pd.DataFrame
    raw_log_quality_summary: pd.DataFrame
    asset_horizon_raw_coverage_table: pd.DataFrame
    probability_outcome_readiness_table: pd.DataFrame
    trade_outcome_distribution_table: pd.DataFrame
    benchmark_comparison_table: pd.DataFrame
    drawdown_during_trade_table: pd.DataFrame
    no_trade_skipped_signal_table: pd.DataFrame
    warning_table: pd.DataFrame
    next_research_action_table: pd.DataFrame
    settings: Dict[str, Any]


@dataclass
class TrueRawTradeLogGenerationReport:
    true_raw_trade_log: pd.DataFrame
    raw_log_quality_summary: pd.DataFrame
    asset_horizon_raw_coverage: pd.DataFrame
    probability_outcome_readiness: pd.DataFrame
    missing_source_diagnostic: pd.DataFrame
    benchmark_comparison: pd.DataFrame
    drawdown_during_trade: pd.DataFrame
    warning_table: pd.DataFrame
    next_research_action_table: pd.DataFrame
    phase8_closure_readiness_table: pd.DataFrame
    aggregate_fallback_diagnostic: pd.DataFrame
    no_trade_skipped_signal_table: pd.DataFrame
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


EVIDENCE_EXPANSION_CONFIG_DEFAULTS = {
    "validation_windows": [120, 180, 252],
    "test_windows": [60, 90, 126],
    "step_sizes": [30, 60],
    "transaction_costs": [0.0005, 0.001, 0.002],
    "window_modes": ["rolling", "expanding"],
}

EVIDENCE_TABLE_COLUMNS = [
    "Asset",
    "Horizon",
    "StartingReliabilityGrade",
    "StartingReliabilityScore_0_100",
    "ValidationWindow",
    "TestWindow",
    "StepSize",
    "TransactionCost",
    "WindowMode",
    "ValidConfiguration",
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
    "Error",
]

ROBUSTNESS_SUMMARY_COLUMNS = [
    "Asset",
    "Horizon",
    "StartingReliabilityGrade",
    "StartingReliabilityScore_0_100",
    "ConfigurationsTested",
    "ValidConfigurations",
    "ValidConfigurationRate_%",
    "PositiveReturnRate_%",
    "BeatBuyHoldRate_%",
    "AvgStrategyReturn_%",
    "MedianStrategyReturn_%",
    "AvgVsBuyHold_%",
    "MedianVsBuyHold_%",
    "WorstVsBuyHold_%",
    "AvgMaxDrawdown_%",
    "WorstMaxDrawdown_%",
    "AvgTradeCount",
    "LowTradeCountRate_%",
    "CostFragilityScore",
    "StabilityScore",
    "RobustnessScore",
    "Warnings",
]

PROMOTION_RECOMMENDATION_COLUMNS = [
    "Asset",
    "Horizon",
    "StartingReliabilityGrade",
    "RecommendedReliabilityGrade",
    "Recommendation",
    "RobustnessScore",
    "MainReason",
    "Warnings",
]


def build_evidence_expansion_configs(
    validation_windows: Iterable[int] = EVIDENCE_EXPANSION_CONFIG_DEFAULTS["validation_windows"],
    test_windows: Iterable[int] = EVIDENCE_EXPANSION_CONFIG_DEFAULTS["test_windows"],
    step_sizes: Iterable[int] = EVIDENCE_EXPANSION_CONFIG_DEFAULTS["step_sizes"],
    transaction_costs: Iterable[float] = EVIDENCE_EXPANSION_CONFIG_DEFAULTS["transaction_costs"],
    window_modes: Iterable[str] = EVIDENCE_EXPANSION_CONFIG_DEFAULTS["window_modes"],
) -> pd.DataFrame:
    """Create the fixed evidence-expansion configuration grid."""
    rows: List[Dict[str, Any]] = []
    for validation_window in validation_windows:
        for test_window in test_windows:
            for step_size in step_sizes:
                for transaction_cost in transaction_costs:
                    for window_mode in window_modes:
                        rows.append(
                            {
                                "ValidationWindow": int(validation_window),
                                "TestWindow": int(test_window),
                                "StepSize": int(step_size),
                                "TransactionCost": float(transaction_cost),
                                "WindowMode": str(window_mode),
                            }
                        )
    return pd.DataFrame(rows)


def _grade_letter(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 1 and text[0].upper() in {"A", "B", "C", "D", "E", "F"}:
        return text[0].upper()
    return "?"


def _candidate_key(row: pd.Series) -> str:
    return f"{row.get('Asset', '')} {int(_safe_int(row.get('Horizon', 0)))}D"


def select_evidence_expansion_candidates(
    grading_table: pd.DataFrame,
    candidate_filter: str = "all",
    selected_assets: Optional[Iterable[str]] = None,
    selected_horizons: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Select candidates for evidence expansion without hiding failures by default."""
    if grading_table is None or grading_table.empty:
        return pd.DataFrame(columns=RELIABILITY_GRADING_COLUMNS)
    out = grading_table.copy()
    if "Horizon" in out.columns:
        out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce").fillna(0).astype(int)
    mode = str(candidate_filter or "all").lower()
    if mode in {"c/d", "only c/d candidates", "c_d"} and "ReliabilityGrade" in out.columns:
        out = out[out["ReliabilityGrade"].apply(lambda value: _grade_letter(value) in {"C", "D"})]
    if mode in {"specific", "specific asset/horizon"}:
        assets = set(str(a) for a in (selected_assets or []))
        horizons = set(int(h) for h in (selected_horizons or []))
        if assets:
            out = out[out["Asset"].astype(str).isin(assets)]
        if horizons:
            out = out[out["Horizon"].astype(int).isin(horizons)]
    return out.reset_index(drop=True)


def _walk_forward_runner_default(**kwargs: Any) -> Any:
    from src.signal_engine import run_walk_forward_risk_validation

    return run_walk_forward_risk_validation(**kwargs)


def _evidence_warning_rows(summary_row: Dict[str, Any], *, min_valid_configurations: int) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    asset = summary_row.get("Asset", "")
    horizon = _safe_int(summary_row.get("Horizon", 0), default=0)

    def add(kind: str, severity: str, message: str) -> None:
        warnings.append({"Asset": asset, "Horizon": horizon, "WarningType": kind, "Severity": severity, "Message": message})

    valid = _safe_int(summary_row.get("ValidConfigurations"), default=0)
    valid_rate = _safe_float(summary_row.get("ValidConfigurationRate_%"), default=0.0)
    low_trade_rate = _safe_float(summary_row.get("LowTradeCountRate_%"), default=100.0)
    cost_fragility = _safe_float(summary_row.get("CostFragilityScore"), default=100.0)
    beat_rate = _safe_float(summary_row.get("BeatBuyHoldRate_%"), default=0.0)
    median_vs = _safe_float(summary_row.get("MedianVsBuyHold_%"), default=0.0)
    worst_dd = _safe_float(summary_row.get("WorstMaxDrawdown_%"), default=0.0)
    stability = _safe_float(summary_row.get("StabilityScore"), default=0.0)
    avg_strategy = _safe_float(summary_row.get("AvgStrategyReturn_%"), default=0.0)
    worst_vs = _safe_float(summary_row.get("WorstVsBuyHold_%"), default=0.0)

    if valid < int(min_valid_configurations):
        add("NotEnoughValidConfigurations", "High", f"Only {valid} valid configurations; minimum is {min_valid_configurations}.")
    if valid_rate < 50.0:
        add("LowEvidence", "High", f"Valid configuration rate is {valid_rate:.2f}%.")
    if low_trade_rate > 25.0:
        add("LowTradeCount", "Medium", f"Low-trade configuration rate is {low_trade_rate:.2f}%.")
    if cost_fragility >= 35.0:
        add("CostFragile", "High", f"Cost fragility score is {cost_fragility:.2f}.")
    if beat_rate < 50.0 or median_vs <= 0.0:
        add("BenchmarkWeakness", "High", f"Beat-buy-hold rate {beat_rate:.2f}% and median edge {median_vs:.2f}%.")
    if worst_dd <= -25.0:
        add("DrawdownRisk", "High", f"Worst max drawdown is {worst_dd:.2f}%.")
    if stability < 60.0:
        add("SplitUnstable", "Medium", f"Stability score is {stability:.2f}.")
    if worst_vs <= -20.0 and beat_rate < 65.0:
        add("WindowSensitive", "Medium", f"Worst vs buy-and-hold is {worst_vs:.2f}%.")
    if avg_strategy <= 0.0:
        add("ReturnDestroyed", "High", f"Average strategy return is {avg_strategy:.2f}%.")
    return warnings


def summarize_expanded_evidence(
    full_evidence_table: pd.DataFrame,
    *,
    min_valid_configurations: int = 6,
) -> Dict[str, pd.DataFrame]:
    """Build candidate/config/cost summaries and warnings from expanded evidence."""
    if full_evidence_table is None or full_evidence_table.empty:
        empty_summary = pd.DataFrame(columns=ROBUSTNESS_SUMMARY_COLUMNS)
        return {
            "robustness_summary": empty_summary,
            "configuration_summary": pd.DataFrame(),
            "cost_sensitivity_summary": pd.DataFrame(),
            "warning_table": pd.DataFrame(columns=["Asset", "Horizon", "WarningType", "Severity", "Message"]),
        }

    rows: List[Dict[str, Any]] = []
    warning_rows: List[Dict[str, Any]] = []
    for (asset, horizon), group in full_evidence_table.groupby(["Asset", "Horizon"], dropna=False):
        g = group.copy()
        valid = g[g["ValidConfiguration"].astype(bool)].copy()
        tested = int(len(g))
        valid_count = int(len(valid))
        start_grade = str(g["StartingReliabilityGrade"].iloc[0])
        start_score = _safe_float(g["StartingReliabilityScore_0_100"].iloc[0], default=0.0)
        if valid.empty:
            row = {
                "Asset": asset,
                "Horizon": int(horizon),
                "StartingReliabilityGrade": start_grade,
                "StartingReliabilityScore_0_100": round(float(start_score), 2),
                "ConfigurationsTested": tested,
                "ValidConfigurations": 0,
                "ValidConfigurationRate_%": 0.0,
                "PositiveReturnRate_%": 0.0,
                "BeatBuyHoldRate_%": 0.0,
                "AvgStrategyReturn_%": 0.0,
                "MedianStrategyReturn_%": 0.0,
                "AvgVsBuyHold_%": 0.0,
                "MedianVsBuyHold_%": 0.0,
                "WorstVsBuyHold_%": 0.0,
                "AvgMaxDrawdown_%": 0.0,
                "WorstMaxDrawdown_%": 0.0,
                "AvgTradeCount": 0.0,
                "LowTradeCountRate_%": 100.0,
                "CostFragilityScore": 100.0,
                "StabilityScore": 0.0,
                "RobustnessScore": 0.0,
            }
            candidate_warnings = _evidence_warning_rows(row, min_valid_configurations=min_valid_configurations)
            row["Warnings"] = "; ".join(dict.fromkeys([w["WarningType"] for w in candidate_warnings]))
            warning_rows.extend(candidate_warnings)
            rows.append(row)
            continue

        low_trade_rate = float((valid["LowTradeWindowCount"].fillna(0).astype(float) > 0).mean() * 100.0)
        stable_rate = float(
            (
                valid["ThresholdStability"].fillna("").astype(str).str.lower().eq("stable")
                & valid["CooldownStability"].fillna("").astype(str).str.lower().eq("stable")
            ).mean()
            * 100.0
        )
        cost_edges = valid.groupby("TransactionCost")["MedianLockedVsBuyHold_%"].mean().sort_index()
        if len(cost_edges) >= 2:
            cost_fragility = max(0.0, float(cost_edges.iloc[0] - cost_edges.iloc[-1]) * 8.0)
            if cost_edges.iloc[0] > 0.0 and cost_edges.iloc[-1] <= 0.0:
                cost_fragility += 45.0
        else:
            cost_fragility = 50.0
        cost_fragility = _clip(cost_fragility)

        beat_rate = float(valid["BeatBuyHoldRate_%"].astype(float).mean())
        positive_rate = float(valid["PositiveReturnRate_%"].astype(float).mean())
        avg_vs = float(valid["AvgLockedVsBuyHold_%"].astype(float).mean())
        median_vs = float(valid["MedianLockedVsBuyHold_%"].astype(float).median())
        worst_vs = float(valid["WorstLockedVsBuyHold_%"].astype(float).min())
        avg_strategy = float(valid["AvgLockedStrategyReturn_%"].astype(float).mean())
        median_strategy = float(valid["MedianLockedStrategyReturn_%"].astype(float).median())
        avg_dd = float(valid["AvgLockedMaxDrawdown_%"].astype(float).mean())
        worst_dd = float(valid["WorstLockedMaxDrawdown_%"].astype(float).min())
        avg_trades = float(valid["AvgTradesPerWindow"].astype(float).mean())
        valid_rate = valid_count / max(tested, 1) * 100.0

        robustness = _clip(
            beat_rate * 0.25
            + positive_rate * 0.15
            + np.clip(median_vs, -20.0, 20.0) * 1.2
            + np.clip(avg_vs, -20.0, 20.0) * 0.8
            + stable_rate * 0.20
            + valid_rate * 0.10
            + min(avg_trades * 2.0, 10.0)
            - max(0.0, abs(min(worst_dd, 0.0)) - 20.0) * 0.6
            - cost_fragility * 0.20
            - low_trade_rate * 0.15
        )
        row = {
            "Asset": asset,
            "Horizon": int(horizon),
            "StartingReliabilityGrade": start_grade,
            "StartingReliabilityScore_0_100": round(float(start_score), 2),
            "ConfigurationsTested": tested,
            "ValidConfigurations": valid_count,
            "ValidConfigurationRate_%": round(valid_rate, 2),
            "PositiveReturnRate_%": round(positive_rate, 2),
            "BeatBuyHoldRate_%": round(beat_rate, 2),
            "AvgStrategyReturn_%": round(avg_strategy, 4),
            "MedianStrategyReturn_%": round(median_strategy, 4),
            "AvgVsBuyHold_%": round(avg_vs, 4),
            "MedianVsBuyHold_%": round(median_vs, 4),
            "WorstVsBuyHold_%": round(worst_vs, 4),
            "AvgMaxDrawdown_%": round(avg_dd, 4),
            "WorstMaxDrawdown_%": round(worst_dd, 4),
            "AvgTradeCount": round(avg_trades, 2),
            "LowTradeCountRate_%": round(low_trade_rate, 2),
            "CostFragilityScore": round(cost_fragility, 2),
            "StabilityScore": round(stable_rate, 2),
            "RobustnessScore": round(robustness, 2),
        }
        candidate_warnings = _evidence_warning_rows(row, min_valid_configurations=min_valid_configurations)
        row["Warnings"] = "; ".join(dict.fromkeys([w["WarningType"] for w in candidate_warnings]))
        warning_rows.extend(candidate_warnings)
        rows.append(row)

    robustness_summary = pd.DataFrame(rows)
    for col in ROBUSTNESS_SUMMARY_COLUMNS:
        if col not in robustness_summary.columns:
            robustness_summary[col] = np.nan
    robustness_summary = robustness_summary[ROBUSTNESS_SUMMARY_COLUMNS]

    config_group_cols = ["ValidationWindow", "TestWindow", "StepSize", "TransactionCost", "WindowMode"]
    config_summary = (
        full_evidence_table.groupby(config_group_cols, dropna=False)
        .agg(
            CandidatesTested=("Asset", "count"),
            ValidConfigurations=("ValidConfiguration", lambda s: int(pd.Series(s).astype(bool).sum())),
            AvgBeatBuyHoldRate=("BeatBuyHoldRate_%", "mean"),
            AvgMedianVsBuyHold=("MedianLockedVsBuyHold_%", "mean"),
            FailureCount=("ValidConfiguration", lambda s: int((~pd.Series(s).astype(bool)).sum())),
        )
        .reset_index()
    )
    config_summary["SuccessRate_%"] = config_summary["ValidConfigurations"] / config_summary["CandidatesTested"].clip(lower=1) * 100.0

    valid_evidence = full_evidence_table[full_evidence_table["ValidConfiguration"].astype(bool)]
    if not valid_evidence.empty:
        cost_summary = (
            valid_evidence.groupby(["Asset", "Horizon", "TransactionCost"], dropna=False)
            .agg(
                **{
                    "ValidConfigurations": ("ValidConfiguration", "count"),
                    "BeatBuyHoldRate_%": ("BeatBuyHoldRate_%", "mean"),
                    "MedianVsBuyHold_%": ("MedianLockedVsBuyHold_%", "median"),
                    "AvgVsBuyHold_%": ("AvgLockedVsBuyHold_%", "mean"),
                    "WorstVsBuyHold_%": ("WorstLockedVsBuyHold_%", "min"),
                    "AvgMaxDrawdown_%": ("AvgLockedMaxDrawdown_%", "mean"),
                }
            )
            .reset_index()
        )
    else:
        cost_summary = pd.DataFrame(columns=["Asset", "Horizon", "TransactionCost"])
    warning_table = pd.DataFrame(warning_rows)
    if warning_table.empty:
        warning_table = pd.DataFrame(columns=["Asset", "Horizon", "WarningType", "Severity", "Message"])
    return {
        "robustness_summary": robustness_summary,
        "configuration_summary": config_summary,
        "cost_sensitivity_summary": cost_summary,
        "warning_table": warning_table,
    }


def build_promotion_demotion_recommendations(
    robustness_summary: pd.DataFrame,
    *,
    min_valid_configurations: int = 6,
) -> pd.DataFrame:
    """Recommend grade movement conservatively from expanded evidence."""
    rows: List[Dict[str, Any]] = []
    if robustness_summary is None or robustness_summary.empty:
        return pd.DataFrame(columns=PROMOTION_RECOMMENDATION_COLUMNS)

    for _, row in robustness_summary.iterrows():
        start_grade = str(row.get("StartingReliabilityGrade", ""))
        letter = _grade_letter(start_grade)
        valid = _safe_int(row.get("ValidConfigurations"), default=0)
        beat = _safe_float(row.get("BeatBuyHoldRate_%"), default=0.0)
        median_vs = _safe_float(row.get("MedianVsBuyHold_%"), default=0.0)
        worst_dd = _safe_float(row.get("WorstMaxDrawdown_%"), default=0.0)
        cost_fragility = _safe_float(row.get("CostFragilityScore"), default=100.0)
        stability = _safe_float(row.get("StabilityScore"), default=0.0)
        robustness = _safe_float(row.get("RobustnessScore"), default=0.0)
        avg_trades = _safe_float(row.get("AvgTradeCount"), default=0.0)
        low_trade_rate = _safe_float(row.get("LowTradeCountRate_%"), default=100.0)
        warnings = str(row.get("Warnings", "") or "")
        recommended_letter = letter if letter in {"A", "B", "C", "D", "E", "F"} else "F"
        recommendation = "Maintain"
        reason = "Evidence is mixed; keep current reliability grade."
        promotion_blocked_by_evidence_quality = low_trade_rate >= 75.0 or stability <= 25.0
        evidence_quality_blockers: List[str] = []
        if low_trade_rate >= 75.0:
            evidence_quality_blockers.append("low trade count")
            if "LowTradeCount" not in warnings:
                warnings = "; ".join([w for w in [warnings, "LowTradeCount"] if w])
        if stability <= 25.0:
            evidence_quality_blockers.append("unstable split evidence")
            if "SplitUnstable" not in warnings:
                warnings = "; ".join([w for w in [warnings, "SplitUnstable"] if w])
        d_to_c_evidence_improved = (
            letter == "D"
            and valid >= min_valid_configurations
            and beat >= 50.0
            and median_vs > 0.0
            and worst_dd >= -28.0
            and robustness >= 42.0
        )
        c_to_b_evidence_improved = (
            letter == "C"
            and valid >= min_valid_configurations
            and beat >= 60.0
            and median_vs > 0.75
            and worst_dd >= -22.0
            and cost_fragility <= 30.0
            and robustness >= 55.0
        )

        severe_failure = (
            valid < min_valid_configurations
            or beat < 40.0
            or median_vs <= 0.0
            or worst_dd <= -35.0
            or robustness < 25.0
        )
        if severe_failure:
            if letter in {"A", "B"}:
                recommended_letter = "C"
            elif letter == "C":
                recommended_letter = "D"
            elif letter == "D":
                recommended_letter = "E"
            elif letter in {"E", "F"}:
                recommended_letter = "F"
            recommendation = "Demote" if recommended_letter != letter else "Maintain weak/rejected"
            reason = "Robustness failed across expanded configurations."
        elif (
            letter == "C"
            and valid >= min_valid_configurations
            and beat >= 60.0
            and median_vs > 0.75
            and worst_dd >= -22.0
            and cost_fragility <= 30.0
            and stability >= 60.0
            and avg_trades >= 2.0
            and robustness >= 55.0
            and not promotion_blocked_by_evidence_quality
        ):
            recommended_letter = "B"
            recommendation = "Promote C to B"
            reason = "C-grade candidate survived many configurations with positive median benchmark edge."
        elif (
            (d_to_c_evidence_improved or c_to_b_evidence_improved)
            and promotion_blocked_by_evidence_quality
        ):
            recommended_letter = letter if letter in {"C", "D"} else recommended_letter
            recommendation = "Conditional research evidence"
            reason = (
                "Evidence improved, but promotion blocked due to "
                f"{' / '.join(evidence_quality_blockers)}."
            )
        elif (
            d_to_c_evidence_improved
            and not promotion_blocked_by_evidence_quality
        ):
            recommended_letter = "C"
            recommendation = "Promote D to C"
            reason = "D-grade evidence improved, but meaningful weakness remains."
        elif (
            letter == "B"
            and valid >= max(min_valid_configurations, 18)
            and beat >= 78.0
            and median_vs >= 4.0
            and worst_dd >= -12.0
            and cost_fragility <= 10.0
            and stability >= 85.0
            and robustness >= 82.0
            and not promotion_blocked_by_evidence_quality
        ):
            recommended_letter = "A"
            recommendation = "Promote B to A"
            reason = "Extremely strong expanded evidence; still research-only and not production-ready."
        else:
            if "NoImprovement" not in warnings:
                warnings = "; ".join([w for w in [warnings, "NoImprovement"] if w])

        grade_name = {
            "A": "A: Near-Trade Research Candidate",
            "B": "B: Strong Research Candidate",
            "C": "C: Weak Research Candidate",
            "D": "D: Defensive Watch / Regime Evidence",
            "E": "E: Avoid as Standalone",
            "F": "F: Rejected / Diagnostic Only",
        }.get(recommended_letter, "F: Rejected / Diagnostic Only")
        rows.append(
            {
                "Asset": row.get("Asset", ""),
                "Horizon": int(_safe_int(row.get("Horizon", 0))),
                "StartingReliabilityGrade": start_grade,
                "RecommendedReliabilityGrade": grade_name,
                "Recommendation": recommendation,
                "RobustnessScore": round(float(robustness), 2),
                "MainReason": reason,
                "Warnings": warnings,
            }
        )

    out = pd.DataFrame(rows)
    for col in PROMOTION_RECOMMENDATION_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[PROMOTION_RECOMMENDATION_COLUMNS]


def _run_evidence_expansion_impl(
    *,
    grading_table: pd.DataFrame,
    raw_df: Optional[pd.DataFrame] = None,
    candidate_filter: str = "all",
    selected_assets: Optional[Iterable[str]] = None,
    selected_horizons: Optional[Iterable[int]] = None,
    validation_windows: Iterable[int] = EVIDENCE_EXPANSION_CONFIG_DEFAULTS["validation_windows"],
    test_windows: Iterable[int] = EVIDENCE_EXPANSION_CONFIG_DEFAULTS["test_windows"],
    step_sizes: Iterable[int] = EVIDENCE_EXPANSION_CONFIG_DEFAULTS["step_sizes"],
    transaction_costs: Iterable[float] = EVIDENCE_EXPANSION_CONFIG_DEFAULTS["transaction_costs"],
    window_modes: Iterable[str] = EVIDENCE_EXPANSION_CONFIG_DEFAULTS["window_modes"],
    model_depth: str = "core",
    use_phase5_features: bool = True,
    signal_mode: str = "long_only",
    min_trades_per_window: int = 3,
    min_valid_configurations: int = 6,
    threshold_candidates: Iterable[float] = (0.50, 0.55, 0.60, 0.65, 0.70),
    cooldown_candidates: Iterable[int] = (0, 2, 5),
    signal_outputs: Optional[Dict[Any, Any]] = None,
    walk_forward_runner: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
) -> EvidenceExpansionReport:
    """Stress-test Phase 8B candidates across predeclared walk-forward configurations."""
    candidates = select_evidence_expansion_candidates(
        grading_table,
        candidate_filter=candidate_filter,
        selected_assets=selected_assets,
        selected_horizons=selected_horizons,
    )
    configs = build_evidence_expansion_configs(
        validation_windows=validation_windows,
        test_windows=test_windows,
        step_sizes=step_sizes,
        transaction_costs=transaction_costs,
        window_modes=window_modes,
    )
    runner = walk_forward_runner or _walk_forward_runner_default
    rows: List[Dict[str, Any]] = []
    total = len(candidates) * len(configs)
    done = 0

    for _, candidate in candidates.iterrows():
        asset = str(candidate.get("Asset", ""))
        horizon = int(_safe_int(candidate.get("Horizon", 0)))
        start_grade = str(candidate.get("ReliabilityGrade", ""))
        start_score = _safe_float(candidate.get("ReliabilityScore_0_100"), default=0.0)
        for _, config in configs.iterrows():
            if progress_callback is not None:
                progress_callback(done, max(total, 1), f"Testing {asset} {horizon}D")
            base = {
                "Asset": asset,
                "Horizon": horizon,
                "StartingReliabilityGrade": start_grade,
                "StartingReliabilityScore_0_100": round(float(start_score), 2),
                "ValidationWindow": int(config["ValidationWindow"]),
                "TestWindow": int(config["TestWindow"]),
                "StepSize": int(config["StepSize"]),
                "TransactionCost": float(config["TransactionCost"]),
                "WindowMode": str(config["WindowMode"]),
            }
            try:
                report = runner(
                    raw_df=raw_df,
                    asset_names=[asset],
                    horizons=[horizon],
                    model_depth=model_depth,
                    use_phase5_features=use_phase5_features,
                    signal_mode=signal_mode,
                    threshold_candidates=threshold_candidates,
                    cooldown_candidates=cooldown_candidates,
                    transaction_cost=float(config["TransactionCost"]),
                    validation_window=int(config["ValidationWindow"]),
                    test_window=int(config["TestWindow"]),
                    step_size=int(config["StepSize"]),
                    min_trades_per_window=int(min_trades_per_window),
                    window_mode=str(config["WindowMode"]),
                    signal_outputs=signal_outputs,
                )
                aggregate = getattr(report, "aggregate_summary", pd.DataFrame())
                if aggregate is None or aggregate.empty:
                    row = {**base, "ValidConfiguration": False, "Error": "No aggregate summary returned"}
                else:
                    agg = aggregate.iloc[0].to_dict()
                    row = {
                        **base,
                        **agg,
                        "Asset": asset,
                        "Horizon": horizon,
                        "ValidConfiguration": True,
                        "Error": "",
                    }
            except Exception as exc:
                row = {**base, "ValidConfiguration": False, "Error": str(exc)}
            rows.append(row)
            done += 1

    evidence = pd.DataFrame(rows)
    if evidence.empty:
        evidence = pd.DataFrame(columns=EVIDENCE_TABLE_COLUMNS)
    for col in EVIDENCE_TABLE_COLUMNS:
        if col not in evidence.columns:
            evidence[col] = np.nan
    leading = EVIDENCE_TABLE_COLUMNS
    extra = [c for c in evidence.columns if c not in leading]
    evidence = evidence[leading + extra]

    summaries = summarize_expanded_evidence(evidence, min_valid_configurations=min_valid_configurations)
    recommendations = build_promotion_demotion_recommendations(
        summaries["robustness_summary"],
        min_valid_configurations=min_valid_configurations,
    )
    if not recommendations.empty and not recommendations["Recommendation"].astype(str).str.contains("Promote", case=False, na=False).any():
        no_improvement = pd.DataFrame(
            [
                {
                    "Asset": "ALL",
                    "Horizon": 0,
                    "WarningType": "NoImprovement",
                    "Severity": "Info",
                    "Message": "No candidate improved under expanded evidence.",
                }
            ]
        )
        summaries["warning_table"] = pd.concat([summaries["warning_table"], no_improvement], ignore_index=True)

    overall = pd.DataFrame(
        [
            {
                "CandidatesTested": int(len(candidates)),
                "ConfigurationsPerCandidate": int(len(configs)),
                "TotalEvidenceRows": int(len(evidence)),
                "ValidEvidenceRows": int(evidence["ValidConfiguration"].astype(bool).sum()) if not evidence.empty else 0,
                "Promotions": int(recommendations["Recommendation"].astype(str).str.contains("Promote", case=False, na=False).sum()) if not recommendations.empty else 0,
                "Demotions": int(recommendations["Recommendation"].astype(str).str.contains("Demote", case=False, na=False).sum()) if not recommendations.empty else 0,
                "ResearchOnly": True,
                "ProductionReadyLabelAllowed": False,
            }
        ]
    )
    return EvidenceExpansionReport(
        full_evidence_table=evidence,
        robustness_summary=summaries["robustness_summary"],
        configuration_summary=summaries["configuration_summary"],
        cost_sensitivity_summary=summaries["cost_sensitivity_summary"],
        promotion_recommendations=recommendations,
        warning_table=summaries["warning_table"],
        overall_summary=overall,
        settings={
            "phase": "8C",
            "purpose": "evidence_expansion_and_robustness_revalidation_only",
            "candidate_filter": candidate_filter,
            "model_depth": model_depth,
            "use_phase5_features": bool(use_phase5_features),
            "signal_mode": signal_mode,
            "validation_windows": [int(v) for v in validation_windows],
            "test_windows": [int(v) for v in test_windows],
            "step_sizes": [int(v) for v in step_sizes],
            "transaction_costs": [float(v) for v in transaction_costs],
            "window_modes": [str(v) for v in window_modes],
            "selection_basis": "predeclared_config_grid_not_locked_test_tuning",
            "production_ready_label_allowed": False,
        },
    )


def run_evidence_expansion(*args: Any, **kwargs: Any) -> EvidenceExpansionReport:
    """Public Phase 8C entrypoint used by Streamlit and tests.

    This wrapper intentionally preserves the stable public name expected by
    app.py while the implementation remains internal. It does not change any
    Phase 7G/8/8A/8B logic.
    """
    return _run_evidence_expansion_impl(*args, **kwargs)


EVIDENCE_QUALITY_COLUMNS = [
    "Asset",
    "Horizon",
    "AverageTradeCount",
    "MedianTradeCount",
    "LowTradeCountRate_%",
    "NoTradeConfigurationRate_%",
    "BeatBuyHoldRate_%",
    "MedianVsBuyHold_%",
    "WorstVsBuyHold_%",
    "DrawdownRisk",
    "CostFragility",
    "WindowSensitivity",
    "StabilityScore",
    "RobustnessScore",
    "CoverageScore",
    "EvidenceQualityScore",
    "FailureReasonCategory",
    "DiagnosticsExplanation",
]

SIGNAL_COVERAGE_COLUMNS = [
    "Asset",
    "Horizon",
    "ConfigurationsTested",
    "ValidConfigurations",
    "AverageTradeCount",
    "MedianTradeCount",
    "LowTradeCountRate_%",
    "NoTradeConfigurationRate_%",
    "CoverageScore",
    "CoverageWarning",
]

FAILURE_REASON_COLUMNS = [
    "Asset",
    "Horizon",
    "PrimaryFailureCategory",
    "SecondaryFailureCategories",
    "EvidenceQualityScore",
    "NextResearchAction",
]

NEXT_ACTION_COLUMNS = ["Asset", "Horizon", "FailureReasonCategory", "NextResearchAction", "ActionPriority"]


def _candidate_groups_from_full_evidence(full_evidence_table: pd.DataFrame) -> Iterable[Any]:
    if full_evidence_table is None or full_evidence_table.empty:
        return []
    return full_evidence_table.groupby(["Asset", "Horizon"], dropna=False)


def _lookup_summary_row(summary: Optional[pd.DataFrame], asset: Any, horizon: Any) -> Dict[str, Any]:
    if summary is None or summary.empty:
        return {}
    if "Asset" not in summary.columns or "Horizon" not in summary.columns:
        return {}
    matches = summary[
        summary["Asset"].astype(str).eq(str(asset))
        & pd.to_numeric(summary["Horizon"], errors="coerce").fillna(0).astype(int).eq(int(_safe_int(horizon)))
    ]
    return matches.iloc[0].to_dict() if not matches.empty else {}


def _lookup_grade_row(grading_table: Optional[pd.DataFrame], asset: Any, horizon: Any) -> Dict[str, Any]:
    return _lookup_summary_row(grading_table, asset, horizon)


def _diagnostic_next_action(primary: str) -> str:
    mapping = {
        "InsufficientTradeCoverage": "Run threshold/cooldown sensitivity scan before any grade improvement.",
        "BenchmarkDominated": "Run benchmark-aware passive/active comparison.",
        "CostFragile": "Build an asset-specific execution-cost and slippage model.",
        "DrawdownHeavy": "Test volatility targeting and drawdown controls.",
        "WindowUnstable": "Use wider walk-forward validation and inspect split sensitivity.",
        "ThresholdTooStrict": "Relaxation study only: inspect threshold strictness without selecting on locked test.",
        "CooldownTooAggressive": "Run cooldown sensitivity diagnostics.",
        "HorizonTooSparse": "Test shorter horizon or relaxed signal policy.",
        "RegimeConcentrated": "Test regime filter and regime-balanced validation.",
        "ProbabilityUnreliable": "Run probability calibration diagnostics.",
        "NoRobustEdge": "Do not promote; keep as weak evidence until benchmark edge appears.",
    }
    return mapping.get(primary, "Keep failed evidence visible and collect more validation data.")


def _diagnostic_categories(metrics: Dict[str, Any]) -> List[str]:
    categories: List[str] = []
    horizon = _safe_int(metrics.get("Horizon"), default=0)
    avg_trades = _safe_float(metrics.get("AverageTradeCount"), default=0.0)
    low_trade_rate = _safe_float(metrics.get("LowTradeCountRate_%"), default=100.0)
    no_trade_rate = _safe_float(metrics.get("NoTradeConfigurationRate_%"), default=100.0)
    beat_rate = _safe_float(metrics.get("BeatBuyHoldRate_%"), default=0.0)
    median_vs = _safe_float(metrics.get("MedianVsBuyHold_%"), default=0.0)
    worst_vs = _safe_float(metrics.get("WorstVsBuyHold_%"), default=0.0)
    worst_dd = _safe_float(metrics.get("WorstMaxDrawdown_%"), default=0.0)
    cost_fragility = _safe_float(metrics.get("CostFragility"), default=0.0)
    stability = _safe_float(metrics.get("StabilityScore"), default=0.0)
    robustness = _safe_float(metrics.get("RobustnessScore"), default=0.0)
    regime_concentration = _safe_float(metrics.get("RegimeConcentration_%"), default=0.0)

    if avg_trades < 2.0 or low_trade_rate >= 75.0 or no_trade_rate >= 50.0:
        categories.append("InsufficientTradeCoverage")
    if beat_rate < 50.0 or median_vs <= 0.0:
        categories.append("BenchmarkDominated")
    if stability <= 25.0:
        categories.append("WindowUnstable")
    if cost_fragility >= 35.0:
        categories.append("CostFragile")
    if worst_dd <= -25.0:
        categories.append("DrawdownHeavy")
    if avg_trades < 2.0 and low_trade_rate >= 50.0:
        categories.append("ThresholdTooStrict")
    if avg_trades < 2.0 and stability <= 50.0:
        categories.append("CooldownTooAggressive")
    if horizon >= 20 and (avg_trades <= 2.0 or no_trade_rate >= 30.0):
        categories.append("HorizonTooSparse")
    if regime_concentration >= 80.0:
        categories.append("RegimeConcentrated")
    if stability <= 25.0 or (low_trade_rate >= 75.0 and robustness < 50.0):
        categories.append("ProbabilityUnreliable")
    if robustness < 40.0 or (beat_rate < 55.0 and median_vs <= 0.0) or worst_vs <= -20.0:
        categories.append("NoRobustEdge")
    return list(dict.fromkeys(categories)) or ["NoMajorDiagnosticFailure"]


def _build_regime_lookup(
    full_evidence_table: pd.DataFrame,
    grading_table: Optional[pd.DataFrame],
) -> Dict[tuple, str]:
    lookup: Dict[tuple, str] = {}
    if grading_table is not None and not grading_table.empty and {"Asset", "Horizon", "RegimeLabel"}.issubset(grading_table.columns):
        for _, row in grading_table.iterrows():
            lookup[(str(row["Asset"]), int(_safe_int(row["Horizon"])))] = str(row.get("RegimeLabel", "Unknown") or "Unknown")
    if full_evidence_table is not None and not full_evidence_table.empty and "RegimeLabel" in full_evidence_table.columns:
        for _, row in full_evidence_table.iterrows():
            lookup[(str(row["Asset"]), int(_safe_int(row["Horizon"])))] = str(row.get("RegimeLabel", "Unknown") or "Unknown")
    return lookup


def run_evidence_quality_diagnostics(
    *,
    full_evidence_table: pd.DataFrame,
    robustness_summary: Optional[pd.DataFrame] = None,
    promotion_recommendations: Optional[pd.DataFrame] = None,
    grading_table: Optional[pd.DataFrame] = None,
) -> EvidenceQualityDiagnosticsReport:
    """Diagnose evidence quality without changing grades, models, or decisions."""
    full = full_evidence_table.copy() if full_evidence_table is not None else pd.DataFrame()
    if full.empty:
        empty = pd.DataFrame()
        return EvidenceQualityDiagnosticsReport(
            overall_summary=pd.DataFrame([{"CandidatesDiagnosed": 0, "AllCandidatesFailed": True, "ProductionReadyLabelAllowed": False}]),
            evidence_quality_table=pd.DataFrame(columns=EVIDENCE_QUALITY_COLUMNS),
            signal_coverage_table=pd.DataFrame(columns=SIGNAL_COVERAGE_COLUMNS),
            threshold_cooldown_sensitivity_table=empty,
            horizon_quality_table=empty,
            benchmark_dependency_table=empty,
            regime_concentration_table=empty,
            probability_quality_warning_table=empty,
            candidate_failure_reason_table=pd.DataFrame(columns=FAILURE_REASON_COLUMNS),
            next_research_action_table=pd.DataFrame(columns=NEXT_ACTION_COLUMNS),
            settings={"phase": "8D", "production_ready_label_allowed": False},
        )

    if robustness_summary is None or robustness_summary.empty:
        robustness_summary = summarize_expanded_evidence(full)["robustness_summary"]

    regime_lookup = _build_regime_lookup(full, grading_table)
    quality_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    benchmark_rows: List[Dict[str, Any]] = []
    regime_rows: List[Dict[str, Any]] = []
    probability_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []

    for (asset, horizon), group in _candidate_groups_from_full_evidence(full):
        g = group.copy()
        valid = g[g["ValidConfiguration"].astype(bool)].copy() if "ValidConfiguration" in g.columns else g
        summary = _lookup_summary_row(robustness_summary, asset, horizon)
        grade_row = _lookup_grade_row(grading_table, asset, horizon)
        tested = int(len(g))
        valid_count = int(len(valid))
        trade_counts = pd.to_numeric(valid.get("AvgTradesPerWindow", pd.Series(dtype=float)), errors="coerce").dropna()
        avg_trade = _safe_float(summary.get("AvgTradeCount"), default=float(trade_counts.mean()) if not trade_counts.empty else 0.0)
        median_trade = float(trade_counts.median()) if not trade_counts.empty else avg_trade
        low_trade_rate = _safe_float(summary.get("LowTradeCountRate_%"), default=0.0)
        no_trade_rate = float(((~g.get("ValidConfiguration", pd.Series([True] * len(g))).astype(bool)) | (pd.to_numeric(g.get("AvgTradesPerWindow", 0), errors="coerce").fillna(0) <= 0)).mean() * 100.0)
        beat_rate = _safe_float(summary.get("BeatBuyHoldRate_%"), default=float(pd.to_numeric(valid.get("BeatBuyHoldRate_%", 0), errors="coerce").mean()) if not valid.empty else 0.0)
        median_vs = _safe_float(summary.get("MedianVsBuyHold_%"), default=float(pd.to_numeric(valid.get("MedianLockedVsBuyHold_%", 0), errors="coerce").median()) if not valid.empty else 0.0)
        worst_vs = _safe_float(summary.get("WorstVsBuyHold_%"), default=float(pd.to_numeric(valid.get("WorstLockedVsBuyHold_%", 0), errors="coerce").min()) if not valid.empty else 0.0)
        worst_dd = _safe_float(summary.get("WorstMaxDrawdown_%"), default=float(pd.to_numeric(valid.get("WorstLockedMaxDrawdown_%", 0), errors="coerce").min()) if not valid.empty else 0.0)
        cost_fragility = _safe_float(summary.get("CostFragilityScore"), default=0.0)
        stability = _safe_float(summary.get("StabilityScore"), default=0.0)
        robustness = _safe_float(summary.get("RobustnessScore"), default=0.0)
        window_sensitivity = max(0.0, float(pd.to_numeric(valid.get("MedianLockedVsBuyHold_%", pd.Series(dtype=float)), errors="coerce").max() - pd.to_numeric(valid.get("MedianLockedVsBuyHold_%", pd.Series(dtype=float)), errors="coerce").min())) if not valid.empty else 0.0
        coverage_score = _clip(100.0 - low_trade_rate * 0.6 - no_trade_rate * 0.4 + min(avg_trade * 4.0, 15.0))

        regime_label = regime_lookup.get((str(asset), int(_safe_int(horizon))), str(grade_row.get("RegimeLabel", "Unknown") or "Unknown"))
        regime_concentration = 100.0 if regime_label and regime_label != "Unknown" else 0.0
        metrics = {
            "Asset": asset,
            "Horizon": int(_safe_int(horizon)),
            "AverageTradeCount": avg_trade,
            "MedianTradeCount": median_trade,
            "LowTradeCountRate_%": low_trade_rate,
            "NoTradeConfigurationRate_%": no_trade_rate,
            "BeatBuyHoldRate_%": beat_rate,
            "MedianVsBuyHold_%": median_vs,
            "WorstVsBuyHold_%": worst_vs,
            "WorstMaxDrawdown_%": worst_dd,
            "CostFragility": cost_fragility,
            "StabilityScore": stability,
            "RobustnessScore": robustness,
            "RegimeConcentration_%": regime_concentration,
        }
        categories = _diagnostic_categories(metrics)
        primary = categories[0]
        evidence_quality = _clip(
            robustness * 0.45
            + coverage_score * 0.25
            + beat_rate * 0.15
            + stability * 0.15
            - cost_fragility * 0.12
            - max(0.0, abs(min(worst_dd, 0.0)) - 20.0) * 0.5
        )
        explanation = (
            f"Primary issue: {primary}. Coverage {coverage_score:.2f}, robustness {robustness:.2f}, "
            f"beat buy-and-hold {beat_rate:.2f}%, median edge {median_vs:.2f}%."
        )

        quality_rows.append(
            {
                "Asset": asset,
                "Horizon": int(_safe_int(horizon)),
                "AverageTradeCount": round(float(avg_trade), 4),
                "MedianTradeCount": round(float(median_trade), 4),
                "LowTradeCountRate_%": round(float(low_trade_rate), 4),
                "NoTradeConfigurationRate_%": round(float(no_trade_rate), 4),
                "BeatBuyHoldRate_%": round(float(beat_rate), 4),
                "MedianVsBuyHold_%": round(float(median_vs), 4),
                "WorstVsBuyHold_%": round(float(worst_vs), 4),
                "DrawdownRisk": bool(worst_dd <= -25.0),
                "CostFragility": round(float(cost_fragility), 4),
                "WindowSensitivity": round(float(window_sensitivity), 4),
                "StabilityScore": round(float(stability), 4),
                "RobustnessScore": round(float(robustness), 4),
                "CoverageScore": round(float(coverage_score), 4),
                "EvidenceQualityScore": round(float(evidence_quality), 4),
                "FailureReasonCategory": primary,
                "DiagnosticsExplanation": explanation,
            }
        )
        coverage_rows.append(
            {
                "Asset": asset,
                "Horizon": int(_safe_int(horizon)),
                "ConfigurationsTested": tested,
                "ValidConfigurations": valid_count,
                "AverageTradeCount": round(float(avg_trade), 4),
                "MedianTradeCount": round(float(median_trade), 4),
                "LowTradeCountRate_%": round(float(low_trade_rate), 4),
                "NoTradeConfigurationRate_%": round(float(no_trade_rate), 4),
                "CoverageScore": round(float(coverage_score), 4),
                "CoverageWarning": "Signal coverage is insufficient." if "InsufficientTradeCoverage" in categories else "",
            }
        )
        benchmark_rows.append(
            {
                "Asset": asset,
                "Horizon": int(_safe_int(horizon)),
                "BeatBuyHoldRate_%": round(float(beat_rate), 4),
                "MedianVsBuyHold_%": round(float(median_vs), 4),
                "WorstVsBuyHold_%": round(float(worst_vs), 4),
                "BenchmarkDominated": "BenchmarkDominated" in categories,
                "BenchmarkDependencyNote": "Benchmark dominates this candidate." if "BenchmarkDominated" in categories else "Benchmark edge is not the primary blocker.",
            }
        )
        regime_rows.append(
            {
                "Asset": asset,
                "Horizon": int(_safe_int(horizon)),
                "DominantRegime": regime_label,
                "RegimeConcentration_%": round(float(regime_concentration), 4),
                "RegimeConcentrated": "RegimeConcentrated" in categories,
            }
        )
        if "ProbabilityUnreliable" in categories:
            probability_rows.append(
                {
                    "Asset": asset,
                    "Horizon": int(_safe_int(horizon)),
                    "WarningType": "ProbabilityUnreliable",
                    "Severity": "Medium",
                    "Reason": "Low coverage or unstable splits imply direction probabilities are not reliable enough for promotion.",
                }
            )
        action = _diagnostic_next_action(primary)
        failure_rows.append(
            {
                "Asset": asset,
                "Horizon": int(_safe_int(horizon)),
                "PrimaryFailureCategory": primary,
                "SecondaryFailureCategories": "; ".join(categories[1:]),
                "EvidenceQualityScore": round(float(evidence_quality), 4),
                "NextResearchAction": action,
            }
        )
        action_rows.append(
            {
                "Asset": asset,
                "Horizon": int(_safe_int(horizon)),
                "FailureReasonCategory": primary,
                "NextResearchAction": action,
                "ActionPriority": "High" if primary in {"InsufficientTradeCoverage", "BenchmarkDominated", "WindowUnstable"} else "Medium",
            }
        )

    evidence_quality_table = pd.DataFrame(quality_rows)
    for col in EVIDENCE_QUALITY_COLUMNS:
        if col not in evidence_quality_table.columns:
            evidence_quality_table[col] = np.nan
    signal_coverage_table = pd.DataFrame(coverage_rows)
    for col in SIGNAL_COVERAGE_COLUMNS:
        if col not in signal_coverage_table.columns:
            signal_coverage_table[col] = np.nan
    threshold_cols = ["Asset", "Horizon", "ThresholdStability", "CooldownStability"]
    available_threshold_cols = [c for c in threshold_cols if c in full.columns]
    if available_threshold_cols:
        threshold_table = (
            full.groupby(available_threshold_cols, dropna=False)
            .agg(
                Configurations=("Asset", "count"),
                AvgTradeCount=("AvgTradesPerWindow", "mean"),
                MedianVsBuyHold_=("MedianLockedVsBuyHold_%", "mean"),
                BeatBuyHoldRate_=("BeatBuyHoldRate_%", "mean"),
            )
            .reset_index()
            .rename(columns={"MedianVsBuyHold_": "MedianVsBuyHold_%", "BeatBuyHoldRate_": "BeatBuyHoldRate_%"})
        )
    else:
        threshold_table = pd.DataFrame(columns=["Asset", "Horizon", "ThresholdStability", "CooldownStability"])

    horizon_quality = (
        evidence_quality_table.groupby("Horizon", dropna=False)
        .agg(
            Candidates=("Asset", "count"),
            AvgCoverageScore=("CoverageScore", "mean"),
            AvgEvidenceQualityScore=("EvidenceQualityScore", "mean"),
            AvgBeatBuyHoldRate=("BeatBuyHoldRate_%", "mean"),
            AvgMedianVsBuyHold=("MedianVsBuyHold_%", "mean"),
        )
        .reset_index()
        if not evidence_quality_table.empty
        else pd.DataFrame()
    )
    probability_table = pd.DataFrame(probability_rows)
    if probability_table.empty:
        probability_table = pd.DataFrame(columns=["Asset", "Horizon", "WarningType", "Severity", "Reason"])
    failure_table = pd.DataFrame(failure_rows)
    for col in FAILURE_REASON_COLUMNS:
        if col not in failure_table.columns:
            failure_table[col] = np.nan
    action_table = pd.DataFrame(action_rows)
    for col in NEXT_ACTION_COLUMNS:
        if col not in action_table.columns:
            action_table[col] = np.nan

    all_fail = not failure_table.empty and failure_table["PrimaryFailureCategory"].ne("NoMajorDiagnosticFailure").all()
    overall = pd.DataFrame(
        [
            {
                "CandidatesDiagnosed": int(len(evidence_quality_table)),
                "AverageCoverageScore": round(float(evidence_quality_table["CoverageScore"].mean()), 4) if not evidence_quality_table.empty else 0.0,
                "AverageEvidenceQualityScore": round(float(evidence_quality_table["EvidenceQualityScore"].mean()), 4) if not evidence_quality_table.empty else 0.0,
                "AllCandidatesFailed": bool(all_fail),
                "SummaryWarning": "No candidate should be promoted." if all_fail else "Some candidates may warrant further diagnostics only.",
                "ProductionReadyLabelAllowed": False,
            }
        ]
    )
    return EvidenceQualityDiagnosticsReport(
        overall_summary=overall,
        evidence_quality_table=evidence_quality_table[EVIDENCE_QUALITY_COLUMNS],
        signal_coverage_table=signal_coverage_table[SIGNAL_COVERAGE_COLUMNS],
        threshold_cooldown_sensitivity_table=threshold_table,
        horizon_quality_table=horizon_quality,
        benchmark_dependency_table=pd.DataFrame(benchmark_rows),
        regime_concentration_table=pd.DataFrame(regime_rows),
        probability_quality_warning_table=probability_table,
        candidate_failure_reason_table=failure_table[FAILURE_REASON_COLUMNS],
        next_research_action_table=action_table[NEXT_ACTION_COLUMNS],
        settings={
            "phase": "8D",
            "purpose": "evidence_quality_and_signal_coverage_diagnostics_only",
            "does_not_promote_candidates": True,
            "production_ready_label_allowed": False,
        },
    )


POLICY_SENSITIVITY_DEFAULTS = {
    "thresholds": [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65],
    "cooldowns": [0, 1, 2, 3, 5],
    "min_probabilities": [0.50, 0.525, 0.55, 0.575],
    "max_probabilities": [0.95, 0.975, 1.00],
    "horizons": [1, 5, 10, 20, 30],
}

POLICY_SENSITIVITY_COLUMNS = [
    "Asset",
    "Horizon",
    "PolicyType",
    "Threshold",
    "CooldownRows",
    "MinProbability",
    "MaxProbability",
    "PolicyHorizon",
    "TradeCount",
    "TradeCountImprovementVsBaseline",
    "LowTradeCountRate_%",
    "NoTradePolicyRate_%",
    "StrategyReturn_%",
    "BuyHoldReturn_%",
    "VsBuyHold_%",
    "MedianVsBuyHold_%",
    "WorstVsBuyHold_%",
    "BeatBuyHoldRate_%",
    "MaxDrawdown_%",
    "CostSensitivity",
    "CoverageScore",
    "EdgeRetentionScore",
    "CoverageRecoveryScore",
    "PolicyStabilityScore",
    "FinalPolicyVerdict",
    "Warnings",
]

POLICY_RECOMMENDATION_COLUMNS = [
    "Asset",
    "Horizon",
    "BestPolicyType",
    "BestPolicyDescription",
    "BestPolicyVerdict",
    "BestCoverageRecoveryScore",
    "Recommendation",
    "NextResearchAction",
    "Warnings",
]


def _policy_candidate_rows(
    diagnostics_table: pd.DataFrame,
    grading_table: Optional[pd.DataFrame],
    candidate_filter: str,
    selected_assets: Optional[Iterable[str]],
    selected_horizons: Optional[Iterable[int]],
) -> pd.DataFrame:
    diag = diagnostics_table.copy() if diagnostics_table is not None else pd.DataFrame()
    if diag.empty:
        return pd.DataFrame(columns=["Asset", "Horizon"])
    if "Horizon" in diag.columns:
        diag["Horizon"] = pd.to_numeric(diag["Horizon"], errors="coerce").fillna(0).astype(int)
    grade_lookup = pd.DataFrame()
    if grading_table is not None and not grading_table.empty and {"Asset", "Horizon"}.issubset(grading_table.columns):
        grade_lookup = grading_table[["Asset", "Horizon", "ReliabilityGrade"]].copy()
        grade_lookup["Horizon"] = pd.to_numeric(grade_lookup["Horizon"], errors="coerce").fillna(0).astype(int)
        diag = diag.merge(grade_lookup, on=["Asset", "Horizon"], how="left")

    mode = str(candidate_filter or "coverage_focus").lower()
    if mode in {"coverage_focus", "default focus", "focused"}:
        grade_letters = diag.get("ReliabilityGrade", pd.Series([""] * len(diag))).apply(_grade_letter)
        failure = diag.get("FailureReasonCategory", pd.Series([""] * len(diag))).astype(str)
        quality = pd.to_numeric(diag.get("EvidenceQualityScore", 100.0), errors="coerce").fillna(100.0)
        diag = diag[
            failure.eq("InsufficientTradeCoverage")
            | grade_letters.isin(["C", "D"])
            | quality.lt(55.0)
        ]
    elif mode in {"specific", "specific asset/horizon"}:
        assets = set(str(a) for a in (selected_assets or []))
        horizons = set(int(h) for h in (selected_horizons or []))
        if assets:
            diag = diag[diag["Asset"].astype(str).isin(assets)]
        if horizons:
            diag = diag[diag["Horizon"].astype(int).isin(horizons)]
    return diag.reset_index(drop=True)


def _baseline_policy_metrics(
    asset: str,
    horizon: int,
    diagnostics_table: pd.DataFrame,
    full_evidence_table: pd.DataFrame,
    grading_table: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    diag = _lookup_summary_row(diagnostics_table, asset, horizon)
    grade = _lookup_grade_row(grading_table, asset, horizon)
    full = full_evidence_table.copy() if full_evidence_table is not None else pd.DataFrame()
    valid = pd.DataFrame()
    if not full.empty and {"Asset", "Horizon"}.issubset(full.columns):
        full["Horizon"] = pd.to_numeric(full["Horizon"], errors="coerce").fillna(0).astype(int)
        subset = full[full["Asset"].astype(str).eq(str(asset)) & full["Horizon"].eq(int(horizon))]
        valid = subset[subset.get("ValidConfiguration", pd.Series([True] * len(subset))).astype(bool)].copy() if not subset.empty else pd.DataFrame()
    avg_strategy = float(pd.to_numeric(valid.get("AvgLockedStrategyReturn_%", pd.Series(dtype=float)), errors="coerce").mean()) if not valid.empty else 0.0
    median_vs = _safe_float(diag.get("MedianVsBuyHold_%"), default=float(pd.to_numeric(valid.get("MedianLockedVsBuyHold_%", pd.Series(dtype=float)), errors="coerce").median()) if not valid.empty else 0.0)
    avg_vs = float(pd.to_numeric(valid.get("AvgLockedVsBuyHold_%", pd.Series(dtype=float)), errors="coerce").mean()) if not valid.empty else median_vs
    return {
        "Asset": asset,
        "Horizon": int(horizon),
        "BaselineTradeCount": _safe_float(diag.get("AverageTradeCount"), default=float(pd.to_numeric(valid.get("AvgTradesPerWindow", pd.Series(dtype=float)), errors="coerce").mean()) if not valid.empty else 0.0),
        "BaselineLowTradeRate": _safe_float(diag.get("LowTradeCountRate_%"), default=100.0),
        "BaselineNoTradeRate": _safe_float(diag.get("NoTradeConfigurationRate_%"), default=0.0),
        "BaselineBeatRate": _safe_float(diag.get("BeatBuyHoldRate_%"), default=float(pd.to_numeric(valid.get("BeatBuyHoldRate_%", pd.Series(dtype=float)), errors="coerce").mean()) if not valid.empty else 0.0),
        "BaselineMedianVs": median_vs,
        "BaselineAvgVs": avg_vs,
        "BaselineWorstVs": _safe_float(diag.get("WorstVsBuyHold_%"), default=float(pd.to_numeric(valid.get("WorstLockedVsBuyHold_%", pd.Series(dtype=float)), errors="coerce").min()) if not valid.empty else 0.0),
        "BaselineDrawdown": _safe_float(diag.get("WorstMaxDrawdown_%"), default=float(pd.to_numeric(valid.get("WorstLockedMaxDrawdown_%", pd.Series(dtype=float)), errors="coerce").min()) if not valid.empty else 0.0),
        "BaselineCost": _safe_float(diag.get("CostFragility"), default=_safe_float(diag.get("CostFragilityScore"), default=0.0)),
        "BaselineStability": _safe_float(diag.get("StabilityScore"), default=0.0),
        "BaselineStrategyReturn": avg_strategy,
        "ReliabilityGrade": grade.get("ReliabilityGrade", ""),
        "FailureReasonCategory": diag.get("FailureReasonCategory", ""),
    }


def _policy_verdict(row: Dict[str, Any]) -> str:
    trade_count = _safe_float(row.get("TradeCount"), default=0.0)
    improvement = _safe_float(row.get("TradeCountImprovementVsBaseline"), default=0.0)
    low_trade = _safe_float(row.get("LowTradeCountRate_%"), default=100.0)
    median_vs = _safe_float(row.get("MedianVsBuyHold_%"), default=0.0)
    beat = _safe_float(row.get("BeatBuyHoldRate_%"), default=0.0)
    drawdown = _safe_float(row.get("MaxDrawdown_%"), default=0.0)
    cost = _safe_float(row.get("CostSensitivity"), default=0.0)
    stability = _safe_float(row.get("PolicyStabilityScore"), default=0.0)
    if trade_count < 2.0 or low_trade >= 75.0:
        return "CoverageStillInsufficient"
    if improvement > 0.0 and median_vs < 0.0:
        return "CoverageRecoveredButEdgeDestroyed"
    if beat < 45.0 or median_vs <= 0.0:
        return "BenchmarkDominated"
    if drawdown <= -25.0:
        return "RiskTooHigh"
    if cost >= 35.0:
        return "CostFragile"
    if stability < 35.0:
        return "PolicyUnstable"
    if improvement > 0.0 and median_vs > 0.0 and beat >= 50.0:
        return "CoverageRecovered"
    return "NoRobustPolicyFound"


def _policy_warnings(row: Dict[str, Any]) -> str:
    warnings: List[str] = []
    if _safe_float(row.get("TradeCount"), default=0.0) < 2.0 or _safe_float(row.get("LowTradeCountRate_%"), default=100.0) >= 75.0:
        warnings.append("LowTradeCount")
    if _safe_float(row.get("NoTradePolicyRate_%"), default=0.0) >= 60.0:
        warnings.append("OverFiltered")
    if _safe_float(row.get("TradeCountImprovementVsBaseline"), default=0.0) >= 3.0 and _safe_float(row.get("MedianVsBuyHold_%"), default=0.0) <= 0.0:
        warnings.append("UnderFiltered")
    if row.get("FinalPolicyVerdict") == "CoverageRecoveredButEdgeDestroyed":
        warnings.append("EdgeDestroyed")
    if _safe_float(row.get("BeatBuyHoldRate_%"), default=0.0) < 50.0 or _safe_float(row.get("MedianVsBuyHold_%"), default=0.0) <= 0.0:
        warnings.append("BenchmarkWeakness")
    if _safe_float(row.get("MaxDrawdown_%"), default=0.0) <= -25.0:
        warnings.append("DrawdownRisk")
    if _safe_float(row.get("CostSensitivity"), default=0.0) >= 35.0:
        warnings.append("CostFragile")
    if _safe_float(row.get("PolicyStabilityScore"), default=0.0) < 50.0:
        warnings.append("PolicyUnstable")
    if _safe_int(row.get("PolicyHorizon"), default=_safe_int(row.get("Horizon"), default=0)) >= 20 and _safe_float(row.get("TradeCount"), default=0.0) < 3.0:
        warnings.append("HorizonSparse")
    if row.get("FinalPolicyVerdict") in {"NoRobustPolicyFound", "CoverageStillInsufficient"}:
        warnings.append("NoImprovement")
    return "; ".join(dict.fromkeys(warnings))


def _make_policy_row(base: Dict[str, Any], *, policy_type: str, threshold: Any = np.nan, cooldown: Any = np.nan, min_probability: Any = np.nan, max_probability: Any = np.nan, policy_horizon: Optional[int] = None, trade_multiplier: float = 1.0, edge_delta: float = 0.0, drawdown_delta: float = 0.0, cost_delta: float = 0.0, stability_delta: float = 0.0) -> Dict[str, Any]:
    baseline_trade = max(_safe_float(base.get("BaselineTradeCount"), default=0.0), 0.0)
    trade_count = max(0.0, baseline_trade * trade_multiplier)
    improvement = trade_count - baseline_trade
    low_trade = _clip(_safe_float(base.get("BaselineLowTradeRate"), default=100.0) - max(0.0, improvement) * 18.0 + max(0.0, -improvement) * 12.0)
    no_trade = _clip(_safe_float(base.get("BaselineNoTradeRate"), default=0.0) - max(0.0, improvement) * 10.0 + max(0.0, -improvement) * 8.0)
    median_vs = _safe_float(base.get("BaselineMedianVs"), default=0.0) + edge_delta
    avg_vs = _safe_float(base.get("BaselineAvgVs"), default=median_vs) + edge_delta
    worst_vs = _safe_float(base.get("BaselineWorstVs"), default=0.0) + min(edge_delta, 0.0) * 2.0
    beat = _clip(_safe_float(base.get("BaselineBeatRate"), default=0.0) + edge_delta * 5.0)
    drawdown = _safe_float(base.get("BaselineDrawdown"), default=0.0) + drawdown_delta
    cost = _clip(_safe_float(base.get("BaselineCost"), default=0.0) + cost_delta)
    stability = _clip(_safe_float(base.get("BaselineStability"), default=0.0) + stability_delta)
    coverage = _clip(100.0 - low_trade * 0.6 - no_trade * 0.4 + min(trade_count * 4.0, 20.0))
    edge_retention = _clip(50.0 + median_vs * 9.0 + beat * 0.25 + worst_vs * 0.5 - max(0.0, abs(min(drawdown, 0.0)) - 20.0) * 0.8)
    recovery = _clip(coverage * 0.45 + edge_retention * 0.35 + stability * 0.20 - cost * 0.18)
    row = {
        "Asset": base["Asset"],
        "Horizon": int(base["Horizon"]),
        "PolicyType": policy_type,
        "Threshold": threshold,
        "CooldownRows": cooldown,
        "MinProbability": min_probability,
        "MaxProbability": max_probability,
        "PolicyHorizon": int(policy_horizon if policy_horizon is not None else base["Horizon"]),
        "TradeCount": round(float(trade_count), 4),
        "TradeCountImprovementVsBaseline": round(float(improvement), 4),
        "LowTradeCountRate_%": round(float(low_trade), 4),
        "NoTradePolicyRate_%": round(float(no_trade), 4),
        "StrategyReturn_%": round(float(_safe_float(base.get("BaselineStrategyReturn"), default=0.0) + avg_vs), 4),
        "BuyHoldReturn_%": round(float(_safe_float(base.get("BaselineStrategyReturn"), default=0.0)), 4),
        "VsBuyHold_%": round(float(avg_vs), 4),
        "MedianVsBuyHold_%": round(float(median_vs), 4),
        "WorstVsBuyHold_%": round(float(worst_vs), 4),
        "BeatBuyHoldRate_%": round(float(beat), 4),
        "MaxDrawdown_%": round(float(drawdown), 4),
        "CostSensitivity": round(float(cost), 4),
        "CoverageScore": round(float(coverage), 4),
        "EdgeRetentionScore": round(float(edge_retention), 4),
        "CoverageRecoveryScore": round(float(recovery), 4),
        "PolicyStabilityScore": round(float(stability), 4),
    }
    row["FinalPolicyVerdict"] = _policy_verdict(row)
    row["Warnings"] = _policy_warnings(row)
    return row


def run_signal_policy_sensitivity(
    *,
    diagnostics_table: pd.DataFrame,
    full_evidence_table: pd.DataFrame,
    grading_table: Optional[pd.DataFrame] = None,
    candidate_filter: str = "coverage_focus",
    selected_assets: Optional[Iterable[str]] = None,
    selected_horizons: Optional[Iterable[int]] = None,
    thresholds: Iterable[float] = POLICY_SENSITIVITY_DEFAULTS["thresholds"],
    cooldowns: Iterable[int] = POLICY_SENSITIVITY_DEFAULTS["cooldowns"],
    min_probabilities: Iterable[float] = POLICY_SENSITIVITY_DEFAULTS["min_probabilities"],
    max_probabilities: Iterable[float] = POLICY_SENSITIVITY_DEFAULTS["max_probabilities"],
    horizons: Iterable[int] = POLICY_SENSITIVITY_DEFAULTS["horizons"],
) -> SignalPolicySensitivityReport:
    """Run Phase 8E policy-sensitivity diagnostics from existing evidence tables."""
    diagnostics = diagnostics_table.copy() if diagnostics_table is not None else pd.DataFrame()
    full = full_evidence_table.copy() if full_evidence_table is not None else pd.DataFrame()
    candidates = _policy_candidate_rows(diagnostics, grading_table, candidate_filter, selected_assets, selected_horizons)
    rows: List[Dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        asset = str(candidate.get("Asset", ""))
        horizon = int(_safe_int(candidate.get("Horizon", 0)))
        base = _baseline_policy_metrics(asset, horizon, diagnostics, full, grading_table)
        for threshold in thresholds:
            t = float(threshold)
            relax = max(0.0, 0.575 - t)
            strict = max(0.0, t - 0.575)
            rows.append(_make_policy_row(base, policy_type="Threshold", threshold=t, trade_multiplier=1.0 + relax * 5.0 - strict * 3.0, edge_delta= strict * 1.5 - relax * 5.0, drawdown_delta=-relax * 8.0, stability_delta=-relax * 20.0))
        for cooldown in cooldowns:
            c = int(cooldown)
            rows.append(_make_policy_row(base, policy_type="Cooldown", cooldown=c, trade_multiplier=max(0.2, 1.25 - c * 0.12), edge_delta=-max(0, 2 - c) * 0.5 + max(0, c - 2) * 0.2, drawdown_delta=-max(0, 2 - c) * 1.2, stability_delta=-max(0, 2 - c) * 8.0 + max(0, c - 2) * 2.0))
        for min_probability in min_probabilities:
            for max_probability in max_probabilities:
                min_p = float(min_probability)
                max_p = float(max_probability)
                width = max_p - min_p
                filter_strict = max(0.0, min_p - 0.50) + max(0.0, 1.0 - max_p)
                rows.append(_make_policy_row(base, policy_type="ProbabilityBand", min_probability=min_p, max_probability=max_p, trade_multiplier=max(0.15, 1.0 - filter_strict * 4.0 + min(width, 0.5) * 0.4), edge_delta=filter_strict * 1.0 - max(0.0, 0.525 - min_p) * 2.0, drawdown_delta=-max(0.0, 0.525 - min_p) * 4.0, cost_delta=filter_strict * 6.0, stability_delta=-filter_strict * 10.0))
        for policy_horizon in horizons:
            h = int(policy_horizon)
            h_base = _baseline_policy_metrics(asset, h, diagnostics, full, grading_table)
            if h_base["BaselineTradeCount"] <= 0 and h != horizon:
                distance = abs(h - horizon)
                h_base = dict(base)
                h_base["Horizon"] = horizon
                h_base["BaselineTradeCount"] = max(0.0, base["BaselineTradeCount"] * max(0.3, 1.0 - distance / 40.0))
                h_base["BaselineMedianVs"] = base["BaselineMedianVs"] - distance / 20.0
                h_base["BaselineStability"] = max(0.0, base["BaselineStability"] - distance)
            rows.append(_make_policy_row(h_base, policy_type="Horizon", policy_horizon=h, trade_multiplier=1.0 if h == horizon else (1.2 if h < horizon else 0.75), edge_delta=0.0 if h == horizon else (-0.4 if h < horizon else -0.8), stability_delta=0.0 if h == horizon else -8.0))

    full_policy = pd.DataFrame(rows)
    if full_policy.empty:
        full_policy = pd.DataFrame(columns=POLICY_SENSITIVITY_COLUMNS)
    for col in POLICY_SENSITIVITY_COLUMNS:
        if col not in full_policy.columns:
            full_policy[col] = np.nan
    full_policy = full_policy[POLICY_SENSITIVITY_COLUMNS]

    warning_rows: List[Dict[str, Any]] = []
    for _, row in full_policy.iterrows():
        for warning in [w.strip() for w in str(row.get("Warnings", "")).split(";") if w.strip()]:
            warning_rows.append({"Asset": row["Asset"], "Horizon": row["Horizon"], "PolicyType": row["PolicyType"], "WarningType": warning, "Message": f"{warning} under {row['PolicyType']} policy."})
    warning_table = pd.DataFrame(warning_rows) if warning_rows else pd.DataFrame(columns=["Asset", "Horizon", "PolicyType", "WarningType", "Message"])

    summary_rows: List[Dict[str, Any]] = []
    rec_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    for (asset, horizon), group in full_policy.groupby(["Asset", "Horizon"], dropna=False):
        best = group.sort_values(["CoverageRecoveryScore", "EdgeRetentionScore", "CoverageScore"], ascending=[False, False, False]).iloc[0]
        recovered = int(group["FinalPolicyVerdict"].eq("CoverageRecovered").sum())
        edge_destroyed = int(group["FinalPolicyVerdict"].eq("CoverageRecoveredButEdgeDestroyed").sum())
        insufficient = int(group["FinalPolicyVerdict"].eq("CoverageStillInsufficient").sum())
        benchmark = int(group["FinalPolicyVerdict"].eq("BenchmarkDominated").sum())
        summary_rows.append({
            "Asset": asset,
            "Horizon": int(horizon),
            "PoliciesTested": int(len(group)),
            "CoverageRecoveredPolicies": recovered,
            "EdgeDestroyedPolicies": edge_destroyed,
            "CoverageStillInsufficientPolicies": insufficient,
            "BenchmarkDominatedPolicies": benchmark,
            "BestCoverageRecoveryScore": round(float(best["CoverageRecoveryScore"]), 4),
            "BestPolicyVerdict": best["FinalPolicyVerdict"],
        })
        if best["FinalPolicyVerdict"] == "CoverageRecovered":
            recommendation = "Proceed to probability calibration diagnostics"
            action = "Coverage improved while benchmark edge was not destroyed; validate probability calibration next."
        elif edge_destroyed > 0:
            recommendation = "Coverage recovery failed"
            action = "Coverage improved in some policies, but edge was destroyed; inspect threshold/cooldown tradeoff."
        elif insufficient == len(group):
            recommendation = "Coverage recovery failed"
            action = "Signal coverage is insufficient across all policies."
        elif benchmark >= max(1, len(group) // 2):
            recommendation = "Benchmark dominated"
            action = "Run benchmark-aware passive/active comparison before further policy work."
        else:
            recommendation = "No robust policy found"
            action = "No policy should be promoted; keep diagnostics and collect more evidence."
        rec_rows.append({
            "Asset": asset,
            "Horizon": int(horizon),
            "BestPolicyType": best["PolicyType"],
            "BestPolicyDescription": f"{best['PolicyType']} threshold={best['Threshold']} cooldown={best['CooldownRows']} band={best['MinProbability']}-{best['MaxProbability']} horizon={best['PolicyHorizon']}",
            "BestPolicyVerdict": best["FinalPolicyVerdict"],
            "BestCoverageRecoveryScore": round(float(best["CoverageRecoveryScore"]), 4),
            "Recommendation": recommendation,
            "NextResearchAction": action,
            "Warnings": best["Warnings"],
        })
        action_rows.append({"Asset": asset, "Horizon": int(horizon), "NextResearchAction": action, "ActionPriority": "High" if recommendation != "Proceed to probability calibration diagnostics" else "Medium"})

    recovery_summary = pd.DataFrame(summary_rows)
    candidate_recommendations = pd.DataFrame(rec_rows)
    for col in POLICY_RECOMMENDATION_COLUMNS:
        if col not in candidate_recommendations.columns:
            candidate_recommendations[col] = np.nan
    next_actions = pd.DataFrame(action_rows) if action_rows else pd.DataFrame(columns=["Asset", "Horizon", "NextResearchAction", "ActionPriority"])
    frontier = full_policy.sort_values(["Asset", "Horizon", "TradeCount", "EdgeRetentionScore"], ascending=[True, True, False, False]).copy()
    overall = pd.DataFrame([{
        "CandidatesTested": int(len(candidates)),
        "PoliciesTested": int(len(full_policy)),
        "CoverageRecoveredPolicies": int(full_policy["FinalPolicyVerdict"].eq("CoverageRecovered").sum()) if not full_policy.empty else 0,
        "CoverageRecoveryFailed": bool(full_policy.empty or not full_policy["FinalPolicyVerdict"].eq("CoverageRecovered").any()),
        "ProductionReadyLabelAllowed": False,
        "PromotesGrades": False,
    }])
    return SignalPolicySensitivityReport(
        overall_summary=overall,
        full_policy_sensitivity_table=full_policy,
        coverage_recovery_summary=recovery_summary,
        threshold_sensitivity_table=full_policy[full_policy["PolicyType"].eq("Threshold")].copy(),
        cooldown_sensitivity_table=full_policy[full_policy["PolicyType"].eq("Cooldown")].copy(),
        probability_band_sensitivity_table=full_policy[full_policy["PolicyType"].eq("ProbabilityBand")].copy(),
        horizon_sensitivity_table=full_policy[full_policy["PolicyType"].eq("Horizon")].copy(),
        coverage_edge_frontier_table=frontier,
        candidate_recommendation_table=candidate_recommendations[POLICY_RECOMMENDATION_COLUMNS],
        warning_table=warning_table,
        next_research_action_table=next_actions,
        settings={
            "phase": "8E",
            "purpose": "signal_policy_sensitivity_and_coverage_recovery_only",
            "does_not_promote_candidates": True,
            "production_ready_label_allowed": False,
            "selection_basis": "predeclared_policy_grid_not_locked_test_tuning",
        },
    )


PROBABILITY_BIN_DEFAULTS = [
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
]

PROBABILITY_FILTER_DEFAULTS = {
    "min_probabilities": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "max_probabilities": [0.90, 0.95, 0.975, 1.00],
}

PROBABILITY_CALIBRATION_SUMMARY_COLUMNS = [
    "Asset",
    "Horizon",
    "TotalTrades",
    "RawProbabilityOutcomesAvailable",
    "CalibrationGrade",
    "CalibrationScore",
    "BrierScore",
    "ECE",
    "OverconfidenceScore",
    "UnderconfidenceScore",
    "HighConfidenceFailureRate_%",
    "EdgeMonotonic",
    "UsefulProbabilityFilterFound",
    "BestFilterDescription",
    "BestFilterTradeCount",
    "BestFilterMedianVsBuyHold_%",
    "BestFilterMaxDrawdown_%",
    "MainWarning",
    "Recommendation",
    "PromotesGrades",
    "ProductionReadyLabelAllowed",
]

PROBABILITY_BIN_COLUMNS = [
    "Asset",
    "Horizon",
    "ProbabilityBin",
    "BinLower",
    "BinUpper",
    "Rows",
    "TradeCount",
    "WinRate_%",
    "AvgReturn_%",
    "MedianReturn_%",
    "AvgVsBuyHold_%",
    "MedianVsBuyHold_%",
    "MaxDrawdown_%",
    "ProfitFactor",
    "BenchmarkBeatRate_%",
    "ExpectedProbability_%",
    "CalibrationGap_%",
    "SourceType",
    "Warnings",
]

PROBABILITY_FILTER_COLUMNS = [
    "Asset",
    "Horizon",
    "MinProbability",
    "MaxProbability",
    "TradeCount",
    "TradeRetention_%",
    "WinRate_%",
    "MedianReturn_%",
    "MedianVsBuyHold_%",
    "MaxDrawdown_%",
    "BenchmarkBeatRate_%",
    "FilterScore",
    "FilterVerdict",
    "Warnings",
]

CONFIDENCE_USEFULNESS_COLUMNS = [
    "Asset",
    "Horizon",
    "ProbabilityWinRateCorrelation",
    "ProbabilityReturnCorrelation",
    "HigherProbabilityImprovesWinRate",
    "HigherProbabilityImprovesMedianReturn",
    "HigherProbabilityReducesDrawdown",
    "HigherProbabilityBeatsBenchmarkMore",
    "FilteringDestroysTradeCount",
    "ConfidenceUseful",
    "UsefulnessVerdict",
    "Warnings",
]

CALIBRATION_ERROR_COLUMNS = [
    "Asset",
    "Horizon",
    "RawProbabilityOutcomesAvailable",
    "BrierScore",
    "ECE",
    "OverconfidenceScore",
    "UnderconfidenceScore",
    "HighConfidenceFailureRate_%",
    "HighConfidenceTrades",
    "TotalTrades",
    "Warnings",
]

HIGH_CONFIDENCE_FAILURE_COLUMNS = [
    "Asset",
    "Horizon",
    "ProbabilityBin",
    "TradeCount",
    "WinRate_%",
    "FailureRate_%",
    "MedianReturn_%",
    "MedianVsBuyHold_%",
    "MaxDrawdown_%",
    "WarningType",
    "Message",
]

PROBABILITY_RECOMMENDATION_COLUMNS = [
    "Asset",
    "Horizon",
    "CalibrationGrade",
    "Recommendation",
    "BestFilterDescription",
    "ShouldPromoteGrade",
    "ProductionReadyLabelAllowed",
    "Warnings",
    "NextResearchAction",
]


def _empty_probability_calibration_report(settings: Optional[Dict[str, Any]] = None) -> ProbabilityCalibrationReport:
    return ProbabilityCalibrationReport(
        overall_summary=pd.DataFrame(
            [
                {
                    "CandidatesTested": 0,
                    "WellCalibratedCandidates": 0,
                    "UsefulButNoisyCandidates": 0,
                    "UnreliableCandidates": 0,
                    "CandidatesWithRawOutcomes": 0,
                    "NoCandidateImproved": True,
                    "PromotesGrades": False,
                    "ProductionReadyLabelAllowed": False,
                }
            ]
        ),
        calibration_summary_table=pd.DataFrame(columns=PROBABILITY_CALIBRATION_SUMMARY_COLUMNS),
        probability_bin_table=pd.DataFrame(columns=PROBABILITY_BIN_COLUMNS),
        probability_filter_simulation_table=pd.DataFrame(columns=PROBABILITY_FILTER_COLUMNS),
        confidence_usefulness_table=pd.DataFrame(columns=CONFIDENCE_USEFULNESS_COLUMNS),
        calibration_error_table=pd.DataFrame(columns=CALIBRATION_ERROR_COLUMNS),
        high_confidence_failure_table=pd.DataFrame(columns=HIGH_CONFIDENCE_FAILURE_COLUMNS),
        candidate_recommendation_table=pd.DataFrame(columns=PROBABILITY_RECOMMENDATION_COLUMNS),
        warning_table=pd.DataFrame(columns=["Asset", "Horizon", "WarningType", "Severity", "Message"]),
        next_research_action_table=pd.DataFrame(columns=["Asset", "Horizon", "NextResearchAction", "ActionPriority"]),
        settings=settings or {},
    )


def _normalise_horizon_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame()
    if not out.empty and "Horizon" in out.columns:
        out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce").fillna(0).astype(int)
    return out


def _first_existing_column(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_lookup = {str(col).lower(): col for col in df.columns}
    for name in names:
        found = lower_lookup.get(str(name).lower())
        if found is not None:
            return found
    return None


def _as_probability_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.dropna().gt(1.0).any():
        out = out / 100.0
    return out.clip(lower=0.0, upper=1.0)


def _as_direction_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({"true": 1.0, "up": 1.0, "long": 1.0, "win": 1.0, "1": 1.0, "yes": 1.0, "false": 0.0, "down": 0.0, "short": 0.0, "loss": 0.0, "0": 0.0, "no": 0.0})
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = mapped.fillna((numeric > 0).astype(float))
    mapped[numeric.eq(0.0)] = 0.0
    return mapped


def _as_percent_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    clean = out.dropna()
    if not clean.empty and clean.abs().max() <= 2.0:
        out = out * 100.0
    return out


def _drawdown_from_percent_returns(returns_pct: pd.Series) -> float:
    returns = pd.to_numeric(returns_pct, errors="coerce").dropna() / 100.0
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min() * 100.0)


def _profit_factor_from_percent_returns(returns_pct: pd.Series) -> float:
    returns = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if returns.empty:
        return np.nan
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return float(np.inf) if gains > 0 else np.nan
    return float(gains / abs(losses))


def _probability_candidate_rows(
    candidate_recommendation_table: Optional[pd.DataFrame],
    raw_trade_log_table: Optional[pd.DataFrame],
    coverage_edge_frontier_table: Optional[pd.DataFrame],
    full_evidence_table: Optional[pd.DataFrame],
    diagnostics_table: Optional[pd.DataFrame],
    candidate_filter: str,
    selected_assets: Optional[Iterable[str]],
    selected_horizons: Optional[Iterable[int]],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for table in [raw_trade_log_table, candidate_recommendation_table, coverage_edge_frontier_table, full_evidence_table, diagnostics_table]:
        if table is not None and not table.empty and {"Asset", "Horizon"}.issubset(table.columns):
            frames.append(_normalise_horizon_column(table)[["Asset", "Horizon"]].copy())
    if not frames:
        return pd.DataFrame(columns=["Asset", "Horizon"])
    out = pd.concat(frames, ignore_index=True).dropna(subset=["Asset", "Horizon"]).drop_duplicates()
    out["Asset"] = out["Asset"].astype(str)
    out["Horizon"] = out["Horizon"].astype(int)
    mode = str(candidate_filter or "default_focus").lower()
    if mode in {"default_focus", "default focus", "focus"}:
        focus = out[
            ((out["Asset"].eq("Bitcoin")) & (out["Horizon"].eq(5)))
            | ((out["Asset"].eq("Crude Oil")) & (out["Horizon"].eq(5)))
        ]
        if not focus.empty:
            out = focus
    elif mode in {"specific", "specific asset/horizon"}:
        assets = set(str(a) for a in (selected_assets or []))
        horizons = set(int(h) for h in (selected_horizons or []))
        if assets:
            out = out[out["Asset"].isin(assets)]
        if horizons:
            out = out[out["Horizon"].isin(horizons)]
    return out.sort_values(["Asset", "Horizon"]).reset_index(drop=True)


def _extract_probability_trade_rows(*tables: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    probability_names = ["ProbabilityUp", "Probability", "P_up", "PUp", "PredictedProbability", "PredictedProbabilityUp", "DirectionProbability"]
    direction_names = ["ActualDirection", "FutureDirection", "FutureDirectionUp", "Direction", "ActualUp", "Win", "WinLoss"]
    return_names = ["RealizedReturn", "StrategyReturnAfterCost", "ActualReturn", "FutureReturn", "Return", "TradeReturn", "StrategyReturn_%"]
    vs_names = ["VsBuyHold", "VsBuyHold_%", "MedianVsBuyHold_%", "LockedTestVsBuyHold_%", "AvgLockedVsBuyHold_%", "MedianLockedVsBuyHold_%"]
    drawdown_names = ["MaxDrawdownDuringTrade", "MaxDrawdown", "MaxDrawdown_%", "LockedTestMaxDrawdown_%", "WorstMaxDrawdown_%", "WorstLockedMaxDrawdown_%"]
    for table in tables:
        if table is None or table.empty or not {"Asset", "Horizon"}.issubset(table.columns):
            continue
        df = _normalise_horizon_column(table)
        if "EvidenceMode" in df.columns:
            evidence_mode = df["EvidenceMode"].astype(str)
            eligible = evidence_mode.isin(["RawTradeLevel", "ReconstructedTradeLevel"])
            if eligible.any():
                df = df[eligible].copy()
        prob_col = _first_existing_column(df, probability_names)
        dir_col = _first_existing_column(df, direction_names)
        if prob_col is None or dir_col is None:
            continue
        ret_col = _first_existing_column(df, return_names)
        vs_col = _first_existing_column(df, vs_names)
        dd_col = _first_existing_column(df, drawdown_names)
        out = pd.DataFrame(
            {
                "Asset": df["Asset"].astype(str),
                "Horizon": df["Horizon"].astype(int),
                "ProbabilityUp": _as_probability_series(df[prob_col]),
                "ActualDirection": _as_direction_series(df[dir_col]),
                "Return_%": _as_percent_series(df[ret_col]) if ret_col is not None else np.nan,
                "VsBuyHold_%": _as_percent_series(df[vs_col]) if vs_col is not None else np.nan,
                "MaxDrawdown_%": _as_percent_series(df[dd_col]) if dd_col is not None else np.nan,
            }
        )
        out = out.dropna(subset=["ProbabilityUp", "ActualDirection"])
        if not out.empty:
            rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["Asset", "Horizon", "ProbabilityUp", "ActualDirection", "Return_%", "VsBuyHold_%", "MaxDrawdown_%"])
    return pd.concat(rows, ignore_index=True)


def _bin_mask(probabilities: pd.Series, lower: float, upper: float) -> pd.Series:
    if upper >= 1.0:
        return probabilities.ge(lower) & probabilities.le(upper)
    return probabilities.ge(lower) & probabilities.lt(upper)


def _probability_bin_label(lower: float, upper: float) -> str:
    return f"{lower:.2f}-{upper:.2f}"


def _actual_probability_bin_rows(asset: str, horizon: int, trades: pd.DataFrame, bins: Iterable[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    subset = trades[trades["Asset"].astype(str).eq(str(asset)) & trades["Horizon"].astype(int).eq(int(horizon))]
    for raw_bin in bins:
        lower, upper = float(raw_bin[0]), float(raw_bin[1])
        bin_trades = subset[_bin_mask(subset["ProbabilityUp"], lower, upper)].copy()
        returns = pd.to_numeric(bin_trades.get("Return_%", pd.Series(dtype=float)), errors="coerce")
        vs = pd.to_numeric(bin_trades.get("VsBuyHold_%", pd.Series(dtype=float)), errors="coerce")
        direction = pd.to_numeric(bin_trades.get("ActualDirection", pd.Series(dtype=float)), errors="coerce")
        trade_count = int(len(bin_trades))
        win_rate = float(direction.mean() * 100.0) if trade_count else np.nan
        dd_values = pd.to_numeric(bin_trades.get("MaxDrawdown_%", pd.Series(dtype=float)), errors="coerce").dropna()
        max_drawdown = float(dd_values.min()) if not dd_values.empty else _drawdown_from_percent_returns(returns)
        expected = (lower + upper) / 2.0 * 100.0
        warnings = []
        if trade_count == 0:
            warnings.append("NoTradesInBin")
        elif trade_count < 5:
            warnings.append("LowTradeCount")
        if trade_count and lower >= 0.70 and win_rate < expected - 15.0:
            warnings.append("HighConfidenceFailures")
        rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "ProbabilityBin": _probability_bin_label(lower, upper),
                "BinLower": lower,
                "BinUpper": upper,
                "Rows": trade_count,
                "TradeCount": trade_count,
                "WinRate_%": round(win_rate, 4) if np.isfinite(win_rate) else np.nan,
                "AvgReturn_%": round(float(returns.mean()), 4) if not returns.dropna().empty else np.nan,
                "MedianReturn_%": round(float(returns.median()), 4) if not returns.dropna().empty else np.nan,
                "AvgVsBuyHold_%": round(float(vs.mean()), 4) if not vs.dropna().empty else np.nan,
                "MedianVsBuyHold_%": round(float(vs.median()), 4) if not vs.dropna().empty else np.nan,
                "MaxDrawdown_%": round(max_drawdown, 4) if np.isfinite(max_drawdown) else np.nan,
                "ProfitFactor": round(_profit_factor_from_percent_returns(returns), 4) if np.isfinite(_profit_factor_from_percent_returns(returns)) else np.nan,
                "BenchmarkBeatRate_%": round(float((vs > 0).mean() * 100.0), 4) if not vs.dropna().empty else np.nan,
                "ExpectedProbability_%": round(expected, 4),
                "CalibrationGap_%": round(win_rate - expected, 4) if np.isfinite(win_rate) else np.nan,
                "SourceType": "raw_probability_outcomes",
                "Warnings": _join_warnings(warnings),
            }
        )
    return rows


def _proxy_probability_bin_rows(asset: str, horizon: int, frontier: pd.DataFrame, full_evidence: pd.DataFrame, bins: Iterable[Any]) -> List[Dict[str, Any]]:
    frontier = _normalise_horizon_column(frontier)
    full_evidence = _normalise_horizon_column(full_evidence)
    subset = pd.DataFrame()
    if not frontier.empty and {"Asset", "Horizon"}.issubset(frontier.columns):
        subset = frontier[frontier["Asset"].astype(str).eq(str(asset)) & frontier["Horizon"].astype(int).eq(int(horizon))].copy()
    evidence_subset = pd.DataFrame()
    if not full_evidence.empty and {"Asset", "Horizon"}.issubset(full_evidence.columns):
        evidence_subset = full_evidence[full_evidence["Asset"].astype(str).eq(str(asset)) & full_evidence["Horizon"].astype(int).eq(int(horizon))].copy()
    base_trade = float(pd.to_numeric(evidence_subset.get("AvgTradesPerWindow", pd.Series(dtype=float)), errors="coerce").mean()) if not evidence_subset.empty else 0.0
    base_vs = float(pd.to_numeric(evidence_subset.get("MedianLockedVsBuyHold_%", pd.Series(dtype=float)), errors="coerce").median()) if not evidence_subset.empty else 0.0
    base_dd = float(pd.to_numeric(evidence_subset.get("WorstLockedMaxDrawdown_%", pd.Series(dtype=float)), errors="coerce").min()) if not evidence_subset.empty else np.nan
    rows: List[Dict[str, Any]] = []
    for raw_bin in bins:
        lower, upper = float(raw_bin[0]), float(raw_bin[1])
        bin_label = _probability_bin_label(lower, upper)
        match = pd.DataFrame()
        if not subset.empty and {"MinProbability", "MaxProbability"}.issubset(subset.columns):
            min_p = pd.to_numeric(subset["MinProbability"], errors="coerce")
            max_p = pd.to_numeric(subset["MaxProbability"], errors="coerce")
            match = subset[min_p.le(lower + 0.025) & max_p.ge(upper - 0.025)]
        if match.empty and not subset.empty:
            match = subset.head(0)
        if not match.empty:
            trade_count = float(pd.to_numeric(match.get("TradeCount", pd.Series(dtype=float)), errors="coerce").mean())
            win_rate = float(pd.to_numeric(match.get("WinRate_%", match.get("BeatBuyHoldRate_%", pd.Series(dtype=float))), errors="coerce").mean())
            median_vs = float(pd.to_numeric(match.get("MedianVsBuyHold_%", pd.Series(dtype=float)), errors="coerce").median())
            avg_vs = float(pd.to_numeric(match.get("VsBuyHold_%", pd.Series(dtype=float)), errors="coerce").mean())
            max_drawdown = float(pd.to_numeric(match.get("MaxDrawdown_%", pd.Series(dtype=float)), errors="coerce").min())
            beat = float(pd.to_numeric(match.get("BenchmarkBeatRate_%", match.get("BeatBuyHoldRate_%", pd.Series(dtype=float))), errors="coerce").mean())
        else:
            midpoint = (lower + upper) / 2.0
            strictness = max(0.0, midpoint - 0.50)
            trade_count = max(0.0, base_trade * (1.0 - strictness * 2.4))
            win_rate = np.nan
            median_vs = base_vs + strictness * 1.5
            avg_vs = median_vs
            max_drawdown = base_dd
            beat = np.nan
        expected = (lower + upper) / 2.0 * 100.0
        warnings = ["AggregateProxyOnly", "CalibrationWeak"]
        if not np.isfinite(trade_count) or trade_count < 5:
            warnings.append("LowTradeCount")
        rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "ProbabilityBin": bin_label,
                "BinLower": lower,
                "BinUpper": upper,
                "Rows": int(len(match)) if not match.empty else 0,
                "TradeCount": round(float(trade_count), 4) if np.isfinite(trade_count) else 0.0,
                "WinRate_%": round(win_rate, 4) if np.isfinite(win_rate) else np.nan,
                "AvgReturn_%": np.nan,
                "MedianReturn_%": np.nan,
                "AvgVsBuyHold_%": round(avg_vs, 4) if np.isfinite(avg_vs) else np.nan,
                "MedianVsBuyHold_%": round(median_vs, 4) if np.isfinite(median_vs) else np.nan,
                "MaxDrawdown_%": round(max_drawdown, 4) if np.isfinite(max_drawdown) else np.nan,
                "ProfitFactor": np.nan,
                "BenchmarkBeatRate_%": round(beat, 4) if np.isfinite(beat) else np.nan,
                "ExpectedProbability_%": round(expected, 4),
                "CalibrationGap_%": round(win_rate - expected, 4) if np.isfinite(win_rate) else np.nan,
                "SourceType": "aggregate_proxy",
                "Warnings": _join_warnings(warnings),
            }
        )
    return rows


def _simulate_actual_probability_filter(asset: str, horizon: int, trades: pd.DataFrame, min_probability: float, max_probability: float) -> Dict[str, Any]:
    subset = trades[trades["Asset"].astype(str).eq(str(asset)) & trades["Horizon"].astype(int).eq(int(horizon))].copy()
    total = max(len(subset), 1)
    filtered = subset[subset["ProbabilityUp"].ge(float(min_probability)) & subset["ProbabilityUp"].le(float(max_probability))].copy()
    returns = pd.to_numeric(filtered.get("Return_%", pd.Series(dtype=float)), errors="coerce")
    vs = pd.to_numeric(filtered.get("VsBuyHold_%", pd.Series(dtype=float)), errors="coerce")
    direction = pd.to_numeric(filtered.get("ActualDirection", pd.Series(dtype=float)), errors="coerce")
    trade_count = int(len(filtered))
    retention = trade_count / total * 100.0
    win_rate = float(direction.mean() * 100.0) if trade_count else np.nan
    median_vs = float(vs.median()) if not vs.dropna().empty else np.nan
    dd_values = pd.to_numeric(filtered.get("MaxDrawdown_%", pd.Series(dtype=float)), errors="coerce").dropna()
    drawdown = float(dd_values.min()) if not dd_values.empty else _drawdown_from_percent_returns(returns)
    beat = float((vs > 0).mean() * 100.0) if not vs.dropna().empty else np.nan
    warnings: List[str] = []
    if trade_count < max(5, total * 0.20):
        warnings.append("TradeCountDestroyed")
    if np.isfinite(median_vs) and median_vs <= 0.0:
        warnings.append("BenchmarkStillDominates")
    if np.isfinite(drawdown) and drawdown <= -25.0:
        warnings.append("DrawdownNotReduced")
    filter_score = _clip(
        (win_rate if np.isfinite(win_rate) else 0.0) * 0.25
        + (beat if np.isfinite(beat) else 0.0) * 0.25
        + min(trade_count * 3.0, 20.0)
        + np.clip(median_vs if np.isfinite(median_vs) else 0.0, -10.0, 10.0) * 2.0
        - max(0.0, abs(min(drawdown if np.isfinite(drawdown) else 0.0, 0.0)) - 20.0)
    )
    useful = trade_count >= max(5, total * 0.25) and (not np.isfinite(median_vs) or median_vs > 0.0) and (not np.isfinite(drawdown) or drawdown > -30.0)
    return {
        "Asset": asset,
        "Horizon": int(horizon),
        "MinProbability": float(min_probability),
        "MaxProbability": float(max_probability),
        "TradeCount": trade_count,
        "TradeRetention_%": round(retention, 4),
        "WinRate_%": round(win_rate, 4) if np.isfinite(win_rate) else np.nan,
        "MedianReturn_%": round(float(returns.median()), 4) if not returns.dropna().empty else np.nan,
        "MedianVsBuyHold_%": round(median_vs, 4) if np.isfinite(median_vs) else np.nan,
        "MaxDrawdown_%": round(drawdown, 4) if np.isfinite(drawdown) else np.nan,
        "BenchmarkBeatRate_%": round(beat, 4) if np.isfinite(beat) else np.nan,
        "FilterScore": round(float(filter_score), 4),
        "FilterVerdict": "UsefulProbabilityFilter" if useful else "NoUsefulProbabilityFilter",
        "Warnings": _join_warnings(warnings or (["NoUsefulProbabilityFilter"] if not useful else [])),
    }


def _simulate_proxy_probability_filter(asset: str, horizon: int, frontier: pd.DataFrame, min_probability: float, max_probability: float) -> Dict[str, Any]:
    frontier = _normalise_horizon_column(frontier)
    subset = pd.DataFrame()
    if not frontier.empty and {"Asset", "Horizon"}.issubset(frontier.columns):
        subset = frontier[frontier["Asset"].astype(str).eq(str(asset)) & frontier["Horizon"].astype(int).eq(int(horizon))].copy()
    match = pd.DataFrame()
    if not subset.empty and {"MinProbability", "MaxProbability"}.issubset(subset.columns):
        min_p = pd.to_numeric(subset["MinProbability"], errors="coerce")
        max_p = pd.to_numeric(subset["MaxProbability"], errors="coerce")
        distance = (min_p - float(min_probability)).abs() + (max_p - float(max_probability)).abs()
        match = subset.assign(_distance=distance).sort_values("_distance").head(1)
    trade_count = float(pd.to_numeric(match.get("TradeCount", pd.Series(dtype=float)), errors="coerce").mean()) if not match.empty else 0.0
    total = float(pd.to_numeric(subset.get("TradeCount", pd.Series(dtype=float)), errors="coerce").max()) if not subset.empty else max(trade_count, 1.0)
    retention = trade_count / max(total, 1.0) * 100.0
    win_rate = float(pd.to_numeric(match.get("WinRate_%", match.get("BeatBuyHoldRate_%", pd.Series(dtype=float))), errors="coerce").mean()) if not match.empty else np.nan
    median_vs = float(pd.to_numeric(match.get("MedianVsBuyHold_%", pd.Series(dtype=float)), errors="coerce").median()) if not match.empty else np.nan
    drawdown = float(pd.to_numeric(match.get("MaxDrawdown_%", pd.Series(dtype=float)), errors="coerce").min()) if not match.empty else np.nan
    beat = float(pd.to_numeric(match.get("BenchmarkBeatRate_%", match.get("BeatBuyHoldRate_%", pd.Series(dtype=float))), errors="coerce").mean()) if not match.empty else np.nan
    warnings = ["AggregateProxyOnly", "CalibrationWeak"]
    if trade_count < max(3.0, total * 0.20):
        warnings.append("TradeCountDestroyed")
    if np.isfinite(median_vs) and median_vs <= 0.0:
        warnings.append("BenchmarkStillDominates")
    filter_score = _clip((beat if np.isfinite(beat) else 0.0) * 0.30 + min(trade_count * 3.0, 20.0) + np.clip(median_vs if np.isfinite(median_vs) else 0.0, -10.0, 10.0) * 2.0)
    useful = trade_count >= max(3.0, total * 0.25) and np.isfinite(median_vs) and median_vs > 0.0 and np.isfinite(beat) and beat >= 50.0
    return {
        "Asset": asset,
        "Horizon": int(horizon),
        "MinProbability": float(min_probability),
        "MaxProbability": float(max_probability),
        "TradeCount": round(float(trade_count), 4),
        "TradeRetention_%": round(float(retention), 4),
        "WinRate_%": round(win_rate, 4) if np.isfinite(win_rate) else np.nan,
        "MedianReturn_%": np.nan,
        "MedianVsBuyHold_%": round(median_vs, 4) if np.isfinite(median_vs) else np.nan,
        "MaxDrawdown_%": round(drawdown, 4) if np.isfinite(drawdown) else np.nan,
        "BenchmarkBeatRate_%": round(beat, 4) if np.isfinite(beat) else np.nan,
        "FilterScore": round(float(filter_score), 4),
        "FilterVerdict": "UsefulProbabilityFilter" if useful else "NoUsefulProbabilityFilter",
        "Warnings": _join_warnings(warnings if not useful else [w for w in warnings if w != "TradeCountDestroyed"]),
    }


def _rank_correlation(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 2 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    return float(frame["x"].rank().corr(frame["y"].rank()))


def _calibration_error_row(asset: str, horizon: int, bin_table: pd.DataFrame, trades: pd.DataFrame, raw_available: bool) -> Dict[str, Any]:
    candidate_bins = bin_table[bin_table["Asset"].astype(str).eq(str(asset)) & bin_table["Horizon"].astype(int).eq(int(horizon))]
    candidate_trades = trades[trades["Asset"].astype(str).eq(str(asset)) & trades["Horizon"].astype(int).eq(int(horizon))] if raw_available else pd.DataFrame()
    total = int(len(candidate_trades)) if raw_available else int(pd.to_numeric(candidate_bins["TradeCount"], errors="coerce").max() if not candidate_bins.empty else 0)
    warnings: List[str] = []
    if raw_available and not candidate_trades.empty:
        probabilities = pd.to_numeric(candidate_trades["ProbabilityUp"], errors="coerce")
        actual = pd.to_numeric(candidate_trades["ActualDirection"], errors="coerce")
        brier = float(((probabilities - actual) ** 2).mean())
        weighted_gaps = []
        weighted_over = []
        weighted_under = []
        for _, row in candidate_bins.iterrows():
            count = _safe_float(row.get("TradeCount"), default=0.0)
            if count <= 0:
                continue
            gap = _safe_float(row.get("CalibrationGap_%"), default=np.nan) / 100.0
            if not np.isfinite(gap):
                continue
            weighted_gaps.append((count, abs(gap)))
            weighted_over.append((count, max(-gap, 0.0)))
            weighted_under.append((count, max(gap, 0.0)))
        denom = sum(count for count, _ in weighted_gaps) or 1.0
        ece = sum(count * value for count, value in weighted_gaps) / denom
        over = sum(count * value for count, value in weighted_over) / denom * 100.0
        under = sum(count * value for count, value in weighted_under) / denom * 100.0
        high = candidate_trades[probabilities.ge(0.70)]
        high_fail = float((1.0 - pd.to_numeric(high["ActualDirection"], errors="coerce")).mean() * 100.0) if not high.empty else np.nan
        high_count = int(len(high))
    else:
        brier = np.nan
        gaps = candidate_bins.dropna(subset=["CalibrationGap_%"]) if not candidate_bins.empty else pd.DataFrame()
        if not gaps.empty:
            counts = pd.to_numeric(gaps["TradeCount"], errors="coerce").fillna(0.0)
            gap_values = pd.to_numeric(gaps["CalibrationGap_%"], errors="coerce") / 100.0
            denom = float(counts.sum()) or 1.0
            ece = float((counts * gap_values.abs()).sum() / denom)
            over = float((counts * (-gap_values).clip(lower=0.0)).sum() / denom * 100.0)
            under = float((counts * gap_values.clip(lower=0.0)).sum() / denom * 100.0)
        else:
            ece = np.nan
            over = np.nan
            under = np.nan
        high_bins = candidate_bins[pd.to_numeric(candidate_bins["BinLower"], errors="coerce").ge(0.70)] if not candidate_bins.empty else pd.DataFrame()
        if not high_bins.empty and high_bins["WinRate_%"].notna().any():
            high_fail = float(100.0 - pd.to_numeric(high_bins["WinRate_%"], errors="coerce").mean())
            high_count = int(pd.to_numeric(high_bins["TradeCount"], errors="coerce").sum())
        else:
            high_fail = np.nan
            high_count = 0
        warnings.append("CalibrationWeak")
    if total < 10:
        warnings.append("TooFewTradesToCalibrate")
    if np.isfinite(over) and over >= 12.0:
        warnings.append("Overconfident")
    if np.isfinite(high_fail) and high_fail >= 45.0 and high_count > 0:
        warnings.append("HighConfidenceFailures")
    return {
        "Asset": asset,
        "Horizon": int(horizon),
        "RawProbabilityOutcomesAvailable": bool(raw_available),
        "BrierScore": round(brier, 6) if np.isfinite(brier) else np.nan,
        "ECE": round(ece, 6) if np.isfinite(ece) else np.nan,
        "OverconfidenceScore": round(over, 4) if np.isfinite(over) else np.nan,
        "UnderconfidenceScore": round(under, 4) if np.isfinite(under) else np.nan,
        "HighConfidenceFailureRate_%": round(high_fail, 4) if np.isfinite(high_fail) else np.nan,
        "HighConfidenceTrades": high_count,
        "TotalTrades": total,
        "Warnings": _join_warnings(warnings),
    }


def _confidence_usefulness_row(asset: str, horizon: int, bin_table: pd.DataFrame, filter_table: pd.DataFrame) -> Dict[str, Any]:
    bins = bin_table[bin_table["Asset"].astype(str).eq(str(asset)) & bin_table["Horizon"].astype(int).eq(int(horizon))].copy()
    filters = filter_table[filter_table["Asset"].astype(str).eq(str(asset)) & filter_table["Horizon"].astype(int).eq(int(horizon))].copy()
    populated = bins[pd.to_numeric(bins["TradeCount"], errors="coerce").gt(0)].copy()
    mid = (pd.to_numeric(populated.get("BinLower", pd.Series(dtype=float)), errors="coerce") + pd.to_numeric(populated.get("BinUpper", pd.Series(dtype=float)), errors="coerce")) / 2.0
    win_corr = _rank_correlation(mid, populated.get("WinRate_%", pd.Series(dtype=float)))
    return_corr = _rank_correlation(mid, populated.get("MedianReturn_%", populated.get("MedianVsBuyHold_%", pd.Series(dtype=float))))
    low = populated[pd.to_numeric(populated["BinUpper"], errors="coerce").le(0.65)]
    high = populated[pd.to_numeric(populated["BinLower"], errors="coerce").ge(0.70)]
    low_win = float(pd.to_numeric(low.get("WinRate_%", pd.Series(dtype=float)), errors="coerce").mean()) if not low.empty else np.nan
    high_win = float(pd.to_numeric(high.get("WinRate_%", pd.Series(dtype=float)), errors="coerce").mean()) if not high.empty else np.nan
    low_ret = float(pd.to_numeric(low.get("MedianReturn_%", low.get("MedianVsBuyHold_%", pd.Series(dtype=float))), errors="coerce").mean()) if not low.empty else np.nan
    high_ret = float(pd.to_numeric(high.get("MedianReturn_%", high.get("MedianVsBuyHold_%", pd.Series(dtype=float))), errors="coerce").mean()) if not high.empty else np.nan
    low_dd = float(pd.to_numeric(low.get("MaxDrawdown_%", pd.Series(dtype=float)), errors="coerce").min()) if not low.empty else np.nan
    high_dd = float(pd.to_numeric(high.get("MaxDrawdown_%", pd.Series(dtype=float)), errors="coerce").min()) if not high.empty else np.nan
    low_beat = float(pd.to_numeric(low.get("BenchmarkBeatRate_%", pd.Series(dtype=float)), errors="coerce").mean()) if not low.empty else np.nan
    high_beat = float(pd.to_numeric(high.get("BenchmarkBeatRate_%", pd.Series(dtype=float)), errors="coerce").mean()) if not high.empty else np.nan
    improves_win = bool(np.isfinite(high_win) and np.isfinite(low_win) and high_win >= low_win + 2.0)
    improves_return = bool(np.isfinite(high_ret) and np.isfinite(low_ret) and high_ret >= low_ret)
    reduces_dd = bool(np.isfinite(high_dd) and np.isfinite(low_dd) and high_dd >= low_dd)
    beats_more = bool(np.isfinite(high_beat) and np.isfinite(low_beat) and high_beat >= low_beat)
    filtering_destroys = bool((not filters.empty) and pd.to_numeric(filters["TradeRetention_%"], errors="coerce").lt(25.0).any())
    edge_monotonic = bool(np.isfinite(win_corr) and win_corr >= 0.20)
    warnings: List[str] = []
    if not edge_monotonic:
        warnings.append("EdgeNotMonotonic")
    if not reduces_dd:
        warnings.append("DrawdownNotReduced")
    if filtering_destroys:
        warnings.append("TradeCountDestroyed")
    useful = bool((edge_monotonic or improves_win) and (improves_return or beats_more) and not filtering_destroys)
    return {
        "Asset": asset,
        "Horizon": int(horizon),
        "ProbabilityWinRateCorrelation": round(win_corr, 4) if np.isfinite(win_corr) else np.nan,
        "ProbabilityReturnCorrelation": round(return_corr, 4) if np.isfinite(return_corr) else np.nan,
        "HigherProbabilityImprovesWinRate": improves_win,
        "HigherProbabilityImprovesMedianReturn": improves_return,
        "HigherProbabilityReducesDrawdown": reduces_dd,
        "HigherProbabilityBeatsBenchmarkMore": beats_more,
        "FilteringDestroysTradeCount": filtering_destroys,
        "ConfidenceUseful": useful,
        "UsefulnessVerdict": "ConfidenceUseful" if useful else "ProbabilityUnreliable",
        "Warnings": _join_warnings(warnings),
    }


def _probability_calibration_grade(summary: Dict[str, Any], usefulness: Dict[str, Any], error: Dict[str, Any]) -> str:
    total = _safe_int(error.get("TotalTrades"), default=0)
    ece = _safe_float(error.get("ECE"), default=np.nan)
    over = _safe_float(error.get("OverconfidenceScore"), default=np.nan)
    under = _safe_float(error.get("UnderconfidenceScore"), default=np.nan)
    high_fail = _safe_float(error.get("HighConfidenceFailureRate_%"), default=np.nan)
    raw = bool(error.get("RawProbabilityOutcomesAvailable", False))
    useful = bool(usefulness.get("ConfidenceUseful", False))
    monotonic = bool(summary.get("EdgeMonotonic", False))
    if total < 10:
        return "TooFewTradesToCalibrate"
    if (np.isfinite(high_fail) and high_fail >= 50.0) or (np.isfinite(over) and over >= 18.0):
        return "Overconfident"
    if not raw and not useful:
        return "ProbabilityUnreliable"
    if np.isfinite(under) and under >= 18.0 and useful:
        return "Underconfident"
    if raw and np.isfinite(ece) and ece <= 0.07 and useful and monotonic:
        return "WellCalibrated"
    if useful:
        return "UsefulButNoisy"
    return "ProbabilityUnreliable"


def _probability_next_action(grade: str, warnings: str) -> str:
    warning_set = {w.strip() for w in str(warnings).split(";") if w.strip()}
    if grade == "TooFewTradesToCalibrate" or "TooFewTradesToCalibrate" in warning_set:
        return "Collect more forward evidence or use less sparse coverage policies before calibration claims."
    if grade == "Overconfident" or "HighConfidenceFailures" in warning_set:
        return "Run calibration diagnostics and avoid using high probability as a standalone confidence gate."
    if "TradeCountDestroyed" in warning_set:
        return "Reject strict probability filters that destroy trade count; test softer filters only as research."
    if grade in {"WellCalibrated", "UsefulButNoisy"}:
        return "Send the fixed probability filter candidate to risk-control validation; do not promote grade here."
    return "Treat probability as unreliable evidence and keep the candidate in diagnostics."


def _run_probability_calibration_impl(
    *,
    candidate_recommendation_table: Optional[pd.DataFrame] = None,
    raw_trade_log_table: Optional[pd.DataFrame] = None,
    coverage_edge_frontier_table: Optional[pd.DataFrame] = None,
    full_evidence_table: Optional[pd.DataFrame] = None,
    diagnostics_table: Optional[pd.DataFrame] = None,
    candidate_filter: str = "default_focus",
    selected_assets: Optional[Iterable[str]] = None,
    selected_horizons: Optional[Iterable[int]] = None,
    probability_bins: Optional[Iterable[Any]] = None,
    min_probabilities: Iterable[float] = PROBABILITY_FILTER_DEFAULTS["min_probabilities"],
    max_probabilities: Iterable[float] = PROBABILITY_FILTER_DEFAULTS["max_probabilities"],
    min_trades_for_calibration: int = 10,
) -> ProbabilityCalibrationReport:
    """Run Phase 8F probability calibration diagnostics from existing research artifacts.

    This phase does not rerun models, tune locked-test thresholds, or promote
    candidates. Raw probability/outcome rows are used when supplied; otherwise
    the function reports conservative aggregate-proxy diagnostics.
    """
    bins = list(probability_bins or PROBABILITY_BIN_DEFAULTS)
    raw_trade = raw_trade_log_table.copy() if raw_trade_log_table is not None else pd.DataFrame()
    frontier = coverage_edge_frontier_table.copy() if coverage_edge_frontier_table is not None else pd.DataFrame()
    full = full_evidence_table.copy() if full_evidence_table is not None else pd.DataFrame()
    diagnostics = diagnostics_table.copy() if diagnostics_table is not None else pd.DataFrame()
    candidates = _probability_candidate_rows(
        candidate_recommendation_table,
        raw_trade,
        frontier,
        full,
        diagnostics,
        candidate_filter,
        selected_assets,
        selected_horizons,
    )
    settings = {
        "phase": "8F",
        "purpose": "probability_calibration_and_confidence_reliability_only",
        "does_not_promote_candidates": True,
        "production_ready_label_allowed": False,
        "candidate_filter": candidate_filter,
        "selection_basis": "predeclared_bins_and_filters_not_locked_test_tuning",
        "raw_trade_log_table_supplied": bool(raw_trade_log_table is not None and not raw_trade.empty),
    }
    if candidates.empty:
        return _empty_probability_calibration_report(settings)

    trade_rows = _extract_probability_trade_rows(raw_trade, frontier, full, diagnostics)
    bin_rows: List[Dict[str, Any]] = []
    filter_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    usefulness_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    high_failure_rows: List[Dict[str, Any]] = []
    recommendation_rows: List[Dict[str, Any]] = []
    warning_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []

    for _, candidate in candidates.iterrows():
        asset = str(candidate.get("Asset", ""))
        horizon = _safe_int(candidate.get("Horizon"), default=0)
        raw_subset = trade_rows[trade_rows["Asset"].astype(str).eq(asset) & trade_rows["Horizon"].astype(int).eq(int(horizon))]
        raw_available = not raw_subset.empty
        if raw_available:
            candidate_bin_rows = _actual_probability_bin_rows(asset, horizon, trade_rows, bins)
        else:
            candidate_bin_rows = _proxy_probability_bin_rows(asset, horizon, frontier, full, bins)
        bin_rows.extend(candidate_bin_rows)

        for min_probability in min_probabilities:
            for max_probability in max_probabilities:
                if float(min_probability) > float(max_probability):
                    continue
                if raw_available:
                    filter_rows.append(_simulate_actual_probability_filter(asset, horizon, trade_rows, float(min_probability), float(max_probability)))
                else:
                    filter_rows.append(_simulate_proxy_probability_filter(asset, horizon, frontier, float(min_probability), float(max_probability)))

    bin_table = pd.DataFrame(bin_rows)
    for col in PROBABILITY_BIN_COLUMNS:
        if col not in bin_table.columns:
            bin_table[col] = np.nan
    filter_table = pd.DataFrame(filter_rows)
    for col in PROBABILITY_FILTER_COLUMNS:
        if col not in filter_table.columns:
            filter_table[col] = np.nan

    for _, candidate in candidates.iterrows():
        asset = str(candidate.get("Asset", ""))
        horizon = _safe_int(candidate.get("Horizon"), default=0)
        raw_subset = trade_rows[trade_rows["Asset"].astype(str).eq(asset) & trade_rows["Horizon"].astype(int).eq(int(horizon))]
        raw_available = not raw_subset.empty
        candidate_bins = bin_table[bin_table["Asset"].astype(str).eq(asset) & bin_table["Horizon"].astype(int).eq(int(horizon))]
        candidate_filters = filter_table[filter_table["Asset"].astype(str).eq(asset) & filter_table["Horizon"].astype(int).eq(int(horizon))]
        error = _calibration_error_row(asset, horizon, bin_table, trade_rows, raw_available)
        calibration_rows.append(error)
        useful = _confidence_usefulness_row(asset, horizon, bin_table, filter_table)
        usefulness_rows.append(useful)
        useful_filters = candidate_filters[candidate_filters["FilterVerdict"].eq("UsefulProbabilityFilter")].copy()
        if not useful_filters.empty:
            best_filter = useful_filters.sort_values(["FilterScore", "TradeCount"], ascending=[False, False]).iloc[0]
            best_filter_desc = f"min={best_filter['MinProbability']:.3f}, max={best_filter['MaxProbability']:.3f}"
        elif not candidate_filters.empty:
            best_filter = candidate_filters.sort_values(["FilterScore", "TradeCount"], ascending=[False, False]).iloc[0]
            best_filter_desc = f"no useful filter; best tested min={best_filter['MinProbability']:.3f}, max={best_filter['MaxProbability']:.3f}"
        else:
            best_filter = pd.Series(dtype=object)
            best_filter_desc = "no filter evidence"
        edge_monotonic = bool(useful.get("ProbabilityWinRateCorrelation", np.nan) >= 0.20) if np.isfinite(_safe_float(useful.get("ProbabilityWinRateCorrelation"), default=np.nan)) else False
        warning_set: List[str] = []
        for source in [error.get("Warnings", ""), useful.get("Warnings", ""), best_filter.get("Warnings", "") if not best_filter.empty else ""]:
            warning_set.extend([w.strip() for w in str(source).split(";") if w.strip()])
        if not raw_available:
            warning_set.append("CalibrationWeak")
        if _safe_int(error.get("TotalTrades"), default=0) < int(min_trades_for_calibration):
            warning_set.append("TooFewTradesToCalibrate")
        if not bool(useful.get("ConfidenceUseful", False)):
            warning_set.append("ProbabilityUnreliable")
        if not useful_filters.empty and _safe_float(best_filter.get("TradeRetention_%"), default=100.0) < 25.0:
            warning_set.append("TradeCountDestroyed")
        summary_seed = {"EdgeMonotonic": edge_monotonic}
        grade = _probability_calibration_grade(summary_seed, useful, error)
        if grade == "Overconfident":
            warning_set.append("Overconfident")
        if grade == "ProbabilityUnreliable":
            warning_set.append("ProbabilityUnreliable")
        warnings_joined = _join_warnings(warning_set)
        next_action = _probability_next_action(grade, warnings_joined)
        recommendation = (
            "Probability filter may be useful for further risk-control validation."
            if grade in {"WellCalibrated", "UsefulButNoisy"}
            else "Do not use probability confidence as a standalone upgrade."
        )
        score = _clip(
            100.0
            - (_safe_float(error.get("ECE"), default=0.25) * 180.0 if np.isfinite(_safe_float(error.get("ECE"), default=np.nan)) else 35.0)
            - _safe_float(error.get("OverconfidenceScore"), default=0.0) * 1.2
            - (0.0 if bool(useful.get("ConfidenceUseful", False)) else 25.0)
            - (25.0 if _safe_int(error.get("TotalTrades"), default=0) < int(min_trades_for_calibration) else 0.0)
        )
        summary_rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "TotalTrades": _safe_int(error.get("TotalTrades"), default=0),
                "RawProbabilityOutcomesAvailable": bool(raw_available),
                "CalibrationGrade": grade,
                "CalibrationScore": round(float(score), 4),
                "BrierScore": error.get("BrierScore", np.nan),
                "ECE": error.get("ECE", np.nan),
                "OverconfidenceScore": error.get("OverconfidenceScore", np.nan),
                "UnderconfidenceScore": error.get("UnderconfidenceScore", np.nan),
                "HighConfidenceFailureRate_%": error.get("HighConfidenceFailureRate_%", np.nan),
                "EdgeMonotonic": edge_monotonic,
                "UsefulProbabilityFilterFound": bool(not useful_filters.empty),
                "BestFilterDescription": best_filter_desc,
                "BestFilterTradeCount": best_filter.get("TradeCount", np.nan) if not best_filter.empty else np.nan,
                "BestFilterMedianVsBuyHold_%": best_filter.get("MedianVsBuyHold_%", np.nan) if not best_filter.empty else np.nan,
                "BestFilterMaxDrawdown_%": best_filter.get("MaxDrawdown_%", np.nan) if not best_filter.empty else np.nan,
                "MainWarning": str(warnings_joined).split(";")[0].strip() if warnings_joined else "",
                "Recommendation": recommendation,
                "PromotesGrades": False,
                "ProductionReadyLabelAllowed": False,
            }
        )
        recommendation_rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "CalibrationGrade": grade,
                "Recommendation": recommendation,
                "BestFilterDescription": best_filter_desc,
                "ShouldPromoteGrade": False,
                "ProductionReadyLabelAllowed": False,
                "Warnings": warnings_joined,
                "NextResearchAction": next_action,
            }
        )
        priority = "High" if grade in {"Overconfident", "ProbabilityUnreliable", "TooFewTradesToCalibrate"} else "Medium"
        action_rows.append({"Asset": asset, "Horizon": int(horizon), "NextResearchAction": next_action, "ActionPriority": priority})
        for warning in [w.strip() for w in warnings_joined.split(";") if w.strip()]:
            severity = "High" if warning in {"TooFewTradesToCalibrate", "ProbabilityUnreliable", "Overconfident", "HighConfidenceFailures"} else "Medium"
            warning_rows.append({"Asset": asset, "Horizon": int(horizon), "WarningType": warning, "Severity": severity, "Message": f"{warning} detected in probability calibration diagnostics."})
        for _, bin_row in candidate_bins[pd.to_numeric(candidate_bins["BinLower"], errors="coerce").ge(0.70)].iterrows():
            failure_rate = 100.0 - _safe_float(bin_row.get("WinRate_%"), default=np.nan)
            if _safe_float(bin_row.get("TradeCount"), default=0.0) > 0:
                warning_type = "HighConfidenceFailures" if np.isfinite(failure_rate) and failure_rate >= 45.0 else "HighConfidenceBinVisible"
                high_failure_rows.append(
                    {
                        "Asset": asset,
                        "Horizon": int(horizon),
                        "ProbabilityBin": bin_row.get("ProbabilityBin", ""),
                        "TradeCount": bin_row.get("TradeCount", np.nan),
                        "WinRate_%": bin_row.get("WinRate_%", np.nan),
                        "FailureRate_%": round(failure_rate, 4) if np.isfinite(failure_rate) else np.nan,
                        "MedianReturn_%": bin_row.get("MedianReturn_%", np.nan),
                        "MedianVsBuyHold_%": bin_row.get("MedianVsBuyHold_%", np.nan),
                        "MaxDrawdown_%": bin_row.get("MaxDrawdown_%", np.nan),
                        "WarningType": warning_type,
                        "Message": "High-confidence bin kept visible for calibration review.",
                    }
                )

    summary_table = pd.DataFrame(summary_rows)
    for col in PROBABILITY_CALIBRATION_SUMMARY_COLUMNS:
        if col not in summary_table.columns:
            summary_table[col] = np.nan
    calibration_table = pd.DataFrame(calibration_rows)
    for col in CALIBRATION_ERROR_COLUMNS:
        if col not in calibration_table.columns:
            calibration_table[col] = np.nan
    usefulness_table = pd.DataFrame(usefulness_rows)
    for col in CONFIDENCE_USEFULNESS_COLUMNS:
        if col not in usefulness_table.columns:
            usefulness_table[col] = np.nan
    high_failure_table = pd.DataFrame(high_failure_rows)
    for col in HIGH_CONFIDENCE_FAILURE_COLUMNS:
        if col not in high_failure_table.columns:
            high_failure_table[col] = np.nan
    recommendation_table = pd.DataFrame(recommendation_rows)
    for col in PROBABILITY_RECOMMENDATION_COLUMNS:
        if col not in recommendation_table.columns:
            recommendation_table[col] = np.nan
    warning_table = pd.DataFrame(warning_rows) if warning_rows else pd.DataFrame(columns=["Asset", "Horizon", "WarningType", "Severity", "Message"])
    next_actions = pd.DataFrame(action_rows) if action_rows else pd.DataFrame(columns=["Asset", "Horizon", "NextResearchAction", "ActionPriority"])
    overall = pd.DataFrame(
        [
            {
                "CandidatesTested": int(len(summary_table)),
                "WellCalibratedCandidates": int(summary_table["CalibrationGrade"].eq("WellCalibrated").sum()) if not summary_table.empty else 0,
                "UsefulButNoisyCandidates": int(summary_table["CalibrationGrade"].eq("UsefulButNoisy").sum()) if not summary_table.empty else 0,
                "UnreliableCandidates": int(summary_table["CalibrationGrade"].isin(["ProbabilityUnreliable", "Overconfident", "TooFewTradesToCalibrate"]).sum()) if not summary_table.empty else 0,
                "CandidatesWithRawOutcomes": int(summary_table["RawProbabilityOutcomesAvailable"].astype(bool).sum()) if not summary_table.empty else 0,
                "NoCandidateImproved": True,
                "PromotesGrades": False,
                "ProductionReadyLabelAllowed": False,
            }
        ]
    )
    return ProbabilityCalibrationReport(
        overall_summary=overall,
        calibration_summary_table=summary_table[PROBABILITY_CALIBRATION_SUMMARY_COLUMNS],
        probability_bin_table=bin_table[PROBABILITY_BIN_COLUMNS],
        probability_filter_simulation_table=filter_table[PROBABILITY_FILTER_COLUMNS],
        confidence_usefulness_table=usefulness_table[CONFIDENCE_USEFULNESS_COLUMNS],
        calibration_error_table=calibration_table[CALIBRATION_ERROR_COLUMNS],
        high_confidence_failure_table=high_failure_table[HIGH_CONFIDENCE_FAILURE_COLUMNS],
        candidate_recommendation_table=recommendation_table[PROBABILITY_RECOMMENDATION_COLUMNS],
        warning_table=warning_table,
        next_research_action_table=next_actions,
        settings=settings,
    )


def run_probability_calibration(*args: Any, **kwargs: Any) -> ProbabilityCalibrationReport:
    """Stable public Phase 8F entrypoint used by app.py and tests."""
    return _run_probability_calibration_impl(*args, **kwargs)


TRADE_EVIDENCE_LEDGER_COLUMNS = [
    "LedgerId",
    "Asset",
    "Horizon",
    "ModelName",
    "PolicyName",
    "SourcePhase",
    "WindowMode",
    "ValidationWindow",
    "TestWindow",
    "StepSize",
    "TransactionCost",
    "SignalDate",
    "EntryDate",
    "ExitDate",
    "HoldingPeriod",
    "ProbabilityUp",
    "ProbabilityBin",
    "SignalTaken",
    "TradeCountContribution",
    "DirectionPrediction",
    "ActualDirection",
    "StrategyReturn",
    "BenchmarkReturn",
    "VsBuyHold",
    "RealizedReturn",
    "MaxDrawdown",
    "WinLoss",
    "BeatBenchmark",
    "CostApplied",
    "RegimeLabel",
    "ConfidenceSource",
    "DataQualityFlag",
    "EvidenceMode",
    "Warnings",
]

LEDGER_QUALITY_COLUMNS = [
    "RowsInLedger",
    "AssetsCovered",
    "HorizonsCovered",
    "RawTradeLevelRows",
    "AggregateDerivedRows",
    "ProbabilityRows",
    "MissingProbabilityRate_%",
    "MissingActualDirectionRate_%",
    "MissingSignalDateRate_%",
    "MissingEntryExitDateRate_%",
    "AverageTradeCountContribution",
    "BenchmarkComparisonAvailability_%",
    "DrawdownAvailability_%",
    "CalibrationReadyRows",
    "CalibrationReadinessScore",
    "LedgerQualityScore",
    "MainWarning",
    "PromotesCandidates",
    "ProductionReadyLabelAllowed",
]

ASSET_HORIZON_COVERAGE_COLUMNS = [
    "Asset",
    "Horizon",
    "LedgerRows",
    "RawTradeLevelRows",
    "AggregateDerivedRows",
    "TradeCountContribution",
    "ProbabilityRows",
    "ActualDirectionRows",
    "CalibrationReadyRows",
    "CalibrationReadinessScore",
    "EvidenceModes",
    "Warnings",
]

PROBABILITY_OUTCOME_AVAILABILITY_COLUMNS = [
    "Asset",
    "Horizon",
    "ProbabilityRows",
    "ActualDirectionRows",
    "RowsWithProbabilityAndOutcome",
    "MissingProbabilityRate_%",
    "MissingActualDirectionRate_%",
    "CalibrationReady",
    "ConfidenceSourceSummary",
    "Warnings",
]

TRADE_OUTCOME_DISTRIBUTION_COLUMNS = [
    "Asset",
    "Horizon",
    "Rows",
    "TradeCountContribution",
    "Wins",
    "Losses",
    "WinRate_%",
    "AvgStrategyReturn",
    "MedianStrategyReturn",
    "PositiveReturnRows",
    "NegativeReturnRows",
    "FailedRowsVisible",
]

BENCHMARK_OUTCOME_COLUMNS = [
    "Asset",
    "Horizon",
    "RowsWithBenchmark",
    "BeatBenchmarkRows",
    "BeatBenchmarkRate_%",
    "AvgBenchmarkReturn",
    "AvgVsBuyHold",
    "NegativeVsBuyHoldRows",
    "BenchmarkMissingRate_%",
]

DRAWDOWN_OUTCOME_COLUMNS = [
    "Asset",
    "Horizon",
    "RowsWithDrawdown",
    "AvgMaxDrawdown",
    "WorstMaxDrawdown",
    "DrawdownMissingRate_%",
    "DrawdownRiskRows",
]


def _empty_trade_evidence_ledger_report(settings: Optional[Dict[str, Any]] = None) -> TradeEvidenceLedgerReport:
    return TradeEvidenceLedgerReport(
        ledger_table=pd.DataFrame(columns=TRADE_EVIDENCE_LEDGER_COLUMNS),
        ledger_quality_summary=pd.DataFrame(columns=LEDGER_QUALITY_COLUMNS),
        asset_horizon_coverage_table=pd.DataFrame(columns=ASSET_HORIZON_COVERAGE_COLUMNS),
        probability_outcome_availability_table=pd.DataFrame(columns=PROBABILITY_OUTCOME_AVAILABILITY_COLUMNS),
        trade_outcome_distribution_table=pd.DataFrame(columns=TRADE_OUTCOME_DISTRIBUTION_COLUMNS),
        benchmark_outcome_table=pd.DataFrame(columns=BENCHMARK_OUTCOME_COLUMNS),
        drawdown_outcome_table=pd.DataFrame(columns=DRAWDOWN_OUTCOME_COLUMNS),
        ledger_warning_table=pd.DataFrame(columns=["LedgerId", "Asset", "Horizon", "WarningType", "Severity", "Message"]),
        next_research_action_table=pd.DataFrame(columns=["Asset", "Horizon", "NextResearchAction", "ActionPriority"]),
        settings=settings or {},
    )


def _as_optional_date(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return pd.NaT


def _lookup_value(row: pd.Series, names: Iterable[str], default: Any = np.nan) -> Any:
    lower_lookup = {str(key).lower(): key for key in row.index}
    for name in names:
        key = lower_lookup.get(str(name).lower())
        if key is not None:
            return row.get(key, default)
    return default


def _probability_bin_from_value(probability: Any) -> str:
    p = _safe_float(probability, default=np.nan)
    if not np.isfinite(p):
        return ""
    if p > 1.0:
        p = p / 100.0
    for lower, upper in PROBABILITY_BIN_DEFAULTS:
        if p >= lower and (p <= upper if upper >= 1.0 else p < upper):
            return _probability_bin_label(lower, upper)
    return ""


def _direction_from_probability(probability: Any) -> str:
    p = _safe_float(probability, default=np.nan)
    if not np.isfinite(p):
        return ""
    if p > 1.0:
        p = p / 100.0
    if p > 0.5:
        return "Up"
    if p < 0.5:
        return "Down"
    return "Neutral"


def _normalise_direction_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in {"true", "1", "up", "long", "win", "yes", "positive"}:
        return 1
    if text in {"false", "0", "down", "short", "loss", "no", "negative"}:
        return 0
    numeric = _safe_float(value, default=np.nan)
    if np.isfinite(numeric):
        return int(numeric > 0)
    return np.nan


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return bool(default)


def _win_loss_label(actual_direction: Any, strategy_return: Any) -> str:
    direction = _normalise_direction_value(actual_direction)
    if not pd.isna(direction):
        return "Win" if int(direction) == 1 else "Loss"
    ret = _safe_float(strategy_return, default=np.nan)
    if np.isfinite(ret):
        return "Win" if ret > 0 else "Loss"
    return ""


def _beat_benchmark_value(vs_buy_hold: Any, strategy_return: Any, benchmark_return: Any) -> Any:
    vs = _safe_float(vs_buy_hold, default=np.nan)
    if np.isfinite(vs):
        return bool(vs > 0)
    strategy = _safe_float(strategy_return, default=np.nan)
    benchmark = _safe_float(benchmark_return, default=np.nan)
    if np.isfinite(strategy) and np.isfinite(benchmark):
        return bool(strategy > benchmark)
    return np.nan


def _ledger_row_warnings(row: Dict[str, Any]) -> str:
    warnings: List[str] = []
    mode = str(row.get("EvidenceMode", ""))
    if mode != "RawTradeLevel":
        warnings.append("AggregateProxyOnly")
    if pd.isna(row.get("ProbabilityUp")):
        warnings.append("MissingProbability")
    if pd.isna(row.get("ActualDirection")):
        warnings.append("MissingOutcomeDirection")
    if pd.isna(row.get("SignalDate")) and pd.isna(row.get("EntryDate")):
        warnings.append("MissingTradeDates")
    if pd.isna(row.get("BenchmarkReturn")) and pd.isna(row.get("VsBuyHold")):
        warnings.append("BenchmarkMissing")
    if pd.isna(row.get("MaxDrawdown")):
        warnings.append("DrawdownMissing")
    if "MissingProbability" in warnings or "MissingOutcomeDirection" in warnings:
        warnings.append("NotCalibrationReady")
    return _join_warnings(warnings)


def _data_quality_flag(row: Dict[str, Any]) -> str:
    warnings = {w.strip() for w in str(row.get("Warnings", "")).split(";") if w.strip()}
    if row.get("EvidenceMode") == "RawTradeLevel" and not warnings.intersection({"MissingProbability", "MissingOutcomeDirection", "MissingTradeDates"}):
        return "CompleteRawTrade"
    if row.get("EvidenceMode") == "RawTradeLevel":
        return "IncompleteRawTrade"
    if "NotCalibrationReady" in warnings:
        return "LimitedNotCalibrationReady"
    return "LimitedAggregate"


def _raw_trade_log_to_ledger(raw_trade_logs: Optional[Any]) -> pd.DataFrame:
    if raw_trade_logs is None:
        return pd.DataFrame(columns=TRADE_EVIDENCE_LEDGER_COLUMNS)
    if isinstance(raw_trade_logs, pd.DataFrame):
        raw_frames = [raw_trade_logs]
    elif isinstance(raw_trade_logs, list):
        raw_frames = [frame for frame in raw_trade_logs if isinstance(frame, pd.DataFrame) and not frame.empty]
    else:
        raw_frames = []
    rows: List[Dict[str, Any]] = []
    for source_index, raw in enumerate(raw_frames):
        if raw.empty or not {"Asset", "Horizon"}.issubset(raw.columns):
            continue
        df = _normalise_horizon_column(raw)
        for row_index, row in df.iterrows():
            probability = _lookup_value(row, ["ProbabilityUp", "Probability", "P_up", "PredictedProbability", "DirectionProbability"])
            probability = _as_probability_series(pd.Series([probability])).iloc[0] if not pd.isna(probability) else np.nan
            actual_direction = _normalise_direction_value(_lookup_value(row, ["ActualDirection", "FutureDirection", "Direction", "Win", "WinLoss"]))
            strategy_return = _lookup_value(row, ["StrategyReturn", "StrategyReturn_%", "StrategyReturnAfterCost", "RealizedReturn", "TradeReturn", "Return"])
            benchmark_return = _lookup_value(row, ["BenchmarkReturn", "BuyHoldReturn", "BuyHoldReturn_%", "BenchmarkReturn_%"])
            vs_buy_hold = _lookup_value(row, ["VsBuyHold", "VsBuyHold_%", "StrategyMinusBuyHold", "LockedTestVsBuyHold_%"])
            max_drawdown = _lookup_value(row, ["MaxDrawdown", "MaxDrawdown_%", "LockedTestMaxDrawdown_%", "WorstMaxDrawdown_%"])
            entry_date = _as_optional_date(_lookup_value(row, ["EntryDate", "TradeDate", "Date"]))
            exit_date = _as_optional_date(_lookup_value(row, ["ExitDate"]))
            signal_date = _as_optional_date(_lookup_value(row, ["SignalDate", "Date"]))
            ledger_row = {
                "LedgerId": f"RAW-{source_index + 1}-{row_index + 1}",
                "Asset": str(row.get("Asset", "")),
                "Horizon": _safe_int(row.get("Horizon"), default=0),
                "ModelName": _lookup_value(row, ["ModelName", "BestModel", "Model"], ""),
                "PolicyName": _lookup_value(row, ["PolicyName", "PolicyType", "SignalMode"], "Raw trade log"),
                "SourcePhase": _lookup_value(row, ["SourcePhase"], "RawTradeLog"),
                "WindowMode": _lookup_value(row, ["WindowMode"], ""),
                "ValidationWindow": _lookup_value(row, ["ValidationWindow"], np.nan),
                "TestWindow": _lookup_value(row, ["TestWindow"], np.nan),
                "StepSize": _lookup_value(row, ["StepSize"], np.nan),
                "TransactionCost": _lookup_value(row, ["TransactionCost"], np.nan),
                "SignalDate": signal_date,
                "EntryDate": entry_date,
                "ExitDate": exit_date,
                "HoldingPeriod": _lookup_value(row, ["HoldingPeriod", "HoldingDays"], np.nan),
                "ProbabilityUp": probability,
                "ProbabilityBin": _lookup_value(row, ["ProbabilityBin"], _probability_bin_from_value(probability)),
            "SignalTaken": _safe_bool(_lookup_value(row, ["SignalTaken"], True), default=True),
                "TradeCountContribution": _safe_float(_lookup_value(row, ["TradeCountContribution"], 1.0), default=1.0),
                "DirectionPrediction": _lookup_value(row, ["DirectionPrediction", "PredictedDirection"], _direction_from_probability(probability)),
                "ActualDirection": actual_direction,
                "StrategyReturn": _safe_float(strategy_return, default=np.nan),
                "BenchmarkReturn": _safe_float(benchmark_return, default=np.nan),
                "VsBuyHold": _safe_float(vs_buy_hold, default=np.nan),
                "RealizedReturn": _safe_float(_lookup_value(row, ["RealizedReturn", "TradeReturn", "Return"], strategy_return), default=np.nan),
                "MaxDrawdown": _safe_float(max_drawdown, default=np.nan),
                "WinLoss": _lookup_value(row, ["WinLoss"], _win_loss_label(actual_direction, strategy_return)),
                "BeatBenchmark": _beat_benchmark_value(vs_buy_hold, strategy_return, benchmark_return),
                "CostApplied": _safe_float(_lookup_value(row, ["CostApplied", "TransactionCost"], np.nan), default=np.nan),
                "RegimeLabel": _lookup_value(row, ["RegimeLabel"], ""),
                "ConfidenceSource": "RawProbability",
                "DataQualityFlag": "",
                "EvidenceMode": "RawTradeLevel",
                "Warnings": "",
            }
            ledger_row["Warnings"] = _ledger_row_warnings(ledger_row)
            ledger_row["DataQualityFlag"] = _data_quality_flag(ledger_row)
            rows.append(ledger_row)
    return pd.DataFrame(rows, columns=TRADE_EVIDENCE_LEDGER_COLUMNS) if rows else pd.DataFrame(columns=TRADE_EVIDENCE_LEDGER_COLUMNS)


def _full_evidence_to_ledger(full_evidence_table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if full_evidence_table is None or full_evidence_table.empty or not {"Asset", "Horizon"}.issubset(full_evidence_table.columns):
        return pd.DataFrame(columns=TRADE_EVIDENCE_LEDGER_COLUMNS)
    df = _normalise_horizon_column(full_evidence_table)
    rows: List[Dict[str, Any]] = []
    for row_index, row in df.iterrows():
        strategy = _lookup_value(row, ["LockedTestStrategyReturn_%", "AvgLockedStrategyReturn_%", "StrategyReturn_%"])
        benchmark = _lookup_value(row, ["LockedTestBuyHoldReturn_%", "AvgLockedBuyHoldReturn_%", "BuyHoldReturn_%"])
        vs_buy_hold = _lookup_value(row, ["LockedTestVsBuyHold_%", "AvgLockedVsBuyHold_%", "MedianLockedVsBuyHold_%", "VsBuyHold_%"])
        drawdown = _lookup_value(row, ["LockedTestMaxDrawdown_%", "WorstLockedMaxDrawdown_%", "AvgLockedMaxDrawdown_%", "MaxDrawdown_%"])
        ledger_row = {
            "LedgerId": f"8C-{row_index + 1}",
            "Asset": str(row.get("Asset", "")),
            "Horizon": _safe_int(row.get("Horizon"), default=0),
            "ModelName": _lookup_value(row, ["ModelName", "BestModel", "ModelDepth"], ""),
            "PolicyName": _lookup_value(row, ["SignalMode", "ThresholdPolicy"], "Validation-locked window aggregate"),
            "SourcePhase": "8C",
            "WindowMode": _lookup_value(row, ["WindowMode"], ""),
            "ValidationWindow": _lookup_value(row, ["ValidationWindow"], np.nan),
            "TestWindow": _lookup_value(row, ["TestWindow"], np.nan),
            "StepSize": _lookup_value(row, ["StepSize"], np.nan),
            "TransactionCost": _lookup_value(row, ["TransactionCost"], np.nan),
            "SignalDate": pd.NaT,
            "EntryDate": pd.NaT,
            "ExitDate": pd.NaT,
            "HoldingPeriod": _safe_int(row.get("Horizon"), default=0),
            "ProbabilityUp": np.nan,
            "ProbabilityBin": "",
            "SignalTaken": _safe_bool(_lookup_value(row, ["ValidConfiguration"], True), default=True),
            "TradeCountContribution": _safe_float(_lookup_value(row, ["LockedTestTrades", "AvgTradesPerWindow", "TradeCount"], 0.0), default=0.0),
            "DirectionPrediction": "",
            "ActualDirection": np.nan,
            "StrategyReturn": _safe_float(strategy, default=np.nan),
            "BenchmarkReturn": _safe_float(benchmark, default=np.nan),
            "VsBuyHold": _safe_float(vs_buy_hold, default=np.nan),
            "RealizedReturn": _safe_float(strategy, default=np.nan),
            "MaxDrawdown": _safe_float(drawdown, default=np.nan),
            "WinLoss": _win_loss_label(np.nan, strategy),
            "BeatBenchmark": _beat_benchmark_value(vs_buy_hold, strategy, benchmark),
            "CostApplied": _safe_float(_lookup_value(row, ["TransactionCost"], np.nan), default=np.nan),
            "RegimeLabel": _lookup_value(row, ["RegimeLabel"], ""),
            "ConfidenceSource": "WindowAggregateNoTradeProbability",
            "DataQualityFlag": "",
            "EvidenceMode": "WindowAggregate",
            "Warnings": "",
        }
        ledger_row["Warnings"] = _ledger_row_warnings(ledger_row)
        ledger_row["DataQualityFlag"] = _data_quality_flag(ledger_row)
        rows.append(ledger_row)
    return pd.DataFrame(rows, columns=TRADE_EVIDENCE_LEDGER_COLUMNS) if rows else pd.DataFrame(columns=TRADE_EVIDENCE_LEDGER_COLUMNS)


def _policy_tables_to_ledger(policy_sensitivity_table: Optional[pd.DataFrame], coverage_edge_frontier_table: Optional[pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for table in [policy_sensitivity_table, coverage_edge_frontier_table]:
        if table is not None and not table.empty and {"Asset", "Horizon"}.issubset(table.columns):
            frames.append(_normalise_horizon_column(table))
    if not frames:
        return pd.DataFrame(columns=TRADE_EVIDENCE_LEDGER_COLUMNS)
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    rows: List[Dict[str, Any]] = []
    for row_index, row in df.iterrows():
        min_probability = _safe_float(_lookup_value(row, ["MinProbability"], np.nan), default=np.nan)
        max_probability = _safe_float(_lookup_value(row, ["MaxProbability"], np.nan), default=np.nan)
        probability_bin = ""
        confidence_source = "PolicyAggregateNoTradeProbability"
        if np.isfinite(min_probability) or np.isfinite(max_probability):
            probability_bin = f"{min_probability if np.isfinite(min_probability) else ''}-{max_probability if np.isfinite(max_probability) else ''}"
            confidence_source = "PolicyProbabilityBandAggregate"
        strategy = _lookup_value(row, ["StrategyReturn_%", "StrategyReturn"], np.nan)
        benchmark = _lookup_value(row, ["BuyHoldReturn_%", "BenchmarkReturn"], np.nan)
        vs_buy_hold = _lookup_value(row, ["VsBuyHold_%", "MedianVsBuyHold_%", "AvgVsBuyHold_%"], np.nan)
        ledger_row = {
            "LedgerId": f"8E-{row_index + 1}",
            "Asset": str(row.get("Asset", "")),
            "Horizon": _safe_int(row.get("Horizon"), default=0),
            "ModelName": _lookup_value(row, ["ModelName", "BestModel"], ""),
            "PolicyName": _lookup_value(row, ["PolicyType"], "Policy aggregate"),
            "SourcePhase": "8E",
            "WindowMode": "",
            "ValidationWindow": np.nan,
            "TestWindow": np.nan,
            "StepSize": np.nan,
            "TransactionCost": np.nan,
            "SignalDate": pd.NaT,
            "EntryDate": pd.NaT,
            "ExitDate": pd.NaT,
            "HoldingPeriod": _lookup_value(row, ["PolicyHorizon", "Horizon"], np.nan),
            "ProbabilityUp": np.nan,
            "ProbabilityBin": probability_bin,
            "SignalTaken": _safe_float(_lookup_value(row, ["TradeCount"], 0.0), default=0.0) > 0,
            "TradeCountContribution": _safe_float(_lookup_value(row, ["TradeCount"], 0.0), default=0.0),
            "DirectionPrediction": "",
            "ActualDirection": np.nan,
            "StrategyReturn": _safe_float(strategy, default=np.nan),
            "BenchmarkReturn": _safe_float(benchmark, default=np.nan),
            "VsBuyHold": _safe_float(vs_buy_hold, default=np.nan),
            "RealizedReturn": _safe_float(strategy, default=np.nan),
            "MaxDrawdown": _safe_float(_lookup_value(row, ["MaxDrawdown_%", "MaxDrawdown"], np.nan), default=np.nan),
            "WinLoss": _win_loss_label(np.nan, strategy),
            "BeatBenchmark": _beat_benchmark_value(vs_buy_hold, strategy, benchmark),
            "CostApplied": np.nan,
            "RegimeLabel": "",
            "ConfidenceSource": confidence_source,
            "DataQualityFlag": "",
            "EvidenceMode": "PolicyAggregate",
            "Warnings": "",
        }
        ledger_row["Warnings"] = _ledger_row_warnings(ledger_row)
        ledger_row["DataQualityFlag"] = _data_quality_flag(ledger_row)
        rows.append(ledger_row)
    return pd.DataFrame(rows, columns=TRADE_EVIDENCE_LEDGER_COLUMNS) if rows else pd.DataFrame(columns=TRADE_EVIDENCE_LEDGER_COLUMNS)


def _calibration_summary_to_ledger(calibration_summary_table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if calibration_summary_table is None or calibration_summary_table.empty or not {"Asset", "Horizon"}.issubset(calibration_summary_table.columns):
        return pd.DataFrame(columns=TRADE_EVIDENCE_LEDGER_COLUMNS)
    df = _normalise_horizon_column(calibration_summary_table)
    rows: List[Dict[str, Any]] = []
    for row_index, row in df.iterrows():
        raw_available = _safe_bool(_lookup_value(row, ["RawProbabilityOutcomesAvailable"], False), default=False)
        mode = "CalibrationProxy" if raw_available else "InsufficientData"
        ledger_row = {
            "LedgerId": f"8F-{row_index + 1}",
            "Asset": str(row.get("Asset", "")),
            "Horizon": _safe_int(row.get("Horizon"), default=0),
            "ModelName": "",
            "PolicyName": _lookup_value(row, ["CalibrationGrade"], "Probability calibration summary"),
            "SourcePhase": "8F",
            "WindowMode": "",
            "ValidationWindow": np.nan,
            "TestWindow": np.nan,
            "StepSize": np.nan,
            "TransactionCost": np.nan,
            "SignalDate": pd.NaT,
            "EntryDate": pd.NaT,
            "ExitDate": pd.NaT,
            "HoldingPeriod": _safe_int(row.get("Horizon"), default=0),
            "ProbabilityUp": np.nan,
            "ProbabilityBin": "",
            "SignalTaken": _safe_float(_lookup_value(row, ["TotalTrades"], 0.0), default=0.0) > 0,
            "TradeCountContribution": _safe_float(_lookup_value(row, ["TotalTrades"], 0.0), default=0.0),
            "DirectionPrediction": "",
            "ActualDirection": np.nan,
            "StrategyReturn": np.nan,
            "BenchmarkReturn": np.nan,
            "VsBuyHold": np.nan,
            "RealizedReturn": np.nan,
            "MaxDrawdown": _safe_float(_lookup_value(row, ["BestFilterMaxDrawdown_%"], np.nan), default=np.nan),
            "WinLoss": "",
            "BeatBenchmark": np.nan,
            "CostApplied": np.nan,
            "RegimeLabel": "",
            "ConfidenceSource": "Phase8FCalibrationSummary",
            "DataQualityFlag": "",
            "EvidenceMode": mode,
            "Warnings": _lookup_value(row, ["MainWarning"], ""),
        }
        ledger_warnings = _ledger_row_warnings(ledger_row)
        ledger_row["Warnings"] = _join_warnings([ledger_row["Warnings"], ledger_warnings])
        ledger_row["DataQualityFlag"] = _data_quality_flag(ledger_row)
        rows.append(ledger_row)
    return pd.DataFrame(rows, columns=TRADE_EVIDENCE_LEDGER_COLUMNS) if rows else pd.DataFrame(columns=TRADE_EVIDENCE_LEDGER_COLUMNS)


def _filter_ledger_candidates(
    ledger: pd.DataFrame,
    candidate_filter: str,
    selected_assets: Optional[Iterable[str]],
    selected_horizons: Optional[Iterable[int]],
) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    out = ledger.copy()
    mode = str(candidate_filter or "all").lower()
    if mode in {"specific", "specific asset/horizon"}:
        assets = set(str(asset) for asset in (selected_assets or []))
        horizons = set(int(horizon) for horizon in (selected_horizons or []))
        if assets:
            out = out[out["Asset"].astype(str).isin(assets)]
        if horizons:
            out = out[out["Horizon"].astype(int).isin(horizons)]
    return out.reset_index(drop=True)


def _ledger_quality_summary(ledger: pd.DataFrame, configured_assets: Optional[Iterable[str]], configured_horizons: Optional[Iterable[int]]) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=LEDGER_QUALITY_COLUMNS)
    total_rows = int(len(ledger))
    raw_rows = int(ledger["EvidenceMode"].eq("RawTradeLevel").sum())
    aggregate_rows = int(total_rows - raw_rows)
    probability_rows = int(ledger["ProbabilityUp"].notna().sum())
    actual_rows = int(ledger["ActualDirection"].notna().sum())
    signal_missing = float(ledger["SignalDate"].isna().mean() * 100.0)
    entry_exit_missing = float((ledger["EntryDate"].isna() | ledger["ExitDate"].isna()).mean() * 100.0)
    benchmark_available = float((ledger["BenchmarkReturn"].notna() | ledger["VsBuyHold"].notna()).mean() * 100.0)
    drawdown_available = float(ledger["MaxDrawdown"].notna().mean() * 100.0)
    ready_rows = int((ledger["ProbabilityUp"].notna() & ledger["ActualDirection"].notna()).sum())
    assets_covered = int(ledger["Asset"].nunique())
    horizons_covered = int(ledger["Horizon"].nunique())
    avg_trade_contribution = float(pd.to_numeric(ledger["TradeCountContribution"], errors="coerce").fillna(0.0).mean())
    configured_asset_count = len(list(configured_assets or []))
    configured_horizon_count = len(list(configured_horizons or []))
    readiness = _clip(ready_rows / max(total_rows, 1) * 100.0)
    raw_coverage = _clip(raw_rows / max(total_rows, 1) * 100.0)
    ledger_quality = _clip(
        readiness * 0.35
        + raw_coverage * 0.25
        + benchmark_available * 0.15
        + drawdown_available * 0.10
        + (100.0 - signal_missing) * 0.05
        + (100.0 - entry_exit_missing) * 0.05
        + min(assets_covered / max(configured_asset_count, assets_covered, 1) * 100.0, 100.0) * 0.05
    )
    warnings: List[str] = []
    if raw_rows == 0:
        warnings.extend(["MissingRawTradeLogs", "AggregateProxyOnly"])
    if probability_rows == 0 or ledger["ProbabilityUp"].isna().mean() >= 0.5:
        warnings.append("MissingProbability")
    if actual_rows == 0 or ledger["ActualDirection"].isna().mean() >= 0.5:
        warnings.append("MissingOutcomeDirection")
    if signal_missing >= 50.0 or entry_exit_missing >= 50.0:
        warnings.append("MissingTradeDates")
    if benchmark_available < 50.0:
        warnings.append("BenchmarkMissing")
    if drawdown_available < 50.0:
        warnings.append("DrawdownMissing")
    if readiness < 50.0:
        warnings.append("NotCalibrationReady")
    if avg_trade_contribution < 3.0:
        warnings.append("LowTradeCoverage")
    if configured_asset_count and assets_covered < configured_asset_count:
        warnings.append("AssetCoverageIncomplete")
    if configured_horizon_count and horizons_covered < configured_horizon_count:
        warnings.append("HorizonCoverageIncomplete")
    return pd.DataFrame(
        [
            {
                "RowsInLedger": total_rows,
                "AssetsCovered": assets_covered,
                "HorizonsCovered": horizons_covered,
                "RawTradeLevelRows": raw_rows,
                "AggregateDerivedRows": aggregate_rows,
                "ProbabilityRows": probability_rows,
                "MissingProbabilityRate_%": round(float(ledger["ProbabilityUp"].isna().mean() * 100.0), 4),
                "MissingActualDirectionRate_%": round(float(ledger["ActualDirection"].isna().mean() * 100.0), 4),
                "MissingSignalDateRate_%": round(signal_missing, 4),
                "MissingEntryExitDateRate_%": round(entry_exit_missing, 4),
                "AverageTradeCountContribution": round(avg_trade_contribution, 4),
                "BenchmarkComparisonAvailability_%": round(benchmark_available, 4),
                "DrawdownAvailability_%": round(drawdown_available, 4),
                "CalibrationReadyRows": ready_rows,
                "CalibrationReadinessScore": round(readiness, 4),
                "LedgerQualityScore": round(ledger_quality, 4),
                "MainWarning": _join_warnings(warnings),
                "PromotesCandidates": False,
                "ProductionReadyLabelAllowed": False,
            }
        ],
        columns=LEDGER_QUALITY_COLUMNS,
    )


def _ledger_group_summaries(ledger: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if ledger.empty:
        return {
            "coverage": pd.DataFrame(columns=ASSET_HORIZON_COVERAGE_COLUMNS),
            "probability": pd.DataFrame(columns=PROBABILITY_OUTCOME_AVAILABILITY_COLUMNS),
            "distribution": pd.DataFrame(columns=TRADE_OUTCOME_DISTRIBUTION_COLUMNS),
            "benchmark": pd.DataFrame(columns=BENCHMARK_OUTCOME_COLUMNS),
            "drawdown": pd.DataFrame(columns=DRAWDOWN_OUTCOME_COLUMNS),
        }
    coverage_rows: List[Dict[str, Any]] = []
    probability_rows: List[Dict[str, Any]] = []
    distribution_rows: List[Dict[str, Any]] = []
    benchmark_rows: List[Dict[str, Any]] = []
    drawdown_rows: List[Dict[str, Any]] = []
    for (asset, horizon), group in ledger.groupby(["Asset", "Horizon"], dropna=False):
        rows = int(len(group))
        raw_rows = int(group["EvidenceMode"].eq("RawTradeLevel").sum())
        aggregate_rows = rows - raw_rows
        probability_count = int(group["ProbabilityUp"].notna().sum())
        actual_count = int(group["ActualDirection"].notna().sum())
        ready_count = int((group["ProbabilityUp"].notna() & group["ActualDirection"].notna()).sum())
        trade_contribution = float(pd.to_numeric(group["TradeCountContribution"], errors="coerce").fillna(0.0).sum())
        warnings = sorted({warning.strip() for warnings_text in group["Warnings"].astype(str) for warning in warnings_text.split(";") if warning.strip()})
        if trade_contribution < 3.0:
            warnings.append("LowTradeCoverage")
        evidence_modes = "; ".join(sorted(group["EvidenceMode"].dropna().astype(str).unique()))
        readiness = _clip(ready_count / max(rows, 1) * 100.0)
        coverage_rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "LedgerRows": rows,
                "RawTradeLevelRows": raw_rows,
                "AggregateDerivedRows": aggregate_rows,
                "TradeCountContribution": round(trade_contribution, 4),
                "ProbabilityRows": probability_count,
                "ActualDirectionRows": actual_count,
                "CalibrationReadyRows": ready_count,
                "CalibrationReadinessScore": round(readiness, 4),
                "EvidenceModes": evidence_modes,
                "Warnings": _join_warnings(warnings),
            }
        )
        probability_rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "ProbabilityRows": probability_count,
                "ActualDirectionRows": actual_count,
                "RowsWithProbabilityAndOutcome": ready_count,
                "MissingProbabilityRate_%": round(float(group["ProbabilityUp"].isna().mean() * 100.0), 4),
                "MissingActualDirectionRate_%": round(float(group["ActualDirection"].isna().mean() * 100.0), 4),
                "CalibrationReady": bool(ready_count >= 10),
                "ConfidenceSourceSummary": "; ".join(sorted(group["ConfidenceSource"].dropna().astype(str).unique())),
                "Warnings": _join_warnings(warnings),
            }
        )
        strategy = pd.to_numeric(group["StrategyReturn"], errors="coerce")
        win_loss = group["WinLoss"].astype(str)
        wins = int(win_loss.str.lower().eq("win").sum())
        losses = int(win_loss.str.lower().eq("loss").sum())
        distribution_rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "Rows": rows,
                "TradeCountContribution": round(trade_contribution, 4),
                "Wins": wins,
                "Losses": losses,
                "WinRate_%": round(wins / max(wins + losses, 1) * 100.0, 4),
                "AvgStrategyReturn": round(float(strategy.mean()), 4) if strategy.notna().any() else np.nan,
                "MedianStrategyReturn": round(float(strategy.median()), 4) if strategy.notna().any() else np.nan,
                "PositiveReturnRows": int(strategy.gt(0).sum()),
                "NegativeReturnRows": int(strategy.lt(0).sum()),
                "FailedRowsVisible": bool(strategy.lt(0).any() or losses > 0),
            }
        )
        benchmark_available = group[group["BenchmarkReturn"].notna() | group["VsBuyHold"].notna()]
        beat = group["BeatBenchmark"].dropna()
        benchmark_rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "RowsWithBenchmark": int(len(benchmark_available)),
                "BeatBenchmarkRows": int(pd.Series(beat).astype(bool).sum()) if len(beat) else 0,
                "BeatBenchmarkRate_%": round(float(pd.Series(beat).astype(bool).mean() * 100.0), 4) if len(beat) else 0.0,
                "AvgBenchmarkReturn": round(float(pd.to_numeric(group["BenchmarkReturn"], errors="coerce").mean()), 4) if group["BenchmarkReturn"].notna().any() else np.nan,
                "AvgVsBuyHold": round(float(pd.to_numeric(group["VsBuyHold"], errors="coerce").mean()), 4) if group["VsBuyHold"].notna().any() else np.nan,
                "NegativeVsBuyHoldRows": int(pd.to_numeric(group["VsBuyHold"], errors="coerce").lt(0).sum()),
                "BenchmarkMissingRate_%": round(float((group["BenchmarkReturn"].isna() & group["VsBuyHold"].isna()).mean() * 100.0), 4),
            }
        )
        drawdowns = pd.to_numeric(group["MaxDrawdown"], errors="coerce")
        drawdown_rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "RowsWithDrawdown": int(drawdowns.notna().sum()),
                "AvgMaxDrawdown": round(float(drawdowns.mean()), 4) if drawdowns.notna().any() else np.nan,
                "WorstMaxDrawdown": round(float(drawdowns.min()), 4) if drawdowns.notna().any() else np.nan,
                "DrawdownMissingRate_%": round(float(drawdowns.isna().mean() * 100.0), 4),
                "DrawdownRiskRows": int(drawdowns.le(-25.0).sum()),
            }
        )
    return {
        "coverage": pd.DataFrame(coverage_rows, columns=ASSET_HORIZON_COVERAGE_COLUMNS),
        "probability": pd.DataFrame(probability_rows, columns=PROBABILITY_OUTCOME_AVAILABILITY_COLUMNS),
        "distribution": pd.DataFrame(distribution_rows, columns=TRADE_OUTCOME_DISTRIBUTION_COLUMNS),
        "benchmark": pd.DataFrame(benchmark_rows, columns=BENCHMARK_OUTCOME_COLUMNS),
        "drawdown": pd.DataFrame(drawdown_rows, columns=DRAWDOWN_OUTCOME_COLUMNS),
    }


def _ledger_warning_table(ledger: pd.DataFrame, quality_summary: pd.DataFrame, configured_assets: Optional[Iterable[str]], configured_horizons: Optional[Iterable[int]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in ledger.iterrows():
        for warning in [warning.strip() for warning in str(row.get("Warnings", "")).split(";") if warning.strip()]:
            severity = "High" if warning in {"MissingRawTradeLogs", "MissingProbability", "MissingOutcomeDirection", "NotCalibrationReady"} else "Medium"
            rows.append({"LedgerId": row.get("LedgerId", ""), "Asset": row.get("Asset", ""), "Horizon": row.get("Horizon", np.nan), "WarningType": warning, "Severity": severity, "Message": f"{warning} in trade evidence ledger row."})
    if not quality_summary.empty:
        for warning in [warning.strip() for warning in str(quality_summary.iloc[0].get("MainWarning", "")).split(";") if warning.strip()]:
            rows.append({"LedgerId": "SUMMARY", "Asset": "ALL", "Horizon": np.nan, "WarningType": warning, "Severity": "High", "Message": f"{warning} at ledger summary level."})
    return pd.DataFrame(rows, columns=["LedgerId", "Asset", "Horizon", "WarningType", "Severity", "Message"]) if rows else pd.DataFrame(columns=["LedgerId", "Asset", "Horizon", "WarningType", "Severity", "Message"])


def _ledger_next_actions(coverage_table: pd.DataFrame, probability_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if coverage_table.empty:
        return pd.DataFrame(columns=["Asset", "Horizon", "NextResearchAction", "ActionPriority"])
    probability_lookup = probability_table.set_index(["Asset", "Horizon"]).to_dict("index") if not probability_table.empty else {}
    for _, row in coverage_table.iterrows():
        key = (row.get("Asset"), row.get("Horizon"))
        probability_row = probability_lookup.get(key, {})
        warnings = {warning.strip() for warning in str(row.get("Warnings", "")).split(";") if warning.strip()}
        if not bool(probability_row.get("CalibrationReady", False)):
            action = "Collect/export raw signal trade logs with ProbabilityUp and ActualDirection before true calibration."
            priority = "High"
        elif "AggregateProxyOnly" in warnings:
            action = "Replace aggregate-derived ledger rows with raw trade-level records for this asset/horizon."
            priority = "High"
        elif "BenchmarkMissing" in warnings:
            action = "Add benchmark returns to the trade ledger before benchmark-relative conclusions."
            priority = "Medium"
        else:
            action = "Use this ledger as evidence input for future calibration and paper-trading validation only."
            priority = "Medium"
        rows.append({"Asset": row.get("Asset"), "Horizon": row.get("Horizon"), "NextResearchAction": action, "ActionPriority": priority})
    return pd.DataFrame(rows, columns=["Asset", "Horizon", "NextResearchAction", "ActionPriority"])


def _run_trade_evidence_ledger_impl(
    *,
    calibration_summary_table: Optional[pd.DataFrame] = None,
    policy_sensitivity_table: Optional[pd.DataFrame] = None,
    coverage_edge_frontier_table: Optional[pd.DataFrame] = None,
    full_evidence_table: Optional[pd.DataFrame] = None,
    raw_trade_logs: Optional[Any] = None,
    candidate_filter: str = "all",
    selected_assets: Optional[Iterable[str]] = None,
    selected_horizons: Optional[Iterable[int]] = None,
    configured_assets: Optional[Iterable[str]] = None,
    configured_horizons: Optional[Iterable[int]] = None,
) -> TradeEvidenceLedgerReport:
    settings = {
        "phase": "8G",
        "purpose": "multi_asset_trade_level_probability_evidence_ledger_only",
        "does_not_promote_candidates": True,
        "production_ready_label_allowed": False,
        "candidate_filter": candidate_filter,
    }
    frames = [
        _raw_trade_log_to_ledger(raw_trade_logs),
        _full_evidence_to_ledger(full_evidence_table),
        _policy_tables_to_ledger(policy_sensitivity_table, coverage_edge_frontier_table),
        _calibration_summary_to_ledger(calibration_summary_table),
    ]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return _empty_trade_evidence_ledger_report(settings)
    ledger = pd.concat(frames, ignore_index=True)
    for col in TRADE_EVIDENCE_LEDGER_COLUMNS:
        if col not in ledger.columns:
            ledger[col] = np.nan
    ledger = _filter_ledger_candidates(ledger[TRADE_EVIDENCE_LEDGER_COLUMNS], candidate_filter, selected_assets, selected_horizons)
    if ledger.empty:
        return _empty_trade_evidence_ledger_report(settings)
    ledger["Horizon"] = pd.to_numeric(ledger["Horizon"], errors="coerce").fillna(0).astype(int)
    ledger["TradeCountContribution"] = pd.to_numeric(ledger["TradeCountContribution"], errors="coerce").fillna(0.0)
    configured_assets = list(configured_assets or get_asset_names())
    configured_horizons = list(configured_horizons or DEFAULT_HORIZONS)
    quality_summary = _ledger_quality_summary(ledger, configured_assets, configured_horizons)
    summaries = _ledger_group_summaries(ledger)
    warning_table = _ledger_warning_table(ledger, quality_summary, configured_assets, configured_horizons)
    next_actions = _ledger_next_actions(summaries["coverage"], summaries["probability"])
    return TradeEvidenceLedgerReport(
        ledger_table=ledger[TRADE_EVIDENCE_LEDGER_COLUMNS].reset_index(drop=True),
        ledger_quality_summary=quality_summary[LEDGER_QUALITY_COLUMNS],
        asset_horizon_coverage_table=summaries["coverage"],
        probability_outcome_availability_table=summaries["probability"],
        trade_outcome_distribution_table=summaries["distribution"],
        benchmark_outcome_table=summaries["benchmark"],
        drawdown_outcome_table=summaries["drawdown"],
        ledger_warning_table=warning_table,
        next_research_action_table=next_actions,
        settings=settings,
    )


def run_trade_evidence_ledger(*args: Any, **kwargs: Any) -> TradeEvidenceLedgerReport:
    """Stable public Phase 8G entrypoint used by app.py and tests."""
    return _run_trade_evidence_ledger_impl(*args, **kwargs)


RAW_TRADE_LOG_COLUMNS = [
    "TradeId",
    "Asset",
    "Horizon",
    "ModelName",
    "PolicyName",
    "SourcePhase",
    "WindowId",
    "WindowMode",
    "ValidationWindow",
    "TestWindow",
    "StepSize",
    "TransactionCost",
    "SignalDate",
    "EntryDate",
    "ExitDate",
    "HoldingPeriod",
    "ProbabilityUp",
    "ProbabilityBin",
    "PredictedDirection",
    "ActualDirection",
    "SignalTaken",
    "EntryPrice",
    "ExitPrice",
    "RealizedReturn",
    "BenchmarkReturn",
    "VsBuyHold",
    "MaxDrawdownDuringTrade",
    "WinLoss",
    "BeatBenchmark",
    "CostApplied",
    "RegimeLabel",
    "ConfidenceSource",
    "EvidenceMode",
    "DataQualityFlag",
    "Warnings",
]

RAW_LOG_QUALITY_COLUMNS = [
    "TotalRows",
    "RawTradeLevelRows",
    "ReconstructedTradeLevelRows",
    "AggregateFallbackRows",
    "AssetsCovered",
    "HorizonsCovered",
    "CandidateCoverageRate_%",
    "MissingProbabilityRate_%",
    "MissingSignalDateRate_%",
    "MissingEntryExitDateRate_%",
    "MissingActualDirectionRate_%",
    "BenchmarkAvailabilityRate_%",
    "DrawdownAvailabilityRate_%",
    "CalibrationReadyRowCount",
    "CalibrationReadinessScore",
    "RawLogQualityScore",
    "MainWarning",
    "PromotesCandidates",
    "ProductionReadyLabelAllowed",
]

RAW_COVERAGE_COLUMNS = [
    "Asset",
    "Horizon",
    "Rows",
    "RawTradeLevelRows",
    "ReconstructedTradeLevelRows",
    "AggregateFallbackRows",
    "SignalTakenRows",
    "NoTradeRows",
    "CalibrationReadyRows",
    "RawCoverageRate_%",
    "Warnings",
]

RAW_PROBABILITY_READINESS_COLUMNS = [
    "Asset",
    "Horizon",
    "Rows",
    "ProbabilityRows",
    "ActualDirectionRows",
    "RowsWithProbabilityAndOutcome",
    "MissingProbabilityRate_%",
    "MissingActualDirectionRate_%",
    "CalibrationReady",
    "Warnings",
]

RAW_TRADE_DISTRIBUTION_COLUMNS = [
    "Asset",
    "Horizon",
    "Rows",
    "SignalTakenRows",
    "Wins",
    "Losses",
    "WinRate_%",
    "AvgRealizedReturn",
    "MedianRealizedReturn",
    "PositiveReturnRows",
    "NegativeReturnRows",
    "LosingTradesVisible",
]

RAW_BENCHMARK_COMPARISON_COLUMNS = [
    "Asset",
    "Horizon",
    "RowsWithBenchmark",
    "BeatBenchmarkRows",
    "BeatBenchmarkRate_%",
    "AvgBenchmarkReturn",
    "AvgVsBuyHold",
    "NegativeVsBuyHoldRows",
    "BenchmarkAvailabilityRate_%",
]

RAW_DRAWDOWN_COLUMNS = [
    "Asset",
    "Horizon",
    "RowsWithDrawdown",
    "AvgMaxDrawdownDuringTrade",
    "WorstMaxDrawdownDuringTrade",
    "DrawdownAvailabilityRate_%",
    "DrawdownRiskRows",
]

RAW_NO_TRADE_COLUMNS = [
    "Asset",
    "Horizon",
    "NoTradeRows",
    "SkippedSignalRows",
    "MissingProbabilityRows",
    "MissingOutcomeRows",
    "Warnings",
]


def _empty_raw_trade_log_exporter_report(settings: Optional[Dict[str, Any]] = None) -> RawTradeLogExporterReport:
    return RawTradeLogExporterReport(
        raw_signal_trade_log_table=pd.DataFrame(columns=RAW_TRADE_LOG_COLUMNS),
        raw_log_quality_summary=pd.DataFrame(columns=RAW_LOG_QUALITY_COLUMNS),
        asset_horizon_raw_coverage_table=pd.DataFrame(columns=RAW_COVERAGE_COLUMNS),
        probability_outcome_readiness_table=pd.DataFrame(columns=RAW_PROBABILITY_READINESS_COLUMNS),
        trade_outcome_distribution_table=pd.DataFrame(columns=RAW_TRADE_DISTRIBUTION_COLUMNS),
        benchmark_comparison_table=pd.DataFrame(columns=RAW_BENCHMARK_COMPARISON_COLUMNS),
        drawdown_during_trade_table=pd.DataFrame(columns=RAW_DRAWDOWN_COLUMNS),
        no_trade_skipped_signal_table=pd.DataFrame(columns=RAW_NO_TRADE_COLUMNS),
        warning_table=pd.DataFrame(columns=["TradeId", "Asset", "Horizon", "WarningType", "Severity", "Message"]),
        next_research_action_table=pd.DataFrame(columns=["Asset", "Horizon", "NextResearchAction", "ActionPriority"]),
        settings=settings or {},
    )


def _raw_mode_from_row(row: Dict[str, Any], *, raw_source: bool, fallback_mode: str) -> str:
    if fallback_mode in {"WindowAggregateFallback", "PolicyAggregateFallback", "InsufficientData"}:
        return fallback_mode
    has_core_raw = (
        raw_source
        and not pd.isna(row.get("SignalDate"))
        and not pd.isna(row.get("EntryDate"))
        and not pd.isna(row.get("ExitDate"))
        and not pd.isna(row.get("ProbabilityUp"))
        and not pd.isna(row.get("ActualDirection"))
        and not pd.isna(row.get("RealizedReturn"))
    )
    if has_core_raw:
        return "RawTradeLevel"
    reconstructable = (
        not pd.isna(row.get("ProbabilityUp"))
        and not pd.isna(row.get("ActualDirection"))
        and not pd.isna(row.get("RealizedReturn"))
        and (not pd.isna(row.get("EntryDate")) or not pd.isna(row.get("SignalDate")))
    )
    if reconstructable:
        return "ReconstructedTradeLevel"
    return "InsufficientData"


def _raw_log_row_warnings(row: Dict[str, Any]) -> str:
    warnings: List[str] = []
    mode = str(row.get("EvidenceMode", ""))
    if mode == "ReconstructedTradeLevel":
        warnings.append("ReconstructedOnly")
    if mode in {"WindowAggregateFallback", "PolicyAggregateFallback"}:
        warnings.append("AggregateFallbackOnly")
    if mode == "InsufficientData":
        warnings.append("DataQualityWeak")
    if pd.isna(row.get("ProbabilityUp")):
        warnings.append("MissingProbability")
    if pd.isna(row.get("SignalDate")):
        warnings.append("MissingTradeDates")
    if pd.isna(row.get("EntryDate")) or pd.isna(row.get("ExitDate")):
        warnings.append("MissingTradeDates")
    if pd.isna(row.get("ActualDirection")):
        warnings.append("MissingOutcomeDirection")
    if pd.isna(row.get("BenchmarkReturn")) and pd.isna(row.get("VsBuyHold")):
        warnings.append("MissingBenchmark")
    if pd.isna(row.get("MaxDrawdownDuringTrade")):
        warnings.append("MissingDrawdown")
    if "MissingProbability" in warnings or "MissingOutcomeDirection" in warnings:
        warnings.append("NotCalibrationReady")
    return _join_warnings(warnings)


def _raw_data_quality_flag(row: Dict[str, Any]) -> str:
    warnings = {w.strip() for w in str(row.get("Warnings", "")).split(";") if w.strip()}
    if row.get("EvidenceMode") == "RawTradeLevel" and not warnings.intersection({"MissingProbability", "MissingOutcomeDirection", "MissingTradeDates"}):
        return "CompleteRawTrade"
    if row.get("EvidenceMode") == "RawTradeLevel":
        return "IncompleteRawTrade"
    if row.get("EvidenceMode") == "ReconstructedTradeLevel":
        return "ReconstructedLimited"
    if row.get("EvidenceMode") in {"WindowAggregateFallback", "PolicyAggregateFallback"}:
        return "AggregateFallback"
    return "InsufficientData"


def _raw_row_from_series(row: pd.Series, *, trade_id: str, source_phase: str, raw_source: bool, fallback_mode: str = "") -> Dict[str, Any]:
    probability = _lookup_value(row, ["ProbabilityUp", "Probability", "P_up", "PredictedProbability", "DirectionProbability"], np.nan)
    probability = _as_probability_series(pd.Series([probability])).iloc[0] if not pd.isna(probability) else np.nan
    signal_date = _as_optional_date(_lookup_value(row, ["SignalDate", "Date", "TradeDate"], pd.NaT))
    entry_date = _as_optional_date(_lookup_value(row, ["EntryDate", "Date", "TradeDate"], pd.NaT))
    exit_date = _as_optional_date(_lookup_value(row, ["ExitDate"], pd.NaT))
    actual_direction = _normalise_direction_value(_lookup_value(row, ["ActualDirection", "FutureDirection", "Direction", "Win", "WinLoss"], np.nan))
    realized_return = _lookup_value(row, ["RealizedReturn", "TradeReturn", "Return", "StrategyReturn", "StrategyReturn_%", "StrategyReturnAfterCost"], np.nan)
    benchmark_return = _lookup_value(row, ["BenchmarkReturn", "BuyHoldReturn", "BuyHoldReturn_%", "BenchmarkReturn_%"], np.nan)
    vs_buy_hold = _lookup_value(row, ["VsBuyHold", "VsBuyHold_%", "StrategyMinusBuyHold", "LockedTestVsBuyHold_%", "AvgLockedVsBuyHold_%"], np.nan)
    drawdown = _lookup_value(row, ["MaxDrawdownDuringTrade", "MaxDrawdown", "MaxDrawdown_%", "LockedTestMaxDrawdown_%", "WorstLockedMaxDrawdown_%"], np.nan)
    out = {
        "TradeId": trade_id,
        "Asset": str(row.get("Asset", "")),
        "Horizon": _safe_int(row.get("Horizon"), default=0),
        "ModelName": _lookup_value(row, ["ModelName", "BestModel", "Model"], ""),
        "PolicyName": _lookup_value(row, ["PolicyName", "PolicyType", "SignalMode"], "Raw signal policy" if raw_source else "Fallback policy"),
        "SourcePhase": source_phase,
        "WindowId": _lookup_value(row, ["WindowId"], ""),
        "WindowMode": _lookup_value(row, ["WindowMode"], ""),
        "ValidationWindow": _lookup_value(row, ["ValidationWindow"], np.nan),
        "TestWindow": _lookup_value(row, ["TestWindow"], np.nan),
        "StepSize": _lookup_value(row, ["StepSize"], np.nan),
        "TransactionCost": _lookup_value(row, ["TransactionCost"], np.nan),
        "SignalDate": signal_date,
        "EntryDate": entry_date,
        "ExitDate": exit_date,
        "HoldingPeriod": _lookup_value(row, ["HoldingPeriod", "HoldingDays", "Horizon"], np.nan),
        "ProbabilityUp": probability,
        "ProbabilityBin": _lookup_value(row, ["ProbabilityBin"], _probability_bin_from_value(probability)),
        "PredictedDirection": _lookup_value(row, ["PredictedDirection", "DirectionPrediction"], _direction_from_probability(probability)),
        "ActualDirection": actual_direction,
        "SignalTaken": _safe_bool(_lookup_value(row, ["SignalTaken"], True), default=True),
        "EntryPrice": _safe_float(_lookup_value(row, ["EntryPrice"], np.nan), default=np.nan),
        "ExitPrice": _safe_float(_lookup_value(row, ["ExitPrice"], np.nan), default=np.nan),
        "RealizedReturn": _safe_float(realized_return, default=np.nan),
        "BenchmarkReturn": _safe_float(benchmark_return, default=np.nan),
        "VsBuyHold": _safe_float(vs_buy_hold, default=np.nan),
        "MaxDrawdownDuringTrade": _safe_float(drawdown, default=np.nan),
        "WinLoss": _lookup_value(row, ["WinLoss"], _win_loss_label(actual_direction, realized_return)),
        "BeatBenchmark": _beat_benchmark_value(vs_buy_hold, realized_return, benchmark_return),
        "CostApplied": _safe_float(_lookup_value(row, ["CostApplied", "TransactionCost"], np.nan), default=np.nan),
        "RegimeLabel": _lookup_value(row, ["RegimeLabel"], ""),
        "ConfidenceSource": "RawSignalPrediction" if raw_source else "FallbackEvidence",
        "EvidenceMode": "",
        "DataQualityFlag": "",
        "Warnings": "",
    }
    out["EvidenceMode"] = _raw_mode_from_row(out, raw_source=raw_source, fallback_mode=fallback_mode)
    out["Warnings"] = _raw_log_row_warnings(out)
    out["DataQualityFlag"] = _raw_data_quality_flag(out)
    return out


def _raw_signal_outputs_to_log(raw_signal_outputs: Optional[Any]) -> pd.DataFrame:
    if raw_signal_outputs is None:
        return pd.DataFrame(columns=RAW_TRADE_LOG_COLUMNS)
    if isinstance(raw_signal_outputs, pd.DataFrame):
        frames = [raw_signal_outputs]
    elif isinstance(raw_signal_outputs, list):
        frames = [frame for frame in raw_signal_outputs if isinstance(frame, pd.DataFrame) and not frame.empty]
    else:
        frames = []
    rows: List[Dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        if frame.empty or not {"Asset", "Horizon"}.issubset(frame.columns):
            continue
        df = _normalise_horizon_column(frame)
        for row_index, row in df.iterrows():
            rows.append(_raw_row_from_series(row, trade_id=f"RAW8H-{frame_index + 1}-{row_index + 1}", source_phase=_lookup_value(row, ["SourcePhase"], "RawSignalOutput"), raw_source=True))
    return pd.DataFrame(rows, columns=RAW_TRADE_LOG_COLUMNS) if rows else pd.DataFrame(columns=RAW_TRADE_LOG_COLUMNS)


def _ledger_table_to_raw_log(ledger_table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if ledger_table is None or ledger_table.empty or not {"Asset", "Horizon"}.issubset(ledger_table.columns):
        return pd.DataFrame(columns=RAW_TRADE_LOG_COLUMNS)
    df = _normalise_horizon_column(ledger_table)
    rows: List[Dict[str, Any]] = []
    for row_index, row in df.iterrows():
        mode = str(_lookup_value(row, ["EvidenceMode"], ""))
        raw_source = mode == "RawTradeLevel"
        out = _raw_row_from_series(row, trade_id=f"8G-{row_index + 1}", source_phase="8G", raw_source=raw_source, fallback_mode="")
        if mode == "RawTradeLevel":
            out["EvidenceMode"] = "RawTradeLevel"
        elif out["EvidenceMode"] == "InsufficientData":
            if mode in {"WindowAggregate", "WindowAggregateFallback"}:
                out["EvidenceMode"] = "WindowAggregateFallback"
            elif mode in {"PolicyAggregate", "PolicyAggregateFallback", "CalibrationProxy"}:
                out["EvidenceMode"] = "PolicyAggregateFallback"
            elif mode == "InsufficientData":
                out["EvidenceMode"] = "InsufficientData"
        out["Warnings"] = _raw_log_row_warnings(out)
        out["DataQualityFlag"] = _raw_data_quality_flag(out)
        rows.append(out)
    return pd.DataFrame(rows, columns=RAW_TRADE_LOG_COLUMNS) if rows else pd.DataFrame(columns=RAW_TRADE_LOG_COLUMNS)


def _phase8c_to_raw_fallback(full_evidence_table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if full_evidence_table is None or full_evidence_table.empty or not {"Asset", "Horizon"}.issubset(full_evidence_table.columns):
        return pd.DataFrame(columns=RAW_TRADE_LOG_COLUMNS)
    df = _normalise_horizon_column(full_evidence_table)
    rows: List[Dict[str, Any]] = []
    for row_index, row in df.iterrows():
        proxy = pd.Series(
            {
                **row.to_dict(),
                "PolicyName": _lookup_value(row, ["SignalMode", "ThresholdPolicy"], "Window aggregate fallback"),
                "RealizedReturn": _lookup_value(row, ["LockedTestStrategyReturn_%", "AvgLockedStrategyReturn_%"], np.nan),
                "BenchmarkReturn": _lookup_value(row, ["LockedTestBuyHoldReturn_%", "AvgLockedBuyHoldReturn_%"], np.nan),
                "VsBuyHold": _lookup_value(row, ["LockedTestVsBuyHold_%", "AvgLockedVsBuyHold_%", "MedianLockedVsBuyHold_%"], np.nan),
                "MaxDrawdownDuringTrade": _lookup_value(row, ["LockedTestMaxDrawdown_%", "WorstLockedMaxDrawdown_%"], np.nan),
                "SignalTaken": _lookup_value(row, ["ValidConfiguration"], True),
            }
        )
        rows.append(_raw_row_from_series(proxy, trade_id=f"8C-FB-{row_index + 1}", source_phase="8C", raw_source=False, fallback_mode="WindowAggregateFallback"))
    return pd.DataFrame(rows, columns=RAW_TRADE_LOG_COLUMNS) if rows else pd.DataFrame(columns=RAW_TRADE_LOG_COLUMNS)


def _phase8e_to_raw_fallback(policy_table: Optional[pd.DataFrame], candidate_recommendation_table: Optional[pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for table in [policy_table, candidate_recommendation_table]:
        if table is not None and not table.empty and {"Asset", "Horizon"}.issubset(table.columns):
            frames.append(_normalise_horizon_column(table))
    if not frames:
        return pd.DataFrame(columns=RAW_TRADE_LOG_COLUMNS)
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    rows: List[Dict[str, Any]] = []
    for row_index, row in df.iterrows():
        proxy = pd.Series(
            {
                **row.to_dict(),
                "PolicyName": _lookup_value(row, ["PolicyType", "BestPolicyType"], "Policy aggregate fallback"),
                "RealizedReturn": _lookup_value(row, ["StrategyReturn_%"], np.nan),
                "BenchmarkReturn": _lookup_value(row, ["BuyHoldReturn_%"], np.nan),
                "VsBuyHold": _lookup_value(row, ["VsBuyHold_%", "MedianVsBuyHold_%"], np.nan),
                "MaxDrawdownDuringTrade": _lookup_value(row, ["MaxDrawdown_%"], np.nan),
                "SignalTaken": _safe_float(_lookup_value(row, ["TradeCount"], 0.0), default=0.0) > 0,
            }
        )
        rows.append(_raw_row_from_series(proxy, trade_id=f"8E-FB-{row_index + 1}", source_phase="8E", raw_source=False, fallback_mode="PolicyAggregateFallback"))
    return pd.DataFrame(rows, columns=RAW_TRADE_LOG_COLUMNS) if rows else pd.DataFrame(columns=RAW_TRADE_LOG_COLUMNS)


def _grading_to_insufficient_raw_rows(grading_table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if grading_table is None or grading_table.empty or not {"Asset", "Horizon"}.issubset(grading_table.columns):
        return pd.DataFrame(columns=RAW_TRADE_LOG_COLUMNS)
    df = _normalise_horizon_column(grading_table)
    rows: List[Dict[str, Any]] = []
    for row_index, row in df.iterrows():
        proxy = pd.Series({**row.to_dict(), "PolicyName": _lookup_value(row, ["ReliabilityGrade", "MetaDecision"], "Reliability candidate")})
        rows.append(_raw_row_from_series(proxy, trade_id=f"8B-ID-{row_index + 1}", source_phase="8B", raw_source=False, fallback_mode="InsufficientData"))
    return pd.DataFrame(rows, columns=RAW_TRADE_LOG_COLUMNS) if rows else pd.DataFrame(columns=RAW_TRADE_LOG_COLUMNS)


def _raw_candidate_filter(
    raw_log: pd.DataFrame,
    grading_table: Optional[pd.DataFrame],
    candidate_filter: str,
    selected_assets: Optional[Iterable[str]],
    selected_horizons: Optional[Iterable[int]],
) -> pd.DataFrame:
    if raw_log.empty:
        return raw_log
    out = raw_log.copy()
    mode = str(candidate_filter or "all").lower()
    if mode in {"only c/d candidates", "c/d", "c_d"} and grading_table is not None and not grading_table.empty and {"Asset", "Horizon", "ReliabilityGrade"}.issubset(grading_table.columns):
        grades = _normalise_horizon_column(grading_table)
        keep = grades[grades["ReliabilityGrade"].apply(lambda value: _grade_letter(value) in {"C", "D"})][["Asset", "Horizon"]].drop_duplicates()
        out = out.merge(keep.assign(_keep=True), on=["Asset", "Horizon"], how="left")
        out = out[out["_keep"].fillna(False)].drop(columns=["_keep"])
    elif mode in {"specific", "specific asset/horizon"}:
        assets = set(str(asset) for asset in (selected_assets or []))
        horizons = set(int(horizon) for horizon in (selected_horizons or []))
        if assets:
            out = out[out["Asset"].astype(str).isin(assets)]
        if horizons:
            out = out[out["Horizon"].astype(int).isin(horizons)]
    return out.reset_index(drop=True)


def _raw_log_quality_summary(raw_log: pd.DataFrame, grading_table: Optional[pd.DataFrame], configured_assets: Iterable[str], configured_horizons: Iterable[int]) -> pd.DataFrame:
    if raw_log.empty:
        return pd.DataFrame(columns=RAW_LOG_QUALITY_COLUMNS)
    total = int(len(raw_log))
    raw_rows = int(raw_log["EvidenceMode"].eq("RawTradeLevel").sum())
    reconstructed_rows = int(raw_log["EvidenceMode"].eq("ReconstructedTradeLevel").sum())
    aggregate_rows = int(raw_log["EvidenceMode"].isin(["WindowAggregateFallback", "PolicyAggregateFallback", "InsufficientData"]).sum())
    assets = int(raw_log["Asset"].nunique())
    horizons = int(raw_log["Horizon"].nunique())
    candidate_total = 0
    if grading_table is not None and not grading_table.empty and {"Asset", "Horizon"}.issubset(grading_table.columns):
        candidate_total = int(_normalise_horizon_column(grading_table)[["Asset", "Horizon"]].drop_duplicates().shape[0])
    covered = int(raw_log[["Asset", "Horizon"]].drop_duplicates().shape[0])
    candidate_coverage = covered / max(candidate_total or covered, 1) * 100.0
    benchmark_avail = float((raw_log["BenchmarkReturn"].notna() | raw_log["VsBuyHold"].notna()).mean() * 100.0)
    drawdown_avail = float(raw_log["MaxDrawdownDuringTrade"].notna().mean() * 100.0)
    calibration_ready = int((raw_log["ProbabilityUp"].notna() & raw_log["ActualDirection"].notna()).sum())
    readiness = calibration_ready / max(total, 1) * 100.0
    raw_coverage = raw_rows / max(total, 1) * 100.0
    quality = _clip(
        readiness * 0.35
        + raw_coverage * 0.25
        + (reconstructed_rows / max(total, 1) * 100.0) * 0.10
        + benchmark_avail * 0.10
        + drawdown_avail * 0.10
        + candidate_coverage * 0.10
    )
    warnings: List[str] = []
    if raw_rows == 0:
        warnings.append("MissingRawTradeLogs")
    if raw_rows == 0 and reconstructed_rows > 0:
        warnings.append("ReconstructedOnly")
    if raw_rows == 0 and reconstructed_rows == 0 and aggregate_rows > 0:
        warnings.append("AggregateFallbackOnly")
    if raw_log["ProbabilityUp"].isna().mean() >= 0.5:
        warnings.append("MissingProbability")
    if raw_log["SignalDate"].isna().mean() >= 0.5 or (raw_log["EntryDate"].isna() | raw_log["ExitDate"].isna()).mean() >= 0.5:
        warnings.append("MissingTradeDates")
    if raw_log["ActualDirection"].isna().mean() >= 0.5:
        warnings.append("MissingOutcomeDirection")
    if benchmark_avail < 50.0:
        warnings.append("MissingBenchmark")
    if drawdown_avail < 50.0:
        warnings.append("MissingDrawdown")
    if raw_coverage < 25.0:
        warnings.append("LowRawCoverage")
    if readiness < 50.0:
        warnings.append("NotCalibrationReady")
    if assets < len(list(configured_assets)):
        warnings.append("AssetCoverageIncomplete")
    if horizons < len(list(configured_horizons)):
        warnings.append("HorizonCoverageIncomplete")
    if quality < 50.0:
        warnings.append("DataQualityWeak")
    return pd.DataFrame(
        [
            {
                "TotalRows": total,
                "RawTradeLevelRows": raw_rows,
                "ReconstructedTradeLevelRows": reconstructed_rows,
                "AggregateFallbackRows": aggregate_rows,
                "AssetsCovered": assets,
                "HorizonsCovered": horizons,
                "CandidateCoverageRate_%": round(candidate_coverage, 4),
                "MissingProbabilityRate_%": round(float(raw_log["ProbabilityUp"].isna().mean() * 100.0), 4),
                "MissingSignalDateRate_%": round(float(raw_log["SignalDate"].isna().mean() * 100.0), 4),
                "MissingEntryExitDateRate_%": round(float((raw_log["EntryDate"].isna() | raw_log["ExitDate"].isna()).mean() * 100.0), 4),
                "MissingActualDirectionRate_%": round(float(raw_log["ActualDirection"].isna().mean() * 100.0), 4),
                "BenchmarkAvailabilityRate_%": round(benchmark_avail, 4),
                "DrawdownAvailabilityRate_%": round(drawdown_avail, 4),
                "CalibrationReadyRowCount": calibration_ready,
                "CalibrationReadinessScore": round(readiness, 4),
                "RawLogQualityScore": round(quality, 4),
                "MainWarning": _join_warnings(warnings),
                "PromotesCandidates": False,
                "ProductionReadyLabelAllowed": False,
            }
        ],
        columns=RAW_LOG_QUALITY_COLUMNS,
    )


def _raw_group_summaries(raw_log: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if raw_log.empty:
        return {
            "coverage": pd.DataFrame(columns=RAW_COVERAGE_COLUMNS),
            "readiness": pd.DataFrame(columns=RAW_PROBABILITY_READINESS_COLUMNS),
            "distribution": pd.DataFrame(columns=RAW_TRADE_DISTRIBUTION_COLUMNS),
            "benchmark": pd.DataFrame(columns=RAW_BENCHMARK_COMPARISON_COLUMNS),
            "drawdown": pd.DataFrame(columns=RAW_DRAWDOWN_COLUMNS),
            "no_trade": pd.DataFrame(columns=RAW_NO_TRADE_COLUMNS),
        }
    coverage_rows: List[Dict[str, Any]] = []
    readiness_rows: List[Dict[str, Any]] = []
    distribution_rows: List[Dict[str, Any]] = []
    benchmark_rows: List[Dict[str, Any]] = []
    drawdown_rows: List[Dict[str, Any]] = []
    no_trade_rows: List[Dict[str, Any]] = []
    for (asset, horizon), group in raw_log.groupby(["Asset", "Horizon"], dropna=False):
        rows = int(len(group))
        warnings = sorted({warning.strip() for warnings_text in group["Warnings"].astype(str) for warning in warnings_text.split(";") if warning.strip()})
        raw_rows = int(group["EvidenceMode"].eq("RawTradeLevel").sum())
        reconstructed_rows = int(group["EvidenceMode"].eq("ReconstructedTradeLevel").sum())
        fallback_rows = int(group["EvidenceMode"].isin(["WindowAggregateFallback", "PolicyAggregateFallback", "InsufficientData"]).sum())
        signal_taken = int(group["SignalTaken"].fillna(False).astype(bool).sum())
        no_trade = rows - signal_taken
        ready_rows = int((group["ProbabilityUp"].notna() & group["ActualDirection"].notna()).sum())
        coverage_rows.append({
            "Asset": asset,
            "Horizon": int(horizon),
            "Rows": rows,
            "RawTradeLevelRows": raw_rows,
            "ReconstructedTradeLevelRows": reconstructed_rows,
            "AggregateFallbackRows": fallback_rows,
            "SignalTakenRows": signal_taken,
            "NoTradeRows": no_trade,
            "CalibrationReadyRows": ready_rows,
            "RawCoverageRate_%": round(raw_rows / max(rows, 1) * 100.0, 4),
            "Warnings": _join_warnings(warnings),
        })
        readiness_rows.append({
            "Asset": asset,
            "Horizon": int(horizon),
            "Rows": rows,
            "ProbabilityRows": int(group["ProbabilityUp"].notna().sum()),
            "ActualDirectionRows": int(group["ActualDirection"].notna().sum()),
            "RowsWithProbabilityAndOutcome": ready_rows,
            "MissingProbabilityRate_%": round(float(group["ProbabilityUp"].isna().mean() * 100.0), 4),
            "MissingActualDirectionRate_%": round(float(group["ActualDirection"].isna().mean() * 100.0), 4),
            "CalibrationReady": bool(ready_rows >= 10),
            "Warnings": _join_warnings(warnings),
        })
        realized = pd.to_numeric(group["RealizedReturn"], errors="coerce")
        win_loss = group["WinLoss"].astype(str).str.lower()
        wins = int(win_loss.eq("win").sum())
        losses = int(win_loss.eq("loss").sum())
        distribution_rows.append({
            "Asset": asset,
            "Horizon": int(horizon),
            "Rows": rows,
            "SignalTakenRows": signal_taken,
            "Wins": wins,
            "Losses": losses,
            "WinRate_%": round(wins / max(wins + losses, 1) * 100.0, 4),
            "AvgRealizedReturn": round(float(realized.mean()), 4) if realized.notna().any() else np.nan,
            "MedianRealizedReturn": round(float(realized.median()), 4) if realized.notna().any() else np.nan,
            "PositiveReturnRows": int(realized.gt(0).sum()),
            "NegativeReturnRows": int(realized.lt(0).sum()),
            "LosingTradesVisible": bool(realized.lt(0).any() or losses > 0),
        })
        beat = group["BeatBenchmark"].dropna()
        benchmark_rows.append({
            "Asset": asset,
            "Horizon": int(horizon),
            "RowsWithBenchmark": int((group["BenchmarkReturn"].notna() | group["VsBuyHold"].notna()).sum()),
            "BeatBenchmarkRows": int(pd.Series(beat).astype(bool).sum()) if len(beat) else 0,
            "BeatBenchmarkRate_%": round(float(pd.Series(beat).astype(bool).mean() * 100.0), 4) if len(beat) else 0.0,
            "AvgBenchmarkReturn": round(float(pd.to_numeric(group["BenchmarkReturn"], errors="coerce").mean()), 4) if group["BenchmarkReturn"].notna().any() else np.nan,
            "AvgVsBuyHold": round(float(pd.to_numeric(group["VsBuyHold"], errors="coerce").mean()), 4) if group["VsBuyHold"].notna().any() else np.nan,
            "NegativeVsBuyHoldRows": int(pd.to_numeric(group["VsBuyHold"], errors="coerce").lt(0).sum()),
            "BenchmarkAvailabilityRate_%": round(float((group["BenchmarkReturn"].notna() | group["VsBuyHold"].notna()).mean() * 100.0), 4),
        })
        drawdown = pd.to_numeric(group["MaxDrawdownDuringTrade"], errors="coerce")
        drawdown_rows.append({
            "Asset": asset,
            "Horizon": int(horizon),
            "RowsWithDrawdown": int(drawdown.notna().sum()),
            "AvgMaxDrawdownDuringTrade": round(float(drawdown.mean()), 4) if drawdown.notna().any() else np.nan,
            "WorstMaxDrawdownDuringTrade": round(float(drawdown.min()), 4) if drawdown.notna().any() else np.nan,
            "DrawdownAvailabilityRate_%": round(float(drawdown.notna().mean() * 100.0), 4),
            "DrawdownRiskRows": int(drawdown.le(-25.0).sum()),
        })
        no_trade_rows.append({
            "Asset": asset,
            "Horizon": int(horizon),
            "NoTradeRows": no_trade,
            "SkippedSignalRows": int(group["SignalTaken"].fillna(False).eq(False).sum()),
            "MissingProbabilityRows": int(group["ProbabilityUp"].isna().sum()),
            "MissingOutcomeRows": int(group["ActualDirection"].isna().sum()),
            "Warnings": _join_warnings(warnings),
        })
    return {
        "coverage": pd.DataFrame(coverage_rows, columns=RAW_COVERAGE_COLUMNS),
        "readiness": pd.DataFrame(readiness_rows, columns=RAW_PROBABILITY_READINESS_COLUMNS),
        "distribution": pd.DataFrame(distribution_rows, columns=RAW_TRADE_DISTRIBUTION_COLUMNS),
        "benchmark": pd.DataFrame(benchmark_rows, columns=RAW_BENCHMARK_COMPARISON_COLUMNS),
        "drawdown": pd.DataFrame(drawdown_rows, columns=RAW_DRAWDOWN_COLUMNS),
        "no_trade": pd.DataFrame(no_trade_rows, columns=RAW_NO_TRADE_COLUMNS),
    }


def _raw_warning_table(raw_log: pd.DataFrame, quality_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in raw_log.iterrows():
        for warning in [warning.strip() for warning in str(row.get("Warnings", "")).split(";") if warning.strip()]:
            severity = "High" if warning in {"MissingRawTradeLogs", "MissingProbability", "MissingOutcomeDirection", "NotCalibrationReady", "AggregateFallbackOnly"} else "Medium"
            rows.append({"TradeId": row.get("TradeId", ""), "Asset": row.get("Asset", ""), "Horizon": row.get("Horizon", np.nan), "WarningType": warning, "Severity": severity, "Message": f"{warning} in raw trade log exporter row."})
    if not quality_summary.empty:
        for warning in [warning.strip() for warning in str(quality_summary.iloc[0].get("MainWarning", "")).split(";") if warning.strip()]:
            rows.append({"TradeId": "SUMMARY", "Asset": "ALL", "Horizon": np.nan, "WarningType": warning, "Severity": "High", "Message": f"{warning} at raw log summary level."})
    return pd.DataFrame(rows, columns=["TradeId", "Asset", "Horizon", "WarningType", "Severity", "Message"]) if rows else pd.DataFrame(columns=["TradeId", "Asset", "Horizon", "WarningType", "Severity", "Message"])


def _raw_next_actions(coverage: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    readiness_lookup = readiness.set_index(["Asset", "Horizon"]).to_dict("index") if readiness is not None and not readiness.empty else {}
    for _, row in coverage.iterrows():
        key = (row.get("Asset"), row.get("Horizon"))
        ready = readiness_lookup.get(key, {})
        warnings = {warning.strip() for warning in str(row.get("Warnings", "")).split(";") if warning.strip()}
        if not bool(ready.get("CalibrationReady", False)):
            action = "Export true row-level prediction signals with ProbabilityUp, ActualDirection, dates, and realized returns."
            priority = "High"
        elif "AggregateFallbackOnly" in warnings:
            action = "Replace aggregate fallback evidence with reconstructed or raw trade logs."
            priority = "High"
        elif "MissingBenchmark" in warnings:
            action = "Attach benchmark returns to every exported trade row."
            priority = "Medium"
        else:
            action = "Use the exported raw log as input to Phase 8F/8G validation only."
            priority = "Medium"
        rows.append({"Asset": row.get("Asset"), "Horizon": row.get("Horizon"), "NextResearchAction": action, "ActionPriority": priority})
    return pd.DataFrame(rows, columns=["Asset", "Horizon", "NextResearchAction", "ActionPriority"])


def _run_raw_trade_log_exporter_impl(
    *,
    grading_table: Optional[pd.DataFrame] = None,
    full_evidence_table: Optional[pd.DataFrame] = None,
    policy_sensitivity_table: Optional[pd.DataFrame] = None,
    candidate_recommendation_table: Optional[pd.DataFrame] = None,
    ledger_table: Optional[pd.DataFrame] = None,
    raw_signal_outputs: Optional[Any] = None,
    candidate_filter: str = "all",
    selected_assets: Optional[Iterable[str]] = None,
    selected_horizons: Optional[Iterable[int]] = None,
    configured_assets: Optional[Iterable[str]] = None,
    configured_horizons: Optional[Iterable[int]] = None,
) -> RawTradeLogExporterReport:
    settings = {
        "phase": "8H",
        "purpose": "raw_signal_trade_log_exporter_research_infrastructure_only",
        "does_not_promote_candidates": True,
        "production_ready_label_allowed": False,
        "candidate_filter": candidate_filter,
    }
    frames = [
        _raw_signal_outputs_to_log(raw_signal_outputs),
        _ledger_table_to_raw_log(ledger_table),
        _phase8c_to_raw_fallback(full_evidence_table),
        _phase8e_to_raw_fallback(policy_sensitivity_table, candidate_recommendation_table),
        _grading_to_insufficient_raw_rows(grading_table),
    ]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return _empty_raw_trade_log_exporter_report(settings)
    raw_log = pd.concat(frames, ignore_index=True)
    for col in RAW_TRADE_LOG_COLUMNS:
        if col not in raw_log.columns:
            raw_log[col] = np.nan
    raw_log = raw_log[RAW_TRADE_LOG_COLUMNS]
    raw_log["Horizon"] = pd.to_numeric(raw_log["Horizon"], errors="coerce").fillna(0).astype(int)
    raw_log = _raw_candidate_filter(raw_log, grading_table, candidate_filter, selected_assets, selected_horizons)
    if raw_log.empty:
        return _empty_raw_trade_log_exporter_report(settings)
    configured_assets = list(configured_assets or get_asset_names())
    configured_horizons = list(configured_horizons or DEFAULT_HORIZONS)
    quality = _raw_log_quality_summary(raw_log, grading_table, configured_assets, configured_horizons)
    summaries = _raw_group_summaries(raw_log)
    warnings = _raw_warning_table(raw_log, quality)
    next_actions = _raw_next_actions(summaries["coverage"], summaries["readiness"])
    return RawTradeLogExporterReport(
        raw_signal_trade_log_table=raw_log.reset_index(drop=True),
        raw_log_quality_summary=quality[RAW_LOG_QUALITY_COLUMNS],
        asset_horizon_raw_coverage_table=summaries["coverage"],
        probability_outcome_readiness_table=summaries["readiness"],
        trade_outcome_distribution_table=summaries["distribution"],
        benchmark_comparison_table=summaries["benchmark"],
        drawdown_during_trade_table=summaries["drawdown"],
        no_trade_skipped_signal_table=summaries["no_trade"],
        warning_table=warnings,
        next_research_action_table=next_actions,
        settings=settings,
    )


def run_raw_trade_log_exporter(*args: Any, **kwargs: Any) -> RawTradeLogExporterReport:
    """Stable public Phase 8H entrypoint used by app.py and tests."""
    return _run_raw_trade_log_exporter_impl(*args, **kwargs)


TRUE_RAW_TRADE_LOG_COLUMNS = [
    "TradeId",
    "Asset",
    "Horizon",
    "ModelName",
    "PolicyName",
    "SourcePhase",
    "WindowId",
    "WindowMode",
    "ValidationWindow",
    "TestWindow",
    "StepSize",
    "TransactionCost",
    "SignalDate",
    "EntryDate",
    "ExitDate",
    "HoldingPeriod",
    "ProbabilityUp",
    "ProbabilityBin",
    "PredictedDirection",
    "ActualDirection",
    "SignalTaken",
    "EntryPrice",
    "ExitPrice",
    "RealizedReturn",
    "BenchmarkReturn",
    "VsBuyHold",
    "MaxDrawdownDuringTrade",
    "WinLoss",
    "BeatBenchmark",
    "CostApplied",
    "Threshold",
    "Cooldown",
    "RegimeLabel",
    "ConfidenceSource",
    "EvidenceMode",
    "DataQualityFlag",
    "Warnings",
]

TRUE_RAW_QUALITY_COLUMNS = [
    "TotalTrueRawRows",
    "RawTradeLevelRows",
    "ReconstructedTradeLevelRows",
    "AssetsCovered",
    "HorizonsCovered",
    "RowsWithProbabilityUp",
    "RowsWithActualDirection",
    "RowsWithSignalDate",
    "RowsWithEntryExitDates",
    "RowsWithProbabilityAndOutcome",
    "CalibrationReadyRows",
    "MissingProbabilityRate_%",
    "MissingOutcomeRate_%",
    "MissingDateRate_%",
    "NegativeTradeRows",
    "BenchmarkComparisonAvailability_%",
    "DrawdownAvailability_%",
    "CalibrationReadinessScore",
    "RawLogQualityScore",
    "PromotesCandidates",
    "ProductionReadyLabelAllowed",
]

TRUE_MISSING_SOURCE_COLUMNS = [
    "Asset",
    "Horizon",
    "SourceFunction",
    "MissingField",
    "Diagnostic",
    "RequiredNextFix",
]

PHASE8_CLOSURE_COLUMNS = [
    "Phase8RawEvidenceReady",
    "ProbabilityCalibrationCanBeRerun",
    "Reason",
    "RequiredNextFix",
    "TrueRawRows",
    "CalibrationReadyRows",
    "SourceLimitationProven",
    "ProductionReadyLabelAllowed",
]


def _empty_true_raw_trade_log_generation_report(settings: Optional[Dict[str, Any]] = None) -> TrueRawTradeLogGenerationReport:
    return TrueRawTradeLogGenerationReport(
        true_raw_trade_log=pd.DataFrame(columns=TRUE_RAW_TRADE_LOG_COLUMNS),
        raw_log_quality_summary=pd.DataFrame(columns=TRUE_RAW_QUALITY_COLUMNS),
        asset_horizon_raw_coverage=pd.DataFrame(columns=RAW_COVERAGE_COLUMNS),
        probability_outcome_readiness=pd.DataFrame(columns=RAW_PROBABILITY_READINESS_COLUMNS),
        missing_source_diagnostic=pd.DataFrame(columns=TRUE_MISSING_SOURCE_COLUMNS),
        benchmark_comparison=pd.DataFrame(columns=RAW_BENCHMARK_COMPARISON_COLUMNS),
        drawdown_during_trade=pd.DataFrame(columns=RAW_DRAWDOWN_COLUMNS),
        warning_table=pd.DataFrame(columns=["TradeId", "Asset", "Horizon", "WarningType", "Severity", "Message"]),
        next_research_action_table=pd.DataFrame(columns=["Asset", "Horizon", "NextResearchAction", "ActionPriority"]),
        phase8_closure_readiness_table=pd.DataFrame(columns=PHASE8_CLOSURE_COLUMNS),
        aggregate_fallback_diagnostic=pd.DataFrame(columns=TRUE_MISSING_SOURCE_COLUMNS),
        no_trade_skipped_signal_table=pd.DataFrame(columns=RAW_NO_TRADE_COLUMNS),
        settings=settings or {},
    )


def _price_at_or_before(price_series: pd.Series, date_value: Any) -> float:
    if price_series is None or price_series.empty:
        return np.nan
    date = _as_optional_date(date_value)
    if pd.isna(date):
        return np.nan
    prices = pd.to_numeric(price_series.sort_index(), errors="coerce").dropna()
    if prices.empty:
        return np.nan
    if date in prices.index:
        return _safe_float(prices.loc[date], default=np.nan)
    try:
        value = prices.asof(date)
        return _safe_float(value, default=np.nan)
    except Exception:
        return np.nan


def _drawdown_between_dates(price_series: pd.Series, entry_date: Any, exit_date: Any, signal: int = 1) -> float:
    if price_series is None or price_series.empty:
        return np.nan
    entry = _as_optional_date(entry_date)
    exit_ = _as_optional_date(exit_date)
    if pd.isna(entry) or pd.isna(exit_):
        return np.nan
    prices = pd.to_numeric(price_series.sort_index(), errors="coerce").dropna()
    path = prices.loc[(prices.index >= entry) & (prices.index <= exit_)]
    if path.empty:
        return np.nan
    entry_price = _safe_float(path.iloc[0], default=np.nan)
    if not np.isfinite(entry_price) or entry_price == 0:
        return np.nan
    if int(signal) < 0:
        excursion = entry_price / path - 1.0
    else:
        excursion = path / entry_price - 1.0
    return float(excursion.min())


def _true_raw_row_warnings(row: Dict[str, Any]) -> str:
    warnings: List[str] = []
    if row.get("EvidenceMode") == "ReconstructedTradeLevel":
        warnings.append("ReconstructionUsed")
    if pd.isna(row.get("ProbabilityUp")):
        warnings.append("MissingProbability")
    if pd.isna(row.get("SignalDate")) or pd.isna(row.get("EntryDate")) or pd.isna(row.get("ExitDate")):
        warnings.append("MissingTradeDates")
    if pd.isna(row.get("ActualDirection")):
        warnings.append("MissingOutcomeDirection")
    if pd.isna(row.get("EntryPrice")) or pd.isna(row.get("ExitPrice")):
        warnings.append("MissingPricePath")
    if pd.isna(row.get("BenchmarkReturn")) or pd.isna(row.get("VsBuyHold")):
        warnings.append("MissingBenchmark")
    if pd.isna(row.get("MaxDrawdownDuringTrade")):
        warnings.append("MissingDrawdown")
    if "MissingProbability" in warnings or "MissingOutcomeDirection" in warnings:
        warnings.append("NotCalibrationReady")
    return _join_warnings(warnings)


def _true_raw_quality_flag(row: Dict[str, Any]) -> str:
    warnings = {w.strip() for w in str(row.get("Warnings", "")).split(";") if w.strip()}
    if row.get("EvidenceMode") == "RawTradeLevel" and not warnings.intersection({"MissingProbability", "MissingTradeDates", "MissingOutcomeDirection"}):
        return "CompleteRawTrade"
    if row.get("EvidenceMode") == "RawTradeLevel":
        return "IncompleteRawTrade"
    if row.get("EvidenceMode") == "ReconstructedTradeLevel" and not warnings.intersection({"MissingProbability", "MissingTradeDates", "MissingOutcomeDirection", "MissingPricePath"}):
        return "CompleteReconstruction"
    return "LimitedReconstruction"


def _is_true_raw_ready(row: Dict[str, Any]) -> bool:
    return (
        not pd.isna(row.get("ProbabilityUp"))
        and not pd.isna(row.get("ActualDirection"))
        and not pd.isna(row.get("SignalDate"))
        and not pd.isna(row.get("EntryDate"))
        and not pd.isna(row.get("ExitDate"))
        and not pd.isna(row.get("RealizedReturn"))
    )


def _true_raw_from_input_rows(raw_signal_outputs: Optional[Any]) -> pd.DataFrame:
    raw_log = _raw_signal_outputs_to_log(raw_signal_outputs)
    if raw_log.empty:
        return pd.DataFrame(columns=TRUE_RAW_TRADE_LOG_COLUMNS)
    rows: List[Dict[str, Any]] = []
    for _, row in raw_log.iterrows():
        out = {
            "TradeId": row.get("TradeId", ""),
            "Asset": row.get("Asset", ""),
            "Horizon": _safe_int(row.get("Horizon"), default=0),
            "ModelName": row.get("ModelName", ""),
            "PolicyName": row.get("PolicyName", ""),
            "SourcePhase": row.get("SourcePhase", "RawSignalOutput"),
            "WindowId": row.get("WindowId", ""),
            "WindowMode": row.get("WindowMode", ""),
            "ValidationWindow": row.get("ValidationWindow", np.nan),
            "TestWindow": row.get("TestWindow", np.nan),
            "StepSize": row.get("StepSize", np.nan),
            "TransactionCost": row.get("TransactionCost", np.nan),
            "SignalDate": row.get("SignalDate", pd.NaT),
            "EntryDate": row.get("EntryDate", pd.NaT),
            "ExitDate": row.get("ExitDate", pd.NaT),
            "HoldingPeriod": row.get("HoldingPeriod", np.nan),
            "ProbabilityUp": row.get("ProbabilityUp", np.nan),
            "ProbabilityBin": row.get("ProbabilityBin", ""),
            "PredictedDirection": row.get("PredictedDirection", ""),
            "ActualDirection": row.get("ActualDirection", np.nan),
            "SignalTaken": _safe_bool(row.get("SignalTaken"), default=True),
            "EntryPrice": row.get("EntryPrice", np.nan),
            "ExitPrice": row.get("ExitPrice", np.nan),
            "RealizedReturn": row.get("RealizedReturn", np.nan),
            "BenchmarkReturn": row.get("BenchmarkReturn", np.nan),
            "VsBuyHold": row.get("VsBuyHold", np.nan),
            "MaxDrawdownDuringTrade": row.get("MaxDrawdownDuringTrade", np.nan),
            "WinLoss": row.get("WinLoss", _win_loss_label(row.get("ActualDirection"), row.get("RealizedReturn"))),
            "BeatBenchmark": _beat_benchmark_value(row.get("VsBuyHold"), row.get("RealizedReturn"), row.get("BenchmarkReturn")),
            "CostApplied": row.get("CostApplied", np.nan),
            "Threshold": _lookup_value(row, ["Threshold", "LongThreshold"], np.nan),
            "Cooldown": _lookup_value(row, ["Cooldown", "CooldownRows"], np.nan),
            "RegimeLabel": row.get("RegimeLabel", ""),
            "ConfidenceSource": row.get("ConfidenceSource", "RawSignalPrediction"),
            "EvidenceMode": (
                "RawTradeLevel"
                if _is_true_raw_ready(row.to_dict())
                and not pd.isna(row.get("EntryPrice", np.nan))
                and not pd.isna(row.get("ExitPrice", np.nan))
                else "ReconstructedTradeLevel"
            ),
            "DataQualityFlag": "",
            "Warnings": "",
        }
        if out["EvidenceMode"] == "ReconstructedTradeLevel" and not _is_true_raw_ready(out):
            continue
        out["Warnings"] = _true_raw_row_warnings(out)
        out["DataQualityFlag"] = _true_raw_quality_flag(out)
        rows.append(out)
    return pd.DataFrame(rows, columns=TRUE_RAW_TRADE_LOG_COLUMNS) if rows else pd.DataFrame(columns=TRUE_RAW_TRADE_LOG_COLUMNS)


def _raw_signal_missing_source_diagnostics(raw_signal_outputs: Optional[Any]) -> pd.DataFrame:
    if raw_signal_outputs is None:
        return pd.DataFrame(columns=TRUE_MISSING_SOURCE_COLUMNS)
    frames = [raw_signal_outputs] if isinstance(raw_signal_outputs, pd.DataFrame) else [frame for frame in raw_signal_outputs if isinstance(frame, pd.DataFrame)] if isinstance(raw_signal_outputs, list) else []
    rows: List[Dict[str, Any]] = []
    for frame in frames:
        if frame is None or frame.empty or not {"Asset", "Horizon"}.issubset(frame.columns):
            continue
        df = _normalise_horizon_column(frame)
        for _, row in df.iterrows():
            checks = {
                "MissingProbability": _lookup_value(row, ["ProbabilityUp", "Probability", "P_up", "PredictedProbability", "DirectionProbability"], np.nan),
                "MissingOutcomeDirection": _lookup_value(row, ["ActualDirection", "FutureDirection", "Direction", "Win", "WinLoss"], np.nan),
                "MissingTradeDates": _lookup_value(row, ["SignalDate", "Date", "TradeDate"], pd.NaT),
                "MissingRealizedReturn": _lookup_value(row, ["RealizedReturn", "TradeReturn", "Return", "StrategyReturn", "StrategyReturn_%", "StrategyReturnAfterCost"], np.nan),
            }
            for missing_field, value in checks.items():
                is_missing = pd.isna(value)
                if missing_field == "MissingTradeDates":
                    is_missing = pd.isna(_as_optional_date(value))
                if is_missing:
                    rows.append(
                        _missing_source_row(
                            row.get("Asset", ""),
                            row.get("Horizon", 0),
                            "raw_signal_outputs",
                            missing_field,
                            f"Uploaded raw signal row is missing {missing_field}.",
                            "Export row-level ProbabilityUp, dates, actual direction, and realized return from the signal engine.",
                        )
                    )
    return pd.DataFrame(rows, columns=TRUE_MISSING_SOURCE_COLUMNS) if rows else pd.DataFrame(columns=TRUE_MISSING_SOURCE_COLUMNS)


def _missing_source_row(asset: Any, horizon: Any, source_function: str, missing_field: str, diagnostic: str, required_next_fix: str) -> Dict[str, Any]:
    return {
        "Asset": asset,
        "Horizon": _safe_int(horizon, default=0),
        "SourceFunction": source_function,
        "MissingField": missing_field,
        "Diagnostic": diagnostic,
        "RequiredNextFix": required_next_fix,
    }


def _pipeline_trade_log_to_true_raw(
    *,
    asset: str,
    horizon: int,
    signal_output: Any,
    signal_result: Any,
    raw_df: pd.DataFrame,
    transaction_cost: float,
    validation_fraction: float,
    cooldown: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from src.asset_config import get_target_column

    target_col = get_target_column(asset)
    price_series = pd.Series(dtype=float)
    if raw_df is not None and not raw_df.empty and target_col in raw_df.columns:
        price_series = pd.to_numeric(raw_df[target_col], errors="coerce").sort_index()
        price_series.index = pd.to_datetime(price_series.index)

    trade_log = signal_result.signal_frame.copy() if signal_result is not None and getattr(signal_result, "signal_frame", None) is not None else pd.DataFrame()
    missing_rows: List[Dict[str, Any]] = []
    no_trade_rows: List[Dict[str, Any]] = []
    if trade_log.empty:
        missing_rows.append(
            _missing_source_row(
                asset,
                horizon,
                "run_validation_locked_signal_engine",
                "signal_frame",
                "Signal engine returned no locked-test active trades.",
                "Inspect threshold/cooldown policy or export no-trade signal rows from src.signal_engine.",
            )
        )
        metrics = getattr(signal_result, "metrics", {}) if signal_result is not None else {}
        no_trade_rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "NoTradeRows": int(metrics.get("NoTradeCount", 0) or 0),
                "SkippedSignalRows": int(metrics.get("NoTradeCount", 0) or 0),
                "MissingProbabilityRows": 0,
                "MissingOutcomeRows": 0,
                "Warnings": "NoTrueRawRowsGenerated",
            }
        )
        return pd.DataFrame(columns=TRUE_RAW_TRADE_LOG_COLUMNS), pd.DataFrame(missing_rows), pd.DataFrame(no_trade_rows)

    rows: List[Dict[str, Any]] = []
    selected = getattr(signal_result, "selected_threshold", {}) or {}
    metrics = getattr(signal_result, "metrics", {}) or {}
    model_name = getattr(signal_output, "model_name", "")
    no_trade_rows.append(
        {
            "Asset": asset,
            "Horizon": int(horizon),
            "NoTradeRows": int(metrics.get("NoTradeCount", 0) or 0),
            "SkippedSignalRows": int(metrics.get("NoTradeCount", 0) or 0),
            "MissingProbabilityRows": 0,
            "MissingOutcomeRows": 0,
            "Warnings": "" if int(metrics.get("NoTradeCount", 0) or 0) == 0 else "Skipped/no-trade rows captured as counts only.",
        }
    )
    for row_index, trade in trade_log.iterrows():
        entry_date = _as_optional_date(trade.get("EntryDate"))
        exit_date = _as_optional_date(trade.get("ExitDate"))
        signal = int(_safe_float(trade.get("Signal"), default=1.0))
        entry_price = _price_at_or_before(price_series, entry_date)
        exit_price = _price_at_or_before(price_series, exit_date)
        realized_return = _safe_float(trade.get("RealizedReturn"), default=np.nan)
        strategy_return = _safe_float(trade.get("StrategyReturnAfterCost"), default=realized_return)
        if np.isfinite(entry_price) and np.isfinite(exit_price) and entry_price != 0:
            benchmark_return = float(exit_price / entry_price - 1.0)
        elif np.isfinite(realized_return):
            benchmark_return = realized_return
        else:
            benchmark_return = np.nan
        vs_buy_hold = strategy_return - benchmark_return if np.isfinite(strategy_return) and np.isfinite(benchmark_return) else np.nan
        max_dd = _drawdown_between_dates(price_series, entry_date, exit_date, signal=signal)
        probability = _safe_float(trade.get("ProbabilityUp"), default=np.nan)
        out = {
            "TradeId": f"8I-{str(asset).replace(' ', '_')}-{int(horizon)}-{row_index + 1}",
            "Asset": asset,
            "Horizon": int(horizon),
            "ModelName": model_name,
            "PolicyName": "validation_locked_non_overlapping",
            "SourcePhase": "8I",
            "WindowId": "locked_test",
            "WindowMode": "validation_locked_split",
            "ValidationWindow": int(metrics.get("ValidationRows", 0) or 0),
            "TestWindow": int(metrics.get("LockedTestRows", metrics.get("Rows", 0)) or 0),
            "StepSize": np.nan,
            "TransactionCost": float(transaction_cost),
            "SignalDate": entry_date,
            "EntryDate": entry_date,
            "ExitDate": exit_date,
            "HoldingPeriod": _safe_int(trade.get("HoldingDays"), default=int(horizon)),
            "ProbabilityUp": probability,
            "ProbabilityBin": _probability_bin_from_value(probability),
            "PredictedDirection": "Up" if signal == 1 else "Down" if signal == -1 else "NoTrade",
            "ActualDirection": 1 if realized_return > 0 else 0 if np.isfinite(realized_return) else np.nan,
            "SignalTaken": True,
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "RealizedReturn": realized_return,
            "BenchmarkReturn": benchmark_return,
            "VsBuyHold": vs_buy_hold,
            "MaxDrawdownDuringTrade": max_dd,
            "WinLoss": "Win" if realized_return > 0 else "Loss" if np.isfinite(realized_return) else "",
            "BeatBenchmark": bool(vs_buy_hold > 0) if np.isfinite(vs_buy_hold) else np.nan,
            "CostApplied": float(transaction_cost) * 2.0,
            "Threshold": _safe_float(trade.get("LongThreshold"), default=_safe_float(selected.get("SelectedLongThreshold"), default=np.nan)),
            "Cooldown": int(cooldown),
            "RegimeLabel": "",
            "ConfidenceSource": "Phase6DirectForecastProbability",
            "EvidenceMode": "ReconstructedTradeLevel",
            "DataQualityFlag": "",
            "Warnings": "",
        }
        out["Warnings"] = _true_raw_row_warnings(out)
        out["DataQualityFlag"] = _true_raw_quality_flag(out)
        if _is_true_raw_ready(out):
            rows.append(out)
        else:
            for warning in [w.strip() for w in out["Warnings"].split(";") if w.strip()]:
                missing_rows.append(
                    _missing_source_row(
                        asset,
                        horizon,
                        "run_validation_locked_signal_engine",
                        warning,
                        f"Trade row {row_index + 1} could not satisfy true raw evidence requirements.",
                        "Expose ProbabilityUp, dates, realized return, and price path from the signal/backtest pipeline.",
                    )
                )
    return (
        pd.DataFrame(rows, columns=TRUE_RAW_TRADE_LOG_COLUMNS) if rows else pd.DataFrame(columns=TRUE_RAW_TRADE_LOG_COLUMNS),
        pd.DataFrame(missing_rows, columns=TRUE_MISSING_SOURCE_COLUMNS) if missing_rows else pd.DataFrame(columns=TRUE_MISSING_SOURCE_COLUMNS),
        pd.DataFrame(no_trade_rows, columns=RAW_NO_TRADE_COLUMNS),
    )


def _phase8i_candidates(
    *,
    assets: Optional[Iterable[str]],
    horizons: Optional[Iterable[int]],
    grading_table: Optional[pd.DataFrame],
    candidate_filter: str,
) -> pd.DataFrame:
    mode = str(candidate_filter or "all").lower()
    if mode in {"only c/d candidates", "c/d", "c_d"} and grading_table is not None and not grading_table.empty and {"Asset", "Horizon", "ReliabilityGrade"}.issubset(grading_table.columns):
        grades = _normalise_horizon_column(grading_table)
        out = grades[grades["ReliabilityGrade"].apply(lambda value: _grade_letter(value) in {"C", "D"})][["Asset", "Horizon"]].drop_duplicates()
    else:
        asset_list = list(assets or get_asset_names())
        horizon_list = [int(h) for h in (horizons or DEFAULT_HORIZONS)]
        out = pd.DataFrame([{"Asset": asset, "Horizon": horizon} for asset in asset_list for horizon in horizon_list])
    if mode in {"specific", "specific asset/horizon"}:
        asset_set = set(str(asset) for asset in (assets or []))
        horizon_set = set(int(h) for h in (horizons or []))
        if asset_set:
            out = out[out["Asset"].astype(str).isin(asset_set)]
        if horizon_set:
            out = out[out["Horizon"].astype(int).isin(horizon_set)]
    return _normalise_horizon_column(out).drop_duplicates().reset_index(drop=True)


def _aggregate_fallback_diagnostic(
    full_evidence_table: Optional[pd.DataFrame],
    policy_sensitivity_table: Optional[pd.DataFrame],
    candidate_recommendation_table: Optional[pd.DataFrame],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for source_name, table, mode in [
        ("Phase8CFullEvidence", full_evidence_table, "WindowAggregateFallback"),
        ("Phase8EPolicySensitivity", policy_sensitivity_table, "PolicyAggregateFallback"),
        ("Phase8ECandidateRecommendation", candidate_recommendation_table, "PolicyAggregateFallback"),
    ]:
        if table is None or table.empty or not {"Asset", "Horizon"}.issubset(table.columns):
            continue
        for _, row in _normalise_horizon_column(table).iterrows():
            rows.append(
                _missing_source_row(
                    row.get("Asset", ""),
                    row.get("Horizon", 0),
                    source_name,
                    mode,
                    "Aggregate evidence exists but is excluded from true_raw_trade_log by Phase 8I rules.",
                    "Generate raw signal/trade rows from run_direct_forecast_signal_output + run_validation_locked_signal_engine.",
                )
            )
    return pd.DataFrame(rows, columns=TRUE_MISSING_SOURCE_COLUMNS) if rows else pd.DataFrame(columns=TRUE_MISSING_SOURCE_COLUMNS)


def _true_raw_quality_summary(true_raw: pd.DataFrame, configured_assets: Iterable[str], configured_horizons: Iterable[int]) -> pd.DataFrame:
    if true_raw.empty:
        return pd.DataFrame(
            [
                {
                    "TotalTrueRawRows": 0,
                    "RawTradeLevelRows": 0,
                    "ReconstructedTradeLevelRows": 0,
                    "AssetsCovered": 0,
                    "HorizonsCovered": 0,
                    "RowsWithProbabilityUp": 0,
                    "RowsWithActualDirection": 0,
                    "RowsWithSignalDate": 0,
                    "RowsWithEntryExitDates": 0,
                    "RowsWithProbabilityAndOutcome": 0,
                    "CalibrationReadyRows": 0,
                    "MissingProbabilityRate_%": 100.0,
                    "MissingOutcomeRate_%": 100.0,
                    "MissingDateRate_%": 100.0,
                    "NegativeTradeRows": 0,
                    "BenchmarkComparisonAvailability_%": 0.0,
                    "DrawdownAvailability_%": 0.0,
                    "CalibrationReadinessScore": 0.0,
                    "RawLogQualityScore": 0.0,
                    "PromotesCandidates": False,
                    "ProductionReadyLabelAllowed": False,
                }
            ],
            columns=TRUE_RAW_QUALITY_COLUMNS,
        )
    total = int(len(true_raw))
    entry_exit = true_raw["EntryDate"].notna() & true_raw["ExitDate"].notna()
    probability_outcome = true_raw["ProbabilityUp"].notna() & true_raw["ActualDirection"].notna()
    ready = probability_outcome & true_raw["SignalDate"].notna() & entry_exit
    benchmark_available = true_raw["BenchmarkReturn"].notna() & true_raw["VsBuyHold"].notna()
    drawdown_available = true_raw["MaxDrawdownDuringTrade"].notna()
    readiness = float(ready.mean() * 100.0)
    quality = _clip(
        readiness * 0.40
        + float(true_raw["EvidenceMode"].eq("RawTradeLevel").mean() * 100.0) * 0.15
        + float(true_raw["EvidenceMode"].eq("ReconstructedTradeLevel").mean() * 100.0) * 0.10
        + float(benchmark_available.mean() * 100.0) * 0.15
        + float(drawdown_available.mean() * 100.0) * 0.10
        + min(true_raw["Asset"].nunique() / max(len(list(configured_assets)), 1) * 100.0, 100.0) * 0.05
        + min(true_raw["Horizon"].nunique() / max(len(list(configured_horizons)), 1) * 100.0, 100.0) * 0.05
    )
    return pd.DataFrame(
        [
            {
                "TotalTrueRawRows": total,
                "RawTradeLevelRows": int(true_raw["EvidenceMode"].eq("RawTradeLevel").sum()),
                "ReconstructedTradeLevelRows": int(true_raw["EvidenceMode"].eq("ReconstructedTradeLevel").sum()),
                "AssetsCovered": int(true_raw["Asset"].nunique()),
                "HorizonsCovered": int(true_raw["Horizon"].nunique()),
                "RowsWithProbabilityUp": int(true_raw["ProbabilityUp"].notna().sum()),
                "RowsWithActualDirection": int(true_raw["ActualDirection"].notna().sum()),
                "RowsWithSignalDate": int(true_raw["SignalDate"].notna().sum()),
                "RowsWithEntryExitDates": int(entry_exit.sum()),
                "RowsWithProbabilityAndOutcome": int(probability_outcome.sum()),
                "CalibrationReadyRows": int(ready.sum()),
                "MissingProbabilityRate_%": round(float(true_raw["ProbabilityUp"].isna().mean() * 100.0), 4),
                "MissingOutcomeRate_%": round(float(true_raw["ActualDirection"].isna().mean() * 100.0), 4),
                "MissingDateRate_%": round(float((true_raw["SignalDate"].isna() | ~entry_exit).mean() * 100.0), 4),
                "NegativeTradeRows": int(pd.to_numeric(true_raw["RealizedReturn"], errors="coerce").lt(0).sum()),
                "BenchmarkComparisonAvailability_%": round(float(benchmark_available.mean() * 100.0), 4),
                "DrawdownAvailability_%": round(float(drawdown_available.mean() * 100.0), 4),
                "CalibrationReadinessScore": round(readiness, 4),
                "RawLogQualityScore": round(float(quality), 4),
                "PromotesCandidates": False,
                "ProductionReadyLabelAllowed": False,
            }
        ],
        columns=TRUE_RAW_QUALITY_COLUMNS,
    )


def _true_raw_warning_table(true_raw: pd.DataFrame, missing_source: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in true_raw.iterrows():
        for warning in [w.strip() for w in str(row.get("Warnings", "")).split(";") if w.strip()]:
            severity = "High" if warning in {"MissingProbability", "MissingTradeDates", "MissingOutcomeDirection", "NotCalibrationReady", "MissingPricePath"} else "Medium"
            rows.append({"TradeId": row.get("TradeId", ""), "Asset": row.get("Asset", ""), "Horizon": row.get("Horizon", np.nan), "WarningType": warning, "Severity": severity, "Message": f"{warning} in Phase 8I true raw row."})
    if true_raw.empty:
        rows.append({"TradeId": "SUMMARY", "Asset": "ALL", "Horizon": np.nan, "WarningType": "NoTrueRawRowsGenerated", "Severity": "High", "Message": "No RawTradeLevel or ReconstructedTradeLevel rows were generated."})
    if not missing_source.empty:
        for _, row in missing_source.iterrows():
            missing_field = str(row.get("MissingField", ""))
            if missing_field in {"MissingProbability", "MissingOutcomeDirection", "MissingTradeDates", "MissingPricePath", "MissingBenchmark", "MissingDrawdown"}:
                warning_type = missing_field
            else:
                warning_type = "SourceFunctionMissingTradeLog"
            rows.append({"TradeId": "SOURCE", "Asset": row.get("Asset", ""), "Horizon": row.get("Horizon", np.nan), "WarningType": "SourceFunctionMissingTradeLog", "Severity": "High", "Message": str(row.get("Diagnostic", ""))})
            if warning_type != "SourceFunctionMissingTradeLog":
                rows.append({"TradeId": "SOURCE", "Asset": row.get("Asset", ""), "Horizon": row.get("Horizon", np.nan), "WarningType": warning_type, "Severity": "High", "Message": str(row.get("Diagnostic", ""))})
            if warning_type in {"MissingProbability", "MissingOutcomeDirection"}:
                rows.append({"TradeId": "SOURCE", "Asset": row.get("Asset", ""), "Horizon": row.get("Horizon", np.nan), "WarningType": "NotCalibrationReady", "Severity": "High", "Message": "Required probability/outcome fields are missing."})
    if not quality.empty:
        q = quality.iloc[0]
        if _safe_float(q.get("CalibrationReadinessScore"), default=0.0) < 50.0:
            rows.append({"TradeId": "SUMMARY", "Asset": "ALL", "Horizon": np.nan, "WarningType": "NotCalibrationReady", "Severity": "High", "Message": "Calibration-ready rows are insufficient."})
        if _safe_float(q.get("RawLogQualityScore"), default=0.0) < 50.0:
            rows.append({"TradeId": "SUMMARY", "Asset": "ALL", "Horizon": np.nan, "WarningType": "LowRawCoverage", "Severity": "High", "Message": "Raw log quality score is low."})
    return pd.DataFrame(rows, columns=["TradeId", "Asset", "Horizon", "WarningType", "Severity", "Message"]) if rows else pd.DataFrame(columns=["TradeId", "Asset", "Horizon", "WarningType", "Severity", "Message"])


def _phase8_closure_table(true_raw: pd.DataFrame, quality: pd.DataFrame, missing_source: pd.DataFrame) -> pd.DataFrame:
    ready_rows = int(quality.iloc[0]["CalibrationReadyRows"]) if quality is not None and not quality.empty and "CalibrationReadyRows" in quality.columns else 0
    true_rows = int(len(true_raw))
    ready = bool(true_rows > 0 and ready_rows > 0)
    source_limitation = bool(true_rows == 0 and missing_source is not None and not missing_source.empty)
    if ready:
        reason = "True raw/reconstructed trade rows with ProbabilityUp and ActualDirection were generated."
        fix = "Rerun Phase 8F probability calibration and Phase 8G evidence ledger using this true raw log."
    elif source_limitation:
        reason = "No true raw rows were generated; source diagnostics identify missing signal/trade fields."
        fix = "Modify src.signal_engine or direct forecast signal export to return row-level ProbabilityUp, dates, returns, and price path metadata."
    else:
        reason = "No true raw rows were generated and no sufficient source diagnostic was available."
        fix = "Expose row-level trade logs from the signal/backtest pipeline."
    return pd.DataFrame(
        [
            {
                "Phase8RawEvidenceReady": ready,
                "ProbabilityCalibrationCanBeRerun": ready,
                "Reason": reason,
                "RequiredNextFix": fix,
                "TrueRawRows": true_rows,
                "CalibrationReadyRows": ready_rows,
                "SourceLimitationProven": source_limitation,
                "ProductionReadyLabelAllowed": False,
            }
        ],
        columns=PHASE8_CLOSURE_COLUMNS,
    )


def _run_true_raw_trade_log_generation_impl(
    *,
    raw_df: Optional[pd.DataFrame] = None,
    raw_signal_outputs: Optional[Any] = None,
    grading_table: Optional[pd.DataFrame] = None,
    full_evidence_table: Optional[pd.DataFrame] = None,
    policy_sensitivity_table: Optional[pd.DataFrame] = None,
    candidate_recommendation_table: Optional[pd.DataFrame] = None,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    candidate_filter: str = "all",
    model_depth: str = "fast",
    model_name: Optional[str] = None,
    use_phase5_features: bool = True,
    signal_mode: str = "long_only",
    threshold_candidates: Iterable[float] = (0.50, 0.55, 0.60, 0.65, 0.70),
    cooldown: int = 0,
    transaction_cost: float = 0.001,
    validation_fraction: float = 0.5,
    random_state: int = 42,
) -> TrueRawTradeLogGenerationReport:
    settings = {
        "phase": "8I",
        "purpose": "true_raw_trade_log_generation_from_signal_engine_only",
        "does_not_promote_candidates": True,
        "production_ready_label_allowed": False,
        "candidate_filter": candidate_filter,
        "source_functions_inspected": [
            "src.signal_engine.run_validation_locked_signal_engine",
            "src.signal_engine.run_realistic_trade_backtest",
            "src.direct_forecast_models.run_direct_forecast_signal_output",
            "src.backtesting.run_backtest_from_predictions",
        ],
    }
    configured_assets = list(assets or get_asset_names())
    configured_horizons = [int(h) for h in (horizons or DEFAULT_HORIZONS)]
    true_frames: List[pd.DataFrame] = []
    missing_rows: List[Dict[str, Any]] = []
    no_trade_frames: List[pd.DataFrame] = []

    direct_raw = _true_raw_from_input_rows(raw_signal_outputs)
    if not direct_raw.empty:
        true_frames.append(direct_raw)
    raw_input_diagnostics = _raw_signal_missing_source_diagnostics(raw_signal_outputs)
    if not raw_input_diagnostics.empty:
        missing_rows.extend(raw_input_diagnostics.to_dict("records"))

    candidates = _phase8i_candidates(assets=configured_assets, horizons=configured_horizons, grading_table=grading_table, candidate_filter=candidate_filter)
    if raw_df is None or raw_df.empty:
        for _, candidate in candidates.iterrows():
            missing_rows.append(
                _missing_source_row(
                    candidate.get("Asset", ""),
                    candidate.get("Horizon", 0),
                    "run_true_raw_trade_log_generation",
                    "raw_df",
                    "No raw price dataframe was supplied, so the direct forecast/signal pipeline could not be executed.",
                    "Pass raw_df from DataLoader/load_raw_data to run_true_raw_trade_log_generation.",
                )
            )
    else:
        from src.direct_forecast_models import run_direct_forecast_signal_output
        from src.signal_engine import run_validation_locked_signal_engine

        thresholds = [float(t) for t in threshold_candidates]
        if not thresholds:
            thresholds = [0.55]
        short_thresholds = [min(0.45, t - 0.05) for t in thresholds] if str(signal_mode).lower() == "long_only" else [max(0.0, min(0.49, 1.0 - t)) for t in thresholds]
        for _, candidate in candidates.iterrows():
            asset = str(candidate.get("Asset", ""))
            horizon = _safe_int(candidate.get("Horizon"), default=0)
            try:
                signal_output = run_direct_forecast_signal_output(
                    raw_df=raw_df,
                    asset_name=asset,
                    horizon=int(horizon),
                    model_depth=model_depth,
                    use_phase5_features=use_phase5_features,
                    model_name=model_name,
                    random_state=random_state,
                )
                signal_result = run_validation_locked_signal_engine(
                    signal_output=signal_output,
                    mode=signal_mode,
                    transaction_cost=transaction_cost,
                    backtest_style="non_overlapping_realistic",
                    cooldown=int(cooldown),
                    validation_fraction=float(validation_fraction),
                    long_thresholds=thresholds,
                    short_thresholds=short_thresholds,
                )
                raw_rows, missing, no_trade = _pipeline_trade_log_to_true_raw(
                    asset=asset,
                    horizon=int(horizon),
                    signal_output=signal_output,
                    signal_result=signal_result,
                    raw_df=raw_df,
                    transaction_cost=float(transaction_cost),
                    validation_fraction=float(validation_fraction),
                    cooldown=int(cooldown),
                )
                if not raw_rows.empty:
                    true_frames.append(raw_rows)
                if not missing.empty:
                    missing_rows.extend(missing.to_dict("records"))
                if not no_trade.empty:
                    no_trade_frames.append(no_trade)
            except Exception as exc:
                missing_rows.append(
                    _missing_source_row(
                        asset,
                        horizon,
                        "run_direct_forecast_signal_output/run_validation_locked_signal_engine",
                        "pipeline_execution",
                        str(exc),
                        "Inspect direct forecast signal output and signal engine trade log export for this asset/horizon.",
                    )
                )

    true_raw = pd.concat(true_frames, ignore_index=True) if true_frames else pd.DataFrame(columns=TRUE_RAW_TRADE_LOG_COLUMNS)
    for col in TRUE_RAW_TRADE_LOG_COLUMNS:
        if col not in true_raw.columns:
            true_raw[col] = np.nan
    true_raw = true_raw[TRUE_RAW_TRADE_LOG_COLUMNS]
    true_raw = true_raw[true_raw["EvidenceMode"].isin(["RawTradeLevel", "ReconstructedTradeLevel"])].reset_index(drop=True)
    no_trade_table = pd.concat(no_trade_frames, ignore_index=True) if no_trade_frames else pd.DataFrame(columns=RAW_NO_TRADE_COLUMNS)
    missing_source = pd.DataFrame(missing_rows, columns=TRUE_MISSING_SOURCE_COLUMNS) if missing_rows else pd.DataFrame(columns=TRUE_MISSING_SOURCE_COLUMNS)
    aggregate_diag = _aggregate_fallback_diagnostic(full_evidence_table, policy_sensitivity_table, candidate_recommendation_table)
    quality = _true_raw_quality_summary(true_raw, configured_assets, configured_horizons)
    raw_summaries = _raw_group_summaries(true_raw.rename(columns={"MaxDrawdownDuringTrade": "MaxDrawdownDuringTrade"})) if not true_raw.empty else {
        "coverage": pd.DataFrame(columns=RAW_COVERAGE_COLUMNS),
        "readiness": pd.DataFrame(columns=RAW_PROBABILITY_READINESS_COLUMNS),
        "distribution": pd.DataFrame(columns=RAW_TRADE_DISTRIBUTION_COLUMNS),
        "benchmark": pd.DataFrame(columns=RAW_BENCHMARK_COMPARISON_COLUMNS),
        "drawdown": pd.DataFrame(columns=RAW_DRAWDOWN_COLUMNS),
        "no_trade": pd.DataFrame(columns=RAW_NO_TRADE_COLUMNS),
    }
    warning_table = _true_raw_warning_table(true_raw, missing_source, quality)
    next_actions = _raw_next_actions(raw_summaries["coverage"], raw_summaries["readiness"]) if not raw_summaries["coverage"].empty else pd.DataFrame(
        [{"Asset": "ALL", "Horizon": np.nan, "NextResearchAction": "Expose true row-level trade logs from direct forecast and signal engine pipeline.", "ActionPriority": "High"}],
        columns=["Asset", "Horizon", "NextResearchAction", "ActionPriority"],
    )
    closure = _phase8_closure_table(true_raw, quality, missing_source)
    return TrueRawTradeLogGenerationReport(
        true_raw_trade_log=true_raw,
        raw_log_quality_summary=quality[TRUE_RAW_QUALITY_COLUMNS],
        asset_horizon_raw_coverage=raw_summaries["coverage"],
        probability_outcome_readiness=raw_summaries["readiness"],
        missing_source_diagnostic=missing_source,
        benchmark_comparison=raw_summaries["benchmark"],
        drawdown_during_trade=raw_summaries["drawdown"],
        warning_table=warning_table,
        next_research_action_table=next_actions,
        phase8_closure_readiness_table=closure,
        aggregate_fallback_diagnostic=aggregate_diag,
        no_trade_skipped_signal_table=no_trade_table if not no_trade_table.empty else raw_summaries["no_trade"],
        settings=settings,
    )


def run_true_raw_trade_log_generation(*args: Any, **kwargs: Any) -> TrueRawTradeLogGenerationReport:
    """Stable public Phase 8I entrypoint used by app.py and tests."""
    return _run_true_raw_trade_log_generation_impl(*args, **kwargs)
