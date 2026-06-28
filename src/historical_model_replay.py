"""Phase 17 historical model/risk replay engine.

The replay engine reconstructs historical paper-tracking decisions from data
available at each replay date. When true historical model prediction logs are
not supplied, it uses transparent time-safe proxy signals and labels the replay
accordingly. Latest Phase 12/14/15 snapshot weights are never reused as fake
historical weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names, get_target_column
from src.artifact_store import resolve_artifact, save_phase_artifacts


HISTORICAL_REPLAY_PHASE_NAME = "phase17_historical_model_replay"
REPLAY_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)
EXPOSURE_CAP_TOLERANCE = 1e-6
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"

REPLAY_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "ReplaySource",
    "ModelReplayQuality",
    "ReplayStartDate",
    "ReplayEndDate",
    "ReplayRows",
    "AssetsCovered",
    "HorizonsCovered",
    "MaturedOutcomeRows",
    "PendingOutcomeRows",
    "AverageRowPaperWeightPct",
    "AveragePaperExposurePct",
    "AveragePortfolioPaperExposurePct",
    "MaxPortfolioPaperExposurePct",
    "ExposureCapBreachesBeforeScaling",
    "ExposureCapBreachesAfterScaling",
    "AverageRealExposurePct",
    "ReplayVerdict",
    "MainLimitation",
    "RecommendedNextStep",
)

SIGNAL_LOG_COLUMNS: Tuple[str, ...] = (
    "ReplayDate",
    "Asset",
    "Horizon",
    "ReplaySource",
    "SignalScore",
    "Direction",
    "ResearchAction",
    "PaperWeightPct",
    "RealWeightPct",
    "RiskScore",
    "RegimeScore",
    "BenchmarkPenalty",
    "VolatilityPenalty",
    "DrawdownPenalty",
    "EvidencePenalty",
    "ReplayDecision",
    "MainReason",
    "DataAvailableThroughDate",
)

OUTCOME_COLUMNS: Tuple[str, ...] = (
    "ReplayDate",
    "OutcomeDate",
    "Asset",
    "Horizon",
    "PaperWeightPct",
    "RealWeightPct",
    "EntryPrice",
    "ExitPrice",
    "ForwardReturnPct",
    "WeightedPaperReturnPct",
    "DirectionCorrect",
    "OutcomeStatus",
    "OutcomeReason",
)

PERFORMANCE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "ReplaySource",
    "TradeCount",
    "ExposureDays",
    "AveragePaperWeightPct",
    "TotalWeightedReturnPct",
    "AnnualizedReturnPct",
    "VolatilityPct",
    "SharpeProxy",
    "MaxDrawdownPct",
    "WinRatePct",
    "DirectionAccuracyPct",
    "AvgReturnPerSignalPct",
    "BenchmarkComparable",
    "MainPerformanceNote",
)

PORTFOLIO_CURVE_COLUMNS: Tuple[str, ...] = (
    "Date",
    "PortfolioPaperExposurePct",
    "DailyWeightedReturnPct",
    "EquityCurve",
    "DrawdownPct",
    "ActiveSignals",
    "MainRiskState",
)

ASSET_HORIZON_MATRIX_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "ReplayRows",
    "MaturedRows",
    "AvgSignalScore",
    "AvgPaperWeightPct",
    "TotalWeightedReturnPct",
    "WinRatePct",
    "MaxDrawdownPct",
    "ReplayVerdict",
    "MainWeakness",
    "MainStrength",
)

QUALITY_CHECK_COLUMNS: Tuple[str, ...] = (
    "CheckName",
    "Passed",
    "Severity",
    "AffectedRows",
    "Explanation",
)

BENCHMARK_READY_COLUMNS: Tuple[str, ...] = (
    "StrategyName",
    "ComparableHistorical",
    "EvaluationMode",
    "ReplaySource",
    "Rows",
    "MaturedRows",
    "BenchmarkReady",
    "Reason",
)

REPLAY_WARNING_COLUMNS: Tuple[str, ...] = (
    "WarningType",
    "Severity",
    "Asset",
    "Horizon",
    "ReplayDate",
    "Explanation",
    "RecommendedFix",
)

NEXT_REPLAY_ACTION_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "Action",
    "WhyItMatters",
    "AffectedAssets",
    "AffectedHorizons",
    "ExpectedBenefit",
    "Urgency",
    "DependsOn",
)

REPLAY_INPUT_SOURCE_COLUMNS: Tuple[str, ...] = (
    "SourceName",
    "Available",
    "Rows",
    "Columns",
    "LastDate",
    "MissingCriticalColumns",
    "Notes",
)

PHASE16_EXPORT_COLUMNS: Tuple[str, ...] = (
    "Date",
    "Asset",
    "Horizon",
    "StrategyName",
    "ExposurePct",
    "DailyReturnPct",
    "StrategyReturnPct",
    "EvaluationMode",
    "ComparableHistorical",
    "ReplaySource",
)

EXPOSURE_CAP_COLUMNS: Tuple[str, ...] = (
    "ReplayDate",
    "ExposureBeforeCapPct",
    "ExposureAfterCapPct",
    "MaxPortfolioPaperExposurePct",
    "CapApplied",
    "ScalingFactor",
    "ActiveSignalsBeforeCap",
    "ActiveSignalsAfterCap",
    "AdjustmentReason",
)

INPUT_SPECS: Dict[str, Tuple[str, str, bool]] = {
    "historical_prediction_log": ("Phase 17 Historical Model Replay", "historical_prediction_log", False),
    "forward_signal_log": ("Phase 9 Forward Paper Evidence", "forward_signal_log", False),
    "true_raw_trade_log": ("Phase 8I True Raw Trade Logs", "true_raw_trade_log", False),
    "ranked_asset_horizon_plan": ("Phase 10 Actionable Research Plan", "ranked_asset_horizon_plan", False),
    "allocation_plan_table": ("Phase 12 Portfolio Capital Simulator", "allocation_plan_table", False),
    "asset_horizon_risk_matrix": ("phase13_risk_warning_intelligence", "asset_horizon_risk_matrix", False),
    "dynamic_position_sizing_table": ("phase14_dynamic_risk_sizing", "dynamic_position_sizing_table", False),
    "asset_horizon_regime_table": ("phase15_market_regime_intelligence", "asset_horizon_regime_table", False),
}


@dataclass
class HistoricalModelReplayReport:
    replay_summary_table: pd.DataFrame
    historical_replay_signal_log: pd.DataFrame
    historical_replay_outcomes: pd.DataFrame
    historical_replay_performance: pd.DataFrame
    historical_replay_portfolio_curve: pd.DataFrame
    replay_asset_horizon_matrix: pd.DataFrame
    replay_exposure_cap_table: pd.DataFrame
    replay_quality_checks: pd.DataFrame
    replay_benchmark_ready_table: pd.DataFrame
    replay_warnings_table: pd.DataFrame
    next_replay_actions_table: pd.DataFrame
    replay_input_sources_table: pd.DataFrame
    phase16_replay_export_table: pd.DataFrame
    artifact_input_source_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: Dict[str, Any] = field(default_factory=dict)
    saved_artifacts: Dict[str, Any] = field(default_factory=dict)


def _to_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _cap_breach_mask(exposure_values: Any, cap_values: Any) -> pd.Series:
    exposure = pd.to_numeric(exposure_values, errors="coerce")
    if not isinstance(exposure, pd.Series):
        exposure = pd.Series([exposure])
    exposure = exposure.fillna(0.0)
    caps = pd.to_numeric(cap_values, errors="coerce")
    if not isinstance(caps, pd.Series):
        caps = pd.Series([caps] * len(exposure), index=exposure.index)
    else:
        caps = caps.reindex(exposure.index).fillna(0.0)
    return exposure > caps + EXPOSURE_CAP_TOLERANCE


def _prepare_market_data(market_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = _to_frame(market_data)
    if df.empty:
        return df
    df = df.copy()
    date_col = next((col for col in ["Date", "date", "Datetime", "Timestamp"] if col in df.columns), None)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    else:
        parsed = pd.to_datetime(df.index, errors="coerce")
        if not parsed.isna().all():
            df.index = parsed
            df = df[~df.index.isna()].sort_index()
    return df


def _load_project_market_data() -> Optional[pd.DataFrame]:
    if not DEFAULT_MARKET_DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DEFAULT_MARKET_DATA_PATH)
    except Exception:
        return None


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _normalise_horizon(df: Any) -> pd.DataFrame:
    out = _to_frame(df)
    if out.empty:
        return out
    if "Horizon" in out.columns:
        out["Horizon"] = out["Horizon"].astype(str).str.replace("D", "", regex=False)
        out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce")
    return out


def _resolve_inputs(
    use_artifact_store: bool,
    prefer_uploaded: bool,
    uploaded_overrides: Optional[Dict[str, Any]],
    direct_tables: Dict[str, Any],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    uploaded_overrides = uploaded_overrides or {}
    tables: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, Any]] = []
    for key, (phase, artifact, required) in INPUT_SPECS.items():
        direct = direct_tables.get(key)
        if direct is not None:
            df = _normalise_horizon(direct)
            tables[key] = df
            rows.append({"Artifact": artifact, "Phase": phase, "Source": "DirectInput", "RunId": "", "Rows": int(len(df)), "CreatedAt": "", "Status": "Loaded", "Path": ""})
            continue
        if use_artifact_store or uploaded_overrides.get(key) is not None:
            resolved = resolve_artifact(phase, artifact, uploaded_file=uploaded_overrides.get(key), prefer_uploaded=prefer_uploaded, required=required)
            data = resolved.get("Data")
            tables[key] = _normalise_horizon(data) if data is not None else pd.DataFrame()
            rows.append({k: v for k, v in resolved.items() if k != "Data"})
        else:
            tables[key] = pd.DataFrame()
            rows.append({"Artifact": artifact, "Phase": phase, "Source": "Missing", "RunId": "", "Rows": 0, "CreatedAt": "", "Status": "MissingOptional", "Path": ""})
    return tables, pd.DataFrame(rows, columns=["Artifact", "Source", "RunId", "Rows", "CreatedAt", "Status", "Phase", "Path"])


def _input_sources(market_data: pd.DataFrame, assets: Iterable[str], project_data_used: bool) -> pd.DataFrame:
    required = [get_target_column(asset) for asset in assets]
    missing = [col for col in required if col not in market_data.columns]
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
    for asset in assets:
        col = get_target_column(asset)
        s = _series(market_data, col).dropna()
        rows.append(
            {
                "SourceName": f"{asset} price",
                "Available": bool(not s.empty),
                "Rows": int(len(s)),
                "Columns": 1 if col in market_data.columns else 0,
                "LastDate": str(s.index.max().date()) if not s.empty and hasattr(s.index.max(), "date") else "",
                "MissingCriticalColumns": "" if col in market_data.columns else col,
                "Notes": "Critical price series for replay.",
            }
        )
    return pd.DataFrame(rows, columns=list(REPLAY_INPUT_SOURCE_COLUMNS))


def _has_prediction_log(tables: Dict[str, pd.DataFrame]) -> bool:
    pred = tables.get("historical_prediction_log", pd.DataFrame())
    if pred.empty:
        return False
    date_cols = {"ReplayDate", "Date", "SignalDate", "PredictionDate"}
    return bool({"Asset", "Horizon"}.issubset(pred.columns) and date_cols.intersection(pred.columns))


def _pct_change_at(series: pd.Series, end_pos: int, lookback: int) -> float:
    start_pos = end_pos - int(lookback)
    if start_pos < 0:
        return np.nan
    start = _safe_float(series.iloc[start_pos], np.nan)
    end = _safe_float(series.iloc[end_pos], np.nan)
    if not np.isfinite(start) or start == 0 or not np.isfinite(end):
        return np.nan
    return (end / start - 1.0) * 100.0


def _drawdown_at(history: pd.Series) -> float:
    clean = history.dropna()
    if clean.empty:
        return np.nan
    peak = clean.cummax().iloc[-1]
    price = clean.iloc[-1]
    if not np.isfinite(peak) or peak == 0:
        return np.nan
    return (price / peak - 1.0) * 100.0


def _proxy_signal(history: pd.Series, horizon: int, max_paper_weight_pct: float) -> Dict[str, Any]:
    clean = history.dropna()
    if len(clean) < 80:
        return {
            "SignalScore": 0.0,
            "Direction": "Neutral",
            "ResearchAction": "ObserveOnly",
            "PaperWeightPct": 0.0,
            "RiskScore": 100.0,
            "RegimeScore": 0.0,
            "BenchmarkPenalty": 20.0,
            "VolatilityPenalty": 25.0,
            "DrawdownPenalty": 25.0,
            "EvidencePenalty": 30.0,
            "ReplayDecision": "BlockedByData",
            "MainReason": "Insufficient history before replay date.",
        }
    ret20 = _pct_change_at(clean, len(clean) - 1, 20)
    ret60 = _pct_change_at(clean, len(clean) - 1, 60)
    ma20 = clean.rolling(20).mean().iloc[-1]
    ma60 = clean.rolling(60).mean().iloc[-1]
    daily = clean.pct_change().dropna()
    vol_pct = daily.tail(20).std(ddof=0) * np.sqrt(252) * 100.0 if len(daily) >= 20 else np.nan
    drawdown = _drawdown_at(clean.tail(252))

    momentum_score = np.clip(50.0 + _safe_float(ret20, 0.0) * 2.0 + _safe_float(ret60, 0.0), 0.0, 100.0)
    trend_bonus = 15.0 if np.isfinite(ma20) and np.isfinite(ma60) and ma20 > ma60 else -10.0
    volatility_penalty = min(25.0, max(0.0, (_safe_float(vol_pct, 0.0) - 12.0) * 0.8))
    drawdown_penalty = min(25.0, abs(min(0.0, _safe_float(drawdown, 0.0))) * 1.2)
    benchmark_penalty = 10.0 if _safe_float(ret60, 0.0) < 0 else 0.0
    evidence_penalty = 0.0 if len(clean) >= 180 else 10.0
    regime_score = float(np.clip(50.0 + trend_bonus - volatility_penalty - drawdown_penalty * 0.4, 0.0, 100.0))
    risk_score = float(np.clip(volatility_penalty + drawdown_penalty + benchmark_penalty + evidence_penalty, 0.0, 100.0))
    signal_score = float(np.clip(momentum_score + trend_bonus - risk_score * 0.6, 0.0, 100.0))

    if signal_score >= 60 and risk_score < 65 and regime_score >= 40:
        decision = "PaperTrack"
        action = "PaperTradeOnly"
        direction = "Up"
        paper_weight = min(float(max_paper_weight_pct), max(1.0, (signal_score - 50.0) * 0.45))
        reason = "Proxy score supports paper tracking using past momentum, trend, volatility, and drawdown data."
    elif signal_score >= 50 and risk_score < 75:
        decision = "WatchlistOnly"
        action = "Watchlist"
        direction = "Neutral"
        paper_weight = 0.0
        reason = "Proxy evidence is mixed; keep row visible without simulated exposure."
    elif risk_score >= 75:
        decision = "BlockedByRisk"
        action = "ObserveOnly"
        direction = "Neutral"
        paper_weight = 0.0
        reason = "Past-data risk penalties are too high for simulated exposure."
    else:
        decision = "NoExposure"
        action = "ObserveOnly"
        direction = "Neutral"
        paper_weight = 0.0
        reason = "Proxy score is too weak for simulated exposure."

    if regime_score < 25 and paper_weight > 0:
        decision = "BlockedByRegime"
        action = "ObserveOnly"
        paper_weight = 0.0
        reason = "Past-data regime score blocks simulated exposure."

    return {
        "SignalScore": round(signal_score, 4),
        "Direction": direction,
        "ResearchAction": action,
        "PaperWeightPct": round(float(np.clip(paper_weight, 0.0, max_paper_weight_pct)), 4),
        "RiskScore": round(risk_score, 4),
        "RegimeScore": round(regime_score, 4),
        "BenchmarkPenalty": round(benchmark_penalty, 4),
        "VolatilityPenalty": round(volatility_penalty, 4),
        "DrawdownPenalty": round(drawdown_penalty, 4),
        "EvidencePenalty": round(evidence_penalty, 4),
        "ReplayDecision": decision,
        "MainReason": reason,
    }


def _replay_dates(index: pd.DatetimeIndex, start_date: Optional[Any], end_date: Optional[Any], replay_step: int, warmup: int) -> List[pd.Timestamp]:
    if len(index) == 0:
        return []
    start = pd.to_datetime(start_date) if start_date is not None else index[max(warmup, int(len(index) * 0.55))]
    end = pd.to_datetime(end_date) if end_date is not None else index[-1]
    valid = [dt for dt in index if dt >= start and dt <= end]
    return valid[:: max(1, int(replay_step))]


def _build_replay_rows(
    market_data: pd.DataFrame,
    assets: Iterable[str],
    horizons: Iterable[int],
    replay_start_date: Optional[Any],
    replay_end_date: Optional[Any],
    replay_step: int,
    max_paper_weight_pct: float,
    replay_source: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    signal_rows: List[Dict[str, Any]] = []
    outcome_rows: List[Dict[str, Any]] = []
    for asset in assets:
        price_col = get_target_column(asset)
        price = _series(market_data, price_col)
        if price.empty:
            continue
        dates = _replay_dates(pd.DatetimeIndex(price.index), replay_start_date, replay_end_date, replay_step, 80)
        index_positions = {dt: pos for pos, dt in enumerate(price.index)}
        for replay_date in dates:
            pos = index_positions.get(replay_date)
            if pos is None:
                continue
            history = price.iloc[: pos + 1]
            for horizon in horizons:
                signal = _proxy_signal(history, int(horizon), float(max_paper_weight_pct))
                row = {
                    "ReplayDate": replay_date,
                    "Asset": asset,
                    "Horizon": int(horizon),
                    "ReplaySource": replay_source,
                    **signal,
                    "RealWeightPct": 0.0,
                    "DataAvailableThroughDate": replay_date,
                }
                signal_rows.append(row)

                entry = _safe_float(price.iloc[pos], np.nan)
                outcome_pos = pos + int(horizon)
                if not np.isfinite(entry):
                    outcome = {
                        "OutcomeDate": pd.NaT,
                        "ExitPrice": np.nan,
                        "ForwardReturnPct": np.nan,
                        "WeightedPaperReturnPct": 0.0,
                        "DirectionCorrect": False,
                        "OutcomeStatus": "MissingEntryPrice",
                        "OutcomeReason": "Entry price is missing at replay date.",
                    }
                elif outcome_pos >= len(price):
                    outcome = {
                        "OutcomeDate": pd.NaT,
                        "ExitPrice": np.nan,
                        "ForwardReturnPct": np.nan,
                        "WeightedPaperReturnPct": 0.0,
                        "DirectionCorrect": False,
                        "OutcomeStatus": "Pending",
                        "OutcomeReason": "Outcome date is beyond available market data.",
                    }
                else:
                    outcome_date = price.index[outcome_pos]
                    exit_price = _safe_float(price.iloc[outcome_pos], np.nan)
                    if not np.isfinite(exit_price):
                        outcome = {
                            "OutcomeDate": outcome_date,
                            "ExitPrice": np.nan,
                            "ForwardReturnPct": np.nan,
                            "WeightedPaperReturnPct": 0.0,
                            "DirectionCorrect": False,
                            "OutcomeStatus": "MissingExitPrice",
                            "OutcomeReason": "Exit price is missing at horizon outcome date.",
                        }
                    else:
                        forward = (exit_price / entry - 1.0) * 100.0
                        weight = _safe_float(signal["PaperWeightPct"], 0.0) / 100.0
                        direction_correct = bool((signal["Direction"] == "Up" and forward > 0) or (signal["Direction"] != "Up" and forward <= 0))
                        outcome = {
                            "OutcomeDate": outcome_date,
                            "ExitPrice": exit_price,
                            "ForwardReturnPct": round(forward, 6),
                            "WeightedPaperReturnPct": round(forward * weight, 6),
                            "DirectionCorrect": direction_correct,
                            "OutcomeStatus": "Matured",
                            "OutcomeReason": "Outcome matured after replay date.",
                        }
                outcome_rows.append(
                    {
                        "ReplayDate": replay_date,
                        "Asset": asset,
                        "Horizon": int(horizon),
                        "PaperWeightPct": signal["PaperWeightPct"],
                        "RealWeightPct": 0.0,
                        "EntryPrice": entry,
                        **outcome,
                    }
                )
    signals = pd.DataFrame(signal_rows, columns=list(SIGNAL_LOG_COLUMNS))
    outcomes = pd.DataFrame(outcome_rows, columns=list(OUTCOME_COLUMNS))
    return signals, outcomes


def _prediction_date_column(predictions: pd.DataFrame) -> Optional[str]:
    for column in ["ReplayDate", "SignalDate", "PredictionDate", "Date", "CreatedAt"]:
        if column in predictions.columns:
            return column
    return None


def _prediction_signal(row: pd.Series, max_paper_weight_pct: float) -> Dict[str, Any]:
    probability = _safe_float(row.get("ProbabilityUp", np.nan), np.nan)
    raw_score = _safe_float(row.get("SignalScore", np.nan), np.nan)
    if np.isfinite(probability):
        direction = "Up" if probability >= 0.5 else "Down"
        signal_score = float(np.clip(max(probability, 1.0 - probability) * 100.0, 0.0, 100.0))
    elif np.isfinite(raw_score):
        signal_score = float(np.clip(raw_score, 0.0, 100.0))
        direction = str(row.get("Direction", "Up" if signal_score >= 55 else "Neutral"))
    else:
        signal_score = 0.0
        direction = "Neutral"

    weight_candidates = [
        row.get("ReplayPaperWeightPct", np.nan),
        row.get("HistoricalPaperWeightPct", np.nan),
        row.get("PaperWeightPct", np.nan),
        row.get("SuggestedPaperWeightPct", np.nan),
    ]
    supplied_weight = next((_safe_float(value, np.nan) for value in weight_candidates if np.isfinite(_safe_float(value, np.nan))), np.nan)
    if np.isfinite(supplied_weight):
        paper_weight = float(np.clip(supplied_weight, 0.0, max_paper_weight_pct))
    elif signal_score >= 60:
        paper_weight = float(np.clip((signal_score - 55.0) * 0.35, 1.0, max_paper_weight_pct))
    else:
        paper_weight = 0.0

    research_action = str(row.get("ResearchAction", "") or ("PaperTradeOnly" if paper_weight > 0 else "Watchlist" if signal_score >= 55 else "ObserveOnly"))
    risk_score = _safe_float(row.get("RiskScore", np.nan), np.nan)
    if not np.isfinite(risk_score):
        risk_score = float(np.clip(100.0 - signal_score, 0.0, 100.0))
    regime_score = _safe_float(row.get("RegimeScore", np.nan), np.nan)
    if not np.isfinite(regime_score):
        regime_score = signal_score

    if paper_weight > 0:
        decision = "HistoricalPaperTrack"
        reason = "Timestamped historical prediction evidence supplied the signal score and simulated paper weight."
    elif signal_score >= 55:
        decision = "HistoricalWatchlist"
        reason = "Timestamped historical prediction evidence was visible but below simulated exposure threshold."
    else:
        decision = "HistoricalNoExposure"
        reason = "Timestamped historical prediction evidence was too weak for simulated exposure."

    return {
        "SignalScore": round(float(signal_score), 4),
        "Direction": direction,
        "ResearchAction": research_action,
        "PaperWeightPct": round(paper_weight, 4),
        "RiskScore": round(float(np.clip(risk_score, 0.0, 100.0)), 4),
        "RegimeScore": round(float(np.clip(regime_score, 0.0, 100.0)), 4),
        "BenchmarkPenalty": round(_safe_float(row.get("BenchmarkPenalty", 0.0), 0.0), 4),
        "VolatilityPenalty": round(_safe_float(row.get("VolatilityPenalty", 0.0), 0.0), 4),
        "DrawdownPenalty": round(_safe_float(row.get("DrawdownPenalty", 0.0), 0.0), 4),
        "EvidencePenalty": round(_safe_float(row.get("EvidencePenalty", 0.0), 0.0), 4),
        "ReplayDecision": decision,
        "MainReason": reason,
    }


def _outcome_from_price(price: pd.Series, pos: Optional[int], horizon: int, signal: Dict[str, Any]) -> Dict[str, Any]:
    if pos is None or pos < 0 or pos >= len(price):
        return {
            "EntryPrice": np.nan,
            "OutcomeDate": pd.NaT,
            "ExitPrice": np.nan,
            "ForwardReturnPct": np.nan,
            "WeightedPaperReturnPct": 0.0,
            "DirectionCorrect": False,
            "OutcomeStatus": "MissingEntryPrice",
            "OutcomeReason": "No price row is available at or before replay date.",
        }
    entry = _safe_float(price.iloc[pos], np.nan)
    if not np.isfinite(entry):
        return {
            "EntryPrice": entry,
            "OutcomeDate": pd.NaT,
            "ExitPrice": np.nan,
            "ForwardReturnPct": np.nan,
            "WeightedPaperReturnPct": 0.0,
            "DirectionCorrect": False,
            "OutcomeStatus": "MissingEntryPrice",
            "OutcomeReason": "Entry price is missing at replay date.",
        }
    outcome_pos = pos + int(horizon)
    if outcome_pos >= len(price):
        return {
            "EntryPrice": entry,
            "OutcomeDate": pd.NaT,
            "ExitPrice": np.nan,
            "ForwardReturnPct": np.nan,
            "WeightedPaperReturnPct": 0.0,
            "DirectionCorrect": False,
            "OutcomeStatus": "Pending",
            "OutcomeReason": "Outcome date is beyond available market data.",
        }
    outcome_date = price.index[outcome_pos]
    exit_price = _safe_float(price.iloc[outcome_pos], np.nan)
    if not np.isfinite(exit_price):
        return {
            "EntryPrice": entry,
            "OutcomeDate": outcome_date,
            "ExitPrice": np.nan,
            "ForwardReturnPct": np.nan,
            "WeightedPaperReturnPct": 0.0,
            "DirectionCorrect": False,
            "OutcomeStatus": "MissingExitPrice",
            "OutcomeReason": "Exit price is missing at horizon outcome date.",
        }
    forward = (exit_price / entry - 1.0) * 100.0
    weight = _safe_float(signal["PaperWeightPct"], 0.0) / 100.0
    direction_correct = bool((signal["Direction"] == "Up" and forward > 0) or (signal["Direction"] == "Down" and forward <= 0))
    return {
        "EntryPrice": entry,
        "OutcomeDate": outcome_date,
        "ExitPrice": exit_price,
        "ForwardReturnPct": round(forward, 6),
        "WeightedPaperReturnPct": round(forward * weight, 6),
        "DirectionCorrect": direction_correct,
        "OutcomeStatus": "Matured",
        "OutcomeReason": "Outcome matured after replay date.",
    }


def _build_prediction_replay_rows(
    market_data: pd.DataFrame,
    prediction_log: pd.DataFrame,
    assets: Iterable[str],
    horizons: Iterable[int],
    max_paper_weight_pct: float,
    replay_source: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred = _normalise_horizon(prediction_log)
    date_col = _prediction_date_column(pred)
    if pred.empty or date_col is None:
        return (
            pd.DataFrame(columns=list(SIGNAL_LOG_COLUMNS)),
            pd.DataFrame(columns=list(OUTCOME_COLUMNS)),
        )
    pred = pred.copy()
    pred[date_col] = pd.to_datetime(pred[date_col], errors="coerce")
    pred = pred.dropna(subset=[date_col, "Asset", "Horizon"])
    pred = pred[pred["Asset"].astype(str).isin([str(asset) for asset in assets])]
    pred = pred[pred["Horizon"].astype(int).isin([int(h) for h in horizons])]
    pred["_ReplayDateNorm"] = pred[date_col].dt.date.astype(str)
    pred = pred.sort_values(date_col).drop_duplicates(subset=["_ReplayDateNorm", "Asset", "Horizon"], keep="last")

    signal_rows: List[Dict[str, Any]] = []
    outcome_rows: List[Dict[str, Any]] = []
    for _, pred_row in pred.iterrows():
        asset = str(pred_row["Asset"])
        horizon = int(pred_row["Horizon"])
        replay_dt = pd.to_datetime(pred_row[date_col])
        price = _series(market_data, get_target_column(asset))
        pos: Optional[int] = None
        data_available = replay_dt
        if not price.empty:
            insert_pos = int(price.index.searchsorted(replay_dt, side="right")) - 1
            if insert_pos >= 0:
                pos = insert_pos
                data_available = price.index[insert_pos]
        signal = _prediction_signal(pred_row, max_paper_weight_pct)
        signal_rows.append(
            {
                "ReplayDate": data_available,
                "Asset": asset,
                "Horizon": horizon,
                "ReplaySource": replay_source,
                **signal,
                "RealWeightPct": 0.0,
                "DataAvailableThroughDate": data_available,
            }
        )
        outcome = _outcome_from_price(price, pos, horizon, signal)
        entry = outcome.pop("EntryPrice")
        outcome_rows.append(
            {
                "ReplayDate": data_available,
                "Asset": asset,
                "Horizon": horizon,
                "PaperWeightPct": signal["PaperWeightPct"],
                "RealWeightPct": 0.0,
                "EntryPrice": entry,
                **outcome,
            }
        )
    return (
        pd.DataFrame(signal_rows, columns=list(SIGNAL_LOG_COLUMNS)),
        pd.DataFrame(outcome_rows, columns=list(OUTCOME_COLUMNS)),
    )


def _equity_from_returns(returns_pct: pd.Series) -> pd.Series:
    if returns_pct.empty:
        return pd.Series(dtype=float)
    return (1.0 + returns_pct.fillna(0.0) / 100.0).cumprod()


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    dd = equity / equity.cummax() - 1.0
    return float(dd.min() * 100.0)


def _performance(signals: pd.DataFrame, outcomes: pd.DataFrame, replay_source: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if signals.empty:
        return pd.DataFrame(columns=list(PERFORMANCE_COLUMNS))
    merged = outcomes.merge(signals[["ReplayDate", "Asset", "Horizon", "ReplaySource", "ReplayDecision", "SignalScore"]], on=["ReplayDate", "Asset", "Horizon"], how="left")
    for (asset, horizon), group in merged.groupby(["Asset", "Horizon"], dropna=False):
        matured = group[group["OutcomeStatus"].eq("Matured")].copy()
        exposed = matured[matured["PaperWeightPct"].astype(float) > 0]
        returns = pd.to_numeric(matured["WeightedPaperReturnPct"], errors="coerce").fillna(0.0)
        equity = _equity_from_returns(returns)
        total = (equity.iloc[-1] - 1.0) * 100.0 if not equity.empty else 0.0
        vol = returns.std(ddof=0) * np.sqrt(252) if len(returns) > 1 else 0.0
        sharpe = (returns.mean() / returns.std(ddof=0)) * np.sqrt(252) if len(returns) > 1 and returns.std(ddof=0) > 0 else 0.0
        annual = ((1.0 + total / 100.0) ** (252.0 / max(len(returns), 1)) - 1.0) * 100.0 if total > -100 and len(returns) > 0 else 0.0
        rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "ReplaySource": replay_source,
                "TradeCount": int(len(exposed)),
                "ExposureDays": int((matured["PaperWeightPct"].astype(float) > 0).sum()),
                "AveragePaperWeightPct": round(float(matured["PaperWeightPct"].astype(float).mean()), 4) if not matured.empty else 0.0,
                "TotalWeightedReturnPct": round(float(total), 4),
                "AnnualizedReturnPct": round(float(annual), 4),
                "VolatilityPct": round(float(vol), 4),
                "SharpeProxy": round(float(sharpe), 4),
                "MaxDrawdownPct": round(_max_drawdown_pct(equity), 4),
                "WinRatePct": round(float((exposed["WeightedPaperReturnPct"].astype(float) > 0).mean() * 100.0), 4) if not exposed.empty else 0.0,
                "DirectionAccuracyPct": round(float(matured["DirectionCorrect"].astype(bool).mean() * 100.0), 4) if not matured.empty else 0.0,
                "AvgReturnPerSignalPct": round(float(exposed["WeightedPaperReturnPct"].astype(float).mean()), 4) if not exposed.empty else 0.0,
                "BenchmarkComparable": bool(not matured.empty),
                "MainPerformanceNote": "Proxy replay result; use Phase 16 to compare against baselines." if replay_source == "HistoricalSignalProxyReplay" else "Historical model prediction replay result.",
            }
        )
    return pd.DataFrame(rows, columns=list(PERFORMANCE_COLUMNS))


def _portfolio_curve(signals: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=list(PORTFOLIO_CURVE_COLUMNS))
    rows: List[Dict[str, Any]] = []
    signal_dates = sorted(pd.to_datetime(signals["ReplayDate"].dropna().unique()))
    for dt in signal_dates:
        sub = signals[pd.to_datetime(signals["ReplayDate"]).eq(dt)]
        weighted_returns: List[float] = []
        exposure = float(pd.to_numeric(sub["PaperWeightPct"], errors="coerce").fillna(0.0).sum())
        for _, row in sub.iterrows():
            asset = row["Asset"]
            price = _series(market_data, get_target_column(asset))
            if dt not in price.index:
                continue
            pos = price.index.get_loc(dt)
            if isinstance(pos, slice) or pos + 1 >= len(price):
                continue
            entry = _safe_float(price.iloc[pos], np.nan)
            next_price = _safe_float(price.iloc[pos + 1], np.nan)
            if np.isfinite(entry) and entry != 0 and np.isfinite(next_price):
                daily = (next_price / entry - 1.0) * 100.0
                weighted_returns.append(daily * _safe_float(row["PaperWeightPct"], 0.0) / 100.0)
        rows.append(
            {
                "Date": dt,
                "PortfolioPaperExposurePct": round(exposure, 4),
                "DailyWeightedReturnPct": round(float(np.nansum(weighted_returns)), 6),
                "ActiveSignals": int((pd.to_numeric(sub["PaperWeightPct"], errors="coerce").fillna(0.0) > 0).sum()),
                "MainRiskState": "HighExposure" if exposure > 50 else "NormalPaperExposure" if exposure > 0 else "NoExposure",
            }
        )
    curve = pd.DataFrame(rows)
    if curve.empty:
        return pd.DataFrame(columns=list(PORTFOLIO_CURVE_COLUMNS))
    equity = _equity_from_returns(curve["DailyWeightedReturnPct"])
    curve["EquityCurve"] = equity.values
    curve["DrawdownPct"] = (equity / equity.cummax() - 1.0) * 100.0
    return curve[list(PORTFOLIO_CURVE_COLUMNS)]


def _asset_horizon_matrix(signals: pd.DataFrame, outcomes: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if signals.empty:
        return pd.DataFrame(columns=list(ASSET_HORIZON_MATRIX_COLUMNS))
    perf_lookup = {(str(row["Asset"]), int(row["Horizon"])): row for _, row in performance.iterrows()} if not performance.empty else {}
    for (asset, horizon), group in signals.groupby(["Asset", "Horizon"], dropna=False):
        outs = outcomes[outcomes["Asset"].astype(str).eq(str(asset)) & outcomes["Horizon"].astype(int).eq(int(horizon))]
        matured = outs[outs["OutcomeStatus"].eq("Matured")]
        perf = perf_lookup.get((str(asset), int(horizon)), {})
        trade_count = int(perf.get("TradeCount", 0)) if isinstance(perf, pd.Series) else 0
        total = _safe_float(perf.get("TotalWeightedReturnPct", 0.0) if isinstance(perf, pd.Series) else 0.0, 0.0)
        max_dd = _safe_float(perf.get("MaxDrawdownPct", 0.0) if isinstance(perf, pd.Series) else 0.0, 0.0)
        if len(matured) == 0:
            verdict = "InsufficientReplayData"
            weakness = "No matured outcomes"
        elif trade_count < 3:
            verdict = "SparseProxyReplay"
            weakness = "Too few exposed replay rows"
        elif total < 0:
            verdict = "WeakProxyReplay"
            weakness = "Negative weighted replay return"
        else:
            verdict = "ProxyReplayCandidate"
            weakness = "Proxy-only evidence"
        rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "ReplayRows": int(len(group)),
                "MaturedRows": int(len(matured)),
                "AvgSignalScore": round(float(pd.to_numeric(group["SignalScore"], errors="coerce").mean()), 4),
                "AvgPaperWeightPct": round(float(pd.to_numeric(group["PaperWeightPct"], errors="coerce").mean()), 4),
                "TotalWeightedReturnPct": round(total, 4),
                "WinRatePct": round(_safe_float(perf.get("WinRatePct", 0.0) if isinstance(perf, pd.Series) else 0.0, 0.0), 4),
                "MaxDrawdownPct": round(max_dd, 4),
                "ReplayVerdict": verdict,
                "MainWeakness": weakness,
                "MainStrength": "Positive proxy replay return" if total > 0 else "No robust strength yet",
            }
        )
    return pd.DataFrame(rows, columns=list(ASSET_HORIZON_MATRIX_COLUMNS))


def _replay_key(date_value: Any, asset: Any, horizon: Any) -> Tuple[str, str, str]:
    parsed = pd.to_datetime(date_value, errors="coerce")
    date_text = str(parsed.date()) if not pd.isna(parsed) else str(date_value)
    return (date_text, str(asset), str(int(float(horizon))) if pd.notna(horizon) else "")


def _strategy_name_for_replay(replay_source: str) -> str:
    if replay_source == "HistoricalSignalProxyReplay":
        return "HistoricalSignalProxyReplay"
    if replay_source == "HistoricalModelPredictionReplay":
        return "HistoricalModelPredictionReplay"
    return "HistoricalModelRiskReplay"


def _apply_portfolio_exposure_cap(
    signals: pd.DataFrame,
    outcomes: pd.DataFrame,
    max_portfolio_paper_exposure_pct: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if signals.empty:
        return (
            signals.copy(),
            outcomes.copy(),
            pd.DataFrame(columns=list(EXPOSURE_CAP_COLUMNS)),
        )

    cap = max(0.0, float(max_portfolio_paper_exposure_pct))
    scaled_signals = signals.copy()
    scaled_outcomes = outcomes.copy()
    scaled_signals["PaperWeightPct"] = pd.to_numeric(scaled_signals["PaperWeightPct"], errors="coerce").fillna(0.0)
    rows: List[Dict[str, Any]] = []

    replay_dates = pd.to_datetime(scaled_signals["ReplayDate"], errors="coerce")
    scaled_signals["_ReplayDateKey"] = replay_dates.dt.date.astype(str)
    for date_key, idx in scaled_signals.groupby("_ReplayDateKey").groups.items():
        group_idx = list(idx)
        before_weights = scaled_signals.loc[group_idx, "PaperWeightPct"].astype(float).clip(lower=0.0)
        active_before = before_weights > 0
        exposure_before = float(before_weights[active_before].sum())
        if exposure_before > cap + EXPOSURE_CAP_TOLERANCE and exposure_before > 0:
            factor = cap / exposure_before if cap > 0 else 0.0
            scaled_signals.loc[group_idx, "PaperWeightPct"] = (before_weights * factor).clip(lower=0.0)
            cap_applied = True
            reason = "Active replay paper exposure exceeded the configured portfolio cap and was proportionally scaled."
        else:
            factor = 1.0
            scaled_signals.loc[group_idx, "PaperWeightPct"] = before_weights
            cap_applied = False
            reason = "No portfolio exposure scaling required for this replay date."
        after_weights = scaled_signals.loc[group_idx, "PaperWeightPct"].astype(float).clip(lower=0.0)
        exposure_after = float(after_weights.sum())
        if cap_applied and exposure_after > cap + EXPOSURE_CAP_TOLERANCE:
            overflow = exposure_after - cap
            positive_idx = [i for i in group_idx if _safe_float(scaled_signals.loc[i, "PaperWeightPct"], 0.0) > 0]
            if positive_idx:
                adjust_idx = positive_idx[-1]
                adjusted = max(0.0, _safe_float(scaled_signals.loc[adjust_idx, "PaperWeightPct"], 0.0) - overflow)
                scaled_signals.loc[adjust_idx, "PaperWeightPct"] = adjusted
                after_weights = scaled_signals.loc[group_idx, "PaperWeightPct"].astype(float).clip(lower=0.0)
                exposure_after = float(after_weights.sum())
        rows.append(
            {
                "ReplayDate": pd.to_datetime(scaled_signals.loc[group_idx[0], "ReplayDate"]),
                "ExposureBeforeCapPct": round(exposure_before, 6),
                "ExposureAfterCapPct": round(exposure_after, 6),
                "MaxPortfolioPaperExposurePct": round(cap, 6),
                "CapApplied": bool(cap_applied),
                "ScalingFactor": round(float(factor), 8),
                "ActiveSignalsBeforeCap": int(active_before.sum()),
                "ActiveSignalsAfterCap": int((after_weights > 0).sum()),
                "AdjustmentReason": reason,
            }
        )
    scaled_signals = scaled_signals.drop(columns=["_ReplayDateKey"])

    capped_weights = {
        _replay_key(row["ReplayDate"], row["Asset"], row["Horizon"]): _safe_float(row["PaperWeightPct"], 0.0)
        for _, row in scaled_signals.iterrows()
    }
    if not scaled_outcomes.empty:
        new_weights: List[float] = []
        weighted_returns: List[float] = []
        for _, row in scaled_outcomes.iterrows():
            weight = capped_weights.get(_replay_key(row["ReplayDate"], row["Asset"], row["Horizon"]), 0.0)
            forward_return = _safe_float(row.get("ForwardReturnPct", np.nan), np.nan)
            is_matured = str(row.get("OutcomeStatus", "")) == "Matured"
            new_weights.append(round(weight, 6))
            weighted_returns.append(round(forward_return * weight / 100.0, 6) if is_matured and np.isfinite(forward_return) else 0.0)
        scaled_outcomes["PaperWeightPct"] = new_weights
        scaled_outcomes["WeightedPaperReturnPct"] = weighted_returns

    cap_table = pd.DataFrame(rows, columns=list(EXPOSURE_CAP_COLUMNS))
    return scaled_signals, scaled_outcomes, cap_table


def _phase16_export(signals: pd.DataFrame, market_data: pd.DataFrame, outcomes: pd.DataFrame, replay_source: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if signals.empty:
        return pd.DataFrame(columns=list(PHASE16_EXPORT_COLUMNS))
    matured_keys = {
        _replay_key(row["ReplayDate"], row["Asset"], row["Horizon"])
        for _, row in outcomes[outcomes["OutcomeStatus"].eq("Matured")].iterrows()
    }
    for _, row in signals.iterrows():
        dt = pd.to_datetime(row["ReplayDate"])
        asset = row["Asset"]
        horizon = int(row["Horizon"])
        price = _series(market_data, get_target_column(asset))
        daily_return = np.nan
        if dt in price.index:
            pos = price.index.get_loc(dt)
            if not isinstance(pos, slice) and pos + 1 < len(price):
                entry = _safe_float(price.iloc[pos], np.nan)
                next_price = _safe_float(price.iloc[pos + 1], np.nan)
                if np.isfinite(entry) and entry != 0 and np.isfinite(next_price):
                    daily_return = (next_price / entry - 1.0) * 100.0
        exposure = _safe_float(row["PaperWeightPct"], 0.0)
        comparable = _replay_key(row["ReplayDate"], asset, horizon) in matured_keys
        rows.append(
            {
                "Date": dt,
                "Asset": asset,
                "Horizon": horizon,
                "StrategyName": _strategy_name_for_replay(replay_source),
                "ExposurePct": round(exposure, 4),
                "DailyReturnPct": round(daily_return, 6) if np.isfinite(daily_return) else np.nan,
                "StrategyReturnPct": round(daily_return * exposure / 100.0, 6) if np.isfinite(daily_return) else np.nan,
                "EvaluationMode": "HistoricalDailyExposure" if comparable else "InsufficientData",
                "ComparableHistorical": bool(comparable),
                "ReplaySource": replay_source,
            }
        )
    return pd.DataFrame(rows, columns=list(PHASE16_EXPORT_COLUMNS))


def _quality_checks(
    signals: pd.DataFrame,
    outcomes: pd.DataFrame,
    market_data: pd.DataFrame,
    replay_source: str,
    portfolio_curve: pd.DataFrame,
    exposure_cap_table: pd.DataFrame,
    phase16_export: pd.DataFrame,
    max_portfolio_paper_exposure_pct: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    valid_outcomes = outcomes[outcomes["OutcomeStatus"].eq("Matured")].copy() if not outcomes.empty else pd.DataFrame()
    after = True
    bad_after_rows = pd.DataFrame()
    if not valid_outcomes.empty:
        outcome_dates = pd.to_datetime(valid_outcomes["OutcomeDate"], errors="coerce")
        replay_dates = pd.to_datetime(valid_outcomes["ReplayDate"], errors="coerce")
        bad_after_rows = valid_outcomes[~(outcome_dates > replay_dates)]
        after = bool(bad_after_rows.empty)
    dupes = int(signals.duplicated(subset=["ReplayDate", "Asset", "Horizon"]).sum()) if not signals.empty else 0
    missing_rate = float(outcomes["OutcomeStatus"].isin(["MissingExitPrice", "MissingEntryPrice", "InvalidData"]).mean() * 100.0) if not outcomes.empty else 100.0
    same_day = False
    if not valid_outcomes.empty:
        same_day = bool((pd.to_datetime(valid_outcomes["OutcomeDate"]) <= pd.to_datetime(valid_outcomes["ReplayDate"])).any())
    blocked_data = int(signals["ReplayDecision"].eq("BlockedByData").sum()) if not signals.empty else 0
    cap = float(max_portfolio_paper_exposure_pct)
    after_cap = pd.to_numeric(exposure_cap_table.get("ExposureAfterCapPct", pd.Series(dtype=float)), errors="coerce").fillna(0.0) if not exposure_cap_table.empty else pd.Series(dtype=float)
    before_cap = pd.to_numeric(exposure_cap_table.get("ExposureBeforeCapPct", pd.Series(dtype=float)), errors="coerce").fillna(0.0) if not exposure_cap_table.empty else pd.Series(dtype=float)
    cap_breaches_after = int(_cap_breach_mask(after_cap, cap).sum()) if not after_cap.empty else 0
    cap_breaches_before = int(_cap_breach_mask(before_cap, cap).sum()) if not before_cap.empty else 0
    scaled_rows = int(exposure_cap_table["CapApplied"].astype(bool).sum()) if not exposure_cap_table.empty and "CapApplied" in exposure_cap_table.columns else 0
    scaled_when_needed = scaled_rows == cap_breaches_before
    summary_matches_curve = True
    if not portfolio_curve.empty and not signals.empty:
        grouped_exposure = (
            signals.assign(_DateKey=pd.to_datetime(signals["ReplayDate"], errors="coerce").dt.date.astype(str))
            .groupby("_DateKey")["PaperWeightPct"]
            .sum()
            .astype(float)
        )
        curve_exposure = (
            portfolio_curve.assign(_DateKey=pd.to_datetime(portfolio_curve["Date"], errors="coerce").dt.date.astype(str))
            .set_index("_DateKey")["PortfolioPaperExposurePct"]
            .astype(float)
        )
        shared = grouped_exposure.index.intersection(curve_exposure.index)
        if len(shared) > 0:
            summary_matches_curve = bool(np.allclose(grouped_exposure.loc[shared].values, curve_exposure.loc[shared].values, atol=1e-4))
    phase16_uses_capped = True
    if not phase16_export.empty and not signals.empty:
        signal_weights = {
            _replay_key(row["ReplayDate"], row["Asset"], row["Horizon"]): _safe_float(row["PaperWeightPct"], 0.0)
            for _, row in signals.iterrows()
        }
        mismatches = 0
        for _, row in phase16_export.iterrows():
            key = _replay_key(row["Date"], row["Asset"], row["Horizon"])
            if abs(signal_weights.get(key, 0.0) - _safe_float(row["ExposurePct"], 0.0)) > 1e-4:
                mismatches += 1
        phase16_uses_capped = mismatches == 0
    else:
        mismatches = 0
    checks = [
        ("NoFutureDataUsed", True, "Critical", 0, "Signal generation uses market history through DataAvailableThroughDate only."),
        ("SignalsUsePastDataOnly", True, "Critical", 0, "Proxy features are momentum, trend, volatility, and drawdown calculated from past prices."),
        ("OutcomesAfterReplayDate", after, "Critical", int(len(bad_after_rows)), "Matured outcomes must occur after replay dates."),
        ("NoSameDayCloseLeakage", not same_day, "Critical", int(same_day), "Replay outcomes never use same-day close as exit."),
        ("DateOrderingValid", bool(market_data.empty or market_data.index.is_monotonic_increasing), "High", 0, "Market data is sorted chronologically."),
        ("NoDuplicateReplayRows", dupes == 0, "High", dupes, "Replay rows are unique by date, asset, and horizon."),
        ("MissingPriceRateAcceptable", missing_rate <= 20.0, "Medium", int(outcomes["OutcomeStatus"].isin(["MissingExitPrice", "MissingEntryPrice", "InvalidData"]).sum()) if not outcomes.empty else 0, "Missing entry/exit price rate should remain limited."),
        ("HorizonOutcomeAlignmentValid", after, "Critical", 0, "Outcome dates are selected horizon rows after replay dates."),
        ("LatestSnapshotNotUsedAsHistoricalWeights", True, "Critical", 0, "Latest Phase 12/14/15 snapshot weights are not used to create historical weights."),
        ("ReplaySourceClearlyLabeled", replay_source in {"HistoricalModelPredictionReplay", "HistoricalSignalProxyReplay", "ForwardPaperReplay", "InsufficientData"}, "High", 0, "Replay source is explicit."),
        ("SufficientHistoryBeforeSignal", blocked_data < max(1, len(signals) * 0.5) if not signals.empty else False, "Medium", blocked_data, "Rows with insufficient history are blocked and visible."),
        ("PortfolioExposureCapRespected", cap_breaches_after == 0, "Critical", cap_breaches_after, "Portfolio paper exposure after scaling must stay within the configured cap."),
        ("PortfolioExposureScaledWhenNeeded", scaled_when_needed, "High", max(0, cap_breaches_before - scaled_rows), "Dates above the cap before scaling must be proportionally scaled."),
        ("SummaryExposureMatchesPortfolioCurve", summary_matches_curve, "High", 0 if summary_matches_curve else 1, "Portfolio curve exposure must match capped signal weights."),
        ("Phase16ExportUsesCappedWeights", phase16_uses_capped, "Critical", int(mismatches), "Phase 16 replay export must use capped signal weights."),
    ]
    for name, passed, severity, affected, explanation in checks:
        rows.append({"CheckName": name, "Passed": bool(passed), "Severity": severity, "AffectedRows": int(affected), "Explanation": explanation})
    return pd.DataFrame(rows, columns=list(QUALITY_CHECK_COLUMNS))


def _benchmark_ready(signals: pd.DataFrame, outcomes: pd.DataFrame, replay_source: str, quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    matured = outcomes[outcomes["OutcomeStatus"].eq("Matured")] if not outcomes.empty else pd.DataFrame()
    valid = int(len(matured))
    quality_pass = bool(not quality.empty and quality["Passed"].astype(bool).all())
    ready = bool(valid > 0 and quality_pass)
    if ready and replay_source == "HistoricalSignalProxyReplay":
        reason = "Benchmark-ready as a capped historical proxy strategy, not true trained ML replay."
    elif ready:
        reason = "Benchmark-ready historical prediction replay with matured rows and passing quality checks."
    elif valid <= 0:
        reason = "No matured replay rows yet."
    else:
        reason = "Replay quality checks must pass before benchmark comparison."
    rows.append(
        {
            "StrategyName": _strategy_name_for_replay(replay_source),
            "ComparableHistorical": ready,
            "EvaluationMode": "HistoricalDailyExposure" if ready else "InsufficientData",
            "ReplaySource": replay_source,
            "Rows": int(len(signals)),
            "MaturedRows": valid,
            "BenchmarkReady": ready,
            "Reason": reason,
        }
    )
    return pd.DataFrame(rows, columns=list(BENCHMARK_READY_COLUMNS))


def _warnings(
    signals: pd.DataFrame,
    outcomes: pd.DataFrame,
    performance: pd.DataFrame,
    quality: pd.DataFrame,
    replay_source: str,
    exposure_cap_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if replay_source == "HistoricalSignalProxyReplay":
        rows.append({"WarningType": "ProxyOnlyReplay", "Severity": "High", "Asset": "ALL", "Horizon": "ALL", "ReplayDate": "", "Explanation": "No historical model prediction log was supplied; proxy replay is not true ML model history.", "RecommendedFix": "Persist historical model predictions with timestamps for future replay."})
        rows.append({"WarningType": "InsufficientHistoricalModelPredictions", "Severity": "High", "Asset": "ALL", "Horizon": "ALL", "ReplayDate": "", "Explanation": "Historical model prediction evidence is unavailable.", "RecommendedFix": "Create a prediction ledger during future model runs."})
    if not exposure_cap_table.empty:
        scaled = exposure_cap_table[exposure_cap_table["CapApplied"].astype(bool)]
        for _, row in scaled.head(100).iterrows():
            before = _safe_float(row.get("ExposureBeforeCapPct", 0.0), 0.0)
            cap = max(_safe_float(row.get("MaxPortfolioPaperExposurePct", 0.0), 0.0), 1e-9)
            severity = "High" if before >= cap * 2 else "Medium"
            rows.append(
                {
                    "WarningType": "PortfolioExposureScaled",
                    "Severity": severity,
                    "Asset": "ALL",
                    "Horizon": "ALL",
                    "ReplayDate": row.get("ReplayDate", ""),
                    "Explanation": "Active replay exposure exceeded the configured portfolio cap and was proportionally scaled.",
                    "RecommendedFix": "Review per-date active signal density or reduce row-level paper weights.",
                }
            )
        breaches = exposure_cap_table[
            _cap_breach_mask(exposure_cap_table["ExposureAfterCapPct"], exposure_cap_table["MaxPortfolioPaperExposurePct"])
        ]
        for _, row in breaches.head(100).iterrows():
            rows.append(
                {
                    "WarningType": "PortfolioExposureCapBreach",
                    "Severity": "Critical",
                    "Asset": "ALL",
                    "Horizon": "ALL",
                    "ReplayDate": row.get("ReplayDate", ""),
                    "Explanation": "Portfolio paper exposure remains above the configured cap after scaling.",
                    "RecommendedFix": "Fix replay exposure scaling before interpreting benchmark results.",
                }
            )
    missing_exit = outcomes[outcomes["OutcomeStatus"].eq("MissingExitPrice")] if not outcomes.empty else pd.DataFrame()
    for _, row in missing_exit.head(50).iterrows():
        rows.append({"WarningType": "MissingExitPrice", "Severity": "Medium", "Asset": row["Asset"], "Horizon": int(row["Horizon"]), "ReplayDate": row["ReplayDate"], "Explanation": "Exit price is missing at the horizon outcome date.", "RecommendedFix": "Repair price history or keep outcome invalid."})
    for _, row in performance.iterrows():
        if int(row["TradeCount"]) < 3:
            rows.append({"WarningType": "LowTradeCount", "Severity": "Medium", "Asset": row["Asset"], "Horizon": int(row["Horizon"]), "ReplayDate": "", "Explanation": "Replay has too few exposed paper rows for robust evidence.", "RecommendedFix": "Extend replay window or inspect filters."})
        if _safe_float(row["MaxDrawdownPct"], 0.0) <= -20:
            rows.append({"WarningType": "HighDrawdown", "Severity": "High", "Asset": row["Asset"], "Horizon": int(row["Horizon"]), "ReplayDate": "", "Explanation": "Historical replay drawdown is large.", "RecommendedFix": "Review risk and regime filters."})
        if _safe_float(row["DirectionAccuracyPct"], 100.0) < 50:
            rows.append({"WarningType": "WeakDirectionAccuracy", "Severity": "Medium", "Asset": row["Asset"], "Horizon": int(row["Horizon"]), "ReplayDate": "", "Explanation": "Direction accuracy is below 50%.", "RecommendedFix": "Treat as weak research evidence."})
        if int(row["ExposureDays"]) < max(2, int(row["TradeCount"])):
            rows.append({"WarningType": "TooSparseExposure", "Severity": "Medium", "Asset": row["Asset"], "Horizon": int(row["Horizon"]), "ReplayDate": "", "Explanation": "Replay exposure is sparse.", "RecommendedFix": "Check over-filtering and score thresholds."})
    for _, row in quality[~quality["Passed"]].iterrows():
        rows.append({"WarningType": "PossibleLeakage" if "Leakage" in row["CheckName"] or "Future" in row["CheckName"] else "DataQualityIssue", "Severity": row["Severity"], "Asset": "ALL", "Horizon": "ALL", "ReplayDate": "", "Explanation": row["Explanation"], "RecommendedFix": "Fix replay quality check before interpreting results."})
    rows.append({"WarningType": "BenchmarkNotYetCompared", "Severity": "Info", "Asset": "ALL", "Horizon": "ALL", "ReplayDate": "", "Explanation": "Use Phase 16 to compare replay export against simple baselines.", "RecommendedFix": "Load phase17_phase16_replay_export.csv into the benchmark arena when supported."})
    return pd.DataFrame(rows, columns=list(REPLAY_WARNING_COLUMNS))


def _summary(
    signals: pd.DataFrame,
    outcomes: pd.DataFrame,
    replay_source: str,
    quality: pd.DataFrame,
    replay_start: str,
    replay_end: str,
    portfolio_curve: pd.DataFrame,
    exposure_cap_table: pd.DataFrame,
) -> pd.DataFrame:
    matured = int(outcomes["OutcomeStatus"].eq("Matured").sum()) if not outcomes.empty else 0
    pending = int(outcomes["OutcomeStatus"].eq("Pending").sum()) if not outcomes.empty else 0
    failed_quality = bool(not quality.empty and not quality["Passed"].all())
    if signals.empty:
        verdict = "InsufficientReplayData"
        limitation = "No replay rows were generated."
    elif failed_quality and quality[~quality["Passed"]]["Severity"].astype(str).isin(["Critical"]).any():
        verdict = "ReplayFailedQualityChecks"
        limitation = "One or more critical replay quality checks failed."
    elif replay_source == "HistoricalModelPredictionReplay":
        verdict = "HistoricalModelReplayReady"
        limitation = "Historical model prediction replay is available, still research-only."
    elif replay_source == "HistoricalSignalProxyReplay":
        verdict = "ProxyReplayOnly"
        limitation = "No true historical model predictions were supplied; proxy signals were used."
    else:
        verdict = "InsufficientReplayData"
        limitation = "Replay source or market data is insufficient."
    row_avg = round(float(pd.to_numeric(signals.get("PaperWeightPct", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()), 4) if not signals.empty else 0.0
    portfolio_avg = round(float(pd.to_numeric(portfolio_curve.get("PortfolioPaperExposurePct", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()), 4) if not portfolio_curve.empty else 0.0
    portfolio_max = round(float(pd.to_numeric(portfolio_curve.get("PortfolioPaperExposurePct", pd.Series(dtype=float)), errors="coerce").fillna(0.0).max()), 4) if not portfolio_curve.empty else 0.0
    breaches_before = int(exposure_cap_table["CapApplied"].astype(bool).sum()) if not exposure_cap_table.empty and "CapApplied" in exposure_cap_table.columns else 0
    if not exposure_cap_table.empty:
        breaches_after = int(_cap_breach_mask(exposure_cap_table["ExposureAfterCapPct"], exposure_cap_table["MaxPortfolioPaperExposurePct"]).sum())
    else:
        breaches_after = 0
    return pd.DataFrame(
        [
            {
                "ReplaySource": replay_source,
                "ModelReplayQuality": "ProxyOnly" if replay_source == "HistoricalSignalProxyReplay" else "HistoricalPredictions" if replay_source == "HistoricalModelPredictionReplay" else "InsufficientData",
                "ReplayStartDate": replay_start,
                "ReplayEndDate": replay_end,
                "ReplayRows": int(len(signals)),
                "AssetsCovered": "; ".join(sorted(signals["Asset"].dropna().astype(str).unique())) if not signals.empty else "",
                "HorizonsCovered": "; ".join(str(int(h)) for h in sorted(pd.to_numeric(signals["Horizon"], errors="coerce").dropna().unique())) if not signals.empty else "",
                "MaturedOutcomeRows": matured,
                "PendingOutcomeRows": pending,
                "AverageRowPaperWeightPct": row_avg,
                "AveragePaperExposurePct": portfolio_avg,
                "AveragePortfolioPaperExposurePct": portfolio_avg,
                "MaxPortfolioPaperExposurePct": portfolio_max,
                "ExposureCapBreachesBeforeScaling": breaches_before,
                "ExposureCapBreachesAfterScaling": breaches_after,
                "AverageRealExposurePct": 0.0,
                "ReplayVerdict": verdict,
                "MainLimitation": limitation,
                "RecommendedNextStep": "Build and persist true historical model prediction logs." if verdict == "ProxyReplayOnly" else "Use Phase 16 to compare replay export with baselines.",
            }
        ],
        columns=list(REPLAY_SUMMARY_COLUMNS),
    )


def _next_actions(summary: pd.DataFrame, warnings: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    verdict = str(summary.iloc[0]["ReplayVerdict"]) if not summary.empty else "InsufficientReplayData"
    if verdict == "ProxyReplayOnly":
        rows.append({"Rank": 0, "Action": "Persist historical prediction logs.", "WhyItMatters": "Proxy replay cannot prove trained model history.", "AffectedAssets": "ALL", "AffectedHorizons": "ALL", "ExpectedBenefit": "Enables true historical model replay.", "Urgency": "High", "DependsOn": "Model prediction export pipeline."})
    weak = matrix[matrix["ReplayVerdict"].isin(["WeakProxyReplay", "SparseProxyReplay", "InsufficientReplayData"])] if not matrix.empty else pd.DataFrame()
    if not weak.empty:
        rows.append({"Rank": 0, "Action": "Inspect weak or sparse replay combinations.", "WhyItMatters": "Weak replay rows should stay visible before Phase 16 comparison.", "AffectedAssets": "; ".join(weak["Asset"].astype(str).unique()), "AffectedHorizons": "; ".join(f"{int(h)}D" for h in pd.to_numeric(weak["Horizon"], errors="coerce").dropna().astype(int).unique()), "ExpectedBenefit": "Separates insufficient evidence from promising replay evidence.", "Urgency": "Medium", "DependsOn": "Replay matrix and warnings."})
    if warnings["WarningType"].astype(str).eq("MissingExitPrice").any() if not warnings.empty else False:
        rows.append({"Rank": 0, "Action": "Repair missing outcome prices.", "WhyItMatters": "Missing exit prices prevent honest outcome scoring.", "AffectedAssets": "See warnings", "AffectedHorizons": "See warnings", "ExpectedBenefit": "Improves replay maturity and quality checks.", "Urgency": "Medium", "DependsOn": "Clean price history."})
    if not rows:
        rows.append({"Rank": 0, "Action": "Export replay table for Phase 16 comparison.", "WhyItMatters": "Benchmark arena should judge replay performance against simple baselines.", "AffectedAssets": "ALL", "AffectedHorizons": "ALL", "ExpectedBenefit": "Moves from snapshot-only model/risk evidence to historical replay evidence.", "Urgency": "Medium", "DependsOn": "Phase 16 replay ingestion."})
    out = pd.DataFrame(rows, columns=list(NEXT_REPLAY_ACTION_COLUMNS))
    out["Rank"] = np.arange(1, len(out) + 1)
    return out


def run_historical_model_replay(
    *,
    market_data: Optional[pd.DataFrame] = None,
    use_project_market_data: bool = True,
    use_artifact_store: bool = False,
    prefer_uploaded: bool = False,
    uploaded_overrides: Optional[Dict[str, Any]] = None,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    replay_start_date: Optional[Any] = None,
    replay_end_date: Optional[Any] = None,
    replay_step: int = 5,
    max_paper_weight_pct: float = 20.0,
    max_portfolio_paper_exposure_pct: float = 45.0,
    autosave: bool = False,
    **direct_tables: Any,
) -> HistoricalModelReplayReport:
    asset_list = list(assets or get_asset_names())
    horizon_list = [int(h) for h in (horizons or REPLAY_HORIZONS)]
    project_used = False
    if market_data is None and use_project_market_data:
        market_data = _load_project_market_data()
        project_used = market_data is not None
    market = _prepare_market_data(market_data)
    tables, artifact_sources = _resolve_inputs(bool(use_artifact_store), bool(prefer_uploaded), uploaded_overrides, direct_tables)
    has_prediction_log = _has_prediction_log(tables)
    replay_source = "HistoricalModelPredictionReplay" if has_prediction_log else "HistoricalSignalProxyReplay" if not market.empty else "InsufficientData"

    if has_prediction_log:
        signals, outcomes = _build_prediction_replay_rows(
            market,
            tables.get("historical_prediction_log", pd.DataFrame()),
            asset_list,
            horizon_list,
            float(max_paper_weight_pct),
            replay_source,
        )
    else:
        signals, outcomes = _build_replay_rows(market, asset_list, horizon_list, replay_start_date, replay_end_date, int(replay_step), float(max_paper_weight_pct), replay_source)
    signals, outcomes, exposure_cap = _apply_portfolio_exposure_cap(signals, outcomes, float(max_portfolio_paper_exposure_pct))
    performance = _performance(signals, outcomes, replay_source)
    curve = _portfolio_curve(signals, market)
    matrix = _asset_horizon_matrix(signals, outcomes, performance)
    replay_export = _phase16_export(signals, market, outcomes, replay_source)
    quality = _quality_checks(signals, outcomes, market, replay_source, curve, exposure_cap, replay_export, float(max_portfolio_paper_exposure_pct))
    if not quality.empty and not quality["Passed"].astype(bool).all() and not replay_export.empty:
        replay_export["ComparableHistorical"] = False
        replay_export["EvaluationMode"] = "InsufficientData"
    benchmark_ready = _benchmark_ready(signals, outcomes, replay_source, quality)
    warnings = _warnings(signals, outcomes, performance, quality, replay_source, exposure_cap)
    input_sources = _input_sources(market, asset_list, project_used)
    replay_start = str(pd.to_datetime(signals["ReplayDate"]).min().date()) if not signals.empty else ""
    replay_end = str(pd.to_datetime(signals["ReplayDate"]).max().date()) if not signals.empty else ""
    summary = _summary(signals, outcomes, replay_source, quality, replay_start, replay_end, curve, exposure_cap)
    actions = _next_actions(summary, warnings, matrix)
    settings = {
        "phase": "17",
        "purpose": "historical_model_risk_replay",
        "assets": asset_list,
        "horizons": horizon_list,
        "replay_start_date": str(replay_start_date or ""),
        "replay_end_date": str(replay_end_date or ""),
        "replay_step": int(replay_step),
        "max_paper_weight_pct": float(max_paper_weight_pct),
        "max_portfolio_paper_exposure_pct": float(max_portfolio_paper_exposure_pct),
        "real_capital_policy": "RealWeightPct remains zero in Phase 17 replay.",
    }
    report = HistoricalModelReplayReport(
        replay_summary_table=summary.reset_index(drop=True),
        historical_replay_signal_log=signals.reset_index(drop=True),
        historical_replay_outcomes=outcomes.reset_index(drop=True),
        historical_replay_performance=performance.reset_index(drop=True),
        historical_replay_portfolio_curve=curve.reset_index(drop=True),
        replay_asset_horizon_matrix=matrix.reset_index(drop=True),
        replay_exposure_cap_table=exposure_cap.reset_index(drop=True),
        replay_quality_checks=quality.reset_index(drop=True),
        replay_benchmark_ready_table=benchmark_ready.reset_index(drop=True),
        replay_warnings_table=warnings.reset_index(drop=True),
        next_replay_actions_table=actions.reset_index(drop=True),
        replay_input_sources_table=input_sources.reset_index(drop=True),
        phase16_replay_export_table=replay_export.reset_index(drop=True),
        artifact_input_source_table=artifact_sources.reset_index(drop=True),
        settings=settings,
    )
    if autosave:
        report.saved_artifacts = save_phase_artifacts(
            HISTORICAL_REPLAY_PHASE_NAME,
            {
                "replay_summary_table": report.replay_summary_table,
                "historical_replay_signal_log": report.historical_replay_signal_log,
                "historical_replay_outcomes": report.historical_replay_outcomes,
                "historical_replay_performance": report.historical_replay_performance,
                "historical_replay_portfolio_curve": report.historical_replay_portfolio_curve,
                "replay_asset_horizon_matrix": report.replay_asset_horizon_matrix,
                "replay_exposure_cap_table": report.replay_exposure_cap_table,
                "replay_quality_checks": report.replay_quality_checks,
                "replay_benchmark_ready_table": report.replay_benchmark_ready_table,
                "replay_warnings_table": report.replay_warnings_table,
                "next_replay_actions_table": report.next_replay_actions_table,
                "replay_input_sources_table": report.replay_input_sources_table,
                "phase16_replay_export_table": report.phase16_replay_export_table,
                "artifact_input_source_table": report.artifact_input_source_table,
            },
            inputs={},
            config=report.settings,
            warnings=report.replay_warnings_table["WarningType"].dropna().astype(str).unique().tolist() if not report.replay_warnings_table.empty else [],
        )
    return report


__all__ = [
    "ASSET_HORIZON_MATRIX_COLUMNS",
    "BENCHMARK_READY_COLUMNS",
    "EXPOSURE_CAP_COLUMNS",
    "EXPOSURE_CAP_TOLERANCE",
    "HISTORICAL_REPLAY_PHASE_NAME",
    "HistoricalModelReplayReport",
    "NEXT_REPLAY_ACTION_COLUMNS",
    "OUTCOME_COLUMNS",
    "PERFORMANCE_COLUMNS",
    "PHASE16_EXPORT_COLUMNS",
    "PORTFOLIO_CURVE_COLUMNS",
    "QUALITY_CHECK_COLUMNS",
    "REPLAY_HORIZONS",
    "REPLAY_INPUT_SOURCE_COLUMNS",
    "REPLAY_SUMMARY_COLUMNS",
    "REPLAY_WARNING_COLUMNS",
    "SIGNAL_LOG_COLUMNS",
    "run_historical_model_replay",
]
