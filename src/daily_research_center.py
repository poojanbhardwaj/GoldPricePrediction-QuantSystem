"""Phase 11 daily research control center.

This module turns Phase 8, Phase 9, and Phase 10 artifacts into a daily
research-control cockpit. It is research decision support only: it does not
execute trades, guarantee returns, or make unconditional capital approvals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names
from src.artifact_store import build_input_source_table, resolve_artifact, save_phase_artifacts


CONTROL_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)

DAILY_RESEARCH_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "ReportDate",
    "TotalActivePaperSignals",
    "PendingOutcomes",
    "MaturedOutcomes",
    "MaturedToday",
    "OverdueOutcomes",
    "PaperTradeCandidates",
    "WatchlistCandidates",
    "RealCapitalEligibleCandidates",
    "AvoidCandidates",
    "MainWarning",
    "DailyActionSummary",
)

CAPITAL_ELIGIBILITY_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "RealCapitalAllowed",
    "CapitalDeploymentStatus",
    "CapitalPlanType",
    "EvidenceGatePassed",
    "CalibrationGatePassed",
    "ForwardEvidenceGatePassed",
    "BenchmarkGatePassed",
    "DrawdownGatePassed",
    "CostStressGatePassed",
    "TradeCountGatePassed",
    "SevereWarningGatePassed",
    "MainCapitalBlocker",
    "WhatWouldAllowRealCapital",
    "MaxAllowedRealCapitalPct",
    "MaxLossPerIdeaPct",
    "ReviewDate",
)

STRUCTURED_CAPITAL_PLAN_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "CapitalPlanType",
    "MaxAllowedRealCapitalPct",
    "MaxLossPerIdeaPct",
    "PositionSizingRule",
    "EntryCondition",
    "ExitCondition",
    "StopOrInvalidationRule",
    "ScalingRule",
    "DeRiskingRule",
    "ReviewDate",
    "RequiredMonitoring",
    "RiskWarning",
)

CAPITAL_BLOCKER_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "CapitalDeploymentStatus",
    "CapitalPlanType",
    "MainCapitalBlocker",
    "FailedGates",
    "WhatWouldAllowRealCapital",
    "ResearchAction",
)

EVIDENCE_HEALTH_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "PendingSignalCount",
    "MaturedForwardCount",
    "WarningCount",
    "PaperTradeEligibility",
    "CapitalDeploymentStatus",
    "NextReviewDate",
    "MainBlocker",
    "NextEvidenceNeeded",
)

DAILY_NEXT_ACTION_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "DailyAction",
    "Priority",
    "Reason",
    "ReviewDate",
)

WARNING_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "WarningType",
    "Severity",
    "Message",
)

COMPARISON_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "ComparisonStatus",
    "PreviousCapitalDeploymentStatus",
    "CurrentCapitalDeploymentStatus",
    "PreviousMainBlocker",
    "CurrentMainBlocker",
    "Reason",
)


@dataclass
class DailyResearchControlCenterReport:
    daily_research_summary: pd.DataFrame
    active_paper_signals_table: pd.DataFrame
    pending_outcomes_table: pd.DataFrame
    matured_today_table: pd.DataFrame
    overdue_outcomes_table: pd.DataFrame
    top_paper_candidates_today: pd.DataFrame
    watchlist_review_table: pd.DataFrame
    capital_eligibility_table: pd.DataFrame
    structured_capital_plan_table: pd.DataFrame
    capital_blocker_table: pd.DataFrame
    degraded_candidates_table: pd.DataFrame
    improved_candidates_table: pd.DataFrame
    blocked_or_avoid_table: pd.DataFrame
    evidence_health_table: pd.DataFrame
    daily_next_actions_table: pd.DataFrame
    warning_table: pd.DataFrame
    input_source_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: Dict[str, Any] = field(default_factory=dict)
    saved_artifacts: Dict[str, Any] = field(default_factory=dict)


def _empty_report(settings: Optional[Dict[str, Any]] = None) -> DailyResearchControlCenterReport:
    return DailyResearchControlCenterReport(
        daily_research_summary=pd.DataFrame(columns=list(DAILY_RESEARCH_SUMMARY_COLUMNS)),
        active_paper_signals_table=pd.DataFrame(),
        pending_outcomes_table=pd.DataFrame(),
        matured_today_table=pd.DataFrame(),
        overdue_outcomes_table=pd.DataFrame(),
        top_paper_candidates_today=pd.DataFrame(),
        watchlist_review_table=pd.DataFrame(),
        capital_eligibility_table=pd.DataFrame(columns=list(CAPITAL_ELIGIBILITY_COLUMNS)),
        structured_capital_plan_table=pd.DataFrame(columns=list(STRUCTURED_CAPITAL_PLAN_COLUMNS)),
        capital_blocker_table=pd.DataFrame(columns=list(CAPITAL_BLOCKER_COLUMNS)),
        degraded_candidates_table=pd.DataFrame(columns=list(COMPARISON_COLUMNS)),
        improved_candidates_table=pd.DataFrame(columns=list(COMPARISON_COLUMNS)),
        blocked_or_avoid_table=pd.DataFrame(),
        evidence_health_table=pd.DataFrame(columns=list(EVIDENCE_HEALTH_COLUMNS)),
        daily_next_actions_table=pd.DataFrame(columns=list(DAILY_NEXT_ACTION_COLUMNS)),
        warning_table=pd.DataFrame(columns=list(WARNING_COLUMNS)),
        settings=settings or {},
    )


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
        out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce").astype("Int64")
    return out


def _subset(df: pd.DataFrame, asset: str, horizon: int) -> pd.DataFrame:
    if df.empty or not {"Asset", "Horizon"}.issubset(df.columns):
        return pd.DataFrame()
    horizon_values = pd.to_numeric(df["Horizon"].astype(str).str.replace("D", "", regex=False), errors="coerce")
    return df[df["Asset"].astype(str).eq(str(asset)) & horizon_values.eq(int(horizon))].copy()


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _first_value(df: pd.DataFrame, names: Iterable[str], default: Any = np.nan) -> Any:
    if df.empty:
        return default
    for name in names:
        if name in df.columns and df[name].notna().any():
            return df[name].dropna().iloc[0]
    return default


def _mean_value(df: pd.DataFrame, names: Iterable[str], default: float = np.nan) -> float:
    if df.empty:
        return default
    for name in names:
        if name in df.columns:
            values = pd.to_numeric(df[name], errors="coerce").dropna()
            if not values.empty:
                return float(values.mean())
    return default


def _min_value(df: pd.DataFrame, names: Iterable[str], default: float = np.nan) -> float:
    if df.empty:
        return default
    for name in names:
        if name in df.columns:
            values = pd.to_numeric(df[name], errors="coerce").dropna()
            if not values.empty:
                return float(values.min())
    return default


def _collect_warnings(*values: Any) -> List[str]:
    warnings: List[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            parts = value
        else:
            parts = str(value).replace(",", ";").split(";")
        for part in parts:
            token = str(part).strip()
            if token and token.lower() not in {"nan", "none", "<na>"} and token not in warnings:
                warnings.append(token)
    return warnings


def _warning_severity(warning: str) -> str:
    high = {
        "NotFinancialAdvice",
        "NotProductionReady",
        "RealCapitalBlocked",
        "ProbabilityUnreliable",
        "BenchmarkDominated",
        "DrawdownRisk",
        "LowTradeCount",
        "NoMaturedForwardEvidenceYet",
        "EvidenceGateFailed",
        "MissingArtifact",
        "TooFewTradesToCalibrate",
        "Overconfident",
    }
    return "High" if str(warning) in high else "Medium"


def _status_series(df: pd.DataFrame) -> pd.Series:
    if df.empty or "Status" not in df.columns:
        return pd.Series(dtype=str)
    return df["Status"].astype(str)


def _date_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return pd.to_datetime(df[column], errors="coerce")


def _report_date(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value if value is not None else pd.Timestamp.today(), errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp.today()
    return pd.Timestamp(ts).tz_localize(None).normalize()


def _review_date(report_date: pd.Timestamp, horizon: int) -> str:
    return str((report_date + pd.Timedelta(days=max(int(horizon), 5))).date())


def _earliest_date(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return ""
    dates = pd.to_datetime(df[column], errors="coerce").dropna()
    if dates.empty:
        return ""
    return str(pd.Timestamp(dates.min()).date())


def _mode_thresholds(mode: str) -> Dict[str, float]:
    text = str(mode or "Conservative").lower()
    if "aggressive" in text:
        return {"brier": 0.28, "win_rate": 53.0, "beat_rate": 53.0, "severe_warnings": 1}
    if "balanced" in text:
        return {"brier": 0.25, "win_rate": 55.0, "beat_rate": 55.0, "severe_warnings": 1}
    return {"brier": 0.22, "win_rate": 57.0, "beat_rate": 57.0, "severe_warnings": 0}


def _artifact_specs() -> Dict[str, Tuple[str, str, bool]]:
    return {
        "true_raw_trade_log": ("Phase 8I True Raw Trade Logs", "true_raw_trade_log", False),
        "probability_calibration_summary": ("Phase 8F Probability Calibration", "probability_calibration_summary", False),
        "probability_calibration_warnings": ("Phase 8F Probability Calibration", "probability_calibration_warnings", False),
        "forward_signal_log": ("Phase 9 Forward Paper Evidence", "forward_signal_log", False),
        "pending_outcome_table": ("Phase 9 Forward Paper Evidence", "pending_outcome_table", False),
        "matured_outcome_table": ("Phase 9 Forward Paper Evidence", "matured_outcome_table", False),
        "forward_accuracy_summary": ("Phase 9 Forward Paper Evidence", "forward_accuracy_summary", False),
        "forward_probability_calibration_summary": ("Phase 9 Forward Paper Evidence", "forward_probability_calibration_summary", False),
        "forward_warning_table": ("Phase 9 Forward Paper Evidence", "warning_table", False),
        "ranked_asset_horizon_plan": ("Phase 10 Actionable Research Plan", "ranked_asset_horizon_plan", False),
        "plan_card_table": ("Phase 10 Actionable Research Plan", "plan_card_table", False),
        "paper_trade_plan_table": ("Phase 10 Actionable Research Plan", "paper_trade_plan_table", False),
        "watchlist_table": ("Phase 10 Actionable Research Plan", "watchlist_table", False),
        "risk_budget_table": ("Phase 10 Actionable Research Plan", "risk_budget_table", False),
        "phase10_warnings_table": ("Phase 10 Actionable Research Plan", "warnings_table", False),
        "previous_capital_eligibility_table": ("Phase 11 Daily Research Control Center", "capital_eligibility_table", False),
    }


def _resolve_inputs(use_artifact_store: bool, prefer_uploaded: bool, uploaded_overrides: Optional[Dict[str, Any]], direct: Dict[str, Any]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    resolved_items: List[Dict[str, Any]] = []
    tables: Dict[str, pd.DataFrame] = {}
    uploaded_overrides = uploaded_overrides or {}
    for key, (phase, artifact, required) in _artifact_specs().items():
        direct_value = direct.get(key)
        if direct_value is not None:
            data = _normalise_horizon(direct_value)
            tables[key] = data
            resolved_items.append(
                {
                    "Artifact": artifact,
                    "Phase": phase,
                    "Source": "DirectInput",
                    "RunId": "",
                    "Rows": int(len(data)),
                    "CreatedAt": "",
                    "Status": "Loaded",
                    "Path": "",
                }
            )
            continue
        if use_artifact_store:
            resolved = resolve_artifact(
                phase,
                artifact,
                uploaded_file=uploaded_overrides.get(key),
                prefer_uploaded=prefer_uploaded,
                required=required,
            )
            resolved_items.append(resolved)
            tables[key] = _normalise_horizon(resolved.get("Data"))
        else:
            uploaded = uploaded_overrides.get(key)
            if uploaded is not None:
                resolved = resolve_artifact(phase, artifact, uploaded_file=uploaded, prefer_uploaded=True, required=False)
                resolved_items.append(resolved)
                tables[key] = _normalise_horizon(resolved.get("Data"))
            else:
                tables[key] = pd.DataFrame()
                resolved_items.append(
                    {
                        "Artifact": artifact,
                        "Phase": phase,
                        "Source": "Missing",
                        "RunId": "",
                        "Rows": 0,
                        "CreatedAt": "",
                        "Status": "MissingOptional",
                        "Path": "",
                    }
                )
    return tables, build_input_source_table(resolved_items)


def _candidate_metrics(asset: str, horizon: int, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    raw = _subset(tables["true_raw_trade_log"], asset, horizon)
    prob = _subset(tables["probability_calibration_summary"], asset, horizon)
    prob_warnings = _subset(tables["probability_calibration_warnings"], asset, horizon)
    fwd = _subset(tables["forward_signal_log"], asset, horizon)
    pending_artifact = _subset(tables["pending_outcome_table"], asset, horizon)
    matured_artifact = _subset(tables["matured_outcome_table"], asset, horizon)
    fwd_acc = _subset(tables["forward_accuracy_summary"], asset, horizon)
    fwd_prob = _subset(tables["forward_probability_calibration_summary"], asset, horizon)
    fwd_warnings = _subset(tables["forward_warning_table"], asset, horizon)
    ranked = _subset(tables["ranked_asset_horizon_plan"], asset, horizon)
    paper = _subset(tables["paper_trade_plan_table"], asset, horizon)
    watch = _subset(tables["watchlist_table"], asset, horizon)
    phase10_warn = _subset(tables["phase10_warnings_table"], asset, horizon)

    status = _status_series(fwd)
    pending = pending_artifact.copy()
    if pending.empty:
        pending = fwd[status.eq("Pending")].copy() if not fwd.empty and not status.empty else pd.DataFrame()
    matured = fwd[status.eq("Matured")].copy() if not fwd.empty and not status.empty else matured_artifact.copy()
    pending_target_date = _earliest_date(pending, "TargetOutcomeDate")

    raw_ready = raw.copy()
    if not raw_ready.empty and {"ProbabilityUp", "ActualDirection"}.issubset(raw_ready.columns):
        raw_ready = raw_ready[raw_ready["ProbabilityUp"].notna() & raw_ready["ActualDirection"].notna()].copy()

    brier = _safe_float(_first_value(prob, ["BrierScore"], np.nan), np.nan)
    if not np.isfinite(brier):
        brier = _safe_float(_first_value(fwd_prob, ["BrierScore"], np.nan), np.nan)
    matured_count = _safe_int(_first_value(fwd_acc, ["MaturedSignals", "MaturedOutcomes"], len(matured)), len(matured))
    pending_count = _safe_int(_first_value(fwd_acc, ["PendingSignals", "PendingOutcomes"], len(pending)), len(pending))
    win_rate = _safe_float(_first_value(fwd_acc, ["WinRate_%", "ForwardWinRate_%"], np.nan), np.nan)
    beat_rate = _safe_float(_first_value(fwd_acc, ["BeatBenchmarkRate_%"], np.nan), np.nan)
    if not np.isfinite(beat_rate) and not matured.empty and "BeatBenchmark" in matured.columns:
        vals = matured["BeatBenchmark"].map(lambda x: str(x).lower() in {"true", "1", "yes", "win"}).astype(float)
        beat_rate = float(vals.mean() * 100.0) if not vals.empty else np.nan
    forward_edge = _safe_float(_first_value(fwd_acc, ["AvgVsBuyHold_%", "MedianVsBuyHold_%", "ForwardBenchmarkEdge_%"], np.nan), np.nan)
    raw_edge = _mean_value(raw_ready, ["VsBuyHold", "VsBenchmark", "VsBenchmarkReturn"], np.nan)
    if np.isfinite(raw_edge) and abs(raw_edge) <= 2:
        raw_edge *= 100.0
    drawdown = _min_value(raw_ready, ["MaxDrawdownDuringTrade", "MaxDrawdown_%", "MaxDrawdown"], np.nan)
    if np.isfinite(drawdown) and abs(drawdown) <= 2:
        drawdown *= 100.0
    probability = _safe_float(_first_value(pending, ["ProbabilityUp", "PredictedProbabilityUp"], np.nan), np.nan)
    if not np.isfinite(probability):
        probability = _safe_float(_first_value(ranked, ["ProbabilityUp"], np.nan), np.nan)

    warnings = ["NotFinancialAdvice", "NotProductionReady"]
    for table in [prob, prob_warnings, fwd, pending, matured, fwd_acc, fwd_prob, fwd_warnings, ranked, paper, watch, phase10_warn]:
        if table.empty:
            continue
        for col in ["Warnings", "MainWarnings", "WarningType", "MainWarning", "CalibrationGrade", "CalibrationVerdict", "EvidenceVerdict"]:
            if col in table.columns:
                warnings.extend(_collect_warnings(*table[col].dropna().tolist()))
    text = " ".join(warnings).lower()
    if "probabilityunreliable" in text or "unreliable" in text:
        warnings.append("ProbabilityUnreliable")
    if "benchmarkdominated" in text or ("benchmark" in text and "dominated" in text):
        warnings.append("BenchmarkDominated")
    if "drawdown" in text or (np.isfinite(drawdown) and drawdown <= -12.0):
        warnings.append("DrawdownRisk")
    if "costfragile" in text or "returndestroyed" in text:
        warnings.append("CostFragile")
    if "overconfident" in text:
        warnings.append("Overconfident")
    if raw_ready.empty:
        warnings.append("LowTradeCount")
    if matured_count <= 0:
        warnings.append("NoMaturedForwardEvidenceYet")
    if pending_count > 0 and matured_count <= 0:
        warnings.append("PendingEvidenceOnly")
    return {
        "asset": asset,
        "horizon": int(horizon),
        "raw_trade_count": int(len(raw_ready)),
        "pending_count": int(pending_count),
        "matured_count": int(matured_count),
        "brier": brier,
        "win_rate": win_rate,
        "beat_rate": beat_rate,
        "forward_edge": forward_edge,
        "raw_edge": raw_edge,
        "drawdown": drawdown,
        "probability": probability,
        "research_action": str(_first_value(ranked, ["ResearchAction", "Decision"], "")),
        "phase10_capital_status": str(_first_value(ranked, ["CapitalDeploymentStatus"], "")),
        "required_evidence": str(_first_value(ranked, ["RequiredEvidenceToUpgrade"], "")),
        "review_date": str(_first_value(ranked, ["ReviewDate"], "")),
        "pending_target_date": pending_target_date,
        "has_forward_log": not fwd.empty or not pending_artifact.empty or not matured_artifact.empty,
        "warnings": _collect_warnings(warnings),
    }


def _capital_eligibility(metrics: Dict[str, Any], settings: Dict[str, Any], report_date: pd.Timestamp) -> Dict[str, Any]:
    thresholds = _mode_thresholds(settings["capital_eligibility_mode"])
    min_matured = int(settings["minimum_matured_forward_outcomes"])
    max_drawdown_allowed = float(settings["max_drawdown_allowed_pct"])
    max_cap = float(settings["max_real_capital_pct"])
    warnings = set(metrics["warnings"])
    severe = {
        "ProbabilityUnreliable",
        "TooFewTradesToCalibrate",
        "Overconfident",
        "BenchmarkDominated",
        "DrawdownRisk",
        "MissingArtifact",
        "EvidenceGateFailed",
        "NoMaturedForwardEvidenceYet",
        "LowTradeCount",
    }
    severe_count = len(warnings & severe)

    evidence_gate = metrics["raw_trade_count"] >= min_matured
    calibration_gate = np.isfinite(metrics["brier"]) and metrics["brier"] <= thresholds["brier"] and not (warnings & {"ProbabilityUnreliable", "TooFewTradesToCalibrate", "Overconfident"})
    forward_gate = metrics["has_forward_log"] and metrics["matured_count"] >= min_matured and np.isfinite(metrics["win_rate"]) and metrics["win_rate"] >= thresholds["win_rate"]
    benchmark_gate = not ("BenchmarkDominated" in warnings) and (
        (np.isfinite(metrics["beat_rate"]) and metrics["beat_rate"] >= thresholds["beat_rate"])
        or (np.isfinite(metrics["forward_edge"]) and metrics["forward_edge"] > 0)
        or (np.isfinite(metrics["raw_edge"]) and metrics["raw_edge"] > 0)
    )
    drawdown_gate = np.isfinite(metrics["drawdown"]) and abs(min(metrics["drawdown"], 0.0)) <= max_drawdown_allowed
    cost_gate = not bool(warnings & {"CostFragile", "ReturnDestroyed"})
    trade_count_gate = metrics["raw_trade_count"] >= min_matured and metrics["matured_count"] >= min_matured
    severe_gate = severe_count <= int(thresholds["severe_warnings"])

    gates = {
        "EvidenceGatePassed": bool(evidence_gate),
        "CalibrationGatePassed": bool(calibration_gate),
        "ForwardEvidenceGatePassed": bool(forward_gate),
        "BenchmarkGatePassed": bool(benchmark_gate),
        "DrawdownGatePassed": bool(drawdown_gate),
        "CostStressGatePassed": bool(cost_gate),
        "TradeCountGatePassed": bool(trade_count_gate),
        "SevereWarningGatePassed": bool(severe_gate),
    }
    failed = [name.replace("GatePassed", "") for name, passed in gates.items() if not passed]
    real_allowed = not failed
    stronger = (
        real_allowed
        and metrics["matured_count"] >= max(min_matured * 2, 20)
        and np.isfinite(metrics["win_rate"])
        and metrics["win_rate"] >= 60.0
        and np.isfinite(metrics["beat_rate"])
        and metrics["beat_rate"] >= 60.0
        and np.isfinite(metrics["brier"])
        and metrics["brier"] <= min(thresholds["brier"], 0.20)
        and np.isfinite(metrics["drawdown"])
        and abs(min(metrics["drawdown"], 0.0)) <= max_drawdown_allowed * 0.75
    )
    if real_allowed and stronger:
        status = "ConditionalResearchCapitalEligible"
        plan_type = "ConditionalResearchAllocation"
        max_allowed = min(max_cap, 2.0)
        max_loss = min(0.5, max_allowed / 2.0)
    elif real_allowed:
        status = "ConditionalMicroCapitalEligible"
        plan_type = "MicroCapitalTrial"
        max_allowed = min(max_cap, 0.5)
        max_loss = min(0.25, max_allowed / 2.0)
    elif metrics["raw_trade_count"] > 0 or metrics["pending_count"] > 0 or metrics["matured_count"] > 0 or np.isfinite(metrics["brier"]):
        status = "NotReady"
        plan_type = "PaperOnly"
        max_allowed = 0.0
        max_loss = 0.0
    else:
        status = "Blocked"
        plan_type = "NoRealCapital"
        max_allowed = 0.0
        max_loss = 0.0
    blocker = failed[0] if failed else "NoCapitalBlocker"
    if not real_allowed and "ProbabilityUnreliable" in warnings:
        blocker = "ProbabilityUnreliable"
    elif not real_allowed and "NoMaturedForwardEvidenceYet" in warnings:
        blocker = "NoMaturedForwardEvidenceYet"
    elif not real_allowed and "BenchmarkDominated" in warnings:
        blocker = "BenchmarkDominated"
    elif not real_allowed and "DrawdownRisk" in warnings:
        blocker = "DrawdownRisk"
    if failed:
        allow = "Pass failed gates: " + "; ".join(failed)
    else:
        allow = "Continue monitoring; eligible status can be downgraded if evidence weakens."
    row = {
        "Asset": metrics["asset"],
        "Horizon": metrics["horizon"],
        "RealCapitalAllowed": bool(real_allowed),
        "CapitalDeploymentStatus": status,
        "CapitalPlanType": plan_type,
        **gates,
        "MainCapitalBlocker": blocker,
        "WhatWouldAllowRealCapital": allow,
        "MaxAllowedRealCapitalPct": round(max_allowed, 4),
        "MaxLossPerIdeaPct": round(max_loss, 4),
        "ReviewDate": _review_date(report_date, metrics["horizon"]),
        "_FailedGates": failed,
    }
    return row


def _structured_plan(row: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    horizon = int(row["Horizon"])
    probability = metrics["probability"]
    threshold = max(0.55, min(0.70, probability if np.isfinite(probability) else 0.60))
    return {
        "Asset": row["Asset"],
        "Horizon": horizon,
        "CapitalPlanType": row["CapitalPlanType"],
        "MaxAllowedRealCapitalPct": row["MaxAllowedRealCapitalPct"],
        "MaxLossPerIdeaPct": row["MaxLossPerIdeaPct"],
        "PositionSizingRule": "Use the smaller of the capital cap and the max-loss-per-idea constraint.",
        "EntryCondition": f"Only if the research signal remains active and ProbabilityUp is at least {threshold:.2f}.",
        "ExitCondition": f"Exit or review after the {horizon}D target outcome date or if the signal is invalidated.",
        "StopOrInvalidationRule": "Stop after two consecutive paper losses, benchmark underperformance, or repeated drawdown warning.",
        "ScalingRule": "No scaling until additional matured forward outcomes preserve calibration, edge, and drawdown gates.",
        "DeRiskingRule": "Reduce to paper-only if calibration, benchmark, drawdown, cost, or trade-count gates fail.",
        "ReviewDate": row["ReviewDate"],
        "RequiredMonitoring": "Monitor probability calibration, matured outcomes, benchmark edge, drawdown, and warnings daily.",
        "RiskWarning": "Conditional research allocation only; not financial advice and not a return guarantee.",
    }


def _comparison_tables(current: pd.DataFrame, previous: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    if previous is None or previous.empty or not {"Asset", "Horizon"}.issubset(previous.columns):
        for _, row in current.iterrows():
            rows.append(
                {
                    "Asset": row["Asset"],
                    "Horizon": row["Horizon"],
                    "ComparisonStatus": "No prior comparison",
                    "PreviousCapitalDeploymentStatus": "",
                    "CurrentCapitalDeploymentStatus": row["CapitalDeploymentStatus"],
                    "PreviousMainBlocker": "",
                    "CurrentMainBlocker": row["MainCapitalBlocker"],
                    "Reason": "No prior Phase 11 capital eligibility artifact was available.",
                }
            )
        empty = pd.DataFrame(columns=list(COMPARISON_COLUMNS))
        return pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS)), empty
    score = {"Blocked": 0, "NotReady": 1, "ConditionalMicroCapitalEligible": 2, "ConditionalResearchCapitalEligible": 3}
    prev = previous.copy()
    prev["Horizon"] = pd.to_numeric(prev["Horizon"].astype(str).str.replace("D", "", regex=False), errors="coerce").astype("Int64")
    for _, row in current.iterrows():
        match = _subset(prev, str(row["Asset"]), int(row["Horizon"]))
        prev_status = str(_first_value(match, ["CapitalDeploymentStatus"], ""))
        curr_status = str(row["CapitalDeploymentStatus"])
        prev_blocker = str(_first_value(match, ["MainCapitalBlocker"], ""))
        curr_blocker = str(row["MainCapitalBlocker"])
        if not prev_status:
            status = "No prior comparison"
            reason = "No prior row for this asset-horizon."
        elif score.get(curr_status, 0) > score.get(prev_status, 0) or (prev_blocker and curr_blocker != prev_blocker):
            status = "Improved"
            reason = "Capital status improved or main blocker changed."
        elif score.get(curr_status, 0) < score.get(prev_status, 0):
            status = "Degraded"
            reason = "Capital status moved toward a more restrictive state."
        else:
            status = "Unchanged"
            reason = "No material eligibility change."
        rows.append(
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "ComparisonStatus": status,
                "PreviousCapitalDeploymentStatus": prev_status,
                "CurrentCapitalDeploymentStatus": curr_status,
                "PreviousMainBlocker": prev_blocker,
                "CurrentMainBlocker": curr_blocker,
                "Reason": reason,
            }
        )
    comparison = pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))
    improved = comparison[comparison["ComparisonStatus"].isin(["Improved", "No prior comparison"])].copy().reset_index(drop=True)
    degraded = comparison[comparison["ComparisonStatus"].eq("Degraded")].copy().reset_index(drop=True)
    return improved, degraded


def _build_warning_table(
    assets: List[str],
    horizons: List[int],
    metrics_rows: List[Dict[str, Any]],
    input_source_table: pd.DataFrame,
    capital_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {"Asset": "ALL", "Horizon": np.nan, "WarningType": "NotFinancialAdvice", "Severity": "High", "Message": "Research decision support only; no financial advice is provided."},
        {"Asset": "ALL", "Horizon": np.nan, "WarningType": "NotProductionReady", "Severity": "High", "Message": "This daily control center does not approve production or blind trading."},
        {"Asset": "ALL", "Horizon": np.nan, "WarningType": "RealCapitalConditionalOnly", "Severity": "High", "Message": "Any real-capital row must remain conditional, capped, monitored, and invalidation-driven."},
    ]
    if not input_source_table.empty:
        for _, row in input_source_table[input_source_table["Status"].astype(str).str.contains("Missing", na=False)].iterrows():
            rows.append(
                {
                    "Asset": "ALL",
                    "Horizon": np.nan,
                    "WarningType": "MissingArtifact",
                    "Severity": "Medium",
                    "Message": f"{row.get('Phase', '')} / {row.get('Artifact', '')} was not loaded.",
                }
            )
    if not any(bool(row["RealCapitalAllowed"]) for row in capital_rows):
        rows.append({"Asset": "ALL", "Horizon": np.nan, "WarningType": "RealCapitalBlocked", "Severity": "High", "Message": "No candidate passed every real-capital eligibility gate."})
    for metrics in metrics_rows:
        for warning in metrics["warnings"]:
            rows.append(
                {
                    "Asset": metrics["asset"],
                    "Horizon": metrics["horizon"],
                    "WarningType": warning,
                    "Severity": _warning_severity(warning),
                    "Message": f"{warning} affects the daily research-control decision.",
                }
            )
    return pd.DataFrame(rows, columns=list(WARNING_COLUMNS)).drop_duplicates().reset_index(drop=True)


def run_daily_research_control_center(
    *,
    true_raw_trade_log: Optional[pd.DataFrame] = None,
    probability_calibration_summary: Optional[pd.DataFrame] = None,
    probability_calibration_warnings: Optional[pd.DataFrame] = None,
    forward_signal_log: Optional[pd.DataFrame] = None,
    pending_outcome_table: Optional[pd.DataFrame] = None,
    matured_outcome_table: Optional[pd.DataFrame] = None,
    forward_accuracy_summary: Optional[pd.DataFrame] = None,
    forward_probability_calibration_summary: Optional[pd.DataFrame] = None,
    forward_warning_table: Optional[pd.DataFrame] = None,
    ranked_asset_horizon_plan: Optional[pd.DataFrame] = None,
    plan_card_table: Optional[pd.DataFrame] = None,
    paper_trade_plan_table: Optional[pd.DataFrame] = None,
    watchlist_table: Optional[pd.DataFrame] = None,
    risk_budget_table: Optional[pd.DataFrame] = None,
    phase10_warnings_table: Optional[pd.DataFrame] = None,
    previous_capital_eligibility_table: Optional[pd.DataFrame] = None,
    use_artifact_store: bool = False,
    prefer_uploaded: bool = False,
    uploaded_overrides: Optional[Dict[str, Any]] = None,
    report_date: Any = None,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    include_blocked_candidates: bool = True,
    capital_eligibility_mode: str = "Conservative",
    minimum_matured_forward_outcomes: int = 10,
    max_drawdown_allowed_pct: float = 12.0,
    max_real_capital_pct: float = 1.0,
    autosave: bool = False,
) -> DailyResearchControlCenterReport:
    """Build the Phase 11 daily research-control cockpit."""
    asset_list = list(assets or get_asset_names())
    horizon_list = [int(h) for h in (horizons or CONTROL_HORIZONS)]
    rd = _report_date(report_date)
    settings = {
        "phase": "11",
        "purpose": "daily_research_control_center",
        "report_date": str(rd.date()),
        "capital_eligibility_mode": capital_eligibility_mode,
        "minimum_matured_forward_outcomes": int(minimum_matured_forward_outcomes),
        "max_drawdown_allowed_pct": float(max_drawdown_allowed_pct),
        "max_real_capital_pct": float(max_real_capital_pct),
        "real_capital_eligibility_hardcoded_false": False,
        "production_ready_label_allowed": False,
        "guaranteed_return_language_allowed": False,
    }
    direct_inputs = {
        "true_raw_trade_log": true_raw_trade_log,
        "probability_calibration_summary": probability_calibration_summary,
        "probability_calibration_warnings": probability_calibration_warnings,
        "forward_signal_log": forward_signal_log,
        "pending_outcome_table": pending_outcome_table,
        "matured_outcome_table": matured_outcome_table,
        "forward_accuracy_summary": forward_accuracy_summary,
        "forward_probability_calibration_summary": forward_probability_calibration_summary,
        "forward_warning_table": forward_warning_table,
        "ranked_asset_horizon_plan": ranked_asset_horizon_plan,
        "plan_card_table": plan_card_table,
        "paper_trade_plan_table": paper_trade_plan_table,
        "watchlist_table": watchlist_table,
        "risk_budget_table": risk_budget_table,
        "phase10_warnings_table": phase10_warnings_table,
        "previous_capital_eligibility_table": previous_capital_eligibility_table,
    }
    tables, input_source_table = _resolve_inputs(use_artifact_store, prefer_uploaded, uploaded_overrides, direct_inputs)

    forward = tables["forward_signal_log"]
    status = _status_series(forward)
    active_paper = forward[status.eq("Pending")].copy() if not forward.empty and not status.empty else pd.DataFrame(columns=forward.columns)
    pending = tables["pending_outcome_table"].copy()
    if pending.empty:
        pending = active_paper.copy()
    matured_all = tables["matured_outcome_table"].copy()
    if matured_all.empty and not forward.empty and not status.empty:
        matured_all = forward[status.eq("Matured")].copy()
    if matured_all.empty:
        matured_today = pd.DataFrame(columns=matured_all.columns)
    else:
        actual_dates = _date_series(matured_all, "ActualOutcomeDate")
        if actual_dates.notna().any():
            matured_today = matured_all[actual_dates.dt.normalize().eq(rd)].copy()
        else:
            matured_today = matured_all[_date_series(matured_all, "TargetOutcomeDate").dt.normalize().eq(rd)].copy()
    overdue = pd.DataFrame(columns=pending.columns)
    if not pending.empty:
        pending_targets = _date_series(pending, "TargetOutcomeDate")
        overdue = pending[pending_targets.dt.normalize().lt(rd)].copy()
        if not overdue.empty:
            overdue["Warnings"] = overdue.get("Warnings", "").astype(str) + "; OutcomeOverdue"

    metrics_rows: List[Dict[str, Any]] = []
    capital_rows: List[Dict[str, Any]] = []
    structured_rows: List[Dict[str, Any]] = []
    health_rows: List[Dict[str, Any]] = []
    next_action_rows: List[Dict[str, Any]] = []
    for asset in asset_list:
        for horizon in horizon_list:
            metrics = _candidate_metrics(asset, int(horizon), tables)
            metrics_rows.append(metrics)
            cap = _capital_eligibility(metrics, settings, rd)
            capital_rows.append(cap)
            if cap["RealCapitalAllowed"]:
                structured_rows.append(_structured_plan(cap, metrics))
            action = metrics["research_action"] or ("PaperTradeOnly" if metrics["pending_count"] > 0 else "ObserveOnly")
            blocker = cap["MainCapitalBlocker"]
            health_rows.append(
                {
                    "Asset": asset,
                    "Horizon": int(horizon),
                    "PendingSignalCount": metrics["pending_count"],
                    "MaturedForwardCount": metrics["matured_count"],
                    "WarningCount": len(metrics["warnings"]),
                    "PaperTradeEligibility": action if action in {"PaperTradeOnly", "Watchlist"} else "NoActivePaperPlan",
                    "CapitalDeploymentStatus": cap["CapitalDeploymentStatus"],
                    "NextReviewDate": metrics["pending_target_date"] or metrics["review_date"] or cap["ReviewDate"],
                    "MainBlocker": blocker,
                    "NextEvidenceNeeded": metrics["required_evidence"] or cap["WhatWouldAllowRealCapital"],
                }
            )
            if cap["RealCapitalAllowed"]:
                next_action = "ReviewMicroCapitalEligibility"
                reason = "Strict capital gates passed; review conditional risk controls."
                priority = "High"
            elif metrics["pending_count"] > 0:
                next_action = "WaitForOutcome"
                reason = "Paper signal is pending and should be reviewed after target outcome date."
                priority = "High" if blocker == "NoMaturedForwardEvidenceYet" else "Medium"
            elif blocker in {"ProbabilityUnreliable", "EvidenceGate", "TradeCount"}:
                next_action = "RecheckCalibrationLater"
                reason = cap["WhatWouldAllowRealCapital"]
                priority = "Medium"
            elif cap["CapitalDeploymentStatus"] == "Blocked":
                next_action = "KeepBlocked"
                reason = cap["WhatWouldAllowRealCapital"]
                priority = "Medium"
            else:
                next_action = "ContinuePaperTracking" if action == "PaperTradeOnly" else "AddToWatchlist"
                reason = cap["WhatWouldAllowRealCapital"]
                priority = "Medium"
            action_review_date = metrics["pending_target_date"] if next_action == "WaitForOutcome" and metrics["pending_target_date"] else metrics["review_date"] or cap["ReviewDate"]
            next_action_rows.append(
                {
                    "Asset": asset,
                    "Horizon": int(horizon),
                    "DailyAction": next_action,
                    "Priority": priority,
                    "Reason": reason,
                    "ReviewDate": action_review_date,
                }
            )

    capital = pd.DataFrame(capital_rows)
    failed_cols = capital.pop("_FailedGates") if "_FailedGates" in capital.columns else pd.Series([[]] * len(capital))
    capital = capital[list(CAPITAL_ELIGIBILITY_COLUMNS)] if not capital.empty else pd.DataFrame(columns=list(CAPITAL_ELIGIBILITY_COLUMNS))
    blocker_rows = []
    for idx, row in capital.iterrows():
        if not bool(row["RealCapitalAllowed"]):
            failed = failed_cols.iloc[idx] if idx < len(failed_cols) else []
            blocker_rows.append(
                {
                    "Asset": row["Asset"],
                    "Horizon": row["Horizon"],
                    "CapitalDeploymentStatus": row["CapitalDeploymentStatus"],
                    "CapitalPlanType": row["CapitalPlanType"],
                    "MainCapitalBlocker": row["MainCapitalBlocker"],
                    "FailedGates": "; ".join(failed),
                    "WhatWouldAllowRealCapital": row["WhatWouldAllowRealCapital"],
                    "ResearchAction": _first_value(_subset(tables["ranked_asset_horizon_plan"], row["Asset"], int(row["Horizon"])), ["ResearchAction", "Decision"], ""),
                }
            )
    blockers = pd.DataFrame(blocker_rows, columns=list(CAPITAL_BLOCKER_COLUMNS))

    structured = pd.DataFrame(structured_rows, columns=list(STRUCTURED_CAPITAL_PLAN_COLUMNS))
    if not include_blocked_candidates:
        blockers = blockers[~blockers["CapitalDeploymentStatus"].eq("Blocked")].copy() if not blockers.empty else blockers

    ranked = tables["ranked_asset_horizon_plan"]
    paper_table = tables["paper_trade_plan_table"].copy()
    if paper_table.empty and not ranked.empty and "ResearchAction" in ranked.columns:
        paper_table = ranked[ranked["ResearchAction"].astype(str).eq("PaperTradeOnly")].copy()
    watch_table = tables["watchlist_table"].copy()
    if watch_table.empty and not ranked.empty and "ResearchAction" in ranked.columns:
        watch_table = ranked[ranked["ResearchAction"].astype(str).eq("Watchlist")].copy()
    avoid_table = ranked[ranked.get("ResearchAction", pd.Series(dtype=str)).astype(str).isin(["Avoid", "BlockedDueToDataFailure"])].copy() if not ranked.empty and "ResearchAction" in ranked.columns else pd.DataFrame()
    if not paper_table.empty:
        sort_cols = [c for c in ["ActionabilityScore", "OpportunityScore", "EvidenceScore"] if c in paper_table.columns]
        top_paper = paper_table.sort_values(sort_cols, ascending=False).head(10) if sort_cols else paper_table.head(10).copy()
    else:
        top_paper = pd.DataFrame(columns=paper_table.columns)
    watch_review = watch_table.copy()

    improved, degraded = _comparison_tables(capital, tables["previous_capital_eligibility_table"])
    evidence_health = pd.DataFrame(health_rows, columns=list(EVIDENCE_HEALTH_COLUMNS))
    next_actions = pd.DataFrame(next_action_rows, columns=list(DAILY_NEXT_ACTION_COLUMNS))
    warning_table = _build_warning_table(asset_list, horizon_list, metrics_rows, input_source_table, capital_rows)

    summary = pd.DataFrame(
        [
            {
                "ReportDate": str(rd.date()),
                "TotalActivePaperSignals": int(len(active_paper)),
                "PendingOutcomes": int(len(pending)),
                "MaturedOutcomes": int(len(matured_all)),
                "MaturedToday": int(len(matured_today)),
                "OverdueOutcomes": int(len(overdue)),
                "PaperTradeCandidates": int(len(top_paper)),
                "WatchlistCandidates": int(len(watch_review)),
                "RealCapitalEligibleCandidates": int(capital["RealCapitalAllowed"].astype(bool).sum()) if not capital.empty else 0,
                "AvoidCandidates": int(len(avoid_table)),
                "MainWarning": "RealCapitalConditionalOnly" if structured_rows else "RealCapitalBlocked",
                "DailyActionSummary": (
                    "Strict gates produced conditional capital plans with caps and invalidation rules."
                    if structured_rows
                    else "No real-capital candidate passed all gates; use paper-only, watchlist, pending-outcome, and blocker review actions."
                ),
            }
        ],
        columns=list(DAILY_RESEARCH_SUMMARY_COLUMNS),
    )

    report = DailyResearchControlCenterReport(
        daily_research_summary=summary,
        active_paper_signals_table=active_paper.reset_index(drop=True),
        pending_outcomes_table=pending.reset_index(drop=True),
        matured_today_table=matured_today.reset_index(drop=True),
        overdue_outcomes_table=overdue.reset_index(drop=True),
        top_paper_candidates_today=top_paper.reset_index(drop=True),
        watchlist_review_table=watch_review.reset_index(drop=True),
        capital_eligibility_table=capital.reset_index(drop=True),
        structured_capital_plan_table=structured.reset_index(drop=True),
        capital_blocker_table=blockers.reset_index(drop=True),
        degraded_candidates_table=degraded.reset_index(drop=True),
        improved_candidates_table=improved.reset_index(drop=True),
        blocked_or_avoid_table=avoid_table.reset_index(drop=True),
        evidence_health_table=evidence_health.reset_index(drop=True),
        daily_next_actions_table=next_actions.reset_index(drop=True),
        warning_table=warning_table.reset_index(drop=True),
        input_source_table=input_source_table,
        settings=settings,
    )
    if autosave:
        saved = save_phase_artifacts(
            "Phase 11 Daily Research Control Center",
            {
                "daily_research_summary": report.daily_research_summary,
                "active_paper_signals_table": report.active_paper_signals_table,
                "pending_outcomes_table": report.pending_outcomes_table,
                "matured_today_table": report.matured_today_table,
                "overdue_outcomes_table": report.overdue_outcomes_table,
                "top_paper_candidates_today": report.top_paper_candidates_today,
                "watchlist_review_table": report.watchlist_review_table,
                "capital_eligibility_table": report.capital_eligibility_table,
                "structured_capital_plan_table": report.structured_capital_plan_table,
                "capital_blocker_table": report.capital_blocker_table,
                "degraded_candidates_table": report.degraded_candidates_table,
                "improved_candidates_table": report.improved_candidates_table,
                "blocked_or_avoid_table": report.blocked_or_avoid_table,
                "evidence_health_table": report.evidence_health_table,
                "daily_next_actions_table": report.daily_next_actions_table,
                "warning_table": report.warning_table,
                "input_source_table": report.input_source_table,
            },
            inputs={},
            config=settings,
            warnings=warning_table["WarningType"].dropna().astype(str).unique().tolist() if not warning_table.empty else [],
        )
        report.saved_artifacts = saved
    return report


__all__ = [
    "DailyResearchControlCenterReport",
    "DAILY_RESEARCH_SUMMARY_COLUMNS",
    "CAPITAL_ELIGIBILITY_COLUMNS",
    "STRUCTURED_CAPITAL_PLAN_COLUMNS",
    "CAPITAL_BLOCKER_COLUMNS",
    "EVIDENCE_HEALTH_COLUMNS",
    "DAILY_NEXT_ACTION_COLUMNS",
    "WARNING_COLUMNS",
    "run_daily_research_control_center",
]
