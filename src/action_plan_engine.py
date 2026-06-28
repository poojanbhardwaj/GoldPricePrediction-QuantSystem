"""Phase 10 actionable research plan engine.

This module converts Phase 8 and Phase 9 research evidence into a structured
trader-assistance research plan. It never executes trades, promotes candidates,
or emits live-deployment language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names


PLAN_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)

EXECUTIVE_DECISION_COLUMNS: Tuple[str, ...] = (
    "DecisionArea",
    "Summary",
    "BestAsset",
    "BestHorizon",
    "TopDecision",
    "TopActionabilityScore",
    "Warning",
)

RANKED_PLAN_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "Asset",
    "Horizon",
    "Decision",
    "CapitalDeploymentStatus",
    "ResearchAction",
    "DeploymentAllowed",
    "RealCapitalRiskCap",
    "PaperTradeAllowed",
    "ReasonCapitalBlocked",
    "ReasonResearchAction",
    "EvidenceScore",
    "OpportunityScore",
    "RiskScore",
    "ActionabilityScore",
    "ConfidenceLabel",
    "CurrentSignalDirection",
    "ProbabilityUp",
    "SuggestedMode",
    "ResearchPlan",
    "EntryTrigger",
    "InvalidationRule",
    "RiskCap",
    "MaxPaperTradeSize",
    "ReviewDate",
    "WhyThisRank",
    "WhyNotDeploymentReady",
    "RequiredEvidenceToUpgrade",
    "MainWarnings",
)

PLAN_CARD_COLUMNS: Tuple[str, ...] = RANKED_PLAN_COLUMNS

ENTRY_TRIGGER_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "Decision",
    "EntryTrigger",
    "SignalDirection",
    "ProbabilityUp",
    "TriggerStatus",
    "Warnings",
)

INVALIDATION_RULE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "InvalidationRule",
    "ReviewDate",
    "Warnings",
)

RISK_BUDGET_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "Decision",
    "CapitalDeploymentStatus",
    "ResearchAction",
    "DeploymentAllowed",
    "RealCapitalRiskCap",
    "RiskCap",
    "MaxPaperTradeSize",
    "RiskReason",
)

EVIDENCE_SCORECARD_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "RawTradeCount",
    "ForwardPendingCount",
    "ForwardMaturedCount",
    "BrierScore",
    "CalibrationGrade",
    "ForwardWinRate_%",
    "ForwardBenchmarkEdge_%",
    "MaxDrawdown_%",
    "WarningCount",
    "EvidenceScore",
    "OpportunityScore",
    "RiskScore",
)

WATCHLIST_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "Decision",
    "CapitalDeploymentStatus",
    "ResearchAction",
    "PaperTradeAllowed",
    "ActionabilityScore",
    "WatchReason",
    "NextReviewTrigger",
    "MainWarnings",
)

NEXT_EVIDENCE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "EvidenceNeeded",
    "MinimumForwardMaturedRows",
    "CalibrationRequirement",
    "Priority",
)

WARNING_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "WarningType",
    "Severity",
    "Message",
)

SAFE_DECISIONS = {
    "Avoid",
    "Blocked",
    "Watchlist",
    "PaperTradeOnly",
    "ObserveOnly",
    "BlockedDueToDataFailure",
}

SAFE_CAPITAL_DEPLOYMENT_STATUSES = {
    "Blocked",
    "NotReady",
    "ResearchOnly",
}

SAFE_RESEARCH_ACTIONS = {
    "PaperTradeOnly",
    "Watchlist",
    "ObserveOnly",
    "Avoid",
    "BlockedDueToDataFailure",
}

SAFE_MODES = {
    "ObserveOnly",
    "PaperTradeOnly",
    "TinyResearchSimulation",
    "Blocked",
    "Avoid",
}


@dataclass
class ActionableResearchPlanReport:
    executive_decision_table: pd.DataFrame
    ranked_asset_horizon_plan: pd.DataFrame
    plan_card_table: pd.DataFrame
    entry_trigger_table: pd.DataFrame
    invalidation_rule_table: pd.DataFrame
    risk_budget_table: pd.DataFrame
    evidence_scorecard: pd.DataFrame
    blocked_candidates_table: pd.DataFrame
    watchlist_table: pd.DataFrame
    paper_trade_plan_table: pd.DataFrame
    next_evidence_needed_table: pd.DataFrame
    warnings_table: pd.DataFrame
    settings: Dict[str, Any] = field(default_factory=dict)


def _empty_report(settings: Optional[Dict[str, Any]] = None) -> ActionableResearchPlanReport:
    return ActionableResearchPlanReport(
        executive_decision_table=pd.DataFrame(columns=list(EXECUTIVE_DECISION_COLUMNS)),
        ranked_asset_horizon_plan=pd.DataFrame(columns=list(RANKED_PLAN_COLUMNS)),
        plan_card_table=pd.DataFrame(columns=list(PLAN_CARD_COLUMNS)),
        entry_trigger_table=pd.DataFrame(columns=list(ENTRY_TRIGGER_COLUMNS)),
        invalidation_rule_table=pd.DataFrame(columns=list(INVALIDATION_RULE_COLUMNS)),
        risk_budget_table=pd.DataFrame(columns=list(RISK_BUDGET_COLUMNS)),
        evidence_scorecard=pd.DataFrame(columns=list(EVIDENCE_SCORECARD_COLUMNS)),
        blocked_candidates_table=pd.DataFrame(columns=list(RANKED_PLAN_COLUMNS)),
        watchlist_table=pd.DataFrame(columns=list(WATCHLIST_COLUMNS)),
        paper_trade_plan_table=pd.DataFrame(columns=list(RANKED_PLAN_COLUMNS)),
        next_evidence_needed_table=pd.DataFrame(columns=list(NEXT_EVIDENCE_COLUMNS)),
        warnings_table=pd.DataFrame(columns=list(WARNING_COLUMNS)),
        settings=settings or {},
    )


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
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


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(np.clip(_safe_float(value, default=0.0), low, high))


def _normalise_horizon(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "Horizon" in out.columns:
        out["Horizon"] = out["Horizon"].astype(str).str.replace("D", "", regex=False)
        out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce").astype("Int64")
    return out


def _subset(table: pd.DataFrame, asset: str, horizon: int) -> pd.DataFrame:
    if table.empty or not {"Asset", "Horizon"}.issubset(table.columns):
        return pd.DataFrame()
    return table[
        table["Asset"].astype(str).eq(str(asset))
        & pd.to_numeric(table["Horizon"], errors="coerce").astype("Int64").eq(int(horizon))
    ].copy()


def _first_value(table: pd.DataFrame, names: Iterable[str], default: Any = np.nan) -> Any:
    if table.empty:
        return default
    for name in names:
        if name in table.columns and table[name].notna().any():
            return table[name].dropna().iloc[0]
    return default


def _mean_value(table: pd.DataFrame, names: Iterable[str], default: float = np.nan) -> float:
    if table.empty:
        return default
    for name in names:
        if name in table.columns:
            values = pd.to_numeric(table[name], errors="coerce").dropna()
            if not values.empty:
                return float(values.mean())
    return default


def _min_value(table: pd.DataFrame, names: Iterable[str], default: float = np.nan) -> float:
    if table.empty:
        return default
    for name in names:
        if name in table.columns:
            values = pd.to_numeric(table[name], errors="coerce").dropna()
            if not values.empty:
                return float(values.min())
    return default


def _collect_warning_tokens(*values: Any) -> List[str]:
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
            if token and token.lower() != "nan" and token not in warnings:
                warnings.append(token)
    return warnings


def _warning_severity(warning: str) -> str:
    high = {
        "ProbabilityUnreliable",
        "InsufficientForwardEvidence",
        "BenchmarkDominated",
        "DrawdownRisk",
        "LowTradeCount",
        "Overconfident",
        "AvoidRealCapital",
        "NoCandidateDeploymentReady",
    }
    return "High" if warning in high else "Medium"


def _warning_score(warnings: Iterable[str]) -> float:
    score = 0.0
    for warning in warnings:
        score += 12.0 if _warning_severity(warning) == "High" else 5.0
    return score


def _signal_strength_points(value: Any) -> float:
    text = str(value or "").strip().lower()
    if text == "high":
        return 10.0
    if text == "medium":
        return 6.0
    if text == "low":
        return 2.0
    return 0.0


def _confidence_label(actionability: float, decision: str) -> str:
    if decision in {"Avoid", "BlockedDueToDataFailure"}:
        return "Blocked"
    if decision == "ObserveOnly":
        return "Low research confidence"
    if actionability >= 70:
        return "Higher research confidence"
    if actionability >= 50:
        return "Moderate research confidence"
    return "Low research confidence"


def _review_date(days: int) -> str:
    return str((pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=int(days))).date())


def _build_candidate_metrics(
    *,
    asset: str,
    horizon: int,
    probability_calibration_summary: pd.DataFrame,
    probability_calibration_warnings: pd.DataFrame,
    probability_calibration_recommendations: pd.DataFrame,
    probability_calibration_bins: pd.DataFrame,
    true_raw_trade_log: pd.DataFrame,
    forward_signal_log: pd.DataFrame,
    forward_accuracy_summary: pd.DataFrame,
    forward_probability_calibration_summary: pd.DataFrame,
    forward_warning_table: pd.DataFrame,
    latest_model_predictions: pd.DataFrame,
) -> Dict[str, Any]:
    prob = _subset(probability_calibration_summary, asset, horizon)
    prob_warn = _subset(probability_calibration_warnings, asset, horizon)
    prob_rec = _subset(probability_calibration_recommendations, asset, horizon)
    prob_bins = _subset(probability_calibration_bins, asset, horizon)
    raw = _subset(true_raw_trade_log, asset, horizon)
    fwd = _subset(forward_signal_log, asset, horizon)
    fwd_acc = _subset(forward_accuracy_summary, asset, horizon)
    fwd_prob = _subset(forward_probability_calibration_summary, asset, horizon)
    fwd_warn = _subset(forward_warning_table, asset, horizon)
    latest_pred = _subset(latest_model_predictions, asset, horizon)

    pending = fwd[fwd.get("Status", pd.Series(dtype=str)).astype(str).eq("Pending")].copy() if not fwd.empty and "Status" in fwd.columns else pd.DataFrame()
    matured = fwd[fwd.get("Status", pd.Series(dtype=str)).astype(str).eq("Matured")].copy() if not fwd.empty and "Status" in fwd.columns else pd.DataFrame()
    latest_pending = pending.sort_values("SignalDate").tail(1) if not pending.empty and "SignalDate" in pending.columns else pd.DataFrame()
    if latest_pending.empty and not latest_pred.empty:
        latest_pending = latest_pred.tail(1)

    raw_ready = raw.copy()
    if not raw_ready.empty:
        if {"ProbabilityUp", "ActualDirection"}.issubset(raw_ready.columns):
            raw_ready = raw_ready[raw_ready["ProbabilityUp"].notna() & raw_ready["ActualDirection"].notna()].copy()

    brier = _safe_float(_first_value(prob, ["BrierScore"], np.nan), np.nan)
    if not np.isfinite(brier):
        brier = _safe_float(_first_value(fwd_prob, ["BrierScore"], np.nan), np.nan)
    calibration_grade = str(_first_value(prob, ["CalibrationGrade", "CalibrationVerdict"], "") or "")
    calibration_verdict = str(_first_value(fwd_prob, ["CalibrationVerdict"], "") or "")
    raw_available = bool(_first_value(prob, ["RawProbabilityOutcomesAvailable"], False))
    useful_filter = bool(_first_value(prob, ["UsefulProbabilityFilterFound", "ConfidenceUseful"], False))
    raw_trade_count = int(len(raw_ready))
    raw_return = _mean_value(raw_ready, ["RealizedReturn", "StrategyReturnAfterCost"], np.nan)
    raw_edge = _mean_value(raw_ready, ["VsBuyHold", "VsBenchmark", "VsBenchmarkReturn"], np.nan)
    raw_drawdown = _min_value(raw_ready, ["MaxDrawdownDuringTrade", "MaxDrawdown_%", "MaxDrawdown"], np.nan)

    forward_pending = int(_safe_float(_first_value(fwd_acc, ["PendingSignals"], len(pending)), len(pending)))
    forward_matured = int(_safe_float(_first_value(fwd_acc, ["MaturedSignals"], len(matured)), len(matured)))
    forward_win_rate = _safe_float(_first_value(fwd_acc, ["WinRate_%"], np.nan), np.nan)
    forward_edge_pct = _safe_float(_first_value(fwd_acc, ["AvgVsBuyHold_%", "MedianVsBuyHold_%"], np.nan), np.nan)
    forward_return_pct = _safe_float(_first_value(fwd_acc, ["AvgRealizedReturn_%", "MedianRealizedReturn_%"], np.nan), np.nan)
    worst_return_pct = _safe_float(_first_value(fwd_acc, ["WorstRealizedReturn_%"], np.nan), np.nan)
    fwd_brier = _safe_float(_first_value(fwd_prob, ["BrierScore"], np.nan), np.nan)

    probability = _safe_float(_first_value(latest_pending, ["ProbabilityUp", "PredictedProbabilityUp"], np.nan), np.nan)
    direction = str(_first_value(latest_pending, ["PredictedDirection", "Signal"], "") or "")
    strength = str(_first_value(latest_pending, ["SignalStrength"], "") or "")
    if not direction and np.isfinite(probability):
        direction = "Up" if probability >= 0.5 else "Down"
    if not strength and np.isfinite(probability):
        distance = abs(probability - 0.5)
        strength = "High" if distance >= 0.20 else "Medium" if distance >= 0.10 else "Low"

    warning_tokens: List[str] = [
        "NotFinancialAdvice",
        "NotProductionReady",
        "AvoidRealCapital",
    ]
    for table in [prob, prob_warn, prob_rec, prob_bins, raw, fwd, fwd_acc, fwd_prob, fwd_warn]:
        if not table.empty:
            for col in ["Warnings", "MainWarning", "WarningType", "CalibrationGrade", "CalibrationVerdict", "EvidenceVerdict"]:
                if col in table.columns:
                    warning_tokens.extend(_collect_warning_tokens(*table[col].dropna().tolist()))

    text_blob = " ".join(warning_tokens).lower()
    if "unreliable" in text_blob or "overconfident" in text_blob or calibration_grade in {"Overconfident", "ProbabilityUnreliable"} or calibration_verdict == "ProbabilityStillUnreliable":
        warning_tokens.append("ProbabilityUnreliable")
    if "overconfident" in text_blob or calibration_grade == "Overconfident":
        warning_tokens.append("Overconfident")
    if "benchmark" in text_blob and ("dominated" in text_blob or "stilldominates" in text_blob):
        warning_tokens.append("BenchmarkDominated")
    if forward_edge_pct < 0 or (np.isfinite(raw_edge) and raw_edge < 0):
        warning_tokens.append("BenchmarkDominated")
    if "drawdown" in text_blob or (np.isfinite(raw_drawdown) and raw_drawdown <= -0.15) or (np.isfinite(worst_return_pct) and worst_return_pct <= -15.0):
        warning_tokens.append("DrawdownRisk")
    if raw_trade_count < 10 and forward_matured < 10:
        warning_tokens.append("LowTradeCount")
    if forward_matured < 5:
        warning_tokens.append("InsufficientForwardEvidence")
    if forward_pending > 0 and forward_matured < 5:
        warning_tokens.append("PendingEvidenceOnly")
    if "paper" in text_blob or forward_pending > 0:
        warning_tokens.append("PaperOnly")

    warning_tokens = _collect_warning_tokens(warning_tokens)
    return {
        "asset": asset,
        "horizon": int(horizon),
        "raw_available": raw_available,
        "useful_filter": useful_filter,
        "raw_trade_count": raw_trade_count,
        "raw_return": raw_return,
        "raw_edge": raw_edge,
        "raw_drawdown": raw_drawdown,
        "brier": brier,
        "fwd_brier": fwd_brier,
        "calibration_grade": calibration_grade,
        "calibration_verdict": calibration_verdict,
        "forward_pending": forward_pending,
        "forward_matured": forward_matured,
        "forward_win_rate": forward_win_rate,
        "forward_edge_pct": forward_edge_pct,
        "forward_return_pct": forward_return_pct,
        "worst_return_pct": worst_return_pct,
        "probability": probability,
        "direction": direction,
        "strength": strength,
        "warnings": warning_tokens,
    }


def _score_candidate(metrics: Dict[str, Any], risk_appetite: str) -> Dict[str, float]:
    warnings = metrics["warnings"]
    raw_count = metrics["raw_trade_count"]
    matured = metrics["forward_matured"]
    pending = metrics["forward_pending"]
    brier = metrics["brier"]
    fwd_brier = metrics["fwd_brier"]
    grade = metrics["calibration_grade"]

    evidence = 0.0
    if metrics["raw_available"]:
        evidence += 12.0
    evidence += min(raw_count * 1.5, 18.0)
    evidence += min(matured * 4.0, 24.0)
    evidence += min(pending * 1.5, 8.0)
    if np.isfinite(brier):
        evidence += max(0.0, (0.35 - brier) / 0.35 * 18.0)
    if np.isfinite(fwd_brier):
        evidence += max(0.0, (0.35 - fwd_brier) / 0.35 * 10.0)
    if grade in {"WellCalibrated", "UsefulButNoisy"}:
        evidence += 12.0
    elif grade in {"Overconfident", "ProbabilityUnreliable"}:
        evidence -= 10.0
    if metrics["useful_filter"]:
        evidence += 6.0
    if np.isfinite(metrics["forward_win_rate"]) and matured > 0:
        evidence += np.clip((metrics["forward_win_rate"] - 45.0) / 20.0, 0.0, 1.0) * 10.0
    evidence -= min(_warning_score(warnings) * 0.35, 22.0)

    probability = metrics["probability"]
    opportunity = 0.0
    if np.isfinite(probability):
        opportunity += min(abs(probability - 0.5) * 180.0, 30.0)
    opportunity += _signal_strength_points(metrics["strength"])
    if pending > 0:
        opportunity += 12.0
    raw_return_pct = metrics["raw_return"] * 100.0 if np.isfinite(metrics["raw_return"]) and abs(metrics["raw_return"]) <= 2 else metrics["raw_return"]
    raw_edge_pct = metrics["raw_edge"] * 100.0 if np.isfinite(metrics["raw_edge"]) and abs(metrics["raw_edge"]) <= 2 else metrics["raw_edge"]
    if np.isfinite(raw_return_pct):
        opportunity += np.clip(raw_return_pct, -5.0, 8.0) * 1.4
    if np.isfinite(raw_edge_pct):
        opportunity += np.clip(raw_edge_pct, -5.0, 8.0) * 1.8
    if np.isfinite(metrics["forward_return_pct"]):
        opportunity += np.clip(metrics["forward_return_pct"], -8.0, 10.0) * 1.4
    if np.isfinite(metrics["forward_edge_pct"]):
        opportunity += np.clip(metrics["forward_edge_pct"], -8.0, 10.0) * 2.0
    if np.isfinite(metrics["forward_win_rate"]) and matured:
        opportunity += np.clip((metrics["forward_win_rate"] - 50.0) / 25.0, -1.0, 1.0) * 12.0

    risk = 18.0
    risk += _warning_score(warnings)
    if raw_count < 5:
        risk += 18.0
    elif raw_count < 10:
        risk += 10.0
    if matured < 5:
        risk += 16.0
    elif matured < 10:
        risk += 8.0
    if np.isfinite(brier) and brier > 0.30:
        risk += 14.0
    if np.isfinite(fwd_brier) and fwd_brier > 0.30:
        risk += 12.0
    drawdown_pct = metrics["raw_drawdown"] * 100.0 if np.isfinite(metrics["raw_drawdown"]) and abs(metrics["raw_drawdown"]) <= 2 else metrics["raw_drawdown"]
    if np.isfinite(drawdown_pct):
        risk += max(0.0, abs(min(drawdown_pct, 0.0)) - 8.0) * 1.2
    if np.isfinite(metrics["worst_return_pct"]):
        risk += max(0.0, abs(min(metrics["worst_return_pct"], 0.0)) - 8.0) * 1.1

    mode = str(risk_appetite or "Conservative").lower()
    risk_tolerance = 0.0
    if "aggressive" in mode:
        risk_tolerance = 10.0
    elif "balanced" in mode:
        risk_tolerance = 5.0
    risk = max(0.0, risk - risk_tolerance)

    evidence = _clip(evidence)
    opportunity = _clip(opportunity)
    risk = _clip(risk)
    actionability = _clip(evidence * 0.42 + opportunity * 0.33 + (100.0 - risk) * 0.25)
    if "ProbabilityUnreliable" in warnings:
        actionability = min(actionability, 58.0)
    if "LowTradeCount" in warnings or "InsufficientForwardEvidence" in warnings:
        actionability = min(actionability, 52.0)
    if "BenchmarkDominated" in warnings:
        actionability = min(actionability, 48.0)
    if "DrawdownRisk" in warnings:
        actionability = min(actionability, 45.0)
    return {
        "EvidenceScore": round(evidence, 2),
        "OpportunityScore": round(opportunity, 2),
        "RiskScore": round(risk, 2),
        "ActionabilityScore": round(actionability, 2),
    }


def _has_usable_research_data(metrics: Dict[str, Any]) -> bool:
    return bool(
        np.isfinite(metrics.get("probability", np.nan))
        or int(metrics.get("forward_pending", 0)) > 0
        or int(metrics.get("forward_matured", 0)) > 0
        or int(metrics.get("raw_trade_count", 0)) > 0
        or bool(metrics.get("raw_available", False))
        or np.isfinite(metrics.get("brier", np.nan))
        or np.isfinite(metrics.get("fwd_brier", np.nan))
    )


def _capital_deployment_status(metrics: Dict[str, Any], scores: Dict[str, float]) -> str:
    return "Blocked"


def _research_action(metrics: Dict[str, Any], scores: Dict[str, float], minimum_evidence_score: float) -> str:
    warnings = set(metrics["warnings"])
    evidence = scores["EvidenceScore"]
    opportunity = scores["OpportunityScore"]
    risk = scores["RiskScore"]
    actionability = scores["ActionabilityScore"]
    min_evidence = float(minimum_evidence_score)
    has_data = _has_usable_research_data(metrics)
    has_probability = np.isfinite(metrics.get("probability", np.nan))
    has_pending = int(metrics.get("forward_pending", 0)) >= 1

    if not has_data:
        return "BlockedDueToDataFailure"
    if opportunity < 40.0 and evidence < 20.0:
        return "Avoid"
    if "BenchmarkDominated" in warnings and risk >= 85.0 and opportunity < 55.0:
        return "Avoid"
    if "DrawdownRisk" in warnings and risk >= 85.0 and opportunity < 60.0:
        return "Avoid"

    if has_pending and has_probability:
        if opportunity >= 70.0 and evidence >= max(35.0, min_evidence):
            return "PaperTradeOnly"
        if opportunity >= 60.0 and evidence >= 20.0:
            return "PaperTradeOnly"
        if opportunity >= 55.0:
            return "Watchlist"
        if opportunity >= 40.0 or actionability >= 42.0:
            return "Watchlist"

    if has_probability and opportunity >= 55.0 and evidence >= 25.0:
        return "Watchlist"
    if actionability >= 45.0 and opportunity >= 45.0:
        return "Watchlist"
    if evidence < min_evidence or "InsufficientForwardEvidence" in warnings or "LowTradeCount" in warnings:
        return "ObserveOnly"
    if opportunity < 40.0:
        return "ObserveOnly"
    return "Watchlist"


def _suggested_mode(research_action: str) -> str:
    if research_action == "PaperTradeOnly":
        return "PaperTradeOnly"
    if research_action == "Watchlist":
        return "ObserveOnly"
    if research_action == "ObserveOnly":
        return "ObserveOnly"
    if research_action == "Avoid":
        return "Avoid"
    return "Blocked"


def _decide(metrics: Dict[str, Any], scores: Dict[str, float], risk_appetite: str, minimum_evidence_score: float) -> Tuple[str, str, str]:
    capital_status = _capital_deployment_status(metrics, scores)
    research_action = _research_action(metrics, scores, minimum_evidence_score)
    return capital_status, research_action, _suggested_mode(research_action)


def _main_warning(warnings: List[str]) -> str:
    priority = [
        "DrawdownRisk",
        "BenchmarkDominated",
        "ProbabilityUnreliable",
        "Overconfident",
        "InsufficientForwardEvidence",
        "LowTradeCount",
        "PendingEvidenceOnly",
        "PaperOnly",
    ]
    for warning in priority:
        if warning in warnings:
            return warning
    return warnings[0] if warnings else "NotFinancialAdvice"


def _entry_trigger(metrics: Dict[str, Any], decision: str) -> str:
    probability = metrics["probability"]
    if decision == "PaperTradeOnly":
        threshold = max(0.55, min(0.70, probability if np.isfinite(probability) else 0.60))
        direction = metrics["direction"] or "model direction"
        return f"Record paper-only signal when ProbabilityUp is at least {threshold:.2f}, direction remains {direction}, and no new severe warning appears."
    if decision == "Watchlist":
        return "No paper entry yet; wait for forward evidence to mature or warnings to reduce."
    if decision == "ObserveOnly":
        return "Observe only; no paper entry trigger until probability and forward evidence improve."
    if decision == "Avoid":
        return "No paper entry; archive unless evidence quality improves."
    if decision == "BlockedDueToDataFailure":
        return "No paper entry; probability, signal, or date evidence is missing."
    threshold = max(0.55, min(0.70, probability if np.isfinite(probability) else 0.60))
    direction = metrics["direction"] or "model direction"
    return f"Record paper-only signal when ProbabilityUp is at least {threshold:.2f}, direction remains {direction}, and no new severe warning appears."


def _invalidation_rule(metrics: Dict[str, Any]) -> str:
    needed = max(10, metrics["forward_matured"] + 5)
    parts = [
        "Invalidate research rank after two consecutive forward paper losses",
        "or if forward benchmark edge stays negative",
        f"or if matured forward rows remain below {needed}",
    ]
    if "DrawdownRisk" in metrics["warnings"]:
        parts.append("or if drawdown warning repeats")
    return " ".join(parts) + "."


def _required_evidence(metrics: Dict[str, Any]) -> str:
    needs: List[str] = []
    if metrics["forward_matured"] < 10:
        needs.append("at least 10 matured forward paper outcomes")
    if "ProbabilityUnreliable" in metrics["warnings"]:
        needs.append("better probability calibration with lower Brier/ECE")
    if "BenchmarkDominated" in metrics["warnings"]:
        needs.append("positive forward edge versus benchmark")
    if "DrawdownRisk" in metrics["warnings"]:
        needs.append("controlled drawdown in paper evidence")
    if not needs:
        needs.append("continued stable forward paper evidence")
    return "; ".join(needs)


def _risk_cap(decision: str) -> Tuple[str, str, bool]:
    if decision == "PaperTradeOnly":
        return "0 real capital", "1 paper unit", True
    if decision == "Watchlist":
        return "0 real capital", "observe only", False
    if decision == "ObserveOnly":
        return "0 real capital", "0 paper units; observation only", False
    return "0 real capital", "0 paper units", False


def _plan_text(asset: str, horizon: int, decision: str, metrics: Dict[str, Any]) -> str:
    if decision == "PaperTradeOnly":
        return (
            f"Track {asset} {horizon}D as paper-only. Record entry from the pending Phase 9 signal. "
            "Review after target outcome date. Invalidate after two consecutive paper losses, "
            "benchmark underperformance, or repeated drawdown warning."
        )
    if decision == "Watchlist":
        return f"Do not paper-enter {asset} {horizon}D yet. Watch until forward evidence matures or warnings reduce."
    if decision == "ObserveOnly":
        return f"Observe {asset} {horizon}D as weak evidence. Re-score after more forward paper outcomes mature."
    if decision == "BlockedDueToDataFailure":
        return f"Fix missing probability, signal, or date evidence for {asset} {horizon}D before research tracking."
    return f"Archive {asset} {horizon}D unless evidence quality improves."


def _why_rank(metrics: Dict[str, Any], scores: Dict[str, float]) -> str:
    return (
        f"Evidence {scores['EvidenceScore']:.1f}, opportunity {scores['OpportunityScore']:.1f}, "
        f"risk {scores['RiskScore']:.1f}; pending {metrics['forward_pending']} and matured {metrics['forward_matured']}."
    )


def _deployment_reason(metrics: Dict[str, Any]) -> str:
    if not _has_usable_research_data(metrics):
        return "Real capital is blocked because usable probability, signal, or date evidence is missing."
    if metrics["forward_matured"] < 10:
        return "Real capital is blocked because forward paper outcomes are still too few."
    if "ProbabilityUnreliable" in metrics["warnings"]:
        return "Real capital is blocked because probability evidence is not reliable enough."
    if "BenchmarkDominated" in metrics["warnings"]:
        return "Real capital is blocked because benchmark comparison is not strong enough."
    if "DrawdownRisk" in metrics["warnings"]:
        return "Real capital is blocked because drawdown evidence is too risky."
    return "Real capital is blocked; this remains research-only until sustained forward evidence is collected."


def _research_action_reason(metrics: Dict[str, Any], scores: Dict[str, float], decision: str) -> str:
    if decision == "PaperTradeOnly":
        return (
            "Paper-only tracking is allowed because opportunity is high enough, a current probability exists, "
            "and at least one forward paper signal is pending."
        )
    if decision == "Watchlist":
        return "Moderate research opportunity exists, but evidence or risk is not strong enough for a paper entry."
    if decision == "ObserveOnly":
        return "Evidence exists, but opportunity or reliability is too weak for active paper tracking."
    if decision == "BlockedDueToDataFailure":
        return "No usable probability, signal, or date evidence exists for research tracking."
    return "Opportunity or evidence quality is too weak; keep visible as rejected research evidence."


def _candidate_row(
    rank: int,
    metrics: Dict[str, Any],
    scores: Dict[str, float],
    capital_status: str,
    research_action: str,
    suggested_mode: str,
) -> Dict[str, Any]:
    asset = metrics["asset"]
    horizon = metrics["horizon"]
    risk_cap, max_size, paper_allowed = _risk_cap(research_action)
    warnings = metrics["warnings"]
    return {
        "Rank": int(rank),
        "Asset": asset,
        "Horizon": int(horizon),
        "Decision": research_action,
        "CapitalDeploymentStatus": capital_status,
        "ResearchAction": research_action,
        "DeploymentAllowed": False,
        "RealCapitalRiskCap": 0,
        "PaperTradeAllowed": bool(paper_allowed),
        "ReasonCapitalBlocked": _deployment_reason(metrics),
        "ReasonResearchAction": _research_action_reason(metrics, scores, research_action),
        "EvidenceScore": scores["EvidenceScore"],
        "OpportunityScore": scores["OpportunityScore"],
        "RiskScore": scores["RiskScore"],
        "ActionabilityScore": scores["ActionabilityScore"],
        "ConfidenceLabel": _confidence_label(scores["ActionabilityScore"], research_action),
        "CurrentSignalDirection": metrics["direction"] or "No current signal",
        "ProbabilityUp": round(metrics["probability"], 6) if np.isfinite(metrics["probability"]) else np.nan,
        "SuggestedMode": suggested_mode,
        "ResearchPlan": _plan_text(asset, horizon, research_action, metrics),
        "EntryTrigger": _entry_trigger(metrics, research_action),
        "InvalidationRule": _invalidation_rule(metrics),
        "RiskCap": risk_cap,
        "MaxPaperTradeSize": max_size,
        "ReviewDate": _review_date(max(int(horizon), 5)),
        "WhyThisRank": _why_rank(metrics, scores),
        "WhyNotDeploymentReady": _deployment_reason(metrics),
        "RequiredEvidenceToUpgrade": _required_evidence(metrics),
        "MainWarnings": "; ".join(warnings),
    }


def _executive_table(ranked: pd.DataFrame, warnings_table: pd.DataFrame) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame(
            [
                {
                    "DecisionArea": "Overall",
                    "Summary": "No real-capital deployment is allowed; research tracking can start once usable evidence appears.",
                    "BestAsset": "",
                    "BestHorizon": np.nan,
                    "TopDecision": "BlockedDueToDataFailure",
                    "TopActionabilityScore": np.nan,
                    "Warning": "NoCandidateDeploymentReady",
                }
            ],
            columns=list(EXECUTIVE_DECISION_COLUMNS),
        )
    top = ranked.iloc[0]
    paper_count = int(ranked["ResearchAction"].eq("PaperTradeOnly").sum()) if "ResearchAction" in ranked.columns else 0
    watch_count = int(ranked["ResearchAction"].eq("Watchlist").sum()) if "ResearchAction" in ranked.columns else 0
    warning = "NoCandidateDeploymentReady"
    summary = (
        f"No real-capital deployment is allowed. Research plan includes {paper_count} paper-only "
        f"candidate(s) and {watch_count} watchlist candidate(s)."
    )
    rows = [
        {
            "DecisionArea": "Overall",
            "Summary": summary,
            "BestAsset": top["Asset"],
            "BestHorizon": int(top["Horizon"]),
            "TopDecision": top["Decision"],
            "TopActionabilityScore": top["ActionabilityScore"],
            "Warning": warning or _main_warning(str(top.get("MainWarnings", "")).split(";")),
        },
        {
            "DecisionArea": "Coverage",
            "Summary": f"{len(ranked)} asset-horizon combinations scored; failed and blocked rows remain visible.",
            "BestAsset": top["Asset"],
            "BestHorizon": int(top["Horizon"]),
            "TopDecision": top["Decision"],
            "TopActionabilityScore": top["ActionabilityScore"],
            "Warning": "NotFinancialAdvice; NotProductionReady",
        },
    ]
    if not warnings_table.empty and warnings_table["WarningType"].eq("NoCandidateDeploymentReady").any():
        rows.append(
            {
                "DecisionArea": "DeploymentGate",
                "Summary": "No candidate is cleared for real-capital deployment; use paper-ledger and watchlist actions only.",
                "BestAsset": top["Asset"],
                "BestHorizon": int(top["Horizon"]),
                "TopDecision": top["Decision"],
                "TopActionabilityScore": top["ActionabilityScore"],
                "Warning": "NoCandidateDeploymentReady",
            }
        )
    return pd.DataFrame(rows, columns=list(EXECUTIVE_DECISION_COLUMNS))


def _warnings_table(candidate_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {
            "Asset": "ALL",
            "Horizon": np.nan,
            "WarningType": "NotFinancialAdvice",
            "Severity": "High",
            "Message": "Research decision support only; no financial advice is provided.",
        },
        {
            "Asset": "ALL",
            "Horizon": np.nan,
            "WarningType": "NotProductionReady",
            "Severity": "High",
            "Message": "Research evidence is not a live-deployment approval.",
        },
        {
            "Asset": "ALL",
            "Horizon": np.nan,
            "WarningType": "AvoidRealCapital",
            "Severity": "High",
            "Message": "Use paper tracking and research review only.",
        },
    ]
    for row in candidate_rows:
        for warning in _collect_warning_tokens(row.get("MainWarnings", "")):
            rows.append(
                {
                    "Asset": row.get("Asset", ""),
                    "Horizon": row.get("Horizon", np.nan),
                    "WarningType": warning,
                    "Severity": _warning_severity(warning),
                    "Message": f"{warning} affects the research plan for this asset-horizon.",
                }
            )
    rows.append(
        {
            "Asset": "ALL",
            "Horizon": np.nan,
            "WarningType": "NoCandidateDeploymentReady",
            "Severity": "High",
            "Message": "No candidate clears the evidence gates for real-capital deployment.",
        }
    )
    return pd.DataFrame(rows, columns=list(WARNING_COLUMNS)).drop_duplicates().reset_index(drop=True)


def run_actionable_research_plan(
    *,
    probability_calibration_summary: Optional[pd.DataFrame] = None,
    probability_calibration_warnings: Optional[pd.DataFrame] = None,
    probability_calibration_recommendations: Optional[pd.DataFrame] = None,
    probability_calibration_bins: Optional[pd.DataFrame] = None,
    true_raw_trade_log: Optional[pd.DataFrame] = None,
    forward_signal_log: Optional[pd.DataFrame] = None,
    forward_accuracy_summary: Optional[pd.DataFrame] = None,
    forward_probability_calibration_summary: Optional[pd.DataFrame] = None,
    forward_warning_table: Optional[pd.DataFrame] = None,
    forward_next_research_actions: Optional[pd.DataFrame] = None,
    forward_evidence_coverage: Optional[pd.DataFrame] = None,
    latest_model_predictions: Optional[pd.DataFrame] = None,
    current_pending_forward_signals: Optional[pd.DataFrame] = None,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    risk_appetite: str = "Conservative",
    minimum_evidence_score: float = 35.0,
    include_blocked_candidates: bool = True,
    top_n_plan_cards: int = 10,
) -> ActionableResearchPlanReport:
    """Build a Phase 10 research-only action plan from Phase 8/9 evidence."""
    asset_list = list(assets or get_asset_names())
    horizon_list = [int(h) for h in (horizons or PLAN_HORIZONS)]
    settings = {
        "phase": "10",
        "purpose": "actionable_research_plan_only",
        "risk_appetite": risk_appetite,
        "minimum_evidence_score": float(minimum_evidence_score),
        "include_blocked_candidates": bool(include_blocked_candidates),
        "production_ready_label_allowed": False,
        "real_money_execution_allowed": False,
    }
    tables = {
        "probability_calibration_summary": _normalise_horizon(probability_calibration_summary),
        "probability_calibration_warnings": _normalise_horizon(probability_calibration_warnings),
        "probability_calibration_recommendations": _normalise_horizon(probability_calibration_recommendations),
        "probability_calibration_bins": _normalise_horizon(probability_calibration_bins),
        "true_raw_trade_log": _normalise_horizon(true_raw_trade_log),
        "forward_signal_log": _normalise_horizon(forward_signal_log),
        "forward_accuracy_summary": _normalise_horizon(forward_accuracy_summary),
        "forward_probability_calibration_summary": _normalise_horizon(forward_probability_calibration_summary),
        "forward_warning_table": _normalise_horizon(forward_warning_table),
        "latest_model_predictions": _normalise_horizon(latest_model_predictions),
    }
    if current_pending_forward_signals is not None and not current_pending_forward_signals.empty:
        pending = _normalise_horizon(current_pending_forward_signals)
        tables["forward_signal_log"] = pd.concat([tables["forward_signal_log"], pending], ignore_index=True)
    if forward_evidence_coverage is not None and not forward_evidence_coverage.empty:
        coverage = _normalise_horizon(forward_evidence_coverage)
        if tables["forward_accuracy_summary"].empty:
            tables["forward_accuracy_summary"] = coverage.copy()
        else:
            tables["forward_accuracy_summary"] = pd.concat([tables["forward_accuracy_summary"], coverage], ignore_index=True)

    candidate_rows: List[Dict[str, Any]] = []
    scorecard_rows: List[Dict[str, Any]] = []
    for asset in asset_list:
        for horizon in horizon_list:
            metrics = _build_candidate_metrics(
                asset=asset,
                horizon=int(horizon),
                probability_calibration_summary=tables["probability_calibration_summary"],
                probability_calibration_warnings=tables["probability_calibration_warnings"],
                probability_calibration_recommendations=tables["probability_calibration_recommendations"],
                probability_calibration_bins=tables["probability_calibration_bins"],
                true_raw_trade_log=tables["true_raw_trade_log"],
                forward_signal_log=tables["forward_signal_log"],
                forward_accuracy_summary=tables["forward_accuracy_summary"],
                forward_probability_calibration_summary=tables["forward_probability_calibration_summary"],
                forward_warning_table=tables["forward_warning_table"],
                latest_model_predictions=tables["latest_model_predictions"],
            )
            scores = _score_candidate(metrics, risk_appetite)
            capital_status, research_action, suggested_mode = _decide(metrics, scores, risk_appetite, minimum_evidence_score)
            candidate_rows.append(_candidate_row(0, metrics, scores, capital_status, research_action, suggested_mode))
            scorecard_rows.append(
                {
                    "Asset": asset,
                    "Horizon": int(horizon),
                    "RawTradeCount": int(metrics["raw_trade_count"]),
                    "ForwardPendingCount": int(metrics["forward_pending"]),
                    "ForwardMaturedCount": int(metrics["forward_matured"]),
                    "BrierScore": round(metrics["brier"], 6) if np.isfinite(metrics["brier"]) else np.nan,
                    "CalibrationGrade": metrics["calibration_grade"],
                    "ForwardWinRate_%": round(metrics["forward_win_rate"], 4) if np.isfinite(metrics["forward_win_rate"]) else np.nan,
                    "ForwardBenchmarkEdge_%": round(metrics["forward_edge_pct"], 4) if np.isfinite(metrics["forward_edge_pct"]) else np.nan,
                    "MaxDrawdown_%": round(metrics["raw_drawdown"] * 100.0 if np.isfinite(metrics["raw_drawdown"]) and abs(metrics["raw_drawdown"]) <= 2 else metrics["raw_drawdown"], 4) if np.isfinite(metrics["raw_drawdown"]) else np.nan,
                    "WarningCount": len(metrics["warnings"]),
                    "EvidenceScore": scores["EvidenceScore"],
                    "OpportunityScore": scores["OpportunityScore"],
                    "RiskScore": scores["RiskScore"],
                }
            )

    ranked = pd.DataFrame(candidate_rows, columns=list(RANKED_PLAN_COLUMNS))
    if ranked.empty:
        return _empty_report(settings)
    ranked = ranked.sort_values(["ActionabilityScore", "EvidenceScore", "OpportunityScore"], ascending=[False, False, False]).reset_index(drop=True)
    ranked["Rank"] = range(1, len(ranked) + 1)
    if not include_blocked_candidates:
        display_ranked = ranked[~ranked["ResearchAction"].isin(["Avoid", "BlockedDueToDataFailure"])].copy()
    else:
        display_ranked = ranked.copy()
    if display_ranked.empty:
        display_ranked = ranked.head(max(int(top_n_plan_cards), 3)).copy()
    display_ranked = display_ranked.reset_index(drop=True)
    display_ranked["Rank"] = range(1, len(display_ranked) + 1)

    warnings = _warnings_table(ranked.to_dict("records"))
    executive = _executive_table(display_ranked, warnings)
    plan_cards = display_ranked.head(int(top_n_plan_cards)).copy()
    if plan_cards.empty:
        plan_cards = ranked.head(max(int(top_n_plan_cards), 3)).copy()
    entry = pd.DataFrame(
        [
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "Decision": row["Decision"],
                "EntryTrigger": row["EntryTrigger"],
                "SignalDirection": row["CurrentSignalDirection"],
                "ProbabilityUp": row["ProbabilityUp"],
                "TriggerStatus": "Paper-only trigger" if row["ResearchAction"] == "PaperTradeOnly" else row["ResearchAction"],
                "Warnings": row["MainWarnings"],
            }
            for _, row in display_ranked.iterrows()
        ],
        columns=list(ENTRY_TRIGGER_COLUMNS),
    )
    invalidation = pd.DataFrame(
        [
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "InvalidationRule": row["InvalidationRule"],
                "ReviewDate": row["ReviewDate"],
                "Warnings": row["MainWarnings"],
            }
            for _, row in display_ranked.iterrows()
        ],
        columns=list(INVALIDATION_RULE_COLUMNS),
    )
    risk_budget = pd.DataFrame(
        [
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "Decision": row["Decision"],
                "CapitalDeploymentStatus": row["CapitalDeploymentStatus"],
                "ResearchAction": row["ResearchAction"],
                "DeploymentAllowed": row["DeploymentAllowed"],
                "RealCapitalRiskCap": row["RealCapitalRiskCap"],
                "RiskCap": row["RiskCap"],
                "MaxPaperTradeSize": row["MaxPaperTradeSize"],
                "RiskReason": row["WhyNotDeploymentReady"],
            }
            for _, row in display_ranked.iterrows()
        ],
        columns=list(RISK_BUDGET_COLUMNS),
    )
    scorecard = pd.DataFrame(scorecard_rows, columns=list(EVIDENCE_SCORECARD_COLUMNS)).sort_values(["EvidenceScore", "OpportunityScore"], ascending=[False, False]).reset_index(drop=True)
    blocked = ranked[ranked["CapitalDeploymentStatus"].eq("Blocked")].copy().reset_index(drop=True)
    watchlist = pd.DataFrame(
        [
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "Decision": row["Decision"],
                "CapitalDeploymentStatus": row["CapitalDeploymentStatus"],
                "ResearchAction": row["ResearchAction"],
                "PaperTradeAllowed": row["PaperTradeAllowed"],
                "ActionabilityScore": row["ActionabilityScore"],
                "WatchReason": row["ReasonResearchAction"],
                "NextReviewTrigger": row["RequiredEvidenceToUpgrade"],
                "MainWarnings": row["MainWarnings"],
            }
            for _, row in ranked[ranked["ResearchAction"].eq("Watchlist")].iterrows()
        ],
        columns=list(WATCHLIST_COLUMNS),
    )
    paper = ranked[ranked["ResearchAction"].eq("PaperTradeOnly")].copy().reset_index(drop=True)
    next_evidence = pd.DataFrame(
        [
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "EvidenceNeeded": row["RequiredEvidenceToUpgrade"],
                "MinimumForwardMaturedRows": 10,
                "CalibrationRequirement": "Brier/ECE must improve or remain stable with raw forward rows.",
                "Priority": "High" if row["ResearchAction"] == "PaperTradeOnly" else "Medium",
            }
            for _, row in ranked.iterrows()
        ],
        columns=list(NEXT_EVIDENCE_COLUMNS),
    )

    return ActionableResearchPlanReport(
        executive_decision_table=executive,
        ranked_asset_horizon_plan=display_ranked[list(RANKED_PLAN_COLUMNS)],
        plan_card_table=plan_cards[list(PLAN_CARD_COLUMNS)],
        entry_trigger_table=entry,
        invalidation_rule_table=invalidation,
        risk_budget_table=risk_budget,
        evidence_scorecard=scorecard,
        blocked_candidates_table=blocked[list(RANKED_PLAN_COLUMNS)] if not blocked.empty else pd.DataFrame(columns=list(RANKED_PLAN_COLUMNS)),
        watchlist_table=watchlist,
        paper_trade_plan_table=paper[list(RANKED_PLAN_COLUMNS)] if not paper.empty else pd.DataFrame(columns=list(RANKED_PLAN_COLUMNS)),
        next_evidence_needed_table=next_evidence,
        warnings_table=warnings,
        settings=settings,
    )


__all__ = [
    "ActionableResearchPlanReport",
    "EXECUTIVE_DECISION_COLUMNS",
    "RANKED_PLAN_COLUMNS",
    "PLAN_CARD_COLUMNS",
    "ENTRY_TRIGGER_COLUMNS",
    "INVALIDATION_RULE_COLUMNS",
    "RISK_BUDGET_COLUMNS",
    "EVIDENCE_SCORECARD_COLUMNS",
    "WATCHLIST_COLUMNS",
    "NEXT_EVIDENCE_COLUMNS",
    "WARNING_COLUMNS",
    "run_actionable_research_plan",
]
