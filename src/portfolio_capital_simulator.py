"""Phase 12 portfolio and capital allocation simulator.

The simulator consumes Phase 10 and Phase 11 research artifacts, then builds a
paper portfolio and any strictly conditional capped capital plan allowed by the
Phase 11 gates. It does not execute trades, guarantee outcomes, or override
capital eligibility evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names
from src.artifact_store import build_input_source_table, resolve_artifact, save_phase_artifacts


SIMULATOR_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)

PORTFOLIO_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "RunDate",
    "TotalCandidates",
    "PaperOnlyCandidates",
    "RealCapitalEligibleCandidates",
    "WatchlistCandidates",
    "BlockedCandidates",
    "PortfolioMode",
    "TotalPaperCapital",
    "TotalPaperAllocatedPct",
    "PaperReservePct",
    "NumberAllocatedPaperCandidates",
    "NumberEligiblePaperCandidates",
    "TotalRealCapitalAllowedPct",
    "MaxPortfolioLossPct",
    "MainPaperAllocationReason",
    "MainRiskWarning",
    "PortfolioActionSummary",
)

ALLOCATION_PLAN_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "Asset",
    "Horizon",
    "ResearchAction",
    "CapitalDeploymentStatus",
    "RealCapitalAllowed",
    "AllocationMode",
    "AllocationScore",
    "SuggestedPaperWeightPct",
    "SuggestedRealWeightPct",
    "PaperAllocationStatus",
    "ZeroWeightReason",
    "PaperReservePct",
    "MaxLossPct",
    "PositionSizingRule",
    "StopRule",
    "ExitRule",
    "InvalidationRule",
    "ReviewDate",
    "MainReason",
    "MainWarning",
    "PortfolioContributionReason",
)

POSITION_SIZING_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "AllocationMode",
    "PaperPositionSize",
    "RealPositionSizePct",
    "MaxLossPct",
    "StopDistanceProxy",
    "RiskUnit",
    "PositionSizingExplanation",
    "Warnings",
)

RISK_BUDGET_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "AllocationMode",
    "SuggestedPaperWeightPct",
    "SuggestedRealWeightPct",
    "MaxLossPct",
    "MaxPortfolioLossPct",
    "RealCapitalAllowed",
    "RiskBudgetStatus",
    "RiskWarning",
)

STRESS_COLUMNS: Tuple[str, ...] = (
    "Scenario",
    "EstimatedPortfolioLossPct",
    "MaxPortfolioLossPct",
    "Breach",
    "Notes",
)

CONCENTRATION_COLUMNS: Tuple[str, ...] = (
    "ConcentrationType",
    "Bucket",
    "ExposurePct",
    "LimitPct",
    "Warning",
)

COST_STRESS_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "AllocationMode",
    "BaseWeightPct",
    "HigherCostImpactPct",
    "StressResult",
    "Warning",
)

STOP_EXIT_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "AllocationMode",
    "StopRule",
    "ExitRule",
    "InvalidationRule",
    "ReviewDate",
)

CAPITAL_BLOCKER_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "ResearchAction",
    "CapitalDeploymentStatus",
    "AllocationMode",
    "MainCapitalBlocker",
    "FailedGates",
    "WhatWouldAllowRealCapital",
)

SCENARIO_COLUMNS: Tuple[str, ...] = (
    "Scenario",
    "PortfolioMode",
    "PaperImpact",
    "RealCapitalImpactPct",
    "ExpectedAction",
    "Warning",
)

NEXT_ACTION_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "NextAction",
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


@dataclass
class PortfolioCapitalSimulatorReport:
    portfolio_summary_table: pd.DataFrame
    allocation_plan_table: pd.DataFrame
    paper_portfolio_table: pd.DataFrame
    conditional_real_capital_table: pd.DataFrame
    position_sizing_table: pd.DataFrame
    risk_budget_table: pd.DataFrame
    portfolio_drawdown_stress_table: pd.DataFrame
    correlation_concentration_table: pd.DataFrame
    cost_slippage_stress_table: pd.DataFrame
    stop_exit_plan_table: pd.DataFrame
    capital_blocker_table: pd.DataFrame
    scenario_analysis_table: pd.DataFrame
    next_actions_table: pd.DataFrame
    warning_table: pd.DataFrame
    input_source_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: Dict[str, Any] = field(default_factory=dict)
    saved_artifacts: Dict[str, Any] = field(default_factory=dict)


def _to_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value)


def _normalise_horizon(value: Any) -> pd.DataFrame:
    out = _to_frame(value)
    if out.empty:
        return out
    if "Horizon" in out.columns:
        out["Horizon"] = out["Horizon"].astype(str).str.replace("D", "", regex=False)
        out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce").astype("Int64")
    return out


def _subset(df: pd.DataFrame, asset: str, horizon: int) -> pd.DataFrame:
    if df.empty or not {"Asset", "Horizon"}.issubset(df.columns):
        return pd.DataFrame()
    values = pd.to_numeric(df["Horizon"].astype(str).str.replace("D", "", regex=False), errors="coerce")
    return df[df["Asset"].astype(str).eq(str(asset)) & values.eq(int(horizon))].copy()


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


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
        "ConcentrationRisk",
        "CorrelationRisk",
        "DrawdownRisk",
        "CostFragile",
        "ProbabilityUnreliable",
        "BenchmarkDominated",
        "LowTradeCount",
        "PendingEvidenceOnly",
    }
    return "High" if str(warning) in high else "Medium"


def _run_date(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value if value is not None else pd.Timestamp.today(), errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp.today()
    return pd.Timestamp(ts).tz_localize(None).normalize()


def _artifact_specs() -> Dict[str, Tuple[str, str, bool]]:
    return {
        "plan_card_table": ("Phase 10 Actionable Research Plan", "plan_card_table", False),
        "ranked_asset_horizon_plan": ("Phase 10 Actionable Research Plan", "ranked_asset_horizon_plan", False),
        "paper_trade_plan_table": ("Phase 10 Actionable Research Plan", "paper_trade_plan_table", False),
        "watchlist_table": ("Phase 10 Actionable Research Plan", "watchlist_table", False),
        "phase10_risk_budget_table": ("Phase 10 Actionable Research Plan", "risk_budget_table", False),
        "capital_eligibility_table": ("Phase 11 Daily Research Control Center", "capital_eligibility_table", False),
        "structured_capital_plan_table": ("Phase 11 Daily Research Control Center", "structured_capital_plan_table", False),
        "capital_blocker_table": ("Phase 11 Daily Research Control Center", "capital_blocker_table", False),
        "active_paper_signals_table": ("Phase 11 Daily Research Control Center", "active_paper_signals_table", False),
        "pending_outcomes_table": ("Phase 11 Daily Research Control Center", "pending_outcomes_table", False),
        "top_paper_candidates_today": ("Phase 11 Daily Research Control Center", "top_paper_candidates_today", False),
        "evidence_health_table": ("Phase 11 Daily Research Control Center", "evidence_health_table", False),
        "phase11_warning_table": ("Phase 11 Daily Research Control Center", "warning_table", False),
        "true_raw_trade_log": ("Phase 8I True Raw Trade Logs", "true_raw_trade_log", False),
        "probability_calibration_summary": ("Phase 8F Probability Calibration", "probability_calibration_summary", False),
        "forward_signal_log": ("Phase 9 Forward Paper Evidence", "forward_signal_log", False),
    }


def _resolve_inputs(use_artifact_store: bool, prefer_uploaded: bool, uploaded_overrides: Optional[Dict[str, Any]], direct: Dict[str, Any]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    uploaded_overrides = uploaded_overrides or {}
    resolved_items: List[Dict[str, Any]] = []
    tables: Dict[str, pd.DataFrame] = {}
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
            resolved = resolve_artifact(phase, artifact, uploaded_file=uploaded_overrides.get(key), prefer_uploaded=prefer_uploaded, required=required)
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
    ranked = _subset(tables["ranked_asset_horizon_plan"], asset, horizon)
    paper = _subset(tables["paper_trade_plan_table"], asset, horizon)
    watch = _subset(tables["watchlist_table"], asset, horizon)
    capital = _subset(tables["capital_eligibility_table"], asset, horizon)
    structured = _subset(tables["structured_capital_plan_table"], asset, horizon)
    blockers = _subset(tables["capital_blocker_table"], asset, horizon)
    health = _subset(tables["evidence_health_table"], asset, horizon)
    raw = _subset(tables["true_raw_trade_log"], asset, horizon)
    prob = _subset(tables["probability_calibration_summary"], asset, horizon)
    forward = _subset(tables["forward_signal_log"], asset, horizon)
    active = _subset(tables["active_paper_signals_table"], asset, horizon)
    pending = _subset(tables["pending_outcomes_table"], asset, horizon)
    top_paper = _subset(tables["top_paper_candidates_today"], asset, horizon)
    phase10_risk = _subset(tables["phase10_risk_budget_table"], asset, horizon)
    warning_table = _subset(tables["phase11_warning_table"], asset, horizon)

    research_action = str(_first_value(ranked, ["ResearchAction", "Decision"], ""))
    if not research_action and not paper.empty:
        research_action = "PaperTradeOnly"
    if not research_action and not watch.empty:
        research_action = "Watchlist"
    if not research_action:
        research_action = "ObserveOnly"
    capital_status = str(_first_value(capital, ["CapitalDeploymentStatus"], _first_value(ranked, ["CapitalDeploymentStatus"], "Blocked")))
    real_allowed = _safe_bool(_first_value(capital, ["RealCapitalAllowed"], False))
    cap_plan = str(_first_value(capital, ["CapitalPlanType"], "NoRealCapital"))
    if real_allowed and cap_plan not in {"MicroCapitalTrial", "ConditionalResearchAllocation"}:
        cap_plan = "MicroCapitalTrial"

    opportunity = _safe_float(_first_value(ranked, ["OpportunityScore"], np.nan), np.nan)
    evidence = _safe_float(_first_value(ranked, ["EvidenceScore"], np.nan), np.nan)
    risk = _safe_float(_first_value(ranked, ["RiskScore"], np.nan), np.nan)
    actionability = _safe_float(_first_value(ranked, ["ActionabilityScore"], np.nan), np.nan)
    brier = _safe_float(_first_value(prob, ["BrierScore"], np.nan), np.nan)
    if not forward.empty and "Status" in forward.columns:
        forward_pending = forward[forward["Status"].astype(str).str.lower().eq("pending")].copy()
    else:
        forward_pending = forward
    pending_count = int(len(pending)) if not pending.empty else int(len(active)) if not active.empty else int(len(forward_pending))
    if health.empty:
        matured_count = 0
    else:
        matured_count = int(_safe_float(_first_value(health, ["MaturedForwardCount"], 0), 0))
    raw_count = int(len(raw))
    drawdown = _min_value(raw, ["MaxDrawdownDuringTrade", "MaxDrawdown_%", "MaxDrawdown"], np.nan)
    if np.isfinite(drawdown) and abs(drawdown) <= 2:
        drawdown *= 100.0
    raw_edge = _mean_value(raw, ["VsBuyHold", "VsBenchmark", "VsBenchmarkReturn"], np.nan)
    if np.isfinite(raw_edge) and abs(raw_edge) <= 2:
        raw_edge *= 100.0

    warnings = ["NotFinancialAdvice", "NotProductionReady"]
    for table in [ranked, paper, watch, capital, structured, blockers, health, raw, prob, forward, active, pending, top_paper, phase10_risk, warning_table]:
        if table.empty:
            continue
        for col in ["Warnings", "MainWarnings", "WarningType", "MainWarning", "MainCapitalBlocker", "CalibrationGrade", "CalibrationVerdict"]:
            if col in table.columns:
                warnings.extend(_collect_warnings(*table[col].dropna().tolist()))
    if raw_count < 10:
        warnings.append("LowTradeCount")
    if pending_count > 0 and matured_count <= 0:
        warnings.append("PendingEvidenceOnly")
    text = " ".join(warnings).lower()
    if "probabilityunreliable" in text or "unreliable" in text:
        warnings.append("ProbabilityUnreliable")
    if "benchmarkdominated" in text or ("benchmark" in text and "dominated" in text):
        warnings.append("BenchmarkDominated")
    if "drawdown" in text or (np.isfinite(drawdown) and drawdown <= -12.0):
        warnings.append("DrawdownRisk")
    if "costfragile" in text or "returndestroyed" in text:
        warnings.append("CostFragile")
    warnings.append("VolatilityMissing")
    warnings.append("ConservativeSizingUsed")

    rank = _safe_float(_first_value(ranked, ["Rank"], np.nan), np.nan)
    probability_up = _safe_float(
        _first_value(
            ranked,
            ["ProbabilityUp"],
            _first_value(
                top_paper,
                ["ProbabilityUp"],
                _first_value(
                    pending,
                    ["ProbabilityUp"],
                    _first_value(forward_pending, ["ProbabilityUp"], _mean_value(raw, ["ProbabilityUp"], np.nan)),
                ),
            ),
        ),
        np.nan,
    )
    forward_pending_count = _safe_float(_first_value(health, ["PendingSignalCount"], pending_count), pending_count)

    return {
        "Asset": asset,
        "Horizon": int(horizon),
        "Rank": rank,
        "ResearchAction": research_action,
        "CapitalDeploymentStatus": capital_status,
        "RealCapitalAllowed": bool(real_allowed),
        "CapitalPlanType": cap_plan,
        "MaxAllowedRealCapitalPct": _safe_float(_first_value(capital, ["MaxAllowedRealCapitalPct"], 0.0), 0.0),
        "MaxLossPerIdeaPct": _safe_float(_first_value(capital, ["MaxLossPerIdeaPct"], 0.0), 0.0),
        "ReviewDate": str(_first_value(capital, ["ReviewDate"], _first_value(ranked, ["ReviewDate"], ""))),
        "StopRule": str(_first_value(structured, ["StopOrInvalidationRule"], "Stop if paper losses, benchmark underperformance, or drawdown warnings repeat.")),
        "ExitRule": str(_first_value(structured, ["ExitCondition"], "Exit or review after the target outcome date or signal invalidation.")),
        "InvalidationRule": str(_first_value(ranked, ["InvalidationRule"], _first_value(structured, ["StopOrInvalidationRule"], "Invalidate if evidence weakens."))),
        "PositionSizingRule": str(_first_value(structured, ["PositionSizingRule"], "Use conservative proxy sizing because volatility data is unavailable.")),
        "MainBlocker": str(_first_value(blockers, ["MainCapitalBlocker"], _first_value(capital, ["MainCapitalBlocker"], ""))),
        "WhatWouldAllowRealCapital": str(_first_value(blockers, ["WhatWouldAllowRealCapital"], _first_value(capital, ["WhatWouldAllowRealCapital"], ""))),
        "FailedGates": str(_first_value(blockers, ["FailedGates"], "")),
        "OpportunityScore": opportunity,
        "EvidenceScore": evidence,
        "RiskScore": risk,
        "ActionabilityScore": actionability,
        "ProbabilityUp": probability_up,
        "ForwardPendingCount": forward_pending_count,
        "BrierScore": brier,
        "RawTradeCount": raw_count,
        "PendingCount": pending_count,
        "MaturedCount": matured_count,
        "DrawdownPct": drawdown,
        "RawEdgePct": raw_edge,
        "Warnings": _collect_warnings(warnings),
    }


def _allocation_mode(metrics: Dict[str, Any], include_watchlist: bool, include_blocked: bool) -> str:
    action = str(metrics["ResearchAction"])
    status = str(metrics["CapitalDeploymentStatus"])
    if metrics["RealCapitalAllowed"]:
        if status == "ConditionalResearchCapitalEligible" or metrics["CapitalPlanType"] == "ConditionalResearchAllocation":
            return "ConditionalResearchCapital"
        return "ConditionalMicroCapital"
    if action == "PaperTradeOnly":
        return "PaperOnly"
    if action == "Watchlist" and include_watchlist:
        return "WatchlistOnly"
    if action in {"Avoid", "BlockedDueToDataFailure"} or status == "Blocked":
        return "NoAllocation"
    return "NoAllocation" if not include_blocked else "NoAllocation"


def _candidate_score(metrics: Dict[str, Any]) -> float:
    action = str(metrics.get("ResearchAction", ""))
    is_paper = action == "PaperTradeOnly"
    is_watchlist = action == "Watchlist"
    opportunity = _safe_float(metrics.get("OpportunityScore"), np.nan)
    evidence = _safe_float(metrics.get("EvidenceScore"), np.nan)
    risk = _safe_float(metrics.get("RiskScore"), np.nan)
    actionability = _safe_float(metrics.get("ActionabilityScore"), np.nan)
    if not np.isfinite(opportunity):
        opportunity = 55.0 if is_paper else 42.0 if is_watchlist else 25.0
    if not np.isfinite(evidence):
        evidence = 35.0 if is_paper else 28.0 if is_watchlist else 18.0
    if not np.isfinite(risk):
        risk = 55.0 if is_paper or is_watchlist else 75.0
    if not np.isfinite(actionability):
        actionability = opportunity * 0.60 + evidence * 0.40
    score = opportunity * 0.35 + evidence * 0.25 + actionability * 0.25 + max(0.0, 100.0 - risk) * 0.15

    rank = _safe_float(metrics.get("Rank"), np.nan)
    if np.isfinite(rank) and rank > 0:
        score += max(0.0, 31.0 - min(rank, 30.0)) / 30.0 * 10.0
    probability_up = _safe_float(metrics.get("ProbabilityUp"), np.nan)
    if np.isfinite(probability_up):
        score += min(abs(probability_up - 0.50) * 120.0, 12.0)
    elif is_paper:
        score -= 4.0
    pending_count = _safe_float(metrics.get("ForwardPendingCount"), metrics.get("PendingCount", 0.0))
    if np.isfinite(pending_count) and pending_count > 0:
        score += min(pending_count * 3.0, 8.0)
    elif is_paper:
        score -= 4.0
    raw_count = _safe_float(metrics.get("RawTradeCount"), 0.0)
    matured_count = _safe_float(metrics.get("MaturedCount"), 0.0)
    score += min(max(raw_count, 0.0) * 0.4 + max(matured_count, 0.0) * 0.8, 8.0)

    penalties = {
        "ProbabilityUnreliable": 18.0,
        "BenchmarkDominated": 16.0,
        "DrawdownRisk": 15.0,
        "CostFragile": 10.0,
        "LowTradeCount": 8.0,
        "PendingEvidenceOnly": 6.0,
    }
    for warning, penalty in penalties.items():
        if warning in metrics["Warnings"]:
            score -= penalty
    if is_paper and not _has_severe_paper_data_failure(metrics):
        score = max(score, 8.0)
    return float(np.clip(score, 0.0, 100.0))


def _paper_mode_settings(portfolio_mode: str) -> Dict[str, float]:
    text = str(portfolio_mode or "Conservative").lower()
    if "aggressive" in text:
        return {"target_count": 10, "base_deploy_pct": 65.0, "min_score": 0.0}
    if "balanced" in text:
        return {"target_count": 6, "base_deploy_pct": 40.0, "min_score": 0.0}
    return {"target_count": 2, "base_deploy_pct": 20.0, "min_score": 0.0}


def _has_severe_paper_data_failure(row: Dict[str, Any]) -> bool:
    action = str(row.get("ResearchAction", ""))
    warnings = " ".join(str(w) for w in row.get("Warnings", []))
    blocker_text = " ".join(
        str(row.get(name, ""))
        for name in ["CapitalDeploymentStatus", "MainBlocker", "FailedGates", "WhatWouldAllowRealCapital"]
    )
    text = f"{action} {warnings} {blocker_text}".lower()
    severe_tokens = [
        "blockedduetodatafailure",
        "datafailure",
        "data failure",
        "missingcritical",
        "missing critical",
        "no usable",
        "nousable",
        "invalid signal",
    ]
    if any(token in text for token in severe_tokens):
        return True
    probability_up = _safe_float(row.get("ProbabilityUp"), np.nan)
    opportunity = _safe_float(row.get("OpportunityScore"), np.nan)
    signal_rows = (
        max(_safe_float(row.get("PendingCount"), 0.0), 0.0)
        + max(_safe_float(row.get("ForwardPendingCount"), 0.0), 0.0)
        + max(_safe_float(row.get("RawTradeCount"), 0.0), 0.0)
        + max(_safe_float(row.get("MaturedCount"), 0.0), 0.0)
    )
    return action == "PaperTradeOnly" and signal_rows <= 0 and not np.isfinite(probability_up) and not np.isfinite(opportunity)


def _normalise_paper_allocation_scores(paper_rows: List[Dict[str, Any]]) -> None:
    if not paper_rows:
        return
    raw_scores = [float(row.get("_Score", 0.0)) for row in paper_rows]
    low = min(raw_scores)
    high = max(raw_scores)
    for row in paper_rows:
        raw = float(row.get("_Score", 0.0))
        if high > low:
            normalized = 30.0 + (raw - low) / (high - low) * 70.0
        else:
            normalized = max(35.0, min(65.0, raw + 25.0))
        row["_AllocationScore"] = float(np.clip(normalized, 0.0, 100.0))


def _paper_reserve_pct(paper_candidates: List[Dict[str, Any]], base_deploy_pct: float) -> float:
    if not paper_candidates:
        return 100.0
    avg_score = float(np.mean([row.get("_AllocationScore", row["_Score"]) for row in paper_candidates]))
    severe_count = sum(
        1
        for row in paper_candidates
        if set(row.get("Warnings", [])) & {"ProbabilityUnreliable", "BenchmarkDominated", "DrawdownRisk", "CostFragile", "LowTradeCount", "PendingEvidenceOnly"}
    )
    deploy = float(base_deploy_pct)
    if avg_score < 30.0:
        deploy *= 0.65
    elif avg_score < 45.0:
        deploy *= 0.80
    elif avg_score >= 70.0:
        deploy = min(90.0, deploy + 10.0)
    if severe_count:
        deploy *= max(0.55, 1.0 - 0.05 * severe_count)
    deploy = float(np.clip(deploy, 10.0, 90.0))
    return round(100.0 - deploy, 4)


def _main_warning(warnings: List[str]) -> str:
    priority = [
        "ProbabilityUnreliable",
        "BenchmarkDominated",
        "DrawdownRisk",
        "CostFragile",
        "LowTradeCount",
        "PendingEvidenceOnly",
        "ConcentrationRisk",
        "CorrelationRisk",
        "VolatilityMissing",
        "ConservativeSizingUsed",
    ]
    for warning in priority:
        if warning in warnings:
            return warning
    return warnings[0] if warnings else "NotFinancialAdvice"


def _main_reason(metrics: Dict[str, Any], mode: str) -> str:
    if mode in {"ConditionalMicroCapital", "ConditionalResearchCapital"}:
        return "Phase 11 capital gates passed; allocation remains conditional, capped, and monitored."
    if mode == "PaperOnly":
        return "Paper allocation only because real-capital gates did not pass."
    if mode == "WatchlistOnly":
        return "Watchlist only; wait for stronger forward evidence or fewer warnings."
    return metrics["WhatWouldAllowRealCapital"] or "No allocation because evidence or eligibility is insufficient."


def _build_allocation_rows(
    metrics_rows: List[Dict[str, Any]],
    *,
    portfolio_mode: str,
    total_paper_capital: float,
    max_real_capital_cap_pct: float,
    max_single_idea_loss_pct: float,
    max_portfolio_loss_pct: float,
    max_single_asset_exposure_pct: float,
    max_single_horizon_exposure_pct: float,
    include_watchlist_candidates: bool,
    include_blocked_candidates: bool,
) -> pd.DataFrame:
    enriched: List[Dict[str, Any]] = []
    for metrics in metrics_rows:
        mode = _allocation_mode(metrics, include_watchlist_candidates, include_blocked_candidates)
        score = _candidate_score(metrics)
        enriched.append({**metrics, "AllocationMode": mode, "_Score": score, "_AllocationScore": score})

    mode_settings = _paper_mode_settings(portfolio_mode)
    all_paper_candidates = [row for row in enriched if row["AllocationMode"] == "PaperOnly"]
    _normalise_paper_allocation_scores(all_paper_candidates)
    paper_candidates = sorted(
        [row for row in all_paper_candidates if not _has_severe_paper_data_failure(row)],
        key=lambda row: (row["_AllocationScore"], row["_Score"]),
        reverse=True,
    )
    target_count = int(mode_settings["target_count"])
    selected_paper = paper_candidates[:target_count]
    selected_keys = {(row["Asset"], row["Horizon"]) for row in selected_paper}
    reserve_pct = _paper_reserve_pct(selected_paper, mode_settings["base_deploy_pct"])
    deploy_pct = 100.0 - reserve_pct
    total_selected_score = sum(max(row["_AllocationScore"], 1.0) for row in selected_paper)
    asset_exposure: Dict[str, float] = {}
    horizon_exposure: Dict[int, float] = {}
    paper_weights: Dict[Tuple[str, int], float] = {}
    paper_reasons: Dict[Tuple[str, int], Tuple[str, str]] = {}
    for row in selected_paper:
        key = (row["Asset"], row["Horizon"])
        raw_weight = deploy_pct * max(row["_AllocationScore"], 1.0) / max(total_selected_score, 1.0)
        available_asset = max_single_asset_exposure_pct - asset_exposure.get(row["Asset"], 0.0)
        available_horizon = max_single_horizon_exposure_pct - horizon_exposure.get(int(row["Horizon"]), 0.0)
        weight = max(0.0, min(raw_weight, available_asset, available_horizon))
        if weight > 0:
            paper_weights[key] = weight
            asset_exposure[row["Asset"]] = asset_exposure.get(row["Asset"], 0.0) + weight
            horizon_exposure[int(row["Horizon"])] = horizon_exposure.get(int(row["Horizon"]), 0.0) + weight
            paper_reasons[key] = ("Allocated", "Diversified paper allocation under asset and horizon caps.")
        else:
            reason = "Asset exposure cap reached." if available_asset <= 0 else "Horizon exposure cap reached." if available_horizon <= 0 else "Portfolio paper reserve retained due to weak evidence."
            paper_reasons[key] = ("EligibleButNotAllocated", reason)

    real_rows = [row for row in enriched if row["AllocationMode"] in {"ConditionalMicroCapital", "ConditionalResearchCapital"}]
    real_caps = [min(max(row["MaxAllowedRealCapitalPct"], 0.0), max_real_capital_cap_pct, max_single_asset_exposure_pct, max_single_horizon_exposure_pct) for row in real_rows]
    real_scale = 1.0
    if sum(real_caps) > max_real_capital_cap_pct and sum(real_caps) > 0:
        real_scale = max_real_capital_cap_pct / sum(real_caps)

    rows: List[Dict[str, Any]] = []
    real_index = 0
    for item in enriched:
        mode = item["AllocationMode"]
        key = (item["Asset"], item["Horizon"])
        paper_weight = float(paper_weights.get(key, 0.0))
        if mode == "PaperOnly":
            if key in paper_reasons:
                paper_status, zero_reason = paper_reasons[key]
                if paper_weight > 0:
                    zero_reason = ""
            elif _has_severe_paper_data_failure(item):
                paper_status = "EligibleButNotAllocated"
                zero_reason = "Severe data failure prevented paper allocation, but the row remains visible for diagnostics."
            elif key not in selected_keys:
                paper_status = "EligibleButNotAllocated"
                zero_reason = f"Outside top {target_count} paper candidates for {portfolio_mode} mode."
            else:
                paper_status = "EligibleButNotAllocated"
                zero_reason = "Portfolio paper reserve retained due to weak evidence."
        elif mode == "WatchlistOnly":
            paper_status = "WatchlistOnly"
            zero_reason = "Watchlist candidates receive no paper allocation until upgraded to PaperTradeOnly."
        else:
            paper_status = "NoAllocation"
            zero_reason = "No paper allocation because candidate is blocked, avoided, or real-capital-only."
        if mode in {"ConditionalMicroCapital", "ConditionalResearchCapital"}:
            real_weight = real_caps[real_index] * real_scale
            real_index += 1
        else:
            real_weight = 0.0
        max_loss = min(max_single_idea_loss_pct, item["MaxLossPerIdeaPct"] if item["MaxLossPerIdeaPct"] > 0 else max_single_idea_loss_pct)
        if real_weight <= 0:
            max_loss = 0.0 if mode != "PaperOnly" else min(max_single_idea_loss_pct, 0.25)
        warning = _main_warning(item["Warnings"])
        rows.append(
            {
                "Rank": 0,
                "Asset": item["Asset"],
                "Horizon": item["Horizon"],
                "ResearchAction": item["ResearchAction"],
                "CapitalDeploymentStatus": item["CapitalDeploymentStatus"],
                "RealCapitalAllowed": bool(item["RealCapitalAllowed"]),
                "AllocationMode": mode,
                "AllocationScore": round(float(item.get("_AllocationScore", item["_Score"])), 4),
                "SuggestedPaperWeightPct": round(float(paper_weight), 4),
                "SuggestedRealWeightPct": round(float(real_weight), 4),
                "PaperAllocationStatus": paper_status,
                "ZeroWeightReason": zero_reason if paper_weight <= 0 else "",
                "PaperReservePct": reserve_pct,
                "MaxLossPct": round(float(max_loss), 4),
                "PositionSizingRule": item["PositionSizingRule"],
                "StopRule": item["StopRule"],
                "ExitRule": item["ExitRule"],
                "InvalidationRule": item["InvalidationRule"],
                "ReviewDate": item["ReviewDate"],
                "MainReason": _main_reason(item, mode),
                "MainWarning": warning,
                "PortfolioContributionReason": (
                    "Allocated as a diversified paper candidate."
                    if paper_weight > 0
                    else zero_reason or _main_reason(item, mode)
                ),
                "_Score": item["_Score"],
                "_AllocationScore": item.get("_AllocationScore", item["_Score"]),
                "_Warnings": item["Warnings"],
                "_MainBlocker": item["MainBlocker"],
                "_FailedGates": item["FailedGates"],
                "_WhatWouldAllowRealCapital": item["WhatWouldAllowRealCapital"],
                "_DrawdownPct": item["DrawdownPct"],
                "_RawEdgePct": item["RawEdgePct"],
                "_TotalPaperCapital": total_paper_capital,
                "_MaxPortfolioLossPct": max_portfolio_loss_pct,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=list(ALLOCATION_PLAN_COLUMNS))
    df = df.sort_values(["SuggestedRealWeightPct", "SuggestedPaperWeightPct", "_AllocationScore"], ascending=[False, False, False]).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df


def _position_sizing_table(allocation: pd.DataFrame, total_paper_capital: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in allocation.iterrows():
        paper_size = round(total_paper_capital * _safe_float(row["SuggestedPaperWeightPct"], 0.0) / 100.0, 2)
        stop_proxy = max(1.0, _safe_float(row["MaxLossPct"], 0.0) * 4.0)
        risk_unit = round(_safe_float(row["MaxLossPct"], 0.0) / max(stop_proxy, 1.0), 6)
        rows.append(
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "AllocationMode": row["AllocationMode"],
                "PaperPositionSize": paper_size,
                "RealPositionSizePct": row["SuggestedRealWeightPct"],
                "MaxLossPct": row["MaxLossPct"],
                "StopDistanceProxy": round(stop_proxy, 4),
                "RiskUnit": risk_unit,
                "PositionSizingExplanation": "Conservative proxy sizing is used because volatility data is unavailable.",
                "Warnings": "VolatilityMissing; ConservativeSizingUsed",
            }
        )
    return pd.DataFrame(rows, columns=list(POSITION_SIZING_COLUMNS))


def _risk_budget_table(allocation: pd.DataFrame, max_portfolio_loss_pct: float) -> pd.DataFrame:
    rows = []
    for _, row in allocation.iterrows():
        risk_warning = row["MainWarning"]
        status = "WithinCaps" if _safe_float(row["MaxLossPct"], 0.0) <= max_portfolio_loss_pct else "LossCapBreach"
        rows.append(
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "AllocationMode": row["AllocationMode"],
                "SuggestedPaperWeightPct": row["SuggestedPaperWeightPct"],
                "SuggestedRealWeightPct": row["SuggestedRealWeightPct"],
                "MaxLossPct": row["MaxLossPct"],
                "MaxPortfolioLossPct": max_portfolio_loss_pct,
                "RealCapitalAllowed": row["RealCapitalAllowed"],
                "RiskBudgetStatus": status,
                "RiskWarning": risk_warning,
            }
        )
    return pd.DataFrame(rows, columns=list(RISK_BUDGET_COLUMNS))


def _concentration_table(allocation: pd.DataFrame, max_asset: float, max_horizon: float) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if allocation.empty:
        return pd.DataFrame(columns=list(CONCENTRATION_COLUMNS)), warnings
    exposure = allocation["SuggestedPaperWeightPct"].astype(float) + allocation["SuggestedRealWeightPct"].astype(float)
    temp = allocation.assign(_Exposure=exposure)
    for asset, group in temp.groupby("Asset"):
        exp = float(group["_Exposure"].sum())
        warn = "ConcentrationRisk" if exp > max_asset else ""
        if warn:
            warnings.append(warn)
        rows.append({"ConcentrationType": "Asset", "Bucket": asset, "ExposurePct": round(exp, 4), "LimitPct": max_asset, "Warning": warn})
    for horizon, group in temp.groupby("Horizon"):
        exp = float(group["_Exposure"].sum())
        warn = "CorrelationRisk" if exp > max_horizon else ""
        if warn:
            warnings.append(warn)
        rows.append({"ConcentrationType": "Horizon", "Bucket": f"{int(horizon)}D", "ExposurePct": round(exp, 4), "LimitPct": max_horizon, "Warning": warn})
    return pd.DataFrame(rows, columns=list(CONCENTRATION_COLUMNS)), _collect_warnings(warnings)


def _drawdown_stress_table(allocation: pd.DataFrame, max_portfolio_loss_pct: float) -> pd.DataFrame:
    real_loss = float((allocation["SuggestedRealWeightPct"].astype(float) * allocation["MaxLossPct"].astype(float) / 100.0).sum()) if not allocation.empty else 0.0
    paper_weight = float(allocation["SuggestedPaperWeightPct"].astype(float).sum()) if not allocation.empty else 0.0
    drawdown_shock = min(max_portfolio_loss_pct * 1.5, real_loss + 2.0)
    rows = [
        {"Scenario": "Base case", "EstimatedPortfolioLossPct": round(real_loss, 4), "MaxPortfolioLossPct": max_portfolio_loss_pct, "Breach": real_loss > max_portfolio_loss_pct, "Notes": "Uses current capped real-capital allocation."},
        {"Scenario": "Worst candidate loss", "EstimatedPortfolioLossPct": round(float(allocation["MaxLossPct"].max()) if not allocation.empty else 0.0, 4), "MaxPortfolioLossPct": max_portfolio_loss_pct, "Breach": (float(allocation["MaxLossPct"].max()) if not allocation.empty else 0.0) > max_portfolio_loss_pct, "Notes": "Largest individual loss cap is applied."},
        {"Scenario": "All paper candidates lose", "EstimatedPortfolioLossPct": 0.0, "MaxPortfolioLossPct": max_portfolio_loss_pct, "Breach": False, "Notes": f"Paper portfolio stress affects simulated capital only; paper weight {paper_weight:.2f}%."},
        {"Scenario": "Drawdown shock", "EstimatedPortfolioLossPct": round(drawdown_shock, 4), "MaxPortfolioLossPct": max_portfolio_loss_pct, "Breach": drawdown_shock > max_portfolio_loss_pct, "Notes": "Shock proxy for clustered losses and drawdown warnings."},
    ]
    return pd.DataFrame(rows, columns=list(STRESS_COLUMNS))


def _cost_stress_table(allocation: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in allocation.iterrows():
        weight = max(_safe_float(row["SuggestedPaperWeightPct"], 0.0), _safe_float(row["SuggestedRealWeightPct"], 0.0))
        impact = min(weight * 0.05, 1.0)
        warning = "CostFragile" if row["MainWarning"] == "CostFragile" else ""
        rows.append(
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "AllocationMode": row["AllocationMode"],
                "BaseWeightPct": round(weight, 4),
                "HigherCostImpactPct": round(impact, 4),
                "StressResult": "Reduce or keep paper-only if cost stress removes edge.",
                "Warning": warning,
            }
        )
    return pd.DataFrame(rows, columns=list(COST_STRESS_COLUMNS))


def _scenario_analysis(allocation: pd.DataFrame, portfolio_mode: str) -> pd.DataFrame:
    real_total = float(allocation["SuggestedRealWeightPct"].sum()) if not allocation.empty else 0.0
    paper_total = float(allocation["SuggestedPaperWeightPct"].sum()) if not allocation.empty else 0.0
    rows = [
        ("Base case", f"Track {paper_total:.2f}% paper allocation and {real_total:.2f}% conditional real-capital allocation.", real_total, "Continue monitoring.", "ConditionalOnly"),
        ("Higher cost/slippage", "Apply extra friction to all allocations.", max(real_total - 0.25, 0.0), "Reduce exposure or keep paper-only if edge weakens.", "CostFragile"),
        ("Worst candidate loss", "Largest candidate hits its loss cap.", min(real_total, 1.0), "Review stop rules and downgrade if warning repeats.", "DrawdownRisk"),
        ("All paper candidates lose", f"All allocated paper rows lose across {paper_total:.2f}% simulated paper exposure; no real-capital effect.", 0.0, "Invalidate weak paper rows and re-score.", "PaperOnly"),
        ("Benchmark dominates", "Passive benchmark comparison worsens.", max(real_total - 0.5, 0.0), "Move affected rows to watchlist or no allocation.", "BenchmarkDominated"),
        ("Drawdown shock", "Clustered drawdown warning appears.", min(real_total + 0.5, 5.0), "De-risk conditional allocations and keep paper rows visible.", "DrawdownRisk"),
        ("Probability calibration failure", "Calibration warning appears or Brier deteriorates.", 0.0, "Block real capital and return to paper/watchlist tracking.", "ProbabilityUnreliable"),
    ]
    return pd.DataFrame(
        [
            {
                "Scenario": scenario,
                "PortfolioMode": portfolio_mode,
                "PaperImpact": paper,
                "RealCapitalImpactPct": round(real_impact, 4),
                "ExpectedAction": action,
                "Warning": warning,
            }
            for scenario, paper, real_impact, action, warning in rows
        ],
        columns=list(SCENARIO_COLUMNS),
    )


def _next_actions(allocation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in allocation.iterrows():
        mode = row["AllocationMode"]
        if mode == "PaperOnly":
            action = "ContinuePaperTracking"
            priority = "High"
            reason = "Paper allocation exists; review after target outcome or invalidation."
        elif mode == "WatchlistOnly":
            action = "WatchlistReview"
            priority = "Medium"
            reason = "No allocation yet; wait for improved evidence or fewer warnings."
        elif mode in {"ConditionalMicroCapital", "ConditionalResearchCapital"}:
            action = "ReviewConditionalCapitalRisk"
            priority = "High"
            reason = "Strict gates passed; confirm caps, stops, and monitoring before any simulated allocation."
        else:
            action = "KeepBlocked"
            priority = "Medium"
            reason = row["MainReason"]
        rows.append({"Asset": row["Asset"], "Horizon": row["Horizon"], "NextAction": action, "Priority": priority, "Reason": reason, "ReviewDate": row["ReviewDate"]})
    return pd.DataFrame(rows, columns=list(NEXT_ACTION_COLUMNS))


def _warning_table(allocation: pd.DataFrame, input_source_table: pd.DataFrame, concentration_warnings: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {"Asset": "ALL", "Horizon": np.nan, "WarningType": "NotFinancialAdvice", "Severity": "High", "Message": "Research and capital-risk planning support only."},
        {"Asset": "ALL", "Horizon": np.nan, "WarningType": "NotProductionReady", "Severity": "High", "Message": "This simulator is not a live-deployment approval."},
        {"Asset": "ALL", "Horizon": np.nan, "WarningType": "ConditionalOnly", "Severity": "High", "Message": "Any real-capital row must remain capped, conditional, monitored, and invalidation-driven."},
        {"Asset": "ALL", "Horizon": np.nan, "WarningType": "VolatilityMissing", "Severity": "Medium", "Message": "Volatility data was unavailable for precise sizing."},
        {"Asset": "ALL", "Horizon": np.nan, "WarningType": "ConservativeSizingUsed", "Severity": "Medium", "Message": "Conservative proxy sizing was used."},
    ]
    if allocation.empty or not allocation["RealCapitalAllowed"].astype(bool).any():
        rows.append({"Asset": "ALL", "Horizon": np.nan, "WarningType": "RealCapitalBlocked", "Severity": "High", "Message": "No candidate passed Phase 11 real-capital gates."})
    for warning in concentration_warnings:
        rows.append({"Asset": "ALL", "Horizon": np.nan, "WarningType": warning, "Severity": _warning_severity(warning), "Message": f"{warning} detected in portfolio exposures."})
    if not input_source_table.empty:
        for _, row in input_source_table[input_source_table["Status"].astype(str).str.contains("Missing", na=False)].iterrows():
            rows.append({"Asset": "ALL", "Horizon": np.nan, "WarningType": "MissingArtifact", "Severity": "Medium", "Message": f"{row.get('Phase', '')} / {row.get('Artifact', '')} was not loaded."})
    for _, row in allocation.iterrows():
        for warning in _collect_warnings(row.get("_Warnings", []), row.get("MainWarning", "")):
            rows.append({"Asset": row["Asset"], "Horizon": row["Horizon"], "WarningType": warning, "Severity": _warning_severity(warning), "Message": f"{warning} affects allocation sizing or eligibility."})
    return pd.DataFrame(rows, columns=list(WARNING_COLUMNS)).drop_duplicates().reset_index(drop=True)


def run_portfolio_capital_simulator(
    *,
    plan_card_table: Optional[pd.DataFrame] = None,
    ranked_asset_horizon_plan: Optional[pd.DataFrame] = None,
    paper_trade_plan_table: Optional[pd.DataFrame] = None,
    watchlist_table: Optional[pd.DataFrame] = None,
    phase10_risk_budget_table: Optional[pd.DataFrame] = None,
    capital_eligibility_table: Optional[pd.DataFrame] = None,
    structured_capital_plan_table: Optional[pd.DataFrame] = None,
    capital_blocker_table: Optional[pd.DataFrame] = None,
    active_paper_signals_table: Optional[pd.DataFrame] = None,
    pending_outcomes_table: Optional[pd.DataFrame] = None,
    top_paper_candidates_today: Optional[pd.DataFrame] = None,
    evidence_health_table: Optional[pd.DataFrame] = None,
    phase11_warning_table: Optional[pd.DataFrame] = None,
    true_raw_trade_log: Optional[pd.DataFrame] = None,
    probability_calibration_summary: Optional[pd.DataFrame] = None,
    forward_signal_log: Optional[pd.DataFrame] = None,
    use_artifact_store: bool = False,
    prefer_uploaded: bool = False,
    uploaded_overrides: Optional[Dict[str, Any]] = None,
    run_date: Any = None,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    portfolio_mode: str = "Conservative",
    total_paper_capital: float = 100000.0,
    max_real_capital_cap_pct: float = 1.0,
    max_single_idea_loss_pct: float = 0.25,
    max_portfolio_loss_pct: float = 1.0,
    max_single_asset_exposure_pct: float = 25.0,
    max_single_horizon_exposure_pct: float = 35.0,
    include_watchlist_candidates: bool = True,
    include_blocked_candidates: bool = True,
    autosave: bool = False,
) -> PortfolioCapitalSimulatorReport:
    """Build the Phase 12 portfolio and capital allocation simulator report."""
    asset_list = list(assets or get_asset_names())
    horizon_list = [int(h) for h in (horizons or SIMULATOR_HORIZONS)]
    rd = _run_date(run_date)
    settings = {
        "phase": "12",
        "purpose": "portfolio_capital_allocation_simulator",
        "run_date": str(rd.date()),
        "portfolio_mode": portfolio_mode,
        "total_paper_capital": float(total_paper_capital),
        "max_real_capital_cap_pct": float(max_real_capital_cap_pct),
        "max_single_idea_loss_pct": float(max_single_idea_loss_pct),
        "max_portfolio_loss_pct": float(max_portfolio_loss_pct),
        "max_single_asset_exposure_pct": float(max_single_asset_exposure_pct),
        "max_single_horizon_exposure_pct": float(max_single_horizon_exposure_pct),
        "real_capital_source_of_truth": "Phase 11 capital gates",
        "production_ready_label_allowed": False,
        "guaranteed_return_language_allowed": False,
    }
    direct = {
        "plan_card_table": plan_card_table,
        "ranked_asset_horizon_plan": ranked_asset_horizon_plan,
        "paper_trade_plan_table": paper_trade_plan_table,
        "watchlist_table": watchlist_table,
        "phase10_risk_budget_table": phase10_risk_budget_table,
        "capital_eligibility_table": capital_eligibility_table,
        "structured_capital_plan_table": structured_capital_plan_table,
        "capital_blocker_table": capital_blocker_table,
        "active_paper_signals_table": active_paper_signals_table,
        "pending_outcomes_table": pending_outcomes_table,
        "top_paper_candidates_today": top_paper_candidates_today,
        "evidence_health_table": evidence_health_table,
        "phase11_warning_table": phase11_warning_table,
        "true_raw_trade_log": true_raw_trade_log,
        "probability_calibration_summary": probability_calibration_summary,
        "forward_signal_log": forward_signal_log,
    }
    tables, input_source_table = _resolve_inputs(use_artifact_store, prefer_uploaded, uploaded_overrides, direct)

    metrics_rows: List[Dict[str, Any]] = []
    for asset in asset_list:
        for horizon in horizon_list:
            metrics_rows.append(_candidate_metrics(asset, int(horizon), tables))

    allocation_full = _build_allocation_rows(
        metrics_rows,
        portfolio_mode=portfolio_mode,
        total_paper_capital=float(total_paper_capital),
        max_real_capital_cap_pct=float(max_real_capital_cap_pct),
        max_single_idea_loss_pct=float(max_single_idea_loss_pct),
        max_portfolio_loss_pct=float(max_portfolio_loss_pct),
        max_single_asset_exposure_pct=float(max_single_asset_exposure_pct),
        max_single_horizon_exposure_pct=float(max_single_horizon_exposure_pct),
        include_watchlist_candidates=bool(include_watchlist_candidates),
        include_blocked_candidates=bool(include_blocked_candidates),
    )
    if allocation_full.empty:
        allocation = pd.DataFrame(columns=list(ALLOCATION_PLAN_COLUMNS))
    else:
        allocation = allocation_full[list(ALLOCATION_PLAN_COLUMNS)].copy()
    paper = allocation[allocation["AllocationMode"].eq("PaperOnly")].copy().reset_index(drop=True) if not allocation.empty else pd.DataFrame(columns=list(ALLOCATION_PLAN_COLUMNS))
    conditional = allocation[allocation["AllocationMode"].isin(["ConditionalMicroCapital", "ConditionalResearchCapital"])].copy().reset_index(drop=True) if not allocation.empty else pd.DataFrame(columns=list(ALLOCATION_PLAN_COLUMNS))
    blockers = pd.DataFrame(
        [
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "ResearchAction": row["ResearchAction"],
                "CapitalDeploymentStatus": row["CapitalDeploymentStatus"],
                "AllocationMode": row["AllocationMode"],
                "MainCapitalBlocker": row.get("_MainBlocker", row["MainWarning"]),
                "FailedGates": row.get("_FailedGates", ""),
                "WhatWouldAllowRealCapital": row.get("_WhatWouldAllowRealCapital", row["MainReason"]),
            }
            for _, row in allocation_full[~allocation_full["RealCapitalAllowed"].astype(bool)].iterrows()
        ],
        columns=list(CAPITAL_BLOCKER_COLUMNS),
    ) if not allocation_full.empty else pd.DataFrame(columns=list(CAPITAL_BLOCKER_COLUMNS))
    if not include_blocked_candidates and not blockers.empty:
        blockers = blockers[~blockers["CapitalDeploymentStatus"].eq("Blocked")].copy()

    position_sizing = _position_sizing_table(allocation, float(total_paper_capital))
    risk_budget = _risk_budget_table(allocation, float(max_portfolio_loss_pct))
    concentration, concentration_warnings = _concentration_table(allocation, float(max_single_asset_exposure_pct), float(max_single_horizon_exposure_pct))
    drawdown_stress = _drawdown_stress_table(allocation, float(max_portfolio_loss_pct))
    cost_stress = _cost_stress_table(allocation)
    stop_exit = allocation[["Asset", "Horizon", "AllocationMode", "StopRule", "ExitRule", "InvalidationRule", "ReviewDate"]].copy() if not allocation.empty else pd.DataFrame(columns=list(STOP_EXIT_COLUMNS))
    scenario = _scenario_analysis(allocation, portfolio_mode)
    next_actions = _next_actions(allocation)
    warnings = _warning_table(allocation_full, input_source_table, concentration_warnings)

    total_real = float(allocation["SuggestedRealWeightPct"].sum()) if not allocation.empty else 0.0
    total_paper_allocated = float(allocation["SuggestedPaperWeightPct"].sum()) if not allocation.empty else 0.0
    paper_reserve = max(0.0, 100.0 - total_paper_allocated)
    allocated_paper_count = int((allocation["SuggestedPaperWeightPct"].astype(float) > 0).sum()) if not allocation.empty else 0
    eligible_paper_count = int(allocation["AllocationMode"].eq("PaperOnly").sum()) if not allocation.empty else 0
    if allocated_paper_count > 0:
        paper_reason = f"Allocated paper capital across {allocated_paper_count} of {eligible_paper_count} eligible paper candidates; reserve retained for weak or capped evidence."
    elif eligible_paper_count > 0:
        paper_reason = "Eligible paper candidates exist, but scores or exposure caps kept paper allocation in reserve."
    else:
        paper_reason = "No PaperTradeOnly candidates were available for paper allocation."
    summary_warning = "ConditionalOnly" if total_real > 0 else "RealCapitalBlocked"
    if concentration_warnings:
        summary_warning = concentration_warnings[0]
    summary = pd.DataFrame(
        [
            {
                "RunDate": str(rd.date()),
                "TotalCandidates": int(len(allocation)),
                "PaperOnlyCandidates": int(len(paper)),
                "RealCapitalEligibleCandidates": int(len(conditional)),
                "WatchlistCandidates": int(allocation["AllocationMode"].eq("WatchlistOnly").sum()) if not allocation.empty else 0,
                "BlockedCandidates": int(allocation["AllocationMode"].eq("NoAllocation").sum()) if not allocation.empty else 0,
                "PortfolioMode": portfolio_mode,
                "TotalPaperCapital": float(total_paper_capital),
                "TotalPaperAllocatedPct": round(total_paper_allocated, 4),
                "PaperReservePct": round(paper_reserve, 4),
                "NumberAllocatedPaperCandidates": allocated_paper_count,
                "NumberEligiblePaperCandidates": eligible_paper_count,
                "TotalRealCapitalAllowedPct": round(total_real, 4),
                "MaxPortfolioLossPct": float(max_portfolio_loss_pct),
                "MainPaperAllocationReason": paper_reason,
                "MainRiskWarning": summary_warning,
                "PortfolioActionSummary": (
                    "Strict Phase 11 gates allow only capped conditional real-capital allocation plus monitored paper allocation."
                    if total_real > 0
                    else "Real-capital gates did not pass; simulator provides paper allocation, watchlist review, and capital blockers."
                ),
            }
        ],
        columns=list(PORTFOLIO_SUMMARY_COLUMNS),
    )

    report = PortfolioCapitalSimulatorReport(
        portfolio_summary_table=summary,
        allocation_plan_table=allocation.reset_index(drop=True),
        paper_portfolio_table=paper.reset_index(drop=True),
        conditional_real_capital_table=conditional.reset_index(drop=True),
        position_sizing_table=position_sizing.reset_index(drop=True),
        risk_budget_table=risk_budget.reset_index(drop=True),
        portfolio_drawdown_stress_table=drawdown_stress.reset_index(drop=True),
        correlation_concentration_table=concentration.reset_index(drop=True),
        cost_slippage_stress_table=cost_stress.reset_index(drop=True),
        stop_exit_plan_table=stop_exit[list(STOP_EXIT_COLUMNS)].reset_index(drop=True) if not stop_exit.empty else pd.DataFrame(columns=list(STOP_EXIT_COLUMNS)),
        capital_blocker_table=blockers.reset_index(drop=True),
        scenario_analysis_table=scenario.reset_index(drop=True),
        next_actions_table=next_actions.reset_index(drop=True),
        warning_table=warnings.reset_index(drop=True),
        input_source_table=input_source_table,
        settings=settings,
    )
    if autosave:
        saved = save_phase_artifacts(
            "Phase 12 Portfolio Capital Simulator",
            {
                "portfolio_summary_table": report.portfolio_summary_table,
                "allocation_plan_table": report.allocation_plan_table,
                "paper_portfolio_table": report.paper_portfolio_table,
                "conditional_real_capital_table": report.conditional_real_capital_table,
                "position_sizing_table": report.position_sizing_table,
                "risk_budget_table": report.risk_budget_table,
                "portfolio_drawdown_stress_table": report.portfolio_drawdown_stress_table,
                "correlation_concentration_table": report.correlation_concentration_table,
                "cost_slippage_stress_table": report.cost_slippage_stress_table,
                "stop_exit_plan_table": report.stop_exit_plan_table,
                "capital_blocker_table": report.capital_blocker_table,
                "scenario_analysis_table": report.scenario_analysis_table,
                "next_actions_table": report.next_actions_table,
                "warning_table": report.warning_table,
                "input_source_table": report.input_source_table,
            },
            inputs={},
            config=settings,
            warnings=report.warning_table["WarningType"].dropna().astype(str).unique().tolist() if not report.warning_table.empty else [],
        )
        report.saved_artifacts = saved
    return report


__all__ = [
    "PortfolioCapitalSimulatorReport",
    "PORTFOLIO_SUMMARY_COLUMNS",
    "ALLOCATION_PLAN_COLUMNS",
    "POSITION_SIZING_COLUMNS",
    "RISK_BUDGET_COLUMNS",
    "STRESS_COLUMNS",
    "CONCENTRATION_COLUMNS",
    "COST_STRESS_COLUMNS",
    "STOP_EXIT_COLUMNS",
    "CAPITAL_BLOCKER_COLUMNS",
    "SCENARIO_COLUMNS",
    "NEXT_ACTION_COLUMNS",
    "WARNING_COLUMNS",
    "run_portfolio_capital_simulator",
]
