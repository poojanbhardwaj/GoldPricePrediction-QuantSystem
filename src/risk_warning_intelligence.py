"""Phase 13 risk and warning intelligence dashboard backend.

This module turns warning-heavy research artifacts into ranked, grouped,
actionable risk intelligence. It does not change signals, capital gates, model
outputs, or evidence. It only classifies and summarizes what prior phases
already produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names
from src.artifact_store import build_input_source_table, resolve_artifact, save_phase_artifacts


RISK_INTELLIGENCE_PHASE_NAME = "phase13_risk_warning_intelligence"
RISK_INTELLIGENCE_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)

RISK_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "RiskCategory",
    "Severity",
    "RiskScore",
    "AffectedAssets",
    "AffectedHorizons",
    "WarningCount",
    "CapitalImpact",
    "PaperImpact",
    "MainReason",
    "RecommendedAction",
)

TOP_RISKS_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "RiskCategory",
    "Severity",
    "RiskScore",
    "Asset",
    "Horizon",
    "CapitalImpact",
    "PaperImpact",
    "EvidenceSource",
    "MainWarning",
    "RecommendedAction",
)

CAPITAL_BLOCKING_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "BlockingReason",
    "Severity",
    "EvidenceSource",
    "RequiredImprovement",
    "CanPaperTrade",
)

PAPER_ONLY_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "PaperStatus",
    "PaperRiskReason",
    "SuggestedPaperHandling",
    "ReviewTrigger",
)

WARNING_GROUP_COLUMNS: Tuple[str, ...] = (
    "WarningType",
    "Severity",
    "Count",
    "AffectedAssets",
    "AffectedHorizons",
    "FirstSeenSource",
    "MainImpact",
    "RecommendedAction",
)

RISK_MATRIX_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "OverallRiskScore",
    "CapitalStatus",
    "PaperStatus",
    "TopRisk",
    "WarningCount",
    "DrawdownRisk",
    "ProbabilityRisk",
    "BenchmarkRisk",
    "EvidenceRisk",
    "CostRisk",
    "NextAction",
)

RISK_STATUS_COLUMNS: Tuple[str, ...] = (
    "RiskCategory",
    "Status",
    "EvidenceAvailable",
    "Explanation",
)

NEXT_ACTION_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "Action",
    "WhyItMatters",
    "ExpectedBenefit",
    "AffectedAssets",
    "AffectedHorizons",
    "Urgency",
    "DependsOn",
)

RAW_WARNING_COLUMNS: Tuple[str, ...] = (
    "EvidenceSource",
    "Asset",
    "Horizon",
    "WarningType",
    "RiskCategory",
    "Severity",
    "RiskScore",
    "CapitalImpact",
    "PaperImpact",
    "Message",
    "RecommendedAction",
)


@dataclass
class RiskWarningIntelligenceReport:
    risk_summary_table: pd.DataFrame
    top_risks_table: pd.DataFrame
    capital_blocking_risks_table: pd.DataFrame
    paper_only_risks_table: pd.DataFrame
    warning_group_table: pd.DataFrame
    asset_horizon_risk_matrix: pd.DataFrame
    risk_trend_or_status_table: pd.DataFrame
    next_risk_actions_table: pd.DataFrame
    raw_warning_evidence: pd.DataFrame
    input_source_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: Dict[str, Any] = field(default_factory=dict)
    saved_artifacts: Dict[str, Any] = field(default_factory=dict)


INPUT_SPECS: Dict[str, Tuple[str, str, bool]] = {
    "probability_calibration_summary": ("Phase 8F Probability Calibration", "probability_calibration_summary", False),
    "probability_calibration_warnings": ("Phase 8F Probability Calibration", "probability_calibration_warnings", False),
    "forward_signal_log": ("Phase 9 Forward Paper Evidence", "forward_signal_log", False),
    "forward_warning_table": ("Phase 9 Forward Paper Evidence", "forward_warning_table", False),
    "pending_outcome_table": ("Phase 9 Forward Paper Evidence", "pending_outcome_table", False),
    "matured_outcome_table": ("Phase 9 Forward Paper Evidence", "matured_outcome_table", False),
    "forward_accuracy_summary": ("Phase 9 Forward Paper Evidence", "forward_accuracy_summary", False),
    "forward_probability_calibration_summary": ("Phase 9 Forward Paper Evidence", "forward_probability_calibration_summary", False),
    "phase10_warnings_table": ("Phase 10 Actionable Research Plan", "warnings_table", False),
    "phase10_blocked_candidates_table": ("Phase 10 Actionable Research Plan", "blocked_candidates_table", False),
    "phase10_risk_budget_table": ("Phase 10 Actionable Research Plan", "risk_budget_table", False),
    "phase10_paper_trade_plan_table": ("Phase 10 Actionable Research Plan", "paper_trade_plan_table", False),
    "phase10_watchlist_table": ("Phase 10 Actionable Research Plan", "watchlist_table", False),
    "capital_eligibility_table": ("Phase 11 Daily Research Control Center", "capital_eligibility_table", False),
    "phase11_capital_blocker_table": ("Phase 11 Daily Research Control Center", "capital_blocker_table", False),
    "phase11_warning_table": ("Phase 11 Daily Research Control Center", "warning_table", False),
    "phase11_pending_outcomes_table": ("Phase 11 Daily Research Control Center", "pending_outcomes_table", False),
    "matured_today_table": ("Phase 11 Daily Research Control Center", "matured_today_table", False),
    "evidence_health_table": ("Phase 11 Daily Research Control Center", "evidence_health_table", False),
    "allocation_plan_table": ("Phase 12 Portfolio Capital Simulator", "allocation_plan_table", False),
    "paper_portfolio_table": ("Phase 12 Portfolio Capital Simulator", "paper_portfolio_table", False),
    "phase12_warning_table": ("Phase 12 Portfolio Capital Simulator", "warning_table", False),
    "phase12_capital_blocker_table": ("Phase 12 Portfolio Capital Simulator", "capital_blocker_table", False),
    "portfolio_drawdown_stress_table": ("Phase 12 Portfolio Capital Simulator", "portfolio_drawdown_stress_table", False),
    "correlation_concentration_table": ("Phase 12 Portfolio Capital Simulator", "correlation_concentration_table", False),
    "cost_slippage_stress_table": ("Phase 12 Portfolio Capital Simulator", "cost_slippage_stress_table", False),
    "scenario_analysis_table": ("Phase 12 Portfolio Capital Simulator", "scenario_analysis_table", False),
    "phase12_risk_budget_table": ("Phase 12 Portfolio Capital Simulator", "risk_budget_table", False),
}

SEVERITY_RANK = {"Info": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}
CAPITAL_IMPACT_RANK = {"Unknown": 0, "NoRealCapitalImpact": 1, "ReducesRealCapital": 2, "BlocksRealCapital": 3}
PAPER_IMPACT_RANK = {"NoPaperImpact": 0, "MonitorOnly": 1, "PaperOnlyAllowed": 2, "ReducesPaperSize": 3, "BlocksPaper": 4}

RISK_RULES: Dict[str, Dict[str, Any]] = {
    "ComplianceNotice": {
        "Severity": "Info",
        "RiskScore": 2,
        "CapitalImpact": "NoRealCapitalImpact",
        "PaperImpact": "NoPaperImpact",
        "RecommendedAction": "Keep disclaimer visible; do not treat as a model or data failure.",
    },
    "ResearchDeploymentLimit": {
        "Severity": "Medium",
        "RiskScore": 35,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "MonitorOnly",
        "RecommendedAction": "Keep research-only deployment limits visible while continuing evidence collection.",
    },
    "ProbabilityUnreliable": {
        "Severity": "High",
        "RiskScore": 72,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "MonitorOnly",
        "RecommendedAction": "Recalibrate probability model before using confidence.",
    },
    "Overconfident": {
        "Severity": "High",
        "RiskScore": 76,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "ReducesPaperSize",
        "RecommendedAction": "Reduce confidence use and inspect high-confidence failures.",
    },
    "BenchmarkDominated": {
        "Severity": "High",
        "RiskScore": 70,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "MonitorOnly",
        "RecommendedAction": "Keep real capital blocked until benchmark dominance improves.",
    },
    "DrawdownRisk": {
        "Severity": "High",
        "RiskScore": 68,
        "CapitalImpact": "ReducesRealCapital",
        "PaperImpact": "ReducesPaperSize",
        "RecommendedAction": "Reduce paper allocation due to drawdown stress.",
    },
    "CostFragile": {
        "Severity": "Medium",
        "RiskScore": 58,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "ReducesPaperSize",
        "RecommendedAction": "Use cost stress before increasing paper exposure.",
    },
    "SplitUnstable": {
        "Severity": "High",
        "RiskScore": 66,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "MonitorOnly",
        "RecommendedAction": "Run wider walk-forward validation before escalation.",
    },
    "LowTradeCount": {
        "Severity": "Medium",
        "RiskScore": 48,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "MonitorOnly",
        "RecommendedAction": "Continue paper tracking until more forward outcomes mature.",
    },
    "OverFiltered": {
        "Severity": "Medium",
        "RiskScore": 45,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "MonitorOnly",
        "RecommendedAction": "Review threshold and cooldown filters for excessive coverage loss.",
    },
    "ReturnDestroyed": {
        "Severity": "High",
        "RiskScore": 74,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "ReducesPaperSize",
        "RecommendedAction": "Remove or avoid candidate until evidence improves.",
    },
    "NoImprovement": {
        "Severity": "Medium",
        "RiskScore": 52,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "MonitorOnly",
        "RecommendedAction": "Keep candidate on watchlist only.",
    },
    "PendingEvidenceOnly": {
        "Severity": "Medium",
        "RiskScore": 38,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "PaperOnlyAllowed",
        "RecommendedAction": "Review asset/horizon after next matured outcome date.",
    },
    "RealCapitalBlocked": {
        "Severity": "High",
        "RiskScore": 70,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "PaperOnlyAllowed",
        "RecommendedAction": "Keep real capital blocked while continuing research tracking where evidence exists.",
    },
    "ConcentrationRisk": {
        "Severity": "Medium",
        "RiskScore": 55,
        "CapitalImpact": "ReducesRealCapital",
        "PaperImpact": "ReducesPaperSize",
        "RecommendedAction": "Reduce concentration and diversify paper tracking.",
    },
    "HorizonConcentration": {
        "Severity": "Medium",
        "RiskScore": 50,
        "CapitalImpact": "ReducesRealCapital",
        "PaperImpact": "ReducesPaperSize",
        "RecommendedAction": "Reduce horizon concentration before increasing exposure.",
    },
    "DataQualityRisk": {
        "Severity": "Critical",
        "RiskScore": 88,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "BlocksPaper",
        "RecommendedAction": "Fix missing or invalid data before using this evidence.",
    },
    "EvidenceInsufficient": {
        "Severity": "Medium",
        "RiskScore": 42,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "MonitorOnly",
        "RecommendedAction": "Collect more forward evidence before changing status.",
    },
    "CalibrationWeak": {
        "Severity": "Medium",
        "RiskScore": 56,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "MonitorOnly",
        "RecommendedAction": "Recalibrate probability model before using confidence.",
    },
    "ForwardEvidenceYoung": {
        "Severity": "Medium",
        "RiskScore": 40,
        "CapitalImpact": "BlocksRealCapital",
        "PaperImpact": "PaperOnlyAllowed",
        "RecommendedAction": "Continue paper tracking until more forward outcomes mature.",
    },
}


def _empty_frame(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _to_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value)


def _normalise_horizon(df: pd.DataFrame) -> pd.DataFrame:
    out = _to_frame(df)
    if out.empty:
        return out
    if "Horizon" in out.columns:
        out["Horizon"] = out["Horizon"].astype(str).str.replace("D", "", regex=False)
        out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce")
    return out


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _split_warning_values(*values: Any) -> List[str]:
    warnings: List[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            warnings.extend(_split_warning_values(*value))
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "false", "0"}:
            continue
        for sep in [";", "|", ","]:
            text = text.replace(sep, "\n")
        warnings.extend(part.strip() for part in text.splitlines() if part.strip())
    clean: List[str] = []
    for warning in warnings:
        if warning not in clean:
            clean.append(warning)
    return clean


def _best_by_rank(values: Iterable[str], rank: Dict[str, int], default: str) -> str:
    best = default
    best_rank = rank.get(default, -1)
    for value in values:
        candidate = str(value or default)
        candidate_rank = rank.get(candidate, -1)
        if candidate_rank > best_rank:
            best = candidate
            best_rank = candidate_rank
    return best


def _category_from_text(text: Any) -> str:
    raw = str(text or "").strip()
    lower = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
    readable = raw.lower()
    if lower in {"notfinancialadvice", "researchonly", "noliveapproval", "nolivetradingapproval", "compliancenotice"}:
        return "ComplianceNotice"
    if lower in {"notproductionready", "researchdeploymentlimit"}:
        return "ResearchDeploymentLimit"
    checks = [
        ("Overconfident", ["overconfident", "highconfidence", "confidencefailure"]),
        ("ProbabilityUnreliable", ["probabilityunreliable", "probabilitystillunreliable", "brier", "probabilityrisk"]),
        ("CalibrationWeak", ["calibrationweak", "calibration", "ece", "calibrationgrade"]),
        ("BenchmarkDominated", ["benchmarkdominated", "benchmarkunderperformance", "benchmark", "vsbuyhold"]),
        ("DrawdownRisk", ["drawdownrisk", "drawdown", "losscap", "maxloss"]),
        ("CostFragile", ["costfragile", "slippage", "highercost", "coststress"]),
        ("SplitUnstable", ["splitunstable", "windowunstable", "stability", "unstable"]),
        ("LowTradeCount", ["lowtradecount", "notenoughforwardevidence", "tradecount"]),
        ("OverFiltered", ["overfiltered", "coverage loss", "coverage"]),
        ("ReturnDestroyed", ["returndestroyed", "return destroyed"]),
        ("NoImprovement", ["noimprovement", "no improvement"]),
        ("PendingEvidenceOnly", ["pendingevidenceonly", "pendingoutcome", "outcomenotmatured"]),
        ("ForwardEvidenceYoung", ["forwardevidenceyoung", "forwardevidence", "maturedforward"]),
        ("RealCapitalBlocked", ["realcapitalblocked", "capitalblocked", "blockedrealcapital", "norealcapital", "capitalgate", "capitalblocker"]),
        ("ConcentrationRisk", ["concentrationrisk", "assetconcentration"]),
        ("HorizonConcentration", ["horizonconcentration", "correlationrisk", "horizon exposure"]),
        (
            "DataQualityRisk",
            [
                "dataquality",
                "missingexitprice",
                "missingentryprice",
                "missingprice",
                "missingcoreprice",
                "invalidtarget",
                "invalidprice",
                "corruptedinput",
                "stalecriticaldata",
                "missingartifact",
            ],
        ),
        ("EvidenceInsufficient", ["evidenceinsufficient", "insufficientevidence", "lowevidence", "not enough"]),
    ]
    for category, tokens in checks:
        if any(token in lower or token in readable for token in tokens):
            return category
    if raw in RISK_RULES:
        return raw
    return "EvidenceInsufficient"


def _rule_for_category(category: str) -> Dict[str, Any]:
    return RISK_RULES.get(category, RISK_RULES["EvidenceInsufficient"])


def _severity_from_score(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 35:
        return "Medium"
    if score >= 10:
        return "Low"
    return "Info"


def _message_from_row(row: pd.Series) -> str:
    for col in [
        "Message",
        "MainReason",
        "Reason",
        "ZeroWeightReason",
        "RequiredImprovement",
        "WhatWouldAllowRealCapital",
        "Notes",
        "StressResult",
        "RiskWarning",
        "PortfolioContributionReason",
        "ExpectedAction",
    ]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return "Warning evidence from upstream research artifact."


def _asset_from_row(row: pd.Series) -> str:
    if "Asset" in row.index and pd.notna(row["Asset"]) and str(row["Asset"]).strip():
        return str(row["Asset"]).strip()
    return "ALL"


def _horizon_from_row(row: pd.Series) -> Any:
    if "Horizon" in row.index and pd.notna(row["Horizon"]):
        value = _safe_float(row["Horizon"], np.nan)
        if np.isfinite(value):
            return int(value)
    return "ALL"


def _warning_values_from_row(row: pd.Series, source_key: str) -> List[str]:
    candidates: List[Any] = []
    for col in [
        "WarningType",
        "MainWarning",
        "MainWarnings",
        "MainRiskWarning",
        "RiskWarning",
        "Warning",
        "Warnings",
        "CalibrationGrade",
        "CalibrationVerdict",
        "MainCapitalBlocker",
        "BlockingReason",
        "FailedGates",
        "PaperRiskReason",
    ]:
        if col in row.index:
            candidates.append(row[col])
    values = _split_warning_values(*candidates)
    if values:
        return values
    if "capital" in source_key and str(row.get("RealCapitalAllowed", "")).lower() in {"false", "0", "no"}:
        return ["RealCapitalBlocked"]
    if "pending" in source_key:
        return ["PendingEvidenceOnly"]
    if "drawdown" in source_key:
        return ["DrawdownRisk"]
    if "cost" in source_key:
        return ["CostFragile"]
    if "concentration" in source_key:
        return ["ConcentrationRisk"]
    return []


def _source_label(key: str) -> str:
    labels = {
        "probability_calibration_summary": "Phase 8F calibration summary",
        "probability_calibration_warnings": "Phase 8F calibration warnings",
        "forward_signal_log": "Phase 9 forward signal log",
        "forward_warning_table": "Phase 9 forward warnings",
        "pending_outcome_table": "Phase 9 pending outcomes",
        "matured_outcome_table": "Phase 9 matured outcomes",
        "forward_accuracy_summary": "Phase 9 forward accuracy",
        "forward_probability_calibration_summary": "Phase 9 forward calibration",
        "phase10_warnings_table": "Phase 10 warnings",
        "phase10_blocked_candidates_table": "Phase 10 blockers",
        "phase10_risk_budget_table": "Phase 10 risk budget",
        "phase10_paper_trade_plan_table": "Phase 10 paper plan",
        "phase10_watchlist_table": "Phase 10 watchlist",
        "capital_eligibility_table": "Phase 11 capital eligibility",
        "phase11_capital_blocker_table": "Phase 11 capital blockers",
        "phase11_warning_table": "Phase 11 warnings",
        "phase11_pending_outcomes_table": "Phase 11 pending outcomes",
        "matured_today_table": "Phase 11 matured outcomes",
        "evidence_health_table": "Phase 11 evidence health",
        "allocation_plan_table": "Phase 12 allocation plan",
        "paper_portfolio_table": "Phase 12 paper portfolio",
        "phase12_warning_table": "Phase 12 warnings",
        "phase12_capital_blocker_table": "Phase 12 capital blockers",
        "portfolio_drawdown_stress_table": "Phase 12 drawdown stress",
        "correlation_concentration_table": "Phase 12 concentration",
        "cost_slippage_stress_table": "Phase 12 cost stress",
        "scenario_analysis_table": "Phase 12 scenarios",
        "phase12_risk_budget_table": "Phase 12 risk budget",
    }
    return labels.get(key, key)


def _extract_rows_from_table(key: str, table: pd.DataFrame) -> List[Dict[str, Any]]:
    df = _normalise_horizon(table)
    if df.empty:
        return []
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        warning_values = _warning_values_from_row(row, key)
        if key == "forward_signal_log" and str(row.get("Status", "")).lower() == "pending":
            warning_values.append("PendingEvidenceOnly")
        if key == "allocation_plan_table" and str(row.get("AllocationMode", "")).lower() == "paperonly":
            warning_values.append("RealCapitalBlocked")
        if key == "capital_eligibility_table" and str(row.get("RealCapitalAllowed", "")).lower() in {"false", "0", "no"}:
            warning_values.append("RealCapitalBlocked")
        warning_values = _split_warning_values(warning_values)
        for warning in warning_values:
            category = _category_from_text(warning)
            rule = _rule_for_category(category)
            score = float(rule["RiskScore"])
            if category != "ComplianceNotice" and "Severity" in row.index and pd.notna(row["Severity"]):
                row_severity = str(row["Severity"])
                if SEVERITY_RANK.get(row_severity, -1) > SEVERITY_RANK.get(str(rule["Severity"]), -1):
                    score = max(score, {"Critical": 88, "High": 68, "Medium": 45, "Low": 22, "Info": 5}.get(row_severity, score))
            severity = _severity_from_score(score)
            rows.append(
                {
                    "EvidenceSource": _source_label(key),
                    "Asset": _asset_from_row(row),
                    "Horizon": _horizon_from_row(row),
                    "WarningType": str(warning),
                    "RiskCategory": category,
                    "Severity": severity,
                    "RiskScore": round(score, 4),
                    "CapitalImpact": rule["CapitalImpact"],
                    "PaperImpact": rule["PaperImpact"],
                    "Message": _message_from_row(row),
                    "RecommendedAction": rule["RecommendedAction"],
                }
            )
    return rows


def _resolve_inputs(
    use_artifact_store: bool,
    prefer_uploaded: bool,
    uploaded_overrides: Optional[Dict[str, Any]],
    direct_tables: Dict[str, Any],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    uploaded_overrides = uploaded_overrides or {}
    resolved_rows: List[Dict[str, Any]] = []
    tables: Dict[str, pd.DataFrame] = {}
    for key, (phase, artifact, required) in INPUT_SPECS.items():
        direct = direct_tables.get(key)
        if direct is not None:
            df = _normalise_horizon(direct)
            tables[key] = df
            resolved_rows.append(
                {
                    "Artifact": artifact,
                    "Phase": phase,
                    "Source": "DirectInput",
                    "RunId": "",
                    "Rows": int(len(df)),
                    "CreatedAt": "",
                    "Status": "Loaded",
                    "Path": "",
                }
            )
            continue
        if use_artifact_store or uploaded_overrides.get(key) is not None:
            resolved = resolve_artifact(
                phase,
                artifact,
                uploaded_file=uploaded_overrides.get(key),
                prefer_uploaded=prefer_uploaded,
                required=required,
            )
            data = resolved.get("Data")
            tables[key] = _normalise_horizon(data) if data is not None else pd.DataFrame()
            resolved_rows.append({k: v for k, v in resolved.items() if k != "Data"})
        else:
            tables[key] = pd.DataFrame()
            resolved_rows.append(
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
    return tables, build_input_source_table(resolved_rows)


def _build_raw_warning_evidence(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key, table in tables.items():
        rows.extend(_extract_rows_from_table(key, table))
    raw = pd.DataFrame(rows, columns=list(RAW_WARNING_COLUMNS))
    if raw.empty:
        return _empty_frame(RAW_WARNING_COLUMNS)
    raw = raw.drop_duplicates().reset_index(drop=True)
    raw["RiskScore"] = pd.to_numeric(raw["RiskScore"], errors="coerce").fillna(0.0)
    return raw[list(RAW_WARNING_COLUMNS)]


def _join_unique(values: Iterable[Any]) -> str:
    clean = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value)
        if not text or text == "ALL":
            continue
        if text not in clean:
            clean.append(text)
    return "; ".join(clean) if clean else "ALL"


def _build_risk_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return _empty_frame(RISK_SUMMARY_COLUMNS)
    rows: List[Dict[str, Any]] = []
    for category, group in raw.groupby("RiskCategory", dropna=False):
        sorted_group = group.sort_values("RiskScore", ascending=False)
        top = sorted_group.iloc[0]
        rows.append(
            {
                "RiskCategory": category,
                "Severity": _best_by_rank(group["Severity"], SEVERITY_RANK, "Info"),
                "RiskScore": round(float(group["RiskScore"].max()), 4),
                "AffectedAssets": _join_unique(group["Asset"]),
                "AffectedHorizons": _join_unique(group["Horizon"]),
                "WarningCount": int(len(group)),
                "CapitalImpact": _best_by_rank(group["CapitalImpact"], CAPITAL_IMPACT_RANK, "Unknown"),
                "PaperImpact": _best_by_rank(group["PaperImpact"], PAPER_IMPACT_RANK, "NoPaperImpact"),
                "MainReason": str(top["Message"]),
                "RecommendedAction": str(top["RecommendedAction"]),
            }
        )
    return pd.DataFrame(rows, columns=list(RISK_SUMMARY_COLUMNS)).sort_values("RiskScore", ascending=False).reset_index(drop=True)


def _build_top_risks(raw: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    if raw.empty:
        return _empty_frame(TOP_RISKS_COLUMNS)
    top = raw.sort_values(["RiskScore", "Severity"], ascending=[False, False]).head(int(limit)).copy()
    top.insert(0, "Rank", range(1, len(top) + 1))
    out = top.rename(columns={"WarningType": "MainWarning"})
    return out[list(TOP_RISKS_COLUMNS)]


def _build_capital_blocking(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return _empty_frame(CAPITAL_BLOCKING_COLUMNS)
    subset = raw[raw["CapitalImpact"].isin(["BlocksRealCapital", "ReducesRealCapital"])].copy()
    if subset.empty:
        return _empty_frame(CAPITAL_BLOCKING_COLUMNS)
    rows = []
    for _, row in subset.sort_values("RiskScore", ascending=False).iterrows():
        rows.append(
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "BlockingReason": row["WarningType"],
                "Severity": row["Severity"],
                "EvidenceSource": row["EvidenceSource"],
                "RequiredImprovement": row["RecommendedAction"],
                "CanPaperTrade": row["PaperImpact"] in {"PaperOnlyAllowed", "MonitorOnly", "ReducesPaperSize", "NoPaperImpact"},
            }
        )
    return pd.DataFrame(rows, columns=list(CAPITAL_BLOCKING_COLUMNS)).drop_duplicates().reset_index(drop=True)


def _build_paper_only(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return _empty_frame(PAPER_ONLY_COLUMNS)
    subset = raw[raw["PaperImpact"].isin(["PaperOnlyAllowed", "MonitorOnly", "ReducesPaperSize"])].copy()
    if subset.empty:
        return _empty_frame(PAPER_ONLY_COLUMNS)
    rows = []
    for _, row in subset.sort_values("RiskScore", ascending=False).iterrows():
        if row["PaperImpact"] == "ReducesPaperSize":
            status = "ReducePaperSize"
        elif row["PaperImpact"] == "PaperOnlyAllowed":
            status = "PaperOnlyAllowed"
        else:
            status = "MonitorOnly"
        rows.append(
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "PaperStatus": status,
                "PaperRiskReason": row["WarningType"],
                "SuggestedPaperHandling": row["RecommendedAction"],
                "ReviewTrigger": "Review after next matured outcome or repeated warning.",
            }
        )
    return pd.DataFrame(rows, columns=list(PAPER_ONLY_COLUMNS)).drop_duplicates().reset_index(drop=True)


def _build_warning_groups(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return _empty_frame(WARNING_GROUP_COLUMNS)
    rows = []
    for warning_type, group in raw.groupby("WarningType", dropna=False):
        top = group.sort_values("RiskScore", ascending=False).iloc[0]
        rows.append(
            {
                "WarningType": warning_type,
                "Severity": _best_by_rank(group["Severity"], SEVERITY_RANK, "Info"),
                "Count": int(len(group)),
                "AffectedAssets": _join_unique(group["Asset"]),
                "AffectedHorizons": _join_unique(group["Horizon"]),
                "FirstSeenSource": str(group["EvidenceSource"].iloc[0]),
                "MainImpact": f"{top['CapitalImpact']} / {top['PaperImpact']}",
                "RecommendedAction": str(top["RecommendedAction"]),
            }
        )
    return pd.DataFrame(rows, columns=list(WARNING_GROUP_COLUMNS)).sort_values(["Count", "Severity"], ascending=[False, False]).reset_index(drop=True)


def _flag_category(group: pd.DataFrame, categories: Iterable[str]) -> str:
    return "Yes" if not group[group["RiskCategory"].isin(list(categories))].empty else "No"


def _asset_horizon_subset(df: pd.DataFrame, asset: str, horizon: int) -> pd.DataFrame:
    if df.empty or "Asset" not in df.columns or "Horizon" not in df.columns:
        return pd.DataFrame()
    h = pd.to_numeric(df["Horizon"].astype(str).str.replace("D", "", regex=False), errors="coerce")
    return df[df["Asset"].astype(str).eq(str(asset)) & h.eq(int(horizon))].copy()


def _allocation_status_for_asset_horizon(tables: Dict[str, pd.DataFrame], asset: str, horizon: int) -> str:
    allocation = _asset_horizon_subset(tables.get("allocation_plan_table", pd.DataFrame()), asset, horizon)
    paper_portfolio = _asset_horizon_subset(tables.get("paper_portfolio_table", pd.DataFrame()), asset, horizon)
    if not paper_portfolio.empty:
        paper_weight = pd.to_numeric(paper_portfolio.get("SuggestedPaperWeightPct", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        if float(paper_weight.max()) > 0:
            return "PaperAllocated"
    if allocation.empty:
        return ""
    paper_weight = pd.to_numeric(allocation.get("SuggestedPaperWeightPct", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    if float(paper_weight.max()) > 0:
        return "PaperAllocated"
    action_values = " ".join(allocation.get("ResearchAction", pd.Series(dtype=str)).fillna("").astype(str).tolist()).lower()
    status_values = " ".join(allocation.get("PaperAllocationStatus", pd.Series(dtype=str)).fillna("").astype(str).tolist()).lower()
    mode_values = " ".join(allocation.get("AllocationMode", pd.Series(dtype=str)).fillna("").astype(str).tolist()).lower()
    if "watchlist" in action_values or "watchlist" in mode_values:
        return "WatchlistOnly"
    if "papertradeonly" in action_values or "allocated" in status_values or "eligiblebutnotallocated" in status_values or "paperonly" in mode_values:
        return "PaperOnlyAllowed"
    return ""


def _has_severe_exact_data_failure(group: pd.DataFrame) -> bool:
    if group.empty:
        return False
    severe = group[
        group["RiskCategory"].eq("DataQualityRisk")
        & group["WarningType"].astype(str).str.lower().str.contains(
            "missingexitprice|missingentryprice|missingprice|missingcoreprice|invalidtarget|invalidprice|corruptedinput|stalecriticaldata",
            regex=True,
            na=False,
        )
    ]
    return not severe.empty


def _paper_status_from_evidence(paper_impact: str, group: pd.DataFrame, allocation_status: str) -> str:
    if _has_severe_exact_data_failure(group):
        return "Blocked"
    if allocation_status:
        return allocation_status
    if paper_impact == "ReducesPaperSize":
        return "ReducePaperSize"
    if paper_impact == "BlocksPaper":
        return "Blocked"
    if paper_impact == "PaperOnlyAllowed":
        return "PaperOnlyAllowed"
    if paper_impact == "MonitorOnly":
        return "MonitorOnly"
    return "MonitorOnly"


def _build_risk_matrix(raw: pd.DataFrame, assets: Iterable[str], horizons: Iterable[int], tables: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    tables = tables or {}
    rows: List[Dict[str, Any]] = []
    for asset in assets:
        for horizon in horizons:
            if raw.empty:
                group = pd.DataFrame(columns=list(RAW_WARNING_COLUMNS))
            else:
                h = pd.to_numeric(raw["Horizon"].astype(str).str.replace("D", "", regex=False), errors="coerce")
                group = raw[raw["Asset"].astype(str).eq(str(asset)) & h.eq(int(horizon))].copy()
            allocation_status = _allocation_status_for_asset_horizon(tables, asset, int(horizon))
            if group.empty:
                rows.append(
                    {
                        "Asset": asset,
                        "Horizon": int(horizon),
                        "OverallRiskScore": 0.0,
                        "CapitalStatus": "NoSpecificBlocker",
                        "PaperStatus": allocation_status or "MonitorOnly",
                        "TopRisk": "None",
                        "WarningCount": 0,
                        "DrawdownRisk": "No",
                        "ProbabilityRisk": "No",
                        "BenchmarkRisk": "No",
                        "EvidenceRisk": "No",
                        "CostRisk": "No",
                        "NextAction": "Monitor future evidence.",
                    }
                )
                continue
            top = group.sort_values("RiskScore", ascending=False).iloc[0]
            capital_impact = _best_by_rank(group["CapitalImpact"], CAPITAL_IMPACT_RANK, "Unknown")
            paper_impact = _best_by_rank(group["PaperImpact"], PAPER_IMPACT_RANK, "NoPaperImpact")
            capital_status = "Blocked" if capital_impact == "BlocksRealCapital" else "Reduced" if capital_impact == "ReducesRealCapital" else "NoSpecificBlocker"
            paper_status = _paper_status_from_evidence(paper_impact, group, allocation_status)
            rows.append(
                {
                    "Asset": asset,
                    "Horizon": int(horizon),
                    "OverallRiskScore": round(float(group["RiskScore"].max()), 4),
                    "CapitalStatus": capital_status,
                    "PaperStatus": paper_status,
                    "TopRisk": str(top["RiskCategory"]),
                    "WarningCount": int(len(group)),
                    "DrawdownRisk": _flag_category(group, ["DrawdownRisk"]),
                    "ProbabilityRisk": _flag_category(group, ["ProbabilityUnreliable", "Overconfident", "CalibrationWeak"]),
                    "BenchmarkRisk": _flag_category(group, ["BenchmarkDominated"]),
                    "EvidenceRisk": _flag_category(group, ["LowTradeCount", "EvidenceInsufficient", "PendingEvidenceOnly", "ForwardEvidenceYoung", "DataQualityRisk"]),
                    "CostRisk": _flag_category(group, ["CostFragile"]),
                    "NextAction": str(top["RecommendedAction"]),
                }
            )
    return pd.DataFrame(rows, columns=list(RISK_MATRIX_COLUMNS))


def _build_risk_status(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    categories = list(RISK_RULES.keys())
    for category in categories:
        match = summary[summary["RiskCategory"].astype(str).eq(category)] if not summary.empty else pd.DataFrame()
        if match.empty:
            rows.append(
                {
                    "RiskCategory": category,
                    "Status": "InsufficientHistory",
                    "EvidenceAvailable": False,
                    "Explanation": "No current evidence for this category in loaded artifacts.",
                }
            )
        else:
            rows.append(
                {
                    "RiskCategory": category,
                    "Status": "InsufficientHistory",
                    "EvidenceAvailable": True,
                    "Explanation": "Current snapshot available; trend needs multiple saved Phase 13 runs.",
                }
            )
    return pd.DataFrame(rows, columns=list(RISK_STATUS_COLUMNS))


def _urgency_from_score(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def _build_next_actions(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return _empty_frame(NEXT_ACTION_COLUMNS)
    rows = []
    for _, row in summary.sort_values("RiskScore", ascending=False).iterrows():
        category = str(row["RiskCategory"])
        action = str(row["RecommendedAction"])
        rows.append(
            {
                "Rank": 0,
                "Action": action,
                "WhyItMatters": f"{category} affects {row['WarningCount']} warning rows.",
                "ExpectedBenefit": "Cleaner risk triage and better paper-evidence interpretation.",
                "AffectedAssets": row["AffectedAssets"],
                "AffectedHorizons": row["AffectedHorizons"],
                "Urgency": _urgency_from_score(float(row["RiskScore"])),
                "DependsOn": "More matured forward outcomes" if category in {"PendingEvidenceOnly", "ForwardEvidenceYoung", "LowTradeCount"} else "Upstream research evidence",
            }
        )
    actions = pd.DataFrame(rows, columns=list(NEXT_ACTION_COLUMNS))
    if not actions.empty:
        actions = actions.drop_duplicates(subset=["Action", "AffectedAssets", "AffectedHorizons"]).reset_index(drop=True)
        actions["Rank"] = range(1, len(actions) + 1)
    return actions


def run_risk_warning_intelligence(
    *,
    use_artifact_store: bool = False,
    prefer_uploaded: bool = False,
    uploaded_overrides: Optional[Dict[str, Any]] = None,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    autosave: bool = False,
    **direct_tables: Any,
) -> RiskWarningIntelligenceReport:
    """Build Phase 13 risk and warning intelligence tables."""
    asset_list = list(assets or get_asset_names())
    horizon_list = [int(h) for h in (horizons or RISK_INTELLIGENCE_HORIZONS)]
    settings = {
        "phase": "13",
        "purpose": "risk_warning_intelligence",
        "assets": asset_list,
        "horizons": horizon_list,
        "use_artifact_store": bool(use_artifact_store),
        "live_deployment_approval": False,
    }
    tables, input_source_table = _resolve_inputs(bool(use_artifact_store), bool(prefer_uploaded), uploaded_overrides, direct_tables)
    raw = _build_raw_warning_evidence(tables)
    summary = _build_risk_summary(raw)
    top = _build_top_risks(raw)
    capital = _build_capital_blocking(raw)
    paper = _build_paper_only(raw)
    groups = _build_warning_groups(raw)
    matrix = _build_risk_matrix(raw, asset_list, horizon_list, tables)
    status = _build_risk_status(summary)
    actions = _build_next_actions(summary)

    report = RiskWarningIntelligenceReport(
        risk_summary_table=summary.reset_index(drop=True),
        top_risks_table=top.reset_index(drop=True),
        capital_blocking_risks_table=capital.reset_index(drop=True),
        paper_only_risks_table=paper.reset_index(drop=True),
        warning_group_table=groups.reset_index(drop=True),
        asset_horizon_risk_matrix=matrix.reset_index(drop=True),
        risk_trend_or_status_table=status.reset_index(drop=True),
        next_risk_actions_table=actions.reset_index(drop=True),
        raw_warning_evidence=raw.reset_index(drop=True),
        input_source_table=input_source_table.reset_index(drop=True),
        settings=settings,
    )
    if autosave:
        report.saved_artifacts = save_phase_artifacts(
            RISK_INTELLIGENCE_PHASE_NAME,
            {
                "risk_summary_table": report.risk_summary_table,
                "top_risks_table": report.top_risks_table,
                "capital_blocking_risks_table": report.capital_blocking_risks_table,
                "paper_only_risks_table": report.paper_only_risks_table,
                "warning_group_table": report.warning_group_table,
                "asset_horizon_risk_matrix": report.asset_horizon_risk_matrix,
                "risk_trend_or_status_table": report.risk_trend_or_status_table,
                "next_risk_actions_table": report.next_risk_actions_table,
                "raw_warning_evidence": report.raw_warning_evidence,
                "input_source_table": report.input_source_table,
            },
            inputs={},
            config=report.settings,
            warnings=report.raw_warning_evidence["WarningType"].dropna().astype(str).unique().tolist()
            if not report.raw_warning_evidence.empty
            else [],
        )
    return report


__all__ = [
    "RiskWarningIntelligenceReport",
    "run_risk_warning_intelligence",
    "RISK_INTELLIGENCE_PHASE_NAME",
]
