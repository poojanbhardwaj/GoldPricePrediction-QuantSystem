"""Phase 14 dynamic position sizing and risk optimizer.

This engine consumes Phase 12 paper allocation plus Phase 13 risk intelligence
and produces risk-adjusted simulated paper sizing. It does not execute trades,
alter model outputs, or relax real-capital gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names
from src.artifact_store import build_input_source_table, resolve_artifact, save_phase_artifacts


DYNAMIC_RISK_SIZING_PHASE_NAME = "phase14_dynamic_risk_sizing"
DYNAMIC_SIZING_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)

DYNAMIC_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "PortfolioMode",
    "StartingPaperExposurePct",
    "OptimizedPaperExposurePct",
    "PaperReservePct",
    "RealCapitalExposurePct",
    "NumberStartingPaperCandidates",
    "NumberOptimizedPaperCandidates",
    "NumberZeroedByRisk",
    "LargestRiskReductionReason",
    "OverallSizingVerdict",
    "MainReason",
)

DYNAMIC_POSITION_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "ResearchAction",
    "Phase12PaperWeightPct",
    "OptimizedPaperWeightPct",
    "SuggestedRealWeightPct",
    "RealCapitalAllowed",
    "PaperStatus",
    "CapitalStatus",
    "OverallRiskScore",
    "OverallRiskMultiplier",
    "DataQualityMultiplier",
    "ProbabilityReliabilityMultiplier",
    "DrawdownMultiplier",
    "BenchmarkMultiplier",
    "EvidenceMultiplier",
    "CostMultiplier",
    "ConcentrationMultiplier",
    "ForwardEvidenceMultiplier",
    "SizeChangePct",
    "SizingDecision",
    "ZeroSizeReason",
    "MainRiskReason",
    "SizingExplanation",
)

RISK_MULTIPLIER_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "RiskCategory",
    "Severity",
    "RiskScore",
    "AppliedMultiplier",
    "ImpactType",
    "Explanation",
)

RISK_MULTIPLIER_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "RiskCategory",
    "Severity",
    "AffectedAssets",
    "AffectedHorizons",
    "Count",
    "MinMultiplier",
    "MedianMultiplier",
    "MaxMultiplier",
    "MainImpactType",
    "Explanation",
)

CAP_ADJUSTMENT_COLUMNS: Tuple[str, ...] = (
    "CapType",
    "LimitPct",
    "BeforeAdjustmentPct",
    "AfterAdjustmentPct",
    "BreachBefore",
    "BreachAfter",
    "AffectedAssets",
    "AffectedHorizons",
    "AdjustmentReason",
)

ZERO_SIZE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "OriginalPaperWeightPct",
    "OptimizedPaperWeightPct",
    "ZeroSizeReason",
    "EvidenceSource",
    "RequiredFixBeforeSizing",
)

OPTIMIZED_PORTFOLIO_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "Asset",
    "Horizon",
    "OptimizedPaperWeightPct",
    "RiskScore",
    "RiskAdjustedScore",
    "MainRisk",
    "ReviewTrigger",
    "ExitOrReduceTrigger",
)

DRAWDOWN_BUDGET_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "OptimizedPaperWeightPct",
    "EstimatedDrawdownShockPct",
    "ContributionToPortfolioDrawdownPct",
    "DrawdownBudgetUsedPct",
    "DrawdownCapPct",
    "Breach",
    "Action",
)

SCENARIO_COLUMNS: Tuple[str, ...] = (
    "Scenario",
    "StartingPortfolioImpact",
    "OptimizedPortfolioImpact",
    "RiskReductionPct",
    "Explanation",
)

NEXT_ACTION_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "Action",
    "AffectedAssets",
    "AffectedHorizons",
    "WhyItMatters",
    "ExpectedEffect",
    "Urgency",
    "DependsOn",
)


@dataclass
class DynamicRiskSizingReport:
    dynamic_sizing_summary_table: pd.DataFrame
    dynamic_position_sizing_table: pd.DataFrame
    risk_multiplier_table: pd.DataFrame
    risk_multiplier_summary_table: pd.DataFrame
    cap_adjustment_table: pd.DataFrame
    zero_size_table: pd.DataFrame
    optimized_portfolio_table: pd.DataFrame
    drawdown_budget_table: pd.DataFrame
    risk_adjusted_scenarios_table: pd.DataFrame
    next_sizing_actions_table: pd.DataFrame
    input_source_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: Dict[str, Any] = field(default_factory=dict)
    saved_artifacts: Dict[str, Any] = field(default_factory=dict)


INPUT_SPECS: Dict[str, Tuple[str, str, bool]] = {
    "allocation_plan_table": ("Phase 12 Portfolio Capital Simulator", "allocation_plan_table", False),
    "paper_portfolio_table": ("Phase 12 Portfolio Capital Simulator", "paper_portfolio_table", False),
    "correlation_concentration_table": ("Phase 12 Portfolio Capital Simulator", "correlation_concentration_table", False),
    "portfolio_drawdown_stress_table": ("Phase 12 Portfolio Capital Simulator", "portfolio_drawdown_stress_table", False),
    "cost_slippage_stress_table": ("Phase 12 Portfolio Capital Simulator", "cost_slippage_stress_table", False),
    "scenario_analysis_table": ("Phase 12 Portfolio Capital Simulator", "scenario_analysis_table", False),
    "phase12_risk_budget_table": ("Phase 12 Portfolio Capital Simulator", "risk_budget_table", False),
    "position_sizing_table": ("Phase 12 Portfolio Capital Simulator", "position_sizing_table", False),
    "phase12_warning_table": ("Phase 12 Portfolio Capital Simulator", "warning_table", False),
    "phase12_capital_blocker_table": ("Phase 12 Portfolio Capital Simulator", "capital_blocker_table", False),
    "risk_summary_table": ("phase13_risk_warning_intelligence", "risk_summary_table", False),
    "top_risks_table": ("phase13_risk_warning_intelligence", "top_risks_table", False),
    "capital_blocking_risks_table": ("phase13_risk_warning_intelligence", "capital_blocking_risks_table", False),
    "paper_only_risks_table": ("phase13_risk_warning_intelligence", "paper_only_risks_table", False),
    "warning_group_table": ("phase13_risk_warning_intelligence", "warning_group_table", False),
    "asset_horizon_risk_matrix": ("phase13_risk_warning_intelligence", "asset_horizon_risk_matrix", False),
    "risk_trend_or_status_table": ("phase13_risk_warning_intelligence", "risk_trend_or_status_table", False),
    "next_risk_actions_table": ("phase13_risk_warning_intelligence", "next_risk_actions_table", False),
    "raw_warning_evidence": ("phase13_risk_warning_intelligence", "raw_warning_evidence", False),
}

SEVERE_DATA_WARNINGS = {
    "missingexitprice",
    "missingentryprice",
    "missingprice",
    "missingcoreprice",
    "invalidtarget",
    "invalidprice",
    "corruptedinput",
    "stalecriticaldata",
}

CATEGORY_TO_MULTIPLIER_BUCKET = {
    "DataQualityRisk": "DataQualityMultiplier",
    "ProbabilityUnreliable": "ProbabilityReliabilityMultiplier",
    "Overconfident": "ProbabilityReliabilityMultiplier",
    "CalibrationWeak": "ProbabilityReliabilityMultiplier",
    "DrawdownRisk": "DrawdownMultiplier",
    "BenchmarkDominated": "BenchmarkMultiplier",
    "EvidenceInsufficient": "EvidenceMultiplier",
    "LowTradeCount": "EvidenceMultiplier",
    "OverFiltered": "EvidenceMultiplier",
    "NoImprovement": "EvidenceMultiplier",
    "ReturnDestroyed": "EvidenceMultiplier",
    "CostFragile": "CostMultiplier",
    "ConcentrationRisk": "ConcentrationMultiplier",
    "HorizonConcentration": "ConcentrationMultiplier",
    "PendingEvidenceOnly": "ForwardEvidenceMultiplier",
    "ForwardEvidenceYoung": "ForwardEvidenceMultiplier",
    "SplitUnstable": "ForwardEvidenceMultiplier",
    "ComplianceNotice": "ComplianceNotice",
    "ResearchDeploymentLimit": "ResearchDeploymentLimit",
    "RealCapitalBlocked": "ResearchDeploymentLimit",
}

BASE_MULTIPLIERS = {
    "ComplianceNotice": 1.0,
    "ResearchDeploymentLimit": 1.0,
    "RealCapitalBlocked": 1.0,
    "DataQualityRisk": 0.0,
    "ProbabilityUnreliable": 0.45,
    "Overconfident": 0.40,
    "CalibrationWeak": 0.70,
    "DrawdownRisk": 0.65,
    "BenchmarkDominated": 0.65,
    "EvidenceInsufficient": 0.75,
    "LowTradeCount": 0.70,
    "OverFiltered": 0.75,
    "NoImprovement": 0.75,
    "ReturnDestroyed": 0.35,
    "CostFragile": 0.75,
    "ConcentrationRisk": 0.80,
    "HorizonConcentration": 0.80,
    "PendingEvidenceOnly": 0.85,
    "ForwardEvidenceYoung": 0.85,
    "SplitUnstable": 0.65,
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


def _subset(df: pd.DataFrame, asset: str, horizon: int) -> pd.DataFrame:
    if df.empty or "Asset" not in df.columns or "Horizon" not in df.columns:
        return pd.DataFrame()
    h = pd.to_numeric(df["Horizon"].astype(str).str.replace("D", "", regex=False), errors="coerce")
    return df[df["Asset"].astype(str).eq(str(asset)) & h.eq(int(horizon))].copy()


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


def _mode_portfolio_limit(portfolio_mode: str, requested_limit: Optional[float]) -> float:
    text = str(portfolio_mode or "Conservative").lower()
    mode_limit = 70.0 if "aggressive" in text else 45.0 if "balanced" in text else 25.0
    if requested_limit is None:
        return mode_limit
    requested = _safe_float(requested_limit, mode_limit)
    return float(np.clip(min(requested, mode_limit), 0.0, 100.0))


def _is_severe_data_issue(warning_type: Any, category: Any, severity: Any) -> bool:
    warning = str(warning_type or "").lower().replace("_", "").replace("-", "").replace(" ", "")
    if str(category) != "DataQualityRisk":
        return False
    if any(token in warning for token in SEVERE_DATA_WARNINGS):
        return True
    return str(severity) == "Critical"


def _candidate_base_row(asset: str, horizon: int, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    allocation = _subset(tables["allocation_plan_table"], asset, horizon)
    paper = _subset(tables["paper_portfolio_table"], asset, horizon)
    risk_matrix = _subset(tables["asset_horizon_risk_matrix"], asset, horizon)
    source = allocation if not allocation.empty else paper

    phase12_weight = _safe_float(_first_value(allocation, ["SuggestedPaperWeightPct"], np.nan), np.nan)
    if not np.isfinite(phase12_weight):
        phase12_weight = _safe_float(_first_value(paper, ["SuggestedPaperWeightPct"], 0.0), 0.0)
    real_allowed = _safe_bool(_first_value(source, ["RealCapitalAllowed"], False))
    phase12_real = _safe_float(_first_value(source, ["SuggestedRealWeightPct"], 0.0), 0.0)
    real_weight = phase12_real if real_allowed else 0.0
    return {
        "Asset": asset,
        "Horizon": int(horizon),
        "ResearchAction": str(_first_value(source, ["ResearchAction"], "")) or "ObserveOnly",
        "Phase12PaperWeightPct": max(0.0, phase12_weight),
        "SuggestedRealWeightPct": max(0.0, real_weight),
        "RealCapitalAllowed": bool(real_allowed),
        "PaperStatus": str(_first_value(risk_matrix, ["PaperStatus"], _first_value(source, ["PaperAllocationStatus"], ""))) or "MonitorOnly",
        "CapitalStatus": str(_first_value(risk_matrix, ["CapitalStatus"], _first_value(source, ["CapitalDeploymentStatus"], ""))) or "NoSpecificBlocker",
        "OverallRiskScore": _safe_float(_first_value(risk_matrix, ["OverallRiskScore"], 0.0), 0.0),
    }


def _matching_risks(asset: str, horizon: int, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    raw = tables.get("raw_warning_evidence", pd.DataFrame())
    if raw.empty:
        return pd.DataFrame()
    return _subset(raw, asset, horizon)


def _drawdown_breach(tables: Dict[str, pd.DataFrame]) -> bool:
    stress = tables.get("portfolio_drawdown_stress_table", pd.DataFrame())
    if stress.empty:
        return False
    if "Breach" in stress.columns:
        return stress["Breach"].astype(str).str.lower().isin({"true", "1", "yes"}).any()
    return False


def _multiplier_for_risk(row: pd.Series, drawdown_breach: bool) -> Tuple[str, float, str]:
    category = str(row.get("RiskCategory", "EvidenceInsufficient"))
    bucket = CATEGORY_TO_MULTIPLIER_BUCKET.get(category, "EvidenceMultiplier")
    multiplier = float(BASE_MULTIPLIERS.get(category, 0.80))
    if category == "DataQualityRisk" and not _is_severe_data_issue(row.get("WarningType"), category, row.get("Severity")):
        multiplier = 0.60
    if category == "DrawdownRisk" and drawdown_breach:
        multiplier = min(multiplier, 0.50)
    if category == "ComplianceNotice":
        multiplier = 1.0
    explanation = f"{category} applies {multiplier:.2f} sizing multiplier."
    return bucket, float(np.clip(multiplier, 0.0, 1.0)), explanation


def _risk_multipliers(asset: str, horizon: int, tables: Dict[str, pd.DataFrame], drawdown_breach: bool) -> Tuple[Dict[str, float], pd.DataFrame, str, str]:
    multipliers = {
        "DataQualityMultiplier": 1.0,
        "ProbabilityReliabilityMultiplier": 1.0,
        "DrawdownMultiplier": 1.0,
        "BenchmarkMultiplier": 1.0,
        "EvidenceMultiplier": 1.0,
        "CostMultiplier": 1.0,
        "ConcentrationMultiplier": 1.0,
        "ForwardEvidenceMultiplier": 1.0,
    }
    rows: List[Dict[str, Any]] = []
    risks = _matching_risks(asset, horizon, tables)
    main_risk = "NoSpecificRisk"
    zero_reason = ""
    if not risks.empty:
        risks = risks.copy()
        risks["RiskScore"] = pd.to_numeric(risks.get("RiskScore", 0.0), errors="coerce").fillna(0.0)
        main_risk = str(risks.sort_values("RiskScore", ascending=False).iloc[0].get("RiskCategory", "EvidenceRisk"))
    for _, risk in risks.iterrows():
        bucket, multiplier, explanation = _multiplier_for_risk(risk, drawdown_breach)
        if bucket in multipliers:
            multipliers[bucket] = min(multipliers[bucket], multiplier)
        category = str(risk.get("RiskCategory", "EvidenceInsufficient"))
        if category == "DataQualityRisk" and multiplier == 0:
            zero_reason = str(risk.get("WarningType", "Severe data issue"))
        rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "RiskCategory": category,
                "Severity": str(risk.get("Severity", "")),
                "RiskScore": _safe_float(risk.get("RiskScore"), 0.0),
                "AppliedMultiplier": round(multiplier, 4),
                "ImpactType": bucket,
                "Explanation": explanation,
            }
        )
    return multipliers, pd.DataFrame(rows, columns=list(RISK_MULTIPLIER_COLUMNS)), main_risk, zero_reason


def _overall_multiplier(multipliers: Dict[str, float]) -> float:
    value = 1.0
    for key in [
        "DataQualityMultiplier",
        "ProbabilityReliabilityMultiplier",
        "DrawdownMultiplier",
        "BenchmarkMultiplier",
        "EvidenceMultiplier",
        "CostMultiplier",
        "ConcentrationMultiplier",
        "ForwardEvidenceMultiplier",
    ]:
        value *= float(multipliers.get(key, 1.0))
    return float(np.clip(value, 0.0, 1.0))


def _pre_cap_position_rows(
    assets: Iterable[str],
    horizons: Iterable[int],
    tables: Dict[str, pd.DataFrame],
    *,
    allow_low_risk_increase: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    multiplier_tables: List[pd.DataFrame] = []
    drawdown_breached = _drawdown_breach(tables)
    for asset in assets:
        for horizon in horizons:
            base = _candidate_base_row(asset, int(horizon), tables)
            multipliers, multiplier_table, main_risk, zero_reason = _risk_multipliers(asset, int(horizon), tables, drawdown_breached)
            multiplier_tables.append(multiplier_table)
            overall = _overall_multiplier(multipliers)
            start = float(base["Phase12PaperWeightPct"])
            preliminary = start * overall
            if allow_low_risk_increase and start > 0 and base["OverallRiskScore"] < 20 and overall >= 0.95:
                preliminary = min(start * 1.05, start + 1.0)
            if zero_reason:
                preliminary = 0.0
            if start <= 0:
                preliminary = 0.0
            size_change = preliminary - start
            if start <= 0 and base["ResearchAction"] == "Watchlist":
                decision = "WatchlistOnly"
            elif start <= 0:
                decision = "NoAllocation"
            elif zero_reason:
                decision = "ZeroDueToDataIssue"
            elif preliminary <= 0:
                decision = "ZeroDueToRisk"
            elif preliminary < start - 0.0001:
                decision = "ReduceSize"
            elif preliminary > start + 0.0001:
                decision = "IncreaseWithinPaperOnlyLimit"
            else:
                decision = "KeepSize"
            explanation = (
                "Compliance notices do not reduce sizing."
                if main_risk == "ComplianceNotice"
                else f"Risk-adjusted from {start:.2f}% using multiplier {overall:.2f}; main risk: {main_risk}."
            )
            row = {
                **base,
                **multipliers,
                "OverallRiskMultiplier": round(overall, 6),
                "_PreCapPaperWeightPct": round(float(preliminary), 6),
                "OptimizedPaperWeightPct": round(float(preliminary), 6),
                "SizeChangePct": round(float(size_change), 6),
                "SizingDecision": decision,
                "ZeroSizeReason": zero_reason if preliminary <= 0 and start > 0 else "",
                "MainRiskReason": main_risk,
                "SizingExplanation": explanation,
            }
            rows.append(row)
    multiplier_table = pd.concat(multiplier_tables, ignore_index=True) if multiplier_tables else _empty_frame(RISK_MULTIPLIER_COLUMNS)
    return pd.DataFrame(rows), multiplier_table


def _cap_positions(
    pre_cap: pd.DataFrame,
    *,
    portfolio_mode: str,
    max_single_asset_exposure_pct: float,
    max_single_horizon_exposure_pct: float,
    max_portfolio_paper_exposure_pct: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pre_cap.empty:
        return _empty_frame(DYNAMIC_POSITION_COLUMNS), _empty_frame(CAP_ADJUSTMENT_COLUMNS)
    rows = pre_cap.copy()
    before_total = float(rows["_PreCapPaperWeightPct"].sum())
    portfolio_limit = _mode_portfolio_limit(portfolio_mode, max_portfolio_paper_exposure_pct)
    if before_total > portfolio_limit and before_total > 0:
        rows["_AfterPortfolioCap"] = rows["_PreCapPaperWeightPct"] * portfolio_limit / before_total
    else:
        rows["_AfterPortfolioCap"] = rows["_PreCapPaperWeightPct"]

    rows = rows.sort_values(["_AfterPortfolioCap", "OverallRiskScore"], ascending=[False, True]).reset_index(drop=True)
    asset_used: Dict[str, float] = {}
    horizon_used: Dict[int, float] = {}
    final_weights: List[float] = []
    cap_notes: List[str] = []
    for _, row in rows.iterrows():
        raw_weight = float(row["_AfterPortfolioCap"])
        asset = str(row["Asset"])
        horizon = int(row["Horizon"])
        available_asset = max(0.0, float(max_single_asset_exposure_pct) - asset_used.get(asset, 0.0))
        available_horizon = max(0.0, float(max_single_horizon_exposure_pct) - horizon_used.get(horizon, 0.0))
        final = max(0.0, min(raw_weight, available_asset, available_horizon))
        final_weights.append(final)
        if final + 0.000001 < raw_weight:
            reason = "asset cap" if available_asset <= available_horizon else "horizon cap"
            cap_notes.append(reason)
        else:
            cap_notes.append("")
        asset_used[asset] = asset_used.get(asset, 0.0) + final
        horizon_used[horizon] = horizon_used.get(horizon, 0.0) + final
    rows["OptimizedPaperWeightPct"] = final_weights
    rows["_CapNote"] = cap_notes
    rows["SizeChangePct"] = rows["OptimizedPaperWeightPct"].astype(float) - rows["Phase12PaperWeightPct"].astype(float)
    for idx, row in rows.iterrows():
        start = float(row["Phase12PaperWeightPct"])
        final = float(row["OptimizedPaperWeightPct"])
        if start > 0 and final <= 0 and not str(row["ZeroSizeReason"]).strip():
            rows.at[idx, "ZeroSizeReason"] = "Exposure cap or portfolio risk cap reduced size to zero."
            rows.at[idx, "SizingDecision"] = "ZeroDueToRisk"
        elif start > 0 and final < start - 0.0001 and rows.at[idx, "SizingDecision"] == "KeepSize":
            rows.at[idx, "SizingDecision"] = "ReduceSize"
        if str(row.get("_CapNote", "")):
            rows.at[idx, "SizingExplanation"] = f"{row['SizingExplanation']} Capped by {row['_CapNote']}."

    cap_rows = [
        {
            "CapType": "PortfolioPaperExposure",
            "LimitPct": round(portfolio_limit, 4),
            "BeforeAdjustmentPct": round(before_total, 4),
            "AfterAdjustmentPct": round(float(rows["_AfterPortfolioCap"].sum()), 4),
            "BreachBefore": bool(before_total > portfolio_limit),
            "BreachAfter": bool(float(rows["_AfterPortfolioCap"].sum()) > portfolio_limit + 0.0001),
            "AffectedAssets": "ALL",
            "AffectedHorizons": "ALL",
            "AdjustmentReason": "Mode and portfolio paper exposure cap.",
        }
    ]
    for asset, group in rows.groupby("Asset"):
        before = float(group["_AfterPortfolioCap"].sum())
        after = float(group["OptimizedPaperWeightPct"].sum())
        cap_rows.append(
            {
                "CapType": "SingleAssetExposure",
                "LimitPct": round(float(max_single_asset_exposure_pct), 4),
                "BeforeAdjustmentPct": round(before, 4),
                "AfterAdjustmentPct": round(after, 4),
                "BreachBefore": bool(before > max_single_asset_exposure_pct),
                "BreachAfter": bool(after > max_single_asset_exposure_pct + 0.0001),
                "AffectedAssets": asset,
                "AffectedHorizons": "ALL",
                "AdjustmentReason": "Single asset exposure cap.",
            }
        )
    for horizon, group in rows.groupby("Horizon"):
        before = float(group["_AfterPortfolioCap"].sum())
        after = float(group["OptimizedPaperWeightPct"].sum())
        cap_rows.append(
            {
                "CapType": "SingleHorizonExposure",
                "LimitPct": round(float(max_single_horizon_exposure_pct), 4),
                "BeforeAdjustmentPct": round(before, 4),
                "AfterAdjustmentPct": round(after, 4),
                "BreachBefore": bool(before > max_single_horizon_exposure_pct),
                "BreachAfter": bool(after > max_single_horizon_exposure_pct + 0.0001),
                "AffectedAssets": "ALL",
                "AffectedHorizons": f"{int(horizon)}D",
                "AdjustmentReason": "Single horizon exposure cap.",
            }
        )
    output = rows[list(DYNAMIC_POSITION_COLUMNS)].copy()
    output["OptimizedPaperWeightPct"] = output["OptimizedPaperWeightPct"].round(4)
    output["SizeChangePct"] = output["SizeChangePct"].round(4)
    return output, pd.DataFrame(cap_rows, columns=list(CAP_ADJUSTMENT_COLUMNS))


def _zero_size_table(position: pd.DataFrame, multipliers: pd.DataFrame) -> pd.DataFrame:
    if position.empty:
        return _empty_frame(ZERO_SIZE_COLUMNS)
    zeroed = position[(position["Phase12PaperWeightPct"].astype(float) > 0) & (position["OptimizedPaperWeightPct"].astype(float) <= 0)].copy()
    rows: List[Dict[str, Any]] = []
    for _, row in zeroed.iterrows():
        risk_match = _subset(multipliers, row["Asset"], int(row["Horizon"]))
        risk_match = risk_match.sort_values("RiskScore", ascending=False) if not risk_match.empty else risk_match
        source = str(risk_match.iloc[0]["RiskCategory"]) if not risk_match.empty else row["MainRiskReason"]
        rows.append(
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "OriginalPaperWeightPct": row["Phase12PaperWeightPct"],
                "OptimizedPaperWeightPct": row["OptimizedPaperWeightPct"],
                "ZeroSizeReason": row["ZeroSizeReason"] or row["MainRiskReason"],
                "EvidenceSource": source,
                "RequiredFixBeforeSizing": "Fix severe data issue or reduce risk evidence before restoring simulated size.",
            }
        )
    return pd.DataFrame(rows, columns=list(ZERO_SIZE_COLUMNS))


def _join_unique(values: Iterable[Any]) -> str:
    clean: List[str] = []
    for value in values:
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value)
        if not text:
            continue
        if text not in clean:
            clean.append(text)
    return "; ".join(clean) if clean else "ALL"


def _severity_rank(value: Any) -> int:
    return {"Info": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}.get(str(value), 0)


def _risk_multiplier_summary(multipliers: pd.DataFrame) -> pd.DataFrame:
    if multipliers.empty:
        return _empty_frame(RISK_MULTIPLIER_SUMMARY_COLUMNS)
    rows: List[Dict[str, Any]] = []
    data = multipliers.copy()
    data["AppliedMultiplier"] = pd.to_numeric(data["AppliedMultiplier"], errors="coerce").fillna(1.0)
    for category, group in data.groupby("RiskCategory", dropna=False):
        severity = sorted(group["Severity"].dropna().astype(str).unique().tolist(), key=_severity_rank, reverse=True)
        impact = group["ImpactType"].dropna().astype(str).mode()
        rows.append(
            {
                "RiskCategory": str(category),
                "Severity": severity[0] if severity else "",
                "AffectedAssets": _join_unique(group["Asset"]),
                "AffectedHorizons": _join_unique([f"{int(h)}D" for h in pd.to_numeric(group["Horizon"], errors="coerce").dropna().astype(int)]),
                "Count": int(len(group)),
                "MinMultiplier": round(float(group["AppliedMultiplier"].min()), 4),
                "MedianMultiplier": round(float(group["AppliedMultiplier"].median()), 4),
                "MaxMultiplier": round(float(group["AppliedMultiplier"].max()), 4),
                "MainImpactType": str(impact.iloc[0]) if not impact.empty else "",
                "Explanation": str(group.sort_values("AppliedMultiplier").iloc[0]["Explanation"]),
            }
        )
    return pd.DataFrame(rows, columns=list(RISK_MULTIPLIER_SUMMARY_COLUMNS)).sort_values(
        ["MinMultiplier", "Count"], ascending=[True, False]
    ).reset_index(drop=True)


def _optimized_portfolio(position: pd.DataFrame) -> pd.DataFrame:
    if position.empty:
        return _empty_frame(OPTIMIZED_PORTFOLIO_COLUMNS)
    active = position[position["OptimizedPaperWeightPct"].astype(float) > 0].copy()
    if active.empty:
        return _empty_frame(OPTIMIZED_PORTFOLIO_COLUMNS)
    active["RiskAdjustedScore"] = active["OptimizedPaperWeightPct"].astype(float) * active["OverallRiskMultiplier"].astype(float) * (100.0 - active["OverallRiskScore"].astype(float).clip(0, 100)) / 100.0
    active = active.sort_values(["OptimizedPaperWeightPct", "RiskAdjustedScore"], ascending=[False, False]).reset_index(drop=True)
    active["Rank"] = range(1, len(active) + 1)
    out = pd.DataFrame(
        {
            "Rank": active["Rank"],
            "Asset": active["Asset"],
            "Horizon": active["Horizon"],
            "OptimizedPaperWeightPct": active["OptimizedPaperWeightPct"],
            "RiskScore": active["OverallRiskScore"],
            "RiskAdjustedScore": active["RiskAdjustedScore"].round(4),
            "MainRisk": active["MainRiskReason"],
            "ReviewTrigger": "Review after next matured outcome or new critical warning.",
            "ExitOrReduceTrigger": "Reduce or zero simulated size if severe data, drawdown, cost, or probability warning worsens.",
        }
    )
    return out[list(OPTIMIZED_PORTFOLIO_COLUMNS)]


def _drawdown_budget(position: pd.DataFrame, max_drawdown_shock_pct: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in position.iterrows():
        weight = _safe_float(row.get("OptimizedPaperWeightPct"), 0.0)
        risk = str(row.get("MainRiskReason", ""))
        shock = 35.0 if risk == "DrawdownRisk" else 25.0 if row.get("DrawdownMultiplier", 1.0) < 1.0 else 15.0
        contribution = weight * shock / 100.0
        used = contribution / max(float(max_drawdown_shock_pct), 0.0001) * 100.0
        breach = contribution > float(max_drawdown_shock_pct)
        rows.append(
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "OptimizedPaperWeightPct": round(weight, 4),
                "EstimatedDrawdownShockPct": round(shock, 4),
                "ContributionToPortfolioDrawdownPct": round(contribution, 4),
                "DrawdownBudgetUsedPct": round(used, 4),
                "DrawdownCapPct": float(max_drawdown_shock_pct),
                "Breach": bool(breach),
                "Action": "Reduce simulated size if drawdown budget is breached." if breach else "Within simulated drawdown budget.",
            }
        )
    return pd.DataFrame(rows, columns=list(DRAWDOWN_BUDGET_COLUMNS))


def _scenario_table(position: pd.DataFrame, zero_size: pd.DataFrame) -> pd.DataFrame:
    starting_total = float(position["Phase12PaperWeightPct"].astype(float).sum()) if not position.empty else 0.0
    optimized_total = float(position["OptimizedPaperWeightPct"].astype(float).sum()) if not position.empty else 0.0
    highest_risk = 0.0
    highest_risk_opt = 0.0
    highest_risk_explanation = "No optimized paper candidates exist."
    if not position.empty:
        active = position[position["OptimizedPaperWeightPct"].astype(float) > 0].copy()
        high = active.sort_values("OverallRiskScore", ascending=False).head(1) if not active.empty else pd.DataFrame()
        if not high.empty:
            highest_risk = float(high["Phase12PaperWeightPct"].iloc[0])
            highest_risk_opt = float(high["OptimizedPaperWeightPct"].iloc[0])
            highest_risk_explanation = (
                f"Uses highest-risk optimized candidate: {high['Asset'].iloc[0]} "
                f"{int(high['Horizon'].iloc[0])}D; main risk {high['MainRiskReason'].iloc[0]}."
            )
    data_blocked = float(zero_size["OriginalPaperWeightPct"].astype(float).sum()) if not zero_size.empty else 0.0
    zero_highest_start = 0.0
    zero_highest_explanation = "No zero-sized candidates exist."
    if not zero_size.empty:
        zero_highest = zero_size.copy()
        if "OriginalPaperWeightPct" in zero_highest.columns:
            zero_highest = zero_highest.sort_values("OriginalPaperWeightPct", ascending=False).head(1)
        else:
            zero_highest = zero_highest.head(1)
        zero_highest_start = float(zero_highest["OriginalPaperWeightPct"].iloc[0])
        zero_highest_explanation = (
            f"Excluded zero-sized candidate: {zero_highest['Asset'].iloc[0]} "
            f"{int(zero_highest['Horizon'].iloc[0])}D; reason {zero_highest['ZeroSizeReason'].iloc[0]}."
        )
    drawdown_start = float((position["Phase12PaperWeightPct"].astype(float) * 0.25).sum()) if not position.empty else 0.0
    drawdown_opt = float((position["OptimizedPaperWeightPct"].astype(float) * 0.25).sum()) if not position.empty else 0.0
    cost_start = float((position["Phase12PaperWeightPct"].astype(float) * (1.0 - position["CostMultiplier"].astype(float))).sum()) if not position.empty else 0.0
    cost_opt = float((position["OptimizedPaperWeightPct"].astype(float) * (1.0 - position["CostMultiplier"].astype(float))).sum()) if not position.empty else 0.0
    prob_start = float((position["Phase12PaperWeightPct"].astype(float) * (1.0 - position["ProbabilityReliabilityMultiplier"].astype(float))).sum()) if not position.empty else 0.0
    prob_opt = float((position["OptimizedPaperWeightPct"].astype(float) * (1.0 - position["ProbabilityReliabilityMultiplier"].astype(float))).sum()) if not position.empty else 0.0
    scenarios = [
        ("All optimized paper candidates lose", starting_total, optimized_total, "Uses total simulated paper exposure."),
        ("Highest risk allocated candidate fails", highest_risk, highest_risk_opt, highest_risk_explanation),
        ("Highest risk zero-sized candidate excluded", zero_highest_start, 0.0, zero_highest_explanation),
        ("Data-quality blocked candidates excluded", data_blocked, 0.0, "Severe data issue rows are excluded from simulated sizing."),
        ("Drawdown shock repeats", drawdown_start, drawdown_opt, "Applies a drawdown shock proxy to paper exposure."),
        ("Cost/slippage stress increases", cost_start, cost_opt, "Estimates exposure tied to cost-fragility multipliers."),
        ("Probability confidence fails", prob_start, prob_opt, "Estimates exposure tied to probability-reliability multipliers."),
    ]
    rows = []
    for scenario, start, opt, explanation in scenarios:
        reduction = 0.0 if start <= 0 else max(0.0, (start - opt) / start * 100.0)
        rows.append(
            {
                "Scenario": scenario,
                "StartingPortfolioImpact": round(start, 4),
                "OptimizedPortfolioImpact": round(opt, 4),
                "RiskReductionPct": round(reduction, 4),
                "Explanation": explanation,
            }
        )
    return pd.DataFrame(rows, columns=list(SCENARIO_COLUMNS))


def _next_actions(position: pd.DataFrame, zero_size: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not zero_size.empty:
        rows.append(
            {
                "Rank": 0,
                "Action": "Fix severe data issues before restoring simulated size.",
                "AffectedAssets": "; ".join(zero_size["Asset"].astype(str).unique()),
                "AffectedHorizons": "; ".join(f"{int(h)}D" for h in pd.to_numeric(zero_size["Horizon"], errors="coerce").dropna().astype(int).unique()),
                "WhyItMatters": "Severe data issues force zero simulated size.",
                "ExpectedEffect": "Allows affected rows to re-enter paper sizing after evidence is valid.",
                "Urgency": "High",
                "DependsOn": "Clean outcome and price data.",
            }
        )
    if not position.empty:
        reduced = position[(position["Phase12PaperWeightPct"].astype(float) > position["OptimizedPaperWeightPct"].astype(float)) & (position["OptimizedPaperWeightPct"].astype(float) > 0)].copy()
        if not reduced.empty:
            top = reduced.sort_values("OverallRiskScore", ascending=False).head(5)
            rows.append(
                {
                    "Rank": 0,
                    "Action": "Review reduced paper rows with highest risk score.",
                    "AffectedAssets": "; ".join(top["Asset"].astype(str).unique()),
                    "AffectedHorizons": "; ".join(f"{int(h)}D" for h in pd.to_numeric(top["Horizon"], errors="coerce").dropna().astype(int).unique()),
                    "WhyItMatters": "These rows still carry simulated exposure after risk reduction.",
                    "ExpectedEffect": "Improves monitoring priority and prevents warning overload.",
                    "Urgency": "Medium",
                    "DependsOn": "New Phase 13 risk evidence and matured outcomes.",
                }
            )
    if not rows:
        rows.append(
            {
                "Rank": 0,
                "Action": "Continue monitoring optimized paper sizing.",
                "AffectedAssets": "ALL",
                "AffectedHorizons": "ALL",
                "WhyItMatters": "No severe sizing issue was detected in loaded evidence.",
                "ExpectedEffect": "Keeps paper sizing aligned with updated warnings.",
                "Urgency": "Low",
                "DependsOn": "Next evidence refresh.",
            }
        )
    actions = pd.DataFrame(rows, columns=list(NEXT_ACTION_COLUMNS))
    actions["Rank"] = range(1, len(actions) + 1)
    return actions


def _summary(position: pd.DataFrame, zero_size: pd.DataFrame, portfolio_mode: str) -> pd.DataFrame:
    starting = float(position["Phase12PaperWeightPct"].astype(float).sum()) if not position.empty else 0.0
    optimized = float(position["OptimizedPaperWeightPct"].astype(float).sum()) if not position.empty else 0.0
    real = float(position["SuggestedRealWeightPct"].astype(float).sum()) if not position.empty else 0.0
    starting_count = int((position["Phase12PaperWeightPct"].astype(float) > 0).sum()) if not position.empty else 0
    optimized_count = int((position["OptimizedPaperWeightPct"].astype(float) > 0).sum()) if not position.empty else 0
    zeroed = int(len(zero_size))
    if zeroed:
        largest_reason = str(zero_size.iloc[0]["ZeroSizeReason"])
    elif not position.empty and starting > optimized:
        reduced = position.assign(_Reduction=position["Phase12PaperWeightPct"].astype(float) - position["OptimizedPaperWeightPct"].astype(float))
        largest_reason = str(reduced.sort_values("_Reduction", ascending=False).iloc[0]["MainRiskReason"])
    else:
        largest_reason = "No major reduction."
    if optimized <= 0:
        verdict = "Full paper reserve due to severe risk."
    elif optimized < 5:
        verdict = "Micro paper tracking mode."
    elif optimized < 15:
        verdict = "Reduced paper tracking mode."
    else:
        verdict = "Active risk-adjusted paper sizing."
    active_risks = []
    if not position.empty:
        risk_terms = {
            "ProbabilityUnreliable": "probability unreliability",
            "Overconfident": "overconfidence",
            "BenchmarkDominated": "benchmark dominance",
            "DrawdownRisk": "drawdown",
            "CostFragile": "cost",
            "EvidenceInsufficient": "evidence weakness",
            "LowTradeCount": "evidence weakness",
            "PendingEvidenceOnly": "young forward evidence",
            "ForwardEvidenceYoung": "young forward evidence",
            "DataQualityRisk": "data-quality issues",
        }
        for risk, label in risk_terms.items():
            if position["MainRiskReason"].astype(str).eq(risk).any() and label not in active_risks:
                active_risks.append(label)
    risk_phrase = ", ".join(active_risks) if active_risks else "current warning evidence"
    main_reason = (
        "Real capital remains blocked unless upstream Phase 11 and Phase 12 gates allow it. "
        f"Paper sizing is intentionally small when risks such as {risk_phrase} are present. "
        "This is simulated paper sizing only."
    )
    return pd.DataFrame(
        [
            {
                "PortfolioMode": portfolio_mode,
                "StartingPaperExposurePct": round(starting, 4),
                "OptimizedPaperExposurePct": round(optimized, 4),
                "PaperReservePct": round(max(0.0, 100.0 - optimized), 4),
                "RealCapitalExposurePct": round(real, 4),
                "NumberStartingPaperCandidates": starting_count,
                "NumberOptimizedPaperCandidates": optimized_count,
                "NumberZeroedByRisk": zeroed,
                "LargestRiskReductionReason": largest_reason,
                "OverallSizingVerdict": verdict,
                "MainReason": main_reason,
            }
        ],
        columns=list(DYNAMIC_SUMMARY_COLUMNS),
    )


def run_dynamic_risk_sizing(
    *,
    use_artifact_store: bool = False,
    prefer_uploaded: bool = False,
    uploaded_overrides: Optional[Dict[str, Any]] = None,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    portfolio_mode: str = "Balanced Research",
    max_single_asset_exposure_pct: float = 25.0,
    max_single_horizon_exposure_pct: float = 35.0,
    max_portfolio_paper_exposure_pct: Optional[float] = None,
    max_drawdown_shock_pct: float = 10.0,
    allow_low_risk_increase: bool = False,
    autosave: bool = False,
    **direct_tables: Any,
) -> DynamicRiskSizingReport:
    """Build the Phase 14 dynamic risk sizing report."""
    asset_list = list(assets or get_asset_names())
    horizon_list = [int(h) for h in (horizons or DYNAMIC_SIZING_HORIZONS)]
    settings = {
        "phase": "14",
        "purpose": "dynamic_risk_sizing",
        "assets": asset_list,
        "horizons": horizon_list,
        "portfolio_mode": portfolio_mode,
        "max_single_asset_exposure_pct": float(max_single_asset_exposure_pct),
        "max_single_horizon_exposure_pct": float(max_single_horizon_exposure_pct),
        "max_portfolio_paper_exposure_pct": _mode_portfolio_limit(portfolio_mode, max_portfolio_paper_exposure_pct),
        "max_drawdown_shock_pct": float(max_drawdown_shock_pct),
        "allow_low_risk_increase": bool(allow_low_risk_increase),
        "live_execution": False,
    }
    tables, input_source_table = _resolve_inputs(bool(use_artifact_store), bool(prefer_uploaded), uploaded_overrides, direct_tables)
    pre_cap, multiplier_table = _pre_cap_position_rows(asset_list, horizon_list, tables, allow_low_risk_increase=bool(allow_low_risk_increase))
    position, cap_adjustments = _cap_positions(
        pre_cap,
        portfolio_mode=portfolio_mode,
        max_single_asset_exposure_pct=float(max_single_asset_exposure_pct),
        max_single_horizon_exposure_pct=float(max_single_horizon_exposure_pct),
        max_portfolio_paper_exposure_pct=max_portfolio_paper_exposure_pct,
    )
    zero_size = _zero_size_table(position, multiplier_table)
    optimized = _optimized_portfolio(position)
    drawdown = _drawdown_budget(position, float(max_drawdown_shock_pct))
    scenarios = _scenario_table(position, zero_size)
    next_actions = _next_actions(position, zero_size)
    summary = _summary(position, zero_size, portfolio_mode)
    multiplier_summary = _risk_multiplier_summary(multiplier_table)

    report = DynamicRiskSizingReport(
        dynamic_sizing_summary_table=summary.reset_index(drop=True),
        dynamic_position_sizing_table=position.reset_index(drop=True),
        risk_multiplier_table=multiplier_table.reset_index(drop=True),
        risk_multiplier_summary_table=multiplier_summary.reset_index(drop=True),
        cap_adjustment_table=cap_adjustments.reset_index(drop=True),
        zero_size_table=zero_size.reset_index(drop=True),
        optimized_portfolio_table=optimized.reset_index(drop=True),
        drawdown_budget_table=drawdown.reset_index(drop=True),
        risk_adjusted_scenarios_table=scenarios.reset_index(drop=True),
        next_sizing_actions_table=next_actions.reset_index(drop=True),
        input_source_table=input_source_table.reset_index(drop=True),
        settings=settings,
    )
    if autosave:
        report.saved_artifacts = save_phase_artifacts(
            DYNAMIC_RISK_SIZING_PHASE_NAME,
            {
                "dynamic_sizing_summary_table": report.dynamic_sizing_summary_table,
                "dynamic_position_sizing_table": report.dynamic_position_sizing_table,
                "risk_multiplier_table": report.risk_multiplier_table,
                "risk_multiplier_summary_table": report.risk_multiplier_summary_table,
                "cap_adjustment_table": report.cap_adjustment_table,
                "zero_size_table": report.zero_size_table,
                "optimized_portfolio_table": report.optimized_portfolio_table,
                "drawdown_budget_table": report.drawdown_budget_table,
                "risk_adjusted_scenarios_table": report.risk_adjusted_scenarios_table,
                "next_sizing_actions_table": report.next_sizing_actions_table,
                "input_source_table": report.input_source_table,
            },
            inputs={},
            config=report.settings,
            warnings=report.risk_multiplier_table["RiskCategory"].dropna().astype(str).unique().tolist()
            if not report.risk_multiplier_table.empty
            else [],
        )
    return report


__all__ = [
    "DYNAMIC_RISK_SIZING_PHASE_NAME",
    "DynamicRiskSizingReport",
    "run_dynamic_risk_sizing",
]
