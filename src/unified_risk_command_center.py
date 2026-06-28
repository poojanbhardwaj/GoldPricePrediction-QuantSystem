"""Phase 21 unified performance and risk intelligence command center.

This module aggregates existing Phase 18, 19, and 20 evidence. It never
retrains a model and never changes upstream results or capital gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.artifact_store import resolve_artifact, save_phase_artifacts


UNIFIED_RISK_COMMAND_CENTER_PHASE_NAME = "phase21_unified_risk_command_center"

UNIFIED_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "CommandCenterVerdict", "BroadEdgeStatus", "BestEvidenceSource", "BestAsset",
    "BestHorizon", "BestModelOrPolicy", "BestNetReturnPct", "BestBaselineGapPct",
    "LeakageStatus", "BenchmarkStatus", "CostFragilityStatus", "DrawdownRiskStatus",
    "OverfitRiskStatus", "RealCapitalStatus", "RecommendedMode", "FinalExplanation",
)

ASSET_HORIZON_SCORECARD_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "TrueMLReturnPct", "TrueMLBeatsBestBaseline",
    "PolicyLabReturnPct", "PolicyBeatsBestBaseline", "BestBaselineName",
    "BestBaselineReturnPct", "LeakagePassed", "CostFragilityFlag",
    "DrawdownRiskFlag", "OverfitRiskFlag", "MaturedTrades", "PendingTrades",
    "EvidenceScore", "RiskScore", "FinalResearchLabel", "Explanation",
)

RISK_REGISTER_COLUMNS: Tuple[str, ...] = (
    "RiskName", "Severity", "AffectedAssets", "AffectedHorizons", "EvidenceSource",
    "WhyItMatters", "Mitigation", "BlocksRealCapital",
)

PAPER_TRACKING_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "CandidateSource", "EvidenceScore", "RiskScore", "Reason",
    "RequiredNextEvidence", "RealCapitalAllowed",
)

REJECTED_CANDIDATE_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "RejectionReason", "FailedChecks", "EvidenceScore",
    "RiskScore", "SuggestedFix",
)

QUALITY_GATE_COLUMNS: Tuple[str, ...] = (
    "GateName", "Passed", "Severity", "Explanation",
)

NEXT_ACTION_COLUMNS: Tuple[str, ...] = (
    "Priority", "Action", "Reason", "ExpectedImpact", "PhaseSuggestion",
)

INPUT_SOURCE_COLUMNS: Tuple[str, ...] = (
    "SourcePhase", "ExpectedFile", "Found", "Rows", "UsedFor", "Warning",
)


INPUT_SPECS: Dict[str, Tuple[str, str, str, str]] = {
    "phase18_summary": ("phase18_replay_benchmark_audit", "replay_benchmark_summary_table", "phase18_replay_benchmark_summary.csv", "Proxy replay context"),
    "phase18_asset_horizon_edge": ("phase18_replay_benchmark_audit", "replay_asset_horizon_edge_table", "phase18_replay_asset_horizon_edge.csv", "Proxy edge context"),
    "phase18_dominance_failures": ("phase18_replay_benchmark_audit", "replay_dominance_failures_table", "phase18_replay_dominance_failures.csv", "Proxy rejection evidence"),
    "phase18_cost_robustness": ("phase18_replay_benchmark_audit", "replay_cost_robustness_table", "phase18_replay_cost_robustness.csv", "Proxy cost risk"),
    "phase18_drawdown": ("phase18_replay_benchmark_audit", "replay_drawdown_comparison_table", "phase18_replay_drawdown_comparison.csv", "Proxy drawdown context"),
    "phase18_quality_gates": ("phase18_replay_benchmark_audit", "replay_quality_gate_table", "phase18_replay_quality_gates.csv", "Proxy evidence quality"),
    "phase19_summary": ("phase19_signal_policy_edge_lab", "policy_lab_summary_table", "phase19_policy_lab_summary.csv", "Policy edge summary"),
    "phase19_asset_horizon_edge": ("phase19_signal_policy_edge_lab", "policy_asset_horizon_edge_table", "phase19_policy_asset_horizon_edge.csv", "Policy scorecard evidence"),
    "phase19_dominance_failures": ("phase19_signal_policy_edge_lab", "policy_dominance_failures_table", "phase19_policy_dominance_failures.csv", "Policy rejection evidence"),
    "phase19_overfit_audit": ("phase19_signal_policy_edge_lab", "policy_overfit_audit_table", "phase19_policy_overfit_audit.csv", "Policy overfit risk"),
    "phase19_cost_sensitivity": ("phase19_signal_policy_edge_lab", "policy_cost_sensitivity_table", "phase19_policy_cost_sensitivity.csv", "Policy cost risk"),
    "phase19_drawdown": ("phase19_signal_policy_edge_lab", "policy_drawdown_table", "phase19_policy_drawdown.csv", "Policy drawdown risk"),
    "phase19_quality_gates": ("phase19_signal_policy_edge_lab", "policy_quality_gates_table", "phase19_policy_quality_gates.csv", "Policy evidence quality"),
    "phase20_summary": ("phase20_true_historical_ml_replay", "phase20_true_ml_summary", "phase20_true_ml_summary.csv", "True ML replay summary"),
    "phase20_performance": ("phase20_true_historical_ml_replay", "phase20_true_ml_performance", "phase20_true_ml_performance.csv", "True ML performance"),
    "phase20_baseline_comparison": ("phase20_true_historical_ml_replay", "phase20_true_ml_baseline_comparison", "phase20_true_ml_baseline_comparison.csv", "True ML baseline comparison"),
    "phase20_strength": ("phase20_true_historical_ml_replay", "phase20_true_ml_strength", "phase20_true_ml_strength.csv", "True ML strength and rejection"),
    "phase20_leakage_audit": ("phase20_true_historical_ml_replay", "phase20_leakage_audit", "phase20_leakage_audit.csv", "Leakage assurance"),
    "phase20_quality_gates": ("phase20_true_historical_ml_replay", "phase20_quality_gates", "phase20_quality_gates.csv", "True ML quality gates"),
}

DIRECT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "phase18_summary": ("replay_benchmark_summary_table",),
    "phase18_asset_horizon_edge": ("replay_asset_horizon_edge_table",),
    "phase18_dominance_failures": ("replay_dominance_failures_table",),
    "phase18_cost_robustness": ("replay_cost_robustness_table",),
    "phase18_drawdown": ("replay_drawdown_comparison_table",),
    "phase18_quality_gates": ("replay_quality_gate_table",),
    "phase19_summary": ("policy_lab_summary_table",),
    "phase19_asset_horizon_edge": ("policy_asset_horizon_edge_table",),
    "phase19_dominance_failures": ("policy_dominance_failures_table",),
    "phase19_overfit_audit": ("policy_overfit_audit_table",),
    "phase19_cost_sensitivity": ("policy_cost_sensitivity_table",),
    "phase19_drawdown": ("policy_drawdown_table",),
    "phase19_quality_gates": ("policy_quality_gates_table",),
    "phase20_summary": ("true_ml_summary_table",),
    "phase20_performance": ("true_ml_performance_table",),
    "phase20_baseline_comparison": ("true_ml_baseline_comparison_table",),
    "phase20_strength": ("true_ml_strength_table",),
    "phase20_leakage_audit": ("leakage_audit_table",),
    "phase20_quality_gates": ("quality_gates_table",),
}


@dataclass
class UnifiedRiskCommandCenterReport:
    unified_summary_table: pd.DataFrame
    asset_horizon_scorecard: pd.DataFrame
    risk_register: pd.DataFrame
    paper_tracking_candidates: pd.DataFrame
    rejected_candidates: pd.DataFrame
    quality_gates: pd.DataFrame
    next_actions: pd.DataFrame
    input_sources: pd.DataFrame
    settings: Dict[str, Any] = field(default_factory=dict)
    saved_artifacts: Dict[str, Any] = field(default_factory=dict)


def _to_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame()
    try:
        return pd.DataFrame(value)
    except Exception:
        return pd.DataFrame()


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        result = float(value)
    except Exception:
        return default
    return result if np.isfinite(result) else default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "passed", "pass"}:
        return True
    if text in {"false", "0", "no", "failed", "fail"}:
        return False
    return default


def _normalize_horizon(value: Any) -> Optional[int]:
    try:
        return int(float(str(value).replace("D", "").strip()))
    except Exception:
        return None


def _find_direct(alias: str, direct_tables: Dict[str, Any]) -> Any:
    for key in (alias,) + DIRECT_ALIASES.get(alias, ()):
        if key in direct_tables and direct_tables[key] is not None:
            return direct_tables[key]
    return None


def _resolve_inputs(
    use_artifact_store: bool,
    prefer_uploaded: bool,
    uploaded_overrides: Optional[Dict[str, Any]],
    direct_tables: Dict[str, Any],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    uploaded_overrides = uploaded_overrides or {}
    tables: Dict[str, pd.DataFrame] = {}
    source_rows: List[Dict[str, Any]] = []
    source_metadata: Dict[str, Any] = {}
    for alias, (phase, artifact, filename, used_for) in INPUT_SPECS.items():
        direct = _find_direct(alias, direct_tables)
        uploaded = uploaded_overrides.get(alias, uploaded_overrides.get(artifact))
        if direct is not None:
            table = _to_frame(direct)
            source = "DirectInput"
            found = True
            warning = ""
            metadata = {"Source": source, "Rows": len(table)}
        elif use_artifact_store or uploaded is not None:
            resolved = resolve_artifact(
                phase,
                artifact,
                uploaded_file=uploaded,
                prefer_uploaded=bool(prefer_uploaded),
                required=False,
            )
            table = _to_frame(resolved.get("Data"))
            source = str(resolved.get("Source", "Missing"))
            found = str(resolved.get("Status", "")).lower() == "loaded"
            warning = "" if found else f"Missing optional evidence: {filename}"
            metadata = {key: value for key, value in resolved.items() if key != "Data"}
        else:
            table = pd.DataFrame()
            source = "Missing"
            found = False
            warning = f"Missing optional evidence: {filename}"
            metadata = {"Source": source, "Rows": 0}
        tables[alias] = table
        source_metadata[alias] = metadata
        source_rows.append(
            {
                "SourcePhase": phase,
                "ExpectedFile": filename,
                "Found": bool(found),
                "Rows": int(len(table)),
                "UsedFor": used_for,
                "Warning": warning,
            }
        )
    return tables, pd.DataFrame(source_rows, columns=list(INPUT_SOURCE_COLUMNS)), source_metadata


def _keys_from_table(table: pd.DataFrame) -> List[Tuple[str, int]]:
    if table.empty or not {"Asset", "Horizon"}.issubset(table.columns):
        return []
    keys: List[Tuple[str, int]] = []
    for _, row in table.iterrows():
        horizon = _normalize_horizon(row.get("Horizon"))
        asset = str(row.get("Asset", "")).strip()
        if asset and horizon is not None:
            keys.append((asset, horizon))
    return keys


def _matching(table: pd.DataFrame, asset: str, horizon: int) -> pd.DataFrame:
    if table.empty or not {"Asset", "Horizon"}.issubset(table.columns):
        return pd.DataFrame()
    normalized = table["Horizon"].map(_normalize_horizon)
    return table[table["Asset"].astype(str).eq(str(asset)) & normalized.eq(int(horizon))].copy()


def _build_scorecard(tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], Dict[str, str]]]:
    key_sources = (
        "phase19_asset_horizon_edge", "phase19_dominance_failures",
        "phase20_performance", "phase20_baseline_comparison", "phase20_strength",
        "phase20_leakage_audit",
    )
    keys = sorted({key for source in key_sources for key in _keys_from_table(tables[source])})
    rows: List[Dict[str, Any]] = []
    identities: Dict[Tuple[str, int], Dict[str, str]] = {}
    for asset, horizon in keys:
        performance = _matching(tables["phase20_performance"], asset, horizon)
        if not performance.empty and "TotalReturnPct" in performance.columns:
            performance = performance.assign(_return=pd.to_numeric(performance["TotalReturnPct"], errors="coerce"))
            performance = performance.sort_values("_return", ascending=False)
        perf = performance.iloc[0] if not performance.empty else pd.Series(dtype=object)
        model = str(perf.get("ModelName", "")) if not perf.empty else ""
        baseline = _matching(tables["phase20_baseline_comparison"], asset, horizon)
        if not baseline.empty and model and "ModelName" in baseline.columns:
            model_baseline = baseline[baseline["ModelName"].astype(str).eq(model)]
            if not model_baseline.empty:
                baseline = model_baseline
        baseline_row = baseline.iloc[0] if not baseline.empty else pd.Series(dtype=object)
        strength = _matching(tables["phase20_strength"], asset, horizon)
        if not strength.empty and model and "ModelName" in strength.columns:
            model_strength = strength[strength["ModelName"].astype(str).eq(model)]
            if not model_strength.empty:
                strength = model_strength
        strength_row = strength.iloc[0] if not strength.empty else pd.Series(dtype=object)
        leakage_rows = _matching(tables["phase20_leakage_audit"], asset, horizon)

        policy_rows = _matching(tables["phase19_asset_horizon_edge"], asset, horizon)
        if not policy_rows.empty and "BestPolicyReturnPct" in policy_rows.columns:
            policy_rows = policy_rows.assign(_return=pd.to_numeric(policy_rows["BestPolicyReturnPct"], errors="coerce")).sort_values("_return", ascending=False)
        policy = policy_rows.iloc[0] if not policy_rows.empty else pd.Series(dtype=object)
        policy_name = str(policy.get("BestPolicy", "")) if not policy.empty else ""
        policy_cost = _matching(tables["phase19_cost_sensitivity"], asset, horizon)
        if not policy_cost.empty and policy_name and "PolicyName" in policy_cost.columns:
            policy_cost = policy_cost[policy_cost["PolicyName"].astype(str).eq(policy_name)]
        policy_drawdown = _matching(tables["phase19_drawdown"], asset, horizon)
        if not policy_drawdown.empty and policy_name and "PolicyName" in policy_drawdown.columns:
            policy_drawdown = policy_drawdown[policy_drawdown["PolicyName"].astype(str).eq(policy_name)]
        policy_overfit = _matching(tables["phase19_overfit_audit"], asset, horizon)
        if not policy_overfit.empty and policy_name and "PolicyName" in policy_overfit.columns:
            policy_overfit = policy_overfit[policy_overfit["PolicyName"].astype(str).eq(policy_name)]

        true_ml_available = not perf.empty
        policy_available = not policy.empty
        true_return = _safe_float(perf.get("TotalReturnPct"), np.nan) if true_ml_available else np.nan
        true_beats = _as_bool(baseline_row.get("BeatsBestBaseline"), False) if not baseline_row.empty else False
        policy_return = _safe_float(policy.get("BestPolicyReturnPct"), np.nan) if policy_available else np.nan
        policy_beats = _as_bool(policy.get("BeatsBestBaseline"), False) if policy_available else False
        baseline_name = str(baseline_row.get("BestBaselineName", policy.get("BestBaseline", "")))
        baseline_return = _safe_float(baseline_row.get("BestBaselineReturnPct", policy.get("BestBaselineReturnPct", np.nan)), np.nan)
        if not leakage_rows.empty and "LeakagePassed" in leakage_rows.columns:
            leakage_passed: Any = bool(leakage_rows["LeakagePassed"].map(_as_bool).all())
        elif not strength_row.empty and "LeakagePassed" in strength_row.index:
            leakage_passed = _as_bool(strength_row.get("LeakagePassed"), False)
        else:
            leakage_passed = np.nan
        matured = int(_safe_float(perf.get("MaturedTrades"), 0.0)) if true_ml_available else 0
        pending = int(_safe_float(perf.get("PendingTrades"), 0.0)) if true_ml_available else 0
        true_cost_fragile = _as_bool(strength_row.get("CostFragile"), False) if not strength_row.empty else False
        policy_cost_fragile = bool(
            not policy_cost.empty
            and "CostFragile" in policy_cost.columns
            and policy_cost["CostFragile"].map(_as_bool).any()
        )
        cost_fragile = bool(true_cost_fragile or policy_cost_fragile)
        true_drawdown = _safe_float(perf.get("MaxDrawdownPct"), np.nan) if true_ml_available else np.nan
        policy_dd = pd.to_numeric(policy_drawdown.get("MaxDrawdownPct", pd.Series(dtype=float)), errors="coerce") if not policy_drawdown.empty else pd.Series(dtype=float)
        policy_drawdown_risk = bool(not policy_dd.empty and policy_dd.min() <= -25.0)
        drawdown_risk = bool((np.isfinite(true_drawdown) and true_drawdown <= -25.0) or policy_drawdown_risk)
        overfit_risk = bool(
            not policy_overfit.empty
            and "OverfitRisk" in policy_overfit.columns
            and policy_overfit["OverfitRisk"].astype(str).str.lower().eq("high").any()
        )

        evidence = 0.0
        if true_ml_available:
            evidence += 20.0
            evidence += 10.0 if true_return > 0 else 0.0
            evidence += 20.0 if true_beats else 0.0
            evidence += 15.0 if matured >= 10 else 8.0 if matured >= 3 else 2.0 if matured > 0 else 0.0
        if leakage_passed is True:
            evidence += 10.0
        if policy_available:
            evidence += 10.0
            evidence += 10.0 if policy_beats else 0.0
            evidence += 5.0 if policy_return > 0 else 0.0
        if not drawdown_risk and (true_ml_available or policy_available):
            evidence += 5.0
        evidence = round(min(100.0, max(0.0, evidence)), 2)

        risk = 10.0
        if true_ml_available and not true_beats:
            risk += 25.0
        if policy_available and not policy_beats:
            risk += 10.0
        if true_ml_available and matured < 3:
            risk += 20.0
        elif true_ml_available and matured < 10:
            risk += 10.0
        if cost_fragile:
            risk += 15.0
        if drawdown_risk:
            risk += 15.0
        if overfit_risk:
            risk += 15.0
        if leakage_passed is False:
            risk += 40.0
        if not true_ml_available:
            risk += 20.0
        if not policy_available:
            risk += 10.0
        risk = round(min(100.0, max(0.0, risk)), 2)

        both_dominated = bool((true_ml_available and not true_beats) and (not policy_available or not policy_beats))
        if leakage_passed is False:
            label = "RejectedForNow"
            explanation = "True ML leakage checks failed, so performance evidence cannot support paper tracking."
        elif not true_ml_available and not policy_available:
            label = "InsufficientEvidence"
            explanation = "Neither true ML replay nor policy-lab evidence is available."
        elif both_dominated:
            label = "BenchmarkDominated"
            explanation = "True ML replay and available policy evidence do not beat the strongest baseline."
        elif evidence >= 65.0 and risk <= 40.0 and (true_beats or policy_beats):
            label = "PaperTrackCandidate"
            explanation = "Past-only replay or policy evidence clears a baseline hurdle with controlled visible risks."
        elif evidence >= 45.0 and risk <= 65.0:
            label = "WatchlistOnly"
            explanation = "Evidence is promising enough to monitor, but risk or sample depth remains limiting."
        elif evidence < 30.0:
            label = "InsufficientEvidence"
            explanation = "Evidence coverage or matured sample size is too limited."
        else:
            label = "ResearchOnly"
            explanation = "Mixed evidence remains useful for research but does not clear conservative tracking gates."

        identities[(asset, horizon)] = {
            "model": model,
            "policy": policy_name,
            "source": "CombinedEvidence" if true_ml_available and policy_available else "TrueHistoricalMLReplay" if true_ml_available else "Phase19PolicyLab",
        }
        rows.append(
            {
                "Asset": asset,
                "Horizon": horizon,
                "TrueMLReturnPct": round(true_return, 6) if np.isfinite(true_return) else np.nan,
                "TrueMLBeatsBestBaseline": true_beats,
                "PolicyLabReturnPct": round(policy_return, 6) if np.isfinite(policy_return) else np.nan,
                "PolicyBeatsBestBaseline": policy_beats,
                "BestBaselineName": baseline_name,
                "BestBaselineReturnPct": round(baseline_return, 6) if np.isfinite(baseline_return) else np.nan,
                "LeakagePassed": leakage_passed,
                "CostFragilityFlag": cost_fragile,
                "DrawdownRiskFlag": drawdown_risk,
                "OverfitRiskFlag": overfit_risk,
                "MaturedTrades": matured,
                "PendingTrades": pending,
                "EvidenceScore": evidence,
                "RiskScore": risk,
                "FinalResearchLabel": label,
                "Explanation": explanation,
            }
        )
    scorecard = pd.DataFrame(rows, columns=list(ASSET_HORIZON_SCORECARD_COLUMNS))
    if not scorecard.empty:
        scorecard = scorecard.sort_values(["EvidenceScore", "RiskScore", "Asset", "Horizon"], ascending=[False, True, True, True]).reset_index(drop=True)
    return scorecard, identities


def _paper_candidates(scorecard: pd.DataFrame, identities: Dict[Tuple[str, int], Dict[str, str]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if scorecard.empty:
        return pd.DataFrame(columns=list(PAPER_TRACKING_COLUMNS))
    eligible = scorecard[
        scorecard["FinalResearchLabel"].eq("PaperTrackCandidate")
        & scorecard["LeakagePassed"].map(lambda value: _as_bool(value, False))
        & (scorecard["RiskScore"] <= 40.0)
    ]
    for _, row in eligible.iterrows():
        key = (str(row["Asset"]), int(row["Horizon"]))
        limited = int(row["MaturedTrades"]) < 10
        rows.append(
            {
                "Asset": row["Asset"],
                "Horizon": int(row["Horizon"]),
                "CandidateSource": identities.get(key, {}).get("source", "CombinedEvidence"),
                "EvidenceScore": row["EvidenceScore"],
                "RiskScore": row["RiskScore"],
                "Reason": "Conservative baseline and leakage gates passed; matured evidence is still limited." if limited else "Conservative baseline, leakage, sample, and risk gates passed.",
                "RequiredNextEvidence": "Increase matured forward-paper outcomes and repeat walk-forward windows." if limited else "Confirm stability with additional windows, costs, and forward-paper outcomes.",
                "RealCapitalAllowed": False,
            }
        )
    return pd.DataFrame(rows, columns=list(PAPER_TRACKING_COLUMNS))


def _rejections(scorecard: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if scorecard.empty:
        rows.append(
            {
                "Asset": "ALL",
                "Horizon": 0,
                "RejectionReason": "InsufficientEvidence",
                "FailedChecks": "Missing Phase 19/20 asset-horizon evidence",
                "EvidenceScore": 0.0,
                "RiskScore": 100.0,
                "SuggestedFix": "Generate and save Phase 19 and Phase 20 artifacts, then rerun the command center.",
            }
        )
    else:
        rejected = scorecard[~scorecard["FinalResearchLabel"].eq("PaperTrackCandidate")]
        for _, row in rejected.iterrows():
            checks: List[str] = []
            if row["LeakagePassed"] is False or str(row["LeakagePassed"]).lower() == "false":
                checks.append("LeakageFailed")
            if not _as_bool(row["TrueMLBeatsBestBaseline"], False) and pd.notna(row["TrueMLReturnPct"]):
                checks.append("TrueMLBenchmarkDominated")
            if not _as_bool(row["PolicyBeatsBestBaseline"], False) and pd.notna(row["PolicyLabReturnPct"]):
                checks.append("PolicyBenchmarkDominated")
            if int(row["MaturedTrades"]) < 3:
                checks.append("LowSampleSize")
            if _as_bool(row["CostFragilityFlag"], False):
                checks.append("CostFragility")
            if _as_bool(row["DrawdownRiskFlag"], False):
                checks.append("DrawdownRisk")
            if _as_bool(row["OverfitRiskFlag"], False):
                checks.append("OverfitRisk")
            if not checks:
                checks.append("ConservativeScoreGateNotMet")
            rows.append(
                {
                    "Asset": row["Asset"],
                    "Horizon": int(row["Horizon"]),
                    "RejectionReason": row["FinalResearchLabel"],
                    "FailedChecks": "; ".join(checks),
                    "EvidenceScore": row["EvidenceScore"],
                    "RiskScore": row["RiskScore"],
                    "SuggestedFix": "Add leakage-safe matured windows, improve baseline edge, and reduce the listed risks.",
                }
            )
    score_lookup = {
        (str(row["Asset"]), int(row["Horizon"])): (float(row["EvidenceScore"]), float(row["RiskScore"]))
        for _, row in scorecard.iterrows()
    } if not scorecard.empty else {}
    for source_alias, reason_prefix, name_column, failed_check in (
        ("phase18_dominance_failures", "ProxyBenchmarkDominated", "DominatingBaseline", "ProxyBenchmarkDominance"),
        ("phase19_dominance_failures", "PolicyBenchmarkDominated", "PolicyName", "PolicyBenchmarkDominance"),
    ):
        source = tables[source_alias]
        if source.empty or not {"Asset", "Horizon"}.issubset(source.columns):
            continue
        for _, source_row in source.iterrows():
            horizon = _normalize_horizon(source_row.get("Horizon"))
            asset = str(source_row.get("Asset", "")).strip()
            if not asset or horizon is None:
                continue
            evidence, risk = score_lookup.get((asset, horizon), (0.0, 90.0))
            identity = str(source_row.get(name_column, "")).strip()
            rows.append(
                {
                    "Asset": asset,
                    "Horizon": horizon,
                    "RejectionReason": f"{reason_prefix}: {identity}" if identity else reason_prefix,
                    "FailedChecks": failed_check,
                    "EvidenceScore": evidence,
                    "RiskScore": risk,
                    "SuggestedFix": "Retain this dominated row as rejection evidence and require a repeatable baseline edge before reconsideration.",
                }
            )
    phase20_strength = tables["phase20_strength"]
    if not phase20_strength.empty and {"Asset", "Horizon", "StrengthClassification"}.issubset(phase20_strength.columns):
        rejected_classes = {"BenchmarkDominated", "LeakageFailed", "CostFragile", "InsufficientTrades"}
        for _, source_row in phase20_strength[phase20_strength["StrengthClassification"].astype(str).isin(rejected_classes)].iterrows():
            horizon = _normalize_horizon(source_row.get("Horizon"))
            asset = str(source_row.get("Asset", "")).strip()
            if not asset or horizon is None:
                continue
            evidence, risk = score_lookup.get((asset, horizon), (0.0, 90.0))
            classification = str(source_row.get("StrengthClassification", "InsufficientEvidence"))
            rows.append(
                {
                    "Asset": asset,
                    "Horizon": horizon,
                    "RejectionReason": f"TrueML: {classification}",
                    "FailedChecks": classification,
                    "EvidenceScore": evidence,
                    "RiskScore": risk,
                    "SuggestedFix": "Add leakage-safe matured replay windows and clear the failed true-ML evidence check.",
                }
            )
    result = pd.DataFrame(rows, columns=list(REJECTED_CANDIDATE_COLUMNS))
    if not result.empty:
        result = result.drop_duplicates(["Asset", "Horizon", "RejectionReason", "FailedChecks"]).reset_index(drop=True)
    return result


def _affected(scorecard: pd.DataFrame, mask: pd.Series) -> Tuple[str, str]:
    affected = scorecard[mask] if not scorecard.empty else pd.DataFrame()
    if affected.empty:
        return "ALL", "ALL"
    assets = "; ".join(sorted(affected["Asset"].astype(str).unique()))
    horizons = "; ".join(f"{int(value)}D" for value in sorted(affected["Horizon"].astype(int).unique()))
    return assets, horizons


def _risk_register(scorecard: pd.DataFrame, inputs: pd.DataFrame, rejections: pd.DataFrame, phase18_present: bool, broad_edge: bool) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(name: str, severity: str, assets: str, horizons: str, source: str, why: str, mitigation: str, blocks: bool = True) -> None:
        rows.append({"RiskName": name, "Severity": severity, "AffectedAssets": assets, "AffectedHorizons": horizons, "EvidenceSource": source, "WhyItMatters": why, "Mitigation": mitigation, "BlocksRealCapital": bool(blocks)})

    if not broad_edge:
        add("NoBroadEdgeProven", "High", "ALL", "ALL", "Unified scorecard", "Evidence does not beat serious baselines broadly across assets and horizons.", "Expand leakage-safe walk-forward and forward-paper evidence.")
    if not scorecard.empty:
        dominated = (~scorecard["TrueMLBeatsBestBaseline"].astype(bool) & scorecard["TrueMLReturnPct"].notna()) | (~scorecard["PolicyBeatsBestBaseline"].astype(bool) & scorecard["PolicyLabReturnPct"].notna())
        if dominated.any():
            assets, horizons = _affected(scorecard, dominated)
            add("BenchmarkDominance", "High", assets, horizons, "Phase 19/20", "Simple baselines remain stronger for affected rows.", "Reject or redesign dominated rows before further tracking.")
        for column, risk_name, severity, mitigation in (
            ("CostFragilityFlag", "CostFragility", "High", "Use asset-specific cost stress and reject fragile edges."),
            ("OverfitRiskFlag", "OverfitRisk", "High", "Increase walk-forward windows and model simplicity."),
            ("DrawdownRiskFlag", "DrawdownRisk", "High", "Review exposure controls and drawdown-aware policy filters."),
        ):
            mask = scorecard[column].astype(bool)
            if mask.any():
                assets, horizons = _affected(scorecard, mask)
                add(risk_name, severity, assets, horizons, "Phase 19/20", f"{risk_name} remains visible in the combined evidence.", mitigation)
        low_sample = scorecard["MaturedTrades"] < 3
        if low_sample.any():
            assets, horizons = _affected(scorecard, low_sample)
            add("LowSampleSize", "High", assets, horizons, "Phase 20", "Too few matured historical paper signals support affected rows.", "Increase replay windows and collect more matured forward outcomes.")
    if not any(row["RiskName"] == "BenchmarkDominance" for row in rows) and not rejections.empty and rejections["FailedChecks"].astype(str).str.contains("Benchmark", case=False).any():
        benchmark_rejections = rejections[rejections["FailedChecks"].astype(str).str.contains("Benchmark", case=False)]
        assets = "; ".join(sorted(benchmark_rejections["Asset"].astype(str).unique()))
        horizons = "; ".join(f"{int(value)}D" for value in sorted(pd.to_numeric(benchmark_rejections["Horizon"], errors="coerce").dropna().astype(int).unique()))
        add("BenchmarkDominance", "High", assets or "ALL", horizons or "ALL", "Phase 18/19/20", "At least one model, policy, or proxy row remains baseline-dominated.", "Keep dominated rows rejected until they show repeatable out-of-sample edge.")
    if phase18_present:
        add("ProxyReplayLimitations", "Medium", "ALL", "ALL", "Phase 18", "Proxy replay is useful context but is not true historical trained-model evidence.", "Use Phase 20 as the primary historical ML source.", False)
    missing = inputs[~inputs["Found"].astype(bool)] if not inputs.empty else pd.DataFrame()
    if not missing.empty:
        add("MissingArtifacts", "High", "ALL", "ALL", "Input sources", "Some expected evidence tables are unavailable.", "Generate or upload missing artifacts and rerun the command center.")
    add("DeploymentNotHardened", "High", "ALL", "ALL", "Project-wide", "Research workflows are not an execution or deployment system.", "Add operational validation, monitoring, and failure recovery before any eligibility review.")
    add("RealCapitalBlocked", "Critical", "ALL", "ALL", "Phase 21", "Current evidence remains research-only.", "Continue paper tracking and preserve upstream capital gates.")
    return pd.DataFrame(rows, columns=list(RISK_REGISTER_COLUMNS))


def _summary(scorecard: pd.DataFrame, candidates: pd.DataFrame, inputs: pd.DataFrame, identities: Dict[Tuple[str, int], Dict[str, str]]) -> pd.DataFrame:
    broad_threshold = max(3, int(np.ceil(len(scorecard) / 3.0))) if len(scorecard) else 3
    broad_edge = bool(len(candidates) >= broad_threshold)
    best = scorecard.iloc[0] if not scorecard.empty else pd.Series(dtype=object)
    key = (str(best.get("Asset", "")), int(best.get("Horizon", 0))) if not best.empty else ("", 0)
    identity = identities.get(key, {})
    true_return = _safe_float(best.get("TrueMLReturnPct"), np.nan) if not best.empty else np.nan
    policy_return = _safe_float(best.get("PolicyLabReturnPct"), np.nan) if not best.empty else np.nan
    if np.isfinite(true_return):
        best_return = true_return
        source = "TrueHistoricalMLReplay"
        best_name = identity.get("model", "")
    elif np.isfinite(policy_return):
        best_return = policy_return
        source = "Phase19PolicyLab"
        best_name = identity.get("policy", "")
    else:
        best_return = np.nan
        source = "InsufficientEvidence"
        best_name = ""
    baseline_return = _safe_float(best.get("BestBaselineReturnPct"), np.nan) if not best.empty else np.nan
    gap = best_return - baseline_return if np.isfinite(best_return) and np.isfinite(baseline_return) else np.nan
    phase20_found = bool(inputs[inputs["ExpectedFile"].eq("phase20_true_ml_performance.csv")]["Found"].any()) if not inputs.empty else False
    leakage_values = scorecard["LeakagePassed"].dropna().map(_as_bool) if not scorecard.empty else pd.Series(dtype=bool)
    leakage_status = "Passed" if phase20_found and not leakage_values.empty and leakage_values.all() else "Failed" if not leakage_values.empty and not leakage_values.all() else "InsufficientEvidence"
    dominated_count = int(scorecard["FinalResearchLabel"].eq("BenchmarkDominated").sum()) if not scorecard.empty else 0
    if scorecard.empty:
        verdict = "InsufficientEvidence"
    elif dominated_count == len(scorecard):
        verdict = "BenchmarkDominated"
    elif not candidates.empty:
        verdict = "NarrowPaperTrackCandidatesOnly" if not broad_edge else "ResearchOnlyContinue"
    elif phase20_found:
        verdict = "NoBroadEdgeProven"
    else:
        verdict = "EvidenceImprovingButInsufficient"
    explanation = (
        "The command center combines true historical ML replay, policy-lab, and proxy audit evidence. "
        f"{len(candidates)} conservative paper-tracking rows passed; real capital remains blocked."
    )
    return pd.DataFrame(
        [{
            "CommandCenterVerdict": verdict,
            "BroadEdgeStatus": "ResearchOnly" if broad_edge else "NoBroadEdgeProven",
            "BestEvidenceSource": source,
            "BestAsset": str(best.get("Asset", "")) if not best.empty else "",
            "BestHorizon": int(best.get("Horizon", 0)) if not best.empty else 0,
            "BestModelOrPolicy": best_name,
            "BestNetReturnPct": round(best_return, 6) if np.isfinite(best_return) else np.nan,
            "BestBaselineGapPct": round(gap, 6) if np.isfinite(gap) else np.nan,
            "LeakageStatus": leakage_status,
            "BenchmarkStatus": "BenchmarkDominated" if dominated_count else "MixedOrNarrowEvidence",
            "CostFragilityStatus": "RiskPresent" if not scorecard.empty and scorecard["CostFragilityFlag"].astype(bool).any() else "NoFlagInAvailableEvidence",
            "DrawdownRiskStatus": "RiskPresent" if not scorecard.empty and scorecard["DrawdownRiskFlag"].astype(bool).any() else "NoFlagInAvailableEvidence",
            "OverfitRiskStatus": "RiskPresent" if not scorecard.empty and scorecard["OverfitRiskFlag"].astype(bool).any() else "NoFlagInAvailableEvidence",
            "RealCapitalStatus": "Blocked",
            "RecommendedMode": "PaperTrack" if not candidates.empty else "WatchlistOnly",
            "FinalExplanation": explanation,
        }],
        columns=list(UNIFIED_SUMMARY_COLUMNS),
    )


def _quality_gates(inputs: pd.DataFrame, scorecard: pd.DataFrame, risk_register: pd.DataFrame, rejections: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    required_phase20_files = {
        "phase20_true_ml_performance.csv",
        "phase20_true_ml_baseline_comparison.csv",
        "phase20_leakage_audit.csv",
    }
    phase20_inputs = inputs[inputs["ExpectedFile"].isin(required_phase20_files)] if not inputs.empty else pd.DataFrame()
    phase20_available = bool(
        len(phase20_inputs) == len(required_phase20_files)
        and phase20_inputs["Found"].astype(bool).all()
    )
    leakage_rows = scorecard["LeakagePassed"].dropna() if not scorecard.empty else pd.Series(dtype=object)
    leakage_passed = bool(phase20_available and not leakage_rows.empty and leakage_rows.map(_as_bool).all())
    baseline_available = bool(not scorecard.empty and scorecard["BestBaselineName"].astype(str).ne("").any())
    weak_visible = bool(not rejections.empty)
    output_text = "\n".join(frame.astype(str).to_csv(index=False) for frame in [summary, scorecard, risk_register, rejections])
    prohibited = re.compile(r"\b(Buy|Strong Buy|Invest Now|Guaranteed Profit|Safe Profit|Production Ready Trading)\b", flags=re.IGNORECASE)
    no_forbidden = prohibited.search(output_text) is None
    gates = [
        ("Phase20Available", phase20_available, "Critical", "True historical ML replay evidence is available."),
        ("LeakageAuditPassed", leakage_passed, "Critical", "Available true ML rows pass chronological leakage checks."),
        ("BaselineComparisonAvailable", baseline_available, "Critical", "Serious baseline comparisons are represented in the scorecard."),
        ("WeakResultsVisible", weak_visible, "High", "Weak or dominated rows remain visible."),
        ("RejectionsVisible", not rejections.empty, "High", "Rejected and insufficient rows are retained."),
        ("RealCapitalBlocked", True, "Critical", "Real capital remains blocked for every Phase 21 result."),
        ("NoForbiddenClaims", no_forbidden, "Critical", "Command-center outputs avoid prohibited claims."),
        ("MissingArtifactsHandled", True, "High", "Missing optional artifacts produce warnings rather than crashes."),
        ("ScorecardGenerated", not scorecard.empty, "High", "Asset-horizon evidence scorecard was generated."),
        ("RiskRegisterGenerated", not risk_register.empty, "High", "Current risks remain visible in a unified register."),
    ]
    return pd.DataFrame([{"GateName": name, "Passed": bool(passed), "Severity": severity, "Explanation": explanation} for name, passed, severity, explanation in gates], columns=list(QUALITY_GATE_COLUMNS))


def _next_actions() -> pd.DataFrame:
    rows = [
        (1, "Increase leakage-safe walk-forward windows.", "Small historical samples weaken confidence.", "Improves stability and sample depth.", "Phase 20 extension"),
        (2, "Test additional lightweight models under identical splits.", "Model diversity should be evaluated without changing chronology.", "Checks whether narrow edge survives model choice.", "Phase 20 model comparison"),
        (3, "Improve asset-specific cost modeling.", "Generic costs can hide execution fragility.", "Makes paper economics more realistic.", "Cost-model research"),
        (4, "Continue command-center UI and report export hardening.", "Evidence must stay understandable and reproducible.", "Improves review quality without changing results.", "Phase 21 UI hardening"),
        (5, "Prepare deployment hardening research without enabling capital.", "Operational failure modes remain untested.", "Defines monitoring, rollback, and data-quality requirements.", "Future infrastructure research"),
    ]
    return pd.DataFrame([{"Priority": priority, "Action": action, "Reason": reason, "ExpectedImpact": impact, "PhaseSuggestion": phase} for priority, action, reason, impact, phase in rows], columns=list(NEXT_ACTION_COLUMNS))


def run_unified_risk_command_center(
    *,
    use_artifact_store: bool = True,
    prefer_uploaded: bool = False,
    uploaded_overrides: Optional[Dict[str, Any]] = None,
    autosave: bool = False,
    **direct_tables: Any,
) -> UnifiedRiskCommandCenterReport:
    """Combine available Phase 18/19/20 evidence without retraining."""
    tables, input_sources, source_metadata = _resolve_inputs(
        bool(use_artifact_store), bool(prefer_uploaded), uploaded_overrides, direct_tables
    )
    scorecard, identities = _build_scorecard(tables)
    candidates = _paper_candidates(scorecard, identities)
    rejections = _rejections(scorecard, tables)
    broad_threshold = max(3, int(np.ceil(len(scorecard) / 3.0))) if len(scorecard) else 3
    broad_edge = bool(len(candidates) >= broad_threshold)
    phase18_present = bool(input_sources[input_sources["SourcePhase"].eq("phase18_replay_benchmark_audit")]["Found"].any())
    risks = _risk_register(scorecard, input_sources, rejections, phase18_present, broad_edge)
    summary = _summary(scorecard, candidates, input_sources, identities)
    quality = _quality_gates(input_sources, scorecard, risks, rejections, summary)
    actions = _next_actions()
    settings = {
        "phase": "21",
        "purpose": "unified_performance_risk_intelligence",
        "real_capital_status": "Blocked",
        "use_artifact_store": bool(use_artifact_store),
        "prefer_uploaded": bool(prefer_uploaded),
    }
    report = UnifiedRiskCommandCenterReport(
        unified_summary_table=summary.reset_index(drop=True),
        asset_horizon_scorecard=scorecard.reset_index(drop=True),
        risk_register=risks.reset_index(drop=True),
        paper_tracking_candidates=candidates.reset_index(drop=True),
        rejected_candidates=rejections.reset_index(drop=True),
        quality_gates=quality.reset_index(drop=True),
        next_actions=actions.reset_index(drop=True),
        input_sources=input_sources.reset_index(drop=True),
        settings=settings,
    )
    if autosave:
        report.saved_artifacts = save_phase_artifacts(
            UNIFIED_RISK_COMMAND_CENTER_PHASE_NAME,
            {
                "phase21_unified_summary": report.unified_summary_table,
                "phase21_asset_horizon_scorecard": report.asset_horizon_scorecard,
                "phase21_risk_register": report.risk_register,
                "phase21_paper_tracking_candidates": report.paper_tracking_candidates,
                "phase21_rejected_candidates": report.rejected_candidates,
                "phase21_quality_gates": report.quality_gates,
                "phase21_next_actions": report.next_actions,
                "phase21_input_sources": report.input_sources,
            },
            inputs=source_metadata,
            config=settings,
            warnings=input_sources.loc[~input_sources["Found"].astype(bool), "Warning"].dropna().astype(str).tolist(),
        )
    return report


__all__ = [
    "UNIFIED_RISK_COMMAND_CENTER_PHASE_NAME",
    "UNIFIED_SUMMARY_COLUMNS",
    "ASSET_HORIZON_SCORECARD_COLUMNS",
    "RISK_REGISTER_COLUMNS",
    "PAPER_TRACKING_COLUMNS",
    "REJECTED_CANDIDATE_COLUMNS",
    "QUALITY_GATE_COLUMNS",
    "NEXT_ACTION_COLUMNS",
    "INPUT_SOURCE_COLUMNS",
    "UnifiedRiskCommandCenterReport",
    "run_unified_risk_command_center",
]
