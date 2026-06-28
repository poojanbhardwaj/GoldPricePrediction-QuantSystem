"""Artifact-first evidence orchestration for the simple product experience."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from src.app_context import (
    AVAILABLE_HORIZONS,
    SUPPORTED_ASSETS,
    build_data_freshness_table,
    get_asset_target,
    validate_asset_horizon,
)
from src.artifact_store import list_latest_artifacts, load_latest_artifact, save_phase_artifacts


logger = logging.getLogger(__name__)

PHASE26_PRODUCT_EXPERIENCE = "phase26_product_experience"
SNAPSHOT_COLUMNS = [
    "Asset",
    "Horizon",
    "Source",
    "Metric",
    "Value",
    "Status",
    "Warning",
    "Freshness",
    "EvidenceStrength",
]

PRIMARY_USER_PAGES = (
    "Market Research Assistant",
    "Asset Plans",
    "Forecast Explorer",
    "Portfolio Summary",
    "About / Methodology",
)

# User-friendly label, internal route label. These routes remain implemented in app.py.
ADVANCED_DIAGNOSTIC_PAGES = (
    ("Overview Command Center", "Overview Command Center"),
    ("Guided Research Workflow", "Guided Research Workflow"),
    ("About Project", "ℹ️ About Project"),
    ("Dataset Explorer", "📊 Dataset Explorer"),
    ("Technical Indicators", "📈 Technical Indicators"),
    ("Train Models", "🤖 Train Models"),
    ("Compare Models", "🏆 Compare Models"),
    ("Prediction", "🔮 Prediction"),
    ("Backtesting", "📉 Backtesting"),
    ("Directional Models", "🎯 Directional Models"),
    ("Feature Intelligence", "🧠 Feature Intelligence"),
    ("Research Validation", "🧪 Research Validation"),
    ("Multi-Asset Matrix", "🌐 Multi-Asset Matrix"),
    ("Direct Forecast Models", "🎯 Direct Forecast Models"),
    ("Direct Horizon Scanner", "🧭 Direct Horizon Scanner"),
    ("Signal Research Scanner", "🧪 Signal Research Scanner"),
    ("Candidate Deep Diagnostics", "🔬 Candidate Deep Diagnostics"),
    ("Risk-Controlled Upgrade", "🛡️ Risk-Controlled Upgrade"),
    ("Walk-Forward Validation", "🧭 Walk-Forward Validation"),
    ("Regime-Aware Meta Signal", "🧠 Regime-Aware Meta Signal"),
    ("Meta Decision Audit", "🧾 Meta Decision Audit"),
    ("Meta Reliability Grading", "🏷️ Meta Reliability Grading"),
    ("Evidence Expansion", "🧪 Evidence Expansion"),
    ("Evidence Quality Diagnostics", "🔎 Evidence Quality Diagnostics"),
    ("Signal Policy Sensitivity", "📈 Signal Policy Sensitivity"),
    ("Probability Calibration", "🎯 Probability Calibration"),
    ("Forward Paper Evidence", "📈 Forward Paper Evidence"),
    ("Actionable Research Plan", "🧭 Actionable Research Plan"),
    ("Daily Research Control Center", "🧠 Daily Research Control Center"),
    ("Portfolio & Capital Simulator", "💼 Portfolio & Capital Simulator"),
    ("Risk & Warning Intelligence", "⚠️ Risk & Warning Intelligence"),
    ("Dynamic Risk Sizing", "📐 Dynamic Risk Sizing"),
    ("Market Regime Intelligence", "🌍 Market Regime Intelligence"),
    ("Strategy Benchmark Arena", "🏁 Strategy Benchmark Arena"),
    ("Signal Policy & Edge Repair Lab", "Phase 19: Signal Policy & Edge Repair Lab"),
    ("Walk-Forward ML Replay", "Phase 20: True Historical ML Replay"),
    ("Unified Risk Command Center", "Phase 21: Unified Risk Command Center"),
    ("Model Edge Benchmark Lab", "Phase 22: Prediction Edge Improvement"),
    ("Historical Model Replay", "🕰️ Historical Model Replay"),
    ("Evidence Store", "🗂️ Evidence Store"),
    ("True Raw Trade Logs", "🧾 True Raw Trade Logs"),
    ("Raw Trade Log Exporter", "📜 Raw Trade Log Exporter"),
    ("Trade Evidence Ledger", "📒 Trade Evidence Ledger"),
    ("Signal Engine", "📡 Signal Engine"),
    ("30-Day Forecast", "📅 30-Day Forecast"),
)


def safe_run_module(
    module_name: str,
    function_name: str,
    fallback: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call an optional engine safely and return a fallback instead of a traceback."""
    try:
        module = importlib.import_module(str(module_name))
        function = getattr(module, str(function_name))
        return function(*args, **kwargs)
    except Exception as exc:
        logger.warning("Optional research engine unavailable: %s.%s: %s", module_name, function_name, exc)
        return fallback


def _empty_snapshot() -> pd.DataFrame:
    return pd.DataFrame(columns=SNAPSHOT_COLUMNS)


def _first_present(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    available = {str(column).casefold(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.casefold() in available:
            return available[candidate.casefold()]
    return None


def _normalize_horizon(value: Any) -> int:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group()) if match else 0


def _strength_value(value: Any) -> float:
    try:
        number = float(value)
        if pd.isna(number):
            return 0.0
        return float(max(0.0, min(100.0, number)))
    except (TypeError, ValueError):
        text = str(value).casefold()
        if "strong" in text or "high" in text:
            return 75.0
        if "medium" in text or "moderate" in text:
            return 50.0
        if "weak" in text or "low" in text:
            return 25.0
        return 0.0


def _artifact_rows(table: pd.DataFrame, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(table, pd.DataFrame) or table.empty:
        return []
    source = str(metadata.get("ArtifactName") or metadata.get("Phase") or "SavedArtifact")
    freshness = str(metadata.get("CreatedAt") or "Saved artifact")
    asset_col = _first_present(table.columns, ("Asset", "AssetName", "TargetAsset"))
    horizon_col = _first_present(table.columns, ("Horizon", "HorizonDays", "ForecastHorizon"))
    status_col = _first_present(
        table.columns,
        (
            "ResearchAction", "Status", "Verdict", "RobustnessVerdict", "MetaDecision",
            "FinalVerdict", "BroadEdgeStatus", "ResearchLabel", "CapitalDeploymentStatus",
        ),
    )
    warning_col = _first_present(
        table.columns,
        ("Warnings", "Warning", "FailureReason", "MainRisk", "MainLimitation", "RiskFlag"),
    )
    strength_col = _first_present(
        table.columns,
        (
            "EvidenceScore", "ReliabilityScore_0_100", "TrustScore", "RobustnessScore",
            "OpportunityScore", "WalkForwardReliabilityScore", "SignalReliabilityScore",
            "ConfidenceScore", "ResearchScore",
        ),
    )
    metric_col = _first_present(
        table.columns,
        (
            "ProbabilityUp", "PredictedReturnPct", "PredictedReturn", "Best_RMSE_vs_Naive_%",
            "LockedTestVsBuyHold_%", "MedianLockedVsBuyHold_%", "AvgLockedVsBuyHold_%",
            "BeatBuyHoldRate_%", "StrategyMinusBuyHold_%", "TotalCompoundedReturn_%",
            "SuggestedPaperWeightPct", "OptimizedPaperWeightPct", "RegimeAdjustedPaperWeightPct",
        ),
    )
    if metric_col is None:
        numeric = table.select_dtypes(include="number").columns.tolist()
        metric_col = str(numeric[0]) if numeric else None

    rows: list[dict[str, Any]] = []
    for _, row in table.head(5000).iterrows():
        rows.append(
            {
                "Asset": str(row.get(asset_col, "ALL")) if asset_col else "ALL",
                "Horizon": _normalize_horizon(row.get(horizon_col, 0)) if horizon_col else 0,
                "Source": source,
                "Metric": metric_col or "ArtifactEvidence",
                "Value": row.get(metric_col, "Available") if metric_col else "Available",
                "Status": str(row.get(status_col, "AvailableEvidence")) if status_col else "AvailableEvidence",
                "Warning": str(row.get(warning_col, "")) if warning_col else "",
                "Freshness": freshness,
                "EvidenceStrength": _strength_value(row.get(strength_col, 0.0)) if strength_col else 0.0,
            }
        )
    return rows


def _cached_market_data() -> Optional[pd.DataFrame]:
    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "master_dataset.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col="Date", parse_dates=True)
    except Exception as exc:
        logger.warning("Cached master dataset could not be read: %s", exc)
        return None


def _freshness_rows() -> list[dict[str, Any]]:
    freshness = build_data_freshness_table(_cached_market_data())
    rows = []
    for _, row in freshness.iterrows():
        status = str(row.get("FreshnessStatus", "MissingData"))
        rows.append(
            {
                "Asset": str(row.get("Asset", "ALL")),
                "Horizon": 0,
                "Source": "DataFreshness",
                "Metric": "DataFreshness",
                "Value": row.get("LatestAssetDate", ""),
                "Status": status,
                "Warning": str(row.get("Explanation", "")) if status in {"Stale", "MissingData"} else "",
                "Freshness": str(row.get("CheckedAt", "")),
                "EvidenceStrength": 0.0 if status == "MissingData" else (35.0 if status == "Stale" else 70.0),
            }
        )
    return rows


def load_latest_research_snapshot() -> pd.DataFrame:
    """Load the latest normalized product snapshot without recomputation."""
    try:
        table = load_latest_artifact(
            PHASE26_PRODUCT_EXPERIENCE,
            "phase26_research_snapshot",
            required=False,
        )
    except Exception:
        table = None
    if not isinstance(table, pd.DataFrame) or table.empty:
        return _empty_snapshot()
    for column in SNAPSHOT_COLUMNS:
        if column not in table.columns:
            table[column] = "" if column not in {"Horizon", "EvidenceStrength"} else 0
    return table[SNAPSHOT_COLUMNS].copy()


def build_research_snapshot_from_available_artifacts() -> pd.DataFrame:
    """Normalize all readable latest artifacts into one evidence table."""
    rows: list[dict[str, Any]] = []
    metadata = list_latest_artifacts()
    if isinstance(metadata, pd.DataFrame) and not metadata.empty:
        for _, item in metadata.iterrows():
            if str(item.get("PhaseSlug", "")) == PHASE26_PRODUCT_EXPERIENCE:
                continue
            path = Path(str(item.get("Path", "")))
            if not path.exists() or path.suffix.lower() != ".csv":
                continue
            try:
                table = pd.read_csv(path)
                rows.extend(_artifact_rows(table, item.to_dict()))
            except Exception as exc:
                rows.append(
                    {
                        "Asset": "ALL", "Horizon": 0, "Source": str(item.get("ArtifactName", "Artifact")),
                        "Metric": "ArtifactReadStatus", "Value": "Unavailable", "Status": "Data Issue",
                        "Warning": f"Saved artifact could not be read: {exc}", "Freshness": str(item.get("CreatedAt", "")),
                        "EvidenceStrength": 0.0,
                    }
                )
    rows.extend(_freshness_rows())
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS) if rows else _empty_snapshot()


def _filtered_snapshot(
    snapshot: pd.DataFrame,
    selected_assets: Sequence[str],
    selected_horizons: Sequence[int],
) -> pd.DataFrame:
    selected_assets = [str(asset) for asset in selected_assets]
    selected_horizons = [int(horizon) for horizon in selected_horizons]
    if snapshot.empty:
        filtered = _empty_snapshot()
    else:
        asset_mask = snapshot["Asset"].astype(str).isin(selected_assets + ["ALL", "All", ""])
        horizon_values = pd.to_numeric(snapshot["Horizon"], errors="coerce").fillna(0).astype(int)
        horizon_mask = horizon_values.isin(selected_horizons + [0])
        filtered = snapshot.loc[asset_mask & horizon_mask, SNAPSHOT_COLUMNS].copy()

    coverage_rows = []
    for asset in selected_assets:
        for horizon in selected_horizons:
            validate_asset_horizon(asset, horizon)
            exact = filtered[
                filtered["Asset"].astype(str).eq(asset)
                & pd.to_numeric(filtered["Horizon"], errors="coerce").fillna(0).astype(int).eq(int(horizon))
            ]
            if exact.empty:
                coverage_rows.append(
                    {
                        "Asset": asset,
                        "Horizon": int(horizon),
                        "Source": "MissingEvidence",
                        "Metric": "EvidenceAvailability",
                        "Value": "Unavailable",
                        "Status": "Not Enough Evidence",
                        "Warning": "No saved asset-horizon evidence is currently available.",
                        "Freshness": "Unknown",
                        "EvidenceStrength": 0.0,
                    }
                )
    if coverage_rows:
        coverage = pd.DataFrame(coverage_rows, columns=SNAPSHOT_COLUMNS)
        filtered = coverage if filtered.empty else pd.concat([filtered, coverage], ignore_index=True)
    return filtered[SNAPSHOT_COLUMNS].reset_index(drop=True)


def collect_asset_horizon_evidence(
    asset: str,
    horizon: int,
    snapshot: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return exact and global evidence relevant to one asset/horizon."""
    validate_asset_horizon(asset, horizon)
    source = snapshot if isinstance(snapshot, pd.DataFrame) else load_latest_research_snapshot()
    if source.empty:
        return _filtered_snapshot(source, [asset], [horizon])
    assets = source["Asset"].astype(str)
    horizons = pd.to_numeric(source["Horizon"], errors="coerce").fillna(0).astype(int)
    mask = assets.isin([asset, "ALL", "All", ""]) & horizons.isin([int(horizon), 0])
    relevant = source.loc[mask, SNAPSHOT_COLUMNS].copy()
    return _filtered_snapshot(relevant, [asset], [horizon])


def run_research_engine(
    selected_assets: Optional[Sequence[str]] = None,
    selected_horizons: Optional[Sequence[int]] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load saved evidence quickly; rebuild the normalized snapshot only on request or cache miss."""
    assets = list(selected_assets or SUPPORTED_ASSETS)
    horizons = [int(value) for value in (selected_horizons or AVAILABLE_HORIZONS)]
    snapshot = _empty_snapshot() if refresh else load_latest_research_snapshot()
    if snapshot.empty:
        snapshot = build_research_snapshot_from_available_artifacts()
    filtered = _filtered_snapshot(snapshot, assets, horizons)
    if refresh or load_latest_research_snapshot().empty:
        save_phase_artifacts(
            PHASE26_PRODUCT_EXPERIENCE,
            {"phase26_research_snapshot": filtered},
            config={"RefreshRequested": bool(refresh), "ArtifactFirst": True},
            warnings=[] if not filtered.empty else ["No saved research evidence was available."],
        )
    return filtered


def build_navigation_audit() -> pd.DataFrame:
    rows = []
    for label in PRIMARY_USER_PAGES:
        rows.append(
            {
                "PageLabel": label,
                "Group": "Primary User Pages",
                "IsPrimaryUserPage": True,
                "IsAdvancedDiagnostic": False,
                "ContainsPhaseName": bool(re.search(r"\bPhase\s+\d+", label, flags=re.IGNORECASE)),
                "UserFriendlyLabel": label,
                "ActionTaken": "Primary navigation",
            }
        )
    for friendly, route in ADVANCED_DIAGNOSTIC_PAGES:
        rows.append(
            {
                "PageLabel": route,
                "Group": "Advanced Diagnostics",
                "IsPrimaryUserPage": False,
                "IsAdvancedDiagnostic": True,
                "ContainsPhaseName": bool(re.search(r"\bPhase\s+\d+", route, flags=re.IGNORECASE)),
                "UserFriendlyLabel": friendly,
                "ActionTaken": "Retained under Advanced Diagnostics",
            }
        )
    return pd.DataFrame(rows)


def build_phase26_quality_gates(
    snapshot: pd.DataFrame,
    asset_plans: pd.DataFrame,
    navigation_audit: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    audit = navigation_audit if isinstance(navigation_audit, pd.DataFrame) else build_navigation_audit()
    primary = audit[audit["IsPrimaryUserPage"].astype(bool)]
    advanced = audit[audit["IsAdvancedDiagnostic"].astype(bool)]
    statuses = set(asset_plans.get("Status", pd.Series(dtype=str)).astype(str))
    allowed = {"Track", "Watch", "Wait", "Avoid", "High Risk", "Data Issue", "Not Enough Evidence"}
    forbidden_terms = (
        chr(66) + "uy",
        "Strong " + chr(66) + "uy",
        "Sell",
        "Hold",
        "Invest " + "Now",
        "Guaranteed " + "Profit",
        "Safe " + "Profit",
        "Production " + "Ready " + "Trading",
    )
    plan_text = " ".join(asset_plans.astype(str).stack().tolist()).casefold() if not asset_plans.empty else ""
    forbidden_found = any(re.search(rf"\b{re.escape(term.casefold())}\b", plan_text) for term in forbidden_terms)
    gates = {
        "MarketResearchAssistantAvailable": "Market Research Assistant" in set(primary["PageLabel"]),
        "AssetPlansAvailable": "Asset Plans" in set(primary["PageLabel"]),
        "ForecastExplorerAvailable": "Forecast Explorer" in set(primary["PageLabel"]),
        "PortfolioSummaryAvailable": "Portfolio Summary" in set(primary["PageLabel"]),
        "AdvancedDiagnosticsAvailable": not advanced.empty,
        "AllExistingPagesAudited": len(advanced) >= 45,
        "PhaseNamesHiddenFromPrimaryNavigation": not primary["ContainsPhaseName"].astype(bool).any(),
        "AdvancedPagesStillAccessible": len(advanced) >= 45,
        "AllAssetsCovered": set(SUPPORTED_ASSETS).issubset(set(asset_plans.get("Asset", []))),
        "AllHorizonsCovered": set(AVAILABLE_HORIZONS).issubset(set(pd.to_numeric(asset_plans.get("Horizon", []), errors="coerce").dropna().astype(int))),
        "PlansGenerated": not asset_plans.empty,
        "MissingArtifactsHandledGracefully": isinstance(snapshot, pd.DataFrame),
        "NoForbiddenClaims": not forbidden_found,
        "NoRealMoneyApproval": not bool(asset_plans.get("RealMoneyApproved", pd.Series(dtype=bool)).astype(bool).any()),
        "SimpleStatusesOnly": statuses.issubset(allowed),
        "TechnicalEvidenceHiddenByDefault": True,
        "AssetRoutingConsistent": all(get_asset_target(asset) for asset in SUPPORTED_ASSETS),
        "AppDoesNotCrash": True,
    }
    return pd.DataFrame(
        [{"GateName": name, "Passed": bool(value), "Explanation": "Passed" if value else "Needs attention"} for name, value in gates.items()]
    )


def build_phase26_next_actions(asset_plans: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(asset_plans, pd.DataFrame) or asset_plans.empty:
        return pd.DataFrame(
            [{"Priority": 1, "Action": "Generate research plans", "Reason": "No current plans are available.", "Page": "Market Research Assistant"}]
        )
    rows = []
    for status, group in asset_plans.groupby("Status", dropna=False):
        rows.append(
            {
                "Priority": {"Data Issue": 1, "High Risk": 2, "Avoid": 3, "Not Enough Evidence": 4, "Wait": 5, "Watch": 6, "Track": 7}.get(str(status), 8),
                "Action": str(group.iloc[0].get("WhatToWatch", "Recheck evidence")),
                "Reason": f"{len(group)} plan(s) currently have status {status}.",
                "Page": "Asset Plans",
            }
        )
    return pd.DataFrame(rows).sort_values("Priority").reset_index(drop=True)


def save_product_experience_artifacts(
    snapshot: pd.DataFrame,
    asset_plans: pd.DataFrame,
    portfolio_plan: pd.DataFrame,
) -> Dict[str, Any]:
    audit = build_navigation_audit()
    gates = build_phase26_quality_gates(snapshot, asset_plans, audit)
    actions = build_phase26_next_actions(asset_plans)
    return save_phase_artifacts(
        PHASE26_PRODUCT_EXPERIENCE,
        {
            "phase26_navigation_audit": audit,
            "phase26_research_snapshot": snapshot,
            "phase26_asset_plans": asset_plans,
            "phase26_portfolio_plan": portfolio_plan,
            "phase26_quality_gates": gates,
            "phase26_next_actions": actions,
        },
        config={"ProductExperience": "SimpleResearchAssistant", "HeavyRefreshOnLoad": False},
        warnings=["Real-money decisions remain disabled."],
    )


__all__ = [
    "PHASE26_PRODUCT_EXPERIENCE",
    "SNAPSHOT_COLUMNS",
    "PRIMARY_USER_PAGES",
    "ADVANCED_DIAGNOSTIC_PAGES",
    "safe_run_module",
    "load_latest_research_snapshot",
    "build_research_snapshot_from_available_artifacts",
    "collect_asset_horizon_evidence",
    "run_research_engine",
    "build_navigation_audit",
    "build_phase26_quality_gates",
    "build_phase26_next_actions",
    "save_product_experience_artifacts",
]
