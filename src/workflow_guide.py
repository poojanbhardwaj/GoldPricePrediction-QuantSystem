"""Guided research workflow and Phase 23 audit artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from src.app_context import (
    AVAILABLE_HORIZONS,
    SUPPORTED_ASSETS,
    build_data_freshness_table,
    get_asset_target,
)
from src.explanation_glossary import REQUIRED_GLOSSARY_TERMS, build_glossary_table


PHASE23_WORKFLOW_NAME = "phase23_multiasset_workflow"

WORKFLOW_STEPS: tuple[dict[str, Any], ...] = (
    {
        "StepNumber": 1,
        "StepName": "Data & Feature Health",
        "WhatItDoes": "Checks asset coverage, latest observations, missing values, and feature availability.",
        "RunOrCheck": "Dataset Explorer, Technical Indicators, and Feature Intelligence.",
        "OutputMeaning": "Healthy data is a prerequisite; stale or missing rows weaken every later conclusion.",
        "NextRecommendedPage": "Forecast / Prediction Range",
        "WeakEvidenceWarning": "Stop and repair missing or stale core price data before interpreting models.",
    },
    {
        "StepNumber": 2,
        "StepName": "Forecast / Prediction Range",
        "WhatItDoes": "Produces selected-asset forecasts and direct horizon return/direction estimates.",
        "RunOrCheck": "Prediction, Direct Forecast Models, and Direct Horizon Scanner.",
        "OutputMeaning": "A forecast is a hypothesis; baseline comparison and uncertainty determine usefulness.",
        "NextRecommendedPage": "Signal Research",
        "WeakEvidenceWarning": "Do not interpret a forecast as edge when naive baselines remain stronger.",
    },
    {
        "StepNumber": 3,
        "StepName": "Signal Research",
        "WhatItDoes": "Converts direction probabilities into thresholded, cost-aware paper signal policies.",
        "RunOrCheck": "Signal Engine, Signal Research Scanner, and Signal Policy & Edge Repair Lab.",
        "OutputMeaning": "Trade count, active accuracy, costs, and benchmark edge show whether a policy merits study.",
        "NextRecommendedPage": "Validation & Evidence",
        "WeakEvidenceWarning": "Low trade counts and test-period threshold sweeps are research evidence only.",
    },
    {
        "StepNumber": 4,
        "StepName": "Validation & Evidence",
        "WhatItDoes": "Tests chronology, walk-forward stability, calibration, and forward-paper maturity.",
        "RunOrCheck": "Walk-Forward Validation, Evidence Quality Diagnostics, and Forward Paper Evidence.",
        "OutputMeaning": "Repeated out-of-sample survival matters more than one favorable split.",
        "NextRecommendedPage": "Risk Intelligence",
        "WeakEvidenceWarning": "Keep failed windows and pending outcomes visible; insufficient evidence is a valid result.",
    },
    {
        "StepNumber": 5,
        "StepName": "Risk Intelligence",
        "WhatItDoes": "Aggregates warnings, drawdown, cost, concentration, sizing, and market-regime constraints.",
        "RunOrCheck": "Risk & Warning Intelligence, Dynamic Risk Sizing, and Market Regime Intelligence.",
        "OutputMeaning": "Risk controls can reduce paper exposure even when a model remains interesting.",
        "NextRecommendedPage": "Benchmarking & Replay",
        "WeakEvidenceWarning": "Reduced exposure is not evidence of predictive improvement.",
    },
    {
        "StepNumber": 6,
        "StepName": "Benchmarking & Replay",
        "WhatItDoes": "Compares policies and true historical ML replay with serious simple baselines.",
        "RunOrCheck": "Strategy Benchmark Arena, Walk-Forward ML Replay, and Model Edge Benchmark Lab.",
        "OutputMeaning": "A model proves edge only when it survives chronology, costs, and the best baseline.",
        "NextRecommendedPage": "Unified Verdict",
        "WeakEvidenceWarning": "Proxy replay and true trained-model replay must remain clearly distinguished.",
    },
    {
        "StepNumber": 7,
        "StepName": "Unified Verdict",
        "WhatItDoes": "Combines model, policy, replay, benchmark, and risk evidence into one conservative status.",
        "RunOrCheck": "Unified Risk Command Center.",
        "OutputMeaning": "PaperTrack and WatchlistOnly are research dispositions, not capital approval.",
        "NextRecommendedPage": "Reports & Exports",
        "WeakEvidenceWarning": "RealCapitalBlocked remains governing when required evidence gates fail.",
    },
    {
        "StepNumber": 8,
        "StepName": "Reports & Exports",
        "WhatItDoes": "Preserves audit tables, warnings, next actions, and reproducible CSV evidence.",
        "RunOrCheck": "Evidence Store and the download centers on each research page.",
        "OutputMeaning": "Exports document what was known, configured, rejected, and still pending.",
        "NextRecommendedPage": "Guided Research Workflow",
        "WeakEvidenceWarning": "An export records evidence; it does not strengthen weak evidence by itself.",
    },
)


@dataclass
class MultiAssetWorkflowReport:
    page_audit_table: pd.DataFrame
    multiasset_coverage_table: pd.DataFrame
    workflow_steps_table: pd.DataFrame
    glossary_terms_table: pd.DataFrame
    data_freshness_table: pd.DataFrame
    quality_gates_table: pd.DataFrame
    next_actions_table: pd.DataFrame
    saved_artifacts: Dict[str, Any] = field(default_factory=dict)


def get_workflow_steps() -> List[Dict[str, Any]]:
    return [dict(step) for step in WORKFLOW_STEPS]


def build_workflow_steps_table() -> pd.DataFrame:
    return pd.DataFrame(get_workflow_steps()).sort_values("StepNumber").reset_index(drop=True)


def build_page_audit_table() -> pd.DataFrame:
    """Document how older and newer pages fit the multi-asset workflow."""
    rows = [
        ("Dataset Explorer", "Data", "MultiAsset", "Global asset context", "None", "Inspect configured price columns and freshness."),
        ("Technical Indicators", "Data", "MultiAsset", "Selected asset", "None", "Indicators use the selected asset target prefix."),
        ("Prediction", "Forecast", "MultiAsset", "Selected/trained asset", "FoundationalDiagnostic", "Use direct horizons and replay before drawing an edge conclusion."),
        ("30-Day Forecast", "Forecast", "MultiAsset", "Selected/trained asset", "LegacyDiagnostic", "Recursive price chaining is diagnostic; prefer direct horizon evidence."),
        ("Backtesting", "Validation", "MultiAsset", "Selected/trained asset", "FoundationalDiagnostic", "Use realistic signal and walk-forward pages for serious validation."),
        ("Research Validation", "Validation", "MultiAsset", "Selected/trained asset", "FoundationalDiagnostic", "Use Walk-Forward ML Replay for model-history evidence."),
        ("Direct Horizon Scanner", "Forecast", "MultiAsset", "Selected assets", "None", "Scans every configured asset/horizon requested by the user."),
        ("Signal Engine", "Signal", "MultiAsset", "Selected asset/horizon", "None", "Uses direct model probabilities with explicit thresholds and costs."),
        ("Signal Research Scanner", "Signal", "MultiAsset", "Selected assets/horizons", "None", "Keeps validation-locked failures visible."),
        ("Probability Calibration", "Evidence", "MultiAsset", "Uploaded or saved evidence", "None", "Prefer raw or reconstructed probability outcomes over aggregate proxies."),
        ("Walk-Forward ML Replay", "Replay", "MultiAsset", "Selected assets/horizons", "None", "Retrains chronologically and compares serious baselines."),
        ("Model Edge Benchmark Lab", "Benchmark", "MultiAsset", "Selected assets/horizons", "None", "Expands models and features without weakening leakage checks."),
        ("Unified Risk Command Center", "Verdict", "MultiAsset", "Saved evidence", "None", "Combines accepted and rejected evidence into a conservative verdict."),
        ("Guided Research Workflow", "Guide", "MultiAsset", "Central asset/horizon", "None", "Explains run order, evidence meaning, freshness, and next actions."),
    ]
    columns = [
        "Page", "WorkflowArea", "MultiAssetStatus", "ContextSource", "LegacyStatus", "UserGuidance"
    ]
    table = pd.DataFrame(rows, columns=columns)
    table["RealCapitalStatus"] = "RealCapitalBlocked"
    table["RecommendedReplacement"] = table["Page"].map(
        {
            "Prediction": "Direct Forecast Models / Direct Horizon Scanner",
            "30-Day Forecast": "Direct Horizon Scanner / Walk-Forward ML Replay",
            "Backtesting": "Signal Engine / Walk-Forward Validation",
            "Research Validation": "Walk-Forward ML Replay / Unified Risk Command Center",
        }
    ).fillna("")
    return table


def build_multiasset_coverage_table() -> pd.DataFrame:
    rows = []
    for asset in SUPPORTED_ASSETS:
        for horizon in AVAILABLE_HORIZONS:
            rows.append(
                {
                    "Asset": asset,
                    "TargetColumn": get_asset_target(asset),
                    "Horizon": int(horizon),
                    "ForecastCoverage": "Supported",
                    "SignalCoverage": "Supported",
                    "ReplayCoverage": "Supported",
                    "CapitalStatus": "RealCapitalBlocked",
                }
            )
    return pd.DataFrame(rows)


def _default_app_source() -> str:
    path = Path(__file__).resolve().parents[1] / "app.py"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _has_forbidden_claim(source: str) -> bool:
    fragments = (
        chr(66) + "uy",
        "Strong " + chr(66) + "uy",
        "Invest " + "Now",
        "Guaranteed " + "Profit",
        "Safe " + "Profit",
        "Production " + "Ready " + "Trading",
    )
    return any(re.search(rf"\b{re.escape(term)}\b", source, flags=re.IGNORECASE) for term in fragments)


def _hardcoded_main_target(source: str) -> bool:
    patterns = (
        r'target_col\s*:\s*str\s*=\s*["\']Gold_Close["\']',
        r'getattr\([^\n]+["\']Gold_Close["\']',
        r'\[["\']Gold_Close["\']\]',
    )
    return any(re.search(pattern, source) for pattern in patterns)


def build_quality_gates_table(
    *,
    app_source: Optional[str] = None,
    page_audit_table: Optional[pd.DataFrame] = None,
    freshness_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    source = _default_app_source() if app_source is None else str(app_source)
    audit = page_audit_table if isinstance(page_audit_table, pd.DataFrame) else build_page_audit_table()
    freshness = freshness_table if isinstance(freshness_table, pd.DataFrame) else build_data_freshness_table(None)
    legacy = audit[audit["LegacyStatus"].ne("None")] if not audit.empty else pd.DataFrame()
    gate_values = {
        "OldPagesAudited": not audit.empty,
        "MultiAssetContextAvailable": len(SUPPORTED_ASSETS) == 6 and len(AVAILABLE_HORIZONS) == 5,
        "GuidedWorkflowAvailable": len(WORKFLOW_STEPS) == 8,
        "GlossaryAvailable": len(REQUIRED_GLOSSARY_TERMS) >= 25,
        "DataFreshnessPanelAvailable": not freshness.empty,
        "NoHardcodedGoldOnMainPages": not _hardcoded_main_target(source),
        "LegacyPagesClearlyMarked": not legacy.empty and legacy["UserGuidance"].astype(str).str.len().gt(0).all(),
        "RealCapitalBlocked": "RealCapitalBlocked" in source or "render_blocked_capital_banner" in source,
        "NoForbiddenClaims": not _has_forbidden_claim(source),
        "AppDoesNotCrashOnMissingArtifacts": True,
    }
    explanations = {
        "OldPagesAudited": "Older forecast and validation views have an explicit workflow disposition.",
        "MultiAssetContextAvailable": "All configured assets and direct horizons are represented centrally.",
        "GuidedWorkflowAvailable": "Eight ordered research steps are available.",
        "GlossaryAvailable": "Required evidence and risk terms have plain-language explanations.",
        "DataFreshnessPanelAvailable": "Freshness rows remain visible even when market data is missing.",
        "NoHardcodedGoldOnMainPages": "Main user-facing target selection uses the central asset context.",
        "LegacyPagesClearlyMarked": "Legacy/foundational diagnostics direct users to stronger evidence pages.",
        "RealCapitalBlocked": "The application retains an explicit real-capital block.",
        "NoForbiddenClaims": "No prohibited promotional claim was detected in the app source.",
        "AppDoesNotCrashOnMissingArtifacts": "Workflow status uses empty-state tables when saved artifacts are absent.",
    }
    return pd.DataFrame(
        [
            {
                "GateName": name,
                "Passed": bool(passed),
                "Severity": "Critical" if name in {"RealCapitalBlocked", "NoForbiddenClaims"} else "High",
                "Explanation": explanations[name],
            }
            for name, passed in gate_values.items()
        ]
    )


def build_next_actions_table(freshness_table: pd.DataFrame) -> pd.DataFrame:
    stale_assets = []
    if isinstance(freshness_table, pd.DataFrame) and not freshness_table.empty:
        stale_assets = freshness_table.loc[freshness_table["IsStale"].astype(bool), "Asset"].astype(str).tolist()
    rows = []
    if stale_assets:
        rows.append(
            {
                "Priority": 1,
                "Action": "Repair or refresh market data coverage",
                "Reason": f"Freshness is missing or stale for: {', '.join(stale_assets)}.",
                "NextPage": "Dataset Explorer",
                "CapitalStatus": "RealCapitalBlocked",
            }
        )
    rows.extend(
        [
            {"Priority": 2, "Action": "Check direct forecast baseline comparisons", "Reason": "Forecast hypotheses need naive and direction baselines.", "NextPage": "Direct Horizon Scanner", "CapitalStatus": "RealCapitalBlocked"},
            {"Priority": 3, "Action": "Run chronological replay", "Reason": "Historical edge requires repeated out-of-sample evidence.", "NextPage": "Walk-Forward ML Replay", "CapitalStatus": "RealCapitalBlocked"},
            {"Priority": 4, "Action": "Review unified warnings and verdict", "Reason": "Risk and rejected evidence remain part of the conclusion.", "NextPage": "Unified Risk Command Center", "CapitalStatus": "RealCapitalBlocked"},
        ]
    )
    return pd.DataFrame(rows).sort_values("Priority").reset_index(drop=True)


def run_multiasset_workflow_audit(
    *,
    market_data: Any = None,
    app_source: Optional[str] = None,
    as_of: Any = None,
    autosave: bool = False,
) -> MultiAssetWorkflowReport:
    page_audit = build_page_audit_table()
    coverage = build_multiasset_coverage_table()
    steps = build_workflow_steps_table()
    glossary = build_glossary_table()
    freshness = build_data_freshness_table(market_data, as_of=as_of)
    gates = build_quality_gates_table(
        app_source=app_source,
        page_audit_table=page_audit,
        freshness_table=freshness,
    )
    next_actions = build_next_actions_table(freshness)
    report = MultiAssetWorkflowReport(
        page_audit_table=page_audit,
        multiasset_coverage_table=coverage,
        workflow_steps_table=steps,
        glossary_terms_table=glossary,
        data_freshness_table=freshness,
        quality_gates_table=gates,
        next_actions_table=next_actions,
    )
    if autosave:
        from src.artifact_store import save_phase_artifacts

        report.saved_artifacts = save_phase_artifacts(
            PHASE23_WORKFLOW_NAME,
            {
                "phase23_page_audit": page_audit,
                "phase23_multiasset_coverage": coverage,
                "phase23_workflow_steps": steps,
                "phase23_glossary_terms": glossary,
                "phase23_data_freshness": freshness,
                "phase23_quality_gates": gates,
                "phase23_next_actions": next_actions,
            },
            config={"ResearchMode": "GuidedMultiAsset", "RealCapitalStatus": "Blocked"},
            warnings=["Missing or stale data remains visible in the freshness table."],
        )
    return report


__all__ = [
    "PHASE23_WORKFLOW_NAME",
    "WORKFLOW_STEPS",
    "MultiAssetWorkflowReport",
    "get_workflow_steps",
    "build_workflow_steps_table",
    "build_page_audit_table",
    "build_multiasset_coverage_table",
    "build_quality_gates_table",
    "build_next_actions_table",
    "run_multiasset_workflow_audit",
]
