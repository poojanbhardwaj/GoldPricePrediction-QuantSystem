from pathlib import Path
import ast
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.app_context import AVAILABLE_HORIZONS, SUPPORTED_ASSETS
from src.research_orchestrator import (
    ADVANCED_DIAGNOSTIC_PAGES,
    PRIMARY_USER_PAGES,
    SNAPSHOT_COLUMNS,
    build_navigation_audit,
    build_phase26_quality_gates,
    collect_asset_horizon_evidence,
    safe_run_module,
)
from src.user_plan_generator import (
    ALLOWED_PLAN_STATUSES,
    generate_all_asset_plans,
    generate_asset_plan,
    generate_portfolio_plan,
    rank_asset_plans,
)


FORBIDDEN = re.compile(
    r"\b(Buy|Strong Buy|Sell|Hold|Invest Now|Guaranteed Profit|Safe Profit|Production Ready Trading)\b",
    flags=re.IGNORECASE,
)
REQUIRED_PLAN_FIELDS = {
    "Summary",
    "Why",
    "MainRisk",
    "WhatToWatch",
    "TrackingCondition",
    "InvalidationCondition",
    "RecheckWhen",
}


def _source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def _evidence_row(**overrides):
    row = {
        "Asset": "Gold",
        "Horizon": 5,
        "Source": "SyntheticEvidence",
        "Metric": "ProbabilityUp",
        "Value": 0.62,
        "Status": "AvailableEvidence",
        "Warning": "",
        "Freshness": "Current",
        "EvidenceStrength": 70.0,
    }
    row.update(overrides)
    return row


def test_phase26_modules_import_and_public_functions_are_callable():
    assert callable(safe_run_module)
    assert callable(generate_asset_plan)
    assert callable(generate_all_asset_plans)
    assert callable(generate_portfolio_plan)
    assert callable(rank_asset_plans)


def test_phase26_missing_optional_module_returns_fallback():
    fallback = {"Status": "Not Enough Evidence"}
    result = safe_run_module("src.module_that_does_not_exist", "missing_function", fallback)
    assert result == fallback


def test_phase26_missing_evidence_generates_all_asset_horizon_plans_without_crash():
    plans = generate_all_asset_plans(pd.DataFrame(columns=SNAPSHOT_COLUMNS))

    assert len(plans) == len(SUPPORTED_ASSETS) * len(AVAILABLE_HORIZONS)
    assert set(plans["Asset"]) == set(SUPPORTED_ASSETS)
    assert set(plans["Horizon"]) == set(AVAILABLE_HORIZONS)
    assert plans["Status"].eq("Not Enough Evidence").all()
    assert not plans["RealMoneyApproved"].astype(bool).any()


def test_phase26_only_simple_allowed_statuses_are_generated():
    evidence = pd.DataFrame([_evidence_row()])
    plans = generate_all_asset_plans(evidence)
    assert set(plans["Status"]).issubset(set(ALLOWED_PLAN_STATUSES))


def test_phase26_high_risk_evidence_creates_high_risk_or_avoid():
    evidence = pd.DataFrame(
        [
            _evidence_row(
                Metric="DrawdownRisk",
                Value=-38.0,
                Status="Critical",
                Warning="Severe risk and probability unreliability remain active.",
                EvidenceStrength=65.0,
            )
        ]
    )
    plan = generate_asset_plan("Gold", 5, evidence)
    assert plan["Status"] in {"High Risk", "Avoid"}
    assert plan["RealMoneyApproved"] is False


def test_phase26_weak_benchmark_evidence_prevents_track():
    evidence = pd.DataFrame(
        [
            _evidence_row(Status="BenchmarkDominated", EvidenceStrength=90.0),
            _evidence_row(Metric="PredictedReturn", Value=2.4, EvidenceStrength=90.0),
        ]
    )
    plan = generate_asset_plan("Gold", 5, evidence)
    assert plan["Status"] in {"Watch", "Wait"}
    assert plan["Status"] != "Track"


def test_phase26_generated_plans_have_required_fields_and_no_forbidden_copy():
    plans = generate_all_asset_plans(pd.DataFrame([_evidence_row()]))
    assert REQUIRED_PLAN_FIELDS.issubset(plans.columns)
    for _, plan in plans.iterrows():
        assert all(str(plan[field]).strip() for field in REQUIRED_PLAN_FIELDS)
        user_text = " ".join(str(value) for value in plan.tolist())
        assert FORBIDDEN.search(user_text) is None


def test_phase26_portfolio_plan_never_approves_real_money():
    plans = generate_all_asset_plans(pd.DataFrame([_evidence_row()]))
    portfolio = generate_portfolio_plan(plans)
    assert not portfolio.empty
    assert not portfolio["RealMoneyApproved"].astype(bool).any()
    assert FORBIDDEN.search(" ".join(portfolio.astype(str).stack().tolist())) is None


def test_phase26_collect_evidence_handles_empty_snapshot():
    evidence = collect_asset_horizon_evidence("Bitcoin", 10, pd.DataFrame(columns=SNAPSHOT_COLUMNS))
    assert not evidence.empty
    assert evidence.iloc[0]["Status"] == "Not Enough Evidence"


def test_phase26_primary_navigation_has_no_phase_names_and_advanced_exists():
    assert PRIMARY_USER_PAGES == (
        "Market Research Assistant",
        "Asset Plans",
        "Forecast Explorer",
        "Portfolio Summary",
        "About / Methodology",
    )
    assert all(re.search(r"\bPhase\s+\d+", label, flags=re.IGNORECASE) is None for label in PRIMARY_USER_PAGES)
    assert len(ADVANCED_DIAGNOSTIC_PAGES) >= 45
    app_source = _source("app.py")
    assert '["Advanced Diagnostics"]' in app_source
    assert "Advanced diagnostic page. Normal users should use Market Research Assistant or Asset Plans first." in app_source


def test_phase26_navigation_audit_covers_every_app_route():
    app_source = _source("app.py")
    route_labels = set(re.findall(r'(?:if|elif) page == "([^"]+)"', app_source))
    audit = build_navigation_audit()
    assert route_labels.issubset(set(audit["PageLabel"]))
    assert audit["IsPrimaryUserPage"].sum() == len(PRIMARY_USER_PAGES)
    assert audit["IsAdvancedDiagnostic"].sum() >= 45


def test_phase26_quality_gates_are_complete():
    plans = generate_all_asset_plans(pd.DataFrame([_evidence_row()]))
    gates = build_phase26_quality_gates(pd.DataFrame([_evidence_row()]), plans)
    expected = {
        "MarketResearchAssistantAvailable", "AssetPlansAvailable", "ForecastExplorerAvailable",
        "PortfolioSummaryAvailable", "AdvancedDiagnosticsAvailable", "AllExistingPagesAudited",
        "PhaseNamesHiddenFromPrimaryNavigation", "AdvancedPagesStillAccessible", "AllAssetsCovered",
        "AllHorizonsCovered", "PlansGenerated", "MissingArtifactsHandledGracefully",
        "NoForbiddenClaims", "NoRealMoneyApproval", "SimpleStatusesOnly",
        "TechnicalEvidenceHiddenByDefault", "AssetRoutingConsistent", "AppDoesNotCrash",
    }
    assert set(gates["GateName"]) == expected
    assert gates["Passed"].astype(bool).all()


def test_phase26_app_compiles_and_primary_technical_evidence_is_hidden_by_default():
    app_source = _source("app.py")
    ast.parse(app_source)
    assert 'show_advanced_evidence = st.checkbox' in app_source
    assert 'value=False' in app_source.split('show_advanced_evidence = st.checkbox', 1)[1].split(')', 1)[0]
    assert 'if show_advanced_evidence:' in app_source
    assert 'explorer_target = get_asset_target(explorer_asset)' in app_source
    assert 'file_name=f"{_safe_filename_part(explorer_asset)}_' in app_source
