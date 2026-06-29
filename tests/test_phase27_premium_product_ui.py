from pathlib import Path
import ast
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.app_context import AVAILABLE_HORIZONS, SUPPORTED_ASSETS
from src.ui_components import (
    inject_premium_css,
    render_advanced_evidence_expander,
    render_asset_plan_card,
    render_confidence_badge,
    render_disclaimer_banner,
    render_empty_state,
    render_glass_container,
    render_hero_section,
    render_metric_card,
    render_monitoring_card,
    render_navigation_card,
    render_opportunity_card,
    render_opportunity_score,
    render_premium_header,
    render_risk_explanation_card,
    render_status_badge,
    render_status_tabs,
)
from src.user_plan_generator import (
    ALLOWED_BLOCK_REASONS,
    build_high_risk_explanations,
    build_monitoring_plan,
    build_phase27_ui_quality_gates,
    generate_all_asset_plans,
    generate_asset_plan,
    generate_portfolio_plan,
    rank_asset_plans,
)


FORBIDDEN = re.compile(
    r"\b(Buy|Strong Buy|Sell|Hold|Invest Now|Guaranteed Profit|Safe Profit|Production Ready Trading)\b",
    flags=re.IGNORECASE,
)


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _evidence_row(asset="Gold", horizon=5, **overrides):
    row = {
        "Asset": asset,
        "Horizon": horizon,
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


def test_phase27_premium_ui_helpers_import_and_are_callable():
    helpers = [
        inject_premium_css, render_premium_header, render_hero_section, render_status_badge,
        render_confidence_badge, render_opportunity_score, render_metric_card, render_asset_plan_card,
        render_opportunity_card, render_risk_explanation_card, render_monitoring_card,
        render_empty_state, render_disclaimer_banner, render_advanced_evidence_expander,
        render_navigation_card, render_status_tabs, render_glass_container,
    ]
    assert all(callable(helper) for helper in helpers)


def test_phase27_opportunity_fields_are_generated_and_bounded():
    plans = generate_all_asset_plans(pd.DataFrame([_evidence_row()]))
    required = {
        "OpportunityScore", "OpportunityGrade", "ClosestToTrackRank", "RecheckPriority",
        "BlockReason", "ImprovementNeeded", "WhatUserShouldMonitorNext", "UserFriendlyNextStep",
    }
    assert required.issubset(plans.columns)
    assert plans["OpportunityScore"].between(0, 100).all()
    assert set(plans["OpportunityGrade"]).issubset({"A", "B", "C", "D", "F"})
    assert set(plans["RecheckPriority"]).issubset({"High", "Medium", "Low"})
    assert set(plans["BlockReason"]).issubset(set(ALLOWED_BLOCK_REASONS))
    assert plans["ImprovementNeeded"].astype(str).str.len().gt(0).all()
    assert plans["WhatUserShouldMonitorNext"].astype(str).str.len().gt(0).all()


def test_phase27_opportunity_rank_covers_every_asset_and_horizon():
    plans = generate_all_asset_plans(pd.DataFrame([_evidence_row()]))
    ranked = rank_asset_plans(plans)
    assert len(ranked) == len(SUPPORTED_ASSETS) * len(AVAILABLE_HORIZONS)
    assert set(ranked["ClosestToTrackRank"]) == set(range(1, len(ranked) + 1))
    assert set(ranked["Asset"]) == set(SUPPORTED_ASSETS)
    assert set(ranked["Horizon"]) == set(AVAILABLE_HORIZONS)


def test_phase27_weak_benchmark_evidence_never_becomes_track():
    evidence = pd.DataFrame([
        _evidence_row(Status="BenchmarkDominated", EvidenceStrength=92.0),
        _evidence_row(Metric="PredictedReturn", Value=2.5, EvidenceStrength=92.0),
    ])
    plan = generate_asset_plan("Gold", 5, evidence)
    assert plan["Status"] in {"Watch", "Wait"}
    assert plan["Status"] != "Track"
    assert plan["BlockReason"] == "Benchmark Weakness"
    assert 0 <= plan["OpportunityScore"] <= 100


def test_phase27_high_risk_plans_keep_status_and_useful_next_steps():
    evidence = pd.DataFrame([
        _evidence_row(
            Metric="DrawdownRisk", Value=-42.0, Status="Critical",
            Warning="Severe risk and probability unreliability remain active.", EvidenceStrength=68.0,
        )
    ])
    plan = generate_asset_plan("Gold", 5, evidence)
    assert plan["Status"] in {"High Risk", "Avoid"}
    assert plan["OpportunityScore"] <= 49
    assert plan["WhatMustImprove"]
    assert plan["WhatUserShouldMonitorNext"]
    assert plan["NextReviewTrigger"]
    assert plan["UserFriendlyNextStep"]
    assert plan["RealMoneyApproved"] is False


def test_phase27_many_high_risk_plans_explain_portfolio_caution():
    evidence = pd.DataFrame([
        _evidence_row(
            asset=asset, horizon=horizon, Metric="DrawdownRisk", Value=-35.0,
            Status="Critical", Warning="Severe risk remains active.", EvidenceStrength=60.0,
        )
        for asset in SUPPORTED_ASSETS
        for horizon in AVAILABLE_HORIZONS
    ])
    plans = generate_all_asset_plans(evidence)
    assert plans["Status"].isin({"High Risk", "Avoid"}).all()
    assert plans["WhyEverythingIsHighRisk"].astype(str).str.contains("risk warnings", case=False).all()
    portfolio = generate_portfolio_plan(plans)
    assert "cautious" in str(portfolio.iloc[0]["WhySystemIsCautious"]).casefold() or "risk" in str(portfolio.iloc[0]["WhySystemIsCautious"]).casefold()


def test_phase27_explanation_and_monitoring_tables_keep_failures_visible():
    evidence = pd.DataFrame([
        _evidence_row(
            asset=asset, horizon=5, Metric="DrawdownRisk", Status="Critical",
            Warning="High risk and overconfident probability evidence.", EvidenceStrength=55.0,
        )
        for asset in SUPPORTED_ASSETS
    ])
    plans = generate_all_asset_plans(evidence)
    explanations = build_high_risk_explanations(plans)
    monitoring = build_monitoring_plan(plans)
    assert not explanations.empty
    assert {"BlockReason", "WhatMustImprove", "WhatUserShouldMonitorNext"}.issubset(explanations.columns)
    assert len(monitoring) == len(plans)


def test_phase27_quality_gates_are_complete_and_pass():
    plans = generate_all_asset_plans(pd.DataFrame([_evidence_row()]))
    gates = build_phase27_ui_quality_gates(plans, _source("app.py"))
    expected = {
        "PremiumHeroAvailable", "PremiumCardsAvailable", "PrimaryPagesUseCards",
        "RawTablesHiddenByDefault", "OpportunityScoresGenerated", "OpportunityGradesGenerated",
        "ClosestToTrackRankingAvailable", "HighRiskExplanationAvailable", "WhatMustImproveAvailable",
        "WhatUserShouldMonitorNextAvailable", "RecheckPriorityGenerated", "NoForbiddenClaims",
        "NoRealMoneyApproval", "AdvancedDiagnosticsStillAvailable", "PhaseNamesHiddenFromPrimaryNavigation",
        "ForecastExplorerAssetRoutingStillCorrect", "DeprecatedStreamlitWidthWarningsReduced", "AppDoesNotCrash",
    }
    assert set(gates["GateName"]) == expected
    assert gates["Passed"].astype(bool).all()


def test_phase27_primary_navigation_is_product_named_and_advanced_remains():
    source = _source("app.py")
    navigation = source.split("NAVIGATION_GROUPS =", 1)[1].split("navigation_group =", 1)[0]
    assert "Advanced Diagnostics" in source
    assert all(label in navigation for label in (
        "Data & Features", "Forecasting & Models", "Signals & Plans", "Risk & Regime",
        "Backtesting & Replay", "Evidence & Quality Gates",
    ))
    for phase_number in ("Phase 23", "Phase 24", "Phase 25", "Phase 26", "Phase 27"):
        assert phase_number not in navigation


def test_phase27_forecast_explorer_asset_routing_is_still_explicit():
    source = _source("app.py")
    forecast = source.split('elif page == "Forecast Explorer"', 1)[1].split('elif page == "Portfolio Summary"', 1)[0]
    assert "explorer_target = get_asset_target(explorer_asset)" in forecast
    assert "market_history[[explorer_target]]" in forecast
    assert 'file_name=f"{_safe_filename_part(explorer_asset)}_' in forecast
    assert "The app will not substitute another asset" in forecast


def test_phase27_public_outputs_have_no_forbidden_claims_or_real_money_approval():
    plans = generate_all_asset_plans(pd.DataFrame([_evidence_row()]))
    portfolio = generate_portfolio_plan(plans)
    public_text = " ".join(plans.astype(str).stack().tolist() + portfolio.astype(str).stack().tolist())
    assert FORBIDDEN.search(public_text) is None
    assert not plans["RealMoneyApproved"].astype(bool).any()
    assert not portfolio["RealMoneyApproved"].astype(bool).any()


def test_phase27_python_files_compile_as_source():
    for path in ["app.py", "src/ui_components.py", "src/user_plan_generator.py", "tests/test_phase27_premium_product_ui.py"]:
        ast.parse(_source(path))

