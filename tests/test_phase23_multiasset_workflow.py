from pathlib import Path
import ast
import re
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.app_context import (
    AVAILABLE_HORIZONS,
    DEFAULT_ASSET,
    DEFAULT_HORIZON,
    SUPPORTED_ASSETS,
    build_data_freshness_table,
    get_asset_target,
    validate_asset_horizon,
)
from src.explanation_glossary import REQUIRED_GLOSSARY_TERMS, explain_term
from src.workflow_guide import (
    build_multiasset_coverage_table,
    build_page_audit_table,
    build_quality_gates_table,
    get_workflow_steps,
    run_multiasset_workflow_audit,
)


EXPECTED_ASSETS = {
    "Gold": "Gold_Close",
    "Silver": "Silver_Close",
    "Crude Oil": "Oil_Close",
    "Bitcoin": "BTC_Close",
    "S&P 500": "SP500_Close",
    "Gold ETF": "GLD_Close",
}
EXPECTED_HORIZONS = (1, 5, 10, 20, 30)
EXPECTED_GATES = {
    "OldPagesAudited",
    "MultiAssetContextAvailable",
    "GuidedWorkflowAvailable",
    "GlossaryAvailable",
    "DataFreshnessPanelAvailable",
    "NoHardcodedGoldOnMainPages",
    "LegacyPagesClearlyMarked",
    "RealCapitalBlocked",
    "NoForbiddenClaims",
    "AppDoesNotCrashOnMissingArtifacts",
}
FORBIDDEN_LANGUAGE = re.compile(
    r"\b(Buy|Strong Buy|Invest Now|Guaranteed Profit|Safe Profit|Production Ready Trading)\b",
    flags=re.IGNORECASE,
)


def _source(path):
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def _synthetic_market_data():
    dates = pd.bdate_range("2026-01-05", periods=20)
    data = {column: 100.0 + np.arange(len(dates)) for column in EXPECTED_ASSETS.values()}
    return pd.DataFrame(data, index=dates)


def test_phase23_context_supports_all_assets_and_horizons():
    assert set(SUPPORTED_ASSETS) == set(EXPECTED_ASSETS)
    assert AVAILABLE_HORIZONS == EXPECTED_HORIZONS
    assert DEFAULT_ASSET in SUPPORTED_ASSETS
    assert DEFAULT_HORIZON in AVAILABLE_HORIZONS
    for asset, target in EXPECTED_ASSETS.items():
        assert get_asset_target(asset) == target
        for horizon in AVAILABLE_HORIZONS:
            assert validate_asset_horizon(asset, horizon) is True


def test_phase23_workflow_steps_are_complete_and_ordered():
    steps = get_workflow_steps()
    assert [step["StepNumber"] for step in steps] == list(range(1, 9))
    assert [step["StepName"] for step in steps] == [
        "Data & Feature Health",
        "Forecast / Prediction Range",
        "Signal Research",
        "Validation & Evidence",
        "Risk Intelligence",
        "Benchmarking & Replay",
        "Unified Verdict",
        "Reports & Exports",
    ]
    assert all(step["NextRecommendedPage"] for step in steps)
    assert all(step["WeakEvidenceWarning"] for step in steps)


def test_phase23_glossary_explains_every_required_term():
    required = {
        "Asset", "Horizon", "Prediction Date", "Target Outcome Date", "Predicted Return",
        "Realized Return", "Net Return", "Baseline", "Hold-only", "Momentum baseline",
        "Moving-average baseline", "Random baseline", "Leakage", "Walk-forward validation",
        "Drawdown", "Cost drag", "Slippage", "Hit rate", "Sharpe-like", "Sortino-like",
        "Calmar-like", "Benchmark dominated", "PaperTrack", "WatchlistOnly",
        "RealCapitalBlocked", "NoBroadEdgeProven",
    }
    assert required.issubset(set(REQUIRED_GLOSSARY_TERMS))
    assert all(explain_term(term) and "No glossary" not in explain_term(term) for term in required)


def test_phase23_data_freshness_and_coverage_are_multiasset():
    market = _synthetic_market_data()
    freshness = build_data_freshness_table(market, as_of=market.index.max())
    coverage = build_multiasset_coverage_table()

    assert set(freshness["Asset"]) == set(EXPECTED_ASSETS)
    assert freshness["LatestAssetDate"].astype(str).str.len().gt(0).all()
    assert len(coverage) == len(EXPECTED_ASSETS) * len(EXPECTED_HORIZONS)
    assert set(coverage["Horizon"]) == set(EXPECTED_HORIZONS)
    assert coverage["CapitalStatus"].eq("RealCapitalBlocked").all()


def test_phase23_report_handles_missing_data_and_exposes_all_tables():
    report = run_multiasset_workflow_audit(market_data=None, app_source=_source("app.py"))

    assert not report.page_audit_table.empty
    assert not report.multiasset_coverage_table.empty
    assert not report.workflow_steps_table.empty
    assert not report.glossary_terms_table.empty
    assert not report.data_freshness_table.empty
    assert not report.quality_gates_table.empty
    assert not report.next_actions_table.empty
    assert report.data_freshness_table["FreshnessStatus"].eq("MissingData").all()


def test_phase23_page_audit_keeps_legacy_diagnostics_visible_and_marked():
    audit = build_page_audit_table()
    legacy = audit[audit["LegacyStatus"].ne("None")]

    assert {"30-Day Forecast", "Backtesting", "Research Validation"}.issubset(set(legacy["Page"]))
    assert legacy["UserGuidance"].astype(str).str.len().gt(0).all()
    assert legacy["RecommendedReplacement"].astype(str).str.len().gt(0).all()


def test_phase23_quality_gates_exist_and_pass_for_current_app():
    gates = build_quality_gates_table(app_source=_source("app.py"))

    assert EXPECTED_GATES == set(gates["GateName"])
    assert gates["Passed"].astype(bool).all()


def test_phase23_app_uses_friendly_visible_navigation_and_blocked_capital_language():
    app_source = _source("app.py")
    ast.parse(app_source)
    navigation = app_source.split("NAVIGATION_GROUPS =", 1)[1].split("navigation_group =", 1)[0]

    assert "Guided Research Workflow" in navigation
    assert "Walk-Forward ML Replay" in navigation
    assert "Model Edge Benchmark Lab" in navigation
    assert "Unified Risk Command Center" in navigation
    for phase_number in ("Phase 19", "Phase 20", "Phase 21", "Phase 22"):
        assert phase_number not in navigation
    assert "RealCapitalBlocked" in app_source
    assert "B.Tech Final Year Project" not in app_source


def test_phase23_main_forecast_paths_do_not_default_directly_to_gold_target():
    app_source = _source("app.py")
    disallowed_patterns = [
        r'target_col\s*:\s*str\s*=\s*["\']Gold_Close["\']',
        r'getattr\([^\n]+["\']Gold_Close["\']',
        r'\[["\']Gold_Close["\']\]',
    ]

    assert all(re.search(pattern, app_source) is None for pattern in disallowed_patterns)
    assert "get_asset_target(selected_asset)" in app_source


def test_phase23_forbidden_claims_are_absent():
    for path in [
        "app.py",
        "src/app_context.py",
        "src/workflow_guide.py",
        "src/explanation_glossary.py",
    ]:
        assert FORBIDDEN_LANGUAGE.search(_source(path)) is None, path
