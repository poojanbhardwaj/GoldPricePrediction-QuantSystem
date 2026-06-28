from pathlib import Path
import ast
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_components import (
    inject_premium_css,
    render_blocked_capital_banner,
    render_download_buttons,
    render_glossary_expander,
    render_metric_grid,
    render_pipeline_stepper,
    render_research_disclaimer,
    render_safe_table,
    render_section_header,
    render_status_card,
)


FORBIDDEN_LANGUAGE = re.compile(
    r"\b(Buy|Strong Buy|Invest Now|Guaranteed Profit|Safe Profit|Production Ready Trading)\b",
    flags=re.IGNORECASE,
)


def _source(path):
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_phase24_ui_helpers_import_and_are_callable():
    helpers = [
        inject_premium_css,
        render_status_card,
        render_metric_grid,
        render_section_header,
        render_research_disclaimer,
        render_blocked_capital_banner,
        render_safe_table,
        render_download_buttons,
        render_pipeline_stepper,
        render_glossary_expander,
    ]
    assert all(callable(helper) for helper in helpers)


def test_phase24_app_ui_and_tests_compile_as_python():
    for path in ["app.py", "src/ui_components.py", "tests/test_phase24_premium_ui.py"]:
        ast.parse(_source(path))


def test_phase24_forbidden_language_is_absent():
    for path in ["app.py", "README.md", "src/ui_components.py"]:
        assert FORBIDDEN_LANGUAGE.search(_source(path)) is None, path


def test_phase24_removes_legacy_academic_and_single_asset_branding():
    app_source = _source("app.py")
    readme_source = _source("README.md")

    assert "B.Tech Final Year Project" not in app_source
    assert "B.Tech Final Year Project" not in readme_source
    assert "Gold Price Prediction AI" not in app_source
    assert "Gold Price Prediction AI" not in readme_source


def test_phase24_visible_navigation_uses_product_names_not_phase_numbers():
    app_source = _source("app.py")
    navigation = app_source.split("NAVIGATION_GROUPS =", 1)[1].split("navigation_group =", 1)[0]

    for phase_number in ("Phase 19", "Phase 20", "Phase 21", "Phase 22", "Phase 23", "Phase 24"):
        assert phase_number not in navigation
    for label in (
        "Guided Research Workflow",
        "Signal Policy & Edge Repair Lab",
        "Walk-Forward ML Replay",
        "Unified Risk Command Center",
        "Model Edge Benchmark Lab",
    ):
        assert label in navigation


def test_phase24_real_capital_block_remains_prominent():
    app_source = _source("app.py")
    ui_source = _source("src/ui_components.py")

    assert "render_blocked_capital_banner" in app_source
    assert "Real capital status: Blocked" in ui_source
    assert "RealCapitalBlocked" in app_source


def test_phase24_overview_uses_executive_cards_and_exact_pipeline():
    app_source = _source("app.py")
    overview = app_source.split('if page == "Overview Command Center"', 1)[1].split("# PAGE: GUIDED RESEARCH WORKFLOW", 1)[0]

    assert "Multi-Asset Market Research &amp; Risk Intelligence Platform" in overview
    assert "render_metric_grid" in overview
    assert "render_pipeline_stepper" in overview
    assert "Data freshness" in overview
    assert "Best Paper-Track Candidates" in overview
    assert "Main Risks" in overview
    expected_steps = ["Data", "Features", "Forecasts", "Signals", "Validation", "Risk", "Benchmarking", "Unified Verdict"]
    assert all(f'"{step}"' in overview for step in expected_steps)


def test_phase24_guided_workflow_uses_step_cards_and_glossary():
    app_source = _source("app.py")
    guided = app_source.split('elif page == "Guided Research Workflow"', 1)[1].split("# PAGE: ABOUT", 1)[0]

    assert "render_pipeline_stepper" in guided
    assert "render_metric_grid" in guided
    assert "render_glossary_expander" in guided
    assert "WeakEvidenceWarning" in guided
    assert "NextRecommendedPage" in guided


def test_phase24_executive_evidence_pages_keep_rejections_visible():
    app_source = _source("app.py")
    unified = app_source.split('elif page == "Phase 21: Unified Risk Command Center"', 1)[1].split("# PAGE: PREDICTION EDGE IMPROVEMENT", 1)[0]
    model_edge = app_source.split('elif page == "Phase 22: Prediction Edge Improvement"', 1)[1].split("# PAGE: HISTORICAL MODEL REPLAY", 1)[0]

    assert "Evidence Health" in unified
    assert "Visible Rejections" in unified
    assert "Benchmark Evidence" in model_edge
    assert "Visible Rejected Models" in model_edge
    assert "render_download_buttons" in unified
    assert "render_download_buttons" in model_edge


def test_phase24_ui_layer_does_not_import_research_calculations():
    ui_source = _source("src/ui_components.py")

    assert "from src." not in ui_source
    assert "run_" not in ui_source
    assert "predict(" not in ui_source
    assert "fit(" not in ui_source


def test_phase24_empty_state_copy_is_explicit_and_honest():
    app_source = _source("app.py")

    assert "No saved unified or model-benchmark summary was found" in app_source
    assert "No conservative candidate is available" in app_source
    assert "No saved workflow audit is loaded" in app_source
    assert "No rejected model rows are available" in app_source
