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
    render_metric_grid,
    render_research_disclaimer,
    render_safe_table,
    render_section_header,
    render_status_card,
)


FORBIDDEN_LANGUAGE = re.compile(
    r"\b(Buy|Strong Buy|Invest Now|Guaranteed Profit|Safe Profit|Production Ready Trading)\b",
    flags=re.IGNORECASE,
)


def _source(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_phase23_ui_component_helpers_import_and_are_callable():
    helpers = [
        inject_premium_css,
        render_status_card,
        render_metric_grid,
        render_section_header,
        render_research_disclaimer,
        render_blocked_capital_banner,
        render_safe_table,
        render_download_buttons,
    ]

    assert all(callable(helper) for helper in helpers)


def test_phase23_app_and_ui_components_parse_as_python():
    ast.parse(_source("app.py"))
    ast.parse(_source("src/ui_components.py"))


def test_phase23_forbidden_phrases_are_not_introduced():
    for path in ["app.py", "README.md", "src/ui_components.py"]:
        assert FORBIDDEN_LANGUAGE.search(_source(path)) is None, path


def test_phase23_removes_legacy_academic_label_from_app():
    assert "B.Tech Final Year Project" not in _source("app.py")


def test_phase23_real_capital_block_remains_prominent():
    app_source = _source("app.py")
    component_source = _source("src/ui_components.py")

    assert "render_blocked_capital_banner" in app_source
    assert "Real capital status: Blocked" in component_source


def test_phase23_navigation_groups_and_overview_page_exist():
    app_source = _source("app.py")
    expected = {
        "Overview Command Center",
        "Forecasting & Prediction",
        "Signal Research",
        "Validation & Evidence",
        "Risk Intelligence",
        "Benchmarking & Replay",
        "Reports & Exports",
    }

    assert expected.issubset({label for label in expected if label in app_source})
    assert 'if page == "Overview Command Center"' in app_source
    assert "Multi-Asset Market Research &amp; Risk Intelligence Platform" in app_source


def test_phase23_pipeline_map_is_present_and_complete():
    app_source = _source("app.py")
    for step in ["Data", "Features", "Forecasts", "Signals", "Validation", "Risk", "Benchmarking", "True ML Replay", "Unified Verdict"]:
        assert step in app_source


def test_phase23_new_navigation_block_has_no_mojibake():
    app_source = _source("app.py")
    navigation = app_source.split("NAVIGATION_GROUPS =", 1)[1].split("asset_names =", 1)[0]

    assert not re.search(r"Ã|â€|ðŸ|ï¸", navigation)


def test_phase23_phase21_and_phase22_use_shared_ui_helpers():
    app_source = _source("app.py")
    phase21 = app_source.split('elif page == "Phase 21: Unified Risk Command Center"', 1)[1].split("# PAGE: PREDICTION EDGE IMPROVEMENT", 1)[0]
    phase22 = app_source.split('elif page == "Phase 22: Prediction Edge Improvement"', 1)[1].split("# PAGE: HISTORICAL MODEL REPLAY", 1)[0]

    for section in [phase21, phase22]:
        assert "render_metric_grid" in section
        assert "render_blocked_capital_banner" in section
        assert "render_safe_table" in section
        assert "render_download_buttons" in section


def test_phase23_ui_module_does_not_import_research_calculations():
    ui_source = _source("src/ui_components.py")

    assert "from src." not in ui_source
    assert "run_" not in ui_source
