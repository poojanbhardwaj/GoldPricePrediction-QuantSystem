from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from streamlit.testing.v1 import AppTest

from src import model_training_workflow


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
SECTION_NAMES = [
    "Data Health",
    "Forecasting & Models",
    "Research Records",
    "User Workspace",
    "System",
]
PANEL_NAMES = [
    "Snapshot Overview",
    "Market Data Refresh",
    "Source Freshness",
    "Artifact Inventory",
    "Forecast Diagnostics",
    "Compare Models",
    "30 Days Forecast",
    "Model Training Diagnostics",
    "Train Models",
    "Candidate Records",
    "Evidence Records",
    "Asset/Horizon Lookup",
    "Saved Plans",
    "Research History",
    "Account Storage",
    "Navigation State",
    "Runtime Health",
    "Safety Checks",
]


def _text(app: AppTest) -> str:
    values = []
    for collection in (
        app.markdown,
        app.caption,
        app.info,
        app.warning,
        app.error,
        app.success,
    ):
        values.extend(str(item.value) for item in collection)
    values.extend(str(item.label) for item in app.expander)
    return "\n".join(values)


def _authenticated_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=90)
    for key, value in {
        "user_unlocked": True,
        "current_app_user_id": 999999,
        "auth_provider": "local_password",
        "auth_user_id": "advanced-routing-user",
        "current_user_label": "Advanced Routing",
        "current_user_email": "advanced-routing@example.com",
        "current_user_is_local_dev": True,
        "selected_asset": "Gold",
        "selected_horizon": 5,
    }.items():
        app.session_state[key] = value
    return app.run(timeout=90)


def _open_advanced(app: AppTest) -> AppTest:
    next(
        button for button in app.sidebar.button
        if button.label == "Advanced Diagnostics"
    ).click().run(timeout=90)
    return app


def test_advanced_diagnostics_renders_all_sections_and_panels_without_redirect():
    app = _open_advanced(_authenticated_app())
    visible = _text(app)

    assert not app.exception
    assert app.session_state["active_product_page"] == "Advanced Diagnostics"
    assert app.session_state["primary_product_navigation"] == "Advanced Diagnostics"
    assert "Advanced Diagnostics" in visible
    for section_name in SECTION_NAMES:
        assert section_name in visible
    for panel_name in PANEL_NAMES:
        assert panel_name in visible
    assert "Multi-Asset Research Dashboard" not in visible


def test_advanced_diagnostics_has_no_internal_route_selectboxes():
    app = _open_advanced(_authenticated_app())
    labels = [control.label for control in app.selectbox]

    assert "Diagnostic area" not in labels
    assert "Diagnostic page" not in labels
    assert "Research Asset" in labels
    assert "Research Horizon" in labels
    assert "advanced_diagnostic_area" not in app.session_state
    assert "advanced_diagnostic_page" not in app.session_state


def test_logged_out_advanced_diagnostics_shows_read_only_preview():
    app = AppTest.from_file(str(APP_PATH), default_timeout=90)
    app.session_state["primary_product_navigation"] = "Advanced Diagnostics"
    app.run(timeout=90)
    visible = _text(app)

    assert not app.exception
    assert app.session_state["active_product_page"] == "Advanced Diagnostics"
    assert "Data Health" in visible
    assert "Forecasting & Models" in visible
    assert "Research Records" in visible
    assert "System" in visible
    assert "Login required for full diagnostics and training controls." in visible
    assert "Start Training" not in [button.label for button in app.button]
    assert "Research Asset" not in [control.label for control in app.selectbox]


def test_start_training_stays_inside_advanced_diagnostics(monkeypatch):
    def fake_training(**kwargs):
        return model_training_workflow.TrainingWorkflowResult(
            asset=kwargs["asset"],
            horizon=kwargs["horizon"],
            target_col=kwargs["target_col"],
            model_families=("ML",),
            model_count=1,
            leaderboard=pd.DataFrame([
                {"ModelFamily": "ML", "Model": "Mock", "RMSE": 1.0}
            ]),
            ml_trainer=SimpleNamespace(results={"Mock": object()}),
            preprocessor=SimpleNamespace(),
            data=SimpleNamespace(),
            feature_frame=pd.DataFrame({"feature": [1.0]}),
        )

    monkeypatch.setattr(model_training_workflow, "run_training_pipeline", fake_training)
    app = _open_advanced(_authenticated_app())
    next(button for button in app.button if button.label == "Start Training").click().run(
        timeout=90
    )

    assert not app.exception
    assert app.session_state["active_product_page"] == "Advanced Diagnostics"
    assert app.session_state["primary_product_navigation"] == "Advanced Diagnostics"
    assert app.session_state["selected_asset"] == "Gold"
    assert app.session_state["selected_horizon"] == 5
    assert "Training complete for Gold" in _text(app)
    assert "Multi-Asset Research Dashboard" not in _text(app)


def test_advanced_diagnostics_renderer_is_isolated_from_product_navigation():
    source = APP_PATH.read_text(encoding="utf-8")
    renderer_start = source.index("def render_advanced_diagnostics(")
    renderer_end = source.index("\ndef _render_public_market_teaser", renderer_start)
    renderer = source[renderer_start:renderer_end]

    assert "request_product_navigation" not in renderer
    assert "normalize_page_name" not in renderer
    assert "st.switch_page" not in renderer
    assert "pending_product_navigation" not in renderer
    assert 'active_product_page"] =' not in renderer
    assert 'primary_product_navigation"] =' not in renderer
    assert "Diagnostic area" not in renderer
    assert "Diagnostic page" not in renderer

