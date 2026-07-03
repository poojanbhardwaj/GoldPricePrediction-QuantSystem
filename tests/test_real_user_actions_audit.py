from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import auth_manager
from src import model_training_workflow
from src.research_history import load_research_history_runs
from src.user_platform import (
    create_password_user,
    load_user_preferences,
    save_user_preferences,
)


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"


class _Session(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _StreamlitStub:
    def __init__(self):
        self.session_state = _Session()


def _login_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run(timeout=90)
    app.sidebar.radio[0].set_value("Login / Sign Up").run(timeout=90)
    return app


def _text(app: AppTest) -> str:
    output = []
    for collection in (app.markdown, app.caption, app.info, app.warning, app.error, app.success):
        output.extend(str(item.value) for item in collection)
    return "\n".join(output)


def _authenticated_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=90)
    for key, value in {
        "user_unlocked": True,
        "current_app_user_id": 999999,
        "auth_provider": "local_password",
        "auth_user_id": "training-audit-user",
        "current_user_label": "Training Audit",
        "current_user_email": "training-audit@example.com",
        "current_user_is_local_dev": True,
        "selected_asset": "Gold",
        "selected_horizon": 1,
    }.items():
        app.session_state[key] = value
    return app.run(timeout=90)


def _open_train_models(app: AppTest) -> AppTest:
    next(button for button in app.sidebar.button if button.label == "Advanced Diagnostics").click().run(
        timeout=90
    )
    next(control for control in app.selectbox if control.label == "Diagnostic area").set_value(
        "Forecasting & Models"
    ).run(timeout=90)
    next(control for control in app.selectbox if control.label == "Diagnostic page").set_value(
        "Train Models"
    ).run(timeout=90)
    return app


def _training_result(
    asset: str, horizon: int, model_families: tuple[str, ...] = ("ML",)
) -> model_training_workflow.TrainingWorkflowResult:
    return model_training_workflow.TrainingWorkflowResult(
        asset=asset,
        horizon=horizon,
        target_col="Silver_Close" if asset == "Silver" else "Gold_Close",
        model_families=model_families,
        model_count=1,
        leaderboard=pd.DataFrame([{"ModelFamily": "ML", "Model": "Mock Model", "RMSE": 1.0}]),
        ml_trainer=SimpleNamespace(results={"Mock Model": object()}),
        preprocessor=SimpleNamespace(),
        data=SimpleNamespace(),
        feature_frame=pd.DataFrame({"feature": [1.0]}),
    )


def test_login_page_rejects_invalid_email_without_external_auth_or_database_write():
    app = _login_app()
    next(item for item in app.text_input if item.key == "signin_email").set_value("invalid-email")
    next(item for item in app.text_input if item.key == "signin_password").set_value("StrongPass123")
    next(button for button in app.button if button.label == "Sign in").click().run(timeout=90)
    assert not app.exception
    assert "valid email" in _text(app).casefold()
    assert app.session_state["user_unlocked"] is False


def test_create_account_rejects_weak_password_before_account_creation():
    app = _login_app()
    next(item for item in app.text_input if item.key == "signup_name").set_value("Audit User")
    next(item for item in app.text_input if item.key == "signup_email").set_value("audit@example.com")
    next(item for item in app.text_input if item.key == "signup_password").set_value("weak")
    next(item for item in app.text_input if item.key == "signup_confirm_password").set_value("weak")
    next(item for item in app.checkbox if item.key == "signup_research_only_ack").check()
    next(button for button in app.button if button.label == "Create account").click().run(timeout=90)
    assert not app.exception
    assert "at least 8 characters" in _text(app)
    assert app.session_state["user_unlocked"] is False


def test_login_page_labels_local_auth_and_rejects_financial_credentials():
    app = _login_app()
    text = _text(app)
    assert "Local development auth is active" in text or "Verified email authentication is enabled" in text
    labels = " ".join(item.label for item in app.text_input).casefold()
    for forbidden in ("broker", "bank", "trading account", "api key", "secret"):
        assert forbidden not in labels


def test_account_preferences_round_trip_is_user_owned(tmp_path):
    db_path = tmp_path / "audit-users.db"
    first = create_password_user("first@example.com", "StrongPass123", "First", db_path=db_path)
    second = create_password_user("second@example.com", "StrongPass123", "Second", db_path=db_path)
    preferences = {
        "default_assets": ["Gold", "Bitcoin"],
        "default_horizon": "5D",
        "explanation_mode": "Detailed",
        "default_page": "Forecast Explorer",
        "risk_display_mode": "Summary",
    }
    save_user_preferences(first["id"], preferences, db_path=db_path)
    assert load_user_preferences(first["id"], db_path=db_path)["default_page"] == "Forecast Explorer"
    assert load_user_preferences(first["id"], db_path=db_path)["default_assets"] == ["Gold", "Bitcoin"]
    assert load_user_preferences(second["id"], db_path=db_path) is None


def test_empty_research_history_is_professional_and_user_scoped(tmp_path):
    db_path = tmp_path / "audit-history.db"
    user = create_password_user("history@example.com", "StrongPass123", "History", db_path=db_path)
    history = load_research_history_runs(user["id"], db_path=db_path)
    assert history.empty
    assert history.columns.tolist() == ["RunId", "RunLabel", "CreatedAt", "AssetCount", "Source"]


def test_logout_clears_auth_and_queues_safe_public_page(monkeypatch):
    stub = _StreamlitStub()
    stub.session_state.update({
        "user_unlocked": True,
        "current_app_user_id": 42,
        "auth_provider": "local_password",
        "demo_user_id": None,
    })
    monkeypatch.setattr(auth_manager, "st", stub)
    monkeypatch.setattr(auth_manager, "is_supabase_configured", lambda: False)
    auth_manager.logout_current_user()
    assert stub.session_state["user_unlocked"] is False
    assert "current_app_user_id" not in stub.session_state
    assert stub.session_state["_pending_product_navigation"] == "Market Research Assistant"


def test_train_models_page_opens_without_dashboard_fallback():
    app = _open_train_models(_authenticated_app())
    text = _text(app)
    assert not app.exception
    assert "Train Models" in text
    assert "Multi-Asset Research Dashboard" not in text
    assert app.session_state["primary_product_navigation"] == "Advanced Diagnostics"
    assert app.session_state["advanced_diagnostic_area"] == "Forecasting & Models"
    assert app.session_state["advanced_diagnostic_page"] == "Train Models"


def test_start_training_calls_workflow_and_preserves_page_and_choices(monkeypatch):
    calls = []

    def fake_training(**kwargs):
        calls.append(kwargs)
        kwargs["progress_callback"](60, "Mock training is running...")
        families = tuple(
            name for name, enabled in (("ML", kwargs["train_ml"]), ("DL", kwargs["train_dl"]))
            if enabled
        )
        return _training_result(kwargs["asset"], kwargs["horizon"], families)

    monkeypatch.setattr(model_training_workflow, "run_training_pipeline", fake_training)
    app = _open_train_models(_authenticated_app())
    next(control for control in app.selectbox if control.label == "Research Asset").set_value(
        "Silver"
    ).run(timeout=90)
    next(control for control in app.selectbox if control.label == "Research Horizon").set_value(
        5
    ).run(timeout=90)
    next(control for control in app.checkbox if control.label.startswith("Train ML Models")).uncheck().run(
        timeout=90
    )
    next(control for control in app.checkbox if control.label.startswith("Train DL Models")).check().run(
        timeout=90
    )
    next(control for control in app.slider if control.label == "DL Epochs (demo)").set_value(7).run(
        timeout=90
    )
    next(button for button in app.button if button.label == "Start Training").click().run(timeout=90)

    text = _text(app)
    assert not app.exception
    assert len(calls) == 1
    assert calls[0]["asset"] == "Silver"
    assert calls[0]["horizon"] == 5
    assert calls[0]["train_ml"] is False
    assert calls[0]["train_dl"] is True
    assert calls[0]["dl_epochs"] == 7
    assert app.session_state["primary_product_navigation"] == "Advanced Diagnostics"
    assert app.session_state["active_product_page"] == "Advanced Diagnostics"
    assert app.session_state["advanced_diagnostic_area"] == "Forecasting & Models"
    assert app.session_state["advanced_diagnostic_page"] == "Train Models"
    assert app.session_state["selected_asset"] == "Silver"
    assert app.session_state["selected_horizon"] == 5
    assert app.session_state["advanced_training_ml_models"] is False
    assert app.session_state["advanced_training_dl_models"] is True
    assert app.session_state["advanced_training_dl_epochs"] == 7
    assert app.session_state["training_in_progress"] is False
    assert app.session_state["last_training_result"]["Status"] == "Complete"
    assert "Training complete for Silver" in text
    assert "Multi-Asset Research Dashboard" not in text
    app.run(timeout=90)
    assert app.session_state["primary_product_navigation"] == "Advanced Diagnostics"
    assert app.session_state["advanced_diagnostic_page"] == "Train Models"
    assert "Training complete for Silver" in _text(app)


def test_training_failure_stays_on_train_models_and_keeps_error_details(monkeypatch):
    def fail_training(**kwargs):
        raise RuntimeError("mock dependency unavailable")

    monkeypatch.setattr(model_training_workflow, "run_training_pipeline", fail_training)
    app = _open_train_models(_authenticated_app())
    next(button for button in app.button if button.label == "Start Training").click().run(timeout=90)

    text = _text(app)
    assert not app.exception
    assert app.session_state["primary_product_navigation"] == "Advanced Diagnostics"
    assert app.session_state["advanced_diagnostic_area"] == "Forecasting & Models"
    assert app.session_state["advanced_diagnostic_page"] == "Train Models"
    assert app.session_state["training_in_progress"] is False
    assert app.session_state["last_training_error"] == "mock dependency unavailable"
    assert "Training failed for the selected configuration" in text
    assert any(expander.label == "Training error details" for expander in app.expander)
    assert "Multi-Asset Research Dashboard" not in text


def test_missing_training_dependency_is_reported_without_navigation(monkeypatch):
    def fail_training(**kwargs):
        raise ModuleNotFoundError("No module named 'tensorflow'")

    monkeypatch.setattr(model_training_workflow, "run_training_pipeline", fail_training)
    app = _open_train_models(_authenticated_app())
    next(button for button in app.button if button.label == "Start Training").click().run(timeout=90)

    assert not app.exception
    assert app.session_state["primary_product_navigation"] == "Advanced Diagnostics"
    assert app.session_state["advanced_diagnostic_page"] == "Train Models"
    assert "Training could not start because required input is unavailable." in _text(app)
    assert "tensorflow" in app.session_state["last_training_error"]


def test_training_block_has_no_dashboard_navigation_mutation():
    source = APP_PATH.read_text(encoding="utf-8")
    start = source.index('elif page == "🤖 Train Models"')
    end = source.index("# PAGE: COMPARE MODELS", start)
    training_block = source[start:end]
    assert 'request_product_navigation("Research Dashboard")' not in training_block
    assert 'active_product_page"] = "Research Dashboard"' not in training_block
    assert 'primary_product_navigation"] = "Research Dashboard"' not in training_block
    assert "model_training_workflow.run_training_pipeline" in training_block
