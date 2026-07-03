from __future__ import annotations

import ast
from html import unescape
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.product_navigation import resolve_navigation_label, select_available_page


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"

PUBLIC_PAGES = ["Research Dashboard", "Login / Sign Up", "About / Methodology"]
AUTHENTICATED_TITLES = {
    "Research Dashboard": "Multi-Asset Research Dashboard",
    "Candidate Watchlist": "Candidate Watchlist",
    "Evidence & Validation": "Evidence & Validation",
    "Forecast Explorer": "Forecast Explorer",
    "Asset Plans": "Asset Plans",
    "Cost & Risk Plan": "Cost & Risk Plan",
    "Goals & Saved Plans": "Goals & Saved Plans",
    "Research History & Changes": "Research History & Changes",
    "Portfolio Research Summary": "Portfolio Research Summary",
    "Paper Research Log": "Paper Research Log",
    "Account & Settings": "Account & Settings",
    "About / Methodology": "About / Methodology",
    "Advanced Diagnostics": "Advanced Diagnostics",
}
ASSET_CONTEXT_PAGES = {
    "Forecast Explorer", "Asset Plans", "Cost & Risk Plan", "Advanced Diagnostics"
}
FORBIDDEN = re.compile(
    r"buy now|sell now|approved trade|guaranteed returns|live price|real-time price|current price",
    re.I,
)
BROKEN_OR_INTERNAL = re.compile(
    r"\bOpportunityScore\b|\bMissingEstimate\b|\bNotEnoughEvidence\b|\bCostManageable\b|"
    r"\bHighRisk\b|\bLowRecheck\b|\bRe view\b|\btar get\b|\bre play\b|\bestim ate\b|"
    r"\bOpp ortunity\b|\bMiss ing\b"
)


def _visible_text(app: AppTest) -> str:
    output = []
    for collection in (
        app.markdown, app.caption, app.info, app.warning, app.success, app.error, app.metric
    ):
        output.extend(str(item.value) for item in collection)
    return unescape("\n".join(output))


def _state(app: AppTest, key: str, default=None):
    try:
        return app.session_state[key]
    except KeyError:
        return default


def _authenticated_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=90)
    state = {
        "user_unlocked": True,
        "current_app_user_id": 999999,
        "auth_provider": "local_password",
        "auth_user_id": "navigation-audit-user",
        "current_user_label": "Navigation Audit",
        "current_user_email": "navigation-audit@example.com",
        "current_user_is_local_dev": True,
        "selected_asset": "Gold",
        "selected_horizon": 1,
    }
    for key, value in state.items():
        app.session_state[key] = value
    return app.run(timeout=90)


def _literal_assignment(name: str):
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} was not found as a literal assignment")


def test_page_registry_and_requested_aliases_are_canonical():
    registry = _literal_assignment("PAGE_REGISTRY")
    aliases = _literal_assignment("PRIMARY_PAGE_ALIASES")
    assert set(AUTHENTICATED_TITLES) | {"Login / Sign Up"} == set(registry)
    assert aliases["Market Research Assistant"] == "Research Dashboard"
    assert aliases["Evidence of Edge"] == "Evidence & Validation"
    assert aliases["Forecast Model"] == "Forecast Explorer"
    assert aliases["Forecast"] == "Forecast Explorer"
    assert aliases["Advanced diagnostics"] == "Advanced Diagnostics"
    assert aliases["Diagnostics"] == "Advanced Diagnostics"

    forecast_labels = (
        "Forecast Explorer", "Forecast", "Forecast Model", "Forecast model",
        "Train Model", "Train model", "Train Models", "Train models",
        "Forecast and Train Model", "Forecast & Train Model", "Forecast / Train Model",
        "Forecasting", "Model Forecast", "Model Training", "Train / Forecast",
        "Training & Forecast",
    )
    diagnostic_labels = (
        "Advanced Diagnostics", "Advanced diagnostics", "Diagnostics", "Advanced",
        "Developer Diagnostics", "Snapshot Diagnostics",
    )
    for label in forecast_labels:
        resolved, note = resolve_navigation_label(label, registry, aliases)
        assert resolved == "Forecast Explorer", label
        assert note is None
    for label in diagnostic_labels:
        resolved, note = resolve_navigation_label(label, registry, aliases)
        assert resolved == "Advanced Diagnostics", label
        assert note is None

    for label in ("Something Forecast Training", "Model Training Forecast Page"):
        assert resolve_navigation_label(label, registry, aliases) == (
            "Forecast Explorer", "Unknown navigation label normalized to: Forecast Explorer"
        )
    for label in ("Developer Advanced Tool", "Snapshot Diagnostics Page"):
        assert resolve_navigation_label(label, registry, aliases) == (
            "Advanced Diagnostics", "Unknown navigation label normalized to: Advanced Diagnostics"
        )
    unknown, note = resolve_navigation_label("Completely Unknown Page XYZ", registry, aliases)
    assert note is None
    assert select_available_page(unknown, registry) == "Research Dashboard"


def test_public_pages_open_and_protected_pages_are_not_exposed():
    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run(timeout=90)
    assert not app.exception
    assert app.sidebar.radio[0].options == PUBLIC_PAGES
    expected = {
        "Research Dashboard": "Multi-Asset Quant Research Platform",
        "Login / Sign Up": "Access your research workspace",
        "About / Methodology": "About / Methodology",
    }
    for page, title in expected.items():
        app.sidebar.radio[0].set_value(page).run(timeout=90)
        assert not app.exception
        assert title in _visible_text(app)
        assert _state(app, "primary_product_navigation") == page
    assert not set(AUTHENTICATED_TITLES).difference({"Research Dashboard", "About / Methodology"}) & set(
        app.sidebar.radio[0].options
    )


def test_unknown_public_route_falls_back_without_leaking_protected_content():
    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run(timeout=90)
    text = _visible_text(app)
    assert not app.exception
    assert _state(app, "primary_product_navigation") == "Research Dashboard"
    assert "Multi-Asset Quant Research Platform" in text
    assert "Advanced Diagnostics" not in text


def test_every_authenticated_page_click_renders_title_and_survives_rerun():
    app = _authenticated_app()
    for page, title in AUTHENTICATED_TITLES.items():
        button = next((item for item in app.sidebar.button if item.label == page), None)
        assert button is not None, page
        button.click().run(timeout=90)
        text = _visible_text(app)
        assert not app.exception, page
        assert _state(app, "primary_product_navigation") == page
        assert title in text, page
        assert "Calling st.rerun() within a callback is a no-op" not in text, page
        assert FORBIDDEN.search(text) is None, page
        if page != "Advanced Diagnostics":
            assert BROKEN_OR_INTERNAL.search(text) is None, page
        selector_labels = {item.label for item in app.selectbox}
        assert ("Research Asset" in selector_labels) == (page in ASSET_CONTEXT_PAGES), page
        app.run(timeout=90)
        assert not app.exception, page
        assert _state(app, "primary_product_navigation") == page
        assert title in _visible_text(app), page


def test_forecast_advanced_account_and_history_never_fall_back_to_dashboard():
    app = _authenticated_app()
    for canonical_page in (
        "Forecast Explorer", "Advanced Diagnostics", "Account & Settings",
        "Research History & Changes",
    ):
        next(button for button in app.sidebar.button if button.label == canonical_page).click().run(
            timeout=90
        )
        text = _visible_text(app)
        assert not app.exception
        assert _state(app, "primary_product_navigation") == canonical_page
        assert AUTHENTICATED_TITLES[canonical_page] in text
        assert "Multi-Asset Research Dashboard" not in text


def test_saved_login_default_applies_once_and_manual_forecast_click_wins():
    app = _authenticated_app()
    app.session_state["_pending_product_navigation"] = "Goals & Saved Plans"
    app.run(timeout=90)
    assert _state(app, "primary_product_navigation") == "Goals & Saved Plans"
    next(button for button in app.sidebar.button if button.label == "Forecast Explorer").click().run(
        timeout=90
    )
    assert _state(app, "primary_product_navigation") == "Forecast Explorer"
    app.run(timeout=90)
    assert _state(app, "primary_product_navigation") == "Forecast Explorer"
    assert "Forecast Explorer" in _visible_text(app)


def test_keyword_fallback_note_is_visible_only_in_advanced_diagnostics():
    app = _authenticated_app()
    note = "Unknown navigation label normalized to: Forecast Explorer"
    app.session_state["_navigation_normalization_note"] = note
    app.run(timeout=90)
    assert note not in _visible_text(app)
    next(button for button in app.sidebar.button if button.label == "Advanced Diagnostics").click().run(
        timeout=90
    )
    assert note in _visible_text(app)


def test_navigation_uses_pending_write_before_widget_and_no_post_widget_mutation():
    source = APP_PATH.read_text(encoding="utf-8")
    apply_call = source.index("apply_pending_product_navigation()", source.index("# Sidebar Navigation"))
    widget = source.index('key="primary_product_navigation"', apply_call)
    assert apply_call < widget
    assert 'st.session_state["_pending_product_navigation"]' in source
    tail = source[widget:]
    assert 'st.session_state["primary_product_navigation"] =' not in tail
    assert "st.session_state.primary_product_navigation =" not in tail
