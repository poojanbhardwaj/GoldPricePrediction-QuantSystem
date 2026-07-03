from __future__ import annotations

from html import unescape
from pathlib import Path
import re

from streamlit.testing.v1 import AppTest


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
ASSET_PAGES = {"Forecast Explorer", "Asset Plans", "Cost & Risk Plan", "Advanced Diagnostics"}
FORBIDDEN = re.compile(
    r"buy now|sell now|approved trade|guaranteed returns|live price|real-time price|current live price",
    re.I,
)


def _text(app: AppTest) -> str:
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
    for key, value in {
        "user_unlocked": True,
        "current_app_user_id": 999999,
        "auth_provider": "local_password",
        "auth_user_id": "real-user-audit",
        "current_user_label": "Real User Audit",
        "current_user_email": "real-user-audit@example.com",
        "current_user_is_local_dev": True,
        "selected_asset": "Gold",
        "selected_horizon": 1,
    }.items():
        app.session_state[key] = value
    return app.run(timeout=90)


def _click_sidebar(app: AppTest, label: str) -> AppTest:
    button = next((item for item in app.sidebar.button if item.label == label), None)
    assert button is not None, f"Missing sidebar button: {label}"
    return button.click().run(timeout=90)


def test_public_real_sidebar_pages_and_gating():
    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run(timeout=90)
    assert app.sidebar.radio[0].options == PUBLIC_PAGES
    expected = {
        "Research Dashboard": "Multi-Asset Quant Research Platform",
        "Login / Sign Up": "Access your research workspace",
        "About / Methodology": "About / Methodology",
    }
    for page, title in expected.items():
        app.sidebar.radio[0].set_value(page).run(timeout=90)
        text = _text(app)
        assert not app.exception
        assert title in text
        assert FORBIDDEN.search(text) is None
    protected = set(AUTHENTICATED_TITLES) - {"Research Dashboard", "About / Methodology"}
    assert protected.isdisjoint(app.sidebar.radio[0].options)


def test_every_authenticated_real_sidebar_page_opens_with_scoped_asset_selector():
    app = _authenticated_app()
    visible_labels = [button.label for button in app.sidebar.button]
    assert visible_labels == list(AUTHENTICATED_TITLES)
    for page, title in AUTHENTICATED_TITLES.items():
        _click_sidebar(app, page)
        text = _text(app)
        assert not app.exception, page
        assert _state(app, "primary_product_navigation") == page
        assert title in text, page
        assert FORBIDDEN.search(text) is None, page
        labels = {control.label for control in app.selectbox}
        assert ("Research Asset" in labels) == (page in ASSET_PAGES), page
        assert "Traceback" not in text


def test_dashboard_ctas_use_real_navigation_callbacks():
    app = _authenticated_app()
    next(button for button in app.button if button.label == "View Cost & Risk Plan").click().run(
        timeout=90
    )
    assert not app.exception
    assert _state(app, "primary_product_navigation") == "Cost & Risk Plan"
    assert "Cost & Risk Plan" in _text(app)

    _click_sidebar(app, "Research Dashboard")
    next(button for button in app.button if button.label == "Open Paper Research Log").click().run(
        timeout=90
    )
    assert not app.exception
    assert _state(app, "primary_product_navigation") == "Paper Research Log"
    assert "Paper Research Log" in _text(app)


def test_public_and_snapshot_card_ctas_use_real_navigation_callbacks():
    public = AppTest.from_file(str(APP_PATH), default_timeout=90).run(timeout=90)
    next(button for button in public.button if button.label == "Sign in / Create account").click().run(
        timeout=90
    )
    assert not public.exception
    assert "Access your research workspace" in _text(public)

    public = AppTest.from_file(str(APP_PATH), default_timeout=90).run(timeout=90)
    next(button for button in public.button if button.label == "View methodology").click().run(
        timeout=90
    )
    assert not public.exception
    assert "About / Methodology" in _text(public)

    app = _authenticated_app()
    next(button for button in app.button if button.label == "View plan").click().run(timeout=90)
    assert not app.exception
    assert _state(app, "primary_product_navigation") == "Asset Plans"
    assert "Asset Plans" in _text(app)


def test_forecast_asset_selector_supports_every_configured_asset():
    app = _authenticated_app()
    _click_sidebar(app, "Forecast Explorer")
    selector = next(control for control in app.selectbox if control.label == "Research Asset")
    expected_assets = ["Gold", "Silver", "Crude Oil", "Bitcoin", "S&P 500", "Gold ETF"]
    assert selector.options == expected_assets
    for asset in expected_assets:
        selector = next(control for control in app.selectbox if control.label == "Research Asset")
        selector.set_value(asset).run(timeout=90)
        assert not app.exception, asset
        assert "Forecast Explorer" in _text(app)
        assert _state(app, "selected_asset") == asset


def test_public_and_account_controls_never_request_financial_credentials():
    public = AppTest.from_file(str(APP_PATH), default_timeout=90).run(timeout=90)
    public.sidebar.radio[0].set_value("Login / Sign Up").run(timeout=90)
    login_labels = " ".join(control.label for control in public.text_input).casefold()
    assert "email" in login_labels
    assert "password" in login_labels
    for forbidden in ("broker", "bank", "trading account", "api key", "secret"):
        assert forbidden not in login_labels

    app = _authenticated_app()
    _click_sidebar(app, "Account & Settings")
    account_labels = " ".join(
        control.label for collection in (app.text_input, app.selectbox, app.multiselect)
        for control in collection
    ).casefold()
    for forbidden in ("broker", "bank", "trading password", "api key", "secret"):
        assert forbidden not in account_labels
    assert "Local development auth" in _text(app)


def test_topbar_logout_returns_to_public_navigation():
    app = _authenticated_app()
    logout = next((button for button in app.button if button.label == "Logout"), None)
    assert logout is not None
    logout.click().run(timeout=90)
    assert not app.exception
    assert _state(app, "user_unlocked") is False
    assert app.sidebar.radio[0].options == PUBLIC_PAGES
    assert "Public preview" in _text(app)
