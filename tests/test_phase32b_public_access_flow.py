from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ui_components import (
    render_clean_footer,
    render_feature_grid,
    render_final_cta,
    render_how_it_works,
    render_methodology_preview,
    render_metric_strip,
    render_professional_hero,
    render_trust_and_safety,
)


PUBLIC_PAGES = [
    "Market Research Assistant",
    "Login / Unlock Demo",
    "About / Methodology",
]


def _all_visible_text(app: AppTest) -> str:
    values = []
    for collection in (
        app.markdown,
        app.caption,
        app.info,
        app.warning,
        app.success,
        app.metric,
    ):
        values.extend(str(item.value) for item in collection)
    return "\n".join(values)


def _public_app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run(timeout=90)


def test_public_navigation_is_narrow_and_preview_is_professional():
    app = _public_app()
    text = _all_visible_text(app)

    assert not app.exception
    assert app.sidebar.radio[0].options == PUBLIC_PAGES
    assert "Multi-Asset Quant Research Platform" in text
    assert "research-only" in text.casefold()
    assert "Public Market Snapshot" in text
    assert "Unlock forecasts" in text or "unlock after demo" in text
    for missing_placeholder in ("Run research", "No saved estimate", "0/100"):
        assert missing_placeholder not in text


def test_public_preview_has_features_and_trust_safety_language():
    app = _public_app()
    text = _all_visible_text(app)

    for feature in (
        "Candidate Watchlist",
        "Evidence of Edge",
        "Personalized Research Plans",
        "Risk & Cost Awareness",
    ):
        assert feature in text
    for safety_text in (
        "No broker credentials",
        "No real-money execution",
        "No return promises",
        "Stale, cached, saved, and refreshed snapshots stay visibly labeled.",
    ):
        assert safety_text in text


def test_public_price_preview_has_date_and_honest_source_without_live_claim():
    app = _public_app()
    text = _all_visible_text(app)
    source_labels = {
        "Cached dataset price",
        "Saved research snapshot",
        "Latest refreshed snapshot",
    }

    assert any(label in text for label in source_labels)
    assert re.search(r"20\d{2}-\d{2}-\d{2}", text)
    assert "live price" not in text.casefold()
    assert "live quote" not in text.casefold()
    assert any("Stale" in str(caption.value) for caption in app.caption)


def test_public_preview_has_unlock_and_methodology_actions_only():
    app = _public_app()
    labels = [button.label for button in app.button]

    assert "Continue as Demo User" in labels
    assert "View methodology" in labels
    assert "Refresh / Rebuild Research" not in labels
    assert "Refresh Market Data" not in labels
    assert "View Cost-Aware Plan" not in labels


def test_login_page_is_demo_only_and_collects_no_password():
    app = _public_app()
    app.sidebar.radio[0].set_value("Login / Unlock Demo").run(timeout=90)
    text = _all_visible_text(app)

    assert not app.exception
    assert "Unlock personalized research plans" in text
    assert "Continue as Demo User" in [button.label for button in app.button]
    assert (
        "Do not enter broker, bank, trading-account credentials, or API secrets. "
        "This app is research-only."
    ) in text
    assert not app.text_input
    assert "password" not in text.casefold()


def test_unlock_restores_full_navigation_and_opens_user_goals():
    app = _public_app()
    next(button for button in app.button if button.label == "Continue as Demo User").click().run(
        timeout=90
    )

    assert not app.exception
    assert app.session_state["user_unlocked"] is True
    assert app.sidebar.radio[0].value == "User Goals & Saved Plans"
    options = app.sidebar.radio[0].options
    for page in (
        "Candidate Watchlist",
        "Evidence of Edge",
        "User Goals & Saved Plans",
        "Asset Plans",
        "Forecast Explorer",
    ):
        assert page in options
    assert "Advanced Diagnostics" in options
    assert len(app.get("form")) == 1
    assert "Generate Personalized Plan" in [button.label for button in app.button]


def test_logout_returns_to_public_mode():
    app = _public_app()
    next(button for button in app.button if button.label == "Continue as Demo User").click().run(
        timeout=90
    )
    next(button for button in app.button if button.label == "Logout").click().run(timeout=90)

    assert not app.exception
    assert app.session_state["user_unlocked"] is False
    assert app.session_state["demo_user_id"] is None
    assert app.sidebar.radio[0].value == "Market Research Assistant"
    assert app.sidebar.radio[0].options == PUBLIC_PAGES


def test_gated_pages_have_a_locked_route_guard():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(source)

    assert "GATED_PRODUCT_PAGES" in source
    assert 'st.info("Unlock demo mode to access this research page.")' in source
    guard = source.split("if not _is_user_unlocked() and (page in GATED_PRODUCT_PAGES", 1)[1]
    assert "_render_unlock_prompt()" in guard.split('if page == "Market Research Assistant"', 1)[0]
    assert "st.stop()" in guard.split('if page == "Market Research Assistant"', 1)[0]


def test_public_landing_avoids_prohibited_promotional_language():
    app = _public_app()
    text = _all_visible_text(app)
    prohibited = re.compile(
        r"buy now|sell now|guaranteed|approved trade",
        flags=re.IGNORECASE,
    )

    assert prohibited.search(text) is None


def test_public_ui_helpers_are_callable():
    helpers = (
        render_professional_hero,
        render_metric_strip,
        render_feature_grid,
        render_how_it_works,
        render_trust_and_safety,
        render_methodology_preview,
        render_final_cta,
        render_clean_footer,
    )
    assert all(callable(helper) for helper in helpers)
