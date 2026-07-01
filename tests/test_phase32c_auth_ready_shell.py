from __future__ import annotations

from pathlib import Path
import re
import sys

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import auth_manager
from src.auth_manager import AuthUser, get_public_user
from src.user_platform import (
    apply_profile_defaults,
    get_or_create_demo_user,
    get_or_create_user_for_auth,
    load_latest_user_plan,
    load_user_profile,
    save_user_plan_run,
    save_user_profile,
)


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _StreamlitStub:
    def __init__(self):
        self.session_state = _SessionState()


def _text(app: AppTest) -> str:
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


def test_public_user_object_is_locked():
    user = get_public_user()

    assert isinstance(user, AuthUser)
    assert user.app_user_id is None
    assert user.auth_provider == "public"
    assert user.is_authenticated is False
    assert user.is_demo is False


def test_demo_unlock_sets_expected_session_keys(monkeypatch, tmp_path):
    stub = _StreamlitStub()
    monkeypatch.setattr(auth_manager, "st", stub)

    assert auth_manager.is_user_unlocked() is False
    user = auth_manager.unlock_demo_user(tmp_path / "app.db")

    assert user.is_authenticated is True
    assert user.is_demo is True
    assert user.app_user_id is not None
    assert stub.session_state["user_unlocked"] is True
    assert stub.session_state["demo_user_id"] == user.app_user_id
    assert stub.session_state["auth_provider"] == "demo"
    assert stub.session_state["auth_user_id"] == "demo-user"
    assert stub.session_state["current_app_user_id"] == user.app_user_id
    assert stub.session_state["primary_product_navigation"] == "User Goals & Saved Plans"


def test_logout_returns_to_public_session(monkeypatch, tmp_path):
    stub = _StreamlitStub()
    monkeypatch.setattr(auth_manager, "st", stub)
    auth_manager.unlock_demo_user(tmp_path / "app.db")

    auth_manager.logout_current_user()

    assert stub.session_state["user_unlocked"] is False
    assert stub.session_state["demo_user_id"] is None
    assert "current_app_user_id" not in stub.session_state
    assert stub.session_state["primary_product_navigation"] == "Market Research Assistant"
    assert auth_manager.get_current_auth_user(tmp_path / "app.db").is_authenticated is False


def test_legacy_unlocked_session_resolves_demo_user(monkeypatch, tmp_path):
    stub = _StreamlitStub()
    stub.session_state["user_unlocked"] = True
    monkeypatch.setattr(auth_manager, "st", stub)

    user = auth_manager.get_current_auth_user(tmp_path / "app.db")

    assert user.is_authenticated is True
    assert user.is_demo is True
    assert stub.session_state["current_app_user_id"] == user.app_user_id


def test_get_or_create_user_for_auth_is_stable_and_demo_compatible(tmp_path):
    db_path = tmp_path / "app.db"
    first = get_or_create_user_for_auth(
        "demo",
        "demo-user",
        "Demo User",
        email="demo@local.app",
        is_demo=True,
        db_path=db_path,
    )
    second = get_or_create_user_for_auth(
        "demo",
        "demo-user",
        "Demo User",
        email="demo@local.app",
        is_demo=True,
        db_path=db_path,
    )
    legacy = get_or_create_demo_user(db_path)

    assert first["id"] == second["id"] == legacy["id"]
    assert first["auth_provider"] == "demo"
    assert first["auth_user_id"] == "demo-user"
    assert first["is_demo"] is True


def test_user_owned_profile_and_plan_require_app_user_id(tmp_path):
    db_path = tmp_path / "app.db"
    profile = apply_profile_defaults({"name": "Demo User"})

    with pytest.raises(ValueError):
        save_user_profile(None, profile, db_path=db_path)
    with pytest.raises(ValueError):
        save_user_plan_run(None, profile, [], db_path=db_path)

    assert load_user_profile(None, db_path=db_path) is None
    assert load_latest_user_plan(None, db_path=db_path).empty


def test_current_demo_user_owns_profile_records(tmp_path):
    db_path = tmp_path / "app.db"
    user = get_or_create_user_for_auth(
        "demo",
        "demo-user",
        "Demo User",
        email="demo@local.app",
        is_demo=True,
        db_path=db_path,
    )
    profile = apply_profile_defaults({"name": "Demo User", "goal_type": "Learn markets"})

    save_user_profile(user["id"], profile, db_path=db_path)
    loaded = load_user_profile(user["id"], db_path=db_path)

    assert loaded is not None
    assert loaded["name"] == "Demo User"
    assert loaded["goal_type"] == "Learning"


def test_topbar_appears_in_public_shell_and_sidebar_remains_navigation_only():
    app = _public_app()
    text = _text(app)

    assert not app.exception
    assert "Quant Research Lab" in text
    assert "Public preview" in text
    assert "Continue as Demo User" in [button.label for button in app.button]
    assert app.sidebar.radio[0].options == [
        "Market Research Assistant",
        "Login / Unlock Demo",
        "About / Methodology",
    ]


def test_unlock_topbar_restores_full_workspace_navigation():
    app = _public_app()
    next(button for button in app.button if button.label == "Continue as Demo User").click().run(
        timeout=90
    )
    text = _text(app)

    assert not app.exception
    assert app.session_state["user_unlocked"] is True
    assert app.session_state["auth_provider"] == "demo"
    assert app.sidebar.radio[0].value == "User Goals & Saved Plans"
    assert "Demo user" in text
    assert "Logout" in [button.label for button in app.button]
    for page in (
        "Candidate Watchlist",
        "Evidence of Edge",
        "User Goals & Saved Plans",
        "Asset Plans",
        "Forecast Explorer",
        "Advanced Diagnostics",
    ):
        assert page in app.sidebar.radio[0].options


def test_login_unlock_page_is_demo_only_and_auth_ready():
    app = _public_app()
    app.sidebar.radio[0].set_value("Login / Unlock Demo").run(timeout=90)
    text = _text(app)

    assert not app.exception
    assert "Access your research workspace" in text
    assert "Unlock personalized research plans" in text
    assert "Continue as Demo User" in [button.label for button in app.button]
    assert "Email login" in text
    assert "coming later" in text.casefold()
    assert not app.text_input
    assert "password" not in text.casefold()
    assert "Do not enter broker, bank, trading-account credentials, or API secrets" in text


def test_locked_gated_page_source_has_unlock_guard():
    source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "GATED_PRODUCT_PAGES" in source
    assert "Unlock demo mode to access this research page." in source
    guard = source.split("if not _is_user_unlocked() and (page in GATED_PRODUCT_PAGES", 1)[1]
    guarded_block = guard.split('if page == "Market Research Assistant"', 1)[0]
    assert "_render_unlock_prompt()" in guarded_block
    assert "st.stop()" in guarded_block


def test_product_shell_avoids_forbidden_trading_language():
    app = _public_app()
    text = _text(app)
    prohibited = re.compile(r"buy now|sell now|approved trade", flags=re.IGNORECASE)

    assert prohibited.search(text) is None
    assert "No broker credentials" in text
    assert "No real-money execution" in text
    assert "No return promises" in text
