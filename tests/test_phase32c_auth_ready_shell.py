from __future__ import annotations

from pathlib import Path
import re
import sqlite3
import sys

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import auth_manager
from src.auth_manager import AuthUser, get_public_user
from src.user_platform import (
    apply_profile_defaults,
    authenticate_password_user,
    create_password_user,
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
        app.error,
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


def test_local_account_creation_stores_hash_not_plaintext(tmp_path):
    db_path = tmp_path / "app.db"
    user = create_password_user(
        "poojan@example.com", "StrongPass123", "Poojan", db_path=db_path
    )

    assert user["email"] == "poojan@example.com"
    assert user["auth_provider"] == "local_password"
    assert user["is_demo"] is False

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT password_salt, password_hash, password_iterations FROM users WHERE id = ?",
            (user["id"],),
        ).fetchone()
    assert row is not None
    assert row[0]
    assert row[1]
    assert row[1] != "StrongPass123"
    assert int(row[2]) >= 100_000


def test_local_account_rejects_invalid_email_formats(tmp_path):
    db_path = tmp_path / "email_validation.db"
    invalid_emails = [
        "",
        "poojan",
        "poojan@",
        "@gmail.com",
        "poojan@gmail",
        "poojan@gmail.",
        "poojan@.com",
        "poojan@@gmail.com",
        "poojan@gmail..com",
        "poojan@-gmail.com",
        "poojan@gmail.c",
        "poojan gmail@gmail.com",
    ]

    for email in invalid_emails:
        with pytest.raises(ValueError):
            create_password_user(email, "StrongPass123", "Poojan", db_path=db_path)

    user = create_password_user(
        "  Poojan.Test+1@Gmail.COM  ",
        "StrongPass123",
        "Poojan",
        db_path=db_path,
    )
    assert user["email"] == "poojan.test+1@gmail.com"


def test_local_password_authentication_and_bad_password(tmp_path):
    db_path = tmp_path / "app.db"
    created = create_password_user(
        "poojan@example.com", "StrongPass123", "Poojan", db_path=db_path
    )
    authed = authenticate_password_user(
        "poojan@example.com", "StrongPass123", db_path=db_path
    )

    assert authed["id"] == created["id"]
    assert authed["auth_provider"] == "local_password"
    with pytest.raises(ValueError):
        authenticate_password_user("poojan@example.com", "WrongPass123", db_path=db_path)


def test_create_account_and_login_sets_expected_session_keys(monkeypatch, tmp_path):
    stub = _StreamlitStub()
    monkeypatch.setattr(auth_manager, "st", stub)

    assert auth_manager.is_user_unlocked() is False
    user = auth_manager.create_account_and_login(
        "poojan@example.com", "StrongPass123", "Poojan", db_path=tmp_path / "app.db"
    )

    assert user.is_authenticated is True
    assert user.is_demo is False
    assert user.auth_provider == "local_password"
    assert user.app_user_id is not None
    assert stub.session_state["user_unlocked"] is True
    assert stub.session_state["demo_user_id"] is None
    assert stub.session_state["current_app_user_id"] == user.app_user_id
    assert stub.session_state["primary_product_navigation"] == "User Goals & Saved Plans"


def test_logout_returns_to_public_session(monkeypatch, tmp_path):
    stub = _StreamlitStub()
    monkeypatch.setattr(auth_manager, "st", stub)
    auth_manager.create_account_and_login(
        "poojan@example.com", "StrongPass123", "Poojan", db_path=tmp_path / "app.db"
    )

    auth_manager.logout_current_user()

    assert stub.session_state["user_unlocked"] is False
    assert stub.session_state["demo_user_id"] is None
    assert "current_app_user_id" not in stub.session_state
    assert stub.session_state["primary_product_navigation"] == "Market Research Assistant"
    assert auth_manager.get_current_auth_user(tmp_path / "app.db").is_authenticated is False


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
    profile = apply_profile_defaults({"name": "Research User"})

    with pytest.raises(ValueError):
        save_user_profile(None, profile, db_path=db_path)
    with pytest.raises(ValueError):
        save_user_plan_run(None, profile, [], db_path=db_path)

    assert load_user_profile(None, db_path=db_path) is None
    assert load_latest_user_plan(None, db_path=db_path).empty


def test_current_logged_in_user_owns_profile_records(tmp_path):
    db_path = tmp_path / "app.db"
    user = create_password_user(
        "poojan@example.com", "StrongPass123", "Poojan", db_path=db_path
    )
    profile = apply_profile_defaults({"name": "Poojan", "goal_type": "Learn markets"})

    save_user_profile(user["id"], profile, db_path=db_path)
    loaded = load_user_profile(user["id"], db_path=db_path)

    assert loaded is not None
    assert loaded["name"] == "Poojan"
    assert loaded["goal_type"] == "Learning"


def test_topbar_appears_in_public_shell_and_sidebar_remains_navigation_only():
    app = _public_app()
    text = _text(app)

    assert not app.exception
    assert "Quant Research Lab" in text
    assert "Public preview" in text
    assert "Sign in / Create account" in [button.label for button in app.button]
    assert "Continue as Demo User" not in [button.label for button in app.button]
    assert app.sidebar.radio[0].options == [
        "Research Dashboard",
        "Login / Sign Up",
        "About / Methodology",
    ]


def test_login_signup_page_requires_app_credentials():
    app = _public_app()
    app.sidebar.radio[0].set_value("Login / Sign Up").run(timeout=90)
    text = _text(app)
    labels = [item.label for item in app.text_input]

    assert not app.exception
    assert "Access your research workspace" in text
    assert "Create account" in text
    assert "Sign in" in text
    assert "Email" in labels
    assert "Password" in labels or "Create password" in labels
    assert "Continue as Demo User" not in [button.label for button in app.button]
    assert "Do not enter broker, bank, trading-account credentials, or API secrets" in text


def test_locked_gated_page_source_has_login_guard():
    source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "GATED_PRODUCT_PAGES" in source
    assert "Log in to access this research page." in source
    guard = source.split("if not _is_user_unlocked() and (page_label in GATED_PRODUCT_PAGES", 1)[1]
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
