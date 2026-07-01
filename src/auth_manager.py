"""Auth/session abstraction for public visitors and local app accounts.

Phase 32D adds a real login gate without external auth dependencies. The app
collects only app-account email/password credentials, stores only salted PBKDF2
hashes, and never asks for broker, bank, trading-account credentials, or API
secrets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st

from src.user_platform import (
    authenticate_password_user,
    create_password_user,
    get_or_create_demo_user,
    init_user_platform_db,
)


@dataclass(frozen=True)
class AuthUser:
    app_user_id: Optional[int]
    auth_provider: str
    auth_user_id: str
    display_name: str
    email: Optional[str]
    is_demo: bool
    is_authenticated: bool


def get_public_user() -> AuthUser:
    """Return the locked/public visitor identity."""
    return AuthUser(
        app_user_id=None,
        auth_provider="public",
        auth_user_id="public-visitor",
        display_name="Public visitor",
        email=None,
        is_demo=False,
        is_authenticated=False,
    )


def is_user_unlocked() -> bool:
    """Whether the Streamlit session currently has authenticated workspace access."""
    return bool(st.session_state.get("user_unlocked", False))


def _write_authenticated_session(user: dict, *, provider: str) -> None:
    """Populate the session state keys used by app pages."""
    user_id = int(user["id"])
    display_name = str(user.get("display_name") or user.get("name") or "Research user")
    email = user.get("email")
    auth_user_id = str(user.get("auth_user_id") or email or f"user-{user_id}")
    st.session_state.user_unlocked = True
    st.session_state.current_app_user_id = user_id
    st.session_state.auth_provider = provider
    st.session_state.auth_user_id = auth_user_id
    st.session_state.current_user_label = display_name
    st.session_state.current_user_email = email
    st.session_state.demo_user_id = user_id if provider == "demo" else None


def get_current_auth_user(db_path: str | Path = "data/app.db") -> AuthUser:
    """Resolve the current app user from session state."""
    if not is_user_unlocked():
        return get_public_user()

    app_user_id = st.session_state.get("current_app_user_id")
    if app_user_id is None:
        # Backward compatibility for old local sessions; do not expose this path
        # in the UI, but keep existing tests/sessions from breaking.
        legacy_demo_id = st.session_state.get("demo_user_id")
        if legacy_demo_id is not None:
            app_user_id = legacy_demo_id
        else:
            return get_public_user()

    provider = str(st.session_state.get("auth_provider") or "local_password")
    display_name = str(st.session_state.get("current_user_label") or "Research user")
    return AuthUser(
        app_user_id=int(app_user_id),
        auth_provider=provider,
        auth_user_id=str(st.session_state.get("auth_user_id") or "local-user"),
        display_name=display_name,
        email=st.session_state.get("current_user_email"),
        is_demo=provider == "demo",
        is_authenticated=True,
    )


def create_account_and_login(
    email: str,
    password: str,
    display_name: str | None = None,
    db_path: str | Path = "data/app.db",
) -> AuthUser:
    """Create a local app account and log it into the current session."""
    init_user_platform_db(db_path)
    user = create_password_user(email, password, display_name, db_path=db_path)
    _write_authenticated_session(user, provider="local_password")
    st.session_state.primary_product_navigation = "User Goals & Saved Plans"
    return get_current_auth_user(db_path)


def login_with_password(
    email: str,
    password: str,
    db_path: str | Path = "data/app.db",
) -> AuthUser:
    """Log in a local app account."""
    init_user_platform_db(db_path)
    user = authenticate_password_user(email, password, db_path=db_path)
    _write_authenticated_session(user, provider="local_password")
    st.session_state.primary_product_navigation = "User Goals & Saved Plans"
    return get_current_auth_user(db_path)


def unlock_demo_user(db_path: str | Path = "data/app.db") -> AuthUser:
    """Backward-compatible demo unlock for tests only; not shown in production UI."""
    init_user_platform_db(db_path)
    user = get_or_create_demo_user(db_path)
    _write_authenticated_session(user, provider="demo")
    st.session_state.primary_product_navigation = "User Goals & Saved Plans"
    return get_current_auth_user(db_path)


def logout_current_user() -> None:
    """Return the session to public preview mode."""
    st.session_state.user_unlocked = False
    st.session_state.demo_user_id = None
    for key in (
        "auth_provider",
        "auth_user_id",
        "current_app_user_id",
        "current_user_label",
        "current_user_email",
    ):
        st.session_state.pop(key, None)
    st.session_state.primary_product_navigation = "Market Research Assistant"


__all__ = [
    "AuthUser",
    "get_public_user",
    "get_current_auth_user",
    "create_account_and_login",
    "login_with_password",
    "unlock_demo_user",
    "logout_current_user",
    "is_user_unlocked",
]
