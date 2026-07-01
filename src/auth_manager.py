"""Session-only auth abstraction for public and local demo users.

This module intentionally does not implement passwords or external auth. It keeps
Phase 32B demo mode working while giving the app a single current-user object
that can later be backed by Supabase/Auth0/Clerk without changing page logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st

from src.user_platform import get_or_create_demo_user, init_user_platform_db


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
    """Whether the Streamlit session currently has workspace access."""
    return bool(st.session_state.get("user_unlocked", False))


def _write_demo_session(user: dict) -> None:
    """Populate the session state keys used by Phase 32B and newer pages."""
    user_id = int(user["id"])
    st.session_state.user_unlocked = True
    st.session_state.demo_user_id = user_id
    st.session_state.auth_provider = "demo"
    st.session_state.auth_user_id = "demo-user"
    st.session_state.current_app_user_id = user_id
    st.session_state.current_user_label = str(
        user.get("display_name") or user.get("name") or "Demo user"
    )
    st.session_state.current_user_email = user.get("email")


def get_current_auth_user(db_path: str | Path = "data/app.db") -> AuthUser:
    """Resolve the current app user from session state.

    Older tests and sessions may only set ``user_unlocked=True``. In that case we
    resolve the backward-compatible demo user instead of leaving the app in a
    half-unlocked state.
    """
    if not is_user_unlocked():
        return get_public_user()

    app_user_id = st.session_state.get(
        "current_app_user_id", st.session_state.get("demo_user_id")
    )
    if app_user_id is None:
        try:
            init_user_platform_db(db_path)
            user = get_or_create_demo_user(db_path)
            _write_demo_session(user)
            app_user_id = st.session_state.get("current_app_user_id")
        except Exception:
            return get_public_user()

    provider = str(st.session_state.get("auth_provider") or "demo")
    return AuthUser(
        app_user_id=int(app_user_id),
        auth_provider=provider,
        auth_user_id=str(st.session_state.get("auth_user_id") or "demo-user"),
        display_name=str(st.session_state.get("current_user_label") or "Demo user"),
        email=st.session_state.get("current_user_email"),
        is_demo=provider == "demo",
        is_authenticated=True,
    )


def unlock_demo_user(db_path: str | Path = "data/app.db") -> AuthUser:
    """Unlock the local demo workspace without collecting credentials."""
    init_user_platform_db(db_path)
    user = get_or_create_demo_user(db_path)
    _write_demo_session(user)
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
    "unlock_demo_user",
    "logout_current_user",
    "is_user_unlocked",
]
