"""Provider-aware authentication with a secure local-development fallback."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional

import streamlit as st

from src.user_platform import (
    authenticate_password_user,
    create_password_user,
    get_or_create_demo_user,
    get_or_create_user_for_auth,
    init_user_platform_db,
    load_user_preferences,
    update_user_last_active,
    validate_email_format as _validate_email_format,
    validate_password_strength as _validate_password_strength,
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
    is_email_verified: bool = False
    is_local_dev: bool = False
    created_at: Optional[str] = None
    last_active_at: Optional[str] = None


def _secret(name: str) -> Optional[str]:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    value = value or os.environ.get(name)
    return str(value).strip() if value else None


def is_supabase_configured() -> bool:
    return bool(_secret("SUPABASE_URL") and _secret("SUPABASE_ANON_KEY"))


def get_supabase_client() -> Any:
    if not is_supabase_configured():
        raise RuntimeError("Supabase is not configured")
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError("Supabase support is configured but the client package is unavailable") from exc
    return create_client(_secret("SUPABASE_URL"), _secret("SUPABASE_ANON_KEY"))


def validate_email_format(email: str) -> str:
    return _validate_email_format(email)


def validate_password_strength(password: str) -> str:
    return _validate_password_strength(password)


def sanitize_display_name(value: str | None) -> str:
    text = re.sub(r"[<>\x00-\x1f]", "", str(value or "")).strip()
    text = re.sub(r"\s+", " ", text)[:80]
    return text or "Research user"


def no_secrets_in_logs(value: Any) -> str:
    text = str(value or "")
    text = re.sub(
        r"(?i)(api[_ -]?key|token|password|secret|authorization)\s*[=:]\s*\S+",
        r"\1=[redacted]",
        text,
    )
    return re.sub(r"([a-z]+://)[^/\s:@]+:[^@\s/]+@", r"\1[redacted]@", text)


def get_public_user() -> AuthUser:
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
    return bool(st.session_state.get("user_unlocked", False))


def _write_authenticated_session(
    user: Mapping[str, Any],
    *,
    provider: str,
    email_verified: bool,
    local_dev: bool,
    db_path: str | Path = "data/app.db",
) -> None:
    user_id = int(user["id"])
    display_name = sanitize_display_name(user.get("display_name") or user.get("name"))
    st.session_state.user_unlocked = True
    st.session_state.current_app_user_id = user_id
    st.session_state.auth_provider = provider
    st.session_state.auth_user_id = str(user.get("auth_user_id") or user.get("email") or f"user-{user_id}")
    st.session_state.current_user_label = display_name
    st.session_state.current_user_email = user.get("email")
    st.session_state.current_user_email_verified = bool(email_verified)
    st.session_state.current_user_is_local_dev = bool(local_dev)
    st.session_state.current_user_created_at = user.get("created_at")
    st.session_state.current_user_last_active_at = user.get("last_active_at")
    st.session_state.demo_user_id = user_id if provider == "demo" else None
    update_user_last_active(user_id, db_path=db_path)


def _set_post_login_page(user_id: int, db_path: str | Path) -> None:
    preferences = load_user_preferences(user_id, db_path=db_path) or {}
    st.session_state.primary_product_navigation = str(
        preferences.get("default_page") or "User Goals & Saved Plans"
    )


def get_current_auth_user(db_path: str | Path = "data/app.db") -> AuthUser:
    if not is_user_unlocked():
        return get_public_user()
    app_user_id = st.session_state.get("current_app_user_id", st.session_state.get("demo_user_id"))
    if app_user_id is None:
        return get_public_user()
    provider = str(st.session_state.get("auth_provider") or "local_password")
    return AuthUser(
        app_user_id=int(app_user_id),
        auth_provider=provider,
        auth_user_id=str(st.session_state.get("auth_user_id") or "local-user"),
        display_name=str(st.session_state.get("current_user_label") or "Research user"),
        email=st.session_state.get("current_user_email"),
        is_demo=provider == "demo",
        is_authenticated=True,
        is_email_verified=bool(st.session_state.get("current_user_email_verified", provider == "supabase")),
        is_local_dev=bool(st.session_state.get("current_user_is_local_dev", provider == "local_password")),
        created_at=st.session_state.get("current_user_created_at"),
        last_active_at=st.session_state.get("current_user_last_active_at"),
    )


def _supabase_user_verified(user: Any) -> bool:
    return bool(
        getattr(user, "email_confirmed_at", None)
        or getattr(user, "confirmed_at", None)
        or (getattr(user, "user_metadata", None) or {}).get("email_verified")
    )


def _map_supabase_user(
    user: Any,
    display_name: str | None = None,
    db_path: str | Path = "data/app.db",
) -> dict[str, Any]:
    provider_user_id = str(getattr(user, "id", "") or "")
    email = validate_email_format(getattr(user, "email", "") or "")
    metadata = getattr(user, "user_metadata", None) or {}
    name = sanitize_display_name(display_name or metadata.get("display_name") or metadata.get("name") or email.split("@", 1)[0])
    return get_or_create_user_for_auth(
        auth_provider="supabase",
        auth_user_id=provider_user_id,
        display_name=name,
        email=email,
        is_demo=False,
        db_path=db_path,
    )


def sign_up_with_email(
    email: str,
    password: str,
    display_name: str | None = None,
    db_path: str | Path = "data/app.db",
) -> AuthUser:
    normalized_email = validate_email_format(email)
    strong_password = validate_password_strength(password)
    clean_name = sanitize_display_name(display_name)
    if is_supabase_configured():
        response = get_supabase_client().auth.sign_up({
            "email": normalized_email,
            "password": strong_password,
            "options": {"data": {"display_name": clean_name}},
        })
        provider_user = getattr(response, "user", None)
        session = getattr(response, "session", None)
        if provider_user is not None and session is not None and _supabase_user_verified(provider_user):
            app_user = _map_supabase_user(provider_user, clean_name, db_path=db_path)
            _write_authenticated_session(
                app_user, provider="supabase", email_verified=True, local_dev=False, db_path=db_path
            )
            _set_post_login_page(int(app_user["id"]), db_path)
            return get_current_auth_user()
        logout_current_user()
        st.session_state.auth_notice = (
            "Verification email sent. Please verify your email before accessing the research workspace."
        )
        return AuthUser(None, "supabase", str(getattr(provider_user, "id", "pending")), clean_name, normalized_email, False, False)

    user = create_password_user(normalized_email, strong_password, clean_name, db_path=db_path)
    _write_authenticated_session(
        user, provider="local_password", email_verified=False, local_dev=True, db_path=db_path
    )
    st.session_state.auth_notice = "Local development auth mode: email ownership is not verified."
    _set_post_login_page(int(user["id"]), db_path)
    return get_current_auth_user()


def sign_in_with_email(
    email: str,
    password: str,
    db_path: str | Path = "data/app.db",
) -> AuthUser:
    normalized_email = validate_email_format(email)
    if is_supabase_configured():
        response = get_supabase_client().auth.sign_in_with_password({
            "email": normalized_email,
            "password": str(password or ""),
        })
        provider_user = getattr(response, "user", None)
        if provider_user is None or not _supabase_user_verified(provider_user):
            logout_current_user()
            raise ValueError("Please verify your email first.")
        app_user = _map_supabase_user(provider_user, db_path=db_path)
        _write_authenticated_session(
            app_user, provider="supabase", email_verified=True, local_dev=False, db_path=db_path
        )
        _set_post_login_page(int(app_user["id"]), db_path)
        return get_current_auth_user()

    user = authenticate_password_user(normalized_email, password, db_path=db_path)
    _write_authenticated_session(
        user, provider="local_password", email_verified=False, local_dev=True, db_path=db_path
    )
    st.session_state.auth_notice = "Local development auth mode: email ownership is not verified."
    _set_post_login_page(int(user["id"]), db_path)
    return get_current_auth_user()


def require_verified_user() -> AuthUser:
    user = get_current_auth_user()
    if not user.is_authenticated:
        return get_public_user()
    if user.is_email_verified or user.is_local_dev or user.is_demo:
        return user
    return get_public_user()


def create_account_and_login(
    email: str,
    password: str,
    display_name: str | None = None,
    db_path: str | Path = "data/app.db",
) -> AuthUser:
    if is_supabase_configured():
        return sign_up_with_email(email, password, display_name, db_path=db_path)
    user = create_password_user(email, password, sanitize_display_name(display_name), db_path=db_path)
    _write_authenticated_session(
        user, provider="local_password", email_verified=False, local_dev=True, db_path=db_path
    )
    st.session_state.auth_notice = "Local development auth mode: email ownership is not verified."
    _set_post_login_page(int(user["id"]), db_path)
    return get_current_auth_user(db_path)


def login_with_password(
    email: str,
    password: str,
    db_path: str | Path = "data/app.db",
) -> AuthUser:
    if is_supabase_configured():
        return sign_in_with_email(email, password, db_path=db_path)
    user = authenticate_password_user(email, password, db_path=db_path)
    _write_authenticated_session(
        user, provider="local_password", email_verified=False, local_dev=True, db_path=db_path
    )
    st.session_state.auth_notice = "Local development auth mode: email ownership is not verified."
    _set_post_login_page(int(user["id"]), db_path)
    return get_current_auth_user(db_path)


def unlock_demo_user(db_path: str | Path = "data/app.db") -> AuthUser:
    init_user_platform_db(db_path)
    user = get_or_create_demo_user(db_path)
    _write_authenticated_session(
        user, provider="demo", email_verified=False, local_dev=True, db_path=db_path
    )
    _set_post_login_page(int(user["id"]), db_path)
    return get_current_auth_user(db_path)


def logout_current_user() -> None:
    if st.session_state.get("auth_provider") == "supabase" and is_supabase_configured():
        try:
            get_supabase_client().auth.sign_out()
        except Exception:
            pass
    st.session_state.user_unlocked = False
    st.session_state.demo_user_id = None
    for key in (
        "auth_provider", "auth_user_id", "current_app_user_id", "current_user_label",
        "current_user_email", "current_user_email_verified", "current_user_is_local_dev",
        "current_user_created_at", "current_user_last_active_at",
    ):
        st.session_state.pop(key, None)
    st.session_state.primary_product_navigation = "Market Research Assistant"


def sign_out() -> None:
    logout_current_user()


__all__ = [
    "AuthUser", "get_public_user", "get_current_auth_user", "is_supabase_configured",
    "get_supabase_client", "validate_email_format", "validate_password_strength",
    "sanitize_display_name", "no_secrets_in_logs", "sign_up_with_email", "sign_in_with_email",
    "sign_out", "require_verified_user", "create_account_and_login", "login_with_password",
    "unlock_demo_user", "logout_current_user", "is_user_unlocked",
]
