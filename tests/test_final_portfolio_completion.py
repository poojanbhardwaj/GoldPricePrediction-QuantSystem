from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sqlite3

import pandas as pd
import pytest

from src import auth_manager
from src.research_history import (
    compare_research_history_runs,
    load_latest_research_history_run,
    load_previous_research_history_run,
    load_research_history_runs,
    normalize_research_snapshot_for_history,
    save_research_history_run,
    summarize_research_changes,
)
from src.user_platform import (
    create_password_user,
    init_user_platform_db,
    load_user_preferences,
    save_user_preferences,
    validate_email_format,
    validate_password_strength,
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


class _FakeSupabaseAuth:
    def __init__(self, user, *, signup_session=None, signin_session=object()):
        self.user = user
        self.signup_session = signup_session
        self.signin_session = signin_session
        self.signed_out = False

    def sign_up(self, _credentials):
        return SimpleNamespace(user=self.user, session=self.signup_session)

    def sign_in_with_password(self, _credentials):
        return SimpleNamespace(user=self.user, session=self.signin_session)

    def sign_out(self):
        self.signed_out = True


def _history_snapshot(category: str, score: float, move: float) -> pd.DataFrame:
    return normalize_research_snapshot_for_history(
        pd.DataFrame([{
            "Asset": "Gold", "LatestPrice": 2300.0, "PredictedPrice": 2350.0,
            "PredictedMovePct": move, "SourceSnapshotDate": "2026-06-30",
        }]),
        watchlist=pd.DataFrame([{
            "Asset": "Gold", "OpportunityScore": score, "CandidateCategory": category,
            "RiskLabel": "Moderate",
        }]),
        edge_table=pd.DataFrame([{
            "Asset": "Gold", "EdgeStatus": "Research candidate", "CostVerdict": "Cost aware",
        }]),
        personalized_plans=pd.DataFrame([{"Asset": "Gold", "PlanType": "Paper Track"}]),
    )


def test_security_validators_reject_invalid_or_weak_values():
    assert validate_email_format("  Person+test@Example.COM ") == "person+test@example.com"
    for bad_email in ("person", "person@", "@example.com"):
        with pytest.raises(ValueError):
            validate_email_format(bad_email)
    for weak in ("short1", "abcdefgh", "12345678", "Password1"):
        with pytest.raises(ValueError):
            validate_password_strength(weak)
    redacted = auth_manager.no_secrets_in_logs("password=StrongPass123 api_key=abcdef")
    assert "StrongPass123" not in redacted
    assert "abcdef" not in redacted


def test_unverified_supabase_signup_does_not_unlock(monkeypatch, tmp_path):
    stub = _StreamlitStub()
    user = SimpleNamespace(
        id="provider-user", email="person@example.com", email_confirmed_at=None,
        confirmed_at=None, user_metadata={"display_name": "Person"},
    )
    client = SimpleNamespace(auth=_FakeSupabaseAuth(user, signup_session=None))
    monkeypatch.setattr(auth_manager, "st", stub)
    monkeypatch.setattr(auth_manager, "is_supabase_configured", lambda: True)
    monkeypatch.setattr(auth_manager, "get_supabase_client", lambda: client)

    result = auth_manager.sign_up_with_email(
        "person@example.com", "StrongPass123", "Person", db_path=tmp_path / "app.db"
    )

    assert result.is_authenticated is False
    assert auth_manager.is_user_unlocked() is False
    assert "Verification email sent" in stub.session_state["auth_notice"]


def test_verified_supabase_user_can_unlock(monkeypatch, tmp_path):
    stub = _StreamlitStub()
    user = SimpleNamespace(
        id="provider-user", email="verified@example.com",
        email_confirmed_at="2026-07-01T00:00:00Z", confirmed_at=None,
        user_metadata={"display_name": "Verified User"},
    )
    client = SimpleNamespace(auth=_FakeSupabaseAuth(user))
    monkeypatch.setattr(auth_manager, "st", stub)
    monkeypatch.setattr(auth_manager, "is_supabase_configured", lambda: True)
    monkeypatch.setattr(auth_manager, "get_supabase_client", lambda: client)

    result = auth_manager.sign_in_with_email(
        "verified@example.com", "StrongPass123", db_path=tmp_path / "app.db"
    )

    assert result.is_authenticated is True
    assert result.is_email_verified is True
    assert result.auth_provider == "supabase"
    assert auth_manager.require_verified_user().app_user_id == result.app_user_id


def test_preferences_are_isolated_per_user(tmp_path):
    db_path = tmp_path / "app.db"
    first = create_password_user("first@example.com", "StrongPass123", db_path=db_path)
    second = create_password_user("second@example.com", "StrongPass123", db_path=db_path)
    save_user_preferences(first["id"], {"default_assets": ["Gold"], "default_horizon": "5D"}, db_path)
    save_user_preferences(second["id"], {"default_assets": ["Bitcoin"], "default_horizon": "20D"}, db_path)

    assert load_user_preferences(first["id"], db_path)["default_assets"] == ["Gold"]
    assert load_user_preferences(second["id"], db_path)["default_assets"] == ["Bitcoin"]
    with pytest.raises(ValueError):
        save_user_preferences(None, {}, db_path)


def test_saved_default_page_is_used_after_local_login(monkeypatch, tmp_path):
    db_path = tmp_path / "app.db"
    user = create_password_user("person@example.com", "StrongPass123", db_path=db_path)
    save_user_preferences(user["id"], {"default_page": "Candidate Watchlist"}, db_path)
    stub = _StreamlitStub()
    monkeypatch.setattr(auth_manager, "st", stub)
    monkeypatch.setattr(auth_manager, "is_supabase_configured", lambda: False)

    auth_manager.sign_in_with_email("person@example.com", "StrongPass123", db_path=db_path)

    assert stub.session_state["primary_product_navigation"] == "Candidate Watchlist"


def test_migration_preserves_existing_user(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO users (id, email, name, created_at) VALUES (1, 'legacy@example.com', 'Legacy', '2025-01-01')"
        )
    init_user_platform_db(db_path)
    with sqlite3.connect(db_path) as connection:
        row = connection.execute("SELECT email, name FROM users WHERE id = 1").fetchone()
    assert row == ("legacy@example.com", "Legacy")


def test_research_history_normalizes_saves_and_compares_per_user(tmp_path):
    db_path = tmp_path / "app.db"
    first = create_password_user("first@example.com", "StrongPass123", db_path=db_path)
    second = create_password_user("second@example.com", "StrongPass123", db_path=db_path)
    older = _history_snapshot("Watchlist", 45.0, 1.0)
    newer = _history_snapshot("High Potential Candidate", 78.0, 4.5)

    first_run = save_research_history_run(first["id"], older, db_path=db_path)
    second_run = save_research_history_run(first["id"], newer, db_path=db_path)
    save_research_history_run(second["id"], older, db_path=db_path)

    assert first_run > 0 and second_run > first_run
    assert len(load_research_history_runs(first["id"], db_path)) == 2
    assert len(load_research_history_runs(second["id"], db_path)) == 1
    latest = load_latest_research_history_run(first["id"], db_path)
    previous = load_previous_research_history_run(first["id"], db_path)
    changes = compare_research_history_runs(previous, latest)
    assert changes.iloc[0]["ChangeSeverity"] in {"Upgrade", "Major Upgrade"}
    assert summarize_research_changes(changes)["Upgrades"] == 1


def test_placeholder_only_history_is_not_saved(tmp_path):
    placeholder = normalize_research_snapshot_for_history(pd.DataFrame([{
        "Asset": "Gold", "PredictedPrice": "Run research",
        "PredictedMovePct": "No saved estimate", "CandidateCategory": "Insufficient Evidence",
    }]))
    assert save_research_history_run(1, placeholder, db_path=tmp_path / "app.db") == 0


def test_no_secret_or_execution_fields_are_added():
    sources = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in ("src/auth_manager.py", "src/user_platform.py", "src/research_history.py")
    ).casefold()
    for forbidden_field in ("broker_password", "bank_password", "trading_api_secret"):
        assert forbidden_field not in sources


def test_product_shell_contains_gated_history_and_account_pages():
    source = Path("app.py").read_text(encoding="utf-8")
    navigation = source.split("PRIMARY_PRODUCT_PAGES = [", 1)[1].split("]", 1)[0]
    assert '"Research History & Changes"' in navigation
    assert '"Account & Settings"' in navigation
    assert 'elif page == "Research History & Changes":' in source
    assert 'elif page == "Account & Settings":' in source
    assert "GATED_PRODUCT_PAGES" in source


def test_primary_navigation_is_grouped_and_legacy_names_are_aliased():
    source = Path("app.py").read_text(encoding="utf-8")
    navigation = source.split("PRIMARY_PRODUCT_PAGES = [", 1)[1].split("]", 1)[0]
    for label in (
        "Research Dashboard", "Candidate Watchlist", "Evidence & Validation",
        "Forecast Explorer", "Asset Plans", "Cost & Risk Plan", "Goals & Saved Plans",
        "Research History & Changes", "Portfolio Research Summary", "Paper Research Log",
        "Account & Settings", "About / Methodology",
    ):
        assert f'"{label}"' in navigation
    for group in ("Dashboard", "Research", "Planning", "Account", "Info", "Advanced"):
        assert f'"{group}":' in source
    assert '"Market Research Assistant": "Research Dashboard"' in source
    assert '"Evidence of Edge": "Evidence & Validation"' in source
    assert '"Paper Research Journey": "Paper Research Log"' in source
    assert "_render_grouped_sidebar_navigation" in source


def test_research_asset_selector_is_page_scoped():
    source = Path("app.py").read_text(encoding="utf-8")
    navigation_block = source.split("page = PAGE_ROUTE_ALIASES.get", 1)[1].split("target_col =", 1)[0]
    account_block = source.split('elif page == "Account & Settings":', 1)[1].split('elif page == "About / Methodology":', 1)[0]
    history_block = source.split('elif page == "Research History & Changes":', 1)[1].split('elif page == "Paper Research Journey":', 1)[0]
    forecast_block = source.split('elif page == "Forecast Explorer":', 1)[1].split('elif page == "Cost-Aware Plan":', 1)[0]
    asset_plan_block = source.split('elif page == "Asset Plans":', 1)[1].split('elif page == "Forecast Explorer":', 1)[0]

    assert 'st.sidebar.selectbox("Research Asset"' not in navigation_block
    assert "_render_asset_selector" not in account_block
    assert "_render_asset_selector" not in history_block
    assert "_render_asset_selector" in forecast_block
    assert "_render_asset_selector" in asset_plan_block


def test_polished_source_and_empty_state_language_is_present():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "Snapshot Source Diagnostics" in source
    assert "Latest Refreshed Research Snapshot" in source
    assert "Saved Research Snapshot" in source
    assert "Cached Market Snapshot" in source
    assert "No candidate table available" in source
    assert "Validation evidence unavailable" in source
    assert "Forecast unavailable" in source
    assert "Local development auth" in source


def test_security_templates_ci_and_optional_auth_dependency_are_present():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    secrets_example = Path(".streamlit/secrets.example.toml").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "supabase>=" in requirements
    for pattern in ("data/*.db", "data/*.db-*", ".streamlit/secrets.toml", ".env.*", "phase*_files/", "*_debug.zip", "*_wip*.txt"):
        assert pattern in ignore
    assert "your-project.supabase.co" in secrets_example
    assert "your-anon-key" in secrets_example
    assert "SERVICE_ROLE" not in secrets_example.upper()
    assert 'python-version: "3.11"' in workflow
    assert "python -m compileall app.py src" in workflow
    assert "python -m pytest tests -q" in workflow


def test_recruiter_documentation_describes_auth_history_and_safety():
    readme = Path("README.md").read_text(encoding="utf-8")
    architecture = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    demo = Path("docs/DEMO_SCRIPT.md").read_text(encoding="utf-8")
    assert "Multi-Asset Quant Research Platform" in readme
    assert "Gold, Silver, Crude Oil, Bitcoin, S&P 500, and GLD" in readme
    assert "Supabase" in readme and "research history" in readme.casefold()
    assert "Authentication Flow" in architecture
    assert "Research History Flow" in architecture
    assert "Research History" in demo
    assert "Account Settings" in demo
