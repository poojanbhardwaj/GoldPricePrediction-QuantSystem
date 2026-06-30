from __future__ import annotations

import ast
from pathlib import Path
import re
import sqlite3
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asset_config import get_asset_names
from src.user_platform import (
    apply_profile_defaults,
    choose_personalized_plan_type,
    get_or_create_demo_user,
    init_user_platform_db,
    list_user_plan_runs,
    load_latest_user_plan,
    load_user_profile,
    personalize_asset_plans,
    save_user_plan_run,
    save_user_profile,
)


BANNED_OUTPUT = re.compile(
    r"buy now|sell now|guaranteed profit|approved trade",
    flags=re.IGNORECASE,
)


def _profile(**overrides) -> dict:
    profile = {
        "name": "Researcher",
        "experience_level": "Beginner",
        "risk_tolerance": "Low",
        "goal_type": "Learning",
        "preferred_assets": ["Gold"],
        "default_horizon": "Auto",
        "simulated_capital": 10000,
        "style": "Conservative",
        "existing_position": "Planning only",
        "wants_simple_language": True,
    }
    profile.update(overrides)
    return profile


def _candidate(**overrides) -> dict:
    row = {
        "Asset": "Gold",
        "Category": "Watchlist Candidate",
        "Direction": "Bullish",
        "BestHorizon": 5,
        "OpportunityScore": 70,
        "Status": "Watch",
        "CostVerdict": "Costs Manageable",
        "Reason": "Validation breadth remains limited.",
        "UpgradeTrigger": "Require another matching validation window.",
    }
    row.update(overrides)
    return row


def _edge(**overrides) -> dict:
    row = {
        "Asset": "Gold",
        "EdgeStatus": "Watch Only",
        "EvidenceGrade": "C",
        "ActiveMinusPassivePct": 0.5,
        "CostVerdict": "Costs Manageable",
        "MainRisk": "Validation breadth remains limited.",
        "RequiredBeforeAction": "Require another matching validation window.",
    }
    row.update(overrides)
    return row


def test_db_init_creates_required_tables(tmp_path):
    db_path = tmp_path / "app.db"
    init_user_platform_db(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    assert {"users", "user_profiles", "user_plan_runs", "user_asset_plans", "audit_events"} <= tables


def test_demo_user_can_be_created_idempotently(tmp_path):
    db_path = tmp_path / "app.db"
    first = get_or_create_demo_user(db_path)
    second = get_or_create_demo_user(db_path)

    assert first == second
    assert first["email"] == "demo@local.app"
    assert first["name"] == "Demo User"


def test_profile_defaults_handle_unsure_answers():
    profile = apply_profile_defaults({
        "experience_level": "I don't know",
        "risk_tolerance": "I don't know",
        "goal_type": "I don't know",
        "preferred_assets": "Choose for me",
        "default_horizon": None,
        "simulated_capital": None,
        "style": "Choose for me",
        "existing_position": "I don't know",
    })

    assert profile["experience_level"] == "Beginner"
    assert profile["risk_tolerance"] == "Low"
    assert profile["goal_type"] == "Learning"
    assert profile["style"] == "Conservative"
    assert profile["preferred_assets"] == get_asset_names()
    assert profile["default_horizon"] == "Auto"
    assert profile["simulated_capital"] == 10000
    assert profile["existing_position"] == "Planning only"


def test_profile_can_be_saved_and_loaded(tmp_path):
    db_path = tmp_path / "app.db"
    user = get_or_create_demo_user(db_path)
    expected = apply_profile_defaults(_profile(preferred_assets=["Silver", "Bitcoin"]))

    profile_id = save_user_profile(user["id"], expected, db_path)
    loaded = load_user_profile(user["id"], db_path)

    assert profile_id > 0
    assert loaded == expected


def test_conservative_beginner_high_risk_is_not_accumulation_plan():
    plan_type, _, language = choose_personalized_plan_type(
        _profile(),
        _candidate(Status="High Risk"),
        _edge(),
    )

    assert plan_type in {"Blocked / Avoid for Now", "Watchlist Only"}
    assert plan_type != "Accumulate Only After Confirmation"
    assert BANNED_OUTPUT.search(language) is None


def test_existing_holder_gets_hold_existing_when_evidence_is_not_terrible():
    plan_type, reason, language = choose_personalized_plan_type(
        _profile(existing_position="Already hold"),
        _candidate(),
        _edge(),
    )

    assert plan_type == "Hold Existing Only"
    assert "already held" in reason
    assert "existing position" in language


def test_no_position_user_never_gets_immediate_action_language():
    plan_type, _, language = choose_personalized_plan_type(
        _profile(style="Balanced", existing_position="Do not currently hold"),
        _candidate(Category="Actionable Candidate"),
        _edge(EdgeStatus="Edge Supported", EvidenceGrade="A"),
    )

    assert plan_type == "Accumulate Only After Confirmation"
    assert BANNED_OUTPUT.search(language) is None
    assert "confirmation" in language.casefold()


def test_benchmark_weak_edge_prefers_passive_comparison():
    plan_type, reason, _ = choose_personalized_plan_type(
        _profile(),
        _candidate(),
        _edge(EdgeStatus="Benchmark Weak", EvidenceGrade="D", ActiveMinusPassivePct=-1.2),
    )

    assert plan_type == "Passive Benchmark Preferred"
    assert "passive" in reason.casefold()


def test_missing_evidence_remains_explicitly_unavailable():
    plan_type, reason, language = choose_personalized_plan_type(
        _profile(),
        _candidate(Category="Insufficient Evidence"),
        _edge(EdgeStatus="Insufficient Evidence", EvidenceGrade="F", ActiveMinusPassivePct=None),
    )

    assert plan_type == "Learn First"
    assert "Evidence unavailable" in reason
    assert "no action is suggested" in language


def test_personalized_plan_run_can_be_saved_and_loaded(tmp_path):
    db_path = tmp_path / "app.db"
    user = get_or_create_demo_user(db_path)
    profile = _profile()
    save_user_profile(user["id"], profile, db_path)
    plans = personalize_asset_plans(
        profile,
        pd.DataFrame([_candidate()]),
        pd.DataFrame([_edge()]),
    )

    run_id = save_user_plan_run(user["id"], profile, plans, db_path=db_path)
    loaded = load_latest_user_plan(user["id"], db_path)
    runs = list_user_plan_runs(user["id"], db_path)

    assert run_id > 0
    assert loaded["Asset"].tolist() == ["Gold"]
    assert loaded.iloc[0]["PlanType"] == "Watchlist Only"
    assert runs.iloc[0]["PlanRunId"] == run_id
    assert runs.iloc[0]["AssetCount"] == 1


def test_all_generated_plan_text_avoids_banned_phrases():
    watch = pd.DataFrame([
        _candidate(Asset="Gold"),
        _candidate(Asset="Silver", Status="High Risk"),
    ])
    edges = pd.DataFrame([
        _edge(Asset="Gold"),
        _edge(Asset="Silver", EdgeStatus="Cost Blocked", EvidenceGrade="F"),
    ])
    plans = personalize_asset_plans(
        _profile(preferred_assets=["Gold", "Silver"]), watch, edges
    )

    text = " ".join(plans.astype(str).to_numpy().ravel())
    assert BANNED_OUTPUT.search(text) is None
    assert set(plans["Asset"]) == {"Gold", "Silver"}


def test_app_navigation_places_user_goals_after_evidence_of_edge():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(app_source)
    navigation = app_source.split("PRIMARY_PRODUCT_PAGES = [", 1)[1].split("]", 1)[0]
    page_block = app_source.split('elif page == "User Goals & Saved Plans":', 1)[1].split(
        'elif page == "Paper Research Journey":', 1
    )[0]

    assert navigation.index('"Evidence of Edge"') < navigation.index('"User Goals & Saved Plans"')
    assert navigation.index('"User Goals & Saved Plans"') < navigation.index('"Asset Plans"')
    assert "_render_user_goals_saved_plans()" in page_block
    assert "Demo user mode" in page_block


def test_user_page_has_no_credential_collection_fields():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    platform_source = (ROOT / "src" / "user_platform.py").read_text(encoding="utf-8")
    combined = app_source + platform_source

    assert re.search(r"(?:text_input|number_input)\([^\n]*(?:password|broker|bank account)", combined, re.IGNORECASE) is None
    assert "password_hash" not in platform_source

