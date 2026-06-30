"""Demo-user profiles and conservative personalized paper-research plans."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.asset_config import get_asset_names


PLAN_TYPES = (
    "Learn First",
    "Watchlist Only",
    "Paper Track",
    "Hold Existing Only",
    "Accumulate Only After Confirmation",
    "Passive Benchmark Preferred",
    "Blocked / Avoid for Now",
)

PERSONALIZED_PLAN_COLUMNS = [
    "Asset",
    "PlanType",
    "Direction",
    "GoalFit",
    "RiskFit",
    "EvidenceGrade",
    "EdgeStatus",
    "PersonalizedPlan",
    "WhatToMonitor",
    "RequiredBeforeAction",
    "ReviewWhen",
    "ExistingPositionGuidance",
    "SimulatedCapital",
    "SavedAt",
]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _database_path(db_path: str | Path) -> str:
    value = str(db_path)
    if value != ":memory:":
        Path(value).expanduser().parent.mkdir(parents=True, exist_ok=True)
    return value


def _connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(_database_path(db_path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_user_platform_db(db_path: str | Path = "data/app.db") -> None:
    """Create the local demo-user schema when it does not already exist."""
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                name TEXT NOT NULL,
                experience_level TEXT NOT NULL,
                risk_tolerance TEXT NOT NULL,
                goal_type TEXT NOT NULL,
                preferred_assets_json TEXT NOT NULL,
                default_horizon TEXT NOT NULL,
                simulated_capital REAL NOT NULL,
                style TEXT NOT NULL,
                existing_position TEXT NOT NULL,
                wants_simple_language INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS user_plan_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                profile_json TEXT NOT NULL,
                source TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS user_asset_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_run_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                asset TEXT NOT NULL,
                plan_type TEXT NOT NULL,
                direction TEXT,
                goal_fit TEXT,
                risk_fit TEXT,
                evidence_grade TEXT,
                edge_status TEXT,
                personalized_plan TEXT,
                what_to_monitor TEXT,
                required_before_action TEXT,
                review_when TEXT,
                existing_position_guidance TEXT,
                simulated_capital REAL,
                saved_at TEXT NOT NULL,
                FOREIGN KEY(plan_run_id) REFERENCES user_plan_runs(id),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                event_type TEXT NOT NULL,
                event_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )


def get_or_create_demo_user(db_path: str | Path = "data/app.db") -> dict[str, Any]:
    """Return the single local demo identity without collecting credentials."""
    init_user_platform_db(db_path)
    timestamp = _now()
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO users (email, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET updated_at = excluded.updated_at
            """,
            ("demo@local.app", "Demo User", timestamp, timestamp),
        )
        row = connection.execute(
            "SELECT id, email, name FROM users WHERE email = ?", ("demo@local.app",)
        ).fetchone()
    return dict(row) if row is not None else {}


def _is_unsure(value: Any) -> bool:
    return value is None or str(value).strip().casefold() in {
        "",
        "i don't know",
        "i dont know",
        "choose for me",
        "auto select",
    }


def _preferred_assets(value: Any) -> list[str]:
    supported = get_asset_names()
    if _is_unsure(value):
        return supported
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            values = parsed if isinstance(parsed, list) else [value]
        except (TypeError, ValueError, json.JSONDecodeError):
            values = [part.strip() for part in value.split(",")]
    elif isinstance(value, Iterable):
        values = list(value)
    else:
        values = []
    selected = [asset for asset in supported if asset in values]
    return selected or supported


def apply_profile_defaults(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Map beginner-facing answers to stable conservative research defaults."""
    raw = dict(profile or {})
    experience = raw.get("experience_level")
    experience = "Beginner" if _is_unsure(experience) else str(experience)

    raw_risk = raw.get("risk_tolerance", raw.get("risk_comfort"))
    risk_map = {
        "i hate losses": "Low",
        "i can handle small swings": "Medium",
        "i can handle high volatility": "High",
    }
    risk = "Low" if _is_unsure(raw_risk) else risk_map.get(str(raw_risk).casefold(), str(raw_risk))

    raw_goal = raw.get("goal_type", raw.get("goal"))
    goal_map = {"learn markets": "Learning"}
    goal = "Learning" if _is_unsure(raw_goal) else goal_map.get(str(raw_goal).casefold(), str(raw_goal))

    style_value = raw.get("style")
    style = "Conservative" if _is_unsure(style_value) else str(style_value)
    horizon_value = raw.get("default_horizon", raw.get("horizon"))
    horizon = "Auto" if _is_unsure(horizon_value) else str(horizon_value)
    position_value = raw.get("existing_position")
    position = "Planning only" if _is_unsure(position_value) else str(position_value)

    capital = pd.to_numeric(pd.Series([raw.get("simulated_capital")]), errors="coerce").iloc[0]
    simulated_capital = 10000.0 if pd.isna(capital) or float(capital) <= 0 else float(capital)
    simple_value = raw.get("wants_simple_language", True)
    if isinstance(simple_value, str):
        wants_simple = simple_value.strip().casefold() not in {"false", "0", "no", "off"}
    else:
        wants_simple = bool(simple_value)

    return {
        "name": str(raw.get("name") or "Demo User"),
        "experience_level": experience,
        "risk_tolerance": risk,
        "goal_type": goal,
        "preferred_assets": _preferred_assets(raw.get("preferred_assets")),
        "default_horizon": horizon,
        "simulated_capital": simulated_capital,
        "style": style,
        "existing_position": position,
        "wants_simple_language": wants_simple,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def save_user_profile(
    user_id: int,
    profile: Mapping[str, Any],
    db_path: str | Path = "data/app.db",
) -> int:
    """Insert or update one demo-user profile and return its row id."""
    init_user_platform_db(db_path)
    normalized = apply_profile_defaults(profile)
    timestamp = _now()
    values = (
        int(user_id),
        normalized["name"],
        normalized["experience_level"],
        normalized["risk_tolerance"],
        normalized["goal_type"],
        json.dumps(normalized["preferred_assets"]),
        normalized["default_horizon"],
        normalized["simulated_capital"],
        normalized["style"],
        normalized["existing_position"],
        int(normalized["wants_simple_language"]),
        timestamp,
        timestamp,
    )
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_profiles (
                user_id, name, experience_level, risk_tolerance, goal_type,
                preferred_assets_json, default_horizon, simulated_capital, style,
                existing_position, wants_simple_language, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name = excluded.name,
                experience_level = excluded.experience_level,
                risk_tolerance = excluded.risk_tolerance,
                goal_type = excluded.goal_type,
                preferred_assets_json = excluded.preferred_assets_json,
                default_horizon = excluded.default_horizon,
                simulated_capital = excluded.simulated_capital,
                style = excluded.style,
                existing_position = excluded.existing_position,
                wants_simple_language = excluded.wants_simple_language,
                updated_at = excluded.updated_at
            """,
            values,
        )
        profile_id = connection.execute(
            "SELECT id FROM user_profiles WHERE user_id = ?", (int(user_id),)
        ).fetchone()["id"]
        connection.execute(
            "INSERT INTO audit_events (user_id, event_type, event_json, created_at) VALUES (?, ?, ?, ?)",
            (int(user_id), "ProfileSaved", json.dumps(_json_safe(normalized)), timestamp),
        )
    return int(profile_id)


def load_user_profile(
    user_id: int, db_path: str | Path = "data/app.db"
) -> dict[str, Any] | None:
    """Load one saved profile, returning None before the first save."""
    init_user_platform_db(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT name, experience_level, risk_tolerance, goal_type,
                   preferred_assets_json, default_horizon, simulated_capital, style,
                   existing_position, wants_simple_language
            FROM user_profiles WHERE user_id = ?
            """,
            (int(user_id),),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    try:
        result["preferred_assets"] = json.loads(result.pop("preferred_assets_json"))
    except (TypeError, ValueError, json.JSONDecodeError):
        result["preferred_assets"] = get_asset_names()
        result.pop("preferred_assets_json", None)
    result["wants_simple_language"] = bool(result["wants_simple_language"])
    return result


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _number(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def choose_personalized_plan_type(
    profile: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    edge_row: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Choose a research-only plan using explicit conservative policy gates."""
    user = apply_profile_defaults(profile)
    candidate = dict(candidate_row or {})
    edge = dict(edge_row or {})
    edge_status = str(edge.get("EdgeStatus") or "Insufficient Evidence")
    grade = str(edge.get("EvidenceGrade") or "F").upper()
    status = _normalized_text(candidate.get("Status", edge.get("Status")))
    category = _normalized_text(candidate.get("Category"))
    risk_text = _normalized_text(
        f"{candidate.get('Reason', '')} {candidate.get('MainRisk', '')} {edge.get('MainRisk', '')}"
    )
    style = _normalized_text(user["style"])
    experience = _normalized_text(user["experience_level"])
    existing = _normalized_text(user["existing_position"])
    cost_verdict = _normalized_text(edge.get("CostVerdict", candidate.get("CostVerdict")))
    gap = _number(edge.get("ActiveMinusPassivePct"))

    if edge_status == "Cost Blocked" or "coststoohighforsignal" in cost_verdict.replace(" ", ""):
        return (
            "Blocked / Avoid for Now",
            "Modeled costs are too large for the saved estimate.",
            "Do not act on this yet. Keep it blocked until the estimated move clearly exceeds costs.",
        )

    if edge_status == "Benchmark Weak" or (gap is not None and gap < 0):
        return (
            "Passive Benchmark Preferred",
            "The passive comparison is stronger than the active saved estimate.",
            "Use the passive benchmark as the research reference and wait for stronger active evidence.",
        )

    high_risk = status == "high risk" or "high risk" in category or "high risk" in risk_text
    if high_risk:
        if style == "aggressive":
            return (
                "Paper Track",
                "Risk is high, so only simulated observation is suitable.",
                "Consider paper-tracking first and require another confirmation before changing the plan.",
            )
        if style == "balanced":
            return (
                "Watchlist Only",
                "The saved risk warning is stronger than the available evidence.",
                "Keep this on the watchlist and wait for the risk warning to improve.",
            )
        return (
            "Blocked / Avoid for Now",
            "This candidate is outside a conservative risk comfort level.",
            "Do not act on this yet. Review it only after the high-risk warning clears.",
        )

    if edge_status == "Insufficient Evidence" or grade == "F":
        if experience == "beginner" or style == "conservative":
            return (
                "Learn First",
                "Evidence unavailable: the saved record is not complete enough for a plan.",
                "Use this only to learn how the asset is tracked; no action is suggested.",
            )
        return (
            "Blocked / Avoid for Now",
            "Evidence unavailable: the saved record is incomplete.",
            "Wait for valid benchmark, cost, risk, and validation evidence.",
        )

    if existing == "already hold":
        return (
            "Hold Existing Only",
            "The profile says this asset is already held and the evidence is not severely blocked.",
            "Monitor the existing position and avoid adding more until the saved evidence improves.",
        )

    if edge_status == "Edge Supported" and grade in {"A", "B"}:
        if style == "conservative":
            return (
                "Paper Track",
                "The saved edge evidence passed the current research gates without a high-risk status.",
                "This is suitable only for paper tracking before any stronger consideration.",
            )
        return (
            "Accumulate Only After Confirmation",
            "The edge is supported, but another confirmation is still required.",
            "Consider only a simulated staged plan after repeated confirmation; this is not a trading recommendation.",
        )

    return (
        "Watchlist Only",
        "The candidate is visible, but its evidence does not support a stronger plan.",
        "Add it to the watchlist and review it after the next saved evidence update.",
    )


def _best_asset_row(frame: pd.DataFrame, asset: str) -> dict[str, Any]:
    if frame.empty or "Asset" not in frame.columns:
        return {}
    rows = frame.loc[frame["Asset"].astype(str).eq(asset)].copy()
    if rows.empty:
        return {}
    score = pd.to_numeric(
        rows.get("OpportunityScore", pd.Series(index=rows.index, dtype=float)), errors="coerce"
    ).fillna(-1.0)
    return rows.loc[score.sort_values(ascending=False).index[0]].to_dict()


def _goal_fit(profile: Mapping[str, Any], plan_type: str, horizon: Any) -> str:
    goal = _normalized_text(profile.get("goal_type"))
    horizon_value = _number(horizon)
    if goal == "learning":
        return "Supports learning and evidence comparison"
    if goal == "risk monitoring":
        return "Aligned with risk monitoring"
    if goal == "short-term monitoring":
        return "Aligned with short-term monitoring" if horizon_value in {1, 5} else "Longer than the preferred monitoring window"
    if goal == "long-term research":
        return "Aligned with long-term monitoring" if plan_type in {"Hold Existing Only", "Passive Benchmark Preferred", "Watchlist Only"} else "Needs longer evidence"
    return "General research fit"


def _risk_fit(profile: Mapping[str, Any], plan_type: str, status: Any) -> str:
    if _normalized_text(status) == "high risk":
        return "Outside stated risk comfort"
    if plan_type in {"Blocked / Avoid for Now", "Learn First"}:
        return "Risk protection applied"
    if _normalized_text(profile.get("risk_tolerance")) == "low" and plan_type == "Accumulate Only After Confirmation":
        return "Requires a more conservative confirmation"
    return "Within paper-research limits"


def personalize_asset_plans(
    user_profile: Mapping[str, Any],
    watchlist: pd.DataFrame,
    edge_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build one beginner-readable, research-only plan for each selected asset."""
    profile = apply_profile_defaults(user_profile)
    watch = watchlist.copy() if isinstance(watchlist, pd.DataFrame) else pd.DataFrame(watchlist)
    edges = edge_table.copy() if isinstance(edge_table, pd.DataFrame) else pd.DataFrame(edge_table)
    available_assets: list[str] = []
    for frame in (watch, edges):
        if "Asset" not in frame.columns:
            continue
        for asset in frame["Asset"].dropna().astype(str):
            if asset not in available_assets:
                available_assets.append(asset)
    assets = [asset for asset in profile["preferred_assets"] if asset in available_assets]
    timestamp = _now()
    output: list[dict[str, Any]] = []

    for asset in assets:
        candidate = _best_asset_row(watch, asset)
        edge = _best_asset_row(edges, asset)
        plan_type, reason, action_language = choose_personalized_plan_type(profile, candidate, edge)
        direction = str(candidate.get("Direction") or edge.get("Direction") or "Neutral")
        grade = str(edge.get("EvidenceGrade") or "Evidence unavailable")
        edge_status = str(edge.get("EdgeStatus") or "Insufficient Evidence")
        required = str(
            edge.get("RequiredBeforeAction")
            or candidate.get("UpgradeTrigger")
            or "Wait for another saved evidence update and repeat the comparison."
        )
        horizon = candidate.get("BestHorizon", candidate.get("Horizon"))
        review_when = (
            f"After the next {int(float(horizon))}-trading-day outcome or saved evidence update."
            if _number(horizon) is not None
            else "After the next saved evidence update."
        )
        if profile["existing_position"] == "Already hold":
            position_guidance = "Monitor the existing position and do not add exposure while blockers remain."
        elif profile["existing_position"] == "Do not currently hold":
            position_guidance = "No current position is assumed; use paper tracking or watchlist review only."
        else:
            position_guidance = "This plan assumes research and simulation only."

        output.append({
            "Asset": asset,
            "PlanType": plan_type,
            "Direction": direction,
            "GoalFit": _goal_fit(profile, plan_type, horizon),
            "RiskFit": _risk_fit(profile, plan_type, candidate.get("Status", edge.get("Status"))),
            "EvidenceGrade": grade,
            "EdgeStatus": edge_status,
            "PersonalizedPlan": f"{action_language} {reason}",
            "WhatToMonitor": str(
                edge.get("MainRisk")
                or candidate.get("Reason")
                or "Monitor risk, costs, and the passive benchmark comparison."
            ),
            "RequiredBeforeAction": required,
            "ReviewWhen": review_when,
            "ExistingPositionGuidance": position_guidance,
            "SimulatedCapital": profile["simulated_capital"],
            "SavedAt": timestamp,
        })

    return pd.DataFrame(output, columns=PERSONALIZED_PLAN_COLUMNS)


def save_user_plan_run(
    user_id: int,
    profile: Mapping[str, Any],
    personalized_plans: pd.DataFrame,
    source: str = "phase32a",
    db_path: str | Path = "data/app.db",
) -> int:
    """Persist an immutable personalized plan run and its asset rows."""
    init_user_platform_db(db_path)
    normalized_profile = apply_profile_defaults(profile)
    plans = personalized_plans.copy() if isinstance(personalized_plans, pd.DataFrame) else pd.DataFrame(personalized_plans)
    saved_at = _now()
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO user_plan_runs (user_id, profile_json, source, saved_at) VALUES (?, ?, ?, ?)",
            (
                int(user_id),
                json.dumps(_json_safe(normalized_profile)),
                str(source or "phase32a"),
                saved_at,
            ),
        )
        run_id = int(cursor.lastrowid)
        for _, row in plans.iterrows():
            values = row.to_dict()
            connection.execute(
                """
                INSERT INTO user_asset_plans (
                    plan_run_id, user_id, asset, plan_type, direction, goal_fit, risk_fit,
                    evidence_grade, edge_status, personalized_plan, what_to_monitor,
                    required_before_action, review_when, existing_position_guidance,
                    simulated_capital, saved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(user_id),
                    str(values.get("Asset", "")),
                    str(values.get("PlanType", "Learn First")),
                    str(values.get("Direction", "Neutral")),
                    str(values.get("GoalFit", "")),
                    str(values.get("RiskFit", "")),
                    str(values.get("EvidenceGrade", "Evidence unavailable")),
                    str(values.get("EdgeStatus", "Insufficient Evidence")),
                    str(values.get("PersonalizedPlan", "")),
                    str(values.get("WhatToMonitor", "")),
                    str(values.get("RequiredBeforeAction", "")),
                    str(values.get("ReviewWhen", "")),
                    str(values.get("ExistingPositionGuidance", "")),
                    _number(values.get("SimulatedCapital")),
                    str(values.get("SavedAt") or saved_at),
                ),
            )
        connection.execute(
            "INSERT INTO audit_events (user_id, event_type, event_json, created_at) VALUES (?, ?, ?, ?)",
            (
                int(user_id),
                "PlanRunSaved",
                json.dumps({"run_id": run_id, "source": str(source), "asset_count": len(plans)}),
                saved_at,
            ),
        )
    return run_id


def load_latest_user_plan(
    user_id: int, db_path: str | Path = "data/app.db"
) -> pd.DataFrame:
    """Load the latest saved asset-plan rows for one demo user."""
    init_user_platform_db(db_path)
    with _connect(db_path) as connection:
        run = connection.execute(
            "SELECT id FROM user_plan_runs WHERE user_id = ? ORDER BY saved_at DESC, id DESC LIMIT 1",
            (int(user_id),),
        ).fetchone()
        if run is None:
            return pd.DataFrame(columns=PERSONALIZED_PLAN_COLUMNS)
        frame = pd.read_sql_query(
            """
            SELECT asset AS Asset, plan_type AS PlanType, direction AS Direction,
                   goal_fit AS GoalFit, risk_fit AS RiskFit, evidence_grade AS EvidenceGrade,
                   edge_status AS EdgeStatus, personalized_plan AS PersonalizedPlan,
                   what_to_monitor AS WhatToMonitor,
                   required_before_action AS RequiredBeforeAction,
                   review_when AS ReviewWhen,
                   existing_position_guidance AS ExistingPositionGuidance,
                   simulated_capital AS SimulatedCapital, saved_at AS SavedAt
            FROM user_asset_plans WHERE plan_run_id = ? ORDER BY id
            """,
            connection,
            params=(int(run["id"]),),
        )
    return frame[PERSONALIZED_PLAN_COLUMNS]


def list_user_plan_runs(
    user_id: int, db_path: str | Path = "data/app.db"
) -> pd.DataFrame:
    """List prior saved plan runs and their asset counts."""
    init_user_platform_db(db_path)
    with _connect(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT r.id AS PlanRunId, r.saved_at AS SavedAt, r.source AS Source,
                   COUNT(p.id) AS AssetCount
            FROM user_plan_runs r
            LEFT JOIN user_asset_plans p ON p.plan_run_id = r.id
            WHERE r.user_id = ?
            GROUP BY r.id, r.saved_at, r.source
            ORDER BY r.saved_at DESC, r.id DESC
            """,
            connection,
            params=(int(user_id),),
        )


__all__ = [
    "PLAN_TYPES",
    "PERSONALIZED_PLAN_COLUMNS",
    "init_user_platform_db",
    "get_or_create_demo_user",
    "apply_profile_defaults",
    "save_user_profile",
    "load_user_profile",
    "choose_personalized_plan_type",
    "personalize_asset_plans",
    "save_user_plan_run",
    "load_latest_user_plan",
    "list_user_plan_runs",
]
