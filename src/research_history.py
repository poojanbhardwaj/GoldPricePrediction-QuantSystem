"""User-owned research snapshot history and transparent change tracking."""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


HISTORY_COLUMNS = [
    "Asset", "LatestPrice", "PredictedPrice", "PredictedMovePct", "OpportunityScore",
    "CandidateCategory", "EdgeStatus", "PlanType", "RiskLabel", "CostVerdict",
    "SourceSnapshotDate", "SavedAt",
]

CHANGE_COLUMNS = [
    "Asset", "PreviousCategory", "LatestCategory", "CategoryChange",
    "PreviousPlanType", "LatestPlanType", "PlanChange", "PreviousMovePct",
    "LatestMovePct", "MoveChangePct", "PreviousOpportunityScore",
    "LatestOpportunityScore", "ScoreChange", "PreviousEdgeStatus",
    "LatestEdgeStatus", "EdgeChange", "ChangeSeverity", "Explanation",
]

_PLACEHOLDERS = {
    "", "none", "nan", "run research", "no saved estimate", "unlock forecast",
    "snapshot unavailable", "estimate unavailable", "score unavailable",
    "insufficient evidence", "evidence unavailable",
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value)
    except (TypeError, ValueError):
        return pd.DataFrame()


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(index=frame.index, dtype=object)


def _canonical_source(value: Any, mapping: Mapping[str, tuple[str, ...]]) -> pd.DataFrame:
    source = _frame(value)
    if source.empty or "Asset" not in source.columns:
        return pd.DataFrame(columns=["Asset", *mapping.keys()])
    source = source.dropna(subset=["Asset"]).copy()
    result = pd.DataFrame({"Asset": source["Asset"].astype(str).str.strip()})
    for output, candidates in mapping.items():
        result[output] = _first_column(source, candidates).values
    return result.drop_duplicates("Asset", keep="last")


def normalize_research_snapshot_for_history(
    prediction_snapshot: Any,
    watchlist: Any = None,
    edge_table: Any = None,
    personalized_plans: Any = None,
) -> pd.DataFrame:
    """Merge current research evidence into one stable row per asset."""
    prediction = _canonical_source(prediction_snapshot, {
        "LatestPrice": ("LatestPrice", "CurrentPrice", "EntryPrice"),
        "PredictedPrice": ("PredictedPrice", "ForecastPrice"),
        "PredictedMovePct": ("PredictedMovePct", "ExpectedMovePct", "ActiveEstimatePct"),
        "OpportunityScore": ("OpportunityScore", "Score"),
        "CandidateCategory": ("CandidateCategory", "Category", "Status"),
        "RiskLabel": ("RiskLabel", "Risk", "Status"),
        "SourceSnapshotDate": ("SourceSnapshotDate", "SnapshotDate", "LatestDate", "AsOfDate"),
    })
    watch = _canonical_source(watchlist, {
        "OpportunityScore": ("OpportunityScore", "Score"),
        "CandidateCategory": ("CandidateCategory", "Category", "Status"),
        "RiskLabel": ("RiskLabel", "Risk", "Status"),
    })
    edge = _canonical_source(edge_table, {
        "EdgeStatus": ("EdgeStatus", "Status", "Verdict"),
        "CostVerdict": ("CostVerdict", "CostStatus"),
    })
    plans = _canonical_source(personalized_plans, {
        "PlanType": ("PlanType", "ResearchAction", "Decision"),
        "RiskLabel": ("RiskLabel", "RiskFit"),
        "CostVerdict": ("CostVerdict",),
    })

    assets: set[str] = set()
    for frame in (prediction, watch, edge, plans):
        if not frame.empty:
            assets.update(frame["Asset"].dropna().astype(str))
    if not assets:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    output = pd.DataFrame({"Asset": sorted(assets)})
    for source in (prediction, watch, edge, plans):
        if source.empty:
            continue
        source = source.set_index("Asset")
        for column in source.columns:
            values = output["Asset"].map(source[column])
            if column not in output.columns:
                output[column] = values
            else:
                output[column] = values.where(values.notna(), output[column])
    for column in HISTORY_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    for column in ("LatestPrice", "PredictedPrice", "PredictedMovePct", "OpportunityScore"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["SavedAt"] = _now()
    return output[HISTORY_COLUMNS].reset_index(drop=True)


def init_research_history_tables(db_path: str | Path = "data/app.db") -> None:
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_history_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                run_label TEXT NOT NULL,
                created_at TEXT NOT NULL,
                asset_count INTEGER NOT NULL,
                source TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_history_assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                asset TEXT NOT NULL,
                latest_price REAL,
                predicted_price REAL,
                predicted_move_pct REAL,
                opportunity_score REAL,
                candidate_category TEXT,
                edge_status TEXT,
                plan_type TEXT,
                risk_label TEXT,
                cost_verdict TEXT,
                source_snapshot_date TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES research_history_runs(id)
            );
            CREATE INDEX IF NOT EXISTS idx_research_history_runs_user
                ON research_history_runs(user_id, created_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_research_history_assets_run
                ON research_history_assets(run_id, user_id);
            """
        )


def _meaningful_text(value: Any) -> bool:
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return str(value or "").strip().casefold() not in _PLACEHOLDERS


def _has_research_evidence(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    numeric = frame[["PredictedPrice", "PredictedMovePct", "OpportunityScore"]].notna().any(axis=1)
    textual = pd.Series(False, index=frame.index)
    for column in ("CandidateCategory", "EdgeStatus", "PlanType", "RiskLabel", "CostVerdict"):
        textual |= frame[column].map(_meaningful_text)
    return bool((numeric | textual).any())


def save_research_history_run(
    user_id: int | None,
    snapshot: Any,
    run_label: str = "Personalized research plan",
    source: str = "app",
    db_path: str | Path = "data/app.db",
) -> int:
    """Save an immutable research run; placeholder-only snapshots are ignored."""
    if user_id is None:
        raise ValueError("An unlocked application user is required to save research history")
    frame = _frame(snapshot)
    if not set(HISTORY_COLUMNS).issubset(frame.columns):
        frame = normalize_research_snapshot_for_history(frame)
    else:
        frame = frame[HISTORY_COLUMNS].copy()
    if not _has_research_evidence(frame):
        return 0
    init_research_history_tables(db_path)
    created_at = _now()
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO research_history_runs (user_id, run_label, created_at, asset_count, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (int(user_id), str(run_label), created_at, int(len(frame)), str(source)),
        )
        run_id = int(cursor.lastrowid)
        for _, row in frame.iterrows():
            connection.execute(
                """
                INSERT INTO research_history_assets (
                    run_id, user_id, asset, latest_price, predicted_price,
                    predicted_move_pct, opportunity_score, candidate_category,
                    edge_status, plan_type, risk_label, cost_verdict,
                    source_snapshot_date, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, int(user_id), str(row["Asset"]),
                    _nullable_number(row["LatestPrice"]), _nullable_number(row["PredictedPrice"]),
                    _nullable_number(row["PredictedMovePct"]), _nullable_number(row["OpportunityScore"]),
                    _nullable_text(row["CandidateCategory"]), _nullable_text(row["EdgeStatus"]),
                    _nullable_text(row["PlanType"]), _nullable_text(row["RiskLabel"]),
                    _nullable_text(row["CostVerdict"]), _nullable_text(row["SourceSnapshotDate"]),
                    created_at,
                ),
            )
    return run_id


def _nullable_number(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def _nullable_text(value: Any) -> str | None:
    return None if pd.isna(value) else str(value)


def load_research_history_runs(
    user_id: int | None, db_path: str | Path = "data/app.db"
) -> pd.DataFrame:
    columns = ["RunId", "RunLabel", "CreatedAt", "AssetCount", "Source"]
    if user_id is None:
        return pd.DataFrame(columns=columns)
    init_research_history_tables(db_path)
    with _connect(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT id AS RunId, run_label AS RunLabel, created_at AS CreatedAt,
                   asset_count AS AssetCount, source AS Source
            FROM research_history_runs WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            connection,
            params=(int(user_id),),
        )


def _load_history_position(
    user_id: int | None, offset: int, db_path: str | Path
) -> pd.DataFrame:
    if user_id is None:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    init_research_history_tables(db_path)
    with _connect(db_path) as connection:
        run = connection.execute(
            """
            SELECT id FROM research_history_runs WHERE user_id = ?
            ORDER BY created_at DESC, id DESC LIMIT 1 OFFSET ?
            """,
            (int(user_id), int(offset)),
        ).fetchone()
        if run is None:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        frame = pd.read_sql_query(
            """
            SELECT asset AS Asset, latest_price AS LatestPrice,
                   predicted_price AS PredictedPrice, predicted_move_pct AS PredictedMovePct,
                   opportunity_score AS OpportunityScore,
                   candidate_category AS CandidateCategory, edge_status AS EdgeStatus,
                   plan_type AS PlanType, risk_label AS RiskLabel,
                   cost_verdict AS CostVerdict, source_snapshot_date AS SourceSnapshotDate,
                   created_at AS SavedAt
            FROM research_history_assets WHERE run_id = ? AND user_id = ? ORDER BY id
            """,
            connection,
            params=(int(run["id"]), int(user_id)),
        )
    return frame[HISTORY_COLUMNS]


def load_latest_research_history_run(
    user_id: int | None, db_path: str | Path = "data/app.db"
) -> pd.DataFrame:
    return _load_history_position(user_id, 0, db_path)


def load_previous_research_history_run(
    user_id: int | None, db_path: str | Path = "data/app.db"
) -> pd.DataFrame:
    return _load_history_position(user_id, 1, db_path)


def _rank(value: Any, kind: str) -> int:
    text = str(value or "").casefold()
    maps = {
        "category": (("high potential", 5), ("candidate", 4), ("paper", 4), ("watch", 3), ("neutral", 2), ("avoid", 0), ("reject", 0)),
        "plan": (("confirmation", 5), ("paper", 4), ("watch", 3), ("hold existing", 2), ("learn", 1), ("blocked", 0), ("avoid", 0)),
        "edge": (("supported", 5), ("promising", 4), ("research", 3), ("weak", 2), ("insufficient", 1), ("blocked", 0)),
    }
    for token, rank in maps[kind]:
        if token in text:
            return rank
    return 1


def compare_research_history_runs(previous: Any, latest: Any) -> pd.DataFrame:
    """Compare two saved runs without modifying or re-scoring their evidence."""
    old = _frame(previous).set_index("Asset") if "Asset" in _frame(previous).columns else pd.DataFrame()
    new = _frame(latest).set_index("Asset") if "Asset" in _frame(latest).columns else pd.DataFrame()
    assets = sorted(set(old.index if not old.empty else []) | set(new.index if not new.empty else []))
    rows: list[dict[str, Any]] = []
    for asset in assets:
        if asset not in old.index:
            severity, explanation = "New Asset", "This asset appears for the first time in saved research history."
            before, after = {}, new.loc[asset].to_dict()
        elif asset not in new.index:
            severity, explanation = "Removed Asset", "This asset is absent from the latest saved research snapshot."
            before, after = old.loc[asset].to_dict(), {}
        else:
            before, after = old.loc[asset].to_dict(), new.loc[asset].to_dict()
            category_delta = _rank(after.get("CandidateCategory"), "category") - _rank(before.get("CandidateCategory"), "category")
            plan_delta = _rank(after.get("PlanType"), "plan") - _rank(before.get("PlanType"), "plan")
            edge_delta = _rank(after.get("EdgeStatus"), "edge") - _rank(before.get("EdgeStatus"), "edge")
            score_delta = (_nullable_number(after.get("OpportunityScore")) or 0) - (_nullable_number(before.get("OpportunityScore")) or 0)
            signal = category_delta + plan_delta + edge_delta
            if signal >= 4 or score_delta >= 20:
                severity = "Major Upgrade"
            elif signal > 0 or score_delta >= 5:
                severity = "Upgrade"
            elif signal <= -4 or score_delta <= -20:
                severity = "Major Downgrade"
            elif signal < 0 or score_delta <= -5:
                severity = "Downgrade"
            else:
                severity = "No Major Change"
            explanation = f"Category, plan, edge, and saved opportunity evidence imply: {severity}."
        previous_move = _nullable_number(before.get("PredictedMovePct"))
        latest_move = _nullable_number(after.get("PredictedMovePct"))
        previous_score = _nullable_number(before.get("OpportunityScore"))
        latest_score = _nullable_number(after.get("OpportunityScore"))
        rows.append({
            "Asset": asset,
            "PreviousCategory": before.get("CandidateCategory"), "LatestCategory": after.get("CandidateCategory"),
            "CategoryChange": f"{before.get('CandidateCategory', 'Unavailable')} -> {after.get('CandidateCategory', 'Unavailable')}",
            "PreviousPlanType": before.get("PlanType"), "LatestPlanType": after.get("PlanType"),
            "PlanChange": f"{before.get('PlanType', 'Unavailable')} -> {after.get('PlanType', 'Unavailable')}",
            "PreviousMovePct": previous_move, "LatestMovePct": latest_move,
            "MoveChangePct": None if previous_move is None or latest_move is None else latest_move - previous_move,
            "PreviousOpportunityScore": previous_score, "LatestOpportunityScore": latest_score,
            "ScoreChange": None if previous_score is None or latest_score is None else latest_score - previous_score,
            "PreviousEdgeStatus": before.get("EdgeStatus"), "LatestEdgeStatus": after.get("EdgeStatus"),
            "EdgeChange": f"{before.get('EdgeStatus', 'Unavailable')} -> {after.get('EdgeStatus', 'Unavailable')}",
            "ChangeSeverity": severity, "Explanation": explanation,
        })
    return pd.DataFrame(rows, columns=CHANGE_COLUMNS)


def summarize_research_changes(changes: Any) -> dict[str, int]:
    frame = _frame(changes)
    severity = frame.get("ChangeSeverity", pd.Series(dtype=str)).astype(str)
    return {
        "Upgrades": int(severity.isin(["Upgrade", "Major Upgrade", "New Asset"]).sum()),
        "Downgrades": int(severity.isin(["Downgrade", "Major Downgrade", "Removed Asset"]).sum()),
        "MajorChanges": int(severity.isin(["Major Upgrade", "Major Downgrade", "New Asset", "Removed Asset"]).sum()),
        "StableAssets": int(severity.eq("No Major Change").sum()),
    }


__all__ = [
    "HISTORY_COLUMNS", "CHANGE_COLUMNS", "normalize_research_snapshot_for_history",
    "init_research_history_tables", "save_research_history_run", "load_research_history_runs",
    "load_latest_research_history_run", "load_previous_research_history_run",
    "compare_research_history_runs", "summarize_research_changes",
]
