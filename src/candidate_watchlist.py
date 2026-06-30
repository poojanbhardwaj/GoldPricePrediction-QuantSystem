"""Conservative candidate ranking over existing saved research outputs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import pandas as pd


WATCHLIST_COLUMNS = [
    "Asset",
    "Category",
    "Direction",
    "PredictedMovePct",
    "PredictedPrice",
    "BestHorizon",
    "OpportunityScore",
    "Status",
    "CostVerdict",
    "PriorityRank",
    "Reason",
    "UpgradeTrigger",
    "AvoidTrigger",
]

CATEGORY_PRIORITY = {
    "Actionable Candidate": 0,
    "Watchlist Candidate": 1,
    "Bearish Watchlist": 2,
    "Avoid / Blocked": 3,
    "Insufficient Evidence": 4,
}

DISPLAY_LABELS = {
    "MissingEstimate": "Estimate unavailable",
    "Not Enough Evidence": "Insufficient evidence",
    "ExpectedDelay": "Recent data delay",
    "CostsTooHighForSignal": "Costs too high for signal",
    "CostsManageable": "Costs manageable",
    "CostsLow": "Costs low",
    "CostsHigh": "Costs high",
}

MANAGEABLE_COST_VERDICTS = {"costslow", "costsmanageable", "manageable", "low"}
WEAK_EVIDENCE_STATUSES = {"not enough evidence", "insufficient evidence", "data issue"}


def format_watchlist_label(value: Any) -> str:
    """Translate internal research enums without changing their stored values."""
    text = str(value if value is not None else "").strip()
    return DISPLAY_LABELS.get(text, text)


def _frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return pd.DataFrame(list(value))
    return pd.DataFrame()


def _number(value: Any) -> Optional[float]:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(result) else float(result)


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def _asset_order(frames: Iterable[pd.DataFrame]) -> list[str]:
    assets: list[str] = []
    for frame in frames:
        if frame.empty or "Asset" not in frame.columns:
            continue
        for asset in frame["Asset"].dropna().astype(str):
            if asset and asset not in assets:
                assets.append(asset)
    return assets


def _best_asset_row(frame: pd.DataFrame, asset: str, horizon: Optional[int] = None) -> dict[str, Any]:
    if frame.empty or "Asset" not in frame.columns:
        return {}
    rows = frame.loc[frame["Asset"].astype(str).eq(asset)].copy()
    if rows.empty:
        return {}
    if horizon is not None:
        horizon_values = pd.to_numeric(
            rows.get("BestHorizon", rows.get("Horizon", pd.Series(index=rows.index, dtype=float))),
            errors="coerce",
        )
        exact = rows.loc[horizon_values.eq(int(horizon))]
        if not exact.empty:
            rows = exact
    rows["_score"] = pd.to_numeric(
        rows.get("OpportunityScore", pd.Series(index=rows.index, dtype=float)), errors="coerce"
    ).fillna(-1.0)
    rows["_move"] = pd.to_numeric(
        rows.get("PredictedMovePct", pd.Series(index=rows.index, dtype=float)), errors="coerce"
    ).abs().fillna(-1.0)
    return rows.sort_values(["_score", "_move"], ascending=[False, False]).iloc[0].drop(
        labels=["_score", "_move"], errors="ignore"
    ).to_dict()


def _first_value(rows: Iterable[Mapping[str, Any]], *fields: str) -> Any:
    for field in fields:
        for row in rows:
            value = row.get(field)
            if _present(value):
                return value
    return None


def _direction(predicted_move: Optional[float]) -> str:
    if predicted_move is None:
        return "Neutral"
    if predicted_move >= 0.35:
        return "Bullish"
    if predicted_move <= -0.35:
        return "Bearish"
    return "Neutral"


def _classify_candidate(
    *,
    prediction_available: bool,
    direction: str,
    predicted_move: Optional[float],
    score: float,
    status: str,
    cost_verdict: str,
    saved_upgrade: str,
    saved_avoid: str,
) -> tuple[str, str, str, str]:
    status_key = status.casefold().strip()
    cost_key = cost_verdict.casefold().strip()
    manageable_cost = cost_key in MANAGEABLE_COST_VERDICTS
    excessive_cost = cost_key == "coststoohighforsignal"
    high_risk = status_key == "high risk"
    weak_evidence = status_key in WEAK_EVIDENCE_STATUSES
    move_size = abs(predicted_move) if predicted_move is not None else 0.0

    if not prediction_available:
        return (
            "Insufficient Evidence",
            "No saved prediction estimate is available.",
            "Run full research or provide valid prediction evidence.",
            "Avoid using this asset until prediction evidence exists.",
        )

    blockers: list[str] = []
    if score < 40:
        blockers.append("opportunity score is below 40")
    if excessive_cost:
        blockers.append("modeled costs are too high for the saved signal")
    if weak_evidence:
        blockers.append("saved evidence is insufficient")
    if direction == "Neutral":
        blockers.append("the predicted move is below the 0.35% direction threshold")
    if blockers:
        return (
            "Avoid / Blocked",
            "Blocked because " + "; ".join(blockers) + ".",
            saved_upgrade or "Improve prediction strength, evidence quality, score, and cost viability.",
            saved_avoid or "Keep blocked while any current blocker remains.",
        )

    if direction == "Bearish":
        return (
            "Bearish Watchlist",
            "The saved estimate points lower; this is a downside watch candidate, not a trading recommendation.",
            saved_upgrade or "Require repeated downside evidence, controlled risk, and benchmark confirmation.",
            saved_avoid or "Remove from the downside watchlist if the move weakens or evidence becomes unstable.",
        )

    if score >= 65 and manageable_cost and not high_risk and move_size >= 0.35:
        return (
            "Actionable Candidate",
            "The saved bullish estimate, opportunity score, and modeled costs pass the research candidate gates; confirmation is still required.",
            saved_upgrade or "Confirm the estimate with fresh data, risk checks, and forward paper evidence.",
            saved_avoid or "Block if risk becomes high, costs become excessive, or the predicted move falls below 0.35%.",
        )

    watch_blockers = []
    if high_risk:
        watch_blockers.append("High Risk status")
    if score < 65:
        watch_blockers.append("score below the actionable threshold")
    if not manageable_cost:
        watch_blockers.append("cost evidence is high or uncertain")
    reason = "Bullish saved evidence merits monitoring, but " + ", ".join(watch_blockers) + " blocks actionable classification."
    return (
        "Watchlist Candidate",
        reason,
        saved_upgrade or "Reduce the listed blockers and confirm the estimate with forward paper evidence.",
        saved_avoid or "Keep blocked if risk, cost, score, or prediction evidence deteriorates.",
    )


def build_candidate_watchlist(
    prediction_snapshot: pd.DataFrame,
    cost_plans: pd.DataFrame | None = None,
    final_user_plans: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one transparent, conservatively classified watchlist row per asset."""
    predictions = _frame(prediction_snapshot)
    costs = _frame(cost_plans)
    plans = _frame(final_user_plans)
    assets = _asset_order((predictions, costs, plans))
    if not assets:
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)

    output: list[dict[str, Any]] = []
    for asset in assets:
        prediction_row = _best_asset_row(predictions, asset)
        horizon = _number(prediction_row.get("BestHorizon", prediction_row.get("Horizon")))
        cost_row = _best_asset_row(costs, asset, int(horizon) if horizon is not None else None)
        plan_row = _best_asset_row(plans, asset, int(horizon) if horizon is not None else None)
        source_rows = (prediction_row, cost_row, plan_row)

        predicted_move = _number(_first_value(source_rows, "PredictedMovePct", "GrossActiveEstimatePct"))
        predicted_price = _number(_first_value(source_rows, "PredictedPrice"))
        score = _number(_first_value(source_rows, "OpportunityScore"))
        score = max(0.0, min(100.0, score if score is not None else 0.0))
        raw_status = str(_first_value((prediction_row, plan_row, cost_row), "Status") or "Not Enough Evidence")
        raw_cost = str(_first_value((cost_row, prediction_row, plan_row), "CostVerdict") or "MissingEstimate")
        saved_upgrade = str(_first_value((plan_row, prediction_row, cost_row), "WhatMustImprove", "ImprovementNeeded") or "")
        saved_avoid = str(_first_value((plan_row, prediction_row, cost_row), "InvalidationCondition", "AvoidTrigger") or "")
        direction = _direction(predicted_move)
        prediction_available = predicted_move is not None or predicted_price is not None
        category, reason, upgrade_trigger, avoid_trigger = _classify_candidate(
            prediction_available=prediction_available,
            direction=direction,
            predicted_move=predicted_move,
            score=score,
            status=raw_status,
            cost_verdict=raw_cost,
            saved_upgrade=saved_upgrade,
            saved_avoid=saved_avoid,
        )
        output.append({
            "Asset": asset,
            "Category": category,
            "Direction": direction,
            "PredictedMovePct": predicted_move,
            "PredictedPrice": predicted_price,
            "BestHorizon": int(horizon) if horizon is not None else None,
            "OpportunityScore": round(score, 2),
            "Status": format_watchlist_label(raw_status),
            "CostVerdict": format_watchlist_label(raw_cost),
            "Reason": reason,
            "UpgradeTrigger": upgrade_trigger,
            "AvoidTrigger": avoid_trigger,
        })

    result = pd.DataFrame(output)
    result["_category"] = result["Category"].map(CATEGORY_PRIORITY).fillna(99)
    result["_move"] = pd.to_numeric(result["PredictedMovePct"], errors="coerce").abs().fillna(-1.0)
    result = result.sort_values(
        ["_category", "OpportunityScore", "_move", "Asset"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    result["PriorityRank"] = range(1, len(result) + 1)
    return result.drop(columns=["_category", "_move"])[WATCHLIST_COLUMNS]


def summarize_watchlist(watchlist: pd.DataFrame) -> dict[str, Any]:
    """Summarize category counts and the highest-ranked candidate."""
    frame = _frame(watchlist)
    if frame.empty:
        return {
            "total_assets": 0,
            "actionable_count": 0,
            "watchlist_count": 0,
            "bearish_watchlist_count": 0,
            "blocked_count": 0,
            "top_candidate_asset": None,
            "top_candidate_category": None,
            "top_candidate_score": None,
        }
    categories = frame.get("Category", pd.Series(index=frame.index, dtype=str)).astype(str)
    ranked = frame.sort_values("PriorityRank") if "PriorityRank" in frame.columns else frame
    top = ranked.iloc[0]
    return {
        "total_assets": int(len(frame)),
        "actionable_count": int(categories.eq("Actionable Candidate").sum()),
        "watchlist_count": int(categories.eq("Watchlist Candidate").sum()),
        "bearish_watchlist_count": int(categories.eq("Bearish Watchlist").sum()),
        "blocked_count": int(categories.isin(["Avoid / Blocked", "Insufficient Evidence"]).sum()),
        "top_candidate_asset": str(top.get("Asset", "")) or None,
        "top_candidate_category": str(top.get("Category", "")) or None,
        "top_candidate_score": _number(top.get("OpportunityScore")),
    }


__all__ = [
    "WATCHLIST_COLUMNS",
    "CATEGORY_PRIORITY",
    "DISPLAY_LABELS",
    "format_watchlist_label",
    "build_candidate_watchlist",
    "summarize_watchlist",
]
