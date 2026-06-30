"""Conservative evidence-of-edge aggregation over saved research tables."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from src.candidate_watchlist import format_watchlist_label


EDGE_EVIDENCE_COLUMNS = [
    "Asset",
    "Category",
    "Direction",
    "EdgeStatus",
    "EvidenceGrade",
    "OpportunityScore",
    "PredictedMovePct",
    "CostVerdict",
    "Status",
    "BenchmarkName",
    "ActiveEstimatePct",
    "PassiveEstimatePct",
    "ActiveMinusPassivePct",
    "CostDragPct",
    "WinRatePct",
    "Sharpe",
    "MaxDrawdownPct",
    "WalkForwardReturnPct",
    "TradeCount",
    "EvidenceSummary",
    "MainRisk",
    "RequiredBeforeAction",
]

ALIASES = {
    "BenchmarkName": ("PassiveBenchmarkName", "BenchmarkName"),
    "ActiveEstimatePct": (
        "NetActiveEstimatePct", "ActiveEstimatePct", "GrossActiveEstimatePct", "PredictedMovePct",
    ),
    "PassiveEstimatePct": (
        "NetPassiveEstimatePct", "PassiveEstimatePct", "GrossPassiveEstimatePct",
    ),
    "ActiveMinusPassivePct": (
        "ActiveMinusPassiveNetPct", "ActiveMinusPassivePct", "ExcessReturnPct",
    ),
    "CostDragPct": ("CostDragPct",),
    "WinRatePct": ("WinRatePct", "WinRate", "HitRatePct"),
    "Sharpe": ("Sharpe", "SharpeRatio"),
    "MaxDrawdownPct": ("MaxDrawdownPct", "MaxDrawdown"),
    "WalkForwardReturnPct": ("WalkForwardReturnPct", "TotalReturnPct", "StrategyReturnPct"),
    "TradeCount": ("TradeCount", "Trades", "NumTrades"),
}

VALIDATION_FIELDS = ("WinRatePct", "Sharpe", "MaxDrawdownPct", "WalkForwardReturnPct")


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


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _asset_order(frames: Iterable[pd.DataFrame]) -> list[str]:
    assets: list[str] = []
    for frame in frames:
        if frame.empty or "Asset" not in frame.columns:
            continue
        for asset in frame["Asset"].dropna().astype(str):
            if asset and asset not in {"ALL", "All"} and asset not in assets:
                assets.append(asset)
    return assets


def _best_row(frame: pd.DataFrame, asset: str, horizon: Optional[int]) -> dict[str, Any]:
    if frame.empty or "Asset" not in frame.columns:
        return {}
    rows = frame.loc[frame["Asset"].astype(str).eq(asset)].copy()
    if rows.empty:
        return {}
    if horizon is not None:
        horizons = pd.to_numeric(
            rows.get("BestHorizon", rows.get("Horizon", pd.Series(index=rows.index, dtype=float))),
            errors="coerce",
        )
        exact = rows.loc[horizons.eq(int(horizon))]
        if not exact.empty:
            rows = exact
    scores = pd.to_numeric(
        rows.get("OpportunityScore", pd.Series(index=rows.index, dtype=float)), errors="coerce"
    ).fillna(-1.0)
    return rows.loc[scores.sort_values(ascending=False).index[0]].to_dict()


def _first_value(rows: Sequence[Mapping[str, Any]], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        for row in rows:
            value = row.get(alias)
            if _present(value):
                return value
    return None


def _long_metric_value(
    research: pd.DataFrame,
    asset: str,
    horizon: Optional[int],
    aliases: Sequence[str],
) -> Optional[float]:
    if research.empty or not {"Metric", "Value"}.issubset(research.columns):
        return None
    assets = research.get("Asset", pd.Series("ALL", index=research.index)).astype(str)
    horizons = pd.to_numeric(
        research.get("Horizon", pd.Series(0, index=research.index)), errors="coerce"
    ).fillna(0).astype(int)
    metrics = research["Metric"].astype(str).map(_normalize)
    target_horizon = int(horizon) if horizon is not None else 0

    for alias in aliases:
        normalized_alias = _normalize(alias)
        for exact_metric in (True, False):
            metric_mask = metrics.eq(normalized_alias) if exact_metric else metrics.str.endswith(normalized_alias)
            for asset_value, horizon_value in (
                (asset, target_horizon), (asset, 0), ("ALL", target_horizon), ("ALL", 0),
                ("All", target_horizon), ("All", 0),
            ):
                matches = research.loc[
                    metric_mask & assets.eq(asset_value) & horizons.eq(horizon_value), "Value"
                ]
                values = pd.to_numeric(matches, errors="coerce").dropna()
                if not values.empty:
                    return float(values.iloc[-1])
    return None


def _numeric_evidence(
    rows: Sequence[Mapping[str, Any]],
    research: pd.DataFrame,
    asset: str,
    horizon: Optional[int],
    field: str,
) -> Optional[float]:
    value = _number(_first_value(rows, ALIASES[field]))
    if value is not None:
        return value
    return _long_metric_value(research, asset, horizon, ALIASES[field])


def classify_edge_status(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return edge status, evidence grade, and an honest evidence summary."""
    category = str(row.get("Category", "Insufficient Evidence"))
    status_key = _normalize(row.get("Status"))
    cost_key = _normalize(row.get("CostVerdict"))
    predicted_move = _number(row.get("PredictedMovePct"))
    active = _number(row.get("ActiveEstimatePct"))
    passive = _number(row.get("PassiveEstimatePct"))
    gap = _number(row.get("ActiveMinusPassivePct"))
    cost_drag = _number(row.get("CostDragPct"))
    validation_count = sum(_number(row.get(field)) is not None for field in VALIDATION_FIELDS)
    validation_available = validation_count > 0
    prediction_available = predicted_move is not None or _number(row.get("PredictedPrice")) is not None
    benchmark_available = gap is not None or active is not None or passive is not None
    meaningful_cost = cost_drag is not None or cost_key not in {"", "missingestimate", "estimateunavailable"}

    cost_blocked = cost_key == "coststoohighforsignal"
    if predicted_move is not None and cost_drag is not None and cost_drag > abs(predicted_move):
        cost_blocked = True
    if cost_blocked:
        return (
            "Cost Blocked",
            "F",
            "Modeled cost drag is too large for the saved prediction evidence.",
        )

    benchmark_weak = gap is not None and gap < 0
    if active is not None and passive is not None and passive > active:
        benchmark_weak = True
    if benchmark_weak:
        return (
            "Benchmark Weak",
            "D",
            "The passive comparison is stronger than the active saved estimate.",
        )

    no_evidence = not prediction_available and not benchmark_available and not meaningful_cost and not validation_available
    if no_evidence or category == "Insufficient Evidence":
        return (
            "Insufficient Evidence",
            "F",
            "Prediction, benchmark, cost, and validation evidence are unavailable.",
        )

    eligible_category = category in {"Actionable Candidate", "Watchlist Candidate"}
    if eligible_category and gap is not None and gap > 0 and status_key != "highrisk" and validation_available:
        grade = "A" if category == "Actionable Candidate" and validation_count >= 3 else "B"
        return (
            "Edge Supported",
            grade,
            "Positive active-versus-passive evidence remains after modeled costs and validation evidence is available.",
        )

    if prediction_available:
        grade = "C" if category in {"Actionable Candidate", "Watchlist Candidate", "Bearish Watchlist"} else "D"
        if not validation_available:
            summary = "The saved candidate is visible, but validation evidence is unavailable."
        elif status_key == "highrisk":
            summary = "The saved candidate is visible, but its High Risk status prevents stronger edge support."
        else:
            summary = "The saved candidate remains incomplete and requires additional benchmark and risk confirmation."
        return "Watch Only", grade, summary

    return (
        "Insufficient Evidence",
        "F",
        "Available fields do not establish a complete prediction and validation record.",
    )


def _required_before_action(edge_status: str, saved_requirement: Any) -> str:
    if _present(saved_requirement):
        return str(saved_requirement)
    defaults = {
        "Edge Supported": "Require repeated forward validation and stable risk evidence before any stronger research use.",
        "Watch Only": "Add validation metrics and confirm benchmark, cost, and risk behavior.",
        "Benchmark Weak": "Active evidence must exceed the passive comparison over matching dates.",
        "Cost Blocked": "The saved move must exceed realistic cost drag with room for uncertainty.",
        "Insufficient Evidence": "Add valid prediction, benchmark, cost, and validation evidence.",
    }
    return defaults.get(edge_status, "Add repeated validation evidence before stronger research use.")


def build_edge_evidence_table(
    watchlist: pd.DataFrame,
    prediction_snapshot: pd.DataFrame | None = None,
    cost_plans: pd.DataFrame | None = None,
    research_snapshot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one evidence-of-edge row per saved asset without inventing missing metrics."""
    watch = _frame(watchlist)
    predictions = _frame(prediction_snapshot)
    costs = _frame(cost_plans)
    research = _frame(research_snapshot)
    assets = _asset_order((watch, predictions, costs, research))
    if not assets:
        return pd.DataFrame(columns=EDGE_EVIDENCE_COLUMNS)

    output: list[dict[str, Any]] = []
    for asset in assets:
        watch_row = _best_row(watch, asset, None)
        horizon = _number(watch_row.get("BestHorizon"))
        prediction_row = _best_row(predictions, asset, int(horizon) if horizon is not None else None)
        if horizon is None:
            horizon = _number(prediction_row.get("BestHorizon", prediction_row.get("Horizon")))
        cost_row = _best_row(costs, asset, int(horizon) if horizon is not None else None)
        research_row = _best_row(research, asset, int(horizon) if horizon is not None else None)
        rows = (cost_row, prediction_row, watch_row, research_row)

        category = str(_first_value((watch_row, prediction_row), ("Category",)) or "Insufficient Evidence")
        direction = str(_first_value((watch_row, prediction_row), ("Direction",)) or "Neutral")
        opportunity_score = _number(_first_value((watch_row, prediction_row, cost_row), ("OpportunityScore",)))
        predicted_move = _number(_first_value((watch_row, prediction_row, cost_row), ("PredictedMovePct",)))
        predicted_price = _number(_first_value((prediction_row, watch_row), ("PredictedPrice",)))
        raw_cost = str(_first_value((cost_row, prediction_row, watch_row), ("CostVerdict",)) or "MissingEstimate")
        raw_status = str(_first_value((watch_row, prediction_row, cost_row), ("Status",)) or "Not Enough Evidence")
        benchmark_name = _first_value(rows, ALIASES["BenchmarkName"])
        main_risk = str(_first_value((prediction_row, cost_row, watch_row), ("MainRisk", "Reason")) or "Evidence risk is not documented.")
        saved_requirement = _first_value(
            (watch_row, prediction_row, cost_row),
            ("UpgradeTrigger", "RequiredBeforeAction", "WhatMustImprove"),
        )

        evidence = {
            field: _numeric_evidence(rows, research, asset, int(horizon) if horizon is not None else None, field)
            for field in ALIASES
            if field != "BenchmarkName"
        }
        if evidence["ActiveMinusPassivePct"] is None:
            active = evidence["ActiveEstimatePct"]
            passive = evidence["PassiveEstimatePct"]
            if active is not None and passive is not None:
                evidence["ActiveMinusPassivePct"] = active - passive

        classification_row = {
            "Category": category,
            "Status": raw_status,
            "CostVerdict": raw_cost,
            "PredictedMovePct": predicted_move,
            "PredictedPrice": predicted_price,
            **evidence,
        }
        edge_status, evidence_grade, evidence_summary = classify_edge_status(classification_row)
        output.append({
            "Asset": asset,
            "Category": category,
            "Direction": direction,
            "EdgeStatus": edge_status,
            "EvidenceGrade": evidence_grade,
            "OpportunityScore": opportunity_score,
            "PredictedMovePct": predicted_move,
            "CostVerdict": format_watchlist_label(raw_cost),
            "Status": format_watchlist_label(raw_status),
            "BenchmarkName": str(benchmark_name) if _present(benchmark_name) else "Evidence unavailable",
            **evidence,
            "EvidenceSummary": evidence_summary,
            "MainRisk": main_risk,
            "RequiredBeforeAction": _required_before_action(edge_status, saved_requirement),
        })

    result = pd.DataFrame(output)
    for column in EDGE_EVIDENCE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result[EDGE_EVIDENCE_COLUMNS]


def summarize_edge_evidence(edge_table: pd.DataFrame) -> dict[str, Any]:
    """Summarize edge classifications without converting missing evidence to zero."""
    frame = _frame(edge_table)
    if frame.empty:
        return {
            "total_assets": 0,
            "edge_supported_count": 0,
            "watch_only_count": 0,
            "insufficient_evidence_count": 0,
            "cost_blocked_count": 0,
            "top_edge_asset": None,
            "top_edge_grade": None,
        }
    statuses = frame.get("EdgeStatus", pd.Series(index=frame.index, dtype=str)).astype(str)
    supported = frame.loc[statuses.eq("Edge Supported")].copy()
    if supported.empty:
        top_asset = None
        top_grade = None
    else:
        grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
        supported["_grade"] = supported["EvidenceGrade"].map(grade_order).fillna(9)
        supported["_score"] = pd.to_numeric(supported.get("OpportunityScore"), errors="coerce").fillna(-1)
        top = supported.sort_values(["_grade", "_score"], ascending=[True, False]).iloc[0]
        top_asset = str(top.get("Asset", "")) or None
        top_grade = str(top.get("EvidenceGrade", "")) or None
    return {
        "total_assets": int(len(frame)),
        "edge_supported_count": int(statuses.eq("Edge Supported").sum()),
        "watch_only_count": int(statuses.eq("Watch Only").sum()),
        "insufficient_evidence_count": int(statuses.eq("Insufficient Evidence").sum()),
        "cost_blocked_count": int(statuses.eq("Cost Blocked").sum()),
        "top_edge_asset": top_asset,
        "top_edge_grade": top_grade,
    }


__all__ = [
    "EDGE_EVIDENCE_COLUMNS",
    "ALIASES",
    "classify_edge_status",
    "build_edge_evidence_table",
    "summarize_edge_evidence",
]
