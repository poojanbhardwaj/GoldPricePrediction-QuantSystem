from __future__ import annotations

import pandas as pd

from src.candidate_watchlist import (
    build_candidate_watchlist,
    format_watchlist_label,
    summarize_watchlist,
)


def _row(asset: str, move, price=100.0, score=50, status="Medium Risk", cost="CostsManageable", horizon=10):
    return {
        "Asset": asset,
        "PredictedMovePct": move,
        "PredictedPrice": price,
        "OpportunityScore": score,
        "Status": status,
        "CostVerdict": cost,
        "BestHorizon": horizon,
    }


def test_actionable_candidate_requires_score_cost_risk_and_move():
    predictions = pd.DataFrame([
        _row("Gold", 1.2, score=72, status="Medium Risk", cost="CostsManageable"),
    ])

    watchlist = build_candidate_watchlist(predictions)

    assert watchlist.loc[0, "Asset"] == "Gold"
    assert watchlist.loc[0, "Category"] == "Actionable Candidate"
    assert watchlist.loc[0, "Direction"] == "Bullish"


def test_high_risk_bullish_asset_is_watchlist_not_actionable():
    predictions = pd.DataFrame([
        _row("Silver", 2.4, score=85, status="High Risk", cost="CostsManageable"),
    ])

    watchlist = build_candidate_watchlist(predictions)

    assert watchlist.loc[0, "Category"] == "Watchlist Candidate"
    assert "High Risk" in watchlist.loc[0, "Reason"]


def test_negative_prediction_becomes_bearish_watchlist_not_actionable_short():
    predictions = pd.DataFrame([
        _row("Crude Oil", -1.1, score=75, status="Medium Risk", cost="CostsManageable"),
    ])

    watchlist = build_candidate_watchlist(predictions)

    assert watchlist.loc[0, "Category"] == "Bearish Watchlist"
    assert watchlist.loc[0, "Direction"] == "Bearish"
    assert "not a trading recommendation" in watchlist.loc[0, "Reason"]


def test_low_score_or_excessive_cost_is_blocked():
    predictions = pd.DataFrame([
        _row("Bitcoin", 1.0, score=35, status="Medium Risk", cost="CostsManageable"),
        _row("Gold ETF", 1.0, score=80, status="Medium Risk", cost="CostsTooHighForSignal"),
    ])

    watchlist = build_candidate_watchlist(predictions)
    categories = dict(zip(watchlist["Asset"], watchlist["Category"]))

    assert categories["Bitcoin"] == "Avoid / Blocked"
    assert categories["Gold ETF"] == "Avoid / Blocked"


def test_missing_prediction_is_insufficient_evidence():
    predictions = pd.DataFrame([
        _row("S&P 500", None, price=None, score=80, status="Medium Risk", cost="CostsManageable"),
    ])

    watchlist = build_candidate_watchlist(predictions)

    assert watchlist.loc[0, "Category"] == "Insufficient Evidence"
    assert watchlist.loc[0, "Direction"] == "Neutral"


def test_sorting_prioritizes_actionable_then_watchlist_then_bearish_then_blocked():
    predictions = pd.DataFrame([
        _row("Blocked", 0.1, score=90, status="Medium Risk", cost="CostsManageable"),
        _row("Bearish", -1.2, score=80, status="Medium Risk", cost="CostsManageable"),
        _row("Watch", 1.5, score=80, status="High Risk", cost="CostsManageable"),
        _row("Action", 1.0, score=70, status="Medium Risk", cost="CostsManageable"),
    ])

    watchlist = build_candidate_watchlist(predictions)

    assert list(watchlist["Category"]) == [
        "Actionable Candidate",
        "Watchlist Candidate",
        "Bearish Watchlist",
        "Avoid / Blocked",
    ]
    assert list(watchlist["PriorityRank"]) == [1, 2, 3, 4]


def test_summary_counts_and_top_candidate_are_correct():
    predictions = pd.DataFrame([
        _row("Gold", 1.0, score=70, status="Medium Risk", cost="CostsManageable"),
        _row("Silver", 2.0, score=80, status="High Risk", cost="CostsManageable"),
        _row("Oil", -1.0, score=65, status="Medium Risk", cost="CostsManageable"),
        _row("BTC", 0.1, score=30, status="Medium Risk", cost="CostsManageable"),
    ])

    watchlist = build_candidate_watchlist(predictions)
    summary = summarize_watchlist(watchlist)

    assert summary["total_assets"] == 4
    assert summary["actionable_count"] == 1
    assert summary["watchlist_count"] == 1
    assert summary["bearish_watchlist_count"] == 1
    assert summary["blocked_count"] == 1
    assert summary["top_candidate_asset"] == "Gold"


def test_display_labels_are_human_readable():
    assert format_watchlist_label("MissingEstimate") == "Estimate unavailable"
    assert format_watchlist_label("Not Enough Evidence") == "Insufficient evidence"
    assert format_watchlist_label("ExpectedDelay") == "Recent data delay"
    assert format_watchlist_label("CostsTooHighForSignal") == "Costs too high for signal"
