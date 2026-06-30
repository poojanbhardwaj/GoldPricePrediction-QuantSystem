from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_watchlist import (
    WATCHLIST_COLUMNS,
    build_candidate_watchlist,
    format_watchlist_label,
    summarize_watchlist,
)


FORBIDDEN = re.compile(
    r"\b(Buy Now|Guaranteed Edge|Guaranteed Profit|Safe Profit|Production Ready Trading)\b",
    flags=re.IGNORECASE,
)


def _row(
    asset: str,
    move: float | None,
    score: float,
    status: str = "Watch",
    cost: str = "CostsManageable",
    price: float | None = 100.0,
) -> dict:
    return {
        "Asset": asset,
        "PredictedMovePct": move,
        "PredictedPrice": price,
        "BestHorizon": 5,
        "OpportunityScore": score,
        "Status": status,
        "CostVerdict": cost,
    }


def test_bullish_high_score_manageable_cost_is_actionable_candidate():
    result = build_candidate_watchlist(pd.DataFrame([
        _row("Gold", 1.2, 72.0, status="Watch", cost="CostsManageable"),
    ]))

    assert result.iloc[0]["Category"] == "Actionable Candidate"
    assert result.iloc[0]["Direction"] == "Bullish"
    assert "confirmation" in result.iloc[0]["Reason"].casefold()


def test_bullish_high_risk_prediction_remains_watchlist_candidate():
    result = build_candidate_watchlist(pd.DataFrame([
        _row("Silver", 1.8, 70.0, status="High Risk", cost="CostsLow"),
    ]))

    assert result.iloc[0]["Category"] == "Watchlist Candidate"
    assert "high risk" in result.iloc[0]["Reason"].casefold()


def test_negative_prediction_is_bearish_watchlist_not_actionable_short():
    result = build_candidate_watchlist(pd.DataFrame([
        _row("Crude Oil", -2.5, 75.0, status="Watch", cost="CostsManageable"),
    ]))

    assert result.iloc[0]["Category"] == "Bearish Watchlist"
    assert result.iloc[0]["Direction"] == "Bearish"
    assert "not a trading recommendation" in result.iloc[0]["Reason"].casefold()


def test_low_score_or_excessive_cost_is_avoid_blocked():
    snapshot = pd.DataFrame([
        _row("Bitcoin", 1.0, 35.0, cost="CostsManageable"),
        _row("S&P 500", 0.9, 70.0, cost="CostsTooHighForSignal"),
    ])
    result = build_candidate_watchlist(snapshot).set_index("Asset")

    assert result.loc["Bitcoin", "Category"] == "Avoid / Blocked"
    assert result.loc["S&P 500", "Category"] == "Avoid / Blocked"
    assert result.loc["S&P 500", "CostVerdict"] == "Costs too high for signal"


def test_missing_prediction_is_insufficient_evidence():
    result = build_candidate_watchlist(pd.DataFrame([
        _row("Gold ETF", None, 60.0, price=None, cost="MissingEstimate"),
    ]))
    row = result.iloc[0]

    assert row["Category"] == "Insufficient Evidence"
    assert row["Direction"] == "Neutral"
    assert row["Reason"] == "No saved prediction estimate is available."
    assert row["CostVerdict"] == "Estimate unavailable"


def test_priority_sorting_follows_category_then_score_and_move_size():
    snapshot = pd.DataFrame([
        _row("Avoid", 0.1, 80.0),
        _row("Bearish", -1.0, 60.0),
        _row("Watch Lower", 0.7, 45.0, status="High Risk"),
        _row("Actionable", 0.8, 75.0),
        _row("Missing", None, 90.0, price=None),
        _row("Watch Higher", 1.2, 55.0, status="High Risk"),
    ])
    result = build_candidate_watchlist(snapshot)

    assert result["Asset"].tolist() == [
        "Actionable", "Watch Higher", "Watch Lower", "Bearish", "Avoid", "Missing",
    ]
    assert result["PriorityRank"].tolist() == list(range(1, 7))


def test_summary_counts_and_top_candidate_are_correct():
    result = build_candidate_watchlist(pd.DataFrame([
        _row("Actionable", 1.0, 80.0),
        _row("Watch", 0.8, 55.0, status="High Risk"),
        _row("Bearish", -0.9, 55.0),
        _row("Blocked", 0.1, 30.0),
        _row("Missing", None, 0.0, price=None),
    ]))
    summary = summarize_watchlist(result)

    assert summary == {
        "total_assets": 5,
        "actionable_count": 1,
        "watchlist_count": 1,
        "bearish_watchlist_count": 1,
        "blocked_count": 2,
        "top_candidate_asset": "Actionable",
        "top_candidate_category": "Actionable Candidate",
        "top_candidate_score": 80.0,
    }


def test_display_labels_and_output_schema_are_human_readable():
    assert format_watchlist_label("ExpectedDelay") == "Recent data delay"
    assert format_watchlist_label("MissingEstimate") == "Estimate unavailable"
    assert format_watchlist_label("Not Enough Evidence") == "Insufficient evidence"
    assert format_watchlist_label("CostsManageable") == "Costs manageable"
    assert format_watchlist_label("CostsHigh") == "Costs high"

    result = build_candidate_watchlist(pd.DataFrame([_row("Gold", 1.0, 70.0)]))
    assert result.columns.tolist() == WATCHLIST_COLUMNS
    assert FORBIDDEN.search(" ".join(result.astype(str).stack().tolist())) is None


def test_candidate_watchlist_is_a_first_class_primary_route():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(source)
    navigation = source.split("PRIMARY_PRODUCT_PAGES = [", 1)[1].split("]", 1)[0]

    assert navigation.index('"Market Research Assistant"') < navigation.index('"Candidate Watchlist"')
    assert navigation.index('"Candidate Watchlist"') < navigation.index('"Asset Plans"')
    assert 'elif page == "Candidate Watchlist":' in source
    assert "_render_candidate_watchlist_section(phase29_snapshot)" in source

