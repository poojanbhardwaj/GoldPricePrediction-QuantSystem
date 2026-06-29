from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.app_context import get_asset_target
from src.cost_aware_plan import default_cost_assumptions, generate_cost_aware_asset_plan
from src.final_user_dashboard import resolve_horizon_estimates, set_plan_navigation_state


FORBIDDEN = re.compile(
    r"\b(Buy|Strong Buy|Sell|Hold|Invest Now|Guaranteed Profit|Safe Profit|Production Ready Trading)\b",
    flags=re.IGNORECASE,
)


def test_cost_plan_computes_net_active_when_predicted_move_exists():
    plan = generate_cost_aware_asset_plan(
        {"Asset": "Silver", "Horizon": 5, "PredictedMovePct": 2.4},
        amount=10000,
        cost_assumptions=default_cost_assumptions("Silver"),
    )

    assert plan["GrossActiveEstimatePct"] == 2.4
    assert plan["NetActiveEstimatePct"] is not None
    assert plan["NetActiveEstimatePct"] < plan["GrossActiveEstimatePct"]
    assert plan["CostDragPct"] > 0
    assert plan["BreakEvenReturnPct"] > 0


def test_cost_plan_derives_move_from_predicted_and_latest_prices():
    plan = generate_cost_aware_asset_plan(
        {"Asset": "Gold", "Horizon": 5, "LatestPrice": 100.0, "PredictedPrice": 103.0},
        amount=10000,
        cost_assumptions=default_cost_assumptions("Gold"),
    )

    assert plan["PredictedMovePct"] == 3.0
    assert plan["GrossActiveEstimatePct"] == 3.0
    assert plan["NetActiveEstimatePct"] is not None


def test_missing_active_estimate_has_clear_missing_estimate_message():
    plan = generate_cost_aware_asset_plan(
        {"Asset": "Bitcoin", "Horizon": 10},
        amount=10000,
        cost_assumptions=default_cost_assumptions("Bitcoin"),
    )

    assert plan["CostVerdict"] == "MissingEstimate"
    assert plan["EstimateComparisonStatus"] == "MissingEstimate"
    assert "Run Full Research first to generate an active estimate" in plan["ActiveEstimateExplanation"]
    assert "No saved estimate for this horizon yet" in plan["ActiveEstimateExplanation"]
    assert plan["CostDragPct"] > 0
    assert plan["BreakEvenReturnPct"] > 0


def test_missing_passive_estimate_keeps_active_net_and_explains_benchmark():
    plan = generate_cost_aware_asset_plan(
        {"Asset": "S&P 500", "Horizon": 20, "PredictedMovePct": 1.5},
        amount=10000,
        cost_assumptions=default_cost_assumptions("S&P 500"),
    )

    assert plan["NetActiveEstimatePct"] is not None
    assert plan["NetPassiveEstimatePct"] is None
    assert plan["EstimateComparisonStatus"] == "MissingEstimate"
    assert plan["PassiveEstimateExplanation"] == (
        "No passive benchmark estimate is available for this horizon yet. "
        "The benchmark is still shown as a comparison reference."
    )
    assert plan["CostComparisonExplanation"] == (
        "Cost comparison will appear after active/passive estimates are available."
    )


def test_exact_horizon_resolver_uses_saved_move_and_market_reference():
    research = pd.DataFrame([
        {"Asset": "Silver", "Horizon": 5, "Metric": "PredictedMovePct", "Value": 2.25},
        {"Asset": "Silver", "Horizon": 10, "Metric": "PredictedMovePct", "Value": 9.5},
    ])
    dates = pd.bdate_range("2026-01-01", periods=15)
    market = pd.DataFrame(
        {get_asset_target("Silver"): 30.0 * (1.0 + np.linspace(0.0, 0.03, len(dates)))},
        index=dates,
    )

    estimates = resolve_horizon_estimates("Silver", 5, research, market)

    assert estimates["GrossActiveEstimatePct"] == 2.25
    assert estimates["PredictedMovePct"] == 2.25
    assert estimates["GrossPassiveEstimatePct"] is not None
    assert estimates["PredictedPrice"] > estimates["LatestPrice"]


def test_plan_navigation_state_sets_asset_horizon_and_target_page():
    cost_state: dict[str, object] = {}
    set_plan_navigation_state(cost_state, "Silver", "Cost-Aware Plan", 5)
    assert cost_state["primary_product_navigation"] == "Cost-Aware Plan"
    assert cost_state["phase29_cost_asset"] == "Silver"
    assert cost_state["phase29_cost_horizon"] == 5
    assert cost_state["selected_asset"] == "Silver"

    asset_state: dict[str, object] = {}
    set_plan_navigation_state(asset_state, "Bitcoin", "Asset Plans", 10)
    assert asset_state["primary_product_navigation"] == "Asset Plans"
    assert asset_state["phase26_asset_plan_focus"] == "Bitcoin"

    journey_state: dict[str, object] = {}
    set_plan_navigation_state(journey_state, "Gold ETF", "Paper Research Journey", 30)
    assert journey_state["primary_product_navigation"] == "Paper Research Journey"
    assert journey_state["phase28_asset"] == "Gold ETF"
    assert journey_state["phase28_horizon"] == 30


def test_app_cost_page_has_helpful_states_and_navigation_rerun():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(source)
    cost_page = source.split('elif page == "Cost-Aware Plan":', 1)[1].split(
        'elif page == "Portfolio Summary":', 1
    )[0]

    assert "resolve_horizon_estimates" in cost_page
    assert "Run Full Research first" in cost_page
    assert "No forward estimate" in cost_page
    assert '"Not available"' not in cost_page
    assert "set_plan_navigation_state" in source
    assert "st.rerun()" in source


def test_phase29b_outputs_and_user_facing_source_avoid_forbidden_phrases():
    plan = generate_cost_aware_asset_plan(
        {"Asset": "Crude Oil", "Horizon": 5, "PredictedMovePct": 1.0},
        amount=10000,
        cost_assumptions=default_cost_assumptions("Crude Oil"),
    )
    text = " ".join(str(value) for value in plan.values())
    assert FORBIDDEN.search(text) is None
