from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.app_context import SUPPORTED_ASSETS, get_asset_target
from src.cost_aware_plan import (
    ALLOWED_COST_VERDICTS,
    calculate_break_even_return,
    calculate_round_trip_cost,
    compare_active_vs_passive_after_costs,
    default_cost_assumptions,
    estimate_net_return,
    explain_opportunity_score,
    generate_cost_aware_asset_plan,
)
from src.final_user_dashboard import (
    FINAL_SNAPSHOT_COLUMNS,
    build_all_asset_prediction_snapshot,
    calculate_asset_changes,
    generate_dashboard_summary,
    generate_final_user_plan,
    get_latest_asset_prices,
    save_final_user_dashboard_artifacts,
)
from src.paper_research_journey import build_passive_benchmark_guide
from src.ui_components import (
    render_active_vs_passive_card,
    render_asset_price_card,
    render_beginner_explanation_box,
    render_cost_assumption_inputs,
    render_cost_summary_card,
    render_market_snapshot_grid,
    render_prediction_snapshot_card,
    render_run_research_panel,
    render_score_explainer_card,
    render_simple_plan_card,
)


FORBIDDEN = re.compile(
    r"\b(Buy|Strong Buy|Sell|Hold|Invest Now|Guaranteed Profit|Safe Profit|Production Ready Trading)\b",
    flags=re.IGNORECASE,
)


def _market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2025-10-01", periods=80)
    starts = {
        "Gold": 2500.0, "Silver": 30.0, "Crude Oil": 70.0,
        "Bitcoin": 80000.0, "S&P 500": 5500.0, "Gold ETF": 230.0,
    }
    return pd.DataFrame({
        get_asset_target(asset): base * (1 + np.linspace(0, 0.06, len(dates)))
        for asset, base in starts.items()
    }, index=dates)


def _plans() -> pd.DataFrame:
    return pd.DataFrame([{
        "Asset": asset,
        "Horizon": 5,
        "Status": "Watch",
        "OpportunityScore": 62.0 - index,
        "OpportunityGrade": "C",
        "Confidence": "Moderate",
        "MainRisk": "Benchmark evidence remains weak.",
        "PositiveEvidence": "The saved direction estimate is constructive.",
        "NegativeEvidence": "Benchmark and cost evidence remain limited.",
        "BlockReason": "Benchmark Weakness",
        "WhatMustImprove": "Repeated net outcomes must compare better with the passive reference.",
        "WhatUserShouldMonitorNext": "Monitor costs, risk, and the same-period passive comparison.",
        "InvalidationCondition": "Risk warnings worsen or benchmark misses repeat.",
        "RecheckWhen": "After the next saved evidence refresh.",
        "PredictedMovePct": 1.25,
        "RealMoneyApproved": False,
    } for index, asset in enumerate(SUPPORTED_ASSETS)])


def _flatten(frame: pd.DataFrame) -> str:
    return " ".join(frame.astype(str).stack().tolist())


def test_phase29_modules_and_ui_helpers_import():
    helpers = [
        get_latest_asset_prices, build_all_asset_prediction_snapshot, default_cost_assumptions,
        calculate_round_trip_cost, render_asset_price_card, render_prediction_snapshot_card,
        render_cost_summary_card, render_score_explainer_card, render_active_vs_passive_card,
        render_simple_plan_card, render_run_research_panel, render_market_snapshot_grid,
        render_cost_assumption_inputs, render_beginner_explanation_box,
    ]
    assert all(callable(helper) for helper in helpers)


def test_latest_asset_price_snapshot_covers_all_assets():
    prices = get_latest_asset_prices(_market_data())
    assert set(prices["Asset"]) == set(SUPPORTED_ASSETS)
    assert prices["LatestPrice"].notna().all()
    assert prices["LatestPriceDate"].astype(str).str.len().gt(0).all()
    assert {"Change1D_pct", "Change5D_pct", "Change30D_pct", "DataFreshness"}.issubset(prices.columns)


def test_asset_changes_are_computed_from_the_selected_asset_column():
    market = _market_data()
    gold = calculate_asset_changes(market, "Gold")
    bitcoin = calculate_asset_changes(market, "Bitcoin")
    assert gold["LatestPrice"] != bitcoin["LatestPrice"]
    assert gold["PriceSource"] == "Latest available dataset price."


def test_all_asset_prediction_snapshot_has_required_columns():
    snapshot = build_all_asset_prediction_snapshot(
        _plans(), pd.DataFrame(), default_cost_assumptions(), _market_data()
    )
    assert set(snapshot["Asset"]) == set(SUPPORTED_ASSETS)
    assert set(FINAL_SNAPSHOT_COLUMNS).issubset(snapshot.columns)
    assert snapshot["PredictedPrice"].notna().all()
    assert snapshot["SimplePlan"].astype(str).str.len().gt(0).all()


def test_default_cost_assumptions_are_complete_and_editable_fields_exist():
    assumptions = default_cost_assumptions("Gold")
    required = {
        "EntryBrokerage", "ExitBrokerage", "EntrySpreadPct", "ExitSpreadPct",
        "EntrySlippagePct", "ExitSlippagePct", "PlatformFee", "TaxesAndChargesPct",
        "ExpenseRatioPct", "CurrencyConversionPct", "OtherCost", "Notes",
    }
    assert required.issubset(assumptions)


def test_round_trip_cost_and_break_even_are_calculated():
    assumptions = default_cost_assumptions()
    total = calculate_round_trip_cost(10000, assumptions)
    break_even = calculate_break_even_return(10000, total)
    assert total > 0
    assert break_even > 0
    assert round(total / 10000 * 100, 4) == break_even


def test_net_return_is_lower_than_gross_when_costs_exist():
    result = estimate_net_return(2.0, 10000, default_cost_assumptions())
    assert result["NetReturnPct"] < result["GrossReturnPct"]
    assert result["EstimatedRoundTripCost"] > 0


def test_active_vs_passive_after_costs_comparison_is_transparent():
    result = compare_active_vs_passive_after_costs(2.0, 3.0, 10000, default_cost_assumptions())
    assert result["NetPassiveEstimatePct"] > result["NetActiveEstimatePct"]
    assert result["ActiveMinusPassiveNetPct"] < 0
    assert "passive benchmark" in result["CostWarning"].casefold()


def test_high_costs_can_block_a_small_saved_estimate():
    assumptions = default_cost_assumptions()
    assumptions.update({"EntrySpreadPct": 2.0, "ExitSpreadPct": 2.0})
    result = compare_active_vs_passive_after_costs(1.0, 0.5, 10000, assumptions)
    assert result["CostVerdict"] == "CostsTooHighForSignal"
    assert result["CostVerdict"] in ALLOWED_COST_VERDICTS


def test_score_explanation_and_cost_aware_plan_are_plain_language():
    plan = _plans().iloc[0].to_dict()
    score = explain_opportunity_score(plan)
    enriched = generate_cost_aware_asset_plan(plan, amount=10000, cost_assumptions=default_cost_assumptions())
    assert score["ScoreMeaning"]
    assert score["ScoreReducedBy"]
    assert enriched["ScorePlainEnglishSummary"]
    assert enriched["PassiveBenchmarkName"]
    assert enriched["RealMoneyApproved"] is False
    assert enriched["BrokerExecutionAllowed"] is False


def test_simple_plan_exists_and_does_not_approve_real_money():
    snapshot = build_all_asset_prediction_snapshot(
        _plans(), pd.DataFrame(), default_cost_assumptions(), _market_data()
    )
    plan = generate_final_user_plan(snapshot.iloc[0].to_dict())
    assert plan
    assert "paper research" in plan.casefold()
    assert "not approved for real-money decisions" in plan.casefold()
    assert FORBIDDEN.search(plan) is None


def test_passive_benchmark_guide_exists_for_every_asset():
    for asset in SUPPORTED_ASSETS:
        guide = build_passive_benchmark_guide(asset)
        assert guide["PassiveBenchmarkName"]
        assert guide["PassiveBenchmarkType"]
        assert guide["WhyThisBenchmarkMatters"]
        assert guide["HowToFollowBenchmarkInResearchMode"]
        assert guide["WhatToCompareAgainstBenchmark"]
        assert guide["BenchmarkWarning"]


def test_phase29_artifact_writer_creates_every_required_table(tmp_path):
    snapshot = build_all_asset_prediction_snapshot(
        _plans(), pd.DataFrame(), default_cost_assumptions(), _market_data()
    )
    summary = generate_dashboard_summary(snapshot)
    paths = save_final_user_dashboard_artifacts(
        snapshot, summary, cost_assumptions=default_cost_assumptions(),
        app_source=(ROOT / "app.py").read_text(encoding="utf-8"), output_dir=tmp_path,
    )
    expected = {
        "phase29_latest_asset_prices.csv", "phase29_all_asset_prediction_snapshot.csv",
        "phase29_final_user_plans.csv", "phase29_cost_aware_asset_plans.csv",
        "phase29_cost_assumptions.csv", "phase29_active_vs_passive_cost_comparison.csv",
        "phase29_score_explanations.csv", "phase29_dashboard_summary.csv", "phase29_quality_gates.csv",
    }
    assert set(paths) == expected
    assert all(Path(path).exists() for path in paths.values())
    gates = pd.read_csv(paths["phase29_quality_gates.csv"])
    assert gates["Passed"].astype(bool).all()


def test_generated_outputs_have_no_forbidden_claims_or_execution_approval():
    snapshot = build_all_asset_prediction_snapshot(
        _plans(), pd.DataFrame(), default_cost_assumptions(), _market_data()
    )
    assert FORBIDDEN.search(_flatten(snapshot)) is None
    assert not snapshot["RealMoneyApproved"].astype(bool).any()
    assert not snapshot["BrokerExecutionAllowed"].astype(bool).any()


def test_phase29_app_has_primary_routes_and_compiles():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert '"Run Full Research"' in source
    assert '"Cost-Aware Plan"' in source
    assert '"Paper Research Journey"' in source
    assert "explorer_target = get_asset_target(explorer_asset)" in source
    assert "Advanced Diagnostics" in source

