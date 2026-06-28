from pathlib import Path
import re
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import src.artifact_store as store
from src.artifact_store import load_latest_artifact
from src.asset_config import get_asset_names, get_target_column
from src.signal_policy_edge_lab import (
    POLICY_ASSET_EDGE_COLUMNS,
    POLICY_ASSET_HORIZON_EDGE_COLUMNS,
    POLICY_COST_SENSITIVITY_COLUMNS,
    POLICY_DOMINANCE_FAILURE_COLUMNS,
    POLICY_DRAWDOWN_COLUMNS,
    POLICY_EDGE_LAB_PHASE_NAME,
    POLICY_INPUT_SOURCE_COLUMNS,
    POLICY_LAB_SUMMARY_COLUMNS,
    POLICY_LEADERBOARD_COLUMNS,
    POLICY_NEXT_ACTION_COLUMNS,
    POLICY_OVERFIT_AUDIT_COLUMNS,
    POLICY_QUALITY_GATES_COLUMNS,
    POLICY_RANDOM_COMPARISON_COLUMNS,
    POLICY_RECOMMENDATION_COLUMNS,
    POLICY_STRENGTH_COLUMNS,
    POLICY_TURNOVER_COLUMNS,
    run_signal_policy_edge_lab,
)


FORBIDDEN_LANGUAGE = re.compile(
    r"\b(Buy|Strong Buy|Invest Now|Production Ready|Guaranteed|Safe Profit)\b",
    flags=re.IGNORECASE,
)


def _with_temp_store(fn):
    old_root = store.ARTIFACT_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        store.ARTIFACT_ROOT = Path(tmp) / "artifacts"
        try:
            return fn()
        finally:
            store.ARTIFACT_ROOT = old_root


def _synthetic_market(rows=240):
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    t = np.arange(rows)
    data = pd.DataFrame({"Date": dates})
    for asset in get_asset_names():
        base = 100.0 + np.sin(t / 8.0) * 2.0
        data[get_target_column(asset)] = base
    data[get_target_column("Gold")] = 100.0 + np.maximum(0, t - 80) * 0.35 + np.sin(t / 4.0)
    data[get_target_column("Silver")] = 130.0 - np.maximum(0, t - 80) * 0.25 + np.sin(t / 5.0)
    data[get_target_column("Crude Oil")] = 75.0 + np.sin(t / 3.0) * 4.0
    data[get_target_column("Bitcoin")] = 100.0 + np.sin(t / 2.0) * 8.0 + np.maximum(0, t - 160) * 0.10
    data[get_target_column("S&P 500")] = 3000.0 + t * 0.4 + np.sin(t / 10.0) * 10.0
    data[get_target_column("Gold ETF")] = data[get_target_column("Gold")] * 0.9 + 5.0
    return data


def _overfit_market(rows=240):
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    t = np.arange(rows)
    data = pd.DataFrame({"Date": dates})
    for asset in get_asset_names():
        first = 100.0 + np.minimum(t, rows // 2) * 0.8
        second = first[rows // 2] - np.maximum(0, t - rows // 2) * 0.9
        data[get_target_column(asset)] = np.where(t <= rows // 2, first, second) + np.sin(t / 3.0)
    return data


def _run_basic(**kwargs):
    params = {
        "market_data": _synthetic_market(),
        "use_project_market_data": False,
        "use_artifact_store": False,
        "assets": ["Gold", "Silver"],
        "horizons": [1, 5],
        "cost_bps": 5.0,
        "slippage_bps": 5.0,
        "random_seed": 123,
        "random_simulations": 30,
        "train_fraction": 0.6,
        "min_trades": 1,
    }
    params.update(kwargs)
    return run_signal_policy_edge_lab(**params)


def _all_output_text(report):
    frames = [
        report.policy_lab_summary_table,
        report.policy_leaderboard_table,
        report.policy_asset_edge_table,
        report.policy_asset_horizon_edge_table,
        report.policy_dominance_failures_table,
        report.policy_strength_table,
        report.policy_overfit_audit_table,
        report.policy_cost_sensitivity_table,
        report.policy_random_comparison_table,
        report.policy_drawdown_table,
        report.policy_turnover_table,
        report.policy_quality_gates_table,
        report.policy_recommendation_table,
        report.policy_next_actions_table,
        report.policy_input_sources_table,
    ]
    return "\n".join(frame.astype(str).to_csv(index=False) for frame in frames)


def test_phase19_outputs_all_required_tables_and_columns():
    report = _run_basic()
    expected = {
        "policy_lab_summary_table": POLICY_LAB_SUMMARY_COLUMNS,
        "policy_leaderboard_table": POLICY_LEADERBOARD_COLUMNS,
        "policy_asset_edge_table": POLICY_ASSET_EDGE_COLUMNS,
        "policy_asset_horizon_edge_table": POLICY_ASSET_HORIZON_EDGE_COLUMNS,
        "policy_dominance_failures_table": POLICY_DOMINANCE_FAILURE_COLUMNS,
        "policy_strength_table": POLICY_STRENGTH_COLUMNS,
        "policy_overfit_audit_table": POLICY_OVERFIT_AUDIT_COLUMNS,
        "policy_cost_sensitivity_table": POLICY_COST_SENSITIVITY_COLUMNS,
        "policy_random_comparison_table": POLICY_RANDOM_COMPARISON_COLUMNS,
        "policy_drawdown_table": POLICY_DRAWDOWN_COLUMNS,
        "policy_turnover_table": POLICY_TURNOVER_COLUMNS,
        "policy_quality_gates_table": POLICY_QUALITY_GATES_COLUMNS,
        "policy_recommendation_table": POLICY_RECOMMENDATION_COLUMNS,
        "policy_next_actions_table": POLICY_NEXT_ACTION_COLUMNS,
        "policy_input_sources_table": POLICY_INPUT_SOURCE_COLUMNS,
    }
    for name, columns in expected.items():
        table = getattr(report, name)
        assert set(columns).issubset(table.columns), name


def test_phase19_supports_all_configured_assets_and_horizons():
    report = _run_basic(assets=get_asset_names(), horizons=[1, 5, 10, 20, 30])

    assert len(report.policy_asset_horizon_edge_table) == len(get_asset_names()) * 5
    assert set(report.policy_asset_horizon_edge_table["Asset"]) == set(get_asset_names())
    assert set(report.policy_asset_horizon_edge_table["Horizon"]) == {1, 5, 10, 20, 30}


def test_phase19_trend_policy_can_beat_no_exposure_on_synthetic_trend():
    report = _run_basic(assets=["Gold"], horizons=[5], min_trades=1)
    row = report.policy_leaderboard_table[
        report.policy_leaderboard_table["PolicyName"].eq("TrendMomentumPolicy")
        & report.policy_leaderboard_table["EvaluationMode"].eq("OutOfSample")
    ].iloc[0]

    assert bool(row["BeatsNoExposure"])
    assert row["NetReturnPct"] >= 0


def test_phase19_bad_policy_appears_in_dominance_failures():
    report = _run_basic()

    assert not report.policy_dominance_failures_table.empty
    assert "DominatingBaseline" in report.policy_dominance_failures_table.columns
    assert "InverseMomentumPolicy" in set(report.policy_dominance_failures_table["PolicyName"])


def test_phase19_out_of_sample_split_and_shift_quality_gates_exist():
    report = _run_basic()
    gates = report.policy_quality_gates_table.set_index("GateName")

    assert "OutOfSample" in set(report.policy_leaderboard_table["EvaluationMode"])
    assert "WalkForward" in set(report.policy_leaderboard_table["EvaluationMode"])
    assert bool(gates.loc["SignalsShiftedBeforeReturns", "Passed"])
    assert bool(gates.loc["TrainTestSplitValid", "Passed"])
    assert bool(gates.loc["NoSameDayCloseLeakage", "Passed"])


def test_phase19_costs_reduce_net_returns():
    report = _run_basic(cost_scenarios_bps=[0, 5, 10, 25, 50])
    subset = report.policy_cost_sensitivity_table[
        report.policy_cost_sensitivity_table["PolicyName"].eq("TrendMomentumPolicy")
        & report.policy_cost_sensitivity_table["Asset"].eq("Gold")
        & report.policy_cost_sensitivity_table["Horizon"].eq(5)
    ].sort_values("CostBps")

    assert subset.iloc[-1]["NetReturnPct"] <= subset.iloc[0]["NetReturnPct"]


def test_phase19_random_comparison_is_reproducible():
    first = _run_basic(random_seed=777)
    second = _run_basic(random_seed=777)

    pd.testing.assert_frame_equal(first.policy_random_comparison_table, second.policy_random_comparison_table)


def test_phase19_overfit_audit_flags_in_sample_win_out_of_sample_failure():
    report = _run_basic(market_data=_overfit_market(), assets=["Gold"], horizons=[5], train_fraction=0.5)

    assert "High" in set(report.policy_overfit_audit_table["OverfitRisk"])


def test_phase19_real_capital_remains_blocked():
    report = _run_basic()

    assert set(report.policy_recommendation_table["RealCapitalStatus"]) == {"Blocked"}
    gates = report.policy_quality_gates_table.set_index("GateName")
    assert bool(gates.loc["RealCapitalBlocked", "Passed"])


def test_phase19_autosaves_outputs_to_artifact_store():
    def run():
        report = _run_basic(autosave=True)
        latest = load_latest_artifact(POLICY_EDGE_LAB_PHASE_NAME, "policy_lab_summary_table", required=True)
        assert report.saved_artifacts["RunId"]
        assert latest["PolicyLabVerdict"].iloc[0] == report.policy_lab_summary_table["PolicyLabVerdict"].iloc[0]

    _with_temp_store(run)


def test_phase19_output_has_no_forbidden_live_trading_language():
    report = _run_basic()

    assert not FORBIDDEN_LANGUAGE.search(_all_output_text(report))
