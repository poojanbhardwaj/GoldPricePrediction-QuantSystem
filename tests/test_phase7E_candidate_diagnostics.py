from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_engine import (
    build_cost_sensitivity_table,
    diagnose_benchmark_dependency,
    run_candidate_deep_diagnostics,
    summarize_trade_diagnostics,
)


def _synthetic_signal_output(asset: str = "Silver", horizon: int = 5):
    n = 90
    probabilities = np.full(n, 0.50, dtype=float)
    probabilities[::3] = 0.72
    probabilities[1::6] = 0.58

    simple_returns = np.zeros(n, dtype=float)
    for i in range(0, n // 2, 6):
        simple_returns[i] = 0.07
    for i in range(3, n // 2, 6):
        simple_returns[i] = -0.04
    for i in range(n // 2, n, 6):
        simple_returns[i] = -0.06
    for i in range(n // 2 + 3, n, 6):
        simple_returns[i] = 0.03

    return SimpleNamespace(
        asset=asset,
        probabilities_up_test=probabilities,
        actual_return_test=np.log1p(simple_returns),
        actual_direction_test=(simple_returns > 0.0).astype(int),
        test_index=pd.date_range("2024-01-01", periods=n, freq="B"),
        direction_baseline_accuracy=50.0,
        horizon=int(horizon),
    )


def test_candidate_deep_diagnostics_outputs_required_tables():
    report = run_candidate_deep_diagnostics(
        asset_name="Silver",
        horizon=5,
        model_depth="core",
        use_phase5_features=True,
        signal_mode="long_only",
        threshold_candidates=(0.55, 0.65),
        cooldown_candidates=(0, 2),
        validation_fraction=0.5,
        signal_output=_synthetic_signal_output("Silver", 5),
    )

    assert not report.candidate_summary.empty
    assert not report.trade_diagnostics.empty
    assert not report.monthly_returns.empty
    assert not report.quarterly_returns.empty
    assert not report.cost_sensitivity.empty
    assert not report.validation_split_sensitivity.empty
    assert not report.probability_diagnostics.empty
    assert "SelectedThreshold" in report.candidate_summary.columns
    assert "ProfitFactor" in report.trade_diagnostics.columns
    assert "MissedLargeBuyHoldRally" in report.monthly_returns.columns
    assert "Drawdown_%" in report.drawdown_curve.columns


def test_cost_sensitivity_keeps_selected_threshold_and_cooldown_fixed():
    report = run_candidate_deep_diagnostics(
        asset_name="Bitcoin",
        horizon=5,
        model_depth="core",
        use_phase5_features=True,
        signal_mode="long_only",
        threshold_candidates=(0.55, 0.65),
        cooldown_candidates=(0, 2),
        validation_fraction=0.5,
        signal_output=_synthetic_signal_output("Bitcoin", 5),
    )

    cost_table = report.cost_sensitivity
    selected = float(report.candidate_summary.iloc[0]["SelectedThreshold"])
    cooldown = int(report.candidate_summary.iloc[0]["SelectedCooldown"])
    assert cost_table["SelectedThreshold"].astype(float).eq(selected).all()
    assert cost_table["SelectedCooldown"].astype(int).eq(cooldown).all()
    assert set(cost_table["TransactionCost_%"].round(2)) >= {0.0, 0.05, 0.10, 0.20, 0.50}


def test_benchmark_dependency_warning_is_explicit():
    warning = diagnose_benchmark_dependency(
        {
            "TotalCompoundedReturn_%": -4.0,
            "StrategyMinusBuyHold_%": 12.0,
        }
    )

    assert warning["BenchmarkWeakness"] is True
    assert "BenchmarkWeakness" in warning["BenchmarkDependencyWarning"]


def test_trade_diagnostics_handles_empty_trade_log():
    diagnostics = summarize_trade_diagnostics(pd.DataFrame(), {"Exposure_%": 0.0})

    assert diagnostics.iloc[0]["NumberOfTrades"] == 0
    assert diagnostics.iloc[0]["ProfitFactor"] == 0.0


def test_build_cost_sensitivity_table_fixed_settings_directly():
    output = _synthetic_signal_output("Gold", 1)
    table = build_cost_sensitivity_table(
        signal_output=output,
        selected_long_threshold=0.55,
        selected_short_threshold=0.45,
        selected_cooldown=2,
        validation_fraction=0.5,
        mode="long_only",
        costs=(0.0, 0.001),
    )

    assert len(table) == 2
    assert table["SelectedThreshold"].astype(float).eq(0.55).all()
    assert table["SelectedCooldown"].astype(int).eq(2).all()
    assert "VsBuyHold_%" in table.columns


if __name__ == "__main__":
    test_candidate_deep_diagnostics_outputs_required_tables()
    test_cost_sensitivity_keeps_selected_threshold_and_cooldown_fixed()
    test_benchmark_dependency_warning_is_explicit()
    test_trade_diagnostics_handles_empty_trade_log()
    test_build_cost_sensitivity_table_fixed_settings_directly()
    print("Phase 7E candidate diagnostics tests passed.")
