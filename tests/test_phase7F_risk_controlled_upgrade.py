from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_engine import (
    RISK_CONTROL_COLUMNS,
    apply_risk_control_variant,
    build_risk_variant_grid,
    run_risk_controlled_candidate_upgrade,
)


def _synthetic_signal_output(asset: str = "Silver", horizon: int = 5):
    n = 96
    probabilities = np.full(n, 0.50, dtype=float)
    probabilities[::3] = 0.72
    probabilities[5::12] = 0.99
    probabilities[6::12] = 0.50

    simple_returns = np.zeros(n, dtype=float)
    for i in range(0, n // 2, 6):
        simple_returns[i] = 0.08
    for i in range(3, n // 2, 12):
        simple_returns[i] = -0.16
    for i in range(n // 2, n, 6):
        simple_returns[i] = 0.04
    for i in range(n // 2 + 3, n, 12):
        simple_returns[i] = -0.10

    return SimpleNamespace(
        asset=asset,
        probabilities_up_test=probabilities,
        actual_return_test=np.log1p(simple_returns),
        actual_direction_test=(simple_returns > 0.0).astype(int),
        test_index=pd.date_range("2024-01-01", periods=n, freq="B"),
        direction_baseline_accuracy=50.0,
        horizon=int(horizon),
    )


def test_build_risk_variant_grid_contains_required_controls():
    grid = build_risk_variant_grid()
    names = set(grid["RiskVariantName"])

    assert "Baseline signal" in names
    assert "Volatility filter" in names
    assert "Drawdown stop" in names
    assert "Loss-streak stop" in names
    assert "Probability band filter" in names
    assert "Position sizing" in names


def test_risk_controlled_upgrade_outputs_baseline_selected_and_cost_stress():
    report = run_risk_controlled_candidate_upgrade(
        asset_name="Silver",
        horizon=5,
        model_depth="core",
        use_phase5_features=True,
        signal_mode="long_only",
        threshold_candidates=(0.55, 0.65),
        cooldown_candidates=(0, 2),
        validation_fraction=0.5,
        risk_variant_names=("baseline", "probability_band_filter", "position_sizing"),
        cost_values=(0.0, 0.001, 0.005),
        signal_output=_synthetic_signal_output("Silver", 5),
    )

    assert not report.full_variant_table.empty
    assert set(RISK_CONTROL_COLUMNS).issubset(set(report.full_variant_table.columns))
    assert not report.baseline_vs_best.empty
    assert not report.cost_stress_table.empty
    assert report.settings["selection_basis"] == "validation_only_risk_variant_score"
    assert report.full_variant_table["SelectedByValidation"].sum() == 1
    assert "Baseline signal" in set(report.full_variant_table["RiskVariantName"])


def test_locked_metrics_only_for_baseline_and_selected_variant():
    report = run_risk_controlled_candidate_upgrade(
        asset_name="Bitcoin",
        horizon=5,
        model_depth="core",
        use_phase5_features=True,
        signal_mode="long_only",
        threshold_candidates=(0.55,),
        cooldown_candidates=(0, 2),
        validation_fraction=0.5,
        risk_variant_names=("baseline", "volatility_filter", "position_sizing"),
        signal_output=_synthetic_signal_output("Bitcoin", 5),
    )

    table = report.full_variant_table
    evaluated = table[table["LockedEvaluationStatus"].ne("Not evaluated on locked test")]
    assert set(evaluated["RiskVariantName"]).issuperset({"Baseline signal"})
    assert len(evaluated) <= 2
    assert table[table["LockedEvaluationStatus"].eq("Not evaluated on locked test")]["LockedVsBuyHold_%"].isna().all()


def test_apply_probability_band_filter_removes_overconfident_trades():
    output = _synthetic_signal_output("Gold", 1)
    baseline = apply_risk_control_variant(
        probabilities_up=output.probabilities_up_test[:48],
        future_returns=output.actual_return_test[:48],
        actual_direction=output.actual_direction_test[:48],
        test_index=output.test_index[:48],
        asset=output.asset,
        horizon=1,
        long_threshold=0.55,
        short_threshold=0.45,
        mode="long_only",
        cooldown=0,
        risk_variant={"RiskVariantName": "Baseline signal", "RiskVariantParams": {"type": "baseline"}},
    )
    filtered = apply_risk_control_variant(
        probabilities_up=output.probabilities_up_test[:48],
        future_returns=output.actual_return_test[:48],
        actual_direction=output.actual_direction_test[:48],
        test_index=output.test_index[:48],
        asset=output.asset,
        horizon=1,
        long_threshold=0.55,
        short_threshold=0.45,
        mode="long_only",
        cooldown=0,
        risk_variant={
            "RiskVariantName": "Probability band filter",
            "RiskVariantParams": {"type": "probability_band_filter", "max_probability": 0.98},
        },
    )

    assert filtered.metrics["NumberOfTrades"] < baseline.metrics["NumberOfTrades"]
    assert filtered.metrics["TradesRemoved"] > 0


if __name__ == "__main__":
    test_build_risk_variant_grid_contains_required_controls()
    test_risk_controlled_upgrade_outputs_baseline_selected_and_cost_stress()
    test_locked_metrics_only_for_baseline_and_selected_variant()
    test_apply_probability_band_filter_removes_overconfident_trades()
    print("Phase 7F risk-controlled upgrade tests passed.")
