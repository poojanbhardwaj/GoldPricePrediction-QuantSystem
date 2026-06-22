from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_engine import run_validation_locked_signal_engine


def _locked_test_would_prefer_lower_threshold_output():
    simple_returns = np.array(
        [
            0.08,
            0.00,
            0.08,
            0.00,
            0.08,
            0.00,
            -0.10,
            0.00,
            -0.10,
            0.00,
            -0.10,
            0.00,
            -0.10,
            0.00,
            0.08,
            0.00,
            0.08,
            0.00,
            0.08,
            0.00,
        ],
        dtype=float,
    )
    probabilities = np.array(
        [
            0.70,
            0.50,
            0.72,
            0.50,
            0.69,
            0.50,
            0.58,
            0.50,
            0.57,
            0.50,
            0.70,
            0.50,
            0.72,
            0.50,
            0.58,
            0.50,
            0.57,
            0.50,
            0.59,
            0.50,
        ],
        dtype=float,
    )
    return SimpleNamespace(
        asset="Crude Oil",
        probabilities_up_test=probabilities,
        actual_return_test=np.log1p(simple_returns),
        actual_direction_test=(simple_returns > 0.0).astype(int),
        test_index=pd.date_range("2024-01-01", periods=len(simple_returns), freq="B"),
        direction_baseline_accuracy=50.0,
        horizon=1,
    )


def _validation_good_locked_test_bad_output():
    probabilities = np.array(([0.70, 0.50] * 20), dtype=float)
    validation_returns = np.array(([0.05, -0.03] * 10), dtype=float)
    locked_returns = np.array(([-0.05, 0.03] * 10), dtype=float)
    simple_returns = np.concatenate([validation_returns, locked_returns])
    return SimpleNamespace(
        asset="Gold",
        probabilities_up_test=probabilities,
        actual_return_test=np.log1p(simple_returns),
        actual_direction_test=(simple_returns > 0.0).astype(int),
        test_index=pd.date_range("2024-03-01", periods=len(simple_returns), freq="B"),
        direction_baseline_accuracy=50.0,
        horizon=1,
    )


def test_validation_locked_selects_from_validation_not_locked_test():
    result = run_validation_locked_signal_engine(
        signal_output=_locked_test_would_prefer_lower_threshold_output(),
        mode="long_only",
        transaction_cost=0.0,
        backtest_style="non_overlapping_realistic",
        cooldown=0,
        validation_fraction=0.5,
        long_thresholds=(0.55, 0.65),
    )

    selected = result.selected_threshold
    assert selected["ThresholdPolicy"] == "validation_locked"
    assert selected["SelectionSource"] == "validation_segment_within_available_out_of_sample"
    assert selected["SelectedLongThreshold"] == 0.65
    assert result.metrics["LongThreshold"] == 0.65

    selected_rows = result.validation_sweep[result.validation_sweep["SelectedLockedThreshold"]]
    assert len(selected_rows) == 1
    assert float(selected_rows.iloc[0]["LongThreshold"]) == 0.65


def test_validation_locked_outputs_tables_and_chronological_split():
    result = run_validation_locked_signal_engine(
        signal_output=_locked_test_would_prefer_lower_threshold_output(),
        mode="long_only",
        transaction_cost=0.0,
        backtest_style="non_overlapping_realistic",
        cooldown=0,
        validation_fraction=0.5,
        long_thresholds=(0.55, 0.65),
    )

    assert not result.validation_sweep.empty
    assert not result.validation_test_comparison.empty
    assert {"Validation Selection", "Locked Test"} == set(result.validation_test_comparison["Segment"])
    assert "ValidationSelectionScore" in result.validation_sweep.columns
    assert "Selection_DrawdownPenalty" in result.validation_sweep.columns
    assert "Selection_LowTradePenalty" in result.validation_sweep.columns

    validation_entries = pd.to_datetime(result.validation_signal_frame["EntryDate"])
    locked_entries = pd.to_datetime(result.signal_frame["EntryDate"])
    assert validation_entries.max() < locked_entries.min()


def test_validation_good_but_locked_test_failure_is_warned():
    result = run_validation_locked_signal_engine(
        signal_output=_validation_good_locked_test_bad_output(),
        mode="long_only",
        transaction_cost=0.0,
        backtest_style="non_overlapping_realistic",
        cooldown=0,
        validation_fraction=0.5,
        long_thresholds=(0.55,),
    )

    assert result.selected_threshold["SelectedLongThreshold"] == 0.55
    assert result.validation_metrics["StrategyMinusBuyHold_%"] > 0
    assert result.metrics["StrategyMinusBuyHold_%"] <= 0
    assert "validation looked promising but locked test failed" in str(result.metrics.get("Warnings", ""))


if __name__ == "__main__":
    test_validation_locked_selects_from_validation_not_locked_test()
    test_validation_locked_outputs_tables_and_chronological_split()
    test_validation_good_but_locked_test_failure_is_warned()
    print("Phase 7C validation-locked signal tests passed.")
