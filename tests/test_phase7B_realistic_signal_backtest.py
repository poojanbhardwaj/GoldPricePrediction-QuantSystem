from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_engine import run_signal_engine, run_threshold_sweep


def _signal_output():
    return SimpleNamespace(
        asset="Crude Oil",
        probabilities_up_test=np.array([0.72, 0.68, 0.62, 0.51, 0.66, 0.35, 0.70, 0.58, 0.40, 0.69, 0.63, 0.57]),
        actual_return_test=np.array([0.03, 0.02, 0.01, -0.005, 0.025, -0.02, 0.015, 0.006, -0.015, 0.02, 0.01, -0.004]),
        actual_direction_test=np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0]),
        test_index=pd.date_range("2024-01-01", periods=12, freq="B"),
        direction_baseline_accuracy=50.0,
        horizon=3,
    )


def test_non_overlapping_realistic_trade_log_and_metrics():
    result = run_signal_engine(
        signal_output=_signal_output(),
        long_threshold=0.60,
        short_threshold=0.40,
        mode="long_only",
        transaction_cost=0.001,
        backtest_style="non_overlapping_realistic",
        cooldown=1,
    )

    metrics = result.metrics
    trade_log = result.signal_frame

    assert metrics["BacktestStyle"] == "non_overlapping_realistic"
    assert metrics["NumberOfTrades"] == len(trade_log)
    assert metrics["NumberOfTrades"] > 0
    assert "TotalCompoundedReturn_%" in metrics
    assert "MedianTradeReturn_%" in metrics
    assert "Exposure_%" in metrics
    assert "ThresholdVerdict" in metrics

    required_cols = {
        "EntryDate",
        "ExitDate",
        "Asset",
        "Horizon",
        "Signal",
        "ProbabilityUp",
        "EntryReturnTarget",
        "RealizedReturn",
        "StrategyReturnAfterCost",
        "HoldingDays",
        "Win/Loss",
        "LongThreshold",
        "ShortThreshold",
        "Mode",
    }
    assert required_cols.issubset(set(trade_log.columns))
    assert trade_log["Asset"].eq("Crude Oil").all()
    assert trade_log["HoldingDays"].eq(3).all()

    entry_rows = trade_log["EntryRow"].to_numpy()
    assert np.all(np.diff(entry_rows) >= 5)  # horizon 3 + one exit row + cooldown 1


def test_realistic_mode_trades_less_than_overlapping_mode():
    output = _signal_output()
    overlapping = run_signal_engine(
        signal_output=output,
        long_threshold=0.60,
        short_threshold=0.40,
        mode="long_only",
        transaction_cost=0.001,
        backtest_style="overlapping_research",
    )
    realistic = run_signal_engine(
        signal_output=output,
        long_threshold=0.60,
        short_threshold=0.40,
        mode="long_only",
        transaction_cost=0.001,
        backtest_style="non_overlapping_realistic",
        cooldown=0,
    )

    assert overlapping.metrics["BacktestStyle"] == "overlapping_research"
    assert realistic.metrics["BacktestStyle"] == "non_overlapping_realistic"
    assert realistic.metrics["NumberOfTrades"] < overlapping.metrics["SignalCount"]


def test_threshold_sweep_reports_both_styles_as_research_only():
    output = _signal_output()
    sweep = run_threshold_sweep(
        probabilities_up=output.probabilities_up_test,
        future_returns=output.actual_return_test,
        actual_direction=output.actual_direction_test,
        test_index=output.test_index,
        baseline_direction_accuracy=output.direction_baseline_accuracy,
        mode="long_only",
        transaction_cost=0.001,
        horizon=output.horizon,
        backtest_style="both",
        cooldown=1,
        asset=output.asset,
    )

    assert not sweep.empty
    assert {"overlapping_research", "non_overlapping_realistic"}.issubset(set(sweep["BacktestStyle"]))
    assert sweep["ResearchOnly"].all()


if __name__ == "__main__":
    test_non_overlapping_realistic_trade_log_and_metrics()
    test_realistic_mode_trades_less_than_overlapping_mode()
    test_threshold_sweep_reports_both_styles_as_research_only()
    print("Phase 7B realistic signal backtest tests passed.")
