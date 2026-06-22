from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_engine import generate_signals, run_signal_engine


def test_generate_signals_supports_long_only_and_long_short():
    probabilities = np.array([0.70, 0.56, 0.50, 0.40, 0.30])

    long_only = generate_signals(
        probabilities,
        long_threshold=0.55,
        short_threshold=0.45,
        mode="long_only",
    )
    assert long_only.tolist() == [1, 1, 0, 0, 0]

    long_short = generate_signals(
        probabilities,
        long_threshold=0.55,
        short_threshold=0.45,
        mode="long_short",
    )
    assert long_short.tolist() == [1, 1, 0, -1, -1]


def test_signal_engine_reports_metrics_and_research_sweep():
    output = SimpleNamespace(
        probabilities_up_test=np.array([0.72, 0.61, 0.53, 0.44, 0.32, 0.67, 0.38, 0.58]),
        actual_return_test=np.array([0.03, 0.02, -0.01, -0.02, -0.03, 0.015, -0.01, 0.02]),
        actual_direction_test=np.array([1, 1, 0, 0, 0, 1, 0, 1]),
        test_index=pd.date_range("2024-01-01", periods=8, freq="B"),
        direction_baseline_accuracy=50.0,
        horizon=5,
    )

    result = run_signal_engine(
        signal_output=output,
        long_threshold=0.55,
        short_threshold=0.45,
        mode="long_short",
        transaction_cost=0.0,
    )

    metrics = result.metrics
    assert metrics["SignalCount"] == 7
    assert metrics["LongCount"] == 4
    assert metrics["ShortCount"] == 3
    assert metrics["NoTradeCount"] == 1
    assert metrics["WinRateActive_%"] > 0
    assert "StrategyMinusBuyHold_%" in metrics
    assert "Precision_UpSignals" in metrics
    assert "Recall_UpSignals" in metrics
    assert "F1_UpSignals" in metrics
    assert "DirectionAccuracyActive_%" in metrics
    assert "BaselineDirectionAccuracy_%" in metrics
    assert "ThresholdVerdict" in metrics

    assert not result.signal_frame.empty
    assert set(result.signal_frame["Signal"]).issubset({-1, 0, 1})

    assert not result.threshold_sweep.empty
    assert result.threshold_sweep["ResearchOnly"].all()
    assert "ThresholdVerdict" in result.threshold_sweep.columns


def test_signal_engine_warns_when_signal_count_is_too_low():
    output = SimpleNamespace(
        probabilities_up_test=np.array([0.51, 0.52, 0.53, 0.54, 0.56]),
        actual_return_test=np.array([0.01, -0.01, 0.005, -0.005, 0.002]),
        actual_direction_test=np.array([1, 0, 1, 0, 1]),
        test_index=pd.date_range("2024-02-01", periods=5, freq="B"),
        direction_baseline_accuracy=60.0,
        horizon=1,
    )

    result = run_signal_engine(
        signal_output=output,
        long_threshold=0.70,
        short_threshold=0.30,
        mode="long_only",
        transaction_cost=0.0,
    )

    assert result.metrics["SignalCount"] == 0
    assert result.metrics["ThresholdVerdict"] == "Do not trust for signals"
    assert "too few active signals" in result.metrics["Warnings"]


if __name__ == "__main__":
    test_generate_signals_supports_long_only_and_long_short()
    test_signal_engine_reports_metrics_and_research_sweep()
    test_signal_engine_warns_when_signal_count_is_too_low()
    print("Phase 7 signal engine tests passed.")
