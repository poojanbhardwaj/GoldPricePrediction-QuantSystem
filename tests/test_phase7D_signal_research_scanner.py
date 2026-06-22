from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_engine import (
    SIGNAL_SCAN_COLUMNS,
    robust_signal_verdict,
    run_signal_research_scan,
    score_validation_threshold,
)


def _synthetic_signal_output(asset: str, horizon: int):
    n = 80
    probabilities = np.full(n, 0.50, dtype=float)
    probabilities[::2] = 0.70

    simple_returns = np.zeros(n, dtype=float)
    for i in range(0, n // 2, 4):
        simple_returns[i] = 0.08
    for i in range(2, n // 2, 4):
        simple_returns[i] = -0.10
    for i in range(n // 2, n, 4):
        simple_returns[i] = -0.10
    for i in range(n // 2 + 2, n, 4):
        simple_returns[i] = 0.08

    return SimpleNamespace(
        asset=asset,
        probabilities_up_test=probabilities,
        actual_return_test=np.log1p(simple_returns),
        actual_direction_test=(simple_returns > 0.0).astype(int),
        test_index=pd.date_range("2024-01-01", periods=n, freq="B"),
        direction_baseline_accuracy=50.0,
        horizon=int(horizon),
    )


def _signal_outputs(assets, horizons):
    return {
        (asset, int(horizon)): _synthetic_signal_output(asset, int(horizon))
        for asset in assets
        for horizon in horizons
    }


def test_signal_research_scan_runs_multi_asset_multi_horizon():
    assets = ["Gold", "Bitcoin"]
    horizons = [1, 5]
    report = run_signal_research_scan(
        asset_names=assets,
        horizons=horizons,
        model_depth="core",
        use_phase5_features=True,
        threshold_candidates=(0.55,),
        cooldown_candidates=(0, 2),
        validation_fraction=0.5,
        signal_outputs=_signal_outputs(assets, horizons),
    )

    assert len(report.full_results) == len(assets) * len(horizons)
    assert set(SIGNAL_SCAN_COLUMNS).issubset(set(report.full_results.columns))
    assert set(report.full_results["Asset"]) == set(assets)
    assert set(report.full_results["Horizon"]) == set(horizons)
    assert report.settings["backtest_style"] == "non_overlapping_realistic"
    assert report.settings["threshold_policy"] == "validation_locked"
    assert report.settings["cooldown_selection_basis"] == "validation_score_only"
    assert not report.candidate_results.empty


def test_cooldown_selection_uses_validation_score_not_locked_test():
    report = run_signal_research_scan(
        asset_names=["Gold"],
        horizons=[1],
        model_depth="core",
        use_phase5_features=True,
        threshold_candidates=(0.55,),
        cooldown_candidates=(0, 2),
        validation_fraction=0.5,
        signal_outputs=_signal_outputs(["Gold"], [1]),
    )

    row = report.full_results.iloc[0]
    assert int(row["CooldownRows"]) == 2
    assert row["SelectionBasis"] == "validation_score_only"
    assert row["ValidationScore"] > 0
    assert row["LockedTestVsBuyHold_%"] < 0
    assert "locked test fails buy-and-hold" in str(row["FailureReason"])

    selected_candidates = report.candidate_results[report.candidate_results["SelectedCooldownForAssetHorizon"].eq(True)]
    assert len(selected_candidates) == 1
    assert int(selected_candidates.iloc[0]["CooldownRows"]) == 2


def test_robust_verdict_and_validation_score_are_conservative():
    validation_metrics = {
        "Rows": 80,
        "NumberOfTrades": 12,
        "StrategyMinusBuyHold_%": 8.0,
        "WinRate_%": 60.0,
        "DirectionAccuracyActive_%": 58.0,
        "BaselineDirectionAccuracy_%": 50.0,
        "MaxDrawdown_%": -8.0,
        "ThresholdVerdict": "Low trust / research only",
    }
    locked_metrics = {
        "Rows": 80,
        "NumberOfTrades": 3,
        "StrategyMinusBuyHold_%": 12.0,
        "WinRate_%": 66.0,
        "DirectionAccuracyActive_%": 60.0,
        "BaselineDirectionAccuracy_%": 50.0,
        "MaxDrawdown_%": -6.0,
        "Sharpe": 1.2,
        "ThresholdVerdict": "Low trust / research only",
    }

    score = score_validation_threshold(validation_metrics)
    verdict = robust_signal_verdict(
        validation_metrics=validation_metrics,
        locked_test_metrics=locked_metrics,
        validation_score=score["ValidationSelectionScore"],
    )

    assert score["ValidationSelectionScore"] < validation_metrics["StrategyMinusBuyHold_%"] + 20
    assert verdict["RobustnessVerdict"] == "Do Not Trust"
    assert verdict["StabilityFlag"] == "LowEvidence"
    assert "locked test trades too few" in verdict["FailureReason"]


if __name__ == "__main__":
    test_signal_research_scan_runs_multi_asset_multi_horizon()
    test_cooldown_selection_uses_validation_score_not_locked_test()
    test_robust_verdict_and_validation_score_are_conservative()
    print("Phase 7D signal research scanner tests passed.")
