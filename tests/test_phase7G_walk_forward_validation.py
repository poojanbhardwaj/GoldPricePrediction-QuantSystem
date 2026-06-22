from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.signal_engine import (
    WALK_FORWARD_AGG_COLUMNS,
    WALK_FORWARD_WINDOW_COLUMNS,
    build_walk_forward_windows,
    evaluate_walk_forward_window,
    run_walk_forward_risk_validation,
    threshold_cooldown_stability,
)


def _synthetic_signal_output(asset: str = "Silver", horizon: int = 1, n: int = 180):
    probabilities = np.full(n, 0.50, dtype=float)
    probabilities[::4] = 0.70
    probabilities[2::4] = 0.58

    simple_returns = np.zeros(n, dtype=float)
    simple_returns[::4] = 0.03
    simple_returns[2::4] = -0.02
    simple_returns[1::16] = 0.01

    return SimpleNamespace(
        asset=asset,
        probabilities_up_test=probabilities,
        actual_return_test=np.log1p(simple_returns),
        actual_direction_test=(simple_returns > 0.0).astype(int),
        test_index=pd.date_range("2023-01-02", periods=n, freq="B"),
        direction_baseline_accuracy=50.0,
        horizon=int(horizon),
    )


def _signal_outputs(assets, horizons):
    return {
        (asset, int(horizon)): _synthetic_signal_output(asset, int(horizon))
        for asset in assets
        for horizon in horizons
    }


def test_build_walk_forward_windows_rolling_and_expanding():
    rolling = build_walk_forward_windows(total_rows=120, validation_window=40, test_window=20, step_size=20, mode="rolling")
    expanding = build_walk_forward_windows(total_rows=120, validation_window=40, test_window=20, step_size=20, mode="expanding")

    assert len(rolling) == 4
    assert rolling.iloc[0]["ValidationStartRow"] == 0
    assert rolling.iloc[1]["ValidationStartRow"] == 20
    assert len(expanding) == 4
    assert expanding.iloc[1]["ValidationStartRow"] == 0
    assert expanding.iloc[1]["ValidationEndRow"] == 60


def test_evaluate_walk_forward_window_selects_settings_from_validation():
    output = _synthetic_signal_output("Silver", 1)
    window = build_walk_forward_windows(total_rows=120, validation_window=40, test_window=20, step_size=20).iloc[0].to_dict()
    row = evaluate_walk_forward_window(
        signal_output=output,
        window=window,
        asset_name="Silver",
        horizon=1,
        threshold_candidates=(0.55, 0.65),
        cooldown_candidates=(0, 2),
        min_trades_per_window=1,
    )

    assert row["SelectedThreshold"] == 0.65
    assert row["SelectedCooldown"] in {0, 2}
    assert row["ValidationTrades"] > 0
    assert row["LockedTrades"] > 0
    assert "WindowVerdict" in row


def test_walk_forward_validation_runs_multi_asset_multi_horizon():
    assets = ["Silver", "Crude Oil"]
    horizons = [1, 5]
    report = run_walk_forward_risk_validation(
        asset_names=assets,
        horizons=horizons,
        model_depth="core",
        use_phase5_features=True,
        signal_mode="long_only",
        threshold_candidates=(0.55, 0.65),
        cooldown_candidates=(0, 2),
        validation_window=40,
        test_window=20,
        step_size=20,
        min_trades_per_window=1,
        signal_outputs=_signal_outputs(assets, horizons),
    )

    assert not report.window_results.empty
    assert not report.aggregate_summary.empty
    assert set(WALK_FORWARD_WINDOW_COLUMNS).issubset(set(report.window_results.columns))
    assert set(WALK_FORWARD_AGG_COLUMNS).issubset(set(report.aggregate_summary.columns))
    assert len(report.aggregate_summary) == len(assets) * len(horizons)
    assert report.settings["selection_basis"] == "validation_only_per_walk_forward_window"
    assert report.aggregate_summary["WalkForwardVerdict"].notna().all()


def test_threshold_cooldown_stability_flags_stable_and_unstable_settings():
    stable = pd.DataFrame({"SelectedThreshold": [0.65, 0.65, 0.65], "SelectedCooldown": [2, 2, 2]})
    unstable = pd.DataFrame({"SelectedThreshold": [0.50, 0.65, 0.70], "SelectedCooldown": [0, 2, 5]})

    assert threshold_cooldown_stability(stable)["ThresholdStability"] == "Stable"
    assert threshold_cooldown_stability(stable)["CooldownStability"] == "Stable"
    assert threshold_cooldown_stability(unstable)["ThresholdStability"] == "Unstable"
    assert threshold_cooldown_stability(unstable)["CooldownStability"] == "Unstable"


if __name__ == "__main__":
    test_build_walk_forward_windows_rolling_and_expanding()
    test_evaluate_walk_forward_window_selects_settings_from_validation()
    test_walk_forward_validation_runs_multi_asset_multi_horizon()
    test_threshold_cooldown_stability_flags_stable_and_unstable_settings()
    print("Phase 7G walk-forward validation tests passed.")
