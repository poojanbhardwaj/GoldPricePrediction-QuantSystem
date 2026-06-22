from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.direct_forecast_models import (
    DIRECT_FORECAST_HORIZONS,
    add_direct_forecast_targets,
    direct_forecast_baseline_board,
    make_direct_forecast_dataset,
    run_direct_forecast_report,
)


def _make_simple_feature_frame(n: int = 180) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    t = np.arange(n, dtype=float)
    price = 100.0 + 0.25 * t + 2.0 * np.sin(t / 7.0)

    return pd.DataFrame(
        {
            "Gold_Close": price,
            "known_feature_1": np.log(price),
            "known_feature_2": np.cos(t / 11.0),
        },
        index=idx,
    )


def _add_ohlcv(df: pd.DataFrame, prefix: str, close: np.ndarray) -> None:
    df[f"{prefix}_Open"] = close * 0.998
    df[f"{prefix}_High"] = close * 1.01
    df[f"{prefix}_Low"] = close * 0.99
    df[f"{prefix}_Close"] = close
    df[f"{prefix}_Volume"] = 100000 + np.arange(len(df)) * 10


def _make_synthetic_market_data(n: int = 360) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    t = np.arange(n, dtype=float)
    df = pd.DataFrame(index=idx)

    gold = 1500 + 0.8 * t + 12 * np.sin(t / 11)
    btc = 9000 + 35 * t + 450 * np.sin(t / 17)
    _add_ohlcv(df, "Gold", gold)
    _add_ohlcv(df, "BTC", btc)

    df["Silver_Close"] = 18 + 0.02 * t + 0.6 * np.sin(t / 9)
    df["Oil_Close"] = 55 + 0.03 * t + 4 * np.sin(t / 19)
    df["SP500_Close"] = 3000 + 2.5 * t + 30 * np.sin(t / 15)
    df["GLD_Close"] = 140 + 0.08 * t + 1.5 * np.sin(t / 10)
    df["DXY_Close"] = 100 + 0.04 * np.sin(t / 13)
    df["VIX_Close"] = 20 + 3 * np.sin(t / 5)
    df["TNX_Close"] = 4 + 0.2 * np.sin(t / 23)
    return df


def test_direct_targets_are_aligned_for_every_horizon():
    df = _make_simple_feature_frame()
    price = df["Gold_Close"]

    for horizon in DIRECT_FORECAST_HORIZONS:
        out = add_direct_forecast_targets(df, target_col="Gold_Close", horizon=horizon)
        return_col = f"future_return_{horizon}d"
        direction_col = f"future_direction_{horizon}d"

        expected = np.log(price.iloc[horizon] / price.iloc[0])
        assert np.isclose(out[return_col].iloc[0], expected)
        assert bool(out[direction_col].iloc[0]) == bool(expected > 0)
        assert out[return_col].tail(horizon).isna().all()


def test_direct_dataset_drops_tail_and_excludes_future_targets():
    df = _make_simple_feature_frame()

    for horizon in DIRECT_FORECAST_HORIZONS:
        dataset = make_direct_forecast_dataset(
            df,
            asset="Gold",
            target_col="Gold_Close",
            horizon=horizon,
        )

        assert len(dataset.df_model) == len(df) - horizon
        assert dataset.dropped_tail_rows == horizon
        assert not any(c.startswith("future_return_") for c in dataset.feature_cols)
        assert not any(c.startswith("future_direction_") for c in dataset.feature_cols)
        assert not any(c.startswith("future_realized_vol_") for c in dataset.feature_cols)


def test_direct_dataset_split_is_time_ordered_and_has_baselines():
    df = _make_simple_feature_frame()
    dataset = make_direct_forecast_dataset(
        df,
        asset="Gold",
        target_col="Gold_Close",
        horizon=10,
    )

    assert dataset.train_index.max() < dataset.val_index.min()
    assert dataset.val_index.max() < dataset.test_index.min()

    baselines = direct_forecast_baseline_board(dataset)
    assert "Zero Return baseline" in set(baselines["Baseline"])
    assert "Majority Train Direction baseline" in set(baselines["Baseline"])
    assert "Always Up baseline" in set(baselines["Baseline"])


def test_direct_forecast_report_runs_for_gold_and_bitcoin():
    raw_df = _make_synthetic_market_data()

    for asset in ["Gold", "Bitcoin"]:
        report = run_direct_forecast_report(
            raw_df=raw_df,
            asset_name=asset,
            horizon=5,
            model_depth="fast",
            use_phase5_features=True,
        )

        assert report.asset == asset
        assert report.horizon == 5
        assert report.feature_count > 0
        assert not report.leaderboard.empty
        assert not report.baseline_board.empty
        assert "RMSE_vs_Naive_%" in report.leaderboard.columns
        assert "Direction_vs_Baseline_%" in report.leaderboard.columns
        assert "Verdict" in report.leaderboard.columns


if __name__ == "__main__":
    test_direct_targets_are_aligned_for_every_horizon()
    test_direct_dataset_drops_tail_and_excludes_future_targets()
    test_direct_dataset_split_is_time_ordered_and_has_baselines()
    test_direct_forecast_report_runs_for_gold_and_bitcoin()
    print("Phase 6 direct forecast model tests passed.")
