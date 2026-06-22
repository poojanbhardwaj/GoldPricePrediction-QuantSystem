from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.direct_forecast_models import (
    FUTURE_TARGET_PREFIXES,
    SCAN_SUMMARY_COLUMNS,
    run_asset_horizon_scan,
)


def _add_ohlcv(df: pd.DataFrame, prefix: str, close: np.ndarray) -> None:
    df[f"{prefix}_Open"] = close * 0.998
    df[f"{prefix}_High"] = close * 1.01
    df[f"{prefix}_Low"] = close * 0.99
    df[f"{prefix}_Close"] = close
    df[f"{prefix}_Volume"] = 100000 + np.arange(len(df)) * 10


def _make_synthetic_market_data(n: int = 380) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    t = np.arange(n, dtype=float)
    df = pd.DataFrame(index=idx)

    gold = 1500 + 0.35 * t + 35 * np.sin(t / 8) + 10 * np.cos(t / 17)
    btc = 9000 + 18 * t + 700 * np.sin(t / 13) + 180 * np.cos(t / 9)
    _add_ohlcv(df, "Gold", gold)
    _add_ohlcv(df, "BTC", btc)

    df["Silver_Close"] = 18 + 0.01 * t + 0.9 * np.sin(t / 9)
    df["Oil_Close"] = 55 + 0.02 * t + 5 * np.sin(t / 19)
    df["SP500_Close"] = 3000 + 1.2 * t + 45 * np.sin(t / 15)
    df["GLD_Close"] = 140 + 0.04 * t + 2.5 * np.sin(t / 10)
    df["DXY_Close"] = 100 + 0.7 * np.sin(t / 13)
    df["VIX_Close"] = 20 + 4 * np.sin(t / 5)
    df["TNX_Close"] = 4 + 0.3 * np.sin(t / 23)
    return df


def test_phase6b_scanner_runs_gold_bitcoin_horizons_1_and_5():
    report = run_asset_horizon_scan(
        raw_df=_make_synthetic_market_data(),
        asset_names=["Gold", "Bitcoin"],
        horizons=[1, 5],
        model_depth="fast",
        use_phase5_features=True,
    )

    summary = report.asset_horizon_summary
    assert len(summary) == 4
    assert set(zip(summary["Asset"], summary["Horizon"])) == {
        ("Gold", 1),
        ("Gold", 5),
        ("Bitcoin", 1),
        ("Bitcoin", 5),
    }

    for col in SCAN_SUMMARY_COLUMNS:
        assert col in summary.columns

    assert summary["FeatureLeakageCount"].fillna(0).astype(int).eq(0).all()
    for direct_report in report.reports.values():
        assert direct_report.dataset is not None
        assert not any(
            feature.startswith(FUTURE_TARGET_PREFIXES)
            for feature in direct_report.dataset.feature_cols
        )

    assert summary["Best_Verdict"].astype(str).str.len().gt(0).all()
    assert "Best_RMSE_vs_Naive_%" in summary.columns
    assert "Best_Direction_vs_Baseline_%" in summary.columns
    assert summary["Best_RMSE_Return"].notna().any()
    assert summary["Best_DirectionBaselineAccuracy"].notna().any()

    assert set(report.status_counts) == {"High", "Medium", "Low", "DoNotTrust"}
    assert report.top_promising is not None
    assert report.worst_failed is not None


if __name__ == "__main__":
    test_phase6b_scanner_runs_gold_bitcoin_horizons_1_and_5()
    print("Phase 6B direct horizon scanner tests passed.")
