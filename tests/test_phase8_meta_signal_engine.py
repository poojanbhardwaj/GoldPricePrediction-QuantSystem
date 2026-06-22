from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.meta_signal_engine import (
    META_SIGNAL_COLUMNS,
    build_regime_features,
    run_regime_aware_meta_signal,
)


def _raw_prices(n: int = 320) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    up = 100.0 + np.linspace(0.0, 35.0, n)
    down = 120.0 - np.linspace(0.0, 25.0, n)
    neutral = 80.0 + np.sin(np.linspace(0.0, 10.0, n)) * 2.0

    return pd.DataFrame(
        {
            "Gold_Close": up,
            "Silver_Close": up * 0.7,
            "Oil_Close": neutral,
            "BTC_Close": down,
            "SP500_Close": up * 1.4,
            "GLD_Close": up * 0.95,
        },
        index=dates,
    )


def _walk_forward_summary() -> pd.DataFrame:
    rows = [
        {
            "Asset": "Gold",
            "Horizon": 1,
            "NumberOfWindows": 6,
            "BeatBuyHoldRate_%": 83.0,
            "PositiveReturnRate_%": 83.0,
            "AvgLockedStrategyReturn_%": 9.0,
            "MedianLockedStrategyReturn_%": 7.0,
            "AvgLockedVsBuyHold_%": 5.5,
            "MedianLockedVsBuyHold_%": 4.0,
            "WorstLockedVsBuyHold_%": -2.0,
            "AvgLockedMaxDrawdown_%": -5.0,
            "WorstLockedMaxDrawdown_%": -8.0,
            "AvgLockedSharpe": 1.25,
            "MedianLockedSharpe": 1.10,
            "AvgTradesPerWindow": 5.0,
            "LowTradeWindowCount": 0,
            "ThresholdStability": "Stable",
            "CooldownStability": "Stable",
            "WalkForwardStabilityScore": 82.0,
            "WalkForwardVerdict": "Strong walk-forward research candidate",
            "FailureReason": "",
        },
        {
            "Asset": "Bitcoin",
            "Horizon": 5,
            "NumberOfWindows": 6,
            "BeatBuyHoldRate_%": 17.0,
            "PositiveReturnRate_%": 33.0,
            "AvgLockedStrategyReturn_%": -6.0,
            "MedianLockedStrategyReturn_%": -5.0,
            "AvgLockedVsBuyHold_%": -8.0,
            "MedianLockedVsBuyHold_%": -7.0,
            "WorstLockedVsBuyHold_%": -22.0,
            "AvgLockedMaxDrawdown_%": -18.0,
            "WorstLockedMaxDrawdown_%": -31.0,
            "AvgLockedSharpe": -0.4,
            "MedianLockedSharpe": -0.3,
            "AvgTradesPerWindow": 4.0,
            "LowTradeWindowCount": 0,
            "ThresholdStability": "Unstable",
            "CooldownStability": "Stable",
            "WalkForwardStabilityScore": 18.0,
            "WalkForwardVerdict": "Do not trust",
            "FailureReason": "locked test fails buy-and-hold badly",
        },
    ]
    return pd.DataFrame(rows)


def test_meta_signal_runs_multi_asset_multi_horizon_and_keeps_rejections_visible():
    report = run_regime_aware_meta_signal(
        raw_df=_raw_prices(),
        walk_forward_summary=_walk_forward_summary(),
        asset_names=["Gold", "Bitcoin"],
        horizons=[1, 5],
        model_depth="core",
        use_phase5_features=True,
        signal_mode="long_only",
    )

    assert len(report.decision_table) == 4
    assert set(META_SIGNAL_COLUMNS).issubset(set(report.decision_table.columns))
    assert set(report.decision_table["Asset"]) == {"Gold", "Bitcoin"}
    assert set(report.decision_table["Horizon"]) == {1, 5}
    assert "Avoid" in set(report.decision_table["MetaDecision"])
    assert report.decision_summary["Count"].sum() == 4


def test_weak_walk_forward_rows_do_not_become_trade():
    report = run_regime_aware_meta_signal(
        raw_df=_raw_prices(),
        walk_forward_summary=_walk_forward_summary(),
        asset_names=["Bitcoin"],
        horizons=[5],
    )

    row = report.decision_table.iloc[0]
    assert row["MetaDecision"] != "Trade"
    assert row["MetaDecision"] == "Avoid"
    assert row["BenchmarkRiskFlag"] is True or bool(row["BenchmarkRiskFlag"]) is True
    assert "Do Not Trust" in row["MainReason"] or "Do Not Trust" in row["Warnings"] or "Do not trust" in row["WalkForwardVerdict"]


def test_strong_rows_can_only_trade_when_risk_benchmark_and_stability_allow():
    report = run_regime_aware_meta_signal(
        raw_df=_raw_prices(),
        walk_forward_summary=_walk_forward_summary(),
        asset_names=["Gold"],
        horizons=[1],
    )

    row = report.decision_table.iloc[0]
    assert row["WalkForwardVerdict"] == "Strong walk-forward research candidate"
    assert row["BenchmarkRiskFlag"] is False or bool(row["BenchmarkRiskFlag"]) is False
    assert row["StabilityFlag"] == "Stable"
    assert row["MetaDecision"] in {"Trade", "Research Only"}
    if row["MetaDecision"] == "Trade":
        assert float(row["MetaConfidenceScore"]) >= 72.0
        assert float(row["MetaRiskScore"]) <= 35.0


def test_regime_features_do_not_require_future_columns():
    raw = _raw_prices()[["Gold_Close", "SP500_Close"]].copy()
    regime = build_regime_features(raw, ["Gold"])
    assert not regime.empty
    assert "future_return_5" not in regime.columns
    assert "RegimeDataWarning" in regime.columns

    report = run_regime_aware_meta_signal(
        raw_df=raw,
        walk_forward_summary=_walk_forward_summary(),
        asset_names=["Gold"],
        horizons=[1],
    )
    assert report.decision_table.iloc[0]["RegimeLabel"]


def test_missing_phase7g_columns_are_handled_with_warnings():
    incomplete = pd.DataFrame(
        [
            {
                "Asset": "Gold",
                "Horizon": 1,
                "WalkForwardVerdict": "Strong walk-forward research candidate",
            }
        ]
    )
    report = run_regime_aware_meta_signal(
        raw_df=_raw_prices(),
        walk_forward_summary=incomplete,
        asset_names=["Gold"],
        horizons=[1],
    )

    row = report.decision_table.iloc[0]
    assert row["MetaDecision"] != "Trade"
    assert "MissingPhase7GColumns" in row["Warnings"]
    assert float(row["MetaConfidenceScore"]) < 60.0


if __name__ == "__main__":
    test_meta_signal_runs_multi_asset_multi_horizon_and_keeps_rejections_visible()
    test_weak_walk_forward_rows_do_not_become_trade()
    test_strong_rows_can_only_trade_when_risk_benchmark_and_stability_allow()
    test_regime_features_do_not_require_future_columns()
    test_missing_phase7g_columns_are_handled_with_warnings()
    print("Phase 8 meta signal engine tests passed.")
