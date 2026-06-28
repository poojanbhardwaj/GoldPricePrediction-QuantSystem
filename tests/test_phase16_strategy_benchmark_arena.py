from pathlib import Path
import re
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import src.artifact_store as store
from src.artifact_store import load_latest_artifact
from src.asset_config import get_asset_names, get_target_column
from src.strategy_benchmark_arena import (
    ASSET_BENCHMARK_COLUMNS,
    ASSET_HORIZON_BENCHMARK_COLUMNS,
    BENCHMARK_DOMINANCE_COLUMNS,
    BENCHMARK_HORIZONS,
    BENCHMARK_INPUT_SOURCE_COLUMNS,
    BENCHMARK_SUMMARY_COLUMNS,
    BENCHMARK_WARNING_COLUMNS,
    COST_SENSITIVITY_COLUMNS,
    LEAKAGE_CHECK_COLUMNS,
    MODEL_STRENGTH_COLUMNS,
    NEXT_BENCHMARK_ACTION_COLUMNS,
    RANDOM_BASELINE_COLUMNS,
    RETURN_SANITY_CHECK_COLUMNS,
    SNAPSHOT_MODEL_IMPACT_COLUMNS,
    STRATEGY_BENCHMARK_PHASE_NAME,
    STRATEGY_LEADERBOARD_COLUMNS,
    run_strategy_benchmark_arena,
)


FORBIDDEN_LANGUAGE = re.compile(
    r"\b(Buy|Strong Buy|Invest Now|Production Ready|Guaranteed|Safe Profit)\b",
    flags=re.IGNORECASE,
)


def _with_temp_store(fn):
    old_root = store.ARTIFACT_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        store.ARTIFACT_ROOT = Path(tmp) / "artifacts"
        try:
            return fn()
        finally:
            store.ARTIFACT_ROOT = old_root


def _synthetic_market_data(rows=260, mode="mixed"):
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    x = np.arange(rows, dtype=float)
    data = pd.DataFrame({"Date": dates})
    if mode == "up":
        base = 100 + x * 0.25
        for asset in get_asset_names():
            data[get_target_column(asset)] = base + x * 0.01
    elif mode == "down":
        base = 160 - x * 0.2
        for asset in get_asset_names():
            data[get_target_column(asset)] = base - x * 0.01
    else:
        data["Gold_Close"] = 100 + x * 0.15
        data["Silver_Close"] = 150 - x * 0.12
        data["Oil_Close"] = 95 + np.sin(x / 10.0) * 2.0
        data["BTC_Close"] = 120 + np.sin(x / 4.0) * 18.0 + np.linspace(0, 20, rows)
        data["SP500_Close"] = 100 + x * 0.08
        data["GLD_Close"] = 50 + x * 0.06
    data["VIX_Close"] = 17 + np.sin(x / 20.0)
    data["DXY_Close"] = 104 - x * 0.002
    data["TNX_Close"] = 4.1 - x * 0.0005
    return data


def _phase15_sizing(weight=25.0):
    return pd.DataFrame(
        [
            {
                "Asset": asset,
                "Horizon": horizon,
                "Phase14OptimizedPaperWeightPct": weight,
                "RegimeSizingMultiplier": 1.0,
                "RegimeAdjustedPaperWeightPct": weight,
                "FinalRegimeSizingDecision": "KeepPhase14Size",
            }
            for asset in get_asset_names()
            for horizon in BENCHMARK_HORIZONS
        ]
    )


def _run_basic(**kwargs):
    params = {
        "market_data": _synthetic_market_data(),
        "use_project_market_data": False,
        "use_artifact_store": False,
        "assets": get_asset_names(),
        "horizons": BENCHMARK_HORIZONS,
        "regime_adjusted_sizing_table": _phase15_sizing(),
        "random_simulations": 25,
        "cost_bps": 5.0,
        "slippage_bps": 5.0,
    }
    params.update(kwargs)
    return run_strategy_benchmark_arena(**params)


def _all_output_text(report):
    frames = [
        report.benchmark_summary_table,
        report.strategy_leaderboard_table,
        report.asset_benchmark_table,
        report.asset_horizon_benchmark_table,
        report.benchmark_dominance_table,
        report.model_strength_table,
        report.cost_sensitivity_table,
        report.random_baseline_table,
        report.return_sanity_check_table,
        report.snapshot_model_impact_table,
        report.leakage_check_table,
        report.benchmark_warning_table,
        report.next_benchmark_actions_table,
        report.benchmark_input_sources_table,
    ]
    return "\n".join(frame.astype(str).to_csv(index=False) for frame in frames)


def test_phase16_outputs_all_required_tables_and_columns():
    report = _run_basic()
    expected = {
        "benchmark_summary_table": BENCHMARK_SUMMARY_COLUMNS,
        "strategy_leaderboard_table": STRATEGY_LEADERBOARD_COLUMNS,
        "asset_benchmark_table": ASSET_BENCHMARK_COLUMNS,
        "asset_horizon_benchmark_table": ASSET_HORIZON_BENCHMARK_COLUMNS,
        "benchmark_dominance_table": BENCHMARK_DOMINANCE_COLUMNS,
        "model_strength_table": MODEL_STRENGTH_COLUMNS,
        "cost_sensitivity_table": COST_SENSITIVITY_COLUMNS,
        "random_baseline_table": RANDOM_BASELINE_COLUMNS,
        "return_sanity_check_table": RETURN_SANITY_CHECK_COLUMNS,
        "snapshot_model_impact_table": SNAPSHOT_MODEL_IMPACT_COLUMNS,
        "leakage_check_table": LEAKAGE_CHECK_COLUMNS,
        "benchmark_warning_table": BENCHMARK_WARNING_COLUMNS,
        "next_benchmark_actions_table": NEXT_BENCHMARK_ACTION_COLUMNS,
        "benchmark_input_sources_table": BENCHMARK_INPUT_SOURCE_COLUMNS,
    }
    for table_name, columns in expected.items():
        table = getattr(report, table_name)
        assert set(columns).issubset(table.columns), table_name

    assert set(report.asset_benchmark_table["Asset"]) == set(get_asset_names())
    assert len(report.asset_horizon_benchmark_table) == len(get_asset_names()) * len(BENCHMARK_HORIZONS)


def test_phase16_hold_only_benchmark_computes_positive_directional_return():
    report = run_strategy_benchmark_arena(
        market_data=_synthetic_market_data(mode="up"),
        use_project_market_data=False,
        assets=["Gold"],
        horizons=[1],
        regime_adjusted_sizing_table=_phase15_sizing(weight=0.0),
        cost_bps=0.0,
        slippage_bps=0.0,
        random_simulations=5,
    )
    hold = report.strategy_leaderboard_table[
        report.strategy_leaderboard_table["StrategyName"].eq("HoldOnlyBenchmark")
    ].iloc[0]
    assert hold["NetReturnPct"] > 0


def test_phase16_hold_only_matches_simple_price_ratio_for_all_horizons():
    data = _synthetic_market_data(mode="up")
    report = run_strategy_benchmark_arena(
        market_data=data,
        use_project_market_data=False,
        assets=["Gold"],
        horizons=BENCHMARK_HORIZONS,
        regime_adjusted_sizing_table=_phase15_sizing(weight=0.0),
        cost_bps=0.0,
        slippage_bps=0.0,
        random_simulations=5,
    )
    price = data["Gold_Close"]
    expected = (price.iloc[-1] / price.iloc[0] - 1.0) * 100.0
    hold_rows = report.strategy_leaderboard_table[
        report.strategy_leaderboard_table["StrategyName"].eq("HoldOnlyBenchmark")
    ]

    assert len(hold_rows) == len(BENCHMARK_HORIZONS)
    assert (hold_rows["TotalReturnPct"] - expected).abs().max() < 0.05
    assert hold_rows["TotalReturnPct"].abs().max() < 100000.0


def test_phase16_return_sanity_checks_pass_for_valid_synthetic_data():
    report = _run_basic(cost_bps=0.0, slippage_bps=0.0)

    assert report.return_sanity_check_table["Passed"].all()


def test_phase16_return_sanity_checks_fail_for_impossible_return_data():
    data = _synthetic_market_data(mode="up")
    data.loc[len(data) - 1, "Gold_Close"] = data["Gold_Close"].iloc[0] * 1_000_000
    report = run_strategy_benchmark_arena(
        market_data=data,
        use_project_market_data=False,
        assets=["Gold"],
        horizons=[1],
        regime_adjusted_sizing_table=_phase15_sizing(weight=0.0),
        cost_bps=0.0,
        slippage_bps=0.0,
        random_simulations=5,
    )

    failed = report.return_sanity_check_table[~report.return_sanity_check_table["Passed"]]
    assert not failed.empty
    assert {"NoAstronomicalReturnExplosion", "PercentDecimalUnitsConsistent"} & set(failed["CheckName"])


def test_phase16_moving_average_and_momentum_are_time_safe():
    report = _run_basic()
    checks = report.leakage_check_table.set_index("CheckName")

    assert bool(checks.loc["Signals shifted before returns", "Passed"])
    assert bool(checks.loc["No future return used in feature/signal", "Passed"])


def test_phase16_random_baseline_is_reproducible_with_fixed_seed():
    first = _run_basic(random_seed=123)
    second = _run_basic(random_seed=123)

    pd.testing.assert_frame_equal(first.random_baseline_table, second.random_baseline_table)


def test_phase16_cost_sensitivity_reduces_returns_as_cost_increases():
    report = _run_basic(cost_scenarios_bps=[0, 5, 10, 25, 50])
    subset = report.cost_sensitivity_table[
        (report.cost_sensitivity_table["StrategyName"].eq("MovingAverageCrossover"))
        & (report.cost_sensitivity_table["Asset"].eq("Gold"))
        & (report.cost_sensitivity_table["Horizon"].eq(1))
    ].sort_values("CostBps")

    assert subset.iloc[-1]["NetReturnPct"] <= subset.iloc[0]["NetReturnPct"]


def test_phase16_snapshot_strategy_is_marked_snapshot_only():
    report = _run_basic()
    model_rows = report.strategy_leaderboard_table[
        report.strategy_leaderboard_table["StrategyName"].eq("Phase15RegimeAdjustedStrategy")
    ]

    assert model_rows["DataQualityFlag"].astype(str).str.contains("SnapshotOnly").all()
    assert model_rows["ComparableHistorical"].eq(False).all()
    assert set(model_rows["EvaluationMode"]) == {"LatestSnapshotOnly"}
    assert report.snapshot_model_impact_table["ComparableHistorical"].eq(False).all()
    assert report.leakage_check_table[
        report.leakage_check_table["CheckName"].eq("SnapshotStrategiesExcludedFromHistoricalWinner")
    ]["Passed"].iloc[0]


def test_phase16_snapshot_only_model_is_excluded_from_overall_winner():
    report = _run_basic()
    summary = report.benchmark_summary_table.iloc[0]

    assert not str(summary["OverallWinner"]).startswith("Phase")
    assert summary["BenchmarkVerdict"] == "InsufficientHistoricalModelEvidence"
    assert "historical replay" in summary["MainReason"]


def test_phase16_benchmark_dominated_when_hold_only_beats_model():
    report = run_strategy_benchmark_arena(
        market_data=_synthetic_market_data(mode="up"),
        use_project_market_data=False,
        assets=["Gold"],
        horizons=[1],
        regime_adjusted_sizing_table=_phase15_sizing(weight=0.0),
        cost_bps=0.0,
        slippage_bps=0.0,
        random_simulations=10,
    )

    assert not report.benchmark_dominance_table.empty


def test_phase16_snapshot_impact_visible_when_snapshot_matches_or_beats_baseline():
    report = run_strategy_benchmark_arena(
        market_data=_synthetic_market_data(mode="down"),
        use_project_market_data=False,
        assets=["Gold"],
        horizons=[1],
        regime_adjusted_sizing_table=_phase15_sizing(weight=0.0),
        cost_bps=0.0,
        slippage_bps=0.0,
        random_simulations=10,
    )

    assert not report.snapshot_model_impact_table.empty
    assert report.benchmark_summary_table.iloc[0]["BenchmarkVerdict"] == "InsufficientHistoricalModelEvidence"


def test_phase16_no_exposure_can_beat_losing_strategy():
    report = run_strategy_benchmark_arena(
        market_data=_synthetic_market_data(mode="down"),
        use_project_market_data=False,
        assets=["Gold"],
        horizons=[1],
        regime_adjusted_sizing_table=_phase15_sizing(weight=100.0),
        cost_bps=0.0,
        slippage_bps=0.0,
        random_simulations=10,
    )
    ah = report.asset_horizon_benchmark_table.iloc[0]

    assert ah["BestBaseline"] == "NoExposureBaseline"
    assert ah["ModelStrategyReturnPct"] < ah["BestBaselineReturnPct"]


def test_phase16_no_forbidden_live_trading_language_in_outputs():
    report = _run_basic()
    assert FORBIDDEN_LANGUAGE.search(_all_output_text(report)) is None


def test_phase16_autosaves_outputs_to_artifact_store():
    def run():
        report = run_strategy_benchmark_arena(
            market_data=_synthetic_market_data(),
            use_project_market_data=False,
            assets=["Gold", "Silver"],
            horizons=[1, 5],
            regime_adjusted_sizing_table=_phase15_sizing(weight=20.0),
            random_simulations=5,
            autosave=True,
        )
        assert report.saved_artifacts
        latest = load_latest_artifact(STRATEGY_BENCHMARK_PHASE_NAME, "benchmark_summary_table", required=True)
        assert not latest.empty

    _with_temp_store(run)
