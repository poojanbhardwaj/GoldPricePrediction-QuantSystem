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
from src.replay_benchmark_audit import (
    REPLAY_ASSET_EDGE_COLUMNS,
    REPLAY_ASSET_HORIZON_EDGE_COLUMNS,
    REPLAY_BENCHMARK_AUDIT_PHASE_NAME,
    REPLAY_BENCHMARK_SUMMARY_COLUMNS,
    REPLAY_COST_ROBUSTNESS_COLUMNS,
    REPLAY_DOMINANCE_FAILURE_COLUMNS,
    REPLAY_DRAWDOWN_COMPARISON_COLUMNS,
    REPLAY_INPUT_SOURCE_COLUMNS,
    REPLAY_NEXT_ACTION_COLUMNS,
    REPLAY_QUALITY_GATE_COLUMNS,
    REPLAY_RANDOM_COMPARISON_COLUMNS,
    REPLAY_REAL_CAPITAL_READINESS_COLUMNS,
    REPLAY_STRENGTH_COLUMNS,
    REPLAY_VS_BASELINE_LEADERBOARD_COLUMNS,
    run_replay_benchmark_audit,
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


def _market(rows=90):
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = pd.DataFrame({"Date": dates})
    for asset in get_asset_names():
        data[get_target_column(asset)] = 100.0
    data["VIX_Close"] = 18.0
    return data


def _replay_export(rows=60, bad_label=False):
    dates = pd.date_range("2024-01-10", periods=rows, freq="D")
    out = []
    for dt in dates:
        out.append(
            {
                "Date": dt,
                "Asset": "Gold",
                "Horizon": 5,
                "StrategyName": "HistoricalModelRiskReplay" if bad_label else "HistoricalSignalProxyReplay",
                "ExposurePct": 10.0,
                "DailyReturnPct": 1.0,
                "StrategyReturnPct": 0.1,
                "EvaluationMode": "HistoricalDailyExposure",
                "ComparableHistorical": True,
                "ReplaySource": "HistoricalSignalProxyReplay",
            }
        )
        out.append(
            {
                "Date": dt,
                "Asset": "Silver",
                "Horizon": 5,
                "StrategyName": "HistoricalSignalProxyReplay",
                "ExposurePct": 10.0,
                "DailyReturnPct": -1.0,
                "StrategyReturnPct": -0.1,
                "EvaluationMode": "HistoricalDailyExposure",
                "ComparableHistorical": True,
                "ReplaySource": "HistoricalSignalProxyReplay",
            }
        )
    return pd.DataFrame(out)


def _phase17_quality(passed=True):
    names = [
        "NoFutureDataUsed",
        "OutcomesAfterReplayDate",
        "NoSameDayCloseLeakage",
        "PortfolioExposureCapRespected",
        "Phase16ExportUsesCappedWeights",
    ]
    return pd.DataFrame(
        [{"CheckName": name, "Passed": bool(passed), "Severity": "Critical" if not passed else "High", "AffectedRows": 0, "Explanation": "synthetic"} for name in names]
    )


def _phase17_summary():
    return pd.DataFrame(
        [
            {
                "ReplaySource": "HistoricalSignalProxyReplay",
                "ModelReplayQuality": "ProxyOnly",
                "ReplayVerdict": "ProxyReplayOnly",
                "ReplayRows": 120,
                "MaturedOutcomeRows": 120,
            }
        ]
    )


def _cap_table(breach=False):
    dates = pd.date_range("2024-01-10", periods=5, freq="D")
    return pd.DataFrame(
        [
            {
                "ReplayDate": dt,
                "ExposureBeforeCapPct": 20.0,
                "ExposureAfterCapPct": 60.0 if breach else 20.0,
                "MaxPortfolioPaperExposurePct": 45.0,
                "CapApplied": bool(breach),
                "ScalingFactor": 1.0,
                "ActiveSignalsBeforeCap": 2,
                "ActiveSignalsAfterCap": 2,
                "AdjustmentReason": "synthetic",
            }
            for dt in dates
        ]
    )


def _run_basic(**kwargs):
    params = {
        "market_data": _market(),
        "use_project_market_data": False,
        "use_artifact_store": False,
        "assets": ["Gold", "Silver"],
        "horizons": [5],
        "phase16_replay_export_table": _replay_export(),
        "replay_summary_table": _phase17_summary(),
        "replay_quality_checks": _phase17_quality(),
        "replay_exposure_cap_table": _cap_table(),
        "random_seed": 123,
        "random_simulations": 25,
        "cost_bps": 0.0,
        "slippage_bps": 0.0,
        "min_trades": 3,
        "min_matured_rows": 10,
    }
    params.update(kwargs)
    return run_replay_benchmark_audit(**params)


def _all_output_text(report):
    frames = [
        report.replay_benchmark_summary_table,
        report.replay_vs_baseline_leaderboard,
        report.replay_asset_edge_table,
        report.replay_asset_horizon_edge_table,
        report.replay_dominance_failures_table,
        report.replay_strength_table,
        report.replay_random_comparison_table,
        report.replay_cost_robustness_table,
        report.replay_drawdown_comparison_table,
        report.replay_quality_gate_table,
        report.replay_real_capital_readiness_table,
        report.replay_next_actions_table,
        report.replay_benchmark_input_sources_table,
    ]
    return "\n".join(frame.astype(str).to_csv(index=False) for frame in frames)


def test_phase18_outputs_all_required_tables_and_columns():
    report = _run_basic()
    expected = {
        "replay_benchmark_summary_table": REPLAY_BENCHMARK_SUMMARY_COLUMNS,
        "replay_vs_baseline_leaderboard": REPLAY_VS_BASELINE_LEADERBOARD_COLUMNS,
        "replay_asset_edge_table": REPLAY_ASSET_EDGE_COLUMNS,
        "replay_asset_horizon_edge_table": REPLAY_ASSET_HORIZON_EDGE_COLUMNS,
        "replay_dominance_failures_table": REPLAY_DOMINANCE_FAILURE_COLUMNS,
        "replay_strength_table": REPLAY_STRENGTH_COLUMNS,
        "replay_random_comparison_table": REPLAY_RANDOM_COMPARISON_COLUMNS,
        "replay_cost_robustness_table": REPLAY_COST_ROBUSTNESS_COLUMNS,
        "replay_drawdown_comparison_table": REPLAY_DRAWDOWN_COMPARISON_COLUMNS,
        "replay_quality_gate_table": REPLAY_QUALITY_GATE_COLUMNS,
        "replay_real_capital_readiness_table": REPLAY_REAL_CAPITAL_READINESS_COLUMNS,
        "replay_next_actions_table": REPLAY_NEXT_ACTION_COLUMNS,
        "replay_benchmark_input_sources_table": REPLAY_INPUT_SOURCE_COLUMNS,
    }
    for name, columns in expected.items():
        table = getattr(report, name)
        assert set(columns).issubset(table.columns), name


def test_phase18_keeps_all_configured_asset_horizon_rows_visible():
    report = _run_basic(assets=get_asset_names(), horizons=[1, 5, 10, 20, 30])

    assert len(report.replay_asset_horizon_edge_table) == len(get_asset_names()) * 5
    missing = report.replay_asset_horizon_edge_table[
        ~(
            report.replay_asset_horizon_edge_table["Asset"].isin(["Gold", "Silver"])
            & report.replay_asset_horizon_edge_table["Horizon"].eq(5)
        )
    ]
    assert not missing.empty
    assert set(missing["EdgeVerdict"]).issubset({"InsufficientData", "InsufficientTrades", "BenchmarkDominated", "ProxyMixed"})


def test_phase18_reports_proxy_strength_and_dominance_failures():
    report = _run_basic()

    strength = report.replay_strength_table
    dominance = report.replay_dominance_failures_table
    assert not strength[strength["Asset"].eq("Gold") & strength["Horizon"].eq(5)].empty
    assert not dominance[dominance["Asset"].eq("Silver") & dominance["Horizon"].eq(5)].empty

    silver_edge = report.replay_asset_horizon_edge_table[
        report.replay_asset_horizon_edge_table["Asset"].eq("Silver")
    ].iloc[0]
    assert not bool(silver_edge["ProxyBeatsNoExposure"])
    assert silver_edge["EdgeVerdict"] == "BenchmarkDominated"


def test_phase18_random_comparison_is_reproducible():
    first = _run_basic(random_seed=777)
    second = _run_basic(random_seed=777)

    pd.testing.assert_frame_equal(first.replay_random_comparison_table, second.replay_random_comparison_table)


def test_phase18_non_positive_price_transitions_are_counted_and_skipped():
    market = _market(rows=12)
    market[get_target_column("Gold")] = [100.0, 102.0, 0.0, 104.0, -5.0, 106.0, 108.0, 107.0, 109.0, 111.0, 110.0, 112.0]
    replay = _replay_export()
    report = _run_basic(
        market_data=market,
        assets=["Gold"],
        horizons=[5],
        phase16_replay_export_table=replay[replay["Asset"].eq("Gold")],
    )

    baseline_rows = report.replay_vs_baseline_leaderboard[
        report.replay_vs_baseline_leaderboard["BenchmarkRole"].isin(["SimpleBaseline", "RandomBaseline"])
    ]
    assert baseline_rows["InvalidReturnRows"].max() > 0
    assert baseline_rows["DataQualityFlag"].astype(str).str.contains("NonPositivePriceReturnInvalid").any()
    assert baseline_rows["ReturnSanityStatus"].astype(str).str.contains("InvalidReturnRowsSkipped").any()
    assert not (
        (pd.to_numeric(baseline_rows["NetReturnPct"], errors="coerce") < -100.0)
        & baseline_rows["DataQualityFlag"].astype(str).eq("")
    ).any()

    gates = report.replay_quality_gate_table.set_index("GateName")
    assert bool(gates.loc["NonPositivePriceHandlingApplied", "Passed"])
    assert bool(gates.loc["BaselineReturnSanityPassed", "Passed"])


def test_phase18_baseline_sanity_fails_when_no_clean_price_returns_exist():
    market = _market(rows=8)
    market[get_target_column("Gold")] = [0.0, -1.0, 0.0, -2.0, 0.0, -3.0, 0.0, -4.0]
    replay = _replay_export()
    report = _run_basic(
        market_data=market,
        assets=["Gold"],
        horizons=[5],
        phase16_replay_export_table=replay[replay["Asset"].eq("Gold")],
    )

    gates = report.replay_quality_gate_table.set_index("GateName")
    assert not bool(gates.loc["BaselineReturnSanityPassed", "Passed"])
    assert "InsufficientCleanReturnData" in str(gates.loc["BaselineReturnSanityPassed", "Explanation"])

    baseline_rows = report.replay_vs_baseline_leaderboard[
        report.replay_vs_baseline_leaderboard["BenchmarkRole"].isin(["SimpleBaseline", "RandomBaseline"])
    ]
    assert baseline_rows["ReturnSanityStatus"].eq("InsufficientCleanReturnData").all()


def test_phase18_random_baseline_leaderboard_uses_representative_metrics():
    market = _market(rows=120)
    trend = 100.0 + np.sin(np.arange(120) / 3.0) * 3.0 + np.arange(120) * 0.08
    market[get_target_column("Gold")] = trend
    replay = _replay_export()
    report = _run_basic(
        market_data=market,
        assets=["Gold"],
        horizons=[5],
        phase16_replay_export_table=replay[replay["Asset"].eq("Gold")],
        random_seed=444,
        random_simulations=50,
    )

    random_row = report.replay_vs_baseline_leaderboard[
        report.replay_vs_baseline_leaderboard["StrategyName"].eq("RandomBaseline")
    ].iloc[0]
    assert random_row["ComparableHistorical"] is True or bool(random_row["ComparableHistorical"])
    assert random_row["TradeCount"] > 0
    assert np.isfinite(float(random_row["MaxDrawdownPct"]))
    assert np.isfinite(float(random_row["WinRatePct"]))
    assert not (
        float(random_row["NetReturnPct"]) < 0
        and float(random_row["MaxDrawdownPct"]) == 0
        and float(random_row["WinRatePct"]) == 0
    )

    gates = report.replay_quality_gate_table.set_index("GateName")
    assert bool(gates.loc["RandomBaselineMetricsValid", "Passed"])


def test_phase18_cost_robustness_worsens_as_costs_increase():
    report = _run_basic(cost_scenarios_bps=[0, 5, 10, 25, 50])
    gold = report.replay_cost_robustness_table[
        report.replay_cost_robustness_table["Asset"].eq("Gold")
    ].sort_values("CostBps")

    assert gold.iloc[-1]["ProxyNetReturnPct"] <= gold.iloc[0]["ProxyNetReturnPct"]


def test_phase18_quality_gates_fail_for_mislabel_and_exposure_breach():
    mislabeled = _run_basic(phase16_replay_export_table=_replay_export(bad_label=True))
    gates = mislabeled.replay_quality_gate_table.set_index("GateName")
    assert not bool(gates.loc["ProxyNotMisrepresentedAsML", "Passed"])
    assert "mislabeled as historical model evidence" in str(gates.loc["ProxyNotMisrepresentedAsML", "Explanation"])

    breached = _run_basic(replay_exposure_cap_table=_cap_table(breach=True))
    breach_gates = breached.replay_quality_gate_table.set_index("GateName")
    assert not bool(breach_gates.loc["Phase17ExposureCapRespected", "Passed"])
    assert breached.replay_benchmark_summary_table.iloc[0]["ReplayBenchmarkVerdict"] == "ReplayQualityFailed"


def test_phase18_valid_proxy_comparable_rows_pass_mislabel_gate():
    report = _run_basic()
    gates = report.replay_quality_gate_table.set_index("GateName")
    replay_rows = report.replay_vs_baseline_leaderboard[report.replay_vs_baseline_leaderboard["BenchmarkRole"].eq("ReplayProxy")]

    assert bool(gates.loc["ProxyNotMisrepresentedAsML", "Passed"])
    assert replay_rows["ComparableHistorical"].eq(True).all()
    assert set(replay_rows["StrategyName"]) == {"HistoricalSignalProxyReplay"}
    assert set(replay_rows["ReplaySource"]) == {"HistoricalSignalProxyReplay"}


def test_phase18_proxy_is_not_misrepresented_and_real_capital_stays_blocked():
    report = _run_basic()

    assert set(report.replay_vs_baseline_leaderboard[report.replay_vs_baseline_leaderboard["BenchmarkRole"].eq("ReplayProxy")]["StrategyName"]) == {
        "HistoricalSignalProxyReplay"
    }
    assert "ConditionalCandidate" not in set(report.replay_real_capital_readiness_table["RealCapitalReadiness"])
    assert set(report.replay_real_capital_readiness_table["RecommendedMode"]).issubset({"Paper research only", "Research review only"})


def test_phase18_autosaves_outputs_to_artifact_store():
    def run():
        report = _run_basic(autosave=True)
        latest = load_latest_artifact(REPLAY_BENCHMARK_AUDIT_PHASE_NAME, "replay_benchmark_summary_table", required=True)
        assert report.saved_artifacts["RunId"]
        assert latest["ReplayBenchmarkVerdict"].iloc[0] == report.replay_benchmark_summary_table["ReplayBenchmarkVerdict"].iloc[0]

    _with_temp_store(run)


def test_phase18_output_has_no_forbidden_live_trading_language():
    report = _run_basic()

    assert not FORBIDDEN_LANGUAGE.search(_all_output_text(report))
