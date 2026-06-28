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
from src.historical_model_replay import (
    ASSET_HORIZON_MATRIX_COLUMNS,
    BENCHMARK_READY_COLUMNS,
    EXPOSURE_CAP_TOLERANCE,
    EXPOSURE_CAP_COLUMNS,
    HISTORICAL_REPLAY_PHASE_NAME,
    NEXT_REPLAY_ACTION_COLUMNS,
    OUTCOME_COLUMNS,
    PERFORMANCE_COLUMNS,
    PHASE16_EXPORT_COLUMNS,
    PORTFOLIO_CURVE_COLUMNS,
    QUALITY_CHECK_COLUMNS,
    REPLAY_HORIZONS,
    REPLAY_INPUT_SOURCE_COLUMNS,
    REPLAY_SUMMARY_COLUMNS,
    REPLAY_WARNING_COLUMNS,
    SIGNAL_LOG_COLUMNS,
    run_historical_model_replay,
)
import src.historical_model_replay as replay_module


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


def _synthetic_market_data(rows=240):
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    x = np.arange(rows, dtype=float)
    data = pd.DataFrame({"Date": dates})
    data["Gold_Close"] = 100 + x * 0.18
    data["Silver_Close"] = 130 - x * 0.08
    data["Oil_Close"] = 85 + np.sin(x / 9.0) * 2.5
    data["BTC_Close"] = 180 + np.sin(x / 4.0) * 12.0 + x * 0.08
    data["SP500_Close"] = 100 + x * 0.07
    data["GLD_Close"] = 50 + x * 0.05
    data["VIX_Close"] = 18 + np.sin(x / 20.0)
    data["DXY_Close"] = 104 - x * 0.003
    data["TNX_Close"] = 4.0 - x * 0.0008
    return data


def _run_basic(**kwargs):
    params = {
        "market_data": _synthetic_market_data(),
        "use_project_market_data": False,
        "use_artifact_store": False,
        "assets": get_asset_names(),
        "horizons": REPLAY_HORIZONS,
        "replay_step": 10,
        "max_paper_weight_pct": 20.0,
    }
    params.update(kwargs)
    return run_historical_model_replay(**params)


def _all_output_text(report):
    frames = [
        report.replay_summary_table,
        report.historical_replay_signal_log,
        report.historical_replay_outcomes,
        report.historical_replay_performance,
        report.historical_replay_portfolio_curve,
        report.replay_exposure_cap_table,
        report.replay_asset_horizon_matrix,
        report.replay_quality_checks,
        report.replay_benchmark_ready_table,
        report.replay_warnings_table,
        report.next_replay_actions_table,
        report.replay_input_sources_table,
        report.phase16_replay_export_table,
    ]
    return "\n".join(frame.astype(str).to_csv(index=False) for frame in frames)


def _replay_key(df, date_col="ReplayDate"):
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)
    return set(zip(dates, df["Asset"].astype(str), df["Horizon"].astype(int).astype(str)))


def test_phase17_outputs_all_required_tables_and_columns():
    report = _run_basic()
    expected = {
        "replay_summary_table": REPLAY_SUMMARY_COLUMNS,
        "historical_replay_signal_log": SIGNAL_LOG_COLUMNS,
        "historical_replay_outcomes": OUTCOME_COLUMNS,
        "historical_replay_performance": PERFORMANCE_COLUMNS,
        "historical_replay_portfolio_curve": PORTFOLIO_CURVE_COLUMNS,
        "replay_asset_horizon_matrix": ASSET_HORIZON_MATRIX_COLUMNS,
        "replay_exposure_cap_table": EXPOSURE_CAP_COLUMNS,
        "replay_quality_checks": QUALITY_CHECK_COLUMNS,
        "replay_benchmark_ready_table": BENCHMARK_READY_COLUMNS,
        "replay_warnings_table": REPLAY_WARNING_COLUMNS,
        "next_replay_actions_table": NEXT_REPLAY_ACTION_COLUMNS,
        "replay_input_sources_table": REPLAY_INPUT_SOURCE_COLUMNS,
        "phase16_replay_export_table": PHASE16_EXPORT_COLUMNS,
    }
    for table_name, columns in expected.items():
        table = getattr(report, table_name)
        assert set(columns).issubset(table.columns), table_name

    assert set(report.replay_asset_horizon_matrix["Asset"]) == set(get_asset_names())
    assert set(report.replay_asset_horizon_matrix["Horizon"].astype(int)) == set(REPLAY_HORIZONS)


def test_phase17_exposure_cap_scales_multiple_active_signals_per_date():
    data = _synthetic_market_data(rows=190)
    replay_date = data["Date"].iloc[120]
    prediction_log = pd.DataFrame(
        [
            {
                "ReplayDate": replay_date,
                "Asset": asset,
                "Horizon": horizon,
                "ProbabilityUp": 0.7,
                "ReplayPaperWeightPct": 5.0,
            }
            for asset in get_asset_names()
            for horizon in REPLAY_HORIZONS
        ]
    )

    report = _run_basic(
        market_data=data,
        historical_prediction_log=prediction_log,
        max_paper_weight_pct=5.0,
        max_portfolio_paper_exposure_pct=45.0,
    )
    cap_row = report.replay_exposure_cap_table.iloc[0]

    assert cap_row["ExposureBeforeCapPct"] == 150.0
    assert abs(cap_row["ExposureAfterCapPct"] - 45.0) <= EXPOSURE_CAP_TOLERANCE
    assert bool(cap_row["CapApplied"])
    assert abs(cap_row["ScalingFactor"] - 0.3) < 1e-6
    assert pd.to_numeric(report.historical_replay_signal_log["PaperWeightPct"], errors="coerce").max() <= 1.5 + 1e-6
    assert pd.to_numeric(report.historical_replay_portfolio_curve["PortfolioPaperExposurePct"], errors="coerce").max() <= 45.0 + 1e-6
    assert (
        pd.to_numeric(report.replay_exposure_cap_table["ExposureAfterCapPct"], errors="coerce")
        <= pd.to_numeric(report.replay_exposure_cap_table["MaxPortfolioPaperExposurePct"], errors="coerce") + EXPOSURE_CAP_TOLERANCE
    ).all()
    assert "PortfolioExposureScaled" in set(report.replay_warnings_table["WarningType"])


def test_phase17_summary_exposure_matches_capped_portfolio_curve():
    report = _run_basic(max_portfolio_paper_exposure_pct=45.0)
    summary = report.replay_summary_table.iloc[0]
    curve = report.historical_replay_portfolio_curve

    expected_avg = round(float(pd.to_numeric(curve["PortfolioPaperExposurePct"], errors="coerce").fillna(0.0).mean()), 4)
    expected_max = round(float(pd.to_numeric(curve["PortfolioPaperExposurePct"], errors="coerce").fillna(0.0).max()), 4)

    assert summary["AveragePortfolioPaperExposurePct"] == expected_avg
    assert summary["AveragePaperExposurePct"] == expected_avg
    assert summary["MaxPortfolioPaperExposurePct"] == expected_max
    assert int(summary["ExposureCapBreachesAfterScaling"]) == 0
    assert (
        pd.to_numeric(report.replay_exposure_cap_table["ExposureAfterCapPct"], errors="coerce")
        <= pd.to_numeric(report.replay_exposure_cap_table["MaxPortfolioPaperExposurePct"], errors="coerce") + EXPOSURE_CAP_TOLERANCE
    ).all()


def test_phase17_proxy_replay_is_clearly_labeled_without_prediction_log():
    report = _run_basic()
    summary = report.replay_summary_table.iloc[0]

    assert summary["ReplaySource"] == "HistoricalSignalProxyReplay"
    assert summary["ModelReplayQuality"] == "ProxyOnly"
    assert summary["ReplayVerdict"] == "ProxyReplayOnly"
    assert "proxy" in str(summary["MainLimitation"]).lower()
    assert "ProxyOnlyReplay" in set(report.replay_warnings_table["WarningType"])


def test_phase17_historical_prediction_log_drives_historical_replay_source():
    data = _synthetic_market_data(rows=180)
    pred = pd.DataFrame(
        [
            {
                "ReplayDate": data["Date"].iloc[120],
                "Asset": "Gold",
                "Horizon": 5,
                "ProbabilityUp": 0.66,
                "ReplayPaperWeightPct": 3.0,
            }
        ]
    )

    report = _run_basic(
        market_data=data,
        assets=["Gold"],
        horizons=[5],
        historical_prediction_log=pred,
    )
    signal = report.historical_replay_signal_log.iloc[0]

    assert report.replay_summary_table.iloc[0]["ReplaySource"] == "HistoricalModelPredictionReplay"
    assert signal["ReplaySource"] == "HistoricalModelPredictionReplay"
    assert signal["SignalScore"] == 66.0
    assert signal["PaperWeightPct"] == 3.0
    assert report.historical_replay_outcomes.iloc[0]["OutcomeStatus"] == "Matured"


def test_phase17_signals_use_past_data_only_and_quality_checks_exist():
    report = _run_basic()
    signals = report.historical_replay_signal_log
    checks = report.replay_quality_checks.set_index("CheckName")

    assert not signals.empty
    assert (pd.to_datetime(signals["DataAvailableThroughDate"]) <= pd.to_datetime(signals["ReplayDate"])).all()
    for check_name in [
        "NoFutureDataUsed",
        "SignalsUsePastDataOnly",
        "OutcomesAfterReplayDate",
        "NoSameDayCloseLeakage",
        "DateOrderingValid",
        "NoDuplicateReplayRows",
        "HorizonOutcomeAlignmentValid",
        "LatestSnapshotNotUsedAsHistoricalWeights",
        "ReplaySourceClearlyLabeled",
        "SufficientHistoryBeforeSignal",
        "PortfolioExposureCapRespected",
        "PortfolioExposureScaledWhenNeeded",
        "SummaryExposureMatchesPortfolioCurve",
        "Phase16ExportUsesCappedWeights",
    ]:
        assert check_name in checks.index
    assert bool(checks.loc["NoFutureDataUsed", "Passed"])
    assert bool(checks.loc["SignalsUsePastDataOnly", "Passed"])
    assert bool(checks.loc["LatestSnapshotNotUsedAsHistoricalWeights", "Passed"])


def test_phase17_matured_outcomes_occur_after_replay_date():
    report = _run_basic()
    outcomes = report.historical_replay_outcomes
    matured = outcomes[outcomes["OutcomeStatus"].eq("Matured")]

    assert not matured.empty
    assert (pd.to_datetime(matured["OutcomeDate"]) > pd.to_datetime(matured["ReplayDate"])).all()
    assert (pd.to_numeric(matured["RealWeightPct"], errors="coerce") == 0).all()


def test_phase17_phase16_export_marks_only_matured_rows_comparable():
    report = _run_basic()
    export = report.phase16_replay_export_table
    matured = report.historical_replay_outcomes[report.historical_replay_outcomes["OutcomeStatus"].eq("Matured")]
    comparable = export[export["ComparableHistorical"].eq(True)]

    assert not export.empty
    assert _replay_key(comparable, date_col="Date") == _replay_key(matured)
    assert comparable["EvaluationMode"].eq("HistoricalDailyExposure").all()
    assert export[export["ComparableHistorical"].eq(False)]["EvaluationMode"].eq("InsufficientData").all()


def test_phase17_phase16_export_uses_capped_weights_and_proxy_strategy_name():
    data = _synthetic_market_data(rows=190)
    replay_date = data["Date"].iloc[120]
    prediction_log = pd.DataFrame(
        [
            {
                "ReplayDate": replay_date,
                "Asset": asset,
                "Horizon": horizon,
                "ProbabilityUp": 0.7,
                "ReplayPaperWeightPct": 5.0,
            }
            for asset in get_asset_names()
            for horizon in REPLAY_HORIZONS
        ]
    )
    report = _run_basic(
        market_data=data,
        historical_prediction_log=prediction_log.iloc[0:0],
        max_paper_weight_pct=5.0,
        max_portfolio_paper_exposure_pct=45.0,
    )
    proxy_report = _run_basic(max_paper_weight_pct=5.0, max_portfolio_paper_exposure_pct=45.0)

    assert set(proxy_report.phase16_replay_export_table["StrategyName"]) == {"HistoricalSignalProxyReplay"}
    assert pd.to_numeric(report.phase16_replay_export_table["ExposurePct"], errors="coerce").max() <= 5.0

    capped_report = _run_basic(
        market_data=data,
        historical_prediction_log=prediction_log,
        max_paper_weight_pct=5.0,
        max_portfolio_paper_exposure_pct=45.0,
    )
    assert pd.to_numeric(capped_report.phase16_replay_export_table["ExposurePct"], errors="coerce").max() <= 1.5 + 1e-6
    quality = capped_report.replay_quality_checks.set_index("CheckName")
    assert bool(quality.loc["Phase16ExportUsesCappedWeights", "Passed"])


def test_phase17_benchmark_ready_requires_quality_and_respected_cap():
    report = _run_basic(max_paper_weight_pct=5.0, max_portfolio_paper_exposure_pct=45.0)
    ready = report.replay_benchmark_ready_table.iloc[0]

    assert ready["StrategyName"] == "HistoricalSignalProxyReplay"
    assert bool(ready["BenchmarkReady"])
    assert "historical proxy" in str(ready["Reason"]).lower()

    failed_quality = pd.DataFrame(
        [{"CheckName": "PortfolioExposureCapRespected", "Passed": False, "Severity": "Critical", "AffectedRows": 1, "Explanation": "breach"}]
    )
    blocked = replay_module._benchmark_ready(
        report.historical_replay_signal_log,
        report.historical_replay_outcomes,
        "HistoricalSignalProxyReplay",
        failed_quality,
    )
    assert not bool(blocked.iloc[0]["BenchmarkReady"])


def test_phase17_real_weight_is_zero_and_paper_weight_respects_cap():
    cap = 7.5
    report = _run_basic(max_paper_weight_pct=cap)
    signals = report.historical_replay_signal_log
    outcomes = report.historical_replay_outcomes

    assert (pd.to_numeric(signals["RealWeightPct"], errors="coerce") == 0).all()
    assert (pd.to_numeric(outcomes["RealWeightPct"], errors="coerce") == 0).all()
    assert pd.to_numeric(signals["PaperWeightPct"], errors="coerce").between(0, cap).all()


def test_phase17_missing_exit_price_stays_visible():
    data = _synthetic_market_data(rows=180)
    replay_date = pd.to_datetime(data["Date"].iloc[120])
    data.loc[125, get_target_column("Gold")] = np.nan

    report = _run_basic(
        market_data=data,
        assets=["Gold"],
        horizons=[5],
        replay_start_date=replay_date,
        replay_end_date=replay_date,
        replay_step=1,
    )

    assert "MissingExitPrice" in set(report.historical_replay_outcomes["OutcomeStatus"])
    assert "MissingExitPrice" in set(report.replay_warnings_table["WarningType"])


def test_phase17_autosaves_replay_artifacts():
    def run():
        report = _run_basic(autosave=True, assets=["Gold", "Bitcoin"], horizons=[1, 5])
        latest = load_latest_artifact(HISTORICAL_REPLAY_PHASE_NAME, "replay_summary_table", required=True)

        assert report.saved_artifacts["RunId"]
        assert latest["ReplaySource"].iloc[0] == report.replay_summary_table["ReplaySource"].iloc[0]

    _with_temp_store(run)


def test_phase17_output_has_no_forbidden_live_trading_language():
    report = _run_basic()

    assert not FORBIDDEN_LANGUAGE.search(_all_output_text(report))
