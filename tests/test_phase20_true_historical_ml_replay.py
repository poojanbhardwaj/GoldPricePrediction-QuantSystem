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
from src.true_historical_ml_replay import (
    BASELINE_COMPARISON_COLUMNS,
    INPUT_SOURCE_COLUMNS,
    LEAKAGE_AUDIT_COLUMNS,
    NEXT_ACTION_COLUMNS,
    PERFORMANCE_COLUMNS,
    PREDICTION_LOG_COLUMNS,
    QUALITY_GATE_COLUMNS,
    STRENGTH_COLUMNS,
    SUMMARY_COLUMNS,
    TRUE_HISTORICAL_ML_REPLAY_PHASE_NAME,
    run_true_historical_ml_replay,
)


FORBIDDEN_LANGUAGE = re.compile(
    r"\b(Buy|Strong Buy|Invest Now|Guaranteed Profit|Safe Profit|Production Ready Trading)\b",
    flags=re.IGNORECASE,
)


def _synthetic_market(rows=190):
    dates = pd.date_range("2023-01-02", periods=rows, freq="B")
    t = np.arange(rows, dtype=float)
    data = pd.DataFrame({"Date": dates})
    profiles = {
        "Gold": 0.20,
        "Silver": -0.08,
        "Crude Oil": 0.03,
        "Bitcoin": 0.25,
        "S&P 500": 0.12,
        "Gold ETF": 0.17,
    }
    for offset, asset in enumerate(get_asset_names(), start=1):
        drift = profiles[asset]
        cycle = np.sin(t / (4.0 + offset)) * (1.0 + offset * 0.15)
        price = 100.0 + offset * 15.0 + drift * t + cycle
        data[get_target_column(asset)] = np.maximum(price, 1.0)
    return data


def _run_basic(**kwargs):
    params = {
        "market_data": _synthetic_market(),
        "use_project_market_data": False,
        "assets": ["Gold", "Bitcoin"],
        "horizons": [1, 5],
        "max_windows": 4,
        "min_train_rows": 40,
        "step_size": 15,
        "model_name": "Ridge",
        "cost_bps": 5.0,
        "slippage_bps": 5.0,
        "random_seed": 123,
    }
    params.update(kwargs)
    return run_true_historical_ml_replay(**params)


def _all_output_text(report):
    tables = [
        report.true_ml_summary_table,
        report.true_ml_prediction_log,
        report.true_ml_performance_table,
        report.true_ml_baseline_comparison_table,
        report.true_ml_strength_table,
        report.leakage_audit_table,
        report.input_sources_table,
        report.quality_gates_table,
        report.next_actions_table,
    ]
    return "\n".join(table.astype(str).to_csv(index=False) for table in tables)


def test_phase20_module_runs_and_required_tables_exist():
    report = _run_basic()
    expected = {
        "true_ml_summary_table": SUMMARY_COLUMNS,
        "true_ml_prediction_log": PREDICTION_LOG_COLUMNS,
        "true_ml_performance_table": PERFORMANCE_COLUMNS,
        "true_ml_baseline_comparison_table": BASELINE_COMPARISON_COLUMNS,
        "true_ml_strength_table": STRENGTH_COLUMNS,
        "leakage_audit_table": LEAKAGE_AUDIT_COLUMNS,
        "input_sources_table": INPUT_SOURCE_COLUMNS,
        "quality_gates_table": QUALITY_GATE_COLUMNS,
        "next_actions_table": NEXT_ACTION_COLUMNS,
    }
    for name, columns in expected.items():
        table = getattr(report, name)
        assert set(columns).issubset(table.columns), name


def test_phase20_prediction_log_is_nonempty_and_multi_asset():
    report = _run_basic()

    assert not report.true_ml_prediction_log.empty
    assert set(report.true_ml_prediction_log["Asset"]) == {"Gold", "Bitcoin"}
    assert set(report.true_ml_prediction_log["Horizon"]) == {1, 5}


def test_phase20_training_prediction_and_outcome_dates_are_ordered():
    report = _run_basic()
    log = report.true_ml_prediction_log

    assert (pd.to_datetime(log["TrainEndDate"]) < pd.to_datetime(log["PredictionDate"])).all()
    assert (pd.to_datetime(log["PredictionDate"]) < pd.to_datetime(log["TargetOutcomeDate"])).all()


def test_phase20_valid_data_has_no_leakage_audit_failures():
    report = _run_basic()
    audit = report.leakage_audit_table

    assert not audit.empty
    assert audit["LeakagePassed"].astype(bool).all()
    assert audit["ScalerFitTrainOnly"].astype(bool).all()
    assert audit["NoFutureRowsUsed"].astype(bool).all()
    assert audit["FutureTargetColumnsExcluded"].astype(bool).all()
    assert not audit["FeatureColumns"].str.contains("future_return_|future_direction_|future_realized_vol_", regex=True).any()


def test_phase20_handles_matured_and_pending_outcomes():
    report = _run_basic()
    log = report.true_ml_prediction_log

    assert log["IsMatured"].astype(bool).any()
    assert (~log["IsMatured"].astype(bool)).any()
    pending = log[~log["IsMatured"].astype(bool)]
    assert pending["RealizedReturnPct"].isna().all()
    assert pending["NetRealizedReturnPct"].isna().all()


def test_phase20_baseline_comparison_and_rejections_remain_visible():
    report = _run_basic()
    comparison = report.true_ml_baseline_comparison_table

    assert not comparison.empty
    assert comparison["BestBaselineName"].isin(
        ["NoExposure", "HoldOnly", "MomentumBaseline", "MovingAverageBaseline", "RandomMedianBaseline"]
    ).all()
    assert comparison["DominanceVerdict"].notna().all()


def test_phase20_supports_all_assets_without_asset_specific_logic():
    report = _run_basic(
        assets=get_asset_names(),
        horizons=[1],
        max_windows=2,
        model_name="SafeMean",
    )

    assert set(report.true_ml_prediction_log["Asset"]) == set(get_asset_names())


def test_phase20_real_capital_remains_blocked():
    report = _run_basic()

    assert set(report.true_ml_summary_table["RealCapitalStatus"]) == {"Blocked"}
    assert set(report.true_ml_strength_table["RealCapitalStatus"]) == {"Blocked"}
    gates = report.quality_gates_table.set_index("GateName")
    assert bool(gates.loc["RealCapitalBlocked", "Passed"])


def test_phase20_ridge_path_does_not_require_optional_heavy_models():
    report = _run_basic(model_name="Ridge")

    assert set(report.true_ml_prediction_log["ModelName"]) == {"Ridge"}


def test_phase20_autosaves_all_required_artifacts():
    old_root = store.ARTIFACT_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        store.ARTIFACT_ROOT = Path(tmp) / "artifacts"
        try:
            report = _run_basic(autosave=True)
            expected = {
                "phase20_true_ml_summary",
                "phase20_true_ml_prediction_log",
                "phase20_true_ml_performance",
                "phase20_true_ml_baseline_comparison",
                "phase20_true_ml_strength",
                "phase20_leakage_audit",
                "phase20_input_sources",
                "phase20_quality_gates",
                "phase20_next_actions",
            }
            assert expected.issubset(report.saved_artifacts["Artifacts"])
            latest = load_latest_artifact(TRUE_HISTORICAL_ML_REPLAY_PHASE_NAME, "phase20_true_ml_summary", required=True)
            assert latest.iloc[0]["ReplayType"] == "TrueHistoricalMLReplay"
        finally:
            store.ARTIFACT_ROOT = old_root


def test_phase20_output_has_no_forbidden_trading_language():
    report = _run_basic()

    assert FORBIDDEN_LANGUAGE.search(_all_output_text(report)) is None
