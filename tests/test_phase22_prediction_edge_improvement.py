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
from src.prediction_edge_improvement import (
    ASSET_HORIZON_SCORECARD_COLUMNS,
    BASELINE_COMPARISON_COLUMNS,
    COST_SENSITIVITY_COLUMNS,
    FEATURE_GROUP_AUDIT_COLUMNS,
    INPUT_SOURCE_COLUMNS,
    LEAKAGE_AUDIT_COLUMNS,
    MODEL_LEADERBOARD_COLUMNS,
    MODEL_SELECTION_AUDIT_COLUMNS,
    NEXT_ACTION_COLUMNS,
    OPTIONAL_MODELS,
    PREDICTION_EDGE_IMPROVEMENT_PHASE_NAME,
    PREDICTION_LOG_COLUMNS,
    QUALITY_GATE_COLUMNS,
    REJECTED_MODEL_COLUMNS,
    SUMMARY_COLUMNS,
    run_prediction_edge_improvement,
)


FORBIDDEN_LANGUAGE = re.compile(
    r"\b(Buy|Strong Buy|Invest Now|Guaranteed Profit|Safe Profit|Production Ready Trading)\b",
    flags=re.IGNORECASE,
)


def _synthetic_market(rows=205):
    dates = pd.date_range("2022-01-03", periods=rows, freq="B")
    t = np.arange(rows, dtype=float)
    data = pd.DataFrame({"Date": dates})
    drifts = {
        "Gold": 0.18,
        "Silver": -0.10,
        "Crude Oil": 0.03,
        "Bitcoin": 0.22,
        "S&P 500": 0.11,
        "Gold ETF": 0.16,
    }
    for offset, asset in enumerate(get_asset_names(), start=1):
        cycle = np.sin(t / (3.5 + offset)) * (0.8 + offset * 0.2)
        secondary = np.cos(t / (10.0 + offset)) * 0.5
        price = 100.0 + offset * 20.0 + drifts[asset] * t + cycle + secondary
        data[get_target_column(asset)] = np.maximum(price, 2.0)
    return data


def _run_basic(**kwargs):
    params = {
        "market_data": _synthetic_market(),
        "use_project_market_data": False,
        "assets": ["Gold", "Silver"],
        "horizons": [1, 5],
        "max_windows": 4,
        "min_train_rows": 40,
        "step_size": 15,
        "cost_bps": 5.0,
        "slippage_bps": 5.0,
        "random_seed": 123,
        "models_to_test": ["Ridge", "ElasticNet", "LinearRegression"],
        "feature_groups": ["PriceReturn", "TechnicalIndicators"],
        "enable_ensemble": True,
    }
    params.update(kwargs)
    return run_prediction_edge_improvement(**params)


def _all_output_text(report):
    tables = [
        report.prediction_edge_summary,
        report.model_leaderboard,
        report.asset_horizon_model_scorecard,
        report.prediction_log,
        report.baseline_comparison,
        report.feature_group_audit,
        report.model_selection_audit,
        report.leakage_audit,
        report.cost_sensitivity,
        report.rejected_models,
        report.quality_gates,
        report.next_actions,
        report.input_sources,
    ]
    return "\n".join(table.astype(str).to_csv(index=False) for table in tables)


def test_phase22_module_runs_and_all_required_tables_exist():
    report = _run_basic()
    expected = {
        "prediction_edge_summary": SUMMARY_COLUMNS,
        "model_leaderboard": MODEL_LEADERBOARD_COLUMNS,
        "asset_horizon_model_scorecard": ASSET_HORIZON_SCORECARD_COLUMNS,
        "prediction_log": PREDICTION_LOG_COLUMNS,
        "baseline_comparison": BASELINE_COMPARISON_COLUMNS,
        "feature_group_audit": FEATURE_GROUP_AUDIT_COLUMNS,
        "model_selection_audit": MODEL_SELECTION_AUDIT_COLUMNS,
        "leakage_audit": LEAKAGE_AUDIT_COLUMNS,
        "cost_sensitivity": COST_SENSITIVITY_COLUMNS,
        "rejected_models": REJECTED_MODEL_COLUMNS,
        "quality_gates": QUALITY_GATE_COLUMNS,
        "next_actions": NEXT_ACTION_COLUMNS,
        "input_sources": INPUT_SOURCE_COLUMNS,
    }
    for name, columns in expected.items():
        assert set(columns).issubset(getattr(report, name).columns), name


def test_phase22_synthetic_multi_asset_replay_evaluates_multiple_models():
    report = _run_basic()

    assert not report.prediction_log.empty
    assert set(report.prediction_log["Asset"]) == {"Gold", "Silver"}
    assert {"Ridge", "ElasticNet", "LinearRegression"}.issubset(set(report.model_leaderboard["ModelName"]))
    assert report.model_leaderboard["FeatureGroup"].nunique() >= 2


def test_phase22_supports_all_configured_assets():
    report = _run_basic(
        assets=get_asset_names(),
        horizons=[1],
        max_windows=2,
        models_to_test=["Ridge"],
        feature_groups=["PriceReturn"],
        enable_ensemble=False,
    )

    assert set(report.asset_horizon_model_scorecard["Asset"]) == set(get_asset_names())


def test_phase22_optional_model_is_used_or_skipped_with_visible_reason():
    report = _run_basic(
        assets=["Gold"],
        horizons=[1],
        max_windows=2,
        models_to_test=["Ridge"],
        feature_groups=["PriceReturn"],
        enable_ensemble=False,
        enable_optional_models=True,
    )

    summary = report.prediction_edge_summary.iloc[0]
    available = {value.strip() for value in str(summary["OptionalModelsAvailable"]).split(";") if value.strip()}
    leaderboard_models = set(report.model_leaderboard["ModelName"].astype(str))
    rejected_models = set(report.rejected_models["ModelName"].astype(str))
    for model in OPTIONAL_MODELS:
        if model in available:
            assert model in leaderboard_models or model in rejected_models
        else:
            rejected = report.rejected_models[report.rejected_models["ModelName"].eq(model)]
            assert not rejected.empty
            assert rejected["RejectionReason"].str.contains("MissingOptionalDependency").any()


def test_phase22_optional_reporting_is_truthful_when_disabled_by_default():
    report = _run_basic(enable_optional_models=False)
    summary = report.prediction_edge_summary.iloc[0]

    assert str(summary["OptionalModelsTested"]).strip() == ""
    for model in OPTIONAL_MODELS:
        assert f"{model} disabled by config" in str(summary["OptionalModelsSkipped"])
        rejected = report.rejected_models[report.rejected_models["ModelName"].eq(model)]
        assert not rejected.empty
        assert rejected["RejectionReason"].eq("OptionalModelDisabledByConfig").any()


def test_phase22_summary_tested_models_match_leaderboard():
    report = _run_basic(enable_optional_models=False)
    summary = report.prediction_edge_summary.iloc[0]
    leaderboard_models = set(report.model_leaderboard["ModelName"].astype(str))
    optional_in_leaderboard = [model for model in OPTIONAL_MODELS if model in leaderboard_models]
    optional_reported = [value.strip() for value in str(summary["OptionalModelsTested"]).split(";") if value.strip()]

    assert int(summary["TotalModelsTested"]) == len(leaderboard_models)
    assert optional_reported == optional_in_leaderboard


def test_phase22_model_selection_is_chronological_and_future_free():
    report = _run_basic()
    audit = report.model_selection_audit

    assert not audit.empty
    assert (pd.to_datetime(audit["ValidationEndDate"]) < pd.to_datetime(audit["PredictionDate"])).all()
    assert not audit["SelectionUsedFutureData"].astype(bool).any()
    assert audit["SelectionPassed"].astype(bool).all()


def test_phase22_leakage_audit_passes_on_valid_data():
    report = _run_basic()
    audit = report.leakage_audit

    assert not audit.empty
    assert audit["TrainEndBeforeValidation"].astype(bool).all()
    assert audit["ValidationEndBeforePrediction"].astype(bool).all()
    assert audit["PredictionBeforeTargetOutcome"].astype(bool).all()
    assert audit["ScalerFitPastOnly"].astype(bool).all()
    assert audit["NoFutureRowsUsed"].astype(bool).all()
    assert audit["FutureTargetColumnsExcluded"].astype(bool).all()
    assert audit["LeakagePassed"].astype(bool).all()


def test_phase22_baselines_and_rejected_models_remain_visible():
    report = _run_basic(models_to_test=["Ridge", "ElasticNet", "UnsupportedResearchModel"])

    assert not report.baseline_comparison.empty
    assert set(["NoExposureReturnPct", "HoldOnlyReturnPct", "MomentumBaselineReturnPct", "MovingAverageBaselineReturnPct", "RandomMedianBaselineReturnPct"]).issubset(report.baseline_comparison.columns)
    assert not report.rejected_models.empty
    assert "UnsupportedResearchModel" in set(report.rejected_models["ModelName"])


def test_phase22_real_capital_remains_blocked():
    report = _run_basic()

    assert set(report.prediction_edge_summary["RealCapitalStatus"]) == {"Blocked"}
    gates = report.quality_gates.set_index("GateName")
    assert bool(gates.loc["RealCapitalBlocked", "Passed"])


def test_phase22_quality_gates_include_required_checks():
    report = _run_basic(models_to_test=["Ridge", "UnsupportedResearchModel"])
    required = {
        "Phase20Available", "TrueReplayUsed", "ChronologicalValidationPassed",
        "LeakageAuditPassed", "BaselinesAvailable", "LosingModelsVisible",
        "RejectedModelsVisible", "CostSensitivityAvailable", "RealCapitalBlocked",
        "NoForbiddenClaims", "OptionalModelsHandledGracefully",
        "AppDoesNotCrashOnMissingArtifacts",
    }

    assert required.issubset(set(report.quality_gates["GateName"]))


def test_phase22_app_page_is_wired_without_auto_run():
    app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

    assert '"Phase 22: Prediction Edge Improvement"' in app_source
    assert "run_prediction_edge_improvement" in app_source
    assert "st.button" in app_source


def test_phase22_autosaves_required_artifacts():
    old_root = store.ARTIFACT_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        store.ARTIFACT_ROOT = Path(tmp) / "artifacts"
        try:
            report = _run_basic(assets=["Gold"], horizons=[1], max_windows=2, autosave=True)
            expected = {
                "phase22_prediction_edge_summary", "phase22_model_leaderboard",
                "phase22_asset_horizon_model_scorecard", "phase22_prediction_log",
                "phase22_baseline_comparison", "phase22_feature_group_audit",
                "phase22_model_selection_audit", "phase22_leakage_audit",
                "phase22_cost_sensitivity", "phase22_rejected_models",
                "phase22_quality_gates", "phase22_next_actions", "phase22_input_sources",
            }
            assert expected.issubset(report.saved_artifacts["Artifacts"])
            latest = load_latest_artifact(PREDICTION_EDGE_IMPROVEMENT_PHASE_NAME, "phase22_prediction_edge_summary", required=True)
            assert latest.iloc[0]["RealCapitalStatus"] == "Blocked"
        finally:
            store.ARTIFACT_ROOT = old_root


def test_phase22_outputs_have_no_forbidden_trading_claims():
    report = _run_basic()

    assert FORBIDDEN_LANGUAGE.search(_all_output_text(report)) is None
