from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.asset_config import get_asset_names
from src.meta_signal_engine import (
    PHASE8_CLOSURE_COLUMNS,
    RAW_BENCHMARK_COMPARISON_COLUMNS,
    RAW_COVERAGE_COLUMNS,
    RAW_DRAWDOWN_COLUMNS,
    RAW_PROBABILITY_READINESS_COLUMNS,
    TRUE_MISSING_SOURCE_COLUMNS,
    TRUE_RAW_QUALITY_COLUMNS,
    TRUE_RAW_TRADE_LOG_COLUMNS,
    run_true_raw_trade_log_generation,
)


def _raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "ModelName": "RandomForest",
                "PolicyName": "ValidationLocked",
                "SignalDate": "2026-01-02",
                "EntryDate": "2026-01-02",
                "ExitDate": "2026-01-09",
                "HoldingPeriod": 5,
                "ProbabilityUp": 0.72,
                "PredictedDirection": "Up",
                "ActualDirection": 1,
                "SignalTaken": True,
                "EntryPrice": 100.0,
                "ExitPrice": 103.0,
                "RealizedReturn": 0.03,
                "BenchmarkReturn": 0.01,
                "VsBuyHold": 0.02,
                "MaxDrawdownDuringTrade": -0.04,
                "CostApplied": 0.001,
                "Threshold": 0.55,
                "Cooldown": 0,
            },
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "ModelName": "RandomForest",
                "PolicyName": "ValidationLocked",
                "SignalDate": "2026-01-10",
                "EntryDate": "2026-01-10",
                "ExitDate": "2026-01-17",
                "HoldingPeriod": 5,
                "ProbabilityUp": 0.91,
                "PredictedDirection": "Up",
                "ActualDirection": 0,
                "SignalTaken": True,
                "EntryPrice": 103.0,
                "ExitPrice": 98.0,
                "RealizedReturn": -0.05,
                "BenchmarkReturn": 0.02,
                "VsBuyHold": -0.07,
                "MaxDrawdownDuringTrade": -0.11,
                "CostApplied": 0.001,
                "Threshold": 0.55,
                "Cooldown": 0,
            },
        ]
    )


def _reconstructed_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "Crude Oil",
                "Horizon": 5,
                "SignalDate": "2026-02-02",
                "EntryDate": "2026-02-02",
                "ExitDate": "2026-02-09",
                "HoldingPeriod": 5,
                "ProbabilityUp": 0.63,
                "ActualDirection": 1,
                "SignalTaken": True,
                "RealizedReturn": 0.02,
                "BenchmarkReturn": 0.01,
                "VsBuyHold": 0.01,
                "MaxDrawdownDuringTrade": -0.05,
            }
        ]
    )


def _missing_probability_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "Gold",
                "Horizon": 5,
                "SignalDate": "2026-03-02",
                "EntryDate": "2026-03-02",
                "ExitDate": "2026-03-09",
                "ActualDirection": 1,
                "RealizedReturn": 0.01,
            }
        ]
    )


def _aggregate_only() -> pd.DataFrame:
    rows = []
    for i, asset in enumerate(get_asset_names()):
        rows.append(
            {
                "Asset": asset,
                "Horizon": 5 if i % 2 == 0 else 10,
                "WindowMode": "rolling",
                "AvgLockedStrategyReturn_%": 1.0 - i,
                "AvgLockedBuyHoldReturn_%": 0.5,
                "AvgLockedVsBuyHold_%": 0.5 - i,
            }
        )
    return pd.DataFrame(rows)


def test_true_raw_generation_outputs_columns_and_multi_asset_support():
    report = run_true_raw_trade_log_generation(
        raw_signal_outputs=pd.concat([_raw_rows(), _reconstructed_rows()], ignore_index=True),
        full_evidence_table=_aggregate_only(),
        assets=get_asset_names(),
        horizons=[1, 5, 10, 20, 30],
    )

    assert set(TRUE_RAW_TRADE_LOG_COLUMNS).issubset(report.true_raw_trade_log.columns)
    assert set(TRUE_RAW_QUALITY_COLUMNS).issubset(report.raw_log_quality_summary.columns)
    assert set(RAW_COVERAGE_COLUMNS).issubset(report.asset_horizon_raw_coverage.columns)
    assert set(RAW_PROBABILITY_READINESS_COLUMNS).issubset(report.probability_outcome_readiness.columns)
    assert set(TRUE_MISSING_SOURCE_COLUMNS).issubset(report.missing_source_diagnostic.columns)
    assert set(RAW_BENCHMARK_COMPARISON_COLUMNS).issubset(report.benchmark_comparison.columns)
    assert set(RAW_DRAWDOWN_COLUMNS).issubset(report.drawdown_during_trade.columns)
    assert set(PHASE8_CLOSURE_COLUMNS).issubset(report.phase8_closure_readiness_table.columns)
    assert {"Bitcoin", "Crude Oil"}.issubset(set(report.true_raw_trade_log["Asset"]))


def test_raw_trade_level_only_when_raw_trade_fields_exist():
    report = run_true_raw_trade_log_generation(raw_signal_outputs=_raw_rows())

    assert set(report.true_raw_trade_log["EvidenceMode"]) == {"RawTradeLevel"}


def test_reconstructed_trade_level_requires_safe_row_level_fields():
    report = run_true_raw_trade_log_generation(raw_signal_outputs=_reconstructed_rows())

    assert set(report.true_raw_trade_log["EvidenceMode"]) == {"ReconstructedTradeLevel"}
    assert report.true_raw_trade_log["ProbabilityUp"].notna().all()
    assert report.true_raw_trade_log["ActualDirection"].notna().all()


def test_aggregate_fallback_excluded_from_true_raw_log():
    report = run_true_raw_trade_log_generation(full_evidence_table=_aggregate_only(), assets=["Gold"], horizons=[5])

    assert report.true_raw_trade_log.empty
    assert not report.aggregate_fallback_diagnostic.empty
    assert "WindowAggregateFallback" in set(report.aggregate_fallback_diagnostic["MissingField"])
    assert report.phase8_closure_readiness_table["Phase8RawEvidenceReady"].iloc[0] == False


def test_missing_probability_and_outcome_trigger_not_calibration_ready():
    report = run_true_raw_trade_log_generation(raw_signal_outputs=_missing_probability_rows(), assets=["Gold"], horizons=[5])
    warnings = set(report.warning_table["WarningType"])

    assert "MissingProbability" in warnings
    assert "NotCalibrationReady" in warnings
    assert report.phase8_closure_readiness_table["ProbabilityCalibrationCanBeRerun"].iloc[0] == False


def test_negative_losing_trades_remain_visible():
    report = run_true_raw_trade_log_generation(raw_signal_outputs=_raw_rows())

    assert report.true_raw_trade_log["RealizedReturn"].lt(0).any()
    assert report.raw_log_quality_summary["NegativeTradeRows"].iloc[0] >= 1


def test_no_promotion_or_production_ready_labels_created():
    report = run_true_raw_trade_log_generation(raw_signal_outputs=_raw_rows())

    assert report.raw_log_quality_summary["PromotesCandidates"].iloc[0] == False
    assert report.raw_log_quality_summary["ProductionReadyLabelAllowed"].iloc[0] == False
    assert report.phase8_closure_readiness_table["ProductionReadyLabelAllowed"].iloc[0] == False
    assert "RecommendedReliabilityGrade" not in report.true_raw_trade_log.columns


if __name__ == "__main__":
    test_true_raw_generation_outputs_columns_and_multi_asset_support()
    test_raw_trade_level_only_when_raw_trade_fields_exist()
    test_reconstructed_trade_level_requires_safe_row_level_fields()
    test_aggregate_fallback_excluded_from_true_raw_log()
    test_missing_probability_and_outcome_trigger_not_calibration_ready()
    test_negative_losing_trades_remain_visible()
    test_no_promotion_or_production_ready_labels_created()
    print("Phase 8I true raw trade log tests passed.")
