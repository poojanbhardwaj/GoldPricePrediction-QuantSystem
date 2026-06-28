from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.asset_config import get_asset_names
from src.meta_signal_engine import (
    RAW_BENCHMARK_COMPARISON_COLUMNS,
    RAW_COVERAGE_COLUMNS,
    RAW_DRAWDOWN_COLUMNS,
    RAW_LOG_QUALITY_COLUMNS,
    RAW_NO_TRADE_COLUMNS,
    RAW_PROBABILITY_READINESS_COLUMNS,
    RAW_TRADE_DISTRIBUTION_COLUMNS,
    RAW_TRADE_LOG_COLUMNS,
    run_raw_trade_log_exporter,
)


def _grading_table() -> pd.DataFrame:
    rows = []
    for i, asset in enumerate(get_asset_names()):
        rows.append(
            {
                "Asset": asset,
                "Horizon": 5 if i % 2 == 0 else 10,
                "ReliabilityGrade": "C: Weak Research Candidate" if i % 2 == 0 else "D: Defensive Watch / Regime Evidence",
            }
        )
    return pd.DataFrame(rows)


def _phase8c_full_evidence() -> pd.DataFrame:
    rows = []
    for i, asset in enumerate(get_asset_names()):
        rows.append(
            {
                "Asset": asset,
                "Horizon": 5 if i % 2 == 0 else 10,
                "WindowMode": "rolling",
                "ValidationWindow": 120,
                "TestWindow": 60,
                "StepSize": 30,
                "TransactionCost": 0.001,
                "ValidConfiguration": True,
                "AvgLockedStrategyReturn_%": 1.0 - i,
                "AvgLockedBuyHoldReturn_%": 0.5,
                "AvgLockedVsBuyHold_%": 0.5 - i,
                "WorstLockedMaxDrawdown_%": -8.0 - i,
            }
        )
    return pd.DataFrame(rows)


def _phase8e_policy_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "PolicyType": "ProbabilityBand",
                "TradeCount": 6,
                "StrategyReturn_%": -2.0,
                "BuyHoldReturn_%": 1.0,
                "VsBuyHold_%": -3.0,
                "MaxDrawdown_%": -13.0,
            },
            {
                "Asset": "Gold",
                "Horizon": 5,
                "PolicyType": "Threshold",
                "TradeCount": 0,
                "SignalTaken": False,
                "StrategyReturn_%": 0.0,
                "BuyHoldReturn_%": 1.0,
                "VsBuyHold_%": -1.0,
                "MaxDrawdown_%": -2.0,
            },
        ]
    )


def _phase8g_ledger_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "LedgerId": "L1",
                "Asset": "Crude Oil",
                "Horizon": 5,
                "PolicyName": "Reconstructable policy",
                "EvidenceMode": "PolicyAggregate",
                "SignalDate": "2026-02-01",
                "EntryDate": "2026-02-01",
                "ExitDate": "2026-02-06",
                "HoldingPeriod": 5,
                "ProbabilityUp": 0.64,
                "ActualDirection": 1,
                "StrategyReturn": 0.02,
                "BenchmarkReturn": 0.01,
                "VsBuyHold": 0.01,
                "MaxDrawdown": -5.0,
                "SignalTaken": True,
            }
        ]
    )


def _raw_signal_outputs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "ModelName": "RandomForest",
                "PolicyName": "ProbabilityBand",
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
                "MaxDrawdownDuringTrade": -4.0,
                "CostApplied": 0.001,
            },
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "ModelName": "RandomForest",
                "PolicyName": "ProbabilityBand",
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
                "MaxDrawdownDuringTrade": -11.0,
                "CostApplied": 0.001,
            },
            {
                "Asset": "Gold",
                "Horizon": 5,
                "SignalDate": "2026-01-20",
                "EntryDate": "2026-01-20",
                "ExitDate": "2026-01-27",
                "HoldingPeriod": 5,
                "ActualDirection": 1,
                "SignalTaken": False,
                "RealizedReturn": 0.0,
                "BenchmarkReturn": 0.01,
                "VsBuyHold": -0.01,
                "MaxDrawdownDuringTrade": -2.0,
            },
        ]
    )


def test_raw_trade_log_exporter_multi_asset_outputs_columns():
    report = run_raw_trade_log_exporter(
        grading_table=_grading_table(),
        full_evidence_table=_phase8c_full_evidence(),
        policy_sensitivity_table=_phase8e_policy_table(),
        configured_assets=get_asset_names(),
        configured_horizons=[1, 5, 10, 20, 30],
    )

    assert set(RAW_TRADE_LOG_COLUMNS).issubset(report.raw_signal_trade_log_table.columns)
    assert set(RAW_LOG_QUALITY_COLUMNS).issubset(report.raw_log_quality_summary.columns)
    assert set(RAW_COVERAGE_COLUMNS).issubset(report.asset_horizon_raw_coverage_table.columns)
    assert set(RAW_PROBABILITY_READINESS_COLUMNS).issubset(report.probability_outcome_readiness_table.columns)
    assert set(RAW_TRADE_DISTRIBUTION_COLUMNS).issubset(report.trade_outcome_distribution_table.columns)
    assert set(RAW_BENCHMARK_COMPARISON_COLUMNS).issubset(report.benchmark_comparison_table.columns)
    assert set(RAW_DRAWDOWN_COLUMNS).issubset(report.drawdown_during_trade_table.columns)
    assert set(RAW_NO_TRADE_COLUMNS).issubset(report.no_trade_skipped_signal_table.columns)
    assert set(get_asset_names()).issubset(set(report.raw_signal_trade_log_table["Asset"]))


def test_raw_rows_marked_raw_only_when_raw_fields_exist():
    report = run_raw_trade_log_exporter(raw_signal_outputs=_raw_signal_outputs())
    raw_rows = report.raw_signal_trade_log_table[report.raw_signal_trade_log_table["EvidenceMode"].eq("RawTradeLevel")]

    assert len(raw_rows) == 2
    assert raw_rows["ProbabilityUp"].notna().all()
    assert raw_rows["ActualDirection"].notna().all()


def test_reconstructed_rows_are_clearly_marked():
    report = run_raw_trade_log_exporter(ledger_table=_phase8g_ledger_table())

    assert "ReconstructedTradeLevel" in set(report.raw_signal_trade_log_table["EvidenceMode"])


def test_aggregate_fallback_rows_are_not_raw():
    report = run_raw_trade_log_exporter(full_evidence_table=_phase8c_full_evidence(), policy_sensitivity_table=_phase8e_policy_table())

    modes = set(report.raw_signal_trade_log_table["EvidenceMode"])
    assert "RawTradeLevel" not in modes
    assert {"WindowAggregateFallback", "PolicyAggregateFallback"}.intersection(modes)
    assert report.raw_signal_trade_log_table["Warnings"].astype(str).str.contains("AggregateFallbackOnly").any()


def test_missing_probability_and_direction_trigger_not_calibration_ready():
    report = run_raw_trade_log_exporter(full_evidence_table=_phase8c_full_evidence())
    warnings = set(report.warning_table["WarningType"])

    assert "MissingProbability" in warnings
    assert "MissingOutcomeDirection" in warnings
    assert "NotCalibrationReady" in warnings
    assert report.raw_log_quality_summary["CalibrationReadinessScore"].iloc[0] < 50


def test_negative_losing_trades_remain_visible():
    report = run_raw_trade_log_exporter(raw_signal_outputs=_raw_signal_outputs())

    assert report.raw_signal_trade_log_table["RealizedReturn"].lt(0).any()
    assert report.trade_outcome_distribution_table["NegativeReturnRows"].sum() >= 1


def test_no_promotion_or_production_ready_labels_created():
    report = run_raw_trade_log_exporter(raw_signal_outputs=_raw_signal_outputs())

    assert report.raw_log_quality_summary["PromotesCandidates"].iloc[0] == False
    assert report.raw_log_quality_summary["ProductionReadyLabelAllowed"].iloc[0] == False
    assert "RecommendedReliabilityGrade" not in report.raw_signal_trade_log_table.columns


if __name__ == "__main__":
    test_raw_trade_log_exporter_multi_asset_outputs_columns()
    test_raw_rows_marked_raw_only_when_raw_fields_exist()
    test_reconstructed_rows_are_clearly_marked()
    test_aggregate_fallback_rows_are_not_raw()
    test_missing_probability_and_direction_trigger_not_calibration_ready()
    test_negative_losing_trades_remain_visible()
    test_no_promotion_or_production_ready_labels_created()
    print("Phase 8H raw trade log exporter tests passed.")
