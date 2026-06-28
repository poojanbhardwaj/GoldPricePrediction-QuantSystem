from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.asset_config import get_asset_names
from src.meta_signal_engine import (
    ASSET_HORIZON_COVERAGE_COLUMNS,
    BENCHMARK_OUTCOME_COLUMNS,
    DRAWDOWN_OUTCOME_COLUMNS,
    LEDGER_QUALITY_COLUMNS,
    PROBABILITY_OUTCOME_AVAILABILITY_COLUMNS,
    TRADE_EVIDENCE_LEDGER_COLUMNS,
    TRADE_OUTCOME_DISTRIBUTION_COLUMNS,
    run_trade_evidence_ledger,
)


def _phase8c_full_evidence() -> pd.DataFrame:
    rows = []
    for i, asset in enumerate(get_asset_names()):
        rows.append(
            {
                "Asset": asset,
                "Horizon": 5 if i % 2 == 0 else 10,
                "ValidConfiguration": True,
                "WindowMode": "rolling",
                "ValidationWindow": 120,
                "TestWindow": 60,
                "StepSize": 30,
                "TransactionCost": 0.001,
                "AvgTradesPerWindow": 2 + i,
                "AvgLockedStrategyReturn_%": 1.0 - i * 0.5,
                "AvgLockedBuyHoldReturn_%": 0.5,
                "AvgLockedVsBuyHold_%": 0.5 - i * 0.3,
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
                "MinProbability": 0.55,
                "MaxProbability": 0.95,
                "TradeCount": 8,
                "StrategyReturn_%": -2.5,
                "BuyHoldReturn_%": 1.0,
                "VsBuyHold_%": -3.5,
                "MedianVsBuyHold_%": -1.2,
                "MaxDrawdown_%": -18.0,
            },
            {
                "Asset": "Crude Oil",
                "Horizon": 5,
                "PolicyType": "Threshold",
                "TradeCount": 5,
                "StrategyReturn_%": 2.0,
                "BuyHoldReturn_%": 0.5,
                "VsBuyHold_%": 1.5,
                "MedianVsBuyHold_%": 0.7,
                "MaxDrawdown_%": -9.0,
            },
        ]
    )


def _phase8f_calibration_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "TotalTrades": 0,
                "RawProbabilityOutcomesAvailable": False,
                "CalibrationGrade": "ProbabilityUnreliable",
                "MainWarning": "CalibrationWeak",
                "BestFilterMaxDrawdown_%": -18.0,
            },
            {
                "Asset": "Crude Oil",
                "Horizon": 5,
                "TotalTrades": 4,
                "RawProbabilityOutcomesAvailable": "False",
                "CalibrationGrade": "TooFewTradesToCalibrate",
                "MainWarning": "TooFewTradesToCalibrate",
                "BestFilterMaxDrawdown_%": -9.0,
            },
        ]
    )


def _raw_trade_logs() -> pd.DataFrame:
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
                "ActualDirection": 1,
                "StrategyReturn": 0.03,
                "BenchmarkReturn": 0.01,
                "VsBuyHold": 0.02,
                "RealizedReturn": 0.03,
                "MaxDrawdown": -4.0,
                "TransactionCost": 0.001,
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
                "ActualDirection": 0,
                "StrategyReturn": -0.05,
                "BenchmarkReturn": 0.02,
                "VsBuyHold": -0.07,
                "RealizedReturn": -0.05,
                "MaxDrawdown": -11.0,
                "TransactionCost": 0.001,
            },
        ]
    )


def _aggregate_report():
    return run_trade_evidence_ledger(
        calibration_summary_table=_phase8f_calibration_summary(),
        policy_sensitivity_table=_phase8e_policy_table(),
        coverage_edge_frontier_table=_phase8e_policy_table(),
        full_evidence_table=_phase8c_full_evidence(),
        configured_assets=get_asset_names(),
        configured_horizons=[1, 5, 10, 20, 30],
    )


def test_trade_evidence_ledger_multi_asset_outputs_columns():
    report = _aggregate_report()

    assert set(TRADE_EVIDENCE_LEDGER_COLUMNS).issubset(report.ledger_table.columns)
    assert set(LEDGER_QUALITY_COLUMNS).issubset(report.ledger_quality_summary.columns)
    assert set(ASSET_HORIZON_COVERAGE_COLUMNS).issubset(report.asset_horizon_coverage_table.columns)
    assert set(PROBABILITY_OUTCOME_AVAILABILITY_COLUMNS).issubset(report.probability_outcome_availability_table.columns)
    assert set(TRADE_OUTCOME_DISTRIBUTION_COLUMNS).issubset(report.trade_outcome_distribution_table.columns)
    assert set(BENCHMARK_OUTCOME_COLUMNS).issubset(report.benchmark_outcome_table.columns)
    assert set(DRAWDOWN_OUTCOME_COLUMNS).issubset(report.drawdown_outcome_table.columns)
    assert set(get_asset_names()).issubset(set(report.ledger_table["Asset"]))


def test_aggregate_derived_rows_are_not_marked_raw():
    report = _aggregate_report()

    assert "RawTradeLevel" not in set(report.ledger_table["EvidenceMode"])
    assert {"WindowAggregate", "PolicyAggregate", "InsufficientData"}.intersection(set(report.ledger_table["EvidenceMode"]))
    assert report.ledger_table["Warnings"].astype(str).str.contains("AggregateProxyOnly").any()


def test_missing_probability_and_direction_block_calibration_readiness():
    report = _aggregate_report()

    warnings = set(report.ledger_warning_table["WarningType"])
    assert "MissingProbability" in warnings
    assert "MissingOutcomeDirection" in warnings
    assert "NotCalibrationReady" in warnings
    assert report.ledger_quality_summary["CalibrationReadinessScore"].iloc[0] < 50


def test_failed_negative_rows_remain_visible_with_raw_logs():
    report = run_trade_evidence_ledger(
        raw_trade_logs=_raw_trade_logs(),
        full_evidence_table=_phase8c_full_evidence(),
        configured_assets=get_asset_names(),
        configured_horizons=[1, 5, 10, 20, 30],
    )

    raw_rows = report.ledger_table[report.ledger_table["EvidenceMode"].eq("RawTradeLevel")]
    assert not raw_rows.empty
    assert raw_rows["StrategyReturn"].lt(0).any()
    bitcoin_outcomes = report.trade_outcome_distribution_table[report.trade_outcome_distribution_table["Asset"].eq("Bitcoin")]
    assert bitcoin_outcomes["NegativeReturnRows"].sum() >= 1


def test_no_promotion_or_production_ready_labels_created():
    report = _aggregate_report()

    assert report.ledger_quality_summary["PromotesCandidates"].iloc[0] == False
    assert report.ledger_quality_summary["ProductionReadyLabelAllowed"].iloc[0] == False
    assert "RecommendedReliabilityGrade" not in report.ledger_table.columns


def test_raw_trade_level_rows_are_marked_raw_and_have_probability_outcomes():
    report = run_trade_evidence_ledger(raw_trade_logs=_raw_trade_logs())
    raw_rows = report.ledger_table[report.ledger_table["EvidenceMode"].eq("RawTradeLevel")]

    assert len(raw_rows) == 2
    assert raw_rows["ProbabilityUp"].notna().all()
    assert raw_rows["ActualDirection"].notna().all()
    assert report.probability_outcome_availability_table["RowsWithProbabilityAndOutcome"].sum() == 2


if __name__ == "__main__":
    test_trade_evidence_ledger_multi_asset_outputs_columns()
    test_aggregate_derived_rows_are_not_marked_raw()
    test_missing_probability_and_direction_block_calibration_readiness()
    test_failed_negative_rows_remain_visible_with_raw_logs()
    test_no_promotion_or_production_ready_labels_created()
    test_raw_trade_level_rows_are_marked_raw_and_have_probability_outcomes()
    print("Phase 8G trade evidence ledger tests passed.")
