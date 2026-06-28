from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.meta_signal_engine import (
    CALIBRATION_ERROR_COLUMNS,
    CONFIDENCE_USEFULNESS_COLUMNS,
    HIGH_CONFIDENCE_FAILURE_COLUMNS,
    PROBABILITY_BIN_COLUMNS,
    PROBABILITY_CALIBRATION_SUMMARY_COLUMNS,
    PROBABILITY_FILTER_COLUMNS,
    PROBABILITY_RECOMMENDATION_COLUMNS,
    run_probability_calibration,
)


def _candidate_recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Asset": "Bitcoin", "Horizon": 5, "Recommendation": "Proceed to probability calibration diagnostics"},
            {"Asset": "Crude Oil", "Horizon": 5, "Recommendation": "Proceed to probability calibration diagnostics"},
            {"Asset": "Gold", "Horizon": 5, "Recommendation": "Too sparse"},
            {"Asset": "Silver", "Horizon": 10, "Recommendation": "Non-monotonic smoke candidate"},
        ]
    )


def _trade_rows() -> pd.DataFrame:
    rows = []

    bitcoin_probs = [0.51, 0.53, 0.56, 0.58, 0.61, 0.63, 0.66, 0.68, 0.72, 0.74, 0.76, 0.78, 0.82, 0.88, 0.93, 0.96]
    bitcoin_actuals = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, (probability, actual) in enumerate(zip(bitcoin_probs, bitcoin_actuals)):
        rows.append(
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "ProbabilityUp": probability,
                "ActualDirection": actual,
                "RealizedReturn": 0.008 if actual else -0.012,
                "VsBuyHold_%": 0.6 if actual else -1.0,
                "MaxDrawdown_%": -6.0 - i * 0.1,
            }
        )

    crude_probs = [0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79, 0.82, 0.85, 0.88, 0.92, 0.96]
    crude_actuals = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    for probability, actual in zip(crude_probs, crude_actuals):
        rows.append(
            {
                "Asset": "Crude Oil",
                "Horizon": 5,
                "ProbabilityUp": probability,
                "ActualDirection": actual,
                "RealizedReturn": 0.01 if actual else -0.006,
                "VsBuyHold_%": 0.8 if actual else -0.4,
                "MaxDrawdown_%": -8.0,
            }
        )

    for probability, actual in [(0.52, 1), (0.58, 0), (0.64, 1), (0.72, 0)]:
        rows.append(
            {
                "Asset": "Gold",
                "Horizon": 5,
                "ProbabilityUp": probability,
                "ActualDirection": actual,
                "RealizedReturn": 0.003 if actual else -0.004,
                "VsBuyHold_%": 0.1 if actual else -0.2,
                "MaxDrawdown_%": -4.0,
            }
        )

    silver_probs = [0.51, 0.53, 0.56, 0.58, 0.61, 0.63, 0.66, 0.68, 0.72, 0.74, 0.76, 0.78]
    silver_actuals = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    for probability, actual in zip(silver_probs, silver_actuals):
        rows.append(
            {
                "Asset": "Silver",
                "Horizon": 10,
                "ProbabilityUp": probability,
                "ActualDirection": actual,
                "RealizedReturn": 0.006 if actual else -0.009,
                "VsBuyHold_%": 0.4 if actual else -0.8,
                "MaxDrawdown_%": -10.0,
            }
        )

    return pd.DataFrame(rows)


def _true_raw_trade_log() -> pd.DataFrame:
    rows = []
    probabilities = [0.51, 0.54, 0.57, 0.59, 0.62, 0.66, 0.69, 0.72, 0.76, 0.81, 0.88, 0.94]
    actuals = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]
    for i, (probability, actual) in enumerate(zip(probabilities, actuals)):
        realized = 0.008 if actual else -0.006
        benchmark = 0.002 if i % 3 else 0.004
        rows.append(
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "ProbabilityUp": probability,
                "ActualDirection": actual,
                "RealizedReturn": realized,
                "BenchmarkReturn": benchmark,
                "VsBuyHold": realized - benchmark,
                "MaxDrawdownDuringTrade": -0.025 - i * 0.001,
                "WinLoss": "Win" if actual else "Loss",
                "BeatBenchmark": realized > benchmark,
                "EvidenceMode": "ReconstructedTradeLevel",
            }
        )
    return pd.DataFrame(rows)


def _report():
    return run_probability_calibration(
        candidate_recommendation_table=_candidate_recommendations(),
        coverage_edge_frontier_table=_trade_rows(),
        full_evidence_table=_trade_rows(),
        candidate_filter="all",
        probability_bins=[
            (0.50, 0.55),
            (0.55, 0.60),
            (0.60, 0.65),
            (0.65, 0.70),
            (0.70, 0.75),
            (0.75, 0.80),
            (0.80, 0.90),
            (0.90, 1.00),
        ],
        min_probabilities=[0.50, 0.75, 0.90],
        max_probabilities=[0.90, 1.00],
    )


def test_true_raw_trade_log_is_used_for_calibration_not_aggregate_proxy():
    report = run_probability_calibration(
        raw_trade_log_table=_true_raw_trade_log(),
        candidate_filter="all",
        probability_bins=[
            (0.50, 0.55),
            (0.55, 0.60),
            (0.60, 0.65),
            (0.65, 0.70),
            (0.70, 0.80),
            (0.80, 1.00),
        ],
        min_probabilities=[0.50],
        max_probabilities=[1.00],
    )
    bitcoin = report.calibration_summary_table[report.calibration_summary_table["Asset"].eq("Bitcoin")].iloc[0]

    assert bitcoin["RawProbabilityOutcomesAvailable"] == True
    assert pd.notna(bitcoin["BrierScore"])
    assert bitcoin["TotalTrades"] == len(_true_raw_trade_log())
    assert report.probability_bin_table["SourceType"].eq("raw_probability_outcomes").all()
    assert not report.probability_bin_table["Warnings"].astype(str).str.contains("AggregateProxyOnly").any()


def test_probability_calibration_multi_asset_outputs_columns():
    report = _report()

    assert set(PROBABILITY_CALIBRATION_SUMMARY_COLUMNS).issubset(report.calibration_summary_table.columns)
    assert set(PROBABILITY_BIN_COLUMNS).issubset(report.probability_bin_table.columns)
    assert set(PROBABILITY_FILTER_COLUMNS).issubset(report.probability_filter_simulation_table.columns)
    assert set(CONFIDENCE_USEFULNESS_COLUMNS).issubset(report.confidence_usefulness_table.columns)
    assert set(CALIBRATION_ERROR_COLUMNS).issubset(report.calibration_error_table.columns)
    assert set(HIGH_CONFIDENCE_FAILURE_COLUMNS).issubset(report.high_confidence_failure_table.columns)
    assert set(PROBABILITY_RECOMMENDATION_COLUMNS).issubset(report.candidate_recommendation_table.columns)
    assert {"Bitcoin", "Crude Oil", "Gold", "Silver"}.issubset(set(report.calibration_summary_table["Asset"]))


def test_too_few_trades_creates_calibration_warning():
    report = _report()
    gold = report.calibration_summary_table[report.calibration_summary_table["Asset"].eq("Gold")].iloc[0]

    assert gold["CalibrationGrade"] == "TooFewTradesToCalibrate"
    assert "TooFewTradesToCalibrate" in set(report.warning_table["WarningType"])


def test_high_confidence_losing_flags_overconfidence():
    report = _report()
    bitcoin = report.calibration_summary_table[report.calibration_summary_table["Asset"].eq("Bitcoin")].iloc[0]

    assert bitcoin["CalibrationGrade"] in {"Overconfident", "ProbabilityUnreliable"}
    assert {"Overconfident", "HighConfidenceFailures"}.intersection(set(report.warning_table["WarningType"]))
    assert not report.high_confidence_failure_table[report.high_confidence_failure_table["Asset"].eq("Bitcoin")].empty


def test_non_monotonic_probability_edge_is_visible():
    report = _report()
    silver = report.confidence_usefulness_table[report.confidence_usefulness_table["Asset"].eq("Silver")].iloc[0]

    assert silver["UsefulnessVerdict"] == "ProbabilityUnreliable"
    assert "EdgeNotMonotonic" in str(silver["Warnings"])


def test_probability_filter_destroying_trade_count_is_rejected():
    report = _report()
    strict_filters = report.probability_filter_simulation_table[
        (report.probability_filter_simulation_table["Asset"].eq("Bitcoin"))
        & (report.probability_filter_simulation_table["MinProbability"].eq(0.90))
    ]

    assert not strict_filters.empty
    assert strict_filters["FilterVerdict"].eq("NoUsefulProbabilityFilter").any()
    assert strict_filters["Warnings"].astype(str).str.contains("TradeCountDestroyed").any()


def test_phase8f_never_promotes_or_claims_production_ready():
    report = _report()

    assert report.overall_summary["PromotesGrades"].iloc[0] == False
    assert report.overall_summary["ProductionReadyLabelAllowed"].iloc[0] == False
    assert report.candidate_recommendation_table["ShouldPromoteGrade"].eq(False).all()
    assert report.candidate_recommendation_table["ProductionReadyLabelAllowed"].eq(False).all()


def test_failed_probability_bins_remain_visible():
    report = _report()
    gold_bins = report.probability_bin_table[report.probability_bin_table["Asset"].eq("Gold")]

    assert len(gold_bins) == 8
    assert gold_bins["Warnings"].astype(str).str.contains("NoTradesInBin").any()


if __name__ == "__main__":
    test_true_raw_trade_log_is_used_for_calibration_not_aggregate_proxy()
    test_probability_calibration_multi_asset_outputs_columns()
    test_too_few_trades_creates_calibration_warning()
    test_high_confidence_losing_flags_overconfidence()
    test_non_monotonic_probability_edge_is_visible()
    test_probability_filter_destroying_trade_count_is_rejected()
    test_phase8f_never_promotes_or_claims_production_ready()
    test_failed_probability_bins_remain_visible()
    print("Phase 8F probability calibration tests passed.")
