from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.meta_signal_engine import (
    POLICY_RECOMMENDATION_COLUMNS,
    POLICY_SENSITIVITY_COLUMNS,
    run_signal_policy_sensitivity,
)


def _diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "Gold",
                "Horizon": 5,
                "AverageTradeCount": 1.0,
                "LowTradeCountRate_%": 100.0,
                "NoTradeConfigurationRate_%": 60.0,
                "BeatBuyHoldRate_%": 62.0,
                "MedianVsBuyHold_%": 1.0,
                "WorstVsBuyHold_%": -8.0,
                "WorstMaxDrawdown_%": -12.0,
                "CostFragility": 5.0,
                "StabilityScore": 70.0,
                "FailureReasonCategory": "InsufficientTradeCoverage",
                "EvidenceQualityScore": 45.0,
            },
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "AverageTradeCount": 2.0,
                "LowTradeCountRate_%": 30.0,
                "NoTradeConfigurationRate_%": 20.0,
                "BeatBuyHoldRate_%": 55.0,
                "MedianVsBuyHold_%": 0.1,
                "WorstVsBuyHold_%": -10.0,
                "WorstMaxDrawdown_%": -14.0,
                "CostFragility": 10.0,
                "StabilityScore": 65.0,
                "FailureReasonCategory": "InsufficientTradeCoverage",
                "EvidenceQualityScore": 44.0,
            },
            {
                "Asset": "Silver",
                "Horizon": 10,
                "AverageTradeCount": 3.0,
                "LowTradeCountRate_%": 0.0,
                "NoTradeConfigurationRate_%": 0.0,
                "BeatBuyHoldRate_%": 35.0,
                "MedianVsBuyHold_%": -1.5,
                "WorstVsBuyHold_%": -18.0,
                "WorstMaxDrawdown_%": -12.0,
                "CostFragility": 10.0,
                "StabilityScore": 70.0,
                "FailureReasonCategory": "BenchmarkDominated",
                "EvidenceQualityScore": 30.0,
            },
            {
                "Asset": "Crude Oil",
                "Horizon": 30,
                "AverageTradeCount": 0.5,
                "LowTradeCountRate_%": 100.0,
                "NoTradeConfigurationRate_%": 80.0,
                "BeatBuyHoldRate_%": 50.0,
                "MedianVsBuyHold_%": 0.2,
                "WorstVsBuyHold_%": -9.0,
                "WorstMaxDrawdown_%": -14.0,
                "CostFragility": 5.0,
                "StabilityScore": 20.0,
                "FailureReasonCategory": "InsufficientTradeCoverage",
                "EvidenceQualityScore": 20.0,
            },
        ]
    )


def _full_evidence() -> pd.DataFrame:
    rows = []
    for _, row in _diagnostics().iterrows():
        rows.append(
            {
                "Asset": row["Asset"],
                "Horizon": row["Horizon"],
                "ValidConfiguration": True,
                "AvgTradesPerWindow": row["AverageTradeCount"],
                "BeatBuyHoldRate_%": row["BeatBuyHoldRate_%"],
                "MedianLockedVsBuyHold_%": row["MedianVsBuyHold_%"],
                "AvgLockedVsBuyHold_%": row["MedianVsBuyHold_%"],
                "WorstLockedVsBuyHold_%": row["WorstVsBuyHold_%"],
                "WorstLockedMaxDrawdown_%": row["WorstMaxDrawdown_%"],
                "AvgLockedStrategyReturn_%": 2.0,
            }
        )
    return pd.DataFrame(rows)


def _grading() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Asset": "Gold", "Horizon": 5, "ReliabilityGrade": "C: Weak Research Candidate"},
            {"Asset": "Bitcoin", "Horizon": 5, "ReliabilityGrade": "C: Weak Research Candidate"},
            {"Asset": "Silver", "Horizon": 10, "ReliabilityGrade": "D: Defensive Watch / Regime Evidence"},
            {"Asset": "Crude Oil", "Horizon": 30, "ReliabilityGrade": "D: Defensive Watch / Regime Evidence"},
        ]
    )


def test_signal_policy_sensitivity_multi_asset_outputs_columns():
    report = run_signal_policy_sensitivity(
        diagnostics_table=_diagnostics(),
        full_evidence_table=_full_evidence(),
        grading_table=_grading(),
        thresholds=[0.50, 0.65],
        cooldowns=[0, 5],
        min_probabilities=[0.50],
        max_probabilities=[1.0],
        horizons=[1, 5],
    )

    assert set(POLICY_SENSITIVITY_COLUMNS).issubset(report.full_policy_sensitivity_table.columns)
    assert set(POLICY_RECOMMENDATION_COLUMNS).issubset(report.candidate_recommendation_table.columns)
    assert {"Gold", "Bitcoin", "Silver", "Crude Oil"}.issubset(set(report.full_policy_sensitivity_table["Asset"]))


def test_failed_policies_remain_visible():
    report = run_signal_policy_sensitivity(
        diagnostics_table=_diagnostics(),
        full_evidence_table=_full_evidence(),
        thresholds=[0.65],
        cooldowns=[5],
        min_probabilities=[0.575],
        max_probabilities=[0.95],
        horizons=[30],
        candidate_filter="all",
    )

    assert not report.full_policy_sensitivity_table.empty
    assert report.full_policy_sensitivity_table["FinalPolicyVerdict"].notna().all()
    assert "Crude Oil" in set(report.full_policy_sensitivity_table["Asset"])


def test_increasing_trade_count_alone_can_destroy_edge():
    report = run_signal_policy_sensitivity(
        diagnostics_table=_diagnostics(),
        full_evidence_table=_full_evidence(),
        candidate_filter="specific asset/horizon",
        selected_assets=["Bitcoin"],
        selected_horizons=[5],
        thresholds=[0.50],
        cooldowns=[0],
        min_probabilities=[0.50],
        max_probabilities=[1.0],
        horizons=[5],
    )

    assert "CoverageRecoveredButEdgeDestroyed" in set(report.full_policy_sensitivity_table["FinalPolicyVerdict"])


def test_coverage_still_insufficient_and_benchmark_dominated_trigger():
    report = run_signal_policy_sensitivity(
        diagnostics_table=_diagnostics(),
        full_evidence_table=_full_evidence(),
        candidate_filter="all",
        thresholds=[0.65],
        cooldowns=[5],
        min_probabilities=[0.575],
        max_probabilities=[0.95],
        horizons=[10, 30],
    )

    verdicts = set(report.full_policy_sensitivity_table["FinalPolicyVerdict"])
    assert "CoverageStillInsufficient" in verdicts
    assert "BenchmarkDominated" in verdicts


def test_no_ab_promotion_happens_in_phase8e():
    report = run_signal_policy_sensitivity(
        diagnostics_table=_diagnostics(),
        full_evidence_table=_full_evidence(),
        grading_table=_grading(),
        candidate_filter="all",
        thresholds=[0.50],
        cooldowns=[0],
        min_probabilities=[0.50],
        max_probabilities=[1.0],
        horizons=[1, 5, 10, 20, 30],
    )

    assert report.settings["does_not_promote_candidates"] is True
    assert report.settings["production_ready_label_allowed"] is False
    assert "RecommendedReliabilityGrade" not in report.candidate_recommendation_table.columns


if __name__ == "__main__":
    test_signal_policy_sensitivity_multi_asset_outputs_columns()
    test_failed_policies_remain_visible()
    test_increasing_trade_count_alone_can_destroy_edge()
    test_coverage_still_insufficient_and_benchmark_dominated_trigger()
    test_no_ab_promotion_happens_in_phase8e()
    print("Phase 8E signal policy sensitivity tests passed.")
