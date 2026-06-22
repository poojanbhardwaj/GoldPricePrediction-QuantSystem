from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.meta_signal_engine import (
    RELIABILITY_GRADING_COLUMNS,
    assign_next_best_action,
    build_meta_reliability_grading,
    compute_reliability_grade,
    compute_reliability_score,
    run_meta_score_calibration,
    summarize_reliability_grades,
)


def _grading_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "S&P 500",
                "Horizon": 10,
                "MetaDecision": "Research Only",
                "Calibrated MetaDecision": "Research Only",
                "MetaConfidenceScore": 69.0,
                "MetaRiskScore": 38.0,
                "WalkForwardReliabilityScore": 78.0,
                "BeatBuyHoldRate_%": 72.0,
                "MedianLockedVsBuyHold_%": 4.5,
                "AvgLockedVsBuyHold_%": 5.5,
                "WorstLockedMaxDrawdown_%": -10.0,
                "AvgLockedSharpe": 1.2,
                "AvgTradesPerWindow": 5.0,
                "ThresholdStability": "Stable",
                "CooldownStability": "Stable",
                "RegimeLabel": "Constructive uptrend",
                "BenchmarkRiskFlag": False,
                "CostFragilityFlag": False,
                "DrawdownRiskFlag": False,
                "StabilityFlag": "Stable",
                "WalkForwardVerdict": "Strong walk-forward research candidate",
                "MainBlockingRule": "Confidence below strict Trade threshold",
            },
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "MetaDecision": "Research Only",
                "Calibrated MetaDecision": "Research Only",
                "MetaConfidenceScore": 52.0,
                "MetaRiskScore": 71.0,
                "WalkForwardReliabilityScore": 60.0,
                "BeatBuyHoldRate_%": 64.0,
                "MedianLockedVsBuyHold_%": 2.0,
                "AvgLockedVsBuyHold_%": 3.0,
                "WorstLockedMaxDrawdown_%": -28.0,
                "AvgLockedSharpe": 0.6,
                "AvgTradesPerWindow": 4.0,
                "ThresholdStability": "Unstable",
                "CooldownStability": "Stable",
                "RegimeLabel": "Risk-off / weak trend",
                "BenchmarkRiskFlag": True,
                "CostFragilityFlag": False,
                "DrawdownRiskFlag": True,
                "StabilityFlag": "ThresholdUnstable",
                "WalkForwardVerdict": "Research candidate",
                "MainBlockingRule": "Benchmark risk flag is active",
            },
            {
                "Asset": "Crude Oil",
                "Horizon": 5,
                "MetaDecision": "Avoid",
                "Calibrated MetaDecision": "Avoid",
                "MetaConfidenceScore": 18.0,
                "MetaRiskScore": 92.0,
                "WalkForwardReliabilityScore": 16.0,
                "BeatBuyHoldRate_%": 20.0,
                "MedianLockedVsBuyHold_%": -8.0,
                "AvgLockedVsBuyHold_%": -9.0,
                "WorstLockedMaxDrawdown_%": -35.0,
                "AvgLockedSharpe": -0.5,
                "AvgTradesPerWindow": 2.0,
                "ThresholdStability": "Unstable",
                "CooldownStability": "Unstable",
                "RegimeLabel": "High-volatility drawdown",
                "BenchmarkRiskFlag": True,
                "CostFragilityFlag": True,
                "DrawdownRiskFlag": True,
                "StabilityFlag": "LowEvidence",
                "WalkForwardVerdict": "Do not trust",
                "MainBlockingRule": "Walk-forward verdict is Do Not Trust",
            },
        ]
    )


def test_meta_reliability_grading_runs_multi_asset_multi_horizon():
    report = run_meta_score_calibration(_grading_rows(), "Balanced")

    assert len(report.grading_table) == 3
    assert set(RELIABILITY_GRADING_COLUMNS).issubset(set(report.grading_table.columns))
    assert set(report.grading_table["Asset"]) == {"S&P 500", "Bitcoin", "Crude Oil"}
    assert set(report.grading_table["Horizon"]) == {5, 10}


def test_weak_rows_do_not_become_near_trade():
    table = build_meta_reliability_grading(_grading_rows(), "Aggressive Research")
    weak = table[table["Asset"].eq("Crude Oil")].iloc[0]

    assert not str(weak["ReliabilityGrade"]).startswith("A:")
    assert weak["PromotionReadiness"] != "Near-Trade Research Candidate"
    assert weak["MetaDecision"] == "Avoid"


def test_strong_rows_can_get_higher_grade_but_not_production_ready():
    report = run_meta_score_calibration(_grading_rows(), "Balanced")
    strong = report.grading_table[report.grading_table["Asset"].eq("S&P 500")].iloc[0]

    assert str(strong["ReliabilityGrade"]).startswith(("A:", "B:"))
    assert "not production-ready" in strong["GradeExplanation"] or report.settings["production_ready_label_allowed"] is False
    assert report.settings["meta_decision_remains_action_gate"] is True


def test_rejected_rows_remain_visible():
    report = run_meta_score_calibration(_grading_rows(), "Conservative")
    rejected = report.grading_table[report.grading_table["ReliabilityGrade"].astype(str).str.startswith("F:")]

    assert len(rejected) >= 1
    assert "Crude Oil" in set(rejected["Asset"])


def test_missing_columns_are_handled_safely():
    incomplete = pd.DataFrame([{"Asset": "Gold", "Horizon": 1, "MetaDecision": "No Trade"}])
    report = run_meta_score_calibration(incomplete, "Conservative")
    row = report.grading_table.iloc[0]

    assert 0.0 <= float(row["ReliabilityScore_0_100"]) <= 100.0
    assert row["GradeExplanation"]
    assert row["NextBestAction"]


def test_reliability_score_is_bounded_and_explained():
    rows = _grading_rows()
    for _, row in rows.iterrows():
        score = compute_reliability_score(row, "Aggressive Research")
        grade = compute_reliability_grade(row, "Aggressive Research")
        action = assign_next_best_action(row, grade, "Aggressive Research")

        assert 0.0 <= score["ReliabilityScore_0_100"] <= 100.0
        assert grade["GradeExplanation"]
        assert action


def test_grade_counts_summarize_correctly():
    table = build_meta_reliability_grading(_grading_rows(), "Balanced")
    summary = summarize_reliability_grades(table)

    assert summary["grade_counts"]["Count"].sum() == len(table)
    assert not summary["next_action_summary"].empty
    assert not summary["score_components"].empty


if __name__ == "__main__":
    test_meta_reliability_grading_runs_multi_asset_multi_horizon()
    test_weak_rows_do_not_become_near_trade()
    test_strong_rows_can_get_higher_grade_but_not_production_ready()
    test_rejected_rows_remain_visible()
    test_missing_columns_are_handled_safely()
    test_reliability_score_is_bounded_and_explained()
    test_grade_counts_summarize_correctly()
    print("Phase 8B meta reliability grading tests passed.")
