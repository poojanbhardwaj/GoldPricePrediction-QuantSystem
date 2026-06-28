from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.meta_signal_engine import (
    EVIDENCE_TABLE_COLUMNS,
    PROMOTION_RECOMMENDATION_COLUMNS,
    ROBUSTNESS_SUMMARY_COLUMNS,
    build_evidence_expansion_configs,
    build_promotion_demotion_recommendations,
    run_evidence_expansion,
)


def _grading_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "Gold",
                "Horizon": 5,
                "ReliabilityGrade": "C: Weak Research Candidate",
                "ReliabilityScore_0_100": 52.0,
                "MetaDecision": "Research Only",
            },
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "ReliabilityGrade": "D: Defensive Watch / Regime Evidence",
                "ReliabilityScore_0_100": 41.0,
                "MetaDecision": "No Trade",
            },
            {
                "Asset": "Crude Oil",
                "Horizon": 1,
                "ReliabilityGrade": "F: Rejected / Diagnostic Only",
                "ReliabilityScore_0_100": 14.0,
                "MetaDecision": "Avoid",
            },
        ]
    )


def _synthetic_runner(**kwargs):
    asset = kwargs["asset_names"][0]
    horizon = int(kwargs["horizons"][0])
    cost = float(kwargs["transaction_cost"])
    validation_window = int(kwargs["validation_window"])
    window_mode = str(kwargs["window_mode"])

    if asset == "Crude Oil":
        raise ValueError("synthetic failure for rejected smoke asset")

    strong = asset == "Gold" and validation_window >= 120
    cost_drag = cost * 1000.0
    beat = 68.0 if strong else 48.0
    median_vs = (2.5 if strong else -0.5) - cost_drag
    avg_vs = (3.0 if strong else -0.2) - cost_drag
    drawdown = -14.0 if strong else -27.0
    stability = "Stable" if window_mode == "rolling" and strong else "Unstable"
    agg = pd.DataFrame(
        [
            {
                "Asset": asset,
                "Horizon": horizon,
                "NumberOfWindows": 4,
                "BeatBuyHoldRate_%": beat,
                "PositiveReturnRate_%": 70.0 if strong else 42.0,
                "AvgLockedStrategyReturn_%": 5.0 if strong else -1.0,
                "MedianLockedStrategyReturn_%": 4.0 if strong else -0.8,
                "AvgLockedVsBuyHold_%": avg_vs,
                "MedianLockedVsBuyHold_%": median_vs,
                "WorstLockedVsBuyHold_%": -4.0 if strong else -18.0,
                "AvgLockedMaxDrawdown_%": drawdown,
                "WorstLockedMaxDrawdown_%": drawdown - 4.0,
                "AvgLockedSharpe": 0.9 if strong else -0.1,
                "MedianLockedSharpe": 0.8 if strong else -0.2,
                "AvgTradesPerWindow": 4.0 if strong else 1.0,
                "LowTradeWindowCount": 0 if strong else 2,
                "ThresholdStability": stability,
                "CooldownStability": stability,
                "WalkForwardStabilityScore": 70.0 if stability == "Stable" else 25.0,
                "WalkForwardVerdict": "Research candidate" if strong else "Weak / unstable research only",
                "FailureReason": "",
            }
        ]
    )
    return SimpleNamespace(aggregate_summary=agg, warnings=[], errors=pd.DataFrame(), settings=kwargs)


def test_expanded_configurations_are_generated_correctly():
    configs = build_evidence_expansion_configs(
        validation_windows=[120, 180],
        test_windows=[60],
        step_sizes=[30, 60],
        transaction_costs=[0.001, 0.002],
        window_modes=["rolling", "expanding"],
    )

    assert len(configs) == 16
    assert {"ValidationWindow", "TestWindow", "StepSize", "TransactionCost", "WindowMode"}.issubset(configs.columns)


def test_evidence_expansion_runs_multi_asset_and_outputs_tables():
    report = run_evidence_expansion(
        grading_table=_grading_table(),
        validation_windows=[120],
        test_windows=[60],
        step_sizes=[30, 60],
        transaction_costs=[0.001],
        window_modes=["rolling"],
        walk_forward_runner=_synthetic_runner,
        min_valid_configurations=1,
    )

    assert set(EVIDENCE_TABLE_COLUMNS).issubset(set(report.full_evidence_table.columns))
    assert set(ROBUSTNESS_SUMMARY_COLUMNS).issubset(set(report.robustness_summary.columns))
    assert set(PROMOTION_RECOMMENDATION_COLUMNS).issubset(set(report.promotion_recommendations.columns))
    assert set(report.full_evidence_table["Asset"]) == {"Gold", "Bitcoin", "Crude Oil"}
    assert not report.configuration_summary.empty
    assert not report.cost_sensitivity_summary.empty


def test_failed_candidates_remain_visible_and_low_evidence_warns():
    report = run_evidence_expansion(
        grading_table=_grading_table(),
        validation_windows=[120],
        test_windows=[60],
        step_sizes=[30],
        transaction_costs=[0.001],
        window_modes=["rolling"],
        walk_forward_runner=_synthetic_runner,
        min_valid_configurations=2,
    )

    crude = report.full_evidence_table[report.full_evidence_table["Asset"].eq("Crude Oil")]
    assert len(crude) == 1
    assert crude.iloc[0]["ValidConfiguration"] is False or bool(crude.iloc[0]["ValidConfiguration"]) is False
    assert "NotEnoughValidConfigurations" in set(report.warning_table["WarningType"])


def test_promotion_does_not_occur_from_one_lucky_configuration():
    report = run_evidence_expansion(
        grading_table=_grading_table().iloc[[0]],
        validation_windows=[120],
        test_windows=[60],
        step_sizes=[30],
        transaction_costs=[0.001],
        window_modes=["rolling"],
        walk_forward_runner=_synthetic_runner,
        min_valid_configurations=2,
    )

    rec = report.promotion_recommendations.iloc[0]
    assert "Promote" not in str(rec["Recommendation"])
    assert "NotEnoughValidConfigurations" in str(rec["Warnings"])


def test_candidate_filter_only_c_d_and_no_leakage_selection_basis():
    report = run_evidence_expansion(
        grading_table=_grading_table(),
        candidate_filter="only c/d candidates",
        validation_windows=[120],
        test_windows=[60],
        step_sizes=[30],
        transaction_costs=[0.001],
        window_modes=["rolling"],
        walk_forward_runner=_synthetic_runner,
        min_valid_configurations=1,
    )

    assert set(report.full_evidence_table["Asset"]) == {"Gold", "Bitcoin"}
    assert report.settings["selection_basis"] == "predeclared_config_grid_not_locked_test_tuning"
    assert report.settings["production_ready_label_allowed"] is False


def test_low_trade_unstable_d_candidate_is_conditional_not_promoted():
    robustness = pd.DataFrame(
        [
            {
                "Asset": "Crude Oil",
                "Horizon": 30,
                "StartingReliabilityGrade": "D: Defensive Watch / Regime Evidence",
                "StartingReliabilityScore_0_100": 44.0,
                "ConfigurationsTested": 8,
                "ValidConfigurations": 8,
                "BeatBuyHoldRate_%": 56.0,
                "MedianVsBuyHold_%": 1.4,
                "WorstMaxDrawdown_%": -18.0,
                "CostFragilityScore": 20.0,
                "StabilityScore": 0.0,
                "RobustnessScore": 48.0,
                "AvgTradeCount": 2.0,
                "LowTradeCountRate_%": 100.0,
                "Warnings": "LowTradeCount; SplitUnstable",
            }
        ]
    )

    rec = build_promotion_demotion_recommendations(robustness, min_valid_configurations=2).iloc[0]
    assert rec["Recommendation"] == "Conditional research evidence"
    assert str(rec["RecommendedReliabilityGrade"]).startswith("D:")
    assert "promotion blocked" in rec["MainReason"]
    assert "LowTradeCount" in rec["Warnings"]
    assert "SplitUnstable" in rec["Warnings"]


if __name__ == "__main__":
    test_expanded_configurations_are_generated_correctly()
    test_evidence_expansion_runs_multi_asset_and_outputs_tables()
    test_failed_candidates_remain_visible_and_low_evidence_warns()
    test_promotion_does_not_occur_from_one_lucky_configuration()
    test_candidate_filter_only_c_d_and_no_leakage_selection_basis()
    test_low_trade_unstable_d_candidate_is_conditional_not_promoted()
    print("Phase 8C evidence expansion tests passed.")
