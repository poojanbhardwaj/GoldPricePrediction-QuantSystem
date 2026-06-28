from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.meta_signal_engine import (
    EVIDENCE_QUALITY_COLUMNS,
    FAILURE_REASON_COLUMNS,
    SIGNAL_COVERAGE_COLUMNS,
    run_evidence_quality_diagnostics,
)


def _full_evidence() -> pd.DataFrame:
    rows = []
    cases = [
        ("Gold", 5, 1.0, 60.0, 1.2, -6.0, -12.0, "Stable", "Stable", True),
        ("Bitcoin", 5, 4.0, 35.0, -2.0, -18.0, -16.0, "Stable", "Stable", True),
        ("Silver", 10, 4.0, 62.0, 1.5, -5.0, -13.0, "Stable", "Stable", True),
        ("Crude Oil", 30, 4.0, 64.0, 1.0, -7.0, -15.0, "Unstable", "Unstable", True),
    ]
    for asset, horizon, trades, beat, median_vs, worst_vs, drawdown, threshold_stability, cooldown_stability, valid in cases:
        for cost in [0.0005, 0.002]:
            rows.append(
                {
                    "Asset": asset,
                    "Horizon": horizon,
                    "ValidConfiguration": valid,
                    "ValidationWindow": 120,
                    "TestWindow": 60,
                    "StepSize": 30,
                    "TransactionCost": cost,
                    "WindowMode": "rolling",
                    "AvgTradesPerWindow": trades,
                    "LowTradeWindowCount": 2 if trades <= 1.0 else 0,
                    "BeatBuyHoldRate_%": beat,
                    "MedianLockedVsBuyHold_%": median_vs - (cost * 200.0 if asset == "Silver" else 0.0),
                    "AvgLockedVsBuyHold_%": median_vs,
                    "WorstLockedVsBuyHold_%": worst_vs,
                    "WorstLockedMaxDrawdown_%": drawdown,
                    "ThresholdStability": threshold_stability,
                    "CooldownStability": cooldown_stability,
                }
            )
    rows.append(
        {
            "Asset": "Gold ETF",
            "Horizon": 20,
            "ValidConfiguration": False,
            "ValidationWindow": 120,
            "TestWindow": 60,
            "StepSize": 30,
            "TransactionCost": 0.001,
            "WindowMode": "rolling",
            "AvgTradesPerWindow": 0.0,
            "LowTradeWindowCount": 1,
            "BeatBuyHoldRate_%": 0.0,
            "MedianLockedVsBuyHold_%": 0.0,
            "AvgLockedVsBuyHold_%": 0.0,
            "WorstLockedVsBuyHold_%": 0.0,
            "WorstLockedMaxDrawdown_%": 0.0,
            "ThresholdStability": "Unstable",
            "CooldownStability": "Unstable",
        }
    )
    return pd.DataFrame(rows)


def _robustness_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "Gold",
                "Horizon": 5,
                "AvgTradeCount": 1.0,
                "LowTradeCountRate_%": 100.0,
                "BeatBuyHoldRate_%": 60.0,
                "MedianVsBuyHold_%": 1.2,
                "WorstVsBuyHold_%": -6.0,
                "WorstMaxDrawdown_%": -12.0,
                "CostFragilityScore": 5.0,
                "StabilityScore": 80.0,
                "RobustnessScore": 48.0,
            },
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "AvgTradeCount": 4.0,
                "LowTradeCountRate_%": 0.0,
                "BeatBuyHoldRate_%": 35.0,
                "MedianVsBuyHold_%": -2.0,
                "WorstVsBuyHold_%": -18.0,
                "WorstMaxDrawdown_%": -16.0,
                "CostFragilityScore": 10.0,
                "StabilityScore": 80.0,
                "RobustnessScore": 32.0,
            },
            {
                "Asset": "Silver",
                "Horizon": 10,
                "AvgTradeCount": 4.0,
                "LowTradeCountRate_%": 0.0,
                "BeatBuyHoldRate_%": 62.0,
                "MedianVsBuyHold_%": 1.5,
                "WorstVsBuyHold_%": -5.0,
                "WorstMaxDrawdown_%": -13.0,
                "CostFragilityScore": 60.0,
                "StabilityScore": 80.0,
                "RobustnessScore": 50.0,
            },
            {
                "Asset": "Crude Oil",
                "Horizon": 30,
                "AvgTradeCount": 4.0,
                "LowTradeCountRate_%": 0.0,
                "BeatBuyHoldRate_%": 64.0,
                "MedianVsBuyHold_%": 1.0,
                "WorstVsBuyHold_%": -7.0,
                "WorstMaxDrawdown_%": -15.0,
                "CostFragilityScore": 10.0,
                "StabilityScore": 0.0,
                "RobustnessScore": 45.0,
            },
        ]
    )


def test_evidence_quality_diagnostics_multi_asset_outputs_columns():
    report = run_evidence_quality_diagnostics(
        full_evidence_table=_full_evidence(),
        robustness_summary=_robustness_summary(),
    )

    assert set(EVIDENCE_QUALITY_COLUMNS).issubset(report.evidence_quality_table.columns)
    assert set(SIGNAL_COVERAGE_COLUMNS).issubset(report.signal_coverage_table.columns)
    assert set(FAILURE_REASON_COLUMNS).issubset(report.candidate_failure_reason_table.columns)
    assert {"Gold", "Bitcoin", "Silver", "Crude Oil", "Gold ETF"}.issubset(set(report.evidence_quality_table["Asset"]))


def test_low_trade_count_produces_insufficient_trade_coverage():
    report = run_evidence_quality_diagnostics(full_evidence_table=_full_evidence(), robustness_summary=_robustness_summary())
    row = report.candidate_failure_reason_table[report.candidate_failure_reason_table["Asset"].eq("Gold")].iloc[0]

    assert row["PrimaryFailureCategory"] == "InsufficientTradeCoverage"
    assert "threshold/cooldown sensitivity" in row["NextResearchAction"]


def test_poor_vs_buyhold_produces_benchmark_dominated():
    report = run_evidence_quality_diagnostics(full_evidence_table=_full_evidence(), robustness_summary=_robustness_summary())
    row = report.candidate_failure_reason_table[report.candidate_failure_reason_table["Asset"].eq("Bitcoin")].iloc[0]

    assert row["PrimaryFailureCategory"] == "BenchmarkDominated"


def test_cost_sensitivity_produces_cost_fragile():
    report = run_evidence_quality_diagnostics(full_evidence_table=_full_evidence(), robustness_summary=_robustness_summary())
    row = report.candidate_failure_reason_table[report.candidate_failure_reason_table["Asset"].eq("Silver")].iloc[0]

    assert row["PrimaryFailureCategory"] == "CostFragile"


def test_unstable_windows_produce_window_unstable():
    report = run_evidence_quality_diagnostics(full_evidence_table=_full_evidence(), robustness_summary=_robustness_summary())
    row = report.candidate_failure_reason_table[report.candidate_failure_reason_table["Asset"].eq("Crude Oil")].iloc[0]

    assert row["PrimaryFailureCategory"] == "WindowUnstable"


def test_phase8d_does_not_promote_candidates_and_keeps_failures_visible():
    report = run_evidence_quality_diagnostics(full_evidence_table=_full_evidence(), robustness_summary=_robustness_summary())

    assert report.settings["does_not_promote_candidates"] is True
    assert report.settings["production_ready_label_allowed"] is False
    assert "Gold ETF" in set(report.evidence_quality_table["Asset"])
    assert not report.next_research_action_table.empty


if __name__ == "__main__":
    test_evidence_quality_diagnostics_multi_asset_outputs_columns()
    test_low_trade_count_produces_insufficient_trade_coverage()
    test_poor_vs_buyhold_produces_benchmark_dominated()
    test_cost_sensitivity_produces_cost_fragile()
    test_unstable_windows_produce_window_unstable()
    test_phase8d_does_not_promote_candidates_and_keeps_failures_visible()
    print("Phase 8D evidence quality diagnostics tests passed.")
