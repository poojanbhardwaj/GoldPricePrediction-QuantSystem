from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.meta_signal_engine import (
    AUDIT_COLUMNS,
    build_meta_decision_audit,
    calibrate_meta_decision_thresholds,
    explain_meta_decision_rules,
    run_meta_decision_audit,
)


def _meta_decision_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "Silver",
                "Horizon": 5,
                "MetaDecision": "Research Only",
                "MetaConfidenceScore": 64.0,
                "MetaRiskScore": 44.0,
                "RegimeLabel": "Constructive uptrend",
                "SignalReliabilityScore": 68.0,
                "WalkForwardReliabilityScore": 72.0,
                "BenchmarkRiskFlag": False,
                "CostFragilityFlag": False,
                "DrawdownRiskFlag": False,
                "StabilityFlag": "Stable",
                "WalkForwardVerdict": "Strong walk-forward research candidate",
                "BeatBuyHoldRate_%": 66.0,
                "PositiveReturnRate_%": 66.0,
                "MedianLockedVsBuyHold_%": 3.0,
                "WorstLockedMaxDrawdown_%": -12.0,
                "AvgTradesPerWindow": 4.0,
                "Warnings": "",
            },
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "MetaDecision": "Avoid",
                "MetaConfidenceScore": 12.0,
                "MetaRiskScore": 91.0,
                "RegimeLabel": "Risk-off / weak trend",
                "SignalReliabilityScore": 15.0,
                "WalkForwardReliabilityScore": 18.0,
                "BenchmarkRiskFlag": True,
                "CostFragilityFlag": True,
                "DrawdownRiskFlag": True,
                "StabilityFlag": "ThresholdUnstable",
                "WalkForwardVerdict": "Do not trust",
                "BeatBuyHoldRate_%": 17.0,
                "PositiveReturnRate_%": 33.0,
                "MedianLockedVsBuyHold_%": -7.0,
                "WorstLockedMaxDrawdown_%": -31.0,
                "AvgTradesPerWindow": 4.0,
                "Warnings": "BenchmarkRisk; DrawdownRisk",
            },
            {
                "Asset": "Gold",
                "Horizon": 1,
                "MetaDecision": "No Trade",
                "MetaConfidenceScore": 41.0,
                "MetaRiskScore": 62.0,
                "RegimeLabel": "High-volatility drawdown",
                "SignalReliabilityScore": 48.0,
                "WalkForwardReliabilityScore": 50.0,
                "BenchmarkRiskFlag": False,
                "CostFragilityFlag": False,
                "DrawdownRiskFlag": False,
                "StabilityFlag": "Stable",
                "WalkForwardVerdict": "Research candidate",
                "BeatBuyHoldRate_%": 55.0,
                "PositiveReturnRate_%": 60.0,
                "MedianLockedVsBuyHold_%": 1.2,
                "WorstLockedMaxDrawdown_%": -14.0,
                "AvgTradesPerWindow": 3.0,
                "Warnings": "",
            },
        ]
    )


def test_blocked_and_passing_rules_are_generated():
    row = _meta_decision_rows().iloc[0]
    audit = explain_meta_decision_rules(row, "Conservative")

    assert audit["PassingRules"]
    assert audit["BlockingRules"]
    assert "Confidence >=" in audit["MainBlockingRule"]
    assert "WhatWouldNeedToImprove" in audit


def test_rejected_rows_remain_visible_in_audit():
    report = run_meta_decision_audit(_meta_decision_rows(), "Conservative")

    assert len(report.audit_table) == 3
    assert set(AUDIT_COLUMNS).issubset(set(report.audit_table.columns))
    rejected = report.audit_table[report.audit_table["Asset"].eq("Bitcoin")]
    assert len(rejected) == 1
    assert rejected.iloc[0]["Current MetaDecision"] == "Avoid"


def test_weak_rows_do_not_become_trade_in_conservative_or_balanced():
    rows = _meta_decision_rows()
    conservative = build_meta_decision_audit(rows, "Conservative")
    balanced = build_meta_decision_audit(rows, "Balanced")

    btc_conservative = conservative[conservative["Asset"].eq("Bitcoin")].iloc[0]
    btc_balanced = balanced[balanced["Asset"].eq("Bitcoin")].iloc[0]
    assert btc_conservative["Calibrated MetaDecision"] != "Trade"
    assert btc_balanced["Calibrated MetaDecision"] != "Trade"
    assert btc_balanced["Calibrated MetaDecision"] == "Avoid"


def test_aggressive_research_is_explicitly_not_production_ready():
    rows = _meta_decision_rows()
    report = run_meta_decision_audit(rows, "Aggressive Research")

    assert report.settings["production_ready_label_allowed"] is False
    assert any("not production-ready" in warning for warning in report.warnings)
    assert "production-ready" in " ".join(report.audit_table["RuleAuditExplanation"].astype(str).tolist()).lower() or (
        report.audit_table["Warnings"].astype(str).str.contains("AggressiveResearchExperimental").any()
    )


def test_missing_columns_are_handled_safely():
    incomplete = pd.DataFrame([{"Asset": "Gold", "Horizon": 1, "MetaDecision": "Research Only"}])
    report = run_meta_decision_audit(incomplete, "Balanced")

    assert len(report.audit_table) == 1
    row = report.audit_table.iloc[0]
    assert row["Calibrated MetaDecision"] != "Trade"
    assert "Missing columns" in row["BlockingRules"]


def test_multi_asset_multi_horizon_and_mode_outputs():
    rows = _meta_decision_rows()
    report = run_meta_decision_audit(rows, "Balanced")
    thresholds = calibrate_meta_decision_thresholds("Balanced")

    assert set(report.audit_table["Asset"]) == {"Silver", "Bitcoin", "Gold"}
    assert set(report.audit_table["Horizon"]) == {1, 5}
    assert set(report.mode_comparison["Mode"]) == {"Conservative", "Balanced", "Aggressive Research"}
    assert report.mode_comparison["Total"].eq(len(rows)).all()
    assert thresholds["ProductionReadyLabelAllowed"] is False
    assert not report.threshold_config.empty
    assert not report.top_blocked_candidates.empty
    assert not report.highest_confidence_candidates.empty
    assert not report.highest_risk_candidates.empty


if __name__ == "__main__":
    test_blocked_and_passing_rules_are_generated()
    test_rejected_rows_remain_visible_in_audit()
    test_weak_rows_do_not_become_trade_in_conservative_or_balanced()
    test_aggressive_research_is_explicitly_not_production_ready()
    test_missing_columns_are_handled_safely()
    test_multi_asset_multi_horizon_and_mode_outputs()
    print("Phase 8A meta decision audit tests passed.")
