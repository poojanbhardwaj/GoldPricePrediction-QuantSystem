from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import src.artifact_store as store
from src.artifact_store import load_latest_artifact
from src.asset_config import get_asset_names
from src.risk_warning_intelligence import (
    RISK_SUMMARY_COLUMNS,
    TOP_RISKS_COLUMNS,
    CAPITAL_BLOCKING_COLUMNS,
    PAPER_ONLY_COLUMNS,
    WARNING_GROUP_COLUMNS,
    RISK_MATRIX_COLUMNS,
    RISK_STATUS_COLUMNS,
    NEXT_ACTION_COLUMNS,
    RAW_WARNING_COLUMNS,
    run_risk_warning_intelligence,
)


def _with_temp_store(fn):
    old_root = store.ARTIFACT_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        store.ARTIFACT_ROOT = Path(tmp) / "artifacts"
        try:
            return fn()
        finally:
            store.ARTIFACT_ROOT = old_root


def _synthetic_inputs():
    return {
        "probability_calibration_warnings": pd.DataFrame(
            [
                {
                    "Asset": "Gold",
                    "Horizon": 1,
                    "WarningType": "ProbabilityUnreliable",
                    "Severity": "High",
                    "Message": "Brier score is weak.",
                },
                {
                    "Asset": "Bitcoin",
                    "Horizon": 5,
                    "WarningType": "Overconfident",
                    "Severity": "High",
                    "Message": "High-confidence failures detected.",
                },
            ]
        ),
        "forward_warning_table": pd.DataFrame(
            [
                {
                    "Asset": "ALL",
                    "Horizon": "ALL",
                    "WarningType": "NotFinancialAdvice",
                    "Severity": "Critical",
                    "Message": "Research and educational disclaimer.",
                },
                {
                    "Asset": "Silver",
                    "Horizon": 5,
                    "WarningType": "PendingEvidenceOnly",
                    "Severity": "Medium",
                    "Message": "Forward evidence has not matured yet.",
                },
                {
                    "Asset": "Crude Oil",
                    "Horizon": 5,
                    "WarningType": "LowTradeCount",
                    "Severity": "Medium",
                    "Message": "Trade count remains low.",
                },
            ]
        ),
        "phase11_capital_blocker_table": pd.DataFrame(
            [
                {
                    "Asset": "Gold ETF",
                    "Horizon": 20,
                    "MainCapitalBlocker": "RealCapitalBlocked",
                    "FailedGates": "ForwardEvidence",
                    "WhatWouldAllowRealCapital": "More matured forward outcomes.",
                    "ResearchAction": "PaperTradeOnly",
                }
            ]
        ),
        "allocation_plan_table": pd.DataFrame(
            [
                {
                    "Asset": "S&P 500",
                    "Horizon": 10,
                    "AllocationMode": "PaperOnly",
                    "SuggestedPaperWeightPct": 12.5,
                    "MainWarning": "DrawdownRisk",
                    "PortfolioContributionReason": "Paper allocation exists but sizing should be reduced.",
                },
                {
                    "Asset": "Gold",
                    "Horizon": 30,
                    "AllocationMode": "NoAllocation",
                    "SuggestedPaperWeightPct": 0.0,
                    "MainWarning": "DataQualityRisk",
                    "ZeroWeightReason": "Missing evidence.",
                },
            ]
        ),
        "portfolio_drawdown_stress_table": pd.DataFrame(
            [
                {
                    "Scenario": "Drawdown shock",
                    "Breach": True,
                    "Warning": "DrawdownRisk",
                    "Notes": "Clustered drawdown warning appears.",
                }
            ]
        ),
        "cost_slippage_stress_table": pd.DataFrame(
            [
                {
                    "Asset": "Bitcoin",
                    "Horizon": 5,
                    "Warning": "CostFragile",
                    "StressResult": "Higher cost removes edge.",
                }
            ]
        ),
        "correlation_concentration_table": pd.DataFrame(
            [
                {
                    "ConcentrationType": "Asset",
                    "Bucket": "Bitcoin",
                    "Warning": "ConcentrationRisk",
                }
            ]
        ),
    }


def test_phase13_produces_all_required_tables():
    report = run_risk_warning_intelligence(**_synthetic_inputs(), assets=get_asset_names(), horizons=[1, 5, 10, 20, 30])

    assert set(RISK_SUMMARY_COLUMNS).issubset(report.risk_summary_table.columns)
    assert set(TOP_RISKS_COLUMNS).issubset(report.top_risks_table.columns)
    assert set(CAPITAL_BLOCKING_COLUMNS).issubset(report.capital_blocking_risks_table.columns)
    assert set(PAPER_ONLY_COLUMNS).issubset(report.paper_only_risks_table.columns)
    assert set(WARNING_GROUP_COLUMNS).issubset(report.warning_group_table.columns)
    assert set(RISK_MATRIX_COLUMNS).issubset(report.asset_horizon_risk_matrix.columns)
    assert set(RISK_STATUS_COLUMNS).issubset(report.risk_trend_or_status_table.columns)
    assert set(NEXT_ACTION_COLUMNS).issubset(report.next_risk_actions_table.columns)
    assert set(RAW_WARNING_COLUMNS).issubset(report.raw_warning_evidence.columns)
    assert not report.risk_summary_table.empty


def test_top_risks_are_ranked_by_score_descending():
    report = run_risk_warning_intelligence(**_synthetic_inputs(), assets=get_asset_names(), horizons=[1, 5])
    scores = report.top_risks_table["RiskScore"].astype(float).tolist()
    assert scores == sorted(scores, reverse=True)
    assert report.top_risks_table["Rank"].tolist() == list(range(1, len(report.top_risks_table) + 1))


def test_capital_blocking_and_paper_only_risks_are_separated():
    report = run_risk_warning_intelligence(**_synthetic_inputs(), assets=get_asset_names(), horizons=[1, 5, 10, 20, 30])

    assert not report.capital_blocking_risks_table.empty
    assert not report.paper_only_risks_table.empty
    blockers = set(report.capital_blocking_risks_table["BlockingReason"].astype(str))
    assert "RealCapitalBlocked" in blockers
    paper_statuses = set(report.paper_only_risks_table["PaperStatus"].astype(str))
    assert {"PaperOnlyAllowed", "MonitorOnly", "ReducePaperSize"} & paper_statuses


def test_real_capital_blocked_is_classified_as_blocks_real_capital():
    report = run_risk_warning_intelligence(**_synthetic_inputs(), assets=["Gold ETF"], horizons=[20])
    match = report.raw_warning_evidence[report.raw_warning_evidence["RiskCategory"].eq("RealCapitalBlocked")]
    assert not match.empty
    assert match["CapitalImpact"].eq("BlocksRealCapital").all()


def test_drawdown_risk_reduces_paper_size_when_paper_allocation_exists():
    report = run_risk_warning_intelligence(**_synthetic_inputs(), assets=["S&P 500"], horizons=[10])
    match = report.raw_warning_evidence[
        report.raw_warning_evidence["Asset"].eq("S&P 500")
        & report.raw_warning_evidence["RiskCategory"].eq("DrawdownRisk")
    ]
    assert not match.empty
    assert "ReducesPaperSize" in set(match["PaperImpact"].astype(str))


def test_pending_evidence_is_not_fake_fatal_error():
    report = run_risk_warning_intelligence(**_synthetic_inputs(), assets=["Silver"], horizons=[5])
    match = report.raw_warning_evidence[report.raw_warning_evidence["RiskCategory"].eq("PendingEvidenceOnly")]
    assert not match.empty
    assert set(match["PaperImpact"].astype(str)).issubset({"PaperOnlyAllowed", "MonitorOnly"})
    assert not match["Severity"].eq("Critical").any()


def test_not_financial_advice_is_info_and_does_not_block_paper():
    report = run_risk_warning_intelligence(**_synthetic_inputs(), assets=get_asset_names(), horizons=[1, 5, 10, 20, 30])
    match = report.raw_warning_evidence[report.raw_warning_evidence["WarningType"].eq("NotFinancialAdvice")]

    assert not match.empty
    assert match["RiskCategory"].eq("ComplianceNotice").all()
    assert match["Severity"].eq("Info").all()
    assert match["RiskScore"].astype(float).le(5).all()
    assert match["CapitalImpact"].eq("NoRealCapitalImpact").all()
    assert match["PaperImpact"].eq("NoPaperImpact").all()
    assert "DataQualityRisk" not in set(match["RiskCategory"].astype(str))


def test_phase12_paper_allocation_sets_matrix_status_to_allocated():
    report = run_risk_warning_intelligence(**_synthetic_inputs(), assets=get_asset_names(), horizons=[1, 5, 10, 20, 30])
    matrix = report.asset_horizon_risk_matrix
    row = matrix[matrix["Asset"].eq("S&P 500") & matrix["Horizon"].astype(int).eq(10)].iloc[0]

    assert row["PaperStatus"] == "PaperAllocated"
    assert not matrix["PaperStatus"].eq("Blocked").all()


def test_missing_exit_price_only_affects_its_own_asset_horizon():
    inputs = _synthetic_inputs()
    inputs["phase12_warning_table"] = pd.DataFrame(
        [
            {
                "Asset": "ALL",
                "Horizon": "ALL",
                "WarningType": "NotFinancialAdvice",
                "Severity": "Critical",
                "Message": "Keep disclaimer visible.",
            },
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "WarningType": "MissingExitPrice",
                "Severity": "Critical",
                "Message": "Exit price missing for this paper row.",
            },
        ]
    )
    inputs["allocation_plan_table"] = pd.DataFrame(
        [
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "AllocationMode": "PaperOnly",
                "ResearchAction": "PaperTradeOnly",
                "SuggestedPaperWeightPct": 0.0,
                "PaperAllocationStatus": "EligibleButNotAllocated",
                "MainWarning": "MissingExitPrice",
            },
            {
                "Asset": "Silver",
                "Horizon": 5,
                "AllocationMode": "PaperOnly",
                "ResearchAction": "PaperTradeOnly",
                "SuggestedPaperWeightPct": 8.0,
                "PaperAllocationStatus": "Allocated",
                "MainWarning": "PendingEvidenceOnly",
            },
        ]
    )

    report = run_risk_warning_intelligence(**inputs, assets=["Bitcoin", "Silver", "Gold"], horizons=[5])
    matrix = report.asset_horizon_risk_matrix
    bitcoin = matrix[matrix["Asset"].eq("Bitcoin") & matrix["Horizon"].astype(int).eq(5)].iloc[0]
    silver = matrix[matrix["Asset"].eq("Silver") & matrix["Horizon"].astype(int).eq(5)].iloc[0]
    gold = matrix[matrix["Asset"].eq("Gold") & matrix["Horizon"].astype(int).eq(5)].iloc[0]

    assert bitcoin["PaperStatus"] == "Blocked"
    assert silver["PaperStatus"] == "PaperAllocated"
    assert gold["PaperStatus"] != "Blocked"


def test_data_quality_risk_only_appears_for_real_data_issues():
    report = run_risk_warning_intelligence(
        forward_warning_table=pd.DataFrame(
            [
                {"Asset": "ALL", "Horizon": "ALL", "WarningType": "NotFinancialAdvice", "Severity": "Critical"},
                {"Asset": "Gold", "Horizon": 1, "WarningType": "MissingExitPrice", "Severity": "Critical"},
            ]
        ),
        assets=["Gold"],
        horizons=[1],
    )

    compliance = report.raw_warning_evidence[report.raw_warning_evidence["WarningType"].eq("NotFinancialAdvice")]
    data_quality = report.raw_warning_evidence[report.raw_warning_evidence["RiskCategory"].eq("DataQualityRisk")]
    assert not compliance.empty
    assert compliance["RiskCategory"].eq("ComplianceNotice").all()
    assert set(data_quality["WarningType"].astype(str)) == {"MissingExitPrice"}


def test_all_configured_assets_can_appear_in_matrix():
    report = run_risk_warning_intelligence(**_synthetic_inputs(), assets=get_asset_names(), horizons=[1, 5, 10, 20, 30])
    assert set(get_asset_names()).issubset(set(report.asset_horizon_risk_matrix["Asset"].astype(str)))
    assert len(report.asset_horizon_risk_matrix) == len(get_asset_names()) * 5


def test_missing_optional_artifacts_are_handled_gracefully():
    report = run_risk_warning_intelligence(assets=["Gold"], horizons=[1])
    assert report.risk_summary_table.empty
    assert len(report.asset_horizon_risk_matrix) == 1
    assert report.asset_horizon_risk_matrix.iloc[0]["OverallRiskScore"] == 0


def test_no_forbidden_live_trading_language_appears():
    report = run_risk_warning_intelligence(**_synthetic_inputs(), assets=get_asset_names(), horizons=[1, 5, 10, 20, 30])
    text = "\n".join(
        table.astype(str).to_csv(index=False)
        for table in [
            report.risk_summary_table,
            report.top_risks_table,
            report.capital_blocking_risks_table,
            report.paper_only_risks_table,
            report.warning_group_table,
            report.asset_horizon_risk_matrix,
            report.next_risk_actions_table,
        ]
    )
    for phrase in ["Buy", "Strong Buy", "Invest Now", "Production Ready", "Guaranteed", "Safe Profit"]:
        assert phrase not in text


def test_autosaves_outputs_to_artifact_store():
    def run():
        report = run_risk_warning_intelligence(**_synthetic_inputs(), assets=["Gold", "Silver"], horizons=[1, 5], autosave=True)
        assert report.saved_artifacts
        latest = load_latest_artifact("phase13_risk_warning_intelligence", "risk_summary_table", required=True)
        assert not latest.empty

    _with_temp_store(run)


if __name__ == "__main__":
    test_phase13_produces_all_required_tables()
    test_top_risks_are_ranked_by_score_descending()
    test_capital_blocking_and_paper_only_risks_are_separated()
    test_real_capital_blocked_is_classified_as_blocks_real_capital()
    test_drawdown_risk_reduces_paper_size_when_paper_allocation_exists()
    test_pending_evidence_is_not_fake_fatal_error()
    test_not_financial_advice_is_info_and_does_not_block_paper()
    test_phase12_paper_allocation_sets_matrix_status_to_allocated()
    test_missing_exit_price_only_affects_its_own_asset_horizon()
    test_data_quality_risk_only_appears_for_real_data_issues()
    test_all_configured_assets_can_appear_in_matrix()
    test_missing_optional_artifacts_are_handled_gracefully()
    test_no_forbidden_live_trading_language_appears()
    test_autosaves_outputs_to_artifact_store()
    print("Phase 13 risk warning intelligence tests passed.")
