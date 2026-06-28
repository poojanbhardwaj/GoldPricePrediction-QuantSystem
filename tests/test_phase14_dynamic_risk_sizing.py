from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import src.artifact_store as store
from src.artifact_store import load_latest_artifact
from src.asset_config import get_asset_names
from src.dynamic_risk_sizing import (
    CAP_ADJUSTMENT_COLUMNS,
    DRAWDOWN_BUDGET_COLUMNS,
    DYNAMIC_POSITION_COLUMNS,
    DYNAMIC_SUMMARY_COLUMNS,
    NEXT_ACTION_COLUMNS,
    OPTIMIZED_PORTFOLIO_COLUMNS,
    RISK_MULTIPLIER_COLUMNS,
    RISK_MULTIPLIER_SUMMARY_COLUMNS,
    SCENARIO_COLUMNS,
    ZERO_SIZE_COLUMNS,
    run_dynamic_risk_sizing,
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
    allocation = pd.DataFrame(
        [
            {
                "Asset": "Gold",
                "Horizon": 1,
                "ResearchAction": "PaperTradeOnly",
                "SuggestedPaperWeightPct": 10.0,
                "SuggestedRealWeightPct": 0.0,
                "RealCapitalAllowed": False,
                "PaperAllocationStatus": "Allocated",
                "CapitalStatus": "Blocked",
            },
            {
                "Asset": "Silver",
                "Horizon": 5,
                "ResearchAction": "PaperTradeOnly",
                "SuggestedPaperWeightPct": 12.0,
                "SuggestedRealWeightPct": 0.0,
                "RealCapitalAllowed": False,
                "PaperAllocationStatus": "Allocated",
                "CapitalStatus": "Blocked",
            },
            {
                "Asset": "Crude Oil",
                "Horizon": 5,
                "ResearchAction": "PaperTradeOnly",
                "SuggestedPaperWeightPct": 8.0,
                "SuggestedRealWeightPct": 0.0,
                "RealCapitalAllowed": False,
                "PaperAllocationStatus": "Allocated",
                "CapitalStatus": "Blocked",
            },
            {
                "Asset": "Bitcoin",
                "Horizon": 10,
                "ResearchAction": "PaperTradeOnly",
                "SuggestedPaperWeightPct": 9.0,
                "SuggestedRealWeightPct": 0.0,
                "RealCapitalAllowed": False,
                "PaperAllocationStatus": "Allocated",
                "CapitalStatus": "Blocked",
            },
            {
                "Asset": "S&P 500",
                "Horizon": 20,
                "ResearchAction": "PaperTradeOnly",
                "SuggestedPaperWeightPct": 6.0,
                "SuggestedRealWeightPct": 0.0,
                "RealCapitalAllowed": False,
                "PaperAllocationStatus": "Allocated",
                "CapitalStatus": "Blocked",
            },
            {
                "Asset": "Gold ETF",
                "Horizon": 30,
                "ResearchAction": "Watchlist",
                "SuggestedPaperWeightPct": 0.0,
                "SuggestedRealWeightPct": 0.0,
                "RealCapitalAllowed": False,
                "PaperAllocationStatus": "WatchlistOnly",
                "CapitalStatus": "Blocked",
            },
        ]
    )
    risk_matrix = pd.DataFrame(
        [
            {"Asset": "Gold", "Horizon": 1, "OverallRiskScore": 2, "PaperStatus": "PaperAllocated", "CapitalStatus": "Blocked"},
            {"Asset": "Silver", "Horizon": 5, "OverallRiskScore": 72, "PaperStatus": "PaperAllocated", "CapitalStatus": "Blocked"},
            {"Asset": "Crude Oil", "Horizon": 5, "OverallRiskScore": 68, "PaperStatus": "PaperAllocated", "CapitalStatus": "Blocked"},
            {"Asset": "Bitcoin", "Horizon": 10, "OverallRiskScore": 88, "PaperStatus": "Blocked", "CapitalStatus": "Blocked"},
            {"Asset": "S&P 500", "Horizon": 20, "OverallRiskScore": 38, "PaperStatus": "PaperOnlyAllowed", "CapitalStatus": "Blocked"},
            {"Asset": "Gold ETF", "Horizon": 30, "OverallRiskScore": 0, "PaperStatus": "WatchlistOnly", "CapitalStatus": "Blocked"},
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "EvidenceSource": "Phase 13",
                "Asset": "Gold",
                "Horizon": 1,
                "WarningType": "NotFinancialAdvice",
                "RiskCategory": "ComplianceNotice",
                "Severity": "Info",
                "RiskScore": 2,
                "CapitalImpact": "NoRealCapitalImpact",
                "PaperImpact": "NoPaperImpact",
                "Message": "Disclaimer.",
                "RecommendedAction": "Keep disclaimer visible.",
            },
            {
                "EvidenceSource": "Phase 13",
                "Asset": "Silver",
                "Horizon": 5,
                "WarningType": "ProbabilityUnreliable",
                "RiskCategory": "ProbabilityUnreliable",
                "Severity": "High",
                "RiskScore": 72,
                "CapitalImpact": "BlocksRealCapital",
                "PaperImpact": "MonitorOnly",
                "Message": "Probability calibration is weak.",
                "RecommendedAction": "Recalibrate probability model before confidence use.",
            },
            {
                "EvidenceSource": "Phase 13",
                "Asset": "Crude Oil",
                "Horizon": 5,
                "WarningType": "DrawdownRisk",
                "RiskCategory": "DrawdownRisk",
                "Severity": "High",
                "RiskScore": 68,
                "CapitalImpact": "ReducesRealCapital",
                "PaperImpact": "ReducesPaperSize",
                "Message": "Drawdown stress breach.",
                "RecommendedAction": "Reduce simulated size.",
            },
            {
                "EvidenceSource": "Phase 13",
                "Asset": "Bitcoin",
                "Horizon": 10,
                "WarningType": "MissingExitPrice",
                "RiskCategory": "DataQualityRisk",
                "Severity": "Critical",
                "RiskScore": 88,
                "CapitalImpact": "BlocksRealCapital",
                "PaperImpact": "BlocksPaper",
                "Message": "Exit price is missing.",
                "RecommendedAction": "Fix missing outcome price.",
            },
            {
                "EvidenceSource": "Phase 13",
                "Asset": "S&P 500",
                "Horizon": 20,
                "WarningType": "PendingEvidenceOnly",
                "RiskCategory": "PendingEvidenceOnly",
                "Severity": "Medium",
                "RiskScore": 38,
                "CapitalImpact": "BlocksRealCapital",
                "PaperImpact": "PaperOnlyAllowed",
                "Message": "Evidence is still young.",
                "RecommendedAction": "Wait for outcome maturation.",
            },
        ]
    )
    return {
        "allocation_plan_table": allocation,
        "paper_portfolio_table": allocation[allocation["SuggestedPaperWeightPct"].gt(0)].copy(),
        "asset_horizon_risk_matrix": risk_matrix,
        "raw_warning_evidence": raw,
        "portfolio_drawdown_stress_table": pd.DataFrame(
            [{"Scenario": "Drawdown shock", "Breach": True, "Notes": "Drawdown stress breach."}]
        ),
        "cost_slippage_stress_table": pd.DataFrame(columns=["Asset", "Horizon", "Warning"]),
        "correlation_concentration_table": pd.DataFrame(columns=["Asset", "Horizon", "Warning"]),
    }


def _run_report(**kwargs):
    params = {
        "assets": get_asset_names(),
        "horizons": [1, 5, 10, 20, 30],
        "portfolio_mode": "Balanced Research",
        "max_single_asset_exposure_pct": 25,
        "max_single_horizon_exposure_pct": 20,
    }
    params.update(kwargs)
    return run_dynamic_risk_sizing(
        **_synthetic_inputs(),
        **params,
    )


def test_phase14_produces_all_required_output_tables():
    report = _run_report()

    assert set(DYNAMIC_SUMMARY_COLUMNS).issubset(report.dynamic_sizing_summary_table.columns)
    assert set(DYNAMIC_POSITION_COLUMNS).issubset(report.dynamic_position_sizing_table.columns)
    assert set(RISK_MULTIPLIER_COLUMNS).issubset(report.risk_multiplier_table.columns)
    assert set(RISK_MULTIPLIER_SUMMARY_COLUMNS).issubset(report.risk_multiplier_summary_table.columns)
    assert set(CAP_ADJUSTMENT_COLUMNS).issubset(report.cap_adjustment_table.columns)
    assert set(ZERO_SIZE_COLUMNS).issubset(report.zero_size_table.columns)
    assert set(OPTIMIZED_PORTFOLIO_COLUMNS).issubset(report.optimized_portfolio_table.columns)
    assert set(DRAWDOWN_BUDGET_COLUMNS).issubset(report.drawdown_budget_table.columns)
    assert set(SCENARIO_COLUMNS).issubset(report.risk_adjusted_scenarios_table.columns)
    assert set(NEXT_ACTION_COLUMNS).issubset(report.next_sizing_actions_table.columns)


def test_real_capital_remains_zero_when_gates_fail():
    report = _run_report()
    assert report.dynamic_position_sizing_table["RealCapitalAllowed"].astype(bool).eq(False).all()
    assert report.dynamic_position_sizing_table["SuggestedRealWeightPct"].astype(float).eq(0).all()
    assert report.dynamic_sizing_summary_table.iloc[0]["RealCapitalExposurePct"] == 0


def test_missing_exit_price_zeros_only_affected_asset_horizon():
    report = _run_report()
    sizing = report.dynamic_position_sizing_table
    bitcoin = sizing[sizing["Asset"].eq("Bitcoin") & sizing["Horizon"].astype(int).eq(10)].iloc[0]
    silver = sizing[sizing["Asset"].eq("Silver") & sizing["Horizon"].astype(int).eq(5)].iloc[0]

    assert bitcoin["OptimizedPaperWeightPct"] == 0
    assert bitcoin["SizingDecision"] == "ZeroDueToDataIssue"
    assert "MissingExitPrice" in str(bitcoin["ZeroSizeReason"])
    assert silver["OptimizedPaperWeightPct"] > 0


def test_compliance_notice_does_not_reduce_paper_size():
    report = _run_report()
    gold = report.dynamic_position_sizing_table[
        report.dynamic_position_sizing_table["Asset"].eq("Gold")
        & report.dynamic_position_sizing_table["Horizon"].astype(int).eq(1)
    ].iloc[0]

    assert gold["Phase12PaperWeightPct"] == gold["OptimizedPaperWeightPct"]
    assert gold["OverallRiskMultiplier"] == 1.0


def test_probability_unreliable_reduces_paper_size():
    report = _run_report()
    silver = report.dynamic_position_sizing_table[
        report.dynamic_position_sizing_table["Asset"].eq("Silver")
        & report.dynamic_position_sizing_table["Horizon"].astype(int).eq(5)
    ].iloc[0]

    assert silver["OptimizedPaperWeightPct"] < silver["Phase12PaperWeightPct"]
    assert silver["ProbabilityReliabilityMultiplier"] < 1


def test_drawdown_risk_reduces_paper_size_when_drawdown_breach_exists():
    report = _run_report()
    crude = report.dynamic_position_sizing_table[
        report.dynamic_position_sizing_table["Asset"].eq("Crude Oil")
        & report.dynamic_position_sizing_table["Horizon"].astype(int).eq(5)
    ].iloc[0]

    assert crude["OptimizedPaperWeightPct"] < crude["Phase12PaperWeightPct"]
    assert crude["DrawdownMultiplier"] <= 0.5


def test_pending_evidence_reduces_but_does_not_zero_paper():
    report = _run_report()
    spx = report.dynamic_position_sizing_table[
        report.dynamic_position_sizing_table["Asset"].eq("S&P 500")
        & report.dynamic_position_sizing_table["Horizon"].astype(int).eq(20)
    ].iloc[0]

    assert 0 < spx["OptimizedPaperWeightPct"] < spx["Phase12PaperWeightPct"]
    assert spx["ForwardEvidenceMultiplier"] < 1


def test_optimized_exposure_is_positive_and_not_above_starting_by_default():
    report = _run_report()
    summary = report.dynamic_sizing_summary_table.iloc[0]

    assert summary["OptimizedPaperExposurePct"] > 0
    assert summary["OptimizedPaperExposurePct"] <= summary["StartingPaperExposurePct"]
    assert not report.optimized_portfolio_table.empty


def test_asset_and_horizon_caps_are_respected():
    report = _run_report(max_single_asset_exposure_pct=8, max_single_horizon_exposure_pct=10)
    sizing = report.dynamic_position_sizing_table

    for _, group in sizing.groupby("Asset"):
        assert group["OptimizedPaperWeightPct"].astype(float).sum() <= 8.0001
    for _, group in sizing.groupby("Horizon"):
        assert group["OptimizedPaperWeightPct"].astype(float).sum() <= 10.0001
    assert not report.cap_adjustment_table.empty
    assert report.cap_adjustment_table["BeforeAdjustmentPct"].notna().all()
    assert report.cap_adjustment_table["AfterAdjustmentPct"].notna().all()


def test_zero_sized_rows_include_reason():
    report = _run_report()
    assert not report.zero_size_table.empty
    assert report.zero_size_table["ZeroSizeReason"].astype(str).str.len().gt(0).all()


def test_risk_adjusted_scenarios_use_optimized_exposure():
    report = _run_report()
    scenario = report.risk_adjusted_scenarios_table[
        report.risk_adjusted_scenarios_table["Scenario"].eq("All optimized paper candidates lose")
    ].iloc[0]
    summary = report.dynamic_sizing_summary_table.iloc[0]

    assert scenario["OptimizedPortfolioImpact"] == summary["OptimizedPaperExposurePct"]


def test_micro_paper_tracking_verdict_when_optimized_exposure_under_five():
    report = _run_report(max_portfolio_paper_exposure_pct=4)
    summary = report.dynamic_sizing_summary_table.iloc[0]

    assert 0 < summary["OptimizedPaperExposurePct"] < 5
    assert summary["OverallSizingVerdict"] == "Micro paper tracking mode."
    assert "Real capital remains blocked" in str(summary["MainReason"])
    assert "simulated paper sizing only" in str(summary["MainReason"])


def test_highest_risk_allocated_candidate_scenario_has_nonzero_optimized_impact():
    report = _run_report()
    scenario = report.risk_adjusted_scenarios_table[
        report.risk_adjusted_scenarios_table["Scenario"].eq("Highest risk allocated candidate fails")
    ].iloc[0]

    assert scenario["OptimizedPortfolioImpact"] > 0
    assert scenario["StartingPortfolioImpact"] >= scenario["OptimizedPortfolioImpact"]
    assert "Silver" in str(scenario["Explanation"])
    assert "5D" in str(scenario["Explanation"])


def test_highest_risk_zero_sized_candidate_scenario_appears_when_zeroed():
    report = _run_report()
    scenario = report.risk_adjusted_scenarios_table[
        report.risk_adjusted_scenarios_table["Scenario"].eq("Highest risk zero-sized candidate excluded")
    ].iloc[0]

    assert scenario["StartingPortfolioImpact"] > 0
    assert scenario["OptimizedPortfolioImpact"] == 0
    assert "Bitcoin" in str(scenario["Explanation"])


def test_risk_multiplier_summary_has_grouped_rows():
    report = _run_report()

    assert not report.risk_multiplier_summary_table.empty
    assert set(RISK_MULTIPLIER_SUMMARY_COLUMNS).issubset(report.risk_multiplier_summary_table.columns)
    assert "ProbabilityUnreliable" in set(report.risk_multiplier_summary_table["RiskCategory"].astype(str))
    assert report.risk_multiplier_summary_table["Count"].astype(int).ge(1).all()


def test_no_forbidden_live_trading_language_appears():
    report = _run_report()
    text = "\n".join(
        table.astype(str).to_csv(index=False)
        for table in [
            report.dynamic_sizing_summary_table,
            report.dynamic_position_sizing_table,
            report.risk_multiplier_summary_table,
            report.risk_multiplier_table,
            report.cap_adjustment_table,
            report.zero_size_table,
            report.optimized_portfolio_table,
            report.drawdown_budget_table,
            report.risk_adjusted_scenarios_table,
            report.next_sizing_actions_table,
        ]
    )
    for phrase in ["Buy", "Strong Buy", "Invest Now", "Production Ready", "Guaranteed", "Safe Profit"]:
        assert phrase not in text


def test_autosaves_outputs_to_artifact_store():
    def run():
        report = run_dynamic_risk_sizing(**_synthetic_inputs(), assets=["Gold", "Silver"], horizons=[1, 5], autosave=True)
        assert report.saved_artifacts
        latest = load_latest_artifact("phase14_dynamic_risk_sizing", "dynamic_sizing_summary_table", required=True)
        assert not latest.empty

    _with_temp_store(run)


if __name__ == "__main__":
    test_phase14_produces_all_required_output_tables()
    test_real_capital_remains_zero_when_gates_fail()
    test_missing_exit_price_zeros_only_affected_asset_horizon()
    test_compliance_notice_does_not_reduce_paper_size()
    test_probability_unreliable_reduces_paper_size()
    test_drawdown_risk_reduces_paper_size_when_drawdown_breach_exists()
    test_pending_evidence_reduces_but_does_not_zero_paper()
    test_optimized_exposure_is_positive_and_not_above_starting_by_default()
    test_asset_and_horizon_caps_are_respected()
    test_zero_sized_rows_include_reason()
    test_risk_adjusted_scenarios_use_optimized_exposure()
    test_micro_paper_tracking_verdict_when_optimized_exposure_under_five()
    test_highest_risk_allocated_candidate_scenario_has_nonzero_optimized_impact()
    test_highest_risk_zero_sized_candidate_scenario_appears_when_zeroed()
    test_risk_multiplier_summary_has_grouped_rows()
    test_no_forbidden_live_trading_language_appears()
    test_autosaves_outputs_to_artifact_store()
    print("Phase 14 dynamic risk sizing tests passed.")
