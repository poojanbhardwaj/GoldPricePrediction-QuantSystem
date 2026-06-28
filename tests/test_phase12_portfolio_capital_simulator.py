from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import src.artifact_store as store
from src.artifact_store import load_latest_artifact, save_phase_artifacts
from src.asset_config import get_asset_names
from src.portfolio_capital_simulator import (
    ALLOCATION_PLAN_COLUMNS,
    POSITION_SIZING_COLUMNS,
    PORTFOLIO_SUMMARY_COLUMNS,
    SCENARIO_COLUMNS,
    WARNING_COLUMNS,
    run_portfolio_capital_simulator,
)


def _with_temp_store(fn):
    old_root = store.ARTIFACT_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        store.ARTIFACT_ROOT = Path(tmp) / "artifacts"
        try:
            return fn()
        finally:
            store.ARTIFACT_ROOT = old_root


def _ranked_rows():
    return pd.DataFrame(
        [
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "ResearchAction": "PaperTradeOnly",
                "Decision": "PaperTradeOnly",
                "CapitalDeploymentStatus": "Blocked",
                "EvidenceScore": 62,
                "OpportunityScore": 75,
                "RiskScore": 35,
                "ActionabilityScore": 70,
                "ReviewDate": "2026-07-01",
                "InvalidationRule": "Invalidate after repeated paper losses.",
                "MainWarnings": "PaperOnly",
            },
            {
                "Asset": "Silver",
                "Horizon": 5,
                "ResearchAction": "Watchlist",
                "Decision": "Watchlist",
                "CapitalDeploymentStatus": "NotReady",
                "EvidenceScore": 40,
                "OpportunityScore": 58,
                "RiskScore": 55,
                "ActionabilityScore": 50,
                "ReviewDate": "2026-07-02",
                "InvalidationRule": "Recheck after forward evidence matures.",
                "MainWarnings": "PendingEvidenceOnly",
            },
            {
                "Asset": "Gold",
                "Horizon": 1,
                "ResearchAction": "Avoid",
                "Decision": "Avoid",
                "CapitalDeploymentStatus": "Blocked",
                "EvidenceScore": 8,
                "OpportunityScore": 12,
                "RiskScore": 90,
                "ActionabilityScore": 15,
                "ReviewDate": "2026-07-03",
                "InvalidationRule": "Archive unless evidence improves.",
                "MainWarnings": "ProbabilityUnreliable; BenchmarkDominated",
            },
            {
                "Asset": "Crude Oil",
                "Horizon": 5,
                "ResearchAction": "PaperTradeOnly",
                "Decision": "PaperTradeOnly",
                "CapitalDeploymentStatus": "ConditionalMicroCapitalEligible",
                "EvidenceScore": 78,
                "OpportunityScore": 74,
                "RiskScore": 24,
                "ActionabilityScore": 76,
                "ReviewDate": "2026-07-04",
                "InvalidationRule": "De-risk if drawdown or benchmark warning appears.",
                "MainWarnings": "ConditionalOnly",
            },
            {
                "Asset": "S&P 500",
                "Horizon": 10,
                "ResearchAction": "PaperTradeOnly",
                "Decision": "PaperTradeOnly",
                "CapitalDeploymentStatus": "NotReady",
                "EvidenceScore": 58,
                "OpportunityScore": 68,
                "RiskScore": 42,
                "ActionabilityScore": 64,
                "ReviewDate": "2026-07-05",
                "InvalidationRule": "Invalidate if benchmark edge weakens.",
                "MainWarnings": "PaperOnly",
            },
            {
                "Asset": "Gold ETF",
                "Horizon": 20,
                "ResearchAction": "PaperTradeOnly",
                "Decision": "PaperTradeOnly",
                "CapitalDeploymentStatus": "NotReady",
                "EvidenceScore": 54,
                "OpportunityScore": 63,
                "RiskScore": 45,
                "ActionabilityScore": 60,
                "ReviewDate": "2026-07-06",
                "InvalidationRule": "Invalidate if drawdown warning repeats.",
                "MainWarnings": "PaperOnly",
            },
        ]
    )


def _paper_table():
    ranked = _ranked_rows()
    return ranked[ranked["ResearchAction"].eq("PaperTradeOnly")].copy()


def _watchlist_table():
    ranked = _ranked_rows()
    return ranked[ranked["ResearchAction"].eq("Watchlist")].copy()


def _capital_table(real_allowed=True):
    return pd.DataFrame(
        [
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "RealCapitalAllowed": False,
                "CapitalDeploymentStatus": "NotReady",
                "CapitalPlanType": "PaperOnly",
                "MainCapitalBlocker": "ForwardEvidence",
                "WhatWouldAllowRealCapital": "More matured forward outcomes.",
                "MaxAllowedRealCapitalPct": 0.0,
                "MaxLossPerIdeaPct": 0.0,
                "ReviewDate": "2026-07-01",
            },
            {
                "Asset": "Silver",
                "Horizon": 5,
                "RealCapitalAllowed": False,
                "CapitalDeploymentStatus": "NotReady",
                "CapitalPlanType": "PaperOnly",
                "MainCapitalBlocker": "PendingEvidenceOnly",
                "WhatWouldAllowRealCapital": "Wait for outcomes.",
                "MaxAllowedRealCapitalPct": 0.0,
                "MaxLossPerIdeaPct": 0.0,
                "ReviewDate": "2026-07-02",
            },
            {
                "Asset": "Gold",
                "Horizon": 1,
                "RealCapitalAllowed": False,
                "CapitalDeploymentStatus": "Blocked",
                "CapitalPlanType": "NoRealCapital",
                "MainCapitalBlocker": "ProbabilityUnreliable",
                "WhatWouldAllowRealCapital": "Improve calibration and benchmark evidence.",
                "MaxAllowedRealCapitalPct": 0.0,
                "MaxLossPerIdeaPct": 0.0,
                "ReviewDate": "2026-07-03",
            },
            {
                "Asset": "Crude Oil",
                "Horizon": 5,
                "RealCapitalAllowed": bool(real_allowed),
                "CapitalDeploymentStatus": "ConditionalMicroCapitalEligible" if real_allowed else "NotReady",
                "CapitalPlanType": "MicroCapitalTrial" if real_allowed else "PaperOnly",
                "MainCapitalBlocker": "NoCapitalBlocker" if real_allowed else "ForwardEvidence",
                "WhatWouldAllowRealCapital": "Continue monitoring.",
                "MaxAllowedRealCapitalPct": 0.8 if real_allowed else 0.0,
                "MaxLossPerIdeaPct": 0.2 if real_allowed else 0.0,
                "ReviewDate": "2026-07-04",
            },
            {
                "Asset": "S&P 500",
                "Horizon": 10,
                "RealCapitalAllowed": False,
                "CapitalDeploymentStatus": "NotReady",
                "CapitalPlanType": "PaperOnly",
                "MainCapitalBlocker": "ForwardEvidence",
                "WhatWouldAllowRealCapital": "More matured outcomes.",
                "MaxAllowedRealCapitalPct": 0.0,
                "MaxLossPerIdeaPct": 0.0,
                "ReviewDate": "2026-07-05",
            },
            {
                "Asset": "Gold ETF",
                "Horizon": 20,
                "RealCapitalAllowed": False,
                "CapitalDeploymentStatus": "NotReady",
                "CapitalPlanType": "PaperOnly",
                "MainCapitalBlocker": "ForwardEvidence",
                "WhatWouldAllowRealCapital": "More matured outcomes.",
                "MaxAllowedRealCapitalPct": 0.0,
                "MaxLossPerIdeaPct": 0.0,
                "ReviewDate": "2026-07-06",
            },
        ]
    )


def _structured_table():
    return pd.DataFrame(
        [
            {
                "Asset": "Crude Oil",
                "Horizon": 5,
                "CapitalPlanType": "MicroCapitalTrial",
                "MaxAllowedRealCapitalPct": 0.8,
                "MaxLossPerIdeaPct": 0.2,
                "PositionSizingRule": "Use the smaller of cap and loss limit.",
                "EntryCondition": "Only if signal remains active.",
                "ExitCondition": "Exit after target date or invalidation.",
                "StopOrInvalidationRule": "Stop after repeated losses or drawdown warning.",
                "ReviewDate": "2026-07-04",
            }
        ]
    )


def _blockers():
    return pd.DataFrame(
        [
            {
                "Asset": "Gold",
                "Horizon": 1,
                "ResearchAction": "Avoid",
                "CapitalDeploymentStatus": "Blocked",
                "MainCapitalBlocker": "ProbabilityUnreliable",
                "FailedGates": "Calibration; Benchmark",
                "WhatWouldAllowRealCapital": "Better probability and benchmark evidence.",
            },
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "ResearchAction": "PaperTradeOnly",
                "CapitalDeploymentStatus": "NotReady",
                "MainCapitalBlocker": "ForwardEvidence",
                "FailedGates": "ForwardEvidence",
                "WhatWouldAllowRealCapital": "More matured outcomes.",
            },
            {
                "Asset": "S&P 500",
                "Horizon": 10,
                "ResearchAction": "PaperTradeOnly",
                "CapitalDeploymentStatus": "NotReady",
                "MainCapitalBlocker": "ForwardEvidence",
                "FailedGates": "ForwardEvidence",
                "WhatWouldAllowRealCapital": "More matured outcomes.",
            },
            {
                "Asset": "Gold ETF",
                "Horizon": 20,
                "ResearchAction": "PaperTradeOnly",
                "CapitalDeploymentStatus": "NotReady",
                "MainCapitalBlocker": "ForwardEvidence",
                "FailedGates": "ForwardEvidence",
                "WhatWouldAllowRealCapital": "More matured outcomes.",
            },
        ]
    )


def _raw():
    rows = []
    for asset, horizon in [("Bitcoin", 5), ("Crude Oil", 5), ("S&P 500", 10), ("Gold ETF", 20)]:
        for i in range(12):
            rows.append(
                {
                    "Asset": asset,
                    "Horizon": horizon,
                    "ProbabilityUp": 0.64,
                    "ActualDirection": 1 if i % 4 else 0,
                    "RealizedReturn": 0.01 if i % 4 else -0.003,
                    "VsBuyHold": 0.004,
                    "MaxDrawdownDuringTrade": -0.04,
                }
            )
    return pd.DataFrame(rows)


def _prob():
    return pd.DataFrame(
        [
            {"Asset": "Bitcoin", "Horizon": 5, "BrierScore": 0.2, "CalibrationGrade": "UsefulButNoisy"},
            {"Asset": "Crude Oil", "Horizon": 5, "BrierScore": 0.18, "CalibrationGrade": "UsefulButNoisy"},
            {"Asset": "Gold", "Horizon": 1, "BrierScore": 0.34, "CalibrationGrade": "ProbabilityUnreliable", "Warnings": "ProbabilityUnreliable"},
            {"Asset": "S&P 500", "Horizon": 10, "BrierScore": 0.22, "CalibrationGrade": "UsefulButNoisy"},
            {"Asset": "Gold ETF", "Horizon": 20, "BrierScore": 0.23, "CalibrationGrade": "UsefulButNoisy"},
        ]
    )


def _base_inputs(real_allowed=True):
    ranked = _ranked_rows()
    return {
        "ranked_asset_horizon_plan": ranked,
        "plan_card_table": ranked,
        "paper_trade_plan_table": _paper_table(),
        "watchlist_table": _watchlist_table(),
        "capital_eligibility_table": _capital_table(real_allowed=real_allowed),
        "structured_capital_plan_table": _structured_table() if real_allowed else pd.DataFrame(),
        "capital_blocker_table": _blockers(),
        "true_raw_trade_log": _raw(),
        "probability_calibration_summary": _prob(),
        "forward_signal_log": pd.DataFrame(
            [
                {"Asset": "Bitcoin", "Horizon": 5, "Status": "Pending", "TargetOutcomeDate": "2026-07-01"},
                {"Asset": "Crude Oil", "Horizon": 5, "Status": "Pending", "TargetOutcomeDate": "2026-07-04"},
                {"Asset": "S&P 500", "Horizon": 10, "Status": "Pending", "TargetOutcomeDate": "2026-07-05"},
                {"Asset": "Gold ETF", "Horizon": 20, "Status": "Pending", "TargetOutcomeDate": "2026-07-06"},
            ]
        ),
        "evidence_health_table": pd.DataFrame(
            [
                {"Asset": "Bitcoin", "Horizon": 5, "PendingSignalCount": 1, "MaturedForwardCount": 4, "WarningCount": 1},
                {"Asset": "Crude Oil", "Horizon": 5, "PendingSignalCount": 1, "MaturedForwardCount": 12, "WarningCount": 0},
                {"Asset": "S&P 500", "Horizon": 10, "PendingSignalCount": 1, "MaturedForwardCount": 6, "WarningCount": 1},
                {"Asset": "Gold ETF", "Horizon": 20, "PendingSignalCount": 1, "MaturedForwardCount": 5, "WarningCount": 1},
            ]
        ),
    }


def test_false_real_capital_forces_zero_real_weight_and_paper_gets_allocation():
    report = run_portfolio_capital_simulator(
        **_base_inputs(real_allowed=False),
        assets=["Bitcoin", "Gold", "S&P 500", "Gold ETF"],
        horizons=[1, 5, 10, 20],
        total_paper_capital=10000,
        portfolio_mode="Balanced Research",
    )

    assert not report.allocation_plan_table.empty
    assert report.allocation_plan_table["SuggestedRealWeightPct"].astype(float).eq(0).all()
    btc = report.allocation_plan_table[report.allocation_plan_table["Asset"].eq("Bitcoin")].iloc[0]
    gold = report.allocation_plan_table[report.allocation_plan_table["Asset"].eq("Gold")].iloc[0]
    assert btc["AllocationMode"] == "PaperOnly"
    assert btc["SuggestedPaperWeightPct"] > 0
    assert gold["AllocationMode"] == "NoAllocation"
    assert gold["SuggestedPaperWeightPct"] == 0
    summary = report.portfolio_summary_table.iloc[0]
    assert summary["TotalPaperAllocatedPct"] > 0
    assert summary["PaperReservePct"] < 100
    assert "AllocationScore" in report.allocation_plan_table.columns
    assert "PaperAllocationStatus" in report.allocation_plan_table.columns
    assert "PaperReservePct" in report.portfolio_summary_table.columns


def test_balanced_mode_allocates_to_multiple_paper_candidates_when_caps_allow():
    report = run_portfolio_capital_simulator(
        **_base_inputs(real_allowed=False),
        assets=["Bitcoin", "S&P 500", "Gold ETF"],
        horizons=[5, 10, 20],
        portfolio_mode="Balanced Research",
        max_single_asset_exposure_pct=40,
        max_single_horizon_exposure_pct=40,
    )

    paper = report.allocation_plan_table[report.allocation_plan_table["AllocationMode"].eq("PaperOnly")]
    allocated = paper[paper["SuggestedPaperWeightPct"].astype(float) > 0]
    assert len(allocated) > 1
    assert allocated["PaperAllocationStatus"].eq("Allocated").all()
    summary = report.portfolio_summary_table.iloc[0]
    assert summary["TotalPaperAllocatedPct"] > 0
    assert summary["NumberAllocatedPaperCandidates"] == len(allocated)
    assert summary["NumberEligiblePaperCandidates"] >= len(allocated)
    assert 0 <= summary["PaperReservePct"] < 100
    assert report.allocation_plan_table["SuggestedRealWeightPct"].astype(float).eq(0).all()
    all_paper_loss = report.scenario_analysis_table[
        report.scenario_analysis_table["Scenario"].eq("All paper candidates lose")
    ].iloc[0]
    expected_exposure = f"{float(summary['TotalPaperAllocatedPct']):.2f}%"
    assert expected_exposure in str(all_paper_loss["PaperImpact"])
    assert "across 0.00% simulated paper exposure" not in str(all_paper_loss["PaperImpact"])


def test_paper_weights_respect_asset_and_horizon_caps():
    report = run_portfolio_capital_simulator(
        **_base_inputs(real_allowed=False),
        assets=["Bitcoin", "S&P 500", "Gold ETF"],
        horizons=[5, 10, 20],
        portfolio_mode="Balanced Research",
        max_single_asset_exposure_pct=25,
        max_single_horizon_exposure_pct=30,
    )

    allocation = report.allocation_plan_table
    for _, group in allocation.groupby("Asset"):
        assert group["SuggestedPaperWeightPct"].astype(float).sum() <= 25.0001
    for _, group in allocation.groupby("Horizon"):
        assert group["SuggestedPaperWeightPct"].astype(float).sum() <= 30.0001


def test_zero_weight_paper_candidates_include_reason():
    report = run_portfolio_capital_simulator(
        **_base_inputs(real_allowed=False),
        assets=["Bitcoin", "S&P 500", "Gold ETF"],
        horizons=[5, 10, 20],
        portfolio_mode="Conservative",
        max_single_asset_exposure_pct=20,
        max_single_horizon_exposure_pct=20,
    )

    paper = report.allocation_plan_table[report.allocation_plan_table["AllocationMode"].eq("PaperOnly")]
    zero_paper = paper[paper["SuggestedPaperWeightPct"].astype(float).eq(0)]
    assert not zero_paper.empty
    assert zero_paper["ZeroWeightReason"].astype(str).str.len().gt(0).all()
    assert set(zero_paper["PaperAllocationStatus"]).issubset({"EligibleButNotAllocated"})


def test_true_real_capital_allows_only_capped_conditional_allocation():
    report = run_portfolio_capital_simulator(
        **_base_inputs(real_allowed=True),
        assets=["Crude Oil"],
        horizons=[5],
        max_real_capital_cap_pct=0.5,
        max_single_idea_loss_pct=0.25,
    )

    row = report.allocation_plan_table.iloc[0]
    assert row["RealCapitalAllowed"] is True or row["RealCapitalAllowed"] == True
    assert row["AllocationMode"] == "ConditionalMicroCapital"
    assert 0 < row["SuggestedRealWeightPct"] <= 0.5
    assert not report.conditional_real_capital_table.empty


def test_watchlist_and_blocked_candidates_remain_visible():
    report = run_portfolio_capital_simulator(
        **_base_inputs(real_allowed=False),
        assets=["Silver", "Gold"],
        horizons=[1, 5],
        include_watchlist_candidates=True,
        include_blocked_candidates=True,
    )

    modes = set(report.allocation_plan_table["AllocationMode"].astype(str))
    assert "WatchlistOnly" in modes
    assert "NoAllocation" in modes
    assert not report.capital_blocker_table.empty


def test_all_assets_and_horizons_supported():
    report = run_portfolio_capital_simulator(assets=get_asset_names(), horizons=[1, 5, 10, 20, 30])

    assert len(report.allocation_plan_table) == len(get_asset_names()) * 5
    assert set(get_asset_names()).issubset(set(report.allocation_plan_table["Asset"].astype(str)))
    assert {1, 5, 10, 20, 30}.issubset(set(report.allocation_plan_table["Horizon"].astype(int)))


def test_missing_optional_artifacts_do_not_crash():
    def run():
        report = run_portfolio_capital_simulator(use_artifact_store=True, assets=["Gold"], horizons=[1])

        assert not report.input_source_table.empty
        assert not report.warning_table.empty
        assert report.input_source_table["Status"].astype(str).str.contains("Missing").any()

    _with_temp_store(run)


def test_loads_latest_artifacts_from_artifact_store():
    def run():
        inputs = _base_inputs(real_allowed=True)
        save_phase_artifacts(
            "Phase 10 Actionable Research Plan",
            {
                "plan_card_table": inputs["plan_card_table"],
                "ranked_asset_horizon_plan": inputs["ranked_asset_horizon_plan"],
                "paper_trade_plan_table": inputs["paper_trade_plan_table"],
                "watchlist_table": inputs["watchlist_table"],
                "risk_budget_table": pd.DataFrame(columns=["Asset", "Horizon"]),
            },
        )
        save_phase_artifacts(
            "Phase 11 Daily Research Control Center",
            {
                "capital_eligibility_table": inputs["capital_eligibility_table"],
                "structured_capital_plan_table": inputs["structured_capital_plan_table"],
                "capital_blocker_table": inputs["capital_blocker_table"],
                "active_paper_signals_table": pd.DataFrame(columns=["Asset", "Horizon"]),
                "pending_outcomes_table": pd.DataFrame(columns=["Asset", "Horizon"]),
                "top_paper_candidates_today": inputs["paper_trade_plan_table"],
                "evidence_health_table": inputs["evidence_health_table"],
                "warning_table": pd.DataFrame(columns=["Asset", "Horizon", "WarningType"]),
            },
        )
        save_phase_artifacts("Phase 8I True Raw Trade Logs", {"true_raw_trade_log": inputs["true_raw_trade_log"]})
        save_phase_artifacts("Phase 8F Probability Calibration", {"probability_calibration_summary": inputs["probability_calibration_summary"]})
        save_phase_artifacts("Phase 9 Forward Paper Evidence", {"forward_signal_log": inputs["forward_signal_log"]})

        report = run_portfolio_capital_simulator(use_artifact_store=True, assets=["Crude Oil"], horizons=[5], max_real_capital_cap_pct=0.5)

        assert not report.input_source_table.empty
        assert "LatestSavedArtifact" in set(report.input_source_table["Source"].astype(str))
        assert report.allocation_plan_table.iloc[0]["Asset"] == "Crude Oil"

    _with_temp_store(run)


def test_position_sizing_scenarios_warnings_and_columns():
    report = run_portfolio_capital_simulator(**_base_inputs(real_allowed=True), assets=["Bitcoin", "Crude Oil"], horizons=[5])

    assert set(PORTFOLIO_SUMMARY_COLUMNS).issubset(report.portfolio_summary_table.columns)
    assert set(ALLOCATION_PLAN_COLUMNS).issubset(report.allocation_plan_table.columns)
    assert set(POSITION_SIZING_COLUMNS).issubset(report.position_sizing_table.columns)
    assert set(SCENARIO_COLUMNS).issubset(report.scenario_analysis_table.columns)
    assert set(WARNING_COLUMNS).issubset(report.warning_table.columns)
    scenarios = set(report.scenario_analysis_table["Scenario"].astype(str))
    assert {
        "Base case",
        "Higher cost/slippage",
        "Worst candidate loss",
        "All paper candidates lose",
        "Benchmark dominates",
        "Drawdown shock",
        "Probability calibration failure",
    }.issubset(scenarios)
    assert not report.warning_table.empty


def test_no_forbidden_live_trading_language_appears():
    report = run_portfolio_capital_simulator(**_base_inputs(real_allowed=True), assets=["Bitcoin", "Crude Oil", "Gold"], horizons=[1, 5])
    text = "\n".join(
        table.astype(str).to_csv(index=False)
        for table in [
            report.portfolio_summary_table,
            report.allocation_plan_table,
            report.position_sizing_table,
            report.scenario_analysis_table,
            report.next_actions_table,
            report.warning_table,
        ]
    )

    for phrase in ["Buy", "Strong Buy", "Invest Now", "Guaranteed", "Safe Profit", "Production Ready"]:
        assert phrase not in text


def test_outputs_autosave_to_artifact_store():
    def run():
        report = run_portfolio_capital_simulator(**_base_inputs(real_allowed=True), assets=["Crude Oil"], horizons=[5], autosave=True)

        assert report.saved_artifacts
        latest = load_latest_artifact("Phase 12 Portfolio Capital Simulator", "portfolio_summary_table", required=True)
        assert not latest.empty

    _with_temp_store(run)


if __name__ == "__main__":
    test_false_real_capital_forces_zero_real_weight_and_paper_gets_allocation()
    test_balanced_mode_allocates_to_multiple_paper_candidates_when_caps_allow()
    test_paper_weights_respect_asset_and_horizon_caps()
    test_zero_weight_paper_candidates_include_reason()
    test_true_real_capital_allows_only_capped_conditional_allocation()
    test_watchlist_and_blocked_candidates_remain_visible()
    test_all_assets_and_horizons_supported()
    test_missing_optional_artifacts_do_not_crash()
    test_loads_latest_artifacts_from_artifact_store()
    test_position_sizing_scenarios_warnings_and_columns()
    test_no_forbidden_live_trading_language_appears()
    test_outputs_autosave_to_artifact_store()
    print("Phase 12 portfolio capital simulator tests passed.")
