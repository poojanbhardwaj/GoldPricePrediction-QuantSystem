from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.asset_config import get_asset_names
from src.action_plan_engine import (
    ENTRY_TRIGGER_COLUMNS,
    EVIDENCE_SCORECARD_COLUMNS,
    INVALIDATION_RULE_COLUMNS,
    NEXT_EVIDENCE_COLUMNS,
    PLAN_CARD_COLUMNS,
    RANKED_PLAN_COLUMNS,
    RISK_BUDGET_COLUMNS,
    WARNING_COLUMNS,
    WATCHLIST_COLUMNS,
    run_actionable_research_plan,
)


def _probability_summary() -> pd.DataFrame:
    rows = []
    for asset in get_asset_names():
        for horizon in [1, 5, 10, 20, 30]:
            rows.append(
                {
                    "Asset": asset,
                    "Horizon": horizon,
                    "RawProbabilityOutcomesAvailable": True,
                    "TotalTrades": 12 if asset == "Bitcoin" and horizon == 5 else 3,
                    "BrierScore": 0.18 if asset == "Bitcoin" and horizon == 5 else 0.34,
                    "ECE": 0.08 if asset == "Bitcoin" and horizon == 5 else 0.22,
                    "CalibrationGrade": "UsefulButNoisy" if asset == "Bitcoin" and horizon == 5 else "ProbabilityUnreliable",
                    "UsefulProbabilityFilterFound": asset == "Bitcoin" and horizon == 5,
                    "Warnings": "" if asset == "Bitcoin" and horizon == 5 else "ProbabilityUnreliable; LowTradeCount",
                }
            )
    return pd.DataFrame(rows)


def _probability_warnings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Asset": "Gold", "Horizon": 5, "WarningType": "ProbabilityUnreliable"},
            {"Asset": "Crude Oil", "Horizon": 5, "WarningType": "Overconfident"},
            {"Asset": "S&P 500", "Horizon": 10, "WarningType": "BenchmarkDominated"},
        ]
    )


def _true_raw() -> pd.DataFrame:
    rows = []
    for i in range(12):
        rows.append(
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "ProbabilityUp": 0.62 + (i % 3) * 0.03,
                "ActualDirection": 1 if i % 4 else 0,
                "RealizedReturn": 0.012 if i % 4 else -0.006,
                "VsBuyHold": 0.006 if i % 4 else -0.002,
                "MaxDrawdownDuringTrade": -0.05,
                "EvidenceMode": "ReconstructedTradeLevel",
            }
        )
    rows.append(
        {
            "Asset": "S&P 500",
            "Horizon": 10,
            "ProbabilityUp": 0.60,
            "ActualDirection": 0,
            "RealizedReturn": -0.01,
            "VsBuyHold": -0.03,
            "MaxDrawdownDuringTrade": -0.08,
            "EvidenceMode": "ReconstructedTradeLevel",
            "Warnings": "BenchmarkDominated",
        }
    )
    return pd.DataFrame(rows)


def _forward_log(include_matured=True) -> pd.DataFrame:
    rows = [
        {
            "SignalId": "btc5_pending",
            "Asset": "Bitcoin",
            "Horizon": 5,
            "SignalDate": "2026-06-20",
            "TargetOutcomeDate": "2026-06-27",
            "ModelName": "UnitModel",
            "ProbabilityUp": 0.69,
            "PredictedDirection": "Up",
            "SignalStrength": "Medium",
            "Status": "Pending",
            "ActualDirection": pd.NA,
            "ExitPrice": pd.NA,
            "WinLoss": "",
            "EvidenceMode": "ForwardPaperSignal",
            "Warnings": "PendingOutcome",
        }
    ]
    if include_matured:
        for i in range(10):
            rows.append(
                {
                    "SignalId": f"btc5_matured_{i}",
                    "Asset": "Bitcoin",
                    "Horizon": 5,
                    "SignalDate": f"2026-05-{i + 1:02d}",
                    "TargetOutcomeDate": f"2026-05-{i + 8:02d}",
                    "ModelName": "UnitModel",
                    "ProbabilityUp": 0.64,
                    "PredictedDirection": "Up",
                    "SignalStrength": "Medium",
                    "Status": "Matured",
                    "ActualDirection": 1 if i != 0 else 0,
                    "RealizedReturn": 0.01 if i != 0 else -0.004,
                    "VsBuyHold": 0.004 if i != 0 else -0.002,
                    "WinLoss": "Win" if i != 0 else "Loss",
                    "BeatBenchmark": i != 0,
                    "EvidenceMode": "ForwardPaperSignal",
                    "Warnings": "",
                }
            )
    return pd.DataFrame(rows)


def _moderate_watchlist_forward_log() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SignalId": "silver5_pending",
                "Asset": "Silver",
                "Horizon": 5,
                "SignalDate": "2026-06-20",
                "TargetOutcomeDate": "2026-06-27",
                "ModelName": "UnitModel",
                "ProbabilityUp": 0.63,
                "PredictedDirection": "Up",
                "SignalStrength": "Medium",
                "Status": "Pending",
                "ActualDirection": pd.NA,
                "ExitPrice": pd.NA,
                "WinLoss": "",
                "EvidenceMode": "ForwardPaperSignal",
                "Warnings": "PendingOutcome",
            }
        ]
    )


def _forward_accuracy(include_matured=True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "TotalSignals": 11 if include_matured else 1,
                "PendingSignals": 1,
                "MaturedSignals": 10 if include_matured else 0,
                "WinRate_%": 90.0 if include_matured else pd.NA,
                "AvgRealizedReturn_%": 0.86 if include_matured else pd.NA,
                "AvgVsBuyHold_%": 0.34 if include_matured else pd.NA,
                "WorstRealizedReturn_%": -0.4 if include_matured else pd.NA,
                "Warnings": "" if include_matured else "NotEnoughForwardEvidence",
            }
        ]
    )


def _forward_probability(include_matured=True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": "Bitcoin",
                "Horizon": 5,
                "Rows": 10 if include_matured else 0,
                "BrierScore": 0.19 if include_matured else pd.NA,
                "ECE": 0.09 if include_matured else pd.NA,
                "CalibrationVerdict": "Forward probability evidence useful" if include_matured else "Not enough forward evidence yet",
                "Warnings": "" if include_matured else "NotEnoughForwardEvidence",
            }
        ]
    )


def _plan(include_matured=True, **kwargs):
    return run_actionable_research_plan(
        probability_calibration_summary=_probability_summary(),
        probability_calibration_warnings=_probability_warnings(),
        true_raw_trade_log=_true_raw(),
        forward_signal_log=_forward_log(include_matured=include_matured),
        forward_accuracy_summary=_forward_accuracy(include_matured=include_matured),
        forward_probability_calibration_summary=_forward_probability(include_matured=include_matured),
        assets=kwargs.pop("assets", get_asset_names()),
        horizons=kwargs.pop("horizons", [1, 5, 10, 20, 30]),
        risk_appetite=kwargs.pop("risk_appetite", "Balanced Research"),
        minimum_evidence_score=kwargs.pop("minimum_evidence_score", 30.0),
        include_blocked_candidates=kwargs.pop("include_blocked_candidates", True),
        top_n_plan_cards=kwargs.pop("top_n_plan_cards", 8),
        **kwargs,
    )


def _all_output_text(report) -> str:
    parts = []
    for table in [
        report.executive_decision_table,
        report.ranked_asset_horizon_plan,
        report.plan_card_table,
        report.entry_trigger_table,
        report.invalidation_rule_table,
        report.risk_budget_table,
        report.evidence_scorecard,
        report.blocked_candidates_table,
        report.watchlist_table,
        report.paper_trade_plan_table,
        report.next_evidence_needed_table,
        report.warnings_table,
    ]:
        parts.append(table.astype(str).to_csv(index=False))
    return "\n".join(parts)


def test_all_assets_and_horizons_supported():
    report = _plan()

    assert len(report.ranked_asset_horizon_plan) == len(get_asset_names()) * 5
    assert set(get_asset_names()).issubset(set(report.ranked_asset_horizon_plan["Asset"]))
    assert {1, 5, 10, 20, 30}.issubset(set(report.ranked_asset_horizon_plan["Horizon"].astype(int)))


def test_weak_evidence_still_returns_top_research_watch_candidates():
    report = run_actionable_research_plan(assets=get_asset_names(), horizons=[1, 5, 10, 20, 30], include_blocked_candidates=True)

    assert not report.plan_card_table.empty
    assert len(report.plan_card_table) >= 3
    assert report.executive_decision_table["Summary"].astype(str).str.contains("research", case=False).any()


def test_capital_blocked_but_paper_plan_not_empty_for_research_opportunity():
    report = _plan(include_matured=True, assets=["Bitcoin"], horizons=[5])

    assert not report.paper_trade_plan_table.empty
    row = report.paper_trade_plan_table.iloc[0]
    assert row["ResearchAction"] == "PaperTradeOnly"
    assert row["Decision"] == "PaperTradeOnly"
    assert row["CapitalDeploymentStatus"] == "Blocked"
    assert row["DeploymentAllowed"] is False or row["DeploymentAllowed"] == False
    assert row["RealCapitalRiskCap"] == 0
    assert row["PaperTradeAllowed"] is True or row["PaperTradeAllowed"] == True
    assert row["MaxPaperTradeSize"] == "1 paper unit"


def test_watchlist_table_populated_for_moderate_research_opportunity():
    report = run_actionable_research_plan(
        probability_calibration_summary=_probability_summary(),
        forward_signal_log=_moderate_watchlist_forward_log(),
        assets=["Silver"],
        horizons=[5],
        include_blocked_candidates=True,
    )

    assert not report.watchlist_table.empty
    row = report.watchlist_table.iloc[0]
    assert row["ResearchAction"] == "Watchlist"
    assert row["PaperTradeAllowed"] is False or row["PaperTradeAllowed"] == False
    assert report.paper_trade_plan_table.empty


def test_no_forbidden_action_language_in_outputs():
    report = _plan()
    text = _all_output_text(report)

    forbidden = ["Buy", "Strong Buy", "Invest Now", "Production Ready", "Guaranteed", "Safe Profit"]
    for phrase in forbidden:
        assert phrase not in text


def test_probability_unreliable_blocks_deployment_style_decision():
    report = _plan(assets=["Gold"], horizons=[5], include_blocked_candidates=True)
    row = report.ranked_asset_horizon_plan.iloc[0]

    assert row["Decision"] in {"PaperTradeOnly", "Watchlist", "ObserveOnly", "Avoid", "BlockedDueToDataFailure"}
    assert row["CapitalDeploymentStatus"] == "Blocked"
    assert row["DeploymentAllowed"] is False or row["DeploymentAllowed"] == False
    assert row["RealCapitalRiskCap"] == 0
    assert row["RiskCap"] == "0 real capital"
    assert "ProbabilityUnreliable" in row["MainWarnings"]


def test_forward_pending_influences_opportunity_not_matured_accuracy():
    pending_report = _plan(include_matured=False, assets=["Bitcoin"], horizons=[5])
    no_pending_report = run_actionable_research_plan(
        probability_calibration_summary=_probability_summary(),
        true_raw_trade_log=_true_raw(),
        assets=["Bitcoin"],
        horizons=[5],
        include_blocked_candidates=True,
    )

    pending_score = pending_report.ranked_asset_horizon_plan.iloc[0]["OpportunityScore"]
    no_pending_score = no_pending_report.ranked_asset_horizon_plan.iloc[0]["OpportunityScore"]
    matured_count = pending_report.evidence_scorecard.iloc[0]["ForwardMaturedCount"]

    assert pending_score > no_pending_score
    assert matured_count == 0


def test_matured_forward_rows_improve_evidence_score():
    matured_report = _plan(include_matured=True, assets=["Bitcoin"], horizons=[5])
    pending_report = _plan(include_matured=False, assets=["Bitcoin"], horizons=[5])

    assert matured_report.evidence_scorecard.iloc[0]["EvidenceScore"] > pending_report.evidence_scorecard.iloc[0]["EvidenceScore"]


def test_blocked_candidates_remain_visible_when_enabled():
    report = _plan(assets=["S&P 500"], horizons=[10], include_blocked_candidates=True)

    assert not report.blocked_candidates_table.empty
    assert report.blocked_candidates_table.iloc[0]["CapitalDeploymentStatus"] == "Blocked"
    assert report.blocked_candidates_table.iloc[0]["RealCapitalRiskCap"] == 0


def test_data_failure_candidates_remain_blocked_or_avoid():
    report = run_actionable_research_plan(assets=["Gold"], horizons=[1], include_blocked_candidates=True)
    row = report.ranked_asset_horizon_plan.iloc[0]

    assert row["ResearchAction"] in {"BlockedDueToDataFailure", "Avoid"}
    assert row["CapitalDeploymentStatus"] == "Blocked"
    assert row["PaperTradeAllowed"] is False or row["PaperTradeAllowed"] == False
    assert row["RealCapitalRiskCap"] == 0


def test_missing_optional_files_do_not_crash():
    report = run_actionable_research_plan(assets=["Gold"], horizons=[1])

    assert not report.ranked_asset_horizon_plan.empty
    assert not report.warnings_table.empty


def test_export_tables_have_required_columns():
    report = _plan()

    assert set(RANKED_PLAN_COLUMNS).issubset(report.ranked_asset_horizon_plan.columns)
    assert set(PLAN_CARD_COLUMNS).issubset(report.plan_card_table.columns)
    assert set(ENTRY_TRIGGER_COLUMNS).issubset(report.entry_trigger_table.columns)
    assert set(INVALIDATION_RULE_COLUMNS).issubset(report.invalidation_rule_table.columns)
    assert set(RISK_BUDGET_COLUMNS).issubset(report.risk_budget_table.columns)
    assert set(EVIDENCE_SCORECARD_COLUMNS).issubset(report.evidence_scorecard.columns)
    assert set(WATCHLIST_COLUMNS).issubset(report.watchlist_table.columns)
    assert set(NEXT_EVIDENCE_COLUMNS).issubset(report.next_evidence_needed_table.columns)
    assert set(WARNING_COLUMNS).issubset(report.warnings_table.columns)


if __name__ == "__main__":
    test_all_assets_and_horizons_supported()
    test_weak_evidence_still_returns_top_research_watch_candidates()
    test_capital_blocked_but_paper_plan_not_empty_for_research_opportunity()
    test_watchlist_table_populated_for_moderate_research_opportunity()
    test_no_forbidden_action_language_in_outputs()
    test_probability_unreliable_blocks_deployment_style_decision()
    test_forward_pending_influences_opportunity_not_matured_accuracy()
    test_matured_forward_rows_improve_evidence_score()
    test_blocked_candidates_remain_visible_when_enabled()
    test_data_failure_candidates_remain_blocked_or_avoid()
    test_missing_optional_files_do_not_crash()
    test_export_tables_have_required_columns()
    print("Phase 10 actionable research plan tests passed.")
