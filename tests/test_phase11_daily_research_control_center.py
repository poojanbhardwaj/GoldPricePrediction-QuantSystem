from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import src.artifact_store as store
from src.artifact_store import load_latest_artifact, save_phase_artifacts
from src.asset_config import get_asset_names
from src.daily_research_center import (
    CAPITAL_ELIGIBILITY_COLUMNS,
    DAILY_NEXT_ACTION_COLUMNS,
    DAILY_RESEARCH_SUMMARY_COLUMNS,
    STRUCTURED_CAPITAL_PLAN_COLUMNS,
    WARNING_COLUMNS,
    run_daily_research_control_center,
)


def _with_temp_store(fn):
    old_root = store.ARTIFACT_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        store.ARTIFACT_ROOT = Path(tmp) / "artifacts"
        try:
            return fn()
        finally:
            store.ARTIFACT_ROOT = old_root


def _strong_raw(asset="Bitcoin", horizon=5, rows=12):
    return pd.DataFrame(
        [
            {
                "Asset": asset,
                "Horizon": horizon,
                "ProbabilityUp": 0.64 + (i % 3) * 0.02,
                "ActualDirection": 1 if i % 5 else 0,
                "RealizedReturn": 0.012 if i % 5 else -0.003,
                "BenchmarkReturn": 0.004,
                "VsBuyHold": 0.006 if i % 5 else 0.001,
                "MaxDrawdownDuringTrade": -0.04,
                "EvidenceMode": "ReconstructedTradeLevel",
            }
            for i in range(rows)
        ]
    )


def _prob_summary(asset="Bitcoin", horizon=5, strong=True):
    return pd.DataFrame(
        [
            {
                "Asset": asset,
                "Horizon": horizon,
                "RawProbabilityOutcomesAvailable": strong,
                "TotalTrades": 12 if strong else 2,
                "BrierScore": 0.18 if strong else 0.34,
                "CalibrationGrade": "UsefulButNoisy" if strong else "ProbabilityUnreliable",
                "Warnings": "" if strong else "ProbabilityUnreliable; LowTradeCount",
            }
        ]
    )


def _prob_warnings(asset="Gold", horizon=5):
    return pd.DataFrame([{"Asset": asset, "Horizon": horizon, "WarningType": "ProbabilityUnreliable"}])


def _forward_log(asset="Bitcoin", horizon=5, report_date="2026-06-25", include_matured=True):
    rows = [
        {
            "SignalId": f"{asset}_{horizon}_pending",
            "Asset": asset,
            "Horizon": horizon,
            "SignalDate": "2026-06-24",
            "TargetOutcomeDate": "2026-06-30",
            "ModelName": "UnitModel",
            "ProbabilityUp": 0.67,
            "PredictedDirection": "Up",
            "SignalStrength": "Medium",
            "Status": "Pending",
            "ActualOutcomeDate": pd.NA,
            "ExitPrice": pd.NA,
            "ActualDirection": pd.NA,
            "WinLoss": "",
            "BeatBenchmark": pd.NA,
            "EvidenceMode": "ForwardPaperSignal",
            "Warnings": "PendingOutcome",
        },
        {
            "SignalId": f"{asset}_{horizon}_overdue",
            "Asset": asset,
            "Horizon": horizon,
            "SignalDate": "2026-06-10",
            "TargetOutcomeDate": "2026-06-15",
            "ModelName": "UnitModel",
            "ProbabilityUp": 0.66,
            "PredictedDirection": "Up",
            "SignalStrength": "Medium",
            "Status": "Pending",
            "ActualOutcomeDate": pd.NA,
            "ExitPrice": pd.NA,
            "ActualDirection": pd.NA,
            "WinLoss": "",
            "BeatBenchmark": pd.NA,
            "EvidenceMode": "ForwardPaperSignal",
            "Warnings": "PendingOutcome",
        },
    ]
    if include_matured:
        for i in range(12):
            rows.append(
                {
                    "SignalId": f"{asset}_{horizon}_matured_{i}",
                    "Asset": asset,
                    "Horizon": horizon,
                    "SignalDate": f"2026-06-{1 + i:02d}",
                    "TargetOutcomeDate": f"2026-06-{6 + i:02d}",
                    "ModelName": "UnitModel",
                    "ProbabilityUp": 0.65,
                    "PredictedDirection": "Up",
                    "SignalStrength": "Medium",
                    "Status": "Matured",
                    "ActualOutcomeDate": report_date if i == 0 else f"2026-06-{6 + i:02d}",
                    "ExitPrice": 105 + i,
                    "ActualDirection": 1 if i % 4 else 0,
                    "RealizedReturn": 0.01 if i % 4 else -0.002,
                    "BenchmarkReturn": 0.003,
                    "VsBuyHold": 0.005 if i % 4 else 0.001,
                    "WinLoss": "Win" if i % 4 else "Loss",
                    "BeatBenchmark": True,
                    "EvidenceMode": "ForwardPaperSignal",
                    "Warnings": "",
                }
            )
    return pd.DataFrame(rows)


def _forward_accuracy(asset="Bitcoin", horizon=5, strong=True):
    return pd.DataFrame(
        [
            {
                "Asset": asset,
                "Horizon": horizon,
                "PendingSignals": 2,
                "MaturedSignals": 12 if strong else 0,
                "WinRate_%": 66.0 if strong else pd.NA,
                "BeatBenchmarkRate_%": 66.0 if strong else pd.NA,
                "AvgVsBuyHold_%": 0.4 if strong else pd.NA,
                "WorstRealizedReturn_%": -0.2 if strong else pd.NA,
                "Warnings": "" if strong else "NotEnoughForwardEvidence",
            }
        ]
    )


def _forward_prob(asset="Bitcoin", horizon=5, strong=True):
    return pd.DataFrame(
        [
            {
                "Asset": asset,
                "Horizon": horizon,
                "Rows": 12 if strong else 0,
                "BrierScore": 0.18 if strong else pd.NA,
                "ECE": 0.08 if strong else pd.NA,
                "CalibrationVerdict": "Forward probability evidence useful" if strong else "Not enough forward evidence yet",
                "Warnings": "" if strong else "NotEnoughForwardEvidence",
            }
        ]
    )


def _phase10_plan(asset="Bitcoin", horizon=5, action="PaperTradeOnly", review_date="2026-06-30"):
    return pd.DataFrame(
        [
            {
                "Rank": 1,
                "Asset": asset,
                "Horizon": horizon,
                "Decision": action,
                "CapitalDeploymentStatus": "Blocked",
                "ResearchAction": action,
                "DeploymentAllowed": False,
                "RealCapitalRiskCap": 0,
                "PaperTradeAllowed": action == "PaperTradeOnly",
                "EvidenceScore": 70,
                "OpportunityScore": 72,
                "RiskScore": 22,
                "ActionabilityScore": 70,
                "ProbabilityUp": 0.67,
                "ReviewDate": review_date,
                "RequiredEvidenceToUpgrade": "continued stable forward paper evidence",
                "MainWarnings": "NotFinancialAdvice; NotProductionReady",
            }
        ]
    )


def _weak_inputs():
    return {
        "true_raw_trade_log": pd.DataFrame(columns=["Asset", "Horizon", "ProbabilityUp", "ActualDirection"]),
        "probability_calibration_summary": _prob_summary("Gold", 5, strong=False),
        "probability_calibration_warnings": _prob_warnings("Gold", 5),
        "forward_signal_log": _forward_log("Gold", 5, include_matured=False),
        "forward_accuracy_summary": _forward_accuracy("Gold", 5, strong=False),
        "forward_probability_calibration_summary": _forward_prob("Gold", 5, strong=False),
        "ranked_asset_horizon_plan": _phase10_plan("Gold", 5, "PaperTradeOnly"),
        "paper_trade_plan_table": _phase10_plan("Gold", 5, "PaperTradeOnly"),
        "watchlist_table": pd.DataFrame(columns=["Asset", "Horizon", "ResearchAction"]),
    }


def _strong_inputs():
    return {
        "true_raw_trade_log": _strong_raw(),
        "probability_calibration_summary": _prob_summary(strong=True),
        "probability_calibration_warnings": pd.DataFrame(columns=["Asset", "Horizon", "WarningType"]),
        "forward_signal_log": _forward_log(include_matured=True),
        "forward_accuracy_summary": _forward_accuracy(strong=True),
        "forward_probability_calibration_summary": _forward_prob(strong=True),
        "ranked_asset_horizon_plan": _phase10_plan(),
        "paper_trade_plan_table": _phase10_plan(),
        "watchlist_table": pd.DataFrame(columns=["Asset", "Horizon", "ResearchAction"]),
    }


def test_pending_matured_and_overdue_outcomes_are_detected():
    report = run_daily_research_control_center(
        **_strong_inputs(),
        assets=["Bitcoin"],
        horizons=[5],
        report_date="2026-06-25",
    )

    assert not report.pending_outcomes_table.empty
    assert not report.matured_today_table.empty
    assert not report.overdue_outcomes_table.empty
    assert report.daily_research_summary.iloc[0]["PendingOutcomes"] >= 2
    assert report.daily_research_summary.iloc[0]["MaturedToday"] >= 1
    assert report.daily_research_summary.iloc[0]["OverdueOutcomes"] >= 1


def test_wait_for_outcome_review_date_uses_earliest_pending_target_date():
    pending = pd.DataFrame(
        [
            {
                "SignalId": "gold5_pending_late",
                "Asset": "Gold",
                "Horizon": 5,
                "SignalDate": "2026-06-18",
                "TargetOutcomeDate": "2026-06-28",
                "ProbabilityUp": 0.62,
                "Status": "Pending",
                "Warnings": "PendingOutcome",
            },
            {
                "SignalId": "gold5_pending_early",
                "Asset": "Gold",
                "Horizon": 5,
                "SignalDate": "2026-06-10",
                "TargetOutcomeDate": "2026-06-18",
                "ProbabilityUp": 0.63,
                "Status": "Pending",
                "Warnings": "PendingOutcome",
            },
        ]
    )
    inputs = _weak_inputs()
    inputs["forward_signal_log"] = pending.copy()
    inputs["pending_outcome_table"] = pending.copy()
    inputs["ranked_asset_horizon_plan"] = _phase10_plan("Gold", 5, "PaperTradeOnly", review_date="2026-07-31")
    inputs["paper_trade_plan_table"] = _phase10_plan("Gold", 5, "PaperTradeOnly", review_date="2026-07-31")

    report = run_daily_research_control_center(
        **inputs,
        assets=["Gold"],
        horizons=[5],
        report_date="2026-06-20",
    )

    wait_rows = report.daily_next_actions_table[report.daily_next_actions_table["DailyAction"].eq("WaitForOutcome")]
    assert not wait_rows.empty
    assert wait_rows.iloc[0]["ReviewDate"] == "2026-06-18"
    assert pd.to_datetime(report.pending_outcomes_table["TargetOutcomeDate"]).min().date().isoformat() == "2026-06-18"
    assert not report.overdue_outcomes_table.empty
    assert "2026-06-18" in set(pd.to_datetime(report.overdue_outcomes_table["TargetOutcomeDate"]).dt.date.astype(str))


def test_real_capital_false_when_evidence_gates_fail_but_paper_plan_visible():
    report = run_daily_research_control_center(
        **_weak_inputs(),
        assets=["Gold"],
        horizons=[5],
        report_date="2026-06-25",
    )

    row = report.capital_eligibility_table.iloc[0]
    assert row["RealCapitalAllowed"] is False or row["RealCapitalAllowed"] == False
    assert row["CapitalDeploymentStatus"] in {"Blocked", "NotReady"}
    assert not report.top_paper_candidates_today.empty
    assert "ProbabilityUnreliable" in set(report.warning_table["WarningType"].astype(str))


def test_real_capital_can_become_true_only_when_strict_synthetic_gates_pass():
    report = run_daily_research_control_center(
        **_strong_inputs(),
        assets=["Bitcoin"],
        horizons=[5],
        report_date="2026-06-25",
        capital_eligibility_mode="Balanced",
        minimum_matured_forward_outcomes=10,
        max_drawdown_allowed_pct=12.0,
        max_real_capital_pct=1.0,
    )

    row = report.capital_eligibility_table.iloc[0]
    assert row["RealCapitalAllowed"] is True or row["RealCapitalAllowed"] == True
    assert row["CapitalDeploymentStatus"] in {"ConditionalMicroCapitalEligible", "ConditionalResearchCapitalEligible"}
    assert not report.structured_capital_plan_table.empty
    plan = report.structured_capital_plan_table.iloc[0]
    assert plan["MaxAllowedRealCapitalPct"] > 0
    assert plan["MaxLossPerIdeaPct"] > 0
    assert "Stop" in plan["StopOrInvalidationRule"]
    assert plan["ReviewDate"]


def test_structured_capital_plan_only_for_eligible_candidates():
    weak = run_daily_research_control_center(**_weak_inputs(), assets=["Gold"], horizons=[5], report_date="2026-06-25")
    strong = run_daily_research_control_center(**_strong_inputs(), assets=["Bitcoin"], horizons=[5], report_date="2026-06-25", capital_eligibility_mode="Balanced")

    assert weak.structured_capital_plan_table.empty
    assert not strong.structured_capital_plan_table.empty


def test_all_assets_and_horizons_supported():
    report = run_daily_research_control_center(assets=get_asset_names(), horizons=[1, 5, 10, 20, 30], report_date="2026-06-25")

    assert len(report.capital_eligibility_table) == len(get_asset_names()) * 5
    assert set(get_asset_names()).issubset(set(report.capital_eligibility_table["Asset"].astype(str)))
    assert {1, 5, 10, 20, 30}.issubset(set(report.capital_eligibility_table["Horizon"].astype(int)))


def test_missing_artifacts_do_not_crash_and_show_diagnostics():
    def run():
        report = run_daily_research_control_center(use_artifact_store=True, assets=["Gold"], horizons=[1], report_date="2026-06-25")

        assert not report.input_source_table.empty
        assert report.input_source_table["Status"].astype(str).str.contains("Missing").any()
        assert "MissingArtifact" in set(report.warning_table["WarningType"].astype(str))

    _with_temp_store(run)


def test_loads_latest_artifacts_from_artifact_store():
    def run():
        inputs = _strong_inputs()
        save_phase_artifacts("Phase 8I True Raw Trade Logs", {"true_raw_trade_log": inputs["true_raw_trade_log"]})
        save_phase_artifacts(
            "Phase 8F Probability Calibration",
            {
                "probability_calibration_summary": inputs["probability_calibration_summary"],
                "probability_calibration_warnings": inputs["probability_calibration_warnings"],
            },
        )
        save_phase_artifacts(
            "Phase 9 Forward Paper Evidence",
            {
                "forward_signal_log": inputs["forward_signal_log"],
                "pending_outcome_table": inputs["forward_signal_log"][inputs["forward_signal_log"]["Status"].eq("Pending")],
                "matured_outcome_table": inputs["forward_signal_log"][inputs["forward_signal_log"]["Status"].eq("Matured")],
                "forward_accuracy_summary": inputs["forward_accuracy_summary"],
                "forward_probability_calibration_summary": inputs["forward_probability_calibration_summary"],
                "warning_table": pd.DataFrame(columns=["Asset", "Horizon", "WarningType"]),
            },
        )
        save_phase_artifacts(
            "Phase 10 Actionable Research Plan",
            {
                "ranked_asset_horizon_plan": inputs["ranked_asset_horizon_plan"],
                "plan_card_table": inputs["ranked_asset_horizon_plan"],
                "paper_trade_plan_table": inputs["paper_trade_plan_table"],
                "watchlist_table": inputs["watchlist_table"],
                "risk_budget_table": pd.DataFrame(columns=["Asset", "Horizon"]),
                "warnings_table": pd.DataFrame(columns=["Asset", "Horizon", "WarningType"]),
            },
        )

        report = run_daily_research_control_center(use_artifact_store=True, assets=["Bitcoin"], horizons=[5], report_date="2026-06-25", capital_eligibility_mode="Balanced")

        assert not report.input_source_table.empty
        assert "LatestSavedArtifact" in set(report.input_source_table["Source"].astype(str))
        assert report.capital_eligibility_table.iloc[0]["Asset"] == "Bitcoin"

    _with_temp_store(run)


def test_autosaves_phase11_outputs_to_artifact_store():
    def run():
        report = run_daily_research_control_center(
            **_strong_inputs(),
            assets=["Bitcoin"],
            horizons=[5],
            report_date="2026-06-25",
            autosave=True,
        )

        assert report.saved_artifacts
        latest = load_latest_artifact("Phase 11 Daily Research Control Center", "daily_research_summary", required=True)
        assert not latest.empty

    _with_temp_store(run)


def test_daily_next_actions_are_non_empty_when_pending_or_blocked():
    report = run_daily_research_control_center(**_weak_inputs(), assets=["Gold"], horizons=[5], report_date="2026-06-25")

    assert not report.daily_next_actions_table.empty
    assert set(report.daily_next_actions_table["DailyAction"]).intersection({"WaitForOutcome", "KeepBlocked", "RecheckCalibrationLater"})


def test_no_forbidden_language_appears():
    report = run_daily_research_control_center(
        **_strong_inputs(),
        assets=["Bitcoin"],
        horizons=[5],
        report_date="2026-06-25",
        capital_eligibility_mode="Balanced",
    )
    text = "\n".join(
        table.astype(str).to_csv(index=False)
        for table in [
            report.daily_research_summary,
            report.capital_eligibility_table,
            report.structured_capital_plan_table,
            report.capital_blocker_table,
            report.evidence_health_table,
            report.daily_next_actions_table,
            report.warning_table,
        ]
    )

    for phrase in ["Buy", "Strong Buy", "Invest Now", "Production Ready", "Safe Profit", "Guaranteed"]:
        assert phrase not in text


def test_output_tables_have_expected_columns():
    report = run_daily_research_control_center(**_strong_inputs(), assets=["Bitcoin"], horizons=[5], report_date="2026-06-25")

    assert set(DAILY_RESEARCH_SUMMARY_COLUMNS).issubset(report.daily_research_summary.columns)
    assert set(CAPITAL_ELIGIBILITY_COLUMNS).issubset(report.capital_eligibility_table.columns)
    assert set(STRUCTURED_CAPITAL_PLAN_COLUMNS).issubset(report.structured_capital_plan_table.columns)
    assert set(DAILY_NEXT_ACTION_COLUMNS).issubset(report.daily_next_actions_table.columns)
    assert set(WARNING_COLUMNS).issubset(report.warning_table.columns)


if __name__ == "__main__":
    test_pending_matured_and_overdue_outcomes_are_detected()
    test_wait_for_outcome_review_date_uses_earliest_pending_target_date()
    test_real_capital_false_when_evidence_gates_fail_but_paper_plan_visible()
    test_real_capital_can_become_true_only_when_strict_synthetic_gates_pass()
    test_structured_capital_plan_only_for_eligible_candidates()
    test_all_assets_and_horizons_supported()
    test_missing_artifacts_do_not_crash_and_show_diagnostics()
    test_loads_latest_artifacts_from_artifact_store()
    test_autosaves_phase11_outputs_to_artifact_store()
    test_daily_next_actions_are_non_empty_when_pending_or_blocked()
    test_no_forbidden_language_appears()
    test_output_tables_have_expected_columns()
    print("Phase 11 daily research control center tests passed.")
