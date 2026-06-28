from pathlib import Path
import re
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import src.artifact_store as store
from src.artifact_store import load_latest_artifact
from src.unified_risk_command_center import (
    ASSET_HORIZON_SCORECARD_COLUMNS,
    INPUT_SOURCE_COLUMNS,
    NEXT_ACTION_COLUMNS,
    PAPER_TRACKING_COLUMNS,
    QUALITY_GATE_COLUMNS,
    REJECTED_CANDIDATE_COLUMNS,
    RISK_REGISTER_COLUMNS,
    UNIFIED_RISK_COMMAND_CENTER_PHASE_NAME,
    UNIFIED_SUMMARY_COLUMNS,
    run_unified_risk_command_center,
)


FORBIDDEN_LANGUAGE = re.compile(
    r"\b(Buy|Strong Buy|Invest Now|Guaranteed Profit|Safe Profit|Production Ready Trading)\b",
    flags=re.IGNORECASE,
)


def _phase19_edge():
    return pd.DataFrame(
        [
            {"Asset": "Gold", "Horizon": 5, "BestPolicy": "TrendMomentumPolicy", "BestPolicyReturnPct": 8.0, "BestBaseline": "HoldOnly", "BestBaselineReturnPct": 3.0, "PolicyVsBestBaselinePct": 5.0, "BeatsBestBaseline": True, "MaxDrawdownPct": -10.0, "TradeCount": 12, "EdgeVerdict": "PolicyEdge", "MainReason": "Synthetic supported row."},
            {"Asset": "Silver", "Horizon": 5, "BestPolicy": "InverseMomentumPolicy", "BestPolicyReturnPct": -3.0, "BestBaseline": "NoExposure", "BestBaselineReturnPct": 0.0, "PolicyVsBestBaselinePct": -3.0, "BeatsBestBaseline": False, "MaxDrawdownPct": -30.0, "TradeCount": 4, "EdgeVerdict": "BenchmarkDominated", "MainReason": "Synthetic rejected row."},
        ]
    )


def _phase19_dominance():
    return pd.DataFrame(
        [{"Asset": "Silver", "Horizon": 5, "PolicyName": "InverseMomentumPolicy", "DominatingBaseline": "NoExposure", "BaselineReturnPct": 0.0, "PolicyReturnPct": -3.0, "GapPct": 3.0}]
    )


def _phase19_overfit():
    return pd.DataFrame(
        [
            {"Asset": "Gold", "Horizon": 5, "PolicyName": "TrendMomentumPolicy", "OverfitRisk": "Low"},
            {"Asset": "Silver", "Horizon": 5, "PolicyName": "InverseMomentumPolicy", "OverfitRisk": "High"},
        ]
    )


def _phase19_cost():
    return pd.DataFrame(
        [
            {"Asset": "Gold", "Horizon": 5, "PolicyName": "TrendMomentumPolicy", "CostBps": 10, "CostFragile": False},
            {"Asset": "Silver", "Horizon": 5, "PolicyName": "InverseMomentumPolicy", "CostBps": 10, "CostFragile": True},
        ]
    )


def _phase19_drawdown():
    return pd.DataFrame(
        [
            {"Asset": "Gold", "Horizon": 5, "PolicyName": "TrendMomentumPolicy", "MaxDrawdownPct": -10.0},
            {"Asset": "Silver", "Horizon": 5, "PolicyName": "InverseMomentumPolicy", "MaxDrawdownPct": -30.0},
        ]
    )


def _phase20_performance():
    return pd.DataFrame(
        [
            {"Asset": "Gold", "Horizon": 5, "ModelName": "Ridge", "TotalReturnPct": 10.0, "MaxDrawdownPct": -12.0, "MaturedTrades": 15, "PendingTrades": 1},
            {"Asset": "Silver", "Horizon": 5, "ModelName": "Ridge", "TotalReturnPct": -5.0, "MaxDrawdownPct": -35.0, "MaturedTrades": 5, "PendingTrades": 1},
        ]
    )


def _phase20_baseline():
    return pd.DataFrame(
        [
            {"Asset": "Gold", "Horizon": 5, "ModelName": "Ridge", "MLReturnPct": 10.0, "BestBaselineName": "HoldOnly", "BestBaselineReturnPct": 4.0, "BeatsBestBaseline": True, "GapPct": 6.0, "DominanceVerdict": "ResearchOnly"},
            {"Asset": "Silver", "Horizon": 5, "ModelName": "Ridge", "MLReturnPct": -5.0, "BestBaselineName": "NoExposure", "BestBaselineReturnPct": 0.0, "BeatsBestBaseline": False, "GapPct": -5.0, "DominanceVerdict": "BenchmarkDominated"},
        ]
    )


def _phase20_strength():
    return pd.DataFrame(
        [
            {"Asset": "Gold", "Horizon": 5, "ModelName": "Ridge", "StrengthClassification": "WeakCandidate", "LeakagePassed": True, "CostFragile": False},
            {"Asset": "Silver", "Horizon": 5, "ModelName": "Ridge", "StrengthClassification": "BenchmarkDominated", "LeakagePassed": True, "CostFragile": True},
        ]
    )


def _phase20_leakage():
    return pd.DataFrame(
        [
            {"Asset": "Gold", "Horizon": 5, "WindowId": "G1", "LeakagePassed": True},
            {"Asset": "Silver", "Horizon": 5, "WindowId": "S1", "LeakagePassed": True},
        ]
    )


def _run_basic(**kwargs):
    params = {
        "use_artifact_store": False,
        "phase19_asset_horizon_edge": _phase19_edge(),
        "phase19_dominance_failures": _phase19_dominance(),
        "phase19_overfit_audit": _phase19_overfit(),
        "phase19_cost_sensitivity": _phase19_cost(),
        "phase19_drawdown": _phase19_drawdown(),
        "phase20_performance": _phase20_performance(),
        "phase20_baseline_comparison": _phase20_baseline(),
        "phase20_strength": _phase20_strength(),
        "phase20_leakage_audit": _phase20_leakage(),
    }
    params.update(kwargs)
    return run_unified_risk_command_center(**params)


def _all_output_text(report):
    tables = [
        report.unified_summary_table,
        report.asset_horizon_scorecard,
        report.risk_register,
        report.paper_tracking_candidates,
        report.rejected_candidates,
        report.quality_gates,
        report.next_actions,
        report.input_sources,
    ]
    return "\n".join(table.astype(str).to_csv(index=False) for table in tables)


def test_phase21_module_runs_and_required_tables_exist():
    report = _run_basic()
    expected = {
        "unified_summary_table": UNIFIED_SUMMARY_COLUMNS,
        "asset_horizon_scorecard": ASSET_HORIZON_SCORECARD_COLUMNS,
        "risk_register": RISK_REGISTER_COLUMNS,
        "paper_tracking_candidates": PAPER_TRACKING_COLUMNS,
        "rejected_candidates": REJECTED_CANDIDATE_COLUMNS,
        "quality_gates": QUALITY_GATE_COLUMNS,
        "next_actions": NEXT_ACTION_COLUMNS,
        "input_sources": INPUT_SOURCE_COLUMNS,
    }
    for name, columns in expected.items():
        assert set(columns).issubset(getattr(report, name).columns), name


def test_phase21_scorecard_combines_phase19_and_phase20_evidence():
    report = _run_basic()
    scorecard = report.asset_horizon_scorecard.set_index(["Asset", "Horizon"])

    assert ("Gold", 5) in scorecard.index
    assert ("Silver", 5) in scorecard.index
    assert scorecard.loc[("Gold", 5), "TrueMLReturnPct"] == 10.0
    assert scorecard.loc[("Gold", 5), "PolicyLabReturnPct"] == 8.0
    assert scorecard["EvidenceScore"].between(0, 100).all()
    assert scorecard["RiskScore"].between(0, 100).all()


def test_phase21_rejections_and_weak_results_remain_visible():
    report = _run_basic()

    assert not report.rejected_candidates.empty
    assert "Silver" in set(report.rejected_candidates["Asset"])
    gates = report.quality_gates.set_index("GateName")
    assert bool(gates.loc["WeakResultsVisible", "Passed"])
    assert bool(gates.loc["RejectionsVisible", "Passed"])


def test_phase21_conservative_paper_candidate_never_allows_real_capital():
    report = _run_basic()

    assert "Gold" in set(report.paper_tracking_candidates["Asset"])
    assert not report.paper_tracking_candidates["RealCapitalAllowed"].astype(bool).any()
    assert set(report.unified_summary_table["RealCapitalStatus"]) == {"Blocked"}


def test_phase21_missing_artifacts_return_warnings_without_crash():
    report = run_unified_risk_command_center(use_artifact_store=False)

    assert not report.unified_summary_table.empty
    assert not report.risk_register.empty
    assert not report.rejected_candidates.empty
    assert (~report.input_sources["Found"].astype(bool)).all()
    assert report.input_sources["Warning"].astype(str).str.len().gt(0).all()


def test_phase21_risk_register_includes_current_material_risks():
    report = _run_basic()
    risks = set(report.risk_register["RiskName"])

    assert "BenchmarkDominance" in risks
    assert "CostFragility" in risks
    assert "OverfitRisk" in risks
    assert "RealCapitalBlocked" in risks


def test_phase21_quality_gates_include_all_required_checks():
    report = _run_basic()
    required = {
        "Phase20Available",
        "LeakageAuditPassed",
        "BaselineComparisonAvailable",
        "WeakResultsVisible",
        "RejectionsVisible",
        "RealCapitalBlocked",
        "NoForbiddenClaims",
        "MissingArtifactsHandled",
        "ScorecardGenerated",
        "RiskRegisterGenerated",
    }

    assert required.issubset(set(report.quality_gates["GateName"]))


def test_phase21_autosaves_required_artifact_names():
    old_root = store.ARTIFACT_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        store.ARTIFACT_ROOT = Path(tmp) / "artifacts"
        try:
            report = _run_basic(autosave=True)
            expected = {
                "phase21_unified_summary",
                "phase21_asset_horizon_scorecard",
                "phase21_risk_register",
                "phase21_paper_tracking_candidates",
                "phase21_rejected_candidates",
                "phase21_quality_gates",
                "phase21_next_actions",
                "phase21_input_sources",
            }
            assert expected.issubset(report.saved_artifacts["Artifacts"])
            latest = load_latest_artifact(UNIFIED_RISK_COMMAND_CENTER_PHASE_NAME, "phase21_unified_summary", required=True)
            assert latest.iloc[0]["RealCapitalStatus"] == "Blocked"
        finally:
            store.ARTIFACT_ROOT = old_root


def test_phase21_outputs_have_no_forbidden_trading_claims():
    report = _run_basic()

    assert FORBIDDEN_LANGUAGE.search(_all_output_text(report)) is None
