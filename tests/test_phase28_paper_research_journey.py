from __future__ import annotations

import py_compile
from pathlib import Path

from src.paper_research_journey import (
    SUPPORTED_ASSETS,
    build_paper_research_journey,
    build_passive_benchmark_guide,
    calculate_trust_score,
    compare_paper_plan_to_passive,
    create_paper_research_plan,
    generate_active_vs_passive_lesson,
    generate_paper_tracking_candidates,
    save_phase28_artifacts,
)

FORBIDDEN = ["Strong Buy", "Invest Now", "Guaranteed Profit", "Safe Profit", "Production Ready Trading"]


def sample_plans():
    return [
        {
            "Asset": "Bitcoin",
            "Horizon": "1D",
            "Status": "High Risk",
            "OpportunityScore": 54,
            "MainRisk": "Volatility risk dominates the forecast.",
            "WhatMustImprove": "Risk warnings should ease and benchmark gap should improve.",
            "WhatUserShouldMonitorNext": "Monitor volatility, benchmark gap, and data freshness.",
            "RecheckPriority": "High",
        },
        {
            "Asset": "Gold",
            "Horizon": "30D",
            "Status": "Data Issue",
            "OpportunityScore": 8,
            "MainRisk": "Freshness issue.",
        },
    ]


def flatten_text(obj):
    if isinstance(obj, dict):
        return " ".join(flatten_text(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(flatten_text(v) for v in obj)
    return str(obj)


def test_every_supported_asset_has_passive_benchmark_guide():
    for asset in SUPPORTED_ASSETS:
        guide = build_passive_benchmark_guide(asset)
        assert guide["Asset"] == asset
        assert guide["PassiveBenchmarkName"]
        assert guide["HowToFollow"]
        assert guide["WhatToCompare"]
        assert guide["Warning"]


def test_unknown_asset_benchmark_does_not_crash():
    guide = build_passive_benchmark_guide("Custom Asset")
    assert guide["Asset"] == "Custom Asset"
    assert guide["PassiveBenchmarkName"]
    assert guide["Warning"]


def test_paper_tracking_candidates_generate_from_asset_plans():
    candidates = generate_paper_tracking_candidates(sample_plans())
    assert len(candidates) == 2
    assert candidates[0]["OpportunityScore"] >= candidates[1]["OpportunityScore"]
    assert candidates[0]["PassiveBenchmarkName"]
    assert candidates[0]["WhatMustImprove"]


def test_paper_research_plan_has_required_fields():
    candidate = generate_paper_tracking_candidates(sample_plans())[0]
    plan = create_paper_research_plan(candidate["Asset"], candidate["Horizon"], candidate, simulated_amount=5000)
    required = [
        "PlanId",
        "CreatedAt",
        "Asset",
        "Horizon",
        "StatusAtStart",
        "OpportunityScoreAtStart",
        "SimulatedAmount",
        "BenchmarkToCompare",
        "HowToFollowBenchmarkInResearchMode",
        "WhatToCompareAgainstBenchmark",
        "BenchmarkWarning",
        "TrackingStatus",
    ]
    for key in required:
        assert key in plan
    assert plan["SimulatedAmount"] == 5000


def test_trust_score_range_and_low_sample_cap():
    outcomes = [
        {"OutcomeStatus": "Completed", "BenchmarkGapPct": 8.0},
        {"OutcomeStatus": "Completed", "BenchmarkGapPct": 4.0},
    ]
    score = calculate_trust_score(outcomes)
    assert 0 <= score["TrustScore"] <= 100
    assert score["TrustScore"] <= 35
    assert score["SampleSizeWarning"]


def test_weak_benchmark_performance_caps_trust():
    outcomes = [
        {"OutcomeStatus": "Completed", "BenchmarkGapPct": -2.0},
        {"OutcomeStatus": "Completed", "BenchmarkGapPct": -5.0},
        {"OutcomeStatus": "Completed", "BenchmarkGapPct": -1.5},
        {"OutcomeStatus": "Completed", "BenchmarkGapPct": -3.0},
        {"OutcomeStatus": "Completed", "BenchmarkGapPct": -1.0},
    ]
    score = calculate_trust_score(outcomes)
    assert score["TrustScore"] <= 45
    assert score["BenchmarkMissCount"] == 5


def test_missing_outcomes_do_not_crash():
    score = calculate_trust_score([])
    assert score["TrustLabel"] in {"New", "Observing"}
    assert score["CompletedPaperPlans"] == 0


def test_active_vs_passive_lesson_exists():
    plan = create_paper_research_plan("Bitcoin", "1D", sample_plans()[0])
    result = compare_paper_plan_to_passive(plan, {"ActiveReturnPct": 1.0}, {"BenchmarkReturnPct": 2.5})
    assert result["ActiveOutperformedBenchmark"] is False
    assert result["Lesson"]
    assert "passive benchmark" in result["Lesson"].lower()
    lesson = generate_active_vs_passive_lesson(plan, {"ActiveOutperformedBenchmark": True, "BenchmarkGapPct": 2.0})
    assert lesson


def test_journey_builds_and_artifacts_save(tmp_path):
    journey = build_paper_research_journey(sample_plans())
    assert journey["Candidates"]
    assert journey["PassiveBenchmarkGuides"]
    assert journey["TrustScorecard"]["TrustScore"] >= 0
    files = save_phase28_artifacts(journey, tmp_path)
    assert "phase28_quality_gates.csv" in files
    assert Path(files["phase28_quality_gates.csv"]).exists()


def test_no_forbidden_phrases_in_generated_outputs():
    journey = build_paper_research_journey(sample_plans())
    text = flatten_text(journey)
    for phrase in FORBIDDEN:
        assert phrase.lower() not in text.lower()


def test_app_py_compiles():
    assert Path("app.py").exists()
    py_compile.compile("app.py", doraise=True)
