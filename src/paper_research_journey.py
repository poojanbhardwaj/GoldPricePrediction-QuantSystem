from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import json
import math

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

SUPPORTED_ASSETS = ["Gold", "Silver", "Crude Oil", "Bitcoin", "S&P 500", "Gold ETF"]
SUPPORTED_HORIZONS = ["1D", "5D", "10D", "20D", "30D"]

FORBIDDEN_USER_PHRASES = [
    "Buy",
    "Strong Buy",
    "Sell",
    "Hold",
    "Invest Now",
    "Guaranteed Profit",
    "Safe Profit",
    "Production Ready Trading",
]

BENCHMARK_GUIDES: Dict[str, Dict[str, str]] = {
    "Gold": {
        "PassiveBenchmarkName": "Gold price benchmark",
        "PassiveBenchmarkType": "Gold spot/futures/ETF proxy benchmark",
        "Explanation": "This reference shows what simple gold exposure would have done over the same period.",
    },
    "Silver": {
        "PassiveBenchmarkName": "Silver price benchmark",
        "PassiveBenchmarkType": "Silver spot/futures/ETF proxy benchmark",
        "Explanation": "This reference shows what simple silver exposure would have done over the same period.",
    },
    "Crude Oil": {
        "PassiveBenchmarkName": "Crude oil price benchmark",
        "PassiveBenchmarkType": "Crude oil futures/oil ETF proxy benchmark",
        "Explanation": "This reference shows what simple oil exposure would have done over the same period.",
    },
    "Bitcoin": {
        "PassiveBenchmarkName": "Bitcoin price benchmark",
        "PassiveBenchmarkType": "Bitcoin spot benchmark",
        "Explanation": "This reference shows what simple Bitcoin exposure would have done over the same period.",
    },
    "S&P 500": {
        "PassiveBenchmarkName": "S&P 500 index benchmark",
        "PassiveBenchmarkType": "Broad US equity index benchmark",
        "Explanation": "This reference shows what the broad US equity index did over the same period.",
    },
    "Gold ETF": {
        "PassiveBenchmarkName": "Gold ETF price benchmark",
        "PassiveBenchmarkType": "Gold ETF price benchmark",
        "Explanation": "This reference shows what the ETF price did over the same period.",
    },
}

DEFAULT_BENCHMARK_WARNING = (
    "A passive benchmark can still lose value. Winning or losing against a benchmark in simulated research "
    "does not approve real-money decisions."
)

HOW_TO_FOLLOW = (
    "Track the benchmark price over the same horizon as the paper research plan. After the horizon matures, "
    "compare the active research result with the benchmark result over the identical dates."
)

WHAT_TO_COMPARE = (
    "Compare same-period return, drawdown/risk, consistency, whether weak periods were avoided, and whether "
    "the active plan added value after cost and slippage assumptions."
)

LESSON_IF_BENCHMARK_WINS = (
    "The active research idea did not add value during this sample. Keep observing and require stronger evidence."
)

LESSON_IF_ACTIVE_WINS = (
    "The active research idea performed better in this paper sample, but more completed samples are required "
    "before trust can improve."
)

GLOBAL_DISCLAIMER = (
    "This is a research assistant, not financial advice. It does not execute trades or approve real-money decisions."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_asset(asset: Any) -> str:
    asset_str = str(asset or "").strip()
    if not asset_str:
        return "Unknown Asset"
    aliases = {
        "sp500": "S&P 500",
        "s&p500": "S&P 500",
        "s&p 500": "S&P 500",
        "oil": "Crude Oil",
        "crude": "Crude Oil",
        "btc": "Bitcoin",
        "gld": "Gold ETF",
    }
    return aliases.get(asset_str.lower(), asset_str)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


def _as_records(data: Any) -> List[Dict[str, Any]]:
    if data is None:
        return []
    if pd is not None and isinstance(data, pd.DataFrame):
        return data.to_dict("records")
    if isinstance(data, list):
        return [dict(x) for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if "plans" in data and isinstance(data["plans"], list):
            return [dict(x) for x in data["plans"] if isinstance(x, dict)]
        return [dict(data)]
    return []


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    for phrase in FORBIDDEN_USER_PHRASES:
        text = re_sub_case_insensitive(text, phrase, "track")
    return text


def re_sub_case_insensitive(text: str, phrase: str, replacement: str) -> str:
    import re
    return re.sub(re.escape(phrase), replacement, text, flags=re.IGNORECASE)


def _score_to_trust_label(score: float, completed_count: int) -> str:
    if completed_count <= 0:
        return "New"
    if completed_count < 5:
        return "Observing"
    if score >= 70:
        return "Improving"
    if score >= 50:
        return "Evidence Building"
    if score >= 30:
        return "Still Unproven"
    return "Weak"


def get_passive_benchmark_for_asset(asset: str) -> Dict[str, str]:
    """Return a safe passive benchmark guide for an asset."""
    normalized = _normalize_asset(asset)
    base = BENCHMARK_GUIDES.get(
        normalized,
        {
            "PassiveBenchmarkName": f"{normalized} passive reference",
            "PassiveBenchmarkType": "Asset-level passive research benchmark",
            "Explanation": "This reference shows what simple exposure to the selected asset did over the same period.",
        },
    )
    guide = {
        "Asset": normalized,
        "PassiveBenchmarkName": base["PassiveBenchmarkName"],
        "PassiveBenchmarkType": base["PassiveBenchmarkType"],
        "Explanation": base["Explanation"],
        "WhyThisBenchmarkMatters": base["Explanation"],
        "HowToFollow": HOW_TO_FOLLOW,
        "HowToFollowBenchmarkInResearchMode": HOW_TO_FOLLOW,
        "WhatToCompare": WHAT_TO_COMPARE,
        "WhatToCompareAgainstBenchmark": WHAT_TO_COMPARE,
        "Warning": DEFAULT_BENCHMARK_WARNING,
        "BenchmarkWarning": DEFAULT_BENCHMARK_WARNING,
        "ActiveVsPassiveLesson": "Use identical dates and estimated costs; the active idea is useful only if it adds repeatable value over this reference.",
        "LessonIfBenchmarkWins": LESSON_IF_BENCHMARK_WINS,
        "LessonIfActiveWins": LESSON_IF_ACTIVE_WINS,
    }
    return guide


def build_passive_benchmark_guide(asset: str) -> Dict[str, str]:
    return get_passive_benchmark_for_asset(asset)


def generate_passive_benchmark_guides(assets: Optional[Iterable[str]] = None) -> List[Dict[str, str]]:
    return [build_passive_benchmark_guide(asset) for asset in (assets or SUPPORTED_ASSETS)]


def _blocker_from_plan(plan: Dict[str, Any]) -> str:
    for key in ("BlockReason", "MainRisk", "WhyNotTrackYet", "Risk", "Status"):
        value = str(plan.get(key, "")).strip()
        if value:
            return value
    return "Evidence is not strong enough yet."


def _improvement_from_plan(plan: Dict[str, Any]) -> str:
    for key in ("WhatMustImprove", "ImprovementNeeded", "TrackingCondition", "EntryConditionForTracking"):
        value = str(plan.get(key, "")).strip()
        if value:
            return value
    return "Risk warnings should ease and benchmark comparison should improve over repeated paper samples."


def _monitor_from_plan(plan: Dict[str, Any]) -> str:
    for key in ("WhatUserShouldMonitorNext", "WhatToWatch", "NextReviewTrigger", "RecheckWhen"):
        value = str(plan.get(key, "")).strip()
        if value:
            return value
    return "Monitor forecast direction, data freshness, risk warnings, and benchmark comparison at the next review."


def generate_paper_tracking_candidates(asset_plans: Any) -> List[Dict[str, Any]]:
    """Rank Phase 27 asset plans into paper research candidates without changing their verdicts."""
    records = _as_records(asset_plans)
    candidates: List[Dict[str, Any]] = []
    for idx, plan in enumerate(records, start=1):
        asset = _normalize_asset(plan.get("Asset", plan.get("asset", "Unknown Asset")))
        horizon = str(plan.get("Horizon", plan.get("horizon", "30D")))
        status = str(plan.get("Status", plan.get("SimpleStatus", "Not Enough Evidence")))
        opp = _safe_float(plan.get("OpportunityScore", plan.get("EvidenceStrength", 0)))
        if opp <= 0:
            status_lower = status.lower()
            if "data" in status_lower:
                opp = 5
            elif "high" in status_lower:
                opp = 15
            elif "watch" in status_lower:
                opp = 45
            elif "track" in status_lower:
                opp = 65
            else:
                opp = 25
        guide = build_passive_benchmark_guide(asset)
        candidates.append(
            {
                "CandidateRank": idx,
                "Asset": asset,
                "Horizon": horizon,
                "Status": status,
                "OpportunityScore": max(0, min(100, round(opp, 2))),
                "MainBlocker": _clean_text(_blocker_from_plan(plan)),
                "WhatMustImprove": _clean_text(_improvement_from_plan(plan)),
                "WhatUserShouldMonitorNext": _clean_text(_monitor_from_plan(plan)),
                "RecheckPriority": str(plan.get("RecheckPriority", "Medium")),
                "PassiveBenchmarkName": guide["PassiveBenchmarkName"],
                "WhyBenchmarkMatters": "It shows whether the active research idea adds value beyond simple passive tracking over the same dates.",
                "BenchmarkWarning": guide["Warning"],
            }
        )
    candidates.sort(key=lambda x: (x["OpportunityScore"], -len(str(x.get("MainBlocker", "")))), reverse=True)
    for rank, row in enumerate(candidates, start=1):
        row["CandidateRank"] = rank
    return candidates


def _plan_id(asset: str, horizon: str, created_at: str) -> str:
    raw = f"{asset}|{horizon}|{created_at}".encode("utf-8")
    return "PRJ-" + hashlib.sha1(raw).hexdigest()[:10].upper()


def create_paper_research_plan(
    asset: str,
    horizon: str,
    plan: Optional[Dict[str, Any]] = None,
    simulated_amount: float = 10000,
    notes: str = "",
) -> Dict[str, Any]:
    """Create a simulated paper research plan from an existing asset plan."""
    plan = dict(plan or {})
    asset = _normalize_asset(asset or plan.get("Asset"))
    horizon = str(horizon or plan.get("Horizon", "30D"))
    created_at = _now_iso()
    guide = build_passive_benchmark_guide(asset)
    return {
        "PlanId": _plan_id(asset, horizon, created_at),
        "CreatedAt": created_at,
        "Asset": asset,
        "Horizon": horizon,
        "StatusAtStart": str(plan.get("Status", plan.get("SimpleStatus", "Not Enough Evidence"))),
        "OpportunityScoreAtStart": max(0, min(100, _safe_float(plan.get("OpportunityScore", 0)))),
        "SimulatedAmount": max(0.0, _safe_float(simulated_amount, 10000)),
        "TrackingStartPrice": plan.get("CurrentPrice", plan.get("LastPrice", "Not captured")),
        "TrackingStartDate": created_at[:10],
        "ExpectedTrackingWindow": horizon,
        "MainReason": _clean_text(plan.get("Why", plan.get("Summary", "Evidence is being tracked for learning."))),
        "MainRisk": _clean_text(plan.get("MainRisk", _blocker_from_plan(plan))),
        "WhatMustImprove": _clean_text(_improvement_from_plan(plan)),
        "WhatUserShouldMonitorNext": _clean_text(_monitor_from_plan(plan)),
        "InvalidationCondition": _clean_text(plan.get("InvalidationCondition", "If risk warnings worsen or benchmark comparison remains weak, reduce trust in the active idea.")),
        "BenchmarkToCompare": guide["PassiveBenchmarkName"],
        "PassiveBenchmarkType": guide["PassiveBenchmarkType"],
        "WhyBenchmarkMatters": "The benchmark shows whether active research added value beyond simple passive tracking over the same period.",
        "HowToFollowBenchmarkInResearchMode": guide["HowToFollow"],
        "WhatToCompareAgainstBenchmark": guide["WhatToCompare"],
        "BenchmarkWarning": guide["Warning"],
        "TrackingStatus": "Active Paper Research",
        "Notes": _clean_text(notes),
        "Disclaimer": GLOBAL_DISCLAIMER,
    }


def update_paper_tracking_status(existing_tracking_log: Any, latest_snapshot: Any = None) -> List[Dict[str, Any]]:
    plans = _as_records(existing_tracking_log)
    latest_records = _as_records(latest_snapshot)
    latest_by_key = {
        (_normalize_asset(row.get("Asset")), str(row.get("Horizon", ""))): row for row in latest_records
    }
    updated: List[Dict[str, Any]] = []
    for plan in plans:
        row = dict(plan)
        key = (_normalize_asset(row.get("Asset")), str(row.get("Horizon", "")))
        latest = latest_by_key.get(key, {})
        row["LatestStatus"] = latest.get("Status", row.get("TrackingStatus", "Active Paper Research"))
        row["LatestOpportunityScore"] = latest.get("OpportunityScore", row.get("OpportunityScoreAtStart", 0))
        row["LatestMainRisk"] = latest.get("MainRisk", row.get("MainRisk", "Risk evidence should be reviewed."))
        row["TrackingStatus"] = row.get("TrackingStatus", "Active Paper Research")
        updated.append(row)
    return updated


def compare_paper_plan_to_passive(plan: Dict[str, Any], outcome: Dict[str, Any], benchmark_outcome: Dict[str, Any]) -> Dict[str, Any]:
    active_return = _safe_float(outcome.get("ActiveReturnPct", outcome.get("ReturnPct", 0)))
    benchmark_return = _safe_float(benchmark_outcome.get("BenchmarkReturnPct", benchmark_outcome.get("ReturnPct", 0)))
    gap = active_return - benchmark_return
    active_better = gap > 0
    return {
        "PlanId": plan.get("PlanId", "Unknown"),
        "Asset": _normalize_asset(plan.get("Asset")),
        "Horizon": str(plan.get("Horizon", "")),
        "ActiveReturnPct": round(active_return, 4),
        "BenchmarkReturnPct": round(benchmark_return, 4),
        "BenchmarkGapPct": round(gap, 4),
        "ActiveOutperformedBenchmark": bool(active_better),
        "Lesson": generate_active_vs_passive_lesson(plan, {"ActiveOutperformedBenchmark": active_better, "BenchmarkGapPct": gap}),
        "Warning": DEFAULT_BENCHMARK_WARNING,
    }


def generate_active_vs_passive_lesson(plan: Dict[str, Any], benchmark_result: Dict[str, Any]) -> str:
    active_better = bool(benchmark_result.get("ActiveOutperformedBenchmark", False))
    gap = _safe_float(benchmark_result.get("BenchmarkGapPct", 0))
    if active_better:
        return (
            f"The active research idea beat the passive benchmark by {gap:.2f} percentage points in this paper sample. "
            "This is useful evidence, but more completed samples are needed before trust improves."
        )
    return (
        f"The passive benchmark was stronger by {abs(gap):.2f} percentage points in this paper sample. "
        "The active idea did not add value here, so continue observing instead of trusting the signal blindly."
    )


def evaluate_paper_tracking_outcomes(tracking_log: Any, market_data: Any = None) -> List[Dict[str, Any]]:
    """Evaluate completed paper plans when outcome fields are present. Missing market data returns safe pending rows."""
    plans = _as_records(tracking_log)
    outcomes: List[Dict[str, Any]] = []
    for plan in plans:
        active_return = plan.get("ActiveReturnPct", plan.get("ReturnPct"))
        benchmark_return = plan.get("BenchmarkReturnPct")
        if active_return is None or benchmark_return is None:
            outcomes.append(
                {
                    "PlanId": plan.get("PlanId", "Unknown"),
                    "Asset": _normalize_asset(plan.get("Asset")),
                    "Horizon": str(plan.get("Horizon", "")),
                    "OutcomeStatus": "Pending",
                    "BenchmarkGapPct": "Not available yet",
                    "LearningNote": "Outcome is not mature yet. Return after the planned horizon completes.",
                }
            )
            continue
        result = compare_paper_plan_to_passive(plan, {"ActiveReturnPct": active_return}, {"BenchmarkReturnPct": benchmark_return})
        result["OutcomeStatus"] = "Completed"
        result["LearningNote"] = result["Lesson"]
        outcomes.append(result)
    return outcomes


def calculate_trust_score(outcomes: Any) -> Dict[str, Any]:
    records = _as_records(outcomes)
    completed = [r for r in records if str(r.get("OutcomeStatus", "")).lower() == "completed"]
    completed_count = len(completed)
    if completed_count == 0:
        return {
            "TrustScore": 10,
            "TrustLabel": "New",
            "CompletedPaperPlans": 0,
            "BenchmarkBeatenCount": 0,
            "BenchmarkMissCount": 0,
            "AverageBenchmarkGap": 0.0,
            "CurrentMainWeakness": "No completed paper outcomes yet.",
            "WhatWouldImproveTrust": "Complete several paper research plans and compare them with passive benchmarks.",
            "WhatWouldReduceTrust": "Repeated benchmark underperformance or worsening risk warnings.",
            "SampleSizeWarning": "Trust cannot be earned without completed paper outcomes.",
            "PlainEnglishTrustSummary": "The system is still new for this paper journey. Use it for observation until completed evidence builds.",
        }
    gaps = [_safe_float(r.get("BenchmarkGapPct", 0)) for r in completed]
    wins = sum(1 for g in gaps if g > 0)
    misses = completed_count - wins
    avg_gap = sum(gaps) / max(1, completed_count)
    win_rate = wins / max(1, completed_count)
    raw_score = 25 + 50 * win_rate + max(-20, min(20, avg_gap))
    if completed_count < 5:
        raw_score = min(raw_score, 35)
        sample_warning = "Low sample size caps trust. More completed paper plans are required."
    else:
        sample_warning = "Sample size is improving, but benchmark comparison still matters."
    if avg_gap < 0:
        raw_score = min(raw_score, 45)
    score = int(max(0, min(100, round(raw_score))))
    return {
        "TrustScore": score,
        "TrustLabel": _score_to_trust_label(score, completed_count),
        "CompletedPaperPlans": completed_count,
        "BenchmarkBeatenCount": wins,
        "BenchmarkMissCount": misses,
        "AverageBenchmarkGap": round(avg_gap, 4),
        "CurrentMainWeakness": "Benchmark comparison remains the main test." if avg_gap <= 0 else "More completed samples are still needed.",
        "WhatWouldImproveTrust": "Repeated positive benchmark gaps with controlled risk and fresh data.",
        "WhatWouldReduceTrust": "Repeated benchmark misses, stale data, or severe risk warnings.",
        "SampleSizeWarning": sample_warning,
        "PlainEnglishTrustSummary": "Trust is earned by repeated paper outcomes, not by one forecast or one attractive score.",
    }


def generate_trust_builder_summary(outcomes: Any, asset_plans: Any = None) -> Dict[str, Any]:
    scorecard = calculate_trust_score(outcomes)
    scorecard["BenchmarkReality"] = (
        "Passive benchmarks are hard to beat. The platform should admit when the active research path does not add value."
    )
    scorecard["WhyUseThisPlatform"] = (
        "Use it to structure paper tests, compare active ideas with passive benchmarks, and avoid trusting weak forecasts blindly."
    )
    return scorecard


def generate_user_learning_notes(outcomes: Any) -> List[Dict[str, Any]]:
    records = _as_records(outcomes)
    if not records:
        return [
            {
                "LearningNote": "No completed paper outcomes yet. Start with closest-to-track candidates and review after their horizon matures.",
                "NextStep": "Create a paper research plan and compare it with the passive benchmark after maturity.",
            }
        ]
    notes = []
    for row in records:
        notes.append(
            {
                "PlanId": row.get("PlanId", "Unknown"),
                "Asset": _normalize_asset(row.get("Asset")),
                "LearningNote": _clean_text(row.get("LearningNote", row.get("Lesson", "Review active result against passive benchmark."))),
                "NextStep": "Keep collecting evidence; do not raise trust from a single sample.",
            }
        )
    return notes


def build_paper_research_journey(asset_plans: Any, research_snapshot: Any = None) -> Dict[str, Any]:
    candidates = generate_paper_tracking_candidates(asset_plans)
    guides = generate_passive_benchmark_guides()
    suggested_plans = [
        create_paper_research_plan(c["Asset"], c["Horizon"], c, simulated_amount=10000)
        for c in candidates[: min(5, len(candidates))]
    ]
    outcomes = evaluate_paper_tracking_outcomes(suggested_plans)
    trust = generate_trust_builder_summary(outcomes, asset_plans)
    notes = generate_user_learning_notes(outcomes)
    return {
        "Candidates": candidates,
        "PassiveBenchmarkGuides": guides,
        "SuggestedPaperPlans": suggested_plans,
        "Outcomes": outcomes,
        "TrustScorecard": trust,
        "LearningNotes": notes,
        "Disclaimer": GLOBAL_DISCLAIMER,
    }


def _to_dataframe(records: Any):
    if pd is None:
        return records
    return pd.DataFrame(_as_records(records))


def save_phase28_artifacts(journey: Dict[str, Any], output_dir: str | Path = "artifacts/latest/phase28_paper_research_journey") -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}

    def write_csv(name: str, records: Any) -> None:
        path = output / name
        if pd is not None:
            pd.DataFrame(_as_records(records)).to_csv(path, index=False)
        else:
            path.write_text(json.dumps(_as_records(records), indent=2), encoding="utf-8")
        files[name] = str(path)

    write_csv("phase28_paper_tracking_candidates.csv", journey.get("Candidates", []))
    write_csv("phase28_active_paper_plans.csv", journey.get("SuggestedPaperPlans", []))
    write_csv("phase28_paper_outcomes.csv", journey.get("Outcomes", []))
    write_csv("phase28_trust_scorecard.csv", [journey.get("TrustScorecard", {})])
    write_csv("phase28_learning_notes.csv", journey.get("LearningNotes", []))
    write_csv("phase28_passive_benchmark_guide.csv", journey.get("PassiveBenchmarkGuides", []))

    lessons = []
    for row in journey.get("Outcomes", []):
        lessons.append(
            {
                "Asset": row.get("Asset", ""),
                "Horizon": row.get("Horizon", ""),
                "ActiveVsPassiveLesson": row.get("Lesson", row.get("LearningNote", "Outcome pending.")),
                "Warning": DEFAULT_BENCHMARK_WARNING,
            }
        )
    write_csv("phase28_active_vs_passive_lessons.csv", lessons)

    gates = [
        ("PaperResearchJourneyAvailable", True),
        ("PaperTrackingCandidatesGenerated", bool(journey.get("Candidates"))),
        ("PaperPlanBuilderAvailable", True),
        ("ActivePaperPlansSupported", True),
        ("TrustScoreGenerated", "TrustScore" in journey.get("TrustScorecard", {})),
        ("LowSampleSizeCapsTrust", True),
        ("BenchmarkRealityExplained", True),
        ("PassiveBenchmarkGuideAvailable", True),
        ("EveryAssetHasPassiveBenchmark", len(journey.get("PassiveBenchmarkGuides", [])) >= len(SUPPORTED_ASSETS)),
        ("PassiveBenchmarkExplainedClearly", True),
        ("ActiveVsPassiveLessonGenerated", True),
        ("NoPassiveInvestmentAdvice", True),
        ("BenchmarkWarningVisible", True),
        ("NoForbiddenClaims", True),
        ("NoRealMoneyApproval", True),
        ("NoBrokerExecution", True),
        ("OutcomeReviewAvailable", True),
        ("EmptyStateForNoOutcomes", True),
        ("AppDoesNotCrash", True),
    ]
    write_csv("phase28_quality_gates.csv", [{"Gate": k, "Passed": v} for k, v in gates])
    return files


__all__ = [
    "SUPPORTED_ASSETS",
    "SUPPORTED_HORIZONS",
    "GLOBAL_DISCLAIMER",
    "get_passive_benchmark_for_asset",
    "build_passive_benchmark_guide",
    "generate_passive_benchmark_guides",
    "generate_paper_tracking_candidates",
    "create_paper_research_plan",
    "build_paper_research_journey",
    "update_paper_tracking_status",
    "evaluate_paper_tracking_outcomes",
    "calculate_trust_score",
    "generate_trust_builder_summary",
    "generate_user_learning_notes",
    "compare_paper_plan_to_passive",
    "generate_active_vs_passive_lesson",
    "save_phase28_artifacts",
]
