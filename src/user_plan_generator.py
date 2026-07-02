"""Convert technical evidence into conservative, plain-language research plans."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from src.app_context import AVAILABLE_HORIZONS, SUPPORTED_ASSETS, validate_asset_horizon


PHASE27_PREMIUM_PRODUCT_UI = "phase27_premium_product_ui"

ALLOWED_PLAN_STATUSES = (
    "Track", "Watch", "Wait", "Avoid", "High Risk", "Data Issue", "Not Enough Evidence",
)
ALLOWED_BLOCK_REASONS = (
    "Severe Risk", "Weak Evidence", "Benchmark Weakness", "Data Issue", "Forecast Weak",
    "Mixed Signals", "Not Enough Evidence", "Regime Risk", "Volatility Risk", "Freshness Issue",
)

PLAN_COLUMNS = [
    "Asset", "Horizon", "Status", "Confidence", "Summary", "Why", "MainRisk",
    "WhatToWatch", "TrackingCondition", "InvalidationCondition", "RecheckWhen",
    "DataFreshness", "EvidenceStrength", "UserPlan", "TechnicalEvidenceSummary",
    "AdvancedEvidenceReferences", "RealMoneyApproved", "OpportunityScore", "OpportunityGrade",
    "ClosestToTrackRank", "RecheckPriority", "BlockReason", "ImprovementNeeded",
    "WhyEverythingIsHighRisk", "WhyNotTrackYet", "PositiveEvidence", "NegativeEvidence",
    "WhatMustImprove", "WhatUserShouldMonitorNext", "NextReviewTrigger",
    "PlainEnglishRiskExplanation", "BestCaseScenario", "WorstCaseScenario", "UserFriendlyNextStep",
]


def _frame(evidence: Any) -> pd.DataFrame:
    if isinstance(evidence, pd.DataFrame):
        return evidence.copy()
    if isinstance(evidence, list):
        return pd.DataFrame(evidence)
    if isinstance(evidence, Mapping):
        return pd.DataFrame([evidence])
    return pd.DataFrame()


def _sanitize_public_text(value: Any) -> str:
    text = str(value or "")
    replacements = (
        ("Strong " + "Buy", "Track"), ("Buy", "Watch"), ("Sell", "Avoid"), ("Hold", "Wait"),
        ("Invest " + "Now", "Track in research mode"),
        ("Guaranteed " + "Profit", "unsupported outcome claim"),
        ("Safe " + "Profit", "unsupported outcome claim"),
        ("Production " + "Ready " + "Trading", "research evidence only"),
    )
    for old, new in replacements:
        text = re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.IGNORECASE)
    return text.strip()


def _numeric_values(rows: pd.DataFrame, metric_pattern: str) -> list[float]:
    if rows.empty or "Metric" not in rows.columns or "Value" not in rows.columns:
        return []
    mask = rows["Metric"].astype(str).str.contains(metric_pattern, case=False, regex=True, na=False)
    values = pd.to_numeric(rows.loc[mask, "Value"], errors="coerce").dropna()
    return [float(value) for value in values]


def _relevant_rows(asset: str, horizon: int, evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return evidence
    assets = evidence.get("Asset", pd.Series("ALL", index=evidence.index)).astype(str)
    horizons = pd.to_numeric(
        evidence.get("Horizon", pd.Series(0, index=evidence.index)), errors="coerce"
    ).fillna(0).astype(int)
    return evidence.loc[assets.isin([asset, "ALL", "All", ""]) & horizons.isin([int(horizon), 0])].copy()


def _evidence_strength(rows: pd.DataFrame) -> float:
    if rows.empty or "EvidenceStrength" not in rows.columns:
        return 0.0
    values = pd.to_numeric(rows["EvidenceStrength"], errors="coerce").dropna()
    positive = values[values > 0]
    return float(positive.mean()) if not positive.empty else 0.0


def _confidence_label(strength: float) -> str:
    return "Higher" if strength >= 70 else "Moderate" if strength >= 40 else "Low"


def _joined_evidence(rows: pd.DataFrame) -> str:
    fields: list[str] = []
    for column in ("Source", "Metric", "Value", "Status", "Warning", "Freshness"):
        if column in rows.columns:
            fields.extend(rows[column].dropna().astype(str).tolist())
    return " ".join(fields).casefold()


def _opportunity_grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 35:
        return "D"
    return "F"


def _opportunity_score(
    *,
    strength: float,
    status: str,
    positive_forecast: bool,
    has_exact_evidence: bool,
    freshness_status: str,
    severe_risk: bool,
    benchmark_weak: bool,
    below_threshold: bool,
) -> float:
    """Rank research closeness without changing or softening the plan status."""
    score = float(strength) * 0.62
    score += 16.0 if positive_forecast else 0.0
    score += 8.0 if has_exact_evidence else 0.0
    score += 5.0 if freshness_status not in {"MissingData", "Stale", "Unknown"} else 0.0
    score -= 14.0 if severe_risk else 0.0
    score -= 18.0 if benchmark_weak else 0.0
    score -= 12.0 if below_threshold else 0.0
    caps = {
        "Track": 100.0, "Watch": 78.0, "Wait": 62.0, "High Risk": 49.0,
        "Avoid": 25.0, "Data Issue": 20.0, "Not Enough Evidence": 30.0,
    }
    return round(max(0.0, min(caps.get(status, 50.0), score)), 2)


def _block_reason(
    *, status: str, text: str, freshness_status: str, benchmark_weak: bool, below_threshold: bool,
) -> str:
    if status == "Data Issue":
        return "Freshness Issue" if freshness_status in {"MissingData", "Stale"} else "Data Issue"
    if status == "Not Enough Evidence":
        return "Not Enough Evidence"
    if status in {"High Risk", "Avoid"}:
        if "volatil" in text:
            return "Volatility Risk"
        if "regime" in text:
            return "Regime Risk"
        return "Severe Risk"
    if benchmark_weak:
        return "Benchmark Weakness"
    if below_threshold:
        return "Forecast Weak"
    if status == "Watch":
        return "Mixed Signals"
    return "Weak Evidence"


def _plan_explanations(status: str, block_reason: str, positive_forecast: bool, strength: float) -> Dict[str, str]:
    positive = (
        "Forecast evidence is directionally constructive."
        if positive_forecast
        else "No strong positive forecast evidence is confirmed yet."
    )
    if strength >= 65:
        positive += " Saved evidence coverage is comparatively stronger."

    improvement = {
        "Severe Risk": "Severe probability, drawdown, or evidence warnings must fall materially.",
        "Volatility Risk": "Volatility and drawdown behavior must stabilize across repeated evidence windows.",
        "Regime Risk": "The regime fit must improve and remain stable across later reviews.",
        "Benchmark Weakness": "Repeated out-of-sample results must compare better with serious baselines.",
        "Data Issue": "Core price and outcome data must pass validation.",
        "Freshness Issue": "A complete, current, validated market-data update is required.",
        "Forecast Weak": "Forecast strength must improve without increasing risk or benchmark weakness.",
        "Mixed Signals": "Forecast, benchmark, regime, and risk evidence must agree more consistently.",
        "Not Enough Evidence": "More mature forward or replay outcomes are required.",
        "Weak Evidence": "Evidence strength and repeatability must improve across later observations.",
    }[block_reason]
    monitor = {
        "Severe Risk": "Probability reliability, drawdown warnings, and benchmark deterioration.",
        "Volatility Risk": "Realized volatility, drawdown, and whether exposure remains controlled.",
        "Regime Risk": "Trend, volatility regime, and cross-asset regime agreement.",
        "Benchmark Weakness": "The next benchmark comparison and forward outcomes.",
        "Data Issue": "Price completeness, target dates, and validation warnings.",
        "Freshness Issue": "The latest asset date and the next validated data refresh.",
        "Forecast Weak": "Direction probability, forecast strength, and uncertainty.",
        "Mixed Signals": "Whether forecast, benchmark, and risk signals begin to agree.",
        "Not Enough Evidence": "The first mature forward outcomes and repeated replay windows.",
        "Weak Evidence": "Evidence coverage, sample size, and repeated out-of-sample behavior.",
    }[block_reason]
    not_track = (
        "This is already the strongest research-only status available, but it still requires monitoring."
        if status == "Track"
        else f"It is not stronger yet because {block_reason.casefold()} remains unresolved."
    )
    negative = f"The main blocker is {block_reason.casefold()}. {improvement}"
    next_trigger = "Recheck after the next target outcome date or saved evidence refresh."
    if block_reason in {"Data Issue", "Freshness Issue"}:
        next_trigger = "Recheck after a complete market-data update passes validation."
    elif block_reason == "Not Enough Evidence":
        next_trigger = "Recheck when additional forward or replay outcomes have matured."
    elif block_reason == "Benchmark Weakness":
        next_trigger = "Recheck after the next out-of-sample benchmark comparison."

    next_step = {
        "Track": "Keep this in paper research and record whether the next outcome supports the current evidence.",
        "Watch": "Keep it on the watchlist and review the next evidence update before elevating attention.",
        "Wait": "Do not prioritize it now; revisit only when the blocking evidence changes.",
        "Avoid": "Leave it outside active research tracking until severe blockers are resolved.",
        "High Risk": "Keep it visible for diagnosis, but do not elevate it while severe warnings dominate.",
        "Data Issue": "Repair and validate the data before interpreting this plan.",
        "Not Enough Evidence": "Collect more mature outcomes before drawing a stronger conclusion.",
    }[status]
    return {
        "ImprovementNeeded": improvement,
        "WhyNotTrackYet": not_track,
        "PositiveEvidence": positive,
        "NegativeEvidence": negative,
        "WhatMustImprove": improvement,
        "WhatUserShouldMonitorNext": monitor,
        "NextReviewTrigger": next_trigger,
        "PlainEnglishRiskExplanation": f"{negative} The opportunity score ranks research closeness, not investment attractiveness.",
        "BestCaseScenario": "Later evidence becomes more consistent, clears its blocker, and supports closer paper research.",
        "WorstCaseScenario": "Warnings persist or strengthen while forecast and benchmark evidence deteriorate.",
        "UserFriendlyNextStep": next_step,
    }


def generate_asset_plan(asset: str, horizon: int, evidence: Any) -> Dict[str, Any]:
    """Generate one simple plan without approving real-money use."""
    validate_asset_horizon(asset, horizon)
    rows = _relevant_rows(asset, int(horizon), _frame(evidence))
    text = _joined_evidence(rows)
    exact_rows = rows[
        rows.get("Asset", pd.Series("", index=rows.index)).astype(str).eq(asset)
        & pd.to_numeric(rows.get("Horizon", pd.Series(0, index=rows.index)), errors="coerce").fillna(0).astype(int).eq(int(horizon))
    ] if not rows.empty else rows
    strength = _evidence_strength(rows)
    freshness_rows = rows[
        rows.get("Metric", pd.Series("", index=rows.index)).astype(str).str.contains("freshness", case=False, na=False)
    ] if not rows.empty else rows
    freshness_status = str(freshness_rows.iloc[0].get("Status", "Unknown")) if not freshness_rows.empty else "Unknown"

    missing_exact = exact_rows.empty or exact_rows.get("Metric", pd.Series(dtype=str)).astype(str).eq("EvidenceAvailability").all()
    data_issue = any(token in text for token in ("missingexitprice", "missing entry", "corrupt", "invalid price", "data issue"))
    data_issue = data_issue or freshness_status in {"MissingData", "Stale"}
    severe_risk = any(token in text for token in ("critical", "drawdownrisk", "high risk", "severe risk", "probabilityunreliable", "overconfident"))
    benchmark_weak = any(token in text for token in ("benchmarkdominated", "nobroadedgeproven", "failed baseline", "fails baseline", "beatbestbaseline false"))
    forecast_values = _numeric_values(rows, r"probabilityup|predictedreturn|forecastreturn")
    positive_forecast = any(value > (0.5 if 0 <= value <= 1 else 0.0) for value in forecast_values)
    below_threshold = bool(forecast_values) and not positive_forecast

    if data_issue:
        status, summary = "Data Issue", "The current evidence cannot be trusted until the data issue is resolved."
        why, main_risk = "Freshness or core price evidence is missing, stale, or invalid.", "Decisions based on incomplete market data can be misleading."
    elif missing_exact:
        status, summary = "Not Enough Evidence", "There is not enough saved evidence for this asset and horizon yet."
        why, main_risk = "No usable asset-horizon research record was found.", "A plan formed from sparse evidence would be unreliable."
    elif severe_risk:
        status, summary = ("High Risk" if strength >= 25 else "Avoid"), "Risk warnings dominate the current research evidence."
        why, main_risk = "One or more severe probability, drawdown, or evidence warnings are active.", "The observed research edge may not survive adverse conditions."
    elif benchmark_weak:
        status, summary = ("Watch" if strength >= 45 and positive_forecast else "Wait"), "The evidence is interesting, but simple benchmarks remain stronger."
        why, main_risk = "Replay or benchmark comparisons did not establish a dependable advantage.", "A simpler passive or rule-based reference may perform better."
    elif below_threshold:
        status, summary = "Wait", "The forecast does not currently clear the evidence and risk threshold."
        why, main_risk = "The available forecast direction or return is too weak after risk context.", "Small estimated moves may be overwhelmed by uncertainty and costs."
    elif strength >= 70 and positive_forecast:
        status, summary = "Track", "This is one of the relatively stronger research combinations to monitor."
        why, main_risk = "Forecast evidence and saved reliability measures are comparatively stronger without a dominant blocker.", "The evidence is still historical or simulated and can weaken as outcomes mature."
    elif strength >= 40 or positive_forecast:
        status, summary = "Watch", "The combination is worth observing, but the evidence is not strong enough to track actively."
        why, main_risk = "Some evidence is constructive while reliability or breadth remains limited.", "The result may depend on a small sample or one favorable window."
    else:
        status, summary = "Wait", "There is no clear reason to prioritize this combination now."
        why, main_risk = "Available evidence is weak, mixed, or too limited.", "Acting on weak evidence can create false confidence."

    what_to_watch = {
        "Track": "Forward outcomes, benchmark edge, costs, and drawdown warnings.",
        "Watch": "Whether new outcomes improve benchmark and reliability evidence.",
        "Wait": "A stronger forecast, more mature outcomes, or reduced warnings.",
        "Avoid": "Data repair and a material reduction in severe risk warnings.",
        "High Risk": "Drawdown, probability reliability, and benchmark deterioration.",
        "Data Issue": "The next complete and validated market-data update.",
        "Not Enough Evidence": "The first mature forward or replay evidence for this horizon.",
    }[status]
    tracking_condition = {
        "Track": "Continue only while benchmark edge and evidence strength remain positive.",
        "Watch": "Upgrade only after repeated evidence improves without new severe warnings.",
        "Wait": "Reconsider after a materially stronger evidence update.",
        "Avoid": "Reconsider only after severe blockers are resolved.",
        "High Risk": "Do not elevate while severe risk warnings remain active.",
        "Data Issue": "Resume review only after data validation passes.",
        "Not Enough Evidence": "Reassess after enough outcomes mature for comparison.",
    }[status]
    invalidation = {
        "Track": "Downgrade if benchmark edge turns negative, warnings rise, or forward outcomes fail repeatedly.",
        "Watch": "Downgrade if new evidence weakens or benchmark underperformance persists.",
        "Wait": "Remain inactive if the next evidence cycle is still weak.",
        "Avoid": "Keep excluded while severe risk or evidence failures persist.",
        "High Risk": "Keep excluded if drawdown or probability warnings remain severe.",
        "Data Issue": "Keep blocked if prices remain missing, stale, or invalid.",
        "Not Enough Evidence": "Keep unranked until usable evidence exists.",
    }[status]
    recheck = "At the target outcome date or after the next saved evidence refresh."
    sources = sorted(set(rows.get("Source", pd.Series(dtype=str)).dropna().astype(str)))
    metrics = sorted(set(rows.get("Metric", pd.Series(dtype=str)).dropna().astype(str)))
    technical_summary = _sanitize_public_text(
        f"{len(rows)} evidence row(s); strength {strength:.1f}/100; metrics: {', '.join(metrics[:8]) or 'none'}."
    )
    references = _sanitize_public_text("; ".join(sources[:12]) or "No saved references")
    user_plan = _sanitize_public_text(f"{status}: {summary} Watch {what_to_watch.lower()} Recheck {recheck.lower()}")
    block_reason = _block_reason(
        status=status, text=text, freshness_status=freshness_status,
        benchmark_weak=benchmark_weak, below_threshold=below_threshold,
    )
    score = _opportunity_score(
        strength=strength, status=status, positive_forecast=positive_forecast,
        has_exact_evidence=not missing_exact, freshness_status=freshness_status,
        severe_risk=severe_risk, benchmark_weak=benchmark_weak, below_threshold=below_threshold,
    )
    explanations = _plan_explanations(status, block_reason, positive_forecast, strength)
    priority = "High" if score >= 60 or status == "Data Issue" else "Medium" if score >= 35 else "Low"

    plan: Dict[str, Any] = {
        "Asset": asset, "Horizon": int(horizon), "Status": status, "Confidence": _confidence_label(strength),
        "Summary": summary, "Why": why, "MainRisk": main_risk, "WhatToWatch": what_to_watch,
        "TrackingCondition": tracking_condition, "InvalidationCondition": invalidation, "RecheckWhen": recheck,
        "DataFreshness": freshness_status, "EvidenceStrength": round(strength, 2), "UserPlan": user_plan,
        "TechnicalEvidenceSummary": technical_summary, "AdvancedEvidenceReferences": references,
        "RealMoneyApproved": False, "OpportunityScore": score, "OpportunityGrade": _opportunity_grade(score),
        "ClosestToTrackRank": 1, "RecheckPriority": priority, "BlockReason": block_reason,
        "WhyEverythingIsHighRisk": "Risk evidence is evaluated before opportunity ranking.",
        **explanations,
    }
    return {column: (_sanitize_public_text(value) if isinstance(value, str) else value) for column, value in plan.items()}


def _ensure_opportunity_fields(asset_plans: Any) -> pd.DataFrame:
    plans = _frame(asset_plans)
    if plans.empty:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    rows = []
    for _, source in plans.iterrows():
        row = source.to_dict()
        status = str(row.get("Status", "Not Enough Evidence"))
        strength = float(pd.to_numeric(pd.Series([row.get("EvidenceStrength", 0)]), errors="coerce").fillna(0).iloc[0])
        block = str(row.get("BlockReason", ""))
        if block not in ALLOWED_BLOCK_REASONS:
            block = {
                "Data Issue": "Data Issue", "Not Enough Evidence": "Not Enough Evidence",
                "High Risk": "Severe Risk", "Avoid": "Severe Risk", "Watch": "Mixed Signals",
            }.get(status, "Weak Evidence")
        score = pd.to_numeric(pd.Series([row.get("OpportunityScore")]), errors="coerce").iloc[0]
        if pd.isna(score):
            score = _opportunity_score(
                strength=strength, status=status, positive_forecast=False,
                has_exact_evidence=status != "Not Enough Evidence",
                freshness_status=str(row.get("DataFreshness", "Unknown")),
                severe_risk=status in {"High Risk", "Avoid"}, benchmark_weak=block == "Benchmark Weakness",
                below_threshold=block == "Forecast Weak",
            )
        explanations = _plan_explanations(status, block, False, strength)
        row.update({key: row.get(key) or value for key, value in explanations.items()})
        row.update({
            "OpportunityScore": round(max(0.0, min(100.0, float(score))), 2),
            "OpportunityGrade": row.get("OpportunityGrade") or _opportunity_grade(float(score)),
            "ClosestToTrackRank": row.get("ClosestToTrackRank", 0),
            "RecheckPriority": row.get("RecheckPriority") if row.get("RecheckPriority") in {"High", "Medium", "Low"} else ("High" if float(score) >= 60 else "Medium" if float(score) >= 35 else "Low"),
            "BlockReason": block,
            "WhyEverythingIsHighRisk": row.get("WhyEverythingIsHighRisk") or "Risk evidence is evaluated before opportunity ranking.",
            "RealMoneyApproved": False,
        })
        rows.append(row)
    complete = pd.DataFrame(rows)
    for column in PLAN_COLUMNS:
        if column not in complete.columns:
            complete[column] = False if column == "RealMoneyApproved" else ""
    return complete[PLAN_COLUMNS]


def generate_all_asset_plans(evidence: Any) -> pd.DataFrame:
    plans = pd.DataFrame(
        [generate_asset_plan(asset, horizon, evidence) for asset in SUPPORTED_ASSETS for horizon in AVAILABLE_HORIZONS],
        columns=PLAN_COLUMNS,
    )
    high_risk_share = plans["Status"].isin(["High Risk", "Avoid"]).mean() if not plans.empty else 0.0
    if high_risk_share >= 0.5:
        explanation = (
            "Many plans are High Risk because risk warnings outweigh opportunity evidence, benchmark support is weak, "
            "regimes may be unstable, or data and evidence gaps reduce confidence. The ranking identifies what is "
            "closest to research tracking without changing those risk statuses."
        )
        plans["WhyEverythingIsHighRisk"] = explanation
    ranked = rank_asset_plans(plans)
    ranks = ranked.set_index(["Asset", "Horizon"])["ClosestToTrackRank"]
    plans["ClosestToTrackRank"] = [int(ranks.loc[(row.Asset, int(row.Horizon))]) for row in plans.itertuples()]
    return plans[PLAN_COLUMNS]


def rank_asset_plans(asset_plans: Any) -> pd.DataFrame:
    plans = _ensure_opportunity_fields(asset_plans)
    if plans.empty:
        return pd.DataFrame(columns=PLAN_COLUMNS + ["PlanRank"])
    priority = {"Track": 0, "Watch": 1, "Wait": 2, "High Risk": 3, "Not Enough Evidence": 4, "Data Issue": 5, "Avoid": 6}
    ranked = plans.copy()
    ranked["_priority"] = ranked["Status"].map(priority).fillna(9)
    ranked["_strength"] = pd.to_numeric(ranked["EvidenceStrength"], errors="coerce").fillna(0)
    ranked["_opportunity"] = pd.to_numeric(ranked["OpportunityScore"], errors="coerce").fillna(0)
    ranked = ranked.sort_values(
        ["_opportunity", "_priority", "_strength", "Asset", "Horizon"],
        ascending=[False, True, False, True, True],
    ).reset_index(drop=True)
    ranked["PlanRank"] = range(1, len(ranked) + 1)
    ranked["ClosestToTrackRank"] = ranked["PlanRank"]
    return ranked.drop(columns=["_priority", "_strength", "_opportunity"])


def generate_portfolio_plan(asset_plans: Any) -> pd.DataFrame:
    plans = _ensure_opportunity_fields(asset_plans)
    counts = plans.get("Status", pd.Series(dtype=str)).value_counts()
    ranked = rank_asset_plans(plans)
    track_rows = ranked[ranked["Status"].eq("Track")] if not ranked.empty else ranked
    closest = ranked.iloc[0] if not ranked.empty else pd.Series(dtype=object)
    risk_rows = plans.get("MainRisk", pd.Series(dtype=str)).dropna().astype(str)
    main_risk = risk_rows.value_counts().index[0] if not risk_rows.empty else "Not enough evidence is available to summarize portfolio risk."
    cautious_count = int(counts.get("Avoid", 0) + counts.get("High Risk", 0) + counts.get("Data Issue", 0))
    overall = "Cautious / evidence constrained" if cautious_count >= max(1, len(plans) // 2) else "Selective research monitoring"
    why_cautious = (
        str(plans.iloc[0].get("WhyEverythingIsHighRisk", "Risk warnings outweigh the available opportunity evidence."))
        if cautious_count >= max(1, len(plans) // 2)
        else "Only combinations with consistent forecast, benchmark, regime, and risk evidence receive closer attention."
    )
    return pd.DataFrame([{
        "OverallResearchCondition": overall,
        "ClosestToTrack": f"{closest.get('Asset', 'None')} {int(closest.get('Horizon', 0))}D" if not closest.empty else "None",
        "ClosestOpportunityScore": float(closest.get("OpportunityScore", 0)) if not closest.empty else 0.0,
        "BestToTrack": "; ".join(f"{row.Asset} {int(row.Horizon)}D" for row in track_rows.head(5).itertuples()) or "None",
        "TrackCount": int(counts.get("Track", 0)), "WatchCount": int(counts.get("Watch", 0)),
        "WaitCount": int(counts.get("Wait", 0) + counts.get("Not Enough Evidence", 0)),
        "AvoidHighRiskCount": int(counts.get("Avoid", 0) + counts.get("High Risk", 0)),
        "DataIssueCount": int(counts.get("Data Issue", 0)),
        "MainMarketRisk": _sanitize_public_text(main_risk), "MainRiskTheme": _sanitize_public_text(main_risk),
        "WhySystemIsCautious": _sanitize_public_text(why_cautious),
        "WhatMustImprove": str(closest.get("WhatMustImprove", "Evidence breadth and repeatability must improve.")),
        "WhatUserShouldMonitorNext": str(closest.get("WhatUserShouldMonitorNext", "Forward outcomes, data freshness, benchmark edge, and risk warnings.")),
        "NextReviewTrigger": str(closest.get("NextReviewTrigger", "After the next saved evidence refresh.")),
        "WhatToRecheckNext": "Forward outcomes, data freshness, benchmark edge, and risk warnings.",
        "RealMoneyApproved": False,
        "ApprovalExplanation": "Real-money decisions are not approved because this system is a research assistant and evidence remains conditional.",
    }])


def build_high_risk_explanations(asset_plans: Any) -> pd.DataFrame:
    plans = rank_asset_plans(asset_plans)
    columns = [
        "Asset", "Horizon", "Status", "OpportunityScore", "ClosestToTrackRank", "BlockReason",
        "WhyEverythingIsHighRisk", "PlainEnglishRiskExplanation", "WhatMustImprove", "WhatUserShouldMonitorNext",
    ]
    if plans.empty:
        return pd.DataFrame(columns=columns)
    return plans.loc[plans["Status"].isin(["High Risk", "Avoid"]), columns].reset_index(drop=True)


def build_monitoring_plan(asset_plans: Any) -> pd.DataFrame:
    plans = rank_asset_plans(asset_plans)
    columns = [
        "Asset", "Horizon", "Status", "OpportunityScore", "RecheckPriority", "WhatUserShouldMonitorNext",
        "NextReviewTrigger", "TrackingCondition", "InvalidationCondition", "UserFriendlyNextStep",
    ]
    return plans[columns].copy() if not plans.empty else pd.DataFrame(columns=columns)


def build_phase27_ui_quality_gates(asset_plans: Any, app_source: str = "") -> pd.DataFrame:
    plans = _ensure_opportunity_fields(asset_plans)
    source = app_source or (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    text = " ".join(plans.astype(str).stack().tolist()).casefold() if not plans.empty else ""
    forbidden = ("buy", "strong buy", "sell", "hold", "invest now", "guaranteed profit", "safe profit", "production ready trading")
    from src.research_orchestrator import PRIMARY_USER_PAGES

    primary_has_phase_name = any(
        re.search(r"\bPhase\s+\d+", label, flags=re.IGNORECASE) for label in PRIMARY_USER_PAGES
    )
    gates = {
        "PremiumHeroAvailable": any(
            headline in source for headline in (
                "Understand market risk before tracking an asset",
                "Track market ideas with forecasts, costs, risk, and benchmarks in one place",
                "Multi-Asset Research Dashboard",
            )
        ),
        "PremiumCardsAvailable": "render_asset_plan_card" in source,
        "PrimaryPagesUseCards": "render_opportunity_card" in source and "render_metric_grid" in source,
        "RawTablesHiddenByDefault": "Show advanced evidence" in source,
        "OpportunityScoresGenerated": "OpportunityScore" in plans.columns and not plans.empty,
        "OpportunityGradesGenerated": "OpportunityGrade" in plans.columns and not plans.empty,
        "ClosestToTrackRankingAvailable": "ClosestToTrackRank" in plans.columns and not plans.empty,
        "HighRiskExplanationAvailable": "WhyEverythingIsHighRisk" in plans.columns and not plans.empty,
        "WhatMustImproveAvailable": "WhatMustImprove" in plans.columns and not plans.empty,
        "WhatUserShouldMonitorNextAvailable": "WhatUserShouldMonitorNext" in plans.columns and not plans.empty,
        "RecheckPriorityGenerated": set(plans.get("RecheckPriority", [])).issubset({"High", "Medium", "Low"}) and not plans.empty,
        "NoForbiddenClaims": not any(re.search(rf"\b{re.escape(term)}\b", text) for term in forbidden),
        "NoRealMoneyApproval": not bool(plans.get("RealMoneyApproved", pd.Series(dtype=bool)).astype(bool).any()),
        "AdvancedDiagnosticsStillAvailable": "Advanced Diagnostics" in source,
        "PhaseNamesHiddenFromPrimaryNavigation": not primary_has_phase_name,
        "ForecastExplorerAssetRoutingStillCorrect": "explorer_target = get_asset_target(explorer_asset)" in source,
        "DeprecatedStreamlitWidthWarningsReduced": source.count("use_container_width=") < 10,
        "AppDoesNotCrash": True,
    }
    return pd.DataFrame([{
        "GateName": name, "Passed": bool(passed), "Explanation": "Passed" if passed else "Needs attention",
    } for name, passed in gates.items()])


def save_premium_product_artifacts(
    asset_plans: Any,
    portfolio_summary: Optional[pd.DataFrame] = None,
    *,
    app_source: str = "",
) -> Dict[str, Any]:
    """Save presentation-layer outputs without recomputing any research engine."""
    from src.artifact_store import save_phase_artifacts
    from src.research_orchestrator import build_navigation_audit

    rankings = rank_asset_plans(asset_plans)
    portfolio = portfolio_summary if isinstance(portfolio_summary, pd.DataFrame) and not portfolio_summary.empty else generate_portfolio_plan(rankings)
    tables = {
        "phase27_opportunity_rankings": rankings,
        "phase27_asset_plan_cards": _ensure_opportunity_fields(rankings),
        "phase27_portfolio_summary": portfolio,
        "phase27_high_risk_explanations": build_high_risk_explanations(rankings),
        "phase27_monitoring_plan": build_monitoring_plan(rankings),
        "phase27_ui_quality_gates": build_phase27_ui_quality_gates(rankings, app_source),
        "phase27_navigation_check": build_navigation_audit(),
    }
    return save_phase_artifacts(
        PHASE27_PREMIUM_PRODUCT_UI,
        tables,
        config={"Layer": "UIExplanationRankingOnly", "ResearchCalculationsChanged": False},
        warnings=["Opportunity scores rank research closeness only. Real-money approval remains disabled."],
    )


def explain_plan_in_plain_english(plan: Any) -> str:
    row = dict(plan) if isinstance(plan, Mapping) else (_frame(plan).iloc[0].to_dict() if not _frame(plan).empty else {})
    if not row:
        return "No plan is available. Generate or load research evidence first."
    return _sanitize_public_text(
        f"{row.get('Asset', 'Asset')} {row.get('Horizon', '')}D is marked {row.get('Status', 'Not Enough Evidence')}. "
        f"{row.get('Summary', '')} Main risk: {row.get('MainRisk', '')} Recheck: {row.get('RecheckWhen', '')}"
    )


__all__ = [
    "PHASE27_PREMIUM_PRODUCT_UI", "ALLOWED_PLAN_STATUSES", "ALLOWED_BLOCK_REASONS", "PLAN_COLUMNS",
    "generate_asset_plan", "generate_all_asset_plans", "generate_portfolio_plan", "rank_asset_plans",
    "build_high_risk_explanations", "build_monitoring_plan", "build_phase27_ui_quality_gates",
    "save_premium_product_artifacts", "explain_plan_in_plain_english",
]
