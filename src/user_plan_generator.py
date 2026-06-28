"""Convert technical evidence into conservative, plain-language research plans."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from src.app_context import AVAILABLE_HORIZONS, SUPPORTED_ASSETS, validate_asset_horizon


ALLOWED_PLAN_STATUSES = (
    "Track",
    "Watch",
    "Wait",
    "Avoid",
    "High Risk",
    "Data Issue",
    "Not Enough Evidence",
)

PLAN_COLUMNS = [
    "Asset", "Horizon", "Status", "Confidence", "Summary", "Why", "MainRisk",
    "WhatToWatch", "TrackingCondition", "InvalidationCondition", "RecheckWhen",
    "DataFreshness", "EvidenceStrength", "UserPlan", "TechnicalEvidenceSummary",
    "AdvancedEvidenceReferences", "RealMoneyApproved",
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
        ("Strong " + "Buy", "Track"),
        ("Buy", "Watch"),
        ("Sell", "Avoid"),
        ("Hold", "Wait"),
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
    horizons = pd.to_numeric(evidence.get("Horizon", pd.Series(0, index=evidence.index)), errors="coerce").fillna(0).astype(int)
    return evidence.loc[assets.isin([asset, "ALL", "All", ""]) & horizons.isin([int(horizon), 0])].copy()


def _evidence_strength(rows: pd.DataFrame) -> float:
    if rows.empty or "EvidenceStrength" not in rows.columns:
        return 0.0
    values = pd.to_numeric(rows["EvidenceStrength"], errors="coerce").dropna()
    positive = values[values > 0]
    return float(positive.mean()) if not positive.empty else 0.0


def _confidence_label(strength: float) -> str:
    if strength >= 70:
        return "Higher"
    if strength >= 40:
        return "Moderate"
    return "Low"


def _joined_evidence(rows: pd.DataFrame) -> str:
    fields = []
    for column in ("Source", "Metric", "Value", "Status", "Warning", "Freshness"):
        if column in rows.columns:
            fields.extend(rows[column].dropna().astype(str).tolist())
    return " ".join(fields).casefold()


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
    freshness_rows = rows[rows.get("Metric", pd.Series("", index=rows.index)).astype(str).str.contains("freshness", case=False, na=False)] if not rows.empty else rows
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
        status = "Data Issue"
        summary = "The current evidence cannot be trusted until the data issue is resolved."
        why = "Freshness or core price evidence is missing, stale, or invalid."
        main_risk = "Decisions based on incomplete market data can be misleading."
    elif missing_exact:
        status = "Not Enough Evidence"
        summary = "There is not enough saved evidence for this asset and horizon yet."
        why = "No usable asset-horizon research record was found."
        main_risk = "A plan formed from sparse evidence would be unreliable."
    elif severe_risk:
        status = "High Risk" if strength >= 25 else "Avoid"
        summary = "Risk warnings dominate the current research evidence."
        why = "One or more severe probability, drawdown, or evidence warnings are active."
        main_risk = "The observed research edge may not survive adverse conditions."
    elif benchmark_weak:
        status = "Watch" if strength >= 45 and positive_forecast else "Wait"
        summary = "The evidence is interesting, but simple benchmarks remain stronger."
        why = "Replay or benchmark comparisons did not establish a dependable advantage."
        main_risk = "A simpler passive or rule-based reference may perform better."
    elif below_threshold:
        status = "Wait"
        summary = "The forecast does not currently clear the evidence and risk threshold."
        why = "The available forecast direction or return is too weak after risk context."
        main_risk = "Small estimated moves may be overwhelmed by uncertainty and costs."
    elif strength >= 70 and positive_forecast:
        status = "Track"
        summary = "This is one of the relatively stronger research combinations to monitor."
        why = "Forecast evidence and saved reliability measures are comparatively stronger without a dominant blocker."
        main_risk = "The evidence is still historical or simulated and can weaken as outcomes mature."
    elif strength >= 40 or positive_forecast:
        status = "Watch"
        summary = "The combination is worth observing, but the evidence is not strong enough to track actively."
        why = "Some evidence is constructive while reliability or breadth remains limited."
        main_risk = "The result may depend on a small sample or one favorable window."
    else:
        status = "Wait"
        summary = "There is no clear reason to prioritize this combination now."
        why = "Available evidence is weak, mixed, or too limited."
        main_risk = "Acting on weak evidence can create false confidence."

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
    user_plan = _sanitize_public_text(
        f"{status}: {summary} Watch {what_to_watch.lower()} Recheck {recheck.lower()}"
    )

    return {
        "Asset": asset,
        "Horizon": int(horizon),
        "Status": status,
        "Confidence": _confidence_label(strength),
        "Summary": _sanitize_public_text(summary),
        "Why": _sanitize_public_text(why),
        "MainRisk": _sanitize_public_text(main_risk),
        "WhatToWatch": _sanitize_public_text(what_to_watch),
        "TrackingCondition": _sanitize_public_text(tracking_condition),
        "InvalidationCondition": _sanitize_public_text(invalidation),
        "RecheckWhen": recheck,
        "DataFreshness": freshness_status,
        "EvidenceStrength": round(strength, 2),
        "UserPlan": user_plan,
        "TechnicalEvidenceSummary": technical_summary,
        "AdvancedEvidenceReferences": references,
        "RealMoneyApproved": False,
    }


def generate_all_asset_plans(evidence: Any) -> pd.DataFrame:
    rows = [
        generate_asset_plan(asset, horizon, evidence)
        for asset in SUPPORTED_ASSETS
        for horizon in AVAILABLE_HORIZONS
    ]
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


def rank_asset_plans(asset_plans: Any) -> pd.DataFrame:
    plans = _frame(asset_plans)
    if plans.empty:
        return pd.DataFrame(columns=PLAN_COLUMNS + ["PlanRank"])
    priority = {"Track": 0, "Watch": 1, "Wait": 2, "Not Enough Evidence": 3, "Data Issue": 4, "High Risk": 5, "Avoid": 6}
    ranked = plans.copy()
    ranked["_priority"] = ranked["Status"].map(priority).fillna(9)
    ranked["_strength"] = pd.to_numeric(ranked.get("EvidenceStrength", 0), errors="coerce").fillna(0)
    ranked = ranked.sort_values(["_priority", "_strength", "Asset", "Horizon"], ascending=[True, False, True, True]).reset_index(drop=True)
    ranked["PlanRank"] = range(1, len(ranked) + 1)
    return ranked.drop(columns=["_priority", "_strength"])


def generate_portfolio_plan(asset_plans: Any) -> pd.DataFrame:
    plans = _frame(asset_plans)
    counts = plans.get("Status", pd.Series(dtype=str)).value_counts()
    ranked = rank_asset_plans(plans)
    track_rows = ranked[ranked["Status"].eq("Track")] if not ranked.empty else ranked
    main_risk = "Not enough evidence is available to summarize portfolio risk."
    if not plans.empty and "MainRisk" in plans.columns:
        risks = plans["MainRisk"].dropna().astype(str)
        if not risks.empty:
            main_risk = risks.value_counts().index[0]
    return pd.DataFrame(
        [
            {
                "BestToTrack": "; ".join(f"{row.Asset} {int(row.Horizon)}D" for row in track_rows.head(5).itertuples()) or "None",
                "TrackCount": int(counts.get("Track", 0)),
                "WatchCount": int(counts.get("Watch", 0)),
                "WaitCount": int(counts.get("Wait", 0) + counts.get("Not Enough Evidence", 0)),
                "AvoidHighRiskCount": int(counts.get("Avoid", 0) + counts.get("High Risk", 0)),
                "DataIssueCount": int(counts.get("Data Issue", 0)),
                "MainMarketRisk": _sanitize_public_text(main_risk),
                "WhatToRecheckNext": "Forward outcomes, data freshness, benchmark edge, and risk warnings.",
                "RealMoneyApproved": False,
                "ApprovalExplanation": "Real-money decisions are not approved because this system is a research assistant and evidence remains conditional.",
            }
        ]
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
    "ALLOWED_PLAN_STATUSES",
    "PLAN_COLUMNS",
    "generate_asset_plan",
    "generate_all_asset_plans",
    "generate_portfolio_plan",
    "rank_asset_plans",
    "explain_plan_in_plain_english",
]
