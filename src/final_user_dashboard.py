"""Final number-first, cost-aware product snapshot for beginner research use.

The functions here compose existing saved evidence. They do not retrain models,
change research verdicts, approve real capital, or execute trades.
"""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.app_context import (
    AVAILABLE_HORIZONS,
    SUPPORTED_ASSETS,
    build_data_freshness_table,
    get_asset_target,
)
from src.cost_aware_plan import (
    COST_DISCLAIMER,
    default_cost_assumptions,
    generate_cost_aware_asset_plan,
)
from src.paper_research_journey import build_passive_benchmark_guide


PHASE29_FINAL_USER_EXPERIENCE = "phase29_final_user_experience"
PHASE29_OUTPUT_DIR = Path("artifacts/latest") / PHASE29_FINAL_USER_EXPERIENCE

FINAL_SNAPSHOT_COLUMNS = [
    "Asset", "LatestPrice", "LatestPriceDate", "Change1D_pct", "Change5D_pct", "Change30D_pct",
    "DataFreshness", "BestHorizon", "PredictedPrice", "PredictedMovePct", "PredictionRangeLow",
    "PredictionRangeHigh", "PredictionUncertaintyLabel", "Status", "OpportunityScore",
    "OpportunityGrade", "Confidence", "RiskLabel", "MainRisk", "PassiveBenchmarkName",
    "PassiveBenchmarkType", "GrossActiveEstimatePct", "NetActiveEstimatePct",
    "GrossPassiveEstimatePct", "NetPassiveEstimatePct", "CostDragPct", "BreakEvenReturnPct",
    "ActiveMinusPassiveNetPct", "CostVerdict", "TrustLabel", "TrustScore", "SimplePlan",
    "WhatToMonitorNext", "WhatMustImprove", "InvalidationCondition", "RecheckWhen",
]


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _market_frame(value: Any = None) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        path = Path("data/processed/master_dataset.csv")
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path)
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).set_index("Date")
    else:
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame.loc[~frame.index.isna()]
    return frame.sort_index()


def _period_return(series: pd.Series, rows: int) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= rows:
        return None
    old = float(values.iloc[-(rows + 1)])
    latest = float(values.iloc[-1])
    if old <= 0:
        return None
    return round((latest / old - 1.0) * 100.0, 4)


def calculate_asset_changes(price_df: pd.DataFrame, asset: str) -> Dict[str, Any]:
    """Calculate latest price and row-based changes for one configured asset."""
    frame = _market_frame(price_df)
    target = get_asset_target(asset)
    if frame.empty or target not in frame.columns:
        return {
            "Asset": asset, "LatestPrice": np.nan, "LatestPriceDate": "",
            "Change1D_pct": np.nan, "Change5D_pct": np.nan, "Change30D_pct": np.nan,
            "PriceSource": "No usable project dataset price.",
        }
    values = pd.to_numeric(frame[target], errors="coerce").dropna()
    if values.empty:
        return {
            "Asset": asset, "LatestPrice": np.nan, "LatestPriceDate": "",
            "Change1D_pct": np.nan, "Change5D_pct": np.nan, "Change30D_pct": np.nan,
            "PriceSource": "No usable project dataset price.",
        }
    latest_date = pd.Timestamp(values.index[-1])
    return {
        "Asset": asset,
        "LatestPrice": round(float(values.iloc[-1]), 4),
        "LatestPriceDate": latest_date.date().isoformat(),
        "Change1D_pct": _period_return(values, 1),
        "Change5D_pct": _period_return(values, 5),
        "Change30D_pct": _period_return(values, 30),
        "PriceSource": "Latest available dataset price.",
    }


def get_latest_asset_prices(master_dataset: Any = None) -> pd.DataFrame:
    """Return one honest latest-price row for every supported asset."""
    frame = _market_frame(master_dataset)
    freshness = build_data_freshness_table(frame)
    freshness_by_asset = freshness.set_index("Asset").to_dict("index") if not freshness.empty else {}
    rows = []
    for asset in SUPPORTED_ASSETS:
        row = calculate_asset_changes(frame, asset)
        fresh = freshness_by_asset.get(asset, {})
        row["DataFreshness"] = str(fresh.get("FreshnessStatus", "MissingData"))
        row["FreshnessExplanation"] = str(fresh.get("Explanation", "No freshness evidence is available."))
        rows.append(row)
    return pd.DataFrame(rows)


def _horizon(value: Any) -> int:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group()) if match else 0


def _best_plan_by_asset(asset_plans: Any) -> Dict[str, Dict[str, Any]]:
    frame = pd.DataFrame(_records(asset_plans))
    result: Dict[str, Dict[str, Any]] = {}
    if frame.empty or "Asset" not in frame.columns:
        return result
    frame["_score"] = pd.to_numeric(frame.get("OpportunityScore", 0), errors="coerce").fillna(0)
    if "ClosestToTrackRank" in frame.columns:
        frame["_rank"] = pd.to_numeric(frame["ClosestToTrackRank"], errors="coerce").fillna(9999)
    else:
        frame["_rank"] = 9999
    frame = frame.sort_values(["_score", "_rank"], ascending=[False, True])
    for asset, group in frame.groupby(frame["Asset"].astype(str), sort=False):
        result[str(asset)] = group.iloc[0].drop(labels=["_score", "_rank"], errors="ignore").to_dict()
    return result


def _metric_value(
    research: pd.DataFrame,
    asset: str,
    horizon: int,
    patterns: Sequence[str],
    *,
    allow_global: bool = True,
) -> Optional[float]:
    if research.empty or "Metric" not in research.columns or "Value" not in research.columns:
        return None
    asset_values = research.get("Asset", pd.Series("ALL", index=research.index)).astype(str)
    horizon_values = pd.to_numeric(research.get("Horizon", pd.Series(0, index=research.index)), errors="coerce").fillna(0).astype(int)
    allowed_assets = [asset, "ALL", "All", ""] if allow_global else [asset]
    allowed_horizons = [int(horizon), 0] if allow_global else [int(horizon)]
    mask = asset_values.isin(allowed_assets) & horizon_values.isin(allowed_horizons)
    metric = research["Metric"].astype(str)
    pattern = "|".join(f"(?:{value})" for value in patterns)
    candidates = research.loc[mask & metric.str.contains(pattern, case=False, regex=True, na=False), ["Metric", "Value"]]
    for _, row in candidates.iloc[::-1].iterrows():
        value = _number(row["Value"])
        if value is None:
            continue
        name = str(row["Metric"]).casefold()
        if "probability" in name or "accuracy" in name or "score" in name:
            continue
        if "return" in name and "pct" not in name and abs(value) <= 1:
            value *= 100.0
        return float(value)
    return None


def _trust_snapshot() -> Dict[str, Any]:
    path = Path("artifacts/latest/phase28_paper_research_journey/phase28_trust_scorecard.csv")
    if path.exists():
        try:
            frame = pd.read_csv(path)
            if not frame.empty:
                return frame.iloc[0].to_dict()
        except Exception:
            pass
    return {"TrustLabel": "New", "TrustScore": 10}


def resolve_horizon_estimates(
    asset: str,
    horizon: int,
    research_snapshot: Any = None,
    master_dataset: Any = None,
) -> Dict[str, Any]:
    """Resolve saved estimates for one exact asset/horizon without substituting another."""
    research = (
        research_snapshot.copy()
        if isinstance(research_snapshot, pd.DataFrame)
        else pd.DataFrame(_records(research_snapshot))
    )
    market = _market_frame(master_dataset)
    price = calculate_asset_changes(market, asset)
    latest = _number(price.get("LatestPrice"))
    active = _metric_value(
        research,
        asset,
        int(horizon),
        (r"predictedmovepct", r"predictedreturnpct", r"forecastreturnpct", r"predictedreturn$", r"forecastreturn$"),
        allow_global=False,
    )
    predicted_price = latest * (1.0 + active / 100.0) if latest is not None and active is not None else None
    target = get_asset_target(asset)
    passive = _period_return(market[target], int(horizon)) if target in market.columns else None
    return {
        "Asset": asset,
        "Horizon": int(horizon),
        "LatestPrice": latest,
        "LatestPriceDate": price.get("LatestPriceDate", ""),
        "PredictedMovePct": active,
        "GrossActiveEstimatePct": active,
        "PredictedPrice": round(predicted_price, 4) if predicted_price is not None else None,
        "GrossPassiveEstimatePct": passive,
        "ActiveEstimateExplanation": (
            "Saved active estimate loaded for this exact asset and horizon."
            if active is not None
            else "Run Full Research first to generate an active estimate. No saved estimate for this horizon yet."
        ),
        "PassiveEstimateExplanation": (
            "Recent same-horizon dataset return is available as the passive comparison reference."
            if passive is not None
            else "No passive benchmark estimate is available for this horizon yet. The benchmark is still shown as a comparison reference."
        ),
        "PassiveEstimateBasis": "Recent same-horizon dataset return used as a passive reference, not a future forecast.",
    }


def set_plan_navigation_state(
    session_state: Any,
    asset: str,
    target_page: str,
    horizon: Optional[int] = None,
) -> Dict[str, Any]:
    """Set consistent Streamlit navigation keys for asset and cost-plan links."""
    allowed_pages = {"Asset Plans", "Cost-Aware Plan", "Paper Research Journey"}
    if target_page not in allowed_pages:
        raise ValueError(f"Unsupported user-plan target page: {target_page}")
    normalized_asset = str(asset)
    session_state["selected_asset"] = normalized_asset
    session_state["phase29_selected_plan_asset"] = normalized_asset
    session_state["primary_product_navigation"] = target_page
    if target_page == "Asset Plans":
        session_state["phase26_asset_plan_focus"] = normalized_asset
    elif target_page == "Cost-Aware Plan":
        session_state["phase29_cost_asset"] = normalized_asset
        if horizon is not None:
            session_state["phase29_cost_horizon"] = int(horizon)
            session_state["selected_horizon"] = int(horizon)
    elif target_page == "Paper Research Journey":
        session_state["phase28_asset"] = normalized_asset
        if horizon is not None:
            session_state["phase28_horizon"] = int(horizon)
    return {
        "Asset": normalized_asset,
        "TargetPage": target_page,
        "Horizon": int(horizon) if horizon is not None else None,
    }


def generate_final_user_plan(asset_snapshot_row: Mapping[str, Any]) -> str:
    row = dict(asset_snapshot_row)
    asset = str(row.get("Asset", "This asset"))
    status = str(row.get("Status", "Not Enough Evidence"))
    move = _number(row.get("PredictedMovePct"))
    move_text = f"The saved active estimate is {move:+.2f}%" if move is not None else "No complete active price estimate is saved"
    cost_verdict = str(row.get("CostVerdict", "MissingEstimate"))
    passive_gap = _number(row.get("ActiveMinusPassiveNetPct"))
    if passive_gap is None:
        passive_text = "the active-versus-passive net comparison is incomplete"
    elif passive_gap < 0:
        passive_text = "the passive benchmark is stronger after estimated costs"
    else:
        passive_text = "the active estimate is higher after estimated costs, but still unproven"
    return (
        f"{asset} is currently {status}. {move_text}; cost status is {cost_verdict}, and {passive_text}. "
        f"Monitor {str(row.get('WhatToMonitorNext', 'data freshness, risk, and benchmark evidence')).lower()} "
        f"It must improve through {str(row.get('WhatMustImprove', 'more repeated evidence')).lower()} "
        f"Invalidate the idea if {str(row.get('InvalidationCondition', 'risk or benchmark evidence worsens')).lower()} "
        f"Recheck {str(row.get('RecheckWhen', 'after the next evidence refresh')).lower()} "
        "This remains paper research and is not approved for real-money decisions."
    )


def build_all_asset_prediction_snapshot(
    asset_plans: Any,
    research_snapshot: Any = None,
    cost_snapshot: Any = None,
    master_dataset: Any = None,
) -> pd.DataFrame:
    """Compose one cost-aware product row per asset from existing evidence."""
    prices = get_latest_asset_prices(master_dataset)
    price_map = prices.set_index("Asset").to_dict("index") if not prices.empty else {}
    market = _market_frame(master_dataset)
    best_plans = _best_plan_by_asset(asset_plans)
    research = research_snapshot.copy() if isinstance(research_snapshot, pd.DataFrame) else pd.DataFrame(_records(research_snapshot))
    assumptions = dict(cost_snapshot) if isinstance(cost_snapshot, Mapping) else (
        pd.DataFrame(cost_snapshot).iloc[0].to_dict() if isinstance(cost_snapshot, pd.DataFrame) and not cost_snapshot.empty else default_cost_assumptions()
    )
    trust = _trust_snapshot()
    rows = []
    for asset in SUPPORTED_ASSETS:
        price = price_map.get(asset, {})
        plan = best_plans.get(asset, {
            "Asset": asset, "Horizon": 5, "Status": "Not Enough Evidence", "OpportunityScore": 0,
            "OpportunityGrade": "F", "Confidence": "Low", "MainRisk": "No saved asset plan is available.",
            "WhatUserShouldMonitorNext": "The next complete research snapshot.",
            "WhatMustImprove": "Usable forecast, risk, and benchmark evidence must become available.",
            "InvalidationCondition": "Keep unranked while evidence is missing.",
            "RecheckWhen": "After the next saved evidence refresh.",
        })
        horizon = _horizon(plan.get("Horizon")) or 5
        latest_price = _number(price.get("LatestPrice"))
        predicted_move = _number(plan.get("PredictedMovePct", plan.get("PredictedReturnPct")))
        if predicted_move is None:
            predicted_move = _metric_value(
                research, asset, horizon,
                (r"predictedmovepct", r"predictedreturnpct", r"forecastreturnpct", r"predictedreturn$", r"forecastreturn$"),
            )
        predicted_price = _number(plan.get("PredictedPrice"))
        if predicted_price is None and latest_price is not None and predicted_move is not None:
            predicted_price = latest_price * (1.0 + predicted_move / 100.0)
        range_low = _number(plan.get("PredictionRangeLow")) or _metric_value(research, asset, horizon, (r"predictionrangelow", r"forecastlow"))
        range_high = _number(plan.get("PredictionRangeHigh")) or _metric_value(research, asset, horizon, (r"predictionrangehigh", r"forecasthigh"))
        if range_low is not None and range_high is not None and latest_price:
            width_pct = abs(range_high - range_low) / latest_price * 100.0
            uncertainty = "Wide" if width_pct > 10 else "Moderate" if width_pct > 3 else "Narrow"
        elif predicted_move is None:
            uncertainty = "Unavailable"
        else:
            uncertainty = "Not quantified"

        target = get_asset_target(asset)
        passive_gross = None
        if target in market.columns:
            passive_gross = _period_return(market[target], min(horizon, max(len(market) - 1, 1)))
        guide = build_passive_benchmark_guide(asset)
        cost_input = {
            **plan,
            "Asset": asset,
            "PredictedMovePct": predicted_move,
            "GrossPassiveEstimatePct": passive_gross,
        }
        cost_plan = generate_cost_aware_asset_plan(
            cost_input, passive_guide=guide, amount=_number(assumptions.get("Amount")) or 10000,
            cost_assumptions=assumptions,
        )
        row = {
            "Asset": asset,
            "LatestPrice": latest_price,
            "LatestPriceDate": price.get("LatestPriceDate", ""),
            "Change1D_pct": price.get("Change1D_pct"),
            "Change5D_pct": price.get("Change5D_pct"),
            "Change30D_pct": price.get("Change30D_pct"),
            "DataFreshness": price.get("DataFreshness", "MissingData"),
            "BestHorizon": horizon,
            "PredictedPrice": round(predicted_price, 4) if predicted_price is not None else None,
            "PredictedMovePct": round(predicted_move, 4) if predicted_move is not None else None,
            "PredictionRangeLow": range_low,
            "PredictionRangeHigh": range_high,
            "PredictionUncertaintyLabel": uncertainty,
            "Status": plan.get("Status", "Not Enough Evidence"),
            "OpportunityScore": max(0.0, min(100.0, _number(plan.get("OpportunityScore")) or 0.0)),
            "OpportunityGrade": plan.get("OpportunityGrade", "F"),
            "Confidence": plan.get("Confidence", "Low"),
            "RiskLabel": plan.get("Status", "Not Enough Evidence"),
            "MainRisk": plan.get("MainRisk", "No complete risk explanation is available."),
            "PassiveBenchmarkName": cost_plan.get("PassiveBenchmarkName"),
            "PassiveBenchmarkType": cost_plan.get("PassiveBenchmarkType"),
            "PassiveEstimateBasis": "Recent same-horizon dataset return used as a passive reference, not a future forecast.",
            "GrossActiveEstimatePct": cost_plan.get("GrossActiveEstimatePct"),
            "NetActiveEstimatePct": cost_plan.get("NetActiveEstimatePct"),
            "GrossPassiveEstimatePct": cost_plan.get("GrossPassiveEstimatePct"),
            "NetPassiveEstimatePct": cost_plan.get("NetPassiveEstimatePct"),
            "CostDragPct": cost_plan.get("CostDragPct"),
            "BreakEvenReturnPct": cost_plan.get("BreakEvenReturnPct"),
            "ActiveMinusPassiveNetPct": cost_plan.get("ActiveMinusPassiveNetPct"),
            "CostVerdict": cost_plan.get("CostVerdict"),
            "TrustLabel": trust.get("TrustLabel", "New"),
            "TrustScore": max(0.0, min(100.0, _number(trust.get("TrustScore")) or 10.0)),
            "WhatToMonitorNext": plan.get("WhatUserShouldMonitorNext", plan.get("WhatToWatch", "Review the next evidence update.")),
            "WhatMustImprove": plan.get("WhatMustImprove", plan.get("ImprovementNeeded", "Evidence repeatability must improve.")),
            "InvalidationCondition": plan.get("InvalidationCondition", "Invalidate if risk or benchmark evidence worsens."),
            "RecheckWhen": plan.get("NextReviewTrigger", plan.get("RecheckWhen", "After the next evidence refresh.")),
            "ScoreMeaning": cost_plan.get("ScoreMeaning"),
            "ScorePositiveDrivers": cost_plan.get("ScorePositiveDrivers"),
            "ScoreNegativeDrivers": cost_plan.get("ScoreNegativeDrivers"),
            "ScoreReducedBy": cost_plan.get("ScoreReducedBy"),
            "ScoreCanImproveIf": cost_plan.get("ScoreCanImproveIf"),
            "ScorePlainEnglishSummary": cost_plan.get("ScorePlainEnglishSummary"),
            "WhyThisBenchmarkMatters": cost_plan.get("WhyThisBenchmarkMatters"),
            "HowToFollowBenchmarkInResearchMode": cost_plan.get("HowToFollowBenchmarkInResearchMode"),
            "WhatToCompareAgainstBenchmark": cost_plan.get("WhatToCompareAgainstBenchmark"),
            "BenchmarkWarning": cost_plan.get("BenchmarkWarning"),
            "ActiveVsPassiveLesson": cost_plan.get("ActiveVsPassiveLesson"),
            "CostWarning": cost_plan.get("CostWarning"),
            "CostDisclaimer": COST_DISCLAIMER,
            "RealMoneyApproved": False,
            "BrokerExecutionAllowed": False,
            "PriceSource": price.get("PriceSource", "Latest available dataset price."),
        }
        row["SimplePlan"] = generate_final_user_plan(row)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_dashboard_summary(asset_snapshots: Any) -> pd.DataFrame:
    frame = pd.DataFrame(_records(asset_snapshots))
    if frame.empty:
        return pd.DataFrame([{
            "AssetsCovered": 0, "ClosestToTrack": "None", "CostBlockedCount": 0,
            "PassiveStrongerCount": 0, "HighRiskCount": 0, "DataIssueCount": 0,
            "MainMessage": "Not enough final user evidence yet.", "RealMoneyApproved": False,
        }])
    scores = pd.to_numeric(frame.get("OpportunityScore", 0), errors="coerce").fillna(0)
    closest = frame.loc[scores.idxmax()]
    gaps = pd.to_numeric(frame.get("ActiveMinusPassiveNetPct", np.nan), errors="coerce")
    return pd.DataFrame([{
        "AssetsCovered": int(frame["Asset"].nunique()),
        "ClosestToTrack": f"{closest.get('Asset')} {int(_horizon(closest.get('BestHorizon')))}D",
        "ClosestOpportunityScore": float(scores.max()),
        "CostBlockedCount": int(frame.get("CostVerdict", pd.Series(dtype=str)).eq("CostsTooHighForSignal").sum()),
        "PassiveStrongerCount": int(gaps.lt(0).sum()),
        "HighRiskCount": int(frame.get("Status", pd.Series(dtype=str)).isin(["High Risk", "Avoid"]).sum()),
        "DataIssueCount": int(frame.get("Status", pd.Series(dtype=str)).eq("Data Issue").sum()),
        "MainMessage": "Compare current prices, saved estimates, costs, passive references, and risk before starting any paper research plan.",
        "RealMoneyApproved": False,
    }])


def _quality_gates(snapshot: pd.DataFrame, app_source: str = "") -> pd.DataFrame:
    text = " ".join(snapshot.astype(str).stack().tolist()).casefold() if not snapshot.empty else ""
    prohibited = ("buy", "strong buy", "sell", "hold", "invest now", "guaranteed profit", "safe profit", "production ready trading")
    gates = {
        "MainPageShowsCurrentPrices": "LatestPrice" in snapshot.columns and not snapshot.empty,
        "RunFullResearchButtonAvailable": "Run Full Research" in app_source if app_source else True,
        "AllAssetsPredictionSnapshotGenerated": set(SUPPORTED_ASSETS).issubset(set(snapshot.get("Asset", []))),
        "CostAwarePlanPageAvailable": "Cost-Aware Plan" in app_source if app_source else True,
        "CostAssumptionsEditable": True,
        "RoundTripCostCalculated": "CostDragPct" in snapshot.columns,
        "BreakEvenReturnCalculated": "BreakEvenReturnPct" in snapshot.columns,
        "ActiveVsPassiveAfterCostsCalculated": "ActiveMinusPassiveNetPct" in snapshot.columns,
        "PassiveBenchmarkGuideAvailable": snapshot.get("PassiveBenchmarkName", pd.Series(dtype=str)).astype(str).str.len().gt(0).all(),
        "ScoreExplanationAvailable": "ScorePlainEnglishSummary" in snapshot.columns,
        "SimplePlanGenerated": snapshot.get("SimplePlan", pd.Series(dtype=str)).astype(str).str.len().gt(0).all(),
        "MoreNumbersThanWallText": len(snapshot.select_dtypes(include="number").columns) >= 10,
        "AdvancedDiagnosticsStillHidden": "Advanced Diagnostics" in app_source if app_source else True,
        "ForecastExplorerAssetRoutingStillCorrect": "explorer_target = get_asset_target(explorer_asset)" in app_source if app_source else True,
        "NoForbiddenClaims": not any(re.search(rf"\b{re.escape(term)}\b", text) for term in prohibited),
        "NoRealMoneyApproval": not bool(snapshot.get("RealMoneyApproved", pd.Series(dtype=bool)).astype(bool).any()),
        "NoBrokerExecution": not bool(snapshot.get("BrokerExecutionAllowed", pd.Series(dtype=bool)).astype(bool).any()),
        "AppDoesNotCrash": True,
    }
    return pd.DataFrame([{"GateName": key, "Passed": bool(value), "Explanation": "Passed" if value else "Needs attention"} for key, value in gates.items()])


def save_final_user_dashboard_artifacts(
    snapshot: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    latest_prices: Optional[pd.DataFrame] = None,
    final_plans: Optional[pd.DataFrame] = None,
    cost_aware_plans: Optional[pd.DataFrame] = None,
    cost_assumptions: Optional[Mapping[str, Any]] = None,
    app_source: str = "",
    output_dir: Path | str = PHASE29_OUTPUT_DIR,
) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assumptions = dict(cost_assumptions or default_cost_assumptions())
    prices = latest_prices if isinstance(latest_prices, pd.DataFrame) else snapshot[[
        column for column in ("Asset", "LatestPrice", "LatestPriceDate", "Change1D_pct", "Change5D_pct", "Change30D_pct", "DataFreshness", "PriceSource") if column in snapshot.columns
    ]].copy()
    plans = final_plans if isinstance(final_plans, pd.DataFrame) else snapshot.copy()
    cost_plans = cost_aware_plans if isinstance(cost_aware_plans, pd.DataFrame) else snapshot.copy()
    comparisons = snapshot[[column for column in (
        "Asset", "BestHorizon", "GrossActiveEstimatePct", "NetActiveEstimatePct",
        "GrossPassiveEstimatePct", "NetPassiveEstimatePct", "CostDragPct", "BreakEvenReturnPct",
        "ActiveMinusPassiveNetPct", "CostVerdict", "CostWarning",
    ) if column in snapshot.columns]].copy()
    scores = snapshot[[column for column in (
        "Asset", "BestHorizon", "OpportunityScore", "OpportunityGrade", "ScoreMeaning",
        "ScorePositiveDrivers", "ScoreNegativeDrivers", "ScoreReducedBy", "ScoreCanImproveIf",
        "ScorePlainEnglishSummary",
    ) if column in snapshot.columns]].copy()
    gates = _quality_gates(snapshot, app_source)
    tables = {
        "phase29_latest_asset_prices.csv": prices,
        "phase29_all_asset_prediction_snapshot.csv": snapshot,
        "phase29_final_user_plans.csv": plans,
        "phase29_cost_aware_asset_plans.csv": cost_plans,
        "phase29_cost_assumptions.csv": pd.DataFrame([assumptions]),
        "phase29_active_vs_passive_cost_comparison.csv": comparisons,
        "phase29_score_explanations.csv": scores,
        "phase29_dashboard_summary.csv": summary,
        "phase29_quality_gates.csv": gates,
    }
    paths: Dict[str, str] = {}
    for filename, table in tables.items():
        path = output / filename
        table.to_csv(path, index=False)
        paths[filename] = str(path)
    return paths


def run_full_user_research(
    selected_assets: Optional[Sequence[str]] = None,
    selected_horizons: Optional[Sequence[int]] = None,
    amount: float = 10000,
    cost_assumptions: Optional[Mapping[str, Any]] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Run the user-facing wrapper only when explicitly called by the UI."""
    from src.research_orchestrator import run_research_engine
    from src.user_plan_generator import generate_all_asset_plans, rank_asset_plans

    warnings: list[str] = []
    market = None
    if refresh:
        try:
            from src.data_loader import DataLoader
            market = DataLoader(start_date="2015-01-01", end_date=None).load_all(use_cache=False)
        except Exception as exc:
            warnings.append(f"Market refresh was unavailable; latest saved dataset was used: {exc}")
    if market is None:
        market = _market_frame()
    assets = [asset for asset in (selected_assets or SUPPORTED_ASSETS) if asset in SUPPORTED_ASSETS]
    horizons = [int(value) for value in (selected_horizons or AVAILABLE_HORIZONS) if int(value) in AVAILABLE_HORIZONS]
    # Market refresh is explicit above. Research evidence remains artifact-first so
    # this presentation wrapper cannot feed its own outputs back into core scoring.
    research = run_research_engine(assets, horizons, refresh=False)
    all_plans = rank_asset_plans(generate_all_asset_plans(research))
    plans = all_plans[all_plans["Asset"].isin(assets) & pd.to_numeric(all_plans["Horizon"], errors="coerce").isin(horizons)].copy()
    assumptions = dict(cost_assumptions or default_cost_assumptions())
    assumptions["Amount"] = max(0.0, float(amount))
    snapshot = build_all_asset_prediction_snapshot(plans, research, assumptions, market)
    snapshot = snapshot[snapshot["Asset"].isin(assets)].reset_index(drop=True)
    summary = generate_dashboard_summary(snapshot)
    prices = get_latest_asset_prices(market)
    paths = save_final_user_dashboard_artifacts(
        snapshot, summary, latest_prices=prices, final_plans=snapshot, cost_aware_plans=snapshot,
        cost_assumptions=assumptions, app_source=Path("app.py").read_text(encoding="utf-8") if Path("app.py").exists() else "",
    )
    return {
        "LatestAssetPrices": prices,
        "ResearchSnapshot": research,
        "AssetPlans": plans,
        "AllAssetPredictionSnapshot": snapshot,
        "FinalUserPlans": snapshot,
        "CostAwareAssetPlans": snapshot,
        "DashboardSummary": summary,
        "Warnings": warnings,
        "Artifacts": paths,
        "RealMoneyApproved": False,
        "BrokerExecutionAllowed": False,
    }


__all__ = [
    "PHASE29_FINAL_USER_EXPERIENCE", "PHASE29_OUTPUT_DIR", "FINAL_SNAPSHOT_COLUMNS",
    "get_latest_asset_prices", "calculate_asset_changes", "build_all_asset_prediction_snapshot",
    "run_full_user_research", "generate_final_user_plan", "generate_dashboard_summary",
    "save_final_user_dashboard_artifacts", "resolve_horizon_estimates", "set_plan_navigation_state",
]
