"""Cost-aware paper-research planning helpers.

This module is an explanation and simulation layer. It does not alter forecasts,
benchmarks, risk verdicts, model outputs, or real-capital eligibility.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from src.paper_research_journey import build_passive_benchmark_guide


COST_DISCLAIMER = (
    "Cost estimates are assumptions. Enter your own broker, platform, tax, spread, "
    "and slippage values for a more realistic paper simulation."
)

COST_ASSUMPTION_FIELDS = (
    "EntryBrokerage", "ExitBrokerage", "EntrySpreadPct", "ExitSpreadPct",
    "EntrySlippagePct", "ExitSlippagePct", "PlatformFee", "TaxesAndChargesPct",
    "ExpenseRatioPct", "CurrencyConversionPct", "OtherCost", "Notes",
)

ALLOWED_COST_VERDICTS = (
    "CostsLow", "CostsManageable", "CostsHigh", "CostsTooHighForSignal", "MissingEstimate",
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return default if not math.isfinite(result) else result
    except (TypeError, ValueError):
        return default


def _optional_number(value: Any) -> Optional[float]:
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


def _first_available(row: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _optional_number(row.get(key))
        if value is not None:
            return value
    return None


def default_cost_assumptions(asset: Optional[str] = None) -> Dict[str, Any]:
    """Return conservative editable defaults shared by every configured asset."""
    return {
        "Asset": str(asset or "All assets"),
        "EntryBrokerage": 10.0,
        "ExitBrokerage": 10.0,
        "EntrySpreadPct": 0.05,
        "ExitSpreadPct": 0.05,
        "EntrySlippagePct": 0.03,
        "ExitSlippagePct": 0.03,
        "PlatformFee": 0.0,
        "TaxesAndChargesPct": 0.10,
        "ExpenseRatioPct": 0.0,
        "CurrencyConversionPct": 0.0,
        "OtherCost": 0.0,
        "Notes": "Editable paper-simulation assumptions; replace with your own cost schedule.",
    }


def _normalized_assumptions(asset: Optional[str], assumptions: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    result = default_cost_assumptions(asset)
    if isinstance(assumptions, Mapping):
        for field in COST_ASSUMPTION_FIELDS:
            if field in assumptions:
                result[field] = assumptions[field]
    for field in COST_ASSUMPTION_FIELDS:
        if field != "Notes":
            result[field] = max(0.0, _number(result.get(field), 0.0))
    result["Notes"] = str(result.get("Notes", ""))
    return result


def _cost_components(amount: float, assumptions: Mapping[str, Any]) -> Dict[str, float]:
    principal = max(0.0, _number(amount, 0.0))
    entry_variable_pct = _number(assumptions.get("EntrySpreadPct")) + _number(assumptions.get("EntrySlippagePct"))
    exit_variable_pct = (
        _number(assumptions.get("ExitSpreadPct"))
        + _number(assumptions.get("ExitSlippagePct"))
        + _number(assumptions.get("TaxesAndChargesPct"))
        + _number(assumptions.get("ExpenseRatioPct"))
        + _number(assumptions.get("CurrencyConversionPct"))
    )
    entry_cost = _number(assumptions.get("EntryBrokerage")) + principal * entry_variable_pct / 100.0
    exit_cost = _number(assumptions.get("ExitBrokerage")) + principal * exit_variable_pct / 100.0
    entry_cost += _number(assumptions.get("PlatformFee")) + _number(assumptions.get("OtherCost"))
    total = entry_cost + exit_cost
    return {
        "EstimatedEntryCost": round(entry_cost, 4),
        "EstimatedExitCost": round(exit_cost, 4),
        "EstimatedRoundTripCost": round(total, 4),
    }


def calculate_round_trip_cost(amount: float, cost_assumptions: Optional[Mapping[str, Any]] = None) -> float:
    assumptions = _normalized_assumptions(None, cost_assumptions)
    return _cost_components(amount, assumptions)["EstimatedRoundTripCost"]


def calculate_break_even_return(amount: float, total_cost: float) -> float:
    principal = max(0.0, _number(amount, 0.0))
    if principal <= 0:
        return 0.0
    return round(max(0.0, _number(total_cost, 0.0)) / principal * 100.0, 4)


def estimate_net_return(
    gross_return_pct: Optional[float],
    amount: float,
    cost_assumptions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    gross = _optional_number(gross_return_pct)
    principal = max(0.0, _number(amount, 0.0))
    assumptions = _normalized_assumptions(None, cost_assumptions)
    components = _cost_components(principal, assumptions)
    cost_drag = calculate_break_even_return(principal, components["EstimatedRoundTripCost"])
    if gross is None:
        return {
            "GrossReturnPct": None, "NetReturnPct": None, "GrossReturnAmount": None,
            "NetReturnAmount": None, "CostDragPct": cost_drag, **components,
        }
    net_pct = gross - cost_drag
    return {
        "GrossReturnPct": round(gross, 4),
        "NetReturnPct": round(net_pct, 4),
        "GrossReturnAmount": round(principal * gross / 100.0, 4),
        "NetReturnAmount": round(principal * net_pct / 100.0, 4),
        "CostDragPct": cost_drag,
        **components,
    }


def _cost_verdict(active_return_pct: Optional[float], cost_drag_pct: float) -> str:
    active = _optional_number(active_return_pct)
    if active is None:
        return "MissingEstimate"
    if abs(active) <= cost_drag_pct or (active > 0 and active - cost_drag_pct <= 0):
        return "CostsTooHighForSignal"
    if cost_drag_pct <= 0.25:
        return "CostsLow"
    if cost_drag_pct <= 0.75:
        return "CostsManageable"
    return "CostsHigh"


def compare_active_vs_passive_after_costs(
    active_return_pct: Optional[float],
    passive_return_pct: Optional[float],
    amount: float,
    cost_assumptions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    principal = max(0.0, _number(amount, 0.0))
    active = estimate_net_return(active_return_pct, principal, cost_assumptions)
    passive = estimate_net_return(passive_return_pct, principal, cost_assumptions)
    active_net = _optional_number(active.get("NetReturnPct"))
    passive_net = _optional_number(passive.get("NetReturnPct"))
    gap = round(active_net - passive_net, 4) if active_net is not None and passive_net is not None else None
    verdict = _cost_verdict(active_return_pct, _number(active.get("CostDragPct")))
    active_available = _optional_number(active_return_pct) is not None
    passive_available = _optional_number(passive_return_pct) is not None
    active_explanation = (
        "Active estimate is available and its net value includes the entered cost assumptions."
        if active_available
        else "Run Full Research first to generate an active estimate. No saved estimate for this horizon yet."
    )
    passive_explanation = (
        "Passive estimate is available and its net value includes the entered cost assumptions."
        if passive_available
        else "No passive benchmark estimate is available for this horizon yet. The benchmark is still shown as a comparison reference."
    )
    if verdict == "MissingEstimate":
        warning = active_explanation
        if not passive_available:
            warning = f"{warning} {passive_explanation}"
    elif verdict == "CostsTooHighForSignal":
        warning = "The estimated active move is not larger than the assumed round-trip cost drag."
        if not passive_available:
            warning = f"{warning} {passive_explanation}"
    elif not passive_available:
        warning = passive_explanation
    elif passive_net is not None and active_net is not None and passive_net > active_net:
        warning = "The passive benchmark is stronger after the same paper-simulation cost assumptions."
    else:
        warning = "The estimate has room beyond assumed costs, but the research edge remains unproven."
    return {
        "Amount": round(principal, 2),
        "EstimatedEntryCost": active["EstimatedEntryCost"],
        "EstimatedExitCost": active["EstimatedExitCost"],
        "EstimatedRoundTripCost": active["EstimatedRoundTripCost"],
        "CostDragPct": active["CostDragPct"],
        "BreakEvenReturnPct": active["CostDragPct"],
        "GrossActiveEstimatePct": _optional_number(active_return_pct),
        "NetActiveEstimatePct": active_net,
        "GrossPassiveEstimatePct": _optional_number(passive_return_pct),
        "NetPassiveEstimatePct": passive_net,
        "ActiveMinusPassiveNetPct": gap,
        "CostVerdict": verdict,
        "EstimateComparisonStatus": "Available" if active_available and passive_available else "MissingEstimate",
        "ActiveEstimateAvailable": active_available,
        "PassiveEstimateAvailable": passive_available,
        "ActiveEstimateExplanation": active_explanation,
        "PassiveEstimateExplanation": passive_explanation,
        "CostComparisonExplanation": (
            "Active and passive after-cost estimates are available for comparison."
            if active_available and passive_available
            else "Cost comparison will appear after active/passive estimates are available."
        ),
        "CostWarning": warning,
        "BeginnerExplanation": (
            f"A simulated amount of {principal:,.2f} needs about {active['CostDragPct']:.2f}% movement just to cover "
            "the entered round-trip assumptions. This is a paper estimate, not a real-money approval."
        ),
        "CostDisclaimer": COST_DISCLAIMER,
    }


def explain_opportunity_score(plan: Mapping[str, Any]) -> Dict[str, str]:
    score = max(0.0, min(100.0, _number(plan.get("OpportunityScore"), 0.0)))
    status = str(plan.get("Status", "Not Enough Evidence"))
    positive = str(plan.get("PositiveEvidence") or "No strong positive evidence is confirmed yet.")
    negative = str(plan.get("NegativeEvidence") or plan.get("MainRisk") or "Evidence remains limited.")
    reduced_by = str(plan.get("BlockReason") or plan.get("MainRisk") or "Weak Evidence")
    improve = str(plan.get("WhatMustImprove") or plan.get("ImprovementNeeded") or "More repeated evidence is required.")
    return {
        "ScoreMeaning": "OpportunityScore does not mean expected profit. It means how close this asset is to becoming worth tracking in research mode.",
        "ScorePositiveDrivers": positive,
        "ScoreNegativeDrivers": negative,
        "ScoreReducedBy": reduced_by,
        "ScoreCanImproveIf": improve,
        "ScorePlainEnglishSummary": (
            f"The score is {score:.0f}/100 and the status remains {status}. Positive evidence helps the rank, "
            f"while {reduced_by.casefold()} prevents a stronger research status."
        ),
    }


def generate_cost_aware_asset_plan(
    plan: Mapping[str, Any],
    passive_guide: Optional[Mapping[str, Any]] = None,
    amount: float = 10000,
    cost_assumptions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    row = dict(plan)
    asset = str(row.get("Asset", "Unknown Asset"))
    assumptions = _normalized_assumptions(asset, cost_assumptions)
    guide = dict(passive_guide or build_passive_benchmark_guide(asset))
    active_return = _first_available(
        row, "PredictedMovePct", "GrossActiveEstimatePct", "PredictedReturnPct", "PredictedReturn",
    )
    latest_price = _first_available(row, "LatestPrice", "CurrentPrice")
    predicted_price = _first_available(row, "PredictedPrice", "NextPredictedPrice")
    if active_return is None and latest_price is not None and predicted_price is not None and latest_price > 0:
        active_return = round((predicted_price / latest_price - 1.0) * 100.0, 4)
        row["PredictedMovePct"] = active_return
        row["ActiveEstimateSource"] = "Computed from predicted price and latest price."
    elif active_return is not None:
        row["ActiveEstimateSource"] = "Loaded from the saved active estimate."
    else:
        row["ActiveEstimateSource"] = "No saved active estimate for this horizon."
    passive_return = _first_available(
        row, "GrossPassiveEstimatePct", "PassiveEstimatePct", "BenchmarkReturnPct",
    )
    comparison = compare_active_vs_passive_after_costs(
        active_return, passive_return, amount, assumptions
    )
    score = explain_opportunity_score(row)
    active_net = comparison.get("NetActiveEstimatePct")
    passive_net = comparison.get("NetPassiveEstimatePct")
    if active_net is None or passive_net is None:
        lesson = "A complete active-versus-passive cost comparison is not available yet."
    elif passive_net > active_net:
        lesson = "The passive reference is stronger after estimated costs in this paper comparison."
    else:
        lesson = "The active estimate is higher after costs, but repeated outcomes are still required before trust can improve."
    return {
        **row,
        **comparison,
        **score,
        "PassiveBenchmarkName": guide.get("PassiveBenchmarkName", f"{asset} passive reference"),
        "PassiveBenchmarkType": guide.get("PassiveBenchmarkType", "Asset-level passive research benchmark"),
        "WhyThisBenchmarkMatters": guide.get("Explanation", "It is a simple passive reference for the same asset."),
        "HowToFollowBenchmarkInResearchMode": guide.get("HowToFollow", "Track it over the same dates and horizon."),
        "WhatToCompareAgainstBenchmark": guide.get("WhatToCompare", "Compare same-period net return and risk."),
        "BenchmarkWarning": guide.get("Warning", "A benchmark can also lose value."),
        "ActiveVsPassiveLesson": lesson,
        "CostAssumptions": assumptions,
        "RealMoneyApproved": False,
        "BrokerExecutionAllowed": False,
    }


def generate_cost_aware_portfolio_plans(
    asset_plans: Any,
    amount: float = 10000,
    cost_assumptions: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    rows = [
        generate_cost_aware_asset_plan(plan, amount=amount, cost_assumptions=cost_assumptions)
        for plan in _records(asset_plans)
    ]
    return pd.DataFrame(rows)


__all__ = [
    "COST_DISCLAIMER", "COST_ASSUMPTION_FIELDS", "ALLOWED_COST_VERDICTS",
    "default_cost_assumptions", "calculate_round_trip_cost", "calculate_break_even_return",
    "estimate_net_return", "compare_active_vs_passive_after_costs", "explain_opportunity_score",
    "generate_cost_aware_asset_plan", "generate_cost_aware_portfolio_plans",
]
