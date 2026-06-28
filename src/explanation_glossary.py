"""Plain-language explanations for forecasting, validation, and risk evidence."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


GLOSSARY: Dict[str, str] = {
    "Asset": "The market instrument being studied, such as Silver, Bitcoin, or the S&P 500.",
    "Horizon": "The number of trading rows between a signal date and its target outcome date.",
    "Prediction Date": "The date when a forecast is formed using information available at that time.",
    "Target Outcome Date": "The future date when a pending forecast becomes eligible for outcome evaluation.",
    "Predicted Return": "The model's estimated price change over the selected horizon.",
    "Realized Return": "The price change that actually occurred after the outcome matured.",
    "Net Return": "Realized strategy return after modeled costs and slippage.",
    "Baseline": "A simple reference strategy that a complex model must outperform to demonstrate useful edge.",
    "Hold-only": "A passive baseline that remains exposed rather than responding to model signals.",
    "Momentum baseline": "A rule that follows the direction of recent price movement without a trained model.",
    "Moving-average baseline": "A rule that compares short and long moving averages to define market direction.",
    "Random baseline": "A seeded chance-based reference used to test whether model behavior exceeds luck.",
    "Leakage": "Use of information that would not have been available when a historical prediction was formed.",
    "Walk-forward validation": "Repeated chronological training and evaluation that preserves the order of time.",
    "Drawdown": "The decline from an equity curve peak to a later trough.",
    "Cost drag": "Performance lost to modeled fees and other execution costs.",
    "Slippage": "The modeled difference between an expected paper entry or exit and a less favorable fill.",
    "Hit rate": "The percentage of evaluated predictions or paper trades with the correct directional outcome.",
    "Sharpe-like": "A research ratio comparing average return with overall return variability.",
    "Sortino-like": "A research ratio comparing average return with harmful downside variability.",
    "Calmar-like": "A research ratio comparing return with maximum drawdown.",
    "Benchmark dominated": "A simple baseline performed better than the model or policy for the same evidence window.",
    "PaperTrack": "A candidate allowed only for simulated forward evidence collection.",
    "WatchlistOnly": "A candidate retained for observation but not currently assigned paper exposure.",
    "RealCapitalBlocked": "The evidence and risk gates do not permit real-capital deployment.",
    "NoBroadEdgeProven": "Results have not demonstrated consistent superiority across enough assets and horizons.",
}

REQUIRED_GLOSSARY_TERMS = tuple(GLOSSARY.keys())


def explain_term(term: str) -> str:
    """Return a case-insensitive glossary explanation."""
    lookup = str(term).strip().casefold()
    for name, explanation in GLOSSARY.items():
        if name.casefold() == lookup:
            return explanation
    return "No glossary explanation is currently available for this term."


def build_glossary_table() -> pd.DataFrame:
    """Return glossary entries as an exportable table."""
    return pd.DataFrame(
        [{"Term": term, "Explanation": explanation} for term, explanation in GLOSSARY.items()]
    )


def glossary_entries(terms: List[str] | None = None) -> Dict[str, str]:
    """Return all entries or a requested subset in canonical display form."""
    if terms is None:
        return dict(GLOSSARY)
    selected = {}
    for term in terms:
        explanation = explain_term(term)
        selected[str(term)] = explanation
    return selected


__all__ = [
    "GLOSSARY",
    "REQUIRED_GLOSSARY_TERMS",
    "explain_term",
    "build_glossary_table",
    "glossary_entries",
]
