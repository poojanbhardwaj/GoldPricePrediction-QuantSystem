# src/portfolio.py

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def simple_score_allocation(
    predicted_returns: Dict[str, float],
    volatilities: Dict[str, float],
    *,
    cash_floor: float = 0.10,
    max_asset_weight: float = 0.35,
) -> pd.DataFrame:
    """
    Simple allocation logic:
        higher predicted return + lower volatility = higher allocation.

    Inputs:
        predicted_returns: decimal returns, e.g. 0.015 for 1.5%
        volatilities: annualized decimal vol, e.g. 0.20 for 20%

    Output:
        DataFrame with weights in percentage.
    """

    if not 0 <= cash_floor <= 1:
        raise ValueError("cash_floor must be between 0 and 1.")

    if not 0 < max_asset_weight <= 1:
        raise ValueError("max_asset_weight must be between 0 and 1.")

    assets = sorted(set(predicted_returns) & set(volatilities))

    if not assets:
        raise ValueError("No overlapping assets between predicted_returns and volatilities.")

    raw_scores = {}
    for asset in assets:
        ret = float(predicted_returns[asset])
        vol = float(volatilities[asset])

        if not np.isfinite(ret) or not np.isfinite(vol) or vol <= 0:
            score = 0.0
        else:
            score = max(ret, 0.0) / vol

        raw_scores[asset] = score

    score_sum = sum(raw_scores.values())

    investable = 1.0 - cash_floor

    if score_sum <= 0:
        weights = {asset: 0.0 for asset in assets}
        cash = 1.0
    else:
        weights = {asset: investable * raw_scores[asset] / score_sum for asset in assets}

        # Cap overweight assets.
        excess = 0.0
        for asset in assets:
            if weights[asset] > max_asset_weight:
                excess += weights[asset] - max_asset_weight
                weights[asset] = max_asset_weight

        uncapped = [a for a in assets if weights[a] < max_asset_weight and raw_scores[a] > 0]
        if excess > 0 and uncapped:
            uncapped_sum = sum(raw_scores[a] for a in uncapped)
            for a in uncapped:
                weights[a] += excess * raw_scores[a] / uncapped_sum
                weights[a] = min(weights[a], max_asset_weight)

        total_weight = sum(weights.values())
        cash = max(0.0, 1.0 - total_weight)

    rows = []
    for asset in assets:
        rows.append(
            {
                "Asset": asset,
                "Predicted_Return_pct": predicted_returns[asset] * 100.0,
                "Volatility_pct": volatilities[asset] * 100.0,
                "Allocation_pct": weights[asset] * 100.0,
            }
        )

    rows.append(
        {
            "Asset": "Cash",
            "Predicted_Return_pct": 0.0,
            "Volatility_pct": 0.0,
            "Allocation_pct": cash * 100.0,
        }
    )

    return pd.DataFrame(rows).sort_values("Allocation_pct", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    pred = {"Gold": 0.012, "Silver": 0.008, "Bitcoin": 0.025, "S&P 500": 0.006}
    vol = {"Gold": 0.16, "Silver": 0.25, "Bitcoin": 0.65, "S&P 500": 0.18}
    print(simple_score_allocation(pred, vol))
