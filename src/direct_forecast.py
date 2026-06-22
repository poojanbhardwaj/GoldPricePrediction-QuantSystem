# src/direct_forecast.py

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def add_direct_horizon_targets(
    df: pd.DataFrame,
    *,
    price_col: str,
    horizons: Iterable[int] = (1, 7, 30),
) -> pd.DataFrame:
    """
    Adds direct multi-horizon return targets:
        target_return_1d
        target_return_7d
        target_return_30d

    Formula:
        future_price / current_price - 1

    Important:
        These target columns must be excluded from feature columns to avoid leakage.
    """

    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")

    out = df.copy()
    price = pd.to_numeric(out[price_col], errors="coerce")

    for h in horizons:
        if h <= 0:
            raise ValueError("Horizon values must be positive.")
        out[f"target_return_{h}d"] = price.shift(-h) / price - 1.0

    return out


def horizon_return_to_price(last_price: float, predicted_return: float) -> float:
    last_price = float(last_price)
    predicted_return = float(predicted_return)

    if not np.isfinite(last_price) or last_price <= 0:
        raise ValueError("last_price must be a positive finite number.")

    if not np.isfinite(predicted_return):
        raise ValueError("predicted_return must be finite.")

    return float(last_price * (1.0 + predicted_return))


if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    demo = pd.DataFrame({"Gold_Close": np.linspace(2000, 2100, len(dates))}, index=dates)
    demo = add_direct_horizon_targets(demo, price_col="Gold_Close")
    print(demo.tail(10))
