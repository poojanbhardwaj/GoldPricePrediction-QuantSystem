# src/risk.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass
class RiskReport:
    volatility_annual_pct: float
    value_at_risk_95_daily_pct: float
    max_drawdown_pct: float
    risk_level: str
    suggested_position_size_pct: float
    suggested_stop_loss_pct: float
    suggested_take_profit_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_max_drawdown_from_prices(prices: pd.Series) -> float:
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    if prices.empty:
        raise ValueError("No valid prices for drawdown calculation.")

    equity = prices / prices.iloc[0]
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def generate_risk_report(
    prices: pd.Series,
    *,
    target_volatility_annual: float = 0.10,
) -> RiskReport:
    prices = pd.to_numeric(prices, errors="coerce").dropna()

    if len(prices) < 30:
        raise ValueError("Need at least 30 price observations for a useful risk report.")

    returns = prices.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        raise ValueError("No valid returns available.")

    daily_vol = float(returns.std(ddof=1))
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    var_95_daily = float(np.percentile(returns, 5))
    max_dd = calculate_max_drawdown_from_prices(prices)

    if annual_vol < 0.15 and abs(max_dd) < 0.10:
        risk_level = "Low"
    elif annual_vol < 0.35 and abs(max_dd) < 0.25:
        risk_level = "Medium"
    else:
        risk_level = "High"

    if annual_vol <= 0:
        position_size = 0.0
    else:
        position_size = min(1.0, target_volatility_annual / annual_vol)

    suggested_stop = max(0.01, min(0.10, 2.0 * daily_vol))
    suggested_take_profit = max(0.02, min(0.20, 4.0 * daily_vol))

    return RiskReport(
        volatility_annual_pct=round(annual_vol * 100.0, 4),
        value_at_risk_95_daily_pct=round(var_95_daily * 100.0, 4),
        max_drawdown_pct=round(max_dd * 100.0, 4),
        risk_level=risk_level,
        suggested_position_size_pct=round(position_size * 100.0, 2),
        suggested_stop_loss_pct=round(-suggested_stop * 100.0, 2),
        suggested_take_profit_pct=round(suggested_take_profit * 100.0, 2),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 300)))
    print(generate_risk_report(prices).to_dict())
