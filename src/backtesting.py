# src/backtesting.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame


def _validate_dataframe(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")

    clean = df.copy()

    if not isinstance(clean.index, pd.DatetimeIndex):
        if "Date" in clean.columns:
            clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
            clean = clean.set_index("Date")
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a Date column.")

    clean = clean.sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]

    clean[price_col] = pd.to_numeric(clean[price_col], errors="coerce")
    clean = clean.dropna(subset=[price_col])

    if clean.empty:
        raise ValueError("No valid price data after cleaning.")

    return clean


def _calculate_drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _annualized_return(total_return: float, periods: int) -> float:
    if periods <= 0:
        return 0.0

    if total_return <= -1:
        return -1.0

    return float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / periods) - 1.0)


def _sharpe_ratio(returns: pd.Series) -> float:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) < 2:
        return 0.0

    std = returns.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return 0.0

    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / std)


def _build_trades_table(bt: pd.DataFrame) -> pd.DataFrame:
    entries = bt.index[(bt["position"].diff().fillna(bt["position"]) > 0)]
    exits = bt.index[(bt["position"].diff().fillna(0) < 0)]

    rows = []
    exits_list = list(exits)

    for entry_date in entries:
        exit_date = None

        for candidate_exit in exits_list:
            if candidate_exit > entry_date:
                exit_date = candidate_exit
                break

        if exit_date is None:
            exit_date = bt.index[-1]

        trade_slice = bt.loc[entry_date:exit_date]

        if trade_slice.empty:
            continue

        trade_return = float((1.0 + trade_slice["strategy_return"]).prod() - 1.0)

        rows.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "days_held": int(len(trade_slice)),
                "entry_price": float(bt.loc[entry_date, "price"]),
                "exit_price": float(bt.loc[exit_date, "price"]),
                "trade_return_pct": trade_return * 100.0,
                "win": trade_return > 0,
            }
        )

    return pd.DataFrame(rows)


def run_backtest_from_predictions(
    df: pd.DataFrame,
    *,
    price_col: str,
    predicted_price_col: Optional[str] = None,
    predicted_return_col: Optional[str] = None,
    threshold: float = 0.002,
    transaction_cost: float = 0.001,
    allow_short: bool = False,
) -> BacktestResult:
    """
    Simple long/cash backtest.

    Signal:
        If predicted_return > threshold, position = 1.
        Else position = 0.

    Alignment:
        This assumes each row's prediction is for the NEXT trading period.
        Today's features -> predicted next-day price/return.
        Strategy position is applied to next period's realized return.

    threshold:
        Decimal form. 0.002 = 0.2%.

    transaction_cost:
        Decimal form. 0.001 = 0.1% per position change.

    allow_short:
        Optional. If True, predicted_return < -threshold opens short position.
    """

    if predicted_price_col is None and predicted_return_col is None:
        raise ValueError("Provide either predicted_price_col or predicted_return_col.")

    if threshold < 0:
        raise ValueError("threshold cannot be negative.")

    if transaction_cost < 0:
        raise ValueError("transaction_cost cannot be negative.")

    clean = _validate_dataframe(df, price_col=price_col)

    bt = pd.DataFrame(index=clean.index)
    bt["price"] = clean[price_col].astype(float)

    if predicted_return_col is not None:
        if predicted_return_col not in clean.columns:
            raise ValueError(f"Missing predicted return column: {predicted_return_col}")

        pred_ret = pd.to_numeric(clean[predicted_return_col], errors="coerce")

        # If user accidentally passes percentage values like 1.2 instead of 0.012,
        # convert when values look too large.
        if pred_ret.dropna().abs().median() > 0.5:
            pred_ret = pred_ret / 100.0

        bt["predicted_return"] = pred_ret

    else:
        if predicted_price_col not in clean.columns:
            raise ValueError(f"Missing predicted price column: {predicted_price_col}")

        pred_price = pd.to_numeric(clean[predicted_price_col], errors="coerce")
        bt["predicted_return"] = pred_price / bt["price"] - 1.0

    if "Actual_Next_Return" in clean.columns:
        bt["realized_return"] = pd.to_numeric(clean["Actual_Next_Return"], errors="coerce")
    else:
        bt["realized_return"] = bt["price"].pct_change().shift(-1)

    bt = bt.replace([np.inf, -np.inf], np.nan)
    bt = bt.dropna(subset=["predicted_return", "realized_return"])

    if bt.empty:
        raise ValueError("No valid rows after aligning predictions and realized returns.")

    if allow_short:
        bt["position"] = np.where(
            bt["predicted_return"] > threshold,
            1,
            np.where(bt["predicted_return"] < -threshold, -1, 0),
        )
    else:
        bt["position"] = np.where(bt["predicted_return"] > threshold, 1, 0)

    bt["position_change"] = bt["position"].diff().abs()
    bt.loc[bt.index[0], "position_change"] = abs(bt.loc[bt.index[0], "position"])

    bt["cost"] = bt["position_change"] * transaction_cost
    bt["strategy_return"] = bt["position"] * bt["realized_return"] - bt["cost"]

    bt["strategy_equity"] = (1.0 + bt["strategy_return"]).cumprod()
    bt["buy_hold_equity"] = (1.0 + bt["realized_return"]).cumprod()

    bt["strategy_drawdown"] = _calculate_drawdown(bt["strategy_equity"])
    bt["buy_hold_drawdown"] = _calculate_drawdown(bt["buy_hold_equity"])

    total_return = float(bt["strategy_equity"].iloc[-1] - 1.0)
    buy_hold_return = float(bt["buy_hold_equity"].iloc[-1] - 1.0)

    trades = _build_trades_table(bt)
    number_of_trades = int(len(trades))

    if number_of_trades > 0:
        win_rate = float(trades["win"].mean())
    else:
        win_rate = 0.0

    metrics = {
        "total_return_pct": round(total_return * 100.0, 4),
        "annualized_return_pct": round(_annualized_return(total_return, len(bt)) * 100.0, 4),
        "sharpe_ratio": round(_sharpe_ratio(bt["strategy_return"]), 4),
        "max_drawdown_pct": round(float(bt["strategy_drawdown"].min()) * 100.0, 4),
        "win_rate_pct": round(win_rate * 100.0, 4),
        "number_of_trades": float(number_of_trades),
        "buy_hold_return_pct": round(buy_hold_return * 100.0, 4),
        "strategy_minus_buy_hold_pct": round((total_return - buy_hold_return) * 100.0, 4),
        "transaction_cost_pct": round(transaction_cost * 100.0, 4),
        "threshold_pct": round(threshold * 100.0, 4),
    }

    equity_curve = bt[
        [
            "price",
            "predicted_return",
            "realized_return",
            "position",
            "strategy_return",
            "strategy_equity",
            "buy_hold_equity",
            "strategy_drawdown",
            "buy_hold_drawdown",
        ]
    ].copy()

    return BacktestResult(
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades,
    )


if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=300, freq="B")

    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.01, size=len(dates))
    prices = 2000 * np.cumprod(1 + returns)

    predicted_returns = returns + rng.normal(0, 0.008, size=len(dates))

    demo = pd.DataFrame(
        {
            "Gold_Close": prices,
            "Predicted_Return": predicted_returns,
        },
        index=dates,
    )

    result = run_backtest_from_predictions(
        demo,
        price_col="Gold_Close",
        predicted_return_col="Predicted_Return",
        threshold=0.002,
        transaction_cost=0.001,
    )

    print("Backtest metrics:")
    for key, value in result.metrics.items():
        print(f"{key}: {value}")

    print("\nEquity curve:")
    print(result.equity_curve.tail())

    print("\nTrades:")
    print(result.trades.tail())
