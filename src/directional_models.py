# src/directional_models.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

TRADING_DAYS_PER_YEAR = 252


@dataclass
class DirectionalModelResult:
    name: str
    model: Any = None
    metrics: Dict[str, float] = field(default_factory=dict)
    probabilities_test: np.ndarray = field(default_factory=lambda: np.array([]))
    predictions_test: np.ndarray = field(default_factory=lambda: np.array([]))
    train_time_sec: float = 0.0


def direction_targets_from_scaled_y(preprocessor, y_scaled: np.ndarray) -> np.ndarray:
    """Convert scaled next-day log-return target to binary direction labels."""
    returns = preprocessor.inverse_transform_target(np.asarray(y_scaled, dtype=float).flatten())
    return (returns > 0.0).astype(int)


def _metrics(y_true: np.ndarray, pred: np.ndarray, proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).flatten()
    pred = np.asarray(pred).astype(int).flatten()

    out = {
        "Accuracy": round(float(accuracy_score(y_true, pred) * 100.0), 2),
        "Precision": round(float(precision_score(y_true, pred, zero_division=0) * 100.0), 2),
        "Recall": round(float(recall_score(y_true, pred, zero_division=0) * 100.0), 2),
        "F1": round(float(f1_score(y_true, pred, zero_division=0) * 100.0), 2),
    }

    if proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                out["AUC"] = round(float(roc_auc_score(y_true, proba) * 100.0), 2)
            else:
                out["AUC"] = np.nan
        except Exception:
            out["AUC"] = np.nan
    else:
        out["AUC"] = np.nan

    return out


def directional_baseline_leaderboard(data, preprocessor) -> pd.DataFrame:
    y_test = direction_targets_from_scaled_y(preprocessor, data.y_test)
    actual_prices = np.asarray(data.prices_test, dtype=float).flatten()
    anchors = np.concatenate([[float(data.last_price_before_test)], actual_prices[:-1]]) if len(actual_prices) else np.array([])
    actual_returns = actual_prices / anchors - 1.0 if len(anchors) else np.array([])

    rows = []

    always_up = np.ones_like(y_test)
    rows.append({"Model": "Always Up baseline", **_metrics(y_test, always_up, always_up.astype(float))})

    always_down = np.zeros_like(y_test)
    rows.append({"Model": "Always Down baseline", **_metrics(y_test, always_down, always_down.astype(float))})

    if len(actual_returns):
        prev_ret = np.concatenate([[0.0], actual_returns[:-1]])
        momentum = (prev_ret > 0).astype(int)
        rows.append({"Model": "Yesterday direction baseline", **_metrics(y_test, momentum, momentum.astype(float))})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def _safe_predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    pred = model.predict(X)
    return np.asarray(pred, dtype=float).flatten()


def _optional_models(random_state: int) -> List[tuple[str, Any]]:
    models: List[tuple[str, Any]] = []

    # Heavy libraries are optional. If installed, they are useful for a stronger
    # direction benchmark; if missing, the core sklearn classifiers still work.
    try:
        import lightgbm as lgb
        models.append((
            "LightGBM Direction",
            lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,
                random_state=random_state,
                verbosity=-1,
                class_weight="balanced",
            ),
        ))
    except Exception:
        pass

    try:
        import xgboost as xgb
        models.append((
            "XGBoost Direction",
            xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                eval_metric="logloss",
            ),
        ))
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier
        models.append((
            "CatBoost Direction",
            CatBoostClassifier(
                iterations=300,
                depth=4,
                learning_rate=0.03,
                random_seed=random_state,
                verbose=0,
                auto_class_weights="Balanced",
            ),
        ))
    except Exception:
        pass

    return models


def train_directional_models(
    data,
    preprocessor,
    *,
    include_heavy: bool = True,
    random_state: int = 42,
) -> Dict[str, DirectionalModelResult]:
    """
    Train separate Up/Down classifiers using the same preprocessed features.

    Regression models optimize RMSE. These classifiers optimize direction.
    A useful trading model should beat simple baselines on direction accuracy,
    F1/AUC, and probability-based backtesting.
    """
    y_train = direction_targets_from_scaled_y(preprocessor, data.y_train)
    y_test = direction_targets_from_scaled_y(preprocessor, data.y_test)

    base_models: List[tuple[str, Any]] = [
        (
            "Logistic Direction",
            make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            ),
        ),
        (
            "RandomForest Direction",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "GradientBoost Direction",
            GradientBoostingClassifier(random_state=random_state),
        ),
    ]

    if include_heavy:
        base_models.extend(_optional_models(random_state))

    results: Dict[str, DirectionalModelResult] = {}
    for name, model in base_models:
        try:
            t0 = time.perf_counter()
            model.fit(data.X_train, y_train)
            train_time = time.perf_counter() - t0

            proba = _safe_predict_proba(model, data.X_test)
            pred = (proba >= 0.5).astype(int)

            results[name] = DirectionalModelResult(
                name=name,
                model=model,
                metrics=_metrics(y_test, pred, proba),
                probabilities_test=proba,
                predictions_test=pred,
                train_time_sec=float(train_time),
            )
        except Exception as exc:
            # Keep the dashboard robust: one failed optional classifier should
            # not break the whole directional page.
            results[name] = DirectionalModelResult(
                name=name,
                model=None,
                metrics={"Error": str(exc)},
                probabilities_test=np.array([]),
                predictions_test=np.array([]),
                train_time_sec=0.0,
            )

    return results


def directional_leaderboard(results: Dict[str, DirectionalModelResult]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        if not res.metrics or "Error" in res.metrics:
            row = {"Model": name, "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan, "AUC": np.nan, "TrainTime(s)": res.train_time_sec}
            if "Error" in res.metrics:
                row["Error"] = res.metrics["Error"]
        else:
            row = {"Model": name, **res.metrics, "TrainTime(s)": round(res.train_time_sec, 4)}
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and "Accuracy" in df.columns:
        df = df.sort_values(["Accuracy", "F1"], ascending=False, na_position="last").reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def _calculate_drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _sharpe_ratio(returns: pd.Series) -> float:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / std)


def _annualized_return(total_return: float, periods: int) -> float:
    if periods <= 0 or total_return <= -1:
        return -1.0 if total_return <= -1 else 0.0
    return float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / periods) - 1.0)


def run_directional_probability_backtest(
    data,
    probabilities_up: np.ndarray,
    *,
    probability_threshold: float = 0.55,
    transaction_cost: float = 0.001,
    allow_short: bool = False,
) -> tuple[Dict[str, float], pd.DataFrame]:
    """
    Backtest from Up probability instead of regression-predicted price.

    Long-only:
        probability_up >= threshold -> long, otherwise cash.

    Long/short:
        probability_up >= threshold -> long
        probability_up <= 1-threshold -> short
        otherwise cash
    """
    actual_prices = np.asarray(data.prices_test, dtype=float).flatten()
    proba = np.asarray(probabilities_up, dtype=float).flatten()

    if len(proba) != len(actual_prices):
        raise ValueError("Probability vector and prices_test length do not match.")

    anchors = np.concatenate([[float(data.last_price_before_test)], actual_prices[:-1]])
    realized_return = actual_prices / anchors - 1.0

    if allow_short:
        position = np.where(proba >= probability_threshold, 1, np.where(proba <= (1.0 - probability_threshold), -1, 0))
    else:
        position = np.where(proba >= probability_threshold, 1, 0)

    position = position.astype(float)
    position_change = np.abs(np.diff(position, prepend=0.0))
    costs = position_change * transaction_cost
    strategy_return = position * realized_return - costs

    equity = pd.DataFrame(
        {
            "anchor_price": anchors,
            "actual_next_price": actual_prices,
            "probability_up": proba,
            "realized_return": realized_return,
            "position": position,
            "strategy_return": strategy_return,
        },
        index=data.test_index,
    )
    equity["strategy_equity"] = (1.0 + equity["strategy_return"]).cumprod()
    equity["buy_hold_equity"] = (1.0 + equity["realized_return"]).cumprod()
    equity["strategy_drawdown"] = _calculate_drawdown(equity["strategy_equity"])
    equity["buy_hold_drawdown"] = _calculate_drawdown(equity["buy_hold_equity"])

    total_return = float(equity["strategy_equity"].iloc[-1] - 1.0) if len(equity) else 0.0
    buy_hold_return = float(equity["buy_hold_equity"].iloc[-1] - 1.0) if len(equity) else 0.0

    active = equity["position"] != 0
    if int(active.sum()) > 0:
        win_rate = float((equity.loc[active, "strategy_return"] > 0).mean())
    else:
        win_rate = 0.0

    trades = int((equity["position"].diff().fillna(equity["position"]).abs() > 0).sum())

    metrics = {
        "total_return_pct": round(total_return * 100.0, 4),
        "annualized_return_pct": round(_annualized_return(total_return, len(equity)) * 100.0, 4),
        "sharpe_ratio": round(_sharpe_ratio(equity["strategy_return"]), 4),
        "max_drawdown_pct": round(float(equity["strategy_drawdown"].min()) * 100.0, 4),
        "win_rate_pct": round(win_rate * 100.0, 4),
        "number_of_trades": float(trades),
        "buy_hold_return_pct": round(buy_hold_return * 100.0, 4),
        "strategy_minus_buy_hold_pct": round((total_return - buy_hold_return) * 100.0, 4),
        "probability_threshold_pct": round(probability_threshold * 100.0, 2),
        "transaction_cost_pct": round(transaction_cost * 100.0, 4),
    }

    return metrics, equity
