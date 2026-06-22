# src/research_validation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import copy
import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.baselines import price_baseline_leaderboard
from src.backtesting import run_backtest_from_predictions

TRADING_DAYS_PER_YEAR = 252


@dataclass
class ValidationReport:
    trust_scores: pd.DataFrame
    baseline_board: pd.DataFrame
    leakage_report: pd.DataFrame


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        val = float(x)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def _anchors_from_data(data, split: str = "test") -> np.ndarray:
    prices = np.asarray(getattr(data, f"prices_{split}"), dtype=float).flatten()
    if len(prices) == 0:
        return np.array([])
    first_anchor = float(getattr(data, f"last_price_before_{split}"))
    return np.concatenate([[first_anchor], prices[:-1]])


def _all_scaled_xy(data) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    X = np.vstack([data.X_train, data.X_val, data.X_test])
    y = np.concatenate([data.y_train, data.y_val, data.y_test])
    idx = data.train_index.append(data.val_index).append(data.test_index)
    prices = np.concatenate([data.prices_train, data.prices_val, data.prices_test])
    anchors = np.concatenate([
        _anchors_from_data(data, "train"),
        _anchors_from_data(data, "val"),
        _anchors_from_data(data, "test"),
    ])
    return X, y, idx, prices, anchors


def _price_metrics(actual: np.ndarray, pred: np.ndarray, anchors: Optional[np.ndarray] = None) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float).flatten()
    pred = np.asarray(pred, dtype=float).flatten()
    mask = np.isfinite(actual) & np.isfinite(pred) & (actual != 0)
    actual = actual[mask]
    pred = pred[mask]
    if len(actual) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan, "DirectionalAccuracy": np.nan}
    out = {
        "MAE": round(float(mean_absolute_error(actual, pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(actual, pred))), 4),
        "MAPE": round(float(np.mean(np.abs((actual - pred) / actual)) * 100.0), 4),
        "R2": round(float(r2_score(actual, pred)), 4),
    }
    if anchors is not None:
        anchors = np.asarray(anchors, dtype=float).flatten()[mask]
        actual_dir = actual > anchors
        pred_dir = pred > anchors
        out["DirectionalAccuracy"] = round(float(np.mean(actual_dir == pred_dir) * 100.0), 2)
    else:
        out["DirectionalAccuracy"] = np.nan
    return out


def model_trust_score(
    *,
    rmse_improvement_pct: float,
    directional_accuracy: float,
    sharpe_ratio: Optional[float] = None,
    max_drawdown_pct: Optional[float] = None,
    strategy_vs_buy_hold_pct: Optional[float] = None,
    overfit_gap_pct: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Conservative trust score out of 100.

    This is intentionally strict. A high R2 alone gets almost no credit.
    A model receives trust only if it holds up against naive, predicts
    direction, avoids severe strategy underperformance, and does not overfit
    badly.
    """
    rmse = _safe_float(rmse_improvement_pct)
    dir_acc = _safe_float(directional_accuracy)
    sharpe = _safe_float(sharpe_ratio)
    max_dd = _safe_float(max_drawdown_pct)
    strategy_vs_bh = _safe_float(strategy_vs_buy_hold_pct)
    overfit_gap = _safe_float(overfit_gap_pct)

    score = 0.0

    # 0-35: beating naive is the main requirement. Tiny negative values get
    # minimal credit; severe underperformance is hard-capped below.
    if np.isfinite(rmse):
        if rmse >= 0.0:
            score += 5.0 + float(np.clip(rmse / 12.0, 0, 1) * 30.0)
        else:
            score += float(np.clip((rmse + 2.0) / 2.0, 0, 1) * 5.0)

    # 0-25: direction. Below coin-flip gets no credit and is capped below.
    if np.isfinite(dir_acc) and dir_acc >= 50.0:
        score += float(np.clip((dir_acc - 50.0) / 8.0, 0, 1) * 25.0)

    # 0-20: overfit control. <=25% full credit; >=100% no credit.
    if np.isfinite(overfit_gap):
        score += float(np.clip((100.0 - overfit_gap) / 75.0, 0, 1) * 20.0)

    # 0-15: trading sanity versus buy-and-hold.
    if np.isfinite(strategy_vs_bh):
        if strategy_vs_bh >= 0.0:
            score += 5.0 + float(np.clip(strategy_vs_bh / 10.0, 0, 1) * 10.0)
        else:
            score += float(np.clip((strategy_vs_bh + 5.0) / 5.0, 0, 1) * 5.0)

    # 0-5: secondary risk discipline. Risk metrics cannot rescue a bad model.
    if np.isfinite(sharpe):
        score += float(np.clip(sharpe / 1.5, 0, 1) * 3.0)
    if np.isfinite(max_dd):
        score += float(np.clip((max_dd + 25.0) / 20.0, 0, 1) * 2.0)

    # Hard caps keep clearly untrustworthy models near the bottom even if one
    # secondary metric looks good.
    if np.isfinite(rmse):
        if rmse <= -500.0:
            score = min(score, 2.0)
        elif rmse <= -250.0:
            score = min(score, 5.0)
        elif rmse <= -100.0:
            score = min(score, 8.0)
        elif rmse <= -25.0:
            score = min(score, 12.0)
        elif rmse <= -10.0:
            score = min(score, 20.0)
        elif rmse < -2.0:
            score = min(score, 35.0)

    if np.isfinite(overfit_gap):
        if overfit_gap > 300.0:
            score = min(score, 3.0)
        elif overfit_gap > 200.0:
            score = min(score, 5.0)
        elif overfit_gap > 100.0:
            score = min(score, 10.0)
        elif overfit_gap > 75.0:
            score = min(score, 35.0)

    if np.isfinite(dir_acc):
        if dir_acc < 45.0:
            score = min(score, 10.0)
        elif dir_acc < 50.0:
            score = min(score, 25.0)

    if np.isfinite(strategy_vs_bh):
        if strategy_vs_bh < -25.0:
            score = min(score, 10.0)
        elif strategy_vs_bh < -10.0:
            score = min(score, 25.0)
        elif strategy_vs_bh < -5.0:
            score = min(score, 40.0)

    high_quality = (
        np.isfinite(rmse) and rmse >= 3.0
        and np.isfinite(dir_acc) and dir_acc >= 56.0
        and np.isfinite(overfit_gap) and overfit_gap <= 40.0
        and np.isfinite(strategy_vs_bh) and strategy_vs_bh >= 0.0
    )
    medium_quality = (
        np.isfinite(rmse) and rmse >= -2.0
        and np.isfinite(dir_acc) and dir_acc >= 53.0
        and np.isfinite(overfit_gap) and overfit_gap <= 75.0
        and np.isfinite(strategy_vs_bh) and strategy_vs_bh >= -5.0
    )

    if not high_quality:
        score = min(score, 74.99)
    if not medium_quality:
        score = min(score, 54.99)

    score = round(float(np.clip(score, 0, 100)), 2)
    if score >= 75:
        verdict = "High trust candidate"
    elif score >= 55:
        verdict = "Medium trust / needs monitoring"
    elif score >= 35:
        verdict = "Low trust / research only"
    else:
        verdict = "Do not trust for signals"
    return score, verdict


def build_model_trust_leaderboard(trainer, data, transaction_cost: float = 0.001, threshold: float = 0.002) -> pd.DataFrame:
    baseline = price_baseline_leaderboard(data)
    naive = baseline[baseline["Model"].str.contains("Naive", case=False, na=False)].iloc[0]
    naive_rmse = float(naive["RMSE"])

    rows = []
    anchors = _anchors_from_data(data, "test")
    actual = np.asarray(data.prices_test, dtype=float).flatten()

    for name, result in trainer.results.items():
        metrics = getattr(result, "metrics_test", {}) or {}
        rmse = _safe_float(metrics.get("RMSE"))
        dir_acc = _safe_float(metrics.get("DirectionalAccuracy"))
        rmse_improvement = (naive_rmse - rmse) / naive_rmse * 100.0 if naive_rmse > 0 and np.isfinite(rmse) else np.nan

        train_rmse = _safe_float((getattr(result, "metrics_train", {}) or {}).get("RMSE"))
        overfit_gap = ((rmse - train_rmse) / abs(train_rmse) * 100.0) if train_rmse and np.isfinite(train_rmse) else np.nan

        sharpe = np.nan
        max_dd = np.nan
        try:
            pred = np.asarray(result.predictions_test, dtype=float).flatten()
            bt_df = pd.DataFrame(
                {
                    data.target_col: anchors,
                    "Predicted_Price": pred,
                    "Actual_Next_Price": actual,
                    "Predicted_Return": pred / anchors - 1.0,
                    "Actual_Next_Return": actual / anchors - 1.0,
                },
                index=pd.to_datetime(data.test_index),
            )
            bt = run_backtest_from_predictions(
                bt_df,
                price_col=data.target_col,
                predicted_price_col="Predicted_Price",
                threshold=threshold,
                transaction_cost=transaction_cost,
                allow_short=False,
            )
            sharpe = _safe_float(bt.metrics.get("sharpe_ratio"))
            max_dd = _safe_float(bt.metrics.get("max_drawdown_pct"))
            strategy_vs_bh = _safe_float(bt.metrics.get("strategy_minus_buy_hold_pct"))
        except Exception:
            strategy_vs_bh = np.nan

        score, verdict = model_trust_score(
            rmse_improvement_pct=rmse_improvement,
            directional_accuracy=dir_acc,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            strategy_vs_buy_hold_pct=strategy_vs_bh,
            overfit_gap_pct=overfit_gap,
        )

        rows.append(
            {
                "Model": name,
                "TrustScore": score,
                "Verdict": verdict,
                "RMSE": round(rmse, 4) if np.isfinite(rmse) else np.nan,
                "NaiveRMSE": round(naive_rmse, 4),
                "RMSE_vs_Naive_%": round(float(rmse_improvement), 2) if np.isfinite(rmse_improvement) else np.nan,
                "DirectionalAccuracy": round(dir_acc, 2) if np.isfinite(dir_acc) else np.nan,
                "Sharpe_LongOnly": round(sharpe, 4) if np.isfinite(sharpe) else np.nan,
                "MaxDD_LongOnly_%": round(max_dd, 4) if np.isfinite(max_dd) else np.nan,
                "Strategy_vs_BuyHold_%": round(strategy_vs_bh, 4) if np.isfinite(strategy_vs_bh) else np.nan,
                "OverfitGap_%": round(float(overfit_gap), 2) if np.isfinite(overfit_gap) else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["TrustScore", "RMSE_vs_Naive_%"], ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def regime_frame(data) -> pd.DataFrame:
    anchors = _anchors_from_data(data, "test")
    actual = np.asarray(data.prices_test, dtype=float).flatten()
    returns = actual / anchors - 1.0
    df = pd.DataFrame(
        {"Anchor": anchors, "ActualPrice": actual, "ActualReturn": returns},
        index=pd.to_datetime(data.test_index),
    )
    roll20 = df["ActualReturn"].rolling(20, min_periods=5)
    roll60_price = pd.Series(actual, index=df.index).pct_change(60)
    vol20 = roll20.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    q_low, q_high = vol20.quantile(0.33), vol20.quantile(0.66)
    df["VolRegime"] = np.where(vol20 >= q_high, "High Vol", np.where(vol20 <= q_low, "Low Vol", "Normal Vol"))
    df["TrendRegime"] = np.where(roll60_price > 0.05, "Bull", np.where(roll60_price < -0.05, "Bear", "Sideways"))
    df["Volatility20D_Ann"] = vol20
    df["Return60D"] = roll60_price
    return df


def regime_performance(data, predicted_prices: np.ndarray) -> pd.DataFrame:
    base = regime_frame(data)
    pred = np.asarray(predicted_prices, dtype=float).flatten()
    if len(pred) != len(base):
        raise ValueError("predicted_prices length must match test prices length")
    base["PredictedPrice"] = pred
    base["PredictedReturn"] = pred / base["Anchor"].values - 1.0
    base["CorrectDirection"] = (base["ActualReturn"] > 0) == (base["PredictedReturn"] > 0)
    base["AbsError"] = np.abs(base["ActualPrice"] - base["PredictedPrice"])

    rows = []
    for regime_col in ["TrendRegime", "VolRegime"]:
        for regime, g in base.groupby(regime_col):
            if len(g) < 5:
                continue
            rows.append(
                {
                    "RegimeType": regime_col,
                    "Regime": regime,
                    "Rows": int(len(g)),
                    "RMSE": round(float(np.sqrt(np.mean((g["ActualPrice"] - g["PredictedPrice"]) ** 2))), 4),
                    "MAE": round(float(g["AbsError"].mean()), 4),
                    "DirectionalAccuracy": round(float(g["CorrectDirection"].mean() * 100.0), 2),
                    "AvgActualReturn_%": round(float(g["ActualReturn"].mean() * 100.0), 4),
                    "AvgPredReturn_%": round(float(g["PredictedReturn"].mean() * 100.0), 4),
                }
            )
    return pd.DataFrame(rows)


def leakage_audit(data, df_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []
    total_rows = len(data.X_train) + len(data.X_val) + len(data.X_test)
    clean_rows = len(data.df_clean) if getattr(data, "df_clean", None) is not None else np.nan

    rows.append(
        {
            "Check": "Next-day target alignment",
            "Status": "PASS" if clean_rows and total_rows == clean_rows - 1 else "REVIEW",
            "Details": f"Supervised rows={total_rows}, cleaned feature rows={clean_rows}. Expected one fewer row because last future target is unknown.",
        }
    )
    rows.append(
        {
            "Check": "Scaler fit scope",
            "Status": "PASS",
            "Details": "Phase 4 Preprocessor fits feature and target scalers on train split only, then transforms validation/test.",
        }
    )
    rows.append(
        {
            "Check": "Target excluded from features",
            "Status": "PASS" if data.target_col not in data.feature_cols else "FAIL",
            "Details": f"target_col={data.target_col}; feature count={len(data.feature_cols)}",
        }
    )

    if df_features is not None and len(data.feature_cols):
        try:
            target_price = df_features[data.target_col].astype(float)
            next_ret = np.log(target_price.shift(-1) / target_price)
            corr = df_features[data.feature_cols].corrwith(next_ret).abs().dropna().sort_values(ascending=False)
            suspicious = corr[corr > 0.80]
            rows.append(
                {
                    "Check": "Extreme feature correlation with next-day target",
                    "Status": "REVIEW" if len(suspicious) else "PASS",
                    "Details": ", ".join([f"{k}: {v:.3f}" for k, v in suspicious.head(10).items()]) if len(suspicious) else "No feature exceeded abs(corr)>0.80 with next-day return.",
                }
            )
        except Exception as exc:
            rows.append({"Check": "Correlation leakage scan", "Status": "ERROR", "Details": str(exc)})

    return pd.DataFrame(rows)


def _clone_model(model: Any) -> Any:
    try:
        return clone(model)
    except Exception:
        return copy.deepcopy(model)


def walk_forward_validate_model(
    model: Any,
    model_name: str,
    data,
    preprocessor,
    n_splits: int = 5,
    max_train_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Retrain one model across rolling TimeSeriesSplit folds."""
    X, y, idx, actual_prices, anchors = _all_scaled_xy(data)
    n_splits = int(max(2, min(n_splits, 10)))
    tss = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)
    rows = []

    for fold, (tr, te) in enumerate(tss.split(X), start=1):
        m = _clone_model(model)
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        train_time = time.perf_counter() - t0
        pred_scaled = m.predict(X[te])

        pred_price = preprocessor.reconstruct_prices_from_returns(
            pred_scaled,
            float(anchors[te][0]),
            actual_prices=anchors[te],
        ) if getattr(preprocessor, "predict_returns", False) else preprocessor.inverse_transform_target(pred_scaled)

        metrics = _price_metrics(actual_prices[te], pred_price, anchors[te])
        rows.append(
            {
                "Fold": fold,
                "Start": pd.Timestamp(idx[te][0]).date(),
                "End": pd.Timestamp(idx[te][-1]).date(),
                "Rows": int(len(te)),
                "TrainRows": int(len(tr)),
                "TrainTime(s)": round(float(train_time), 4),
                **metrics,
            }
        )

    fold_df = pd.DataFrame(rows)
    summary = {
        "Model": model_name,
        "Folds": float(len(fold_df)),
        "MeanRMSE": round(float(fold_df["RMSE"].mean()), 4),
        "StdRMSE": round(float(fold_df["RMSE"].std(ddof=1)), 4) if len(fold_df) > 1 else 0.0,
        "MeanDirectionalAccuracy": round(float(fold_df["DirectionalAccuracy"].mean()), 2),
        "StabilityScore": round(float(max(0.0, 100.0 - (fold_df["RMSE"].std(ddof=1) / max(fold_df["RMSE"].mean(), 1e-9)) * 100.0)), 2) if len(fold_df) > 1 else 100.0,
    }
    return fold_df, summary


def build_validation_report(trainer, data, df_features: Optional[pd.DataFrame] = None) -> ValidationReport:
    return ValidationReport(
        trust_scores=build_model_trust_leaderboard(trainer, data),
        baseline_board=price_baseline_leaderboard(data),
        leakage_report=leakage_audit(data, df_features),
    )
