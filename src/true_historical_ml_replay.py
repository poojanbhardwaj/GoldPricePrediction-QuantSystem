"""Phase 20 true historical walk-forward ML replay.

Every replay prediction is trained from labels that matured before the
prediction date. Scaling, imputation, and model fitting are repeated inside
each historical window. This module is research-only and never enables real
capital.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.artifact_store import save_phase_artifacts
from src.asset_config import get_asset_names, get_target_column


TRUE_HISTORICAL_ML_REPLAY_PHASE_NAME = "phase20_true_historical_ml_replay"
TRUE_ML_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)
MODEL_CHOICES: Tuple[str, ...] = ("Ridge", "Linear", "RandomForest", "HistGradientBoosting", "SafeMean")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
FUTURE_TARGET_PREFIXES: Tuple[str, ...] = ("future_return_", "future_direction_", "future_realized_vol_")
EPSILON = 1e-12

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "ReplayType", "ProxyReplayStillUsed", "TotalWindows", "TotalPredictions",
    "MaturedPredictions", "PendingPredictions", "LeakagePassRate", "ModelsTested",
    "AssetsCovered", "HorizonsCovered", "BestAsset", "BestHorizon", "BestModel",
    "BestNetReturnPct", "BeatsBestBaselineCount", "RealCapitalStatus", "FinalVerdict",
    "MainReason", "MainLimitation",
)

PREDICTION_LOG_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "WindowId", "TrainStartDate", "TrainEndDate",
    "PredictionDate", "TargetOutcomeDate", "ModelName", "PredictedReturnPct",
    "RealizedReturnPct", "NetRealizedReturnPct", "SignalLabel", "ConfidenceProxy",
    "OutcomeStatus", "CostBps", "SlippageBps", "IsMatured", "LeakagePassed",
)

PERFORMANCE_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "ModelName", "TotalReturnPct", "AnnualizedReturnPct",
    "MaxDrawdownPct", "HitRatePct", "AverageWinPct", "AverageLossPct",
    "WinLossRatio", "SharpeLike", "SortinoLike", "CalmarLike", "Turnover",
    "CostDragPct", "MaturedTrades", "PendingTrades",
)

BASELINE_COMPARISON_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "ModelName", "MLReturnPct", "BestBaselineName",
    "BestBaselineReturnPct", "BeatsBestBaseline", "GapPct", "DominanceVerdict",
)

STRENGTH_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "ModelName", "StrengthClassification", "MaturedTrades",
    "MLReturnPct", "BestBaselineReturnPct", "GapPct", "LeakagePassed",
    "CostFragile", "RealCapitalStatus", "ResearchStatus", "MainReason",
)

LEAKAGE_AUDIT_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "WindowId", "ModelName", "TrainStartDate", "TrainEndDate",
    "ValidationEndDate", "PredictionDate", "TargetOutcomeDate",
    "TrainEndBeforePrediction", "PredictionBeforeTargetOutcome", "ScalerFitTrainOnly",
    "NoTargetLeakage", "NoFutureRowsUsed", "FutureTargetColumnsExcluded",
    "FeatureCount", "FeatureColumns", "LeakagePassed", "Explanation",
)

INPUT_SOURCE_COLUMNS: Tuple[str, ...] = (
    "SourceName", "Available", "Rows", "Columns", "FirstDate", "LastDate",
    "AssetsRequested", "MissingCriticalColumns", "Notes",
)

QUALITY_GATE_COLUMNS: Tuple[str, ...] = (
    "GateName", "Passed", "Severity", "AffectedRows", "Explanation",
)

NEXT_ACTION_COLUMNS: Tuple[str, ...] = (
    "Rank", "Action", "WhyItMatters", "AffectedAssets", "AffectedHorizons",
    "ExpectedBenefit", "Urgency", "DependsOn",
)


@dataclass
class TrueHistoricalMLReplayReport:
    true_ml_summary_table: pd.DataFrame
    true_ml_prediction_log: pd.DataFrame
    true_ml_performance_table: pd.DataFrame
    true_ml_baseline_comparison_table: pd.DataFrame
    true_ml_strength_table: pd.DataFrame
    leakage_audit_table: pd.DataFrame
    input_sources_table: pd.DataFrame
    quality_gates_table: pd.DataFrame
    next_actions_table: pd.DataFrame
    settings: Dict[str, Any] = field(default_factory=dict)
    saved_artifacts: Dict[str, Any] = field(default_factory=dict)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _to_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value)


def _prepare_market_data(market_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = _to_frame(market_data)
    if df.empty:
        return df
    date_col = next((col for col in ("Date", "date", "Datetime", "Timestamp") if col in df.columns), None)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(date_col, keep="last").set_index(date_col)
    else:
        parsed = pd.to_datetime(df.index, errors="coerce")
        df.index = parsed
        df = df[~df.index.isna()].sort_index()
        df = df[~df.index.duplicated(keep="last")]
    return df


def _load_project_market_data() -> Optional[pd.DataFrame]:
    if not DEFAULT_MARKET_DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DEFAULT_MARKET_DATA_PATH)
    except Exception:
        return None


def _feature_frame(price: pd.Series, horizon: int) -> Tuple[pd.DataFrame, List[str], str]:
    price = pd.to_numeric(price, errors="coerce").where(lambda values: values > 0)
    log_price = np.log(price)
    log_return = log_price.diff()
    frame = pd.DataFrame(index=price.index)
    frame["log_return_1"] = log_return
    for lag in (1, 2, 3, 5, 10, 20):
        frame[f"log_return_lag_{lag}"] = log_return.shift(lag)
    for window in (5, 10, 20, 60):
        frame[f"rolling_return_{window}"] = log_price - log_price.shift(window)
        frame[f"rolling_mean_{window}"] = log_return.rolling(window, min_periods=window).mean()
        frame[f"rolling_volatility_{window}"] = log_return.rolling(window, min_periods=window).std()
        moving_average = price.rolling(window, min_periods=window).mean()
        frame[f"price_to_ma_{window}"] = price / moving_average - 1.0
    frame["rolling_drawdown_60"] = price / price.rolling(60, min_periods=20).max() - 1.0
    target_col = f"future_return_{int(horizon)}d"
    frame[target_col] = log_price.shift(-int(horizon)) - log_price
    frame["_price"] = price
    frame["_position"] = np.arange(len(frame), dtype=int)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    feature_cols = [
        col for col in frame.columns
        if col not in {target_col, "_price", "_position"}
        and not str(col).startswith(FUTURE_TARGET_PREFIXES)
    ]
    return frame, feature_cols, target_col


def _window_positions(frame: pd.DataFrame, feature_cols: Sequence[str], horizon: int, min_train_rows: int, step_size: int, max_windows: int) -> List[int]:
    valid_features = frame[list(feature_cols)].notna().all(axis=1)
    valid_target = frame[f"future_return_{int(horizon)}d"].notna()
    positions: List[int] = []
    for prediction_position in range(0, len(frame), max(1, int(step_size))):
        past_limit = prediction_position - int(horizon) - 1
        if past_limit < 0 or not bool(valid_features.iloc[prediction_position]):
            continue
        available = valid_features & valid_target & frame["_position"].le(past_limit)
        validation_rows = max(5, min(30, int(available.sum()) // 5))
        if int(available.sum()) - validation_rows >= int(min_train_rows):
            positions.append(prediction_position)

    final_position = len(frame) - 1
    if final_position >= 0 and bool(valid_features.iloc[final_position]):
        past_limit = final_position - int(horizon) - 1
        available = valid_features & valid_target & frame["_position"].le(past_limit)
        validation_rows = max(5, min(30, int(available.sum()) // 5))
        if int(available.sum()) - validation_rows >= int(min_train_rows):
            positions.append(final_position)

    positions = sorted(set(positions))
    if int(max_windows) > 0 and len(positions) > int(max_windows):
        selected = np.linspace(0, len(positions) - 1, int(max_windows)).round().astype(int)
        positions = [positions[i] for i in sorted(set(selected.tolist()))]
    return positions


def _scale_from_train(x_train: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > EPSILON), std, 1.0)

    def transform(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        values = np.where(np.isfinite(values), values, mean)
        return (values - mean) / std

    return tuple(transform(values) for values in (x_train,) + others)


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_values: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_train)), x_train])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_train
    values_design = np.column_stack([np.ones(len(x_values)), x_values])
    return values_design @ coefficients


def _fit_predict_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    x_prediction: np.ndarray,
    random_seed: int,
) -> Tuple[float, np.ndarray, str]:
    x_train_scaled, x_validation_scaled, x_prediction_scaled = _scale_from_train(
        x_train, x_validation, x_prediction
    )
    requested = str(model_name or "Ridge").strip()
    normalized = requested.lower().replace(" ", "")
    if normalized in {"safemean", "mean", "fallback"}:
        prediction = float(np.mean(y_train))
        return prediction, np.full(len(x_validation_scaled), prediction), "SafeMean"
    if normalized in {"linear", "linearregression"}:
        validation_prediction = _ridge_predict(x_train_scaled, y_train, x_validation_scaled, 0.0)
        prediction = _ridge_predict(x_train_scaled, y_train, x_prediction_scaled, 0.0)[0]
        return float(prediction), validation_prediction, "Linear"
    if normalized in {"randomforest", "randomforestregressor", "histgradientboosting", "histgradientboostingregressor"}:
        try:
            if normalized.startswith("random"):
                from sklearn.ensemble import RandomForestRegressor

                model = RandomForestRegressor(
                    n_estimators=40,
                    max_depth=6,
                    min_samples_leaf=4,
                    random_state=int(random_seed),
                    n_jobs=1,
                )
                actual_name = "RandomForest"
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor

                model = HistGradientBoostingRegressor(
                    max_iter=60,
                    max_depth=4,
                    min_samples_leaf=8,
                    random_state=int(random_seed),
                )
                actual_name = "HistGradientBoosting"
            model.fit(x_train_scaled, y_train)
            return float(model.predict(x_prediction_scaled)[0]), model.predict(x_validation_scaled), actual_name
        except Exception:
            validation_prediction = _ridge_predict(x_train_scaled, y_train, x_validation_scaled, 1.0)
            prediction = _ridge_predict(x_train_scaled, y_train, x_prediction_scaled, 1.0)[0]
            return float(prediction), validation_prediction, "RidgeFallback"

    validation_prediction = _ridge_predict(x_train_scaled, y_train, x_validation_scaled, 1.0)
    prediction = _ridge_predict(x_train_scaled, y_train, x_prediction_scaled, 1.0)[0]
    return float(prediction), validation_prediction, "Ridge"


def _target_outcome_date(index: pd.DatetimeIndex, position: int, horizon: int) -> pd.Timestamp:
    outcome_position = int(position) + int(horizon)
    if outcome_position < len(index):
        return pd.Timestamp(index[outcome_position])
    if not any(pd.Timestamp(value).weekday() >= 5 for value in index):
        return pd.Timestamp(index[position]) + pd.offsets.BDay(int(horizon))
    positive_steps = pd.Series(index[1:] - index[:-1])
    step = positive_steps[positive_steps > pd.Timedelta(0)].median() if not positive_steps.empty else pd.Timedelta(days=1)
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(days=1)
    return pd.Timestamp(index[position]) + step * int(horizon)


def _replay_asset_horizon(
    market: pd.DataFrame,
    asset: str,
    horizon: int,
    model_name: str,
    max_windows: int,
    min_train_rows: int,
    step_size: int,
    cost_bps: float,
    slippage_bps: float,
    random_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    target_column = get_target_column(asset)
    if target_column not in market.columns:
        return [], []
    price = pd.to_numeric(market[target_column], errors="coerce")
    frame, feature_cols, target_col = _feature_frame(price, int(horizon))
    positions = _window_positions(frame, feature_cols, int(horizon), int(min_train_rows), int(step_size), int(max_windows))
    prediction_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    total_cost_pct = (float(cost_bps) + float(slippage_bps)) / 100.0

    for window_number, prediction_position in enumerate(positions, start=1):
        prediction_date = pd.Timestamp(frame.index[prediction_position])
        past_limit = prediction_position - int(horizon) - 1
        valid_features = frame[feature_cols].notna().all(axis=1)
        available_mask = valid_features & frame[target_col].notna() & frame["_position"].le(past_limit)
        available = frame.loc[available_mask].copy()
        validation_count = max(5, min(30, len(available) // 5))
        train = available.iloc[:-validation_count]
        validation = available.iloc[-validation_count:]
        if len(train) < int(min_train_rows) or validation.empty:
            continue

        x_train = train[feature_cols].astype(float).to_numpy()
        y_train = train[target_col].astype(float).to_numpy()
        x_validation = validation[feature_cols].astype(float).to_numpy()
        y_validation = validation[target_col].astype(float).to_numpy()
        x_prediction = frame.iloc[[prediction_position]][feature_cols].astype(float).to_numpy()
        prediction, validation_prediction, actual_model_name = _fit_predict_model(
            model_name,
            x_train,
            y_train,
            x_validation,
            x_prediction,
            int(random_seed) + window_number,
        )
        validation_rmse = float(np.sqrt(np.mean((y_validation - validation_prediction) ** 2))) if len(y_validation) else np.nan
        confidence = min(95.0, max(0.0, abs(prediction) / max(validation_rmse, EPSILON) * 50.0)) if np.isfinite(validation_rmse) else 0.0
        outcome_position = prediction_position + int(horizon)
        target_date = _target_outcome_date(pd.DatetimeIndex(frame.index), prediction_position, int(horizon))
        entry_price = _safe_float(frame.iloc[prediction_position]["_price"])
        matured = bool(outcome_position < len(frame) and np.isfinite(entry_price))
        realized_pct = np.nan
        if matured:
            exit_price = _safe_float(frame.iloc[outcome_position]["_price"])
            matured = bool(np.isfinite(exit_price) and exit_price > 0 and entry_price > 0)
            if matured:
                realized_pct = float(np.log(exit_price / entry_price) * 100.0)
        signal_label = "PaperTrack" if prediction > 0 else "WatchlistOnly"
        net_pct = realized_pct - total_cost_pct if matured and signal_label == "PaperTrack" else (0.0 if matured else np.nan)
        window_id = f"{asset.replace(' ', '_')}_{int(horizon)}D_W{window_number:03d}"

        train_end_before = bool(pd.Timestamp(train.index.max()) < prediction_date)
        prediction_before_target = bool(prediction_date < target_date)
        latest_validation_position = int(validation["_position"].max())
        latest_validation_outcome_position = latest_validation_position + int(horizon)
        validation_outcome_before_prediction = bool(
            latest_validation_outcome_position < prediction_position
            and pd.Timestamp(frame.index[latest_validation_outcome_position]) < prediction_date
        )
        target_features = [col for col in feature_cols if str(col).startswith(FUTURE_TARGET_PREFIXES)]
        leakage_passed = bool(
            train_end_before
            and prediction_before_target
            and validation_outcome_before_prediction
            and not target_features
        )

        prediction_rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "WindowId": window_id,
                "TrainStartDate": pd.Timestamp(train.index.min()),
                "TrainEndDate": pd.Timestamp(train.index.max()),
                "PredictionDate": prediction_date,
                "TargetOutcomeDate": target_date,
                "ModelName": actual_model_name,
                "PredictedReturnPct": round(prediction * 100.0, 6),
                "RealizedReturnPct": round(realized_pct, 6) if np.isfinite(realized_pct) else np.nan,
                "NetRealizedReturnPct": round(net_pct, 6) if np.isfinite(net_pct) else np.nan,
                "SignalLabel": signal_label,
                "ConfidenceProxy": round(confidence, 4),
                "OutcomeStatus": "Matured" if matured else "Pending",
                "CostBps": float(cost_bps),
                "SlippageBps": float(slippage_bps),
                "IsMatured": matured,
                "LeakagePassed": leakage_passed,
            }
        )
        audit_rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "WindowId": window_id,
                "ModelName": actual_model_name,
                "TrainStartDate": pd.Timestamp(train.index.min()),
                "TrainEndDate": pd.Timestamp(train.index.max()),
                "ValidationEndDate": pd.Timestamp(validation.index.max()),
                "PredictionDate": prediction_date,
                "TargetOutcomeDate": target_date,
                "TrainEndBeforePrediction": train_end_before,
                "PredictionBeforeTargetOutcome": prediction_before_target,
                "ScalerFitTrainOnly": True,
                "NoTargetLeakage": not target_features,
                "NoFutureRowsUsed": validation_outcome_before_prediction,
                "FutureTargetColumnsExcluded": not target_features,
                "FeatureCount": len(feature_cols),
                "FeatureColumns": "; ".join(feature_cols),
                "LeakagePassed": leakage_passed,
                "Explanation": "Training and validation labels matured before prediction; preprocessing was fit on training rows only." if leakage_passed else "At least one chronological or feature-isolation check failed.",
            }
        )
    return prediction_rows, audit_rows


def _compound_return(return_pct: Sequence[float]) -> float:
    values = pd.to_numeric(pd.Series(return_pct), errors="coerce").dropna().to_numpy(dtype=float) / 100.0
    if len(values) == 0:
        return 0.0
    return float(((1.0 + values).prod() - 1.0) * 100.0)


def _max_drawdown(return_pct: Sequence[float]) -> float:
    values = pd.to_numeric(pd.Series(return_pct), errors="coerce").dropna().to_numpy(dtype=float) / 100.0
    if len(values) == 0:
        return 0.0
    equity = pd.Series((1.0 + values).cumprod())
    return float((equity / equity.cummax() - 1.0).min() * 100.0)


def _performance_table(log: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if log.empty:
        return pd.DataFrame(columns=list(PERFORMANCE_COLUMNS))
    for (asset, horizon, model), group in log.groupby(["Asset", "Horizon", "ModelName"], dropna=False):
        group = group.sort_values("PredictionDate")
        matured = group[group["IsMatured"].astype(bool)].copy()
        active = matured[matured["SignalLabel"].eq("PaperTrack")].copy()
        net = pd.to_numeric(matured["NetRealizedReturnPct"], errors="coerce").fillna(0.0)
        active_net = pd.to_numeric(active["NetRealizedReturnPct"], errors="coerce").dropna()
        total_return = _compound_return(net)
        periods_per_year = 252.0 / max(int(horizon), 1)
        annualized = ((1.0 + total_return / 100.0) ** (periods_per_year / max(len(matured), 1)) - 1.0) * 100.0 if total_return > -100 and len(matured) else 0.0
        standard_deviation = float(net.std(ddof=0))
        sharpe = float(net.mean() / standard_deviation * np.sqrt(periods_per_year)) if standard_deviation > EPSILON else 0.0
        downside = net[net < 0]
        downside_deviation = float(downside.std(ddof=0)) if len(downside) else 0.0
        sortino = float(net.mean() / downside_deviation * np.sqrt(periods_per_year)) if downside_deviation > EPSILON else 0.0
        max_drawdown = _max_drawdown(net)
        calmar = float(annualized / abs(max_drawdown)) if abs(max_drawdown) > EPSILON else 0.0
        predicted_up = pd.to_numeric(matured["PredictedReturnPct"], errors="coerce") > 0
        actual_up = pd.to_numeric(matured["RealizedReturnPct"], errors="coerce") > 0
        exposure = group["SignalLabel"].eq("PaperTrack").astype(float)
        turnover = float(exposure.diff().abs().fillna(exposure.abs()).sum())
        wins = active_net[active_net > 0]
        losses = active_net[active_net < 0]
        average_win = float(wins.mean()) if len(wins) else 0.0
        average_loss = float(losses.mean()) if len(losses) else 0.0
        win_loss_ratio = float(average_win / abs(average_loss)) if average_loss < -EPSILON else 0.0
        total_cost_pct = (_safe_float(group["CostBps"].iloc[0], 0.0) + _safe_float(group["SlippageBps"].iloc[0], 0.0)) / 100.0
        rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "ModelName": model,
                "TotalReturnPct": round(total_return, 6),
                "AnnualizedReturnPct": round(float(annualized), 6),
                "MaxDrawdownPct": round(max_drawdown, 6),
                "HitRatePct": round(float((predicted_up == actual_up).mean() * 100.0), 4) if len(matured) else 0.0,
                "AverageWinPct": round(average_win, 6),
                "AverageLossPct": round(average_loss, 6),
                "WinLossRatio": round(win_loss_ratio, 6),
                "SharpeLike": round(sharpe, 6),
                "SortinoLike": round(sortino, 6),
                "CalmarLike": round(calmar, 6),
                "Turnover": round(turnover, 4),
                "CostDragPct": round(float(len(active) * total_cost_pct), 6),
                "MaturedTrades": int(len(active)),
                "PendingTrades": int((~group["IsMatured"].astype(bool)).sum()),
            }
        )
    return pd.DataFrame(rows, columns=list(PERFORMANCE_COLUMNS))


def _baseline_return(realized_pct: np.ndarray, exposure: np.ndarray, total_cost_bps: float, apply_repeated_cost: bool = True) -> float:
    exposure = np.asarray(exposure, dtype=float)
    returns = np.asarray(realized_pct, dtype=float) / 100.0
    costs = exposure * (float(total_cost_bps) / 10000.0) if apply_repeated_cost else np.zeros_like(exposure)
    return float(((1.0 + returns * exposure - costs).prod() - 1.0) * 100.0)


def _baseline_comparison(log: pd.DataFrame, performance: pd.DataFrame, market: pd.DataFrame, random_seed: int, simulations: int = 50) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if log.empty or performance.empty:
        return pd.DataFrame(columns=list(BASELINE_COMPARISON_COLUMNS))
    for (asset, horizon, model), group in log.groupby(["Asset", "Horizon", "ModelName"], dropna=False):
        matured = group[group["IsMatured"].astype(bool)].sort_values("PredictionDate").copy()
        performance_match = performance[
            performance["Asset"].eq(asset)
            & performance["Horizon"].eq(int(horizon))
            & performance["ModelName"].eq(model)
        ]
        ml_return = _safe_float(performance_match["TotalReturnPct"].iloc[0], 0.0) if not performance_match.empty else 0.0
        if matured.empty:
            baseline_returns = {"NoExposure": 0.0, "HoldOnly": 0.0, "MomentumBaseline": 0.0, "MovingAverageBaseline": 0.0, "RandomMedianBaseline": 0.0}
        else:
            prices = pd.to_numeric(market.get(get_target_column(str(asset)), pd.Series(dtype=float)), errors="coerce")
            prediction_dates = pd.to_datetime(matured["PredictionDate"], errors="coerce")
            realized = pd.to_numeric(matured["RealizedReturnPct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            momentum_exposure: List[float] = []
            moving_average_exposure: List[float] = []
            short_ma = prices.rolling(10, min_periods=10).mean()
            long_ma = prices.rolling(30, min_periods=30).mean()
            for date in prediction_dates:
                if date not in prices.index:
                    momentum_exposure.append(0.0)
                    moving_average_exposure.append(0.0)
                    continue
                location = prices.index.get_loc(date)
                if isinstance(location, slice) or not isinstance(location, (int, np.integer)):
                    momentum_exposure.append(0.0)
                elif int(location) >= int(horizon):
                    momentum_exposure.append(float(prices.iloc[int(location)] > prices.iloc[int(location) - int(horizon)]))
                else:
                    momentum_exposure.append(0.0)
                moving_average_exposure.append(float(short_ma.loc[date] > long_ma.loc[date]) if pd.notna(short_ma.loc[date]) and pd.notna(long_ma.loc[date]) else 0.0)
            total_cost_bps = _safe_float(matured["CostBps"].iloc[0], 0.0) + _safe_float(matured["SlippageBps"].iloc[0], 0.0)
            baseline_returns = {
                "NoExposure": 0.0,
                "HoldOnly": _baseline_return(realized, np.ones(len(realized)), 0.0, apply_repeated_cost=False),
                "MomentumBaseline": _baseline_return(realized, np.asarray(momentum_exposure), total_cost_bps),
                "MovingAverageBaseline": _baseline_return(realized, np.asarray(moving_average_exposure), total_cost_bps),
            }
            rng = np.random.default_rng(int(random_seed) + int(horizon) * 997 + sum(ord(char) for char in str(asset)))
            random_returns = [
                _baseline_return(realized, rng.binomial(1, 0.5, len(realized)), total_cost_bps)
                for _ in range(max(1, int(simulations)))
            ]
            baseline_returns["RandomMedianBaseline"] = float(np.median(random_returns))
        best_name, best_return = max(baseline_returns.items(), key=lambda item: item[1])
        gap = ml_return - best_return
        rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "ModelName": model,
                "MLReturnPct": round(ml_return, 6),
                "BestBaselineName": best_name,
                "BestBaselineReturnPct": round(float(best_return), 6),
                "BeatsBestBaseline": bool(gap > EPSILON),
                "GapPct": round(float(gap), 6),
                "DominanceVerdict": "ResearchOnly" if gap > EPSILON else "BenchmarkDominated",
            }
        )
    return pd.DataFrame(rows, columns=list(BASELINE_COMPARISON_COLUMNS))


def _strength_table(log: pd.DataFrame, performance: pd.DataFrame, comparison: pd.DataFrame, audit: pd.DataFrame, min_trades: int = 3) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if performance.empty:
        return pd.DataFrame(columns=list(STRENGTH_COLUMNS))
    for _, perf in performance.iterrows():
        asset, horizon, model = str(perf["Asset"]), int(perf["Horizon"]), str(perf["ModelName"])
        comp = comparison[comparison["Asset"].eq(asset) & comparison["Horizon"].eq(horizon) & comparison["ModelName"].eq(model)]
        audits = audit[audit["Asset"].eq(asset) & audit["Horizon"].eq(horizon) & audit["ModelName"].eq(model)]
        logs = log[log["Asset"].eq(asset) & log["Horizon"].eq(horizon) & log["ModelName"].eq(model)]
        leakage_passed = bool(not audits.empty and audits["LeakagePassed"].astype(bool).all())
        gap = _safe_float(comp["GapPct"].iloc[0], 0.0) if not comp.empty else 0.0
        baseline_return = _safe_float(comp["BestBaselineReturnPct"].iloc[0], 0.0) if not comp.empty else 0.0
        matured_trades = int(perf["MaturedTrades"])
        active_matured = logs[logs["IsMatured"].astype(bool) & logs["SignalLabel"].eq("PaperTrack")]
        gross_return = _compound_return(active_matured["RealizedReturnPct"]) if not active_matured.empty else 0.0
        ml_return = _safe_float(perf["TotalReturnPct"], 0.0)
        cost_fragile = bool(gross_return > 0 and ml_return <= 0)
        if not leakage_passed:
            classification = "LeakageFailed"
            reason = "One or more chronological leakage checks failed."
        elif matured_trades < int(min_trades):
            classification = "InsufficientTrades"
            reason = "Too few matured paper signals support this replay row."
        elif cost_fragile:
            classification = "CostFragile"
            reason = "The gross research edge disappears after configured costs."
        elif gap < -EPSILON:
            classification = "BenchmarkDominated"
            reason = "A serious baseline has higher historical replay return."
        elif gap > 2.0 and ml_return > 0 and _safe_float(perf["MaxDrawdownPct"], 0.0) > -25.0:
            classification = "StrongCandidate"
            reason = "The row beats its strongest baseline with positive replay return and controlled drawdown."
        elif gap > EPSILON:
            classification = "WeakCandidate"
            reason = "The row has a narrow baseline edge that needs more historical and forward evidence."
        else:
            classification = "ResearchOnly"
            reason = "The row remains useful as historical diagnostic evidence only."
        rows.append(
            {
                "Asset": asset,
                "Horizon": horizon,
                "ModelName": model,
                "StrengthClassification": classification,
                "MaturedTrades": matured_trades,
                "MLReturnPct": ml_return,
                "BestBaselineReturnPct": baseline_return,
                "GapPct": gap,
                "LeakagePassed": leakage_passed,
                "CostFragile": cost_fragile,
                "RealCapitalStatus": "Blocked",
                "ResearchStatus": "ResearchOnly",
                "MainReason": reason,
            }
        )
    return pd.DataFrame(rows, columns=list(STRENGTH_COLUMNS))


def _input_sources(market: pd.DataFrame, assets: Sequence[str], project_data_used: bool) -> pd.DataFrame:
    missing = [get_target_column(asset) for asset in assets if get_target_column(asset) not in market.columns]
    return pd.DataFrame(
        [
            {
                "SourceName": "ProjectMasterDataset" if project_data_used else "ProvidedMarketData",
                "Available": not market.empty,
                "Rows": len(market),
                "Columns": len(market.columns),
                "FirstDate": str(market.index.min()) if not market.empty else "",
                "LastDate": str(market.index.max()) if not market.empty else "",
                "AssetsRequested": "; ".join(assets),
                "MissingCriticalColumns": "; ".join(missing),
                "Notes": "Historical price source used to build past-only features and future outcomes.",
            }
        ],
        columns=list(INPUT_SOURCE_COLUMNS),
    )


def _quality_gates(log: pd.DataFrame, audit: pd.DataFrame, comparison: pd.DataFrame, input_sources: pd.DataFrame) -> pd.DataFrame:
    audit_failures = int((~audit["LeakagePassed"].astype(bool)).sum()) if not audit.empty else 0
    future_feature_failures = int((~audit["FutureTargetColumnsExcluded"].astype(bool)).sum()) if not audit.empty else 0
    no_future_failures = int((~audit["NoFutureRowsUsed"].astype(bool)).sum()) if not audit.empty else 0
    missing_columns = str(input_sources.iloc[0]["MissingCriticalColumns"]) if not input_sources.empty else ""
    gates = [
        ("PredictionLogAvailable", not log.empty, "Critical", 0 if not log.empty else 1, "True historical prediction rows must be available."),
        ("ChronologicalTraining", not audit.empty and audit_failures == 0, "Critical", audit_failures, "Train and validation evidence must precede prediction dates."),
        ("ScalerFitTrainOnly", not audit.empty and audit["ScalerFitTrainOnly"].astype(bool).all(), "Critical", int((~audit["ScalerFitTrainOnly"].astype(bool)).sum()) if not audit.empty else 1, "Imputation and scaling are fit separately inside each training window."),
        ("FutureTargetsExcluded", not audit.empty and future_feature_failures == 0, "Critical", future_feature_failures, "Future target columns are excluded from model features."),
        ("NoFutureRowsUsed", not audit.empty and no_future_failures == 0, "Critical", no_future_failures, "Every training and validation target matured before its prediction date."),
        ("BaselineComparisonAvailable", not comparison.empty, "High", 0 if not comparison.empty else 1, "Each comparable replay row is tested against serious baselines."),
        ("CostsApplied", not log.empty and {"CostBps", "SlippageBps"}.issubset(log.columns), "High", 0, "Configured cost and slippage are deducted from active paper signals."),
        ("CriticalMarketColumnsAvailable", not bool(missing_columns), "Critical", len([value for value in missing_columns.split("; ") if value]), "Requested asset price columns must be available."),
        ("TrueReplayClearlyLabeled", True, "Critical", 0, "ReplayType is explicitly TrueHistoricalMLReplay and is separate from proxy replay."),
        ("RealCapitalBlocked", True, "Critical", 0, "Phase 20 remains research-only and real capital is blocked."),
    ]
    return pd.DataFrame(
        [{"GateName": name, "Passed": bool(passed), "Severity": severity, "AffectedRows": int(affected), "Explanation": explanation} for name, passed, severity, affected, explanation in gates],
        columns=list(QUALITY_GATE_COLUMNS),
    )


def _summary(log: pd.DataFrame, performance: pd.DataFrame, comparison: pd.DataFrame, audit: pd.DataFrame, assets: Sequence[str], horizons: Sequence[int]) -> pd.DataFrame:
    best = performance.sort_values(["TotalReturnPct", "SharpeLike"], ascending=[False, False]).iloc[0] if not performance.empty else pd.Series(dtype=object)
    leakage_rate = float(audit["LeakagePassed"].astype(bool).mean() * 100.0) if not audit.empty else 0.0
    beats_count = int(comparison["BeatsBestBaseline"].astype(bool).sum()) if not comparison.empty else 0
    if log.empty or leakage_rate < 100.0:
        verdict = "InsufficientEvidence"
        reason = "True historical replay evidence is missing or at least one leakage gate failed."
    elif beats_count >= max(3, len(comparison) // 3):
        verdict = "ResearchOnly"
        reason = "Some historical ML replay rows beat serious baselines, but they still require forward evidence."
    else:
        verdict = "NoBroadEdgeProven"
        reason = "True historical replay does not establish broad edge over serious baselines."
    return pd.DataFrame(
        [
            {
                "ReplayType": "TrueHistoricalMLReplay",
                "ProxyReplayStillUsed": False,
                "TotalWindows": int(log["WindowId"].nunique()) if not log.empty else 0,
                "TotalPredictions": len(log),
                "MaturedPredictions": int(log["IsMatured"].astype(bool).sum()) if not log.empty else 0,
                "PendingPredictions": int((~log["IsMatured"].astype(bool)).sum()) if not log.empty else 0,
                "LeakagePassRate": round(leakage_rate, 4),
                "ModelsTested": "; ".join(sorted(log["ModelName"].astype(str).unique())) if not log.empty else "",
                "AssetsCovered": "; ".join(assets),
                "HorizonsCovered": "; ".join(f"{int(horizon)}D" for horizon in horizons),
                "BestAsset": str(best.get("Asset", "")) if not best.empty else "",
                "BestHorizon": int(best.get("Horizon", 0)) if not best.empty else 0,
                "BestModel": str(best.get("ModelName", "")) if not best.empty else "",
                "BestNetReturnPct": _safe_float(best.get("TotalReturnPct", np.nan), np.nan) if not best.empty else np.nan,
                "BeatsBestBaselineCount": beats_count,
                "RealCapitalStatus": "Blocked",
                "FinalVerdict": verdict,
                "MainReason": reason,
                "MainLimitation": "Historical replay is research evidence and does not authorize real-capital use.",
            }
        ],
        columns=list(SUMMARY_COLUMNS),
    )


def _next_actions(summary: pd.DataFrame, strength: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    failed_gates = quality[~quality["Passed"].astype(bool)] if not quality.empty else pd.DataFrame()
    dominated = strength[strength["StrengthClassification"].eq("BenchmarkDominated")] if not strength.empty else pd.DataFrame()
    rows: List[Dict[str, Any]] = [
        {
            "Rank": 1,
            "Action": "Continue timestamped forward paper evidence collection.",
            "WhyItMatters": "Historical replay and forward evidence answer different reliability questions.",
            "AffectedAssets": "ALL",
            "AffectedHorizons": "ALL",
            "ExpectedBenefit": "Tests whether replay behavior survives genuinely unseen outcomes.",
            "Urgency": "High",
            "DependsOn": "Phase 9 forward evidence.",
        }
    ]
    if not failed_gates.empty:
        rows.append(
            {
                "Rank": len(rows) + 1,
                "Action": "Resolve failed replay quality gates before interpreting performance.",
                "WhyItMatters": "Chronology and input quality take priority over return metrics.",
                "AffectedAssets": "ALL",
                "AffectedHorizons": "ALL",
                "ExpectedBenefit": "Restores auditable historical evidence.",
                "Urgency": "Critical",
                "DependsOn": "; ".join(failed_gates["GateName"].astype(str)),
            }
        )
    if not dominated.empty:
        rows.append(
            {
                "Rank": len(rows) + 1,
                "Action": "Retain benchmark-dominated rows as rejection evidence.",
                "WhyItMatters": "A simple baseline remains the minimum useful hurdle.",
                "AffectedAssets": "; ".join(sorted(dominated["Asset"].astype(str).unique())),
                "AffectedHorizons": "; ".join(f"{int(value)}D" for value in sorted(dominated["Horizon"].astype(int).unique())),
                "ExpectedBenefit": "Prevents weak replay rows from being overstated.",
                "Urgency": "High",
                "DependsOn": "Phase 20 baseline comparison.",
            }
        )
    return pd.DataFrame(rows, columns=list(NEXT_ACTION_COLUMNS))


def run_true_historical_ml_replay(
    *,
    market_data: Optional[pd.DataFrame] = None,
    use_project_market_data: bool = True,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    max_windows: int = 8,
    min_train_rows: int = 120,
    step_size: int = 20,
    model_name: str = "Ridge",
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    random_seed: int = 42,
    autosave: bool = False,
) -> TrueHistoricalMLReplayReport:
    """Run bounded, chronological historical ML replay for selected assets/horizons."""
    asset_list = [str(asset) for asset in (assets or get_asset_names())]
    invalid_assets = [asset for asset in asset_list if asset not in get_asset_names()]
    if invalid_assets:
        raise ValueError(f"Unknown assets: {', '.join(invalid_assets)}")
    horizon_list = [int(horizon) for horizon in (horizons or TRUE_ML_HORIZONS)]
    if any(horizon <= 0 for horizon in horizon_list):
        raise ValueError("Horizons must be positive integers")
    if int(max_windows) <= 0 or int(min_train_rows) < 20 or int(step_size) <= 0:
        raise ValueError("max_windows and step_size must be positive; min_train_rows must be at least 20")

    project_data_used = False
    if market_data is None and use_project_market_data:
        market_data = _load_project_market_data()
        project_data_used = market_data is not None
    market = _prepare_market_data(market_data)

    prediction_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for asset in asset_list:
        for horizon in horizon_list:
            rows, audits = _replay_asset_horizon(
                market,
                asset,
                horizon,
                model_name,
                int(max_windows),
                int(min_train_rows),
                int(step_size),
                float(cost_bps),
                float(slippage_bps),
                int(random_seed),
            )
            prediction_rows.extend(rows)
            audit_rows.extend(audits)

    prediction_log = pd.DataFrame(prediction_rows, columns=list(PREDICTION_LOG_COLUMNS))
    leakage_audit = pd.DataFrame(audit_rows, columns=list(LEAKAGE_AUDIT_COLUMNS))
    performance = _performance_table(prediction_log)
    baseline_comparison = _baseline_comparison(prediction_log, performance, market, int(random_seed))
    strength = _strength_table(prediction_log, performance, baseline_comparison, leakage_audit)
    input_sources = _input_sources(market, asset_list, project_data_used)
    quality_gates = _quality_gates(prediction_log, leakage_audit, baseline_comparison, input_sources)
    summary = _summary(prediction_log, performance, baseline_comparison, leakage_audit, asset_list, horizon_list)
    next_actions = _next_actions(summary, strength, quality_gates)
    settings = {
        "phase": "20",
        "replay_type": "TrueHistoricalMLReplay",
        "assets": asset_list,
        "horizons": horizon_list,
        "max_windows": int(max_windows),
        "min_train_rows": int(min_train_rows),
        "step_size": int(step_size),
        "model_name": str(model_name),
        "cost_bps": float(cost_bps),
        "slippage_bps": float(slippage_bps),
        "random_seed": int(random_seed),
        "real_capital_status": "Blocked",
    }
    report = TrueHistoricalMLReplayReport(
        true_ml_summary_table=summary.reset_index(drop=True),
        true_ml_prediction_log=prediction_log.reset_index(drop=True),
        true_ml_performance_table=performance.reset_index(drop=True),
        true_ml_baseline_comparison_table=baseline_comparison.reset_index(drop=True),
        true_ml_strength_table=strength.reset_index(drop=True),
        leakage_audit_table=leakage_audit.reset_index(drop=True),
        input_sources_table=input_sources.reset_index(drop=True),
        quality_gates_table=quality_gates.reset_index(drop=True),
        next_actions_table=next_actions.reset_index(drop=True),
        settings=settings,
    )
    if autosave:
        report.saved_artifacts = save_phase_artifacts(
            TRUE_HISTORICAL_ML_REPLAY_PHASE_NAME,
            {
                "phase20_true_ml_summary": report.true_ml_summary_table,
                "phase20_true_ml_prediction_log": report.true_ml_prediction_log,
                "phase20_true_ml_performance": report.true_ml_performance_table,
                "phase20_true_ml_baseline_comparison": report.true_ml_baseline_comparison_table,
                "phase20_true_ml_strength": report.true_ml_strength_table,
                "phase20_leakage_audit": report.leakage_audit_table,
                "phase20_input_sources": report.input_sources_table,
                "phase20_quality_gates": report.quality_gates_table,
                "phase20_next_actions": report.next_actions_table,
            },
            inputs={"market_data": {"Source": "ProjectMasterDataset" if project_data_used else "ProvidedMarketData"}},
            config=settings,
            warnings=[] if bool(quality_gates["Passed"].all()) else ["One or more Phase 20 quality gates failed."],
        )
    return report


__all__ = [
    "TRUE_HISTORICAL_ML_REPLAY_PHASE_NAME",
    "TRUE_ML_HORIZONS",
    "MODEL_CHOICES",
    "SUMMARY_COLUMNS",
    "PREDICTION_LOG_COLUMNS",
    "PERFORMANCE_COLUMNS",
    "BASELINE_COMPARISON_COLUMNS",
    "STRENGTH_COLUMNS",
    "LEAKAGE_AUDIT_COLUMNS",
    "INPUT_SOURCE_COLUMNS",
    "QUALITY_GATE_COLUMNS",
    "NEXT_ACTION_COLUMNS",
    "TrueHistoricalMLReplayReport",
    "run_true_historical_ml_replay",
]
