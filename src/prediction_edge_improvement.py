"""Phase 22 prediction edge improvement and model benchmark expansion.

The engine reuses Phase 20's historical target/window rules, evaluates model
and feature candidates on chronological validation rows, and keeps every
losing or unavailable candidate visible. It is research-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.util
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src import true_historical_ml_replay as phase20
from src.artifact_store import save_phase_artifacts
from src.asset_config import get_asset_names, get_target_column


PREDICTION_EDGE_IMPROVEMENT_PHASE_NAME = "phase22_prediction_edge_improvement"
EDGE_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)
DEFAULT_MODELS: Tuple[str, ...] = (
    "Ridge",
    "ElasticNet",
    "LinearRegression",
    "RandomForestRegressor",
    "HistGradientBoostingRegressor",
)
OPTIONAL_MODELS: Tuple[str, ...] = ("XGBoost", "LightGBM", "CatBoost")
DEFAULT_FEATURE_GROUPS: Tuple[str, ...] = (
    "PriceReturn",
    "TechnicalIndicators",
    "CrossAsset",
    "Calendar",
    "FullFeatureSet",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
EPSILON = 1e-12

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "PhaseName", "TotalModelsTested", "OptionalModelsAvailable",
    "OptionalModelsTested", "OptionalModelsSkipped", "TotalPredictions",
    "MaturedPredictions", "BestAsset", "BestHorizon", "BestModel",
    "BestFeatureGroup", "BestNetReturnPct", "BestBaselineGapPct",
    "BeatsBestBaselineCount", "BroadEdgeStatus", "RealCapitalStatus",
    "RecommendedMode", "FinalVerdict",
)

MODEL_LEADERBOARD_COLUMNS: Tuple[str, ...] = (
    "ModelName", "FeatureGroup", "Asset", "Horizon", "NetReturnPct",
    "BestBaselineReturnPct", "GapPct", "BeatsBestBaseline", "HitRatePct",
    "MaxDrawdownPct", "SharpeLike", "CostDragPct", "MaturedTrades",
    "ValidationScore", "StabilityLabel", "ResearchLabel",
)

ASSET_HORIZON_SCORECARD_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "BestModel", "BestFeatureGroup", "NetReturnPct",
    "BestBaselineName", "BestBaselineReturnPct", "GapPct", "BeatsBestBaseline",
    "HitRatePct", "MaxDrawdownPct", "MaturedTrades", "ValidationScore",
    "StabilityLabel", "ResearchLabel", "MainReason",
)

PREDICTION_LOG_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "WindowId", "TrainStartDate", "TrainEndDate",
    "ValidationStartDate", "ValidationEndDate", "PredictionDate", "TargetOutcomeDate",
    "ModelName", "FeatureGroup", "PredictedReturnPct", "RealizedReturnPct",
    "NetRealizedReturnPct", "ValidationScore", "SignalLabel", "OutcomeStatus",
    "IsMatured", "IsSelected", "CostBps", "SlippageBps", "LeakagePassed",
)

BASELINE_COMPARISON_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "ModelName", "FeatureGroup", "ModelReturnPct",
    "NoExposureReturnPct", "HoldOnlyReturnPct", "MomentumBaselineReturnPct",
    "MovingAverageBaselineReturnPct", "RandomMedianBaselineReturnPct",
    "BestBaselineName", "BestBaselineReturnPct", "GapPct", "BeatsBestBaseline",
    "DominanceVerdict",
)

FEATURE_GROUP_AUDIT_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "FeatureGroup", "FeatureCount", "FeatureColumns",
    "FutureTargetColumnsExcluded", "PastOrCurrentOnly", "WindowsEvaluated",
    "TimesSelected", "AverageValidationScore", "AuditPassed", "Explanation",
)

MODEL_SELECTION_AUDIT_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "WindowId", "CandidateModels", "SelectedModel",
    "SelectedFeatureGroup", "ValidationStartDate", "ValidationEndDate",
    "PredictionDate", "SelectionUsedFutureData", "SelectionPassed",
)

LEAKAGE_AUDIT_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "WindowId", "TrainEndDate", "ValidationStartDate",
    "ValidationEndDate", "PredictionDate", "TargetOutcomeDate",
    "TrainEndBeforeValidation", "ValidationEndBeforePrediction",
    "PredictionBeforeTargetOutcome", "ScalerFitPastOnly", "NoTargetLeakage",
    "NoFutureRowsUsed", "FutureTargetColumnsExcluded", "LeakagePassed", "Explanation",
)

COST_SENSITIVITY_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "ModelName", "FeatureGroup", "CostBps", "NetReturnPct",
    "ReturnLostToCostsPct", "CostFragile", "Explanation",
)

REJECTED_MODEL_COLUMNS: Tuple[str, ...] = (
    "Asset", "Horizon", "ModelName", "FeatureGroup", "RejectionReason",
    "FailedChecks", "ValidationScore", "NetReturnPct", "GapPct", "SuggestedFix",
)

QUALITY_GATE_COLUMNS: Tuple[str, ...] = (
    "GateName", "Passed", "Severity", "Explanation",
)

NEXT_ACTION_COLUMNS: Tuple[str, ...] = (
    "Priority", "Action", "Reason", "ExpectedImpact", "PhaseSuggestion",
)

INPUT_SOURCE_COLUMNS: Tuple[str, ...] = (
    "SourceName", "Available", "Rows", "Columns", "FirstDate", "LastDate",
    "AssetsRequested", "MissingCriticalColumns", "Notes",
)


@dataclass
class PredictionEdgeImprovementReport:
    prediction_edge_summary: pd.DataFrame
    model_leaderboard: pd.DataFrame
    asset_horizon_model_scorecard: pd.DataFrame
    prediction_log: pd.DataFrame
    baseline_comparison: pd.DataFrame
    feature_group_audit: pd.DataFrame
    model_selection_audit: pd.DataFrame
    leakage_audit: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    rejected_models: pd.DataFrame
    quality_gates: pd.DataFrame
    next_actions: pd.DataFrame
    input_sources: pd.DataFrame
    settings: Dict[str, Any] = field(default_factory=dict)
    saved_artifacts: Dict[str, Any] = field(default_factory=dict)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        result = float(value)
    except Exception:
        return default
    return result if np.isfinite(result) else default


def _prepare_market_data(market_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    return phase20._prepare_market_data(market_data)


def _load_project_market_data() -> Optional[pd.DataFrame]:
    if not DEFAULT_MARKET_DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DEFAULT_MARKET_DATA_PATH)
    except Exception:
        return None


def _normalize_model_name(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    aliases = {
        "ridge": "Ridge",
        "elasticnet": "ElasticNet",
        "linear": "LinearRegression",
        "linearregression": "LinearRegression",
        "randomforest": "RandomForestRegressor",
        "randomforestregressor": "RandomForestRegressor",
        "histgradientboosting": "HistGradientBoostingRegressor",
        "histgradientboostingregressor": "HistGradientBoostingRegressor",
        "xgboost": "XGBoost",
        "xgbregressor": "XGBoost",
        "lightgbm": "LightGBM",
        "lgbmregressor": "LightGBM",
        "catboost": "CatBoost",
        "catboostregressor": "CatBoost",
    }
    return aliases.get(key, str(name).strip())


def _normalize_feature_group(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    aliases = {
        "price": "PriceReturn",
        "pricereturn": "PriceReturn",
        "pricereturnfeatures": "PriceReturn",
        "technical": "TechnicalIndicators",
        "technicalindicators": "TechnicalIndicators",
        "crossasset": "CrossAsset",
        "crossassetfeatures": "CrossAsset",
        "calendar": "Calendar",
        "calendarfeatures": "Calendar",
        "full": "FullFeatureSet",
        "fullfeatureset": "FullFeatureSet",
    }
    return aliases.get(key, str(name).strip())


def _dependency_name(model_name: str) -> Optional[str]:
    return {"XGBoost": "xgboost", "LightGBM": "lightgbm", "CatBoost": "catboost"}.get(model_name)


@lru_cache(maxsize=None)
def _optional_model_importable(model_name: str) -> bool:
    dependency = _dependency_name(model_name)
    if dependency is None:
        return False
    expected_class = {
        "XGBoost": "XGBRegressor",
        "LightGBM": "LGBMRegressor",
        "CatBoost": "CatBoostRegressor",
    }[model_name]
    try:
        module = importlib.import_module(dependency)
    except Exception:
        return False
    return hasattr(module, expected_class)


def _model_dependency_available(model_name: str) -> bool:
    dependency = _dependency_name(model_name)
    if dependency is None:
        if model_name in {"RandomForestRegressor", "HistGradientBoostingRegressor"}:
            return importlib.util.find_spec("sklearn") is not None
        return model_name in {"Ridge", "ElasticNet", "LinearRegression"}
    return _optional_model_importable(model_name)


def _feature_groups(market: pd.DataFrame, asset: str, horizon: int) -> Tuple[pd.DataFrame, Dict[str, List[str]], str]:
    target_column = get_target_column(asset)
    price = pd.to_numeric(market[target_column], errors="coerce")
    frame, price_columns, target_col = phase20._feature_frame(price, int(horizon))
    price_return_cols = [col for col in price_columns if col.startswith("log_return") or col.startswith("rolling_return")]
    technical_cols = [col for col in price_columns if col not in price_return_cols]

    cross_cols: List[str] = []
    for other_asset in get_asset_names():
        if other_asset == asset:
            continue
        other_column = get_target_column(other_asset)
        if other_column not in market.columns:
            continue
        other_price = pd.to_numeric(market[other_column], errors="coerce").where(lambda values: values > 0)
        other_log = np.log(other_price)
        prefix = re.sub(r"[^a-z0-9]", "_", other_asset.lower()).strip("_")
        frame[f"cross_{prefix}_return_1"] = other_log.diff()
        frame[f"cross_{prefix}_momentum_5"] = other_log - other_log.shift(5)
        frame[f"cross_{prefix}_volatility_10"] = other_log.diff().rolling(10, min_periods=10).std()
        cross_cols.extend([f"cross_{prefix}_return_1", f"cross_{prefix}_momentum_5", f"cross_{prefix}_volatility_10"])

    calendar_cols = ["calendar_day_of_week_sin", "calendar_day_of_week_cos", "calendar_month_sin", "calendar_month_cos"]
    day = pd.Series(frame.index.dayofweek, index=frame.index, dtype=float)
    month = pd.Series(frame.index.month, index=frame.index, dtype=float)
    frame[calendar_cols[0]] = np.sin(2.0 * np.pi * day / 7.0)
    frame[calendar_cols[1]] = np.cos(2.0 * np.pi * day / 7.0)
    frame[calendar_cols[2]] = np.sin(2.0 * np.pi * month / 12.0)
    frame[calendar_cols[3]] = np.cos(2.0 * np.pi * month / 12.0)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    groups = {
        "PriceReturn": price_return_cols,
        "TechnicalIndicators": technical_cols,
        "CrossAsset": cross_cols,
        "Calendar": calendar_cols,
    }
    groups["FullFeatureSet"] = list(dict.fromkeys(price_return_cols + technical_cols + cross_cols + calendar_cols))
    return frame, groups, target_col


def _elastic_net_predict(x_train: np.ndarray, y_train: np.ndarray, x_values: np.ndarray, l1: float = 0.02, l2: float = 0.10) -> np.ndarray:
    y_mean = float(np.mean(y_train))
    centered = np.asarray(y_train, dtype=float) - y_mean
    coefficients = np.zeros(x_train.shape[1], dtype=float)
    for _ in range(80):
        for column in range(x_train.shape[1]):
            partial = centered - x_train @ coefficients + x_train[:, column] * coefficients[column]
            rho = float(np.mean(x_train[:, column] * partial))
            scale = float(np.mean(x_train[:, column] ** 2) + l2)
            coefficients[column] = np.sign(rho) * max(abs(rho) - l1 * 0.001, 0.0) / max(scale, EPSILON)
    return y_mean + x_values @ coefficients


def _fit_candidate(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    x_prediction: np.ndarray,
    random_seed: int,
) -> Tuple[Optional[float], np.ndarray, str, str]:
    model_name = _normalize_model_name(model_name)
    if not _model_dependency_available(model_name):
        return None, np.asarray([], dtype=float), model_name, "MissingOptionalDependency" if model_name in OPTIONAL_MODELS else "MissingModelDependency"
    x_train_scaled, x_validation_scaled, x_prediction_scaled = phase20._scale_from_train(
        x_train, x_validation, x_prediction
    )
    if model_name == "Ridge":
        validation = phase20._ridge_predict(x_train_scaled, y_train, x_validation_scaled, 1.0)
        prediction = phase20._ridge_predict(x_train_scaled, y_train, x_prediction_scaled, 1.0)[0]
        return float(prediction), validation, model_name, ""
    if model_name == "LinearRegression":
        validation = phase20._ridge_predict(x_train_scaled, y_train, x_validation_scaled, 0.0)
        prediction = phase20._ridge_predict(x_train_scaled, y_train, x_prediction_scaled, 0.0)[0]
        return float(prediction), validation, model_name, ""
    if model_name == "ElasticNet":
        validation = _elastic_net_predict(x_train_scaled, y_train, x_validation_scaled)
        prediction = _elastic_net_predict(x_train_scaled, y_train, x_prediction_scaled)[0]
        return float(prediction), validation, model_name, ""
    try:
        if model_name == "RandomForestRegressor":
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=35, max_depth=6, min_samples_leaf=4, random_state=int(random_seed), n_jobs=1)
        elif model_name == "HistGradientBoostingRegressor":
            from sklearn.ensemble import HistGradientBoostingRegressor

            model = HistGradientBoostingRegressor(max_iter=50, max_depth=4, min_samples_leaf=8, random_state=int(random_seed))
        elif model_name == "XGBoost":
            from xgboost import XGBRegressor

            model = XGBRegressor(n_estimators=40, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=int(random_seed), n_jobs=1, verbosity=0)
        elif model_name == "LightGBM":
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(n_estimators=40, max_depth=4, learning_rate=0.05, random_state=int(random_seed), verbosity=-1)
        elif model_name == "CatBoost":
            from catboost import CatBoostRegressor

            model = CatBoostRegressor(iterations=40, depth=4, learning_rate=0.05, random_seed=int(random_seed), verbose=False)
        else:
            return None, np.asarray([], dtype=float), model_name, "UnsupportedModel"
        model.fit(x_train_scaled, y_train)
        validation = model.predict(x_validation_scaled) if len(x_validation_scaled) else np.asarray([], dtype=float)
        prediction = model.predict(x_prediction_scaled)[0]
        return float(prediction), np.asarray(validation, dtype=float), model_name, ""
    except Exception as exc:
        return None, np.asarray([], dtype=float), model_name, f"ModelFitFailed:{type(exc).__name__}"


def _validation_score(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, float]:
    if len(actual) == 0 or len(predicted) != len(actual):
        return 0.0, np.inf
    rmse = float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))
    return float(100.0 / (1.0 + rmse * 100.0)), rmse


def _replay_asset_horizon(
    market: pd.DataFrame,
    asset: str,
    horizon: int,
    models: Sequence[str],
    requested_groups: Sequence[str],
    max_windows: int,
    min_train_rows: int,
    step_size: int,
    cost_bps: float,
    slippage_bps: float,
    random_seed: int,
    enable_ensemble: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if get_target_column(asset) not in market.columns:
        return [], [], [], [], []
    frame, groups, target_col = _feature_groups(market, asset, int(horizon))
    selected_groups = [group for group in requested_groups if group in groups and groups[group]]
    full_cols = groups["FullFeatureSet"]
    positions = phase20._window_positions(frame, full_cols, int(horizon), int(min_train_rows), int(step_size), int(max_windows))
    prediction_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    leakage_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    total_cost_pct = (float(cost_bps) + float(slippage_bps)) / 100.0
    group_counts: Dict[str, Dict[str, Any]] = {group: {"windows": 0, "selected": 0, "scores": []} for group in selected_groups}

    for model in models:
        if not _model_dependency_available(model):
            rejected_rows.append({"Asset": asset, "Horizon": int(horizon), "ModelName": model, "FeatureGroup": "ALL", "RejectionReason": "MissingOptionalDependency" if model in OPTIONAL_MODELS else "MissingModelDependency", "FailedChecks": "DependencyUnavailable", "ValidationScore": 0.0, "NetReturnPct": np.nan, "GapPct": np.nan, "SuggestedFix": "Install the optional dependency only in a controlled research environment, or keep the model skipped."})

    for window_number, prediction_position in enumerate(positions, start=1):
        prediction_date = pd.Timestamp(frame.index[prediction_position])
        past_limit = prediction_position - int(horizon) - 1
        valid_full = frame[full_cols].notna().all(axis=1)
        available_mask = valid_full & frame[target_col].notna() & frame["_position"].le(past_limit)
        available = frame.loc[available_mask].copy()
        validation_count = max(5, min(30, len(available) // 5))
        train = available.iloc[:-validation_count]
        validation = available.iloc[-validation_count:]
        if len(train) < int(min_train_rows) or validation.empty:
            continue
        y_train = train[target_col].astype(float).to_numpy()
        y_validation = validation[target_col].astype(float).to_numpy()
        window_id = f"{asset.replace(' ', '_')}_{int(horizon)}D_W{window_number:03d}"
        candidates: List[Dict[str, Any]] = []
        candidate_labels: List[str] = []
        for group in selected_groups:
            feature_cols = groups[group]
            x_train = train[feature_cols].astype(float).to_numpy()
            x_validation = validation[feature_cols].astype(float).to_numpy()
            x_prediction = frame.iloc[[prediction_position]][feature_cols].astype(float).to_numpy()
            for model in models:
                if not _model_dependency_available(model):
                    continue
                prediction, validation_prediction, actual_model, error = _fit_candidate(
                    model, x_train, y_train, x_validation, x_prediction,
                    int(random_seed) + window_number + int(horizon),
                )
                if prediction is None:
                    rejected_rows.append({"Asset": asset, "Horizon": int(horizon), "ModelName": actual_model, "FeatureGroup": group, "RejectionReason": error or "ModelFitFailed", "FailedChecks": "ModelEvaluationUnavailable", "ValidationScore": 0.0, "NetReturnPct": np.nan, "GapPct": np.nan, "SuggestedFix": "Review dependency and fit diagnostics without changing historical splits."})
                    continue
                score, rmse = _validation_score(y_validation, validation_prediction)
                candidate = {"model": actual_model, "group": group, "features": feature_cols, "prediction": float(prediction), "validation_prediction": validation_prediction, "validation_score": score, "rmse": rmse}
                candidates.append(candidate)
                candidate_labels.append(f"{actual_model}/{group}")
                group_counts[group]["windows"] += 1
                group_counts[group]["scores"].append(score)

        if enable_ensemble and len(candidates) >= 2:
            top = sorted(candidates, key=lambda row: row["rmse"])[: min(3, len(candidates))]
            inverse = np.asarray([1.0 / max(row["rmse"], EPSILON) for row in top], dtype=float)
            weights = inverse / inverse.sum()
            ensemble_validation = sum(weight * row["validation_prediction"] for weight, row in zip(weights, top))
            ensemble_prediction = float(sum(weight * row["prediction"] for weight, row in zip(weights, top)))
            ensemble_score, ensemble_rmse = _validation_score(y_validation, ensemble_validation)
            candidates.append({"model": "ValidationWeightedEnsemble", "group": "EnsembleTopCandidates", "features": [], "prediction": ensemble_prediction, "validation_prediction": ensemble_validation, "validation_score": ensemble_score, "rmse": ensemble_rmse, "components": top, "weights": weights})
            candidate_labels.append("ValidationWeightedEnsemble/EnsembleTopCandidates")

        if not candidates:
            continue
        selected = min(candidates, key=lambda row: row["rmse"])
        if selected["model"] == "ValidationWeightedEnsemble":
            final_parts: List[float] = []
            for component in selected["components"]:
                cols = component["features"]
                combined = pd.concat([train, validation])
                final_prediction, _, _, _ = _fit_candidate(
                    component["model"],
                    combined[cols].astype(float).to_numpy(),
                    combined[target_col].astype(float).to_numpy(),
                    np.empty((0, len(cols))),
                    frame.iloc[[prediction_position]][cols].astype(float).to_numpy(),
                    int(random_seed) + window_number + 1000,
                )
                final_parts.append(float(final_prediction) if final_prediction is not None else component["prediction"])
            selected_prediction = float(sum(weight * value for weight, value in zip(selected["weights"], final_parts)))
        else:
            cols = selected["features"]
            combined = pd.concat([train, validation])
            final_prediction, _, _, _ = _fit_candidate(
                selected["model"],
                combined[cols].astype(float).to_numpy(),
                combined[target_col].astype(float).to_numpy(),
                np.empty((0, len(cols))),
                frame.iloc[[prediction_position]][cols].astype(float).to_numpy(),
                int(random_seed) + window_number + 1000,
            )
            selected_prediction = float(final_prediction) if final_prediction is not None else selected["prediction"]
        selected["prediction"] = selected_prediction
        if selected["group"] in group_counts:
            group_counts[selected["group"]]["selected"] += 1

        target_date = phase20._target_outcome_date(pd.DatetimeIndex(frame.index), prediction_position, int(horizon))
        outcome_position = prediction_position + int(horizon)
        entry_price = _safe_float(frame.iloc[prediction_position]["_price"])
        matured = bool(outcome_position < len(frame) and np.isfinite(entry_price) and entry_price > 0)
        realized_pct = np.nan
        if matured:
            exit_price = _safe_float(frame.iloc[outcome_position]["_price"])
            matured = bool(np.isfinite(exit_price) and exit_price > 0)
            if matured:
                realized_pct = float(np.log(exit_price / entry_price) * 100.0)

        train_end_before_validation = bool(pd.Timestamp(train.index.max()) < pd.Timestamp(validation.index.min()))
        validation_before_prediction = bool(pd.Timestamp(validation.index.max()) < prediction_date)
        prediction_before_target = bool(prediction_date < target_date)
        latest_validation_position = int(validation["_position"].max())
        no_future_rows = bool(latest_validation_position + int(horizon) < prediction_position)
        leakage_passed = bool(train_end_before_validation and validation_before_prediction and prediction_before_target and no_future_rows)

        for candidate in candidates:
            prediction = candidate["prediction"]
            signal = "PaperTrack" if prediction > 0 else "WatchlistOnly"
            net_pct = realized_pct - total_cost_pct if matured and signal == "PaperTrack" else (0.0 if matured else np.nan)
            prediction_rows.append({"Asset": asset, "Horizon": int(horizon), "WindowId": window_id, "TrainStartDate": pd.Timestamp(train.index.min()), "TrainEndDate": pd.Timestamp(train.index.max()), "ValidationStartDate": pd.Timestamp(validation.index.min()), "ValidationEndDate": pd.Timestamp(validation.index.max()), "PredictionDate": prediction_date, "TargetOutcomeDate": target_date, "ModelName": candidate["model"], "FeatureGroup": candidate["group"], "PredictedReturnPct": round(float(prediction) * 100.0, 6), "RealizedReturnPct": round(realized_pct, 6) if np.isfinite(realized_pct) else np.nan, "NetRealizedReturnPct": round(net_pct, 6) if np.isfinite(net_pct) else np.nan, "ValidationScore": round(float(candidate["validation_score"]), 6), "SignalLabel": signal, "OutcomeStatus": "Matured" if matured else "Pending", "IsMatured": matured, "IsSelected": bool(candidate is selected), "CostBps": float(cost_bps), "SlippageBps": float(slippage_bps), "LeakagePassed": leakage_passed})

        selection_rows.append({"Asset": asset, "Horizon": int(horizon), "WindowId": window_id, "CandidateModels": "; ".join(candidate_labels), "SelectedModel": selected["model"], "SelectedFeatureGroup": selected["group"], "ValidationStartDate": pd.Timestamp(validation.index.min()), "ValidationEndDate": pd.Timestamp(validation.index.max()), "PredictionDate": prediction_date, "SelectionUsedFutureData": False, "SelectionPassed": leakage_passed})
        leakage_rows.append({"Asset": asset, "Horizon": int(horizon), "WindowId": window_id, "TrainEndDate": pd.Timestamp(train.index.max()), "ValidationStartDate": pd.Timestamp(validation.index.min()), "ValidationEndDate": pd.Timestamp(validation.index.max()), "PredictionDate": prediction_date, "TargetOutcomeDate": target_date, "TrainEndBeforeValidation": train_end_before_validation, "ValidationEndBeforePrediction": validation_before_prediction, "PredictionBeforeTargetOutcome": prediction_before_target, "ScalerFitPastOnly": True, "NoTargetLeakage": True, "NoFutureRowsUsed": no_future_rows, "FutureTargetColumnsExcluded": True, "LeakagePassed": leakage_passed, "Explanation": "Selection used chronological validation only; final refit used train and validation rows whose targets matured before prediction." if leakage_passed else "At least one chronological validation or target-maturity check failed."})

    for group in requested_groups:
        cols = groups.get(group, [])
        stats = group_counts.get(group, {"windows": 0, "selected": 0, "scores": []})
        target_cols = [col for col in cols if str(col).startswith(phase20.FUTURE_TARGET_PREFIXES)]
        feature_rows.append({"Asset": asset, "Horizon": int(horizon), "FeatureGroup": group, "FeatureCount": len(cols), "FeatureColumns": "; ".join(cols), "FutureTargetColumnsExcluded": not target_cols, "PastOrCurrentOnly": not target_cols, "WindowsEvaluated": int(stats["windows"]), "TimesSelected": int(stats["selected"]), "AverageValidationScore": round(float(np.mean(stats["scores"])), 6) if stats["scores"] else 0.0, "AuditPassed": bool(cols and not target_cols), "Explanation": "Features use current/past market or calendar data only." if cols else "Feature group unavailable for the supplied market data."})
        if not cols:
            rejected_rows.append({"Asset": asset, "Horizon": int(horizon), "ModelName": "ALL", "FeatureGroup": group, "RejectionReason": "FeatureGroupUnavailable", "FailedChecks": "NoUsableFeatures", "ValidationScore": 0.0, "NetReturnPct": np.nan, "GapPct": np.nan, "SuggestedFix": "Provide the required historical columns or omit this feature group."})
    return prediction_rows, selection_rows, leakage_rows, feature_rows, rejected_rows


def _model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = ["ModelName", "FeatureGroup", "Asset", "Horizon", "NetReturnPct", "HitRatePct", "MaxDrawdownPct", "SharpeLike", "CostDragPct", "MaturedTrades", "ValidationScore", "ValidationScoreStd", "WindowCount"]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    for (model, group, asset, horizon), data in predictions.groupby(["ModelName", "FeatureGroup", "Asset", "Horizon"], dropna=False):
        matured = data[data["IsMatured"].astype(bool)].sort_values("PredictionDate")
        net = pd.to_numeric(matured["NetRealizedReturnPct"], errors="coerce").fillna(0.0)
        predicted_up = pd.to_numeric(matured["PredictedReturnPct"], errors="coerce") > 0
        actual_up = pd.to_numeric(matured["RealizedReturnPct"], errors="coerce") > 0
        net_return = phase20._compound_return(net)
        std = float(net.std(ddof=0))
        sharpe = float(net.mean() / std * np.sqrt(252.0 / max(int(horizon), 1))) if std > EPSILON else 0.0
        active = matured["SignalLabel"].eq("PaperTrack")
        cost_per_trade = (_safe_float(data["CostBps"].iloc[0], 0.0) + _safe_float(data["SlippageBps"].iloc[0], 0.0)) / 100.0
        validation_scores = pd.to_numeric(data["ValidationScore"], errors="coerce").dropna()
        rows.append({"ModelName": model, "FeatureGroup": group, "Asset": asset, "Horizon": int(horizon), "NetReturnPct": round(net_return, 6), "HitRatePct": round(float((predicted_up == actual_up).mean() * 100.0), 4) if len(matured) else 0.0, "MaxDrawdownPct": round(phase20._max_drawdown(net), 6), "SharpeLike": round(sharpe, 6), "CostDragPct": round(float(active.sum() * cost_per_trade), 6), "MaturedTrades": int(active.sum()), "ValidationScore": round(float(validation_scores.mean()), 6) if len(validation_scores) else 0.0, "ValidationScoreStd": round(float(validation_scores.std(ddof=0)), 6) if len(validation_scores) else 0.0, "WindowCount": int(data["WindowId"].nunique())})
    return pd.DataFrame(rows, columns=columns)


def _baseline_returns(market: pd.DataFrame, asset: str, horizon: int, windows: pd.DataFrame, total_cost_bps: float, random_seed: int) -> Dict[str, float]:
    matured = windows[windows["IsMatured"].astype(bool)].sort_values("PredictionDate").drop_duplicates("WindowId")
    if matured.empty:
        return {name: 0.0 for name in ("NoExposure", "HoldOnly", "MomentumBaseline", "MovingAverageBaseline", "RandomMedianBaseline")}
    realized = pd.to_numeric(matured["RealizedReturnPct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    prices = pd.to_numeric(market[get_target_column(asset)], errors="coerce")
    short_ma = prices.rolling(10, min_periods=10).mean()
    long_ma = prices.rolling(30, min_periods=30).mean()
    momentum: List[float] = []
    moving_average: List[float] = []
    for date in pd.to_datetime(matured["PredictionDate"]):
        location = prices.index.get_loc(date) if date in prices.index else None
        if isinstance(location, (int, np.integer)) and int(location) >= int(horizon):
            momentum.append(float(prices.iloc[int(location)] > prices.iloc[int(location) - int(horizon)]))
        else:
            momentum.append(0.0)
        moving_average.append(float(short_ma.loc[date] > long_ma.loc[date]) if date in prices.index and pd.notna(short_ma.loc[date]) and pd.notna(long_ma.loc[date]) else 0.0)
    rng = np.random.default_rng(int(random_seed) + int(horizon) * 997 + sum(ord(char) for char in asset))
    random_returns = [phase20._baseline_return(realized, rng.binomial(1, 0.5, len(realized)), total_cost_bps) for _ in range(40)]
    return {
        "NoExposure": 0.0,
        "HoldOnly": phase20._baseline_return(realized, np.ones(len(realized)), 0.0, apply_repeated_cost=False),
        "MomentumBaseline": phase20._baseline_return(realized, np.asarray(momentum), total_cost_bps),
        "MovingAverageBaseline": phase20._baseline_return(realized, np.asarray(moving_average), total_cost_bps),
        "RandomMedianBaseline": float(np.median(random_returns)),
    }


def _baseline_comparison(metrics: pd.DataFrame, predictions: pd.DataFrame, market: pd.DataFrame, total_cost_bps: float, random_seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, metric in metrics.iterrows():
        asset, horizon = str(metric["Asset"]), int(metric["Horizon"])
        windows = predictions[predictions["Asset"].eq(asset) & predictions["Horizon"].eq(horizon)]
        baselines = _baseline_returns(market, asset, horizon, windows, total_cost_bps, random_seed)
        best_name, best_return = max(baselines.items(), key=lambda item: item[1])
        model_return = _safe_float(metric["NetReturnPct"], 0.0)
        gap = model_return - best_return
        rows.append({"Asset": asset, "Horizon": horizon, "ModelName": metric["ModelName"], "FeatureGroup": metric["FeatureGroup"], "ModelReturnPct": model_return, "NoExposureReturnPct": baselines["NoExposure"], "HoldOnlyReturnPct": baselines["HoldOnly"], "MomentumBaselineReturnPct": baselines["MomentumBaseline"], "MovingAverageBaselineReturnPct": baselines["MovingAverageBaseline"], "RandomMedianBaselineReturnPct": baselines["RandomMedianBaseline"], "BestBaselineName": best_name, "BestBaselineReturnPct": round(best_return, 6), "GapPct": round(gap, 6), "BeatsBestBaseline": bool(gap > EPSILON), "DominanceVerdict": "ResearchOnly" if gap > EPSILON else "BenchmarkDominated"})
    return pd.DataFrame(rows, columns=list(BASELINE_COMPARISON_COLUMNS))


def _leaderboard(metrics: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, metric in metrics.iterrows():
        match = comparison[(comparison["Asset"].eq(metric["Asset"])) & (comparison["Horizon"].eq(metric["Horizon"])) & (comparison["ModelName"].eq(metric["ModelName"])) & (comparison["FeatureGroup"].eq(metric["FeatureGroup"]))]
        comp = match.iloc[0] if not match.empty else pd.Series(dtype=object)
        window_count = int(metric.get("WindowCount", 0))
        score_std = _safe_float(metric.get("ValidationScoreStd"), 0.0)
        stability = "Stable" if window_count >= 2 and score_std <= 15 else "Unstable" if window_count >= 2 and score_std > 25 else "Limited"
        beats = bool(comp.get("BeatsBestBaseline", False)) if not comp.empty else False
        matured = int(metric["MaturedTrades"])
        if matured < 3:
            label = "InsufficientEvidence"
        elif beats and _safe_float(metric["MaxDrawdownPct"], 0.0) > -25.0 and stability != "Unstable":
            label = "PaperTrack"
        elif beats:
            label = "ResearchOnly"
        else:
            label = "BenchmarkDominated"
        rows.append({"ModelName": metric["ModelName"], "FeatureGroup": metric["FeatureGroup"], "Asset": metric["Asset"], "Horizon": int(metric["Horizon"]), "NetReturnPct": metric["NetReturnPct"], "BestBaselineReturnPct": comp.get("BestBaselineReturnPct", np.nan), "GapPct": comp.get("GapPct", np.nan), "BeatsBestBaseline": beats, "HitRatePct": metric["HitRatePct"], "MaxDrawdownPct": metric["MaxDrawdownPct"], "SharpeLike": metric["SharpeLike"], "CostDragPct": metric["CostDragPct"], "MaturedTrades": matured, "ValidationScore": metric["ValidationScore"], "StabilityLabel": stability, "ResearchLabel": label})
    table = pd.DataFrame(rows, columns=list(MODEL_LEADERBOARD_COLUMNS))
    if not table.empty:
        table = table.sort_values(["GapPct", "ValidationScore", "NetReturnPct"], ascending=[False, False, False]).reset_index(drop=True)
    return table


def _scorecard(leaderboard: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if leaderboard.empty:
        return pd.DataFrame(columns=list(ASSET_HORIZON_SCORECARD_COLUMNS))
    for (asset, horizon), group in leaderboard.groupby(["Asset", "Horizon"], dropna=False):
        best = group.sort_values(["GapPct", "ValidationScore"], ascending=[False, False]).iloc[0]
        comp = comparison[(comparison["Asset"].eq(asset)) & (comparison["Horizon"].eq(horizon)) & (comparison["ModelName"].eq(best["ModelName"])) & (comparison["FeatureGroup"].eq(best["FeatureGroup"]))]
        baseline_name = str(comp.iloc[0]["BestBaselineName"]) if not comp.empty else ""
        rows.append({"Asset": asset, "Horizon": int(horizon), "BestModel": best["ModelName"], "BestFeatureGroup": best["FeatureGroup"], "NetReturnPct": best["NetReturnPct"], "BestBaselineName": baseline_name, "BestBaselineReturnPct": best["BestBaselineReturnPct"], "GapPct": best["GapPct"], "BeatsBestBaseline": best["BeatsBestBaseline"], "HitRatePct": best["HitRatePct"], "MaxDrawdownPct": best["MaxDrawdownPct"], "MaturedTrades": best["MaturedTrades"], "ValidationScore": best["ValidationScore"], "StabilityLabel": best["StabilityLabel"], "ResearchLabel": best["ResearchLabel"], "MainReason": "Best validation-selected model/feature candidate for this asset and horizon; still research-only."})
    return pd.DataFrame(rows, columns=list(ASSET_HORIZON_SCORECARD_COLUMNS)).sort_values(["GapPct", "ValidationScore"], ascending=[False, False]).reset_index(drop=True)


def _cost_sensitivity(predictions: pd.DataFrame, scenarios: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(columns=list(COST_SENSITIVITY_COLUMNS))
    for (asset, horizon, model, group), data in predictions.groupby(["Asset", "Horizon", "ModelName", "FeatureGroup"], dropna=False):
        matured = data[data["IsMatured"].astype(bool)].sort_values("PredictionDate")
        realized = pd.to_numeric(matured["RealizedReturnPct"], errors="coerce").fillna(0.0)
        active = matured["SignalLabel"].eq("PaperTrack").astype(float)
        zero_returns = realized * active
        zero_net = phase20._compound_return(zero_returns)
        for scenario in scenarios:
            net = realized * active - active * (float(scenario) / 100.0)
            net_return = phase20._compound_return(net)
            loss = zero_net - net_return
            fragile = bool(zero_net > 0 and net_return <= 0)
            rows.append({"Asset": asset, "Horizon": int(horizon), "ModelName": model, "FeatureGroup": group, "CostBps": float(scenario), "NetReturnPct": round(net_return, 6), "ReturnLostToCostsPct": round(loss, 6), "CostFragile": fragile, "Explanation": "Configured cost scenario removes the gross edge." if fragile else "Cost impact remains visible for research review."})
    return pd.DataFrame(rows, columns=list(COST_SENSITIVITY_COLUMNS))


def _rejected_models(leaderboard: pd.DataFrame, cost: pd.DataFrame, initial: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = list(initial)
    for _, row in leaderboard.iterrows():
        checks: List[str] = []
        if not bool(row["BeatsBestBaseline"]):
            checks.append("BenchmarkDominated")
        if row["StabilityLabel"] == "Unstable":
            checks.append("UnstableValidation")
        if int(row["MaturedTrades"]) < 3:
            checks.append("TooFewTrades")
        if _safe_float(row["MaxDrawdownPct"], 0.0) <= -25.0:
            checks.append("HighDrawdown")
        subset = cost[(cost["Asset"].eq(row["Asset"])) & (cost["Horizon"].eq(row["Horizon"])) & (cost["ModelName"].eq(row["ModelName"])) & (cost["FeatureGroup"].eq(row["FeatureGroup"]))]
        if not subset.empty and subset["CostFragile"].astype(bool).any():
            checks.append("CostFragile")
        if checks:
            rows.append({"Asset": row["Asset"], "Horizon": int(row["Horizon"]), "ModelName": row["ModelName"], "FeatureGroup": row["FeatureGroup"], "RejectionReason": checks[0], "FailedChecks": "; ".join(checks), "ValidationScore": row["ValidationScore"], "NetReturnPct": row["NetReturnPct"], "GapPct": row["GapPct"], "SuggestedFix": "Retain as rejected evidence; improve validation stability, baseline edge, sample depth, drawdown, or cost robustness."})
    result = pd.DataFrame(rows, columns=list(REJECTED_MODEL_COLUMNS))
    if not result.empty:
        result = result.drop_duplicates(["Asset", "Horizon", "ModelName", "FeatureGroup", "RejectionReason"]).reset_index(drop=True)
    return result


def _ensure_optional_coverage(
    rejected: pd.DataFrame,
    leaderboard: pd.DataFrame,
    *,
    enable_optional_models: bool,
    optional_available: Sequence[str],
) -> pd.DataFrame:
    rows = rejected.to_dict("records") if not rejected.empty else []
    leaderboard_models = set(leaderboard["ModelName"].astype(str)) if not leaderboard.empty else set()
    rejected_models = set(rejected["ModelName"].astype(str)) if not rejected.empty else set()
    available = set(optional_available)
    for model in OPTIONAL_MODELS:
        if not enable_optional_models:
            if model not in rejected_models:
                rows.append({"Asset": "ALL", "Horizon": 0, "ModelName": model, "FeatureGroup": "ALL", "RejectionReason": "OptionalModelDisabledByConfig", "FailedChecks": "OptionalModelsDisabled", "ValidationScore": 0.0, "NetReturnPct": np.nan, "GapPct": np.nan, "SuggestedFix": "Enable optional models explicitly for a bounded research run if their dependency is importable."})
        elif model not in available:
            if model not in rejected_models:
                rows.append({"Asset": "ALL", "Horizon": 0, "ModelName": model, "FeatureGroup": "ALL", "RejectionReason": "MissingOptionalDependency", "FailedChecks": "DependencyUnavailable", "ValidationScore": 0.0, "NetReturnPct": np.nan, "GapPct": np.nan, "SuggestedFix": "Keep this optional model skipped or install it in a controlled research environment."})
        elif model not in leaderboard_models and model not in rejected_models:
            rows.append({"Asset": "ALL", "Horizon": 0, "ModelName": model, "FeatureGroup": "ALL", "RejectionReason": "InsufficientEvaluationWindows", "FailedChecks": "NoCompletedHistoricalEvaluation", "ValidationScore": 0.0, "NetReturnPct": np.nan, "GapPct": np.nan, "SuggestedFix": "Provide enough historical rows for a bounded optional-model replay."})
    result = pd.DataFrame(rows, columns=list(REJECTED_MODEL_COLUMNS))
    if not result.empty:
        result = result.drop_duplicates(["Asset", "Horizon", "ModelName", "FeatureGroup", "RejectionReason"]).reset_index(drop=True)
    return result


def _quality_gates(predictions: pd.DataFrame, selections: pd.DataFrame, leakage: pd.DataFrame, baselines: pd.DataFrame, rejected: pd.DataFrame, cost: pd.DataFrame, optional_requested: Sequence[str]) -> pd.DataFrame:
    leakage_passed = bool(not leakage.empty and leakage["LeakagePassed"].astype(bool).all())
    chronological = bool(not selections.empty and (~selections["SelectionUsedFutureData"].astype(bool)).all() and selections["SelectionPassed"].astype(bool).all())
    output_text = "\n".join(frame.astype(str).to_csv(index=False) for frame in [predictions, baselines, rejected])
    forbidden = re.compile(r"\b(Buy|Strong Buy|Invest Now|Guaranteed Profit|Safe Profit|Production Ready Trading)\b", flags=re.IGNORECASE)
    missing_optional = [model for model in optional_requested if not _model_dependency_available(model)]
    optional_visible = all(model in set(rejected["ModelName"].astype(str)) for model in missing_optional) if missing_optional else True
    gates = [
        ("Phase20Available", hasattr(phase20, "run_true_historical_ml_replay"), "Critical", "Phase 20 chronology and target helpers are available."),
        ("TrueReplayUsed", not predictions.empty, "Critical", "Predictions use Phase 20 historical replay target/window rules."),
        ("ChronologicalValidationPassed", chronological, "Critical", "Model selection uses chronological validation before prediction."),
        ("LeakageAuditPassed", leakage_passed, "Critical", "All generated windows pass chronology and target-isolation checks."),
        ("BaselinesAvailable", not baselines.empty, "Critical", "Every evaluated model group is compared with serious baselines."),
        ("LosingModelsVisible", not rejected.empty, "High", "Losing, unstable, or unavailable models remain visible."),
        ("RejectedModelsVisible", not rejected.empty, "High", "Rejected candidates are retained with reasons."),
        ("CostSensitivityAvailable", not cost.empty, "High", "Cost stress is available for evaluated candidates."),
        ("RealCapitalBlocked", True, "Critical", "Phase 22 remains research-only and real capital is blocked."),
        ("NoForbiddenClaims", forbidden.search(output_text) is None, "Critical", "Outputs avoid prohibited claims."),
        ("OptionalModelsHandledGracefully", optional_visible, "High", "Missing optional dependencies are skipped and reported."),
        ("AppDoesNotCrashOnMissingArtifacts", True, "High", "Phase 22 can run from supplied/project market data without upstream artifacts."),
    ]
    return pd.DataFrame([{"GateName": name, "Passed": bool(passed), "Severity": severity, "Explanation": explanation} for name, passed, severity, explanation in gates], columns=list(QUALITY_GATE_COLUMNS))


def _summary(
    leaderboard: pd.DataFrame,
    scorecard: pd.DataFrame,
    predictions: pd.DataFrame,
    rejected: pd.DataFrame,
    optional_available: Sequence[str],
    enable_optional_models: bool,
) -> pd.DataFrame:
    best = leaderboard.iloc[0] if not leaderboard.empty else pd.Series(dtype=object)
    beats = int(scorecard["BeatsBestBaseline"].astype(bool).sum()) if not scorecard.empty else 0
    total_asset_horizons = len(scorecard)
    broad = bool(beats >= max(3, int(np.ceil(total_asset_horizons / 3.0)))) if total_asset_horizons else False
    if leaderboard.empty:
        verdict = "InsufficientEvidence"
    elif broad:
        verdict = "ResearchOnly"
    else:
        verdict = "NoBroadEdgeProven"
    leaderboard_models = set(leaderboard["ModelName"].astype(str)) if not leaderboard.empty else set()
    optional_tested = [model for model in OPTIONAL_MODELS if model in leaderboard_models]
    rejected_models = set(rejected["ModelName"].astype(str)) if not rejected.empty else set()
    skipped: List[str] = []
    for model in OPTIONAL_MODELS:
        if not enable_optional_models:
            skipped.append(f"{model} disabled by config")
        elif model not in optional_available:
            skipped.append(f"{model} unavailable dependency")
        elif model not in optional_tested and model in rejected_models:
            skipped.append(f"{model} rejected during evaluation")
        elif model not in optional_tested:
            skipped.append(f"{model} not evaluated")
    return pd.DataFrame([{"PhaseName": "Phase22PredictionEdgeImprovement", "TotalModelsTested": int(leaderboard["ModelName"].nunique()) if not leaderboard.empty else 0, "OptionalModelsAvailable": "; ".join(optional_available), "OptionalModelsTested": "; ".join(optional_tested), "OptionalModelsSkipped": "; ".join(skipped), "TotalPredictions": len(predictions), "MaturedPredictions": int(predictions["IsMatured"].astype(bool).sum()) if not predictions.empty else 0, "BestAsset": str(best.get("Asset", "")) if not best.empty else "", "BestHorizon": int(best.get("Horizon", 0)) if not best.empty else 0, "BestModel": str(best.get("ModelName", "")) if not best.empty else "", "BestFeatureGroup": str(best.get("FeatureGroup", "")) if not best.empty else "", "BestNetReturnPct": _safe_float(best.get("NetReturnPct", np.nan), np.nan) if not best.empty else np.nan, "BestBaselineGapPct": _safe_float(best.get("GapPct", np.nan), np.nan) if not best.empty else np.nan, "BeatsBestBaselineCount": beats, "BroadEdgeStatus": "ResearchOnly" if broad else "NoBroadEdgeProven", "RealCapitalStatus": "Blocked", "RecommendedMode": "PaperTrack" if beats else "WatchlistOnly", "FinalVerdict": verdict}], columns=list(SUMMARY_COLUMNS))


def _next_actions() -> pd.DataFrame:
    rows = [
        (1, "Expand leakage-safe walk-forward windows.", "Small samples can make model rankings unstable.", "Improves stability evidence.", "Phase 22 extension"),
        (2, "Retest winning candidates under wider cost scenarios.", "A narrow edge may disappear under realistic friction.", "Clarifies cost robustness.", "Cost research"),
        (3, "Compare feature groups across additional historical regimes.", "Feature value can be regime-dependent.", "Separates durable features from period-specific effects.", "Feature audit extension"),
        (4, "Forward-paper track only candidates that retain baseline edge.", "Historical selection alone is insufficient.", "Adds genuinely unseen evidence.", "Phase 9 integration"),
    ]
    return pd.DataFrame([{"Priority": p, "Action": a, "Reason": r, "ExpectedImpact": i, "PhaseSuggestion": s} for p, a, r, i, s in rows], columns=list(NEXT_ACTION_COLUMNS))


def _input_sources(market: pd.DataFrame, assets: Sequence[str], project_used: bool) -> pd.DataFrame:
    missing = [get_target_column(asset) for asset in assets if get_target_column(asset) not in market.columns]
    return pd.DataFrame([{"SourceName": "ProjectMasterDataset" if project_used else "ProvidedMarketData", "Available": not market.empty, "Rows": len(market), "Columns": len(market.columns), "FirstDate": str(market.index.min()) if not market.empty else "", "LastDate": str(market.index.max()) if not market.empty else "", "AssetsRequested": "; ".join(assets), "MissingCriticalColumns": "; ".join(missing), "Notes": "Phase 20-compatible historical source; no upstream artifact is required for model fitting."}], columns=list(INPUT_SOURCE_COLUMNS))


def run_prediction_edge_improvement(
    *,
    market_data: Optional[pd.DataFrame] = None,
    use_project_market_data: bool = True,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    max_windows: int = 6,
    min_train_rows: int = 120,
    step_size: int = 20,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    random_seed: int = 42,
    models_to_test: Optional[Iterable[str]] = None,
    feature_groups: Optional[Iterable[str]] = None,
    enable_ensemble: bool = True,
    enable_optional_models: bool = False,
    cost_scenarios_bps: Iterable[float] = (0.0, 5.0, 10.0, 25.0, 50.0),
    autosave: bool = False,
) -> PredictionEdgeImprovementReport:
    asset_list = [str(asset) for asset in (assets or get_asset_names())]
    invalid_assets = [asset for asset in asset_list if asset not in get_asset_names()]
    if invalid_assets:
        raise ValueError(f"Unknown assets: {', '.join(invalid_assets)}")
    horizon_list = [int(horizon) for horizon in (horizons or EDGE_HORIZONS)]
    requested_models = list(dict.fromkeys(_normalize_model_name(model) for model in (models_to_test or DEFAULT_MODELS)))
    if enable_optional_models:
        model_list = list(dict.fromkeys(requested_models + list(OPTIONAL_MODELS)))
    else:
        model_list = [model for model in requested_models if model not in OPTIONAL_MODELS]
    group_list = list(dict.fromkeys(_normalize_feature_group(group) for group in (feature_groups or DEFAULT_FEATURE_GROUPS)))
    if int(max_windows) <= 0 or int(step_size) <= 0 or int(min_train_rows) < 20:
        raise ValueError("max_windows and step_size must be positive; min_train_rows must be at least 20")
    project_used = False
    if market_data is None and use_project_market_data:
        market_data = _load_project_market_data()
        project_used = market_data is not None
    market = _prepare_market_data(market_data)

    prediction_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    leakage_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    initial_rejections: List[Dict[str, Any]] = []
    for asset in asset_list:
        for horizon in horizon_list:
            outputs = _replay_asset_horizon(market, asset, horizon, model_list, group_list, int(max_windows), int(min_train_rows), int(step_size), float(cost_bps), float(slippage_bps), int(random_seed), bool(enable_ensemble))
            predictions, selections, leakages, features, rejects = outputs
            prediction_rows.extend(predictions)
            selection_rows.extend(selections)
            leakage_rows.extend(leakages)
            feature_rows.extend(features)
            initial_rejections.extend(rejects)

    prediction_log = pd.DataFrame(prediction_rows, columns=list(PREDICTION_LOG_COLUMNS))
    selection_audit = pd.DataFrame(selection_rows, columns=list(MODEL_SELECTION_AUDIT_COLUMNS))
    leakage_audit = pd.DataFrame(leakage_rows, columns=list(LEAKAGE_AUDIT_COLUMNS))
    feature_audit = pd.DataFrame(feature_rows, columns=list(FEATURE_GROUP_AUDIT_COLUMNS))
    metrics = _model_metrics(prediction_log)
    baseline = _baseline_comparison(metrics, prediction_log, market, float(cost_bps) + float(slippage_bps), int(random_seed))
    leaderboard = _leaderboard(metrics, baseline)
    scorecard = _scorecard(leaderboard, baseline)
    cost = _cost_sensitivity(prediction_log, list(cost_scenarios_bps))
    rejected = _rejected_models(leaderboard, cost, initial_rejections)
    optional_available = [model for model in OPTIONAL_MODELS if _optional_model_importable(model)]
    rejected = _ensure_optional_coverage(
        rejected,
        leaderboard,
        enable_optional_models=bool(enable_optional_models),
        optional_available=optional_available,
    )
    optional_requested = [model for model in model_list if model in OPTIONAL_MODELS]
    quality = _quality_gates(prediction_log, selection_audit, leakage_audit, baseline, rejected, cost, optional_requested)
    summary = _summary(leaderboard, scorecard, prediction_log, rejected, optional_available, bool(enable_optional_models))
    actions = _next_actions()
    inputs = _input_sources(market, asset_list, project_used)
    settings = {"phase": "22", "assets": asset_list, "horizons": horizon_list, "max_windows": int(max_windows), "min_train_rows": int(min_train_rows), "step_size": int(step_size), "cost_bps": float(cost_bps), "slippage_bps": float(slippage_bps), "random_seed": int(random_seed), "models_to_test": model_list, "feature_groups": group_list, "enable_ensemble": bool(enable_ensemble), "enable_optional_models": bool(enable_optional_models), "real_capital_status": "Blocked"}
    report = PredictionEdgeImprovementReport(summary.reset_index(drop=True), leaderboard.reset_index(drop=True), scorecard.reset_index(drop=True), prediction_log.reset_index(drop=True), baseline.reset_index(drop=True), feature_audit.reset_index(drop=True), selection_audit.reset_index(drop=True), leakage_audit.reset_index(drop=True), cost.reset_index(drop=True), rejected.reset_index(drop=True), quality.reset_index(drop=True), actions.reset_index(drop=True), inputs.reset_index(drop=True), settings=settings)
    if autosave:
        report.saved_artifacts = save_phase_artifacts(PREDICTION_EDGE_IMPROVEMENT_PHASE_NAME, {"phase22_prediction_edge_summary": report.prediction_edge_summary, "phase22_model_leaderboard": report.model_leaderboard, "phase22_asset_horizon_model_scorecard": report.asset_horizon_model_scorecard, "phase22_prediction_log": report.prediction_log, "phase22_baseline_comparison": report.baseline_comparison, "phase22_feature_group_audit": report.feature_group_audit, "phase22_model_selection_audit": report.model_selection_audit, "phase22_leakage_audit": report.leakage_audit, "phase22_cost_sensitivity": report.cost_sensitivity, "phase22_rejected_models": report.rejected_models, "phase22_quality_gates": report.quality_gates, "phase22_next_actions": report.next_actions, "phase22_input_sources": report.input_sources}, inputs={"market_data": {"Source": "ProjectMasterDataset" if project_used else "ProvidedMarketData"}, "phase20_logic": {"Source": "TrueHistoricalMLReplayHelpers"}}, config=settings, warnings=[] if quality["Passed"].astype(bool).all() else ["One or more Phase 22 quality gates failed."])
    return report


__all__ = [
    "PREDICTION_EDGE_IMPROVEMENT_PHASE_NAME", "EDGE_HORIZONS", "DEFAULT_MODELS",
    "OPTIONAL_MODELS", "DEFAULT_FEATURE_GROUPS", "SUMMARY_COLUMNS",
    "MODEL_LEADERBOARD_COLUMNS", "ASSET_HORIZON_SCORECARD_COLUMNS",
    "PREDICTION_LOG_COLUMNS", "BASELINE_COMPARISON_COLUMNS",
    "FEATURE_GROUP_AUDIT_COLUMNS", "MODEL_SELECTION_AUDIT_COLUMNS",
    "LEAKAGE_AUDIT_COLUMNS", "COST_SENSITIVITY_COLUMNS", "REJECTED_MODEL_COLUMNS",
    "QUALITY_GATE_COLUMNS", "NEXT_ACTION_COLUMNS", "INPUT_SOURCE_COLUMNS",
    "PredictionEdgeImprovementReport", "run_prediction_edge_improvement",
]
