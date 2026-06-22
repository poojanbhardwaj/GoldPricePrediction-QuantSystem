# src/direct_forecast_models.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.asset_config import get_asset_names, get_target_column
from src.feature_engineering import FeatureEngineer
from src.feature_intelligence import add_phase5_feature_intelligence
from src.indicators import TechnicalIndicators


DIRECT_FORECAST_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)
FUTURE_TARGET_PREFIXES: Tuple[str, ...] = (
    "future_return_",
    "future_direction_",
    "future_realized_vol_",
)
SCAN_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "BestModel",
    "Rows",
    "Features",
    "TestRows",
    "Best_RMSE_vs_Naive_%",
    "Best_Direction_vs_Baseline_%",
    "Best_DirectionalAccuracy",
    "Best_OverfitGap_%",
    "Best_TrustScore",
    "Best_Verdict",
    "Best_RMSE_Return",
    "Best_Naive_RMSE_Return",
    "Best_DirectionBaselineAccuracy",
    "Best_AUC",
    "Best_F1",
)


@dataclass
class DirectForecastDataset:
    asset: str
    target_col: str
    horizon: int
    return_target_col: str
    direction_target_col: str
    volatility_target_col: str
    df_model: pd.DataFrame
    feature_cols: List[str]
    feature_scaler: Any
    return_scaler: Any
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_return_train: np.ndarray
    y_return_val: np.ndarray
    y_return_test: np.ndarray
    y_return_train_scaled: np.ndarray
    y_return_val_scaled: np.ndarray
    y_return_test_scaled: np.ndarray
    y_direction_train: np.ndarray
    y_direction_val: np.ndarray
    y_direction_test: np.ndarray
    train_index: pd.DatetimeIndex
    val_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    dropped_tail_rows: int


@dataclass
class DirectForecastReport:
    asset: str
    target_col: str
    horizon: int
    model_depth: str
    use_phase5_features: bool
    rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    feature_count: int
    leaderboard: pd.DataFrame
    baseline_board: pd.DataFrame
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    dataset: Optional[DirectForecastDataset] = None


@dataclass
class DirectForecastSignalOutput:
    asset: str
    target_col: str
    horizon: int
    model_depth: str
    use_phase5_features: bool
    model_name: str
    probabilities_up_test: np.ndarray
    predicted_direction_test: np.ndarray
    actual_direction_test: np.ndarray
    actual_return_test: np.ndarray
    test_index: pd.DatetimeIndex
    direction_baseline_accuracy: float
    feature_cols: List[str]
    feature_leakage_columns: List[str]
    leaderboard: pd.DataFrame
    baseline_board: pd.DataFrame
    report: DirectForecastReport


@dataclass
class AssetHorizonScanReport:
    asset_horizon_summary: pd.DataFrame
    status_counts: Dict[str, int]
    top_promising: pd.DataFrame
    worst_failed: pd.DataFrame
    errors: pd.DataFrame
    settings: Dict[str, Any]
    reports: Dict[Tuple[str, int], DirectForecastReport] = field(default_factory=dict)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _target_prefix(target_col: str) -> str:
    return str(target_col).replace("_Close", "")


def _is_future_target_col(col: str) -> bool:
    return str(col).startswith(FUTURE_TARGET_PREFIXES)


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float).flatten()
    pred = np.asarray(pred, dtype=float).flatten()
    mask = np.isfinite(actual) & np.isfinite(pred)
    if int(mask.sum()) == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(actual[mask], pred[mask])))


def build_direct_feature_frame(
    raw_df: pd.DataFrame,
    *,
    target_col: str,
    use_phase5_features: bool = True,
) -> pd.DataFrame:
    """
    Build the same current/past feature frame used by the research matrix.

    Future horizon labels are deliberately added later by
    add_direct_forecast_targets(), never inside feature generation.
    """
    if target_col not in raw_df.columns:
        raise ValueError(f"Target column {target_col!r} not found in raw data")

    prefix = _target_prefix(target_col)
    out = TechnicalIndicators(prefix=prefix).add_all(raw_df.copy())
    out = out.sort_index().ffill()
    out = FeatureEngineer(target_col=target_col).build_features(out)

    if use_phase5_features:
        out = add_phase5_feature_intelligence(out, target_col=target_col)

    return out


def add_direct_forecast_targets(
    df: pd.DataFrame,
    *,
    target_col: str,
    horizon: int,
    include_realized_volatility: bool = True,
) -> pd.DataFrame:
    """
    Add direct future return/direction targets for one trading horizon.

    Target formula:
        future_return_h = log(price.shift(-h) / price)

    The shift(-h) columns created here are targets only. They must be excluded
    from feature_cols before model training.
    """
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be a positive integer")
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found")

    out = df.sort_index().copy()
    price = pd.to_numeric(out[target_col], errors="coerce")

    return_col = f"future_return_{h}d"
    direction_col = f"future_direction_{h}d"
    vol_col = f"future_realized_vol_{h}d"

    future_return = np.log(price.shift(-h) / price)
    out[return_col] = future_return
    out[direction_col] = future_return > 0.0

    if include_realized_volatility:
        daily_log_return = np.log(price / price.shift(1))
        out[vol_col] = daily_log_return.rolling(h, min_periods=h).std().shift(-h) * np.sqrt(252.0)

    return out


def _select_direct_feature_cols(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    return [c for c in numeric_cols if not _is_future_target_col(c)]


def make_direct_forecast_dataset(
    feature_df: pd.DataFrame,
    *,
    asset: str,
    target_col: str,
    horizon: int,
    test_size: float = 0.20,
    val_size: float = 0.10,
) -> DirectForecastDataset:
    """
    Convert a current/past feature frame into a direct horizon dataset.

    Leakage controls:
    - shift(-h) is used only in target columns.
    - the final h rows are dropped because their future labels are unknown.
    - every future_* target column is excluded from feature_cols.
    - train/val/test splits preserve chronological order.
    - feature and return scalers are fit on train only.
    """
    h = int(horizon)
    target_df = add_direct_forecast_targets(feature_df, target_col=target_col, horizon=h)
    if len(target_df) <= h:
        raise ValueError("Not enough rows to drop the final horizon labels")

    return_col = f"future_return_{h}d"
    direction_col = f"future_direction_{h}d"
    vol_col = f"future_realized_vol_{h}d"

    known_df = target_df.iloc[:-h].copy()
    feature_cols = _select_direct_feature_cols(known_df)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for direct forecast models")

    known_df = known_df.replace([np.inf, -np.inf], np.nan)
    required_cols = feature_cols + [return_col, direction_col]
    model_df = known_df.dropna(subset=required_cols).copy()

    zero_var = [c for c in feature_cols if model_df[c].std(ddof=0) < 1e-12]
    if zero_var:
        feature_cols = [c for c in feature_cols if c not in zero_var]
        model_df = model_df.drop(columns=zero_var)

    n = len(model_df)
    if n < 80:
        raise ValueError(f"Not enough rows after direct target alignment: {n}")

    test_n = int(n * float(test_size))
    val_n = int(n * float(val_size))
    train_n = n - val_n - test_n
    if train_n <= 0 or val_n < 0 or test_n <= 0:
        raise ValueError("Invalid chronological split sizes")

    X_raw = model_df[feature_cols].astype(float).to_numpy()
    y_return = model_df[return_col].astype(float).to_numpy()
    y_direction = model_df[direction_col].astype(bool).astype(int).to_numpy()

    X_train_raw = X_raw[:train_n]
    X_val_raw = X_raw[train_n: train_n + val_n]
    X_test_raw = X_raw[train_n + val_n:]

    y_return_train = y_return[:train_n]
    y_return_val = y_return[train_n: train_n + val_n]
    y_return_test = y_return[train_n + val_n:]

    y_direction_train = y_direction[:train_n]
    y_direction_val = y_direction[train_n: train_n + val_n]
    y_direction_test = y_direction[train_n + val_n:]

    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train_raw)
    X_val = feature_scaler.transform(X_val_raw) if len(X_val_raw) else np.array([]).reshape(0, X_train.shape[1])
    X_test = feature_scaler.transform(X_test_raw)

    return_scaler = StandardScaler()
    y_return_train_scaled = return_scaler.fit_transform(y_return_train.reshape(-1, 1)).flatten()
    y_return_val_scaled = return_scaler.transform(y_return_val.reshape(-1, 1)).flatten() if len(y_return_val) else np.array([])
    y_return_test_scaled = return_scaler.transform(y_return_test.reshape(-1, 1)).flatten()

    index = pd.DatetimeIndex(model_df.index)
    return DirectForecastDataset(
        asset=asset,
        target_col=target_col,
        horizon=h,
        return_target_col=return_col,
        direction_target_col=direction_col,
        volatility_target_col=vol_col,
        df_model=model_df,
        feature_cols=feature_cols,
        feature_scaler=feature_scaler,
        return_scaler=return_scaler,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_return_train=y_return_train,
        y_return_val=y_return_val,
        y_return_test=y_return_test,
        y_return_train_scaled=y_return_train_scaled,
        y_return_val_scaled=y_return_val_scaled,
        y_return_test_scaled=y_return_test_scaled,
        y_direction_train=y_direction_train,
        y_direction_val=y_direction_val,
        y_direction_test=y_direction_test,
        train_index=index[:train_n],
        val_index=index[train_n: train_n + val_n],
        test_index=index[train_n + val_n:],
        dropped_tail_rows=h,
    )


def direct_forecast_baseline_board(dataset: DirectForecastDataset) -> pd.DataFrame:
    """Return honest return and direction baselines for the direct horizon."""
    rows: List[Dict[str, Any]] = []

    y_return = np.asarray(dataset.y_return_test, dtype=float)
    zero_pred = np.zeros_like(y_return)
    rows.append(
        {
            "Baseline": "Zero Return baseline",
            "Type": "Return",
            "RMSE_Return": round(_rmse(y_return, zero_pred), 8),
            "DirectionalAccuracy": np.nan,
            "F1": np.nan,
            "AUC": np.nan,
        }
    )

    y_dir = np.asarray(dataset.y_direction_test, dtype=int)
    train_dir = np.asarray(dataset.y_direction_train, dtype=int)
    majority_class = int(np.mean(train_dir) >= 0.5) if len(train_dir) else 1

    direction_baselines = [
        ("Majority Train Direction baseline", np.full_like(y_dir, majority_class)),
        ("Always Up baseline", np.ones_like(y_dir)),
        ("Always Down baseline", np.zeros_like(y_dir)),
    ]

    for name, pred in direction_baselines:
        rows.append(
            {
                "Baseline": name,
                "Type": "Direction",
                "RMSE_Return": np.nan,
                "DirectionalAccuracy": round(float(accuracy_score(y_dir, pred) * 100.0), 2) if len(y_dir) else np.nan,
                "F1": round(float(f1_score(y_dir, pred, zero_division=0) * 100.0), 2) if len(y_dir) else np.nan,
                "AUC": np.nan,
            }
        )

    return pd.DataFrame(rows)


def _direction_baseline_accuracy(baseline_board: pd.DataFrame) -> float:
    if baseline_board.empty:
        return np.nan
    direction_rows = baseline_board[baseline_board["Type"].eq("Direction")]
    if direction_rows.empty:
        return np.nan
    return _safe_float(direction_rows["DirectionalAccuracy"].max())


def _model_specs(model_depth: str, random_state: int = 42) -> List[Tuple[str, Any, Any]]:
    key = str(model_depth).lower().strip()
    if key not in {"fast", "core"}:
        raise ValueError("model_depth must be 'fast' or 'core'")

    specs: List[Tuple[str, Any, Any]] = [
        (
            "Linear Regression + Logistic Direction",
            LinearRegression(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        ),
        (
            "Decision Tree Direct",
            DecisionTreeRegressor(max_depth=6, min_samples_leaf=12, random_state=random_state),
            DecisionTreeClassifier(max_depth=6, min_samples_leaf=12, class_weight="balanced", random_state=random_state),
        ),
    ]

    if key == "core":
        specs.extend(
            [
                (
                    "Random Forest Direct",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=8,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "Gradient Boosting Direct",
                    GradientBoostingRegressor(random_state=random_state),
                    GradientBoostingClassifier(random_state=random_state),
                ),
            ]
        )

    return specs


def _inverse_return_scale(dataset: DirectForecastDataset, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1, 1)
    return dataset.return_scaler.inverse_transform(arr).flatten()


def _direction_predictions(model: Any, dataset: DirectForecastDataset) -> Tuple[np.ndarray, np.ndarray]:
    y_train = np.asarray(dataset.y_direction_train, dtype=int)
    unique = np.unique(y_train)

    if len(unique) < 2:
        proba = np.full(len(dataset.y_direction_test), float(unique[0]))
        return (proba >= 0.5).astype(int), proba

    model.fit(dataset.X_train, y_train)

    if hasattr(model, "predict_proba"):
        proba_raw = model.predict_proba(dataset.X_test)
        if getattr(proba_raw, "ndim", 1) == 2 and proba_raw.shape[1] >= 2:
            proba = np.asarray(proba_raw[:, 1], dtype=float)
        else:
            proba = np.asarray(proba_raw, dtype=float).flatten()
        pred = (proba >= 0.5).astype(int)
    else:
        pred = np.asarray(model.predict(dataset.X_test), dtype=int).flatten()
        proba = pred.astype(float)

    return pred, proba


def direct_forecast_trust_score(
    *,
    rmse_vs_naive_pct: float,
    directional_accuracy: float,
    direction_vs_baseline_pct: float,
    overfit_gap_pct: float,
    f1: float = np.nan,
    auc: float = np.nan,
) -> Tuple[float, str]:
    """Conservative trust score for direct horizon models."""
    rmse_edge = _safe_float(rmse_vs_naive_pct)
    dir_acc = _safe_float(directional_accuracy)
    dir_edge = _safe_float(direction_vs_baseline_pct)
    overfit_gap = _safe_float(overfit_gap_pct)
    f1_val = _safe_float(f1)
    auc_val = _safe_float(auc)

    score = 0.0
    if np.isfinite(rmse_edge):
        if rmse_edge >= 0.0:
            score += 5.0 + float(np.clip(rmse_edge / 10.0, 0, 1) * 35.0)
        else:
            score += float(np.clip((rmse_edge + 2.0) / 2.0, 0, 1) * 5.0)

    if np.isfinite(dir_edge) and dir_edge > 0.0:
        score += float(np.clip(dir_edge / 8.0, 0, 1) * 25.0)

    if np.isfinite(dir_acc) and dir_acc > 50.0:
        score += float(np.clip((dir_acc - 50.0) / 8.0, 0, 1) * 10.0)

    if np.isfinite(overfit_gap):
        score += float(np.clip((100.0 - overfit_gap) / 75.0, 0, 1) * 20.0)

    if np.isfinite(f1_val):
        score += float(np.clip(f1_val / 60.0, 0, 1) * 3.0)
    if np.isfinite(auc_val):
        score += float(np.clip((auc_val - 50.0) / 15.0, 0, 1) * 2.0)

    if np.isfinite(rmse_edge):
        if rmse_edge <= -100.0:
            score = min(score, 5.0)
        elif rmse_edge <= -25.0:
            score = min(score, 10.0)
        elif rmse_edge < 0.0:
            score = min(score, 40.0)

    if np.isfinite(dir_edge) and dir_edge <= 0.0:
        score = min(score, 35.0)
    if np.isfinite(dir_acc) and dir_acc < 50.0:
        score = min(score, 30.0)

    if np.isfinite(overfit_gap):
        if overfit_gap > 200.0:
            score = min(score, 5.0)
        elif overfit_gap > 100.0:
            score = min(score, 10.0)
        elif overfit_gap > 75.0:
            score = min(score, 35.0)

    high_quality = (
        np.isfinite(rmse_edge) and rmse_edge >= 3.0
        and np.isfinite(dir_edge) and dir_edge >= 3.0
        and np.isfinite(dir_acc) and dir_acc >= 53.0
        and np.isfinite(overfit_gap) and overfit_gap <= 40.0
    )
    medium_quality = (
        np.isfinite(rmse_edge) and rmse_edge >= 0.0
        and np.isfinite(dir_edge) and dir_edge > 0.0
        and np.isfinite(dir_acc) and dir_acc >= 50.0
        and np.isfinite(overfit_gap) and overfit_gap <= 75.0
    )

    if not high_quality:
        score = min(score, 74.99)
    if not medium_quality:
        score = min(score, 54.99)

    score = round(float(np.clip(score, 0.0, 100.0)), 2)
    if score >= 75.0:
        verdict = "High direct-horizon candidate"
    elif score >= 55.0:
        verdict = "Medium direct-horizon candidate"
    elif score >= 35.0:
        verdict = "Low trust / research only"
    else:
        verdict = "Do not trust for signals"

    return score, verdict


def _warning_text(rmse_vs_naive: float, direction_vs_baseline: float, overfit_gap: float) -> str:
    warnings: List[str] = []
    if np.isfinite(rmse_vs_naive) and rmse_vs_naive < 0.0:
        warnings.append("fails return baseline")
    if np.isfinite(direction_vs_baseline) and direction_vs_baseline <= 0.0:
        warnings.append("fails direction baseline")
    if np.isfinite(overfit_gap) and overfit_gap > 100.0:
        warnings.append("overfit gap >100%")
    return "; ".join(warnings)


def train_direct_forecast_models(
    dataset: DirectForecastDataset,
    *,
    model_depth: str = "fast",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_board = direct_forecast_baseline_board(dataset)
    naive_rmse = _safe_float(
        baseline_board.loc[baseline_board["Baseline"].eq("Zero Return baseline"), "RMSE_Return"].iloc[0]
    )
    direction_baseline_acc = _direction_baseline_accuracy(baseline_board)

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for name, return_model, direction_model in _model_specs(model_depth, random_state=random_state):
        try:
            t0 = time.perf_counter()
            return_model.fit(dataset.X_train, dataset.y_return_train_scaled)
            pred_train_return = _inverse_return_scale(dataset, return_model.predict(dataset.X_train))
            pred_test_return = _inverse_return_scale(dataset, return_model.predict(dataset.X_test))
            pred_direction, proba_up = _direction_predictions(direction_model, dataset)
            train_time = time.perf_counter() - t0

            rmse_train = _rmse(dataset.y_return_train, pred_train_return)
            rmse_test = _rmse(dataset.y_return_test, pred_test_return)
            rmse_vs_naive = (naive_rmse - rmse_test) / naive_rmse * 100.0 if naive_rmse > 0 and np.isfinite(rmse_test) else np.nan
            overfit_gap = ((rmse_test - rmse_train) / abs(rmse_train) * 100.0) if rmse_train and np.isfinite(rmse_train) else np.nan

            y_dir_test = np.asarray(dataset.y_direction_test, dtype=int)
            dir_acc = float(accuracy_score(y_dir_test, pred_direction) * 100.0)
            dir_vs_baseline = dir_acc - direction_baseline_acc if np.isfinite(direction_baseline_acc) else np.nan
            f1 = float(f1_score(y_dir_test, pred_direction, zero_division=0) * 100.0)

            try:
                auc = float(roc_auc_score(y_dir_test, proba_up) * 100.0) if len(np.unique(y_dir_test)) == 2 else np.nan
            except Exception:
                auc = np.nan

            trust_score, verdict = direct_forecast_trust_score(
                rmse_vs_naive_pct=rmse_vs_naive,
                directional_accuracy=dir_acc,
                direction_vs_baseline_pct=dir_vs_baseline,
                overfit_gap_pct=overfit_gap,
                f1=f1,
                auc=auc,
            )

            rows.append(
                {
                    "Model": name,
                    "Horizon": f"{dataset.horizon}D",
                    "RMSE_Return": round(rmse_test, 8) if np.isfinite(rmse_test) else np.nan,
                    "Naive_RMSE_Return": round(naive_rmse, 8) if np.isfinite(naive_rmse) else np.nan,
                    "RMSE_vs_Naive_%": round(rmse_vs_naive, 2) if np.isfinite(rmse_vs_naive) else np.nan,
                    "DirectionalAccuracy": round(dir_acc, 2) if np.isfinite(dir_acc) else np.nan,
                    "DirectionBaselineAccuracy": round(direction_baseline_acc, 2) if np.isfinite(direction_baseline_acc) else np.nan,
                    "Direction_vs_Baseline_%": round(dir_vs_baseline, 2) if np.isfinite(dir_vs_baseline) else np.nan,
                    "AUC": round(auc, 2) if np.isfinite(auc) else np.nan,
                    "F1": round(f1, 2) if np.isfinite(f1) else np.nan,
                    "OverfitGap_%": round(overfit_gap, 2) if np.isfinite(overfit_gap) else np.nan,
                    "TrustScore": trust_score,
                    "Verdict": verdict,
                    "Warning": _warning_text(rmse_vs_naive, dir_vs_baseline, overfit_gap),
                    "TrainTime(s)": round(float(train_time), 4),
                }
            )
        except Exception as exc:
            errors.append({"Model": name, "Error": str(exc)})

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            ["TrustScore", "RMSE_vs_Naive_%", "Direction_vs_Baseline_%"],
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)
        leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))

    return leaderboard, baseline_board, pd.DataFrame(errors)


def run_direct_forecast_report(
    *,
    raw_df: pd.DataFrame,
    asset_name: str,
    horizon: int,
    model_depth: str = "fast",
    use_phase5_features: bool = True,
    random_state: int = 42,
) -> DirectForecastReport:
    if int(horizon) not in DIRECT_FORECAST_HORIZONS:
        raise ValueError(f"horizon must be one of {DIRECT_FORECAST_HORIZONS}")

    target_col = get_target_column(asset_name)
    feature_df = build_direct_feature_frame(
        raw_df,
        target_col=target_col,
        use_phase5_features=use_phase5_features,
    )
    dataset = make_direct_forecast_dataset(
        feature_df,
        asset=asset_name,
        target_col=target_col,
        horizon=int(horizon),
    )
    leaderboard, baseline_board, errors = train_direct_forecast_models(
        dataset,
        model_depth=model_depth,
        random_state=random_state,
    )

    return DirectForecastReport(
        asset=asset_name,
        target_col=target_col,
        horizon=int(horizon),
        model_depth=str(model_depth),
        use_phase5_features=bool(use_phase5_features),
        rows=int(len(dataset.df_model)),
        train_rows=int(len(dataset.train_index)),
        val_rows=int(len(dataset.val_index)),
        test_rows=int(len(dataset.test_index)),
        feature_count=int(len(dataset.feature_cols)),
        leaderboard=leaderboard,
        baseline_board=baseline_board,
        errors=errors,
        dataset=dataset,
    )


def run_direct_forecast_signal_output(
    *,
    raw_df: pd.DataFrame,
    asset_name: str,
    horizon: int,
    model_depth: str = "fast",
    use_phase5_features: bool = True,
    model_name: Optional[str] = None,
    random_state: int = 42,
) -> DirectForecastSignalOutput:
    """
    Return safe test-set P(up) from the Phase 6 direct direction model path.

    The selected direction model is fit only on the Phase 6 train split and
    predicts only the Phase 6 test split. Threshold decisions are intentionally
    left to src.signal_engine so sweeps are reported as research diagnostics,
    not used here to tune production thresholds on test data.
    """
    report = run_direct_forecast_report(
        raw_df=raw_df,
        asset_name=asset_name,
        horizon=horizon,
        model_depth=model_depth,
        use_phase5_features=use_phase5_features,
        random_state=random_state,
    )
    if report.dataset is None:
        raise ValueError("Direct forecast report did not produce a dataset")
    if report.leaderboard is None or report.leaderboard.empty:
        raise ValueError("Direct forecast report did not produce a model leaderboard")

    selected_model = str(model_name or report.leaderboard.iloc[0]["Model"])
    direction_models = {name: direction_model for name, _, direction_model in _model_specs(model_depth, random_state=random_state)}
    if selected_model not in direction_models:
        valid = ", ".join(direction_models.keys())
        raise ValueError(f"Unknown direct direction model {selected_model!r}. Valid models: {valid}")

    dataset = report.dataset
    predicted_direction, probabilities_up = _direction_predictions(direction_models[selected_model], dataset)
    direction_baseline_acc = _direction_baseline_accuracy(report.baseline_board)
    leak_cols = [c for c in dataset.feature_cols if _is_future_target_col(c)]

    return DirectForecastSignalOutput(
        asset=asset_name,
        target_col=report.target_col,
        horizon=int(horizon),
        model_depth=str(model_depth),
        use_phase5_features=bool(use_phase5_features),
        model_name=selected_model,
        probabilities_up_test=np.asarray(probabilities_up, dtype=float).flatten(),
        predicted_direction_test=np.asarray(predicted_direction, dtype=int).flatten(),
        actual_direction_test=np.asarray(dataset.y_direction_test, dtype=int).flatten(),
        actual_return_test=np.asarray(dataset.y_return_test, dtype=float).flatten(),
        test_index=pd.DatetimeIndex(dataset.test_index),
        direction_baseline_accuracy=direction_baseline_acc,
        feature_cols=list(dataset.feature_cols),
        feature_leakage_columns=leak_cols,
        leaderboard=report.leaderboard.copy(),
        baseline_board=report.baseline_board.copy(),
        report=report,
    )


def _verdict_bucket(verdict: Any) -> str:
    text = str(verdict).lower()
    if "high" in text:
        return "High"
    if "medium" in text:
        return "Medium"
    if "low" in text:
        return "Low"
    return "DoNotTrust"


def _failed_scan_row(asset: str, horizon: int, error: str) -> Dict[str, Any]:
    row = {col: np.nan for col in SCAN_SUMMARY_COLUMNS}
    row.update(
        {
            "Asset": asset,
            "Horizon": int(horizon),
            "BestModel": "",
            "Rows": 0,
            "Features": 0,
            "TestRows": 0,
            "Best_TrustScore": 0.0,
            "Best_Verdict": "Do not trust for signals",
            "Warning": str(error),
            "FeatureLeakageCount": np.nan,
            "FeatureLeakageColumns": "",
        }
    )
    return row


def _scan_row_from_report(report: DirectForecastReport) -> Dict[str, Any]:
    leaderboard = report.leaderboard if report.leaderboard is not None else pd.DataFrame()
    dataset = report.dataset
    leak_cols = []
    if dataset is not None:
        leak_cols = [c for c in dataset.feature_cols if _is_future_target_col(c)]

    if leaderboard.empty:
        row = _failed_scan_row(report.asset, report.horizon, "No valid direct forecast models")
        row.update(
            {
                "Rows": int(report.rows),
                "Features": int(report.feature_count),
                "TestRows": int(report.test_rows),
                "FeatureLeakageCount": len(leak_cols),
                "FeatureLeakageColumns": ", ".join(leak_cols),
            }
        )
        return row

    best = leaderboard.iloc[0]
    return {
        "Asset": report.asset,
        "Horizon": int(report.horizon),
        "BestModel": best.get("Model", ""),
        "Rows": int(report.rows),
        "Features": int(report.feature_count),
        "TestRows": int(report.test_rows),
        "Best_RMSE_vs_Naive_%": best.get("RMSE_vs_Naive_%", np.nan),
        "Best_Direction_vs_Baseline_%": best.get("Direction_vs_Baseline_%", np.nan),
        "Best_DirectionalAccuracy": best.get("DirectionalAccuracy", np.nan),
        "Best_OverfitGap_%": best.get("OverfitGap_%", np.nan),
        "Best_TrustScore": best.get("TrustScore", np.nan),
        "Best_Verdict": best.get("Verdict", ""),
        "Best_RMSE_Return": best.get("RMSE_Return", np.nan),
        "Best_Naive_RMSE_Return": best.get("Naive_RMSE_Return", np.nan),
        "Best_DirectionBaselineAccuracy": best.get("DirectionBaselineAccuracy", np.nan),
        "Best_AUC": best.get("AUC", np.nan),
        "Best_F1": best.get("F1", np.nan),
        "Warning": best.get("Warning", ""),
        "FeatureLeakageCount": len(leak_cols),
        "FeatureLeakageColumns": ", ".join(leak_cols),
    }


def summarize_scan_results(scan_table: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
    """Summarize an asset x horizon direct forecast scan without hiding failures."""
    counts = {"High": 0, "Medium": 0, "Low": 0, "DoNotTrust": 0}
    if scan_table is None or scan_table.empty:
        return {
            "counts": counts,
            "top_promising": pd.DataFrame(),
            "worst_failed": pd.DataFrame(),
            "all_do_not_trust": False,
        }

    df = scan_table.copy()
    buckets = df["Best_Verdict"].map(_verdict_bucket) if "Best_Verdict" in df.columns else pd.Series([], dtype=str)
    for key in counts:
        counts[key] = int((buckets == key).sum())

    sort_cols = ["Best_TrustScore", "Best_RMSE_vs_Naive_%", "Best_Direction_vs_Baseline_%"]
    available_sort_cols = [c for c in sort_cols if c in df.columns]

    non_dnt = df[buckets != "DoNotTrust"].copy()
    if not non_dnt.empty and available_sort_cols:
        top_promising = non_dnt.sort_values(
            available_sort_cols,
            ascending=False,
            na_position="last",
        ).head(int(top_n))
    else:
        top_promising = pd.DataFrame(columns=df.columns)

    if available_sort_cols:
        worst_failed = df.sort_values(
            available_sort_cols,
            ascending=True,
            na_position="first",
        ).head(int(top_n))
    else:
        worst_failed = df.head(int(top_n))

    return {
        "counts": counts,
        "top_promising": top_promising.reset_index(drop=True),
        "worst_failed": worst_failed.reset_index(drop=True),
        "all_do_not_trust": bool(len(df) > 0 and counts["DoNotTrust"] == len(df)),
    }


def run_asset_horizon_scan(
    *,
    raw_df: pd.DataFrame,
    asset_names: Optional[Iterable[str]] = None,
    horizons: Iterable[int] = DIRECT_FORECAST_HORIZONS,
    model_depth: str = "core",
    use_phase5_features: bool = True,
    random_state: int = 42,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    keep_reports: bool = True,
) -> AssetHorizonScanReport:
    """
    Run the Phase 6 direct forecast report across every requested asset/horizon.

    This scanner reuses Phase 6 target creation, dataset construction,
    train-only scaling, baselines, model metrics, and trust verdicts. It does
    not create any new target alignment path.
    """
    assets = list(asset_names) if asset_names is not None else get_asset_names()
    horizon_list = [int(h) for h in horizons]
    if not assets:
        raise ValueError("Select at least one asset")
    if not horizon_list:
        raise ValueError("Select at least one horizon")

    invalid_horizons = [h for h in horizon_list if h not in DIRECT_FORECAST_HORIZONS]
    if invalid_horizons:
        raise ValueError(f"Invalid horizons {invalid_horizons}; expected subset of {DIRECT_FORECAST_HORIZONS}")

    total = len(assets) * len(horizon_list)
    done = 0
    scan_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    reports: Dict[Tuple[str, int], DirectForecastReport] = {}

    for asset in assets:
        try:
            target_col = get_target_column(asset)
            feature_df = build_direct_feature_frame(
                raw_df,
                target_col=target_col,
                use_phase5_features=use_phase5_features,
            )
        except Exception as exc:
            for horizon in horizon_list:
                done += 1
                if progress_callback:
                    progress_callback(done, total, f"Failed feature build for {asset} {horizon}D")
                scan_rows.append(_failed_scan_row(asset, horizon, str(exc)))
                error_rows.append(
                    {
                        "Asset": asset,
                        "Horizon": int(horizon),
                        "Stage": "feature_build",
                        "Error": str(exc),
                    }
                )
            continue

        for horizon in horizon_list:
            done += 1
            if progress_callback:
                progress_callback(done, total, f"Scanning {asset} {horizon}D")

            try:
                dataset = make_direct_forecast_dataset(
                    feature_df,
                    asset=asset,
                    target_col=target_col,
                    horizon=int(horizon),
                )
                leaderboard, baseline_board, errors = train_direct_forecast_models(
                    dataset,
                    model_depth=model_depth,
                    random_state=random_state,
                )
                report = DirectForecastReport(
                    asset=asset,
                    target_col=target_col,
                    horizon=int(horizon),
                    model_depth=str(model_depth),
                    use_phase5_features=bool(use_phase5_features),
                    rows=int(len(dataset.df_model)),
                    train_rows=int(len(dataset.train_index)),
                    val_rows=int(len(dataset.val_index)),
                    test_rows=int(len(dataset.test_index)),
                    feature_count=int(len(dataset.feature_cols)),
                    leaderboard=leaderboard,
                    baseline_board=baseline_board,
                    errors=errors,
                    dataset=dataset,
                )
                if keep_reports:
                    reports[(asset, int(horizon))] = report
                scan_rows.append(_scan_row_from_report(report))

                if errors is not None and not errors.empty:
                    for _, err in errors.iterrows():
                        error_rows.append(
                            {
                                "Asset": asset,
                                "Horizon": int(horizon),
                                "Stage": "model_training",
                                "Model": err.get("Model", ""),
                                "Error": err.get("Error", ""),
                            }
                        )
            except Exception as exc:
                scan_rows.append(_failed_scan_row(asset, horizon, str(exc)))
                error_rows.append(
                    {
                        "Asset": asset,
                        "Horizon": int(horizon),
                        "Stage": "horizon_scan",
                        "Error": str(exc),
                    }
                )

    summary = pd.DataFrame(scan_rows)
    required_plus = list(SCAN_SUMMARY_COLUMNS) + ["Warning", "FeatureLeakageCount", "FeatureLeakageColumns"]
    for col in required_plus:
        if col not in summary.columns:
            summary[col] = np.nan
    summary = summary[required_plus]
    if not summary.empty:
        summary = summary.sort_values(
            ["Best_TrustScore", "Best_RMSE_vs_Naive_%", "Best_Direction_vs_Baseline_%"],
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    scan_summary = summarize_scan_results(summary)
    return AssetHorizonScanReport(
        asset_horizon_summary=summary,
        status_counts=scan_summary["counts"],
        top_promising=scan_summary["top_promising"],
        worst_failed=scan_summary["worst_failed"],
        errors=pd.DataFrame(error_rows),
        settings={
            "assets": assets,
            "horizons": horizon_list,
            "model_depth": str(model_depth),
            "use_phase5_features": bool(use_phase5_features),
        },
        reports=reports,
    )
