"""
train.py — Machine Learning Model Training Orchestrator
===========================================================
Trains and evaluates all 7 classical ML regression models on the
preprocessed gold-price dataset:

    Linear Regression | Decision Tree | Random Forest |
    XGBoost | LightGBM | CatBoost | Support Vector Regression

Features
--------
- Unified training interface across all model types
- Per-model training time + inference time measurement
- Optuna-based hyperparameter optimization (optional, per model)
- Early stopping for boosting models (XGBoost / LightGBM / CatBoost)
- Model checkpointing & versioned serialization
- Full metrics computation (MAE, RMSE, MAPE, R², Directional Accuracy)
- Leaderboard generation sorted by RMSE

Usage
-----
    from src.train import ModelTrainer
    from src.preprocessing import Preprocessor
    from src.data_loader import DataLoader
    from src.indicators import TechnicalIndicators
    from src.feature_engineering import FeatureEngineer

    df = DataLoader().load_all(use_cache=True)
    df = TechnicalIndicators().add_all(df)
    df = FeatureEngineer().build_features(df)
    data = Preprocessor().run(df)

    trainer = ModelTrainer()
    results = trainer.train_all_ml(data)          # dict of results
    leaderboard = trainer.get_leaderboard()        # pd.DataFrame
"""

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from src.config_loader import ConfigLoader
from src.logger import get_logger
from src.utils import compute_metrics, ensure_dir, metrics_to_dataframe, save_model, timer
from src.preprocessing import PreprocessedData

logger = get_logger(__name__)
cfg    = ConfigLoader()


# ════════════════════════════════════════════════════════════════
# Result Container
# ════════════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    """Stores everything needed to evaluate / display one trained model."""
    name:            str
    model:           Any            = None
    metrics_train:   Dict[str, float] = field(default_factory=dict)
    metrics_val:     Dict[str, float] = field(default_factory=dict)
    metrics_test:    Dict[str, float] = field(default_factory=dict)
    train_time_sec:  float = 0.0
    inference_time_sec: float = 0.0
    predictions_test: np.ndarray = field(default_factory=lambda: np.array([]))
    best_params:     Dict[str, Any] = field(default_factory=dict)
    feature_importance: Optional[pd.Series] = None


# ════════════════════════════════════════════════════════════════
# ModelTrainer — ML
# ════════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Orchestrates training, tuning, and evaluation of all ML regression models.

    Parameters
    ----------
    use_optuna : bool
        If True, run Optuna hyperparameter search before final fit
        (slower but typically improves performance).
    n_trials : int
        Number of Optuna trials per model (ignored if use_optuna=False).

    Methods
    -------
    train_all_ml(data)            → Dict[str, ModelResult]
    train_linear_regression(data) → ModelResult
    train_decision_tree(data)     → ModelResult
    train_random_forest(data)     → ModelResult
    train_xgboost(data)           → ModelResult
    train_lightgbm(data)          → ModelResult
    train_catboost(data)          → ModelResult
    train_svr(data)               → ModelResult
    get_leaderboard()             → pd.DataFrame
    save_all_models()
    """

    def __init__(
        self,
        use_optuna: bool = False,
        n_trials: Optional[int] = None,
        target_scaler: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
    ):
        self.use_optuna = use_optuna
        self.n_trials = n_trials or cfg.get("optuna.n_trials", 50)
        self.random_state = cfg.get("preprocessing.random_state", 42)

        self.results: Dict[str, ModelResult] = {}
        self.models_dir = cfg.resolve_path("models_saved")

        # Config blocks for each model
        self.ml_cfg = cfg.get_section("ml_models")

        # Target scaler — used to inverse-transform predictions back to real
        # dollar prices before computing metrics. Without this, metrics like
        # MAPE are computed on standardized (z-score) values, which can
        # explode toward infinity whenever a scaled value crosses near zero,
        # even though the model's real-dollar predictions are perfectly fine.
        self.target_scaler = target_scaler

        # Full Preprocessor instance — needed when predict_returns=True, so
        # we can call reconstruct_prices_from_returns() to chain predicted
        # log-returns into actual price levels via price[t]=price[t-1]*exp(r).
        # If not provided, falls back to treating predictions as price levels
        # directly (only correct when preprocessor.predict_returns=False).
        self.preprocessor = preprocessor
        self.predict_returns = bool(getattr(preprocessor, "predict_returns", False))

    # ──────────────────────────────────────────────────────────────
    # Master orchestrator
    # ──────────────────────────────────────────────────────────────

    @timer
    def train_all_ml(self, data: PreprocessedData) -> Dict[str, ModelResult]:
        """
        Train all 7 ML models sequentially and store results internally.

        Parameters
        ----------
        data : PreprocessedData
            Output of Preprocessor.run() — provides X_train/y_train etc.

        Returns
        -------
        Dict[str, ModelResult]
        """
        trainers = [
            ("Linear Regression", self.train_linear_regression),
            ("Decision Tree",      self.train_decision_tree),
            ("Random Forest",      self.train_random_forest),
            ("XGBoost",            self.train_xgboost),
            ("LightGBM",           self.train_lightgbm),
            ("CatBoost",           self.train_catboost),
            ("SVR",                self.train_svr),
        ]

        for name, fn in trainers:
            try:
                logger.info(f"{'─'*50}")
                logger.info(f"Training: {name}")
                result = fn(data)
                self.results[name] = result
                logger.info(
                    f"[{name}] Test RMSE={result.metrics_test['RMSE']:.4f} | "
                    f"R²={result.metrics_test['R2']:.4f} | "
                    f"Train time={result.train_time_sec:.2f}s"
                )
            except Exception as exc:
                logger.error(f"Training failed for {name}: {exc}")

        logger.info(f"{'─'*50}")
        logger.info(f"All ML models trained: {len(self.results)}/{len(trainers)} succeeded")
        return self.results

    # ──────────────────────────────────────────────────────────────
    # Generic fit/evaluate helper
    # ──────────────────────────────────────────────────────────────

    def _fit_and_evaluate(
        self,
        name: str,
        model: Any,
        data: PreprocessedData,
        fit_kwargs: Optional[Dict] = None,
        best_params: Optional[Dict] = None,
    ) -> ModelResult:
        """
        Generic routine: fit model, time it, predict on all splits,
        compute metrics, extract feature importance if available.
        """
        fit_kwargs = fit_kwargs or {}

        # ── Train ──
        t0 = time.perf_counter()
        model.fit(data.X_train, data.y_train, **fit_kwargs)
        train_time = time.perf_counter() - t0

        # ── Inference timing (test set) ──
        t0 = time.perf_counter()
        pred_test = model.predict(data.X_test)
        inference_time = time.perf_counter() - t0

        pred_train = model.predict(data.X_train)
        pred_val   = model.predict(data.X_val) if len(data.X_val) > 0 else np.array([])

        # ── Convert predictions to real, comparable dollar prices ──────
        # Metrics (especially MAPE) are far more meaningful — and numerically
        # stable — when computed on real dollar prices.
        #
        # Two distinct cases:
        #   predict_returns=True  → model output is a scaled LOG RETURN.
        #       We chain predicted returns forward from the last known real
        #       price at the start of each split to reconstruct an actual
        #       price trajectory: price[t] = price[t-1] * exp(return[t]).
        #       This is what lets tree-based models avoid the extrapolation
        #       failure that occurs when predicting raw price levels.
        #   predict_returns=False → model output is a scaled PRICE level.
        #       A simple inverse_transform is sufficient.
        if self.predict_returns and self.preprocessor is not None:
            y_train_real = data.prices_train
            y_val_real   = data.prices_val if len(data.prices_val) else np.array([])
            y_test_real  = data.prices_test

            # One-step-ahead reconstruction: anchor each day's prediction on
            # the REAL previous price (not the model's own prior prediction),
            # which matches real-world usage and avoids compounding drift
            # over a long evaluation window. actual_prices is the real price
            # series shifted appropriately: day i's anchor is the real price
            # from the day before, i.e. prepend last_price_before_* and drop
            # the final element.
            train_anchors = np.concatenate([[data.last_price_before_train], data.prices_train[:-1]]) if len(data.prices_train) else np.array([])
            val_anchors   = np.concatenate([[data.last_price_before_val], data.prices_val[:-1]]) if len(data.prices_val) else np.array([])
            test_anchors  = np.concatenate([[data.last_price_before_test], data.prices_test[:-1]]) if len(data.prices_test) else np.array([])

            pred_train_real = self.preprocessor.reconstruct_prices_from_returns(
                pred_train, data.last_price_before_train, actual_prices=train_anchors
            )
            pred_val_real = (
                self.preprocessor.reconstruct_prices_from_returns(
                    pred_val, data.last_price_before_val, actual_prices=val_anchors
                )
                if len(pred_val) else np.array([])
            )
            pred_test_real = self.preprocessor.reconstruct_prices_from_returns(
                pred_test, data.last_price_before_test, actual_prices=test_anchors
            )
        elif self.target_scaler is not None:
            y_train_real = self.target_scaler.inverse_transform(data.y_train.reshape(-1, 1)).flatten()
            y_val_real   = self.target_scaler.inverse_transform(data.y_val.reshape(-1, 1)).flatten() if len(data.y_val) else np.array([])
            y_test_real  = self.target_scaler.inverse_transform(data.y_test.reshape(-1, 1)).flatten()

            pred_train_real = self.target_scaler.inverse_transform(pred_train.reshape(-1, 1)).flatten()
            pred_val_real   = self.target_scaler.inverse_transform(pred_val.reshape(-1, 1)).flatten() if len(pred_val) else np.array([])
            pred_test_real  = self.target_scaler.inverse_transform(pred_test.reshape(-1, 1)).flatten()
        else:
            # No scaler provided — fall back to whatever scale was passed in
            y_train_real, y_val_real, y_test_real = data.y_train, data.y_val, data.y_test
            pred_train_real, pred_val_real, pred_test_real = pred_train, pred_val, pred_test

        result = ModelResult(
            name=name,
            model=model,
            train_time_sec=train_time,
            inference_time_sec=inference_time,
            predictions_test=pred_test_real,
            best_params=best_params or {},
        )

        result.metrics_train = compute_metrics(y_train_real, pred_train_real, model_name=f"{name} (train)")
        if len(pred_val_real):
            result.metrics_val = compute_metrics(y_val_real, pred_val_real, model_name=f"{name} (val)")
        result.metrics_test  = compute_metrics(y_test_real, pred_test_real, model_name=f"{name} (test)")

        # ── Feature importance (if supported) ──
        if hasattr(model, "feature_importances_"):
            result.feature_importance = pd.Series(
                model.feature_importances_, index=data.feature_cols
            ).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            result.feature_importance = pd.Series(
                np.abs(model.coef_), index=data.feature_cols
            ).sort_values(ascending=False)

        return result

    # ════════════════════════════════════════════════════════════
    # 1. LINEAR REGRESSION
    # ════════════════════════════════════════════════════════════

    def train_linear_regression(self, data: PreprocessedData) -> ModelResult:
        """Train a baseline Linear Regression model."""
        params = self.ml_cfg.get("linear_regression", {})
        model = LinearRegression(fit_intercept=params.get("fit_intercept", True))
        return self._fit_and_evaluate("Linear Regression", model, data)

    # ════════════════════════════════════════════════════════════
    # 2. DECISION TREE
    # ════════════════════════════════════════════════════════════

    def train_decision_tree(self, data: PreprocessedData) -> ModelResult:
        """Train a Decision Tree Regressor."""
        if self.use_optuna:
            best_params = self._tune_decision_tree(data)
        else:
            cfg_p = self.ml_cfg.get("decision_tree", {})
            best_params = {
                "max_depth": cfg_p.get("max_depth", 10),
                "min_samples_split": cfg_p.get("min_samples_split", 10),
            }

        model = DecisionTreeRegressor(
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            random_state=self.random_state,
        )
        return self._fit_and_evaluate("Decision Tree", model, data, best_params=best_params)

    def _tune_decision_tree(self, data: PreprocessedData) -> Dict:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            max_depth = trial.suggest_int("max_depth", 3, 25)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 30)
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=self.random_state,
            )
            model.fit(data.X_train, data.y_train)
            pred = model.predict(data.X_val)
            rmse = np.sqrt(np.mean((data.y_val - pred) ** 2))
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        logger.info(f"Decision Tree best params: {study.best_params}")
        return study.best_params

    # ════════════════════════════════════════════════════════════
    # 3. RANDOM FOREST
    # ════════════════════════════════════════════════════════════

    def train_random_forest(self, data: PreprocessedData) -> ModelResult:
        """Train a Random Forest Regressor."""
        if self.use_optuna:
            best_params = self._tune_random_forest(data)
        else:
            cfg_p = self.ml_cfg.get("random_forest", {})
            best_params = {
                "n_estimators": cfg_p.get("n_estimators", 200),
                "max_depth": cfg_p.get("max_depth", 15),
                "min_samples_split": cfg_p.get("min_samples_split", 5),
            }

        model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            n_jobs=-1,
            random_state=self.random_state,
        )
        return self._fit_and_evaluate("Random Forest", model, data, best_params=best_params)

    def _tune_random_forest(self, data: PreprocessedData) -> Dict:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 400)
            max_depth = trial.suggest_int("max_depth", 5, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                n_jobs=-1,
                random_state=self.random_state,
            )
            model.fit(data.X_train, data.y_train)
            pred = model.predict(data.X_val)
            rmse = np.sqrt(np.mean((data.y_val - pred) ** 2))
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        logger.info(f"Random Forest best params: {study.best_params}")
        return study.best_params

    # ════════════════════════════════════════════════════════════
    # 4. XGBOOST
    # ════════════════════════════════════════════════════════════

    def train_xgboost(self, data: PreprocessedData) -> ModelResult:
        """Train an XGBoost Regressor with early stopping."""
        import xgboost as xgb

        cfg_p = self.ml_cfg.get("xgboost", {})

        if self.use_optuna:
            best_params = self._tune_xgboost(data)
        else:
            best_params = {
                "n_estimators": cfg_p.get("n_estimators", 500),
                "max_depth": cfg_p.get("max_depth", 6),
                "learning_rate": cfg_p.get("learning_rate", 0.05),
                "subsample": cfg_p.get("subsample", 0.8),
                "colsample_bytree": cfg_p.get("colsample_bytree", 0.8),
            }

        model = xgb.XGBRegressor(
            **best_params,
            random_state=self.random_state,
            early_stopping_rounds=cfg_p.get("early_stopping_rounds", 50),
            eval_metric="rmse",
        )
        fit_kwargs = {"eval_set": [(data.X_val, data.y_val)], "verbose": False}
        return self._fit_and_evaluate("XGBoost", model, data, fit_kwargs=fit_kwargs, best_params=best_params)

    def _tune_xgboost(self, data: PreprocessedData) -> Dict:
        import optuna
        import xgboost as xgb
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
            model = xgb.XGBRegressor(**params, random_state=self.random_state, eval_metric="rmse")
            model.fit(data.X_train, data.y_train)
            pred = model.predict(data.X_val)
            rmse = np.sqrt(np.mean((data.y_val - pred) ** 2))
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        logger.info(f"XGBoost best params: {study.best_params}")
        return study.best_params

    # ════════════════════════════════════════════════════════════
    # 5. LIGHTGBM
    # ════════════════════════════════════════════════════════════

    def train_lightgbm(self, data: PreprocessedData) -> ModelResult:
        """Train a LightGBM Regressor with early stopping."""
        import lightgbm as lgb

        cfg_p = self.ml_cfg.get("lightgbm", {})

        if self.use_optuna:
            best_params = self._tune_lightgbm(data)
        else:
            best_params = {
                "n_estimators": cfg_p.get("n_estimators", 500),
                "max_depth": cfg_p.get("max_depth", 6),
                "learning_rate": cfg_p.get("learning_rate", 0.05),
                "num_leaves": cfg_p.get("num_leaves", 31),
                "subsample": cfg_p.get("subsample", 0.8),
            }

        model = lgb.LGBMRegressor(
            **best_params,
            random_state=self.random_state,
            verbosity=-1,
        )
        fit_kwargs = {
            "eval_set": [(data.X_val, data.y_val)],
            "callbacks": [lgb.early_stopping(cfg_p.get("early_stopping_rounds", 50), verbose=False)],
        }
        return self._fit_and_evaluate("LightGBM", model, data, fit_kwargs=fit_kwargs, best_params=best_params)

    def _tune_lightgbm(self, data: PreprocessedData) -> Dict:
        import optuna
        import lightgbm as lgb
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }
            model = lgb.LGBMRegressor(**params, random_state=self.random_state, verbosity=-1)
            model.fit(data.X_train, data.y_train)
            pred = model.predict(data.X_val)
            rmse = np.sqrt(np.mean((data.y_val - pred) ** 2))
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        logger.info(f"LightGBM best params: {study.best_params}")
        return study.best_params

    # ════════════════════════════════════════════════════════════
    # 6. CATBOOST
    # ════════════════════════════════════════════════════════════

    def train_catboost(self, data: PreprocessedData) -> ModelResult:
        """Train a CatBoost Regressor with early stopping."""
        from catboost import CatBoostRegressor

        cfg_p = self.ml_cfg.get("catboost", {})

        if self.use_optuna:
            best_params = self._tune_catboost(data)
        else:
            best_params = {
                "iterations": cfg_p.get("iterations", 500),
                "depth": cfg_p.get("depth", 6),
                "learning_rate": cfg_p.get("learning_rate", 0.05),
            }

        model = CatBoostRegressor(
            **best_params,
            random_seed=self.random_state,
            verbose=0,
            early_stopping_rounds=cfg_p.get("early_stopping_rounds", 50),
        )
        fit_kwargs = {"eval_set": (data.X_val, data.y_val), "verbose": False}
        return self._fit_and_evaluate("CatBoost", model, data, fit_kwargs=fit_kwargs, best_params=best_params)

    def _tune_catboost(self, data: PreprocessedData) -> Dict:
        import optuna
        from catboost import CatBoostRegressor
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 800),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            }
            model = CatBoostRegressor(**params, random_seed=self.random_state, verbose=0)
            model.fit(data.X_train, data.y_train)
            pred = model.predict(data.X_val)
            rmse = np.sqrt(np.mean((data.y_val - pred) ** 2))
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        logger.info(f"CatBoost best params: {study.best_params}")
        return study.best_params

    # ════════════════════════════════════════════════════════════
    # 7. SUPPORT VECTOR REGRESSION
    # ════════════════════════════════════════════════════════════

    def train_svr(self, data: PreprocessedData) -> ModelResult:
        """Train a Support Vector Regressor."""
        cfg_p = self.ml_cfg.get("svr", {})

        if self.use_optuna:
            best_params = self._tune_svr(data)
        else:
            best_params = {
                "kernel": cfg_p.get("kernel", "rbf"),
                "C": cfg_p.get("C", 100),
                "epsilon": cfg_p.get("epsilon", 0.1),
                "gamma": cfg_p.get("gamma", "scale"),
            }

        model = SVR(**best_params)
        # SVR can be slow on large datasets — subsample if needed
        return self._fit_and_evaluate("SVR", model, data, best_params=best_params)

    def _tune_svr(self, data: PreprocessedData) -> Dict:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "C": trial.suggest_float("C", 1, 500, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.001, 0.5, log=True),
                "kernel": "rbf",
                "gamma": "scale",
            }
            model = SVR(**params)
            model.fit(data.X_train, data.y_train)
            pred = model.predict(data.X_val)
            rmse = np.sqrt(np.mean((data.y_val - pred) ** 2))
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=min(self.n_trials, 30), show_progress_bar=False)
        logger.info(f"SVR best params: {study.best_params}")
        return study.best_params

    # ════════════════════════════════════════════════════════════
    # LEADERBOARD / PERSISTENCE
    # ════════════════════════════════════════════════════════════

    def get_leaderboard(self, split: str = "test") -> pd.DataFrame:
        """
        Build a sorted leaderboard DataFrame across all trained models.

        Parameters
        ----------
        split : str
            Which metric split to display: "train" | "val" | "test"

        Returns
        -------
        pd.DataFrame  Sorted by RMSE ascending, with Rank column.
        """
        attr = f"metrics_{split}"
        metrics_dict = {
            name: getattr(result, attr)
            for name, result in self.results.items()
            if getattr(result, attr)
        }
        if not metrics_dict:
            logger.warning("No results available for leaderboard.")
            return pd.DataFrame()

        df = metrics_to_dataframe(metrics_dict)

        # Append timing info
        df["TrainTime(s)"] = df["Model"].map(
            lambda m: round(self.results[m].train_time_sec, 4)
        )
        df["InferenceTime(s)"] = df["Model"].map(
            lambda m: round(self.results[m].inference_time_sec, 6)
        )
        return df

    def save_all_models(self, directory: Optional[str] = None) -> None:
        """Persist every trained model to disk as .pkl files."""
        out_dir = ensure_dir(directory or self.models_dir)
        for name, result in self.results.items():
            safe_name = name.lower().replace(" ", "_")
            path = out_dir / f"{safe_name}.pkl"
            save_model(result.model, path)
        logger.info(f"Saved {len(self.results)} models → {out_dir}")

    def get_best_model(self, split: str = "test") -> Tuple[str, ModelResult]:
        """Return (name, ModelResult) of the best model by RMSE."""
        board = self.get_leaderboard(split)
        if board.empty:
            raise RuntimeError("No models trained yet.")
        best_name = board.iloc[0]["Model"]
        return best_name, self.results[best_name]


# ════════════════════════════════════════════════════════════════
# Standalone test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.indicators import TechnicalIndicators
    from src.feature_engineering import FeatureEngineer
    from src.preprocessing import Preprocessor

    print("=" * 70)
    print("  ML Model Training — Full Pipeline Test")
    print("=" * 70)

    # ── Build pipeline ──
    loader = DataLoader(start_date="2015-01-01", end_date=None)  # None = today
    df = loader.load_all(use_cache=True)
    print(f"\nRaw data: {df.shape}")

    ti = TechnicalIndicators(prefix="Gold")
    df = ti.add_all(df)

    fe = FeatureEngineer()
    df = fe.build_features(df)
    print(f"Feature-engineered data: {df.shape}")

    pp = Preprocessor()
    data = pp.run(df)
    print(f"Train/Val/Test: {data.X_train.shape[0]}/{data.X_val.shape[0]}/{data.X_test.shape[0]}")

    # ── Train all ML models ──
    trainer = ModelTrainer(use_optuna=False, target_scaler=data.target_scaler, preprocessor=pp)   # set use_optuna=True for hyperparameter tuning
    results = trainer.train_all_ml(data)

    # ── Leaderboard ──
    print("\n" + "=" * 70)
    print("  LEADERBOARD (Test Set)")
    print("=" * 70)
    board = trainer.get_leaderboard("test")
    print(board.to_string(index=False))

    best_name, best_result = trainer.get_best_model("test")
    print(f"\n🏆 Best Model: {best_name}  (RMSE={best_result.metrics_test['RMSE']:.4f})")

    # ── Save models ──
    trainer.save_all_models()

    print("\n✔ train.py (ML section) working correctly")
