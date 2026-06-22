"""
predict.py — Inference & Multi-Day Forecasting Engine
========================================================
Loads trained models and generates:
  • Single-step (next-day) predictions on new/held-out data
  • Rolling N-day-ahead forecasts (default 30 days) by recursively
    feeding each day's prediction back in as input for the next

Handles both representations transparently:
  • predict_returns=True  → predicts log-returns, reconstructs prices
  • predict_returns=False → predicts price levels directly

Usage
-----
    from src.predict import Predictor

    predictor = Predictor(model=trained_model, preprocessor=pp)
    next_day_price = predictor.predict_next_day(latest_features)
    forecast_df = predictor.forecast(df, n_days=30)
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config_loader import ConfigLoader
from src.logger import get_logger
from src.utils import ensure_dir, load_model, timer
from src.prediction_ranges import calculate_prediction_range
from src.signals import generate_trading_signal

logger = get_logger(__name__)
cfg    = ConfigLoader()


class Predictor:
    """
    Unified inference engine for both ML and DL models.

    Parameters
    ----------
    model : Any
        A trained model exposing .predict(X) -> array. Works for sklearn,
        XGBoost/LightGBM/CatBoost, and Keras models alike.
    preprocessor : Preprocessor
        The fitted Preprocessor used to build the training data — required
        for scaling new inputs consistently and reconstructing real prices
        from predicted returns.
    is_sequence_model : bool
        True for DL models that expect 3-D (samples, seq_len, features)
        input; False for flat 2-D ML models.

    Methods
    -------
    predict_next_day(X_latest)        → float (next day's predicted price)
    predict_batch(X)                  → np.ndarray (predicted prices)
    forecast(df, n_days)              → pd.DataFrame (rolling N-day forecast)
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        is_sequence_model: bool = False,
    ):
        self.model = model
        self.pp = preprocessor
        self.is_sequence_model = is_sequence_model
        self.predict_returns = bool(getattr(preprocessor, "predict_returns", False))
        self.seq_len = getattr(preprocessor, "seq_len", cfg.get("dl_models.sequence_length", 60))

    # ──────────────────────────────────────────────────────────────
    # Single / batch prediction
    # ──────────────────────────────────────────────────────────────

    def predict_next_day(
        self,
        X_latest: np.ndarray,
        last_known_price: float,
    ) -> float:
        """
        Predict the next trading day's price.

        Parameters
        ----------
        X_latest : np.ndarray
            Most recent feature row(s), already scaled with the SAME
            feature scaler used during training. For sequence models,
            shape must be (1, seq_len, n_features); for flat models,
            shape (1, n_features).
        last_known_price : float
            Today's actual closing price (the anchor for reconstructing
            tomorrow's price from a predicted return).

        Returns
        -------
        float  Predicted price for the next trading day.
        """
        raw_pred = self._raw_predict(X_latest)

        if self.predict_returns:
            price = self.pp.reconstruct_prices_from_returns(
                np.array([raw_pred]), last_known_price
            )[0]
        else:
            price = self.pp.inverse_transform_target(np.array([raw_pred]))[0]

        return float(price)

    def predict_next_day_report(
        self,
        X_latest: np.ndarray,
        last_known_price: float,
        *,
        model_used: str,
        rmse: Optional[float] = None,
        residuals: Optional[np.ndarray] = None,
        confidence_level: float = 0.68,
    ) -> Dict[str, Any]:
        """
        Predict next-day price and return a production-friendly report with:
        price range, return %, signal, confidence, and risk label.

        Use held-out validation/test RMSE or recent residuals. Do not pass
        training RMSE because that would make uncertainty look too optimistic.
        """
        predicted_price = self.predict_next_day(X_latest, last_known_price)

        prediction_range = calculate_prediction_range(
            last_price=last_known_price,
            predicted_price=predicted_price,
            rmse=rmse,
            residuals=residuals,
            confidence_level=confidence_level,
            model_used=model_used,
        )

        trading_signal = generate_trading_signal(
            predicted_return_pct=prediction_range.predicted_return_pct,
            lower_return_pct=prediction_range.lower_return_pct,
            upper_return_pct=prediction_range.upper_return_pct,
        )

        return {
            "last_known_price": prediction_range.last_price,
            "predicted_next_day_price": prediction_range.predicted_price,
            "predicted_return_pct": prediction_range.predicted_return_pct,
            "expected_lower_bound": prediction_range.lower_bound,
            "expected_upper_bound": prediction_range.upper_bound,
            "lower_return_pct": prediction_range.lower_return_pct,
            "upper_return_pct": prediction_range.upper_return_pct,
            "confidence_level": prediction_range.confidence_level,
            "error_used": prediction_range.error_used,
            "error_source": prediction_range.error_source,
            "model_used": prediction_range.model_used,
            "signal": trading_signal.signal,
            "signal_confidence": trading_signal.confidence_label,
            "confidence_score": trading_signal.confidence_score,
            "risk_label": trading_signal.risk_label,
            "signal_explanation": trading_signal.explanation,
        }

    def predict_batch(
        self,
        X: np.ndarray,
        last_known_price: float,
        actual_prices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict prices for a batch of feature rows (e.g. an entire test set).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, already scaled.
        last_known_price : float
            Price immediately preceding the first row in X.
        actual_prices : np.ndarray, optional
            If provided, uses one-step-ahead reconstruction (anchored on
            real prices) instead of pure chaining — recommended whenever
            ground-truth prices are available, e.g. for backtesting.

        Returns
        -------
        np.ndarray  Predicted prices, one per row of X.
        """
        raw_preds = self._raw_predict(X)

        if self.predict_returns:
            if actual_prices is not None:
                anchors = np.concatenate([[last_known_price], actual_prices[:-1]])
                prices = self.pp.reconstruct_prices_from_returns(
                    raw_preds, last_known_price, actual_prices=anchors
                )
            else:
                prices = self.pp.reconstruct_prices_from_returns(raw_preds, last_known_price)
        else:
            prices = self.pp.inverse_transform_target(raw_preds)

        return prices

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """Call the underlying model's .predict(), handling Keras verbosity."""
        try:
            # Keras models accept a verbose kwarg; sklearn-style ones don't.
            preds = self.model.predict(X, verbose=0)
        except TypeError:
            preds = self.model.predict(X)
        return np.array(preds).flatten()

    # ──────────────────────────────────────────────────────────────
    # Rolling N-day forecast
    # ──────────────────────────────────────────────────────────────

    @timer
    def forecast(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_days: Optional[int] = None,
        indicators_engine: Optional[Any] = None,
        feature_engineer: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Generate a rolling N-day-ahead forecast by recursively predicting
        one day at a time and fully recomputing engineered features (lags,
        rolling stats, technical indicators) before each subsequent step.

        IMPORTANT — this is genuine multi-step forecasting: beyond day 1,
        there is no ground truth available, so predicted prices must be
        chained (pure compounding mode), which means forecast uncertainty
        grows with each additional day. This is expected and should be
        communicated to end users (e.g. wider confidence bands in the
        dashboard for day 30 vs day 1).

        Without indicators_engine/feature_engineer, this falls back to a
        simplified mode that only updates the target column and OHLC
        stand-ins between steps — engineered features (lags, rolling
        stats, RSI, etc.) stay frozen at their last historical values,
        which causes the model to repeat the same prediction every step
        (a flat compounding rate). Passing both engines recomputes the
        FULL feature set after each predicted day, so lag/rolling/momentum
        features genuinely evolve and the forecast responds day-to-day
        rather than just compounding a single fixed rate.

        Parameters
        ----------
        df : pd.DataFrame
            Full feature-engineered historical DataFrame (output of
            FeatureEngineer.build_features()), used to seed the forecast
            with real, recent feature values.
        feature_cols : List[str]
            Exact feature column order used during training
            (data.feature_cols from PreprocessedData).
        n_days : int, optional
            Number of trading days to forecast ahead (default from config).
        indicators_engine : TechnicalIndicators, optional
            If provided (along with feature_engineer), recomputes the full
            indicator set after each predicted day for a more realistic,
            non-flat forecast.
        feature_engineer : FeatureEngineer, optional
            If provided (along with indicators_engine), recomputes lag/
            rolling/calendar features after each predicted day.

        Returns
        -------
        pd.DataFrame with columns: Date, Predicted_Price
            Date index continues from the last historical business day.
        """
        n_days = n_days or cfg.get("forecasting.forecast_days", 30)
        target_col = self.pp.target_col
        target_prefix = str(target_col).replace("_Close", "")
        full_recompute = indicators_engine is not None and feature_engineer is not None

        if not full_recompute:
            logger.warning(
                "forecast() called without indicators_engine/feature_engineer — "
                "falling back to simplified mode. Engineered features will stay "
                "frozen between steps, which typically produces an unrealistic "
                "flat compounding rate. Pass both engines for a proper forecast."
            )

        if full_recompute:
            # Identify the RAW columns (OHLCV + macro) that indicators/features
            # are actually derived from, so we can keep growing a raw history
            # and re-run the full pipeline on it each step — rather than
            # mutating the already-feature-engineered `df`, which has already
            # had its NaN warm-up rows dropped and would shrink to empty if we
            # called build_features() on it repeatedly.
            raw_cols = [
                c for c in df.columns
                if not any(
                    marker in c for marker in (
                        "_lag", "_roll_", "Return", "Volatility", "Ratio",
                        "Trend", "Acceleration", "DayOfWeek", "Month", "Quarter",
                        "WeekOfYear", "IsMonth", "IsQuarter", "is_holiday",
                        "_sin", "_cos", "SMA_", "EMA_", "RSI", "ROC", "Momentum",
                        "MACD", "BB_", "ATR", "ADX", "CCI", "StochRSI", "WilliamsR",
                        "Ichimoku_", "OBV", "VWAP", "MFI",
                    )
                )
            ]
            raw_history = df[raw_cols].copy()
        else:
            history = df.copy()

        last_price = float(df[target_col].iloc[-1])
        last_date  = df.index[-1]
        start_price = last_price  # anchor for the cumulative stability bound
        logger.info(f"[DEBUG] forecast() called with df.index[-1] = {last_date}, last_price = {last_price:.2f}")

        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
        logger.info(f"[DEBUG] future_dates[0] = {future_dates[0]}, future_dates[-1] = {future_dates[-1]}")
        forecasts = []

        for date in future_dates:
            if full_recompute:
                # Re-run the full indicator + feature pipeline on the raw
                # history (which has NOT had NaN rows dropped repeatedly —
                # only the original warm-up period was trimmed once, by the
                # caller, before model training). This recomputes lags,
                # rolling stats, RSI, etc. so they genuinely reflect the
                # latest predicted price rather than staying frozen.
                enriched = indicators_engine.add_all(raw_history)
                enriched = feature_engineer.build_features(enriched)
                latest_row = enriched[feature_cols].values[-1:].copy()
                window_source = enriched
            else:
                latest_row = history[feature_cols].values[-1:].copy()
                window_source = history

            if self.is_sequence_model:
                window = window_source[feature_cols].values[-self.seq_len:]
                X_scaled = self.pp._feature_scaler.transform(window)
                X_input = X_scaled.reshape(1, self.seq_len, len(feature_cols))
            else:
                X_input = self.pp._feature_scaler.transform(latest_row)

            next_price = self.predict_next_day(X_input, last_price)

            # ── Stability safeguard ─────────────────────────────────
            # Recursive (multi-step) forecasting feeds each day's prediction
            # back in as input for the next day. If a model is even slightly
            # unstable (e.g. Linear Regression with many correlated
            # technical-indicator features can have large coefficients),
            # small per-day errors compound multiplicatively and can diverge
            # to absurd values within a handful of days.
            #
            # A flat per-day clamp alone isn't sufficient: even capping each
            # day to a generous ±8% still permits 1.08^30 ≈ 10x growth over a
            # 30-day forecast if a biased model hits the ceiling every single
            # day. We therefore enforce TWO bounds together:
            #   1. Per-day move capped at max_daily_move (catches single-day
            #      pathological jumps).
            #   2. CUMULATIVE deviation from the forecast's starting price
            #      capped at max_cumulative_move (catches the compounding
            #      case where a biased model hits the per-day cap repeatedly).
            # Once the cumulative bound is reached, further days are held
            # essentially flat (tiny residual drift only) rather than
            # continuing to compound away from a once-reasonable price.
            max_daily_move = 0.08
            max_cumulative_move = 0.25  # ±25% over the whole forecast horizon

            implied_change = (next_price / last_price) - 1
            if abs(implied_change) > max_daily_move:
                implied_change = np.clip(implied_change, -max_daily_move, max_daily_move)
                next_price = last_price * (1 + implied_change)

            cumulative_change = (next_price / start_price) - 1
            if abs(cumulative_change) > max_cumulative_move:
                clamped_cumulative = np.clip(cumulative_change, -max_cumulative_move, max_cumulative_move)
                clamped_price = start_price * (1 + clamped_cumulative)
                logger.warning(
                    f"Forecast day {date.date()}: cumulative change from forecast start "
                    f"({cumulative_change:+.1%}) exceeds the {max_cumulative_move:.0%} stability "
                    f"bound — capping to {clamped_cumulative:+.1%} (${next_price:.2f} → ${clamped_price:.2f}). "
                    f"This indicates the selected model is compounding a systematic bias over the "
                    f"forecast horizon; a tree-based or deep learning model is typically more stable "
                    f"for multi-step recursive forecasting than Linear Regression / SVR."
                )
                next_price = clamped_price

            forecasts.append({"Date": date, "Predicted_Price": next_price})

            # Append a new row with the predicted price to whichever history
            # we're tracking, so the next iteration sees it.
            if full_recompute:
                new_row = raw_history.iloc[-1:].copy()
                new_row.index = [date]
                new_row[target_col] = next_price
                for suffix in ("Open", "High", "Low"):
                    col = f"{target_prefix}_{suffix}"
                    if col in raw_history.columns:
                        new_row[col] = next_price
                raw_history = pd.concat([raw_history, new_row])
            else:
                new_row = history.iloc[-1:].copy()
                new_row.index = [date]
                new_row[target_col] = next_price
                for suffix in ("Open", "High", "Low"):
                    col = f"{target_prefix}_{suffix}"
                    if col in history.columns:
                        new_row[col] = next_price
                history = pd.concat([history, new_row])

            last_price = next_price

        forecast_df = pd.DataFrame(forecasts).set_index("Date")
        logger.info(
            f"Generated {n_days}-day forecast "
            f"({'full recompute' if full_recompute else 'simplified'}): "
            f"{forecast_df['Predicted_Price'].iloc[0]:.2f} → {forecast_df['Predicted_Price'].iloc[-1]:.2f}"
        )
        return forecast_df

    # ──────────────────────────────────────────────────────────────
    # Confidence bands (simple heuristic based on historical volatility)
    # ──────────────────────────────────────────────────────────────

    def add_confidence_bands(
        self,
        forecast_df: pd.DataFrame,
        historical_volatility: float,
        z_score: float = 1.96,
    ) -> pd.DataFrame:
        """
        Add naive widening confidence bands to a forecast, reflecting that
        uncertainty compounds with each additional day of pure chaining.

        Parameters
        ----------
        forecast_df : pd.DataFrame
            Output of .forecast() — must have a 'Predicted_Price' column.
        historical_volatility : float
            Daily return standard deviation (e.g. df['Daily_Return'].std()).
        z_score : float
            Confidence interval multiplier (1.96 ≈ 95% CI).

        Returns
        -------
        pd.DataFrame with added Lower_Bound / Upper_Bound columns.
        """
        df = forecast_df.copy()
        n = len(df)
        # Uncertainty grows with sqrt(days ahead) — standard random-walk
        # compounding assumption for daily returns.
        days_ahead = np.arange(1, n + 1)
        pct_uncertainty = z_score * historical_volatility * np.sqrt(days_ahead)

        df["Lower_Bound"] = df["Predicted_Price"] * (1 - pct_uncertainty)
        df["Upper_Bound"] = df["Predicted_Price"] * (1 + pct_uncertainty)
        return df


# ════════════════════════════════════════════════════════════════
# Model loading helper
# ════════════════════════════════════════════════════════════════

def load_trained_model(model_name: str, models_dir: Optional[str] = None) -> Any:
    """
    Load a previously saved model by name.

    Parameters
    ----------
    model_name : str
        e.g. "lightgbm", "xgboost", "lstm" (matches saved filename stem).
    models_dir : str, optional
        Override the default models/saved directory.

    Returns
    -------
    Loaded model object (sklearn/XGBoost/LightGBM/CatBoost via pickle,
    or a Keras model via .keras format).
    """
    directory = Path(models_dir or cfg.resolve_path("models_saved"))
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")

    keras_path = directory / f"{safe_name}.keras"
    pkl_path   = directory / f"{safe_name}.pkl"

    if keras_path.exists():
        from tensorflow.keras.models import load_model as keras_load_model
        logger.info(f"Loading Keras model ← {keras_path}")
        return keras_load_model(keras_path)
    elif pkl_path.exists():
        return load_model(pkl_path)
    else:
        raise FileNotFoundError(f"No saved model found for '{model_name}' in {directory}")


# ════════════════════════════════════════════════════════════════
# Standalone test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.indicators import TechnicalIndicators
    from src.feature_engineering import FeatureEngineer
    from src.preprocessing import Preprocessor
    from src.train import ModelTrainer

    print("=" * 70)
    print("  Predictor — Full Pipeline Test")
    print("=" * 70)

    loader = DataLoader(start_date="2015-01-01", end_date=None)  # None = today
    df = loader.load_all(use_cache=True)
    ti = TechnicalIndicators(prefix="Gold")
    df = ti.add_all(df)
    fe = FeatureEngineer()
    df = fe.build_features(df)

    pp = Preprocessor()
    data = pp.run(df)

    # Load your actual trained best model instead of fitting a throwaway one.
    # Linear Regression is unstable for recursive forecasting -- large
    # coefficients on correlated technical indicators can produce wild
    # single-step predictions that compound across the forecast horizon.
    model = load_trained_model("lightgbm")

    predictor = Predictor(model=model, preprocessor=pp, is_sequence_model=False)

    # Test batch prediction on test set
    preds = predictor.predict_batch(
        data.X_test, data.last_price_before_test, actual_prices=data.prices_test
    )
    print(f"\n✔ Batch prediction shape: {preds.shape}")
    print(f"  First 5 predicted prices: {preds[:5]}")
    print(f"  First 5 actual prices:    {data.prices_test[:5]}")

    # Test 30-day forecast (full recompute mode — realistic, non-flat forecast)
    forecast_df = predictor.forecast(
        df, feature_cols=data.feature_cols, n_days=30,
        indicators_engine=ti, feature_engineer=fe,
    )
    print(f"\n✔ 30-day forecast generated:")
    print(forecast_df.head(10))

    vol = df["Daily_Return"].std()
    forecast_with_bands = predictor.add_confidence_bands(forecast_df, historical_volatility=vol)
    print(f"\n✔ Forecast with confidence bands:")
    print(forecast_with_bands.head(5))
    print(forecast_with_bands.tail(5))

    print("\n✔ predict.py working correctly")
