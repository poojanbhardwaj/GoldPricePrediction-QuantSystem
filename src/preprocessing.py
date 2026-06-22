"""
preprocessing.py — Data Cleaning, Scaling & Splitting Pipeline
==============================================================
Handles every preprocessing step required before model training:
  1.  Missing value imputation  (forward-fill, interpolation, mean)
  2.  Duplicate removal
  3.  Outlier detection & clipping  (IQR  /  Z-score  / Isolation Forest)
  4.  Feature scaling               (MinMax / Standard / Robust)
  5.  Train / Validation / Test split using TimeSeriesSplit
  6.  Walk-Forward Validation fold generator
  7.  Sequence creation for LSTM / GRU / Transformer

Classes
-------
Preprocessor
    End-to-end preprocessing orchestrator.

Usage
-----
    from src.preprocessing import Preprocessor
    pp  = Preprocessor()
    out = pp.run(df)          # returns PreprocessedData namedtuple
    X_train, y_train = out.X_train, out.y_train
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

from src.config_loader import ConfigLoader
from src.logger import get_logger
from src.utils import ensure_dir, timer

logger = get_logger(__name__)
cfg    = ConfigLoader()


# ════════════════════════════════════════════════════════════════
# Data Container
# ════════════════════════════════════════════════════════════════

@dataclass
class PreprocessedData:
    """
    Container returned by Preprocessor.run().
    Holds all splits + metadata needed by train.py.
    """
    # Full cleaned feature matrix (before split)
    df_clean:   pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_cols: List[str]  = field(default_factory=list)
    target_col: str          = "Gold_Close"

    # ML splits (2-D numpy arrays)
    X_train:    np.ndarray   = field(default_factory=lambda: np.array([]))
    X_val:      np.ndarray   = field(default_factory=lambda: np.array([]))
    X_test:     np.ndarray   = field(default_factory=lambda: np.array([]))
    y_train:    np.ndarray   = field(default_factory=lambda: np.array([]))
    y_val:      np.ndarray   = field(default_factory=lambda: np.array([]))
    y_test:     np.ndarray   = field(default_factory=lambda: np.array([]))

    # DL sequences (3-D: samples × timesteps × features)
    X_train_seq: np.ndarray  = field(default_factory=lambda: np.array([]))
    X_val_seq:   np.ndarray  = field(default_factory=lambda: np.array([]))
    X_test_seq:  np.ndarray  = field(default_factory=lambda: np.array([]))
    y_train_seq: np.ndarray  = field(default_factory=lambda: np.array([]))
    y_val_seq:   np.ndarray  = field(default_factory=lambda: np.array([]))
    y_test_seq:  np.ndarray  = field(default_factory=lambda: np.array([]))

    # Scalers (needed for inverse_transform at inference time)
    feature_scaler: Optional[object] = None
    target_scaler:  Optional[object] = None

    # Actual price level immediately preceding each split (needed to
    # reconstruct price levels from predicted returns: price[t] = price[t-1] * exp(return[t]))
    last_price_before_train: float = 0.0
    last_price_before_val:   float = 0.0
    last_price_before_test:  float = 0.0
    prices_train: np.ndarray = field(default_factory=lambda: np.array([]))
    prices_val:   np.ndarray = field(default_factory=lambda: np.array([]))
    prices_test:  np.ndarray = field(default_factory=lambda: np.array([]))

    # Index slices (for plotting)
    train_index: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    val_index:   pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    test_index:  pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))

    # Walk-forward folds
    wf_folds: List[Tuple] = field(default_factory=list)

    # Stats
    stats: Dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════
# Preprocessor
# ════════════════════════════════════════════════════════════════

class Preprocessor:
    """
    End-to-end preprocessing pipeline for time-series financial data.

    Parameters
    ----------
    config_override : dict | None
        Override any preprocessing config keys at runtime.

    Methods (public)
    ----------------
    run(df)               → PreprocessedData
    handle_missing(df)    → pd.DataFrame
    remove_duplicates(df) → pd.DataFrame
    handle_outliers(df)   → pd.DataFrame
    scale_features(df)    → Tuple[np.ndarray, np.ndarray, scalers]
    split(X, y, index)    → PreprocessedData (splits filled)
    create_sequences(X,y) → (X_seq, y_seq)
    walk_forward_folds(X,y)→ list of (train_idx, test_idx)
    """

    # ── Constructor ────────────────────────────────────────────────

    def __init__(self, target_col: Optional[str] = None, config_override: Optional[Dict] = None):
        # Backward compatibility: older code may call Preprocessor({...})
        # with config_override as the first positional argument.
        if isinstance(target_col, dict) and config_override is None:
            config_override = target_col
            target_col = None

        pp = cfg.get_section("preprocessing")
        if config_override:
            pp.update(config_override)

        self.missing_strategy   = pp.get("missing_value_strategy", "forward_fill")
        self.outlier_method     = pp.get("outlier_method", "iqr")
        self.outlier_threshold  = float(pp.get("outlier_threshold", 3.0))
        self.scaling_method     = pp.get("scaling_method", "minmax")
        self.test_size          = float(pp.get("test_size", 0.2))
        self.val_size           = float(pp.get("validation_size", 0.1))
        self.n_splits           = int(pp.get("n_splits", 5))
        self.random_state       = int(pp.get("random_state", 42))

        self.seq_len            = int(cfg.get("dl_models.sequence_length", 60))
        self.target_col         = target_col or cfg.get("data.target_column", "Gold_Close")

        # If True, models are trained to predict the next-day LOG RETURN
        # (a stationary, scale-invariant quantity) instead of the raw price
        # level. This is the standard quant-finance approach and avoids the
        # extrapolation failure that tree-based models (Random Forest,
        # XGBoost, LightGBM, CatBoost) suffer on trending series: a model
        # trained on prices up to $2050 has no learned rule for a test-set
        # price of $2788, but a "+0.8% today" return looks identical whether
        # gold is at $1200 or $2700, so the model can generalize correctly.
        # Predictions are converted back to price levels via
        # price[t] = price[t-1] * exp(predicted_return[t]) at evaluation time.
        self.predict_returns    = bool(pp.get("predict_returns", True))

        # Scalers (initialised during run)
        self._feature_scaler: Optional[object] = None
        self._target_scaler:  Optional[object] = None

    # ── Orchestrator ───────────────────────────────────────────────

    @timer
    def run(self, df: pd.DataFrame) -> PreprocessedData:
        """
        Execute the complete preprocessing pipeline.

        Steps
        -----
        1. Validate input
        2. Handle missing values
        3. Remove duplicates
        4. Handle outliers
        5. Select features
        6. Scale
        7. Split (train / val / test)
        8. Create DL sequences
        9. Generate walk-forward folds

        Parameters
        ----------
        df : pd.DataFrame  Feature-engineered DataFrame (date-indexed).

        Returns
        -------
        PreprocessedData
        """
        logger.info("Preprocessing pipeline started")
        self._validate(df)

        df = self.handle_missing(df.copy())
        df = self.remove_duplicates(df)
        df = self.handle_outliers(df)

        feature_cols = self._select_features(df)
        logger.info(f"Feature columns selected: {len(feature_cols)}")

        # Build raw supervised arrays first, then fit scalers on TRAIN ONLY.
        # This avoids future-distribution leakage from MinMax/Standard scaling.
        X_raw, y_raw, split_index = self._make_supervised_arrays(df, feature_cols)
        result = self._split_and_scale(X_raw, y_raw, split_index, feature_cols)

        result.df_clean     = df
        result.feature_cols = feature_cols
        result.target_col   = self.target_col

        result = self._add_sequences(result)
        X_all_scaled = np.vstack([result.X_train, result.X_val, result.X_test])
        y_all_scaled = np.concatenate([result.y_train, result.y_val, result.y_test])
        result.wf_folds = self.walk_forward_folds(X_all_scaled, y_all_scaled)
        result.stats    = self._compute_stats(df, result)

        logger.info(
            f"Preprocessing done | "
            f"train={result.X_train.shape[0]} val={result.X_val.shape[0]} "
            f"test={result.X_test.shape[0]} | features={len(feature_cols)}"
        )
        return result

    # ── Step 1: Validate ──────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if self.target_col not in df.columns:
            available = [c for c in df.columns if "close" in c.lower() or "gold" in c.lower()]
            if available:
                self.target_col = available[0]
                logger.warning(f"Target column reset to '{self.target_col}'")
            else:
                raise ValueError(f"Target column '{self.target_col}' not in DataFrame.")
        logger.info(f"Input shape: {df.shape} | Target: {self.target_col}")

    # ── Step 2: Missing Values ────────────────────────────────────

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values.

        Strategy options (from config)
        --------------------------------
        forward_fill  — carry last observation forward (good for prices)
        interpolate   — linear interpolation
        mean          — column mean
        """
        null_before = df.isnull().sum().sum()

        if self.missing_strategy == "forward_fill":
            df.ffill(inplace=True)
            df.bfill(inplace=True)          # fill any remaining at start

        elif self.missing_strategy == "interpolate":
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method="time")
            df.bfill(inplace=True)

        elif self.missing_strategy == "mean":
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        else:
            df.ffill(inplace=True)
            df.bfill(inplace=True)

        # Drop any column that is still >50 % null
        null_pct = df.isnull().mean()
        drop_cols = null_pct[null_pct > 0.5].index.tolist()
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
            logger.warning(f"Dropped high-null columns: {drop_cols}")

        null_after = df.isnull().sum().sum()
        logger.info(f"Missing values: {null_before} → {null_after}")
        return df

    # ── Step 3: Duplicates ────────────────────────────────────────

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate index entries (same date)."""
        n_before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        removed = n_before - len(df)
        if removed:
            logger.info(f"Removed {removed} duplicate date entries")
        return df

    # ── Step 4: Outliers ──────────────────────────────────────────

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and clip outliers in numeric columns (excluding target).

        Methods
        -------
        iqr          — clip to [Q1 - k*IQR, Q3 + k*IQR]
        zscore       — clip values with |z| > threshold
        isolation_f  — flag via Isolation Forest (then clip)
        """
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Don't alter the target column
        cols_to_check = [c for c in numeric_cols if c != self.target_col]

        if self.outlier_method == "iqr":
            df = self._clip_iqr(df, cols_to_check)

        elif self.outlier_method == "zscore":
            df = self._clip_zscore(df, cols_to_check)

        elif self.outlier_method == "isolation_forest":
            df = self._clip_isolation_forest(df, cols_to_check)

        logger.info(f"Outlier handling ({self.outlier_method}) applied to {len(cols_to_check)} columns")
        return df

    def _clip_iqr(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        k = self.outlier_threshold / 2   # typically 1.5
        for col in cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr     = q3 - q1
            lower, upper = q1 - k * iqr, q3 + k * iqr
            df[col] = df[col].clip(lower, upper)
        return df

    def _clip_zscore(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for col in cols:
            z = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
            df[col] = df[col].where(z.abs() < self.outlier_threshold, df[col].median())
        return df

    def _clip_isolation_forest(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        try:
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=0.02, random_state=self.random_state)
            labels = iso.fit_predict(df[cols].fillna(0))
            mask   = labels == -1
            logger.info(f"Isolation Forest flagged {mask.sum()} outlier rows")
            # Clip flagged rows to column medians
            for col in cols:
                df.loc[mask, col] = df[col].median()
        except Exception as exc:
            logger.warning(f"Isolation Forest failed, falling back to IQR: {exc}")
            df = self._clip_iqr(df, cols)
        return df

    # ── Step 5: Feature Selection ─────────────────────────────────

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select numeric feature columns (exclude target, retain all others).
        Also drops columns with zero variance and columns suspiciously
        correlated with the target (likely data leakage / duplicate columns).
        """
        numeric = df.select_dtypes(include=np.number).columns.tolist()

        # Remove zero-variance columns
        zero_var = [c for c in numeric if df[c].std() < 1e-9]
        if zero_var:
            logger.warning(f"Dropping zero-variance columns: {zero_var}")

        candidates = [c for c in numeric if c not in zero_var and c != self.target_col]

        # ── Leakage guard ──────────────────────────────────────────
        # Any feature with |correlation| > 0.999 against the target is almost
        # certainly a duplicate/derived copy of the target itself (e.g. a
        # mislabeled column, or — as happened with synthetic fallback data —
        # two series sharing the same random seed). Legitimate predictive
        # features (lags, indicators, cross-asset prices) are never this close.
        leakage_threshold = 0.999
        correlations = df[candidates].corrwith(df[self.target_col]).abs()
        leaky = correlations[correlations > leakage_threshold].index.tolist()
        # Lag-1 of the target itself is expected to be highly correlated and is
        # intentionally informative (not leakage) — never drop it.
        leaky = [c for c in leaky if f"{self.target_col}_lag" not in c]

        if leaky:
            logger.warning(
                f"Dropping {len(leaky)} feature(s) with suspicious "
                f"target correlation (>{leakage_threshold}) — likely leakage: {leaky}"
            )
            candidates = [c for c in candidates if c not in leaky]

        return candidates

    # ── Step 6A: Supervised Array Builder ─────────────────────────

    def _make_supervised_arrays(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Build raw X/y arrays with correct next-day target alignment.

        Critical production rule:
        - features at date t predict the target price/return at date t+1
        - the final feature row is excluded from training/evaluation because
          its next-day label is not known yet
        - scalers are NOT fitted here; scaling happens after chronological
          split so future distribution cannot leak into train data
        """
        X_raw = df[feature_cols].astype(float).values

        if self.predict_returns:
            price = df[self.target_col].astype(float)
            next_price = price.shift(-1)
            log_returns = np.log(next_price / price)

            valid_mask = log_returns.notna() & np.isfinite(log_returns)
            if int(valid_mask.sum()) < 10:
                raise ValueError("Not enough valid next-day targets after alignment.")

            valid_positions = np.flatnonzero(valid_mask.to_numpy())
            target_positions = valid_positions + 1

            X_raw = X_raw[valid_mask.to_numpy()]
            y_raw = log_returns.loc[valid_mask].values.astype(float)

            # Actual target prices P[t+1] and known anchor prices P[t].
            self._price_series = next_price.loc[valid_mask].values.astype(float)
            self._anchor_price_series = price.loc[valid_mask].values.astype(float)
            self._feature_index = pd.DatetimeIndex(df.index[valid_positions])
            self._target_index = pd.DatetimeIndex(df.index[target_positions])
            self._latest_price = float(price.iloc[-1])
            split_index = self._target_index
        else:
            y_raw = df[self.target_col].astype(float).values
            self._price_series = y_raw.copy()
            self._anchor_price_series = None
            split_index = pd.DatetimeIndex(df.index[: len(X_raw)])

        return X_raw, y_raw, split_index

    def _split_and_scale(
        self,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        index: pd.DatetimeIndex,
        feature_cols: List[str],
    ) -> PreprocessedData:
        """
        Chronological split, then fit feature/target scalers on TRAIN ONLY.

        This is a major anti-leakage improvement over fitting scalers on the
        full dataset before splitting. Val/test data are transformed using
        train-fitted scalers exactly as in real production inference.
        """
        n = len(X_raw)
        if n < 50:
            raise ValueError("Not enough rows for train/validation/test split.")

        test_n = int(n * self.test_size)
        val_n = int(n * self.val_size)
        train_n = n - val_n - test_n
        if train_n <= 0 or test_n <= 0:
            raise ValueError("Invalid split sizes; adjust test_size/validation_size.")

        ScalerClass = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "robust": RobustScaler,
        }.get(self.scaling_method, MinMaxScaler)

        X_train_raw = X_raw[:train_n]
        X_val_raw = X_raw[train_n: train_n + val_n]
        X_test_raw = X_raw[train_n + val_n:]

        y_train_raw = y_raw[:train_n]
        y_val_raw = y_raw[train_n: train_n + val_n]
        y_test_raw = y_raw[train_n + val_n:]

        self._feature_scaler = ScalerClass()
        X_train = self._feature_scaler.fit_transform(X_train_raw)
        X_val = self._feature_scaler.transform(X_val_raw) if len(X_val_raw) else np.array([]).reshape(0, X_train.shape[1])
        X_test = self._feature_scaler.transform(X_test_raw) if len(X_test_raw) else np.array([]).reshape(0, X_train.shape[1])

        self._target_scaler = StandardScaler()
        y_train = self._target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        y_val = self._target_scaler.transform(y_val_raw.reshape(-1, 1)).flatten() if len(y_val_raw) else np.array([])
        y_test = self._target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten() if len(y_test_raw) else np.array([])

        data = PreprocessedData(
            feature_scaler=self._feature_scaler,
            target_scaler=self._target_scaler,
        )

        data.X_train, data.X_val, data.X_test = X_train, X_val, X_test
        data.y_train, data.y_val, data.y_test = y_train, y_val, y_test

        data.train_index = index[:train_n]
        data.val_index = index[train_n: train_n + val_n]
        data.test_index = index[train_n + val_n:]

        prices = np.asarray(self._price_series, dtype=float)
        data.prices_train = prices[:train_n]
        data.prices_val = prices[train_n: train_n + val_n]
        data.prices_test = prices[train_n + val_n:]

        anchors = getattr(self, "_anchor_price_series", None)
        if anchors is not None and len(anchors) == len(prices):
            anchors = np.asarray(anchors, dtype=float)
            data.last_price_before_train = float(anchors[0])
            data.last_price_before_val = float(anchors[train_n]) if val_n > 0 and train_n < len(anchors) else float(data.prices_train[-1])
            test_start = train_n + val_n
            data.last_price_before_test = float(anchors[test_start]) if test_start < len(anchors) else float(data.prices_val[-1] if len(data.prices_val) else data.prices_train[-1])
        else:
            data.last_price_before_train = float(prices[0])
            data.last_price_before_val = float(prices[train_n - 1]) if train_n > 0 else float(prices[0])
            data.last_price_before_test = float(prices[train_n + val_n - 1]) if (train_n + val_n) > 0 else float(prices[0])

        logger.info(
            f"Scaling: {self.scaling_method} | X={X_train.shape[0] + X_val.shape[0] + X_test.shape[0], X_train.shape[1]} "
            f"y={(len(y_train) + len(y_val) + len(y_test),)} | target_mode={'log_returns' if self.predict_returns else 'price_level'} | scaler_fit=train_only"
        )
        logger.info(
            f"Split sizes — Train:{train_n} ({train_n/n:.0%})  "
            f"Val:{val_n} ({val_n/n:.0%})  Test:{test_n} ({test_n/n:.0%})"
        )

        return data

    # ── Step 6: Scaling ───────────────────────────────────────────


    def _scale(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features and target independently.
        Stores scalers as instance attributes for later inverse_transform.

        Note on target representation
        -------------------------------
        Controlled by self.predict_returns (config: preprocessing.predict_returns):

        predict_returns=True (default, recommended for tree models):
            Target = true next-day log return: log(price[t+1] / price[t]).
            Today's features are therefore used to predict tomorrow's move.
            This is stationary — its distribution doesn't shift as the
            price trends upward over years — so models never need to
            extrapolate beyond a price range they've seen. Price levels
            are reconstructed afterward via price[t] = price[t-1] * exp(r).

        predict_returns=False (legacy / simpler interpretation):
            Target = raw price level, scaled with StandardScaler. Works
            fine for Linear Regression and DL models with recent lag
            features, but tree-based models (Random Forest, XGBoost,
            LightGBM, CatBoost) will fail to extrapolate on strongly
            trending series — a model trained on prices up to $2050 has
            no learned split for an unseen test price of $2788.
        """
        ScalerClass = {
            "minmax":   MinMaxScaler,
            "standard": StandardScaler,
            "robust":   RobustScaler,
        }.get(self.scaling_method, MinMaxScaler)

        self._feature_scaler = ScalerClass()
        X = df[feature_cols].values
        X_scaled = self._feature_scaler.fit_transform(X)

        if self.predict_returns:
            # TRUE next-day target:
            # features at date t -> target log return from price[t] to price[t+1].
            # This removes the same-day leakage that happens when the target is
            # log(price[t] / price[t-1]) while features also contain date-t data.
            price = df[self.target_col].astype(float)
            next_price = price.shift(-1)
            log_returns = np.log(next_price / price)

            valid_mask = log_returns.notna() & np.isfinite(log_returns)
            if int(valid_mask.sum()) < 10:
                raise ValueError("Not enough valid next-day targets after alignment.")

            # Keep only rows whose next-day target is known. The last feature
            # row is intentionally excluded from training/evaluation because
            # its future price is unknown; the app uses it separately for the
            # live next-day prediction.
            valid_positions = np.flatnonzero(valid_mask.to_numpy())
            target_positions = valid_positions + 1

            X_scaled = X_scaled[valid_mask.to_numpy()]
            log_returns = log_returns.loc[valid_mask].values.reshape(-1, 1)

            self._target_scaler = StandardScaler()
            y_scaled = self._target_scaler.fit_transform(log_returns).flatten()

            # price_series = actual next-day prices P[t+1] for evaluation.
            # anchor_price_series = known prices P[t] used to reconstruct each
            # predicted next-day price from the predicted return.
            self._price_series = next_price.loc[valid_mask].values
            self._anchor_price_series = price.loc[valid_mask].values
            self._feature_index = pd.DatetimeIndex(df.index[valid_positions])
            self._target_index = pd.DatetimeIndex(df.index[target_positions])
            self._latest_price = float(price.iloc[-1])
        else:
            self._target_scaler = StandardScaler()
            y = df[[self.target_col]].values
            y_scaled = self._target_scaler.fit_transform(y).flatten()
            self._price_series = df[self.target_col].values

        logger.info(
            f"Scaling: {self.scaling_method} | X={X_scaled.shape} y={y_scaled.shape} "
            f"| target_mode={'log_returns' if self.predict_returns else 'price_level'}"
        )
        return X_scaled, y_scaled

    # ── Step 7: Train / Val / Test Split ─────────────────────────

    def _split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        index: pd.DatetimeIndex,
        feature_cols: List[str],
    ) -> PreprocessedData:
        """
        Chronological split (no shuffling — required for time series).
        train | val | test in sequential order.
        """
        n = len(X)
        test_n = int(n * self.test_size)
        val_n  = int(n * self.val_size)
        train_n = n - val_n - test_n

        data = PreprocessedData(
            feature_scaler = self._feature_scaler,
            target_scaler  = self._target_scaler,
        )

        data.X_train = X[:train_n]
        data.X_val   = X[train_n: train_n + val_n]
        data.X_test  = X[train_n + val_n:]

        data.y_train = y[:train_n]
        data.y_val   = y[train_n: train_n + val_n]
        data.y_test  = y[train_n + val_n:]

        data.train_index = index[:train_n]
        data.val_index   = index[train_n: train_n + val_n]
        data.test_index  = index[train_n + val_n:]

        # Carry actual target prices through each split.
        # When predict_returns=True after the next-day fix, prices are P[t+1]
        # and anchors are P[t]. This lets training/evaluation reconstruct
        # predicted next-day prices without using same-day leakage.
        prices = self._price_series
        data.prices_train = prices[:train_n]
        data.prices_val    = prices[train_n: train_n + val_n]
        data.prices_test   = prices[train_n + val_n:]

        anchors = getattr(self, "_anchor_price_series", None)
        if anchors is not None and len(anchors) == len(prices):
            data.last_price_before_train = float(anchors[0])
            data.last_price_before_val = float(anchors[train_n]) if val_n > 0 and train_n < len(anchors) else float(data.prices_train[-1])
            test_start = train_n + val_n
            data.last_price_before_test = float(anchors[test_start]) if test_start < len(anchors) else float(data.prices_val[-1] if len(data.prices_val) else data.prices_train[-1])
        else:
            # Legacy fallback for price-level target mode.
            data.last_price_before_train = float(prices[0])
            data.last_price_before_val   = float(prices[train_n - 1]) if train_n > 0 else float(prices[0])
            data.last_price_before_test  = float(prices[train_n + val_n - 1]) if (train_n + val_n) > 0 else float(prices[0])

        logger.info(
            f"Split sizes — Train:{train_n} ({train_n/n:.0%})  "
            f"Val:{val_n} ({val_n/n:.0%})  Test:{test_n} ({test_n/n:.0%})"
        )
        return data

    # ── Step 8: DL Sequences ──────────────────────────────────────

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert flat arrays into sliding-window sequences for LSTM/GRU/Transformer.

        Parameters
        ----------
        X       : (n_samples, n_features)
        y       : (n_samples,)
        seq_len : lookback window length (default from config)

        Returns
        -------
        X_seq : (n_samples - seq_len, seq_len, n_features)
        y_seq : (n_samples - seq_len,)
        """
        L = seq_len or self.seq_len
        xs, ys = [], []
        for i in range(L, len(X)):
            xs.append(X[i - L: i])
            ys.append(y[i])
        return np.array(xs), np.array(ys)

    def _add_sequences(self, data: PreprocessedData) -> PreprocessedData:
        """Add DL sequence arrays to the PreprocessedData container."""
        # Concatenate train+val+test to create sequences across boundaries
        X_all = np.vstack([data.X_train, data.X_val, data.X_test])
        y_all = np.concatenate([data.y_train, data.y_val, data.y_test])

        X_seq, y_seq = self.create_sequences(X_all, y_all, self.seq_len)

        # Re-split sequences chronologically
        n  = len(X_seq)
        t  = len(data.X_train) - self.seq_len
        v  = len(data.X_val)

        t  = max(1, t)
        data.X_train_seq = X_seq[:t]
        data.X_val_seq   = X_seq[t: t + v]
        data.X_test_seq  = X_seq[t + v:]
        data.y_train_seq = y_seq[:t]
        data.y_val_seq   = y_seq[t: t + v]
        data.y_test_seq  = y_seq[t + v:]

        logger.info(
            f"DL sequences | train:{data.X_train_seq.shape} "
            f"val:{data.X_val_seq.shape} test:{data.X_test_seq.shape}"
        )
        return data

    # ── Step 9: Walk-Forward Validation ──────────────────────────

    def walk_forward_folds(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate Walk-Forward Validation folds using TimeSeriesSplit.
        Each fold is (train_indices, test_indices).

        Returns
        -------
        list of (train_idx, test_idx) numpy arrays
        """
        tss   = TimeSeriesSplit(n_splits=self.n_splits)
        folds = [(tr, te) for tr, te in tss.split(X)]
        logger.info(f"Walk-forward: {len(folds)} folds generated")
        return folds

    # ── Inverse Transform ─────────────────────────────────────────

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Convert scaled model output back to its original units.

        IMPORTANT: when self.predict_returns=True, this returns LOG
        RETURNS (e.g. 0.008 = +0.8% that day), NOT price levels. To get
        actual dollar prices from returns, use
        reconstruct_prices_from_returns() instead, which needs a starting
        price anchor to chain the returns into a price trajectory.
        """
        if self._target_scaler is None:
            raise RuntimeError("Target scaler not fitted. Run .run() first.")
        arr = y_scaled.reshape(-1, 1)
        return self._target_scaler.inverse_transform(arr).flatten()

    def reconstruct_prices_from_returns(
        self,
        returns_scaled: np.ndarray,
        last_known_price: float,
        actual_prices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert a sequence of scaled predicted log-returns back into actual
        price levels.

        Two modes
        ---------
        actual_prices=None (pure forecast chaining):
            price[0] = last_known_price * exp(return[0])
            price[1] = price[0]         * exp(return[1])
            ... each day's reconstruction builds on the PREVIOUS PREDICTED
            price. Small per-day errors compound multiplicatively across the
            whole sequence — over hundreds of days this can drift far from
            reality even when each individual day's return prediction is
            quite accurate. Only appropriate for genuine multi-step-ahead
            forecasting (e.g. the 30-day forecast feature), where no actual
            future prices are available to anchor against.

        actual_prices=array (one-step-ahead reconstruction, RECOMMENDED for
        backtesting/evaluation):
            price[t] = actual_price[t-1] * exp(predicted_return[t])
            Each day's reconstruction anchors on the PREVIOUS DAY'S REAL
            price, not the model's own prior prediction. This matches how
            the model would actually be used in production — each morning
            you know yesterday's real closing price and predict only
            today's change from it — and prevents compounding drift from
            making a genuinely good day-ahead model look artificially bad
            over a long test set.

        Parameters
        ----------
        returns_scaled : np.ndarray
            Model's predicted returns, still in scaled (z-score) units.
        last_known_price : float
            Real price on the day immediately before this sequence starts.
        actual_prices : np.ndarray, optional
            Real prices for this same period, shifted one day back, used as
            reconstruction anchors. If provided, this becomes proper
            one-step-ahead evaluation. If omitted, falls back to pure
            chaining (compounds drift — only use for true forecasting).

        Returns
        -------
        np.ndarray of reconstructed price levels, same length as input.
        """
        if self._target_scaler is None:
            raise RuntimeError("Target scaler not fitted. Run .run() first.")

        log_returns = self.inverse_transform_target(returns_scaled)
        prices = np.empty(len(log_returns))

        if actual_prices is not None:
            # One-step-ahead mode: actual_prices must contain the anchor price
            # for each prediction row, e.g. [last_price_before_test, actual_0,
            # actual_1, ...]. This avoids compounding drift during evaluation
            # and matches how the model is used in production.
            anchors = np.asarray(actual_prices, dtype=float).flatten()
            if len(anchors) != len(log_returns):
                raise ValueError(
                    f"actual_prices/anchors length ({len(anchors)}) must match "
                    f"returns length ({len(log_returns)})."
                )
            prices = anchors * np.exp(log_returns)
        else:
            # Pure chaining: each prediction builds on the previous
            # PREDICTION. Compounds drift — intended for genuine multi-step
            # forecasting only (e.g. predicting 30 days with no ground truth).
            current_price = last_known_price
            for i, r in enumerate(log_returns):
                current_price = current_price * np.exp(r)
                prices[i] = current_price

        return prices

    def inverse_transform_features(self, X_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled feature matrix back to original scale."""
        if self._feature_scaler is None:
            raise RuntimeError("Feature scaler not fitted. Run .run() first.")
        return self._feature_scaler.inverse_transform(X_scaled)

    # ── Save / Load Scalers ───────────────────────────────────────

    def save_scalers(self, directory: Optional[str] = None) -> None:
        """Persist scalers to disk for inference-time use."""
        import joblib
        out = ensure_dir(directory or cfg.resolve_path("models_saved"))
        joblib.dump(self._feature_scaler, out / "feature_scaler.pkl")
        joblib.dump(self._target_scaler,  out / "target_scaler.pkl")
        logger.info(f"Scalers saved → {out}")

    def load_scalers(self, directory: Optional[str] = None) -> None:
        """Load pre-fitted scalers from disk."""
        import joblib
        src = directory or cfg.resolve_path("models_saved")
        self._feature_scaler = joblib.load(Path(src) / "feature_scaler.pkl")
        self._target_scaler  = joblib.load(Path(src) / "target_scaler.pkl")
        logger.info(f"Scalers loaded ← {src}")

    # ── Stats ─────────────────────────────────────────────────────

    def _compute_stats(self, df: pd.DataFrame, data: PreprocessedData) -> Dict:
        return {
            "n_total":    len(df),
            "n_features": len(data.feature_cols),
            "n_train":    len(data.X_train),
            "n_val":      len(data.X_val),
            "n_test":     len(data.X_test),
            "seq_len":    self.seq_len,
            "target_min": float(df[self.target_col].min()),
            "target_max": float(df[self.target_col].max()),
            "target_mean":float(df[self.target_col].mean()),
        }


# ════════════════════════════════════════════════════════════════
# Standalone test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.data_loader import DataLoader

    print("=" * 60)
    print("  Preprocessing Pipeline — Smoke Test")
    print("=" * 60)

    loader = DataLoader(start_date="2018-01-01", end_date=None)  # None = today
    df_raw = loader.load_all(use_cache=True)
    print(f"\nRaw data: {df_raw.shape}")

    pp  = Preprocessor()
    out = pp.run(df_raw)

    print(f"\n✔  Train  : X={out.X_train.shape}  y={out.y_train.shape}")
    print(f"✔  Val    : X={out.X_val.shape}    y={out.y_val.shape}")
    print(f"✔  Test   : X={out.X_test.shape}   y={out.y_test.shape}")
    print(f"✔  DL Seq : X_train={out.X_train_seq.shape}")
    print(f"✔  WF Folds: {len(out.wf_folds)}")
    print(f"✔  Features: {len(out.feature_cols)}")
    print(f"\n   Stats: {out.stats}")
    print("\n✔ preprocessing.py working correctly")
