"""
train_dl.py — Deep Learning Model Training Orchestrator
===========================================================
Trains and evaluates all deep learning sequence models on the
preprocessed gold-price dataset using the 3-D windowed sequences
prepared by preprocessing.py:

    LSTM | Bidirectional LSTM | GRU | CNN-LSTM | Transformer

Features
--------
- Unified Keras Sequential / Functional API architectures
- Early stopping + model checkpointing (best weights restored)
- ReduceLROnPlateau for adaptive learning rate decay
- Training history capture for loss-curve plotting
- One-step-ahead price reconstruction (same convention as train.py)
- Full metrics computation (MAE, RMSE, MAPE, R², Directional Accuracy)
- Seamlessly extends the same leaderboard format as ModelTrainer

Usage
-----
    from src.train_dl import DLModelTrainer
    from src.preprocessing import Preprocessor
    ...
    data = Preprocessor().run(df)

    dl_trainer = DLModelTrainer(preprocessor=pp)
    dl_results = dl_trainer.train_all_dl(data)
    leaderboard = dl_trainer.get_leaderboard()
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config_loader import ConfigLoader
from src.logger import get_logger
from src.utils import compute_metrics, ensure_dir, metrics_to_dataframe, timer
from src.preprocessing import PreprocessedData

logger = get_logger(__name__)
cfg    = ConfigLoader()

# Keras imports are deferred into functions where possible to keep module
# import light for code paths that don't need TensorFlow (e.g. ML-only runs).


@dataclass
class DLModelResult:
    """Stores everything needed to evaluate / display one trained DL model."""
    name:               str
    model:              Any = None
    history:            Optional[Dict[str, List[float]]] = None
    metrics_train:      Dict[str, float] = field(default_factory=dict)
    metrics_val:        Dict[str, float] = field(default_factory=dict)
    metrics_test:       Dict[str, float] = field(default_factory=dict)
    train_time_sec:     float = 0.0
    inference_time_sec: float = 0.0
    predictions_test:   np.ndarray = field(default_factory=lambda: np.array([]))
    epochs_trained:     int = 0


class DLModelTrainer:
    """
    Orchestrates training and evaluation of all deep learning models.

    Parameters
    ----------
    preprocessor : Preprocessor
        The fitted Preprocessor instance used to build `data`. Required to
        reconstruct real price levels from predicted log-returns.
    epochs : int
        Max training epochs per model (early stopping may end sooner).
    batch_size : int
        Mini-batch size for gradient updates.
    verbose : int
        Keras verbosity (0=silent, 1=progress bar, 2=one line per epoch).

    Methods
    -------
    train_all_dl(data)      → Dict[str, DLModelResult]
    train_lstm(data)        → DLModelResult
    train_bilstm(data)      → DLModelResult
    train_gru(data)         → DLModelResult
    train_cnn_lstm(data)    → DLModelResult
    train_transformer(data) → DLModelResult
    get_leaderboard()       → pd.DataFrame
    """

    def __init__(
        self,
        preprocessor: Optional[Any] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        dl_cfg = cfg.get_section("dl_models")
        self.epochs      = epochs or dl_cfg.get("epochs", 100)
        self.batch_size  = batch_size or dl_cfg.get("batch_size", 32)
        self.patience     = dl_cfg.get("patience", 15)
        self.learning_rate = dl_cfg.get("learning_rate", 0.001)
        self.dropout_rate  = dl_cfg.get("dropout_rate", 0.2)
        self.verbose      = verbose

        self.preprocessor = preprocessor
        self.predict_returns = bool(getattr(preprocessor, "predict_returns", False))

        self.results: Dict[str, DLModelResult] = {}
        self.checkpoints_dir = cfg.resolve_path("models_checkpoints")
        self.models_dir       = cfg.resolve_path("models_saved")

        self.dl_cfg = dl_cfg

    # ──────────────────────────────────────────────────────────────
    # Master orchestrator
    # ──────────────────────────────────────────────────────────────

    @timer
    def train_all_dl(self, data: PreprocessedData) -> Dict[str, DLModelResult]:
        """Train all 5 DL models sequentially and store results internally."""
        trainers = [
            ("LSTM",            self.train_lstm),
            ("BiLSTM",          self.train_bilstm),
            ("GRU",              self.train_gru),
            ("CNN-LSTM",        self.train_cnn_lstm),
            ("Transformer",     self.train_transformer),
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
                    f"Epochs={result.epochs_trained} | "
                    f"Train time={result.train_time_sec:.2f}s"
                )
            except Exception as exc:
                logger.error(f"Training failed for {name}: {exc}")

        logger.info(f"{'─'*50}")
        logger.info(f"All DL models trained: {len(self.results)}/{len(trainers)} succeeded")
        return self.results

    # ──────────────────────────────────────────────────────────────
    # Shared callbacks
    # ──────────────────────────────────────────────────────────────

    def _build_callbacks(self, model_name: str):
        """Early stopping + LR reduction + checkpointing — shared across all DL models."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        ckpt_path = self.checkpoints_dir / f"{safe_name}_best.keras"

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(3, self.patience // 3),
                min_lr=1e-6,
                verbose=0,
            ),
            ModelCheckpoint(
                filepath=str(ckpt_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]
        return callbacks

    # ──────────────────────────────────────────────────────────────
    # Generic fit/evaluate helper (mirrors train.py's ML version)
    # ──────────────────────────────────────────────────────────────

    def _fit_and_evaluate(
        self,
        name: str,
        model,
        data: PreprocessedData,
    ) -> DLModelResult:
        """
        Generic routine for DL models: fit with callbacks, time it, predict
        on all splits, reconstruct real prices, compute metrics.
        """
        callbacks = self._build_callbacks(name)

        t0 = time.perf_counter()
        history = model.fit(
            data.X_train_seq, data.y_train_seq,
            validation_data=(data.X_val_seq, data.y_val_seq),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=self.verbose,
        )
        train_time = time.perf_counter() - t0
        epochs_trained = len(history.history.get("loss", []))

        t0 = time.perf_counter()
        pred_test = model.predict(data.X_test_seq, verbose=0).flatten()
        inference_time = time.perf_counter() - t0

        pred_train = model.predict(data.X_train_seq, verbose=0).flatten()
        pred_val   = model.predict(data.X_val_seq, verbose=0).flatten() if len(data.X_val_seq) else np.array([])

        # ── Reconstruct real prices ─────────────────────────────────
        # DL sequences are offset by seq_len from the flat arrays (the
        # first seq_len rows are consumed as lookback context), so we
        # align prices/anchors accordingly using the tail-aligned slices
        # preprocessing.py already produces for *_seq splits.
        seq_len = self.preprocessor.seq_len if self.preprocessor else cfg.get("dl_models.sequence_length", 60)

        if self.predict_returns and self.preprocessor is not None:
            # Align price arrays to match the sequence-shifted target arrays.
            # y_*_seq are built by create_sequences(), which drops the first
            # seq_len samples; the corresponding real prices are data.prices_*
            # but offset relative to data.y_* — since sequences are built
            # from the concatenated train+val+test arrays in preprocessing.py,
            # the safest approach is to use the tail `len(pred_*)` entries of
            # each split's price array, which line up correctly because both
            # were sliced from the same chronological position.
            n_train_seq = len(pred_train)
            n_val_seq   = len(pred_val)
            n_test_seq  = len(pred_test)

            prices_train_aligned = data.prices_train[-n_train_seq:] if n_train_seq else np.array([])
            prices_val_aligned   = data.prices_val[-n_val_seq:] if n_val_seq else np.array([])
            prices_test_aligned  = data.prices_test[-n_test_seq:] if n_test_seq else np.array([])

            anchor_train = (
                np.concatenate([[data.prices_train[-n_train_seq - 1]], prices_train_aligned[:-1]])
                if n_train_seq and (len(data.prices_train) > n_train_seq) else
                np.concatenate([[data.last_price_before_train], prices_train_aligned[:-1]]) if n_train_seq else np.array([])
            )
            anchor_val = (
                np.concatenate([[data.prices_val[-n_val_seq - 1]], prices_val_aligned[:-1]])
                if n_val_seq and (len(data.prices_val) > n_val_seq) else
                np.concatenate([[data.last_price_before_val], prices_val_aligned[:-1]]) if n_val_seq else np.array([])
            )
            anchor_test = (
                np.concatenate([[data.prices_test[-n_test_seq - 1]], prices_test_aligned[:-1]])
                if n_test_seq and (len(data.prices_test) > n_test_seq) else
                np.concatenate([[data.last_price_before_test], prices_test_aligned[:-1]]) if n_test_seq else np.array([])
            )

            y_train_real = prices_train_aligned
            y_val_real   = prices_val_aligned
            y_test_real  = prices_test_aligned

            pred_train_real = self.preprocessor.reconstruct_prices_from_returns(
                pred_train, anchor_train[0] if len(anchor_train) else data.last_price_before_train,
                actual_prices=anchor_train
            )
            pred_val_real = (
                self.preprocessor.reconstruct_prices_from_returns(
                    pred_val, anchor_val[0] if len(anchor_val) else data.last_price_before_val,
                    actual_prices=anchor_val
                ) if n_val_seq else np.array([])
            )
            pred_test_real = self.preprocessor.reconstruct_prices_from_returns(
                pred_test, anchor_test[0] if len(anchor_test) else data.last_price_before_test,
                actual_prices=anchor_test
            )
        elif self.preprocessor is not None and self.preprocessor._target_scaler is not None:
            scaler = self.preprocessor._target_scaler
            y_train_real = scaler.inverse_transform(data.y_train_seq.reshape(-1, 1)).flatten()
            y_val_real   = scaler.inverse_transform(data.y_val_seq.reshape(-1, 1)).flatten() if len(data.y_val_seq) else np.array([])
            y_test_real  = scaler.inverse_transform(data.y_test_seq.reshape(-1, 1)).flatten()

            pred_train_real = scaler.inverse_transform(pred_train.reshape(-1, 1)).flatten()
            pred_val_real   = scaler.inverse_transform(pred_val.reshape(-1, 1)).flatten() if len(pred_val) else np.array([])
            pred_test_real  = scaler.inverse_transform(pred_test.reshape(-1, 1)).flatten()
        else:
            y_train_real, y_val_real, y_test_real = data.y_train_seq, data.y_val_seq, data.y_test_seq
            pred_train_real, pred_val_real, pred_test_real = pred_train, pred_val, pred_test

        result = DLModelResult(
            name=name,
            model=model,
            history=history.history,
            train_time_sec=train_time,
            inference_time_sec=inference_time,
            predictions_test=pred_test_real,
            epochs_trained=epochs_trained,
        )

        result.metrics_train = compute_metrics(y_train_real, pred_train_real, model_name=f"{name} (train)")
        if len(pred_val_real):
            result.metrics_val = compute_metrics(y_val_real, pred_val_real, model_name=f"{name} (val)")
        result.metrics_test  = compute_metrics(y_test_real, pred_test_real, model_name=f"{name} (test)")

        return result

    # ════════════════════════════════════════════════════════════
    # 1. LSTM
    # ════════════════════════════════════════════════════════════

    def train_lstm(self, data: PreprocessedData) -> DLModelResult:
        """Train a stacked LSTM model."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        units = self.dl_cfg.get("lstm", {}).get("units", [128, 64])
        n_features = data.X_train_seq.shape[2]
        seq_len = data.X_train_seq.shape[1]

        model = Sequential(name="LSTM")
        model.add(LSTM(units[0], return_sequences=True, input_shape=(seq_len, n_features)))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(units[1], return_sequences=False))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse", metrics=["mae"])
        return self._fit_and_evaluate("LSTM", model, data)

    # ════════════════════════════════════════════════════════════
    # 2. BIDIRECTIONAL LSTM
    # ════════════════════════════════════════════════════════════

    def train_bilstm(self, data: PreprocessedData) -> DLModelResult:
        """Train a Bidirectional LSTM model."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        units = self.dl_cfg.get("bilstm", {}).get("units", [128, 64])
        n_features = data.X_train_seq.shape[2]
        seq_len = data.X_train_seq.shape[1]

        model = Sequential(name="BiLSTM")
        model.add(Bidirectional(LSTM(units[0], return_sequences=True), input_shape=(seq_len, n_features)))
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(LSTM(units[1], return_sequences=False)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse", metrics=["mae"])
        return self._fit_and_evaluate("BiLSTM", model, data)

    # ════════════════════════════════════════════════════════════
    # 3. GRU
    # ════════════════════════════════════════════════════════════

    def train_gru(self, data: PreprocessedData) -> DLModelResult:
        """Train a stacked GRU model."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        units = self.dl_cfg.get("gru", {}).get("units", [128, 64])
        n_features = data.X_train_seq.shape[2]
        seq_len = data.X_train_seq.shape[1]

        model = Sequential(name="GRU")
        model.add(GRU(units[0], return_sequences=True, input_shape=(seq_len, n_features)))
        model.add(Dropout(self.dropout_rate))
        model.add(GRU(units[1], return_sequences=False))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse", metrics=["mae"])
        return self._fit_and_evaluate("GRU", model, data)

    # ════════════════════════════════════════════════════════════
    # 4. CNN-LSTM
    # ════════════════════════════════════════════════════════════

    def train_cnn_lstm(self, data: PreprocessedData) -> DLModelResult:
        """Train a hybrid CNN + LSTM model — CNN extracts local patterns, LSTM models temporal dependency."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        cnn_cfg = self.dl_cfg.get("cnn_lstm", {})
        filters     = cnn_cfg.get("cnn_filters", 64)
        kernel_size = cnn_cfg.get("cnn_kernel_size", 3)
        lstm_units  = cnn_cfg.get("lstm_units", 64)
        pool_size   = cnn_cfg.get("pool_size", 2)

        n_features = data.X_train_seq.shape[2]
        seq_len = data.X_train_seq.shape[1]

        model = Sequential(name="CNN_LSTM")
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation="relu",
                          input_shape=(seq_len, n_features), padding="same"))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(LSTM(lstm_units, return_sequences=False))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse", metrics=["mae"])
        return self._fit_and_evaluate("CNN-LSTM", model, data)

    # ════════════════════════════════════════════════════════════
    # 5. TRANSFORMER
    # ════════════════════════════════════════════════════════════

    def train_transformer(self, data: PreprocessedData) -> DLModelResult:
        """Train a Transformer encoder model with multi-head self-attention."""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Dense, Dropout, LayerNormalization,
            MultiHeadAttention, GlobalAveragePooling1D, Add
        )
        from tensorflow.keras.optimizers import Adam

        t_cfg = self.dl_cfg.get("transformer", {})
        num_heads   = t_cfg.get("num_heads", 4)
        key_dim     = t_cfg.get("key_dim", 32)
        ff_dim      = t_cfg.get("ff_dim", 128)
        num_blocks  = t_cfg.get("num_transformer_blocks", 2)
        mlp_units   = t_cfg.get("mlp_units", [128, 64])

        n_features = data.X_train_seq.shape[2]
        seq_len = data.X_train_seq.shape[1]

        inputs = Input(shape=(seq_len, n_features))
        x = inputs

        for _ in range(num_blocks):
            # Multi-head self-attention block with residual connection
            attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
            attn_out = Dropout(self.dropout_rate)(attn_out)
            x1 = Add()([x, attn_out])
            x1 = LayerNormalization(epsilon=1e-6)(x1)

            # Feed-forward block with residual connection
            ff = Dense(ff_dim, activation="relu")(x1)
            ff = Dense(n_features)(ff)
            ff = Dropout(self.dropout_rate)(ff)
            x2 = Add()([x1, ff])
            x = LayerNormalization(epsilon=1e-6)(x2)

        x = GlobalAveragePooling1D()(x)
        for units in mlp_units:
            x = Dense(units, activation="relu")(x)
            x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs, name="Transformer")
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse", metrics=["mae"])
        return self._fit_and_evaluate("Transformer", model, data)

    # ════════════════════════════════════════════════════════════
    # LEADERBOARD / PERSISTENCE
    # ════════════════════════════════════════════════════════════

    def get_leaderboard(self, split: str = "test") -> pd.DataFrame:
        """Build a sorted leaderboard DataFrame across all trained DL models."""
        attr = f"metrics_{split}"
        metrics_dict = {
            name: getattr(result, attr)
            for name, result in self.results.items()
            if getattr(result, attr)
        }
        if not metrics_dict:
            logger.warning("No DL results available for leaderboard.")
            return pd.DataFrame()

        df = metrics_to_dataframe(metrics_dict)
        df["TrainTime(s)"] = df["Model"].map(lambda m: round(self.results[m].train_time_sec, 4))
        df["InferenceTime(s)"] = df["Model"].map(lambda m: round(self.results[m].inference_time_sec, 6))
        df["Epochs"] = df["Model"].map(lambda m: self.results[m].epochs_trained)
        return df

    def save_all_models(self, directory: Optional[str] = None) -> None:
        """Persist every trained DL model to disk in Keras format."""
        out_dir = ensure_dir(directory or self.models_dir)
        for name, result in self.results.items():
            safe_name = name.lower().replace(" ", "_").replace("-", "_")
            path = out_dir / f"{safe_name}.keras"
            result.model.save(path)
            logger.info(f"Model saved → {path}")
        logger.info(f"Saved {len(self.results)} DL models → {out_dir}")

    def get_best_model(self, split: str = "test") -> Tuple[str, DLModelResult]:
        """Return (name, DLModelResult) of the best DL model by RMSE."""
        board = self.get_leaderboard(split)
        if board.empty:
            raise RuntimeError("No DL models trained yet.")
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
    print("  Deep Learning Model Training — Full Pipeline Test")
    print("=" * 70)

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
    print(f"Sequences: train={data.X_train_seq.shape} val={data.X_val_seq.shape} test={data.X_test_seq.shape}")

    # ── Train all DL models (reduced epochs for quick smoke test) ──
    dl_trainer = DLModelTrainer(preprocessor=pp, epochs=15, batch_size=32, verbose=0)
    results = dl_trainer.train_all_dl(data)

    print("\n" + "=" * 70)
    print("  DL LEADERBOARD (Test Set)")
    print("=" * 70)
    board = dl_trainer.get_leaderboard("test")
    print(board.to_string(index=False))

    if not board.empty:
        best_name, best_result = dl_trainer.get_best_model("test")
        print(f"\n🏆 Best DL Model: {best_name}  (RMSE={best_result.metrics_test['RMSE']:.4f})")

    dl_trainer.save_all_models()
    print("\n✔ train_dl.py working correctly")
