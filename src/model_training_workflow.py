"""UI-safe orchestration for the existing session model-training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from src.train import ModelTrainer


ProgressCallback = Callable[[int, str], None]


class TrainingInputUnavailable(RuntimeError):
    """Raised when the existing training pipeline has no usable input."""


@dataclass
class TrainingWorkflowResult:
    """Result metadata plus the live trainer objects used by existing pages."""

    asset: str
    horizon: int
    target_col: str
    model_families: tuple[str, ...]
    model_count: int
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)
    ml_trainer: Optional[Any] = None
    dl_trainer: Optional[Any] = None
    preprocessor: Optional[Any] = None
    data: Optional[Any] = None
    feature_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifact_paths: tuple[str, ...] = ()


def _notify(callback: Optional[ProgressCallback], value: int, message: str) -> None:
    if callback is not None:
        callback(int(value), str(message))


def run_training_pipeline(
    *,
    asset: str,
    horizon: int,
    target_col: str,
    train_ml: bool,
    train_dl: bool,
    dl_epochs: int,
    load_data: Callable[..., pd.DataFrame],
    build_feature_frame: Callable[..., pd.DataFrame],
    prepare_data: Callable[..., tuple[Any, Any]],
    progress_callback: Optional[ProgressCallback] = None,
) -> TrainingWorkflowResult:
    """Run the project's existing ML/DL trainers without changing their logic."""
    if not train_ml and not train_dl:
        raise TrainingInputUnavailable("Select at least one model family before starting training.")

    _notify(progress_callback, 5, "Loading market data...")
    raw_data = load_data("2015-01-01", use_cache=True)
    if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
        raise TrainingInputUnavailable(
            "Training could not start because required market data is unavailable."
        )

    _notify(progress_callback, 20, "Engineering historical features...")
    feature_frame = build_feature_frame(raw_data, target_col=target_col)
    if not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
        raise TrainingInputUnavailable(
            f"Training could not start because no usable features were generated for {asset}."
        )

    _notify(progress_callback, 40, "Preparing chronological train, validation, and test data...")
    preprocessor, data = prepare_data(feature_frame, target_col=target_col)

    model_families: list[str] = []
    leaderboards: list[pd.DataFrame] = []
    ml_trainer = None
    dl_trainer = None
    model_count = 0

    if train_ml:
        model_families.append("ML")
        _notify(progress_callback, 55, "Training machine-learning models...")
        ml_trainer = ModelTrainer(
            use_optuna=False,
            target_scaler=data.target_scaler,
            preprocessor=preprocessor,
        )
        ml_trainer.train_all_ml(data)
        model_count += len(ml_trainer.results)
        ml_board = ml_trainer.get_leaderboard("test")
        if isinstance(ml_board, pd.DataFrame) and not ml_board.empty:
            ml_board = ml_board.copy()
            ml_board.insert(0, "ModelFamily", "ML")
            leaderboards.append(ml_board)

    if train_dl:
        model_families.append("DL")
        _notify(progress_callback, 72, "Training deep-learning models...")
        from src.train_dl import DLModelTrainer

        dl_trainer = DLModelTrainer(
            preprocessor=preprocessor,
            epochs=int(dl_epochs),
            verbose=0,
        )
        dl_trainer.train_all_dl(data)
        model_count += len(dl_trainer.results)
        dl_board = dl_trainer.get_leaderboard("test")
        if isinstance(dl_board, pd.DataFrame) and not dl_board.empty:
            dl_board = dl_board.copy()
            dl_board.insert(0, "ModelFamily", "DL")
            leaderboards.append(dl_board)

    if model_count == 0:
        raise RuntimeError(
            "Training finished without a successful model. Review the training error details and dependencies."
        )

    _notify(progress_callback, 92, "Finalizing session training results...")
    leaderboard = pd.concat(leaderboards, ignore_index=True) if leaderboards else pd.DataFrame()
    return TrainingWorkflowResult(
        asset=str(asset),
        horizon=int(horizon),
        target_col=str(target_col),
        model_families=tuple(model_families),
        model_count=int(model_count),
        leaderboard=leaderboard,
        ml_trainer=ml_trainer,
        dl_trainer=dl_trainer,
        preprocessor=preprocessor,
        data=data,
        feature_frame=feature_frame,
    )
