"""UI-safe orchestration for the existing session model-training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from src.train import ModelTrainer
from src.trained_model_registry import (
    build_failure_entry,
    build_registry_entries,
    registry_to_leaderboard,
    save_trained_model_registry,
)


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
    model_registry: list[dict[str, Any]] = field(default_factory=list)
    warning_count: int = 0


def _notify(callback: Optional[ProgressCallback], value: int, message: str) -> None:
    if callback is not None:
        callback(int(value), str(message))


def _prefixed_leaderboard(board: pd.DataFrame, family: str) -> pd.DataFrame:
    if not isinstance(board, pd.DataFrame) or board.empty:
        return pd.DataFrame()
    out = board.copy()
    if "ModelFamily" not in out.columns:
        out.insert(0, "ModelFamily", str(family).upper())
    return out


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
    """Run the project's existing ML/DL trainers and register all results once.

    DL remains optional. If TensorFlow/Keras is unavailable, the app returns
    clean warning rows for DL models instead of crashing ML training or app
    startup. Successful ML and DL models are collected into one shared registry
    used by Compare Models and Forecast pages.
    """
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
    dl_failures: dict[str, str] = {}
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
        ml_board = _prefixed_leaderboard(ml_trainer.get_leaderboard("test"), "ML")
        if not ml_board.empty:
            leaderboards.append(ml_board)

    if train_dl:
        model_families.append("DL")
        _notify(progress_callback, 72, "Training deep-learning models...")
        try:
            from src.train_dl import DLModelTrainer

            dl_trainer = DLModelTrainer(
                preprocessor=preprocessor,
                epochs=int(dl_epochs),
                verbose=0,
            )
            dl_trainer.train_all_dl(data)
            model_count += len(dl_trainer.results)
            dl_failures = dict(getattr(dl_trainer, "failures", {}) or {})
            dl_board = _prefixed_leaderboard(dl_trainer.get_leaderboard("test"), "DL")
            if not dl_board.empty:
                leaderboards.append(dl_board)
        except Exception as exc:
            # Import/setup-level failures are converted into one clean warning
            # entry instead of crashing the Streamlit app.
            dl_failures = {
                "DL setup": (
                    f"Deep-learning training is unavailable in this environment: "
                    f"{type(exc).__name__}: {exc}"
                )
            }

    _notify(progress_callback, 88, "Registering trained models...")
    registry_entries = build_registry_entries(
        asset=str(asset),
        horizon=int(horizon),
        target_col=str(target_col),
        preprocessor=preprocessor,
        data=data,
        feature_frame=feature_frame,
        ml_trainer=ml_trainer,
        dl_trainer=dl_trainer,
        dl_failures=dl_failures,
    )

    # If a setup-level DL failure happened before a DL trainer object existed,
    # build_registry_entries cannot see it, so add it explicitly.
    if train_dl and dl_trainer is None and dl_failures:
        for name, warning in dl_failures.items():
            registry_entries.append(
                build_failure_entry(
                    model_name=name,
                    model_family="DL",
                    asset=str(asset),
                    target_col=str(target_col),
                    horizon=int(horizon),
                    warning=warning,
                    preprocessor=preprocessor,
                    data=data,
                    feature_frame=feature_frame,
                )
            )

    warning_count = sum(1 for entry in registry_entries if str(entry.get("Warning") or "").strip())
    registry_board = registry_to_leaderboard(registry_entries, include_unusable=True)
    artifact_paths = tuple(save_trained_model_registry(registry_entries)) if registry_entries else ()

    if model_count == 0 and not registry_entries:
        raise RuntimeError(
            "Training finished without a successful model and no warning entries were produced. "
            "Review the training error details and dependencies."
        )

    _notify(progress_callback, 92, "Finalizing session training results...")
    legacy_leaderboard = pd.concat(leaderboards, ignore_index=True) if leaderboards else pd.DataFrame()
    leaderboard = registry_board if isinstance(registry_board, pd.DataFrame) and not registry_board.empty else legacy_leaderboard
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
        artifact_paths=artifact_paths,
        model_registry=registry_entries,
        warning_count=int(warning_count),
    )
