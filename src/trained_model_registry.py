"""Shared trained-model registry for ML and optional DL workflows.

The registry has two layers:
- live session entries that keep the actual model/preprocessor/data objects for
  Compare Models and Forecast pages during the Streamlit session;
- lightweight CSV/JSON metadata under artifacts/latest/model_registry so a run
  can be audited without pickling arbitrary objects into the repo.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional
import json

import numpy as np
import pandas as pd

REGISTRY_SESSION_KEY = "trained_model_registry"
REGISTRY_METADATA_SESSION_KEY = "trained_model_registry_metadata"
REGISTRY_PHASE_DIR = Path("artifacts/latest/model_registry")
REGISTRY_JSON = REGISTRY_PHASE_DIR / "trained_model_registry.json"
REGISTRY_CSV = REGISTRY_PHASE_DIR / "trained_model_registry.csv"

METRIC_COLUMNS = ("RMSE", "MAE", "MAPE", "R2", "DirectionalAccuracy")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _metrics_from_result(result: Any) -> dict[str, Optional[float]]:
    metrics = getattr(result, "metrics_test", None) or {}
    return {name: _safe_float(metrics.get(name)) for name in METRIC_COLUMNS}


def _display_name(family: str, model_name: str, asset: str, horizon: int) -> str:
    return f"{family} · {model_name} · {asset} · {int(horizon)}D"


def _target_mode(preprocessor: Any) -> str:
    return "log_returns" if bool(getattr(preprocessor, "predict_returns", False)) else "price_level"


def _feature_columns(data: Any) -> list[str]:
    return [str(col) for col in list(getattr(data, "feature_cols", []) or [])]


def build_success_entry(
    *,
    model_name: str,
    model_family: str,
    result: Any,
    asset: str,
    target_col: str,
    horizon: int,
    preprocessor: Any,
    data: Any,
    feature_frame: pd.DataFrame,
    trained_at: Optional[str] = None,
) -> dict[str, Any]:
    """Create one live registry entry for a successfully trained model."""
    family = str(model_family).upper()
    seq_len = _safe_int(getattr(preprocessor, "seq_len", None), default=0)
    feature_cols = _feature_columns(data)
    model_object = getattr(result, "model", None)
    metrics = _metrics_from_result(result)
    warning = "" if model_object is not None else "Model object was not retained in session."
    usable = bool(model_object is not None and feature_cols and not warning)

    entry = {
        "ModelName": str(model_name),
        "DisplayName": _display_name(family, str(model_name), str(asset), int(horizon)),
        "ModelFamily": family,
        "Asset": str(asset),
        "TargetColumn": str(target_col),
        "Horizon": int(horizon),
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "MAPE": metrics["MAPE"],
        "R2": metrics["R2"],
        "DirectionalAccuracy": metrics["DirectionalAccuracy"],
        "ModelObject": model_object,
        "ModelPath": None,
        "ScalerPath": None,
        "FeatureColumns": feature_cols,
        "TargetMode": _target_mode(preprocessor),
        "SequenceLength": seq_len,
        "IsSequenceModel": bool(family == "DL"),
        "TrainedAt": trained_at or _now_iso(),
        "UsableForForecast": bool(usable),
        "Warning": warning,
        "ResultObject": result,
        "PreprocessorObject": preprocessor,
        "DataObject": data,
        "FeatureFrame": feature_frame.copy() if isinstance(feature_frame, pd.DataFrame) else pd.DataFrame(),
        "PredictionsTest": getattr(result, "predictions_test", np.array([])),
    }
    return entry


def build_failure_entry(
    *,
    model_name: str,
    model_family: str,
    asset: str,
    target_col: str,
    horizon: int,
    warning: str,
    preprocessor: Any = None,
    data: Any = None,
    feature_frame: Optional[pd.DataFrame] = None,
    trained_at: Optional[str] = None,
) -> dict[str, Any]:
    """Create one registry entry for a model that failed cleanly."""
    family = str(model_family).upper()
    return {
        "ModelName": str(model_name),
        "DisplayName": _display_name(family, str(model_name), str(asset), int(horizon)),
        "ModelFamily": family,
        "Asset": str(asset),
        "TargetColumn": str(target_col),
        "Horizon": int(horizon),
        "RMSE": None,
        "MAE": None,
        "MAPE": None,
        "R2": None,
        "DirectionalAccuracy": None,
        "ModelObject": None,
        "ModelPath": None,
        "ScalerPath": None,
        "FeatureColumns": _feature_columns(data),
        "TargetMode": _target_mode(preprocessor) if preprocessor is not None else "unknown",
        "SequenceLength": _safe_int(getattr(preprocessor, "seq_len", None), default=0),
        "IsSequenceModel": bool(family == "DL"),
        "TrainedAt": trained_at or _now_iso(),
        "UsableForForecast": False,
        "Warning": str(warning or "Training failed."),
        "ResultObject": None,
        "PreprocessorObject": preprocessor,
        "DataObject": data,
        "FeatureFrame": feature_frame.copy() if isinstance(feature_frame, pd.DataFrame) else pd.DataFrame(),
        "PredictionsTest": np.array([]),
    }


def build_registry_entries(
    *,
    asset: str,
    horizon: int,
    target_col: str,
    preprocessor: Any,
    data: Any,
    feature_frame: pd.DataFrame,
    ml_trainer: Any = None,
    dl_trainer: Any = None,
    dl_failures: Optional[Mapping[str, str]] = None,
) -> list[dict[str, Any]]:
    """Build a single ML+DL registry from trainer objects and failure warnings."""
    trained_at = _now_iso()
    entries: list[dict[str, Any]] = []

    if ml_trainer is not None:
        for name, result in (getattr(ml_trainer, "results", {}) or {}).items():
            entries.append(
                build_success_entry(
                    model_name=name,
                    model_family="ML",
                    result=result,
                    asset=asset,
                    target_col=target_col,
                    horizon=horizon,
                    preprocessor=preprocessor,
                    data=data,
                    feature_frame=feature_frame,
                    trained_at=trained_at,
                )
            )

    if dl_trainer is not None:
        for name, result in (getattr(dl_trainer, "results", {}) or {}).items():
            entries.append(
                build_success_entry(
                    model_name=name,
                    model_family="DL",
                    result=result,
                    asset=asset,
                    target_col=target_col,
                    horizon=horizon,
                    preprocessor=preprocessor,
                    data=data,
                    feature_frame=feature_frame,
                    trained_at=trained_at,
                )
            )

    failures = dict(dl_failures or {})
    if dl_trainer is not None:
        failures.update(getattr(dl_trainer, "failures", {}) or {})
    for name, warning in failures.items():
        if not any(e["ModelFamily"] == "DL" and e["ModelName"] == str(name) for e in entries):
            entries.append(
                build_failure_entry(
                    model_name=name,
                    model_family="DL",
                    asset=asset,
                    target_col=target_col,
                    horizon=horizon,
                    warning=warning,
                    preprocessor=preprocessor,
                    data=data,
                    feature_frame=feature_frame,
                    trained_at=trained_at,
                )
            )

    return entries


def metadata_record(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON/CSV-safe metadata for one registry entry."""
    feature_cols = list(entry.get("FeatureColumns") or [])
    return {
        "ModelName": entry.get("ModelName"),
        "DisplayName": entry.get("DisplayName"),
        "ModelFamily": entry.get("ModelFamily"),
        "Asset": entry.get("Asset"),
        "TargetColumn": entry.get("TargetColumn"),
        "Horizon": _safe_int(entry.get("Horizon"), default=0),
        "RMSE": _safe_float(entry.get("RMSE")),
        "MAE": _safe_float(entry.get("MAE")),
        "MAPE": _safe_float(entry.get("MAPE")),
        "R2": _safe_float(entry.get("R2")),
        "DirectionalAccuracy": _safe_float(entry.get("DirectionalAccuracy")),
        "ModelPath": entry.get("ModelPath"),
        "ScalerPath": entry.get("ScalerPath"),
        "FeatureColumnCount": len(feature_cols),
        "FeatureColumns": feature_cols,
        "TargetMode": entry.get("TargetMode"),
        "SequenceLength": _safe_int(entry.get("SequenceLength"), default=0),
        "IsSequenceModel": bool(entry.get("IsSequenceModel")),
        "TrainedAt": entry.get("TrainedAt"),
        "UsableForForecast": bool(entry.get("UsableForForecast")),
        "Warning": entry.get("Warning") or "",
    }


def registry_to_metadata_frame(entries: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([metadata_record(entry) for entry in list(entries or [])])


def registry_to_leaderboard(
    entries: Iterable[Mapping[str, Any]],
    *,
    asset: Optional[str] = None,
    include_unusable: bool = True,
    sort_by: str = "RMSE",
) -> pd.DataFrame:
    """Build a Compare Models leaderboard from the shared registry."""
    frame = registry_to_metadata_frame(entries)
    if frame.empty:
        return frame
    if asset is not None and "Asset" in frame.columns:
        frame = frame[frame["Asset"].astype(str) == str(asset)].copy()
    if not include_unusable and "UsableForForecast" in frame.columns:
        frame = frame[frame["UsableForForecast"].astype(bool)].copy()
    if frame.empty:
        return frame

    frame["Model"] = frame["DisplayName"]
    cols = [
        "Model", "ModelName", "ModelFamily", "Asset", "Horizon", "RMSE", "MAE", "MAPE", "R2",
        "DirectionalAccuracy", "TargetMode", "SequenceLength", "IsSequenceModel",
        "UsableForForecast", "Warning", "TrainedAt",
    ]
    cols = [col for col in cols if col in frame.columns]
    frame = frame[cols].copy()

    metric = sort_by if sort_by in frame.columns else "RMSE"
    ascending = False if metric in {"R2", "DirectionalAccuracy"} else True
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.sort_values(["UsableForForecast", metric], ascending=[False, ascending], na_position="last")
    frame.insert(0, "Rank", range(1, len(frame) + 1))
    return frame.reset_index(drop=True)


def get_forecast_ready_models(entries: Iterable[Mapping[str, Any]], *, asset: Optional[str] = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in list(entries or []):
        if asset is not None and str(entry.get("Asset")) != str(asset):
            continue
        if bool(entry.get("UsableForForecast")) and entry.get("ModelObject") is not None:
            result.append(dict(entry))
    return result


def get_entry_by_display_name(entries: Iterable[Mapping[str, Any]], display_name: str) -> Optional[dict[str, Any]]:
    for entry in list(entries or []):
        if str(entry.get("DisplayName")) == str(display_name):
            return dict(entry)
    return None


def set_trained_model_registry(session_state: MutableMapping[str, Any], entries: Iterable[Mapping[str, Any]]) -> None:
    live_entries = [dict(entry) for entry in list(entries or [])]
    session_state[REGISTRY_SESSION_KEY] = live_entries
    session_state[REGISTRY_METADATA_SESSION_KEY] = registry_to_metadata_frame(live_entries)


def get_trained_model_registry(session_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(entry) for entry in list(session_state.get(REGISTRY_SESSION_KEY, []) or [])]


def save_trained_model_registry(entries: Iterable[Mapping[str, Any]], directory: Path | str = REGISTRY_PHASE_DIR) -> tuple[str, ...]:
    """Save lightweight registry metadata to artifacts/latest/model_registry."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = registry_to_metadata_frame(entries)
    csv_path = out_dir / "trained_model_registry.csv"
    json_path = out_dir / "trained_model_registry.json"
    if not frame.empty:
        frame.to_csv(csv_path, index=False)
        payload = frame.to_dict("records")
    else:
        payload = []
        pd.DataFrame().to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return (str(csv_path), str(json_path))


def load_saved_registry_metadata(directory: Path | str = REGISTRY_PHASE_DIR) -> pd.DataFrame:
    path = Path(directory) / "trained_model_registry.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
