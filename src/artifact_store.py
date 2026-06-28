"""Permanent artifact store for research evidence tables.

The store writes every important table to project-local ``artifacts/`` files,
keeps immutable run history, and maintains latest pointers for Streamlit pages
that need to recover after restarts or cache clears.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
REGISTRY_FILENAME = "registry.json"


class ArtifactNotFoundError(FileNotFoundError):
    """Controlled error for required missing artifacts."""


def _now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _slugify(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "artifact"


def _artifact_root() -> Path:
    return Path(ARTIFACT_ROOT)


def _registry_path() -> Path:
    return _artifact_root() / REGISTRY_FILENAME


def _ensure_dirs() -> None:
    root = _artifact_root()
    (root / "latest").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)


def _empty_registry() -> Dict[str, Any]:
    return {
        "Version": 1,
        "CreatedAt": _now(),
        "UpdatedAt": _now(),
        "Runs": {},
        "Latest": {},
    }


def _read_registry() -> Dict[str, Any]:
    _ensure_dirs()
    path = _registry_path()
    if not path.exists():
        registry = _empty_registry()
        _write_registry(registry)
        return registry
    try:
        with path.open("r", encoding="utf-8") as fh:
            registry = json.load(fh)
    except Exception:
        registry = _empty_registry()
    registry.setdefault("Runs", {})
    registry.setdefault("Latest", {})
    registry.setdefault("Version", 1)
    registry.setdefault("CreatedAt", _now())
    registry.setdefault("UpdatedAt", _now())
    return registry


def _write_registry(registry: Dict[str, Any]) -> None:
    _ensure_dirs()
    registry["UpdatedAt"] = _now()
    with _registry_path().open("w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, sort_keys=True)


def _to_frame(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame({"Value": [data]})


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, (list, tuple, dict, set)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _detect_assets(df: pd.DataFrame) -> List[str]:
    if "Asset" not in df.columns:
        return []
    return sorted(str(v) for v in df["Asset"].dropna().astype(str).unique())


def _detect_horizons(df: pd.DataFrame) -> List[int]:
    if "Horizon" not in df.columns:
        return []
    raw = df["Horizon"].astype(str).str.replace("D", "", regex=False)
    values = pd.to_numeric(raw, errors="coerce").dropna().astype(int).unique().tolist()
    return sorted(int(v) for v in values)


def _artifact_filename(artifact_name: str, artifact_type: str) -> str:
    suffix = "json" if str(artifact_type).lower() == "json" else "csv"
    return f"{_slugify(artifact_name)}.{suffix}"


def create_run_id(phase_name: str) -> str:
    """Create a run id with the requested timestamp/phase slug shape."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slugify(phase_name)}"


def _unique_run_id(phase_name: str, requested: Optional[str] = None) -> str:
    run_id = requested or create_run_id(phase_name)
    runs_root = _artifact_root() / "runs"
    if not (runs_root / run_id).exists():
        return run_id
    i = 1
    while (runs_root / f"{run_id}_{i:03d}").exists():
        i += 1
    return f"{run_id}_{i:03d}"


def _write_manifest(run_id: str, phase_name: str, phase_slug: str, artifacts: Dict[str, Any], inputs: Dict[str, Any], config: Dict[str, Any], warnings: List[Any]) -> None:
    run_root = _artifact_root() / "runs" / run_id
    manifest = {
        "RunId": run_id,
        "Phase": phase_name,
        "PhaseSlug": phase_slug,
        "CreatedAt": _now(),
        "Artifacts": artifacts,
        "InputArtifactsUsed": _json_safe(inputs or {}),
        "ConfigUsed": _json_safe(config or {}),
        "Warnings": _json_safe(warnings or []),
    }
    with (run_root / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def _merge_phase_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    existing: Dict[str, Any] = {"Artifacts": {}}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except Exception:
            existing = {"Artifacts": {}}
    existing.setdefault("Artifacts", {})
    existing["UpdatedAt"] = _now()
    existing["Phase"] = metadata.get("Phase")
    existing["PhaseSlug"] = metadata.get("PhaseSlug")
    existing["RunId"] = metadata.get("RunId")
    existing["Artifacts"][metadata["ArtifactName"]] = metadata
    with path.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, sort_keys=True)


def save_artifact(run_id, phase_name, artifact_name, data, artifact_type="csv", metadata=None) -> str:
    """Save one artifact into immutable run history and latest folder."""
    _ensure_dirs()
    phase_slug = _slugify(phase_name)
    artifact_slug = _slugify(artifact_name)
    artifact_type = str(artifact_type or "csv").lower()
    filename = _artifact_filename(artifact_name, artifact_type)
    run_root = _artifact_root() / "runs" / str(run_id)
    run_phase = run_root / phase_slug
    latest_phase = _artifact_root() / "latest" / phase_slug
    run_phase.mkdir(parents=True, exist_ok=True)
    latest_phase.mkdir(parents=True, exist_ok=True)

    df = _to_frame(data)
    run_path = run_phase / filename
    latest_path = latest_phase / filename
    if artifact_type == "json":
        payload = _json_safe(data)
        with run_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        shutil.copy2(run_path, latest_path)
    else:
        df.to_csv(run_path, index=False)
        shutil.copy2(run_path, latest_path)

    meta = {
        "Phase": str(phase_name),
        "PhaseSlug": phase_slug,
        "ArtifactName": str(artifact_name),
        "ArtifactSlug": artifact_slug,
        "ArtifactType": artifact_type,
        "RunId": str(run_id),
        "CreatedAt": _now(),
        "Path": str(run_path),
        "LatestPath": str(latest_path),
        "Rows": int(len(df)),
        "Columns": list(df.columns),
        "AssetsCovered": _detect_assets(df),
        "HorizonsCovered": _detect_horizons(df),
        "InputArtifactsUsed": _json_safe((metadata or {}).get("InputArtifactsUsed", {})),
        "UploadedOverridesUsed": _json_safe((metadata or {}).get("UploadedOverridesUsed", {})),
        "ConfigUsed": _json_safe((metadata or {}).get("ConfigUsed", {})),
        "Warnings": _json_safe((metadata or {}).get("Warnings", [])),
        "IsLatestValid": True,
    }
    artifact_meta_path = run_phase / f"{artifact_slug}.metadata.json"
    latest_artifact_meta_path = latest_phase / f"{artifact_slug}.metadata.json"
    with artifact_meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    with latest_artifact_meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    _merge_phase_metadata(run_phase / "metadata.json", meta)
    _merge_phase_metadata(latest_phase / "metadata.json", meta)

    registry = _read_registry()
    registry.setdefault("Runs", {}).setdefault(
        str(run_id),
        {
            "RunId": str(run_id),
            "Phase": str(phase_name),
            "PhaseSlug": phase_slug,
            "CreatedAt": _now(),
            "ManifestPath": str(run_root / "manifest.json"),
            "Artifacts": {},
        },
    )
    registry["Runs"][str(run_id)]["Artifacts"][str(artifact_name)] = meta
    registry.setdefault("Latest", {}).setdefault(phase_slug, {"Phase": str(phase_name), "PhaseSlug": phase_slug, "Artifacts": {}})
    registry["Latest"][phase_slug]["Artifacts"][str(artifact_name)] = meta
    _write_registry(registry)
    return str(run_path)


def save_phase_artifacts(phase_name, artifacts: dict, inputs: dict = None, config: dict = None, warnings: list = None, run_id: str = None) -> dict:
    """Save all tables for one phase and update registry/latest pointers."""
    _ensure_dirs()
    artifacts = artifacts or {}
    resolved_run_id = _unique_run_id(str(phase_name), run_id)
    phase_slug = _slugify(phase_name)
    saved: Dict[str, Any] = {}
    shared_metadata = {
        "InputArtifactsUsed": inputs or {},
        "UploadedOverridesUsed": {},
        "ConfigUsed": config or {},
        "Warnings": warnings or [],
    }
    # Keep UploadedOverridesUsed simple and JSON-safe even when callers pass a source table.
    uploaded = {}
    for key, value in (inputs or {}).items():
        if isinstance(value, dict) and value.get("Source") == "UploadedOverride":
            uploaded[key] = value
    shared_metadata["UploadedOverridesUsed"] = uploaded
    for artifact_name, data in artifacts.items():
        path = save_artifact(resolved_run_id, phase_name, artifact_name, data, artifact_type="csv", metadata=shared_metadata)
        latest = _read_registry()["Latest"][phase_slug]["Artifacts"][str(artifact_name)]
        latest["Path"] = path
        saved[str(artifact_name)] = latest
    _write_manifest(resolved_run_id, str(phase_name), phase_slug, saved, inputs or {}, config or {}, warnings or [])
    registry = _read_registry()
    if resolved_run_id in registry.get("Runs", {}):
        registry["Runs"][resolved_run_id]["ManifestPath"] = str(_artifact_root() / "runs" / resolved_run_id / "manifest.json")
    _write_registry(registry)
    return {"RunId": resolved_run_id, "PhaseSlug": phase_slug, "Artifacts": saved}


def get_artifact_registry():
    """Return the registry dictionary, creating it when missing."""
    return _read_registry()


def _find_latest_meta(phase_name: str, artifact_name: str) -> Optional[Dict[str, Any]]:
    registry = _read_registry()
    phase_slug = _slugify(phase_name)
    phase_latest = registry.get("Latest", {}).get(phase_slug, {})
    artifacts = phase_latest.get("Artifacts", {})
    if artifact_name in artifacts:
        return artifacts[artifact_name]
    artifact_slug = _slugify(artifact_name)
    for name, meta in artifacts.items():
        if _slugify(name) == artifact_slug:
            return meta
    return None


def load_latest_artifact(phase_name, artifact_name, required=False):
    """Load the latest saved CSV artifact as a DataFrame."""
    meta = _find_latest_meta(str(phase_name), str(artifact_name))
    if meta is None:
        if required:
            raise ArtifactNotFoundError(f"Missing required latest artifact: {phase_name} / {artifact_name}")
        return None
    path = Path(meta.get("LatestPath") or meta.get("Path", ""))
    if not path.exists():
        if required:
            raise ArtifactNotFoundError(f"Latest artifact path is missing: {path}")
        return None
    if str(meta.get("ArtifactType", "csv")).lower() == "json":
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return pd.read_csv(path)


def list_latest_artifacts(phase_name=None):
    """Return latest artifact metadata as a DataFrame."""
    registry = _read_registry()
    rows: List[Dict[str, Any]] = []
    phase_filter = _slugify(phase_name) if phase_name else None
    for phase_slug, phase_data in registry.get("Latest", {}).items():
        if phase_filter and phase_slug != phase_filter:
            continue
        for artifact_name, meta in phase_data.get("Artifacts", {}).items():
            rows.append(
                {
                    "Phase": meta.get("Phase", phase_data.get("Phase", phase_slug)),
                    "PhaseSlug": phase_slug,
                    "ArtifactName": artifact_name,
                    "RunId": meta.get("RunId", ""),
                    "CreatedAt": meta.get("CreatedAt", ""),
                    "Rows": meta.get("Rows", 0),
                    "Columns": len(meta.get("Columns", [])),
                    "AssetsCovered": "; ".join(meta.get("AssetsCovered", [])),
                    "HorizonsCovered": "; ".join(str(h) for h in meta.get("HorizonsCovered", [])),
                    "Path": meta.get("LatestPath") or meta.get("Path", ""),
                    "Warnings": "; ".join(str(w) for w in meta.get("Warnings", [])),
                    "IsLatestValid": meta.get("IsLatestValid", False),
                }
            )
    return pd.DataFrame(rows)


def _read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def _uploaded_name(uploaded_file: Any) -> str:
    return str(getattr(uploaded_file, "name", "uploaded_file"))


def resolve_artifact(phase_name, artifact_name, uploaded_file=None, prefer_uploaded=False, required=False):
    """Resolve an artifact from uploaded override or latest saved evidence.

    Returns a dictionary with Data, Source, RunId, Rows, CreatedAt, Status, and
    Path so pages can display exactly where their inputs came from.
    """
    if uploaded_file is not None and prefer_uploaded:
        data = _read_uploaded_csv(uploaded_file)
        return {
            "Artifact": str(artifact_name),
            "Phase": str(phase_name),
            "Data": data,
            "Source": "UploadedOverride",
            "RunId": "",
            "Rows": int(len(data)),
            "CreatedAt": "",
            "Status": "Loaded",
            "Path": _uploaded_name(uploaded_file),
        }
    meta = _find_latest_meta(str(phase_name), str(artifact_name))
    if meta is not None:
        data = load_latest_artifact(phase_name, artifact_name, required=required)
        if data is not None:
            rows = int(len(data)) if hasattr(data, "__len__") and not isinstance(data, dict) else int(meta.get("Rows", 0))
            return {
                "Artifact": str(artifact_name),
                "Phase": str(phase_name),
                "Data": data,
                "Source": "LatestSavedArtifact",
                "RunId": meta.get("RunId", ""),
                "Rows": rows,
                "CreatedAt": meta.get("CreatedAt", ""),
                "Status": "Loaded",
                "Path": meta.get("LatestPath") or meta.get("Path", ""),
            }
    if uploaded_file is not None:
        data = _read_uploaded_csv(uploaded_file)
        return {
            "Artifact": str(artifact_name),
            "Phase": str(phase_name),
            "Data": data,
            "Source": "UploadedOverride",
            "RunId": "",
            "Rows": int(len(data)),
            "CreatedAt": "",
            "Status": "Loaded",
            "Path": _uploaded_name(uploaded_file),
        }
    if required:
        raise ArtifactNotFoundError(f"Missing required artifact: {phase_name} / {artifact_name}")
    return {
        "Artifact": str(artifact_name),
        "Phase": str(phase_name),
        "Data": None,
        "Source": "Missing",
        "RunId": "",
        "Rows": 0,
        "CreatedAt": "",
        "Status": "MissingRequired" if required else "MissingOptional",
        "Path": "",
    }


def validate_required_artifacts(required_specs):
    """Return diagnostics for required/optional artifact availability."""
    rows: List[Dict[str, Any]] = []
    for spec in required_specs or []:
        phase = spec.get("phase_name") or spec.get("Phase") or spec.get("phase")
        artifact = spec.get("artifact_name") or spec.get("Artifact") or spec.get("artifact")
        required = bool(spec.get("required", False))
        try:
            resolved = resolve_artifact(phase, artifact, required=required)
        except ArtifactNotFoundError as exc:
            resolved = {
                "Artifact": artifact,
                "Phase": phase,
                "Source": "Missing",
                "RunId": "",
                "Rows": 0,
                "CreatedAt": "",
                "Status": "MissingRequired",
                "Path": "",
                "Error": str(exc),
            }
        rows.append({k: v for k, v in resolved.items() if k != "Data"})
    return pd.DataFrame(rows)


def build_input_source_table(required_or_optional_inputs):
    """Build display table: Artifact | Source | RunId | Rows | CreatedAt | Status."""
    rows: List[Dict[str, Any]] = []
    for item in required_or_optional_inputs or []:
        if isinstance(item, dict) and "Data" in item:
            rows.append(
                {
                    "Artifact": item.get("Artifact", ""),
                    "Source": item.get("Source", "Missing"),
                    "RunId": item.get("RunId", ""),
                    "Rows": item.get("Rows", 0),
                    "CreatedAt": item.get("CreatedAt", ""),
                    "Status": item.get("Status", ""),
                    "Phase": item.get("Phase", ""),
                    "Path": item.get("Path", ""),
                }
            )
        elif isinstance(item, dict):
            phase = item.get("phase_name") or item.get("Phase") or item.get("phase")
            artifact = item.get("artifact_name") or item.get("Artifact") or item.get("artifact")
            required = bool(item.get("required", False))
            try:
                resolved = resolve_artifact(phase, artifact, uploaded_file=item.get("uploaded_file"), prefer_uploaded=bool(item.get("prefer_uploaded", False)), required=required)
            except ArtifactNotFoundError:
                resolved = {"Artifact": artifact, "Phase": phase, "Source": "Missing", "RunId": "", "Rows": 0, "CreatedAt": "", "Status": "MissingRequired", "Path": ""}
            rows.append({k: v for k, v in resolved.items() if k != "Data"})
    columns = ["Artifact", "Source", "RunId", "Rows", "CreatedAt", "Status", "Phase", "Path"]
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "ARTIFACT_ROOT",
    "ArtifactNotFoundError",
    "build_input_source_table",
    "create_run_id",
    "get_artifact_registry",
    "list_latest_artifacts",
    "load_latest_artifact",
    "resolve_artifact",
    "save_artifact",
    "save_phase_artifacts",
    "validate_required_artifacts",
]
