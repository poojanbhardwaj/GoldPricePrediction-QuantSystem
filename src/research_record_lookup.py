"""Canonical asset and horizon matching for saved research records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Iterable

import pandas as pd

from src.asset_config import ASSETS


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _asset_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for asset, config in ASSETS.items():
        values = {asset, config.display_name, config.target_col}
        values.update(part.strip() for part in config.symbol_hint.split("/") if part.strip())
        for value in values:
            aliases[_token(value)] = asset
    aliases.update(
        {
            "btc": "Bitcoin",
            "bitcoinusd": "Bitcoin",
            "sp500": "S&P 500",
            "sandp500": "S&P 500",
            "oil": "Crude Oil",
            "crudeoil": "Crude Oil",
            "gld": "Gold ETF",
        }
    )
    return aliases


_ASSET_ALIASES = _asset_aliases()
_HORIZON_PATTERN = re.compile(r"^(\d+)\s*(?:d|day|days|tradingday|tradingdays)?$", re.I)


def normalize_asset_key(asset: Any) -> str:
    """Return the configured canonical asset key, or a cleaned unknown label."""
    raw = str(asset or "").strip()
    if _token(raw) in {"all", ""}:
        return "ALL" if _token(raw) == "all" else ""
    return _ASSET_ALIASES.get(_token(raw), raw)


def normalize_asset_display_name(asset: Any) -> str:
    """Return the configured user-facing asset name."""
    key = normalize_asset_key(asset)
    return ASSETS[key].display_name if key in ASSETS else key


def normalize_horizon_key(horizon: Any) -> int:
    """Normalize common horizon labels to an integer trading-day key."""
    if horizon is None or (isinstance(horizon, float) and pd.isna(horizon)):
        return 0
    if isinstance(horizon, (int, float)) and not isinstance(horizon, bool):
        return int(horizon)
    compact = re.sub(r"[-_]", " ", str(horizon).strip())
    compact = re.sub(r"\s+", "", compact)
    match = _HORIZON_PATTERN.fullmatch(compact)
    if not match:
        raise ValueError(f"Unsupported horizon label: {horizon!r}")
    return int(match.group(1))


def _records_frame(records: Any) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, Mapping):
        return pd.DataFrame([dict(records)])
    if isinstance(records, Iterable) and not isinstance(records, (str, bytes)):
        return pd.DataFrame(list(records))
    return pd.DataFrame()


def find_asset_horizon_records(records: Any, asset: Any, horizon: Any) -> pd.DataFrame:
    """Return all exact canonical matches while preserving partial record fields."""
    frame = _records_frame(records)
    if frame.empty or "Asset" not in frame.columns:
        return frame.iloc[0:0].copy()
    asset_key = normalize_asset_key(asset)
    horizon_key = normalize_horizon_key(horizon)
    asset_mask = frame["Asset"].map(normalize_asset_key).eq(asset_key)
    horizon_column = "Horizon" if "Horizon" in frame.columns else (
        "BestHorizon" if "BestHorizon" in frame.columns else None
    )
    if horizon_column is None:
        return frame.loc[asset_mask].copy()
    normalized_horizons = frame[horizon_column].map(
        lambda value: normalize_horizon_key(value) if pd.notna(value) else 0
    )
    return frame.loc[asset_mask & normalized_horizons.eq(horizon_key)].copy()


def find_asset_horizon_record(records: Any, asset: Any, horizon: Any) -> dict[str, Any] | None:
    """Return the strongest exact record without requiring every optional field."""
    matches = find_asset_horizon_records(records, asset, horizon)
    if matches.empty:
        return None
    score = pd.to_numeric(
        matches.get("OpportunityScore", pd.Series(index=matches.index, dtype=float)),
        errors="coerce",
    ).fillna(-1.0)
    completeness = matches.notna().sum(axis=1)
    ranked = matches.assign(_lookup_score=score, _lookup_fields=completeness).sort_values(
        ["_lookup_score", "_lookup_fields"], ascending=[False, False], kind="stable"
    )
    return ranked.iloc[0].drop(labels=["_lookup_score", "_lookup_fields"]).to_dict()


def available_asset_keys(records: Any) -> list[str]:
    frame = _records_frame(records)
    if frame.empty or "Asset" not in frame.columns:
        return []
    return list(dict.fromkeys(frame["Asset"].dropna().map(normalize_asset_key)))


def available_horizon_keys(records: Any) -> list[int]:
    frame = _records_frame(records)
    column = "Horizon" if "Horizon" in frame.columns else (
        "BestHorizon" if "BestHorizon" in frame.columns else None
    )
    if frame.empty or column is None:
        return []
    values = []
    for value in frame[column].dropna():
        try:
            key = normalize_horizon_key(value)
        except ValueError:
            continue
        if key not in values:
            values.append(key)
    return sorted(values)


__all__ = [
    "available_asset_keys",
    "available_horizon_keys",
    "find_asset_horizon_record",
    "find_asset_horizon_records",
    "normalize_asset_display_name",
    "normalize_asset_key",
    "normalize_horizon_key",
]
