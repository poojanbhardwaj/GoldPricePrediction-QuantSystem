"""Canonical multi-asset and horizon context for the Streamlit application."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.asset_config import ASSETS, get_target_column


SUPPORTED_ASSETS: Tuple[str, ...] = tuple(ASSETS.keys())
ASSET_DISPLAY_NAMES: Dict[str, str] = {
    asset: config.display_name for asset, config in ASSETS.items()
}
ASSET_TARGET_COLUMNS: Dict[str, str] = {
    asset: config.target_col for asset, config in ASSETS.items()
}
AVAILABLE_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)
DEFAULT_ASSET = "Gold"
DEFAULT_HORIZON = 5


def get_supported_assets() -> List[str]:
    """Return configured assets in stable display order."""
    return list(SUPPORTED_ASSETS)


def get_available_horizons() -> List[int]:
    """Return supported direct-forecast horizons in stable order."""
    return list(AVAILABLE_HORIZONS)


def get_asset_target(asset: str) -> str:
    """Return the canonical target column for an asset."""
    return get_target_column(str(asset))


def validate_asset_horizon(asset: str, horizon: Any) -> bool:
    """Raise a readable error for unsupported context and return True otherwise."""
    if asset not in SUPPORTED_ASSETS:
        raise ValueError(
            f"Unsupported asset {asset!r}. Supported assets: {', '.join(SUPPORTED_ASSETS)}"
        )
    try:
        normalized_horizon = int(horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid horizon {horizon!r}; expected a trading-day integer.") from exc
    if normalized_horizon not in AVAILABLE_HORIZONS:
        supported = ", ".join(f"{value}D" for value in AVAILABLE_HORIZONS)
        raise ValueError(
            f"Unsupported horizon {normalized_horizon}D. Supported horizons: {supported}"
        )
    return True


def _market_frame(market_data: Any) -> pd.DataFrame:
    if not isinstance(market_data, pd.DataFrame) or market_data.empty:
        return pd.DataFrame()
    frame = market_data.copy()
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).set_index("Date")
    else:
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame.loc[~frame.index.isna()]
    return frame.sort_index()


def _normalized_as_of(as_of: Any = None) -> pd.Timestamp:
    value = as_of if as_of is not None else datetime.now()
    return pd.Timestamp(value).tz_localize(None).normalize()


def _expected_latest_date(asset: str, as_of: pd.Timestamp) -> tuple[pd.Timestamp, str]:
    if asset == "Bitcoin":
        return as_of, "Calendar-day market; a short publication delay can still be normal."
    expected = as_of
    while expected.weekday() >= 5:
        expected -= pd.Timedelta(days=1)
    if expected != as_of:
        return expected, "Weekend adjustment to the latest expected weekday session."
    return expected, "Latest weekday session; exchange holidays are not inferred here."


def _session_lag(asset: str, latest: pd.Timestamp, expected: pd.Timestamp) -> int:
    if latest >= expected:
        return 0
    if asset == "Bitcoin":
        return max(int((expected - latest).days), 0)
    return len(pd.bdate_range(latest + pd.Timedelta(days=1), expected))


def build_data_freshness_table(
    market_data: Any,
    *,
    as_of: Any = None,
    stale_after_sessions: int = 2,
) -> pd.DataFrame:
    """Report per-asset freshness without claiming exchange-calendar precision."""
    frame = _market_frame(market_data)
    as_of_date = _normalized_as_of(as_of)
    master_latest = frame.index.max() if not frame.empty else pd.NaT
    rows = []

    for asset in SUPPORTED_ASSETS:
        target_col = get_asset_target(asset)
        expected, delay_reason = _expected_latest_date(asset, as_of_date)
        latest = pd.NaT
        if target_col in frame.columns:
            valid = pd.to_numeric(frame[target_col], errors="coerce").dropna()
            if not valid.empty:
                latest = pd.Timestamp(valid.index.max()).tz_localize(None).normalize()

        if pd.isna(latest):
            session_lag = None
            status = "MissingData"
            is_stale = True
            explanation = f"No usable observations were found for {target_col}."
        else:
            session_lag = _session_lag(asset, latest, expected)
            is_stale = session_lag > int(stale_after_sessions)
            if is_stale:
                status = "Stale"
                explanation = (
                    f"Latest observation trails the expected session by {session_lag} session(s)."
                )
            elif session_lag == 0:
                status = "Current"
                explanation = "Latest observation reaches the expected date for this simple calendar check."
            else:
                status = "ExpectedDelay"
                explanation = (
                    f"Latest observation trails by {session_lag} session(s), within the configured tolerance."
                )

        rows.append(
            {
                "Asset": asset,
                "TargetColumn": target_col,
                "MasterLatestDate": master_latest.date().isoformat() if not pd.isna(master_latest) else "",
                "LatestAssetDate": latest.date().isoformat() if not pd.isna(latest) else "",
                "ExpectedLatestDate": expected.date().isoformat(),
                "MarketSessionLag": session_lag,
                "FreshnessStatus": status,
                "IsStale": bool(is_stale),
                "ExpectedDelayReason": delay_reason,
                "Explanation": explanation,
                "CheckedAt": as_of_date.date().isoformat(),
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "SUPPORTED_ASSETS",
    "ASSET_DISPLAY_NAMES",
    "ASSET_TARGET_COLUMNS",
    "AVAILABLE_HORIZONS",
    "DEFAULT_ASSET",
    "DEFAULT_HORIZON",
    "get_supported_assets",
    "get_available_horizons",
    "get_asset_target",
    "validate_asset_horizon",
    "build_data_freshness_table",
]
