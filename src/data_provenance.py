"""Honest source and freshness labels for saved and refreshed research views."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def _source_label(source_type: Any) -> str:
    source = str(source_type or "").casefold()
    if any(token in source for token in ("session", "refresh", "latest")):
        return "Latest refreshed research snapshot"
    if any(token in source for token in ("saved", "artifact", "last_good", "checked-in")):
        return "Saved research snapshot"
    if any(token in source for token in ("cached", "dataset", "master", "placeholder")):
        return "Cached market snapshot"
    return "Source unavailable"


def get_snapshot_freshness_label(
    latest_date: Any,
    source_type: Any = None,
    today: Any = None,
) -> dict[str, Any]:
    """Return source, date, age, and freshness as separate truthful fields."""
    parsed = pd.to_datetime(latest_date, errors="coerce")
    if pd.isna(parsed):
        return {
            "source_label": _source_label(source_type),
            "latest_date_label": "Latest source date: Unknown",
            "age_label": "Snapshot age: Unknown",
            "freshness_label": "Freshness: Unknown",
            "freshness_status": "Unknown",
            "is_stale": False,
        }
    reference = pd.Timestamp(today if today is not None else date.today()).normalize()
    source_date = pd.Timestamp(parsed).tz_localize(None).normalize()
    age = max(int((reference - source_date).days), 0)
    freshness = "Recent" if age <= 2 else "Delayed" if age <= 5 else "Stale"
    return {
        "source_label": _source_label(source_type),
        "latest_date_label": f"Latest source date: {source_date.date().isoformat()}",
        "age_label": f"Snapshot age: {age} calendar day{'s' if age != 1 else ''}",
        "freshness_label": f"Freshness: {freshness}",
        "freshness_status": freshness,
        "is_stale": freshness == "Stale",
    }


__all__ = ["get_snapshot_freshness_label"]
