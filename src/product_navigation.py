"""Pure navigation-label resolution shared by the Streamlit product shell."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import Any


def resolve_navigation_label(
    page_name: Any,
    canonical_pages: Collection[str],
    aliases: Mapping[str, str],
    *,
    default_page: str = "Research Dashboard",
) -> tuple[str, str | None]:
    """Resolve canonical, legacy, and known-looking labels without hiding random input."""
    label = str(page_name or default_page).strip()
    canonical_lookup = {str(page).casefold(): str(page) for page in canonical_pages}
    alias_lookup = {str(alias).casefold(): str(target) for alias, target in aliases.items()}
    known = canonical_lookup.get(label.casefold()) or alias_lookup.get(label.casefold())
    if known:
        return known, None

    normalized = re.sub(r"[^a-z0-9]+", " ", label.casefold()).strip()
    if any(token in normalized for token in ("forecast", "train", "model")):
        target = "Forecast Explorer"
    elif any(token in normalized for token in ("advanced", "diagnostic")):
        target = "Advanced Diagnostics"
    else:
        return label, None
    return target, f"Unknown navigation label normalized to: {target}"


def select_available_page(
    resolved_page: str,
    available_pages: Collection[str],
    *,
    default_page: str = "Research Dashboard",
) -> str:
    """Keep valid pages and reserve Dashboard fallback for truly unavailable labels."""
    return resolved_page if resolved_page in set(available_pages) else default_page


__all__ = ["resolve_navigation_label", "select_available_page"]
