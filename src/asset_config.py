# src/asset_config.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AssetConfig:
    display_name: str
    target_col: str
    symbol_hint: str


ASSETS: Dict[str, AssetConfig] = {
    "Gold": AssetConfig("Gold", "Gold_Close", "GC=F / XAUUSD"),
    "Silver": AssetConfig("Silver", "Silver_Close", "SI=F / XAGUSD"),
    "Crude Oil": AssetConfig("Crude Oil", "Oil_Close", "CL=F"),
    "Bitcoin": AssetConfig("Bitcoin", "BTC_Close", "BTC-USD"),
    "S&P 500": AssetConfig("S&P 500", "SP500_Close", "^GSPC"),
    "Gold ETF": AssetConfig("Gold ETF", "GLD_Close", "GLD"),
}


def get_asset_names() -> List[str]:
    return list(ASSETS.keys())


def get_target_column(asset_name: str) -> str:
    if asset_name not in ASSETS:
        valid = ", ".join(get_asset_names())
        raise ValueError(f"Unknown asset: {asset_name}. Valid assets: {valid}")
    return ASSETS[asset_name].target_col


def get_asset_config(asset_name: str) -> AssetConfig:
    if asset_name not in ASSETS:
        valid = ", ".join(get_asset_names())
        raise ValueError(f"Unknown asset: {asset_name}. Valid assets: {valid}")
    return ASSETS[asset_name]
