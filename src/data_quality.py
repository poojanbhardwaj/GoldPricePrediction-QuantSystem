# src/data_quality.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    status: str
    last_updated: str
    rows: int
    columns: int
    missing_values: int
    duplicate_dates: int
    stale_data_warning: bool
    outlier_count: int
    available_assets: List[str]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_data_quality(
    df: pd.DataFrame,
    *,
    asset_columns: List[str],
    max_stale_days: int = 7,
    outlier_z_threshold: float = 5.0,
) -> DataQualityReport:
    if df is None or df.empty:
        return DataQualityReport(
            status="Error",
            last_updated="N/A",
            rows=0,
            columns=0,
            missing_values=0,
            duplicate_dates=0,
            stale_data_warning=True,
            outlier_count=0,
            available_assets=[],
            message="DataFrame is empty.",
        )

    clean = df.copy()

    if not isinstance(clean.index, pd.DatetimeIndex):
        if "Date" in clean.columns:
            clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
            clean = clean.set_index("Date")
        else:
            return DataQualityReport(
                status="Error",
                last_updated="N/A",
                rows=len(df),
                columns=len(df.columns),
                missing_values=int(df.isna().sum().sum()),
                duplicate_dates=0,
                stale_data_warning=True,
                outlier_count=0,
                available_assets=[],
                message="Missing DatetimeIndex or Date column.",
            )

    clean = clean.sort_index()

    missing_values = int(clean.isna().sum().sum())
    duplicate_dates = int(clean.index.duplicated().sum())

    valid_dates = clean.index.dropna()
    last_updated = valid_dates.max() if len(valid_dates) else None

    if last_updated is None:
        stale = True
        last_updated_str = "N/A"
    else:
        last_updated_str = str(last_updated.date())
        now = pd.Timestamp.now(tz=None).normalize()
        stale = (now - pd.Timestamp(last_updated).normalize()).days > max_stale_days

    available_assets = [col for col in asset_columns if col in clean.columns]

    outlier_count = 0
    for col in available_assets:
        series = pd.to_numeric(clean[col], errors="coerce").dropna()
        returns = series.pct_change().dropna()
        if len(returns) < 30:
            continue
        z = (returns - returns.mean()) / returns.std(ddof=1)
        outlier_count += int((z.abs() > outlier_z_threshold).sum())

    if missing_values == 0 and duplicate_dates == 0 and not stale and outlier_count == 0:
        status = "Healthy"
        message = "Data looks healthy."
    elif duplicate_dates > 0 or stale or missing_values > 0 or outlier_count > 0:
        status = "Warning"
        message = "Data has warnings. Review missing values, stale data, duplicate dates, or outliers."
    else:
        status = "Error"
        message = "Data health check failed."

    return DataQualityReport(
        status=status,
        last_updated=last_updated_str,
        rows=int(len(clean)),
        columns=int(len(clean.columns)),
        missing_values=missing_values,
        duplicate_dates=duplicate_dates,
        stale_data_warning=bool(stale),
        outlier_count=int(outlier_count),
        available_assets=available_assets,
        message=message,
    )


if __name__ == "__main__":
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100, freq="B")
    demo = pd.DataFrame({"Gold_Close": np.linspace(2000, 2200, len(dates))}, index=dates)
    print(check_data_quality(demo, asset_columns=["Gold_Close", "Silver_Close"]).to_dict())
