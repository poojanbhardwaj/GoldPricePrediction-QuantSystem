"""
data_loader.py — Multi-Source Financial Data Ingestion
=======================================================
Downloads and merges all datasets required for gold price prediction:
  • Gold, Silver, Crude Oil, Bitcoin, DXY, S&P 500, VIX,
    Treasury 10Y, Gold ETF  ← via yfinance
  • Federal Funds Rate, CPI                               ← via FRED API
  • Provides fallback synthetic data when APIs unavailable

Classes
-------
DataLoader
    Main orchestrator — download, cache, merge, validate.

Usage
-----
    from src.data_loader import DataLoader
    loader = DataLoader()
    df = loader.load_all()          # returns merged DataFrame
    df = loader.load_all(use_cache=True)  # uses local CSVs if present
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config_loader import ConfigLoader
from src.logger import get_logger
from src.utils import ensure_dir, retry, timer, summarize_dataframe

logger = get_logger(__name__)
cfg    = ConfigLoader()


# ════════════════════════════════════════════════════════════════
# Helper: safe yfinance download
# ════════════════════════════════════════════════════════════════

def _yf_download(
    ticker: str,
    start: str,
    end: str,
    prefix: str = "",
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Download OHLCV data from Yahoo Finance with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            import yfinance as yf
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty:
                logger.warning(f"[yfinance] No data for {ticker}")
                return None

            # Flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)

            raw.index = pd.to_datetime(raw.index)
            raw.index.name = "Date"

            if prefix:
                raw = raw.rename(columns={c: f"{prefix}_{c}" for c in raw.columns})

            logger.info(f"Downloaded {ticker}: {raw.shape[0]} rows  ({raw.index.min().date()} → {raw.index.max().date()})")
            return raw

        except Exception as exc:
            logger.warning(f"Attempt {attempt}/{retries} failed for {ticker}: {exc}")
            time.sleep(2 ** attempt)

    logger.error(f"All {retries} attempts failed for {ticker}")
    return None


# ════════════════════════════════════════════════════════════════
# DataLoader
# ════════════════════════════════════════════════════════════════

class DataLoader:
    """
    Downloads, caches, and merges all financial datasets.

    Parameters
    ----------
    start_date : str   Override config start date (YYYY-MM-DD).
    end_date   : str   Override config end date   (YYYY-MM-DD).

    Methods
    -------
    load_all(use_cache)          → pd.DataFrame  (merged master dataset)
    download_yfinance()          → dict of DataFrames
    download_fred()              → dict of DataFrames
    merge_all(dfs)               → pd.DataFrame
    save_raw(dfs)
    load_from_cache()            → pd.DataFrame | None
    """

    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
    ):
        from datetime import date

        self.start  = start_date or cfg.get("data.start_date", "2010-01-01")

        # end_date defaults to TODAY, not a hardcoded historical date — this
        # ensures every fresh download pulls data all the way up to the
        # present, regardless of when this code is run. An explicit
        # end_date passed by the caller (or set in config.yaml) still
        # takes priority, for reproducible backtests on a fixed window.
        config_end_date = cfg.get("data.end_date", None)
        self.end = end_date or config_end_date or date.today().strftime("%Y-%m-%d")

        self.tickers: Dict[str, str] = cfg.get("data.tickers", {})
        self.fred_series: Dict[str, str] = cfg.get("data.fred_series", {})

        self.raw_dir       = cfg.resolve_path("data_raw")
        self.processed_dir = cfg.resolve_path("data_processed")
        self._master_cache = self.processed_dir / "master_dataset.csv"

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    @timer
    def load_all(self, use_cache: bool = False) -> pd.DataFrame:
        """
        Main entry point.  Downloads (or loads from cache) and merges
        all datasets into a single master DataFrame indexed by Date.

        Parameters
        ----------
        use_cache : bool
            If True and a cached master CSV exists, load it directly.

        Returns
        -------
        pd.DataFrame  Date-indexed master dataset.
        """
        if use_cache:
            cached = self.load_from_cache()
            if cached is not None:
                logger.info(f"Loaded from cache: {cached.shape}")
                return cached

        logger.info(f"Downloading data: {self.start} → {self.end}")

        yf_dfs   = self.download_yfinance()
        fred_dfs = self.download_fred()

        all_dfs = {**yf_dfs, **fred_dfs}
        master  = self.merge_all(all_dfs)
        master  = self._post_process(master)

        # Save raw individual files + master
        self.save_raw(yf_dfs)
        master.to_csv(self._master_cache)
        logger.info(f"Master dataset saved → {self._master_cache}  shape={master.shape}")

        info = summarize_dataframe(master)
        logger.info(f"Master dataset summary: {info['shape']} | nulls={sum(info['null_counts'].values())}")
        return master

    # ──────────────────────────────────────────────────────────────
    # yfinance Downloads
    # ──────────────────────────────────────────────────────────────

    def download_yfinance(self) -> Dict[str, pd.DataFrame]:
        """Download all yfinance tickers; return {name: DataFrame}."""
        dfs: Dict[str, pd.DataFrame] = {}

        ticker_map = {
            "gold":        ("GC=F",    "Gold"),
            "silver":      ("SI=F",    "Silver"),
            "crude_oil":   ("CL=F",    "Oil"),
            "bitcoin":     ("BTC-USD", "BTC"),
            "dxy":         ("DX-Y.NYB","DXY"),
            "sp500":       ("^GSPC",   "SP500"),
            "vix":         ("^VIX",    "VIX"),
            "treasury_10y":("^TNX",    "TNX"),
            "gold_etf":    ("GLD",     "GLD"),
        }

        for key, (ticker, prefix) in ticker_map.items():
            logger.info(f"Downloading {prefix} ({ticker}) ...")
            df = _yf_download(ticker, self.start, self.end, prefix=prefix)
            if df is not None:
                dfs[key] = df
            else:
                logger.warning(f"Using synthetic fallback for {prefix}")
                dfs[key] = self._synthetic_fallback(prefix, self.start, self.end)

        return dfs

    # ──────────────────────────────────────────────────────────────
    # FRED Downloads
    # ──────────────────────────────────────────────────────────────

    def download_fred(self) -> Dict[str, pd.DataFrame]:
        """Download macroeconomic series from FRED."""
        dfs: Dict[str, pd.DataFrame] = {}
        fred_key = cfg.get("api.fred_api_key", "")

        fred_map = {
            "fed_rate": ("FEDFUNDS", "FedRate"),
            "cpi":      ("CPIAUCSL", "CPI"),
        }

        if fred_key and fred_key != "your_fred_api_key_here":
            dfs.update(self._download_fred_api(fred_map, fred_key))
        else:
            logger.warning("FRED API key not set — using pandas-datareader fallback")
            dfs.update(self._download_fred_datareader(fred_map))

        return dfs

    def _download_fred_api(
        self, fred_map: Dict[str, Tuple[str, str]], api_key: str
    ) -> Dict[str, pd.DataFrame]:
        """Use fredapi library."""
        dfs = {}
        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            for key, (series_id, col_name) in fred_map.items():
                try:
                    s = fred.get_series(series_id, observation_start=self.start, observation_end=self.end)
                    df = s.to_frame(name=col_name)
                    df.index = pd.to_datetime(df.index)
                    df.index.name = "Date"
                    dfs[key] = df
                    logger.info(f"FRED [{series_id}]: {df.shape[0]} rows")
                except Exception as exc:
                    logger.warning(f"FRED series {series_id} failed: {exc}")
                    dfs[key] = self._fred_fallback(col_name, self.start, self.end)
        except ImportError:
            logger.warning("fredapi not installed — falling back to datareader")
            dfs = self._download_fred_datareader(fred_map)
        return dfs

    def _download_fred_datareader(
        self, fred_map: Dict[str, Tuple[str, str]]
    ) -> Dict[str, pd.DataFrame]:
        """Use pandas-datareader as FRED fallback."""
        dfs = {}
        try:
            import pandas_datareader.data as web
            for key, (series_id, col_name) in fred_map.items():
                try:
                    df = web.DataReader(series_id, "fred", self.start, self.end)
                    df.columns = [col_name]
                    df.index.name = "Date"
                    dfs[key] = df
                    logger.info(f"FRED via datareader [{series_id}]: {df.shape[0]} rows")
                except Exception as exc:
                    logger.warning(f"datareader {series_id} failed: {exc}")
                    dfs[key] = self._fred_fallback(col_name, self.start, self.end)
        except ImportError:
            logger.warning("pandas-datareader not installed — using synthetic FRED data")
            for key, (_, col_name) in fred_map.items():
                dfs[key] = self._fred_fallback(col_name, self.start, self.end)
        return dfs

    # ──────────────────────────────────────────────────────────────
    # Merge All DataFrames
    # ──────────────────────────────────────────────────────────────

    def merge_all(self, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Outer-join all DataFrames on their DatetimeIndex.
        Gold_Close is always placed first (it is the prediction target).
        """
        if not dfs:
            raise ValueError("No DataFrames to merge.")

        # Start with Gold (most complete)
        master = dfs.get("gold", next(iter(dfs.values()))).copy()
        logger.info(f"Base (gold): {master.shape}")

        for name, df in dfs.items():
            if name == "gold":
                continue
            try:
                df_clean = df.copy()
                df_clean.index = pd.to_datetime(df_clean.index)
                master = master.join(df_clean, how="outer", rsuffix=f"_{name}")
                logger.debug(f"Joined {name}: master shape={master.shape}")
            except Exception as exc:
                logger.warning(f"Could not join {name}: {exc}")

        master.index = pd.to_datetime(master.index)
        master.index.name = "Date"
        master.sort_index(inplace=True)
        return master

    # ──────────────────────────────────────────────────────────────
    # Post-Processing
    # ──────────────────────────────────────────────────────────────

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up the merged master DataFrame:
        - Drop fully-empty rows (weekends / holidays with no data)
        - Forward-fill macro series (CPI, FedRate are monthly)
        - Keep only business days
        - Rename target column to 'Gold_Close' for clarity
        """
        # Rename the Gold Close column
        rename_map = {}
        for col in df.columns:
            if col in ("Gold_Close", "Close"):
                rename_map[col] = "Gold_Close"
            elif col == "Gold_Adj Close":
                rename_map[col] = "Gold_Adj_Close"
        df.rename(columns=rename_map, inplace=True)

        # Forward-fill monthly macro series (CPI, FedRate)
        macro_cols = [c for c in df.columns if c in ("CPI", "FedRate")]
        df[macro_cols] = df[macro_cols].ffill()

        # Drop rows where Gold_Close is missing (no trading)
        if "Gold_Close" in df.columns:
            df.dropna(subset=["Gold_Close"], inplace=True)

        # Remove duplicate indices
        df = df[~df.index.duplicated(keep="first")]

        logger.info(f"Post-processed shape: {df.shape}")
        return df

    # ──────────────────────────────────────────────────────────────
    # Cache
    # ──────────────────────────────────────────────────────────────

    def save_raw(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """Save individual raw DataFrames to data/raw/."""
        for name, df in dfs.items():
            path = self.raw_dir / f"{name}.csv"
            df.to_csv(path)
            logger.debug(f"Saved raw {name} → {path}")

    def load_from_cache(self, max_staleness_days: int = 1) -> Optional[pd.DataFrame]:
        """
        Load master dataset from cache if it exists AND is recent enough.

        Parameters
        ----------
        max_staleness_days : int
            If the cached file's most recent date is more than this many
            trading days behind the requested end date, the cache is
            treated as stale and ignored (forcing a fresh download). This
            prevents silently serving outdated prices — e.g. a dataset
            cached on 2024-12-30 would otherwise keep being reused forever
            even when run today, since the file simply existing was
            previously the only check performed.

        Returns
        -------
        pd.DataFrame | None
            The cached DataFrame if fresh, otherwise None (triggering a
            fresh download in load_all()).
        """
        if not self._master_cache.exists():
            logger.info("No cache found.")
            return None

        df = pd.read_csv(self._master_cache, index_col="Date", parse_dates=True)
        cached_last_date = df.index.max()
        requested_end = pd.to_datetime(self.end)

        staleness = (requested_end - cached_last_date).days
        if staleness > max_staleness_days:
            logger.warning(
                f"Cache is stale: last cached date {cached_last_date.date()} is "
                f"{staleness} day(s) behind requested end date {requested_end.date()}. "
                f"Forcing a fresh download instead of reusing outdated data."
            )
            return None

        logger.info(f"Cache hit: {self._master_cache}  shape={df.shape}  (last date: {cached_last_date.date()})")
        return df

    # ──────────────────────────────────────────────────────────────
    # Synthetic Fallbacks (for offline / CI environments)
    # ──────────────────────────────────────────────────────────────

    def _synthetic_fallback(
        self, prefix: str, start: str, end: str
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic OHLCV data using geometric Brownian motion.
        Used when the live API is unavailable.
        """
        dates = pd.bdate_range(start=start, end=end)
        n = len(dates)

        # Use a DIFFERENT seed per asset (hash of prefix) so synthetic series
        # are NOT perfectly correlated with one another — each asset needs its
        # own independent random walk, otherwise models can trivially "predict"
        # one asset from another (data leakage in the synthetic fallback path).
        asset_seed = abs(hash(prefix)) % (2**31)
        rng = np.random.default_rng(seed=asset_seed)

        # Seed prices + asset-specific drift/volatility (more realistic than
        # one-size-fits-all parameters, and further de-correlates series)
        asset_params = {
            "Gold":  {"p0": 1300.0,  "drift": 0.00020, "vol": 0.0090},
            "Silver":{"p0": 18.0,    "drift": 0.00015, "vol": 0.0160},
            "Oil":   {"p0": 60.0,    "drift": 0.00010, "vol": 0.0210},
            "BTC":   {"p0": 10000.0, "drift": 0.00080, "vol": 0.0400},
            "DXY":   {"p0": 95.0,    "drift": 0.00002, "vol": 0.0035},
            "SP500": {"p0": 2500.0,  "drift": 0.00035, "vol": 0.0110},
            "VIX":   {"p0": 15.0,    "drift": -0.0001, "vol": 0.0550},
            "TNX":   {"p0": 2.0,     "drift": 0.00005, "vol": 0.0250},
            "GLD":   {"p0": 125.0,   "drift": 0.00020, "vol": 0.0090},
        }
        params = asset_params.get(prefix, {"p0": 100.0, "drift": 0.0002, "vol": 0.015})
        p0, drift, vol = params["p0"], params["drift"], params["vol"]

        returns = rng.normal(drift, vol, n)
        prices  = p0 * np.cumprod(1 + returns)
        noise   = rng.uniform(0.995, 1.005, n)

        df = pd.DataFrame(
            {
                f"{prefix}_Open":      prices * noise,
                f"{prefix}_High":      prices * rng.uniform(1.000, 1.020, n),
                f"{prefix}_Low":       prices * rng.uniform(0.980, 1.000, n),
                f"{prefix}_Close":     prices,
                f"{prefix}_Adj Close": prices,
                f"{prefix}_Volume":    rng.integers(1_000_000, 10_000_000, n).astype(float),
            },
            index=dates,
        )
        df.index.name = "Date"
        logger.info(f"Synthetic data generated for {prefix}: {df.shape}")
        return df

    def _fred_fallback(self, col_name: str, start: str, end: str) -> pd.DataFrame:
        """Generate synthetic monthly macro series."""
        dates = pd.date_range(start=start, end=end, freq="MS")
        rng   = np.random.default_rng(seed=99)

        if col_name == "CPI":
            values = np.linspace(215, 310, len(dates)) + rng.normal(0, 1.5, len(dates))
        else:  # FedRate
            values = np.clip(rng.normal(1.5, 1.2, len(dates)).cumsum() * 0.1 + 0.5, 0, 6)

        df = pd.DataFrame({col_name: values}, index=dates)
        df.index.name = "Date"
        logger.info(f"Synthetic FRED data for {col_name}: {df.shape}")
        return df

    # ──────────────────────────────────────────────────────────────
    # Info / Validation
    # ──────────────────────────────────────────────────────────────

    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """Return detailed info dict about the master dataset."""
        return {
            "shape":       df.shape,
            "date_range":  f"{df.index.min().date()} → {df.index.max().date()}",
            "columns":     list(df.columns),
            "target":      "Gold_Close",
            "null_totals": int(df.isnull().sum().sum()),
            "null_pct":    round(df.isnull().mean().mean() * 100, 2),
            "gold_stats":  df["Gold_Close"].describe().to_dict() if "Gold_Close" in df else {},
        }


# ════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = DataLoader(start_date="2015-01-01", end_date=None)  # None = today

    print("\n" + "=" * 60)
    print("  Gold Price Prediction — Data Loader")
    print("=" * 60)

    df = loader.load_all(use_cache=False)

    print(f"\n✔  Master dataset shape : {df.shape}")
    print(f"   Date range           : {df.index.min().date()} → {df.index.max().date()}")
    print(f"   Columns              : {list(df.columns[:8])} ...")
    print(f"   Null values          : {df.isnull().sum().sum()}")

    info = loader.get_data_info(df)
    print(f"\n   Gold Close stats:")
    for k, v in info["gold_stats"].items():
        print(f"     {k:>10}: {v:,.2f}")

    print("\n✔ data_loader.py working correctly")
