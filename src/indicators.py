"""
indicators.py — Technical Indicator Generation Engine
========================================================
Computes the full suite of technical analysis indicators used by
professional quant traders, applied to the Gold OHLCV columns.

Indicators implemented
-----------------------
Trend       : SMA, EMA, ADX, Ichimoku Cloud
Momentum    : RSI, Stochastic RSI, Williams %R, ROC, Momentum
Volatility  : Bollinger Bands, ATR
Volume      : OBV, VWAP, Money Flow Index (MFI)
Composite   : MACD, CCI

All indicators are computed from scratch with pandas/numpy
(no closed-source TA-Lib dependency) so the project remains
pip-installable everywhere, including Streamlit Cloud.

Usage
-----
    from src.indicators import TechnicalIndicators
    ti = TechnicalIndicators()
    df = ti.add_all(df)          # adds ~40 indicator columns
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config_loader import ConfigLoader
from src.logger import get_logger
from src.utils import timer

logger = get_logger(__name__)
cfg    = ConfigLoader()


class TechnicalIndicators:
    """
    Computes technical indicators on OHLCV price data.

    Expects columns prefixed with the asset name passed to the constructor,
    e.g. for Gold: Gold_Open, Gold_High, Gold_Low, Gold_Close, Gold_Volume.

    Parameters
    ----------
    prefix : str
        Column prefix for the asset being analyzed (default "Gold").

    Methods
    -------
    add_all(df)              → DataFrame with all indicators appended
    sma(df, period)          → pd.Series
    ema(df, period)          → pd.Series
    rsi(df, period)          → pd.Series
    macd(df)                 → (macd_line, signal_line, histogram)
    bollinger_bands(df)      → (upper, middle, lower)
    atr(df, period)          → pd.Series
    adx(df, period)          → pd.Series
    cci(df, period)          → pd.Series
    stochastic_rsi(df)       → pd.Series
    williams_r(df, period)   → pd.Series
    money_flow_index(df)     → pd.Series
    obv(df)                  → pd.Series
    vwap(df)                 → pd.Series
    ichimoku(df)             → dict of 5 Series
    roc(df, period)          → pd.Series
    momentum(df, period)     → pd.Series
    """

    def __init__(self, prefix: str = "Gold"):
        self.prefix = prefix
        self.col_open   = f"{prefix}_Open"
        self.col_high   = f"{prefix}_High"
        self.col_low    = f"{prefix}_Low"
        self.col_close  = f"{prefix}_Close"
        self.col_volume = f"{prefix}_Volume"

        ind_cfg = cfg.get_section("indicators")
        self.sma_periods       = ind_cfg.get("sma_periods", [5, 10, 20, 50, 100, 200])
        self.ema_periods       = ind_cfg.get("ema_periods", [5, 10, 20, 50, 100])
        self.rsi_period        = ind_cfg.get("rsi_period", 14)
        self.macd_fast          = ind_cfg.get("macd_fast", 12)
        self.macd_slow          = ind_cfg.get("macd_slow", 26)
        self.macd_signal        = ind_cfg.get("macd_signal", 9)
        self.bb_period          = ind_cfg.get("bb_period", 20)
        self.bb_std             = ind_cfg.get("bb_std", 2)
        self.atr_period         = ind_cfg.get("atr_period", 14)
        self.adx_period         = ind_cfg.get("adx_period", 14)
        self.cci_period         = ind_cfg.get("cci_period", 20)
        self.stoch_period       = ind_cfg.get("stoch_period", 14)
        self.williams_period    = ind_cfg.get("williams_period", 14)
        self.mfi_period         = ind_cfg.get("mfi_period", 14)
        self.roc_period         = ind_cfg.get("roc_period", 10)
        self.momentum_period    = ind_cfg.get("momentum_period", 10)

    # ──────────────────────────────────────────────────────────────
    # Column existence guard
    # ──────────────────────────────────────────────────────────────

    def _has_ohlc(self, df: pd.DataFrame) -> bool:
        required = [self.col_open, self.col_high, self.col_low, self.col_close]
        return all(c in df.columns for c in required)

    # ──────────────────────────────────────────────────────────────
    # Master orchestrator
    # ──────────────────────────────────────────────────────────────

    @timer
    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add every technical indicator to the DataFrame.
        Falls back gracefully if OHLC columns are not fully present
        (will use Close-only indicators in that case).

        Returns
        -------
        pd.DataFrame  Original df + indicator columns.
        """
        df = df.copy()

        if self.col_close not in df.columns:
            logger.error(f"Close column '{self.col_close}' not found — skipping indicators.")
            return df

        has_ohlc   = self._has_ohlc(df)
        has_volume = self.col_volume in df.columns

        n_before = df.shape[1]

        # ── Trend: SMA / EMA ─────────────────────────────────────
        for p in self.sma_periods:
            df[f"SMA_{p}"] = self.sma(df, p)
        for p in self.ema_periods:
            df[f"EMA_{p}"] = self.ema(df, p)

        # ── Momentum: RSI, ROC, Momentum ────────────────────────
        df["RSI"]      = self.rsi(df, self.rsi_period)
        df["ROC"]      = self.roc(df, self.roc_period)
        df["Momentum"] = self.momentum(df, self.momentum_period)

        # ── MACD ─────────────────────────────────────────────────
        macd_line, signal_line, hist = self.macd(df)
        df["MACD"]        = macd_line
        df["MACD_Signal"] = signal_line
        df["MACD_Hist"]   = hist

        # ── Bollinger Bands ──────────────────────────────────────
        upper, mid, lower = self.bollinger_bands(df)
        df["BB_Upper"]  = upper
        df["BB_Middle"] = mid
        df["BB_Lower"]  = lower
        df["BB_Width"]  = (upper - lower) / (mid + 1e-9)
        df["BB_PctB"]   = (df[self.col_close] - lower) / ((upper - lower) + 1e-9)

        if has_ohlc:
            # ── ATR / ADX / CCI ──────────────────────────────────
            df["ATR"] = self.atr(df, self.atr_period)
            df["ADX"] = self.adx(df, self.adx_period)
            df["CCI"] = self.cci(df, self.cci_period)

            # ── Stochastic RSI / Williams %R ─────────────────────
            df["StochRSI"]  = self.stochastic_rsi(df)
            df["WilliamsR"] = self.williams_r(df, self.williams_period)

            # ── Ichimoku Cloud ────────────────────────────────────
            ichimoku = self.ichimoku(df)
            for name, series in ichimoku.items():
                df[name] = series
        else:
            logger.warning("Full OHLC not available — skipping ATR/ADX/CCI/Stoch/Ichimoku")

        if has_volume:
            df["OBV"]  = self.obv(df)
            df["VWAP"] = self.vwap(df)
            if has_ohlc:
                df["MFI"] = self.money_flow_index(df)
        else:
            logger.warning(f"No volume column '{self.col_volume}' — skipping OBV/VWAP/MFI")

        n_after = df.shape[1]
        logger.info(f"Indicators added: {n_after - n_before} new columns (total={n_after})")
        return df

    # ════════════════════════════════════════════════════════════
    # TREND INDICATORS
    # ════════════════════════════════════════════════════════════

    def sma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Simple Moving Average."""
        return df[self.col_close].rolling(window=period, min_periods=1).mean()

    def ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return df[self.col_close].ewm(span=period, adjust=False).mean()

    def adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index — measures trend strength (0–100).
        """
        high, low, close = df[self.col_high], df[self.col_low], df[self.col_close]

        up_move   = high.diff()
        down_move = -low.diff()

        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = self._true_range(df)
        atr = tr.rolling(period, min_periods=1).mean()

        plus_di  = 100 * pd.Series(plus_dm, index=df.index).rolling(period, min_periods=1).mean() / (atr + 1e-9)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period, min_periods=1).mean() / (atr + 1e-9)

        dx  = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
        adx = dx.rolling(period, min_periods=1).mean()
        return adx

    def ichimoku(self, df: pd.DataFrame) -> dict:
        """
        Ichimoku Cloud — 4 components used as model features.

        Returns
        -------
        dict with keys: Ichimoku_Tenkan, Ichimoku_Kijun,
                        Ichimoku_SenkouA, Ichimoku_SenkouB

        Note on Chikou Span
        --------------------
        The traditional 5th Ichimoku component, the Chikou (lagging) span,
        is intentionally NOT included here. It's defined as
        close.shift(-kijun_period) — a BACKWARD shift that plots today's
        close 26 periods in the past purely for visual chart confirmation.
        Using it as a model FEATURE is problematic for two reasons:
        (1) it produces NaN for the most recent `kijun_period` rows (no
        future data exists to shift from), which — combined with a
        dropna()-based pipeline — silently deletes the newest rows right
        when they're needed most, e.g. for live forecasting; and (2) by
        construction it's just a future close price relocated, which is
        not a legitimate predictive input. Chart/visualization code can
        still compute it separately for display purposes if desired.
        """
        high, low, close = df[self.col_high], df[self.col_low], df[self.col_close]

        tenkan_period, kijun_period, senkou_b_period = 9, 26, 52

        tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
        kijun  = (high.rolling(kijun_period).max()  + low.rolling(kijun_period).min())  / 2
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
        senkou_b = ((high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2).shift(kijun_period)

        return {
            "Ichimoku_Tenkan":   tenkan,
            "Ichimoku_Kijun":    kijun,
            "Ichimoku_SenkouA":  senkou_a,
            "Ichimoku_SenkouB":  senkou_b,
        }

    # ════════════════════════════════════════════════════════════
    # MOMENTUM INDICATORS
    # ════════════════════════════════════════════════════════════

    def rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Relative Strength Index (0–100)."""
        delta = df[self.col_close].diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs  = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def stochastic_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Stochastic RSI — RSI normalized to its own rolling range (0–100)."""
        rsi_vals = self.rsi(df, period)
        min_rsi  = rsi_vals.rolling(period, min_periods=1).min()
        max_rsi  = rsi_vals.rolling(period, min_periods=1).max()
        stoch_rsi = 100 * (rsi_vals - min_rsi) / ((max_rsi - min_rsi) + 1e-9)
        return stoch_rsi

    def williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R — momentum oscillator (-100 to 0)."""
        highest_high = df[self.col_high].rolling(period, min_periods=1).max()
        lowest_low   = df[self.col_low].rolling(period, min_periods=1).min()
        wr = -100 * (highest_high - df[self.col_close]) / ((highest_high - lowest_low) + 1e-9)
        return wr

    def roc(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Rate of Change (%)."""
        return df[self.col_close].pct_change(periods=period) * 100

    def momentum(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Simple Momentum — price difference over N periods."""
        return df[self.col_close].diff(periods=period)

    # ════════════════════════════════════════════════════════════
    # VOLATILITY INDICATORS
    # ════════════════════════════════════════════════════════════

    def bollinger_bands(self, df: pd.DataFrame):
        """
        Bollinger Bands.

        Returns
        -------
        (upper, middle, lower) : Tuple[pd.Series, pd.Series, pd.Series]
        """
        middle = df[self.col_close].rolling(self.bb_period, min_periods=1).mean()
        std    = df[self.col_close].rolling(self.bb_period, min_periods=1).std()
        upper  = middle + self.bb_std * std
        lower  = middle - self.bb_std * std
        return upper, middle, lower

    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        """Helper: True Range used by ATR and ADX."""
        high, low, close = df[self.col_high], df[self.col_low], df[self.col_close]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr

    def atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range — volatility measure."""
        tr = self._true_range(df)
        return tr.rolling(window=period, min_periods=1).mean()

    # ════════════════════════════════════════════════════════════
    # COMPOSITE / MACD / CCI
    # ════════════════════════════════════════════════════════════

    def macd(self, df: pd.DataFrame):
        """
        MACD — Moving Average Convergence Divergence.

        Returns
        -------
        (macd_line, signal_line, histogram)
        """
        ema_fast = df[self.col_close].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df[self.col_close].ewm(span=self.macd_slow, adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram   = macd_line - signal_line
        return macd_line, signal_line, histogram

    def cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        typical_price = (df[self.col_high] + df[self.col_low] + df[self.col_close]) / 3
        sma_tp  = typical_price.rolling(period, min_periods=1).mean()
        mad     = typical_price.rolling(period, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * (mad + 1e-9))
        return cci

    # ════════════════════════════════════════════════════════════
    # VOLUME INDICATORS
    # ════════════════════════════════════════════════════════════

    def obv(self, df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(df[self.col_close].diff()).fillna(0)
        obv = (direction * df[self.col_volume]).fillna(0).cumsum()
        return obv

    def vwap(self, df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price (cumulative)."""
        typical_price = (df[self.col_high] + df[self.col_low] + df[self.col_close]) / 3 \
            if self._has_ohlc(df) else df[self.col_close]
        cum_vol_price = (typical_price * df[self.col_volume]).cumsum()
        cum_vol       = df[self.col_volume].cumsum()
        return cum_vol_price / (cum_vol + 1e-9)

    def money_flow_index(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Money Flow Index — volume-weighted RSI (0–100)."""
        typical_price = (df[self.col_high] + df[self.col_low] + df[self.col_close]) / 3
        raw_money_flow = typical_price * df[self.col_volume]

        price_diff = typical_price.diff()
        pos_flow = raw_money_flow.where(price_diff > 0, 0.0)
        neg_flow = raw_money_flow.where(price_diff < 0, 0.0)

        pos_sum = pos_flow.rolling(period, min_periods=1).sum()
        neg_sum = neg_flow.rolling(period, min_periods=1).sum()

        money_ratio = pos_sum / (neg_sum + 1e-9)
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

    # ════════════════════════════════════════════════════════════
    # Convenience: list all generated indicator column names
    # ════════════════════════════════════════════════════════════

    def get_indicator_columns(self, df: pd.DataFrame) -> List[str]:
        """Return list of all indicator columns currently in df."""
        known_prefixes = (
            "SMA_", "EMA_", "RSI", "ROC", "Momentum", "MACD",
            "BB_", "ATR", "ADX", "CCI", "StochRSI", "WilliamsR",
            "Ichimoku_", "OBV", "VWAP", "MFI",
        )
        return [c for c in df.columns if c.startswith(known_prefixes)]


# ════════════════════════════════════════════════════════════════
# Standalone test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.data_loader import DataLoader

    print("=" * 60)
    print("  Technical Indicators — Smoke Test")
    print("=" * 60)

    loader = DataLoader(start_date="2020-01-01", end_date=None)  # None = today
    df = loader.load_all(use_cache=True)
    print(f"\nInput shape: {df.shape}")

    ti = TechnicalIndicators(prefix="Gold")
    df_ind = ti.add_all(df)

    print(f"\n✔  Output shape: {df_ind.shape}")
    indicator_cols = ti.get_indicator_columns(df_ind)
    print(f"✔  Indicators added ({len(indicator_cols)}):")
    for i, col in enumerate(indicator_cols, 1):
        print(f"   {i:>2}. {col}")

    print(f"\nSample values (last row):")
    print(df_ind[indicator_cols].iloc[-1].round(3))

    print("\n✔ indicators.py working correctly")
