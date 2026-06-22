"""
utils.py — Shared Utility Functions
=====================================
Helper functions used across multiple modules:
  - Timer / profiling decorator
  - Safe directory creation
  - DataFrame validation helpers
  - Metric computation
  - Progress bar wrapper
  - Model serialization helpers
  - Date helpers

Usage:
    from src.utils import timer, ensure_dir, compute_metrics
"""

import os
import time
import json
import pickle
import functools
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# 1.  DECORATORS
# ════════════════════════════════════════════════════════════════

def timer(func: Callable) -> Callable:
    """
    Decorator that logs the execution time of any function.

    Usage
    -----
    @timer
    def train_model(): ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        logger.info(f"▶ Starting: {func.__name__}")
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"✔ Finished: {func.__name__} in {elapsed:.2f}s")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 2.0, exceptions=(Exception,)):
    """
    Decorator that retries a function on failure.

    Parameters
    ----------
    max_attempts : int
    delay        : float  seconds between retries
    exceptions   : tuple  exception types to catch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    logger.warning(f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {exc}")
                    if attempt < max_attempts:
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                        raise
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════
# 2.  FILE / PATH HELPERS
# ════════════════════════════════════════════════════════════════

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_project_root() -> Path:
    """Return the absolute path to the project root."""
    return Path(__file__).resolve().parent.parent


def timestamped_filename(base: str, ext: str = "") -> str:
    """
    Generate a filename with a timestamp suffix.

    Examples
    --------
    >>> timestamped_filename("model", ".pkl")
    'model_20240115_143022.pkl'
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}{ext}"


# ════════════════════════════════════════════════════════════════
# 3.  DATAFRAME HELPERS
# ════════════════════════════════════════════════════════════════

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 100,
) -> bool:
    """
    Validate a DataFrame for basic sanity checks.

    Parameters
    ----------
    df               : pd.DataFrame
    required_columns : list of column names that must be present
    min_rows         : minimum acceptable row count

    Returns
    -------
    bool  True if valid, raises ValueError otherwise.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty.")

    if len(df) < min_rows:
        raise ValueError(f"DataFrame has only {len(df)} rows (minimum: {min_rows}).")

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    return True


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a summary dict of a DataFrame (shape, dtypes, nulls, etc.)."""
    return {
        "shape":          df.shape,
        "columns":        list(df.columns),
        "dtypes":         df.dtypes.astype(str).to_dict(),
        "null_counts":    df.isnull().sum().to_dict(),
        "null_pct":       (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates":     int(df.duplicated().sum()),
        "memory_mb":      round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "date_range":     (
            f"{df.index.min()} → {df.index.max()}"
            if isinstance(df.index, pd.DatetimeIndex)
            else "N/A"
        ),
    }


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str = "Date",
    how: str = "outer",
) -> pd.DataFrame:
    """
    Merge two DataFrames on a date column with robust error handling.

    Parameters
    ----------
    left, right : DataFrames to merge
    on          : Column or index to merge on
    how         : Merge type (outer / inner / left / right)

    Returns
    -------
    pd.DataFrame  Merged result.
    """
    try:
        # Ensure date index
        for df in (left, right):
            if on in df.columns:
                df.index = pd.to_datetime(df[on])
                df.drop(columns=[on], inplace=True, errors="ignore")
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

        merged = left.join(right, how=how)
        logger.debug(f"Merged shape: {merged.shape}")
        return merged
    except Exception as exc:
        logger.error(f"Merge failed: {exc}")
        raise


# ════════════════════════════════════════════════════════════════
# 4.  METRICS
# ════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Compute all regression evaluation metrics.

    Parameters
    ----------
    y_true      : Ground-truth values
    y_pred      : Model predictions
    model_name  : Label for logging

    Returns
    -------
    dict with keys: MAE, RMSE, MAPE, R2, DirectionalAccuracy
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae   = float(mean_absolute_error(y_true, y_pred))
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2    = float(r2_score(y_true, y_pred))

    # MAPE — guard against zero division
    mask  = y_true != 0
    mape  = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    # Directional accuracy
    if len(y_true) > 1:
        actual_dir    = np.sign(np.diff(y_true))
        predicted_dir = np.sign(np.diff(y_pred))
        dir_acc = float(np.mean(actual_dir == predicted_dir) * 100)
    else:
        dir_acc = 0.0

    metrics = {
        "MAE":               round(mae, 4),
        "RMSE":              round(rmse, 4),
        "MAPE":              round(mape, 4),
        "R2":                round(r2, 4),
        "DirectionalAccuracy": round(dir_acc, 2),
    }

    logger.info(
        f"[{model_name}] MAE={mae:.4f} | RMSE={rmse:.4f} | "
        f"MAPE={mape:.2f}% | R²={r2:.4f} | DirAcc={dir_acc:.2f}%"
    )
    return metrics


def metrics_to_dataframe(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Convert a {model_name: metrics_dict} mapping into a sorted DataFrame.

    Parameters
    ----------
    metrics_dict : e.g. {"XGBoost": {"MAE": 1.2, "RMSE": 1.8, ...}, ...}

    Returns
    -------
    pd.DataFrame  Sorted by RMSE ascending.
    """
    df = pd.DataFrame(metrics_dict).T
    df.index.name = "Model"
    df = df.sort_values("RMSE").reset_index()
    df["Rank"] = range(1, len(df) + 1)
    cols = ["Rank", "Model"] + [c for c in df.columns if c not in ("Rank", "Model")]
    return df[cols]


# ════════════════════════════════════════════════════════════════
# 5.  MODEL SERIALIZATION
# ════════════════════════════════════════════════════════════════

def save_model(model: Any, path: Union[str, Path]) -> None:
    """Serialize any sklearn/xgb/lgb model to disk using pickle."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Model saved → {path}")


def load_model(path: Union[str, Path]) -> Any:
    """Deserialize a model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded ← {path}")
    return model


def save_json(data: Dict, path: Union[str, Path]) -> None:
    """Save a dictionary as JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"JSON saved → {path}")


def load_json(path: Union[str, Path]) -> Dict:
    """Load a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════
# 6.  DATE / TIME HELPERS
# ════════════════════════════════════════════════════════════════

def get_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Return business days between two date strings."""
    return pd.bdate_range(start=start, end=end)


def is_holiday(date: pd.Timestamp, country: str = "US") -> bool:
    """Check whether a date is a public holiday."""
    try:
        import holidays
        country_holidays = holidays.country_holidays(country)
        return date.date() in country_holidays
    except ImportError:
        return False


def add_holiday_flag(df: pd.DataFrame, country: str = "US") -> pd.DataFrame:
    """Add a binary 'is_holiday' column to a date-indexed DataFrame."""
    try:
        import holidays
        country_holidays = holidays.country_holidays(country)
        df["is_holiday"] = df.index.map(lambda d: int(d.date() in country_holidays))
    except ImportError:
        logger.warning("'holidays' package not installed — skipping holiday flag.")
        df["is_holiday"] = 0
    return df


# ════════════════════════════════════════════════════════════════
# 7.  PROGRESS BAR WRAPPER
# ════════════════════════════════════════════════════════════════

def progress_bar(iterable, description: str = "", total: Optional[int] = None):
    """
    Wrap any iterable with a tqdm progress bar.

    Usage
    -----
    for item in progress_bar(items, "Training"):
        ...
    """
    return tqdm(iterable, desc=description, total=total, ncols=100, colour="green")


# ════════════════════════════════════════════════════════════════
# 8.  FORMATTING HELPERS
# ════════════════════════════════════════════════════════════════

def format_currency(value: float, symbol: str = "$") -> str:
    """Format a float as currency string, e.g. $1,234.56"""
    return f"{symbol}{value:,.2f}"


def format_pct(value: float) -> str:
    """Format a float as percentage string, e.g. 12.34%"""
    return f"{value:.2f}%"


def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section header to stdout."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


if __name__ == "__main__":
    # Quick smoke-test
    print_section("Utils Module Test")

    y_true = np.array([1900, 1920, 1910, 1930, 1950])
    y_pred = np.array([1905, 1915, 1915, 1925, 1945])
    m = compute_metrics(y_true, y_pred, model_name="TestModel")
    print(f"\nMetrics: {m}")

    df = pd.DataFrame({"A": [1, 2, None], "B": [4, None, 6]})
    print(f"\nSummary: {summarize_dataframe(df)}")
    print("\n✓ utils.py is working correctly")
