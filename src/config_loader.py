"""
config_loader.py — YAML Configuration Loader
=============================================
Singleton class that loads config/config.yaml once and
provides typed access throughout the project.
Automatically overlays values from .env environment variables.

Usage:
    from src.config_loader import ConfigLoader
    cfg = ConfigLoader()
    start = cfg.get("data.start_date")
    models_dir = cfg.paths.models_saved
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


# ── Locate project root (parent of src/) ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


class _Namespace:
    """Dot-access wrapper around a dictionary."""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _Namespace(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"_Namespace({self.__dict__})"

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            result[k] = v.to_dict() if isinstance(v, _Namespace) else v
        return result


class ConfigLoader:
    """
    Singleton YAML configuration loader.

    Features
    --------
    - Loads config/config.yaml on first instantiation, reuses thereafter.
    - Merges .env environment variables for sensitive keys (API keys, passwords).
    - Provides both dict-style and dot-style access.
    - Supports nested key lookup via dot notation (e.g. "data.start_date").

    Examples
    --------
    >>> cfg = ConfigLoader()
    >>> cfg.get("data.start_date")
    '2010-01-01'
    >>> cfg.paths.models_saved
    'models/saved'
    """

    _instance: Optional["ConfigLoader"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    # ── Internal ───────────────────────────────────────────────────
    def _load(self) -> None:
        """Load YAML config and merge environment variables."""
        # Load .env if it exists
        if ENV_PATH.exists():
            load_dotenv(ENV_PATH)

        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {CONFIG_PATH}\n"
                "Make sure you run from the project root."
            )

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

        # Overlay environment variables for sensitive fields
        self._overlay_env()

        # Build dot-access namespaces for common sections
        self._build_namespaces()

    def _overlay_env(self) -> None:
        """Replace config API keys with environment variable values."""
        env_map = {
            "FRED_API_KEY":   ("api", "fred_api_key"),
            "NEWS_API_KEY":   ("api", "news_api_key"),
            "EMAIL_SENDER":   ("api", "email_sender"),
            "EMAIL_PASSWORD": ("api", "email_password"),
        }
        for env_var, (section, key) in env_map.items():
            value = os.getenv(env_var)
            if value:
                self._config.setdefault(section, {})[key] = value

    def _build_namespaces(self) -> None:
        """Attach dot-access namespaces to self."""
        for section, data in self._config.items():
            if isinstance(data, dict):
                setattr(self, section, _Namespace(data))
            else:
                setattr(self, section, data)

    # ── Public API ─────────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value using dot-notation key path.

        Parameters
        ----------
        key : str
            Dot-separated path, e.g. "data.start_date" or "ml_models.xgboost.n_estimators".
        default : Any
            Value returned when the key is not found.

        Returns
        -------
        Any
            The config value or *default*.
        """
        parts = key.split(".")
        current: Any = self._config
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Return an entire top-level config section as a dict."""
        return self._config.get(section, {})

    def all(self) -> Dict[str, Any]:
        """Return the entire config as a plain dictionary."""
        return self._config

    def reload(self) -> None:
        """Force reload from disk (useful during development)."""
        ConfigLoader._instance = None
        ConfigLoader._config = {}
        self._load()

    # ── Helpers ────────────────────────────────────────────────────
    def resolve_path(self, path_key: str) -> Path:
        """
        Resolve a relative path from the 'paths' section to an absolute Path.

        Parameters
        ----------
        path_key : str
            Key inside paths section, e.g. "data_raw".

        Returns
        -------
        Path
            Absolute path under PROJECT_ROOT.
        """
        rel = self.get(f"paths.{path_key}", "")
        full = PROJECT_ROOT / rel
        full.mkdir(parents=True, exist_ok=True)
        return full

    def __repr__(self) -> str:
        sections = list(self._config.keys())
        return f"ConfigLoader(sections={sections}, source='{CONFIG_PATH}')"


# ── Convenience singleton instance ────────────────────────────────
config = ConfigLoader()


if __name__ == "__main__":
    cfg = ConfigLoader()
    print(cfg)
    print(f"\nProject Name  : {cfg.get('project.name')}")
    print(f"Start Date    : {cfg.get('data.start_date')}")
    print(f"Gold Ticker   : {cfg.get('data.tickers.gold')}")
    print(f"LSTM Units    : {cfg.get('dl_models.lstm.units')}")
    print(f"Models Dir    : {cfg.resolve_path('models_saved')}")
    print(f"\nFull paths section: {cfg.get_section('paths')}")
