"""
Configuration loader for BSC Leverage Bot.

Provides centralized configuration management with .env overrides.

Usage:
    from config.loader import get_config

    config = get_config()
    chain_config = config.get_chain_config(56)
    aave_config = config.get_aave_config()
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# Resolve config directory relative to this file
_CONFIG_DIR = Path(__file__).parent
_PROJECT_ROOT = _CONFIG_DIR.parent


def _load_json(filepath: Path) -> dict[str, Any]:
    """Load a JSON config file. Returns empty dict if file doesn't exist."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[CONFIG_WARN] Config file not found: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        print(f"[CONFIG_ERROR] Invalid JSON in {filepath}: {e}")
        return {}


def get_env_var(var_name: str, default_value: Any, var_type: type) -> Any:
    """Get environment variable with type conversion and fallback."""
    value = os.getenv(var_name, None)
    if value is None:
        return default_value
    try:
        if var_type is bool:
            return value.lower() in ("true", "1", "yes")
        return var_type(value)
    except (ValueError, TypeError):
        return default_value


class ConfigLoader:
    """
    Central configuration manager for the BSC Leverage Bot.

    Loads configuration from JSON files in the config/ directory with .env overrides.
    All accessor methods are cached via @lru_cache for performance.
    """

    _instance: Optional["ConfigLoader"] = None

    def __init__(self):
        self._config_dir = _CONFIG_DIR
        self._project_root = _PROJECT_ROOT

    @classmethod
    def get_instance(cls) -> "ConfigLoader":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Core config file loaders (cached)
    # ------------------------------------------------------------------

    @lru_cache(maxsize=8)
    def get_chain_config(self, chain_id: int = 56) -> dict[str, Any]:
        """Load chain-specific config (BSC = 56)."""
        return _load_json(self._config_dir / "chains" / f"{chain_id}.json")

    @lru_cache(maxsize=1)
    def get_app_config(self) -> dict[str, Any]:
        """Load general application settings."""
        return _load_json(self._config_dir / "app.json")

    @lru_cache(maxsize=1)
    def get_timing_config(self) -> dict[str, Any]:
        """Load timing intervals and timeouts."""
        return _load_json(self._config_dir / "timing.json")

    @lru_cache(maxsize=1)
    def get_rate_limit_config(self) -> dict[str, Any]:
        """Load RPC and API rate limiting configuration."""
        return _load_json(self._config_dir / "rate_limits.json")

    @lru_cache(maxsize=1)
    def get_positions_config(self) -> dict[str, Any]:
        """Load position management config (health factor thresholds, max leverage)."""
        return _load_json(self._config_dir / "positions.json")

    @lru_cache(maxsize=1)
    def get_aave_config(self) -> dict[str, Any]:
        """Load Aave V3 BSC lending config (risk params, assets, flash loan premium)."""
        return _load_json(self._config_dir / "aave.json")

    @lru_cache(maxsize=1)
    def get_aggregator_config(self) -> dict[str, Any]:
        """Load DEX aggregator config (1inch, OpenOcean, ParaSwap endpoints)."""
        return _load_json(self._config_dir / "aggregator.json")

    @lru_cache(maxsize=1)
    def get_signals_config(self) -> dict[str, Any]:
        """Load signal engine config (indicators, thresholds, data sources)."""
        return _load_json(self._config_dir / "signals.json")

    # ------------------------------------------------------------------
    # ABI loader
    # ------------------------------------------------------------------

    @lru_cache(maxsize=32)
    def get_abi(self, abi_name: str) -> list:
        """Load ABI from config/abis/<abi_name>.json."""
        data = _load_json(self._config_dir / "abis" / f"{abi_name}.json")
        # ABI files are either raw arrays or {"abi": [...]}
        if isinstance(data, list):
            return data
        return data.get("abi", [])

    # ------------------------------------------------------------------
    # Arbitrary config file loader
    # ------------------------------------------------------------------

    @lru_cache(maxsize=16)
    def get_config_file(self, config_name: str) -> dict[str, Any]:
        """Load an arbitrary JSON config file from config/ directory."""
        return _load_json(self._config_dir / f"{config_name}.json")

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear all cached configurations (useful for testing)."""
        for method_name in dir(self):
            method = getattr(self, method_name)
            if hasattr(method, "cache_clear"):
                method.cache_clear()


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def get_config() -> ConfigLoader:
    """Get the singleton ConfigLoader instance."""
    return ConfigLoader.get_instance()
