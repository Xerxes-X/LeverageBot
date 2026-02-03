"""
Configuration loader for BSC Leverage Bot.

Adapted from ArbitrageTestBot/config/loader.py (~893 lines).
Provides centralized configuration management with .env overrides.

Usage:
    from config.loader import get_config, get_channel

    config = get_config()
    chain_config = config.get_chain_config(56)
    channel = get_channel("raw_v2_events")
"""

import json
import os
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

# Resolve config directory relative to this file
_CONFIG_DIR = Path(__file__).parent
_PROJECT_ROOT = _CONFIG_DIR.parent


def _load_json(filepath: Path) -> Dict[str, Any]:
    """Load a JSON config file. Returns empty dict if file doesn't exist."""
    try:
        with open(filepath, "r") as f:
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
        if var_type == bool:
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
    def get_chain_config(self, chain_id: int = 56) -> Dict[str, Any]:
        """Load chain-specific config (BSC = 56)."""
        return _load_json(self._config_dir / "chains" / f"{chain_id}.json")

    @lru_cache(maxsize=16)
    def get_protocol_config(self, protocol: str) -> Dict[str, Any]:
        """Load protocol config (v2, v3, aave_v3, curve, dodo, wombat, thena)."""
        return _load_json(self._config_dir / "protocols" / f"{protocol}.json")

    @lru_cache(maxsize=1)
    def get_redis_channels(self) -> Dict[str, Any]:
        """Load Redis channel definitions."""
        return _load_json(self._config_dir / "redis_channels.json")

    @lru_cache(maxsize=1)
    def get_app_config(self) -> Dict[str, Any]:
        """Load general application settings."""
        return _load_json(self._config_dir / "app.json")

    @lru_cache(maxsize=1)
    def get_timing_config(self) -> Dict[str, Any]:
        """Load timing intervals and timeouts."""
        return _load_json(self._config_dir / "timing.json")

    @lru_cache(maxsize=1)
    def get_websocket_config(self) -> Dict[str, Any]:
        """Load WebSocket connection settings."""
        return _load_json(self._config_dir / "websocket.json")

    @lru_cache(maxsize=1)
    def get_gas_config(self) -> Dict[str, Any]:
        """Load gas price tracking settings."""
        return _load_json(self._config_dir / "gas.json")

    @lru_cache(maxsize=1)
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Load RPC rate limiting configuration."""
        return _load_json(self._config_dir / "rate_limits.json")

    @lru_cache(maxsize=1)
    def get_cache_config(self) -> Dict[str, Any]:
        """Load in-memory cache settings."""
        return _load_json(self._config_dir / "cache.json")

    @lru_cache(maxsize=1)
    def get_mev_config(self) -> Dict[str, Any]:
        """Load BSC MEV/Builder API settings."""
        return _load_json(self._config_dir / "mev.json")

    @lru_cache(maxsize=1)
    def get_positions_config(self) -> Dict[str, Any]:
        """Load position management config (health factor thresholds, max leverage)."""
        return _load_json(self._config_dir / "positions.json")

    @lru_cache(maxsize=1)
    def get_optimizer_config(self) -> Dict[str, Any]:
        """Load split routing optimizer parameters."""
        return _load_json(self._config_dir / "optimizer.json")

    @lru_cache(maxsize=1)
    def get_aave_config(self) -> Dict[str, Any]:
        """Load Aave V3 BSC lending config."""
        return self.get_protocol_config("aave_v3")

    @lru_cache(maxsize=1)
    def get_multiprocessing_config(self) -> Dict[str, Any]:
        """Load multiprocessing/CPU affinity settings."""
        return _load_json(self._config_dir / "multiprocessing.json")

    @lru_cache(maxsize=1)
    def get_verification_config(self) -> Dict[str, Any]:
        """Load on-chain verification settings."""
        return _load_json(self._config_dir / "verification.json")

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
    # Redis channel/signal/key helpers
    # ------------------------------------------------------------------

    def get_channel_name(self, channel_key: str) -> str:
        """Get a Redis channel name by key."""
        channels = self.get_redis_channels()
        return channels.get("channels", {}).get(channel_key, f"bsc:{channel_key}")

    def get_signal_value(self, signal_key: str) -> str:
        """Get a Redis signal value by key."""
        channels = self.get_redis_channels()
        return channels.get("signals", {}).get(signal_key, signal_key.upper())

    def get_cache_key(self, key_name: str) -> str:
        """Get a Redis cache key by name."""
        channels = self.get_redis_channels()
        return channels.get("cache_keys", {}).get(key_name, f"cache:{key_name}")

    # ------------------------------------------------------------------
    # Arbitrary config file loader
    # ------------------------------------------------------------------

    @lru_cache(maxsize=16)
    def get_config_file(self, config_name: str) -> Dict[str, Any]:
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
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_config() -> ConfigLoader:
    """Get the singleton ConfigLoader instance."""
    return ConfigLoader.get_instance()


def get_channel(channel_key: str) -> str:
    """Get a Redis channel name by key (convenience function)."""
    return get_config().get_channel_name(channel_key)


def get_signal(signal_key: str) -> str:
    """Get a Redis signal value by key (convenience function)."""
    return get_config().get_signal_value(signal_key)


def get_key(key_name: str) -> str:
    """Get a Redis cache key by name (convenience function)."""
    return get_config().get_cache_key(key_name)
