"""
Unit tests for config/loader.py and config/validate.py.

Tests cover:
- JSON config file loading
- Environment variable overrides with type coercion
- Missing file graceful fallback (empty dict)
- Singleton pattern for ConfigLoader
- Config validation (validate_all_configs)
- Cache management
- ABI loading
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from config.loader import ConfigLoader, get_config, get_env_var
from config.validate import (
    ConfigValidationError,
    validate_aave_config,
    validate_aggregator_config,
    validate_all_configs,
    validate_chain_config,
    validate_positions_config,
    validate_signals_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the ConfigLoader singleton between tests."""
    ConfigLoader._instance = None
    yield
    ConfigLoader._instance = None


# ===========================================================================
# ConfigLoader tests
# ===========================================================================


class TestConfigLoaderSingleton:
    def test_get_instance_returns_same_object(self):
        """Singleton pattern should return the same instance."""
        a = ConfigLoader.get_instance()
        b = ConfigLoader.get_instance()
        assert a is b

    def test_get_config_returns_singleton(self):
        """Module-level get_config() should return singleton."""
        cfg = get_config()
        assert cfg is ConfigLoader.get_instance()

    def test_fresh_instance_after_reset(self):
        """After resetting _instance, a new one is created."""
        first = ConfigLoader.get_instance()
        ConfigLoader._instance = None
        second = ConfigLoader.get_instance()
        assert first is not second


class TestConfigLoading:
    def test_get_chain_config_returns_dict(self):
        """Chain config should load BSC chain data."""
        cfg = get_config()
        chain = cfg.get_chain_config(56)
        assert isinstance(chain, dict)
        assert chain.get("chain_id") == 56

    def test_get_positions_config(self):
        """Positions config should have required keys."""
        cfg = get_config()
        pos = cfg.get_positions_config()
        assert isinstance(pos, dict)
        assert "dry_run" in pos

    def test_get_aave_config(self):
        """Aave config should load."""
        cfg = get_config()
        aave = cfg.get_aave_config()
        assert isinstance(aave, dict)

    def test_get_aggregator_config(self):
        """Aggregator config should load."""
        cfg = get_config()
        agg = cfg.get_aggregator_config()
        assert isinstance(agg, dict)

    def test_get_signals_config(self):
        """Signals config should load."""
        cfg = get_config()
        sig = cfg.get_signals_config()
        assert isinstance(sig, dict)
        assert "enabled" in sig

    def test_get_timing_config(self):
        """Timing config should load."""
        cfg = get_config()
        timing = cfg.get_timing_config()
        assert isinstance(timing, dict)

    def test_get_app_config(self):
        """App config should load."""
        cfg = get_config()
        app = cfg.get_app_config()
        assert isinstance(app, dict)

    def test_get_rate_limit_config(self):
        """Rate limit config should load."""
        cfg = get_config()
        rl = cfg.get_rate_limit_config()
        assert isinstance(rl, dict)

    def test_missing_config_returns_empty_dict(self):
        """Loading a nonexistent config file should return {}."""
        cfg = get_config()
        result = cfg.get_config_file("nonexistent_config_xyz")
        assert result == {}


class TestABILoading:
    def test_load_erc20_abi(self):
        """ERC-20 ABI should load as a list."""
        cfg = get_config()
        abi = cfg.get_abi("erc20")
        assert isinstance(abi, list)
        assert len(abi) > 0

    def test_load_aave_pool_abi(self):
        """Aave V3 pool ABI should load."""
        cfg = get_config()
        abi = cfg.get_abi("aave_v3_pool")
        assert isinstance(abi, list)

    def test_missing_abi_returns_empty(self):
        """Missing ABI file should return empty list."""
        cfg = get_config()
        abi = cfg.get_abi("nonexistent_abi_xyz")
        assert abi == [] or abi == {}


class TestCacheManagement:
    def test_cache_produces_same_result(self):
        """Cached calls should return the same object."""
        cfg = get_config()
        a = cfg.get_positions_config()
        b = cfg.get_positions_config()
        assert a is b

    def test_clear_cache(self):
        """clear_cache should not raise and should allow fresh loads."""
        cfg = get_config()
        _ = cfg.get_positions_config()
        cfg.clear_cache()
        # After clearing, calling again should work without error
        result = cfg.get_positions_config()
        assert isinstance(result, dict)


# ===========================================================================
# get_env_var tests
# ===========================================================================


class TestGetEnvVar:
    def test_bool_true_values(self):
        """Boolean 'true', '1', 'yes' should parse as True."""
        for val in ("true", "True", "TRUE", "1", "yes"):
            with patch.dict(os.environ, {"TEST_BOOL": val}):
                assert get_env_var("TEST_BOOL", False, bool) is True

    def test_bool_false_values(self):
        """Boolean 'false', '0', 'no' should parse as False."""
        for val in ("false", "False", "0", "no"):
            with patch.dict(os.environ, {"TEST_BOOL": val}):
                assert get_env_var("TEST_BOOL", True, bool) is False

    def test_int_conversion(self):
        """Integer env vars should be converted."""
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            assert get_env_var("TEST_INT", 0, int) == 42

    def test_float_conversion(self):
        """Float env vars should be converted."""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            assert get_env_var("TEST_FLOAT", 0.0, float) == pytest.approx(3.14)

    def test_missing_var_returns_default(self):
        """Missing env var should return the default."""
        # Ensure the var doesn't exist
        os.environ.pop("DEFINITELY_NOT_SET_XYZ", None)
        assert get_env_var("DEFINITELY_NOT_SET_XYZ", "fallback", str) == "fallback"

    def test_invalid_int_returns_default(self):
        """Non-numeric value for int should return default."""
        with patch.dict(os.environ, {"TEST_BAD_INT": "abc"}):
            assert get_env_var("TEST_BAD_INT", 99, int) == 99


# ===========================================================================
# Config validation tests
# ===========================================================================


class TestValidateChainConfig:
    def test_valid_chain_config(self):
        """Valid chain config should return no errors."""
        cfg = {
            "chain_id": 56,
            "block_time_seconds": 3,
            "rpc": {"http_url": "https://bsc-dataseed1.binance.org/"},
            "contracts": {
                "aave_v3_pool": "0x123",
                "aave_v3_data_provider": "0x456",
            },
            "chainlink_feeds": {},
            "tokens": {},
        }
        assert validate_chain_config(cfg) == []

    def test_missing_chain_id(self):
        """Missing chain_id should be flagged."""
        errors = validate_chain_config({})
        assert "chain_id" in errors


class TestValidateAaveConfig:
    def test_valid_aave_config(self):
        cfg = {"flash_loan_premium_bps": 5, "supported_assets": []}
        assert validate_aave_config(cfg) == []

    def test_missing_fields(self):
        errors = validate_aave_config({})
        assert len(errors) > 0


class TestValidatePositionsConfig:
    def test_valid_positions_config(self):
        cfg = {
            "dry_run": True,
            "max_flash_loan_usd": 5000,
            "max_position_usd": 10000,
            "max_leverage_ratio": "3.0",
            "min_health_factor": "1.5",
            "deleverage_threshold": "1.4",
            "close_threshold": "1.25",
            "max_gas_price_gwei": 10,
        }
        assert validate_positions_config(cfg) == []

    def test_missing_dry_run(self):
        errors = validate_positions_config({})
        assert "dry_run" in errors


class TestValidateAggregatorConfig:
    def test_valid_aggregator_config(self):
        cfg = {"providers": [{"name": "openocean", "enabled": True}]}
        assert validate_aggregator_config(cfg) == []

    def test_empty_providers(self):
        errors = validate_aggregator_config({"providers": []})
        assert len(errors) > 0

    def test_no_enabled_providers(self):
        errors = validate_aggregator_config({"providers": [{"name": "x", "enabled": False}]})
        assert len(errors) > 0


class TestValidateSignalsConfig:
    def test_valid_signals_config(self):
        cfg = {"enabled": True, "mode": "blended", "data_source": {}, "indicators": {}}
        assert validate_signals_config(cfg) == []

    def test_missing_enabled(self):
        errors = validate_signals_config({})
        assert "enabled" in errors


class TestValidateAllConfigs:
    def test_all_configs_valid(self):
        """validate_all_configs should pass with real config files."""
        # This loads the actual project configs which should be valid
        validate_all_configs()  # Should not raise

    def test_raises_on_invalid(self):
        """validate_all_configs should raise ConfigValidationError on bad configs."""
        mock_loader = MagicMock()
        mock_loader.get_chain_config.return_value = {}  # Missing keys
        mock_loader.get_aave_config.return_value = {}
        mock_loader.get_positions_config.return_value = {}
        mock_loader.get_aggregator_config.return_value = {}
        mock_loader.get_signals_config.return_value = {}

        with (
            patch("config.validate.get_config", return_value=mock_loader),
            pytest.raises(ConfigValidationError, match="validation failed"),
        ):
            validate_all_configs()

    def test_error_message_includes_details(self):
        """Error message should list which configs and keys failed."""
        mock_loader = MagicMock()
        # Non-empty dict missing required keys (empty {} is falsy, triggers different message)
        mock_loader.get_chain_config.return_value = {"block_time_seconds": 3}
        mock_loader.get_aave_config.return_value = {
            "flash_loan_premium_bps": 5,
            "supported_assets": [],
        }
        mock_loader.get_positions_config.return_value = {
            "dry_run": True,
            "max_flash_loan_usd": 5000,
            "max_position_usd": 10000,
            "max_leverage_ratio": "3.0",
            "min_health_factor": "1.5",
            "deleverage_threshold": "1.4",
            "close_threshold": "1.25",
            "max_gas_price_gwei": 10,
        }
        mock_loader.get_aggregator_config.return_value = {
            "providers": [{"name": "x", "enabled": True}]
        }
        mock_loader.get_signals_config.return_value = {
            "enabled": True,
            "mode": "blended",
            "data_source": {},
            "indicators": {},
        }

        with (
            patch("config.validate.get_config", return_value=mock_loader),
            pytest.raises(ConfigValidationError, match="chain_id"),
        ):
            validate_all_configs()
