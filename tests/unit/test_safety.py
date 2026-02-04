"""
Unit tests for core/safety.py.

Tests verify default-to-safe behavior, position checks, gas gating,
cooldown enforcement, tx rate limiting, global pause, and sentinel file.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STANDARD_CONFIG = {
    "dry_run": False,
    "max_position_usd": 10000,
    "max_leverage_ratio": "3.0",
    "max_gas_price_gwei": 10,
    "cooldown_between_actions_seconds": 30,
    "max_transactions_per_24h": 50,
    "oracle_max_staleness_seconds": 60,
}


def _make_safety(config: dict | None = None):
    """Create a SafetyState with patched config and logger."""
    if config is None:
        config = STANDARD_CONFIG
    with (
        patch("core.safety.get_config") as mock_cfg,
        patch("core.safety.setup_module_logger") as mock_logger,
    ):
        mock_loader = MagicMock()
        mock_loader.get_positions_config.return_value = config
        mock_cfg.return_value = mock_loader
        mock_logger.return_value = MagicMock()

        from core.safety import SafetyState

        return SafetyState()


@pytest.fixture
def safety_with_config():
    return _make_safety(STANDARD_CONFIG)


@pytest.fixture
def safety_missing_config():
    return _make_safety({})


@pytest.fixture
def safety_dry_run():
    return _make_safety({**STANDARD_CONFIG, "dry_run": True})


# ---------------------------------------------------------------------------
# Default-to-safe tests
# ---------------------------------------------------------------------------


class TestDefaultToSafe:

    def test_missing_config_defaults_to_safe(self, safety_missing_config):
        assert safety_missing_config.is_dry_run is True

    def test_missing_config_blocks_all_positions(self, safety_missing_config):
        check = safety_missing_config.can_open_position(Decimal("1"), Decimal("1.0"))
        assert check.can_proceed is False


# ---------------------------------------------------------------------------
# can_open_position tests
# ---------------------------------------------------------------------------


class TestCanOpenPosition:

    def test_allows_valid_position(self, safety_with_config):
        check = safety_with_config.can_open_position(Decimal("5000"), Decimal("2.0"))
        assert check.can_proceed is True
        assert check.reason == "All checks passed"

    def test_blocks_excessive_amount(self, safety_with_config):
        check = safety_with_config.can_open_position(Decimal("15000"), Decimal("2.0"))
        assert check.can_proceed is False
        assert "exceeds max" in check.reason

    def test_blocks_excessive_leverage(self, safety_with_config):
        check = safety_with_config.can_open_position(Decimal("5000"), Decimal("4.0"))
        assert check.can_proceed is False
        assert "Leverage" in check.reason

    def test_blocks_during_cooldown(self, safety_with_config):
        safety_with_config.record_action()
        check = safety_with_config.can_open_position(Decimal("5000"), Decimal("2.0"))
        assert check.can_proceed is False
        assert "Cooldown" in check.reason

    def test_blocks_after_24h_tx_limit(self, safety_with_config):
        # Fill up the 24h window with 50 actions
        for _ in range(50):
            safety_with_config._action_timestamps.append(__import__("time").time())
        # Reset cooldown so that doesn't block first
        safety_with_config._last_action_time = 0.0

        check = safety_with_config.can_open_position(Decimal("5000"), Decimal("2.0"))
        assert check.can_proceed is False
        assert "24h tx limit" in check.reason


# ---------------------------------------------------------------------------
# can_submit_tx tests
# ---------------------------------------------------------------------------


class TestCanSubmitTx:

    def test_allows_acceptable_gas(self, safety_with_config):
        check = safety_with_config.can_submit_tx(gas_price_gwei=5)
        assert check.can_proceed is True

    def test_blocks_high_gas(self, safety_with_config):
        check = safety_with_config.can_submit_tx(gas_price_gwei=15)
        assert check.can_proceed is False
        assert "Gas price" in check.reason


# ---------------------------------------------------------------------------
# Pause and resume tests
# ---------------------------------------------------------------------------


class TestPauseAndResume:

    def test_global_pause_blocks_everything(self, safety_with_config):
        safety_with_config.trigger_global_pause("test pause")

        pos_check = safety_with_config.can_open_position(Decimal("5000"), Decimal("2.0"))
        tx_check = safety_with_config.can_submit_tx(gas_price_gwei=5)

        assert pos_check.can_proceed is False
        assert "Global pause" in pos_check.reason
        assert tx_check.can_proceed is False

    def test_resume_clears_pause(self, safety_with_config):
        safety_with_config.trigger_global_pause("test pause")
        safety_with_config.resume()

        check = safety_with_config.can_open_position(Decimal("5000"), Decimal("2.0"))
        assert check.can_proceed is True


# ---------------------------------------------------------------------------
# Sentinel file test
# ---------------------------------------------------------------------------


class TestSentinelFile:

    def test_pause_sentinel_file_triggers_pause(self, safety_with_config):
        with patch("core.safety._SENTINEL_FILE") as mock_path:
            mock_path.exists.return_value = True
            assert safety_with_config.is_paused is True


# ---------------------------------------------------------------------------
# Dry run test
# ---------------------------------------------------------------------------


class TestDryRun:

    def test_dry_run_blocks_position_opening(self, safety_dry_run):
        check = safety_dry_run.can_open_position(Decimal("5000"), Decimal("2.0"))
        assert check.can_proceed is False
        assert "Dry run" in check.reason
