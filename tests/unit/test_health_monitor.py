"""
Unit tests for core/health_monitor.py.

Tests cover tier determination, poll intervals, oracle freshness validation,
HF prediction via compound interest, and the async run loop.
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.constants import RAY
from shared.types import (
    BorrowRateInfo,
    HealthStatus,
    HFTier,
    PositionDirection,
    PositionState,
    ReserveData,
    UserAccountData,
)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SAMPLE_USER = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

SAMPLE_ACCOUNT = UserAccountData(
    total_collateral_usd=Decimal("10000"),
    total_debt_usd=Decimal("5000"),
    available_borrow_usd=Decimal("2000"),
    current_liquidation_threshold=Decimal("0.8"),
    ltv=Decimal("0.75"),
    health_factor=Decimal("1.8"),
)

SAMPLE_RESERVE = ReserveData(
    variable_borrow_rate=Decimal("3.5"),
    utilization_rate=Decimal("0.8"),
    isolation_mode_enabled=False,
    debt_ceiling=Decimal("0"),
    current_isolated_debt=Decimal("0"),
)

CHAINLINK_FEED_ADDR = "0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_aave_client():
    client = MagicMock()
    client._w3 = MagicMock()
    client._w3.eth.contract = MagicMock(return_value=MagicMock())
    client.get_user_account_data = AsyncMock(return_value=SAMPLE_ACCOUNT)
    client.get_reserve_data = AsyncMock(return_value=SAMPLE_RESERVE)
    return client


@pytest.fixture
def mock_safety():
    safety = MagicMock()
    safety.trigger_global_pause = MagicMock()
    return safety


@pytest.fixture
def signal_queue():
    return asyncio.Queue()


@pytest.fixture
def health_monitor(mock_aave_client, mock_safety, signal_queue):
    with patch("core.health_monitor.get_config") as mock_cfg, \
         patch("core.health_monitor.setup_module_logger") as mock_logger:
        mock_loader = MagicMock()
        mock_loader.get_timing_config.return_value = {
            "health_monitoring": {
                "safe_interval_seconds": 15,
                "watch_interval_seconds": 5,
                "warning_interval_seconds": 2,
                "critical_interval_seconds": 1,
                "stale_data_threshold_failures": 5,
            }
        }
        mock_loader.get_positions_config.return_value = {
            "oracle_max_staleness_seconds": 60,
        }
        mock_loader.get_chain_config.return_value = {
            "chainlink_feeds": {
                "BNB_USD": {
                    "address": CHAINLINK_FEED_ADDR,
                    "heartbeat_seconds": 27,
                },
            }
        }
        mock_loader.get_abi.return_value = []
        mock_cfg.return_value = mock_loader
        mock_logger.return_value = MagicMock()

        from core.health_monitor import HealthMonitor
        monitor = HealthMonitor(
            aave_client=mock_aave_client,
            safety=mock_safety,
            user_address=SAMPLE_USER,
            signal_queue=signal_queue,
        )
    return monitor


# ---------------------------------------------------------------------------
# A. _determine_tier tests
# ---------------------------------------------------------------------------

class TestDetermineTier:

    def test_safe_tier(self, health_monitor):
        assert health_monitor._determine_tier(Decimal("2.5")) == HFTier.SAFE

    def test_boundary_2_0_is_watch(self, health_monitor):
        assert health_monitor._determine_tier(Decimal("2.0")) == HFTier.WATCH

    def test_watch_tier(self, health_monitor):
        assert health_monitor._determine_tier(Decimal("1.7")) == HFTier.WATCH

    def test_boundary_1_5_is_watch(self, health_monitor):
        assert health_monitor._determine_tier(Decimal("1.5")) == HFTier.WATCH

    def test_boundary_1_3_is_warning(self, health_monitor):
        assert health_monitor._determine_tier(Decimal("1.3")) == HFTier.WARNING

    def test_critical_tier(self, health_monitor):
        assert health_monitor._determine_tier(Decimal("1.29")) == HFTier.CRITICAL


# ---------------------------------------------------------------------------
# B. _get_poll_interval tests
# ---------------------------------------------------------------------------

class TestGetPollInterval:

    def test_safe_interval(self, health_monitor):
        assert health_monitor._get_poll_interval(HFTier.SAFE) == 15

    def test_watch_interval(self, health_monitor):
        assert health_monitor._get_poll_interval(HFTier.WATCH) == 5

    def test_warning_interval(self, health_monitor):
        assert health_monitor._get_poll_interval(HFTier.WARNING) == 2

    def test_critical_interval(self, health_monitor):
        assert health_monitor._get_poll_interval(HFTier.CRITICAL) == 1


# ---------------------------------------------------------------------------
# C. _poll_once test
# ---------------------------------------------------------------------------

class TestPollOnce:

    async def test_poll_once_returns_health_status(self, health_monitor):
        # Mock the Chainlink feed so oracle freshness passes
        feed_info = health_monitor._chainlink_feeds["BNB_USD"]
        feed_info["contract"].functions.latestRoundData.return_value.call = AsyncMock(
            return_value=(1, 60000000000, int(time.time()), int(time.time()) - 5, 1)
        )

        status = await health_monitor._poll_once()

        assert isinstance(status, HealthStatus)
        assert status.health_factor == Decimal("1.8")
        assert status.tier == HFTier.WATCH
        assert status.collateral_usd == Decimal("10000")
        assert status.debt_usd == Decimal("5000")
        assert status.timestamp > 0


# ---------------------------------------------------------------------------
# D. Oracle freshness tests
# ---------------------------------------------------------------------------

class TestOracleFreshness:

    async def test_fresh_oracle_returns_true(self, health_monitor, mock_safety):
        feed_info = health_monitor._chainlink_feeds["BNB_USD"]
        now = int(time.time())
        feed_info["contract"].functions.latestRoundData.return_value.call = AsyncMock(
            return_value=(1, 60000000000, now, now - 10, 1)
        )

        result = await health_monitor.check_oracle_freshness(CHAINLINK_FEED_ADDR)

        assert result is True
        mock_safety.trigger_global_pause.assert_not_called()

    async def test_stale_oracle_triggers_pause(self, health_monitor, mock_safety):
        feed_info = health_monitor._chainlink_feeds["BNB_USD"]
        now = int(time.time())
        feed_info["contract"].functions.latestRoundData.return_value.call = AsyncMock(
            return_value=(1, 60000000000, now - 300, now - 300, 1)
        )

        result = await health_monitor.check_oracle_freshness(CHAINLINK_FEED_ADDR)

        assert result is False
        mock_safety.trigger_global_pause.assert_called_once()
        call_reason = mock_safety.trigger_global_pause.call_args[0][0]
        assert "stale" in call_reason.lower()

    async def test_incomplete_round_warns_but_does_not_pause(
        self, health_monitor, mock_safety
    ):
        feed_info = health_monitor._chainlink_feeds["BNB_USD"]
        now = int(time.time())
        # answeredInRound (0) < roundId (1) = incomplete
        feed_info["contract"].functions.latestRoundData.return_value.call = AsyncMock(
            return_value=(1, 60000000000, now, now - 5, 0)
        )

        result = await health_monitor.check_oracle_freshness(CHAINLINK_FEED_ADDR)

        assert result is True
        mock_safety.trigger_global_pause.assert_not_called()


# ---------------------------------------------------------------------------
# E. predict_hf_at tests
# ---------------------------------------------------------------------------

class TestPredictHfAt:

    def test_zero_borrow_rate_unchanged(self, health_monitor):
        position = PositionState(
            direction=PositionDirection.LONG,
            debt_token="USDT",
            collateral_token="WBNB",
            debt_usd=Decimal("5000"),
            collateral_usd=Decimal("10000"),
            initial_debt_usd=Decimal("5000"),
            initial_collateral_usd=Decimal("10000"),
            health_factor=Decimal("1.6"),
            borrow_rate_ray=Decimal("0"),
            liquidation_threshold=Decimal("0.8"),
        )
        predicted = health_monitor.predict_hf_at(3600, position)
        assert predicted == Decimal("1.6")

    def test_borrow_rate_decreases_hf(self, health_monitor):
        # 5% APR in RAY format
        rate_ray = Decimal("0.05") * RAY
        position = PositionState(
            direction=PositionDirection.LONG,
            debt_token="USDT",
            collateral_token="WBNB",
            debt_usd=Decimal("5000"),
            collateral_usd=Decimal("10000"),
            initial_debt_usd=Decimal("5000"),
            initial_collateral_usd=Decimal("10000"),
            health_factor=Decimal("1.6"),
            borrow_rate_ray=rate_ray,
            liquidation_threshold=Decimal("0.8"),
        )
        # 24 hours ahead
        predicted = health_monitor.predict_hf_at(86400, position)
        assert predicted < Decimal("1.6")
        # Should still be close to 1.6 (5% APR over 1 day ~ 0.014% decrease)
        assert predicted > Decimal("1.59")

    def test_zero_debt_returns_same_hf(self, health_monitor):
        position = PositionState(
            direction=PositionDirection.LONG,
            debt_token="USDT",
            collateral_token="WBNB",
            debt_usd=Decimal("0"),
            collateral_usd=Decimal("10000"),
            initial_debt_usd=Decimal("0"),
            initial_collateral_usd=Decimal("10000"),
            health_factor=Decimal("999"),
            borrow_rate_ray=Decimal("0.05") * RAY,
            liquidation_threshold=Decimal("0.8"),
        )
        predicted = health_monitor.predict_hf_at(86400, position)
        assert predicted == Decimal("999")


# ---------------------------------------------------------------------------
# F. Run loop tests
# ---------------------------------------------------------------------------

class TestRunLoop:

    async def test_run_pushes_to_queue(self, health_monitor, signal_queue):
        # Mock oracle freshness to pass
        feed_info = health_monitor._chainlink_feeds["BNB_USD"]
        now = int(time.time())
        feed_info["contract"].functions.latestRoundData.return_value.call = AsyncMock(
            return_value=(1, 60000000000, now, now - 5, 1)
        )

        # Patch sleep to stop after first iteration
        call_count = 0

        async def fake_sleep(interval):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                health_monitor.stop()

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await health_monitor.run()

        assert not signal_queue.empty()
        status = signal_queue.get_nowait()
        assert isinstance(status, HealthStatus)
        assert status.health_factor == Decimal("1.8")

    async def test_consecutive_failures_trigger_pause(
        self, health_monitor, mock_safety
    ):
        # Make get_user_account_data always fail
        health_monitor._aave_client.get_user_account_data = AsyncMock(
            side_effect=Exception("RPC timeout")
        )
        health_monitor._stale_data_threshold = 3  # Lower for faster test

        call_count = 0

        async def fake_sleep(interval):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                health_monitor.stop()

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await health_monitor.run()

        mock_safety.trigger_global_pause.assert_called_once()
        reason = mock_safety.trigger_global_pause.call_args[0][0]
        assert "consecutive" in reason.lower()


# ---------------------------------------------------------------------------
# G. get_borrow_rate test
# ---------------------------------------------------------------------------

class TestGetBorrowRate:

    async def test_returns_borrow_rate_info(self, health_monitor):
        result = await health_monitor.get_borrow_rate("0xSomeAsset")

        assert isinstance(result, BorrowRateInfo)
        assert result.variable_rate_apr == Decimal("3.5")
        assert result.utilization_rate == Decimal("0.8")
        assert result.optimal_utilization == Decimal("0.8")
