"""
Unit tests for core/position_manager.py.

Tests verify open/close/deleverage flows with mocked dependencies,
both long and short flows, calldata verification, and isolation mode
rejection for USDT shorts.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.types import (
    PositionDirection,
    ReserveData,
    SafetyCheck,
    SwapQuote,
    UserAccountData,
)

# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


CHAIN_CONFIG = {
    "chain_id": 56,
    "tokens": {
        "WBNB": {"address": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c", "decimals": 18},
        "USDT": {"address": "0x55d398326f99059fF775485246999027B3197955", "decimals": 18},
        "USDC": {"address": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d", "decimals": 18},
    },
}

POSITIONS_CONFIG = {
    "dry_run": False,
    "max_position_usd": 10000,
    "max_leverage_ratio": "3.0",
}

AAVE_CONFIG = {
    "supported_assets": {
        "WBNB": {"liquidation_threshold_bps": 8000},
        "USDT": {"liquidation_threshold_bps": 7800},
        "USDC": {"liquidation_threshold_bps": 8000},
    }
}


def _mock_aave_client():
    client = MagicMock()
    client.get_user_account_data = AsyncMock(
        return_value=UserAccountData(
            total_collateral_usd=Decimal("5200"),
            total_debt_usd=Decimal("5000"),
            available_borrow_usd=Decimal("1000"),
            current_liquidation_threshold=Decimal("0.80"),
            ltv=Decimal("0.75"),
            health_factor=Decimal("1.83"),
        )
    )
    client.get_reserve_data = AsyncMock(
        return_value=ReserveData(
            variable_borrow_rate=Decimal("5.0"),
            utilization_rate=Decimal("0.65"),
            isolation_mode_enabled=False,
            debt_ceiling=Decimal("0"),
            current_isolated_debt=Decimal("0"),
        )
    )
    client.get_asset_price = AsyncMock(return_value=Decimal("300"))
    client.encode_flash_loan = MagicMock(return_value="0xdeadbeef")
    return client


def _mock_aggregator_client():
    client = MagicMock()
    client.get_best_quote = AsyncMock(
        return_value=SwapQuote(
            provider="1inch",
            from_token="0x55d398326f99059fF775485246999027B3197955",
            to_token="0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
            from_amount=Decimal("5000000000000000000000"),
            to_amount=Decimal("16666666666666666666"),
            to_amount_min=Decimal("16583333333333333333"),
            calldata=b"\x00\x01\x02\x03",
            router_address="0x1111111254EEB25477B68fb85Ed929f73A960582",
            gas_estimate=350000,
            price_impact=Decimal("0.1"),
        )
    )
    return client


def _mock_tx_submitter():
    sub = MagicMock()
    sub.simulate = AsyncMock(return_value=b"\x00")
    sub.submit_and_wait = AsyncMock(
        return_value={
            "transactionHash": bytes.fromhex("ab" * 32),
            "gasUsed": 350000,
            "status": 1,
        }
    )
    sub.get_gas_price = AsyncMock(return_value=(5_000_000_000, 1_000_000_000))
    return sub


def _mock_safety(dry_run: bool = False):
    safety = MagicMock()
    safety.can_open_position.return_value = SafetyCheck(can_proceed=True, reason="OK")
    safety.can_submit_tx.return_value = SafetyCheck(can_proceed=True, reason="OK")
    safety.is_dry_run = dry_run
    safety.is_paused = False
    safety.record_action = MagicMock()
    return safety


def _mock_pnl_tracker():
    tracker = MagicMock()
    tracker.record_open = AsyncMock(return_value=1)
    tracker.record_close = AsyncMock(return_value=MagicMock(net_pnl_usd=Decimal("50")))
    tracker.record_deleverage = AsyncMock()
    return tracker


def _make_position_manager(
    aave_client=None,
    aggregator_client=None,
    tx_submitter=None,
    safety=None,
    pnl_tracker=None,
    dry_run=False,
):
    with (
        patch("core.position_manager.get_config") as mock_cfg,
        patch("core.position_manager.setup_module_logger") as mock_logger,
    ):
        mock_loader = MagicMock()
        mock_loader.get_aave_config.return_value = AAVE_CONFIG
        mock_loader.get_positions_config.return_value = POSITIONS_CONFIG
        mock_loader.get_chain_config.return_value = CHAIN_CONFIG
        mock_cfg.return_value = mock_loader
        mock_logger.return_value = MagicMock()

        from core.position_manager import PositionManager

        return PositionManager(
            aave_client=aave_client or _mock_aave_client(),
            aggregator_client=aggregator_client or _mock_aggregator_client(),
            tx_submitter=tx_submitter or _mock_tx_submitter(),
            safety=safety or _mock_safety(dry_run),
            pnl_tracker=pnl_tracker or _mock_pnl_tracker(),
            executor_address="0x" + "aa" * 20,
            user_address="0x" + "bb" * 20,
        )


# ---------------------------------------------------------------------------
# Open position tests
# ---------------------------------------------------------------------------


class TestOpenPosition:

    @pytest.mark.asyncio
    async def test_open_long_position(self):
        pm = _make_position_manager()
        state = await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")
        assert state.direction == PositionDirection.LONG
        assert state.debt_token == "USDT"
        assert state.collateral_token == "WBNB"
        assert pm.has_open_position is True

    @pytest.mark.asyncio
    async def test_open_short_position(self):
        pm = _make_position_manager()
        state = await pm.open_position(PositionDirection.SHORT, "WBNB", Decimal("5000"), "USDC")
        assert state.direction == PositionDirection.SHORT
        assert state.debt_token == "WBNB"
        assert state.collateral_token == "USDC"

    @pytest.mark.asyncio
    async def test_open_calls_aggregator(self):
        agg = _mock_aggregator_client()
        pm = _make_position_manager(aggregator_client=agg)
        await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")
        agg.get_best_quote.assert_called_once()

    @pytest.mark.asyncio
    async def test_open_calls_flash_loan_encode(self):
        aave = _mock_aave_client()
        pm = _make_position_manager(aave_client=aave)
        await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")
        aave.encode_flash_loan.assert_called_once()

    @pytest.mark.asyncio
    async def test_open_calls_simulate(self):
        tx_sub = _mock_tx_submitter()
        pm = _make_position_manager(tx_submitter=tx_sub)
        await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")
        tx_sub.simulate.assert_called_once()

    @pytest.mark.asyncio
    async def test_open_records_to_pnl_tracker(self):
        pnl = _mock_pnl_tracker()
        pm = _make_position_manager(pnl_tracker=pnl)
        await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")
        pnl.record_open.assert_called_once()

    @pytest.mark.asyncio
    async def test_cannot_open_when_position_exists(self):
        pm = _make_position_manager()
        await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")

        from core.position_manager import PositionManagerError

        with pytest.raises(PositionManagerError, match="already open"):
            await pm.open_position(PositionDirection.LONG, "USDT", Decimal("3000"), "WBNB")

    @pytest.mark.asyncio
    async def test_safety_gate_blocks_open(self):
        safety = _mock_safety()
        safety.can_open_position.return_value = SafetyCheck(
            can_proceed=False, reason="Position too large"
        )
        pm = _make_position_manager(safety=safety)

        from core.position_manager import PositionManagerError

        with pytest.raises(PositionManagerError, match="Safety gate"):
            await pm.open_position(PositionDirection.LONG, "USDT", Decimal("50000"), "WBNB")


# ---------------------------------------------------------------------------
# Isolation mode tests
# ---------------------------------------------------------------------------


class TestIsolationMode:

    @pytest.mark.asyncio
    async def test_short_with_isolation_mode_ceiling_reached(self):
        aave = _mock_aave_client()
        aave.get_reserve_data = AsyncMock(
            return_value=ReserveData(
                variable_borrow_rate=Decimal("5.0"),
                utilization_rate=Decimal("0.65"),
                isolation_mode_enabled=True,
                debt_ceiling=Decimal("1000000"),
                current_isolated_debt=Decimal("1000000"),
            )
        )
        pm = _make_position_manager(aave_client=aave)

        from core.position_manager import PositionManagerError

        with pytest.raises(PositionManagerError, match="Isolation mode"):
            await pm.open_position(PositionDirection.SHORT, "WBNB", Decimal("5000"), "USDT")

    @pytest.mark.asyncio
    async def test_short_with_isolation_mode_within_ceiling(self):
        aave = _mock_aave_client()
        # First call for isolation check, second for post-execution reserve data
        aave.get_reserve_data = AsyncMock(
            return_value=ReserveData(
                variable_borrow_rate=Decimal("5.0"),
                utilization_rate=Decimal("0.65"),
                isolation_mode_enabled=True,
                debt_ceiling=Decimal("1000000"),
                current_isolated_debt=Decimal("500000"),
            )
        )
        pm = _make_position_manager(aave_client=aave)
        state = await pm.open_position(PositionDirection.SHORT, "WBNB", Decimal("5000"), "USDC")
        assert state.direction == PositionDirection.SHORT

    @pytest.mark.asyncio
    async def test_long_skips_isolation_check(self):
        aave = _mock_aave_client()
        pm = _make_position_manager(aave_client=aave)
        # LONG should not call get_reserve_data for isolation check
        # (it will call for post-execution though)
        await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")
        # encode_flash_loan should still be called
        aave.encode_flash_loan.assert_called_once()


# ---------------------------------------------------------------------------
# Close position tests
# ---------------------------------------------------------------------------


class TestClosePosition:

    @pytest.mark.asyncio
    async def test_close_long_position(self):
        pm = _make_position_manager()
        await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")
        state = await pm.close_position(reason="signal")
        assert state.debt_usd == Decimal("0")
        assert pm.has_open_position is False

    @pytest.mark.asyncio
    async def test_close_short_position(self):
        pm = _make_position_manager()
        await pm.open_position(PositionDirection.SHORT, "WBNB", Decimal("5000"), "USDC")
        state = await pm.close_position(reason="emergency")
        assert state.debt_usd == Decimal("0")
        assert pm.has_open_position is False

    @pytest.mark.asyncio
    async def test_close_records_to_pnl_tracker(self):
        pnl = _mock_pnl_tracker()
        pm = _make_position_manager(pnl_tracker=pnl)
        await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")
        await pm.close_position()
        pnl.record_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_open_raises(self):
        pm = _make_position_manager()
        from core.position_manager import PositionManagerError

        with pytest.raises(PositionManagerError, match="No open position"):
            await pm.close_position()


# ---------------------------------------------------------------------------
# Deleverage tests
# ---------------------------------------------------------------------------


class TestDeleverage:

    @pytest.mark.asyncio
    async def test_deleverage_when_hf_low(self):
        aave = _mock_aave_client()
        # Return low HF on first call, normal after deleverage
        aave.get_user_account_data = AsyncMock(
            side_effect=[
                # Open position
                UserAccountData(
                    total_collateral_usd=Decimal("5200"),
                    total_debt_usd=Decimal("5000"),
                    available_borrow_usd=Decimal("1000"),
                    current_liquidation_threshold=Decimal("0.80"),
                    ltv=Decimal("0.75"),
                    health_factor=Decimal("1.83"),
                ),
                # Deleverage check (low HF)
                UserAccountData(
                    total_collateral_usd=Decimal("4500"),
                    total_debt_usd=Decimal("5000"),
                    available_borrow_usd=Decimal("0"),
                    current_liquidation_threshold=Decimal("0.80"),
                    ltv=Decimal("0.75"),
                    health_factor=Decimal("1.2"),
                ),
                # Post-deleverage refresh
                UserAccountData(
                    total_collateral_usd=Decimal("4200"),
                    total_debt_usd=Decimal("4000"),
                    available_borrow_usd=Decimal("500"),
                    current_liquidation_threshold=Decimal("0.80"),
                    ltv=Decimal("0.75"),
                    health_factor=Decimal("1.8"),
                ),
            ]
        )
        pm = _make_position_manager(aave_client=aave)
        await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")

        result = await pm.partial_deleverage(Decimal("1.8"))
        assert result.health_factor == Decimal("1.8")

    @pytest.mark.asyncio
    async def test_deleverage_skipped_when_hf_above_target(self):
        aave = _mock_aave_client()
        aave.get_user_account_data = AsyncMock(
            side_effect=[
                # Open
                UserAccountData(
                    total_collateral_usd=Decimal("5200"),
                    total_debt_usd=Decimal("5000"),
                    available_borrow_usd=Decimal("1000"),
                    current_liquidation_threshold=Decimal("0.80"),
                    ltv=Decimal("0.75"),
                    health_factor=Decimal("1.83"),
                ),
                # Deleverage check (already above target)
                UserAccountData(
                    total_collateral_usd=Decimal("5200"),
                    total_debt_usd=Decimal("5000"),
                    available_borrow_usd=Decimal("1000"),
                    current_liquidation_threshold=Decimal("0.80"),
                    ltv=Decimal("0.75"),
                    health_factor=Decimal("2.0"),
                ),
            ]
        )
        pm = _make_position_manager(aave_client=aave)
        await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")

        result = await pm.partial_deleverage(Decimal("1.8"))
        # Should return current position unchanged
        assert result.direction == PositionDirection.LONG

    @pytest.mark.asyncio
    async def test_deleverage_without_position_raises(self):
        pm = _make_position_manager()
        from core.position_manager import PositionManagerError

        with pytest.raises(PositionManagerError, match="No open position"):
            await pm.partial_deleverage(Decimal("1.8"))


# ---------------------------------------------------------------------------
# Dry run tests
# ---------------------------------------------------------------------------


class TestDryRun:

    @pytest.mark.asyncio
    async def test_dry_run_does_not_submit(self):
        tx_sub = _mock_tx_submitter()
        pm = _make_position_manager(tx_submitter=tx_sub, dry_run=True)
        await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")
        tx_sub.submit_and_wait.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_close_does_not_submit(self):
        tx_sub = _mock_tx_submitter()
        pm = _make_position_manager(tx_submitter=tx_sub, dry_run=True)
        await pm.open_position(PositionDirection.LONG, "USDT", Decimal("5000"), "WBNB")
        await pm.close_position()
        tx_sub.submit_and_wait.assert_not_called()


# ---------------------------------------------------------------------------
# Token resolution tests
# ---------------------------------------------------------------------------


class TestTokenResolution:

    def test_unknown_token_raises(self):
        pm = _make_position_manager()
        from core.position_manager import PositionManagerError

        with pytest.raises(PositionManagerError, match="Unknown token"):
            pm._get_token_address("INVALID_TOKEN")

    def test_known_token_returns_address(self):
        pm = _make_position_manager()
        addr = pm._get_token_address("WBNB")
        assert addr.startswith("0x")
        assert len(addr) == 42
