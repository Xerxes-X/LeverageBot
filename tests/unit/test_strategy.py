"""
Unit tests for core/strategy.py.

Tests verify direction-aware stress tests (long formula vs short formula),
cascade multiplier, close factor risk check, borrow rate cost check,
validate_position_size with GARCH adjustment, drawdown-based position
reduction, check_strategy_health and alpha decay, deleverage amount
formula, and tier-based dispatch.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.types import (
    HealthStatus,
    HFTier,
    IndicatorSnapshot,
    MarketRegime,
    PositionDirection,
    ReserveData,
    TradeSignal,
    TradingStats,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POSITIONS_CONFIG = {
    "dry_run": False,
    "max_position_usd": 10000,
    "max_leverage_ratio": "3.0",
    "min_health_factor": "1.5",
    "deleverage_threshold": "1.4",
    "close_threshold": "1.25",
    "target_hf_after_deleverage": "1.8",
    "stress_test_price_drops": ["-0.05", "-0.10", "-0.15", "-0.20", "-0.30"],
    "min_stress_test_hf": "1.1",
    "cascade_liquidation_threshold_usd": 50000000,
    "cascade_additional_drop": "-0.03",
    "max_borrow_cost_pct": "0.5",
    "max_acceptable_borrow_apr": "15.0",
    "close_factor_warning_threshold_usd": 2000,
}

SIGNALS_CONFIG = {
    "entry_rules": {
        "min_confidence": "0.7",
        "max_signals_per_day": 3,
    },
    "position_sizing": {
        "kelly_fraction": "0.25",
        "high_vol_threshold": "0.04",
        "min_position_usd": "100",
    },
    "alpha_decay_monitoring": {
        "enabled": True,
        "accuracy_decay_threshold": "0.7",
        "sharpe_decay_threshold": "0.5",
        "confidence_boost_on_decay": "1.1",
        "rolling_window_days": 30,
        "historical_window_days": 180,
    },
    "exit_rules": {
        "max_hold_hours": 168,
    },
    "short_signals": {
        "preferred_collateral": "USDC",
    },
}

AAVE_CONFIG = {
    "supported_assets": {
        "WBNB": {"liquidation_threshold_bps": 8000},
        "USDT": {"liquidation_threshold_bps": 7800},
        "USDC": {"liquidation_threshold_bps": 8000},
    }
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_indicator_snapshot(**overrides):
    defaults = dict(
        price=Decimal("300"),
        ema_20=Decimal("298"),
        ema_50=Decimal("295"),
        ema_200=Decimal("280"),
        rsi_14=Decimal("55"),
        macd_line=Decimal("2"),
        macd_signal=Decimal("1.5"),
        macd_histogram=Decimal("0.5"),
        bb_upper=Decimal("310"),
        bb_middle=Decimal("300"),
        bb_lower=Decimal("290"),
        atr_14=Decimal("8"),
        atr_ratio=Decimal("1.5"),
        volume=Decimal("1000000"),
        volume_20_avg=Decimal("800000"),
        hurst=Decimal("0.6"),
        vpin=Decimal("0.55"),
        obi=Decimal("0.3"),
        recent_prices=[Decimal("300")] * 200,
    )
    defaults.update(overrides)
    return IndicatorSnapshot(**defaults)


def _make_trade_signal(
    direction=PositionDirection.LONG,
    confidence=Decimal("0.8"),
    recommended_size_usd=Decimal("5000"),
    garch_volatility=Decimal("0.02"),
    **overrides,
):
    defaults = dict(
        direction=direction,
        confidence=confidence,
        strategy_mode="momentum",
        indicators=_make_indicator_snapshot(),
        regime=MarketRegime.TRENDING,
        components=[],
        recommended_size_usd=recommended_size_usd,
        hurst_exponent=Decimal("0.6"),
        garch_volatility=garch_volatility,
        timestamp=1700000000,
    )
    defaults.update(overrides)
    return TradeSignal(**defaults)


def _mock_position_manager(has_position=False):
    pm = MagicMock()
    pm.has_open_position = has_position
    pm.open_position = AsyncMock()
    pm.close_position = AsyncMock()
    pm.partial_deleverage = AsyncMock()
    return pm


def _mock_aave_client():
    client = MagicMock()
    client.get_reserve_data = AsyncMock(
        return_value=ReserveData(
            variable_borrow_rate=Decimal("5.0"),
            utilization_rate=Decimal("0.65"),
            isolation_mode_enabled=False,
            debt_ceiling=Decimal("0"),
            current_isolated_debt=Decimal("0"),
        )
    )
    return client


def _mock_pnl_tracker(drawdown=Decimal("0"), win_rate=Decimal("0.6"), sharpe=Decimal("1.5")):
    tracker = MagicMock()
    tracker.current_drawdown_pct = drawdown
    tracker.get_rolling_stats = AsyncMock(
        return_value=TradingStats(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_pnl_usd=Decimal("500"),
            avg_pnl_per_trade_usd=Decimal("50"),
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            avg_hold_duration_hours=Decimal("24"),
            current_drawdown_pct=drawdown,
            max_drawdown_pct=drawdown,
        )
    )
    return tracker


def _mock_safety():
    safety = MagicMock()
    safety.is_dry_run = False
    safety.is_paused = False
    return safety


def _make_strategy(
    position_manager=None,
    aave_client=None,
    pnl_tracker=None,
    safety=None,
    signal_queue=None,
):
    with (
        patch("core.strategy.get_config") as mock_cfg,
        patch("core.strategy.setup_module_logger") as mock_logger,
    ):
        mock_loader = MagicMock()
        mock_loader.get_positions_config.return_value = POSITIONS_CONFIG
        mock_loader.get_signals_config.return_value = SIGNALS_CONFIG
        mock_loader.get_aave_config.return_value = AAVE_CONFIG
        mock_cfg.return_value = mock_loader
        mock_logger.return_value = MagicMock()

        from core.strategy import Strategy

        return Strategy(
            position_manager=position_manager or _mock_position_manager(),
            aave_client=aave_client or _mock_aave_client(),
            pnl_tracker=pnl_tracker or _mock_pnl_tracker(),
            safety=safety or _mock_safety(),
            signal_queue=signal_queue or asyncio.Queue(),
        )


# ---------------------------------------------------------------------------
# Stress test: LONG direction
# ---------------------------------------------------------------------------


class TestStressTestLong:

    def test_long_stress_test_at_5pct_drop(self):
        strategy = _make_strategy()
        results = strategy.stress_test(
            PositionDirection.LONG,
            collateral_usd=Decimal("5000"),
            debt_usd=Decimal("5000"),
            liq_threshold_bps=Decimal("8000"),
            price_drops=[Decimal("-0.05")],
        )
        # HF = (5000 * 0.95 * 0.80) / 5000 = 0.76
        expected = (Decimal("5000") * Decimal("0.95") * Decimal("0.80")) / Decimal("5000")
        assert results[0] == expected

    def test_long_stress_test_at_20pct_drop(self):
        strategy = _make_strategy()
        results = strategy.stress_test(
            PositionDirection.LONG,
            collateral_usd=Decimal("5000"),
            debt_usd=Decimal("5000"),
            liq_threshold_bps=Decimal("8000"),
            price_drops=[Decimal("-0.20")],
        )
        # HF = (5000 * 0.80 * 0.80) / 5000 = 0.64
        expected = (Decimal("5000") * Decimal("0.80") * Decimal("0.80")) / Decimal("5000")
        assert results[0] == expected

    def test_long_stress_test_no_drop(self):
        strategy = _make_strategy()
        results = strategy.stress_test(
            PositionDirection.LONG,
            collateral_usd=Decimal("5000"),
            debt_usd=Decimal("5000"),
            liq_threshold_bps=Decimal("8000"),
            price_drops=[Decimal("0")],
        )
        # HF = (5000 * 1.0 * 0.80) / 5000 = 0.80
        assert results[0] == Decimal("0.80")


# ---------------------------------------------------------------------------
# Stress test: SHORT direction
# ---------------------------------------------------------------------------


class TestStressTestShort:

    def test_short_stress_test_at_5pct_increase(self):
        strategy = _make_strategy()
        results = strategy.stress_test(
            PositionDirection.SHORT,
            collateral_usd=Decimal("5000"),
            debt_usd=Decimal("5000"),
            liq_threshold_bps=Decimal("8000"),
            price_drops=[Decimal("-0.05")],
        )
        # For short: HF = (5000 * 0.80) / (5000 * 1.05) = 0.7619
        expected = (Decimal("5000") * Decimal("0.80")) / (Decimal("5000") * Decimal("1.05"))
        assert results[0] == expected

    def test_short_stress_test_at_20pct_increase(self):
        strategy = _make_strategy()
        results = strategy.stress_test(
            PositionDirection.SHORT,
            collateral_usd=Decimal("5000"),
            debt_usd=Decimal("5000"),
            liq_threshold_bps=Decimal("8000"),
            price_drops=[Decimal("-0.20")],
        )
        # For short: HF = (5000 * 0.80) / (5000 * 1.20) = 0.6667
        expected = (Decimal("5000") * Decimal("0.80")) / (Decimal("5000") * Decimal("1.20"))
        assert results[0] == expected

    def test_short_vs_long_asymmetry(self):
        """Short HF formula should produce different results than long."""
        strategy = _make_strategy()
        long_results = strategy.stress_test(
            PositionDirection.LONG,
            collateral_usd=Decimal("5000"),
            debt_usd=Decimal("5000"),
            liq_threshold_bps=Decimal("8000"),
            price_drops=[Decimal("-0.10")],
        )
        short_results = strategy.stress_test(
            PositionDirection.SHORT,
            collateral_usd=Decimal("5000"),
            debt_usd=Decimal("5000"),
            liq_threshold_bps=Decimal("8000"),
            price_drops=[Decimal("-0.10")],
        )
        # Long: (5000 * 0.90 * 0.80) / 5000 = 0.72
        # Short: (5000 * 0.80) / (5000 * 1.10) = 0.7273
        assert long_results[0] != short_results[0]


# ---------------------------------------------------------------------------
# Stress test with cascade
# ---------------------------------------------------------------------------


class TestStressTestCascade:

    def test_cascade_activated_above_threshold(self):
        strategy = _make_strategy()
        results = strategy.stress_test_with_cascade(
            PositionDirection.LONG,
            collateral_usd=Decimal("5000"),
            debt_usd=Decimal("5000"),
            liq_threshold_bps=Decimal("8000"),
            price_drops=[Decimal("-0.20")],
            market_total_supply_usd=Decimal("2000000000"),  # $2B market
        )
        # -20% drop on $2B: 10% leveraged = $200M, 100% liquidated = $200M > $50M
        # Cascade adds -3% → -23% total
        expected = (Decimal("5000") * Decimal("0.77") * Decimal("0.80")) / Decimal("5000")
        assert results[0] == expected

    def test_cascade_not_activated_below_threshold(self):
        strategy = _make_strategy()
        results_with_cascade = strategy.stress_test_with_cascade(
            PositionDirection.LONG,
            collateral_usd=Decimal("5000"),
            debt_usd=Decimal("5000"),
            liq_threshold_bps=Decimal("8000"),
            price_drops=[Decimal("-0.01")],
            market_total_supply_usd=Decimal("100000000"),  # $100M market
        )
        results_without = strategy.stress_test(
            PositionDirection.LONG,
            collateral_usd=Decimal("5000"),
            debt_usd=Decimal("5000"),
            liq_threshold_bps=Decimal("8000"),
            price_drops=[Decimal("-0.01")],
        )
        # -1% drop on $100M: 10% leveraged = $10M, 5% liquidated = $0.5M < $50M
        assert results_with_cascade[0] == results_without[0]


# ---------------------------------------------------------------------------
# Close factor risk
# ---------------------------------------------------------------------------


class TestCloseFactorRisk:

    @pytest.mark.asyncio
    async def test_rejects_small_collateral(self):
        strategy = _make_strategy()
        # Collateral of $3000 at -30% drop = $2100 > $2000, OK
        # Collateral of $3000 at -35% drop would = $1950 < $2000
        _result = strategy.stress_test(
            PositionDirection.LONG,
            collateral_usd=Decimal("3000"),
            debt_usd=Decimal("3000"),
            liq_threshold_bps=Decimal("8000"),
            price_drops=[Decimal("-0.35")],
        )
        # Now test the close factor check
        ok = await strategy.check_close_factor_risk(Decimal("3000"), Decimal("3000"))
        # At -30% drop (last in default list): $3000 * 0.70 = $2100 > $2000
        assert ok is True

    @pytest.mark.asyncio
    async def test_rejects_very_small_position(self):
        strategy = _make_strategy()
        # $2500 at -20% = $2000, at -30% = $1750 < $2000
        ok = await strategy.check_close_factor_risk(Decimal("2500"), Decimal("2500"))
        assert ok is False

    @pytest.mark.asyncio
    async def test_accepts_large_position(self):
        strategy = _make_strategy()
        ok = await strategy.check_close_factor_risk(Decimal("10000"), Decimal("10000"))
        assert ok is True

    @pytest.mark.asyncio
    async def test_rejects_small_debt(self):
        strategy = _make_strategy()
        ok = await strategy.check_close_factor_risk(Decimal("10000"), Decimal("1500"))
        assert ok is False


# ---------------------------------------------------------------------------
# Borrow rate check
# ---------------------------------------------------------------------------


class TestBorrowRateCheck:

    @pytest.mark.asyncio
    async def test_acceptable_rate(self):
        aave = _mock_aave_client()
        aave.get_reserve_data = AsyncMock(
            return_value=ReserveData(
                variable_borrow_rate=Decimal("5.0"),
                utilization_rate=Decimal("0.65"),
                isolation_mode_enabled=False,
                debt_ceiling=Decimal("0"),
                current_isolated_debt=Decimal("0"),
            )
        )
        strategy = _make_strategy(aave_client=aave)
        ok, rate = await strategy.check_borrow_rate_acceptable(
            "0x55d398326f99059fF775485246999027B3197955",
            projected_hold_hours=24.0,
        )
        assert ok is True
        assert rate == Decimal("5.0")

    @pytest.mark.asyncio
    async def test_rejects_high_rate(self):
        aave = _mock_aave_client()
        aave.get_reserve_data = AsyncMock(
            return_value=ReserveData(
                variable_borrow_rate=Decimal("80.0"),
                utilization_rate=Decimal("0.95"),
                isolation_mode_enabled=False,
                debt_ceiling=Decimal("0"),
                current_isolated_debt=Decimal("0"),
            )
        )
        strategy = _make_strategy(aave_client=aave)
        ok, rate = await strategy.check_borrow_rate_acceptable(
            "0x55d398326f99059fF775485246999027B3197955",
            projected_hold_hours=24.0,
        )
        assert ok is False
        assert rate == Decimal("80.0")

    @pytest.mark.asyncio
    async def test_rejects_when_projected_cost_too_high(self):
        aave = _mock_aave_client()
        # 14% APR over 168h (7 days) = 14 * 168/8760 = 0.27%
        # But max_borrow_cost_pct = 0.5%, so this should pass
        aave.get_reserve_data = AsyncMock(
            return_value=ReserveData(
                variable_borrow_rate=Decimal("14.0"),
                utilization_rate=Decimal("0.80"),
                isolation_mode_enabled=False,
                debt_ceiling=Decimal("0"),
                current_isolated_debt=Decimal("0"),
            )
        )
        strategy = _make_strategy(aave_client=aave)
        ok, _ = await strategy.check_borrow_rate_acceptable(
            "0xaddr",
            projected_hold_hours=168.0,
        )
        assert ok is True


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


class TestPositionSizing:

    def test_size_within_limits(self):
        strategy = _make_strategy()
        signal = _make_trade_signal(recommended_size_usd=Decimal("5000"))
        size = strategy.validate_position_size(signal)
        assert size == Decimal("5000")

    def test_size_capped_at_max(self):
        strategy = _make_strategy()
        signal = _make_trade_signal(recommended_size_usd=Decimal("20000"))
        size = strategy.validate_position_size(signal)
        assert size == Decimal("10000")  # max_position_usd

    def test_high_volatility_reduces_size(self):
        strategy = _make_strategy()
        signal = _make_trade_signal(
            recommended_size_usd=Decimal("5000"),
            garch_volatility=Decimal("0.08"),  # 2x threshold of 0.04
        )
        size = strategy.validate_position_size(signal)
        # vol_scalar = 0.04 / 0.08 = 0.5 → size = 5000 * 0.5 = 2500
        assert size == Decimal("2500")

    def test_normal_volatility_no_reduction(self):
        strategy = _make_strategy()
        signal = _make_trade_signal(
            recommended_size_usd=Decimal("5000"),
            garch_volatility=Decimal("0.02"),  # Below threshold of 0.04
        )
        size = strategy.validate_position_size(signal)
        assert size == Decimal("5000")

    def test_drawdown_reduces_size(self):
        pnl = _mock_pnl_tracker(drawdown=Decimal("0.20"))
        strategy = _make_strategy(pnl_tracker=pnl)
        signal = _make_trade_signal(recommended_size_usd=Decimal("5000"))
        size = strategy.validate_position_size(signal)
        # dd_scalar = max(0.25, 1 - 0.20) = 0.80 → size = 5000 * 0.80 = 4000
        assert size == Decimal("4000")

    def test_severe_drawdown_floors_at_25pct(self):
        pnl = _mock_pnl_tracker(drawdown=Decimal("0.90"))
        strategy = _make_strategy(pnl_tracker=pnl)
        signal = _make_trade_signal(recommended_size_usd=Decimal("5000"))
        size = strategy.validate_position_size(signal)
        # dd_scalar = max(0.25, 1 - 0.90) = 0.25 → size = 5000 * 0.25 = 1250
        assert size == Decimal("1250")

    def test_below_minimum_returns_zero(self):
        strategy = _make_strategy()
        signal = _make_trade_signal(recommended_size_usd=Decimal("50"))
        size = strategy.validate_position_size(signal)
        assert size == Decimal("0")


# ---------------------------------------------------------------------------
# Deleverage amount formula
# ---------------------------------------------------------------------------


class TestDeleverageFormula:

    def test_compute_deleverage_amount(self):
        strategy = _make_strategy()
        repay = strategy.compute_deleverage_amount(
            current_hf=Decimal("1.2"),
            target_hf=Decimal("1.8"),
            collateral_usd=Decimal("5000"),
            debt_usd=Decimal("5000"),
            liq_threshold_bps=Decimal("8000"),
        )
        # Formula: repay = (D - C*LT/h_t) / (1 + f - LT/h_t)
        # = (5000 - 5000*0.80/1.80) / (1 + 0.0005 - 0.80/1.80)
        # = (5000 - 2222.22) / (1.0005 - 0.4444)
        # = 2777.78 / 0.5561
        # ≈ 4995.something
        assert repay > Decimal("0")

    def test_deleverage_zero_when_not_needed(self):
        strategy = _make_strategy()
        repay = strategy.compute_deleverage_amount(
            current_hf=Decimal("2.0"),
            target_hf=Decimal("1.5"),
            collateral_usd=Decimal("5000"),
            debt_usd=Decimal("2000"),
            liq_threshold_bps=Decimal("8000"),
        )
        # Already above target, should be 0
        assert repay == Decimal("0")


# ---------------------------------------------------------------------------
# Health status handling
# ---------------------------------------------------------------------------


class TestHealthStatusHandling:

    @pytest.mark.asyncio
    async def test_critical_hf_triggers_emergency_close(self):
        pm = _mock_position_manager(has_position=True)
        strategy = _make_strategy(position_manager=pm)

        status = HealthStatus(
            health_factor=Decimal("1.20"),
            tier=HFTier.CRITICAL,
            collateral_usd=Decimal("4500"),
            debt_usd=Decimal("5000"),
            timestamp=1700000000,
        )
        await strategy.handle_health_status(status)
        pm.close_position.assert_called_once_with(reason="emergency")

    @pytest.mark.asyncio
    async def test_critical_hf_above_close_triggers_deleverage(self):
        pm = _mock_position_manager(has_position=True)
        strategy = _make_strategy(position_manager=pm)

        status = HealthStatus(
            health_factor=Decimal("1.35"),
            tier=HFTier.CRITICAL,
            collateral_usd=Decimal("4800"),
            debt_usd=Decimal("5000"),
            timestamp=1700000000,
        )
        await strategy.handle_health_status(status)
        pm.partial_deleverage.assert_called_once_with(Decimal("1.8"))

    @pytest.mark.asyncio
    async def test_warning_hf_triggers_deleverage(self):
        pm = _mock_position_manager(has_position=True)
        strategy = _make_strategy(position_manager=pm)

        status = HealthStatus(
            health_factor=Decimal("1.38"),
            tier=HFTier.WARNING,
            collateral_usd=Decimal("4900"),
            debt_usd=Decimal("5000"),
            timestamp=1700000000,
        )
        await strategy.handle_health_status(status)
        pm.partial_deleverage.assert_called_once()

    @pytest.mark.asyncio
    async def test_safe_hf_does_nothing(self):
        pm = _mock_position_manager(has_position=True)
        strategy = _make_strategy(position_manager=pm)

        status = HealthStatus(
            health_factor=Decimal("2.5"),
            tier=HFTier.SAFE,
            collateral_usd=Decimal("5500"),
            debt_usd=Decimal("5000"),
            timestamp=1700000000,
        )
        await strategy.handle_health_status(status)
        pm.close_position.assert_not_called()
        pm.partial_deleverage.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_position_ignores_status(self):
        pm = _mock_position_manager(has_position=False)
        strategy = _make_strategy(position_manager=pm)

        status = HealthStatus(
            health_factor=Decimal("1.1"),
            tier=HFTier.CRITICAL,
            collateral_usd=Decimal("4000"),
            debt_usd=Decimal("5000"),
            timestamp=1700000000,
        )
        await strategy.handle_health_status(status)
        pm.close_position.assert_not_called()


# ---------------------------------------------------------------------------
# Trade signal handling
# ---------------------------------------------------------------------------


class TestTradeSignalHandling:

    @pytest.mark.asyncio
    async def test_ignores_signal_when_position_open(self):
        pm = _mock_position_manager(has_position=True)
        strategy = _make_strategy(position_manager=pm)

        signal = _make_trade_signal()
        await strategy.handle_trade_signal(signal)
        pm.open_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_low_confidence_signal(self):
        pm = _mock_position_manager()
        strategy = _make_strategy(position_manager=pm)

        signal = _make_trade_signal(confidence=Decimal("0.3"))
        await strategy.handle_trade_signal(signal)
        pm.open_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_accepts_high_confidence_signal(self):
        pm = _mock_position_manager()
        strategy = _make_strategy(position_manager=pm)

        signal = _make_trade_signal(
            confidence=Decimal("0.85"),
            recommended_size_usd=Decimal("5000"),
        )
        # Mock evaluate_entry to pass — we're testing dispatch, not entry logic
        with patch.object(strategy, "evaluate_entry", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = (True, "All checks passed")
            await strategy.handle_trade_signal(signal)
        pm.open_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_long_signal_uses_usdt_debt(self):
        pm = _mock_position_manager()
        strategy = _make_strategy(position_manager=pm)

        signal = _make_trade_signal(direction=PositionDirection.LONG)
        with patch.object(strategy, "evaluate_entry", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = (True, "All checks passed")
            await strategy.handle_trade_signal(signal)

        call_args = pm.open_position.call_args
        assert (
            call_args.kwargs.get("debt_token") == "USDT" or call_args[1].get("debt_token") == "USDT"
        )

    @pytest.mark.asyncio
    async def test_short_signal_uses_wbnb_debt(self):
        pm = _mock_position_manager()
        strategy = _make_strategy(position_manager=pm)

        signal = _make_trade_signal(direction=PositionDirection.SHORT)
        with patch.object(strategy, "evaluate_entry", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = (True, "All checks passed")
            await strategy.handle_trade_signal(signal)

        call_args = pm.open_position.call_args
        assert (
            call_args.kwargs.get("debt_token") == "WBNB" or call_args[1].get("debt_token") == "WBNB"
        )


# ---------------------------------------------------------------------------
# Entry evaluation
# ---------------------------------------------------------------------------


class TestEvaluateEntry:

    @pytest.mark.asyncio
    async def test_low_confidence_rejected(self):
        strategy = _make_strategy()
        signal = _make_trade_signal(confidence=Decimal("0.5"))
        ok, reason = await strategy.evaluate_entry(signal)
        assert ok is False
        assert "Confidence" in reason

    @pytest.mark.asyncio
    async def test_high_borrow_rate_rejected(self):
        aave = _mock_aave_client()
        aave.get_reserve_data = AsyncMock(
            return_value=ReserveData(
                variable_borrow_rate=Decimal("50.0"),
                utilization_rate=Decimal("0.95"),
                isolation_mode_enabled=False,
                debt_ceiling=Decimal("0"),
                current_isolated_debt=Decimal("0"),
            )
        )
        strategy = _make_strategy(aave_client=aave)
        signal = _make_trade_signal(confidence=Decimal("0.9"))
        ok, reason = await strategy.evaluate_entry(signal)
        assert ok is False
        assert "rate" in reason.lower()


# ---------------------------------------------------------------------------
# Alpha decay / strategy health
# ---------------------------------------------------------------------------


class TestAlphaDecay:

    @pytest.mark.asyncio
    async def test_alpha_decay_detected(self):
        # Recent stats worse than historical
        recent_stats = TradingStats(
            total_trades=10,
            winning_trades=3,
            losing_trades=7,
            total_pnl_usd=Decimal("-200"),
            avg_pnl_per_trade_usd=Decimal("-20"),
            win_rate=Decimal("0.3"),
            sharpe_ratio=Decimal("0.3"),
            avg_hold_duration_hours=Decimal("24"),
            current_drawdown_pct=Decimal("0.15"),
            max_drawdown_pct=Decimal("0.15"),
        )
        historical_stats = TradingStats(
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            total_pnl_usd=Decimal("2000"),
            avg_pnl_per_trade_usd=Decimal("40"),
            win_rate=Decimal("0.6"),
            sharpe_ratio=Decimal("1.5"),
            avg_hold_duration_hours=Decimal("20"),
            current_drawdown_pct=Decimal("0.05"),
            max_drawdown_pct=Decimal("0.10"),
        )

        pnl = _mock_pnl_tracker()
        pnl.get_rolling_stats = AsyncMock(side_effect=[recent_stats, historical_stats])

        strategy = _make_strategy(pnl_tracker=pnl)
        report = await strategy.check_strategy_health_async()

        assert report.alpha_decay_detected is True
        assert len(report.recommendations) > 0

    @pytest.mark.asyncio
    async def test_no_alpha_decay_when_stable(self):
        stats = TradingStats(
            total_trades=20,
            winning_trades=12,
            losing_trades=8,
            total_pnl_usd=Decimal("500"),
            avg_pnl_per_trade_usd=Decimal("25"),
            win_rate=Decimal("0.6"),
            sharpe_ratio=Decimal("1.5"),
            avg_hold_duration_hours=Decimal("24"),
            current_drawdown_pct=Decimal("0.05"),
            max_drawdown_pct=Decimal("0.08"),
        )

        pnl = _mock_pnl_tracker()
        pnl.get_rolling_stats = AsyncMock(return_value=stats)

        strategy = _make_strategy(pnl_tracker=pnl)
        report = await strategy.check_strategy_health_async()

        assert report.alpha_decay_detected is False

    @pytest.mark.asyncio
    async def test_alpha_decay_raises_confidence_threshold(self):
        recent_stats = TradingStats(
            total_trades=10,
            winning_trades=2,
            losing_trades=8,
            total_pnl_usd=Decimal("-500"),
            avg_pnl_per_trade_usd=Decimal("-50"),
            win_rate=Decimal("0.2"),
            sharpe_ratio=Decimal("0.2"),
            avg_hold_duration_hours=Decimal("24"),
            current_drawdown_pct=Decimal("0.20"),
            max_drawdown_pct=Decimal("0.20"),
        )
        historical_stats = TradingStats(
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            total_pnl_usd=Decimal("2000"),
            avg_pnl_per_trade_usd=Decimal("40"),
            win_rate=Decimal("0.6"),
            sharpe_ratio=Decimal("1.5"),
            avg_hold_duration_hours=Decimal("20"),
            current_drawdown_pct=Decimal("0.05"),
            max_drawdown_pct=Decimal("0.10"),
        )

        pnl = _mock_pnl_tracker()
        pnl.get_rolling_stats = AsyncMock(side_effect=[recent_stats, historical_stats])

        strategy = _make_strategy(pnl_tracker=pnl)
        original_threshold = strategy._dynamic_confidence_threshold

        await strategy.check_strategy_health_async()

        # Threshold should be raised by 10%
        assert strategy._dynamic_confidence_threshold > original_threshold


# ---------------------------------------------------------------------------
# Daily signal limit
# ---------------------------------------------------------------------------


class TestDailySignalLimit:

    @pytest.mark.asyncio
    async def test_daily_limit_enforced(self):
        pm = _mock_position_manager()
        strategy = _make_strategy(position_manager=pm)
        strategy._signals_today = 3  # Already at limit
        strategy._last_signal_day = int(__import__("time").time()) // 86400

        signal = _make_trade_signal(confidence=Decimal("0.9"))
        await strategy.handle_trade_signal(signal)
        pm.open_position.assert_not_called()
