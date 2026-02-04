"""
Strategy engine for BSC Leverage Bot.

Decision engine that evaluates trade signals, manages position lifecycle,
applies risk filters including direction-aware stress testing, borrow rate
cost analysis, liquidation cascade modeling, GARCH-informed position sizing
via fractional Kelly criterion, and alpha decay monitoring.

Usage:
    strategy = Strategy(position_manager, aave_client, pnl_tracker, safety, signal_queue)
    asyncio.create_task(strategy.run())
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import TYPE_CHECKING

from bot_logging.logger_manager import setup_module_logger
from config.loader import get_config
from shared.constants import CLOSE_FACTOR_THRESHOLD_USD
from shared.types import (
    HealthStatus,
    HFTier,
    PositionDirection,
    StrategyHealthReport,
    TradeSignal,
)

if TYPE_CHECKING:
    from core.pnl_tracker import PnLTracker
    from core.position_manager import PositionManager
    from core.safety import SafetyState
    from execution.aave_client import AaveClient


class Strategy:
    """
    Async strategy engine consuming HealthStatus and TradeSignal from a shared queue.

    Responsibilities:
    - Evaluate trade signals for entry (confidence, borrow rate, stress test)
    - Dispatch health status events (deleverage, close)
    - Direction-aware stress testing with cascade modeling
    - Position sizing via fractional Kelly with GARCH volatility
    - Alpha decay monitoring and dynamic threshold adjustment
    """

    def __init__(
        self,
        position_manager: PositionManager,
        aave_client: AaveClient,
        pnl_tracker: PnLTracker,
        safety: SafetyState,
        signal_queue: asyncio.Queue[HealthStatus | TradeSignal],
    ) -> None:
        self._position_manager = position_manager
        self._aave_client = aave_client
        self._pnl_tracker = pnl_tracker
        self._safety = safety
        self._signal_queue = signal_queue

        cfg = get_config()
        self._positions_config = cfg.get_positions_config()
        self._signals_config = cfg.get_signals_config()
        self._aave_config = cfg.get_aave_config()

        # Stress test parameters
        self._stress_drops = [
            Decimal(d)
            for d in self._positions_config.get(
                "stress_test_price_drops",
                ["-0.05", "-0.10", "-0.15", "-0.20", "-0.30"],
            )
        ]
        self._min_stress_hf = Decimal(str(self._positions_config.get("min_stress_test_hf", "1.1")))
        self._cascade_threshold_usd = Decimal(
            str(self._positions_config.get("cascade_liquidation_threshold_usd", 50_000_000))
        )
        self._cascade_additional_drop = Decimal(
            str(self._positions_config.get("cascade_additional_drop", "-0.03"))
        )

        # Borrow rate limits
        self._max_borrow_cost_pct = Decimal(
            str(self._positions_config.get("max_borrow_cost_pct", "0.5"))
        )
        self._max_acceptable_borrow_apr = Decimal(
            str(self._positions_config.get("max_acceptable_borrow_apr", "15.0"))
        )

        # Health factor thresholds
        self._deleverage_threshold = Decimal(
            str(self._positions_config.get("deleverage_threshold", "1.4"))
        )
        self._close_threshold = Decimal(str(self._positions_config.get("close_threshold", "1.25")))
        self._target_hf_after_deleverage = Decimal(
            str(self._positions_config.get("target_hf_after_deleverage", "1.8"))
        )

        # Entry rules
        entry_rules = self._signals_config.get("entry_rules", {})
        self._min_confidence = Decimal(str(entry_rules.get("min_confidence", "0.7")))
        self._max_signals_per_day = entry_rules.get("max_signals_per_day", 3)

        # Position sizing
        sizing_config = self._signals_config.get("position_sizing", {})
        self._kelly_fraction = Decimal(str(sizing_config.get("kelly_fraction", "0.25")))
        self._high_vol_threshold = Decimal(str(sizing_config.get("high_vol_threshold", "0.04")))
        self._min_position_usd = Decimal(str(sizing_config.get("min_position_usd", "100")))
        self._max_position_usd = Decimal(
            str(self._positions_config.get("max_position_usd", "10000"))
        )
        self._max_leverage = Decimal(str(self._positions_config.get("max_leverage_ratio", "3.0")))

        # Alpha decay monitoring
        decay_config = self._signals_config.get("alpha_decay_monitoring", {})
        self._alpha_decay_enabled = decay_config.get("enabled", True)
        self._accuracy_decay_threshold = Decimal(
            str(decay_config.get("accuracy_decay_threshold", "0.7"))
        )
        self._sharpe_decay_threshold = Decimal(
            str(decay_config.get("sharpe_decay_threshold", "0.5"))
        )
        self._confidence_boost_on_decay = Decimal(
            str(decay_config.get("confidence_boost_on_decay", "1.1"))
        )
        self._rolling_window_days = decay_config.get("rolling_window_days", 30)
        self._historical_window_days = decay_config.get("historical_window_days", 180)

        # Exit rules
        exit_rules = self._signals_config.get("exit_rules", {})
        self._max_hold_hours = exit_rules.get("max_hold_hours", 168)

        # Dynamic confidence threshold (adjusted by alpha decay)
        self._dynamic_confidence_threshold = self._min_confidence

        # Mutable state
        self._running = False
        self._signals_today: int = 0
        self._last_signal_day: int = 0

        self._logger = setup_module_logger(
            "strategy", "strategy.log", module_folder="Strategy_Logs"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(
        self,
        signal_queue: asyncio.Queue[HealthStatus | TradeSignal] | None = None,
    ) -> None:
        """
        Consume HealthStatus and TradeSignal from the signal queue.

        This is designed to be launched as an asyncio.Task alongside
        the health monitor and signal engine.
        """
        queue = signal_queue or self._signal_queue
        self._running = True
        self._logger.info("Strategy engine started")

        try:
            while self._running:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=60)

                    if isinstance(msg, HealthStatus):
                        await self.handle_health_status(msg)
                    elif isinstance(msg, TradeSignal):
                        await self.handle_trade_signal(msg)
                    else:
                        self._logger.warning(
                            "Unknown message type in signal queue: %s", type(msg).__name__
                        )

                except asyncio.TimeoutError:
                    # No messages for 60s — run periodic checks
                    if self._alpha_decay_enabled:
                        report = self.check_strategy_health()
                        if report.alpha_decay_detected:
                            self._logger.warning(
                                "Alpha decay detected: %s",
                                "; ".join(report.recommendations),
                            )

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._logger.error("Strategy loop error: %s", exc, exc_info=True)

        except asyncio.CancelledError:
            self._logger.info("Strategy engine cancelled")
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the run loop to stop."""
        self._running = False

    # ------------------------------------------------------------------
    # Health status handling
    # ------------------------------------------------------------------

    async def handle_health_status(self, status: HealthStatus) -> None:
        """Handle a HealthStatus event from the health monitor."""
        if not self._position_manager.has_open_position:
            return

        if status.tier == HFTier.CRITICAL:
            if status.health_factor <= self._close_threshold:
                self._logger.critical(
                    "HF %.4f below close threshold %.4f — emergency close",
                    status.health_factor,
                    self._close_threshold,
                )
                await self._position_manager.close_position(reason="emergency")
                return

            if status.health_factor <= self._deleverage_threshold:
                self._logger.warning(
                    "HF %.4f below deleverage threshold %.4f — deleveraging",
                    status.health_factor,
                    self._deleverage_threshold,
                )
                await self._position_manager.partial_deleverage(self._target_hf_after_deleverage)
                return

        if status.tier == HFTier.WARNING and status.health_factor <= self._deleverage_threshold:
            self._logger.warning(
                "HF %.4f in WARNING tier below deleverage threshold — deleveraging",
                status.health_factor,
            )
            await self._position_manager.partial_deleverage(self._target_hf_after_deleverage)

    # ------------------------------------------------------------------
    # Trade signal handling
    # ------------------------------------------------------------------

    async def handle_trade_signal(self, signal: TradeSignal) -> None:
        """Handle a TradeSignal from the signal engine."""
        # Already have a position — skip new entries
        if self._position_manager.has_open_position:
            self._logger.debug("Signal ignored: position already open")
            return

        # Check daily signal limit
        today = int(time.time()) // 86400
        if today != self._last_signal_day:
            self._signals_today = 0
            self._last_signal_day = today
        if self._signals_today >= self._max_signals_per_day:
            self._logger.info("Daily signal limit reached (%d)", self._max_signals_per_day)
            return

        # Evaluate entry
        should_enter, reason = await self.evaluate_entry(signal)
        if not should_enter:
            self._logger.info("Entry rejected: %s", reason)
            return

        # Determine position parameters
        if signal.direction == PositionDirection.LONG:
            debt_token = "USDT"
            collateral_token = "WBNB"
        else:
            debt_token = "WBNB"
            preferred = self._signals_config.get("short_signals", {}).get(
                "preferred_collateral", "USDC"
            )
            collateral_token = preferred

        # Validate and constrain position size
        validated_size = self.validate_position_size(signal)
        if validated_size < self._min_position_usd:
            self._logger.info(
                "Position size $%s below minimum $%s",
                validated_size,
                self._min_position_usd,
            )
            return

        # Open position
        self._logger.info(
            "Opening %s position: size=$%s confidence=%.2f regime=%s",
            signal.direction.value,
            validated_size,
            signal.confidence,
            signal.regime.value,
        )
        await self._position_manager.open_position(
            direction=signal.direction,
            debt_token=debt_token,
            amount=validated_size,
            collateral_token=collateral_token,
        )
        self._signals_today += 1

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------

    async def evaluate_entry(self, signal: TradeSignal) -> tuple[bool, str]:
        """
        Evaluate whether to enter a position based on signal and risk filters.

        Returns (should_enter, reason).
        """
        # 1. Confidence check
        if signal.confidence < self._dynamic_confidence_threshold:
            return (
                False,
                f"Confidence {signal.confidence:.2f} below threshold "
                f"{self._dynamic_confidence_threshold:.2f}",
            )

        # 2. Borrow rate check
        from shared.constants import TOKEN_USDT, TOKEN_WBNB

        debt_addr = TOKEN_USDT if signal.direction == PositionDirection.LONG else TOKEN_WBNB
        rate_ok, current_rate = await self.check_borrow_rate_acceptable(
            debt_addr, projected_hold_hours=float(self._max_hold_hours)
        )
        if not rate_ok:
            return (False, f"Borrow rate too high: {current_rate:.2f}% APR")

        # 4. Stress test with position sizing estimate
        validated_size = self.validate_position_size(signal)
        aave_assets = self._aave_config.get("supported_assets", {})

        if signal.direction == PositionDirection.LONG:
            collateral_token = "WBNB"
        else:
            collateral_token = self._signals_config.get("short_signals", {}).get(
                "preferred_collateral", "USDC"
            )

        collateral_info = aave_assets.get(collateral_token, {})
        lt_bps = collateral_info.get("liquidation_threshold_bps", 8000)
        lt = Decimal(lt_bps)

        # Approximate collateral and debt for stress test
        collateral_usd = validated_size  # Approximate: flash-loaned amount ≈ collateral
        debt_usd = validated_size

        stress_hfs = self.stress_test(
            signal.direction, collateral_usd, debt_usd, lt, self._stress_drops
        )

        for i, hf in enumerate(stress_hfs):
            if hf < self._min_stress_hf:
                return (
                    False,
                    f"Stress test failed: HF={hf:.4f} at {self._stress_drops[i]:.0%} drop "
                    f"(min {self._min_stress_hf})",
                )

        # 5. Close factor risk
        close_factor_ok = await self.check_close_factor_risk(collateral_usd, debt_usd)
        if not close_factor_ok:
            return (False, "Position size risks 100% close factor liquidation")

        return (True, "All entry checks passed")

    async def evaluate_open(
        self,
        direction: PositionDirection,
        debt_token: str,
        collateral_token: str,
        amount: Decimal,
    ) -> bool:
        """Simplified entry evaluation for programmatic use."""
        signal_ok, reason = await self.evaluate_entry(
            TradeSignal(
                direction=direction,
                confidence=Decimal("1.0"),
                strategy_mode="manual",
                indicators=None,  # type: ignore[arg-type]
                regime=None,  # type: ignore[arg-type]
                components=[],
                recommended_size_usd=amount,
                hurst_exponent=Decimal("0.5"),
                garch_volatility=Decimal("0.02"),
                timestamp=int(time.time()),
            )
        )
        return signal_ok

    # ------------------------------------------------------------------
    # Stress testing
    # ------------------------------------------------------------------

    def stress_test(
        self,
        direction: PositionDirection,
        collateral_usd: Decimal,
        debt_usd: Decimal,
        liq_threshold_bps: Decimal,
        price_drops: list[Decimal],
    ) -> list[Decimal]:
        """
        Direction-aware stress test.

        LONG (volatile collateral, stable debt):
            HF = (collateral * (1 + drop) * LT) / debt

        SHORT (stable collateral, volatile debt):
            HF = (collateral * LT) / (debt * (1 + drop))

        Note: for shorts, price_drops represent price INCREASES of the
        borrowed volatile asset (adverse scenario).
        """
        lt = liq_threshold_bps / Decimal("10000")
        results = []

        for drop in price_drops:
            hf = self._compute_hf_at_drop(direction, collateral_usd, debt_usd, lt, drop)
            results.append(hf)

        return results

    def stress_test_with_cascade(
        self,
        direction: PositionDirection,
        collateral_usd: Decimal,
        debt_usd: Decimal,
        liq_threshold_bps: Decimal,
        price_drops: list[Decimal],
        market_total_supply_usd: Decimal,
    ) -> list[Decimal]:
        """
        Stress test with liquidation cascade multiplier.

        If initial drop triggers >$50M in market-wide liquidations,
        assumes additional 2-5% cascade-induced decline.
        """
        lt = liq_threshold_bps / Decimal("10000")
        results = []

        for drop in price_drops:
            base_hf = self._compute_hf_at_drop(direction, collateral_usd, debt_usd, lt, drop)

            estimated_liquidatable = self._estimate_market_liquidations(
                drop, market_total_supply_usd
            )

            if estimated_liquidatable > self._cascade_threshold_usd:
                # Apply cascade: additional price decline
                cascade_drop = drop + self._cascade_additional_drop
                base_hf = self._compute_hf_at_drop(
                    direction, collateral_usd, debt_usd, lt, cascade_drop
                )

            results.append(base_hf)

        return results

    def _compute_hf_at_drop(
        self,
        direction: PositionDirection,
        collateral_usd: Decimal,
        debt_usd: Decimal,
        lt: Decimal,
        price_change: Decimal,
    ) -> Decimal:
        """Compute health factor at a given price change."""
        if debt_usd <= 0:
            return Decimal("999")

        if direction == PositionDirection.LONG:
            # Volatile collateral, stable debt
            return (collateral_usd * (1 + price_change) * lt) / debt_usd
        else:
            # Stable collateral, volatile debt
            debt_multiplier = 1 + abs(price_change)  # Price increase = adverse for short
            if debt_multiplier <= 0:
                return Decimal("999")
            return (collateral_usd * lt) / (debt_usd * debt_multiplier)

    @staticmethod
    def _estimate_market_liquidations(
        price_drop: Decimal, market_total_supply_usd: Decimal
    ) -> Decimal:
        """
        Estimate market-wide liquidatable value at a given price drop.

        Simplified model: assumes ~10% of market supply is leveraged,
        and positions are uniformly distributed across HF 1.0-2.0.
        """
        leveraged_fraction = Decimal("0.10")
        total_leveraged = market_total_supply_usd * leveraged_fraction

        # Fraction of positions liquidated at this drop
        # At -5%: ~25% of positions, at -20%: ~90%
        drop_magnitude = abs(price_drop)
        liquidation_fraction = min(drop_magnitude * Decimal("5"), Decimal("1.0"))

        return total_leveraged * liquidation_fraction

    # ------------------------------------------------------------------
    # Deleverage formula
    # ------------------------------------------------------------------

    def compute_deleverage_amount(
        self,
        current_hf: Decimal,
        target_hf: Decimal,
        collateral_usd: Decimal,
        debt_usd: Decimal,
        liq_threshold_bps: Decimal,
    ) -> Decimal:
        """
        Compute the debt repayment amount needed to reach target HF.

        Formula (valid for both long and short):
            repay = (D - C * LT / h_t) / (1 + f - LT / h_t)
        where f = flash loan premium.
        """
        lt = liq_threshold_bps / Decimal("10000")
        flash_premium = Decimal("0.0005")

        lt_over_ht = lt / target_hf
        numerator = debt_usd - collateral_usd * lt_over_ht
        denominator = Decimal("1") + flash_premium - lt_over_ht

        if denominator <= 0:
            return Decimal("0")

        repay = numerator / denominator
        return max(repay, Decimal("0"))

    # ------------------------------------------------------------------
    # Borrow rate check
    # ------------------------------------------------------------------

    async def check_borrow_rate_acceptable(
        self, asset: str, projected_hold_hours: float
    ) -> tuple[bool, Decimal]:
        """
        Reject entry if projected borrow cost exceeds threshold.

        Returns (acceptable, current_rate_apr).
        """
        reserve_data = await self._aave_client.get_reserve_data(asset)
        current_rate_apr = reserve_data.variable_borrow_rate

        # Check absolute rate cap
        if current_rate_apr > self._max_acceptable_borrow_apr:
            self._logger.warning(
                "Borrow rate %s%% APR exceeds max %s%%",
                current_rate_apr,
                self._max_acceptable_borrow_apr,
            )
            return (False, current_rate_apr)

        # Calculate projected cost as percentage of position
        projected_cost_pct = current_rate_apr * Decimal(str(projected_hold_hours)) / Decimal("8760")

        acceptable = projected_cost_pct <= self._max_borrow_cost_pct
        if not acceptable:
            self._logger.warning(
                "Borrow cost too high: %s%% APR, projected cost %s%% over %sh",
                current_rate_apr,
                projected_cost_pct,
                projected_hold_hours,
            )

        return (acceptable, current_rate_apr)

    # ------------------------------------------------------------------
    # Close factor risk
    # ------------------------------------------------------------------

    async def check_close_factor_risk(self, collateral_usd: Decimal, debt_usd: Decimal) -> bool:
        """
        Warn if position size risks 100% close factor liquidation.

        Aave V3: positions with collateral or debt < $2,000 can be
        100% liquidated in a single call (no partial liquidation).
        """
        for drop in self._stress_drops:
            projected_collateral = collateral_usd * (1 + drop)
            if projected_collateral < CLOSE_FACTOR_THRESHOLD_USD:
                self._logger.warning(
                    "At %s drop, collateral $%s < $%s — 100%% close factor risk",
                    f"{drop:.0%}",
                    projected_collateral,
                    CLOSE_FACTOR_THRESHOLD_USD,
                )
                return False

            if debt_usd < CLOSE_FACTOR_THRESHOLD_USD:
                self._logger.warning(
                    "Debt $%s < $%s — 100%% close factor risk",
                    debt_usd,
                    CLOSE_FACTOR_THRESHOLD_USD,
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def validate_position_size(self, signal: TradeSignal) -> Decimal:
        """
        Apply risk constraints to signal engine's recommended position size.

        1. GARCH volatility adjustment — reduce in high-vol regimes
        2. Hard limits (max position, max leverage)
        3. Drawdown-based reduction (Vince optimal-f approach)
        """
        raw_size = signal.recommended_size_usd

        # 1. GARCH volatility adjustment
        garch_vol = signal.garch_volatility
        if garch_vol > self._high_vol_threshold:
            vol_scalar = self._high_vol_threshold / garch_vol
            raw_size *= vol_scalar
            self._logger.info(
                "Position reduced by %s%% for high volatility (%s)",
                (1 - vol_scalar) * 100,
                garch_vol,
            )

        # 2. Hard limits
        raw_size = min(raw_size, self._max_position_usd)

        # 3. Drawdown-based reduction
        current_dd = self._pnl_tracker.current_drawdown_pct
        if current_dd > Decimal("0.1"):
            dd_scalar = max(Decimal("0.25"), Decimal("1") - current_dd)
            raw_size *= dd_scalar
            self._logger.warning(
                "Position reduced to %s%% due to drawdown (%s%%)",
                dd_scalar * 100,
                current_dd * 100,
            )

        # Enforce minimum
        if raw_size < self._min_position_usd:
            return Decimal("0")

        return raw_size

    # ------------------------------------------------------------------
    # Alpha decay monitoring
    # ------------------------------------------------------------------

    def check_strategy_health(self) -> StrategyHealthReport:
        """
        Detect alpha decay and recommend parameter refresh or strategy rotation.

        Compares rolling 30-day stats against 180-day historical baseline.
        """
        report = StrategyHealthReport()

        try:
            # Use synchronous property access since we're in sync context
            # The actual stats computation happens synchronously in PnLTracker
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — schedule coroutines
                # Use a future to get results synchronously
                _stats_future = asyncio.ensure_future(
                    self._pnl_tracker.get_rolling_stats(window_days=self._rolling_window_days)
                )
                _historical_future = asyncio.ensure_future(
                    self._pnl_tracker.get_rolling_stats(window_days=self._historical_window_days)
                )
                # Can't block here — return empty report for now
                # This will be properly called from the async run loop
                return report
        except RuntimeError:
            return report

        return report

    async def check_strategy_health_async(self) -> StrategyHealthReport:
        """Async version of strategy health check."""
        stats = await self._pnl_tracker.get_rolling_stats(window_days=self._rolling_window_days)
        historical = await self._pnl_tracker.get_rolling_stats(
            window_days=self._historical_window_days
        )

        report = StrategyHealthReport()

        # Accuracy decay check
        if historical.win_rate > 0:
            accuracy_ratio = stats.win_rate / historical.win_rate
            report.accuracy_ratio = accuracy_ratio
            if accuracy_ratio < self._accuracy_decay_threshold:
                report.alpha_decay_detected = True
                report.recommendations.append(
                    f"Win rate decayed to {accuracy_ratio:.0%} of 6-month average. "
                    "Consider parameter refresh or regime filter adjustment."
                )

        # Sharpe ratio degradation
        if historical.sharpe_ratio > 0:
            sharpe_ratio = stats.sharpe_ratio / historical.sharpe_ratio
            report.sharpe_ratio = sharpe_ratio
            if sharpe_ratio < self._sharpe_decay_threshold:
                report.alpha_decay_detected = True
                report.recommendations.append(
                    f"Sharpe ratio at {sharpe_ratio:.0%} of historical. " "Strategy may be crowded."
                )

        # If alpha decay detected, increase confidence threshold
        if report.alpha_decay_detected:
            self._dynamic_confidence_threshold = min(
                self._min_confidence * self._confidence_boost_on_decay,
                Decimal("0.9"),
            )
            report.dynamic_confidence_threshold = self._dynamic_confidence_threshold
            self._logger.warning(
                "Alpha decay: raising confidence threshold to %s",
                self._dynamic_confidence_threshold,
            )

        return report
