"""
Health factor monitoring loop for BSC Leverage Bot.

Tiered HF polling: adjusts poll frequency based on health factor tier.
Validates Chainlink oracle freshness and predicts HF degradation via
Aave's compound interest model (3-term Taylor series matching MathUtils.sol).

Usage:
    monitor = HealthMonitor(aave_client, safety, user_address, signal_queue)
    asyncio.create_task(monitor.run())
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import TYPE_CHECKING

from web3 import Web3

from bot_logging.logger_manager import setup_module_logger
from config.loader import get_config
from shared.constants import RAY, SECONDS_PER_YEAR
from shared.types import (
    BorrowRateInfo,
    HealthStatus,
    HFTier,
    PositionState,
)

if TYPE_CHECKING:
    from core.safety import SafetyState
    from execution.aave_client import AaveClient


class HealthMonitor:
    """
    Async health factor polling loop with tier-based frequency.

    Tiers:
        SAFE     (HF > 2.0)  — poll every 15s
        WATCH    (1.5-2.0)   — poll every 5s
        WARNING  (1.3-1.5)   — poll every 2s
        CRITICAL (< 1.3)     — poll every 1s
    """

    def __init__(
        self,
        aave_client: AaveClient,
        safety: SafetyState,
        user_address: str,
        signal_queue: asyncio.Queue,
    ) -> None:
        self._aave_client = aave_client
        self._safety = safety
        self._user_address = user_address
        self._signal_queue = signal_queue

        cfg = get_config()
        timing_cfg = cfg.get_timing_config().get("health_monitoring", {})
        positions_cfg = cfg.get_positions_config()
        chain_cfg = cfg.get_chain_config(56)

        # Poll intervals per tier
        self._safe_interval: float = timing_cfg.get("safe_interval_seconds", 15)
        self._watch_interval: float = timing_cfg.get("watch_interval_seconds", 5)
        self._warning_interval: float = timing_cfg.get("warning_interval_seconds", 2)
        self._critical_interval: float = timing_cfg.get("critical_interval_seconds", 1)
        self._stale_data_threshold: int = timing_cfg.get("stale_data_threshold_failures", 5)

        # Oracle staleness threshold
        self._max_staleness_seconds: int = positions_cfg.get(
            "oracle_max_staleness_seconds", 60
        )

        # Build Chainlink feed contracts for oracle freshness checks
        chainlink_abi = cfg.get_abi("chainlink_aggregator_v3")
        self._chainlink_feeds: dict[str, dict] = {}
        for feed_name, feed_info in chain_cfg.get("chainlink_feeds", {}).items():
            contract = aave_client._w3.eth.contract(
                address=Web3.to_checksum_address(feed_info["address"]),
                abi=chainlink_abi,
            )
            self._chainlink_feeds[feed_name] = {
                "contract": contract,
                "address": feed_info["address"],
                "heartbeat_seconds": feed_info.get("heartbeat_seconds", 60),
            }

        # Mutable state
        self._consecutive_failures: int = 0
        self._current_tier: HFTier = HFTier.SAFE
        self._running: bool = False

        self._logger = setup_module_logger(
            "health_monitor", "health_monitor.log", module_folder="Health_Monitor_Logs"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main polling loop — designed to be launched as an asyncio.Task."""
        self._running = True
        self._logger.info("Health monitor started for %s", self._user_address)
        try:
            while self._running:
                try:
                    status = await self._poll_once()
                    self._consecutive_failures = 0

                    await self._signal_queue.put(status)

                    # Log tier transitions
                    if status.tier != self._current_tier:
                        self._logger.warning(
                            "Tier transition: %s -> %s (HF=%.4f)",
                            self._current_tier.value,
                            status.tier.value,
                            status.health_factor,
                        )
                    self._current_tier = status.tier

                    interval = self._get_poll_interval(status.tier)
                    self._logger.debug(
                        "HF=%.4f tier=%s next_poll=%.1fs",
                        status.health_factor,
                        status.tier.value,
                        interval,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._consecutive_failures += 1
                    self._logger.error(
                        "Poll failed (%d/%d): %s",
                        self._consecutive_failures,
                        self._stale_data_threshold,
                        exc,
                    )
                    if self._consecutive_failures >= self._stale_data_threshold:
                        self._safety.trigger_global_pause(
                            f"Health monitor: {self._consecutive_failures} consecutive "
                            f"poll failures — last error: {exc}"
                        )
                    interval = self._get_poll_interval(self._current_tier)

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            self._logger.info("Health monitor cancelled")
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the run loop to stop."""
        self._running = False

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_once(self) -> HealthStatus:
        """Execute a single poll cycle: read HF, validate oracle, return status."""
        account = await self._aave_client.get_user_account_data(self._user_address)
        tier = self._determine_tier(account.health_factor)

        # Validate oracle freshness for all configured feeds
        for feed_name, feed_info in self._chainlink_feeds.items():
            is_fresh = await self.check_oracle_freshness(feed_info["address"])
            if not is_fresh:
                self._logger.warning(
                    "Oracle %s is stale — pause already triggered", feed_name
                )

        return HealthStatus(
            health_factor=account.health_factor,
            tier=tier,
            collateral_usd=account.total_collateral_usd,
            debt_usd=account.total_debt_usd,
            timestamp=int(time.time()),
        )

    # ------------------------------------------------------------------
    # Tier classification
    # ------------------------------------------------------------------

    def _determine_tier(self, hf: Decimal) -> HFTier:
        """Classify health factor into a monitoring tier (pure function)."""
        if hf > Decimal("2.0"):
            return HFTier.SAFE
        if hf >= Decimal("1.5"):
            return HFTier.WATCH
        if hf >= Decimal("1.3"):
            return HFTier.WARNING
        return HFTier.CRITICAL

    def _get_poll_interval(self, tier: HFTier) -> float:
        """Return poll interval in seconds for the given tier."""
        intervals = {
            HFTier.SAFE: self._safe_interval,
            HFTier.WATCH: self._watch_interval,
            HFTier.WARNING: self._warning_interval,
            HFTier.CRITICAL: self._critical_interval,
        }
        return intervals.get(tier, self._safe_interval)

    # ------------------------------------------------------------------
    # Oracle freshness
    # ------------------------------------------------------------------

    async def check_oracle_freshness(self, feed_address: str) -> bool:
        """
        Validate Chainlink oracle freshness via latestRoundData.

        Returns False and triggers global pause if data is stale.
        Logs warning (but does NOT pause) for incomplete rounds.
        """
        # Find the matching configured feed
        feed_contract = None
        heartbeat = self._max_staleness_seconds
        for feed_info in self._chainlink_feeds.values():
            if feed_info["address"].lower() == feed_address.lower():
                feed_contract = feed_info["contract"]
                heartbeat = feed_info.get("heartbeat_seconds", self._max_staleness_seconds)
                break

        if feed_contract is None:
            # Address not in configured feeds — create ad-hoc contract
            chainlink_abi = get_config().get_abi("chainlink_aggregator_v3")
            feed_contract = self._aave_client._w3.eth.contract(
                address=Web3.to_checksum_address(feed_address),
                abi=chainlink_abi,
            )

        result = await feed_contract.functions.latestRoundData().call()
        round_id, _answer, _started_at, updated_at, answered_in_round = result

        # Check staleness
        staleness = time.time() - updated_at
        max_allowed = max(heartbeat, self._max_staleness_seconds)
        if staleness > max_allowed:
            self._logger.critical(
                "Oracle %s stale: %.0fs since update (max %ds)",
                feed_address,
                staleness,
                max_allowed,
            )
            self._safety.trigger_global_pause(
                f"Chainlink oracle {feed_address} stale by {staleness:.0f}s"
            )
            return False

        # Check incomplete round (warning only — transient condition)
        if answered_in_round < round_id:
            self._logger.warning(
                "Oracle %s incomplete round: answeredInRound=%d < roundId=%d",
                feed_address,
                answered_in_round,
                round_id,
            )

        return True

    # ------------------------------------------------------------------
    # HF prediction
    # ------------------------------------------------------------------

    def predict_hf_at(self, seconds_ahead: int, position: PositionState) -> Decimal:
        """
        Predict health factor at t+seconds_ahead using compound interest.

        Uses Aave's 3-term Taylor series (matches on-chain MathUtils.sol):
            compound = 1 + r*dt + (r*dt)^2 / 2
        where r = borrow_rate_ray / RAY / SECONDS_PER_YEAR.
        """
        if position.debt_usd <= 0:
            return position.health_factor

        rate_per_second = position.borrow_rate_ray / RAY / SECONDS_PER_YEAR
        dt = Decimal(seconds_ahead)

        # 3-term Taylor approximation of e^(r*dt)
        r_dt = rate_per_second * dt
        compound = Decimal("1") + r_dt + (r_dt ** 2) / Decimal("2")

        projected_debt = position.debt_usd * compound

        if projected_debt <= 0:
            return position.health_factor

        return (position.collateral_usd * position.liquidation_threshold) / projected_debt

    # ------------------------------------------------------------------
    # Borrow rate
    # ------------------------------------------------------------------

    async def get_borrow_rate(self, asset: str) -> BorrowRateInfo:
        """Get current borrow rate and utilization for early warning of rate spikes."""
        reserve = await self._aave_client.get_reserve_data(asset)
        return BorrowRateInfo(
            variable_rate_apr=reserve.variable_borrow_rate,
            utilization_rate=reserve.utilization_rate,
            optimal_utilization=Decimal("0.8"),  # Aave V3 BSC default
            rate_at_kink=Decimal("0"),  # Would need interest rate strategy contract
        )
