"""
Safety gate-keeper for BSC Leverage Bot.

Centralized kill switches checked before every action (entry, exit,
deleverage, tx submission). Default-to-safe: if config is missing
or corrupt, defaults to dry_run=True, max_position=0.

Usage:
    from core.safety import SafetyState

    safety = SafetyState()
    check = safety.can_open_position(amount_usd=Decimal("5000"), leverage=Decimal("2.0"))
    if not check.can_proceed:
        print(f"Blocked: {check.reason}")
"""

from __future__ import annotations

import time
from collections import deque
from decimal import Decimal
from pathlib import Path

from bot_logging.logger_manager import setup_module_logger
from config.loader import get_config
from shared.constants import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_DRY_RUN,
    DEFAULT_MAX_GAS_PRICE_GWEI,
    DEFAULT_MAX_LEVERAGE_RATIO,
    DEFAULT_MAX_POSITION_USD,
    DEFAULT_MAX_TX_PER_24H,
)
from shared.types import SafetyCheck

_PROJECT_ROOT = Path(__file__).parent.parent
_SENTINEL_FILE = _PROJECT_ROOT / "PAUSE"

_SECONDS_PER_DAY = 86400


class SafetyState:
    """
    Centralized safety controls for all trading actions.

    Two tiers of defaults:
    - Config present but key missing: use DEFAULT_* constants (permissive operational defaults)
    - Config file missing/empty: lockdown (dry_run=True, max_position=0, max_leverage=1.0)
    """

    def __init__(self) -> None:
        cfg = get_config().get_positions_config()

        if cfg:
            # Config present — use operational defaults for missing keys
            self._dry_run: bool = cfg.get("dry_run", DEFAULT_DRY_RUN)
            self._max_position_usd = Decimal(
                str(cfg.get("max_position_usd", DEFAULT_MAX_POSITION_USD))
            )
            self._max_leverage_ratio = Decimal(
                str(cfg.get("max_leverage_ratio", DEFAULT_MAX_LEVERAGE_RATIO))
            )
            self._max_gas_price_gwei: int = cfg.get(
                "max_gas_price_gwei", DEFAULT_MAX_GAS_PRICE_GWEI
            )
            self._cooldown_seconds: int = cfg.get(
                "cooldown_between_actions_seconds", DEFAULT_COOLDOWN_SECONDS
            )
            self._max_tx_per_24h: int = cfg.get(
                "max_transactions_per_24h", DEFAULT_MAX_TX_PER_24H
            )
        else:
            # Config missing/corrupt — lockdown mode
            self._dry_run = True
            self._max_position_usd = Decimal("0")
            self._max_leverage_ratio = Decimal("1.0")
            self._max_gas_price_gwei = 0
            self._cooldown_seconds = DEFAULT_COOLDOWN_SECONDS
            self._max_tx_per_24h = 0

        # Mutable state
        self._global_pause = False
        self._pause_reason = ""
        self._last_action_time: float = 0.0
        self._action_timestamps: deque[float] = deque()

        self._logger = setup_module_logger(
            "safety", "safety.log", module_folder="Safety_Logs"
        )
        self._logger.info(
            "SafetyState initialized: dry_run=%s max_position=$%s max_leverage=%sx "
            "max_gas=%d gwei cooldown=%ds max_tx_24h=%d",
            self._dry_run,
            self._max_position_usd,
            self._max_leverage_ratio,
            self._max_gas_price_gwei,
            self._cooldown_seconds,
            self._max_tx_per_24h,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        """Check if trading is globally paused (includes sentinel file)."""
        self.check_pause_sentinel()
        return self._global_pause

    @property
    def is_dry_run(self) -> bool:
        return self._dry_run

    # ------------------------------------------------------------------
    # Gate checks
    # ------------------------------------------------------------------

    def can_open_position(
        self, amount_usd: Decimal, leverage: Decimal
    ) -> SafetyCheck:
        """Check whether a new position can be opened."""
        if self.is_paused:
            return SafetyCheck(
                can_proceed=False,
                reason=f"Global pause active: {self._pause_reason}",
            )

        if self._dry_run:
            return SafetyCheck(can_proceed=False, reason="Dry run mode active")

        if amount_usd > self._max_position_usd:
            return SafetyCheck(
                can_proceed=False,
                reason=(
                    f"Position ${amount_usd} exceeds max ${self._max_position_usd}"
                ),
            )

        if leverage > self._max_leverage_ratio:
            return SafetyCheck(
                can_proceed=False,
                reason=(
                    f"Leverage {leverage}x exceeds max {self._max_leverage_ratio}x"
                ),
            )

        # Cooldown check (monotonic clock — immune to NTP adjustments)
        elapsed = time.monotonic() - self._last_action_time
        if self._last_action_time > 0 and elapsed < self._cooldown_seconds:
            remaining = self._cooldown_seconds - elapsed
            return SafetyCheck(
                can_proceed=False,
                reason=f"Cooldown active: {remaining:.0f}s remaining",
            )

        # 24h transaction rate limit (wall-clock time for calendar boundaries)
        self._prune_old_timestamps()
        if len(self._action_timestamps) >= self._max_tx_per_24h:
            return SafetyCheck(
                can_proceed=False,
                reason=(
                    f"24h tx limit reached: {len(self._action_timestamps)}"
                    f"/{self._max_tx_per_24h}"
                ),
            )

        return SafetyCheck(can_proceed=True, reason="All checks passed")

    def can_submit_tx(self, gas_price_gwei: int) -> SafetyCheck:
        """Check whether a transaction can be submitted at the given gas price."""
        if self.is_paused:
            return SafetyCheck(
                can_proceed=False,
                reason=f"Global pause active: {self._pause_reason}",
            )

        if gas_price_gwei > self._max_gas_price_gwei:
            return SafetyCheck(
                can_proceed=False,
                reason=(
                    f"Gas price {gas_price_gwei} gwei exceeds "
                    f"max {self._max_gas_price_gwei} gwei"
                ),
            )

        return SafetyCheck(can_proceed=True, reason="Gas price acceptable")

    # ------------------------------------------------------------------
    # State mutations
    # ------------------------------------------------------------------

    def record_action(self) -> None:
        """Record that an action was taken (for cooldown and rate limiting)."""
        self._last_action_time = time.monotonic()
        self._action_timestamps.append(time.time())
        self._logger.debug("Action recorded; 24h count: %d", len(self._action_timestamps))

    def trigger_global_pause(self, reason: str) -> None:
        """Activate the emergency kill switch."""
        self._global_pause = True
        self._pause_reason = reason
        self._logger.critical("GLOBAL PAUSE TRIGGERED: %s", reason)

    def resume(self) -> None:
        """Clear the global pause (manual recovery)."""
        self._global_pause = False
        self._pause_reason = ""
        self._logger.warning("Global pause CLEARED — manual resume invoked")

    def check_pause_sentinel(self) -> bool:
        """Check for PAUSE file in project root (emergency manual override)."""
        exists = _SENTINEL_FILE.exists()
        if exists and not self._global_pause:
            self.trigger_global_pause("PAUSE sentinel file detected")
        return exists

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_old_timestamps(self) -> None:
        """Remove action timestamps older than 24 hours."""
        cutoff = time.time() - _SECONDS_PER_DAY
        while self._action_timestamps and self._action_timestamps[0] < cutoff:
            self._action_timestamps.popleft()
