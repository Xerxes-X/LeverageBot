"""
Position lifecycle manager for BSC Leverage Bot.

Orchestrates position actions (open, close, deleverage, increase) by
composing aggregator quotes, Aave calldata, and tx submission. Handles
both LONG and SHORT positions using the same contract with parameterized
token roles.

Usage:
    pm = PositionManager(
        aave_client, aggregator_client, tx_submitter, safety, pnl_tracker, executor_address
    )
    state = await pm.open_position(PositionDirection.LONG, USDT, Decimal("5000"), WBNB)
    state = await pm.close_position()
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from web3 import Web3

from bot_logging.logger_manager import setup_module_logger
from config.loader import get_config
from shared.constants import (
    AAVE_V3_POOL,
    DEFAULT_FLASH_LOAN_PREMIUM,
    FLASH_LOAN_MODE_NO_DEBT,
    FLASH_LOAN_MODE_VARIABLE_DEBT,
    WAD,
)
from shared.types import PositionDirection, PositionState, SwapQuote

if TYPE_CHECKING:
    from core.pnl_tracker import PnLTracker
    from core.safety import SafetyState
    from execution.aave_client import AaveClient
    from execution.aggregator_client import AggregatorClient
    from execution.tx_submitter import TxSubmitter


class PositionManagerError(Exception):
    """Raised when a position operation fails."""


class PositionManager:
    """
    Direction-aware position lifecycle manager.

    Handles both LONG (volatile collateral, stable debt) and SHORT
    (stable collateral, volatile debt) positions through the same
    LeverageExecutor contract with parameterized token roles.
    """

    def __init__(
        self,
        aave_client: AaveClient,
        aggregator_client: AggregatorClient,
        tx_submitter: TxSubmitter,
        safety: SafetyState,
        pnl_tracker: PnLTracker,
        executor_address: str,
        user_address: str,
    ) -> None:
        self._aave_client = aave_client
        self._aggregator_client = aggregator_client
        self._tx_submitter = tx_submitter
        self._safety = safety
        self._pnl_tracker = pnl_tracker
        self._executor_address = Web3.to_checksum_address(executor_address)
        self._user_address = Web3.to_checksum_address(user_address)

        cfg = get_config()
        self._aave_config = cfg.get_aave_config()
        self._positions_config = cfg.get_positions_config()
        self._chain_config = cfg.get_chain_config(56)

        # Token metadata from chain config
        self._tokens = self._chain_config.get("tokens", {})

        # Current position state (single position at a time)
        self._current_position: PositionState | None = None
        self._current_position_id: int | None = None

        self._logger = setup_module_logger(
            "position_manager",
            "position_manager.log",
            module_folder="Position_Manager_Logs",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_open_position(self) -> bool:
        return self._current_position is not None

    @property
    def current_position(self) -> PositionState | None:
        return self._current_position

    # ------------------------------------------------------------------
    # Open position
    # ------------------------------------------------------------------

    async def open_position(
        self,
        direction: PositionDirection,
        debt_token: str,
        amount: Decimal,
        collateral_token: str,
    ) -> PositionState:
        """
        Open a new leveraged position.

        LONG: flash loan debt_token (stable) → swap to collateral_token (volatile) → supply
        SHORT: flash loan debt_token (volatile) → swap to collateral_token (stable) → supply

        Args:
            direction: LONG or SHORT.
            debt_token: Symbol of the token to borrow (e.g. "USDT" for long, "WBNB" for short).
            amount: Amount to flash-borrow in USD terms.
            collateral_token: Symbol of the collateral (e.g. "WBNB" for long, "USDC" for short).

        Returns:
            PositionState with post-execution data.
        """
        if self.has_open_position:
            raise PositionManagerError("Cannot open: position already open")

        # 1. Isolation mode check (SHORT only)
        if direction == PositionDirection.SHORT:
            collateral_addr = self._get_token_address(collateral_token)
            iso_ok, iso_msg = await self._check_isolation_mode(collateral_addr)
            if not iso_ok:
                raise PositionManagerError(f"Isolation mode check failed: {iso_msg}")
            self._logger.info("Isolation mode check passed: %s", iso_msg)

        # 2. Resolve token addresses and decimals
        debt_addr = self._get_token_address(debt_token)
        collateral_addr = self._get_token_address(collateral_token)
        debt_decimals = self._get_token_decimals(debt_token)
        collateral_decimals = self._get_token_decimals(collateral_token)

        # Convert amount to native token units
        amount_native = int(amount * Decimal(10**debt_decimals))

        # 3. Get best swap quote (debt → collateral)
        quote = await self._aggregator_client.get_best_quote(
            from_token=debt_addr,
            to_token=collateral_addr,
            from_amount=amount_native,
            from_decimals=debt_decimals,
            to_decimals=collateral_decimals,
        )

        # 4. Validate DEX-Oracle divergence (already done inside get_best_quote)
        self._logger.info(
            "Best quote: provider=%s to_amount=%s",
            quote.provider,
            quote.to_amount,
        )

        # 5. Encode flash loan calldata
        # Encode swap params for the executor contract
        params = Web3.solidity_keccak(
            ["address", "bytes", "uint8"],
            [quote.router_address, quote.calldata, 0 if direction == PositionDirection.LONG else 1],
        )

        flash_loan_calldata = self._aave_client.encode_flash_loan(
            receiver=self._executor_address,
            assets=[debt_addr],
            amounts=[amount_native],
            modes=[FLASH_LOAN_MODE_VARIABLE_DEBT],
            on_behalf_of=self._executor_address,
            params=params,
        )

        # 6. Safety gate
        check = self._safety.can_open_position(
            amount_usd=amount,
            leverage=Decimal("2.0"),  # Approximate — actual depends on LTV
        )
        if not check.can_proceed:
            raise PositionManagerError(f"Safety gate blocked: {check.reason}")

        # 7. Build transaction
        tx = {
            "from": self._user_address,
            "to": Web3.to_checksum_address(AAVE_V3_POOL),
            "data": flash_loan_calldata,
            "gas": 800_000,
            "value": 0,
        }

        # 8. Simulate
        await self._tx_submitter.simulate(tx)
        self._logger.info("Simulation passed for %s open", direction.value)

        # 9. Submit (if not dry run)
        if self._safety.is_dry_run:
            self._logger.info("DRY RUN: Would submit open %s position", direction.value)
            position = self._build_position_state(
                direction, debt_token, collateral_token, amount, quote
            )
            self._current_position = position
            return position

        receipt = await self._tx_submitter.submit_and_wait(tx)
        tx_hash = receipt.get("transactionHash", "").hex() if receipt.get("transactionHash") else ""
        gas_used = receipt.get("gasUsed", 0)

        # 10. Verify post-execution state
        account = await self._aave_client.get_user_account_data(self._executor_address)
        reserve = await self._aave_client.get_reserve_data(debt_addr)

        position = PositionState(
            direction=direction,
            debt_token=debt_token,
            collateral_token=collateral_token,
            debt_usd=account.total_debt_usd,
            collateral_usd=account.total_collateral_usd,
            initial_debt_usd=amount,
            initial_collateral_usd=account.total_collateral_usd,
            health_factor=account.health_factor,
            borrow_rate_ray=reserve.variable_borrow_rate,
            liquidation_threshold=account.current_liquidation_threshold,
        )

        self._current_position = position
        self._safety.record_action()

        # 11. Record to P&L tracker
        gas_cost_usd = await self._estimate_gas_cost_usd(gas_used)
        self._current_position_id = await self._pnl_tracker.record_open(
            position, tx_hash, gas_cost_usd
        )

        self._logger.info(
            "Position opened: direction=%s HF=%.4f collateral=$%s debt=$%s",
            direction.value,
            account.health_factor,
            account.total_collateral_usd,
            account.total_debt_usd,
        )
        return position

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------

    async def close_position(self, reason: str = "signal") -> PositionState:
        """
        Close the current position entirely.

        LONG close: flash loan debt (mode=0) → repay → withdraw collateral → swap → repay flash
        SHORT close: flash loan debt (mode=0) → repay → withdraw collateral → swap → repay flash
        """
        if not self.has_open_position:
            raise PositionManagerError("No open position to close")

        pos = self._current_position
        assert pos is not None

        debt_addr = self._get_token_address(pos.debt_token)
        collateral_addr = self._get_token_address(pos.collateral_token)
        debt_decimals = self._get_token_decimals(pos.debt_token)
        collateral_decimals = self._get_token_decimals(pos.collateral_token)

        # Get current debt amount (with accrued interest)
        account = await self._aave_client.get_user_account_data(self._executor_address)
        debt_amount_native = int(account.total_debt_usd * Decimal(10**debt_decimals))

        # Add buffer for interest that accrues during tx (0.1%)
        debt_with_buffer = int(debt_amount_native * Decimal("1.001"))

        # Get swap quote: collateral → debt (to repay flash loan)
        collateral_amount_native = int(
            account.total_collateral_usd * Decimal(10**collateral_decimals)
        )
        quote = await self._aggregator_client.get_best_quote(
            from_token=collateral_addr,
            to_token=debt_addr,
            from_amount=collateral_amount_native,
            from_decimals=collateral_decimals,
            to_decimals=debt_decimals,
        )

        # Encode flash loan for closing (mode=0: must repay in same tx)
        params = Web3.solidity_keccak(
            ["address", "bytes", "uint8"],
            [quote.router_address, quote.calldata, 2],  # 2 = close operation
        )

        flash_loan_calldata = self._aave_client.encode_flash_loan(
            receiver=self._executor_address,
            assets=[debt_addr],
            amounts=[debt_with_buffer],
            modes=[FLASH_LOAN_MODE_NO_DEBT],
            on_behalf_of=self._executor_address,
            params=params,
        )

        tx = {
            "from": self._user_address,
            "to": Web3.to_checksum_address(AAVE_V3_POOL),
            "data": flash_loan_calldata,
            "gas": 900_000,
            "value": 0,
        }

        # Simulate
        await self._tx_submitter.simulate(tx)
        self._logger.info("Close simulation passed for %s position", pos.direction.value)

        if self._safety.is_dry_run:
            self._logger.info("DRY RUN: Would close %s position", pos.direction.value)
            closed = PositionState(
                direction=pos.direction,
                debt_token=pos.debt_token,
                collateral_token=pos.collateral_token,
                debt_usd=Decimal("0"),
                collateral_usd=Decimal("0"),
                initial_debt_usd=pos.initial_debt_usd,
                initial_collateral_usd=pos.initial_collateral_usd,
                health_factor=Decimal("0"),
                borrow_rate_ray=Decimal("0"),
                liquidation_threshold=pos.liquidation_threshold,
            )
            self._current_position = None
            self._current_position_id = None
            return closed

        receipt = await self._tx_submitter.submit_and_wait(tx)
        tx_hash = receipt.get("transactionHash", "").hex() if receipt.get("transactionHash") else ""
        gas_used = receipt.get("gasUsed", 0)

        gas_cost_usd = await self._estimate_gas_cost_usd(gas_used)
        self._safety.record_action()

        # Compute tokens received (surplus after repaying flash loan)
        # This is an approximation — actual surplus comes from contract events
        tokens_received = quote.to_amount / Decimal(10**debt_decimals) - account.total_debt_usd
        if tokens_received < 0:
            tokens_received = Decimal("0")

        # Record to P&L tracker
        if self._current_position_id is not None:
            pnl = await self._pnl_tracker.record_close(
                self._current_position_id,
                tx_hash,
                gas_cost_usd,
                tokens_received,
                close_reason=reason,
            )
            self._logger.info("Realized P&L: net=$%s", pnl.net_pnl_usd)

        closed = PositionState(
            direction=pos.direction,
            debt_token=pos.debt_token,
            collateral_token=pos.collateral_token,
            debt_usd=Decimal("0"),
            collateral_usd=Decimal("0"),
            initial_debt_usd=pos.initial_debt_usd,
            initial_collateral_usd=pos.initial_collateral_usd,
            health_factor=Decimal("0"),
            borrow_rate_ray=Decimal("0"),
            liquidation_threshold=pos.liquidation_threshold,
        )

        self._current_position = None
        self._current_position_id = None

        self._logger.info("Position closed: direction=%s reason=%s", pos.direction.value, reason)
        return closed

    # ------------------------------------------------------------------
    # Partial deleverage
    # ------------------------------------------------------------------

    async def partial_deleverage(self, target_hf: Decimal) -> PositionState:
        """
        Reduce position size to bring health factor back to target.

        Uses the deleverage formula:
            repay_amount = (D - C * LT / h_t) / (1 + f - LT / h_t)
        where D=debt, C=collateral, LT=liq threshold, h_t=target HF, f=flash premium.
        """
        if not self.has_open_position:
            raise PositionManagerError("No open position to deleverage")

        pos = self._current_position
        assert pos is not None

        # Refresh position data
        account = await self._aave_client.get_user_account_data(self._executor_address)
        current_hf = account.health_factor

        if current_hf >= target_hf:
            self._logger.info(
                "HF %.4f already above target %.4f, no deleverage needed",
                current_hf,
                target_hf,
            )
            return pos

        # Compute repay amount using analytical formula
        flash_premium = DEFAULT_FLASH_LOAN_PREMIUM
        lt = pos.liquidation_threshold
        debt = account.total_debt_usd
        collateral = account.total_collateral_usd

        lt_over_ht = lt / target_hf
        numerator = debt - collateral * lt_over_ht
        denominator = Decimal("1") + flash_premium - lt_over_ht

        if denominator <= 0:
            raise PositionManagerError(
                f"Deleverage formula denominator <= 0: target_hf={target_hf} " f"may be unreachable"
            )

        repay_amount_usd = numerator / denominator
        if repay_amount_usd <= 0:
            self._logger.info("Computed repay amount <= 0, no deleverage needed")
            return pos

        self._logger.info(
            "Deleverage: repay $%s to reach HF %.4f (current %.4f)",
            repay_amount_usd,
            target_hf,
            current_hf,
        )

        # Execute: the deleverage flow is similar to a partial close
        debt_addr = self._get_token_address(pos.debt_token)
        collateral_addr = self._get_token_address(pos.collateral_token)
        debt_decimals = self._get_token_decimals(pos.debt_token)
        collateral_decimals = self._get_token_decimals(pos.collateral_token)

        repay_native = int(repay_amount_usd * Decimal(10**debt_decimals))

        # Swap collateral → debt for repayment
        # Estimate collateral to sell based on current price ratio
        collateral_to_sell = int(
            repay_amount_usd
            / collateral
            * account.total_collateral_usd
            * Decimal(10**collateral_decimals)
        )

        quote = await self._aggregator_client.get_best_quote(
            from_token=collateral_addr,
            to_token=debt_addr,
            from_amount=collateral_to_sell,
            from_decimals=collateral_decimals,
            to_decimals=debt_decimals,
        )

        params = Web3.solidity_keccak(
            ["address", "bytes", "uint8"],
            [quote.router_address, quote.calldata, 3],  # 3 = deleverage
        )

        flash_loan_calldata = self._aave_client.encode_flash_loan(
            receiver=self._executor_address,
            assets=[debt_addr],
            amounts=[repay_native],
            modes=[FLASH_LOAN_MODE_NO_DEBT],
            on_behalf_of=self._executor_address,
            params=params,
        )

        tx = {
            "from": self._user_address,
            "to": Web3.to_checksum_address(AAVE_V3_POOL),
            "data": flash_loan_calldata,
            "gas": 900_000,
            "value": 0,
        }

        await self._tx_submitter.simulate(tx)

        if self._safety.is_dry_run:
            self._logger.info("DRY RUN: Would deleverage %s position", pos.direction.value)
            return pos

        receipt = await self._tx_submitter.submit_and_wait(tx)
        tx_hash = receipt.get("transactionHash", "").hex() if receipt.get("transactionHash") else ""
        gas_used = receipt.get("gasUsed", 0)
        gas_cost_usd = await self._estimate_gas_cost_usd(gas_used)

        self._safety.record_action()

        # Record deleverage
        if self._current_position_id is not None:
            await self._pnl_tracker.record_deleverage(
                self._current_position_id, tx_hash, gas_cost_usd
            )

        # Refresh position state
        account = await self._aave_client.get_user_account_data(self._executor_address)
        reserve = await self._aave_client.get_reserve_data(debt_addr)

        updated = PositionState(
            direction=pos.direction,
            debt_token=pos.debt_token,
            collateral_token=pos.collateral_token,
            debt_usd=account.total_debt_usd,
            collateral_usd=account.total_collateral_usd,
            initial_debt_usd=pos.initial_debt_usd,
            initial_collateral_usd=pos.initial_collateral_usd,
            health_factor=account.health_factor,
            borrow_rate_ray=reserve.variable_borrow_rate,
            liquidation_threshold=account.current_liquidation_threshold,
        )

        self._current_position = updated

        self._logger.info(
            "Deleverage complete: new HF=%.4f debt=$%s collateral=$%s",
            account.health_factor,
            account.total_debt_usd,
            account.total_collateral_usd,
        )
        return updated

    # ------------------------------------------------------------------
    # Increase position
    # ------------------------------------------------------------------

    async def increase_position(self, additional_amount: Decimal) -> PositionState:
        """Add to existing position with additional flash-borrowed debt."""
        if not self.has_open_position:
            raise PositionManagerError("No open position to increase")

        pos = self._current_position
        assert pos is not None

        # Reuse open_position logic with the same direction and tokens
        # but first clear current position to allow re-open
        saved_position = pos
        _saved_id = self._current_position_id

        # We don't close and re-open; instead we do an additional flash loan
        debt_addr = self._get_token_address(pos.debt_token)
        collateral_addr = self._get_token_address(pos.collateral_token)
        debt_decimals = self._get_token_decimals(pos.debt_token)
        collateral_decimals = self._get_token_decimals(pos.collateral_token)

        amount_native = int(additional_amount * Decimal(10**debt_decimals))

        quote = await self._aggregator_client.get_best_quote(
            from_token=debt_addr,
            to_token=collateral_addr,
            from_amount=amount_native,
            from_decimals=debt_decimals,
            to_decimals=collateral_decimals,
        )

        params = Web3.solidity_keccak(
            ["address", "bytes", "uint8"],
            [
                quote.router_address,
                quote.calldata,
                0 if pos.direction == PositionDirection.LONG else 1,
            ],
        )

        flash_loan_calldata = self._aave_client.encode_flash_loan(
            receiver=self._executor_address,
            assets=[debt_addr],
            amounts=[amount_native],
            modes=[FLASH_LOAN_MODE_VARIABLE_DEBT],
            on_behalf_of=self._executor_address,
            params=params,
        )

        check = self._safety.can_open_position(
            amount_usd=additional_amount,
            leverage=Decimal("2.0"),
        )
        if not check.can_proceed:
            raise PositionManagerError(f"Safety gate blocked increase: {check.reason}")

        tx = {
            "from": self._user_address,
            "to": Web3.to_checksum_address(AAVE_V3_POOL),
            "data": flash_loan_calldata,
            "gas": 800_000,
            "value": 0,
        }

        await self._tx_submitter.simulate(tx)

        if self._safety.is_dry_run:
            self._logger.info(
                "DRY RUN: Would increase %s position by $%s", pos.direction.value, additional_amount
            )
            return pos

        _receipt = await self._tx_submitter.submit_and_wait(tx)
        self._safety.record_action()

        # Refresh state
        account = await self._aave_client.get_user_account_data(self._executor_address)
        reserve = await self._aave_client.get_reserve_data(debt_addr)

        updated = PositionState(
            direction=pos.direction,
            debt_token=pos.debt_token,
            collateral_token=pos.collateral_token,
            debt_usd=account.total_debt_usd,
            collateral_usd=account.total_collateral_usd,
            initial_debt_usd=saved_position.initial_debt_usd + additional_amount,
            initial_collateral_usd=saved_position.initial_collateral_usd,
            health_factor=account.health_factor,
            borrow_rate_ray=reserve.variable_borrow_rate,
            liquidation_threshold=account.current_liquidation_threshold,
        )

        self._current_position = updated
        self._logger.info(
            "Position increased by $%s: new HF=%.4f",
            additional_amount,
            account.health_factor,
        )
        return updated

    # ------------------------------------------------------------------
    # Isolation mode check
    # ------------------------------------------------------------------

    async def _check_isolation_mode(self, collateral_token_address: str) -> tuple[bool, str]:
        """
        Check if collateral asset is restricted by Isolation Mode on Aave V3 BSC.

        Returns (ok, message).
        """
        reserve_data = await self._aave_client.get_reserve_data(collateral_token_address)

        if reserve_data.isolation_mode_enabled:
            if (
                reserve_data.debt_ceiling > 0
                and reserve_data.current_isolated_debt >= reserve_data.debt_ceiling
            ):
                return (
                    False,
                    f"Isolation debt ceiling reached: "
                    f"{reserve_data.current_isolated_debt} >= {reserve_data.debt_ceiling}",
                )
            return (
                True,
                "Asset is in isolation mode but within debt ceiling",
            )

        return (True, "Asset is not in isolation mode")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_token_address(self, symbol: str) -> str:
        """Resolve token symbol to checksummed address."""
        token_info = self._tokens.get(symbol)
        if token_info is None:
            raise PositionManagerError(f"Unknown token symbol: {symbol}")
        return Web3.to_checksum_address(token_info["address"])

    def _get_token_decimals(self, symbol: str) -> int:
        """Resolve token symbol to decimal count."""
        token_info = self._tokens.get(symbol)
        if token_info is None:
            raise PositionManagerError(f"Unknown token symbol: {symbol}")
        return int(token_info.get("decimals", 18))

    def _build_position_state(
        self,
        direction: PositionDirection,
        debt_token: str,
        collateral_token: str,
        amount: Decimal,
        quote: SwapQuote,
    ) -> PositionState:
        """Build a PositionState from open parameters (for dry-run)."""
        collateral_decimals = self._get_token_decimals(collateral_token)
        collateral_usd = quote.to_amount / Decimal(10**collateral_decimals)

        # Approximate LT from config
        aave_assets = self._aave_config.get("supported_assets", {})
        collateral_info = aave_assets.get(collateral_token, {})
        lt_bps = collateral_info.get("liquidation_threshold_bps", 8000)
        lt = Decimal(lt_bps) / Decimal("10000")

        # Approximate HF
        hf = (collateral_usd * lt) / amount if amount > 0 else Decimal("0")

        return PositionState(
            direction=direction,
            debt_token=debt_token,
            collateral_token=collateral_token,
            debt_usd=amount,
            collateral_usd=collateral_usd,
            initial_debt_usd=amount,
            initial_collateral_usd=collateral_usd,
            health_factor=hf,
            borrow_rate_ray=Decimal("0"),
            liquidation_threshold=lt,
        )

    async def _estimate_gas_cost_usd(self, gas_used: int) -> Decimal:
        """Estimate gas cost in USD from gas units used."""
        if gas_used == 0:
            return Decimal("0")
        max_fee, _ = await self._tx_submitter.get_gas_price()
        gas_cost_wei = gas_used * max_fee
        gas_cost_bnb = Decimal(gas_cost_wei) / WAD
        # Get BNB price
        from shared.constants import TOKEN_WBNB

        bnb_price = await self._aave_client.get_asset_price(TOKEN_WBNB)
        return gas_cost_bnb * bnb_price

    # ------------------------------------------------------------------
    # State refresh
    # ------------------------------------------------------------------

    async def refresh_position(self) -> PositionState | None:
        """Refresh the current position's on-chain state."""
        if not self.has_open_position:
            return None

        pos = self._current_position
        assert pos is not None

        account = await self._aave_client.get_user_account_data(self._executor_address)
        debt_addr = self._get_token_address(pos.debt_token)
        reserve = await self._aave_client.get_reserve_data(debt_addr)

        updated = PositionState(
            direction=pos.direction,
            debt_token=pos.debt_token,
            collateral_token=pos.collateral_token,
            debt_usd=account.total_debt_usd,
            collateral_usd=account.total_collateral_usd,
            initial_debt_usd=pos.initial_debt_usd,
            initial_collateral_usd=pos.initial_collateral_usd,
            health_factor=account.health_factor,
            borrow_rate_ray=reserve.variable_borrow_rate,
            liquidation_threshold=account.current_liquidation_threshold,
        )

        self._current_position = updated
        return updated
