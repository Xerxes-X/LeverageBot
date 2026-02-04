"""
Transaction submission layer for BSC Leverage Bot.

Signs, simulates (eth_call), submits via MEV-protected RPC, and confirms
transactions with robust nonce management and stuck-tx recovery.

Usage:
    submitter = TxSubmitter(w3, safety, private_key, user_address)
    receipt = await submitter.submit_and_wait(tx)
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

from eth_typing import HexStr
from web3 import AsyncHTTPProvider, AsyncWeb3, Web3
from web3.types import TxParams

from bot_logging.logger_manager import setup_module_logger
from config.loader import get_config
from shared.constants import MEV_PROTECTED_RPC

if TYPE_CHECKING:
    from core.safety import SafetyState

# Error(string) function selector — first 4 bytes of keccak256("Error(string)")
_ERROR_SELECTOR = bytes.fromhex("08c379a0")


class TxSubmitterError(Exception):
    """Base error for transaction submission failures."""


class SimulationFailedError(TxSubmitterError):
    """Raised when eth_call simulation reverts."""


class TxRevertedError(TxSubmitterError):
    """Raised when a confirmed transaction has status=0 (reverted)."""


class TxTimeoutError(TxSubmitterError):
    """Raised when a transaction is not confirmed within the timeout."""


class TxSubmitter:
    """
    Transaction submitter with nonce management and MEV protection.

    Uses a separate MEV-protected RPC for submission to avoid sandwich
    attacks, while the normal RPC is used for simulation and reads.
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        safety: SafetyState,
        private_key: str,
        user_address: str,
    ) -> None:
        self._w3 = w3
        self._safety = safety
        self._private_key = private_key
        self._user_address = Web3.to_checksum_address(user_address)

        cfg = get_config()
        timing_cfg = cfg.get_timing_config()
        chain_cfg = cfg.get_chain_config(56)

        # Timing
        tx_timing = timing_cfg.get("transaction", {})
        self._confirmation_timeout: int = tx_timing.get("confirmation_timeout_seconds", 60)
        self._simulation_timeout: int = tx_timing.get("simulation_timeout_seconds", 15)
        self._nonce_refresh_interval: int = tx_timing.get("nonce_refresh_interval_seconds", 30)

        # Chain
        self._chain_id: int = chain_cfg.get("chain_id", 56)
        mev_url = chain_cfg.get("rpc", {}).get("mev_protected_url", MEV_PROTECTED_RPC)

        # MEV-protected web3 instance for submission
        self._mev_w3 = AsyncWeb3(AsyncHTTPProvider(mev_url))

        # Gas
        self._gas_price_buffer: float = 1.1  # 10% safety buffer

        # Nonce state
        self._nonce: int | None = None
        self._nonce_lock = asyncio.Lock()

        self._logger = setup_module_logger(
            "tx_submitter", "tx_submitter.log", module_folder="TX_Submitter_Logs"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def simulate(self, tx: dict[str, Any]) -> bytes:
        """
        Dry-run a transaction via eth_call.

        Returns output bytes on success.  Raises ``SimulationFailedError``
        on revert or timeout.
        """
        try:
            result = await asyncio.wait_for(
                self._w3.eth.call(cast(TxParams, tx)),
                timeout=self._simulation_timeout,
            )
            self._logger.debug("Simulation succeeded: %d bytes output", len(result))
            return result
        except asyncio.TimeoutError as exc:
            raise SimulationFailedError(
                f"Simulation timed out after {self._simulation_timeout}s"
            ) from exc
        except Exception as exc:
            reason = self.decode_revert_reason(getattr(exc, "data", b""))
            raise SimulationFailedError(f"Simulation reverted: {reason}") from exc

    async def submit(self, tx: dict[str, Any]) -> str:
        """
        Sign and submit a transaction via MEV-protected RPC.

        Assigns nonce, chain ID, and gas price automatically.
        Returns the transaction hash as a hex string.
        """
        nonce = await self._get_next_nonce()
        max_fee, priority_fee = await self.get_gas_price()

        tx = {
            **tx,
            "chainId": self._chain_id,
            "nonce": nonce,
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority_fee,
            "type": 2,  # EIP-1559
        }

        signed = self._w3.eth.account.sign_transaction(tx, self._private_key)
        tx_hash = await self._mev_w3.eth.send_raw_transaction(signed.raw_transaction)

        tx_hash_hex = tx_hash.hex()
        self._logger.info(
            "TX submitted: hash=%s nonce=%d maxFee=%d priorityFee=%d",
            tx_hash_hex,
            nonce,
            max_fee,
            priority_fee,
        )
        return tx_hash_hex

    async def wait_for_receipt(self, tx_hash: str, timeout: int | None = None) -> dict[str, Any]:
        """
        Poll for transaction receipt until confirmed or timeout.

        Raises ``TxRevertedError`` if status=0, ``TxTimeoutError`` on timeout.
        """
        if timeout is None:
            timeout = self._confirmation_timeout

        start = time.monotonic()

        while True:
            try:
                receipt = await self._w3.eth.get_transaction_receipt(cast(HexStr, tx_hash))
            except Exception:
                receipt = None

            if receipt is not None:
                if receipt.get("status") == 1:
                    self._logger.info(
                        "TX confirmed: hash=%s gasUsed=%s",
                        tx_hash,
                        receipt.get("gasUsed"),
                    )
                    return dict(receipt)

                # status == 0 → reverted on-chain
                raise TxRevertedError(f"TX reverted on-chain: {tx_hash}")

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TxTimeoutError(f"TX {tx_hash} not confirmed after {timeout}s")

            await asyncio.sleep(1)

    async def submit_and_wait(self, tx: dict[str, Any]) -> dict[str, Any]:
        """
        Full submission flow: simulate → safety check → submit → wait.

        Returns the transaction receipt dict.
        """
        # 1. Simulate
        await self.simulate(tx)

        # 2. Safety gate
        max_fee, _ = await self.get_gas_price()
        gas_price_gwei = max_fee // 10**9
        check = self._safety.can_submit_tx(int(gas_price_gwei))
        if not check.can_proceed:
            raise TxSubmitterError(f"Safety gate blocked: {check.reason}")

        # 3. Submit
        tx_hash = await self.submit(tx)

        # 4. Wait for receipt
        return await self.wait_for_receipt(tx_hash)

    async def get_gas_price(self) -> tuple[int, int]:
        """
        Get current gas price as (maxFeePerGas, maxPriorityFeePerGas) in Wei.

        Applies a 10% buffer to the base gas price.
        """
        base_price = await self._w3.eth.gas_price
        max_fee = int(base_price * self._gas_price_buffer)
        priority_fee = max(int(base_price * 0.1), 1)
        max_fee = max(max_fee, priority_fee)
        return max_fee, priority_fee

    # ------------------------------------------------------------------
    # Nonce management
    # ------------------------------------------------------------------

    async def _get_next_nonce(self) -> int:
        """Thread-safe nonce increment with asyncio.Lock."""
        async with self._nonce_lock:
            if self._nonce is None:
                self._nonce = await self._w3.eth.get_transaction_count(
                    self._user_address, "pending"
                )
                self._logger.info("Nonce initialized from chain: %d", self._nonce)
            nonce = self._nonce
            self._nonce += 1
            return nonce

    async def _recover_nonce(self) -> None:
        """Re-sync local nonce counter from chain state."""
        async with self._nonce_lock:
            pending = await self._w3.eth.get_transaction_count(self._user_address, "pending")
            latest = await self._w3.eth.get_transaction_count(self._user_address, "latest")
            old_nonce = self._nonce
            self._nonce = pending
            self._logger.warning(
                "Nonce recovered: local=%s pending=%d latest=%d",
                old_nonce,
                pending,
                latest,
            )

    async def _replace_stuck_tx(self, nonce: int, gas_bump_pct: float = 12.5) -> str:
        """
        Replace a stuck transaction by re-submitting at the same nonce
        with higher gas (zero-value self-transfer).
        """
        max_fee, priority_fee = await self.get_gas_price()
        bumped_max_fee = int(max_fee * (1 + gas_bump_pct / 100))
        bumped_priority_fee = int(priority_fee * (1 + gas_bump_pct / 100))

        replacement_tx = {
            "from": self._user_address,
            "to": self._user_address,
            "value": 0,
            "gas": 21000,
            "chainId": self._chain_id,
            "nonce": nonce,
            "maxFeePerGas": bumped_max_fee,
            "maxPriorityFeePerGas": bumped_priority_fee,
            "type": 2,
        }

        signed = self._w3.eth.account.sign_transaction(replacement_tx, self._private_key)
        tx_hash = await self._mev_w3.eth.send_raw_transaction(signed.raw_transaction)

        tx_hash_hex = tx_hash.hex()
        self._logger.warning(
            "Stuck TX replaced: nonce=%d new_hash=%s bumped_gas=%d",
            nonce,
            tx_hash_hex,
            bumped_max_fee,
        )
        return tx_hash_hex

    # ------------------------------------------------------------------
    # Revert decoding
    # ------------------------------------------------------------------

    @staticmethod
    def decode_revert_reason(data: bytes | str) -> str:
        """
        Decode a Solidity revert reason from raw data.

        Handles ``Error(string)`` selector (0x08c379a0).
        Returns "Unknown revert" for empty or unrecognized data.
        """
        if not data:
            return "Unknown revert"

        if isinstance(data, str):
            data = bytes.fromhex(data.removeprefix("0x"))

        if len(data) < 4:
            return data.hex()

        if data[:4] == _ERROR_SELECTOR and len(data) >= 68:
            # ABI-encoded Error(string): selector(4) + offset(32) + length(32) + data
            try:
                str_len = int.from_bytes(data[36:68], "big")
                return data[68 : 68 + str_len].decode("utf-8", errors="replace")
            except Exception:
                pass

        return data.hex()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the MEV web3 provider session."""
        if hasattr(self._mev_w3.provider, "cache_allowed_requests"):
            # AsyncHTTPProvider cleanup
            pass
        self._logger.debug("TxSubmitter closed")
