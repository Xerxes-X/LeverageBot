"""
Unit tests for execution/tx_submitter.py.

All tests mock AsyncWeb3 and SafetyState to avoid real RPC connections.
Tests verify nonce management, simulation, submission, receipt polling,
gas pricing, safety gating, and revert reason decoding.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from shared.types import SafetyCheck

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

SAMPLE_USER = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
SAMPLE_PRIVATE_KEY = "0x" + "ab" * 32
SAMPLE_TX_HASH = bytes.fromhex("cd" * 32)
SAMPLE_TX_HASH_HEX = "cd" * 32

SAMPLE_TX = {
    "to": "0x6807dc923806fE8Fd134338EABCA509979a7e0cB",
    "data": "0xdeadbeef",
    "value": 0,
    "gas": 500000,
}

SAMPLE_RECEIPT_SUCCESS = {
    "status": 1,
    "transactionHash": SAMPLE_TX_HASH,
    "gasUsed": 200000,
    "blockNumber": 12345,
}

SAMPLE_RECEIPT_FAILED = {
    "status": 0,
    "transactionHash": SAMPLE_TX_HASH,
    "gasUsed": 150000,
    "blockNumber": 12345,
}

# 5 gwei base gas price
SAMPLE_GAS_PRICE = 5 * 10**9


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _gas_price_coro():
    """Coroutine that returns sample gas price (used as property mock)."""
    return SAMPLE_GAS_PRICE


@pytest.fixture
def mock_w3():
    w3 = MagicMock()
    w3.eth = MagicMock()
    w3.eth.call = AsyncMock(return_value=b"\x00" * 32)
    # gas_price is a property in web3 that returns a coroutine
    type(w3.eth).gas_price = PropertyMock(side_effect=lambda: _gas_price_coro())
    w3.eth.get_transaction_count = AsyncMock(return_value=42)
    w3.eth.send_raw_transaction = AsyncMock(return_value=SAMPLE_TX_HASH)
    w3.eth.get_transaction_receipt = AsyncMock(return_value=SAMPLE_RECEIPT_SUCCESS)

    # Account signing
    signed_mock = MagicMock()
    signed_mock.raw_transaction = b"\x00" * 100
    w3.eth.account = MagicMock()
    w3.eth.account.sign_transaction = MagicMock(return_value=signed_mock)

    return w3


@pytest.fixture
def mock_safety():
    safety = MagicMock()
    safety.can_submit_tx = MagicMock(
        return_value=SafetyCheck(can_proceed=True, reason="Gas price acceptable")
    )
    return safety


@pytest.fixture
def tx_submitter(mock_w3, mock_safety):
    with (
        patch("execution.tx_submitter.get_config") as mock_cfg,
        patch("execution.tx_submitter.setup_module_logger") as mock_logger,
        patch("execution.tx_submitter.AsyncWeb3") as mock_async_web3,
        patch("execution.tx_submitter.AsyncHTTPProvider"),
    ):
        mock_loader = MagicMock()
        mock_loader.get_timing_config.return_value = {
            "transaction": {
                "confirmation_timeout_seconds": 60,
                "simulation_timeout_seconds": 15,
                "nonce_refresh_interval_seconds": 30,
            },
        }
        mock_loader.get_chain_config.return_value = {
            "chain_id": 56,
            "rpc": {
                "mev_protected_url": "https://rpc.48.club",
            },
        }
        mock_cfg.return_value = mock_loader
        mock_logger.return_value = MagicMock()

        # MEV web3 mock
        mev_w3 = MagicMock()
        mev_w3.eth = MagicMock()
        mev_w3.eth.send_raw_transaction = AsyncMock(return_value=SAMPLE_TX_HASH)
        mock_async_web3.return_value = mev_w3

        from execution.tx_submitter import TxSubmitter

        submitter = TxSubmitter(
            w3=mock_w3,
            safety=mock_safety,
            private_key=SAMPLE_PRIVATE_KEY,
            user_address=SAMPLE_USER,
        )

    return submitter


# ---------------------------------------------------------------------------
# A. Nonce management tests
# ---------------------------------------------------------------------------


class TestNonceManagement:

    async def test_nonce_initializes_from_pending(self, tx_submitter, mock_w3):
        nonce = await tx_submitter._get_next_nonce()

        assert nonce == 42
        mock_w3.eth.get_transaction_count.assert_called_once_with(
            tx_submitter._user_address, "pending"
        )

    async def test_nonce_increments_atomically(self, tx_submitter):
        n1 = await tx_submitter._get_next_nonce()
        n2 = await tx_submitter._get_next_nonce()

        assert n2 == n1 + 1

    async def test_nonce_recovery_resyncs(self, tx_submitter, mock_w3):
        # Initialize nonce
        await tx_submitter._get_next_nonce()

        # Chain says pending is now 50
        mock_w3.eth.get_transaction_count = AsyncMock(return_value=50)

        await tx_submitter._recover_nonce()

        # Next nonce should be 50
        nonce = await tx_submitter._get_next_nonce()
        assert nonce == 50

    async def test_replace_stuck_tx_bumps_gas(self, tx_submitter, mock_w3):
        tx_hash = await tx_submitter._replace_stuck_tx(nonce=42, gas_bump_pct=12.5)

        assert tx_hash == SAMPLE_TX_HASH_HEX

        # Verify sign_transaction was called with correct nonce
        call_args = mock_w3.eth.account.sign_transaction.call_args
        replacement_tx = call_args[0][0]
        assert replacement_tx["nonce"] == 42
        assert replacement_tx["to"] == tx_submitter._user_address
        assert replacement_tx["value"] == 0
        assert replacement_tx["gas"] == 21000

        # Gas should be bumped by 12.5%
        expected_max_fee = int(int(SAMPLE_GAS_PRICE * 1.1) * 1.125)
        assert replacement_tx["maxFeePerGas"] == expected_max_fee


# ---------------------------------------------------------------------------
# B. Simulation tests
# ---------------------------------------------------------------------------


class TestSimulation:

    async def test_simulate_success_returns_bytes(self, tx_submitter):
        result = await tx_submitter.simulate(SAMPLE_TX)

        assert isinstance(result, bytes)
        assert len(result) == 32

    async def test_simulate_revert_raises_error(self, tx_submitter, mock_w3):
        from execution.tx_submitter import SimulationFailedError

        mock_w3.eth.call = AsyncMock(side_effect=Exception("execution reverted"))

        with pytest.raises(SimulationFailedError, match="Simulation reverted"):
            await tx_submitter.simulate(SAMPLE_TX)

    async def test_simulate_timeout_raises_error(self, tx_submitter, mock_w3):
        from execution.tx_submitter import SimulationFailedError

        async def slow_call(*args, **kwargs):
            await asyncio.sleep(999)

        mock_w3.eth.call = slow_call
        tx_submitter._simulation_timeout = 0.01  # Very short timeout for test

        with pytest.raises(SimulationFailedError, match="timed out"):
            await tx_submitter.simulate(SAMPLE_TX)


# ---------------------------------------------------------------------------
# C. Submission tests
# ---------------------------------------------------------------------------


class TestSubmission:

    async def test_submit_signs_and_sends(self, tx_submitter, mock_w3):
        tx_hash = await tx_submitter.submit(SAMPLE_TX)

        assert tx_hash == SAMPLE_TX_HASH_HEX
        mock_w3.eth.account.sign_transaction.assert_called_once()
        tx_submitter._mev_w3.eth.send_raw_transaction.assert_called_once()

    async def test_submit_assigns_nonce_and_gas(self, tx_submitter, mock_w3):
        await tx_submitter.submit(SAMPLE_TX)

        call_args = mock_w3.eth.account.sign_transaction.call_args
        signed_tx = call_args[0][0]
        assert signed_tx["nonce"] == 42
        assert signed_tx["chainId"] == 56
        assert "maxFeePerGas" in signed_tx
        assert "maxPriorityFeePerGas" in signed_tx
        assert signed_tx["type"] == 2


# ---------------------------------------------------------------------------
# D. Wait for receipt tests
# ---------------------------------------------------------------------------


class TestWaitForReceipt:

    async def test_wait_success(self, tx_submitter):
        receipt = await tx_submitter.wait_for_receipt(SAMPLE_TX_HASH_HEX)

        assert receipt["status"] == 1
        assert receipt["gasUsed"] == 200000

    async def test_wait_reverted_raises_error(self, tx_submitter, mock_w3):
        from execution.tx_submitter import TxRevertedError

        mock_w3.eth.get_transaction_receipt = AsyncMock(return_value=SAMPLE_RECEIPT_FAILED)

        with pytest.raises(TxRevertedError, match="reverted on-chain"):
            await tx_submitter.wait_for_receipt(SAMPLE_TX_HASH_HEX)

    async def test_wait_timeout_raises_error(self, tx_submitter, mock_w3):
        from execution.tx_submitter import TxTimeoutError

        mock_w3.eth.get_transaction_receipt = AsyncMock(return_value=None)

        with pytest.raises(TxTimeoutError, match="not confirmed"):
            await tx_submitter.wait_for_receipt(SAMPLE_TX_HASH_HEX, timeout=1)


# ---------------------------------------------------------------------------
# E. Submit and wait tests
# ---------------------------------------------------------------------------


class TestSubmitAndWait:

    async def test_full_flow_success(self, tx_submitter):
        receipt = await tx_submitter.submit_and_wait(SAMPLE_TX)

        assert receipt["status"] == 1

    async def test_safety_gate_blocks_submission(self, tx_submitter, mock_safety):
        from execution.tx_submitter import TxSubmitterError

        mock_safety.can_submit_tx = MagicMock(
            return_value=SafetyCheck(
                can_proceed=False, reason="Gas price 15 gwei exceeds max 10 gwei"
            )
        )

        with pytest.raises(TxSubmitterError, match="Safety gate blocked"):
            await tx_submitter.submit_and_wait(SAMPLE_TX)

    async def test_simulation_failure_prevents_submission(self, tx_submitter, mock_w3):
        from execution.tx_submitter import SimulationFailedError

        mock_w3.eth.call = AsyncMock(side_effect=Exception("revert"))

        with pytest.raises(SimulationFailedError):
            await tx_submitter.submit_and_wait(SAMPLE_TX)

        # Submit should NOT have been called
        tx_submitter._mev_w3.eth.send_raw_transaction.assert_not_called()


# ---------------------------------------------------------------------------
# F. Gas price tests
# ---------------------------------------------------------------------------


class TestGasPrice:

    async def test_gas_price_with_buffer(self, tx_submitter):
        max_fee, priority_fee = await tx_submitter.get_gas_price()

        # Base is 5 gwei, buffer is 1.1x → max_fee = 5.5 gwei
        expected_max_fee = int(SAMPLE_GAS_PRICE * 1.1)
        expected_priority = max(int(SAMPLE_GAS_PRICE * 0.1), 1)

        assert max_fee == expected_max_fee
        assert priority_fee == expected_priority

    async def test_gas_price_to_gwei_conversion(self, tx_submitter):
        max_fee, _ = await tx_submitter.get_gas_price()
        gas_gwei = max_fee // 10**9

        # 5 gwei * 1.1 = 5.5 gwei → int division = 5
        assert gas_gwei == 5


# ---------------------------------------------------------------------------
# G. Revert decoding tests
# ---------------------------------------------------------------------------


class TestRevertDecoding:

    def test_decode_error_string(self):
        from execution.tx_submitter import TxSubmitter

        # Build Error(string) encoded data for "Insufficient output"
        message = b"Insufficient output"
        # selector(4) + offset(32) + length(32) + data(padded to 32)
        selector = bytes.fromhex("08c379a0")
        offset = (32).to_bytes(32, "big")
        length = len(message).to_bytes(32, "big")
        padded_msg = message + b"\x00" * (32 - len(message))
        data = selector + offset + length + padded_msg

        result = TxSubmitter.decode_revert_reason(data)
        assert result == "Insufficient output"

    def test_decode_empty_returns_unknown(self):
        from execution.tx_submitter import TxSubmitter

        assert TxSubmitter.decode_revert_reason(b"") == "Unknown revert"
        assert TxSubmitter.decode_revert_reason("") == "Unknown revert"

    def test_decode_hex_string_input(self):
        from execution.tx_submitter import TxSubmitter

        # Build the same Error(string) but pass as hex string
        message = b"Bad swap"
        selector = bytes.fromhex("08c379a0")
        offset = (32).to_bytes(32, "big")
        length = len(message).to_bytes(32, "big")
        padded_msg = message + b"\x00" * (32 - len(message))
        data = selector + offset + length + padded_msg

        result = TxSubmitter.decode_revert_reason("0x" + data.hex())
        assert result == "Bad swap"
