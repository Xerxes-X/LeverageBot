"""
Shared pytest configuration and fixtures for BSC Leverage Bot tests.

Provides common helpers used across both unit and integration test suites.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from shared.types import (
    UserAccountData,
)

# ---------------------------------------------------------------------------
# Decimal helper
# ---------------------------------------------------------------------------


def _d(v) -> Decimal:
    """Shorthand Decimal factory."""
    return Decimal(str(v))


# ---------------------------------------------------------------------------
# Standard mock configs (can be overridden per test via fixture params)
# ---------------------------------------------------------------------------

STANDARD_POSITIONS_CONFIG = {
    "dry_run": True,
    "max_flash_loan_usd": 5000,
    "max_position_usd": 10000,
    "max_leverage_ratio": "3.0",
    "min_health_factor": "1.5",
    "deleverage_threshold": "1.4",
    "close_threshold": "1.25",
    "target_hf_after_deleverage": "1.8",
    "max_gas_price_gwei": 10,
    "max_slippage_bps": 50,
    "cooldown_between_actions_seconds": 30,
    "max_transactions_per_24h": 50,
    "oracle_max_staleness_seconds": 60,
}


STANDARD_USER_ACCOUNT = UserAccountData(
    total_collateral_usd=_d("10000"),
    total_debt_usd=_d("5000"),
    available_borrow_usd=_d("2000"),
    current_liquidation_threshold=_d("0.8"),
    ltv=_d("0.75"),
    health_factor=_d("1.8"),
)

SAMPLE_USER_ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
SAMPLE_EXECUTOR_ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"


# ---------------------------------------------------------------------------
# Config loader fixture (patched singleton)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config_loader():
    """
    Provide a mock ConfigLoader that returns standard configs.

    Usage in tests:
        def test_something(mock_config_loader):
            mock_config_loader.get_positions_config.return_value = {...}
    """
    loader = MagicMock()
    loader.get_positions_config.return_value = STANDARD_POSITIONS_CONFIG.copy()
    loader.get_aave_config.return_value = {"flash_loan_premium_bps": 5, "supported_assets": []}
    loader.get_aggregator_config.return_value = {
        "providers": [{"name": "openocean", "enabled": True}]
    }
    loader.get_signals_config.return_value = {
        "enabled": True,
        "mode": "blended",
        "data_source": {},
        "indicators": {},
    }
    loader.get_chain_config.return_value = {
        "chain_id": 56,
        "block_time_seconds": 3,
        "rpc": {"http_url": "https://bsc-dataseed1.binance.org/"},
        "contracts": {
            "aave_v3_pool": "0x6807dc923806fE8Fd134338EABCA509979a7e0cB",
            "aave_v3_data_provider": "0x41585C50524fb8c3899B43D7D797d9486AAc94DB",
        },
        "chainlink_feeds": {},
        "tokens": {},
    }
    loader.get_timing_config.return_value = {}
    loader.get_app_config.return_value = {"logging": {"log_dir": "logs"}}
    loader.get_rate_limit_config.return_value = {"binance_spot": {"requests_per_minute": 1200}}
    loader.get_abi.return_value = []
    return loader


# ---------------------------------------------------------------------------
# asyncio.Queue fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def signal_queue():
    """Shared asyncio.Queue for test signal/health message passing."""
    return asyncio.Queue()
