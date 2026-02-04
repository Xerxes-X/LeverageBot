"""
Unit tests for execution/aave_client.py.

All tests mock the AsyncWeb3 instance and contract calls to avoid
real RPC connections. Tests verify WAD/RAY/USD_DECIMALS conversions,
isolation mode bitmap extraction, error wrapping, and ABI encoding.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.constants import DEFAULT_FLASH_LOAN_PREMIUM, RAY, WAD

# ---------------------------------------------------------------------------
# Test constants — realistic Aave V3 return values
# ---------------------------------------------------------------------------

# getUserAccountData: (collateral, debt, availableBorrow, liquidationThreshold, ltv, hf)
# $500 collateral, $250 debt, $100 borrow headroom, 80% LT, 75% LTV, HF 1.8
SAMPLE_ACCOUNT_DATA = (
    500 * 10**8,  # totalCollateralBase (8 decimals)
    250 * 10**8,  # totalDebtBase
    100 * 10**8,  # availableBorrowsBase
    8000,  # currentLiquidationThreshold (bps)
    7500,  # ltv (bps)
    int(Decimal("1.8") * WAD),  # healthFactor (WAD)
)

# DataProvider.getReserveData: 12 flat values
# 3.5% variable borrow rate in RAY, 80% utilization
_TOTAL_ATOKEN = 10_000 * 10**18
_TOTAL_STABLE_DEBT = 1_000 * 10**18
_TOTAL_VARIABLE_DEBT = 7_000 * 10**18
_VARIABLE_RATE_RAY = int(Decimal("0.035") * RAY)  # 3.5% APR
SAMPLE_DP_RESERVE_DATA = (
    0,  # unbacked
    0,  # accruedToTreasuryScaled
    _TOTAL_ATOKEN,  # totalAToken
    _TOTAL_STABLE_DEBT,  # totalStableDebt
    _TOTAL_VARIABLE_DEBT,  # totalVariableDebt
    0,  # liquidityRate
    _VARIABLE_RATE_RAY,  # variableBorrowRate
    0,  # stableBorrowRate
    0,  # averageStableBorrowRate
    0,  # liquidityIndex
    0,  # variableBorrowIndex
    0,  # lastUpdateTimestamp
)


def _make_pool_reserve_data(debt_ceiling: int = 0, isolated_debt: int = 0) -> tuple:
    """Build a 15-field Pool.getReserveData return tuple with custom isolation mode values."""
    # Pack debt_ceiling into configuration bitmap at bits 212-255
    configuration = debt_ceiling << 212
    return (
        configuration,  # [0]  configuration
        0,  # [1]  liquidityIndex
        0,  # [2]  currentLiquidityRate
        0,  # [3]  variableBorrowIndex
        0,  # [4]  currentVariableBorrowRate
        0,  # [5]  currentStableBorrowRate
        0,  # [6]  lastUpdateTimestamp
        0,  # [7]  id
        "0x" + "00" * 20,  # [8]  aTokenAddress
        "0x" + "00" * 20,  # [9]  stableDebtTokenAddress
        "0x" + "00" * 20,  # [10] variableDebtTokenAddress
        "0x" + "00" * 20,  # [11] interestRateStrategyAddress
        0,  # [12] accruedToTreasury
        0,  # [13] unbacked
        isolated_debt,  # [14] isolationModeTotalDebt
    )


SAMPLE_USER = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_contracts():
    """Create mock contract objects for Pool, DataProvider, Oracle."""
    pool = MagicMock()
    data_provider = MagicMock()
    oracle = MagicMock()

    # Default: getUserAccountData returns SAMPLE_ACCOUNT_DATA
    pool.functions.getUserAccountData.return_value.call = AsyncMock(
        return_value=SAMPLE_ACCOUNT_DATA
    )
    # Default: DataProvider.getReserveData returns SAMPLE_DP_RESERVE_DATA
    data_provider.functions.getReserveData.return_value.call = AsyncMock(
        return_value=SAMPLE_DP_RESERVE_DATA
    )
    # Default: Pool.getReserveData returns non-isolation data
    pool.functions.getReserveData.return_value.call = AsyncMock(
        return_value=_make_pool_reserve_data()
    )
    # Default: getAssetPrice returns BNB at $612.34
    oracle.functions.getAssetPrice.return_value.call = AsyncMock(return_value=61234000000)
    # Default: getAssetsPrices returns [BNB, USDT]
    oracle.functions.getAssetsPrices.return_value.call = AsyncMock(
        return_value=[61234000000, 100000000]
    )
    # Default: FLASHLOAN_PREMIUM_TOTAL returns 5 bps
    pool.functions.FLASHLOAN_PREMIUM_TOTAL.return_value.call = AsyncMock(return_value=5)
    # encodeABI for encode methods
    pool.encodeABI = MagicMock(return_value="0xdeadbeef")

    return pool, data_provider, oracle


@pytest.fixture
def aave_client(mock_contracts):
    """Create an AaveClient with mocked web3 and config."""
    pool, data_provider, oracle = mock_contracts
    mock_w3 = MagicMock(spec_set=["eth"])
    mock_w3.eth = MagicMock()

    # Map contract creation to return our mocks based on address
    call_count = {"n": 0}
    contracts = [pool, data_provider, oracle]

    def _contract_factory(**kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return contracts[idx]

    mock_w3.eth.contract = MagicMock(side_effect=_contract_factory)

    with (
        patch("execution.aave_client.get_config") as mock_cfg,
        patch("execution.aave_client.setup_module_logger") as mock_logger,
    ):
        mock_loader = MagicMock()
        mock_loader.get_abi.return_value = []
        mock_loader.get_aave_config.return_value = {
            "flash_loan_premium_bps": 5,
            "flash_loan_premium": "0.0005",
        }
        mock_cfg.return_value = mock_loader
        mock_logger.return_value = MagicMock()

        from execution.aave_client import AaveClient

        client = AaveClient(mock_w3)

    return client


# ---------------------------------------------------------------------------
# A. get_user_account_data tests
# ---------------------------------------------------------------------------


class TestGetUserAccountData:

    async def test_normal_conversion(self, aave_client):
        result = await aave_client.get_user_account_data(SAMPLE_USER)
        assert result.total_collateral_usd == Decimal("500")
        assert result.total_debt_usd == Decimal("250")
        assert result.available_borrow_usd == Decimal("100")
        assert result.current_liquidation_threshold == Decimal("0.8")
        assert result.ltv == Decimal("0.75")
        assert result.health_factor == Decimal("1.8")

    async def test_zero_position(self, aave_client):
        aave_client._pool.functions.getUserAccountData.return_value.call = AsyncMock(
            return_value=(0, 0, 0, 0, 0, 0)
        )
        result = await aave_client.get_user_account_data(SAMPLE_USER)
        assert result.total_collateral_usd == Decimal("0")
        assert result.total_debt_usd == Decimal("0")
        assert result.health_factor == Decimal("0")

    async def test_max_health_factor(self, aave_client):
        """When debt=0 but collateral exists, Aave returns max uint256 for HF."""
        max_uint = 2**256 - 1
        aave_client._pool.functions.getUserAccountData.return_value.call = AsyncMock(
            return_value=(1000 * 10**8, 0, 500 * 10**8, 8000, 7500, max_uint)
        )
        result = await aave_client.get_user_account_data(SAMPLE_USER)
        assert result.health_factor == Decimal(max_uint) / WAD
        assert result.total_collateral_usd == Decimal("1000")

    async def test_low_health_factor(self, aave_client):
        """HF near liquidation threshold (1.05)."""
        hf_wad = int(Decimal("1.05") * WAD)
        aave_client._pool.functions.getUserAccountData.return_value.call = AsyncMock(
            return_value=(2000 * 10**8, 1500 * 10**8, 0, 8000, 7500, hf_wad)
        )
        result = await aave_client.get_user_account_data(SAMPLE_USER)
        assert result.health_factor == Decimal("1.05")

    async def test_rpc_error_raises_aave_client_error(self, aave_client):
        from execution.aave_client import AaveClientError

        aave_client._pool.functions.getUserAccountData.return_value.call = AsyncMock(
            side_effect=ConnectionError("RPC timeout")
        )
        with pytest.raises(AaveClientError, match="getUserAccountData failed"):
            await aave_client.get_user_account_data(SAMPLE_USER)


# ---------------------------------------------------------------------------
# B. get_reserve_data tests
# ---------------------------------------------------------------------------

SAMPLE_ASSET = "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"  # WBNB


class TestGetReserveData:

    async def test_normal_conversion(self, aave_client):
        result = await aave_client.get_reserve_data(SAMPLE_ASSET)
        # 3.5% APR from RAY
        assert result.variable_borrow_rate == Decimal(_VARIABLE_RATE_RAY) / RAY * 100
        # Utilization = (1000 + 7000) / 10000 = 0.8
        assert result.utilization_rate == Decimal("0.8")
        assert result.isolation_mode_enabled is False
        assert result.debt_ceiling == Decimal("0")
        assert result.current_isolated_debt == Decimal("0")

    async def test_isolation_mode_enabled(self, aave_client):
        """Non-zero debt ceiling in bitmap means isolation mode."""
        # Set debt ceiling to 1_000_000 (in 2-decimal units)
        aave_client._pool.functions.getReserveData.return_value.call = AsyncMock(
            return_value=_make_pool_reserve_data(debt_ceiling=1_000_000, isolated_debt=500_000)
        )
        result = await aave_client.get_reserve_data(SAMPLE_ASSET)
        assert result.isolation_mode_enabled is True
        assert result.debt_ceiling == Decimal("1000000")
        assert result.current_isolated_debt == Decimal("500000")

    async def test_no_isolation_mode(self, aave_client):
        """Zero debt ceiling means no isolation mode."""
        result = await aave_client.get_reserve_data(SAMPLE_ASSET)
        assert result.isolation_mode_enabled is False
        assert result.debt_ceiling == Decimal("0")

    async def test_zero_liquidity(self, aave_client):
        """When totalAToken=0, utilization should be 0 (no division error)."""
        zero_liquidity = list(SAMPLE_DP_RESERVE_DATA)
        zero_liquidity[2] = 0  # totalAToken = 0
        aave_client._data_provider.functions.getReserveData.return_value.call = AsyncMock(
            return_value=tuple(zero_liquidity)
        )
        result = await aave_client.get_reserve_data(SAMPLE_ASSET)
        assert result.utilization_rate == Decimal("0")


# ---------------------------------------------------------------------------
# C. get_asset_price tests
# ---------------------------------------------------------------------------


class TestGetAssetPrice:

    async def test_bnb_price(self, aave_client):
        result = await aave_client.get_asset_price(SAMPLE_ASSET)
        assert result == Decimal("612.34")

    async def test_stablecoin_price(self, aave_client):
        aave_client._oracle.functions.getAssetPrice.return_value.call = AsyncMock(
            return_value=100000000
        )
        result = await aave_client.get_asset_price(
            "0x55d398326f99059fF775485246999027B3197955"  # USDT
        )
        assert result == Decimal("1")

    async def test_rpc_error(self, aave_client):
        from execution.aave_client import AaveClientError

        aave_client._oracle.functions.getAssetPrice.return_value.call = AsyncMock(
            side_effect=Exception("Oracle unreachable")
        )
        with pytest.raises(AaveClientError, match="getAssetPrice failed"):
            await aave_client.get_asset_price(SAMPLE_ASSET)


# ---------------------------------------------------------------------------
# D. get_assets_prices tests
# ---------------------------------------------------------------------------


class TestGetAssetsPrices:

    async def test_batch_prices(self, aave_client):
        result = await aave_client.get_assets_prices([SAMPLE_ASSET, SAMPLE_USER])
        assert result == [Decimal("612.34"), Decimal("1")]


# ---------------------------------------------------------------------------
# E. get_flash_loan_premium tests
# ---------------------------------------------------------------------------


class TestGetFlashLoanPremium:

    async def test_from_chain(self, aave_client):
        result = await aave_client.get_flash_loan_premium()
        assert result == Decimal("0.0005")

    async def test_fallback_on_error(self, aave_client):
        aave_client._pool.functions.FLASHLOAN_PREMIUM_TOTAL.return_value.call = AsyncMock(
            side_effect=Exception("Method not found")
        )
        result = await aave_client.get_flash_loan_premium()
        assert result == DEFAULT_FLASH_LOAN_PREMIUM


# ---------------------------------------------------------------------------
# F. Encode method tests
# ---------------------------------------------------------------------------

RECEIVER = "0x1234567890abcdef1234567890abcdef12345678"
ON_BEHALF = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"


class TestEncodeMethods:

    def test_encode_flash_loan(self, aave_client):
        result = aave_client.encode_flash_loan(
            receiver=RECEIVER,
            assets=[SAMPLE_ASSET],
            amounts=[10**18],
            modes=[0],
            on_behalf_of=ON_BEHALF,
            params=b"\x00",
        )
        assert isinstance(result, str)
        aave_client._pool.encodeABI.assert_called_once()
        call_args = aave_client._pool.encodeABI.call_args
        assert call_args.kwargs["fn_name"] == "flashLoan"

    def test_encode_supply(self, aave_client):
        result = aave_client.encode_supply(
            asset=SAMPLE_ASSET,
            amount=10**18,
            on_behalf_of=ON_BEHALF,
        )
        assert isinstance(result, str)
        call_args = aave_client._pool.encodeABI.call_args
        assert call_args.kwargs["fn_name"] == "supply"

    def test_encode_withdraw(self, aave_client):
        result = aave_client.encode_withdraw(
            asset=SAMPLE_ASSET,
            amount=10**18,
            to=ON_BEHALF,
        )
        assert isinstance(result, str)
        call_args = aave_client._pool.encodeABI.call_args
        assert call_args.kwargs["fn_name"] == "withdraw"

    def test_encode_repay(self, aave_client):
        result = aave_client.encode_repay(
            asset=SAMPLE_ASSET,
            amount=10**18,
            rate_mode=2,
            on_behalf_of=ON_BEHALF,
        )
        assert isinstance(result, str)
        call_args = aave_client._pool.encodeABI.call_args
        assert call_args.kwargs["fn_name"] == "repay"


# ---------------------------------------------------------------------------
# G. Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_constructor_loads_correct_abis(self):
        """Verify constructor calls get_abi with the right names."""
        mock_w3 = MagicMock(spec_set=["eth"])
        mock_w3.eth = MagicMock()
        mock_w3.eth.contract = MagicMock(return_value=MagicMock())

        with (
            patch("execution.aave_client.get_config") as mock_cfg,
            patch("execution.aave_client.setup_module_logger"),
        ):
            mock_loader = MagicMock()
            mock_loader.get_abi.return_value = []
            mock_loader.get_aave_config.return_value = {}
            mock_cfg.return_value = mock_loader

            from execution.aave_client import AaveClient

            AaveClient(mock_w3)

            abi_calls = [c.args[0] for c in mock_loader.get_abi.call_args_list]
            assert "aave_v3_pool" in abi_calls
            assert "aave_v3_data_provider" in abi_calls
            assert "aave_v3_oracle" in abi_calls

    async def test_lowercase_address_works(self, aave_client):
        """Lowercase addresses are checksum-converted internally."""
        lower_addr = SAMPLE_USER.lower()
        result = await aave_client.get_user_account_data(lower_addr)
        assert result.total_collateral_usd == Decimal("500")
