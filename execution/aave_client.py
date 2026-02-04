"""
Aave V3 BSC client — thin read + encode wrapper.

Reads on-chain state (user positions, reserve data, oracle prices) via async calls
and encodes calldata for Aave operations (flash loans, supply, withdraw, repay).
No transaction submission — that is handled by TxSubmitter (Phase 5).

Usage:
    from web3 import AsyncWeb3, AsyncHTTPProvider
    from execution.aave_client import AaveClient

    w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url))
    client = AaveClient(w3)
    account = await client.get_user_account_data(user_address)
"""

from __future__ import annotations

from decimal import Decimal

from web3 import AsyncWeb3, Web3

from bot_logging.logger_manager import setup_module_logger
from config.loader import get_config
from shared.constants import (
    AAVE_V3_DATA_PROVIDER,
    AAVE_V3_ORACLE,
    AAVE_V3_POOL,
    DEFAULT_FLASH_LOAN_PREMIUM,
    RAY,
    USD_DECIMALS,
    WAD,
)
from shared.types import ReserveData, UserAccountData

# Aave V3 ReserveConfiguration bitmap constants
_DEBT_CEILING_START_BIT = 212
_DEBT_CEILING_MASK = (1 << 44) - 1  # 44-bit mask

_BPS = Decimal("10000")


class AaveClientError(Exception):
    """Raised when an Aave V3 contract call fails."""


class AaveClient:
    """
    Async read + sync encode wrapper for Aave V3 contracts on BSC.

    Accepts an AsyncWeb3 instance via dependency injection so the same
    connection can be shared across clients. All read methods are async
    (RPC calls). All encode methods are sync (local ABI encoding only).
    """

    def __init__(self, w3: AsyncWeb3) -> None:
        self._w3 = w3
        cfg = get_config()

        self._pool = w3.eth.contract(
            address=Web3.to_checksum_address(AAVE_V3_POOL),
            abi=cfg.get_abi("aave_v3_pool"),
        )
        self._data_provider = w3.eth.contract(
            address=Web3.to_checksum_address(AAVE_V3_DATA_PROVIDER),
            abi=cfg.get_abi("aave_v3_data_provider"),
        )
        self._oracle = w3.eth.contract(
            address=Web3.to_checksum_address(AAVE_V3_ORACLE),
            abi=cfg.get_abi("aave_v3_oracle"),
        )

        self._aave_config = cfg.get_aave_config()
        self._logger = setup_module_logger(
            "aave_client", "aave_client.log", module_folder="Aave_Client_Logs"
        )

    # ------------------------------------------------------------------
    # Read operations (async — RPC calls)
    # ------------------------------------------------------------------

    async def get_user_account_data(self, user: str) -> UserAccountData:
        """
        Query Aave V3 Pool for a user's aggregate position data.

        Returns UserAccountData with USD values, LTV/LT ratios, and health factor.
        All raw Aave values are converted from WAD (1e18) / basis points / USD-8-decimals.
        """
        try:
            checksum = Web3.to_checksum_address(user)
            result = await self._pool.functions.getUserAccountData(checksum).call()
            return UserAccountData(
                total_collateral_usd=Decimal(result[0]) / USD_DECIMALS,
                total_debt_usd=Decimal(result[1]) / USD_DECIMALS,
                available_borrow_usd=Decimal(result[2]) / USD_DECIMALS,
                current_liquidation_threshold=Decimal(result[3]) / _BPS,
                ltv=Decimal(result[4]) / _BPS,
                health_factor=Decimal(result[5]) / WAD,
            )
        except AaveClientError:
            raise
        except Exception as e:
            self._logger.error("Failed to get user account data for %s: %s", user, e)
            raise AaveClientError(f"getUserAccountData failed: {e}") from e

    async def get_reserve_data(self, asset: str) -> ReserveData:
        """
        Query reserve data for an asset from both DataProvider and Pool.

        Uses DataProvider for borrow rate and utilization (flat tuple),
        and Pool for isolation mode info (configuration bitmap).
        """
        try:
            checksum = Web3.to_checksum_address(asset)

            # DataProvider.getReserveData → 12 flat values
            dp_result = await self._data_provider.functions.getReserveData(checksum).call()
            variable_borrow_rate = Decimal(dp_result[6]) / RAY * Decimal("100")

            total_atoken = Decimal(dp_result[2])
            total_stable_debt = Decimal(dp_result[3])
            total_variable_debt = Decimal(dp_result[4])
            total_debt = total_stable_debt + total_variable_debt

            utilization = total_debt / total_atoken if total_atoken > 0 else Decimal("0")

            # Pool.getReserveData → 15-field struct for isolation mode bitmap
            pool_result = await self._pool.functions.getReserveData(checksum).call()
            configuration = pool_result[0]
            debt_ceiling_raw = (configuration >> _DEBT_CEILING_START_BIT) & _DEBT_CEILING_MASK
            isolation_mode_enabled = debt_ceiling_raw > 0
            current_isolated_debt = Decimal(pool_result[14])

            return ReserveData(
                variable_borrow_rate=variable_borrow_rate,
                utilization_rate=utilization,
                isolation_mode_enabled=isolation_mode_enabled,
                debt_ceiling=Decimal(debt_ceiling_raw),
                current_isolated_debt=current_isolated_debt,
            )
        except AaveClientError:
            raise
        except Exception as e:
            self._logger.error("Failed to get reserve data for %s: %s", asset, e)
            raise AaveClientError(f"getReserveData failed for {asset}: {e}") from e

    async def get_asset_price(self, asset: str) -> Decimal:
        """
        Get asset price in USD from Aave Oracle (Chainlink 8-decimal format).
        """
        try:
            checksum = Web3.to_checksum_address(asset)
            raw_price = await self._oracle.functions.getAssetPrice(checksum).call()
            price = Decimal(raw_price) / USD_DECIMALS
            self._logger.debug("Asset price for %s: $%s", asset, price)
            return price
        except AaveClientError:
            raise
        except Exception as e:
            self._logger.error("Failed to get asset price for %s: %s", asset, e)
            raise AaveClientError(f"getAssetPrice failed for {asset}: {e}") from e

    async def get_assets_prices(self, assets: list[str]) -> list[Decimal]:
        """
        Get multiple asset prices in a single batched Oracle call.
        """
        try:
            checksums = [Web3.to_checksum_address(a) for a in assets]
            raw_prices = await self._oracle.functions.getAssetsPrices(checksums).call()
            return [Decimal(p) / USD_DECIMALS for p in raw_prices]
        except AaveClientError:
            raise
        except Exception as e:
            self._logger.error("Failed to get assets prices: %s", e)
            raise AaveClientError(f"getAssetsPrices failed: {e}") from e

    async def get_flash_loan_premium(self) -> Decimal:
        """
        Get flash loan premium from Pool contract, with config fallback.

        Returns the premium as a ratio (e.g. Decimal("0.0005") for 5 bps).
        """
        try:
            raw_premium = await self._pool.functions.FLASHLOAN_PREMIUM_TOTAL().call()
            premium = Decimal(raw_premium) / _BPS
            self._logger.debug("Flash loan premium: %s (%s bps)", premium, raw_premium)
            return premium
        except Exception as e:
            self._logger.warning(
                "Failed to query flash loan premium from chain, using config fallback: %s", e
            )
            return DEFAULT_FLASH_LOAN_PREMIUM

    # ------------------------------------------------------------------
    # Encode operations (sync — local ABI encoding, no RPC)
    # ------------------------------------------------------------------

    def encode_flash_loan(
        self,
        receiver: str,
        assets: list[str],
        amounts: list[int],
        modes: list[int],
        on_behalf_of: str,
        params: bytes,
        referral: int = 0,
    ) -> str:
        """
        Encode calldata for Pool.flashLoan().

        Args:
            receiver: Contract implementing IFlashLoanReceiver (LeverageExecutor).
            assets: Token addresses to flash-borrow.
            amounts: Amounts in native token decimals (wei).
            modes: 0 = repay same tx (5 bps fee), 2 = variable debt stays open.
            on_behalf_of: Address to open debt for (usually receiver).
            params: Arbitrary bytes passed to executeOperation callback.
            referral: Referral code (default 0).

        Returns:
            Hex-encoded calldata string.
        """
        try:
            return self._pool.encodeABI(
                fn_name="flashLoan",
                args=[
                    Web3.to_checksum_address(receiver),
                    [Web3.to_checksum_address(a) for a in assets],
                    amounts,
                    modes,
                    Web3.to_checksum_address(on_behalf_of),
                    params,
                    referral,
                ],
            )
        except Exception as e:
            self._logger.error("Failed to encode flashLoan: %s", e)
            raise AaveClientError(f"encode flashLoan failed: {e}") from e

    def encode_supply(
        self,
        asset: str,
        amount: int,
        on_behalf_of: str,
        referral: int = 0,
    ) -> str:
        """Encode calldata for Pool.supply()."""
        try:
            return self._pool.encodeABI(
                fn_name="supply",
                args=[
                    Web3.to_checksum_address(asset),
                    amount,
                    Web3.to_checksum_address(on_behalf_of),
                    referral,
                ],
            )
        except Exception as e:
            self._logger.error("Failed to encode supply: %s", e)
            raise AaveClientError(f"encode supply failed: {e}") from e

    def encode_withdraw(self, asset: str, amount: int, to: str) -> str:
        """Encode calldata for Pool.withdraw()."""
        try:
            return self._pool.encodeABI(
                fn_name="withdraw",
                args=[
                    Web3.to_checksum_address(asset),
                    amount,
                    Web3.to_checksum_address(to),
                ],
            )
        except Exception as e:
            self._logger.error("Failed to encode withdraw: %s", e)
            raise AaveClientError(f"encode withdraw failed: {e}") from e

    def encode_repay(
        self,
        asset: str,
        amount: int,
        rate_mode: int,
        on_behalf_of: str,
    ) -> str:
        """Encode calldata for Pool.repay()."""
        try:
            return self._pool.encodeABI(
                fn_name="repay",
                args=[
                    Web3.to_checksum_address(asset),
                    amount,
                    rate_mode,
                    Web3.to_checksum_address(on_behalf_of),
                ],
            )
        except Exception as e:
            self._logger.error("Failed to encode repay: %s", e)
            raise AaveClientError(f"encode repay failed: {e}") from e
