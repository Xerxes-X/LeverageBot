//! Aave V3 BSC client — typed read + calldata encoding layer.
//!
//! Reads on-chain state (user positions, reserve data, oracle prices) via
//! async calls and encodes calldata for Aave operations (flash loans, supply,
//! withdraw, repay). No transaction submission — that is handled by
//! TxSubmitter (Phase 6).
//!
//! Key difference from Python: uses Alloy `sol!` macro for compile-time ABI
//! generation instead of runtime JSON ABI parsing via web3.py.

use alloy::primitives::{Address, Bytes, U256};
use alloy::providers::RootProvider;
use anyhow::{Context, Result};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error, warn};

use crate::config::{AaveConfig, ChainConfig};
use crate::constants::DEFAULT_FLASH_LOAN_PREMIUM;
use crate::types::wad_ray::{bps_to_decimal, price_to_decimal, ray_to_decimal, wad_to_decimal};
use crate::types::{ReserveData, UserAccountData};

use super::contracts::{IAaveOracle, IAggregatorV3, IPool, IPoolDataProvider};

/// Concrete provider type: Alloy HTTP provider over Ethereum network.
pub type HttpProvider = RootProvider;

// Aave V3 ReserveConfiguration bitmap: debt ceiling starts at bit 212, 44 bits wide.
const DEBT_CEILING_START_BIT: u32 = 212;
const DEBT_CEILING_MASK: u64 = (1u64 << 44) - 1;

/// Parameters for encoding a flash loan call.
pub struct FlashLoanParams {
    pub receiver: Address,
    pub assets: Vec<Address>,
    pub amounts: Vec<U256>,
    pub modes: Vec<U256>,
    pub on_behalf_of: Address,
    pub params: Bytes,
    pub referral_code: u16,
}

/// Async read + sync encode wrapper for Aave V3 contracts on BSC.
///
/// Accepts an Alloy HTTP provider via dependency injection so the same
/// connection can be shared across clients. All read methods are async
/// (RPC calls). All encode methods are sync (local ABI encoding only).
pub struct AaveClient {
    pool: IPool::IPoolInstance<HttpProvider>,
    data_provider: IPoolDataProvider::IPoolDataProviderInstance<HttpProvider>,
    oracle: IAaveOracle::IAaveOracleInstance<HttpProvider>,
    config: AaveConfig,
}

impl AaveClient {
    /// Construct with an Alloy HTTP provider and config references.
    pub fn new(provider: HttpProvider, aave_config: &AaveConfig, chain_config: &ChainConfig) -> Self {
        let pool_addr: Address = chain_config
            .contracts
            .aave_v3_pool
            .parse()
            .expect("invalid aave_v3_pool address in config");
        let dp_addr: Address = chain_config
            .contracts
            .aave_v3_data_provider
            .parse()
            .expect("invalid aave_v3_data_provider address in config");
        let oracle_addr: Address = chain_config
            .contracts
            .aave_v3_oracle
            .parse()
            .expect("invalid aave_v3_oracle address in config");

        Self {
            pool: IPool::new(pool_addr, provider.clone()),
            data_provider: IPoolDataProvider::new(dp_addr, provider.clone()),
            oracle: IAaveOracle::new(oracle_addr, provider),
            config: aave_config.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Read operations (async — RPC calls)
    // -----------------------------------------------------------------------

    /// Query Aave V3 Pool for a user's aggregate position data.
    ///
    /// Converts raw on-chain values:
    /// - USD amounts: from 8-decimal base currency to `Decimal`
    /// - LTV/LT: from basis points to fraction
    /// - Health factor: from WAD (1e18) to `Decimal`
    pub async fn get_user_account_data(&self, user: Address) -> Result<UserAccountData> {
        let result = self
            .pool
            .getUserAccountData(user)
            .call()
            .await
            .context("getUserAccountData RPC call failed")?;

        Ok(UserAccountData {
            total_collateral_usd: price_to_decimal(result.totalCollateralBase),
            total_debt_usd: price_to_decimal(result.totalDebtBase),
            available_borrow_usd: price_to_decimal(result.availableBorrowsBase),
            current_liquidation_threshold: bps_to_decimal(result.currentLiquidationThreshold),
            ltv: bps_to_decimal(result.ltv),
            health_factor: wad_to_decimal(result.healthFactor),
        })
    }

    /// Query reserve data for an asset from both DataProvider and Pool.
    ///
    /// Uses DataProvider for borrow rate and utilization (12-field flat tuple),
    /// and Pool for isolation mode info (configuration bitmap).
    pub async fn get_reserve_data(&self, asset: Address) -> Result<ReserveData> {
        // DataProvider.getReserveData → 12 flat values
        let dp = self
            .data_provider
            .getReserveData(asset)
            .call()
            .await
            .context("DataProvider.getReserveData RPC call failed")?;

        let variable_borrow_rate = ray_to_decimal(dp.variableBorrowRate) * dec!(100);

        let total_atoken = Decimal::from(dp.totalAToken.to::<u128>());
        let total_stable_debt = Decimal::from(dp.totalStableDebt.to::<u128>());
        let total_variable_debt = Decimal::from(dp.totalVariableDebt.to::<u128>());
        let total_debt = total_stable_debt + total_variable_debt;

        let utilization = if total_atoken > Decimal::ZERO {
            total_debt / total_atoken
        } else {
            Decimal::ZERO
        };

        // Pool.getReserveData → 15-field struct for isolation mode bitmap
        let pool_data = self
            .pool
            .getReserveData(asset)
            .call()
            .await
            .context("Pool.getReserveData RPC call failed")?;

        let configuration = pool_data.configuration;
        let debt_ceiling_raw =
            (configuration >> DEBT_CEILING_START_BIT) & U256::from(DEBT_CEILING_MASK);
        let isolation_mode_enabled = debt_ceiling_raw > U256::ZERO;
        let current_isolated_debt = Decimal::from(pool_data.isolationModeTotalDebt);

        let result = ReserveData {
            variable_borrow_rate,
            utilization_rate: utilization,
            isolation_mode_enabled,
            debt_ceiling: Decimal::from(debt_ceiling_raw.to::<u128>()),
            current_isolated_debt,
        };

        debug!(
            asset = %asset,
            rate = %variable_borrow_rate,
            utilization = %utilization,
            isolation = isolation_mode_enabled,
            "reserve data"
        );

        Ok(result)
    }

    /// Get asset price in USD from Aave Oracle (Chainlink 8-decimal format).
    pub async fn get_asset_price(&self, asset: Address) -> Result<Decimal> {
        let result = self
            .oracle
            .getAssetPrice(asset)
            .call()
            .await
            .context("getAssetPrice RPC call failed")?;

        let price = price_to_decimal(result);
        debug!(asset = %asset, price = %price, "asset price");
        Ok(price)
    }

    /// Get multiple asset prices in a single batched Oracle call.
    pub async fn get_assets_prices(&self, assets: &[Address]) -> Result<Vec<Decimal>> {
        let result = self
            .oracle
            .getAssetsPrices(assets.to_vec())
            .call()
            .await
            .context("getAssetsPrices RPC call failed")?;

        Ok(result.iter().map(|p| price_to_decimal(*p)).collect())
    }

    /// Get flash loan premium from Pool contract, with config fallback.
    ///
    /// Returns the premium as a ratio (e.g. `0.0005` for 5 bps).
    pub async fn get_flash_loan_premium(&self) -> Decimal {
        match self.pool.FLASHLOAN_PREMIUM_TOTAL().call().await {
            Ok(result) => {
                let premium = Decimal::from(result) / dec!(10_000);
                debug!(premium = %premium, "flash loan premium from chain");
                premium
            }
            Err(e) => {
                warn!(
                    error = %e,
                    fallback = %DEFAULT_FLASH_LOAN_PREMIUM,
                    "failed to query flash loan premium, using config fallback"
                );
                self.config.flash_loan_premium
            }
        }
    }

    // -----------------------------------------------------------------------
    // Oracle freshness (Chainlink direct)
    // -----------------------------------------------------------------------

    /// Check Chainlink feed freshness.
    ///
    /// Returns `true` if the feed data is fresh (age ≤ `max_staleness_seconds`).
    /// Ported from Python, per Deng et al., ICSE 2024, "Safeguarding DeFi
    /// Smart Contracts against Oracle Deviations".
    pub async fn check_oracle_freshness(
        provider: &HttpProvider,
        feed_address: Address,
        max_staleness_seconds: u64,
    ) -> Result<bool> {
        let feed = IAggregatorV3::new(feed_address, provider.clone());
        let data = feed
            .latestRoundData()
            .call()
            .await
            .context("latestRoundData RPC call failed")?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_secs();
        let updated_at = data.updatedAt.to::<u64>();
        let age = now.saturating_sub(updated_at);

        if age > max_staleness_seconds {
            error!(
                age,
                max_staleness_seconds,
                feed = %feed_address,
                "oracle data stale"
            );
            return Ok(false);
        }

        let round_id = data.roundId.to::<u64>();
        let answered_in_round = data.answeredInRound.to::<u64>();
        if answered_in_round < round_id {
            warn!(
                answered_in_round,
                round_id,
                feed = %feed_address,
                "oracle round incomplete"
            );
        }

        Ok(true)
    }

    // -----------------------------------------------------------------------
    // Encode operations (sync — local ABI encoding, no RPC)
    // -----------------------------------------------------------------------

    /// Encode calldata for `Pool.flashLoan()`.
    pub fn encode_flash_loan(params: &FlashLoanParams) -> Bytes {
        let call = IPool::flashLoanCall {
            receiverAddress: params.receiver,
            assets: params.assets.clone(),
            amounts: params.amounts.clone(),
            interestRateModes: params.modes.clone(),
            onBehalfOf: params.on_behalf_of,
            params: params.params.clone(),
            referralCode: params.referral_code,
        };
        Bytes::from(alloy::sol_types::SolCall::abi_encode(&call))
    }

    /// Encode calldata for `Pool.supply()`.
    pub fn encode_supply(
        asset: Address,
        amount: U256,
        on_behalf_of: Address,
        referral_code: u16,
    ) -> Bytes {
        let call = IPool::supplyCall {
            asset,
            amount,
            onBehalfOf: on_behalf_of,
            referralCode: referral_code,
        };
        Bytes::from(alloy::sol_types::SolCall::abi_encode(&call))
    }

    /// Encode calldata for `Pool.withdraw()`.
    pub fn encode_withdraw(asset: Address, amount: U256, to: Address) -> Bytes {
        let call = IPool::withdrawCall { asset, amount, to };
        Bytes::from(alloy::sol_types::SolCall::abi_encode(&call))
    }

    /// Encode calldata for `Pool.repay()`.
    pub fn encode_repay(
        asset: Address,
        amount: U256,
        interest_rate_mode: U256,
        on_behalf_of: Address,
    ) -> Bytes {
        let call = IPool::repayCall {
            asset,
            amount,
            interestRateMode: interest_rate_mode,
            onBehalfOf: on_behalf_of,
        };
        Bytes::from(alloy::sol_types::SolCall::abi_encode(&call))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy::primitives::address;

    #[test]
    fn test_encode_supply_selector() {
        let data = AaveClient::encode_supply(
            address!("bb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"),
            U256::from(1_000_000_000_000_000_000u128),
            address!("0000000000000000000000000000000000000001"),
            0,
        );
        // Pool.supply(address,uint256,address,uint16) selector = 0x617ba037
        assert_eq!(&data[..4], &[0x61, 0x7b, 0xa0, 0x37]);
    }

    #[test]
    fn test_encode_withdraw_selector() {
        let data = AaveClient::encode_withdraw(
            address!("bb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"),
            U256::from(1_000_000_000_000_000_000u128),
            address!("0000000000000000000000000000000000000001"),
        );
        // Pool.withdraw(address,uint256,address) selector = 0x69328dec
        assert_eq!(&data[..4], &[0x69, 0x32, 0x8d, 0xec]);
    }

    #[test]
    fn test_encode_repay_selector() {
        let data = AaveClient::encode_repay(
            address!("bb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"),
            U256::from(1_000_000_000_000_000_000u128),
            U256::from(2u64), // variable rate mode
            address!("0000000000000000000000000000000000000001"),
        );
        // Pool.repay(address,uint256,uint256,address) selector = 0x573ade81
        assert_eq!(&data[..4], &[0x57, 0x3a, 0xde, 0x81]);
    }

    #[test]
    fn test_encode_flash_loan_selector() {
        let params = FlashLoanParams {
            receiver: address!("0000000000000000000000000000000000000001"),
            assets: vec![address!("bb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c")],
            amounts: vec![U256::from(1_000_000_000_000_000_000u128)],
            modes: vec![U256::from(0u64)],
            on_behalf_of: address!("0000000000000000000000000000000000000001"),
            params: Bytes::new(),
            referral_code: 0,
        };
        let data = AaveClient::encode_flash_loan(&params);
        // Pool.flashLoan(...) selector = 0xab9c4b5d
        assert_eq!(&data[..4], &[0xab, 0x9c, 0x4b, 0x5d]);
    }

    #[test]
    fn test_encode_supply_roundtrip() {
        let asset = address!("bb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c");
        let amount = U256::from(5_000_000_000_000_000_000u128);
        let on_behalf_of = address!("1234567890123456789012345678901234567890");

        let data = AaveClient::encode_supply(asset, amount, on_behalf_of, 0);

        // Decode the calldata back to verify round-trip correctness.
        let decoded =
            <IPool::supplyCall as alloy::sol_types::SolCall>::abi_decode(&data).unwrap();
        assert_eq!(decoded.asset, asset);
        assert_eq!(decoded.amount, amount);
        assert_eq!(decoded.onBehalfOf, on_behalf_of);
        assert_eq!(decoded.referralCode, 0);
    }

    #[test]
    fn test_encode_flash_loan_roundtrip() {
        let receiver = address!("0000000000000000000000000000000000000001");
        let asset = address!("bb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c");
        let amount = U256::from(10_000_000_000_000_000_000u128);
        let mode = U256::from(2u64); // keep as variable debt
        let on_behalf = address!("0000000000000000000000000000000000000002");
        let inner_params = Bytes::from(vec![0x01, 0x02, 0x03]);

        let params = FlashLoanParams {
            receiver,
            assets: vec![asset],
            amounts: vec![amount],
            modes: vec![mode],
            on_behalf_of: on_behalf,
            params: inner_params.clone(),
            referral_code: 42,
        };
        let data = AaveClient::encode_flash_loan(&params);

        let decoded =
            <IPool::flashLoanCall as alloy::sol_types::SolCall>::abi_decode(&data).unwrap();
        assert_eq!(decoded.receiverAddress, receiver);
        assert_eq!(decoded.assets, vec![asset]);
        assert_eq!(decoded.amounts, vec![amount]);
        assert_eq!(decoded.interestRateModes, vec![mode]);
        assert_eq!(decoded.onBehalfOf, on_behalf);
        assert_eq!(decoded.params.as_ref(), inner_params.as_ref());
        assert_eq!(decoded.referralCode, 42);
    }
}
