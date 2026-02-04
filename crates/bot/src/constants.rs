use alloy::primitives::{address, Address};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

// ---------------------------------------------------------------------------
// Numeric Constants
// ---------------------------------------------------------------------------

/// WAD: 1e18 — standard EVM fixed-point scale for amounts, prices, health factors.
pub const WAD: Decimal = dec!(1_000_000_000_000_000_000);

/// RAY: 1e27 — Aave interest rate scale.
pub const RAY: Decimal = dec!(1_000_000_000_000_000_000_000_000_000);

/// USD_DECIMALS: 1e8 — Chainlink price feed scale / Aave base currency unit.
pub const USD_DECIMALS: Decimal = dec!(100_000_000);

/// Seconds in a non-leap year.
pub const SECONDS_PER_YEAR: u64 = 31_536_000;

// ---------------------------------------------------------------------------
// Flash Loan Modes (Aave V3)
// ---------------------------------------------------------------------------

/// Mode 0: must repay flash-loaned amount + premium in the same transaction.
pub const FLASH_LOAN_MODE_NO_DEBT: u8 = 0;

/// Mode 2: flash-loaned amount becomes variable-rate debt (no in-tx repayment).
pub const FLASH_LOAN_MODE_VARIABLE_DEBT: u8 = 2;

// ---------------------------------------------------------------------------
// Aave V3 BSC Addresses
// ---------------------------------------------------------------------------

pub const AAVE_V3_POOL: Address = address!("6807dc923806fE8Fd134338EABCA509979a7e0cB");
pub const AAVE_V3_POOL_ADDRESSES_PROVIDER: Address =
    address!("ff75B6da14FfbbfD355Daf7a2731456b3562Ba6D");
pub const AAVE_V3_ORACLE: Address = address!("39bc1bfDa2130d6Bb6DBEfd366939b4c7aa7C697");
pub const AAVE_V3_DATA_PROVIDER: Address =
    address!("c90Df74A7c16245c5F5C5870327Ceb38Fe5d5328");
pub const AAVE_V3_ACL_MANAGER: Address =
    address!("2D97F8FA96886Fd923c065F5457F9DDd494e3877");

// ---------------------------------------------------------------------------
// Chainlink BSC Feed Addresses
// ---------------------------------------------------------------------------

pub const CHAINLINK_BNB_USD: Address = address!("0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE");
pub const CHAINLINK_BTC_USD: Address = address!("264990fbd0A4796A3E3d8E37C4d5F87a3aCa5Ebf");
pub const CHAINLINK_ETH_USD: Address = address!("9ef1B8c0E4F7dc8bF5719Ea496883DC6401d5b2e");

// ---------------------------------------------------------------------------
// BSC Token Addresses
// ---------------------------------------------------------------------------

pub const TOKEN_WBNB: Address = address!("bb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c");
pub const TOKEN_USDT: Address = address!("55d398326f99059fF775485246999027B3197955");
pub const TOKEN_USDC: Address = address!("8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d");
pub const TOKEN_BTCB: Address = address!("7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c");
pub const TOKEN_ETH: Address = address!("2170Ed0880ac9A755fd29B2688956BD959F933F8");
pub const TOKEN_FDUSD: Address = address!("c5f0f7b66764F6ec8C8Dff7BA683102295E16409");
pub const TOKEN_CAKE: Address = address!("0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82");

// ---------------------------------------------------------------------------
// BSC Infrastructure
// ---------------------------------------------------------------------------

pub const MULTICALL3: Address = address!("cA11bde05977b3631167028862bE2a173976CA11");
pub const MEV_PROTECTED_RPC: &str = "https://rpc.48.club";

// ---------------------------------------------------------------------------
// Default Safety Values
// ---------------------------------------------------------------------------

pub const DEFAULT_DRY_RUN: bool = true;
pub const DEFAULT_MAX_POSITION_USD: Decimal = dec!(10_000);
pub const DEFAULT_MAX_LEVERAGE_RATIO: Decimal = dec!(3.0);
pub const DEFAULT_MIN_HEALTH_FACTOR: Decimal = dec!(1.5);
pub const DEFAULT_MAX_GAS_PRICE_GWEI: u64 = 10;
pub const DEFAULT_MAX_SLIPPAGE_BPS: u32 = 50;
pub const DEFAULT_COOLDOWN_SECONDS: u64 = 30;
pub const DEFAULT_MAX_TX_PER_24H: u32 = 50;

/// Aave V3 close factor threshold: positions below $2,000 face 100% liquidation.
pub const CLOSE_FACTOR_THRESHOLD_USD: Decimal = dec!(2_000);

pub const DEFAULT_FLASH_LOAN_PREMIUM_BPS: u32 = 5;
pub const DEFAULT_FLASH_LOAN_PREMIUM: Decimal = dec!(0.0005);
