"""
Shared constants for BSC Leverage Bot.

Protocol addresses, numeric constants, and default values used across all modules.
"""

from decimal import Decimal

# ---------------------------------------------------------------------------
# Numeric Constants
# ---------------------------------------------------------------------------

WAD = Decimal("1_000_000_000_000_000_000")  # 1e18 (Aave HF, token amounts)
RAY = Decimal("1_000_000_000_000_000_000_000_000_000")  # 1e27 (Aave rates)
USD_DECIMALS = Decimal("100_000_000")  # 1e8 (Chainlink USD price feeds)
SECONDS_PER_YEAR = 31_536_000

# ---------------------------------------------------------------------------
# Flash Loan Modes (Aave V3)
# ---------------------------------------------------------------------------

FLASH_LOAN_MODE_NO_DEBT = 0  # Must repay in same tx (for closing positions)
FLASH_LOAN_MODE_VARIABLE_DEBT = 2  # Debt stays open (for opening positions)

# ---------------------------------------------------------------------------
# Aave V3 BSC Addresses
# ---------------------------------------------------------------------------

AAVE_V3_POOL = "0x6807dc923806fE8Fd134338EABCA509979a7e0cB"
AAVE_V3_POOL_ADDRESSES_PROVIDER = "0xff75B6da14FfbbfD355Daf7a2731456b3562Ba6D"
AAVE_V3_ORACLE = "0x39bc1bfDa2130d6Bb6DBEfd366939b4c7aa7C697"
AAVE_V3_DATA_PROVIDER = "0xc90Df74A7c16245c5F5C5870327Ceb38Fe5d5328"
AAVE_V3_ACL_MANAGER = "0x2D97F8FA96886Fd923c065F5457F9DDd494e3877"

# ---------------------------------------------------------------------------
# Chainlink BSC Feed Addresses
# ---------------------------------------------------------------------------

CHAINLINK_BNB_USD = "0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE"
CHAINLINK_BTC_USD = "0x264990fbd0A4796A3E3d8E37C4d5F87a3aCa5Ebf"
CHAINLINK_ETH_USD = "0x9ef1B8c0E4F7dc8bF5719Ea496883DC6401d5b2e"

# ---------------------------------------------------------------------------
# BSC Token Addresses
# ---------------------------------------------------------------------------

TOKEN_WBNB = "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"
TOKEN_USDT = "0x55d398326f99059fF775485246999027B3197955"
TOKEN_USDC = "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d"
TOKEN_BTCB = "0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c"
TOKEN_ETH = "0x2170Ed0880ac9A755fd29B2688956BD959F933F8"
TOKEN_FDUSD = "0xc5f0f7b66764F6ec8C8Dff7BA683102295E16409"
TOKEN_CAKE = "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82"

# ---------------------------------------------------------------------------
# BSC Infrastructure
# ---------------------------------------------------------------------------

MULTICALL3 = "0xcA11bde05977b3631167028862bE2a173976CA11"
MEV_PROTECTED_RPC = "https://rpc.48.club"

# ---------------------------------------------------------------------------
# Aave V3 Risk Parameters
# ---------------------------------------------------------------------------

CLOSE_FACTOR_THRESHOLD_USD = Decimal("2000")
DEFAULT_FLASH_LOAN_PREMIUM_BPS = 5
DEFAULT_FLASH_LOAN_PREMIUM = Decimal("0.0005")

# ---------------------------------------------------------------------------
# Default Safety Values
# ---------------------------------------------------------------------------

DEFAULT_DRY_RUN = True
DEFAULT_MAX_POSITION_USD = Decimal("10000")
DEFAULT_MAX_LEVERAGE_RATIO = Decimal("3.0")
DEFAULT_MIN_HEALTH_FACTOR = Decimal("1.5")
DEFAULT_MAX_GAS_PRICE_GWEI = 10
DEFAULT_MAX_SLIPPAGE_BPS = 50
DEFAULT_COOLDOWN_SECONDS = 30
DEFAULT_MAX_TX_PER_24H = 50
