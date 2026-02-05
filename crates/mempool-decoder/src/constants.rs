//! Monitored DEX routers, swap function selectors, and token addresses for BSC mainnet.

use alloy::primitives::{address, Address};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

// ---------------------------------------------------------------------------
// Monitored DEX Routers (12 total)
// ---------------------------------------------------------------------------

pub const PANCAKESWAP_V2: Address = address!("10ED43C718714eb63d5aA57B78B54704E256024E");
pub const PANCAKESWAP_V3_SWAP: Address = address!("1b81D678ffb9C0263b24A97847620C99d213eB14");
pub const PANCAKESWAP_V3_SMART: Address = address!("13f4EA83D0bd40E75C8222255bc855a974568Dd4");
pub const PANCAKESWAP_UNIVERSAL: Address = address!("1a0a18ac4BECDDbd6389559687d1a73d8927e416");
pub const PANCAKESWAP_UNIVERSAL_2: Address = address!("d9c500dff816a1da21a48a732d3498bf09dc9aeb");
pub const BISWAP: Address = address!("3a6d8cA21D1CF76F653A67577FA0D27453350dD8");
pub const APESWAP: Address = address!("cF0feBd3f17CEf5b47b0cD257aCf6025c5BFf3b7");
pub const ONEINCH_V4: Address = address!("1111111254fb6c44bac0bed2854e76f90643097d");
pub const PARASWAP_AUGUSTUS_V5: Address = address!("DEF171Fe48CF0115B1d80b88dc8eAB59176FEe57");
pub const KYBERSWAP_META_V2: Address = address!("6131B5fae19EA4f9D964eAc0408E4408b66337b5");
pub const OPENOCEAN_V2: Address = address!("6352a56caadc4f1e25cd6c75970fa768a3304e64");
pub const FIREBIRD: Address = address!("92e4f29be975c1b1eb72e77de24dccf11432a5bd");

/// All monitored router addresses.
pub const ROUTERS: [Address; 12] = [
    PANCAKESWAP_V2,
    PANCAKESWAP_V3_SWAP,
    PANCAKESWAP_V3_SMART,
    PANCAKESWAP_UNIVERSAL,
    PANCAKESWAP_UNIVERSAL_2,
    BISWAP,
    APESWAP,
    ONEINCH_V4,
    PARASWAP_AUGUSTUS_V5,
    KYBERSWAP_META_V2,
    OPENOCEAN_V2,
    FIREBIRD,
];

/// Check if an address is a monitored DEX router.
pub fn is_monitored_router(addr: &Address) -> bool {
    ROUTERS.contains(addr)
}

// ---------------------------------------------------------------------------
// Swap Function Selectors (26 total)
// ---------------------------------------------------------------------------

// V2 selectors (9)
pub const SEL_SWAP_EXACT_ETH_FOR_TOKENS: [u8; 4] = [0x7f, 0xf3, 0x6a, 0xb5];
pub const SEL_SWAP_EXACT_TOKENS_FOR_ETH: [u8; 4] = [0x18, 0xcb, 0xaf, 0xe5];
pub const SEL_SWAP_EXACT_TOKENS_FOR_TOKENS: [u8; 4] = [0x38, 0xed, 0x17, 0x39];
pub const SEL_SWAP_TOKENS_FOR_EXACT_TOKENS: [u8; 4] = [0x88, 0x03, 0xdb, 0xee];
pub const SEL_SWAP_EXACT_ETH_FOR_TOKENS_FEE: [u8; 4] = [0xb6, 0xf9, 0xde, 0x95];
pub const SEL_SWAP_ETH_FOR_EXACT_TOKENS: [u8; 4] = [0xfb, 0x3b, 0xdb, 0x41];
pub const SEL_SWAP_EXACT_TOKENS_FOR_TOKENS_FEE: [u8; 4] = [0x5c, 0x11, 0xd7, 0x95];
pub const SEL_SWAP_EXACT_TOKENS_FOR_ETH_FEE: [u8; 4] = [0x79, 0x1a, 0xc9, 0x47];
pub const SEL_SWAP_TOKENS_FOR_EXACT_ETH: [u8; 4] = [0x4a, 0x25, 0xd9, 0x4a];

// V3 standard selectors (4)
pub const SEL_EXACT_INPUT_SINGLE: [u8; 4] = [0x41, 0x4b, 0xf3, 0x89];
pub const SEL_EXACT_INPUT: [u8; 4] = [0xc0, 0x4b, 0x8d, 0x59];
pub const SEL_EXACT_OUTPUT_SINGLE: [u8; 4] = [0xdb, 0x3e, 0x21, 0x98];
pub const SEL_EXACT_OUTPUT: [u8; 4] = [0xf2, 0x8c, 0x04, 0x98];

// SmartRouter selectors (6)
pub const SEL_SMART_EXACT_INPUT_SINGLE: [u8; 4] = [0x04, 0xe4, 0x5a, 0xaf];
pub const SEL_SMART_EXACT_OUTPUT_SINGLE: [u8; 4] = [0xb8, 0x58, 0x18, 0x3f];
pub const SEL_SMART_EXACT_INPUT: [u8; 4] = [0x50, 0x23, 0xb4, 0xdf];
pub const SEL_SMART_EXACT_OUTPUT: [u8; 4] = [0x09, 0xb8, 0x13, 0x46];
pub const SEL_SMART_MULTICALL_DEADLINE: [u8; 4] = [0x47, 0x2b, 0x43, 0xf3];
pub const SEL_SMART_MULTICALL: [u8; 4] = [0x42, 0x71, 0x2a, 0x67];

// Universal Router selectors (3)
pub const SEL_EXECUTE_DEADLINE: [u8; 4] = [0x35, 0x93, 0x56, 0x4c];
pub const SEL_EXECUTE: [u8; 4] = [0x24, 0x85, 0x69, 0x96];
pub const SEL_EXECUTE_V2: [u8; 4] = [0x24, 0x85, 0x6b, 0xc3];

// Aggregator selectors (4)
pub const SEL_1INCH_SWAP: [u8; 4] = [0x12, 0xaa, 0x3c, 0xaf];
pub const SEL_1INCH_UNOSWAP: [u8; 4] = [0x05, 0x02, 0xb1, 0xc5];
pub const SEL_1INCH_V3_SWAP: [u8; 4] = [0xe4, 0x49, 0x02, 0x2e];
pub const SEL_PARASWAP_SIMPLE_SWAP: [u8; 4] = [0x5f, 0x57, 0x55, 0x29];

// ---------------------------------------------------------------------------
// Token Addresses (BSC mainnet)
// ---------------------------------------------------------------------------

// Stablecoins
pub const USDT: Address = address!("55d398326f99059fF775485246999027B3197955");
pub const USDC: Address = address!("8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d");
pub const FDUSD: Address = address!("c5f0f7b66764F6ec8C8Dff7BA683102295E16409");
pub const BUSD: Address = address!("e9e7CEA3DedcA5984780Bafc599bD69ADd087D56");

// Volatile tokens
pub const WBNB: Address = address!("bb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c");
pub const BTCB: Address = address!("7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c");
pub const ETH: Address = address!("2170Ed0880ac9A755fd29B2688956BD959F933F8");

pub const STABLES: [Address; 4] = [USDT, USDC, FDUSD, BUSD];
pub const VOLATILES: [Address; 3] = [WBNB, BTCB, ETH];

/// All tokens have 18 decimals on BSC.
pub const TOKEN_DECIMALS: u8 = 18;

/// Whale threshold in USD.
pub const WHALE_THRESHOLD_USD: Decimal = dec!(10_000);

// ---------------------------------------------------------------------------
// Universal Router command bytes
// ---------------------------------------------------------------------------

pub const UR_V3_SWAP_EXACT_IN: u8 = 0x00;
pub const UR_V3_SWAP_EXACT_OUT: u8 = 0x01;
pub const UR_V2_SWAP_EXACT_IN: u8 = 0x08;
pub const UR_V2_SWAP_EXACT_OUT: u8 = 0x09;

