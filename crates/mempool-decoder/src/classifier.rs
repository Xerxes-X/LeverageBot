//! Token classification and USD value estimation for decoded swaps.
//!
//! Classifies swap direction (buy/sell/skip) based on token roles:
//! - stable → volatile = BUY
//! - volatile → stable = SELL
//! - same category = SKIP

use alloy::primitives::{Address, U256};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

use crate::constants::*;
use crate::types::{DecodedSwap, SwapDirection, TokenRole};

/// Classify the direction of a swap based on token roles.
pub fn classify_direction(token_in: Address, token_out: Address) -> SwapDirection {
    let role_in = token_role(token_in);
    let role_out = token_role(token_out);

    match (role_in, role_out) {
        (TokenRole::Stable, TokenRole::Volatile) => SwapDirection::Buy,
        (TokenRole::Volatile, TokenRole::Stable) => SwapDirection::Sell,
        _ => SwapDirection::Skip,
    }
}

/// Determine the role of a token address.
pub fn token_role(addr: Address) -> TokenRole {
    if STABLES.contains(&addr) {
        TokenRole::Stable
    } else if VOLATILES.contains(&addr) {
        TokenRole::Volatile
    } else {
        TokenRole::Unknown
    }
}

/// Map a volatile token address to its symbol name.
pub fn volatile_symbol(addr: Address) -> Option<&'static str> {
    match addr {
        a if a == WBNB => Some("WBNB"),
        a if a == BTCB => Some("BTCB"),
        a if a == ETH => Some("ETH"),
        _ => None,
    }
}

/// Get the volatile token address from a buy or sell swap.
///
/// For buys: token_out is volatile. For sells: token_in is volatile.
pub fn get_volatile_token(swap: &DecodedSwap) -> Option<Address> {
    match swap.direction {
        SwapDirection::Buy => Some(swap.token_out),
        SwapDirection::Sell => Some(swap.token_in),
        SwapDirection::Skip => None,
    }
}

/// Estimate the USD value of a swap using cached prices.
///
/// The `prices` map should contain entries like "BNBUSDT" → price.
/// All monitored BSC tokens use 18 decimals.
pub fn estimate_usd_value(swap: &DecodedSwap, prices: &HashMap<String, Decimal>) -> Decimal {
    let decimals_factor = dec!(1_000_000_000_000_000_000); // 10^18

    // Convert raw amount to human-readable.
    let amount_in = u256_to_decimal(swap.amount_in_raw);
    let human_amount = amount_in / decimals_factor;

    // Determine price based on token_in.
    let usd_price = match swap.token_in {
        addr if addr == WBNB => prices.get("BNBUSDT").copied().unwrap_or(Decimal::ZERO),
        addr if addr == BTCB => prices.get("BTCUSDT").copied().unwrap_or(Decimal::ZERO),
        addr if addr == ETH => prices.get("ETHUSDT").copied().unwrap_or(Decimal::ZERO),
        // Stablecoins: 1 USD each.
        addr if STABLES.contains(&addr) => Decimal::ONE,
        _ => Decimal::ZERO,
    };

    human_amount * usd_price
}

/// Convert a U256 to Decimal (may lose precision for very large values).
fn u256_to_decimal(value: U256) -> Decimal {
    // U256 can be huge; Decimal supports up to ~28 digits.
    // For typical swap amounts, converting via string is safe.
    let s = value.to_string();
    s.parse::<Decimal>().unwrap_or(Decimal::ZERO)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_buy() {
        assert_eq!(classify_direction(USDT, WBNB), SwapDirection::Buy);
        assert_eq!(classify_direction(USDC, BTCB), SwapDirection::Buy);
        assert_eq!(classify_direction(FDUSD, ETH), SwapDirection::Buy);
    }

    #[test]
    fn test_classify_sell() {
        assert_eq!(classify_direction(WBNB, USDT), SwapDirection::Sell);
        assert_eq!(classify_direction(BTCB, USDC), SwapDirection::Sell);
        assert_eq!(classify_direction(ETH, BUSD), SwapDirection::Sell);
    }

    #[test]
    fn test_classify_skip() {
        assert_eq!(classify_direction(WBNB, BTCB), SwapDirection::Skip);
        assert_eq!(classify_direction(USDT, USDC), SwapDirection::Skip);
        assert_eq!(classify_direction(Address::ZERO, WBNB), SwapDirection::Skip);
    }

    #[test]
    fn test_volatile_symbol() {
        assert_eq!(volatile_symbol(WBNB), Some("WBNB"));
        assert_eq!(volatile_symbol(BTCB), Some("BTCB"));
        assert_eq!(volatile_symbol(ETH), Some("ETH"));
        assert_eq!(volatile_symbol(USDT), None);
    }

    #[test]
    fn test_u256_to_decimal() {
        let one_ether = U256::from(1_000_000_000_000_000_000u128);
        let dec = u256_to_decimal(one_ether);
        assert_eq!(dec, dec!(1_000_000_000_000_000_000));
    }
}
