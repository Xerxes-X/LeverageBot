//! Sandwich detection via poison scoring.
//!
//! Assigns a suspicion score [0.0, 1.0] to each decoded swap based on
//! characteristics commonly associated with sandwich attacks:
//! - Abnormally high gas price (front-running premium)
//! - Very short deadline (time pressure)
//! - Zero slippage tolerance (exact output swaps used by bots)
//! - Small USD value combined with high gas (unprofitable for retail)

use alloy::primitives::U256;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::types::DecodedSwap;

/// Default median gas price on BSC (3 Gwei) for comparison when
/// no block-level gas data is available.
const DEFAULT_MEDIAN_GAS_PRICE: u128 = 3_000_000_000; // 3 Gwei

/// Compute a poison/sandwich suspicion score for a decoded swap.
///
/// Returns a value in [0.0, 1.0] where higher values indicate greater
/// suspicion of being part of a sandwich attack.
///
/// # Arguments
/// * `swap` - The decoded swap to score.
/// * `median_gas_price` - Optional median gas price for the current block.
///   If `None`, uses the default BSC median (3 Gwei).
pub fn compute_poison_score(swap: &DecodedSwap, median_gas_price: Option<u128>) -> f64 {
    let mut score = 0.0;
    let median = median_gas_price.unwrap_or(DEFAULT_MEDIAN_GAS_PRICE);

    // High gas premium: gas price > 2x median.
    if swap.gas_price > median.saturating_mul(2) {
        score += 0.3;
    }

    // Short deadline: expires within 60 seconds.
    if let Some(deadline) = swap.deadline {
        let now = swap.timestamp as u64;
        if deadline > 0 && deadline < now.saturating_add(60) {
            score += 0.3;
        }
    }

    // No slippage tolerance: amountOutMin == 0.
    if swap.amount_out_min_raw == U256::ZERO {
        score += 0.2;
    }

    // Small amount + high gas: suggests bot, not retail.
    let high_gas = swap.gas_price > median.saturating_mul(3);
    let small_amount = swap.usd_value > Decimal::ZERO && swap.usd_value < dec!(100);
    if small_amount && high_gas {
        score += 0.2;
    }

    // Cap at 1.0.
    if score > 1.0 {
        score = 1.0;
    }

    score
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloy::primitives::{Address, B256};
    use crate::types::SwapDirection;

    fn make_swap(gas_price: u128, deadline: Option<u64>, amount_out_min: U256, usd_value: Decimal) -> DecodedSwap {
        DecodedSwap {
            tx_hash: B256::ZERO,
            router: Address::ZERO,
            selector: [0; 4],
            token_in: Address::ZERO,
            token_out: Address::ZERO,
            amount_in_raw: U256::from(1_000_000_000_000_000_000u128),
            amount_out_min_raw: amount_out_min,
            gas_price,
            deadline,
            direction: SwapDirection::Buy,
            usd_value,
            poison_score: 0.0,
            timestamp: 1700000000,
        }
    }

    #[test]
    fn test_normal_swap_low_score() {
        let swap = make_swap(
            3_000_000_000,     // normal gas
            Some(1700000300),  // 5 min deadline
            U256::from(1u64),  // nonzero slippage
            dec!(500),         // reasonable amount
        );
        let score = compute_poison_score(&swap, None);
        assert!(score < 0.1, "normal swap should have low score, got {score}");
    }

    #[test]
    fn test_high_gas_adds_score() {
        let swap = make_swap(
            10_000_000_000,    // high gas
            Some(1700000300),
            U256::from(1u64),
            dec!(500),
        );
        let score = compute_poison_score(&swap, None);
        assert!((score - 0.3).abs() < 0.01, "high gas should add 0.3, got {score}");
    }

    #[test]
    fn test_all_flags_caps_at_1() {
        let swap = make_swap(
            15_000_000_000,    // very high gas (>2x and >3x)
            Some(1700000030),  // very short deadline (30s)
            U256::ZERO,        // zero slippage
            dec!(50),          // small amount
        );
        let score = compute_poison_score(&swap, None);
        assert!((score - 1.0).abs() < 0.01, "max score should be 1.0, got {score}");
    }
}
