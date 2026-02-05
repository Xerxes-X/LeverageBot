//! Top-level swap transaction decoder.
//!
//! Dispatches to the appropriate sub-decoder based on the router address
//! and function selector.

pub mod aggregator;
pub mod universal;
pub mod v2;
pub mod v3;

use alloy::primitives::{Address, B256, U256};
use tracing::trace;

use crate::constants::*;
use crate::types::DecodedSwap;

/// Attempt to decode a pending transaction as a DEX swap.
///
/// Returns `None` if:
/// - The `to` address is not a monitored router
/// - The calldata is too short (< 4 bytes)
/// - The function selector is not recognized
/// - ABI decoding fails
///
/// These are all expected cases — the vast majority of BSC transactions
/// are not DEX swaps on monitored routers.
pub fn decode_transaction(
    to: Address,
    calldata: &[u8],
    value: U256,
    tx_hash: B256,
    gas_price: u128,
) -> Option<DecodedSwap> {
    // Quick reject: not a monitored router.
    if !is_monitored_router(&to) {
        return None;
    }

    // Need at least 4 bytes for the function selector.
    if calldata.len() < 4 {
        return None;
    }

    let selector: [u8; 4] = calldata[..4]
        .try_into()
        .expect("slice is exactly 4 bytes");

    // Determine selector category and dispatch.
    // Pass full calldata (including selector) to sub-decoders.
    let result = match categorize_selector(selector) {
        Some(Category::V2) => {
            v2::decode_v2(selector, calldata, value, tx_hash, to, gas_price)
        }
        Some(Category::V3) | Some(Category::SmartRouter) => {
            v3::decode_v3(selector, calldata, value, tx_hash, to, gas_price)
        }
        Some(Category::Universal) => {
            universal::decode_universal(selector, calldata, value, tx_hash, to, gas_price)
        }
        Some(Category::Aggregator) => {
            aggregator::decode_aggregator(selector, calldata, value, tx_hash, to, gas_price)
        }
        None => None,
    };

    if result.is_some() {
        trace!(
            tx = %tx_hash,
            router = %to,
            selector = %hex::encode(selector),
            "decoded swap"
        );
    }

    result
}

#[derive(Debug, Clone, Copy)]
enum Category {
    V2,
    V3,
    SmartRouter,
    Universal,
    Aggregator,
}

fn categorize_selector(sel: [u8; 4]) -> Option<Category> {
    match sel {
        // V2 selectors
        SEL_SWAP_EXACT_ETH_FOR_TOKENS
        | SEL_SWAP_EXACT_TOKENS_FOR_ETH
        | SEL_SWAP_EXACT_TOKENS_FOR_TOKENS
        | SEL_SWAP_TOKENS_FOR_EXACT_TOKENS
        | SEL_SWAP_EXACT_ETH_FOR_TOKENS_FEE
        | SEL_SWAP_ETH_FOR_EXACT_TOKENS
        | SEL_SWAP_EXACT_TOKENS_FOR_TOKENS_FEE
        | SEL_SWAP_EXACT_TOKENS_FOR_ETH_FEE
        | SEL_SWAP_TOKENS_FOR_EXACT_ETH => Some(Category::V2),

        // V3 standard selectors
        SEL_EXACT_INPUT_SINGLE
        | SEL_EXACT_INPUT
        | SEL_EXACT_OUTPUT_SINGLE
        | SEL_EXACT_OUTPUT => Some(Category::V3),

        // SmartRouter selectors
        SEL_SMART_EXACT_INPUT_SINGLE
        | SEL_SMART_EXACT_OUTPUT_SINGLE
        | SEL_SMART_EXACT_INPUT
        | SEL_SMART_EXACT_OUTPUT
        | SEL_SMART_MULTICALL_DEADLINE
        | SEL_SMART_MULTICALL => Some(Category::SmartRouter),

        // Universal Router selectors
        SEL_EXECUTE_DEADLINE
        | SEL_EXECUTE
        | SEL_EXECUTE_V2 => Some(Category::Universal),

        // Aggregator selectors
        SEL_1INCH_SWAP
        | SEL_1INCH_UNOSWAP
        | SEL_1INCH_V3_SWAP
        | SEL_PARASWAP_SIMPLE_SWAP => Some(Category::Aggregator),

        _ => None,
    }
}
