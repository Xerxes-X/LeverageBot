//! ABI decoding for Uniswap V2-style swap functions (9 selectors).
//!
//! Used by PancakeSwap V2, Biswap, ApeSwap, and similar V2-fork routers.

use alloy::primitives::{Address, B256, U256};
use alloy::sol;

use crate::constants::*;
use crate::types::{DecodedSwap, SwapDirection};

// ---------------------------------------------------------------------------
// ABI definitions via sol! macro
// ---------------------------------------------------------------------------

sol! {
    // Standard V2 swaps
    function swapExactTokensForTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] path,
        address to,
        uint256 deadline
    );

    function swapTokensForExactTokens(
        uint256 amountOut,
        uint256 amountInMax,
        address[] path,
        address to,
        uint256 deadline
    );

    function swapExactETHForTokens(
        uint256 amountOutMin,
        address[] path,
        address to,
        uint256 deadline
    );

    function swapExactTokensForETH(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] path,
        address to,
        uint256 deadline
    );

    function swapETHForExactTokens(
        uint256 amountOut,
        address[] path,
        address to,
        uint256 deadline
    );

    function swapTokensForExactETH(
        uint256 amountOut,
        uint256 amountInMax,
        address[] path,
        address to,
        uint256 deadline
    );

    // Fee-on-transfer variants
    function swapExactETHForTokensSupportingFeeOnTransferTokens(
        uint256 amountOutMin,
        address[] path,
        address to,
        uint256 deadline
    );

    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] path,
        address to,
        uint256 deadline
    );

    function swapExactTokensForTokensSupportingFeeOnTransferTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] path,
        address to,
        uint256 deadline
    );
}

use alloy::sol_types::SolCall;

/// Decode a V2-style swap from calldata.
///
/// Returns `None` if the selector doesn't match or decoding fails.
pub fn decode_v2(
    selector: [u8; 4],
    calldata: &[u8],
    value: U256,
    tx_hash: B256,
    router: Address,
    gas_price: u128,
) -> Option<DecodedSwap> {
    // calldata includes the 4-byte selector prefix; abi_decode handles it.
    match selector {
        // --- ETH-input functions: msg.value is the amount in, path[0] = WBNB ---
        SEL_SWAP_EXACT_ETH_FOR_TOKENS => {
            let call = swapExactETHForTokensCall::abi_decode(calldata).ok()?;
            let path = &call.path;
            if path.len() < 2 { return None; }
            Some(build_swap(
                tx_hash, router, selector, gas_price,
                path[0], *path.last().expect("path len >= 2"),
                U256::from(value), call.amountOutMin,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }
        SEL_SWAP_EXACT_ETH_FOR_TOKENS_FEE => {
            let call = swapExactETHForTokensSupportingFeeOnTransferTokensCall::abi_decode(calldata).ok()?;
            let path = &call.path;
            if path.len() < 2 { return None; }
            Some(build_swap(
                tx_hash, router, selector, gas_price,
                path[0], *path.last().expect("path len >= 2"),
                U256::from(value), call.amountOutMin,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }
        SEL_SWAP_ETH_FOR_EXACT_TOKENS => {
            let call = swapETHForExactTokensCall::abi_decode(calldata).ok()?;
            let path = &call.path;
            if path.len() < 2 { return None; }
            Some(build_swap(
                tx_hash, router, selector, gas_price,
                path[0], *path.last().expect("path len >= 2"),
                U256::from(value), call.amountOut,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }

        // --- Standard token-input functions ---
        SEL_SWAP_EXACT_TOKENS_FOR_TOKENS => {
            let call = swapExactTokensForTokensCall::abi_decode(calldata).ok()?;
            let path = &call.path;
            if path.len() < 2 { return None; }
            Some(build_swap(
                tx_hash, router, selector, gas_price,
                path[0], *path.last().expect("path len >= 2"),
                call.amountIn, call.amountOutMin,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }
        SEL_SWAP_EXACT_TOKENS_FOR_ETH => {
            let call = swapExactTokensForETHCall::abi_decode(calldata).ok()?;
            let path = &call.path;
            if path.len() < 2 { return None; }
            Some(build_swap(
                tx_hash, router, selector, gas_price,
                path[0], *path.last().expect("path len >= 2"),
                call.amountIn, call.amountOutMin,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }
        SEL_SWAP_EXACT_TOKENS_FOR_TOKENS_FEE => {
            let call = swapExactTokensForTokensSupportingFeeOnTransferTokensCall::abi_decode(calldata).ok()?;
            let path = &call.path;
            if path.len() < 2 { return None; }
            Some(build_swap(
                tx_hash, router, selector, gas_price,
                path[0], *path.last().expect("path len >= 2"),
                call.amountIn, call.amountOutMin,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }
        SEL_SWAP_EXACT_TOKENS_FOR_ETH_FEE => {
            let call = swapExactTokensForETHSupportingFeeOnTransferTokensCall::abi_decode(calldata).ok()?;
            let path = &call.path;
            if path.len() < 2 { return None; }
            Some(build_swap(
                tx_hash, router, selector, gas_price,
                path[0], *path.last().expect("path len >= 2"),
                call.amountIn, call.amountOutMin,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }

        // --- Exact-output functions: amountOut is known, amountInMax is the limit ---
        SEL_SWAP_TOKENS_FOR_EXACT_TOKENS => {
            let call = swapTokensForExactTokensCall::abi_decode(calldata).ok()?;
            let path = &call.path;
            if path.len() < 2 { return None; }
            Some(build_swap(
                tx_hash, router, selector, gas_price,
                path[0], *path.last().expect("path len >= 2"),
                call.amountInMax, call.amountOut,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }
        SEL_SWAP_TOKENS_FOR_EXACT_ETH => {
            let call = swapTokensForExactETHCall::abi_decode(calldata).ok()?;
            let path = &call.path;
            if path.len() < 2 { return None; }
            Some(build_swap(
                tx_hash, router, selector, gas_price,
                path[0], *path.last().expect("path len >= 2"),
                call.amountInMax, call.amountOut,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }

        _ => None,
    }
}

/// Build a `DecodedSwap` with default classification fields.
fn build_swap(
    tx_hash: B256,
    router: Address,
    selector: [u8; 4],
    gas_price: u128,
    token_in: Address,
    token_out: Address,
    amount_in_raw: U256,
    amount_out_min_raw: U256,
    deadline: Option<u64>,
) -> DecodedSwap {
    DecodedSwap {
        tx_hash,
        router,
        selector,
        token_in,
        token_out,
        amount_in_raw,
        amount_out_min_raw,
        gas_price,
        deadline,
        direction: SwapDirection::Skip, // filled later by classifier
        usd_value: rust_decimal::Decimal::ZERO,
        poison_score: 0.0,
        timestamp: chrono::Utc::now().timestamp(),
    }
}
