//! ABI decoding for Uniswap V3 and PancakeSwap SmartRouter swap functions.
//!
//! V3 standard: 4 selectors (exactInputSingle, exactInput, exactOutputSingle, exactOutput).
//! SmartRouter: 6 selectors (similar but different parameter ordering + multicall).

use alloy::primitives::{Address, B256, U256};
use alloy::sol;
use alloy::sol_types::SolCall;

use crate::constants::*;
use crate::types::{DecodedSwap, SwapDirection};

// ---------------------------------------------------------------------------
// V3 standard ABI definitions
// ---------------------------------------------------------------------------

sol! {
    #[allow(missing_docs)]
    function exactInputSingleV3(
        address tokenIn,
        address tokenOut,
        uint24 fee,
        address recipient,
        uint256 deadline,
        uint256 amountIn,
        uint256 amountOutMinimum,
        uint160 sqrtPriceLimitX96
    );

    #[allow(missing_docs)]
    function exactInputV3(
        bytes path,
        address recipient,
        uint256 deadline,
        uint256 amountIn,
        uint256 amountOutMinimum
    );

    #[allow(missing_docs)]
    function exactOutputSingleV3(
        address tokenIn,
        address tokenOut,
        uint24 fee,
        address recipient,
        uint256 deadline,
        uint256 amountOut,
        uint256 amountInMaximum,
        uint160 sqrtPriceLimitX96
    );

    #[allow(missing_docs)]
    function exactOutputV3(
        bytes path,
        address recipient,
        uint256 deadline,
        uint256 amountOut,
        uint256 amountInMaximum
    );
}

// ---------------------------------------------------------------------------
// SmartRouter ABI definitions (different param ordering — no deadline in some)
// ---------------------------------------------------------------------------

sol! {
    #[allow(missing_docs)]
    function smartExactInputSingle(
        address tokenIn,
        address tokenOut,
        uint24 fee,
        address recipient,
        uint256 amountIn,
        uint256 amountOutMinimum,
        uint160 sqrtPriceLimitX96
    );

    #[allow(missing_docs)]
    function smartExactOutputSingle(
        address tokenIn,
        address tokenOut,
        uint24 fee,
        address recipient,
        uint256 amountOut,
        uint256 amountInMaximum,
        uint160 sqrtPriceLimitX96
    );

    #[allow(missing_docs)]
    function smartExactInput(
        bytes path,
        address recipient,
        uint256 amountIn,
        uint256 amountOutMinimum
    );

    #[allow(missing_docs)]
    function smartExactOutput(
        bytes path,
        address recipient,
        uint256 amountOut,
        uint256 amountInMaximum
    );

    #[allow(missing_docs)]
    function multicallWithDeadline(
        uint256 deadline,
        bytes[] data
    );

    #[allow(missing_docs)]
    function multicallPlain(
        bytes[] data
    );
}

/// Decode a V3 or SmartRouter swap from calldata.
pub fn decode_v3(
    selector: [u8; 4],
    calldata: &[u8],
    _value: U256,
    tx_hash: B256,
    router: Address,
    gas_price: u128,
) -> Option<DecodedSwap> {
    match selector {
        // --- V3 standard ---
        SEL_EXACT_INPUT_SINGLE => {
            let call = exactInputSingleV3Call::abi_decode(calldata).ok()?;
            Some(build_v3_swap(
                tx_hash, router, selector, gas_price,
                call.tokenIn, call.tokenOut,
                call.amountIn, call.amountOutMinimum,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }
        SEL_EXACT_INPUT => {
            let call = exactInputV3Call::abi_decode(calldata).ok()?;
            let (token_in, token_out) = parse_v3_path(&call.path)?;
            Some(build_v3_swap(
                tx_hash, router, selector, gas_price,
                token_in, token_out,
                call.amountIn, call.amountOutMinimum,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }
        SEL_EXACT_OUTPUT_SINGLE => {
            let call = exactOutputSingleV3Call::abi_decode(calldata).ok()?;
            Some(build_v3_swap(
                tx_hash, router, selector, gas_price,
                call.tokenIn, call.tokenOut,
                call.amountInMaximum, call.amountOut,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }
        SEL_EXACT_OUTPUT => {
            let call = exactOutputV3Call::abi_decode(calldata).ok()?;
            // V3 exactOutput path is reversed: token_out is first, token_in is last
            let (token_out, token_in) = parse_v3_path(&call.path)?;
            Some(build_v3_swap(
                tx_hash, router, selector, gas_price,
                token_in, token_out,
                call.amountInMaximum, call.amountOut,
                Some(call.deadline.saturating_to::<u64>()),
            ))
        }

        // --- SmartRouter ---
        SEL_SMART_EXACT_INPUT_SINGLE => {
            let call = smartExactInputSingleCall::abi_decode(calldata).ok()?;
            Some(build_v3_swap(
                tx_hash, router, selector, gas_price,
                call.tokenIn, call.tokenOut,
                call.amountIn, call.amountOutMinimum,
                None,
            ))
        }
        SEL_SMART_EXACT_OUTPUT_SINGLE => {
            let call = smartExactOutputSingleCall::abi_decode(calldata).ok()?;
            Some(build_v3_swap(
                tx_hash, router, selector, gas_price,
                call.tokenIn, call.tokenOut,
                call.amountInMaximum, call.amountOut,
                None,
            ))
        }
        SEL_SMART_EXACT_INPUT => {
            let call = smartExactInputCall::abi_decode(calldata).ok()?;
            let (token_in, token_out) = parse_v3_path(&call.path)?;
            Some(build_v3_swap(
                tx_hash, router, selector, gas_price,
                token_in, token_out,
                call.amountIn, call.amountOutMinimum,
                None,
            ))
        }
        SEL_SMART_EXACT_OUTPUT => {
            let call = smartExactOutputCall::abi_decode(calldata).ok()?;
            let (token_out, token_in) = parse_v3_path(&call.path)?;
            Some(build_v3_swap(
                tx_hash, router, selector, gas_price,
                token_in, token_out,
                call.amountInMaximum, call.amountOut,
                None,
            ))
        }
        SEL_SMART_MULTICALL_DEADLINE => {
            let call = multicallWithDeadlineCall::abi_decode(calldata).ok()?;
            decode_first_swap_from_multicall(
                &call.data,
                tx_hash, router, selector, gas_price,
                Some(call.deadline.saturating_to::<u64>()),
            )
        }
        SEL_SMART_MULTICALL => {
            let call = multicallPlainCall::abi_decode(calldata).ok()?;
            decode_first_swap_from_multicall(
                &call.data,
                tx_hash, router, selector, gas_price,
                None,
            )
        }
        _ => None,
    }
}

/// Parse a V3 packed path to extract the first and last token addresses.
///
/// V3 path format: `token0 (20 bytes) || fee (3 bytes) || token1 (20 bytes) || ...`
/// Minimum path length: 20 + 3 + 20 = 43 bytes.
pub fn parse_v3_path(path: &[u8]) -> Option<(Address, Address)> {
    if path.len() < 43 {
        return None;
    }
    let first = Address::from_slice(&path[..20]);
    let last = Address::from_slice(&path[path.len() - 20..]);
    Some((first, last))
}

/// Try to decode the first swap call inside a multicall's data array.
fn decode_first_swap_from_multicall(
    calls: &[alloy::primitives::Bytes],
    tx_hash: B256,
    router: Address,
    outer_selector: [u8; 4],
    gas_price: u128,
    deadline: Option<u64>,
) -> Option<DecodedSwap> {
    for inner_calldata in calls {
        if inner_calldata.len() < 4 {
            continue;
        }
        let inner_sel: [u8; 4] = inner_calldata[..4].try_into().ok()?;
        let inner_data = &inner_calldata[4..];

        // Try SmartRouter single/path selectors inside the multicall.
        let result = match inner_sel {
            SEL_SMART_EXACT_INPUT_SINGLE => {
                let call = smartExactInputSingleCall::abi_decode(inner_data).ok()?;
                Some(build_v3_swap(
                    tx_hash, router, outer_selector, gas_price,
                    call.tokenIn, call.tokenOut,
                    call.amountIn, call.amountOutMinimum,
                    deadline,
                ))
            }
            SEL_SMART_EXACT_INPUT => {
                let call = smartExactInputCall::abi_decode(inner_data).ok()?;
                let (token_in, token_out) = parse_v3_path(&call.path)?;
                Some(build_v3_swap(
                    tx_hash, router, outer_selector, gas_price,
                    token_in, token_out,
                    call.amountIn, call.amountOutMinimum,
                    deadline,
                ))
            }
            _ => None,
        };

        if result.is_some() {
            return result;
        }
    }
    None
}

fn build_v3_swap(
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
        direction: SwapDirection::Skip,
        usd_value: rust_decimal::Decimal::ZERO,
        poison_score: 0.0,
        timestamp: chrono::Utc::now().timestamp(),
    }
}
