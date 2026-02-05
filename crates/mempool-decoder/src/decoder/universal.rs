//! ABI decoding for PancakeSwap Universal Router `execute` functions.
//!
//! The Universal Router uses a command-based dispatch pattern:
//! - `execute(bytes commands, bytes[] inputs, uint256 deadline)`
//! - `execute(bytes commands, bytes[] inputs)`
//!
//! Each byte in `commands` identifies a command type, and the corresponding
//! `inputs[i]` contains the ABI-encoded parameters for that command.

use alloy::primitives::{Address, B256, U256};
use alloy::sol;
use alloy::sol_types::SolCall;

use crate::constants::*;
use crate::decoder::v3::parse_v3_path;
use crate::types::{DecodedSwap, SwapDirection};

// ---------------------------------------------------------------------------
// Universal Router execute ABI
// ---------------------------------------------------------------------------

sol! {
    #[allow(missing_docs)]
    function executeWithDeadline(
        bytes commands,
        bytes[] inputs,
        uint256 deadline
    );

    #[allow(missing_docs)]
    function executePlain(
        bytes commands,
        bytes[] inputs
    );
}

// Inner command parameter structures (decoded from inputs[i])
sol! {
    // V2 swap exact in: (address recipient, uint256 amountIn, uint256 amountOutMin, address[] path, bool payerIsUser)
    #[allow(missing_docs)]
    function v2SwapExactIn(
        address recipient,
        uint256 amountIn,
        uint256 amountOutMin,
        address[] path,
        bool payerIsUser
    );

    // V2 swap exact out: (address recipient, uint256 amountOut, uint256 amountInMax, address[] path, bool payerIsUser)
    #[allow(missing_docs)]
    function v2SwapExactOut(
        address recipient,
        uint256 amountOut,
        uint256 amountInMax,
        address[] path,
        bool payerIsUser
    );

    // V3 swap exact in: (address recipient, uint256 amountIn, uint256 amountOutMin, bytes path, bool payerIsUser)
    #[allow(missing_docs)]
    function v3SwapExactIn(
        address recipient,
        uint256 amountIn,
        uint256 amountOutMin,
        bytes path,
        bool payerIsUser
    );

    // V3 swap exact out: (address recipient, uint256 amountOut, uint256 amountInMax, bytes path, bool payerIsUser)
    #[allow(missing_docs)]
    function v3SwapExactOut(
        address recipient,
        uint256 amountOut,
        uint256 amountInMax,
        bytes path,
        bool payerIsUser
    );
}

/// Decode a Universal Router `execute` call.
pub fn decode_universal(
    selector: [u8; 4],
    calldata: &[u8],
    value: U256,
    tx_hash: B256,
    router: Address,
    gas_price: u128,
) -> Option<DecodedSwap> {
    let (commands, inputs, deadline) = match selector {
        SEL_EXECUTE_DEADLINE => {
            let call = executeWithDeadlineCall::abi_decode(calldata).ok()?;
            (
                call.commands.to_vec(),
                call.inputs,
                Some(call.deadline.saturating_to::<u64>()),
            )
        }
        SEL_EXECUTE | SEL_EXECUTE_V2 => {
            let call = executePlainCall::abi_decode(calldata).ok()?;
            (call.commands.to_vec(), call.inputs, None)
        }
        _ => return None,
    };

    // Find the first swap command and decode it.
    for (i, &cmd) in commands.iter().enumerate() {
        // The lower 5 bits hold the command type.
        let cmd_type = cmd & 0x1f;

        if i >= inputs.len() {
            break;
        }
        let input = &inputs[i];

        match cmd_type {
            UR_V2_SWAP_EXACT_IN => {
                let decoded = v2SwapExactInCall::abi_decode(input).ok()?;
                let path = &decoded.path;
                if path.len() < 2 {
                    continue;
                }
                let amount_in = if decoded.amountIn == U256::ZERO {
                    // When amountIn is 0 for V2, it means use msg.value (BNB)
                    value
                } else {
                    decoded.amountIn
                };
                return Some(build_ur_swap(
                    tx_hash, router, selector, gas_price,
                    path[0], *path.last().expect("path len >= 2"),
                    amount_in, decoded.amountOutMin,
                    deadline,
                ));
            }
            UR_V2_SWAP_EXACT_OUT => {
                let decoded = v2SwapExactOutCall::abi_decode(input).ok()?;
                let path = &decoded.path;
                if path.len() < 2 {
                    continue;
                }
                return Some(build_ur_swap(
                    tx_hash, router, selector, gas_price,
                    path[0], *path.last().expect("path len >= 2"),
                    decoded.amountInMax, decoded.amountOut,
                    deadline,
                ));
            }
            UR_V3_SWAP_EXACT_IN => {
                let decoded = v3SwapExactInCall::abi_decode(input).ok()?;
                let (token_in, token_out) = parse_v3_path(&decoded.path)?;
                let amount_in = if decoded.amountIn == U256::ZERO {
                    value
                } else {
                    decoded.amountIn
                };
                return Some(build_ur_swap(
                    tx_hash, router, selector, gas_price,
                    token_in, token_out,
                    amount_in, decoded.amountOutMin,
                    deadline,
                ));
            }
            UR_V3_SWAP_EXACT_OUT => {
                let decoded = v3SwapExactOutCall::abi_decode(input).ok()?;
                let (token_out, token_in) = parse_v3_path(&decoded.path)?;
                return Some(build_ur_swap(
                    tx_hash, router, selector, gas_price,
                    token_in, token_out,
                    decoded.amountInMax, decoded.amountOut,
                    deadline,
                ));
            }
            _ => continue,
        }
    }

    None
}

fn build_ur_swap(
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
