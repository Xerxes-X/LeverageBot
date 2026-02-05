//! ABI decoding for DEX aggregator swap functions (1inch, ParaSwap).
//!
//! Best-effort decoding — these aggregators have complex, evolving ABIs.
//! We extract token_in, token_out, and amount_in where possible.

use alloy::primitives::{Address, Bytes, B256, U256};
use alloy::sol;
use alloy::sol_types::SolCall;

use crate::constants::*;
use crate::types::{DecodedSwap, SwapDirection};

// ---------------------------------------------------------------------------
// 1inch V4 ABI definitions
// ---------------------------------------------------------------------------

sol! {
    /// 1inch swap description struct.
    struct SwapDescription {
        address srcToken;
        address dstToken;
        address srcReceiver;
        address dstReceiver;
        uint256 amount;
        uint256 minReturnAmount;
        uint256 flags;
    }

    /// 1inch swap function (generic swap via executor).
    #[allow(missing_docs)]
    function swap(
        address executor,
        SwapDescription desc,
        bytes permit,
        bytes data
    ) returns (uint256 returnAmount, uint256 spentAmount);

    /// 1inch unoswap (single-pool swap).
    #[allow(missing_docs)]
    function unoswap(
        address srcToken,
        uint256 amount,
        uint256 minReturn,
        uint256[] pools
    ) returns (uint256 returnAmount);

    /// 1inch uniswapV3Swap.
    #[allow(missing_docs)]
    function uniswapV3Swap(
        uint256 amount,
        uint256 minReturn,
        uint256[] pools
    ) returns (uint256 returnAmount);
}

// ---------------------------------------------------------------------------
// ParaSwap ABI definitions
// ---------------------------------------------------------------------------

sol! {
    /// ParaSwap simple swap data struct.
    struct SimpleData {
        address fromToken;
        address toToken;
        uint256 fromAmount;
        uint256 toAmount;
        uint256 expectedAmount;
        address[] callees;
        bytes exchangeData;
        uint256[] startIndexes;
        uint256[] values;
        address beneficiary;
        address partner;
        uint256 feePercent;
        bytes permit;
        uint256 deadline;
        bytes16 uuid;
    }

    /// ParaSwap simpleSwap function.
    #[allow(missing_docs)]
    function simpleSwap(SimpleData data) returns (uint256 receivedAmount);
}

/// Decode an aggregator swap from calldata.
pub fn decode_aggregator(
    selector: [u8; 4],
    calldata: &[u8],
    _value: U256,
    tx_hash: B256,
    router: Address,
    gas_price: u128,
) -> Option<DecodedSwap> {
    match selector {
        SEL_1INCH_SWAP => {
            let call = swapCall::abi_decode(calldata).ok()?;
            Some(build_agg_swap(
                tx_hash,
                router,
                selector,
                gas_price,
                call.desc.srcToken,
                call.desc.dstToken,
                call.desc.amount,
                call.desc.minReturnAmount,
                None,
            ))
        }
        SEL_1INCH_UNOSWAP => {
            let call = unoswapCall::abi_decode(calldata).ok()?;
            // unoswap doesn't expose dstToken directly — we only know srcToken.
            // Set token_out to zero address; the classifier will mark it as Skip
            // unless we can infer it from context.
            Some(build_agg_swap(
                tx_hash,
                router,
                selector,
                gas_price,
                call.srcToken,
                Address::ZERO,
                call.amount,
                call.minReturn,
                None,
            ))
        }
        SEL_1INCH_V3_SWAP => {
            let call = uniswapV3SwapCall::abi_decode(calldata).ok()?;
            // uniswapV3Swap uses pool addresses packed in uint256 — no direct token info.
            // Best-effort: cannot determine tokens without pool lookup.
            // Return None — this is an acceptable miss for aggregate signal quality.
            let _ = call;
            None
        }
        SEL_PARASWAP_SIMPLE_SWAP => {
            let call = simpleSwapCall::abi_decode(calldata).ok()?;
            Some(build_agg_swap(
                tx_hash,
                router,
                selector,
                gas_price,
                call.data.fromToken,
                call.data.toToken,
                call.data.fromAmount,
                call.data.toAmount,
                Some(call.data.deadline.saturating_to::<u64>()),
            ))
        }
        _ => None,
    }
}

fn build_agg_swap(
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
