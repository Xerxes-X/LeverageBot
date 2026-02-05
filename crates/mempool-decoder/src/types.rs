//! Core types for the mempool decoder.

use alloy::primitives::{Address, B256, U256};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Decoded swap from ABI decoding
// ---------------------------------------------------------------------------

/// A decoded DEX swap transaction from the BSC mempool.
#[derive(Debug, Clone)]
pub struct DecodedSwap {
    /// Transaction hash.
    pub tx_hash: B256,
    /// Router address the transaction was sent to.
    pub router: Address,
    /// Function selector (first 4 bytes of calldata).
    pub selector: [u8; 4],
    /// Input token address.
    pub token_in: Address,
    /// Output token address.
    pub token_out: Address,
    /// Raw amount in (wei/smallest unit).
    pub amount_in_raw: U256,
    /// Raw minimum output amount (wei/smallest unit).
    pub amount_out_min_raw: U256,
    /// Gas price in wei.
    pub gas_price: u128,
    /// Deadline timestamp (if available from calldata).
    pub deadline: Option<u64>,
    /// Swap direction after classification.
    pub direction: SwapDirection,
    /// Estimated USD value (filled after classification).
    pub usd_value: Decimal,
    /// Poison/sandwich suspicion score (filled after analysis).
    pub poison_score: f64,
    /// Timestamp when the swap was decoded.
    pub timestamp: i64,
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Direction of a swap relative to volatile tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapDirection {
    /// Buying volatile with stable (stable → volatile).
    Buy,
    /// Selling volatile for stable (volatile → stable).
    Sell,
    /// Not directionally informative (both volatile or both stable).
    Skip,
}

/// Role of a token in the swap classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenRole {
    Stable,
    Volatile,
    Unknown,
}

// ---------------------------------------------------------------------------
// Selector routing category
// ---------------------------------------------------------------------------

/// Which decoder module handles a given selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectorCategory {
    V2,
    V3,
    SmartRouter,
    Universal,
    Aggregator,
}

// ---------------------------------------------------------------------------
// Wire-format types — must match bot's types/mempool.rs exactly
// ---------------------------------------------------------------------------

/// Per-token aggregate mempool order flow signal from the Rust decoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MempoolTokenSignal {
    #[serde(with = "rust_decimal::serde::str")]
    pub buy_volume_1m_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub sell_volume_1m_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub buy_volume_5m_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub sell_volume_5m_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub buy_volume_15m_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub sell_volume_15m_usd: Decimal,
    /// Direction score in [-1.0, 1.0].
    #[serde(with = "rust_decimal::serde::str")]
    pub direction_score_5m: Decimal,
    /// Buy / (Buy + Sell) ratio in [0.0, 1.0].
    #[serde(with = "rust_decimal::serde::str")]
    pub buy_sell_ratio_5m: Decimal,
    pub tx_count_buy_5m: u32,
    pub tx_count_sell_5m: u32,
    pub whale_buy_count_15m: u32,
    pub whale_sell_count_15m: u32,
    /// Current 5 m volume / trailing 30 m average.
    #[serde(with = "rust_decimal::serde::str")]
    pub volume_acceleration: Decimal,
    pub total_swaps_seen: u32,
}

/// Aggregate mempool signal across all monitored token pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MempoolSignal {
    pub pairs: HashMap<String, MempoolTokenSignal>,
    pub timestamp: i64,
    pub data_age_seconds: u64,
}
