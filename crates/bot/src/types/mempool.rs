use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
