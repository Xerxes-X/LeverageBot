use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// A single OHLCV candle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    pub timestamp: i64,
    #[serde(with = "rust_decimal::serde::str")]
    pub open: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub high: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub low: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub close: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub volume: Decimal,
}

/// Individual trade from exchange (for VPIN computation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    #[serde(with = "rust_decimal::serde::str")]
    pub price: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub quantity: Decimal,
    pub timestamp: i64,
    /// `true` = sell-initiated (maker was buyer).
    pub is_buyer_maker: bool,
}

/// Order book snapshot (for OBI signal).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    /// (price, quantity) sorted by price descending.
    pub bids: Vec<(Decimal, Decimal)>,
    /// (price, quantity) sorted by price ascending.
    pub asks: Vec<(Decimal, Decimal)>,
    pub timestamp: i64,
}

/// Exchange flow data for directional signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeFlows {
    #[serde(with = "rust_decimal::serde::str")]
    pub inflow_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub outflow_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub avg_hourly_flow: Decimal,
    pub data_age_seconds: u64,
}

/// Pending swap volume from mempool (Tier 3 stub).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingSwapVolume {
    #[serde(with = "rust_decimal::serde::str")]
    pub volume_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub avg_volume_usd: Decimal,
    /// Net buy ratio in [0.0, 1.0].
    #[serde(with = "rust_decimal::serde::str")]
    pub net_buy_ratio: Decimal,
    pub window_seconds: u64,
}

/// Liquidation level from Aave V3 position data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidationLevel {
    #[serde(with = "rust_decimal::serde::str")]
    pub price: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub total_collateral_at_risk_usd: Decimal,
    pub position_count: u32,
}
