use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::position::PositionDirection;

/// Market regime detected by the Hurst exponent (Layer 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarketRegime {
    /// Hurst > 0.55, ATR ratio 1.0–3.0x — momentum signals boosted.
    Trending,
    /// Hurst < 0.45 — mean-reversion signals boosted.
    MeanReverting,
    /// Hurst 0.45–0.55, ATR ratio < 1.0 — all weights reduced to 0.8x.
    Ranging,
    /// ATR ratio > 3.0 — all weights reduced to 0.7x.
    Volatile,
}

/// A single signal source's contribution (Layer 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalComponent {
    /// Source identifier, e.g. "order_book_imbalance", "vpin", "technical".
    pub source: String,
    /// Tier: 1 (highest reliability), 2, or 3.
    pub tier: u8,
    pub direction: PositionDirection,
    /// Signal strength in [-1.0, 1.0].
    #[serde(with = "rust_decimal::serde::str")]
    pub strength: Decimal,
    /// Tier-dependent weight.
    #[serde(with = "rust_decimal::serde::str")]
    pub weight: Decimal,
    /// Self-assessed confidence in [0.0, 1.0].
    #[serde(with = "rust_decimal::serde::str")]
    pub confidence: Decimal,
    /// Freshness of the underlying data.
    pub data_age_seconds: u64,
}

/// Snapshot of all computed indicators at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorSnapshot {
    #[serde(with = "rust_decimal::serde::str")]
    pub price: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub ema_20: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub ema_50: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub ema_200: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub rsi_14: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub macd_line: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub macd_signal: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub macd_histogram: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub bb_upper: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub bb_middle: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub bb_lower: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub atr_14: Decimal,
    /// ATR(14) / 50-period ATR average — regime classifier input.
    #[serde(with = "rust_decimal::serde::str")]
    pub atr_ratio: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub volume: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub volume_20_avg: Decimal,
    /// Hurst exponent from R/S analysis.
    #[serde(with = "rust_decimal::serde::str")]
    pub hurst: Decimal,
    /// VPIN value.
    #[serde(with = "rust_decimal::serde::str")]
    pub vpin: Decimal,
    /// Order book imbalance in [-1, 1].
    #[serde(with = "rust_decimal::serde::str")]
    pub obi: Decimal,
    /// Recent close prices for Hurst computation (last 200).
    pub recent_prices: Vec<Decimal>,
}

/// Composite trade signal emitted after Layer 4 (position sizing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeSignal {
    pub direction: PositionDirection,
    /// Ensemble confidence in [0.0, 1.0].
    #[serde(with = "rust_decimal::serde::str")]
    pub confidence: Decimal,
    /// "momentum", "mean_reversion", "blended", or "manual".
    pub strategy_mode: String,
    pub regime: MarketRegime,
    pub components: Vec<SignalComponent>,
    /// Kelly-derived position size in USD.
    #[serde(with = "rust_decimal::serde::str")]
    pub recommended_size_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub hurst_exponent: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub garch_volatility: Decimal,
    pub timestamp: i64,
}
