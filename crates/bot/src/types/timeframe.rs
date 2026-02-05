//! Timeframe and trading style types for multi-timeframe analysis.
//!
//! Supports 8 timeframes from 1-minute to 6-hour candles with
//! associated metadata for signal weighting and data fetching.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported trading timeframes.
///
/// Ordered from shortest to longest for iteration purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Timeframe {
    /// 1-minute candles (scalping)
    M1,
    /// 5-minute candles (scalping)
    M5,
    /// 15-minute candles (mid-frequency)
    M15,
    /// 30-minute candles (mid-frequency)
    M30,
    /// 1-hour candles (swing)
    H1,
    /// 2-hour candles (swing)
    H2,
    /// 4-hour candles (swing)
    H4,
    /// 6-hour candles (swing/position)
    H6,
}

impl Timeframe {
    /// All timeframes in ascending order (shortest to longest).
    pub const ALL: [Timeframe; 8] = [
        Timeframe::M1,
        Timeframe::M5,
        Timeframe::M15,
        Timeframe::M30,
        Timeframe::H1,
        Timeframe::H2,
        Timeframe::H4,
        Timeframe::H6,
    ];

    /// Binance kline interval string representation.
    #[must_use]
    pub fn as_binance_interval(&self) -> &'static str {
        match self {
            Self::M1 => "1m",
            Self::M5 => "5m",
            Self::M15 => "15m",
            Self::M30 => "30m",
            Self::H1 => "1h",
            Self::H2 => "2h",
            Self::H4 => "4h",
            Self::H6 => "6h",
        }
    }

    /// Parse from Binance interval string.
    #[must_use]
    pub fn from_binance_interval(s: &str) -> Option<Self> {
        match s {
            "1m" => Some(Self::M1),
            "5m" => Some(Self::M5),
            "15m" => Some(Self::M15),
            "30m" => Some(Self::M30),
            "1h" => Some(Self::H1),
            "2h" => Some(Self::H2),
            "4h" => Some(Self::H4),
            "6h" => Some(Self::H6),
            _ => None,
        }
    }

    /// Duration in seconds.
    #[must_use]
    pub const fn duration_secs(&self) -> u64 {
        match self {
            Self::M1 => 60,
            Self::M5 => 300,
            Self::M15 => 900,
            Self::M30 => 1800,
            Self::H1 => 3600,
            Self::H2 => 7200,
            Self::H4 => 14400,
            Self::H6 => 21600,
        }
    }

    /// Duration in milliseconds.
    #[must_use]
    pub const fn duration_ms(&self) -> u64 {
        self.duration_secs() * 1000
    }

    /// Default weight in hierarchical signal aggregation.
    ///
    /// Weights are calibrated so that:
    /// - Higher timeframes (4h, 6h) provide trend direction (33% combined)
    /// - Middle timeframes (1h, 2h) provide momentum confirmation (32% combined)
    /// - Lower timeframes (15m, 30m) provide entry refinement (22% combined)
    /// - Micro timeframes (1m, 5m) provide execution timing (13% combined)
    #[must_use]
    pub fn default_weight(&self) -> Decimal {
        match self {
            Self::M1 => dec!(0.05),
            Self::M5 => dec!(0.08),
            Self::M15 => dec!(0.10),
            Self::M30 => dec!(0.12),
            Self::H1 => dec!(0.20),
            Self::H2 => dec!(0.12),
            Self::H4 => dec!(0.18),
            Self::H6 => dec!(0.15),
        }
    }

    /// Whether this timeframe should use WebSocket streaming.
    ///
    /// Sub-hourly timeframes benefit from real-time updates,
    /// while hourly+ timeframes can use REST polling.
    #[must_use]
    pub const fn uses_websocket(&self) -> bool {
        matches!(self, Self::M1 | Self::M5 | Self::M15 | Self::M30)
    }

    /// Recommended number of historical candles for indicator computation.
    ///
    /// Shorter timeframes need more candles to cover the same time period
    /// for regime detection (Hurst requires 100+ data points).
    #[must_use]
    pub const fn required_candles(&self) -> u32 {
        match self {
            Self::M1 => 500,
            Self::M5 => 400,
            Self::M15 => 300,
            Self::M30 => 250,
            Self::H1 => 200,
            Self::H2 => 150,
            Self::H4 => 100,
            Self::H6 => 100,
        }
    }

    /// Cache TTL for OHLCV data based on timeframe.
    ///
    /// Shorter timeframes have shorter TTLs for fresher data.
    #[must_use]
    pub const fn cache_ttl_secs(&self) -> u64 {
        match self {
            Self::M1 => 5,
            Self::M5 => 10,
            Self::M15 => 20,
            Self::M30 => 30,
            Self::H1 => 60,
            Self::H2 => 90,
            Self::H4 => 120,
            Self::H6 => 180,
        }
    }

    /// Maximum allowed data staleness before signal is discarded.
    #[must_use]
    pub const fn max_staleness_secs(&self) -> u64 {
        match self {
            Self::M1 => 15,
            Self::M5 => 30,
            Self::M15 => 60,
            Self::M30 => 120,
            Self::H1 => 180,
            Self::H2 => 300,
            Self::H4 => 600,
            Self::H6 => 900,
        }
    }

    /// Get the trading style this timeframe primarily belongs to.
    #[must_use]
    pub const fn primary_style(&self) -> TradingStyle {
        match self {
            Self::M1 | Self::M5 => TradingStyle::Scalping,
            Self::M15 | Self::M30 => TradingStyle::MidFrequency,
            Self::H1 | Self::H2 | Self::H4 | Self::H6 => TradingStyle::Swing,
        }
    }

    /// Check if this is a higher timeframe (used for trend direction).
    #[must_use]
    pub const fn is_higher_tf(&self) -> bool {
        matches!(self, Self::H4 | Self::H6)
    }

    /// Check if this is a lower timeframe (used for entry timing).
    #[must_use]
    pub const fn is_lower_tf(&self) -> bool {
        matches!(self, Self::M1 | Self::M5 | Self::M15)
    }

    /// EMA period scaling factor relative to 1H base.
    ///
    /// Used to scale indicator parameters for different timeframes.
    /// Shorter TFs use larger periods to capture similar time windows.
    #[must_use]
    pub fn ema_scale_factor(&self) -> f64 {
        match self {
            Self::M1 => 60.0,  // 60x more candles per hour
            Self::M5 => 12.0,  // 12x more candles per hour
            Self::M15 => 4.0,  // 4x more candles per hour
            Self::M30 => 2.0,  // 2x more candles per hour
            Self::H1 => 1.0,   // Base reference
            Self::H2 => 0.5,   // Half as many candles per hour
            Self::H4 => 0.25,  // Quarter as many
            Self::H6 => 0.167, // 1/6 as many
        }
    }
}

impl fmt::Display for Timeframe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_binance_interval())
    }
}

/// Trading style classification.
///
/// Determines the primary timeframes used for signal generation
/// and the evaluation interval for the signal engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TradingStyle {
    /// High-frequency: 1m-5m primary, seconds to minutes holding
    Scalping,
    /// Medium-frequency: 15m-30m primary, minutes to hours holding
    #[default]
    MidFrequency,
    /// Low-frequency: 1h-6h primary, hours to days holding
    Swing,
}

impl TradingStyle {
    /// Primary timeframes for signal generation.
    #[must_use]
    pub const fn primary_timeframes(&self) -> &'static [Timeframe] {
        match self {
            Self::Scalping => &[Timeframe::M1, Timeframe::M5],
            Self::MidFrequency => &[Timeframe::M15, Timeframe::M30],
            Self::Swing => &[Timeframe::H1, Timeframe::H2, Timeframe::H4, Timeframe::H6],
        }
    }

    /// Higher timeframes for trend direction confirmation.
    #[must_use]
    pub const fn confirmation_timeframes(&self) -> &'static [Timeframe] {
        match self {
            Self::Scalping => &[Timeframe::M15, Timeframe::M30, Timeframe::H1],
            Self::MidFrequency => &[Timeframe::H1, Timeframe::H2, Timeframe::H4],
            Self::Swing => &[Timeframe::H4, Timeframe::H6],
        }
    }

    /// Signal engine evaluation interval in seconds.
    ///
    /// Scalping evaluates most frequently, swing least frequently.
    #[must_use]
    pub const fn evaluation_interval_secs(&self) -> u64 {
        match self {
            Self::Scalping => 5,
            Self::MidFrequency => 30,
            Self::Swing => 60,
        }
    }

    /// Recommended holding period range in hours.
    #[must_use]
    pub const fn holding_period_hours(&self) -> (u32, u32) {
        match self {
            Self::Scalping => (0, 1),      // < 1 hour
            Self::MidFrequency => (1, 8),  // 1-8 hours
            Self::Swing => (8, 168),       // 8 hours to 1 week
        }
    }

    /// Parse from string.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "scalping" | "scalp" => Some(Self::Scalping),
            "mid_frequency" | "midfrequency" | "mid" | "mft" => Some(Self::MidFrequency),
            "swing" | "position" => Some(Self::Swing),
            _ => None,
        }
    }
}

impl fmt::Display for TradingStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalping => write!(f, "scalping"),
            Self::MidFrequency => write!(f, "mid_frequency"),
            Self::Swing => write!(f, "swing"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_binance_interval() {
        assert_eq!(Timeframe::M1.as_binance_interval(), "1m");
        assert_eq!(Timeframe::H4.as_binance_interval(), "4h");
        assert_eq!(Timeframe::from_binance_interval("15m"), Some(Timeframe::M15));
        assert_eq!(Timeframe::from_binance_interval("invalid"), None);
    }

    #[test]
    fn test_timeframe_duration() {
        assert_eq!(Timeframe::M1.duration_secs(), 60);
        assert_eq!(Timeframe::H1.duration_secs(), 3600);
        assert_eq!(Timeframe::H6.duration_secs(), 21600);
    }

    #[test]
    fn test_timeframe_weights_sum_to_one() {
        let total: Decimal = Timeframe::ALL.iter().map(|tf| tf.default_weight()).sum();
        assert_eq!(total, dec!(1.00));
    }

    #[test]
    fn test_timeframe_uses_websocket() {
        assert!(Timeframe::M1.uses_websocket());
        assert!(Timeframe::M30.uses_websocket());
        assert!(!Timeframe::H1.uses_websocket());
        assert!(!Timeframe::H6.uses_websocket());
    }

    #[test]
    fn test_timeframe_ordering() {
        assert!(Timeframe::M1 < Timeframe::M5);
        assert!(Timeframe::M5 < Timeframe::H1);
        assert!(Timeframe::H1 < Timeframe::H6);
    }

    #[test]
    fn test_trading_style_timeframes() {
        let scalp = TradingStyle::Scalping;
        assert!(scalp.primary_timeframes().contains(&Timeframe::M1));
        assert!(!scalp.primary_timeframes().contains(&Timeframe::H1));

        let swing = TradingStyle::Swing;
        assert!(swing.primary_timeframes().contains(&Timeframe::H4));
        assert!(!swing.primary_timeframes().contains(&Timeframe::M5));
    }

    #[test]
    fn test_trading_style_parse() {
        assert_eq!(TradingStyle::from_str("scalping"), Some(TradingStyle::Scalping));
        assert_eq!(TradingStyle::from_str("MID_FREQUENCY"), Some(TradingStyle::MidFrequency));
        assert_eq!(TradingStyle::from_str("swing"), Some(TradingStyle::Swing));
        assert_eq!(TradingStyle::from_str("invalid"), None);
    }

    #[test]
    fn test_primary_style() {
        assert_eq!(Timeframe::M1.primary_style(), TradingStyle::Scalping);
        assert_eq!(Timeframe::M15.primary_style(), TradingStyle::MidFrequency);
        assert_eq!(Timeframe::H4.primary_style(), TradingStyle::Swing);
    }
}
