use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::position::PositionDirection;
use super::timeframe::{Timeframe, TradingStyle};

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

// ============================================================================
// Multi-Timeframe Signal Types
// ============================================================================

/// Indicator snapshot tagged with its source timeframe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeframeIndicatorSnapshot {
    /// The timeframe this snapshot belongs to.
    pub timeframe: Timeframe,
    /// The computed indicators.
    pub snapshot: IndicatorSnapshot,
    /// Data freshness in milliseconds (time since last candle close).
    pub data_freshness_ms: u64,
}

/// A signal component with timeframe context for multi-TF aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTfSignalComponent {
    /// Source identifier, e.g. "order_book_imbalance", "vpin", "technical".
    pub source: String,
    /// Tier: 1 (highest reliability), 2, or 3.
    pub tier: u8,
    /// The timeframe this signal was computed from.
    pub timeframe: Timeframe,
    pub direction: PositionDirection,
    /// Signal strength in [-1.0, 1.0].
    #[serde(with = "rust_decimal::serde::str")]
    pub strength: Decimal,
    /// Tier-dependent weight (before timeframe weighting).
    #[serde(with = "rust_decimal::serde::str")]
    pub weight: Decimal,
    /// Self-assessed confidence in [0.0, 1.0].
    #[serde(with = "rust_decimal::serde::str")]
    pub confidence: Decimal,
    /// Freshness of the underlying data.
    pub data_age_seconds: u64,
}

impl From<MultiTfSignalComponent> for SignalComponent {
    fn from(mtf: MultiTfSignalComponent) -> Self {
        Self {
            source: mtf.source,
            tier: mtf.tier,
            direction: mtf.direction,
            strength: mtf.strength,
            weight: mtf.weight,
            confidence: mtf.confidence,
            data_age_seconds: mtf.data_age_seconds,
        }
    }
}

/// Multi-timeframe aggregated trade signal.
///
/// Extends TradeSignal with per-timeframe confidence breakdown and
/// trading style metadata for hierarchical signal evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTfTradeSignal {
    /// Final aggregated direction (Long or Short).
    pub direction: PositionDirection,
    /// Overall ensemble confidence in [0.0, 1.0].
    #[serde(with = "rust_decimal::serde::str")]
    pub confidence: Decimal,
    /// "momentum", "mean_reversion", "blended", or "manual".
    pub strategy_mode: String,
    /// Active trading style (Scalping, MidFrequency, Swing).
    pub trading_style: TradingStyle,
    /// Market regime detected from higher timeframes.
    pub regime: MarketRegime,
    /// Per-timeframe confidence breakdown.
    /// Positive = bullish, negative = bearish, abs value = confidence.
    #[serde(with = "timeframe_decimal_map")]
    pub timeframe_confidence: HashMap<Timeframe, Decimal>,
    /// All signal components across all timeframes.
    pub components: Vec<MultiTfSignalComponent>,
    /// Kelly-derived position size in USD.
    #[serde(with = "rust_decimal::serde::str")]
    pub recommended_size_usd: Decimal,
    /// Hurst exponent from the highest timeframe.
    #[serde(with = "rust_decimal::serde::str")]
    pub hurst_exponent: Decimal,
    /// GARCH volatility forecast.
    #[serde(with = "rust_decimal::serde::str")]
    pub garch_volatility: Decimal,
    /// Primary timeframe that triggered the entry.
    pub entry_timeframe: Timeframe,
    /// Number of timeframes agreeing on direction.
    pub agreeing_timeframes: u8,
    /// Total number of evaluated timeframes.
    pub total_timeframes: u8,
    pub timestamp: i64,
}

impl MultiTfTradeSignal {
    /// Calculate the agreement ratio (0.0 to 1.0).
    #[must_use]
    pub fn agreement_ratio(&self) -> Decimal {
        if self.total_timeframes == 0 {
            return Decimal::ZERO;
        }
        Decimal::from(self.agreeing_timeframes) / Decimal::from(self.total_timeframes)
    }

    /// Check if higher timeframes (H4, H6) agree with the signal direction.
    #[must_use]
    pub fn higher_tf_aligned(&self) -> bool {
        let higher_tfs = [Timeframe::H4, Timeframe::H6];
        higher_tfs.iter().all(|tf| {
            self.timeframe_confidence.get(tf).map_or(true, |conf| {
                match self.direction {
                    PositionDirection::Long => *conf >= Decimal::ZERO,
                    PositionDirection::Short => *conf <= Decimal::ZERO,
                }
            })
        })
    }

    /// Convert to a legacy single-TF TradeSignal (for backwards compatibility).
    #[must_use]
    pub fn to_trade_signal(&self) -> TradeSignal {
        TradeSignal {
            direction: self.direction,
            confidence: self.confidence,
            strategy_mode: self.strategy_mode.clone(),
            regime: self.regime,
            components: self.components.iter().cloned().map(Into::into).collect(),
            recommended_size_usd: self.recommended_size_usd,
            hurst_exponent: self.hurst_exponent,
            garch_volatility: self.garch_volatility,
            timestamp: self.timestamp,
        }
    }
}

/// Custom serde module for HashMap<Timeframe, Decimal> serialization.
mod timeframe_decimal_map {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(
        map: &HashMap<Timeframe, Decimal>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;
        let mut ser_map = serializer.serialize_map(Some(map.len()))?;
        for (k, v) in map {
            ser_map.serialize_entry(&k.as_binance_interval(), &v.to_string())?;
        }
        ser_map.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<Timeframe, Decimal>, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{MapAccess, Visitor};
        use std::fmt;

        struct MapVisitor;

        impl<'de> Visitor<'de> for MapVisitor {
            type Value = HashMap<Timeframe, Decimal>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a map of timeframe to decimal")
            }

            fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut map = HashMap::new();
                while let Some((key, value)) = access.next_entry::<String, String>()? {
                    let tf = Timeframe::from_binance_interval(&key)
                        .ok_or_else(|| serde::de::Error::custom(format!("invalid timeframe: {key}")))?;
                    let dec: Decimal = value
                        .parse()
                        .map_err(|e| serde::de::Error::custom(format!("invalid decimal: {e}")))?;
                    map.insert(tf, dec);
                }
                Ok(map)
            }
        }

        deserializer.deserialize_map(MapVisitor)
    }
}
