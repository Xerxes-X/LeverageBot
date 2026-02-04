use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::signal::TradeSignal;

/// Health factor tier determining polling frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HFTier {
    /// HF > 2.0 — poll every 15 s.
    Safe,
    /// 1.5–2.0 — poll every 5 s.
    Watch,
    /// 1.3–1.5 — poll every 2 s.
    Warning,
    /// < 1.3 — poll every 1 s + Chainlink events.
    Critical,
}

/// Health status emitted by the health monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    #[serde(with = "rust_decimal::serde::str")]
    pub health_factor: Decimal,
    pub tier: HFTier,
    #[serde(with = "rust_decimal::serde::str")]
    pub collateral_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub debt_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub borrow_rate_apr: Decimal,
    pub oracle_fresh: bool,
    /// Predicted health factor 10 minutes ahead (compound interest model).
    #[serde(with = "rust_decimal::serde::str")]
    pub predicted_hf_10m: Decimal,
    pub timestamp: i64,
}

/// Events consumed by the Strategy task from the shared bounded channel.
#[derive(Debug, Clone)]
pub enum SignalEvent {
    Health(HealthStatus),
    Trade(TradeSignal),
    Shutdown,
}
