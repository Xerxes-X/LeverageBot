use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Realized P&L computed when a position is closed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealizedPnL {
    #[serde(with = "rust_decimal::serde::str")]
    pub gross_pnl_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub accrued_interest_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub gas_costs_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub flash_loan_premiums_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub net_pnl_usd: Decimal,
}

/// Rolling trading statistics for alpha-decay monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingStats {
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    #[serde(with = "rust_decimal::serde::str")]
    pub total_pnl_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub avg_pnl_per_trade_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub win_rate: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub sharpe_ratio: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub sortino_ratio: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub calmar_ratio: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub avg_hold_duration_hours: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub current_drawdown_pct: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub max_drawdown_pct: Decimal,
}

/// Strategy health report from alpha-decay monitoring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrategyHealthReport {
    pub alpha_decay_detected: bool,
    #[serde(with = "rust_decimal::serde::str")]
    pub accuracy_ratio: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub sharpe_ratio: Decimal,
    pub recommendations: Vec<String>,
    pub dynamic_confidence_threshold: Option<Decimal>,
}
