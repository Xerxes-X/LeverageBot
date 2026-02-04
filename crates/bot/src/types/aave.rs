use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Result of `Pool.getUserAccountData()` — values in USD (8 decimals).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAccountData {
    #[serde(with = "rust_decimal::serde::str")]
    pub total_collateral_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub total_debt_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub available_borrow_usd: Decimal,
    /// Average weighted liquidation threshold (basis points / 10000).
    #[serde(with = "rust_decimal::serde::str")]
    pub current_liquidation_threshold: Decimal,
    /// Average weighted LTV (basis points / 10000).
    #[serde(with = "rust_decimal::serde::str")]
    pub ltv: Decimal,
    /// Health factor in WAD (1e18 = HF of 1.0).
    #[serde(with = "rust_decimal::serde::str")]
    pub health_factor: Decimal,
}

/// Reserve-level data from Aave V3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReserveData {
    /// Variable borrow rate in RAY.
    #[serde(with = "rust_decimal::serde::str")]
    pub variable_borrow_rate: Decimal,
    /// Current utilization as a fraction.
    #[serde(with = "rust_decimal::serde::str")]
    pub utilization_rate: Decimal,
    pub isolation_mode_enabled: bool,
    #[serde(with = "rust_decimal::serde::str")]
    pub debt_ceiling: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub current_isolated_debt: Decimal,
}

/// Borrow rate details for cost analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorrowRateInfo {
    #[serde(with = "rust_decimal::serde::str")]
    pub variable_rate_apr: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub utilization_rate: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub optimal_utilization: Decimal,
    /// Rate if utilization were at the kink point.
    #[serde(with = "rust_decimal::serde::str")]
    pub rate_at_kink: Decimal,
}
