use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Direction of a leveraged position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PositionDirection {
    /// Collateral = volatile (WBNB), Debt = stable (USDT/USDC).
    Long,
    /// Collateral = stable (USDC), Debt = volatile (WBNB).
    Short,
}

impl PositionDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Long => "long",
            Self::Short => "short",
        }
    }
}

/// Lifecycle action on a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PositionAction {
    Open,
    Close,
    Deleverage,
    Increase,
}

/// Current state of an open position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionState {
    pub direction: PositionDirection,
    pub debt_token: String,
    pub collateral_token: String,
    #[serde(with = "rust_decimal::serde::str")]
    pub debt_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub collateral_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub initial_debt_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub initial_collateral_usd: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub health_factor: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub borrow_rate_ray: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub liquidation_threshold: Decimal,
    pub open_timestamp: i64,
}
