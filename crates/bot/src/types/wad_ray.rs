//! Newtype wrappers for on-chain fixed-point values.
//!
//! Prevents accidental mixing of WAD-scaled (18 decimals) and RAY-scaled
//! (27 decimals) values at the type level.

use alloy::primitives::{uint, U256};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::fmt;
use std::str::FromStr;

// ---------------------------------------------------------------------------
// WAD (1e18) — health factors, prices, amounts
// ---------------------------------------------------------------------------

/// WAD-scaled value (18 decimals). Used for health factors, prices, amounts.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Wad(pub U256);

const WAD_SCALE: Decimal = dec!(1_000_000_000_000_000_000);

impl Wad {
    pub const ONE: Wad = Wad(U256::from_limbs([1_000_000_000_000_000_000, 0, 0, 0]));
    pub const ZERO: Wad = Wad(U256::ZERO);

    /// Convert on-chain WAD (U256) to off-chain `Decimal`.
    pub fn to_decimal(self) -> Decimal {
        let raw = Decimal::from_str(&self.0.to_string()).unwrap_or_default();
        raw / WAD_SCALE
    }

    /// Create from a `U256` that is already WAD-scaled.
    pub fn from_raw(val: U256) -> Self {
        Self(val)
    }

    /// Inner `U256`.
    pub fn raw(self) -> U256 {
        self.0
    }
}

impl fmt::Debug for Wad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Wad({})", self.to_decimal())
    }
}

impl fmt::Display for Wad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_decimal())
    }
}

impl From<U256> for Wad {
    fn from(val: U256) -> Self {
        Self(val)
    }
}

// ---------------------------------------------------------------------------
// RAY (1e27) — Aave interest rates
// ---------------------------------------------------------------------------

/// RAY-scaled value (27 decimals). Used for Aave interest rates.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ray(pub U256);

const RAY_SCALE: Decimal = dec!(1_000_000_000_000_000_000_000_000_000);

impl Ray {
    pub const ONE: Ray = Ray(uint!(1_000_000_000_000_000_000_000_000_000_U256));
    pub const ZERO: Ray = Ray(U256::ZERO);

    /// Convert on-chain RAY (U256) to off-chain `Decimal`.
    pub fn to_decimal(self) -> Decimal {
        let raw = Decimal::from_str(&self.0.to_string()).unwrap_or_default();
        raw / RAY_SCALE
    }

    /// Convert RAY interest rate to APR percentage (× 100).
    pub fn to_apr_percent(self) -> Decimal {
        self.to_decimal() * dec!(100)
    }

    /// Create from a `U256` that is already RAY-scaled.
    pub fn from_raw(val: U256) -> Self {
        Self(val)
    }

    /// Inner `U256`.
    pub fn raw(self) -> U256 {
        self.0
    }
}

impl fmt::Debug for Ray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ray({})", self.to_decimal())
    }
}

impl fmt::Display for Ray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_decimal())
    }
}

impl From<U256> for Ray {
    fn from(val: U256) -> Self {
        Self(val)
    }
}

// ---------------------------------------------------------------------------
// Free-standing conversion helpers
// ---------------------------------------------------------------------------

/// Convert a raw U256 WAD value to `Decimal`. Convenience alias for `Wad::to_decimal`.
pub fn wad_to_decimal(wad: U256) -> Decimal {
    Wad(wad).to_decimal()
}

/// Convert a raw U256 RAY value to `Decimal`. Convenience alias for `Ray::to_decimal`.
pub fn ray_to_decimal(ray: U256) -> Decimal {
    Ray(ray).to_decimal()
}

/// Convert a Chainlink 8-decimal price (U256) to `Decimal`.
pub fn price_to_decimal(raw: U256) -> Decimal {
    let raw_dec = Decimal::from_str(&raw.to_string()).unwrap_or_default();
    raw_dec / dec!(100_000_000)
}

/// Convert Aave base-currency (8-decimal USD) U256 to `Decimal`.
pub fn base_currency_to_decimal(raw: U256) -> Decimal {
    price_to_decimal(raw)
}

/// Convert basis points (u256) to a fraction `Decimal`.
pub fn bps_to_decimal(bps: U256) -> Decimal {
    let raw_dec = Decimal::from_str(&bps.to_string()).unwrap_or_default();
    raw_dec / dec!(10_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wad_one_is_one() {
        assert_eq!(Wad::ONE.to_decimal(), dec!(1));
    }

    #[test]
    fn test_wad_zero() {
        assert_eq!(Wad::ZERO.to_decimal(), dec!(0));
    }

    #[test]
    fn test_wad_health_factor() {
        // 1.5e18 = health factor of 1.5
        let hf = Wad::from_raw(U256::from(1_500_000_000_000_000_000u128));
        assert_eq!(hf.to_decimal(), dec!(1.5));
    }

    #[test]
    fn test_ray_one_is_one() {
        assert_eq!(Ray::ONE.to_decimal(), dec!(1));
    }

    #[test]
    fn test_ray_to_apr() {
        // 3% APR in RAY = 0.03 * 1e27 = 3e25
        let rate = Ray::from_raw(U256::from(30_000_000_000_000_000_000_000_000u128));
        assert_eq!(rate.to_apr_percent(), dec!(3));
    }

    #[test]
    fn test_wad_to_decimal_free_fn() {
        let val = U256::from(2_500_000_000_000_000_000u128);
        assert_eq!(wad_to_decimal(val), dec!(2.5));
    }

    #[test]
    fn test_ray_to_decimal_free_fn() {
        let val = U256::from(500_000_000_000_000_000_000_000_000u128);
        assert_eq!(ray_to_decimal(val), dec!(0.5));
    }

    #[test]
    fn test_price_to_decimal() {
        // $2500.00 in 8-decimal format = 250_000_000_000
        let raw = U256::from(250_000_000_000u128);
        assert_eq!(price_to_decimal(raw), dec!(2500));
    }

    #[test]
    fn test_bps_to_decimal() {
        // 7500 bps = 0.75
        assert_eq!(bps_to_decimal(U256::from(7500u64)), dec!(0.75));
    }
}
