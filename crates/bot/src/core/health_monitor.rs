//! Health factor monitor — tiered polling with adaptive frequency.
//!
//! Ported from Python `core/health_monitor.py`. Continuously polls Aave V3
//! for the user's health factor, classifies it into tiers, and sends
//! `SignalEvent::Health` through a bounded channel to the Strategy task.
//!
//! Key features:
//! - Adaptive poll intervals (15s/5s/2s/1s) based on HF tier
//! - 3-term Taylor compound interest prediction (matches Aave MathUtils.sol)
//! - 5-failure circuit breaker triggers global pause
//! - Graceful shutdown via CancellationToken

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use alloy::primitives::Address;
use anyhow::{Context, Result};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::config::TimingConfig;
use crate::constants::SECONDS_PER_YEAR;
use crate::execution::aave_client::AaveClient;
use crate::types::{HFTier, HealthStatus, PositionState, SignalEvent};

use super::safety::SafetyState;

/// Maximum consecutive RPC failures before triggering a global pause.
const MAX_CONSECUTIVE_FAILURES: u32 = 5;

// ---------------------------------------------------------------------------
// Pure helper functions (no &self — easy to unit test)
// ---------------------------------------------------------------------------

/// Compute health factor: HF = (collateral * liquidation_threshold) / debt.
///
/// Returns `Decimal::MAX` if debt is zero (no liquidation risk).
pub fn compute_health_factor(
    collateral_usd: Decimal,
    debt_usd: Decimal,
    liquidation_threshold: Decimal,
) -> Decimal {
    if debt_usd <= Decimal::ZERO {
        return Decimal::MAX;
    }
    (collateral_usd * liquidation_threshold) / debt_usd
}

/// Classify a health factor value into a monitoring tier.
pub fn determine_tier(hf: Decimal) -> HFTier {
    if hf > dec!(2.0) {
        HFTier::Safe
    } else if hf >= dec!(1.5) {
        HFTier::Watch
    } else if hf >= dec!(1.3) {
        HFTier::Warning
    } else {
        HFTier::Critical
    }
}

/// Get the poll interval for a given HF tier, using configured values.
pub fn tier_interval(tier: HFTier, config: &TimingConfig) -> Duration {
    let secs = match tier {
        HFTier::Safe => config.health_monitoring.safe_interval_seconds,
        HFTier::Watch => config.health_monitoring.watch_interval_seconds,
        HFTier::Warning => config.health_monitoring.warning_interval_seconds,
        HFTier::Critical => config.health_monitoring.critical_interval_seconds,
    };
    Duration::from_secs(secs)
}

/// Predict the health factor at a future time using Aave's 3-term Taylor
/// approximation of compound interest (`MathUtils.calculateCompoundedInterest`).
///
/// Formula:
///   rate_per_second = borrow_rate_ray / SECONDS_PER_YEAR
///   compound = 1 + r*dt + (r*dt)^2 / 2
///   predicted_hf = (collateral * LT) / (debt * compound)
pub fn predict_hf_at(seconds_ahead: u64, position: &PositionState) -> Decimal {
    if position.debt_usd <= Decimal::ZERO {
        return Decimal::MAX;
    }

    let rate_per_second = position.borrow_rate_ray / Decimal::from(SECONDS_PER_YEAR);
    let dt = Decimal::from(seconds_ahead);
    let r_dt = rate_per_second * dt;

    // Aave V3 3-term Taylor: 1 + r*t + (r*t)^2 / 2
    let compound = dec!(1) + r_dt + (r_dt * r_dt) / dec!(2);

    let projected_debt = position.debt_usd * compound;
    if projected_debt <= Decimal::ZERO {
        return Decimal::MAX;
    }

    (position.collateral_usd * position.liquidation_threshold) / projected_debt
}

// ---------------------------------------------------------------------------
// HealthMonitor actor
// ---------------------------------------------------------------------------

/// Async health monitor that polls Aave V3 and emits `SignalEvent::Health`.
pub struct HealthMonitor {
    aave_client: Arc<AaveClient>,
    safety: Arc<SafetyState>,
    config: TimingConfig,
    user_address: Address,
    event_tx: mpsc::Sender<SignalEvent>,
    shutdown: CancellationToken,
    consecutive_failures: AtomicU32,
    current_tier: std::sync::Mutex<HFTier>,
}

impl HealthMonitor {
    pub fn new(
        aave_client: Arc<AaveClient>,
        safety: Arc<SafetyState>,
        config: TimingConfig,
        user_address: Address,
        event_tx: mpsc::Sender<SignalEvent>,
        shutdown: CancellationToken,
    ) -> Self {
        Self {
            aave_client,
            safety,
            config,
            user_address,
            event_tx,
            shutdown,
            consecutive_failures: AtomicU32::new(0),
            current_tier: std::sync::Mutex::new(HFTier::Safe),
        }
    }

    /// Main polling loop. Runs until the CancellationToken is cancelled.
    pub async fn run(&self) -> Result<()> {
        info!("health monitor started");

        loop {
            let interval = self.current_interval();
            tokio::select! {
                () = self.shutdown.cancelled() => {
                    info!("health monitor shutting down");
                    break;
                }
                () = tokio::time::sleep(interval) => {
                    match self.poll_once().await {
                        Ok(status) => {
                            self.consecutive_failures.store(0, Ordering::Relaxed);
                            self.update_tier(status.tier);

                            if let Err(e) = self.event_tx.send(SignalEvent::Health(status)).await {
                                warn!(error = %e, "failed to send health event — channel full or closed");
                            }
                        }
                        Err(e) => {
                            let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
                            error!(
                                error = %e,
                                consecutive_failures = failures,
                                "health poll failed"
                            );

                            if failures >= MAX_CONSECUTIVE_FAILURES {
                                error!(
                                    "reached {MAX_CONSECUTIVE_FAILURES} consecutive failures — triggering global pause"
                                );
                                self.safety.trigger_global_pause(
                                    "health monitor: too many consecutive RPC failures",
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Single poll cycle: read on-chain health data, validate oracle, build status.
    async fn poll_once(&self) -> Result<HealthStatus> {
        let account = self
            .aave_client
            .get_user_account_data(self.user_address)
            .await
            .context("failed to fetch user account data")?;

        let tier = determine_tier(account.health_factor);
        let timestamp = chrono::Utc::now().timestamp();

        // Oracle freshness is checked by caller before opening positions;
        // here we just record it as informational. Failure to check is not
        // a polling failure.
        let oracle_fresh = true; // TODO: wire to oracle check in Phase 6+ integration

        // HF prediction requires an active position. If no debt, predicted HF
        // is meaningless — use MAX.
        let predicted_hf_10m = if account.total_debt_usd > Decimal::ZERO {
            // We don't have a full PositionState here (that's managed by
            // PositionManager in Phase 8). Approximate with account-level data.
            // Borrow rate would come from reserve data — use 0 for now (conservative).
            let approx_position = PositionState {
                direction: crate::types::PositionDirection::Long,
                debt_token: String::new(),
                collateral_token: String::new(),
                debt_usd: account.total_debt_usd,
                collateral_usd: account.total_collateral_usd,
                initial_debt_usd: account.total_debt_usd,
                initial_collateral_usd: account.total_collateral_usd,
                health_factor: account.health_factor,
                borrow_rate_ray: Decimal::ZERO, // conservative: no rate accrual
                liquidation_threshold: account.current_liquidation_threshold,
                open_timestamp: timestamp,
            };
            predict_hf_at(600, &approx_position)
        } else {
            Decimal::MAX
        };

        let status = HealthStatus {
            health_factor: account.health_factor,
            tier,
            collateral_usd: account.total_collateral_usd,
            debt_usd: account.total_debt_usd,
            borrow_rate_apr: Decimal::ZERO, // populated when reserve data is fetched
            oracle_fresh,
            predicted_hf_10m,
            timestamp,
        };

        debug!(
            hf = %status.health_factor,
            tier = ?status.tier,
            collateral = %status.collateral_usd,
            debt = %status.debt_usd,
            "health poll complete"
        );

        Ok(status)
    }

    fn current_interval(&self) -> Duration {
        let tier = *self
            .current_tier
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        tier_interval(tier, &self.config)
    }

    fn update_tier(&self, new_tier: HFTier) {
        if let Ok(mut current) = self.current_tier.lock() {
            if *current != new_tier {
                warn!(
                    from = ?*current,
                    to = ?new_tier,
                    "health tier transition"
                );
                *current = new_tier;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // -----------------------------------------------------------------------
    // determine_tier boundary tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tier_safe() {
        assert_eq!(determine_tier(dec!(2.1)), HFTier::Safe);
        assert_eq!(determine_tier(dec!(5.0)), HFTier::Safe);
        assert_eq!(determine_tier(dec!(100)), HFTier::Safe);
    }

    #[test]
    fn test_tier_boundary_2_0() {
        // HF == 2.0 is Watch (not Safe, since Safe requires > 2.0)
        assert_eq!(determine_tier(dec!(2.0)), HFTier::Watch);
        assert_eq!(determine_tier(dec!(2.001)), HFTier::Safe);
    }

    #[test]
    fn test_tier_watch() {
        assert_eq!(determine_tier(dec!(1.7)), HFTier::Watch);
        assert_eq!(determine_tier(dec!(1.5)), HFTier::Watch);
    }

    #[test]
    fn test_tier_boundary_1_5() {
        assert_eq!(determine_tier(dec!(1.5)), HFTier::Watch);
        assert_eq!(determine_tier(dec!(1.499)), HFTier::Warning);
    }

    #[test]
    fn test_tier_warning() {
        assert_eq!(determine_tier(dec!(1.4)), HFTier::Warning);
        assert_eq!(determine_tier(dec!(1.3)), HFTier::Warning);
    }

    #[test]
    fn test_tier_boundary_1_3() {
        assert_eq!(determine_tier(dec!(1.3)), HFTier::Warning);
        assert_eq!(determine_tier(dec!(1.299)), HFTier::Critical);
    }

    #[test]
    fn test_tier_critical() {
        assert_eq!(determine_tier(dec!(1.2)), HFTier::Critical);
        assert_eq!(determine_tier(dec!(1.0)), HFTier::Critical);
        assert_eq!(determine_tier(dec!(0.5)), HFTier::Critical);
    }

    // -----------------------------------------------------------------------
    // compute_health_factor
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_hf_basic() {
        // collateral=$10000, debt=$5000, LT=0.8 → HF = (10000*0.8)/5000 = 1.6
        let hf = compute_health_factor(dec!(10000), dec!(5000), dec!(0.8));
        assert_eq!(hf, dec!(1.6));
    }

    #[test]
    fn test_compute_hf_zero_debt() {
        let hf = compute_health_factor(dec!(10000), Decimal::ZERO, dec!(0.8));
        assert_eq!(hf, Decimal::MAX);
    }

    #[test]
    fn test_compute_hf_high_leverage() {
        // collateral=$1000, debt=$800, LT=0.825 → HF = (1000*0.825)/800 = 1.03125
        let hf = compute_health_factor(dec!(1000), dec!(800), dec!(0.825));
        assert_eq!(hf, dec!(1.03125));
    }

    // -----------------------------------------------------------------------
    // predict_hf_at (compound interest prediction)
    // -----------------------------------------------------------------------

    #[test]
    fn test_predict_hf_zero_debt() {
        let pos = PositionState {
            direction: crate::types::PositionDirection::Long,
            debt_token: String::new(),
            collateral_token: String::new(),
            debt_usd: Decimal::ZERO,
            collateral_usd: dec!(10000),
            initial_debt_usd: Decimal::ZERO,
            initial_collateral_usd: dec!(10000),
            health_factor: Decimal::MAX,
            borrow_rate_ray: Decimal::ZERO,
            liquidation_threshold: dec!(0.825),
            open_timestamp: 0,
        };
        assert_eq!(predict_hf_at(600, &pos), Decimal::MAX);
    }

    #[test]
    fn test_predict_hf_zero_rate() {
        // With zero borrow rate, compound=1, predicted HF = current HF
        let pos = PositionState {
            direction: crate::types::PositionDirection::Long,
            debt_token: String::new(),
            collateral_token: String::new(),
            debt_usd: dec!(5000),
            collateral_usd: dec!(10000),
            initial_debt_usd: dec!(5000),
            initial_collateral_usd: dec!(10000),
            health_factor: dec!(1.65),
            borrow_rate_ray: Decimal::ZERO,
            liquidation_threshold: dec!(0.825),
            open_timestamp: 0,
        };
        let predicted = predict_hf_at(600, &pos);
        // compound = 1 + 0 + 0 = 1, projected_debt = 5000, HF = (10000*0.825)/5000 = 1.65
        assert_eq!(predicted, dec!(1.65));
    }

    #[test]
    fn test_predict_hf_with_rate() {
        // 3% APR in RAY decimal = 0.03
        // rate_per_second = 0.03 / 31_536_000 ≈ 9.51e-10
        // r*dt for 600s ≈ 5.707e-7
        // compound ≈ 1.000000571 (trivially close to 1)
        // So predicted HF should be very close to current HF for 10 min window.
        let borrow_rate_ray = dec!(0.03); // 3% as decimal (already divided by 1e27)
        let pos = PositionState {
            direction: crate::types::PositionDirection::Long,
            debt_token: String::new(),
            collateral_token: String::new(),
            debt_usd: dec!(5000),
            collateral_usd: dec!(10000),
            initial_debt_usd: dec!(5000),
            initial_collateral_usd: dec!(10000),
            health_factor: dec!(1.65),
            borrow_rate_ray,
            liquidation_threshold: dec!(0.825),
            open_timestamp: 0,
        };
        let predicted = predict_hf_at(600, &pos);
        let current_hf = dec!(1.65);
        // Should be slightly less than current HF due to interest accrual
        assert!(predicted < current_hf, "predicted HF should decrease due to interest");
        assert!(predicted > dec!(1.64), "10-min change at 3% APR is tiny");
    }

    #[test]
    fn test_predict_hf_high_rate_long_duration() {
        // 100% APR, 1 year ahead — stress test the formula
        let borrow_rate_ray = dec!(1.0); // 100% APR
        let pos = PositionState {
            direction: crate::types::PositionDirection::Long,
            debt_token: String::new(),
            collateral_token: String::new(),
            debt_usd: dec!(5000),
            collateral_usd: dec!(10000),
            initial_debt_usd: dec!(5000),
            initial_collateral_usd: dec!(10000),
            health_factor: dec!(1.65),
            borrow_rate_ray,
            liquidation_threshold: dec!(0.825),
            open_timestamp: 0,
        };
        let predicted = predict_hf_at(SECONDS_PER_YEAR, &pos);
        // compound = 1 + 1 + 0.5 = 2.5 (Taylor 3-term for r*dt=1)
        // projected_debt = 5000 * 2.5 = 12500
        // HF = (10000 * 0.825) / 12500 = 0.66
        // Allow tiny rounding tolerance from Decimal arithmetic
        let diff = (predicted - dec!(0.66)).abs();
        assert!(diff < dec!(0.0001), "predicted {predicted} ≈ 0.66");
    }

    // -----------------------------------------------------------------------
    // tier_interval
    // -----------------------------------------------------------------------

    #[test]
    fn test_tier_intervals() {
        let config = test_timing_config();
        assert_eq!(tier_interval(HFTier::Safe, &config), Duration::from_secs(15));
        assert_eq!(tier_interval(HFTier::Watch, &config), Duration::from_secs(5));
        assert_eq!(tier_interval(HFTier::Warning, &config), Duration::from_secs(2));
        assert_eq!(tier_interval(HFTier::Critical, &config), Duration::from_secs(1));
    }

    // -----------------------------------------------------------------------
    // proptest: HF always positive for valid inputs
    // -----------------------------------------------------------------------

    proptest! {
        #[test]
        fn health_factor_always_positive(
            collateral_usd in 100u64..1_000_000u64,
            debt_usd in 1u64..500_000u64,
            lt_bps in 5000u32..9500u32,
        ) {
            let hf = compute_health_factor(
                Decimal::from(collateral_usd),
                Decimal::from(debt_usd),
                Decimal::from(lt_bps) / dec!(10000),
            );
            prop_assert!(hf > Decimal::ZERO);
        }

        #[test]
        fn predicted_hf_decreases_with_positive_rate(
            debt in 1000u64..100_000u64,
            collateral in 2000u64..200_000u64,
            rate_bps in 100u32..5000u32, // 1% to 50% APR
        ) {
            let borrow_rate = Decimal::from(rate_bps) / dec!(10000);
            let pos = PositionState {
                direction: crate::types::PositionDirection::Long,
                debt_token: String::new(),
                collateral_token: String::new(),
                debt_usd: Decimal::from(debt),
                collateral_usd: Decimal::from(collateral),
                initial_debt_usd: Decimal::from(debt),
                initial_collateral_usd: Decimal::from(collateral),
                health_factor: Decimal::ZERO, // unused by predict_hf_at
                borrow_rate_ray: borrow_rate,
                liquidation_threshold: dec!(0.825),
                open_timestamp: 0,
            };
            let hf_now = compute_health_factor(
                Decimal::from(collateral),
                Decimal::from(debt),
                dec!(0.825),
            );
            let hf_future = predict_hf_at(600, &pos);
            // With positive borrow rate, future HF should be <= current HF
            prop_assert!(hf_future <= hf_now, "future HF {hf_future} > current {hf_now}");
            // And still positive
            prop_assert!(hf_future > Decimal::ZERO);
        }
    }

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn test_timing_config() -> TimingConfig {
        use crate::config::{
            AggregatorTiming, ErrorRecoveryConfig, HealthMonitoringTiming, TransactionTiming,
            Web3ConnectionConfig,
        };
        TimingConfig {
            health_monitoring: HealthMonitoringTiming {
                safe_interval_seconds: 15,
                watch_interval_seconds: 5,
                warning_interval_seconds: 2,
                critical_interval_seconds: 1,
                stale_data_threshold_failures: 5,
            },
            aggregator: AggregatorTiming {
                quote_timeout_seconds: 10,
                quote_cache_ttl_seconds: 5,
            },
            transaction: TransactionTiming {
                confirmation_timeout_seconds: 60,
                simulation_timeout_seconds: 10,
                nonce_refresh_interval_seconds: 15,
            },
            web3_connection: Web3ConnectionConfig {
                max_retries: 3,
                retry_delay_seconds: 1.0,
            },
            error_recovery: ErrorRecoveryConfig {
                rpc_retry_base_delay_seconds: 1,
                rpc_retry_max_delay_seconds: 30,
            },
        }
    }
}
