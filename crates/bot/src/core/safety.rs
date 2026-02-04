//! Safety gate — default-to-deny position and transaction validation.
//!
//! Ported from Python `core/safety.py`. Every trading action must pass through
//! `SafetyState` before execution. On any internal error or missing config the
//! system blocks the action (fail-closed, not fail-open).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tracing::{error, info, warn};

use crate::config::PositionConfig;
use crate::errors::BotError;

const SECONDS_PER_DAY: u64 = 86_400;

/// Centralized safety controls for all trading actions.
///
/// Thread-safe: uses atomics for hot-path flags and `Mutex` only for the
/// cooldown timer and action timestamps. All checks return
/// `Err(BotError::SafetyBlocked)` on failure — callers never need to
/// interpret a boolean.
pub struct SafetyState {
    dry_run: bool,
    max_position_usd: Decimal,
    max_leverage_ratio: Decimal,
    min_health_factor: Decimal,
    max_gas_price_gwei: u64,
    global_pause: AtomicBool,
    last_action_time: Mutex<Option<Instant>>,
    cooldown_seconds: u64,
    daily_tx_count: AtomicU32,
    max_tx_per_24h: u32,
    /// Wall-clock timestamps (seconds since process start) of recent actions for 24h rate limiting.
    action_timestamps: Mutex<VecDeque<Instant>>,
}

impl SafetyState {
    /// Build from validated config. If config is unavailable, use `SafetyState::lockdown()`.
    pub fn from_config(config: &PositionConfig) -> Self {
        Self {
            dry_run: config.dry_run,
            max_position_usd: Decimal::from(config.max_position_usd),
            max_leverage_ratio: config.max_leverage_ratio,
            min_health_factor: config.min_health_factor,
            max_gas_price_gwei: config.max_gas_price_gwei,
            global_pause: AtomicBool::new(false),
            last_action_time: Mutex::new(None),
            cooldown_seconds: config.cooldown_between_actions_seconds,
            daily_tx_count: AtomicU32::new(0),
            max_tx_per_24h: config.max_transactions_per_24h,
            action_timestamps: Mutex::new(VecDeque::new()),
        }
    }

    /// Lockdown mode: dry_run=true, max_position=0, max_leverage=1.0.
    /// Used when config is missing or corrupt.
    pub fn lockdown() -> Self {
        warn!("SafetyState entering lockdown mode — no actions will be permitted");
        Self {
            dry_run: true,
            max_position_usd: Decimal::ZERO,
            max_leverage_ratio: dec!(1.0),
            min_health_factor: dec!(2.0),
            max_gas_price_gwei: 0,
            global_pause: AtomicBool::new(true),
            last_action_time: Mutex::new(None),
            cooldown_seconds: u64::MAX,
            daily_tx_count: AtomicU32::new(0),
            max_tx_per_24h: 0,
            action_timestamps: Mutex::new(VecDeque::new()),
        }
    }

    // -----------------------------------------------------------------------
    // Position validation
    // -----------------------------------------------------------------------

    /// Validate whether a new position can be opened.
    ///
    /// Checks (in order): global pause, dry-run, position size, leverage,
    /// cooldown, and 24h rate limit.
    pub fn can_open_position(
        &self,
        amount_usd: Decimal,
        leverage: Decimal,
    ) -> Result<(), BotError> {
        // 1. Global pause
        if self.is_paused() {
            return Err(BotError::SafetyBlocked {
                reason: "global pause is active".into(),
            });
        }

        // 2. Dry-run mode blocks real execution
        if self.dry_run {
            return Err(BotError::SafetyBlocked {
                reason: "dry-run mode is active".into(),
            });
        }

        // 3. Position size cap
        if amount_usd > self.max_position_usd {
            return Err(BotError::SafetyBlocked {
                reason: format!(
                    "position size ${amount_usd} exceeds max ${max}",
                    max = self.max_position_usd
                ),
            });
        }

        // 4. Leverage cap
        if leverage > self.max_leverage_ratio {
            return Err(BotError::SafetyBlocked {
                reason: format!(
                    "leverage {leverage}x exceeds max {max}x",
                    max = self.max_leverage_ratio
                ),
            });
        }

        // 5. Cooldown
        self.check_cooldown()?;

        // 6. 24h rate limit
        self.check_daily_limit()?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Transaction validation
    // -----------------------------------------------------------------------

    /// Validate whether a transaction can be submitted at the given gas price.
    pub fn can_submit_tx(&self, gas_price_gwei: u64) -> Result<(), BotError> {
        if self.is_paused() {
            return Err(BotError::SafetyBlocked {
                reason: "global pause is active".into(),
            });
        }

        if gas_price_gwei > self.max_gas_price_gwei {
            return Err(BotError::SafetyBlocked {
                reason: format!(
                    "gas price {gas_price_gwei} gwei exceeds max {max} gwei",
                    max = self.max_gas_price_gwei
                ),
            });
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Health factor gate
    // -----------------------------------------------------------------------

    /// Block action if current health factor is below the configured minimum.
    pub fn check_health_factor(&self, current_hf: Decimal) -> Result<(), BotError> {
        if current_hf < self.min_health_factor {
            return Err(BotError::SafetyBlocked {
                reason: format!(
                    "health factor {current_hf} below minimum {min}",
                    min = self.min_health_factor
                ),
            });
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Global pause
    // -----------------------------------------------------------------------

    /// Activate the global pause (emergency kill switch).
    pub fn trigger_global_pause(&self, reason: &str) {
        self.global_pause.store(true, Ordering::SeqCst);
        error!(reason, "GLOBAL PAUSE activated");
    }

    /// Clear the global pause (manual recovery only).
    pub fn resume(&self) {
        self.global_pause.store(false, Ordering::SeqCst);
        info!("global pause cleared — trading resumed");
    }

    /// Check if the system is paused.
    pub fn is_paused(&self) -> bool {
        self.global_pause.load(Ordering::SeqCst)
    }

    /// Whether the system is in dry-run mode.
    pub fn is_dry_run(&self) -> bool {
        self.dry_run
    }

    // -----------------------------------------------------------------------
    // Action recording
    // -----------------------------------------------------------------------

    /// Record that an action was taken. Updates cooldown timer and 24h counter.
    pub fn record_action(&self) {
        let now = Instant::now();

        // Update cooldown
        if let Ok(mut last) = self.last_action_time.lock() {
            *last = Some(now);
        }

        // Push to 24h window
        if let Ok(mut timestamps) = self.action_timestamps.lock() {
            timestamps.push_back(now);
            self.daily_tx_count
                .store(timestamps.len() as u32, Ordering::Relaxed);
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn check_cooldown(&self) -> Result<(), BotError> {
        let last = self
            .last_action_time
            .lock()
            .map_err(|_| BotError::SafetyBlocked {
                reason: "cooldown lock poisoned".into(),
            })?;

        if let Some(last_time) = *last {
            let elapsed = last_time.elapsed().as_secs();
            if elapsed < self.cooldown_seconds {
                let remaining = self.cooldown_seconds - elapsed;
                return Err(BotError::SafetyBlocked {
                    reason: format!("cooldown: {remaining}s remaining"),
                });
            }
        }
        Ok(())
    }

    fn check_daily_limit(&self) -> Result<(), BotError> {
        let mut timestamps = self
            .action_timestamps
            .lock()
            .map_err(|_| BotError::SafetyBlocked {
                reason: "daily limit lock poisoned".into(),
            })?;

        // Prune entries older than 24 hours
        let cutoff = Instant::now()
            .checked_sub(std::time::Duration::from_secs(SECONDS_PER_DAY))
            .unwrap_or_else(Instant::now);
        while timestamps.front().is_some_and(|t| *t < cutoff) {
            timestamps.pop_front();
        }
        self.daily_tx_count
            .store(timestamps.len() as u32, Ordering::Relaxed);

        if timestamps.len() as u32 >= self.max_tx_per_24h {
            return Err(BotError::SafetyBlocked {
                reason: format!(
                    "daily transaction limit reached ({max}/24h)",
                    max = self.max_tx_per_24h
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a SafetyState with sensible test defaults.
    fn test_safety() -> SafetyState {
        SafetyState {
            dry_run: false,
            max_position_usd: dec!(10_000),
            max_leverage_ratio: dec!(3.0),
            min_health_factor: dec!(1.5),
            max_gas_price_gwei: 10,
            global_pause: AtomicBool::new(false),
            last_action_time: Mutex::new(None),
            cooldown_seconds: 30,
            daily_tx_count: AtomicU32::new(0),
            max_tx_per_24h: 50,
            action_timestamps: Mutex::new(VecDeque::new()),
        }
    }

    #[test]
    fn test_lockdown_blocks_everything() {
        let s = SafetyState::lockdown();
        let err = s
            .can_open_position(dec!(1), dec!(1))
            .expect_err("lockdown should block");
        assert!(err.to_string().contains("global pause"));
    }

    #[test]
    fn test_dry_run_blocks_open() {
        let mut s = test_safety();
        s.dry_run = true;
        let err = s
            .can_open_position(dec!(100), dec!(2))
            .expect_err("dry-run should block");
        assert!(err.to_string().contains("dry-run"));
    }

    #[test]
    fn test_position_size_cap() {
        let s = test_safety();
        assert!(s.can_open_position(dec!(5_000), dec!(2)).is_ok());
        let err = s
            .can_open_position(dec!(15_000), dec!(2))
            .expect_err("over-size should block");
        assert!(err.to_string().contains("position size"));
    }

    #[test]
    fn test_leverage_cap() {
        let s = test_safety();
        assert!(s.can_open_position(dec!(1_000), dec!(3.0)).is_ok());
        let err = s
            .can_open_position(dec!(1_000), dec!(3.5))
            .expect_err("over-leverage should block");
        assert!(err.to_string().contains("leverage"));
    }

    #[test]
    fn test_gas_price_cap() {
        let s = test_safety();
        assert!(s.can_submit_tx(5).is_ok());
        assert!(s.can_submit_tx(10).is_ok());
        let err = s.can_submit_tx(15).expect_err("high gas should block");
        assert!(err.to_string().contains("gas price"));
    }

    #[test]
    fn test_global_pause_blocks() {
        let s = test_safety();
        assert!(s.can_open_position(dec!(100), dec!(2)).is_ok());
        s.trigger_global_pause("test");
        assert!(s.is_paused());
        assert!(s.can_open_position(dec!(100), dec!(2)).is_err());
        assert!(s.can_submit_tx(5).is_err());

        // Resume clears pause
        s.resume();
        assert!(!s.is_paused());
        assert!(s.can_open_position(dec!(100), dec!(2)).is_ok());
    }

    #[test]
    fn test_cooldown_enforcement() {
        let s = test_safety();
        // First action should succeed
        assert!(s.can_open_position(dec!(100), dec!(2)).is_ok());
        s.record_action();

        // Immediately after, cooldown should block
        let err = s
            .can_open_position(dec!(100), dec!(2))
            .expect_err("cooldown should block");
        assert!(err.to_string().contains("cooldown"));
    }

    #[test]
    fn test_daily_tx_limit() {
        let s = SafetyState {
            dry_run: false,
            max_position_usd: dec!(10_000),
            max_leverage_ratio: dec!(3.0),
            min_health_factor: dec!(1.5),
            max_gas_price_gwei: 10,
            global_pause: AtomicBool::new(false),
            last_action_time: Mutex::new(None),
            cooldown_seconds: 0, // disable cooldown for this test
            daily_tx_count: AtomicU32::new(0),
            max_tx_per_24h: 3,
            action_timestamps: Mutex::new(VecDeque::new()),
        };

        // Record 3 actions — should hit the limit
        for _ in 0..3 {
            s.record_action();
        }
        let err = s
            .can_open_position(dec!(100), dec!(2))
            .expect_err("daily limit should block");
        assert!(err.to_string().contains("daily transaction limit"));
    }

    #[test]
    fn test_health_factor_gate() {
        let s = test_safety();
        assert!(s.check_health_factor(dec!(2.0)).is_ok());
        assert!(s.check_health_factor(dec!(1.5)).is_ok());
        let err = s
            .check_health_factor(dec!(1.4))
            .expect_err("low HF should block");
        assert!(err.to_string().contains("health factor"));
    }

    #[test]
    fn test_default_to_deny_on_poison() {
        // If a mutex is poisoned, the safety check should block (fail-closed).
        let s = test_safety();
        // Poison the cooldown mutex by panicking inside it.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = s.last_action_time.lock().unwrap();
            panic!("deliberate poison");
        }));
        assert!(result.is_err());

        // After poisoning, can_open_position should fail (not succeed).
        assert!(s.can_open_position(dec!(100), dec!(2)).is_err());
    }
}
