//! P&L tracking and position history for BSC Leverage Bot.
//!
//! SQLite-backed position lifecycle tracking. Records opens, closes,
//! deleverages, and periodic snapshots. Computes realized/unrealized P&L
//! including accrued interest costs.
//!
//! Uses `sqlx::query()` runtime queries (not compile-time `query!` macros)
//! since the database schema is created via `sqlx::migrate!`.

use anyhow::{Context, Result};
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Row, SqlitePool};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

use crate::types::{PositionDirection, PositionState, RealizedPnL, TradingStats};

const SECONDS_PER_HOUR: i64 = 3600;

/// SQLite-backed P&L tracker for position lifecycle management.
///
/// Tracks opens, closes, deleverages, and periodic snapshots.
/// Computes realized/unrealized P&L including accrued interest costs.
pub struct PnLTracker {
    pool: SqlitePool,
}

impl PnLTracker {
    /// Create a new tracker backed by the given SQLite database path.
    ///
    /// Creates the database file if it doesn't exist (`mode=rwc`).
    /// Runs migrations from `migrations/` to ensure the schema is up to date.
    pub async fn new(db_path: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(1) // SQLite is single-writer
            .connect(&format!("sqlite:{db_path}?mode=rwc"))
            .await
            .context("failed to connect to SQLite database")?;

        sqlx::migrate!("../../migrations")
            .run(&pool)
            .await
            .context("failed to run database migrations")?;

        info!(db_path, "PnL tracker initialized");
        Ok(Self { pool })
    }

    // -----------------------------------------------------------------------
    // Record operations
    // -----------------------------------------------------------------------

    /// Record a new position opening.
    ///
    /// Inserts into both `positions` and `transactions` tables atomically.
    /// Returns the auto-generated position ID.
    pub async fn record_open(
        &self,
        position: &PositionState,
        tx_hash: &str,
        gas_cost_usd: Decimal,
    ) -> Result<i64> {
        let now = now_unix();
        let direction = position.direction.as_str();
        let initial_debt = position.initial_debt_usd.to_string();
        let initial_collateral = position.initial_collateral_usd.to_string();
        let gas_cost = gas_cost_usd.to_string();
        let borrow_rate = position.borrow_rate_ray.to_string();

        let mut tx = self.pool.begin().await?;

        let result = sqlx::query(
            "INSERT INTO positions (direction, open_timestamp, debt_token, collateral_token, \
             initial_debt_amount, initial_collateral_amount, total_gas_costs_usd, \
             open_tx_hash, open_borrow_rate_apr) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(direction)
        .bind(now)
        .bind(&position.debt_token)
        .bind(&position.collateral_token)
        .bind(&initial_debt)
        .bind(&initial_collateral)
        .bind(&gas_cost)
        .bind(tx_hash)
        .bind(&borrow_rate)
        .execute(&mut *tx)
        .await?;

        let position_id = result.last_insert_rowid();

        sqlx::query(
            "INSERT INTO transactions \
             (position_id, tx_hash, timestamp, tx_type, gas_cost_usd, success) \
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(position_id)
        .bind(tx_hash)
        .bind(now)
        .bind("open")
        .bind(&gas_cost)
        .bind(true)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        info!(
            position_id,
            direction,
            debt_token = %position.debt_token,
            collateral_token = %position.collateral_token,
            tx_hash,
            "position opened"
        );
        Ok(position_id)
    }

    /// Record position close and compute realized P&L.
    ///
    /// `tokens_received` is the surplus returned after repaying the flash loan
    /// (i.e. the profit/loss amount before gas costs).
    pub async fn record_close(
        &self,
        position_id: i64,
        tx_hash: &str,
        gas_cost_usd: Decimal,
        tokens_received: Decimal,
        close_reason: &str,
    ) -> Result<RealizedPnL> {
        let now = now_unix();

        let row = sqlx::query("SELECT * FROM positions WHERE id = ?")
            .bind(position_id)
            .fetch_one(&self.pool)
            .await
            .context("position not found")?;

        let prev_gas: Decimal = row
            .get::<String, _>("total_gas_costs_usd")
            .parse()
            .unwrap_or(Decimal::ZERO);
        let total_gas = prev_gas + gas_cost_usd;

        let gross_pnl = tokens_received;
        let accrued_interest = Decimal::ZERO;
        let flash_premiums = Decimal::ZERO;
        let net_pnl = gross_pnl - total_gas - flash_premiums - accrued_interest;

        let mut tx = self.pool.begin().await?;

        sqlx::query(
            "UPDATE positions SET close_timestamp = ?, close_tx_hash = ?, close_reason = ?, \
             realized_pnl_usd = ?, total_gas_costs_usd = ? WHERE id = ?",
        )
        .bind(now)
        .bind(tx_hash)
        .bind(close_reason)
        .bind(net_pnl.to_string())
        .bind(total_gas.to_string())
        .bind(position_id)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "INSERT INTO transactions \
             (position_id, tx_hash, timestamp, tx_type, gas_cost_usd, success) \
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(position_id)
        .bind(tx_hash)
        .bind(now)
        .bind("close")
        .bind(gas_cost_usd.to_string())
        .bind(true)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        let result = RealizedPnL {
            gross_pnl_usd: gross_pnl,
            accrued_interest_usd: accrued_interest,
            gas_costs_usd: total_gas,
            flash_loan_premiums_usd: flash_premiums,
            net_pnl_usd: net_pnl,
        };

        info!(
            position_id,
            net_pnl = %net_pnl,
            close_reason,
            tx_hash,
            "position closed"
        );
        Ok(result)
    }

    /// Record a deleverage event for an existing position.
    ///
    /// Accumulates gas costs and records the transaction.
    pub async fn record_deleverage(
        &self,
        position_id: i64,
        tx_hash: &str,
        gas_cost_usd: Decimal,
    ) -> Result<()> {
        let now = now_unix();

        let row = sqlx::query("SELECT total_gas_costs_usd FROM positions WHERE id = ?")
            .bind(position_id)
            .fetch_one(&self.pool)
            .await
            .context("position not found")?;

        let prev_gas: Decimal = row
            .get::<String, _>("total_gas_costs_usd")
            .parse()
            .unwrap_or(Decimal::ZERO);
        let total_gas = prev_gas + gas_cost_usd;

        let mut tx = self.pool.begin().await?;

        sqlx::query("UPDATE positions SET total_gas_costs_usd = ? WHERE id = ?")
            .bind(total_gas.to_string())
            .bind(position_id)
            .execute(&mut *tx)
            .await?;

        sqlx::query(
            "INSERT INTO transactions \
             (position_id, tx_hash, timestamp, tx_type, gas_cost_usd, success) \
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(position_id)
        .bind(tx_hash)
        .bind(now)
        .bind("deleverage")
        .bind(gas_cost_usd.to_string())
        .bind(true)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        info!(
            position_id,
            gas_cost = %gas_cost_usd,
            tx_hash,
            "deleverage recorded"
        );
        Ok(())
    }

    /// Take a periodic snapshot of an open position's state.
    pub async fn snapshot(
        &self,
        position_id: i64,
        position: &PositionState,
    ) -> Result<()> {
        let unrealized = self.get_unrealized_pnl(position);
        let now = now_unix();

        sqlx::query(
            "INSERT INTO position_snapshots \
             (position_id, timestamp, collateral_value_usd, debt_value_usd, \
              health_factor, borrow_rate_apr, unrealized_pnl_usd) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(position_id)
        .bind(now)
        .bind(position.collateral_usd.to_string())
        .bind(position.debt_usd.to_string())
        .bind(position.health_factor.to_string())
        .bind(position.borrow_rate_ray.to_string())
        .bind(unrealized.to_string())
        .execute(&self.pool)
        .await?;

        debug!(position_id, unrealized = %unrealized, "snapshot recorded");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // P&L computation (sync — pure math, no I/O)
    // -----------------------------------------------------------------------

    /// Compute unrealized P&L for an open position.
    ///
    /// - **LONG**: profit = (current_collateral - initial_collateral) - accrued_interest
    /// - **SHORT**: profit = (initial_debt - current_debt) - accrued_interest
    pub fn get_unrealized_pnl(&self, position: &PositionState) -> Decimal {
        let accrued = self.get_accrued_interest(position);
        match position.direction {
            PositionDirection::Long => {
                (position.collateral_usd - position.initial_collateral_usd) - accrued
            }
            PositionDirection::Short => {
                (position.initial_debt_usd - position.debt_usd) - accrued
            }
        }
    }

    /// Estimate accrued interest from the difference between current and initial debt.
    pub fn get_accrued_interest(&self, position: &PositionState) -> Decimal {
        if position.debt_usd <= position.initial_debt_usd {
            Decimal::ZERO
        } else {
            position.debt_usd - position.initial_debt_usd
        }
    }

    // -----------------------------------------------------------------------
    // Query operations
    // -----------------------------------------------------------------------

    /// Compute trading statistics for a rolling window.
    ///
    /// `window_days = None` computes all-time statistics.
    pub async fn get_rolling_stats(&self, window_days: Option<u32>) -> Result<TradingStats> {
        let rows = if let Some(days) = window_days {
            let cutoff = now_unix() - (days as i64 * 86400);
            sqlx::query(
                "SELECT * FROM positions WHERE close_timestamp IS NOT NULL \
                 AND close_timestamp >= ? ORDER BY close_timestamp DESC",
            )
            .bind(cutoff)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT * FROM positions WHERE close_timestamp IS NOT NULL \
                 ORDER BY close_timestamp DESC",
            )
            .fetch_all(&self.pool)
            .await?
        };

        let total_trades = rows.len() as u32;
        if total_trades == 0 {
            return Ok(TradingStats {
                total_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
                total_pnl_usd: Decimal::ZERO,
                avg_pnl_per_trade_usd: Decimal::ZERO,
                win_rate: Decimal::ZERO,
                sharpe_ratio: Decimal::ZERO,
                avg_hold_duration_hours: Decimal::ZERO,
                current_drawdown_pct: Decimal::ZERO,
                max_drawdown_pct: Decimal::ZERO,
            });
        }

        let mut pnls = Vec::with_capacity(rows.len());
        let mut hold_durations = Vec::with_capacity(rows.len());
        let mut winning = 0u32;
        let mut losing = 0u32;

        for row in &rows {
            let pnl: Decimal = row
                .get::<String, _>("realized_pnl_usd")
                .parse()
                .unwrap_or(Decimal::ZERO);
            pnls.push(pnl);

            if pnl > Decimal::ZERO {
                winning += 1;
            } else if pnl < Decimal::ZERO {
                losing += 1;
            }

            let close_ts: i64 = row.get("close_timestamp");
            let open_ts: i64 = row.get("open_timestamp");
            hold_durations.push(close_ts - open_ts);
        }

        let total_pnl: Decimal = pnls.iter().copied().sum();
        let avg_pnl = total_pnl / Decimal::from(total_trades);

        let win_rate = Decimal::from(winning) / Decimal::from(total_trades);

        let avg_hold_hours = if !hold_durations.is_empty() {
            let sum: i64 = hold_durations.iter().sum();
            Decimal::from(sum)
                / Decimal::from(hold_durations.len() as i64)
                / Decimal::from(SECONDS_PER_HOUR)
        } else {
            Decimal::ZERO
        };

        let sharpe = compute_sharpe(&pnls);
        let (current_dd, max_dd) = compute_drawdowns(&pnls);

        Ok(TradingStats {
            total_trades,
            winning_trades: winning,
            losing_trades: losing,
            total_pnl_usd: total_pnl,
            avg_pnl_per_trade_usd: avg_pnl,
            win_rate,
            sharpe_ratio: sharpe,
            avg_hold_duration_hours: avg_hold_hours,
            current_drawdown_pct: current_dd,
            max_drawdown_pct: max_dd,
        })
    }

    /// Get the current drawdown percentage from all-time data.
    pub async fn current_drawdown_pct(&self) -> Result<Decimal> {
        let rows = sqlx::query(
            "SELECT realized_pnl_usd FROM positions \
             WHERE close_timestamp IS NOT NULL \
             ORDER BY close_timestamp ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        let pnls: Vec<Decimal> = rows
            .iter()
            .map(|r| {
                r.get::<String, _>("realized_pnl_usd")
                    .parse()
                    .unwrap_or(Decimal::ZERO)
            })
            .collect();

        if pnls.is_empty() {
            return Ok(Decimal::ZERO);
        }

        let (current_dd, _) = compute_drawdowns(&pnls);
        Ok(current_dd)
    }

    /// Get the current open position from the database, if any.
    pub async fn get_open_position_id(&self) -> Result<Option<i64>> {
        let row = sqlx::query(
            "SELECT id FROM positions WHERE close_timestamp IS NULL \
             ORDER BY open_timestamp DESC LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| r.get("id")))
    }

    /// Direct access to the underlying pool (for advanced queries).
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

// ---------------------------------------------------------------------------
// Free functions — pure computation, no I/O
// ---------------------------------------------------------------------------

/// Get current UNIX timestamp in seconds.
fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_secs() as i64
}

/// Compute Sharpe ratio from a list of P&L values.
///
/// Uses sample standard deviation (N-1 denominator).
fn compute_sharpe(pnls: &[Decimal]) -> Decimal {
    if pnls.len() < 2 {
        return Decimal::ZERO;
    }

    let n = Decimal::from(pnls.len() as u32);
    let mean: Decimal = pnls.iter().copied().sum::<Decimal>() / n;
    let variance: Decimal = pnls
        .iter()
        .map(|p| {
            let diff = *p - mean;
            diff * diff
        })
        .sum::<Decimal>()
        / (n - dec!(1));

    if variance <= Decimal::ZERO {
        return Decimal::ZERO;
    }

    // rust_decimal "maths" feature provides sqrt()
    match variance.sqrt() {
        Some(std_dev) if std_dev > Decimal::ZERO => mean / std_dev,
        _ => Decimal::ZERO,
    }
}

/// Compute current and max drawdown from a chronological P&L series.
///
/// Returns `(current_drawdown_pct, max_drawdown_pct)` as positive fractions
/// (e.g. `0.15` = 15% drawdown).
fn compute_drawdowns(pnls: &[Decimal]) -> (Decimal, Decimal) {
    if pnls.is_empty() {
        return (Decimal::ZERO, Decimal::ZERO);
    }

    let mut cumulative = Decimal::ZERO;
    let mut peak = Decimal::ZERO;
    let mut max_drawdown = Decimal::ZERO;

    for pnl in pnls {
        cumulative += pnl;
        if cumulative > peak {
            peak = cumulative;
        }
        if peak > Decimal::ZERO {
            let dd = (peak - cumulative) / peak;
            if dd > max_drawdown {
                max_drawdown = dd;
            }
        }
    }

    let current_dd = if peak > Decimal::ZERO {
        ((peak - cumulative) / peak).max(Decimal::ZERO)
    } else {
        Decimal::ZERO
    };

    (current_dd, max_drawdown)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- compute_sharpe -------------------------------------------------------

    #[test]
    fn sharpe_empty_returns_zero() {
        assert_eq!(compute_sharpe(&[]), Decimal::ZERO);
    }

    #[test]
    fn sharpe_single_value_returns_zero() {
        assert_eq!(compute_sharpe(&[dec!(100)]), Decimal::ZERO);
    }

    #[test]
    fn sharpe_positive_returns_positive() {
        let pnls = vec![dec!(100), dec!(200), dec!(150), dec!(180)];
        let sharpe = compute_sharpe(&pnls);
        assert!(sharpe > Decimal::ZERO);
    }

    #[test]
    fn sharpe_zero_variance_returns_zero() {
        let pnls = vec![dec!(100), dec!(100), dec!(100)];
        assert_eq!(compute_sharpe(&pnls), Decimal::ZERO);
    }

    #[test]
    fn sharpe_negative_mean_returns_negative() {
        let pnls = vec![dec!(-100), dec!(-200), dec!(-50)];
        let sharpe = compute_sharpe(&pnls);
        assert!(sharpe < Decimal::ZERO);
    }

    // -- compute_drawdowns ----------------------------------------------------

    #[test]
    fn drawdowns_empty_returns_zero() {
        let (current, max) = compute_drawdowns(&[]);
        assert_eq!(current, Decimal::ZERO);
        assert_eq!(max, Decimal::ZERO);
    }

    #[test]
    fn drawdowns_all_positive_no_drawdown() {
        let pnls = vec![dec!(100), dec!(50), dec!(75)];
        let (current, max) = compute_drawdowns(&pnls);
        assert_eq!(current, Decimal::ZERO);
        assert_eq!(max, Decimal::ZERO);
    }

    #[test]
    fn drawdowns_with_loss() {
        // cumulative: 100, 50, 125
        // peak:       100, 100, 125
        // dd:          0,  0.5, 0
        let pnls = vec![dec!(100), dec!(-50), dec!(75)];
        let (current, max) = compute_drawdowns(&pnls);
        assert_eq!(current, Decimal::ZERO);
        assert_eq!(max, dec!(0.5));
    }

    #[test]
    fn drawdowns_ending_in_loss() {
        // cumulative: 100, 60
        // peak:       100, 100
        // dd:          0,  0.4
        let pnls = vec![dec!(100), dec!(-40)];
        let (current, max) = compute_drawdowns(&pnls);
        assert_eq!(current, dec!(0.4));
        assert_eq!(max, dec!(0.4));
    }

    #[test]
    fn drawdowns_multiple_peaks() {
        // cumulative: 100, 50, 120, 60
        // peak:       100, 100, 120, 120
        // dd:          0,  0.5,  0,  0.5
        let pnls = vec![dec!(100), dec!(-50), dec!(70), dec!(-60)];
        let (current, max) = compute_drawdowns(&pnls);
        assert_eq!(current, dec!(0.5));
        assert_eq!(max, dec!(0.5));
    }

    // -- unrealized P&L -------------------------------------------------------

    #[test]
    fn unrealized_pnl_long_profit() {
        let position = PositionState {
            direction: PositionDirection::Long,
            debt_token: "USDT".into(),
            collateral_token: "WBNB".into(),
            debt_usd: dec!(5000),
            collateral_usd: dec!(6000),
            initial_debt_usd: dec!(5000),
            initial_collateral_usd: dec!(5200),
            health_factor: dec!(1.8),
            borrow_rate_ray: dec!(5),
            liquidation_threshold: dec!(0.80),
            open_timestamp: 0,
        };

        // Using a zeroed pool is fine — get_unrealized_pnl is a pure function
        // that doesn't use the pool.
        let unrealized = compute_unrealized_pnl_long(&position);
        // (6000 - 5200) - 0 = 800 (no accrued interest since debt == initial_debt)
        assert_eq!(unrealized, dec!(800));
    }

    #[test]
    fn unrealized_pnl_short_profit() {
        let position = PositionState {
            direction: PositionDirection::Short,
            debt_token: "WBNB".into(),
            collateral_token: "USDC".into(),
            debt_usd: dec!(4500),
            collateral_usd: dec!(5000),
            initial_debt_usd: dec!(5000),
            initial_collateral_usd: dec!(5200),
            health_factor: dec!(1.8),
            borrow_rate_ray: dec!(5),
            liquidation_threshold: dec!(0.80),
            open_timestamp: 0,
        };

        let unrealized = compute_unrealized_pnl_short(&position);
        // (5000 - 4500) - 0 = 500
        assert_eq!(unrealized, dec!(500));
    }

    /// Test helper: compute unrealized PnL for a long position without needing PnLTracker.
    fn compute_unrealized_pnl_long(position: &PositionState) -> Decimal {
        let accrued = if position.debt_usd > position.initial_debt_usd {
            position.debt_usd - position.initial_debt_usd
        } else {
            Decimal::ZERO
        };
        (position.collateral_usd - position.initial_collateral_usd) - accrued
    }

    /// Test helper: compute unrealized PnL for a short position.
    fn compute_unrealized_pnl_short(position: &PositionState) -> Decimal {
        let accrued = if position.debt_usd > position.initial_debt_usd {
            position.debt_usd - position.initial_debt_usd
        } else {
            Decimal::ZERO
        };
        (position.initial_debt_usd - position.debt_usd) - accrued
    }
}
