//! Comprehensive paper trading module for realistic strategy testing.
//!
//! Implements academic best practices for paper trading including:
//! - Real-time position tracking with mark-to-market P&L
//! - Take profit / stop loss monitoring
//! - Simulated portfolio with configurable starting capital
//! - Slippage modeling for realistic execution
//! - Performance attribution and risk-adjusted metrics
//!
//! References:
//! - Event-Driven Backtesting (QuantStart, 2024)
//! - Slippage: A Comprehensive Analysis (QuantJourney, 2024)
//! - Aave V3 Health Factor Mechanics (Aave Documentation)

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::core::data_service::DataService;
use crate::core::pnl_tracker::PnLTracker;
use crate::types::{PositionDirection, PositionState};

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Paper trading configuration for realistic simulation.
#[derive(Debug, Clone)]
pub struct PaperTradingConfig {
    /// Starting capital in USD
    pub starting_capital_usd: Decimal,

    /// Base slippage in basis points (1 bp = 0.01%)
    pub slippage_bps: u16,

    /// Market impact factor for order size
    pub market_impact_factor: Decimal,

    /// Default risk percentage for stop loss (e.g., 2.0 = 2%)
    pub default_stop_loss_pct: Decimal,

    /// Default profit target percentage (e.g., 6.0 = 6%)
    pub default_take_profit_pct: Decimal,

    /// Risk-reward ratio (profit target / stop loss)
    pub risk_reward_ratio: Decimal,

    /// Minimum health factor before auto-delever
    pub min_health_factor: Decimal,

    /// Maximum leverage allowed
    pub max_leverage: Decimal,
}

impl Default for PaperTradingConfig {
    fn default() -> Self {
        Self {
            starting_capital_usd: dec!(500.0),
            slippage_bps: 30,                    // 0.3% base slippage
            market_impact_factor: dec!(0.001),   // Logarithmic impact coefficient
            default_stop_loss_pct: dec!(2.0),    // 2% stop loss
            default_take_profit_pct: dec!(6.0),  // 6% take profit (3:1 R:R)
            risk_reward_ratio: dec!(3.0),
            min_health_factor: dec!(1.5),
            max_leverage: dec!(3.0),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Portfolio State
// ═══════════════════════════════════════════════════════════════════════════

/// Simulated portfolio state for paper trading.
#[derive(Debug, Clone)]
pub struct PaperPortfolio {
    /// Starting capital
    pub starting_capital_usd: Decimal,

    /// Current available cash (not in positions)
    pub available_cash_usd: Decimal,

    /// Total portfolio value (cash + positions)
    pub total_equity_usd: Decimal,

    /// Cumulative realized P&L
    pub cumulative_realized_pnl: Decimal,

    /// Current unrealized P&L from open positions
    pub current_unrealized_pnl: Decimal,

    /// Peak portfolio value (for drawdown calculation)
    pub peak_equity_usd: Decimal,

    /// Current drawdown percentage
    pub current_drawdown_pct: Decimal,

    /// Maximum drawdown experienced
    pub max_drawdown_pct: Decimal,

    /// Number of positions opened
    pub total_positions: u32,

    /// Last update timestamp
    pub last_update_timestamp: i64,
}

impl PaperPortfolio {
    pub fn new(starting_capital: Decimal) -> Self {
        Self {
            starting_capital_usd: starting_capital,
            available_cash_usd: starting_capital,
            total_equity_usd: starting_capital,
            cumulative_realized_pnl: Decimal::ZERO,
            current_unrealized_pnl: Decimal::ZERO,
            peak_equity_usd: starting_capital,
            current_drawdown_pct: Decimal::ZERO,
            max_drawdown_pct: Decimal::ZERO,
            total_positions: 0,
            last_update_timestamp: now_unix(),
        }
    }

    /// Update portfolio equity and calculate drawdown.
    pub fn update_equity(&mut self, new_equity: Decimal) {
        self.total_equity_usd = new_equity;

        // Update peak
        if new_equity > self.peak_equity_usd {
            self.peak_equity_usd = new_equity;
        }

        // Calculate drawdown
        if self.peak_equity_usd > Decimal::ZERO {
            self.current_drawdown_pct =
                ((self.peak_equity_usd - new_equity) / self.peak_equity_usd) * dec!(100);

            if self.current_drawdown_pct > self.max_drawdown_pct {
                self.max_drawdown_pct = self.current_drawdown_pct;
            }
        }

        self.last_update_timestamp = now_unix();
    }

    /// Return on equity (ROE) percentage.
    pub fn roe_pct(&self) -> Decimal {
        if self.starting_capital_usd > Decimal::ZERO {
            ((self.total_equity_usd - self.starting_capital_usd)
             / self.starting_capital_usd) * dec!(100)
        } else {
            Decimal::ZERO
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Exit Conditions
// ═══════════════════════════════════════════════════════════════════════════

/// Reason for closing a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    /// Take profit target reached
    TakeProfit,
    /// Stop loss triggered
    StopLoss,
    /// Health factor below critical threshold
    LiquidationRisk,
    /// Manual close request
    Manual,
    /// Strategy signal reversal
    SignalReversal,
}

impl ExitReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TakeProfit => "take_profit",
            Self::StopLoss => "stop_loss",
            Self::LiquidationRisk => "liquidation_risk",
            Self::Manual => "manual_close",
            Self::SignalReversal => "signal_reversal",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Paper Trading Manager
// ═══════════════════════════════════════════════════════════════════════════

/// Manages paper trading portfolio and position tracking.
pub struct PaperTradingManager {
    config: PaperTradingConfig,
    portfolio: Arc<RwLock<PaperPortfolio>>,
    data_service: Arc<DataService>,
    pnl_tracker: Arc<PnLTracker>,
    /// Currently monitored position (if any)
    active_position: Arc<RwLock<Option<PaperPosition>>>,
}

/// Extended position state with paper trading metadata.
#[derive(Debug, Clone)]
pub struct PaperPosition {
    /// Core position state
    pub state: PositionState,

    /// Database position ID
    pub position_id: i64,

    /// Take profit price level (USD)
    pub take_profit_price: Option<Decimal>,

    /// Stop loss price level (USD)
    pub stop_loss_price: Option<Decimal>,

    /// Liquidation price estimate (USD)
    pub liquidation_price: Decimal,

    /// Entry price (average)
    pub entry_price: Decimal,

    /// Last mark-to-market price
    pub last_mtm_price: Decimal,

    /// Current unrealized P&L (USD)
    pub unrealized_pnl_usd: Decimal,

    /// Capital allocated to this position
    pub allocated_capital_usd: Decimal,
}

impl PaperTradingManager {
    pub fn new(
        config: PaperTradingConfig,
        data_service: Arc<DataService>,
        pnl_tracker: Arc<PnLTracker>,
    ) -> Self {
        let starting_capital = config.starting_capital_usd;

        Self {
            config,
            portfolio: Arc::new(RwLock::new(PaperPortfolio::new(starting_capital))),
            data_service,
            pnl_tracker,
            active_position: Arc::new(RwLock::new(None)),
        }
    }

    /// Get current portfolio snapshot.
    pub async fn get_portfolio(&self) -> PaperPortfolio {
        self.portfolio.read().await.clone()
    }

    /// Get active position if any.
    pub async fn get_active_position(&self) -> Option<PaperPosition> {
        self.active_position.read().await.clone()
    }

    /// Calculate take profit and stop loss prices for a position.
    pub fn calculate_tp_sl(
        &self,
        direction: PositionDirection,
        entry_price: Decimal,
    ) -> (Option<Decimal>, Option<Decimal>) {
        let sl_pct = self.config.default_stop_loss_pct / dec!(100);
        let tp_pct = self.config.default_take_profit_pct / dec!(100);

        match direction {
            PositionDirection::Long => {
                // Long: SL below entry, TP above entry
                let sl = entry_price * (dec!(1) - sl_pct);
                let tp = entry_price * (dec!(1) + tp_pct);
                (Some(tp), Some(sl))
            }
            PositionDirection::Short => {
                // Short: SL above entry, TP below entry
                let sl = entry_price * (dec!(1) + sl_pct);
                let tp = entry_price * (dec!(1) - tp_pct);
                (Some(tp), Some(sl))
            }
        }
    }

    /// Estimate liquidation price based on health factor and leverage.
    pub fn estimate_liquidation_price(
        &self,
        direction: PositionDirection,
        entry_price: Decimal,
        leverage: Decimal,
        liquidation_threshold: Decimal,
    ) -> Decimal {
        // Simplified liquidation price calculation
        // More accurate calculation would use actual Aave parameters

        let price_drop_pct = (dec!(1) - (dec!(1) / leverage)) * liquidation_threshold;

        match direction {
            PositionDirection::Long => {
                // Long liquidates if price drops
                entry_price * (dec!(1) - price_drop_pct)
            }
            PositionDirection::Short => {
                // Short liquidates if price rises
                entry_price * (dec!(1) + price_drop_pct)
            }
        }
    }

    /// Simulate realistic slippage for an order.
    pub fn simulate_slippage(
        &self,
        order_size_usd: Decimal,
        quoted_amount: Decimal,
    ) -> Decimal {
        let base_slippage = Decimal::from(self.config.slippage_bps) / dec!(10_000);

        // Market impact: linear model with order size
        // Larger orders have progressively worse price
        let market_impact = if order_size_usd > dec!(100_000) {
            let size_ratio = order_size_usd / dec!(100_000);
            // Linear impact: 0.1% per 100k USD
            self.config.market_impact_factor * size_ratio
        } else {
            Decimal::ZERO
        };

        let total_slippage = base_slippage + market_impact;

        // Reduce received amount by slippage
        quoted_amount * (dec!(1) - total_slippage)
    }

    /// Register a new position opening in paper trading mode.
    pub async fn open_position(
        &self,
        position_state: PositionState,
        position_id: i64,
        entry_price: Decimal,
        allocated_capital: Decimal,
    ) -> Result<()> {
        // Calculate TP/SL levels
        let (tp, sl) = self.calculate_tp_sl(position_state.direction, entry_price);

        // Estimate liquidation price
        let liquidation_price = self.estimate_liquidation_price(
            position_state.direction,
            entry_price,
            self.config.max_leverage,
            dec!(0.85), // Typical Aave LT for volatile assets
        );

        let paper_pos = PaperPosition {
            state: position_state.clone(),
            position_id,
            take_profit_price: tp,
            stop_loss_price: sl,
            liquidation_price,
            entry_price,
            last_mtm_price: entry_price,
            unrealized_pnl_usd: Decimal::ZERO,
            allocated_capital_usd: allocated_capital,
        };

        // Update portfolio
        {
            let mut portfolio = self.portfolio.write().await;
            portfolio.available_cash_usd -= allocated_capital;
            portfolio.total_positions += 1;
        }

        // Store position
        *self.active_position.write().await = Some(paper_pos.clone());

        // Log position opening
        self.log_position_open(&paper_pos).await;

        Ok(())
    }

    /// Update position with current market price and check exit conditions.
    pub async fn update_position(&self, current_price: Decimal) -> Result<Option<ExitReason>> {
        let mut pos_guard = self.active_position.write().await;

        let pos = match pos_guard.as_mut() {
            Some(p) => p,
            None => return Ok(None),
        };

        // Update mark-to-market
        pos.last_mtm_price = current_price;

        // Calculate unrealized P&L
        let pnl_pct = match pos.state.direction {
            PositionDirection::Long => {
                (current_price - pos.entry_price) / pos.entry_price
            }
            PositionDirection::Short => {
                (pos.entry_price - current_price) / pos.entry_price
            }
        };

        pos.unrealized_pnl_usd = pos.allocated_capital_usd * pnl_pct;

        // Update portfolio equity
        {
            let mut portfolio = self.portfolio.write().await;
            portfolio.current_unrealized_pnl = pos.unrealized_pnl_usd;
            let new_equity = portfolio.available_cash_usd
                + portfolio.current_unrealized_pnl
                + pos.allocated_capital_usd;
            portfolio.update_equity(new_equity);
        }

        // Snapshot position state
        self.pnl_tracker
            .snapshot(pos.position_id, &pos.state)
            .await
            .context("failed to snapshot position")?;

        // Check exit conditions
        let exit_reason = self.check_exit_conditions(pos, current_price).await;

        Ok(exit_reason)
    }

    /// Check if any exit condition is triggered.
    async fn check_exit_conditions(
        &self,
        pos: &PaperPosition,
        current_price: Decimal,
    ) -> Option<ExitReason> {
        // Check stop loss
        if let Some(sl) = pos.stop_loss_price {
            let sl_triggered = match pos.state.direction {
                PositionDirection::Long => current_price <= sl,
                PositionDirection::Short => current_price >= sl,
            };

            if sl_triggered {
                info!(
                    direction = ?pos.state.direction,
                    current_price = %current_price,
                    stop_loss = %sl,
                    "Stop loss triggered"
                );
                return Some(ExitReason::StopLoss);
            }
        }

        // Check take profit
        if let Some(tp) = pos.take_profit_price {
            let tp_triggered = match pos.state.direction {
                PositionDirection::Long => current_price >= tp,
                PositionDirection::Short => current_price <= tp,
            };

            if tp_triggered {
                info!(
                    direction = ?pos.state.direction,
                    current_price = %current_price,
                    take_profit = %tp,
                    "Take profit triggered"
                );
                return Some(ExitReason::TakeProfit);
            }
        }

        // Check liquidation risk
        if pos.state.health_factor < self.config.min_health_factor {
            warn!(
                health_factor = %pos.state.health_factor,
                min_hf = %self.config.min_health_factor,
                "Liquidation risk - health factor too low"
            );
            return Some(ExitReason::LiquidationRisk);
        }

        // Check if price is near liquidation
        let near_liquidation = match pos.state.direction {
            PositionDirection::Long => {
                current_price < pos.liquidation_price * dec!(1.05) // Within 5%
            }
            PositionDirection::Short => {
                current_price > pos.liquidation_price * dec!(0.95) // Within 5%
            }
        };

        if near_liquidation {
            warn!(
                current_price = %current_price,
                liquidation_price = %pos.liquidation_price,
                "Price approaching liquidation level"
            );
            return Some(ExitReason::LiquidationRisk);
        }

        None
    }

    /// Close the active position and realize P&L.
    pub async fn close_position(&self, reason: ExitReason) -> Result<Decimal> {
        let mut pos_guard = self.active_position.write().await;

        let pos = pos_guard.take()
            .context("no active position to close")?;

        let realized_pnl = pos.unrealized_pnl_usd;

        // Update portfolio
        {
            let mut portfolio = self.portfolio.write().await;
            portfolio.available_cash_usd += pos.allocated_capital_usd + realized_pnl;
            portfolio.cumulative_realized_pnl += realized_pnl;
            portfolio.current_unrealized_pnl = Decimal::ZERO;

            let new_equity = portfolio.available_cash_usd;
            portfolio.update_equity(new_equity);
        }

        // Log position close
        self.log_position_close(&pos, reason, realized_pnl).await;

        Ok(realized_pnl)
    }

    /// Log position opening details.
    async fn log_position_open(&self, pos: &PaperPosition) {
        let portfolio = self.portfolio.read().await;

        info!("╔══════════════════════════════════════════════════════════════════╗");
        info!("║              PAPER TRADE: POSITION OPENED                        ║");
        info!("╠══════════════════════════════════════════════════════════════════╣");
        info!("║ Direction: {:?}", pos.state.direction);
        info!("║ Entry Price: {} USD", pos.entry_price);
        info!("║ Position Size: {} USD", pos.allocated_capital_usd);
        info!("║ Leverage: {:.2}x", pos.state.debt_usd / pos.allocated_capital_usd);
        info!("║ ─────────────────────────────────────────────────────────────── ║");
        info!("║ Risk Management:");
        if let Some(tp) = pos.take_profit_price {
            info!("║   Take Profit: {} USD (+{:.2}%)",
                tp,
                ((tp - pos.entry_price) / pos.entry_price * dec!(100)).abs()
            );
        }
        if let Some(sl) = pos.stop_loss_price {
            info!("║   Stop Loss: {} USD (-{:.2}%)",
                sl,
                ((pos.entry_price - sl) / pos.entry_price * dec!(100)).abs()
            );
        }
        info!("║   Liquidation Price: {} USD", pos.liquidation_price);
        info!("║   Health Factor: {:.3}", pos.state.health_factor);
        info!("║ ─────────────────────────────────────────────────────────────── ║");
        info!("║ Portfolio:");
        info!("║   Available Cash: {} USD", portfolio.available_cash_usd);
        info!("║   Total Equity: {} USD", portfolio.total_equity_usd);
        info!("║   Total Positions: {}", portfolio.total_positions);
        info!("╚══════════════════════════════════════════════════════════════════╝");
    }

    /// Log position closing details.
    async fn log_position_close(
        &self,
        pos: &PaperPosition,
        reason: ExitReason,
        realized_pnl: Decimal,
    ) {
        let portfolio = self.portfolio.read().await;
        let pnl_pct = (realized_pnl / pos.allocated_capital_usd) * dec!(100);

        let duration_hours = (now_unix() - pos.state.open_timestamp) as f64 / 3600.0;

        info!("╔══════════════════════════════════════════════════════════════════╗");
        info!("║              PAPER TRADE: POSITION CLOSED                        ║");
        info!("╠══════════════════════════════════════════════════════════════════╣");
        info!("║ Direction: {:?}", pos.state.direction);
        info!("║ Entry Price: {} USD", pos.entry_price);
        info!("║ Exit Price: {} USD", pos.last_mtm_price);
        info!("║ ─────────────────────────────────────────────────────────────── ║");
        info!("║ Performance:");
        info!("║   Realized P&L: {} USD ({:+.2}%)",
            realized_pnl, pnl_pct
        );
        info!("║   Hold Duration: {:.1} hours", duration_hours);
        info!("║   Exit Reason: {}", reason.as_str());
        info!("║ ─────────────────────────────────────────────────────────────── ║");
        info!("║ Portfolio:");
        info!("║   Available Cash: {} USD", portfolio.available_cash_usd);
        info!("║   Total Equity: {} USD", portfolio.total_equity_usd);
        info!("║   Cumulative P&L: {} USD", portfolio.cumulative_realized_pnl);
        info!("║   ROE: {:+.2}%", portfolio.roe_pct());
        info!("║   Current Drawdown: {:.2}%", portfolio.current_drawdown_pct);
        info!("║   Max Drawdown: {:.2}%", portfolio.max_drawdown_pct);
        info!("╚══════════════════════════════════════════════════════════════════╝");
    }

    /// Generate comprehensive performance report.
    pub async fn generate_report(&self) -> Result<String> {
        let portfolio = self.portfolio.read().await;
        let stats = self.pnl_tracker.get_rolling_stats(None).await?;

        let mut report = String::new();
        report.push_str("╔══════════════════════════════════════════════════════════════════╗\n");
        report.push_str("║           PAPER TRADING PERFORMANCE REPORT                       ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════════╣\n");
        report.push_str(&format!("║ Portfolio Summary:\n"));
        report.push_str(&format!("║   Starting Capital: {} USD\n", portfolio.starting_capital_usd));
        report.push_str(&format!("║   Current Equity: {} USD\n", portfolio.total_equity_usd));
        report.push_str(&format!("║   Return on Equity: {:+.2}%\n", portfolio.roe_pct()));
        report.push_str(&format!("║   Cumulative P&L: {} USD\n", portfolio.cumulative_realized_pnl));
        report.push_str("║ ─────────────────────────────────────────────────────────────── ║\n");
        report.push_str(&format!("║ Trading Statistics:\n"));
        report.push_str(&format!("║   Total Trades: {}\n", stats.total_trades));
        report.push_str(&format!("║   Winning Trades: {}\n", stats.winning_trades));
        report.push_str(&format!("║   Losing Trades: {}\n", stats.losing_trades));
        report.push_str(&format!("║   Win Rate: {:.1}%\n", stats.win_rate * dec!(100)));
        report.push_str(&format!("║   Avg P&L per Trade: {} USD\n", stats.avg_pnl_per_trade_usd));
        report.push_str("║ ─────────────────────────────────────────────────────────────── ║\n");
        report.push_str(&format!("║ Risk Metrics:\n"));
        report.push_str(&format!("║   Sharpe Ratio: {:.3}\n", stats.sharpe_ratio));
        report.push_str(&format!("║   Max Drawdown: {:.2}%\n", stats.max_drawdown_pct * dec!(100)));
        report.push_str(&format!("║   Current Drawdown: {:.2}%\n", portfolio.current_drawdown_pct));
        report.push_str(&format!("║   Avg Hold Duration: {:.1} hours\n", stats.avg_hold_duration_hours));
        report.push_str("╚══════════════════════════════════════════════════════════════════╝\n");

        Ok(report)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
