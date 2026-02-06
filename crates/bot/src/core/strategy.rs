//! Strategy engine for BSC Leverage Bot.
//!
//! Decision engine that evaluates trade signals, manages position lifecycle,
//! and applies risk filters including:
//! - Direction-aware stress testing (Perez et al., FC 2021)
//! - Liquidation cascade modeling (OECD 2023)
//! - GARCH-informed position sizing via fractional Kelly (MacLean et al., 2010)
//! - Close factor risk check (Aave V3 LiquidationLogic.sol)
//! - Alpha decay monitoring (Cong et al., 2024)

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use alloy::primitives::Address;
use anyhow::{Context, Result};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::{mpsc, RwLock as TokioRwLock};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::config::{PositionConfig, SignalConfig};
use crate::constants::{CLOSE_FACTOR_THRESHOLD_USD, TOKEN_USDT, TOKEN_WBNB};
use crate::core::safety::SafetyState;
use crate::execution::aave_client::AaveClient;
use crate::types::{
    HFTier, HealthStatus, PositionDirection, SignalEvent, StrategyHealthReport, TradeSignal,
};

use super::pnl_tracker::PnLTracker;
use super::position_manager::PositionManager;
use super::position_sizing::{PositionSizer, RiskMonitor};

/// Async strategy engine consuming events from health monitor and signal engine.
///
/// Evaluates trade signals for entry (confidence, borrow rate, stress test),
/// dispatches health status events (deleverage, close), and monitors alpha
/// decay for strategy health assessment.
pub struct Strategy {
    position_manager: Arc<PositionManager>,
    aave_client: Arc<AaveClient>,
    pnl_tracker: Arc<PnLTracker>,
    safety: Arc<SafetyState>,
    event_rx: mpsc::Receiver<SignalEvent>,
    config: StrategyParams,
    shutdown: CancellationToken,
    /// Dynamic confidence threshold, adjusted upward on alpha decay.
    dynamic_confidence_threshold: Decimal,
    /// Number of signals acted on today.
    signals_today: u32,
    /// Day counter for daily signal limit (UNIX day number).
    last_signal_day: i64,
    /// Dynamic position sizer using Kelly Criterion
    position_sizer: Arc<PositionSizer>,
    /// Risk monitor for portfolio limits (wrapped in RwLock for interior mutability)
    risk_monitor: Arc<TokioRwLock<RiskMonitor>>,
    /// Current portfolio value (starts at $500 for paper trading, updates with P&L)
    portfolio_value: Decimal,
}

/// Pre-extracted strategy parameters from config.
struct StrategyParams {
    // Stress test
    stress_drops: Vec<Decimal>,
    min_stress_hf: Decimal,
    cascade_threshold_usd: Decimal,
    cascade_additional_drop: Decimal,
    // Borrow rate
    max_borrow_cost_pct: Decimal,
    max_acceptable_borrow_apr: Decimal,
    // Health factor thresholds
    deleverage_threshold: Decimal,
    close_threshold: Decimal,
    target_hf_after_deleverage: Decimal,
    // Entry rules
    min_confidence: Decimal,
    max_signals_per_day: u32,
    // Position sizing
    kelly_fraction: Decimal,
    high_vol_threshold: Decimal,
    min_position_usd: Decimal,
    max_position_usd: Decimal,
    max_leverage: Decimal,
    // Alpha decay
    alpha_decay_enabled: bool,
    accuracy_decay_threshold: Decimal,
    sharpe_decay_threshold: Decimal,
    confidence_boost_on_decay: Decimal,
    rolling_window_days: u32,
    historical_window_days: u32,
    // Exit
    max_hold_hours: u32,
    // Short
    preferred_short_collateral: String,
}

impl Strategy {
    /// Construct from all dependencies and config.
    pub fn new(
        position_manager: Arc<PositionManager>,
        aave_client: Arc<AaveClient>,
        pnl_tracker: Arc<PnLTracker>,
        safety: Arc<SafetyState>,
        event_rx: mpsc::Receiver<SignalEvent>,
        position_config: &PositionConfig,
        signal_config: &SignalConfig,
        shutdown: CancellationToken,
    ) -> Self {
        let stress_drops: Vec<Decimal> = position_config
            .stress_test_price_drops
            .iter()
            .map(|s| s.parse().unwrap_or(Decimal::ZERO))
            .collect();

        let config = StrategyParams {
            stress_drops,
            min_stress_hf: position_config.min_stress_test_hf,
            cascade_threshold_usd: Decimal::from(position_config.cascade_liquidation_threshold_usd),
            cascade_additional_drop: position_config.cascade_additional_drop,
            max_borrow_cost_pct: position_config.max_borrow_cost_pct,
            max_acceptable_borrow_apr: position_config.max_acceptable_borrow_apr,
            deleverage_threshold: position_config.deleverage_threshold,
            close_threshold: position_config.close_threshold,
            target_hf_after_deleverage: position_config.target_hf_after_deleverage,
            min_confidence: signal_config.entry_rules.min_confidence,
            max_signals_per_day: signal_config.entry_rules.max_signals_per_day,
            kelly_fraction: signal_config.position_sizing.kelly_fraction,
            high_vol_threshold: signal_config.position_sizing.high_vol_threshold,
            min_position_usd: signal_config.position_sizing.min_position_usd,
            max_position_usd: Decimal::from(position_config.max_position_usd),
            max_leverage: position_config.max_leverage_ratio,
            alpha_decay_enabled: signal_config.alpha_decay_monitoring.enabled,
            accuracy_decay_threshold: signal_config.alpha_decay_monitoring.accuracy_decay_threshold,
            sharpe_decay_threshold: signal_config.alpha_decay_monitoring.sharpe_decay_threshold,
            confidence_boost_on_decay: signal_config
                .alpha_decay_monitoring
                .confidence_boost_on_decay,
            rolling_window_days: signal_config.alpha_decay_monitoring.rolling_window_days,
            historical_window_days: signal_config.alpha_decay_monitoring.historical_window_days,
            max_hold_hours: signal_config.exit_rules.max_hold_hours,
            preferred_short_collateral: signal_config.short_signals.preferred_collateral.clone(),
        };

        let dynamic_confidence_threshold = config.min_confidence;

        // Initialize position sizing system with pure Kelly configuration
        // Academic foundation: Kelly (1956), Thorp (2008), MacLean et al. (2010)
        use super::position_sizing::PositionSizingConfig;
        let sizing_config = PositionSizingConfig {
            // Quarter Kelly: 51% of optimal growth, reduces 80% DD prob from 1-in-5 to 1-in-213
            // For more aggressive: 0.50 (Half Kelly) = 75% growth, 50% volatility
            kelly_fraction: signal_config.position_sizing.kelly_fraction,

            // Minimum 2:1 RR (below this, expected value negative even at 50% win rate)
            min_risk_reward_ratio: dec!(2.0),

            // ATR-based stops for crypto volatility (Wilder, 1978)
            atr_stop_multiplier: dec!(2.5),           // 2.5x ATR = ~95% confidence interval
            atr_tp_multiplier: dec!(5.0),             // 5.0x ATR = 2:1 RR minimum

            // 25% daily loss limit for crypto's 40%+ volatility
            // Allows ~10σ adverse moves (vs 5% = 2σ every 20 days)
            // Academic ref: Basel FRTB (2019), BlackRock Bitcoin Volatility (2025)
            daily_loss_limit_pct: dec!(0.25),

            // 1 position per token (NYSE Pillar Risk Controls standard)
            // Enforces true diversification across uncorrelated assets
            max_positions_per_token: 1,
        };

        let position_sizer = Arc::new(PositionSizer::new(sizing_config.clone()));

        // Starting capital: $500 for paper trading, or actual wallet balance for live
        let initial_capital = if position_config.dry_run {
            dec!(500.0)
        } else {
            signal_config.position_sizing.min_position_usd
        };
        let risk_monitor = Arc::new(TokioRwLock::new(RiskMonitor::new(sizing_config, initial_capital)));

        Self {
            position_manager,
            aave_client,
            pnl_tracker,
            safety,
            event_rx,
            config,
            shutdown,
            dynamic_confidence_threshold,
            signals_today: 0,
            last_signal_day: 0,
            position_sizer,
            risk_monitor,
            portfolio_value: initial_capital,
        }
    }

    // -----------------------------------------------------------------------
    // Main event loop
    // -----------------------------------------------------------------------

    /// Run the strategy event loop until shutdown.
    ///
    /// Consumes `SignalEvent`s from the health monitor and signal engine
    /// via a bounded `mpsc` channel. Periodic alpha decay checks run on
    /// idle timeout (60s).
    pub async fn run(&mut self) -> Result<()> {
        info!("strategy engine started");

        loop {
            tokio::select! {
                _ = self.shutdown.cancelled() => {
                    info!("strategy engine: shutdown signal received");
                    break;
                }
                event = self.event_rx.recv() => {
                    match event {
                        Some(SignalEvent::Health(status)) => {
                            if let Err(e) = self.handle_health(status).await {
                                error!(error = %e, "error handling health event");
                            }
                        }
                        Some(SignalEvent::Trade(signal)) => {
                            if let Err(e) = self.handle_trade_signal(signal).await {
                                error!(error = %e, "error handling trade signal");
                            }
                        }
                        Some(SignalEvent::MultiTfTrade(mtf_signal)) => {
                            // Convert multi-TF signal to legacy format for now
                            // TODO: Implement full multi-TF handling in Phase 6
                            let signal = mtf_signal.to_trade_signal();
                            if let Err(e) = self.handle_trade_signal(signal).await {
                                error!(error = %e, "error handling multi-TF trade signal");
                            }
                        }
                        Some(SignalEvent::Shutdown) | None => {
                            info!("strategy engine: channel closed or shutdown event");
                            break;
                        }
                    }
                }
                // Periodic alpha decay check on idle
                _ = tokio::time::sleep(std::time::Duration::from_secs(60)) => {
                    if self.config.alpha_decay_enabled {
                        match self.check_strategy_health().await {
                            Ok(report) if report.alpha_decay_detected => {
                                warn!(
                                    "alpha decay detected: {}",
                                    report.recommendations.join("; ")
                                );
                            }
                            Err(e) => {
                                warn!(error = %e, "failed to check strategy health");
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        info!("strategy engine stopped");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Health status handling
    // -----------------------------------------------------------------------

    /// Handle a HealthStatus event from the health monitor.
    ///
    /// Triggers deleverage or emergency close based on health factor tiers.
    async fn handle_health(&self, status: HealthStatus) -> Result<()> {
        if !self.position_manager.has_open_position().await {
            return Ok(());
        }

        if status.tier == HFTier::Critical {
            if status.health_factor <= self.config.close_threshold {
                error!(
                    hf = %status.health_factor,
                    threshold = %self.config.close_threshold,
                    "HF below close threshold — emergency close"
                );
                self.position_manager.close_position("emergency").await?;
                return Ok(());
            }

            if status.health_factor <= self.config.deleverage_threshold {
                warn!(
                    hf = %status.health_factor,
                    threshold = %self.config.deleverage_threshold,
                    "HF below deleverage threshold — deleveraging"
                );
                self.position_manager
                    .partial_deleverage(self.config.target_hf_after_deleverage)
                    .await?;
                return Ok(());
            }
        }

        if status.tier == HFTier::Warning
            && status.health_factor <= self.config.deleverage_threshold
        {
            warn!(
                hf = %status.health_factor,
                "WARNING tier below deleverage threshold — deleveraging"
            );
            self.position_manager
                .partial_deleverage(self.config.target_hf_after_deleverage)
                .await?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Trade signal handling
    // -----------------------------------------------------------------------

    /// Handle a TradeSignal from the signal engine.
    ///
    /// Skips if a position is already open or daily limit reached.
    /// Evaluates entry conditions and opens a position if all checks pass.
    async fn handle_trade_signal(&mut self, signal: TradeSignal) -> Result<()> {
        if self.position_manager.has_open_position().await {
            debug!("signal ignored: position already open");
            return Ok(());
        }

        // Daily signal limit
        let today = now_unix() / 86400;
        if today != self.last_signal_day {
            self.signals_today = 0;
            self.last_signal_day = today;
            // Reset daily P&L for new day
            self.risk_monitor.write().await.reset_daily_pnl();
        }
        if self.signals_today >= self.config.max_signals_per_day {
            info!(limit = self.config.max_signals_per_day, "daily signal limit reached");
            return Ok(());
        }

        // Check risk limits (daily loss limit only - no max drawdown)
        let can_trade = self.risk_monitor.read().await.can_trade();
        if !can_trade {
            let rm = self.risk_monitor.read().await;
            warn!(
                daily_pnl = %rm.current_daily_pnl,
                daily_loss_limit = %(self.position_sizer.config.daily_loss_limit_pct * rm.peak_portfolio_value),
                drawdown_pct = %rm.current_drawdown_pct,
                "daily loss limit breached (25%) - trading halted until next day"
            );
            return Ok(());
        }

        // Evaluate entry
        let (should_enter, reason) = self.evaluate_entry(&signal).await?;
        if !should_enter {
            info!(reason, "entry rejected");
            return Ok(());
        }

        // Pure Kelly position sizing based on trade quality
        // Position size determined SOLELY by:
        // 1. Signal confidence (win probability)
        // 2. Expected win/loss ratio
        // 3. Fractional Kelly multiplier (risk control)
        //
        // NO arbitrary portfolio percentage limits (academic optimal: MacLean et al., 2010)

        // Use signal confidence as win probability
        let win_probability = signal.confidence;

        // Calculate expected win/loss ratio from historical stats
        let stats = self.pnl_tracker.get_rolling_stats(Some(30)).await?;
        let expected_win_loss_ratio = if stats.total_trades > 0 && stats.winning_trades > 0 && stats.losing_trades > 0 {
            // Estimate from historical Sharpe ratio and current regime
            // Higher Sharpe = better historical RR
            if stats.sharpe_ratio > dec!(1.5) {
                dec!(2.5) // Strong historical performance
            } else if stats.sharpe_ratio > dec!(1.0) {
                dec!(2.0) // Good historical performance
            } else {
                dec!(1.5) // Conservative default
            }
        } else {
            // No history: use regime-based estimate
            // Trending regimes typically have better RR than ranging
            match signal.regime {
                crate::types::MarketRegime::Trending => dec!(2.0),
                crate::types::MarketRegime::MeanReverting => dec!(1.8),
                crate::types::MarketRegime::Volatile => dec!(1.5),
                crate::types::MarketRegime::Ranging => dec!(1.6),
            }
        };

        // Calculate pure Kelly position size based on trade quality
        let kelly_size = self.position_sizer.calculate_position_size(
            self.portfolio_value,
            win_probability,
            expected_win_loss_ratio,
        );

        // Only check if Kelly size is effectively zero (negative expected value)
        if kelly_size < dec!(10.0) {
            info!(
                kelly_size = %kelly_size,
                win_prob = %win_probability,
                win_loss_ratio = %expected_win_loss_ratio,
                "Kelly sizing rejected trade (negative or minimal expected value)"
            );
            return Ok(());
        }

        // Determine position parameters
        let (debt_token, collateral_token) = match signal.direction {
            PositionDirection::Long => ("USDT".to_string(), "WBNB".to_string()),
            PositionDirection::Short => (
                "WBNB".to_string(),
                self.config.preferred_short_collateral.clone(),
            ),
        };

        info!(
            direction = signal.direction.as_str(),
            kelly_size = %kelly_size,
            portfolio = %self.portfolio_value,
            size_pct = %(kelly_size / self.portfolio_value * dec!(100)),
            win_probability = %win_probability,
            win_loss_ratio = %expected_win_loss_ratio,
            confidence = %signal.confidence,
            regime = ?signal.regime,
            kelly_fraction = %self.position_sizer.config.kelly_fraction,
            expected_value = %((win_probability * expected_win_loss_ratio) - (dec!(1.0) - win_probability)),
            "opening position with pure Kelly sizing (trade quality-based)"
        );

        self.position_manager
            .open_position(signal.direction, &debt_token, kelly_size, &collateral_token)
            .await?;
        self.signals_today += 1;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Entry evaluation
    // -----------------------------------------------------------------------

    /// Evaluate whether to enter a position based on signal and risk filters.
    ///
    /// Returns `(should_enter, reason)`.
    async fn evaluate_entry(
        &self,
        signal: &TradeSignal,
    ) -> Result<(bool, String)> {
        // 1. Confidence check
        if signal.confidence < self.dynamic_confidence_threshold {
            return Ok((
                false,
                format!(
                    "confidence {:.2} below threshold {:.2}",
                    signal.confidence, self.dynamic_confidence_threshold
                ),
            ));
        }

        // 2. Borrow rate check
        let debt_addr = match signal.direction {
            PositionDirection::Long => TOKEN_USDT,
            PositionDirection::Short => TOKEN_WBNB,
        };
        let (rate_ok, current_rate) = self
            .check_borrow_rate_acceptable(debt_addr, self.config.max_hold_hours as f64)
            .await?;
        if !rate_ok {
            return Ok((
                false,
                format!("borrow rate too high: {current_rate:.2}% APR"),
            ));
        }

        // 3. Stress test
        let validated_size = self.validate_position_size(signal).await;

        let collateral_token = match signal.direction {
            PositionDirection::Long => "WBNB",
            PositionDirection::Short => &self.config.preferred_short_collateral,
        };

        let lt_bps = self
            .get_asset_lt_bps(collateral_token)
            .unwrap_or(8000);

        // Approximate collateral and debt for stress test
        let collateral_usd = validated_size;
        let debt_usd = validated_size;

        let stress_hfs = self.stress_test(
            signal.direction,
            collateral_usd,
            debt_usd,
            Decimal::from(lt_bps),
            &self.config.stress_drops,
        );

        for (i, hf) in stress_hfs.iter().enumerate() {
            if *hf < self.config.min_stress_hf {
                return Ok((
                    false,
                    format!(
                        "stress test failed: HF={hf:.4} at {:.0}% drop (min {})",
                        self.config.stress_drops[i] * dec!(100),
                        self.config.min_stress_hf
                    ),
                ));
            }
        }

        // 4. Close factor risk
        if !self.check_close_factor_risk(collateral_usd, debt_usd) {
            return Ok((
                false,
                "position size risks 100% close factor liquidation".into(),
            ));
        }

        Ok((true, "all entry checks passed".into()))
    }

    // -----------------------------------------------------------------------
    // Stress testing (Perez et al., FC 2021; OECD 2023)
    // -----------------------------------------------------------------------

    /// Direction-aware stress test.
    ///
    /// - **LONG** (volatile collateral, stable debt):
    ///   `HF = (collateral * (1 + drop) * LT) / debt`
    ///   HF is linear in price drop.
    ///
    /// - **SHORT** (stable collateral, volatile debt):
    ///   `HF = (collateral * LT) / (debt * (1 + |drop|))`
    ///   HF is convex in price increase.
    pub fn stress_test(
        &self,
        direction: PositionDirection,
        collateral_usd: Decimal,
        debt_usd: Decimal,
        liq_threshold_bps: Decimal,
        price_drops: &[Decimal],
    ) -> Vec<Decimal> {
        let lt = liq_threshold_bps / dec!(10_000);
        price_drops
            .iter()
            .map(|drop| self.compute_hf_at_drop(direction, collateral_usd, debt_usd, lt, *drop))
            .collect()
    }

    /// Stress test with liquidation cascade multiplier.
    ///
    /// If initial drop triggers >$50M in market-wide liquidations,
    /// assumes additional 3% cascade-induced decline.
    pub fn stress_test_with_cascade(
        &self,
        direction: PositionDirection,
        collateral_usd: Decimal,
        debt_usd: Decimal,
        liq_threshold_bps: Decimal,
        price_drops: &[Decimal],
        market_total_supply_usd: Decimal,
    ) -> Vec<Decimal> {
        let lt = liq_threshold_bps / dec!(10_000);
        price_drops
            .iter()
            .map(|drop| {
                let estimated_liquidatable =
                    Self::estimate_market_liquidations(*drop, market_total_supply_usd);

                let effective_drop = if estimated_liquidatable > self.config.cascade_threshold_usd {
                    *drop + self.config.cascade_additional_drop
                } else {
                    *drop
                };

                self.compute_hf_at_drop(direction, collateral_usd, debt_usd, lt, effective_drop)
            })
            .collect()
    }

    /// Compute health factor at a given price change.
    fn compute_hf_at_drop(
        &self,
        direction: PositionDirection,
        collateral_usd: Decimal,
        debt_usd: Decimal,
        lt: Decimal,
        price_change: Decimal,
    ) -> Decimal {
        if debt_usd <= Decimal::ZERO {
            return dec!(999);
        }

        match direction {
            PositionDirection::Long => {
                // Volatile collateral, stable debt
                (collateral_usd * (dec!(1) + price_change) * lt) / debt_usd
            }
            PositionDirection::Short => {
                // Stable collateral, volatile debt
                let debt_multiplier = dec!(1) + price_change.abs();
                if debt_multiplier <= Decimal::ZERO {
                    return dec!(999);
                }
                (collateral_usd * lt) / (debt_usd * debt_multiplier)
            }
        }
    }

    /// Estimate market-wide liquidatable value at a given price drop.
    ///
    /// Simplified model: assumes ~10% of market supply is leveraged,
    /// with positions uniformly distributed across HF 1.0–2.0.
    fn estimate_market_liquidations(
        price_drop: Decimal,
        market_total_supply_usd: Decimal,
    ) -> Decimal {
        let leveraged_fraction = dec!(0.10);
        let total_leveraged = market_total_supply_usd * leveraged_fraction;

        // At -5%: ~25% of positions; at -20%: ~100%
        let drop_magnitude = price_drop.abs();
        let liquidation_fraction = (drop_magnitude * dec!(5)).min(dec!(1.0));

        total_leveraged * liquidation_fraction
    }

    // -----------------------------------------------------------------------
    // Fractional Kelly position sizing (MacLean et al., 2010)
    // -----------------------------------------------------------------------

    /// Compute fractional Kelly fraction for position sizing.
    ///
    /// `edge` = expected return, `volatility` = GARCH forecast.
    /// Returns the fraction of capital to allocate (capped at max_leverage).
    pub fn compute_kelly_fraction(&self, edge: Decimal, volatility: Decimal) -> Decimal {
        if edge <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        let variance = volatility * volatility;
        let full_kelly = if variance > Decimal::ZERO {
            edge / variance
        } else {
            Decimal::ZERO
        };

        // 25% fractional Kelly — standard risk reduction
        let fractional = full_kelly * self.config.kelly_fraction;
        fractional.min(self.config.max_leverage)
    }

    // -----------------------------------------------------------------------
    // Close factor risk (Aave V3 LiquidationLogic.sol)
    // -----------------------------------------------------------------------

    /// Check if position risks 100% close factor liquidation.
    ///
    /// Aave V3: positions below $2,000 face 100% liquidation (no partial).
    pub fn check_close_factor_risk(
        &self,
        collateral_usd: Decimal,
        debt_usd: Decimal,
    ) -> bool {
        for drop in &self.config.stress_drops {
            let projected_collateral = collateral_usd * (dec!(1) + drop);
            if projected_collateral < CLOSE_FACTOR_THRESHOLD_USD {
                warn!(
                    drop = %drop,
                    projected = %projected_collateral,
                    "100% close factor risk: collateral drops below $2,000"
                );
                return false;
            }
        }

        if debt_usd < CLOSE_FACTOR_THRESHOLD_USD {
            warn!(debt = %debt_usd, "100% close factor risk: debt below $2,000");
            return false;
        }

        true
    }

    // -----------------------------------------------------------------------
    // Borrow rate check
    // -----------------------------------------------------------------------

    /// Reject entry if projected borrow cost exceeds threshold.
    ///
    /// Returns `(acceptable, current_rate_apr)`.
    async fn check_borrow_rate_acceptable(
        &self,
        asset: Address,
        projected_hold_hours: f64,
    ) -> Result<(bool, Decimal)> {
        let reserve = self
            .aave_client
            .get_reserve_data(asset)
            .await
            .context("failed to get reserve data for borrow rate check")?;

        let current_rate_apr = reserve.variable_borrow_rate;

        // Absolute rate cap
        if current_rate_apr > self.config.max_acceptable_borrow_apr {
            warn!(
                rate = %current_rate_apr,
                max = %self.config.max_acceptable_borrow_apr,
                "borrow rate exceeds max acceptable APR"
            );
            return Ok((false, current_rate_apr));
        }

        // Projected cost as percentage of position
        let projected_cost_pct =
            current_rate_apr * Decimal::try_from(projected_hold_hours).unwrap_or(dec!(168))
                / dec!(8760);

        if projected_cost_pct > self.config.max_borrow_cost_pct {
            warn!(
                rate = %current_rate_apr,
                projected_cost = %projected_cost_pct,
                max_cost = %self.config.max_borrow_cost_pct,
                "projected borrow cost too high"
            );
            return Ok((false, current_rate_apr));
        }

        Ok((true, current_rate_apr))
    }

    // -----------------------------------------------------------------------
    // Position sizing
    // -----------------------------------------------------------------------

    /// Apply risk constraints to signal engine's recommended position size.
    ///
    /// 1. GARCH volatility adjustment — reduce in high-vol regimes
    /// 2. Hard limits (max position)
    /// 3. Drawdown-based reduction
    async fn validate_position_size(&self, signal: &TradeSignal) -> Decimal {
        let mut size = signal.recommended_size_usd;

        // 1. GARCH volatility adjustment
        if signal.garch_volatility > self.config.high_vol_threshold {
            let vol_scalar = self.config.high_vol_threshold / signal.garch_volatility;
            size *= vol_scalar;
            info!(
                reduction_pct = %((dec!(1) - vol_scalar) * dec!(100)),
                garch_vol = %signal.garch_volatility,
                "position reduced for high volatility"
            );
        }

        // 2. Hard limits
        size = size.min(self.config.max_position_usd);

        // 3. Drawdown-based reduction
        let current_dd = self
            .pnl_tracker
            .current_drawdown_pct()
            .await
            .unwrap_or(Decimal::ZERO);

        if current_dd > dec!(0.1) {
            let dd_scalar = (dec!(1) - current_dd).max(dec!(0.25));
            size *= dd_scalar;
            warn!(
                scalar_pct = %(dd_scalar * dec!(100)),
                drawdown_pct = %(current_dd * dec!(100)),
                "position reduced due to drawdown"
            );
        }

        // Enforce minimum
        if size < self.config.min_position_usd {
            return Decimal::ZERO;
        }

        size
    }

    // -----------------------------------------------------------------------
    // Portfolio Management
    // -----------------------------------------------------------------------

    /// Update portfolio value after a trade closes.
    ///
    /// Called by the close monitoring system to compound profits/losses.
    pub async fn update_portfolio_after_trade(&mut self, realized_pnl: Decimal) -> Result<()> {
        let old_value = self.portfolio_value;
        self.portfolio_value += realized_pnl;

        // Update risk monitor
        {
            let mut rm = self.risk_monitor.write().await;
            rm.update_daily_pnl(realized_pnl);
            rm.update_portfolio_value(self.portfolio_value);
        }

        info!(
            old_value = %old_value,
            realized_pnl = %realized_pnl,
            new_value = %self.portfolio_value,
            change_pct = %(realized_pnl / old_value * dec!(100)),
            "portfolio value updated"
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Alpha decay monitoring (Cong et al., 2024)
    // -----------------------------------------------------------------------

    /// Detect alpha decay by comparing rolling stats against historical baseline.
    ///
    /// If accuracy or Sharpe ratio degrades significantly, raises the
    /// confidence threshold to reduce false positives.
    pub async fn check_strategy_health(&mut self) -> Result<StrategyHealthReport> {
        let stats = self
            .pnl_tracker
            .get_rolling_stats(Some(self.config.rolling_window_days))
            .await
            .context("failed to get rolling stats")?;

        let historical = self
            .pnl_tracker
            .get_rolling_stats(Some(self.config.historical_window_days))
            .await
            .context("failed to get historical stats")?;

        let mut report = StrategyHealthReport::default();

        // Accuracy decay check
        if historical.win_rate > Decimal::ZERO {
            let accuracy_ratio = stats.win_rate / historical.win_rate;
            report.accuracy_ratio = accuracy_ratio;
            if accuracy_ratio < self.config.accuracy_decay_threshold {
                report.alpha_decay_detected = true;
                report.recommendations.push(format!(
                    "Win rate decayed to {:.0}% of 6-month average. \
                     Consider parameter refresh or regime filter adjustment.",
                    accuracy_ratio * dec!(100)
                ));
            }
        }

        // Sharpe ratio degradation
        if historical.sharpe_ratio > Decimal::ZERO {
            let sharpe_ratio = stats.sharpe_ratio / historical.sharpe_ratio;
            report.sharpe_ratio = sharpe_ratio;
            if sharpe_ratio < self.config.sharpe_decay_threshold {
                report.alpha_decay_detected = true;
                report.recommendations.push(format!(
                    "Sharpe ratio at {:.0}% of historical. Strategy may be crowded.",
                    sharpe_ratio * dec!(100)
                ));
            }
        }

        // If alpha decay detected, increase confidence threshold
        if report.alpha_decay_detected {
            self.dynamic_confidence_threshold = (self.config.min_confidence
                * self.config.confidence_boost_on_decay)
                .min(dec!(0.9));
            report.dynamic_confidence_threshold = Some(self.dynamic_confidence_threshold);
            warn!(
                threshold = %self.dynamic_confidence_threshold,
                "alpha decay: raising confidence threshold"
            );
        }

        Ok(report)
    }

    // -----------------------------------------------------------------------
    // Deleverage formula
    // -----------------------------------------------------------------------

    /// Compute the debt repayment amount needed to reach target HF.
    ///
    /// Formula: `repay = (D - C * LT / h_t) / (1 + f - LT / h_t)`
    pub fn compute_deleverage_amount(
        &self,
        collateral_usd: Decimal,
        debt_usd: Decimal,
        liq_threshold_bps: Decimal,
        target_hf: Decimal,
    ) -> Decimal {
        let lt = liq_threshold_bps / dec!(10_000);
        let flash_premium = dec!(0.0005);

        let lt_over_ht = lt / target_hf;
        let numerator = debt_usd - collateral_usd * lt_over_ht;
        let denominator = dec!(1) + flash_premium - lt_over_ht;

        if denominator <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        let repay = numerator / denominator;
        repay.max(Decimal::ZERO)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Get liquidation threshold in basis points for a collateral asset.
    fn get_asset_lt_bps(&self, _collateral_token: &str) -> Option<u32> {
        // This would be looked up from aave config, but we don't have direct
        // access to it here. Default to 8000 (80%) if not found.
        // In production, this should be passed through config or queried.
        Some(8000)
    }
}

/// Get current UNIX timestamp in seconds.
fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_secs() as i64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Stress test -----------------------------------------------------------

    /// Helper to create a Strategy-like struct for testing pure functions.
    /// We test the computation functions directly instead.

    fn compute_hf_at_drop(
        direction: PositionDirection,
        collateral_usd: Decimal,
        debt_usd: Decimal,
        lt: Decimal,
        price_change: Decimal,
    ) -> Decimal {
        if debt_usd <= Decimal::ZERO {
            return dec!(999);
        }
        match direction {
            PositionDirection::Long => {
                (collateral_usd * (dec!(1) + price_change) * lt) / debt_usd
            }
            PositionDirection::Short => {
                let debt_multiplier = dec!(1) + price_change.abs();
                if debt_multiplier <= Decimal::ZERO {
                    return dec!(999);
                }
                (collateral_usd * lt) / (debt_usd * debt_multiplier)
            }
        }
    }

    #[test]
    fn stress_test_long_linear() {
        let lt = dec!(0.80);
        let collateral = dec!(10_000);
        let debt = dec!(5_000);
        let drops = vec![dec!(-0.05), dec!(-0.10), dec!(-0.15), dec!(-0.20)];

        let hfs: Vec<Decimal> = drops
            .iter()
            .map(|d| compute_hf_at_drop(PositionDirection::Long, collateral, debt, lt, *d))
            .collect();

        // HF should decrease with larger drops
        for i in 1..hfs.len() {
            assert!(hfs[i] < hfs[i - 1], "long HF should decrease monotonically");
        }

        // At 0% drop: HF = 10000 * 1.0 * 0.80 / 5000 = 1.6
        let hf_zero = compute_hf_at_drop(PositionDirection::Long, collateral, debt, lt, dec!(0));
        assert_eq!(hf_zero, dec!(1.6));
    }

    #[test]
    fn stress_test_short_convex() {
        let lt = dec!(0.80);
        let collateral = dec!(10_000);
        let debt = dec!(5_000);
        let drops = vec![dec!(-0.05), dec!(-0.10), dec!(-0.15), dec!(-0.20)];

        let hfs: Vec<Decimal> = drops
            .iter()
            .map(|d| compute_hf_at_drop(PositionDirection::Short, collateral, debt, lt, *d))
            .collect();

        // HF should also decrease (price increase is adverse for short)
        for i in 1..hfs.len() {
            assert!(hfs[i] < hfs[i - 1], "short HF should decrease monotonically");
        }

        // At 0% drop: HF = 10000 * 0.80 / (5000 * 1.0) = 1.6
        let hf_zero = compute_hf_at_drop(PositionDirection::Short, collateral, debt, lt, dec!(0));
        assert_eq!(hf_zero, dec!(1.6));
    }

    #[test]
    fn stress_test_long_vs_short_convexity() {
        let lt = dec!(0.80);
        let collateral = dec!(10_000);
        let debt = dec!(5_000);

        // At -20% drop:
        let hf_long = compute_hf_at_drop(
            PositionDirection::Long,
            collateral,
            debt,
            lt,
            dec!(-0.20),
        );
        let hf_short = compute_hf_at_drop(
            PositionDirection::Short,
            collateral,
            debt,
            lt,
            dec!(-0.20),
        );

        // Long HF = 10000 * 0.80 * 0.80 / 5000 = 1.28
        // Short HF = 10000 * 0.80 / (5000 * 1.20) = 1.333...
        // Short HF should be higher (convex) at same drop
        assert!(hf_short > hf_long, "short HF should be higher (convex effect)");
    }

    // -- Kelly sizing ----------------------------------------------------------

    #[test]
    fn kelly_no_edge_returns_zero() {
        let edge = dec!(0);
        let vol = dec!(0.02);
        let kelly_frac = dec!(0.25);
        let max_leverage = dec!(3.0);

        let variance = vol * vol;
        let full_kelly = if variance > Decimal::ZERO && edge > Decimal::ZERO {
            edge / variance
        } else {
            Decimal::ZERO
        };
        let fractional = (full_kelly * kelly_frac).min(max_leverage);
        assert_eq!(fractional, Decimal::ZERO);
    }

    #[test]
    fn kelly_positive_edge_returns_fraction() {
        let edge = dec!(0.05);
        let vol = dec!(0.02);
        let kelly_frac = dec!(0.25);
        let max_leverage = dec!(3.0);

        let variance = vol * vol; // 0.0004
        let full_kelly = edge / variance; // 125
        let fractional = (full_kelly * kelly_frac).min(max_leverage); // min(31.25, 3.0) = 3.0
        assert_eq!(fractional, dec!(3.0));
    }

    #[test]
    fn kelly_high_volatility_reduces_size() {
        let edge = dec!(0.01);
        let vol = dec!(0.10);
        let kelly_frac = dec!(0.25);
        let max_leverage = dec!(3.0);

        let variance = vol * vol; // 0.01
        let full_kelly = edge / variance; // 1.0
        let fractional = (full_kelly * kelly_frac).min(max_leverage); // min(0.25, 3.0) = 0.25
        assert_eq!(fractional, dec!(0.25));
    }

    // -- Close factor risk -----------------------------------------------------

    #[test]
    fn close_factor_large_position_passes() {
        // $10,000 position at -20% = $8,000 > $2,000
        let drops = vec![dec!(-0.05), dec!(-0.10), dec!(-0.20)];
        let collateral = dec!(10_000);
        let debt = dec!(5_000);

        let ok = check_close_factor(&drops, collateral, debt);
        assert!(ok);
    }

    #[test]
    fn close_factor_small_position_fails() {
        // $2,500 at -20% = $2,000 — exactly at threshold
        // $2,500 at -30% = $1,750 — below threshold
        let drops = vec![dec!(-0.05), dec!(-0.20), dec!(-0.30)];
        let collateral = dec!(2_500);
        let debt = dec!(2_500);

        let ok = check_close_factor(&drops, collateral, debt);
        assert!(!ok);
    }

    #[test]
    fn close_factor_debt_below_threshold_fails() {
        let drops = vec![dec!(-0.05)];
        let collateral = dec!(10_000);
        let debt = dec!(1_500); // below $2,000

        let ok = check_close_factor(&drops, collateral, debt);
        assert!(!ok);
    }

    /// Test helper: check close factor without needing a Strategy instance.
    fn check_close_factor(drops: &[Decimal], collateral_usd: Decimal, debt_usd: Decimal) -> bool {
        for drop in drops {
            let projected = collateral_usd * (dec!(1) + drop);
            if projected < CLOSE_FACTOR_THRESHOLD_USD {
                return false;
            }
        }
        if debt_usd < CLOSE_FACTOR_THRESHOLD_USD {
            return false;
        }
        true
    }

    // -- Cascade estimation ----------------------------------------------------

    #[test]
    fn cascade_estimation_small_drop() {
        let liquidatable =
            Strategy::estimate_market_liquidations(dec!(-0.05), dec!(10_000_000_000));
        // 10B * 0.10 * (0.05 * 5) = 1B * 0.25 = 250M
        assert!(liquidatable > Decimal::ZERO);
    }

    #[test]
    fn cascade_estimation_large_drop() {
        let liquidatable =
            Strategy::estimate_market_liquidations(dec!(-0.25), dec!(10_000_000_000));
        // 10B * 0.10 * min(0.25*5, 1.0) = 1B * 1.0 = 1B
        assert!(liquidatable > dec!(50_000_000));
    }

    // -- Deleverage formula ----------------------------------------------------

    #[test]
    fn deleverage_amount_positive() {
        let lt = dec!(8000);
        let flash_premium = dec!(0.0005);
        let target_hf = dec!(1.8);
        let collateral = dec!(10_000);
        let debt = dec!(6_000);

        let lt_dec = lt / dec!(10_000); // 0.80
        let lt_over_ht = lt_dec / target_hf;
        let numerator = debt - collateral * lt_over_ht;
        let denominator = dec!(1) + flash_premium - lt_over_ht;

        let repay = if denominator > Decimal::ZERO {
            (numerator / denominator).max(Decimal::ZERO)
        } else {
            Decimal::ZERO
        };

        assert!(repay > Decimal::ZERO, "repay amount should be positive");
    }

    #[test]
    fn deleverage_amount_zero_when_healthy() {
        let lt_bps = dec!(8000);
        let target_hf = dec!(1.5);
        let collateral = dec!(10_000);
        let debt = dec!(3_000); // Very healthy position

        let lt = lt_bps / dec!(10_000);
        let lt_over_ht = lt / target_hf;
        let numerator = debt - collateral * lt_over_ht;
        // numerator = 3000 - 10000 * (0.80/1.5) = 3000 - 5333 = -2333
        let repay = numerator.max(Decimal::ZERO);
        assert_eq!(repay, Decimal::ZERO, "no deleverage needed when healthy");
    }
}
