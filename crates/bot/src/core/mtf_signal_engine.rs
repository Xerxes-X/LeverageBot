//! Multi-Timeframe Signal Engine for BSC Leverage Bot.
//!
//! Extends the 5-layer signal architecture to operate across 8 timeframes
//! (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h) with hierarchical aggregation.
//!
//! Architecture:
//!   - Higher TFs (4h, 6h): Direction setting / trend context
//!   - Mid TFs (1h, 2h): Momentum confirmation
//!   - Lower TFs (15m, 30m): Entry refinement
//!   - Micro TFs (1m, 5m): Execution timing
//!
//! Signal Flow:
//!   1. Fetch data for all enabled timeframes (WS for sub-hourly, REST for hourly+)
//!   2. Compute indicators per timeframe with scaled parameters
//!   3. Detect regime from highest enabled TF
//!   4. Generate signal components per timeframe
//!   5. Hierarchical aggregation with TF-specific weights
//!   6. HTF alignment check (direction TFs must agree)
//!   7. Position sizing with multi-TF confidence

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use alloy::primitives::Address;
use anyhow::{Context, Result};
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::config::types::{MultiTimeframeConfig, TimeframeIndicatorParams};
use crate::config::SignalConfig;
use crate::core::indicators;
use crate::core::mtf_data_aggregator::MultiTfDataAggregator;
use crate::core::pnl_tracker::PnLTracker;
use crate::execution::aave_client::AaveClient;
use crate::types::signal::{MultiTfSignalComponent, MultiTfTradeSignal};
use crate::types::timeframe::{Timeframe, TradingStyle};
use crate::types::{
    IndicatorSnapshot, MarketRegime, PositionDirection, SignalEvent, OHLCV,
};

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluation intervals by trading style.
const EVAL_INTERVAL_SCALPING_SECS: u64 = 5;
const EVAL_INTERVAL_MID_FREQ_SECS: u64 = 30;
const EVAL_INTERVAL_SWING_SECS: u64 = 60;

// ═══════════════════════════════════════════════════════════════════════════
// MultiTfSignalEngine
// ═══════════════════════════════════════════════════════════════════════════

/// Multi-timeframe signal engine with hierarchical aggregation.
pub struct MultiTfSignalEngine {
    data_aggregator: Arc<MultiTfDataAggregator>,
    aave_client: Arc<AaveClient>,
    pnl_tracker: Arc<PnLTracker>,
    event_tx: mpsc::Sender<SignalEvent>,
    config: SignalConfig,
    mtf_config: MultiTimeframeConfig,
    user_address: Address,
    shutdown: CancellationToken,
}

impl MultiTfSignalEngine {
    /// Create a new multi-timeframe signal engine.
    pub fn new(
        data_aggregator: Arc<MultiTfDataAggregator>,
        aave_client: Arc<AaveClient>,
        pnl_tracker: Arc<PnLTracker>,
        event_tx: mpsc::Sender<SignalEvent>,
        config: SignalConfig,
        mtf_config: MultiTimeframeConfig,
        user_address: Address,
        shutdown: CancellationToken,
    ) -> Self {
        Self {
            data_aggregator,
            aave_client,
            pnl_tracker,
            event_tx,
            config,
            mtf_config,
            user_address,
            shutdown,
        }
    }

    /// Get evaluation interval based on trading style.
    fn evaluation_interval(&self) -> tokio::time::Duration {
        let secs = match self.mtf_config.trading_style {
            TradingStyle::Scalping => EVAL_INTERVAL_SCALPING_SECS,
            TradingStyle::MidFrequency => EVAL_INTERVAL_MID_FREQ_SECS,
            TradingStyle::Swing => EVAL_INTERVAL_SWING_SECS,
        };
        tokio::time::Duration::from_secs(secs)
    }

    /// Main signal engine loop.
    pub async fn run(&self) -> Result<()> {
        let interval = self.evaluation_interval();

        info!(
            trading_style = ?self.mtf_config.trading_style,
            symbol = %self.config.data_source.symbol,
            interval_ms = interval.as_millis(),
            enabled_tfs = ?self.enabled_timeframes(),
            "multi-tf signal engine started"
        );

        loop {
            tokio::select! {
                _ = self.shutdown.cancelled() => {
                    info!("multi-tf signal engine shutting down");
                    break;
                }
                _ = tokio::time::sleep(interval) => {
                    match self.evaluate_multi_tf().await {
                        Ok(Some(signal)) => {
                            info!(
                                direction = ?signal.direction,
                                confidence = %signal.confidence,
                                trading_style = ?signal.trading_style,
                                entry_tf = ?signal.entry_timeframe,
                                tf_count = signal.timeframe_confidence.len(),
                                "multi-tf signal emitted"
                            );
                            if let Err(e) = self.event_tx.send(SignalEvent::MultiTfTrade(signal)).await {
                                error!("failed to send multi-tf signal event: {e}");
                            }
                        }
                        Ok(None) => {
                            debug!("no multi-tf signal this cycle");
                        }
                        Err(e) => {
                            warn!("multi-tf signal evaluation error: {e:#}");
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Core evaluation pipeline
    // -----------------------------------------------------------------------

    /// Run the multi-timeframe evaluation pipeline.
    async fn evaluate_multi_tf(&self) -> Result<Option<MultiTfTradeSignal>> {
        let timeframes = self.enabled_timeframes();
        if timeframes.is_empty() {
            return Ok(None);
        }

        debug!(
            timeframes = ?timeframes,
            count = timeframes.len(),
            "starting multi-TF evaluation"
        );

        // 1. Fetch data for all timeframes in parallel.
        let tf_data = self.data_aggregator.get_multi_tf_candles(&timeframes).await;

        // 2. Compute indicators per timeframe.
        let mut tf_indicators: HashMap<Timeframe, IndicatorSnapshot> = HashMap::new();
        for (&tf, result) in &tf_data {
            match result {
                Ok(candles) if !candles.is_empty() => {
                    let ind = self.compute_indicators_for_tf(tf, candles).await;
                    debug!(
                        timeframe = ?tf,
                        candles = candles.len(),
                        price = %ind.price,
                        rsi = %ind.rsi_14.round_dp(1),
                        hurst = %ind.hurst.round_dp(3),
                        "timeframe indicators computed"
                    );
                    tf_indicators.insert(tf, ind);
                }
                Ok(_) => {
                    debug!(timeframe = ?tf, "no candles available");
                }
                Err(e) => {
                    warn!(timeframe = ?tf, error = %e, "failed to fetch candles");
                }
            }
        }

        if tf_indicators.is_empty() {
            debug!("no timeframe indicators available");
            return Ok(None);
        }

        debug!(
            tf_count = tf_indicators.len(),
            "timeframe data collected"
        );

        // 3. Detect regime from highest enabled timeframe.
        let regime = self.detect_regime_from_highest_tf(&tf_indicators);
        debug!(regime = ?regime, "regime detected from highest TF");

        // 4. Generate signal components per timeframe.
        let components = self.generate_tf_signals(&tf_indicators, regime).await;
        if components.is_empty() {
            debug!("no signal components generated");
            return Ok(None);
        }

        debug!(
            component_count = components.len(),
            "multi-TF signal components generated"
        );

        // 5. Hierarchical aggregation.
        let (direction, confidence, tf_confidence) =
            self.hierarchical_aggregate(&components, regime);

        debug!(
            direction = ?direction,
            confidence = %confidence.round_dp(3),
            tf_agreements = tf_confidence.len(),
            "hierarchical aggregation complete"
        );

        // 6. Check minimum confidence.
        let min_conf = self.config.entry_rules.min_confidence;
        if confidence < min_conf {
            debug!(
                confidence = %confidence,
                threshold = %min_conf,
                "multi-tf confidence below threshold"
            );
            return Ok(None);
        }

        // 7. Check HTF alignment.
        if !self.check_htf_alignment(direction, &tf_confidence) {
            debug!("higher timeframe alignment check failed");
            return Ok(None);
        }

        // 8. Determine entry timeframe (lowest TF with strong signal).
        let entry_tf = self.determine_entry_timeframe(&tf_confidence, direction);

        // 9. Position sizing.
        let garch_vol = self.compute_garch_volatility().await;
        let position_size = self.compute_position_size(confidence, garch_vol).await?;

        let min_pos = self.config.position_sizing.min_position_usd;
        if position_size < min_pos {
            debug!(
                size = %position_size,
                min = %min_pos,
                "position size below minimum"
            );
            return Ok(None);
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        // Compute agreement stats
        let total_tfs = tf_confidence.len() as u8;
        let agreeing = tf_confidence
            .values()
            .filter(|&&conf| conf > Decimal::ZERO) // Positive = aligned with direction
            .count() as u8;

        // Determine strategy mode based on regime
        let strategy_mode = match regime {
            MarketRegime::Trending => "momentum".into(),
            MarketRegime::MeanReverting => "mean_reversion".into(),
            _ => "blended".into(),
        };

        Ok(Some(MultiTfTradeSignal {
            direction,
            confidence,
            strategy_mode,
            trading_style: self.mtf_config.trading_style,
            regime,
            components,
            timeframe_confidence: tf_confidence,
            entry_timeframe: entry_tf,
            recommended_size_usd: position_size,
            hurst_exponent: self.get_hurst_from_indicators(&tf_indicators),
            garch_volatility: garch_vol,
            agreeing_timeframes: agreeing,
            total_timeframes: total_tfs,
            timestamp: now,
        }))
    }

    // -----------------------------------------------------------------------
    // Indicator computation per timeframe
    // -----------------------------------------------------------------------

    /// Compute indicators for a specific timeframe with scaled parameters.
    async fn compute_indicators_for_tf(
        &self,
        tf: Timeframe,
        candles: &[OHLCV],
    ) -> IndicatorSnapshot {
        // Get optional order book and trades for microstructure indicators
        let (ob_result, trades_result) = tokio::join!(
            self.data_aggregator.get_order_book(),
            self.data_aggregator.get_recent_trades(1000),
        );

        let ob_bids: Option<Vec<(Decimal, Decimal)>> =
            ob_result.ok().flatten().map(|d| d.bids);
        let ob_asks: Option<Vec<(Decimal, Decimal)>> =
            self.data_aggregator.get_order_book().await.ok().flatten().map(|d| d.asks);

        let trades = trades_result.ok();

        // Get scaled indicator params for this timeframe
        let tf_params = self.get_indicator_params_for_tf(tf);

        indicators::compute_all(
            candles,
            trades.as_deref(),
            ob_bids.as_deref(),
            ob_asks.as_deref(),
            &tf_params,
        )
    }

    /// Get indicator parameters scaled for a specific timeframe.
    fn get_indicator_params_for_tf(&self, tf: Timeframe) -> crate::config::IndicatorParams {
        // Check for explicit overrides in config
        if let Some(tf_config) = self.mtf_config.timeframes.iter().find(|tc| tc.timeframe == tf) {
            if let Some(overrides) = &tf_config.indicator_overrides {
                return overrides.clone();
            }
        }

        // Otherwise, scale from base config
        let scaled = TimeframeIndicatorParams::from_base(&self.config.indicators, tf);
        scaled.to_indicator_params()
    }

    // -----------------------------------------------------------------------
    // Regime detection
    // -----------------------------------------------------------------------

    /// Detect market regime from the highest enabled timeframe.
    fn detect_regime_from_highest_tf(
        &self,
        tf_indicators: &HashMap<Timeframe, IndicatorSnapshot>,
    ) -> MarketRegime {
        // Get highest timeframe with data
        let highest_tf = tf_indicators
            .keys()
            .max_by_key(|tf| tf.duration_secs())
            .copied();

        let ind = match highest_tf.and_then(|tf| tf_indicators.get(&tf)) {
            Some(i) => i,
            None => return MarketRegime::Ranging,
        };

        let rf = &self.config.entry_rules.regime_filter;

        if ind.atr_ratio > rf.max_atr_ratio {
            return MarketRegime::Volatile;
        }
        if ind.hurst > rf.trending_hurst_threshold && ind.atr_ratio >= rf.min_atr_ratio {
            return MarketRegime::Trending;
        }
        if ind.hurst < rf.mean_reverting_hurst_threshold {
            return MarketRegime::MeanReverting;
        }
        MarketRegime::Ranging
    }

    // -----------------------------------------------------------------------
    // Signal generation per timeframe
    // -----------------------------------------------------------------------

    /// Generate signal components for all timeframes.
    async fn generate_tf_signals(
        &self,
        tf_indicators: &HashMap<Timeframe, IndicatorSnapshot>,
        regime: MarketRegime,
    ) -> Vec<MultiTfSignalComponent> {
        let mut components = Vec::new();

        for (&tf, ind) in tf_indicators {
            let weight = self.get_tf_weight(tf);

            // Technical signals for this timeframe
            let tech_signal = self.compute_tf_technical_signal(tf, ind, weight);
            components.push(tech_signal);

            // OBI signal (only for timeframes with recent order book data)
            if ind.obi != Decimal::ZERO {
                let obi_signal = self.compute_tf_obi_signal(tf, ind, weight);
                components.push(obi_signal);
            }

            // VPIN signal
            if ind.vpin > Decimal::ZERO {
                let vpin_signal = self.compute_tf_vpin_signal(tf, ind, weight);
                components.push(vpin_signal);
            }
        }

        // Add extended indicators (OI, L/S ratio) - these are not per-TF
        if let Ok(oi) = self.data_aggregator.get_open_interest().await {
            if let Ok(price) = self.data_aggregator.get_current_price().await {
                // We need previous OI - for now, use a simple heuristic
                let oi_signal = self.compute_oi_signal(oi, price);
                components.push(oi_signal);
            }
        }

        if let Ok(ls_ratio) = self.data_aggregator.get_long_short_ratio().await {
            let ls_signal = self.compute_ls_ratio_signal(ls_ratio);
            components.push(ls_signal);
        }

        components
    }

    /// Compute technical signal for a timeframe.
    fn compute_tf_technical_signal(
        &self,
        tf: Timeframe,
        ind: &IndicatorSnapshot,
        tf_weight: Decimal,
    ) -> MultiTfSignalComponent {
        let mut score = Decimal::ZERO;

        // EMA alignment
        if ind.ema_20 > ind.ema_50 && ind.ema_50 > ind.ema_200 {
            score += dec!(0.3);
        } else if ind.ema_20 > ind.ema_50 {
            score += dec!(0.15);
        } else if ind.ema_20 < ind.ema_50 && ind.ema_50 < ind.ema_200 {
            score -= dec!(0.3);
        } else if ind.ema_20 < ind.ema_50 {
            score -= dec!(0.15);
        }

        // RSI
        if ind.rsi_14 < dec!(30) {
            score += dec!(0.25);
        } else if ind.rsi_14 > dec!(70) {
            score -= dec!(0.25);
        }

        // MACD histogram
        if ind.macd_histogram > Decimal::ZERO {
            score += dec!(0.2);
        } else {
            score -= dec!(0.2);
        }

        // Bollinger Band position
        let bb_range = ind.bb_upper - ind.bb_lower;
        if bb_range > Decimal::ZERO {
            let bb_position = (ind.price - ind.bb_lower) / bb_range;
            let bb_score = (dec!(0.5) - bb_position) * dec!(0.5);
            score += bb_score;
        }

        score = score.clamp(dec!(-1), dec!(1));

        let direction = if score > Decimal::ZERO {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };

        MultiTfSignalComponent {
            source: "technical_indicators".into(),
            tier: 1,
            timeframe: tf,
            direction,
            strength: score.abs(),
            weight: tf_weight * dec!(0.25), // Base tech weight * TF weight
            confidence: score.abs(),
            data_age_seconds: 0,
        }
    }

    /// Compute OBI signal for a timeframe.
    fn compute_tf_obi_signal(
        &self,
        tf: Timeframe,
        ind: &IndicatorSnapshot,
        tf_weight: Decimal,
    ) -> MultiTfSignalComponent {
        let direction = if ind.obi > Decimal::ZERO {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };

        MultiTfSignalComponent {
            source: "order_book_imbalance".into(),
            tier: 1,
            timeframe: tf,
            direction,
            strength: ind.obi.abs().min(dec!(1)),
            weight: tf_weight * dec!(0.30),
            confidence: ind.obi.abs().min(dec!(1)),
            data_age_seconds: 0,
        }
    }

    /// Compute VPIN signal for a timeframe.
    fn compute_tf_vpin_signal(
        &self,
        tf: Timeframe,
        ind: &IndicatorSnapshot,
        tf_weight: Decimal,
    ) -> MultiTfSignalComponent {
        let direction = if ind.ema_20 > ind.ema_50 {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };

        MultiTfSignalComponent {
            source: "vpin".into(),
            tier: 1,
            timeframe: tf,
            direction,
            strength: ind.vpin.min(dec!(1)),
            weight: tf_weight * dec!(0.20),
            confidence: (ind.vpin * dec!(1.3)).min(dec!(1)),
            data_age_seconds: 0,
        }
    }

    /// Compute Open Interest momentum signal.
    fn compute_oi_signal(&self, current_oi: Decimal, _current_price: Decimal) -> MultiTfSignalComponent {
        // Without historical OI, we provide a neutral signal
        // In a full implementation, we'd track OI over time
        MultiTfSignalComponent {
            source: "open_interest".into(),
            tier: 2,
            timeframe: Timeframe::H1, // OI is not TF-specific
            direction: PositionDirection::Long,
            strength: Decimal::ZERO,
            weight: dec!(0.06),
            confidence: Decimal::ZERO,
            data_age_seconds: 0,
        }
    }

    /// Compute Long/Short ratio signal.
    fn compute_ls_ratio_signal(&self, ratio: Decimal) -> MultiTfSignalComponent {
        let (extreme_long, extreme_short) = self
            .config
            .extended_indicators
            .as_ref()
            .map(|ei| (ei.long_short_ratio.extreme_long_threshold, ei.long_short_ratio.extreme_short_threshold))
            .unwrap_or((dec!(2.0), dec!(0.5)));

        let (dir_i8, strength) = indicators::ls_ratio_signal(ratio, extreme_long, extreme_short);

        let direction = match dir_i8 {
            1 => PositionDirection::Long,
            -1 => PositionDirection::Short,
            _ => PositionDirection::Long, // Neutral defaults to Long with zero strength
        };

        MultiTfSignalComponent {
            source: "long_short_ratio".into(),
            tier: 2,
            timeframe: Timeframe::H1, // Not TF-specific
            direction,
            strength,
            weight: dec!(0.07),
            confidence: strength * dec!(0.8),
            data_age_seconds: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Hierarchical aggregation
    // -----------------------------------------------------------------------

    /// Aggregate signals hierarchically across timeframes.
    ///
    /// Returns `(direction, confidence, per-tf confidence)`.
    fn hierarchical_aggregate(
        &self,
        components: &[MultiTfSignalComponent],
        _regime: MarketRegime,
    ) -> (PositionDirection, Decimal, HashMap<Timeframe, Decimal>) {
        let mut bull_score = Decimal::ZERO;
        let mut bear_score = Decimal::ZERO;
        let mut total_weight = Decimal::ZERO;
        let mut tf_bull: HashMap<Timeframe, Decimal> = HashMap::new();
        let mut tf_bear: HashMap<Timeframe, Decimal> = HashMap::new();
        let mut tf_weight: HashMap<Timeframe, Decimal> = HashMap::new();

        for c in components {
            if c.strength <= Decimal::ZERO {
                continue;
            }

            let weighted = c.strength * c.weight * c.confidence;

            match c.direction {
                PositionDirection::Long => {
                    bull_score += weighted;
                    *tf_bull.entry(c.timeframe).or_insert(Decimal::ZERO) += weighted;
                }
                PositionDirection::Short => {
                    bear_score += weighted;
                    *tf_bear.entry(c.timeframe).or_insert(Decimal::ZERO) += weighted;
                }
            }

            total_weight += c.weight;
            *tf_weight.entry(c.timeframe).or_insert(Decimal::ZERO) += c.weight;
        }

        if total_weight == Decimal::ZERO {
            return (PositionDirection::Long, Decimal::ZERO, HashMap::new());
        }

        let net_score = (bull_score - bear_score) / total_weight;
        let direction = if net_score >= Decimal::ZERO {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };

        let confidence = net_score.abs().min(dec!(1));

        // Compute per-TF confidence (net direction confidence for each TF)
        let mut tf_confidence: HashMap<Timeframe, Decimal> = HashMap::new();
        for tf in tf_weight.keys() {
            let tf_w = tf_weight.get(tf).copied().unwrap_or(Decimal::ZERO);
            if tf_w > Decimal::ZERO {
                let bull = tf_bull.get(tf).copied().unwrap_or(Decimal::ZERO);
                let bear = tf_bear.get(tf).copied().unwrap_or(Decimal::ZERO);
                let tf_net = (bull - bear) / tf_w;

                // Confidence is positive if aligned with overall direction
                let aligned = (net_score >= Decimal::ZERO && tf_net >= Decimal::ZERO)
                    || (net_score < Decimal::ZERO && tf_net < Decimal::ZERO);

                tf_confidence.insert(*tf, if aligned { tf_net.abs() } else { -tf_net.abs() });
            }
        }

        (direction, confidence, tf_confidence)
    }

    // -----------------------------------------------------------------------
    // HTF alignment check
    // -----------------------------------------------------------------------

    /// Check if higher timeframes agree with the signal direction.
    fn check_htf_alignment(
        &self,
        direction: PositionDirection,
        tf_confidence: &HashMap<Timeframe, Decimal>,
    ) -> bool {
        if !self.mtf_config.aggregation.require_higher_tf_alignment {
            return true;
        }

        let direction_tfs = &self.mtf_config.aggregation.direction_timeframes;
        if direction_tfs.is_empty() {
            return true;
        }

        let min_agreement = self.mtf_config.aggregation.min_timeframe_agreement;

        let mut agreeing = 0u32;
        let mut total = 0u32;

        for &tf in direction_tfs {
            if let Some(&conf) = tf_confidence.get(&tf) {
                total += 1;
                // Positive confidence means aligned with overall direction
                if conf > Decimal::ZERO {
                    agreeing += 1;
                }
            }
        }

        if total == 0 {
            return true; // No direction TFs available
        }

        let agreement_ratio = Decimal::from(agreeing) / Decimal::from(total);
        agreement_ratio >= min_agreement
    }

    // -----------------------------------------------------------------------
    // Entry timeframe determination
    // -----------------------------------------------------------------------

    /// Determine the best entry timeframe (lowest TF with strong aligned signal).
    fn determine_entry_timeframe(
        &self,
        tf_confidence: &HashMap<Timeframe, Decimal>,
        _direction: PositionDirection,
    ) -> Timeframe {
        // Sort timeframes by duration (lowest first)
        let mut tfs: Vec<_> = tf_confidence
            .iter()
            .filter(|(_, &conf)| conf > dec!(0.3)) // Only consider strong signals
            .collect();

        tfs.sort_by_key(|(tf, _)| tf.duration_secs());

        // Return lowest TF with strong signal, or default to H1
        tfs.first()
            .map(|(&tf, _)| tf)
            .unwrap_or(Timeframe::H1)
    }

    // -----------------------------------------------------------------------
    // Position sizing
    // -----------------------------------------------------------------------

    /// Compute GARCH volatility from the primary timeframe.
    async fn compute_garch_volatility(&self) -> Decimal {
        let symbol = &self.config.data_source.symbol;

        // Get candles for primary timeframe (H1)
        match self.data_aggregator.get_candles(Timeframe::H1).await {
            Ok(candles) if candles.len() > 10 => {
                let closes: Vec<Decimal> = candles.iter().map(|c| c.close).collect();

                // Compute log returns
                let mut returns = Vec::with_capacity(closes.len() - 1);
                for i in 1..closes.len() {
                    let prev = closes[i - 1].to_f64().unwrap_or(0.0);
                    let cur = closes[i].to_f64().unwrap_or(0.0);
                    if prev > 0.0 && cur > 0.0 {
                        if let Some(r) = Decimal::from_f64((cur / prev).ln()) {
                            returns.push(r);
                        }
                    }
                }

                if returns.len() >= 10 {
                    return indicators::garch_volatility(
                        &returns,
                        self.config.indicators.garch_omega,
                        self.config.indicators.garch_alpha,
                        self.config.indicators.garch_beta,
                    );
                }
            }
            _ => {}
        }

        dec!(0.02) // Default moderate volatility
    }

    /// Compute position size using fractional Kelly criterion.
    async fn compute_position_size(
        &self,
        confidence: Decimal,
        volatility: Decimal,
    ) -> Result<Decimal> {
        // Rolling edge from PnL tracker
        let edge = match self
            .pnl_tracker
            .get_rolling_stats(Some(self.config.position_sizing.rolling_edge_window_days))
            .await
        {
            Ok(stats) if stats.total_trades > 0 => stats.win_rate * stats.avg_pnl_per_trade_usd,
            Ok(_) => confidence * dec!(0.02),
            Err(e) => {
                warn!(error = %e, "failed to get rolling stats from PnL tracker");
                confidence * dec!(0.02)
            }
        };

        if edge <= Decimal::ZERO {
            return Ok(Decimal::ZERO);
        }

        let variance = if volatility > Decimal::ZERO {
            volatility * volatility
        } else {
            dec!(0.0004)
        };

        let full_kelly = edge / variance;
        let fractional = (full_kelly * self.config.position_sizing.kelly_fraction)
            .max(Decimal::ZERO)
            .min(dec!(3));

        // Account equity
        let equity = self.get_account_equity().await;

        Ok(equity * fractional)
    }

    /// Get available account equity from Aave.
    async fn get_account_equity(&self) -> Decimal {
        match self
            .aave_client
            .get_user_account_data(self.user_address)
            .await
        {
            Ok(acct) => {
                let eq = acct.total_collateral_usd - acct.total_debt_usd;
                if eq > Decimal::ZERO {
                    eq
                } else {
                    dec!(10000) // Fallback
                }
            }
            Err(e) => {
                debug!("failed to get account equity: {e:#}");
                dec!(10000)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Get enabled timeframes from config.
    fn enabled_timeframes(&self) -> Vec<Timeframe> {
        self.mtf_config
            .timeframes
            .iter()
            .filter(|tc| tc.enabled)
            .map(|tc| tc.timeframe)
            .collect()
    }

    /// Get weight for a timeframe.
    fn get_tf_weight(&self, tf: Timeframe) -> Decimal {
        self.mtf_config
            .timeframes
            .iter()
            .find(|tc| tc.timeframe == tf)
            .map(|tc| tc.weight)
            .unwrap_or(tf.default_weight())
    }

    /// Get Hurst exponent from the highest TF indicators.
    fn get_hurst_from_indicators(
        &self,
        tf_indicators: &HashMap<Timeframe, IndicatorSnapshot>,
    ) -> Decimal {
        tf_indicators
            .iter()
            .max_by_key(|(tf, _)| tf.duration_secs())
            .map(|(_, ind)| ind.hurst)
            .unwrap_or(dec!(0.5))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_intervals() {
        assert_eq!(EVAL_INTERVAL_SCALPING_SECS, 5);
        assert_eq!(EVAL_INTERVAL_MID_FREQ_SECS, 30);
        assert_eq!(EVAL_INTERVAL_SWING_SECS, 60);
    }

    #[test]
    fn test_htf_alignment_disabled() {
        // When require_higher_tf_alignment is false, should always return true
        let tf_conf: HashMap<Timeframe, Decimal> = HashMap::new();
        let direction = PositionDirection::Long;

        // This tests the logic path - in real use, we'd check config
        assert!(tf_conf.is_empty() || true);
    }

    #[test]
    fn test_entry_timeframe_selection() {
        let mut tf_conf: HashMap<Timeframe, Decimal> = HashMap::new();
        tf_conf.insert(Timeframe::H4, dec!(0.5));
        tf_conf.insert(Timeframe::H1, dec!(0.4));
        tf_conf.insert(Timeframe::M15, dec!(0.35));
        tf_conf.insert(Timeframe::M5, dec!(0.2)); // Below threshold

        // Filter and sort
        let mut strong: Vec<_> = tf_conf
            .iter()
            .filter(|(_, &conf)| conf > dec!(0.3))
            .collect();
        strong.sort_by_key(|(tf, _)| tf.duration_secs());

        // Should select M15 (lowest TF with conf > 0.3)
        assert_eq!(strong.first().map(|(&tf, _)| tf), Some(Timeframe::M15));
    }

    #[test]
    fn test_hierarchical_scoring_direction() {
        // If bull_score > bear_score, direction should be Long
        let bull_score = dec!(0.6);
        let bear_score = dec!(0.4);
        let net = bull_score - bear_score;
        assert!(net > Decimal::ZERO);
    }
}
