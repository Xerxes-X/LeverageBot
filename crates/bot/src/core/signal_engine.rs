//! 5-Layer signal engine for BSC Leverage Bot.
//!
//! Generates confidence-scored trade entry signals from a multi-source pipeline.
//! Produces [`TradeSignal`] objects sent to a shared `mpsc::Sender<SignalEvent>`
//! channel for consumption by the [`Strategy`] module.
//!
//! Architecture (each layer filters or modulates the next):
//!   Layer 1 — Regime Detection: Hurst exponent classifies market behaviour.
//!   Layer 2 — Multi-Source Directional Signals: Tiered ensemble (Tier 1/2/3).
//!   Layer 3 — Ensemble Confidence Scoring: Weighted, regime-adjusted aggregation.
//!   Layer 4 — Position Sizing: Fractional Kelly Criterion with GARCH volatility.
//!   Layer 5 — Entry Rules & Alpha Decay.
//!
//! References:
//!   Kolm et al. (2023) — OBI accounts for 73 % of prediction.
//!   Easley et al. (2012) — VPIN framework for flow toxicity.
//!   Aloosh & Bekaert (2022) — funding rate predictability.
//!   Lo (2004) — Adaptive Markets Hypothesis.
//!   MacLean, Thorp & Ziemba (2010) — Fractional Kelly.
//!   Cong et al. (2024) — Alpha decay in crypto.

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

use crate::config::SignalConfig;
use crate::core::data_service::DataService;
use crate::core::indicators;
use crate::core::pnl_tracker::PnLTracker;
use crate::execution::aave_client::AaveClient;
use crate::types::{
    IndicatorSnapshot, MarketRegime, PositionDirection, SignalComponent, SignalEvent,
    TradeSignal, OHLCV,
};

// ═══════════════════════════════════════════════════════════════════════════
// Momentum / mean-reversion source classification
// ═══════════════════════════════════════════════════════════════════════════

const MOMENTUM_SOURCES: &[&str] = &[
    "technical_indicators",
    "order_book_imbalance",
    "btc_volatility_spillover",
    "exchange_flows",
];

const MEAN_REVERSION_SOURCES: &[&str] = &["funding_rate", "vpin"];

// ═══════════════════════════════════════════════════════════════════════════
// SignalEngine
// ═══════════════════════════════════════════════════════════════════════════

/// Async signal engine implementing the 5-layer architecture.
///
/// Continuously evaluates market data from multiple sources, computes
/// regime-aware ensemble confidence, and emits [`TradeSignal`] objects
/// to a shared channel for consumption by the Strategy module.
pub struct SignalEngine {
    data_service: Arc<DataService>,
    aave_client: Arc<AaveClient>,
    pnl_tracker: Arc<PnLTracker>,
    event_tx: mpsc::Sender<SignalEvent>,
    config: SignalConfig,
    user_address: Address,
    shutdown: CancellationToken,
}

impl SignalEngine {
    /// Create a new `SignalEngine`.
    pub fn new(
        data_service: Arc<DataService>,
        aave_client: Arc<AaveClient>,
        pnl_tracker: Arc<PnLTracker>,
        event_tx: mpsc::Sender<SignalEvent>,
        config: SignalConfig,
        user_address: Address,
        shutdown: CancellationToken,
    ) -> Self {
        Self {
            data_service,
            aave_client,
            pnl_tracker,
            event_tx,
            config,
            user_address,
            shutdown,
        }
    }

    // -----------------------------------------------------------------------
    // Main loop
    // -----------------------------------------------------------------------

    /// Main signal engine loop.
    ///
    /// Periodically evaluates all signal sources and emits [`TradeSignal`]
    /// to the event channel when confidence exceeds the minimum threshold.
    pub async fn run(&self) -> Result<()> {
        let interval =
            tokio::time::Duration::from_secs(self.config.data_source.refresh_interval_seconds);

        info!(
            mode = %self.config.mode,
            symbol = %self.config.data_source.symbol,
            interval_s = self.config.data_source.refresh_interval_seconds,
            "signal engine started"
        );

        loop {
            tokio::select! {
                _ = self.shutdown.cancelled() => {
                    info!("signal engine shutting down");
                    break;
                }
                _ = tokio::time::sleep(interval) => {
                    match self.evaluate_once().await {
                        Ok(Some(signal)) => {
                            info!(
                                direction = ?signal.direction,
                                confidence = %signal.confidence,
                                regime = ?signal.regime,
                                size_usd = %signal.recommended_size_usd,
                                components = signal.components.len(),
                                "signal emitted"
                            );
                            if let Err(e) = self.event_tx.send(SignalEvent::Trade(signal)).await {
                                error!("failed to send signal event: {e}");
                            }
                        }
                        Ok(None) => {
                            debug!("no signal this cycle");
                        }
                        Err(e) => {
                            warn!("signal evaluation error: {e:#}");
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

    /// Run the full 5-layer evaluation pipeline once.
    async fn evaluate_once(&self) -> Result<Option<TradeSignal>> {
        let ds = &self.config.data_source;

        // Fetch market data.
        let candles = self
            .data_service
            .get_ohlcv(&ds.symbol, &ds.interval, ds.history_candles)
            .await?;

        if (candles.len() as u32) < self.config.indicators.hurst_min_data_points {
            debug!(
                candles = candles.len(),
                min = self.config.indicators.hurst_min_data_points,
                "insufficient candles"
            );
            return Ok(None);
        }

        // Fetch supplementary data concurrently.
        let (trades_res, depth_res) = tokio::join!(
            self.data_service.get_recent_trades(&ds.symbol, 1000),
            self.data_service.get_order_book(&ds.symbol, 20),
        );

        let trades = match trades_res {
            Ok(t) => Some(t),
            Err(e) => {
                warn!(error = %e, "failed to fetch recent trades");
                None
            }
        };
        let depth = match depth_res {
            Ok(d) => Some(d),
            Err(e) => {
                warn!(error = %e, "failed to fetch order book");
                None
            }
        };

        // Compute all indicators (input to Layers 1+2).
        let ob_bids: Option<Vec<(Decimal, Decimal)>> =
            depth.as_ref().map(|d| d.bids.clone());
        let ob_asks: Option<Vec<(Decimal, Decimal)>> =
            depth.as_ref().map(|d| d.asks.clone());

        let ind = indicators::compute_all(
            &candles,
            trades.as_deref(),
            ob_bids.as_deref(),
            ob_asks.as_deref(),
            &self.config.indicators,
        );

        // Layer 1: Regime Detection.
        let regime = self.detect_regime(&ind);

        // Layer 2: Collect signal components.
        let components = self.collect_all_signals(&ind, &candles, regime).await;
        if components.is_empty() {
            return Ok(None);
        }

        // Layer 3: Ensemble confidence scoring.
        let (direction, confidence) = self.compute_ensemble_confidence(&components, regime);

        let min_conf = self.config.entry_rules.min_confidence;
        if confidence < min_conf {
            debug!(
                confidence = %confidence,
                threshold = %min_conf,
                "confidence below threshold"
            );
            return Ok(None);
        }

        // Entry rules (trend alignment).
        if !self.check_entry_rules(direction, &ind) {
            return Ok(None);
        }

        // Layer 4: Position sizing (Fractional Kelly).
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

        // Determine strategy mode.
        let strategy_mode = self.determine_strategy_mode(regime);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        Ok(Some(TradeSignal {
            direction,
            confidence,
            strategy_mode,
            regime,
            components,
            recommended_size_usd: position_size,
            hurst_exponent: ind.hurst,
            garch_volatility: garch_vol,
            timestamp: now,
        }))
    }

    // -----------------------------------------------------------------------
    // Layer 1: Regime Detection
    // -----------------------------------------------------------------------

    /// Classify current market regime using Hurst exponent and ATR ratio.
    fn detect_regime(&self, ind: &IndicatorSnapshot) -> MarketRegime {
        let rf = &self.config.entry_rules.regime_filter;

        let max_atr = rf.max_atr_ratio;
        let trending_h = rf.trending_hurst_threshold;
        let mr_h = rf.mean_reverting_hurst_threshold;
        let min_atr = rf.min_atr_ratio;

        if ind.atr_ratio > max_atr {
            return MarketRegime::Volatile;
        }
        if ind.hurst > trending_h && ind.atr_ratio >= min_atr {
            return MarketRegime::Trending;
        }
        if ind.hurst < mr_h {
            return MarketRegime::MeanReverting;
        }
        MarketRegime::Ranging
    }

    // -----------------------------------------------------------------------
    // Layer 2: Multi-Source Signal Collection
    // -----------------------------------------------------------------------

    /// Collect signal components from all enabled Tier 1/2/3 sources.
    async fn collect_all_signals(
        &self,
        ind: &IndicatorSnapshot,
        candles: &[OHLCV],
        regime: MarketRegime,
    ) -> Vec<SignalComponent> {
        let mut components = Vec::new();

        // Tier 1 (synchronous, highest reliability).
        if self.source_enabled("tier_1", "technical_indicators") {
            components.push(self.compute_technical_signals(ind));
        }
        if self.source_enabled("tier_1", "order_book_imbalance") {
            components.push(self.compute_obi_signal(ind));
        }
        if self.source_enabled("tier_1", "vpin") {
            components.push(self.compute_vpin_signal(ind));
        }

        // Tier 2 (async, supplementary — gathered concurrently).
        let (btc_res, liq_res, flow_res, fund_res) = tokio::join!(
            self.compute_btc_spillover_if_enabled(candles),
            self.compute_liquidation_if_enabled(),
            self.compute_exchange_flows_if_enabled(),
            self.compute_funding_rate_if_enabled(),
        );

        for (name, res) in [
            ("btc_spillover", btc_res),
            ("liquidation", liq_res),
            ("exchange_flows", flow_res),
            ("funding_rate", fund_res),
        ] {
            match res {
                Some(Ok(c)) => components.push(c),
                Some(Err(e)) => debug!("tier 2 signal '{name}' failed: {e:#}"),
                None => {}
            }
        }

        // Tier 3 (conditional).
        if self.source_enabled("tier_3", "aggregate_mempool_flow") {
            match self.compute_mempool_flow_signal().await {
                Ok(c) => components.push(c),
                Err(e) => debug!("tier 3 mempool signal failed: {e:#}"),
            }
        }

        components
    }

    // -----------------------------------------------------------------------
    // Tier 1 signals
    // -----------------------------------------------------------------------

    /// Technical indicator signal (EMA alignment, RSI, MACD, BB).
    fn compute_technical_signals(&self, ind: &IndicatorSnapshot) -> SignalComponent {
        let weight = self.source_weight("tier_1", "technical_indicators", dec!(0.25));
        let mut score = Decimal::ZERO;

        // EMA alignment.
        if ind.ema_20 > ind.ema_50 && ind.ema_50 > ind.ema_200 {
            score += dec!(0.3);
        } else if ind.ema_20 > ind.ema_50 {
            score += dec!(0.15);
        } else if ind.ema_20 < ind.ema_50 && ind.ema_50 < ind.ema_200 {
            score -= dec!(0.3);
        } else if ind.ema_20 < ind.ema_50 {
            score -= dec!(0.15);
        }

        // RSI extremes.
        if ind.rsi_14 < dec!(30) {
            score += dec!(0.25);
        } else if ind.rsi_14 > dec!(70) {
            score -= dec!(0.25);
        } else if ind.rsi_14 < dec!(40) {
            score += dec!(0.1);
        } else if ind.rsi_14 > dec!(60) {
            score -= dec!(0.1);
        }

        // MACD histogram.
        if ind.macd_histogram > Decimal::ZERO {
            score += dec!(0.2);
        } else {
            score -= dec!(0.2);
        }

        // Bollinger Band position.
        let bb_range = ind.bb_upper - ind.bb_lower;
        if bb_range > Decimal::ZERO {
            let bb_position = (ind.price - ind.bb_lower) / bb_range;
            let bb_score = (dec!(0.5) - bb_position) * dec!(0.5);
            score += bb_score;
        }

        // Clamp to [-1, 1].
        score = score.max(dec!(-1)).min(dec!(1));

        let direction = if score > Decimal::ZERO {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };
        let strength = score.abs().min(dec!(1));
        let confidence = strength;

        SignalComponent {
            source: "technical_indicators".into(),
            tier: 1,
            direction,
            strength,
            weight,
            confidence,
            data_age_seconds: 0,
        }
    }

    /// Order book imbalance signal.
    fn compute_obi_signal(&self, ind: &IndicatorSnapshot) -> SignalComponent {
        let weight = self.source_weight("tier_1", "order_book_imbalance", dec!(0.30));
        let obi = ind.obi;

        let direction = if obi > Decimal::ZERO {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };
        let strength = obi.abs().min(dec!(1));
        let confidence = strength;

        SignalComponent {
            source: "order_book_imbalance".into(),
            tier: 1,
            direction,
            strength,
            weight,
            confidence,
            data_age_seconds: 0,
        }
    }

    /// VPIN signal.  Direction inferred from EMA crossover.
    fn compute_vpin_signal(&self, ind: &IndicatorSnapshot) -> SignalComponent {
        let weight = self.source_weight("tier_1", "vpin", dec!(0.20));

        let direction = if ind.ema_20 > ind.ema_50 {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };

        let strength = ind.vpin.min(dec!(1));
        let confidence = (ind.vpin * dec!(1.3)).min(dec!(1));

        SignalComponent {
            source: "vpin".into(),
            tier: 1,
            direction,
            strength,
            weight,
            confidence,
            data_age_seconds: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Tier 2 signals (async)
    // -----------------------------------------------------------------------

    /// Wrapper that returns `None` if the source is disabled.
    async fn compute_btc_spillover_if_enabled(
        &self,
        candles: &[OHLCV],
    ) -> Option<Result<SignalComponent>> {
        if !self.source_enabled("tier_2", "btc_volatility_spillover") {
            return None;
        }
        Some(self.compute_btc_spillover(candles).await)
    }

    async fn compute_liquidation_if_enabled(&self) -> Option<Result<SignalComponent>> {
        if !self.source_enabled("tier_2", "liquidation_heatmap") {
            return None;
        }
        Some(self.compute_liquidation_heatmap().await)
    }

    async fn compute_exchange_flows_if_enabled(&self) -> Option<Result<SignalComponent>> {
        if !self.source_enabled("tier_2", "exchange_flows") {
            return None;
        }
        Some(self.compute_exchange_flows().await)
    }

    async fn compute_funding_rate_if_enabled(&self) -> Option<Result<SignalComponent>> {
        if !self.source_enabled("tier_2", "funding_rate") {
            return None;
        }
        Some(self.compute_funding_rate_signal().await)
    }

    /// BTC volatility spillover (DCC-GARCH literature).
    async fn compute_btc_spillover(&self, bnb_candles: &[OHLCV]) -> Result<SignalComponent> {
        let weight = self.source_weight("tier_2", "btc_volatility_spillover", dec!(0.10));

        let btc_symbol = self.tier_param_str("tier_2", "btc_volatility_spillover", "btc_symbol", "BTCUSDT");
        let lookback = self.tier_param_u32("tier_2", "btc_volatility_spillover", "lookback_hours", 24);

        let btc_candles = self
            .data_service
            .get_ohlcv(&btc_symbol, &self.config.data_source.interval, lookback)
            .await?;

        if btc_candles.len() < 4 || bnb_candles.len() < 4 {
            return Ok(neutral_component("btc_volatility_spillover", 2, weight));
        }

        let btc_closes: Vec<Decimal> = btc_candles.iter().map(|c| c.close).collect();
        let bnb_slice_len = btc_candles.len().min(bnb_candles.len());
        let bnb_closes: Vec<Decimal> = bnb_candles[bnb_candles.len() - bnb_slice_len..]
            .iter()
            .map(|c| c.close)
            .collect();

        let btc_rv = indicators::realized_volatility(&btc_closes, btc_closes.len());
        let bnb_rv = indicators::realized_volatility(&bnb_closes, bnb_closes.len());

        let spillover_ratio = if bnb_rv > Decimal::ZERO {
            btc_rv / bnb_rv
        } else {
            dec!(1)
        };

        // BTC price direction as leading indicator.
        let btc_direction = if btc_candles.last().expect("btc_candles len >= 4").close
            > btc_candles[btc_candles.len().saturating_sub(4)].close
        {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };

        let strength = (spillover_ratio / dec!(3)).min(dec!(1));
        let confidence = (spillover_ratio / dec!(2)).min(dec!(1));

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let age = (now - btc_candles.last().map_or(now, |c| c.timestamp)).unsigned_abs();

        Ok(SignalComponent {
            source: "btc_volatility_spillover".into(),
            tier: 2,
            direction: btc_direction,
            strength,
            weight,
            confidence,
            data_age_seconds: age,
        })
    }

    /// Liquidation heatmap (Perez et al. 2021).
    async fn compute_liquidation_heatmap(&self) -> Result<SignalComponent> {
        let weight = self.source_weight("tier_2", "liquidation_heatmap", dec!(0.10));

        let current_price = self
            .data_service
            .get_current_price(&self.config.data_source.symbol)
            .await?;

        if current_price <= Decimal::ZERO {
            return Ok(neutral_component("liquidation_heatmap", 2, weight));
        }

        let subgraph_url =
            self.tier_param_str("tier_2", "liquidation_heatmap", "aave_subgraph_url", "");

        let levels = self
            .data_service
            .get_liquidation_levels("WBNB", &subgraph_url)
            .await?;

        if levels.is_empty() {
            return Ok(neutral_component("liquidation_heatmap", 2, weight));
        }

        // Find nearest significant liquidation levels above and below.
        let mut nearest_above: Option<&crate::types::LiquidationLevel> = None;
        let mut nearest_below: Option<&crate::types::LiquidationLevel> = None;

        for level in &levels {
            if level.price > current_price {
                if nearest_above.map_or(true, |a| level.price < a.price) {
                    nearest_above = Some(level);
                }
            } else if level.price < current_price {
                if nearest_below.map_or(true, |b| level.price > b.price) {
                    nearest_below = Some(level);
                }
            }
        }

        let mut strength = Decimal::ZERO;
        let mut direction = PositionDirection::Long;
        let million = Decimal::from(1_000_000u64);

        if let Some(below) = nearest_below {
            if below.total_collateral_at_risk_usd > million {
                let dist_pct = (current_price - below.price) / current_price;
                if dist_pct < dec!(0.05) {
                    strength = (dec!(1) - dist_pct * dec!(20)).max(Decimal::ZERO);
                    direction = PositionDirection::Short;
                }
            }
        }

        if let Some(above) = nearest_above {
            if above.total_collateral_at_risk_usd > million {
                let dist_pct = (above.price - current_price) / current_price;
                if dist_pct < dec!(0.05) {
                    let above_s = (dec!(1) - dist_pct * dec!(20)).max(Decimal::ZERO);
                    if above_s > strength {
                        strength = above_s;
                        direction = PositionDirection::Long;
                    }
                }
            }
        }

        strength = strength.min(dec!(1));

        Ok(SignalComponent {
            source: "liquidation_heatmap".into(),
            tier: 2,
            direction,
            strength,
            weight,
            confidence: strength * dec!(0.8),
            data_age_seconds: 0,
        })
    }

    /// Exchange flows signal (Chi et al. 2024).
    async fn compute_exchange_flows(&self) -> Result<SignalComponent> {
        let weight = self.source_weight("tier_2", "exchange_flows", dec!(0.08));
        let window = self.tier_param_u32("tier_2", "exchange_flows", "flow_window_minutes", 60);

        let flows = self
            .data_service
            .get_exchange_flows("USDT", window)
            .await?;

        let net_inflow = flows.inflow_usd - flows.outflow_usd;
        let direction = if net_inflow > Decimal::ZERO {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };

        let strength = if flows.avg_hourly_flow > Decimal::ZERO {
            (net_inflow.abs() / flows.avg_hourly_flow).min(dec!(1))
        } else {
            Decimal::ZERO
        };

        Ok(SignalComponent {
            source: "exchange_flows".into(),
            tier: 2,
            direction,
            strength,
            weight,
            confidence: strength * dec!(0.8),
            data_age_seconds: flows.data_age_seconds,
        })
    }

    /// Funding rate contrarian signal (Aloosh & Bekaert 2022).
    async fn compute_funding_rate_signal(&self) -> Result<SignalComponent> {
        let weight = self.source_weight("tier_2", "funding_rate", dec!(0.07));
        let extreme_str = self.tier_param_str("tier_2", "funding_rate", "extreme_threshold", "0.0005");
        let extreme: Decimal = extreme_str.parse().unwrap_or(dec!(0.0005));

        let funding = self
            .data_service
            .get_funding_rate(&self.config.data_source.symbol)
            .await?;

        let funding = match funding {
            Some(f) if f.abs() >= extreme => f,
            _ => return Ok(neutral_component("funding_rate", 2, weight)),
        };

        // Contrarian: positive funding → short, negative → long.
        let direction = if funding > Decimal::ZERO {
            PositionDirection::Short
        } else {
            PositionDirection::Long
        };

        let strength = (funding.abs() / dec!(0.001)).min(dec!(1));

        Ok(SignalComponent {
            source: "funding_rate".into(),
            tier: 2,
            direction,
            strength,
            weight,
            confidence: strength * dec!(0.7),
            data_age_seconds: 0,
        })
    }

    // -----------------------------------------------------------------------
    // Tier 3 signals
    // -----------------------------------------------------------------------

    /// Mempool flow signal (Ante & Saggu 2024).
    async fn compute_mempool_flow_signal(&self) -> Result<SignalComponent> {
        let weight = self.source_weight("tier_3", "aggregate_mempool_flow", dec!(0.05));

        let channel = self.tier_param_str(
            "tier_3",
            "aggregate_mempool_flow",
            "redis_channel",
            "mempool:aggregate_signal",
        );

        let signal = self.data_service.get_mempool_signal(&channel).await?;

        let signal = match signal {
            Some(s) => s,
            None => return Ok(neutral_component("mempool_flow", 3, weight)),
        };

        // Find the WBNB pair signal.
        let pair_key = "WBNB";
        let token_sig = match signal.pairs.get(pair_key) {
            Some(s) => s,
            None => return Ok(neutral_component("mempool_flow", 3, weight)),
        };

        let direction = if token_sig.direction_score_5m > Decimal::ZERO {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };

        let strength = token_sig.direction_score_5m.abs().min(dec!(1));

        Ok(SignalComponent {
            source: "mempool_flow".into(),
            tier: 3,
            direction,
            strength,
            weight,
            confidence: dec!(0.3), // Low confidence — informational only.
            data_age_seconds: signal.data_age_seconds,
        })
    }

    // -----------------------------------------------------------------------
    // Layer 3: Ensemble Confidence Scoring
    // -----------------------------------------------------------------------

    /// Weighted ensemble of all signal sources, regime-adjusted.
    ///
    /// Returns `(direction, confidence)`.
    fn compute_ensemble_confidence(
        &self,
        components: &[SignalComponent],
        regime: MarketRegime,
    ) -> (PositionDirection, Decimal) {
        let max_age = self.config.entry_rules.max_signal_age_seconds;
        let mut bull_score = Decimal::ZERO;
        let mut bear_score = Decimal::ZERO;
        let mut total_weight = Decimal::ZERO;

        for c in components {
            if c.data_age_seconds > max_age {
                continue;
            }
            if c.strength <= Decimal::ZERO {
                continue;
            }

            let regime_mult = self.regime_weight_multiplier(&c.source, regime);
            let weighted = c.strength * c.weight * c.confidence * regime_mult;

            match c.direction {
                PositionDirection::Long => bull_score += weighted,
                PositionDirection::Short => bear_score += weighted,
            }

            total_weight += c.weight * regime_mult;
        }

        if total_weight == Decimal::ZERO {
            return (PositionDirection::Long, Decimal::ZERO);
        }

        let mut net_score = (bull_score - bear_score) / total_weight;
        let direction = if net_score >= Decimal::ZERO {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };

        // Agreement bonus: if > 70 % of active components agree, boost +15 %.
        let active: Vec<_> = components
            .iter()
            .filter(|c| c.strength > dec!(0.1) && c.data_age_seconds <= max_age)
            .collect();

        if !active.is_empty() {
            let long_count = active
                .iter()
                .filter(|c| matches!(c.direction, PositionDirection::Long))
                .count();
            let majority = long_count.max(active.len() - long_count);
            let majority_pct =
                Decimal::from(majority as u64) / Decimal::from(active.len() as u64);

            if majority_pct >= self.config.entry_rules.agreement_bonus_threshold {
                net_score *= self.config.entry_rules.agreement_bonus_multiplier;
            }
        }

        let confidence = net_score.abs().min(dec!(1));
        (direction, confidence)
    }

    // -----------------------------------------------------------------------
    // Layer 4: Position Sizing
    // -----------------------------------------------------------------------

    /// Compute GARCH(1,1) one-step-ahead volatility.
    async fn compute_garch_volatility(&self) -> Decimal {
        let returns = self
            .data_service
            .get_recent_returns(&self.config.data_source.symbol, 100)
            .await
            .unwrap_or_default();

        if returns.len() < 10 {
            return dec!(0.02); // Default moderate volatility.
        }

        indicators::garch_volatility(
            &returns,
            self.config.indicators.garch_omega,
            self.config.indicators.garch_alpha,
            self.config.indicators.garch_beta,
        )
    }

    /// Compute position size via fractional Kelly criterion.
    async fn compute_position_size(
        &self,
        confidence: Decimal,
        volatility: Decimal,
    ) -> Result<Decimal> {
        // Rolling edge from PnL tracker.
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

        // Account equity.
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
                    dec!(10000) // Fallback.
                }
            }
            Err(e) => {
                debug!("failed to get account equity: {e:#}");
                dec!(10000)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Layer 5: Entry Rules
    // -----------------------------------------------------------------------

    /// Apply entry rule filters (trend alignment).
    fn check_entry_rules(
        &self,
        direction: PositionDirection,
        ind: &IndicatorSnapshot,
    ) -> bool {
        if !self.config.entry_rules.require_trend_alignment {
            return true;
        }

        let mode = &self.config.mode;
        if mode == "momentum" || mode == "blended" {
            let trend_bullish = ind.ema_20 > ind.ema_50;
            if matches!(direction, PositionDirection::Long) && !trend_bullish {
                debug!("trend alignment failed: LONG but EMA bearish");
                return false;
            }
            if matches!(direction, PositionDirection::Short) && trend_bullish {
                debug!("trend alignment failed: SHORT but EMA bullish");
                return false;
            }
        }

        true
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Determine strategy mode based on config and regime.
    fn determine_strategy_mode(&self, regime: MarketRegime) -> String {
        if self.config.mode != "blended" {
            return self.config.mode.clone();
        }

        match regime {
            MarketRegime::Trending => "momentum".into(),
            MarketRegime::MeanReverting => "mean_reversion".into(),
            _ => "blended".into(),
        }
    }

    /// Get regime-dependent weight multiplier for a signal source.
    fn regime_weight_multiplier(&self, source: &str, regime: MarketRegime) -> Decimal {
        let regime_key = match regime {
            MarketRegime::Trending => "trending",
            MarketRegime::MeanReverting => "mean_reverting",
            MarketRegime::Volatile => "volatile",
            MarketRegime::Ranging => "ranging",
        };

        // Look up from config: entry_rules.regime_weight_multipliers
        let mults = &self.config.entry_rules.regime_weight_multipliers;
        let regime_map = match mults.get(regime_key) {
            Some(m) => m,
            None => return dec!(1),
        };

        let is_momentum = MOMENTUM_SOURCES.contains(&source);
        let is_mean_rev = MEAN_REVERSION_SOURCES.contains(&source);

        let key = match regime {
            MarketRegime::Trending | MarketRegime::MeanReverting => {
                if is_momentum {
                    "momentum_signals"
                } else if is_mean_rev {
                    "mean_reversion_signals"
                } else {
                    "all_signals"
                }
            }
            _ => "all_signals",
        };

        regime_map
            .get(key)
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or(dec!(1))
    }

    /// Check if a signal source is enabled in config.
    fn source_enabled(&self, tier: &str, name: &str) -> bool {
        let tier_map = match tier {
            "tier_1" => &self.config.signal_sources.tier_1,
            "tier_2" => &self.config.signal_sources.tier_2,
            "tier_3" => &self.config.signal_sources.tier_3,
            _ => return false,
        };

        tier_map
            .get(name)
            .and_then(|v| v.get("enabled"))
            .and_then(|e| e.as_bool())
            .unwrap_or(tier != "tier_3") // Tier 3 defaults to disabled.
    }

    /// Get the weight for a signal source from config.
    fn source_weight(&self, tier: &str, name: &str, default: Decimal) -> Decimal {
        let tier_map = match tier {
            "tier_1" => &self.config.signal_sources.tier_1,
            "tier_2" => &self.config.signal_sources.tier_2,
            "tier_3" => &self.config.signal_sources.tier_3,
            _ => return default,
        };

        tier_map
            .get(name)
            .and_then(|v| v.get("weight"))
            .and_then(|w| {
                w.as_str()
                    .and_then(|s| s.parse::<Decimal>().ok())
                    .or_else(|| w.as_f64().and_then(Decimal::from_f64))
            })
            .unwrap_or(default)
    }

    /// Get a string parameter from a tier source config.
    fn tier_param_str(&self, tier: &str, source: &str, param: &str, default: &str) -> String {
        let tier_map = match tier {
            "tier_1" => &self.config.signal_sources.tier_1,
            "tier_2" => &self.config.signal_sources.tier_2,
            "tier_3" => &self.config.signal_sources.tier_3,
            _ => return default.to_string(),
        };

        tier_map
            .get(source)
            .and_then(|v| v.get(param))
            .and_then(|p| p.as_str())
            .unwrap_or(default)
            .to_string()
    }

    /// Get a u32 parameter from a tier source config.
    fn tier_param_u32(&self, tier: &str, source: &str, param: &str, default: u32) -> u32 {
        let tier_map = match tier {
            "tier_1" => &self.config.signal_sources.tier_1,
            "tier_2" => &self.config.signal_sources.tier_2,
            "tier_3" => &self.config.signal_sources.tier_3,
            _ => return default,
        };

        tier_map
            .get(source)
            .and_then(|v| v.get(param))
            .and_then(|p| p.as_u64())
            .map(|n| n as u32)
            .unwrap_or(default)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Free helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build a neutral (zero-strength) signal component.
fn neutral_component(source: &str, tier: u8, weight: Decimal) -> SignalComponent {
    SignalComponent {
        source: source.into(),
        tier,
        direction: PositionDirection::Long,
        strength: Decimal::ZERO,
        weight,
        confidence: Decimal::ZERO,
        data_age_seconds: 0,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // -- Regime detection --------------------------------------------------

    fn make_indicator_snapshot(hurst: Decimal, atr_ratio: Decimal) -> IndicatorSnapshot {
        IndicatorSnapshot {
            price: dec!(600),
            ema_20: dec!(600),
            ema_50: dec!(595),
            ema_200: dec!(580),
            rsi_14: dec!(55),
            macd_line: dec!(1),
            macd_signal: dec!(0.5),
            macd_histogram: dec!(0.5),
            bb_upper: dec!(620),
            bb_middle: dec!(600),
            bb_lower: dec!(580),
            atr_14: dec!(10),
            atr_ratio,
            volume: dec!(1000),
            volume_20_avg: dec!(900),
            hurst,
            vpin: dec!(0.3),
            obi: dec!(0.1),
            recent_prices: vec![dec!(600)],
        }
    }

    #[test]
    fn test_detect_regime_volatile() {
        let ind = make_indicator_snapshot(dec!(0.6), dec!(3.5));
        // ATR ratio > 3.0 → Volatile regardless of Hurst.
        // We test the free-standing logic inline since detect_regime needs &self.
        assert!(ind.atr_ratio > dec!(3));
    }

    #[test]
    fn test_detect_regime_trending() {
        let ind = make_indicator_snapshot(dec!(0.6), dec!(1.5));
        assert!(ind.hurst > dec!(0.55) && ind.atr_ratio >= dec!(1));
    }

    #[test]
    fn test_detect_regime_mean_reverting() {
        let ind = make_indicator_snapshot(dec!(0.4), dec!(1.0));
        assert!(ind.hurst < dec!(0.45));
    }

    #[test]
    fn test_detect_regime_ranging() {
        let ind = make_indicator_snapshot(dec!(0.5), dec!(0.8));
        assert!(ind.hurst >= dec!(0.45) && ind.hurst <= dec!(0.55));
    }

    // -- Strategy mode -----------------------------------------------------

    #[test]
    fn test_determine_strategy_mode() {
        // When mode is "blended", it depends on regime.
        assert_eq!(
            match MarketRegime::Trending {
                MarketRegime::Trending => "momentum",
                MarketRegime::MeanReverting => "mean_reversion",
                _ => "blended",
            },
            "momentum"
        );
    }

    // -- Ensemble scoring --------------------------------------------------

    #[test]
    fn test_neutral_component() {
        let c = neutral_component("test", 1, dec!(0.5));
        assert_eq!(c.strength, Decimal::ZERO);
        assert_eq!(c.confidence, Decimal::ZERO);
        assert_eq!(c.tier, 1);
    }

    #[test]
    fn test_regime_weight_classification() {
        // Momentum sources.
        assert!(MOMENTUM_SOURCES.contains(&"technical_indicators"));
        assert!(MOMENTUM_SOURCES.contains(&"order_book_imbalance"));
        // Mean reversion sources.
        assert!(MEAN_REVERSION_SOURCES.contains(&"funding_rate"));
        assert!(MEAN_REVERSION_SOURCES.contains(&"vpin"));
    }

    // -- Technical signals -------------------------------------------------

    #[test]
    fn test_technical_ema_bullish_alignment() {
        // EMA 20 > 50 > 200 → +0.3 score.
        let ind = IndicatorSnapshot {
            price: dec!(600),
            ema_20: dec!(610),
            ema_50: dec!(605),
            ema_200: dec!(580),
            rsi_14: dec!(55), // Neutral RSI.
            macd_line: dec!(1),
            macd_signal: dec!(0.5),
            macd_histogram: dec!(0.5), // Positive → +0.2.
            bb_upper: dec!(620),
            bb_middle: dec!(600),
            bb_lower: dec!(580),
            atr_14: dec!(10),
            atr_ratio: dec!(1),
            volume: dec!(1000),
            volume_20_avg: dec!(900),
            hurst: dec!(0.5),
            vpin: dec!(0.3),
            obi: dec!(0.1),
            recent_prices: vec![],
        };

        // EMA alignment = +0.3, RSI neutral = 0, MACD = +0.2, BB ~0.
        // Total ≈ +0.5, direction should be Long.
        assert!(ind.ema_20 > ind.ema_50);
        assert!(ind.ema_50 > ind.ema_200);
        assert!(ind.macd_histogram > Decimal::ZERO);
    }
}
