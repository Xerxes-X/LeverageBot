//! Startup validation and diagnostic logging.
//!
//! Validates that all data sources are accessible and logs their status
//! at bot startup. This helps ensure the signal pipeline has all required
//! inputs before trading begins.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use rust_decimal::Decimal;
use tracing::{error, info, warn};

use crate::config::SignalConfig;
use crate::core::data_service::DataService;
use crate::core::mtf_data_aggregator::MultiTfDataAggregator;
use crate::core::websocket_manager::WebSocketManager;
use crate::types::timeframe::Timeframe;

/// Result of validating a single data source.
#[derive(Debug)]
pub struct DataSourceStatus {
    pub name: String,
    pub available: bool,
    pub latency_ms: u64,
    pub message: String,
}

/// Comprehensive startup validation results.
#[derive(Debug)]
pub struct StartupValidationResult {
    pub all_critical_passed: bool,
    pub sources: Vec<DataSourceStatus>,
    pub warnings: Vec<String>,
}

/// Startup validator for verifying data source availability.
pub struct StartupValidator {
    data_service: Arc<DataService>,
    config: SignalConfig,
}

impl StartupValidator {
    pub fn new(data_service: Arc<DataService>, config: SignalConfig) -> Self {
        Self {
            data_service,
            config,
        }
    }

    /// Run all startup validations and log results.
    pub async fn validate_all(&self) -> StartupValidationResult {
        info!("═══════════════════════════════════════════════════════════════");
        info!("          STARTUP VALIDATION - DATA SOURCE CHECK");
        info!("═══════════════════════════════════════════════════════════════");

        let mut sources = Vec::new();
        let mut warnings = Vec::new();
        let mut critical_failures = 0;

        let symbol = &self.config.data_source.symbol;

        // 1. Validate OHLCV data (CRITICAL)
        let ohlcv_status = self.validate_ohlcv(symbol).await;
        if !ohlcv_status.available {
            critical_failures += 1;
        }
        sources.push(ohlcv_status);

        // 2. Validate current price (CRITICAL)
        let price_status = self.validate_current_price(symbol).await;
        if !price_status.available {
            critical_failures += 1;
        }
        sources.push(price_status);

        // 3. Validate order book (WARNING if unavailable)
        let ob_status = self.validate_order_book(symbol).await;
        if !ob_status.available {
            warnings.push("Order book unavailable - OBI signals disabled".into());
        }
        sources.push(ob_status);

        // 4. Validate recent trades (WARNING if unavailable)
        let trades_status = self.validate_recent_trades(symbol).await;
        if !trades_status.available {
            warnings.push("Recent trades unavailable - VPIN signals disabled".into());
        }
        sources.push(trades_status);

        // 5. Validate funding rate (OPTIONAL)
        let funding_status = self.validate_funding_rate(symbol).await;
        sources.push(funding_status);

        // 6. Validate Open Interest (OPTIONAL)
        let oi_status = self.validate_open_interest(symbol).await;
        sources.push(oi_status);

        // 7. Validate Long/Short Ratio (OPTIONAL)
        let ls_status = self.validate_long_short_ratio(symbol).await;
        sources.push(ls_status);

        // Log summary
        self.log_validation_summary(&sources, &warnings, critical_failures);

        StartupValidationResult {
            all_critical_passed: critical_failures == 0,
            sources,
            warnings,
        }
    }

    async fn validate_ohlcv(&self, symbol: &str) -> DataSourceStatus {
        let start = Instant::now();
        let interval = &self.config.data_source.interval;
        let limit = self.config.data_source.history_candles;

        match self.data_service.get_ohlcv(symbol, interval, limit).await {
            Ok(candles) => {
                let latency = start.elapsed().as_millis() as u64;
                let last_candle = candles.last();
                let msg = format!(
                    "Received {} candles, latest close: {}",
                    candles.len(),
                    last_candle.map(|c| c.close.to_string()).unwrap_or("N/A".into())
                );
                info!(
                    source = "OHLCV",
                    symbol = symbol,
                    interval = interval,
                    candles = candles.len(),
                    latency_ms = latency,
                    "✓ OHLCV data validated"
                );
                DataSourceStatus {
                    name: format!("OHLCV ({} {})", symbol, interval),
                    available: true,
                    latency_ms: latency,
                    message: msg,
                }
            }
            Err(e) => {
                error!(
                    source = "OHLCV",
                    symbol = symbol,
                    error = %e,
                    "✗ OHLCV data FAILED"
                );
                DataSourceStatus {
                    name: format!("OHLCV ({} {})", symbol, interval),
                    available: false,
                    latency_ms: 0,
                    message: format!("Error: {}", e),
                }
            }
        }
    }

    async fn validate_current_price(&self, symbol: &str) -> DataSourceStatus {
        let start = Instant::now();

        match self.data_service.get_current_price(symbol).await {
            Ok(price) => {
                let latency = start.elapsed().as_millis() as u64;
                info!(
                    source = "Price",
                    symbol = symbol,
                    price = %price,
                    latency_ms = latency,
                    "✓ Current price validated"
                );
                DataSourceStatus {
                    name: format!("Current Price ({})", symbol),
                    available: true,
                    latency_ms: latency,
                    message: format!("Price: {}", price),
                }
            }
            Err(e) => {
                error!(
                    source = "Price",
                    symbol = symbol,
                    error = %e,
                    "✗ Current price FAILED"
                );
                DataSourceStatus {
                    name: format!("Current Price ({})", symbol),
                    available: false,
                    latency_ms: 0,
                    message: format!("Error: {}", e),
                }
            }
        }
    }

    async fn validate_order_book(&self, symbol: &str) -> DataSourceStatus {
        let start = Instant::now();

        match self.data_service.get_order_book(symbol, 20).await {
            Ok(ob) => {
                let latency = start.elapsed().as_millis() as u64;
                let best_bid = ob.bids.first().map(|(p, _)| *p).unwrap_or(Decimal::ZERO);
                let best_ask = ob.asks.first().map(|(p, _)| *p).unwrap_or(Decimal::ZERO);
                let spread = if best_bid > Decimal::ZERO {
                    ((best_ask - best_bid) / best_bid * rust_decimal_macros::dec!(100))
                        .round_dp(4)
                } else {
                    Decimal::ZERO
                };

                info!(
                    source = "OrderBook",
                    symbol = symbol,
                    bid_levels = ob.bids.len(),
                    ask_levels = ob.asks.len(),
                    best_bid = %best_bid,
                    best_ask = %best_ask,
                    spread_pct = %spread,
                    latency_ms = latency,
                    "✓ Order book validated"
                );
                DataSourceStatus {
                    name: format!("Order Book ({})", symbol),
                    available: true,
                    latency_ms: latency,
                    message: format!(
                        "Bids: {}, Asks: {}, Spread: {}%",
                        ob.bids.len(),
                        ob.asks.len(),
                        spread
                    ),
                }
            }
            Err(e) => {
                warn!(
                    source = "OrderBook",
                    symbol = symbol,
                    error = %e,
                    "⚠ Order book unavailable"
                );
                DataSourceStatus {
                    name: format!("Order Book ({})", symbol),
                    available: false,
                    latency_ms: 0,
                    message: format!("Error: {}", e),
                }
            }
        }
    }

    async fn validate_recent_trades(&self, symbol: &str) -> DataSourceStatus {
        let start = Instant::now();

        match self.data_service.get_recent_trades(symbol, 100).await {
            Ok(trades) => {
                let latency = start.elapsed().as_millis() as u64;
                let buy_count = trades.iter().filter(|t| !t.is_buyer_maker).count();
                let sell_count = trades.len() - buy_count;
                let total_volume: Decimal = trades.iter().map(|t| t.quantity).sum();

                info!(
                    source = "Trades",
                    symbol = symbol,
                    trade_count = trades.len(),
                    buy_count = buy_count,
                    sell_count = sell_count,
                    total_volume = %total_volume,
                    latency_ms = latency,
                    "✓ Recent trades validated"
                );
                DataSourceStatus {
                    name: format!("Recent Trades ({})", symbol),
                    available: true,
                    latency_ms: latency,
                    message: format!(
                        "Trades: {}, Buys: {}, Sells: {}, Vol: {}",
                        trades.len(),
                        buy_count,
                        sell_count,
                        total_volume
                    ),
                }
            }
            Err(e) => {
                warn!(
                    source = "Trades",
                    symbol = symbol,
                    error = %e,
                    "⚠ Recent trades unavailable"
                );
                DataSourceStatus {
                    name: format!("Recent Trades ({})", symbol),
                    available: false,
                    latency_ms: 0,
                    message: format!("Error: {}", e),
                }
            }
        }
    }

    async fn validate_funding_rate(&self, symbol: &str) -> DataSourceStatus {
        let start = Instant::now();

        match self.data_service.get_funding_rate(symbol).await {
            Ok(Some(rate)) => {
                let latency = start.elapsed().as_millis() as u64;
                let rate_pct = rate * rust_decimal_macros::dec!(100);
                info!(
                    source = "FundingRate",
                    symbol = symbol,
                    rate = %rate,
                    rate_pct = %rate_pct,
                    latency_ms = latency,
                    "✓ Funding rate validated"
                );
                DataSourceStatus {
                    name: format!("Funding Rate ({})", symbol),
                    available: true,
                    latency_ms: latency,
                    message: format!("Rate: {}%", rate_pct.round_dp(4)),
                }
            }
            Ok(None) => {
                info!(
                    source = "FundingRate",
                    symbol = symbol,
                    "○ Funding rate not available (no futures data)"
                );
                DataSourceStatus {
                    name: format!("Funding Rate ({})", symbol),
                    available: false,
                    latency_ms: 0,
                    message: "No futures data available".into(),
                }
            }
            Err(e) => {
                warn!(
                    source = "FundingRate",
                    symbol = symbol,
                    error = %e,
                    "⚠ Funding rate error"
                );
                DataSourceStatus {
                    name: format!("Funding Rate ({})", symbol),
                    available: false,
                    latency_ms: 0,
                    message: format!("Error: {}", e),
                }
            }
        }
    }

    async fn validate_open_interest(&self, symbol: &str) -> DataSourceStatus {
        let start = Instant::now();

        match self.data_service.get_open_interest(symbol).await {
            Ok(oi) => {
                let latency = start.elapsed().as_millis() as u64;
                info!(
                    source = "OpenInterest",
                    symbol = symbol,
                    open_interest = %oi,
                    latency_ms = latency,
                    "✓ Open Interest validated"
                );
                DataSourceStatus {
                    name: format!("Open Interest ({})", symbol),
                    available: true,
                    latency_ms: latency,
                    message: format!("OI: {}", oi),
                }
            }
            Err(e) => {
                warn!(
                    source = "OpenInterest",
                    symbol = symbol,
                    error = %e,
                    "⚠ Open Interest unavailable"
                );
                DataSourceStatus {
                    name: format!("Open Interest ({})", symbol),
                    available: false,
                    latency_ms: 0,
                    message: format!("Error: {}", e),
                }
            }
        }
    }

    async fn validate_long_short_ratio(&self, symbol: &str) -> DataSourceStatus {
        let start = Instant::now();

        match self.data_service.get_long_short_ratio(symbol).await {
            Ok(ratio) => {
                let latency = start.elapsed().as_millis() as u64;
                let bias = if ratio > rust_decimal_macros::dec!(1.0) {
                    "Long-biased"
                } else if ratio < rust_decimal_macros::dec!(1.0) {
                    "Short-biased"
                } else {
                    "Neutral"
                };
                info!(
                    source = "LongShortRatio",
                    symbol = symbol,
                    ratio = %ratio,
                    bias = bias,
                    latency_ms = latency,
                    "✓ Long/Short Ratio validated"
                );
                DataSourceStatus {
                    name: format!("Long/Short Ratio ({})", symbol),
                    available: true,
                    latency_ms: latency,
                    message: format!("Ratio: {} ({})", ratio.round_dp(3), bias),
                }
            }
            Err(e) => {
                warn!(
                    source = "LongShortRatio",
                    symbol = symbol,
                    error = %e,
                    "⚠ Long/Short Ratio unavailable"
                );
                DataSourceStatus {
                    name: format!("Long/Short Ratio ({})", symbol),
                    available: false,
                    latency_ms: 0,
                    message: format!("Error: {}", e),
                }
            }
        }
    }

    fn log_validation_summary(
        &self,
        sources: &[DataSourceStatus],
        warnings: &[String],
        critical_failures: usize,
    ) {
        info!("───────────────────────────────────────────────────────────────");
        info!("                    VALIDATION SUMMARY");
        info!("───────────────────────────────────────────────────────────────");

        let available_count = sources.iter().filter(|s| s.available).count();
        let total_count = sources.len();

        for source in sources {
            let status = if source.available { "✓" } else { "✗" };
            info!(
                "{} {} [{}ms] - {}",
                status, source.name, source.latency_ms, source.message
            );
        }

        if !warnings.is_empty() {
            info!("───────────────────────────────────────────────────────────────");
            info!("WARNINGS:");
            for warning in warnings {
                warn!("  ⚠ {}", warning);
            }
        }

        info!("───────────────────────────────────────────────────────────────");
        if critical_failures > 0 {
            error!(
                "RESULT: {}/{} data sources available - {} CRITICAL FAILURES",
                available_count, total_count, critical_failures
            );
        } else {
            info!(
                "RESULT: {}/{} data sources available - ALL CRITICAL CHECKS PASSED",
                available_count, total_count
            );
        }
        info!("═══════════════════════════════════════════════════════════════");
    }
}

/// Validate multi-timeframe data sources.
pub struct MultiTfStartupValidator {
    aggregator: Arc<MultiTfDataAggregator>,
    ws_manager: Option<Arc<WebSocketManager>>,
}

impl MultiTfStartupValidator {
    pub fn new(
        aggregator: Arc<MultiTfDataAggregator>,
        ws_manager: Option<Arc<WebSocketManager>>,
    ) -> Self {
        Self {
            aggregator,
            ws_manager,
        }
    }

    /// Validate all timeframe data sources.
    pub async fn validate_timeframes(&self) -> HashMap<Timeframe, bool> {
        info!("═══════════════════════════════════════════════════════════════");
        info!("       MULTI-TIMEFRAME DATA SOURCE VALIDATION");
        info!("═══════════════════════════════════════════════════════════════");

        let mut results = HashMap::new();
        let timeframes = self.aggregator.enabled_timeframes();

        for tf in &timeframes {
            let start = Instant::now();
            match self.aggregator.get_candles(*tf).await {
                Ok(candles) if !candles.is_empty() => {
                    let latency = start.elapsed().as_millis();
                    let source = if tf.uses_websocket() && self.ws_manager.is_some() {
                        "WebSocket"
                    } else {
                        "REST"
                    };
                    info!(
                        timeframe = ?tf,
                        candles = candles.len(),
                        source = source,
                        latency_ms = latency,
                        last_close = %candles.last().map(|c| c.close).unwrap_or_default(),
                        "✓ Timeframe data validated"
                    );
                    results.insert(*tf, true);
                }
                Ok(_) => {
                    warn!(
                        timeframe = ?tf,
                        "⚠ Timeframe has no candles yet"
                    );
                    results.insert(*tf, false);
                }
                Err(e) => {
                    error!(
                        timeframe = ?tf,
                        error = %e,
                        "✗ Timeframe data FAILED"
                    );
                    results.insert(*tf, false);
                }
            }
        }

        // Log WebSocket status if available
        if let Some(ws) = &self.ws_manager {
            let stats = ws.get_buffer_stats().await;
            info!("───────────────────────────────────────────────────────────────");
            info!("WebSocket Buffer Status:");
            info!("  Kline buffers: {:?}", stats.kline_counts);
            info!("  Trade buffer: {} trades", stats.trade_count);
            info!("  Depth available: {}", stats.has_depth);
        }

        let success_count = results.values().filter(|&&v| v).count();
        info!("───────────────────────────────────────────────────────────────");
        info!(
            "RESULT: {}/{} timeframes validated successfully",
            success_count,
            timeframes.len()
        );
        info!("═══════════════════════════════════════════════════════════════");

        results
    }
}

/// Log indicator computation results for debugging.
pub fn log_indicator_snapshot(
    snapshot: &crate::types::IndicatorSnapshot,
    symbol: &str,
    timeframe: Option<Timeframe>,
) {
    let tf_str = timeframe
        .map(|t| format!("{:?}", t))
        .unwrap_or_else(|| "default".into());

    info!(
        symbol = symbol,
        timeframe = tf_str,
        price = %snapshot.price,
        ema_20 = %snapshot.ema_20,
        ema_50 = %snapshot.ema_50,
        ema_200 = %snapshot.ema_200,
        rsi = %snapshot.rsi_14.round_dp(2),
        macd = %snapshot.macd_line.round_dp(4),
        macd_signal = %snapshot.macd_signal.round_dp(4),
        macd_hist = %snapshot.macd_histogram.round_dp(4),
        bb_upper = %snapshot.bb_upper.round_dp(2),
        bb_lower = %snapshot.bb_lower.round_dp(2),
        atr = %snapshot.atr_14.round_dp(4),
        atr_ratio = %snapshot.atr_ratio.round_dp(2),
        hurst = %snapshot.hurst.round_dp(3),
        vpin = %snapshot.vpin.round_dp(3),
        obi = %snapshot.obi.round_dp(3),
        "Indicator snapshot computed"
    );
}

/// Log signal component details for debugging.
pub fn log_signal_component(component: &crate::types::SignalComponent) {
    info!(
        source = %component.source,
        tier = component.tier,
        direction = ?component.direction,
        strength = %component.strength.round_dp(3),
        weight = %component.weight.round_dp(3),
        confidence = %component.confidence.round_dp(3),
        age_secs = component.data_age_seconds,
        "Signal component generated"
    );
}

/// Log multi-TF signal component details.
pub fn log_mtf_signal_component(component: &crate::types::signal::MultiTfSignalComponent) {
    info!(
        source = %component.source,
        tier = component.tier,
        timeframe = ?component.timeframe,
        direction = ?component.direction,
        strength = %component.strength.round_dp(3),
        weight = %component.weight.round_dp(3),
        confidence = %component.confidence.round_dp(3),
        "Multi-TF signal component generated"
    );
}
