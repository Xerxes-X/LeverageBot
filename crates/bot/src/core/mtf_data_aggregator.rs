//! Multi-timeframe data aggregator for the signal engine.
//!
//! Coordinates data fetching across multiple timeframes, using WebSocket
//! streams for sub-hourly data (M1-M30) and REST API for hourly+ (H1-H6).
//!
//! The aggregator provides a unified interface for the signal engine to
//! access OHLCV data across all configured timeframes without worrying
//! about the underlying data source.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use rust_decimal::Decimal;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::config::types::MultiTimeframeConfig;
use crate::core::data_service::DataService;
use crate::core::websocket_manager::WebSocketManager;
use crate::types::market_data::{OrderBookSnapshot, Trade, OHLCV};
use crate::types::timeframe::Timeframe;

/// Data freshness threshold in milliseconds.
/// Data older than this is considered stale.
const FRESHNESS_THRESHOLD_MS: u64 = 120_000; // 2 minutes

/// Multi-timeframe data aggregator.
///
/// Routes data requests to either WebSocket (real-time) or REST (polling)
/// based on the timeframe's data source preference.
pub struct MultiTfDataAggregator {
    data_service: Arc<DataService>,
    ws_manager: Option<Arc<WebSocketManager>>,
    config: MultiTimeframeConfig,
    symbol: String,
    /// Cache for REST-fetched candles (H1+).
    rest_cache: Arc<RwLock<HashMap<Timeframe, Vec<OHLCV>>>>,
}

impl MultiTfDataAggregator {
    /// Create a new multi-timeframe data aggregator.
    pub fn new(
        data_service: Arc<DataService>,
        ws_manager: Option<Arc<WebSocketManager>>,
        config: MultiTimeframeConfig,
        symbol: String,
    ) -> Self {
        Self {
            data_service,
            ws_manager,
            config,
            symbol,
            rest_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get enabled timeframes from configuration.
    pub fn enabled_timeframes(&self) -> Vec<Timeframe> {
        self.config
            .timeframes
            .iter()
            .filter(|tc| tc.enabled)
            .map(|tc| tc.timeframe)
            .collect()
    }

    /// Get candles for a specific timeframe.
    ///
    /// Routes to WebSocket for M1-M30 (if available), REST for H1+.
    pub async fn get_candles(&self, tf: Timeframe) -> Result<Vec<OHLCV>> {
        // Check if this timeframe uses WebSocket
        if tf.uses_websocket() {
            if let Some(ws) = &self.ws_manager {
                if let Some(candles) = ws.get_candles(tf).await {
                    if !candles.is_empty() {
                        debug!(timeframe = ?tf, count = candles.len(), "got candles from websocket");
                        return Ok(candles);
                    }
                }
                // Fall through to REST if WebSocket has no data
                debug!(timeframe = ?tf, "websocket has no data, falling back to REST");
            }
        }

        // Use REST API
        self.get_candles_rest(tf).await
    }

    /// Fetch candles from REST API.
    async fn get_candles_rest(&self, tf: Timeframe) -> Result<Vec<OHLCV>> {
        let limit = self.get_history_candles(tf);
        let interval = tf.as_binance_interval();

        let candles = self
            .data_service
            .get_ohlcv(&self.symbol, interval, limit)
            .await
            .with_context(|| format!("fetch OHLCV for {} {}", self.symbol, interval))?;

        // Update cache
        {
            let mut cache = self.rest_cache.write().await;
            cache.insert(tf, candles.clone());
        }

        debug!(timeframe = ?tf, count = candles.len(), "got candles from REST");
        Ok(candles)
    }

    /// Get candles for multiple timeframes in parallel.
    pub async fn get_multi_tf_candles(
        &self,
        timeframes: &[Timeframe],
    ) -> HashMap<Timeframe, Result<Vec<OHLCV>>> {
        let mut results = HashMap::new();

        // Spawn parallel fetches
        let mut handles = Vec::new();
        for &tf in timeframes {
            let this = self.clone_arc_fields(tf);
            handles.push(tokio::spawn(async move {
                let result = this.get_candles(tf).await;
                (tf, result)
            }));
        }

        // Collect results
        for handle in handles {
            match handle.await {
                Ok((tf, result)) => {
                    results.insert(tf, result);
                }
                Err(e) => {
                    warn!(error = %e, "task join error during multi-tf fetch");
                }
            }
        }

        results
    }

    /// Get the order book snapshot.
    ///
    /// Uses WebSocket if available, otherwise REST.
    pub async fn get_order_book(&self) -> Result<Option<OrderBookSnapshot>> {
        // Try WebSocket first
        if let Some(ws) = &self.ws_manager {
            if let Some(depth) = ws.get_depth().await {
                return Ok(Some(depth));
            }
        }

        // Fall back to REST
        let depth = self
            .data_service
            .get_order_book(&self.symbol, 20)
            .await
            .ok();

        Ok(depth)
    }

    /// Get recent trades.
    ///
    /// Uses WebSocket if available, otherwise REST.
    pub async fn get_recent_trades(&self, limit: usize) -> Result<Vec<Trade>> {
        // Try WebSocket first
        if let Some(ws) = &self.ws_manager {
            let trades = ws.get_trades(limit).await;
            if !trades.is_empty() {
                return Ok(trades);
            }
        }

        // Fall back to REST
        self.data_service
            .get_recent_trades(&self.symbol, limit as u32)
            .await
    }

    /// Get open interest for the symbol.
    pub async fn get_open_interest(&self) -> Result<Decimal> {
        self.data_service.get_open_interest(&self.symbol).await
    }

    /// Get long/short ratio for the symbol.
    pub async fn get_long_short_ratio(&self) -> Result<Decimal> {
        self.data_service.get_long_short_ratio(&self.symbol).await
    }

    /// Get current price.
    pub async fn get_current_price(&self) -> Result<Decimal> {
        self.data_service.get_current_price(&self.symbol).await
    }

    /// Get funding rate.
    pub async fn get_funding_rate(&self) -> Result<Option<Decimal>> {
        self.data_service.get_funding_rate(&self.symbol).await
    }

    /// Check data freshness for a timeframe.
    ///
    /// Returns `true` if data is fresh (within threshold), `false` if stale.
    pub async fn is_data_fresh(&self, tf: Timeframe) -> bool {
        if tf.uses_websocket() {
            if let Some(ws) = &self.ws_manager {
                if let Some(freshness_ms) = ws.get_data_freshness_ms(tf).await {
                    return freshness_ms < FRESHNESS_THRESHOLD_MS;
                }
            }
        }

        // For REST data, check cache age
        let cache = self.rest_cache.read().await;
        if let Some(candles) = cache.get(&tf) {
            if let Some(last) = candles.last() {
                let now = chrono::Utc::now().timestamp();
                let age_secs = (now - last.timestamp).max(0) as u64;
                let expected_interval_secs = tf.duration_secs();
                // Allow up to 2x the interval before considering stale
                return age_secs < expected_interval_secs * 2;
            }
        }

        false
    }

    /// Get data freshness stats for all enabled timeframes.
    pub async fn get_freshness_stats(&self) -> HashMap<Timeframe, DataFreshness> {
        let mut stats = HashMap::new();
        let enabled = self.enabled_timeframes();

        for tf in enabled {
            let is_fresh = self.is_data_fresh(tf).await;
            let source = if tf.uses_websocket() && self.ws_manager.is_some() {
                DataSource::WebSocket
            } else {
                DataSource::Rest
            };

            let candle_count = self.get_candle_count(tf).await;

            stats.insert(
                tf,
                DataFreshness {
                    is_fresh,
                    source,
                    candle_count,
                },
            );
        }

        stats
    }

    /// Get the number of cached candles for a timeframe.
    async fn get_candle_count(&self, tf: Timeframe) -> usize {
        if tf.uses_websocket() {
            if let Some(ws) = &self.ws_manager {
                if let Some(candles) = ws.get_candles(tf).await {
                    return candles.len();
                }
            }
        }

        let cache = self.rest_cache.read().await;
        cache.get(&tf).map(|c| c.len()).unwrap_or(0)
    }

    /// Get history candles setting for a timeframe.
    fn get_history_candles(&self, tf: Timeframe) -> u32 {
        self.config
            .timeframes
            .iter()
            .find(|tc| tc.timeframe == tf)
            .map(|tc| tc.history_candles)
            .unwrap_or(tf.required_candles())
    }

    /// Clone Arc fields for spawning tasks.
    fn clone_arc_fields(&self, _tf: Timeframe) -> MultiTfDataAggregatorInner {
        MultiTfDataAggregatorInner {
            data_service: Arc::clone(&self.data_service),
            ws_manager: self.ws_manager.as_ref().map(Arc::clone),
            symbol: self.symbol.clone(),
            rest_cache: Arc::clone(&self.rest_cache),
            config_timeframes: self.config.timeframes.clone(),
        }
    }
}

/// Inner struct for spawning tasks without full config clone.
struct MultiTfDataAggregatorInner {
    data_service: Arc<DataService>,
    ws_manager: Option<Arc<WebSocketManager>>,
    symbol: String,
    rest_cache: Arc<RwLock<HashMap<Timeframe, Vec<OHLCV>>>>,
    config_timeframes: Vec<crate::config::types::TimeframeConfigEntry>,
}

impl MultiTfDataAggregatorInner {
    async fn get_candles(&self, tf: Timeframe) -> Result<Vec<OHLCV>> {
        // Check if this timeframe uses WebSocket
        if tf.uses_websocket() {
            if let Some(ws) = &self.ws_manager {
                if let Some(candles) = ws.get_candles(tf).await {
                    if !candles.is_empty() {
                        return Ok(candles);
                    }
                }
            }
        }

        // Use REST API
        let limit = self.get_history_candles(tf);
        let interval = tf.as_binance_interval();

        let candles = self
            .data_service
            .get_ohlcv(&self.symbol, interval, limit)
            .await
            .with_context(|| format!("fetch OHLCV for {} {}", self.symbol, interval))?;

        // Update cache
        {
            let mut cache = self.rest_cache.write().await;
            cache.insert(tf, candles.clone());
        }

        Ok(candles)
    }

    fn get_history_candles(&self, tf: Timeframe) -> u32 {
        self.config_timeframes
            .iter()
            .find(|tc| tc.timeframe == tf)
            .map(|tc| tc.history_candles)
            .unwrap_or(tf.required_candles())
    }
}

/// Data source type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataSource {
    WebSocket,
    Rest,
}

/// Data freshness information for a timeframe.
#[derive(Debug, Clone)]
pub struct DataFreshness {
    pub is_fresh: bool,
    pub source: DataSource,
    pub candle_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{TimeframeAggregationConfig, TimeframeConfigEntry};
    use rust_decimal_macros::dec;

    fn make_test_config() -> MultiTimeframeConfig {
        MultiTimeframeConfig {
            enabled: true,
            trading_style: crate::types::timeframe::TradingStyle::MidFrequency,
            timeframes: vec![
                TimeframeConfigEntry {
                    timeframe: Timeframe::M1,
                    enabled: true,
                    weight: dec!(0.05),
                    history_candles: 500,
                    indicator_overrides: None,
                },
                TimeframeConfigEntry {
                    timeframe: Timeframe::H1,
                    enabled: true,
                    weight: dec!(0.20),
                    history_candles: 200,
                    indicator_overrides: None,
                },
                TimeframeConfigEntry {
                    timeframe: Timeframe::H4,
                    enabled: false,
                    weight: dec!(0.18),
                    history_candles: 100,
                    indicator_overrides: None,
                },
            ],
            aggregation: TimeframeAggregationConfig {
                weight_mode: "fixed".to_string(),
                min_timeframe_agreement: dec!(0.5),
                require_higher_tf_alignment: true,
                direction_timeframes: vec![Timeframe::H4, Timeframe::H6],
            },
        }
    }

    #[test]
    fn test_enabled_timeframes() {
        // This test would require a DataService instance which needs network setup.
        // For now, just verify the config parsing logic.
        let config = make_test_config();
        let enabled: Vec<_> = config
            .timeframes
            .iter()
            .filter(|tc| tc.enabled)
            .map(|tc| tc.timeframe)
            .collect();

        assert_eq!(enabled.len(), 2);
        assert!(enabled.contains(&Timeframe::M1));
        assert!(enabled.contains(&Timeframe::H1));
        assert!(!enabled.contains(&Timeframe::H4));
    }

    #[test]
    fn test_data_source_routing() {
        // M1-M30 should prefer WebSocket
        assert!(Timeframe::M1.uses_websocket());
        assert!(Timeframe::M5.uses_websocket());
        assert!(Timeframe::M15.uses_websocket());
        assert!(Timeframe::M30.uses_websocket());

        // H1+ should use REST
        assert!(!Timeframe::H1.uses_websocket());
        assert!(!Timeframe::H2.uses_websocket());
        assert!(!Timeframe::H4.uses_websocket());
        assert!(!Timeframe::H6.uses_websocket());
    }

    #[test]
    fn test_history_candles_from_config() {
        let config = make_test_config();

        // M1 has explicit 500
        let m1_candles = config
            .timeframes
            .iter()
            .find(|tc| tc.timeframe == Timeframe::M1)
            .map(|tc| tc.history_candles)
            .unwrap_or(0);
        assert_eq!(m1_candles, 500);

        // H1 has explicit 200
        let h1_candles = config
            .timeframes
            .iter()
            .find(|tc| tc.timeframe == Timeframe::H1)
            .map(|tc| tc.history_candles)
            .unwrap_or(0);
        assert_eq!(h1_candles, 200);
    }
}
