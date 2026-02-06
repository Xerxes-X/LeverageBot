//! WebSocket manager for real-time Binance market data streaming.
//!
//! Handles kline, depth, and trade streams for sub-hourly timeframes.
//! Uses tokio-tungstenite for async WebSocket connections with automatic
//! reconnection and graceful shutdown support.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use serde::Deserialize;
use tokio::sync::RwLock;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, trace, warn};

use crate::config::types::WebSocketConfig;
use crate::types::market_data::{OrderBookSnapshot, Trade, OHLCV};
use crate::types::timeframe::Timeframe;

/// Maximum number of candles to buffer per timeframe.
const MAX_CANDLE_BUFFER: usize = 500;

/// Maximum number of trades to buffer.
const MAX_TRADE_BUFFER: usize = 2000;

/// WebSocket manager for streaming Binance market data.
pub struct WebSocketManager {
    config: WebSocketConfig,
    symbol: String,
    /// Per-timeframe candle buffers.
    kline_buffers: Arc<RwLock<HashMap<Timeframe, Vec<OHLCV>>>>,
    /// Latest order book snapshot.
    depth_buffer: Arc<RwLock<Option<OrderBookSnapshot>>>,
    /// Recent trades buffer (ring buffer behavior).
    trades_buffer: Arc<RwLock<Vec<Trade>>>,
    shutdown: CancellationToken,
    /// Message statistics for monitoring.
    stats: Arc<WebSocketStats>,
}

/// Statistics for monitoring WebSocket health.
pub struct WebSocketStats {
    /// Total messages received since connection.
    messages_received: AtomicU64,
    /// Total kline messages received.
    kline_messages: AtomicU64,
    /// Total depth messages received.
    depth_messages: AtomicU64,
    /// Total trade messages received.
    trade_messages: AtomicU64,
    /// Total errors encountered.
    errors: AtomicU64,
    /// Connection start time (will be set when connected).
    connected_at: RwLock<Option<Instant>>,
}

impl WebSocketStats {
    fn new() -> Self {
        Self {
            messages_received: AtomicU64::new(0),
            kline_messages: AtomicU64::new(0),
            depth_messages: AtomicU64::new(0),
            trade_messages: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            connected_at: RwLock::new(None),
        }
    }
}

impl WebSocketManager {
    /// Create a new WebSocket manager.
    pub fn new(config: WebSocketConfig, symbol: String, shutdown: CancellationToken) -> Self {
        Self {
            config,
            symbol,
            kline_buffers: Arc::new(RwLock::new(HashMap::new())),
            depth_buffer: Arc::new(RwLock::new(None)),
            trades_buffer: Arc::new(RwLock::new(Vec::with_capacity(MAX_TRADE_BUFFER))),
            shutdown,
            stats: Arc::new(WebSocketStats::new()),
        }
    }

    /// Build the combined stream URL for all subscriptions.
    fn build_stream_url(&self) -> String {
        let mut streams = Vec::new();
        let symbol_lower = self.symbol.to_lowercase();

        // Add kline streams for each enabled timeframe
        for tf in &self.config.subscriptions.klines {
            streams.push(format!("{}@kline_{}", symbol_lower, tf.as_binance_interval()));
        }

        // Add depth stream
        if self.config.subscriptions.depth {
            streams.push(format!("{}@depth20@100ms", symbol_lower));
        }

        // Add trade stream
        if self.config.subscriptions.trades {
            streams.push(format!("{}@aggTrade", symbol_lower));
        }

        format!(
            "{}/stream?streams={}",
            self.config.binance_ws_url,
            streams.join("/")
        )
    }

    /// Main run loop with automatic reconnection.
    pub async fn run(&self) -> Result<()> {
        if !self.config.enabled {
            info!("websocket manager disabled, exiting");
            return Ok(());
        }

        let url = self.build_stream_url();

        // Log subscription details
        info!("═══════════════════════════════════════════════════════════════");
        info!("            WEBSOCKET MANAGER STARTING");
        info!("═══════════════════════════════════════════════════════════════");
        info!(
            symbol = %self.symbol,
            ws_url = %self.config.binance_ws_url,
            "WebSocket configuration"
        );
        info!(
            kline_streams = ?self.config.subscriptions.klines,
            depth_enabled = self.config.subscriptions.depth,
            trades_enabled = self.config.subscriptions.trades,
            "Stream subscriptions"
        );
        info!(
            reconnect_delay_ms = self.config.reconnect_delay_ms,
            max_reconnect_attempts = self.config.max_reconnect_attempts,
            ping_interval_seconds = self.config.ping_interval_seconds,
            "Connection parameters"
        );
        info!("Full stream URL: {}", url);
        info!("───────────────────────────────────────────────────────────────");

        let mut reconnect_attempts = 0u32;

        loop {
            if self.shutdown.is_cancelled() {
                info!("websocket manager: shutdown signal received");
                break;
            }

            match self.connect_and_process(&url).await {
                Ok(_) => {
                    // Clean exit (e.g., shutdown signal)
                    break;
                }
                Err(e) => {
                    self.stats.errors.fetch_add(1, Ordering::Relaxed);
                    reconnect_attempts += 1;

                    if reconnect_attempts > self.config.max_reconnect_attempts {
                        error!(
                            error = %e,
                            attempts = reconnect_attempts,
                            total_errors = self.stats.errors.load(Ordering::Relaxed),
                            "max reconnect attempts exceeded, giving up"
                        );
                        return Err(e);
                    }

                    // Calculate backoff delay
                    let delay_ms = self.config.reconnect_delay_ms * (1 << reconnect_attempts.min(5));

                    warn!(
                        error = %e,
                        attempt = reconnect_attempts,
                        max_attempts = self.config.max_reconnect_attempts,
                        next_retry_ms = delay_ms,
                        "websocket disconnected, reconnecting..."
                    );

                    let delay = Duration::from_millis(delay_ms);

                    tokio::select! {
                        _ = self.shutdown.cancelled() => break,
                        _ = tokio::time::sleep(delay) => {}
                    }
                }
            }
        }

        // Log final statistics
        self.log_final_stats().await;

        info!("websocket manager stopped");
        Ok(())
    }

    /// Log final statistics when shutting down.
    async fn log_final_stats(&self) {
        let uptime = if let Some(connected_at) = *self.stats.connected_at.read().await {
            connected_at.elapsed().as_secs()
        } else {
            0
        };

        info!("───────────────────────────────────────────────────────────────");
        info!("           WEBSOCKET FINAL STATISTICS");
        info!("───────────────────────────────────────────────────────────────");
        info!(
            uptime_secs = uptime,
            total_messages = self.stats.messages_received.load(Ordering::Relaxed),
            kline_messages = self.stats.kline_messages.load(Ordering::Relaxed),
            depth_messages = self.stats.depth_messages.load(Ordering::Relaxed),
            trade_messages = self.stats.trade_messages.load(Ordering::Relaxed),
            errors = self.stats.errors.load(Ordering::Relaxed),
            "Session statistics"
        );

        let stats = self.get_buffer_stats().await;
        info!(
            kline_buffers = ?stats.kline_counts,
            trade_count = stats.trade_count,
            has_depth = stats.has_depth,
            "Buffer statistics"
        );
        info!("═══════════════════════════════════════════════════════════════");
    }

    /// Connect to WebSocket and process messages until disconnection.
    async fn connect_and_process(&self, url: &str) -> Result<()> {
        let connect_start = Instant::now();

        info!("Attempting WebSocket connection...");

        let (ws_stream, response) = connect_async(url)
            .await
            .context("failed to connect to binance websocket")?;

        let connect_time = connect_start.elapsed();

        // Mark connection time
        *self.stats.connected_at.write().await = Some(Instant::now());

        info!(
            connect_time_ms = connect_time.as_millis() as u64,
            status = ?response.status(),
            "WebSocket connected successfully"
        );

        let (mut write, mut read) = ws_stream.split();

        // Ping task to keep connection alive
        let ping_interval = Duration::from_secs(self.config.ping_interval_seconds);
        let shutdown_clone = self.shutdown.clone();

        let ping_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(ping_interval);
            loop {
                tokio::select! {
                    _ = shutdown_clone.cancelled() => break,
                    _ = interval.tick() => {
                        trace!("websocket ping interval tick");
                    }
                }
            }
        });

        // Periodic stats logging task
        let stats_clone = self.stats.clone();
        let stats_shutdown = self.shutdown.clone();
        let stats_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = stats_shutdown.cancelled() => break,
                    _ = interval.tick() => {
                        let msgs = stats_clone.messages_received.load(Ordering::Relaxed);
                        let klines = stats_clone.kline_messages.load(Ordering::Relaxed);
                        let depth = stats_clone.depth_messages.load(Ordering::Relaxed);
                        let trades = stats_clone.trade_messages.load(Ordering::Relaxed);
                        let errors = stats_clone.errors.load(Ordering::Relaxed);

                        if msgs > 0 {
                            debug!(
                                total_messages = msgs,
                                klines = klines,
                                depth = depth,
                                trades = trades,
                                errors = errors,
                                "WebSocket stats (1m interval)"
                            );
                        }
                    }
                }
            }
        });

        // Message processing loop
        loop {
            tokio::select! {
                _ = self.shutdown.cancelled() => {
                    debug!("websocket: shutdown during message loop");
                    // Send close frame
                    let _ = write.send(Message::Close(None)).await;
                    break;
                }
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            self.stats.messages_received.fetch_add(1, Ordering::Relaxed);
                            if let Err(e) = self.process_message(&text).await {
                                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                                warn!(error = %e, "failed to process websocket message");
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            trace!("received ping, sending pong");
                            // Respond to ping with pong
                            if let Err(e) = write.send(Message::Pong(data)).await {
                                warn!(error = %e, "failed to send pong");
                            }
                        }
                        Some(Ok(Message::Close(frame))) => {
                            info!(
                                close_code = ?frame.as_ref().map(|f| f.code),
                                reason = ?frame.as_ref().map(|f| f.reason.to_string()),
                                "websocket: received close frame"
                            );
                            break;
                        }
                        Some(Err(e)) => {
                            return Err(anyhow::anyhow!("websocket error: {}", e));
                        }
                        None => {
                            return Err(anyhow::anyhow!("websocket stream ended"));
                        }
                        _ => {}
                    }
                }
            }
        }

        ping_task.abort();
        stats_task.abort();
        Ok(())
    }

    /// Process a single WebSocket message.
    async fn process_message(&self, text: &str) -> Result<()> {
        // Binance combined stream format: {"stream":"...","data":{...}}
        let wrapper: StreamWrapper = serde_json::from_str(text)
            .context("failed to parse websocket message")?;

        let stream = &wrapper.stream;

        if stream.contains("@kline_") {
            self.handle_kline_message(&wrapper.data).await?;
        } else if stream.contains("@depth") {
            self.handle_depth_message(&wrapper.data).await?;
        } else if stream.contains("@aggTrade") {
            self.handle_trade_message(&wrapper.data).await?;
        }

        Ok(())
    }

    /// Handle kline/candlestick stream message.
    async fn handle_kline_message(&self, data: &serde_json::Value) -> Result<()> {
        self.stats.kline_messages.fetch_add(1, Ordering::Relaxed);

        let kline_data: KlineMessage = serde_json::from_value(data.clone())
            .context("failed to parse kline message")?;

        let k = &kline_data.k;

        // Parse timeframe from interval string
        let tf = Timeframe::from_binance_interval(&k.i)
            .ok_or_else(|| anyhow::anyhow!("unknown interval: {}", k.i))?;

        let candle = OHLCV {
            timestamp: k.t / 1000, // Convert ms to seconds
            open: parse_decimal(&k.o)?,
            high: parse_decimal(&k.h)?,
            low: parse_decimal(&k.l)?,
            close: parse_decimal(&k.c)?,
            volume: parse_decimal(&k.v)?,
        };

        trace!(
            timeframe = ?tf,
            timestamp = candle.timestamp,
            close = %candle.close,
            volume = %candle.volume,
            "received kline update"
        );

        let mut buffers = self.kline_buffers.write().await;
        let buffer = buffers.entry(tf).or_insert_with(Vec::new);

        let is_new_candle;

        // Check if this is an update to the last candle or a new candle
        if let Some(last) = buffer.last_mut() {
            if last.timestamp == candle.timestamp {
                // Update existing candle
                *last = candle.clone();
                is_new_candle = false;
            } else if candle.timestamp > last.timestamp {
                // New candle
                is_new_candle = true;
                buffer.push(candle.clone());
                // Trim buffer if too large
                if buffer.len() > MAX_CANDLE_BUFFER {
                    buffer.remove(0);
                }
            } else {
                is_new_candle = false;
            }
        } else {
            buffer.push(candle.clone());
            is_new_candle = true;
        }

        // Log new candles at debug level
        if is_new_candle {
            debug!(
                timeframe = ?tf,
                timestamp = candle.timestamp,
                open = %candle.open,
                high = %candle.high,
                low = %candle.low,
                close = %candle.close,
                volume = %candle.volume,
                buffer_size = buffer.len(),
                "new candle completed"
            );
        }

        Ok(())
    }

    /// Handle order book depth stream message.
    async fn handle_depth_message(&self, data: &serde_json::Value) -> Result<()> {
        self.stats.depth_messages.fetch_add(1, Ordering::Relaxed);

        let depth: DepthMessage = serde_json::from_value(data.clone())
            .context("failed to parse depth message")?;

        let bids: Vec<(Decimal, Decimal)> = depth
            .bids
            .iter()
            .filter_map(|level| {
                let price = parse_decimal(&level[0]).ok()?;
                let qty = parse_decimal(&level[1]).ok()?;
                Some((price, qty))
            })
            .collect();

        let asks: Vec<(Decimal, Decimal)> = depth
            .asks
            .iter()
            .filter_map(|level| {
                let price = parse_decimal(&level[0]).ok()?;
                let qty = parse_decimal(&level[1]).ok()?;
                Some((price, qty))
            })
            .collect();

        let best_bid = bids.first().map(|(p, _)| *p).unwrap_or(Decimal::ZERO);
        let best_ask = asks.first().map(|(p, _)| *p).unwrap_or(Decimal::ZERO);

        trace!(
            bid_levels = bids.len(),
            ask_levels = asks.len(),
            best_bid = %best_bid,
            best_ask = %best_ask,
            "received depth update"
        );

        let snapshot = OrderBookSnapshot {
            bids,
            asks,
            timestamp: chrono::Utc::now().timestamp(),
        };

        *self.depth_buffer.write().await = Some(snapshot);
        Ok(())
    }

    /// Handle aggregate trade stream message.
    async fn handle_trade_message(&self, data: &serde_json::Value) -> Result<()> {
        self.stats.trade_messages.fetch_add(1, Ordering::Relaxed);

        let trade_msg: AggTradeMessage = serde_json::from_value(data.clone())
            .context("failed to parse trade message")?;

        let trade = Trade {
            price: parse_decimal(&trade_msg.p)?,
            quantity: parse_decimal(&trade_msg.q)?,
            timestamp: trade_msg.trade_time / 1000, // Convert ms to seconds
            is_buyer_maker: trade_msg.m,
        };

        let side = if trade.is_buyer_maker { "SELL" } else { "BUY" };

        trace!(
            price = %trade.price,
            quantity = %trade.quantity,
            side = side,
            "received trade"
        );

        let mut trades = self.trades_buffer.write().await;
        trades.push(trade);

        // Trim if buffer is too large (ring buffer behavior)
        if trades.len() > MAX_TRADE_BUFFER {
            trades.remove(0);
        }

        Ok(())
    }

    /// Get candles for a specific timeframe.
    pub async fn get_candles(&self, tf: Timeframe) -> Option<Vec<OHLCV>> {
        self.kline_buffers.read().await.get(&tf).cloned()
    }

    /// Get the latest order book snapshot.
    pub async fn get_depth(&self) -> Option<OrderBookSnapshot> {
        self.depth_buffer.read().await.clone()
    }

    /// Get recent trades (newest first).
    pub async fn get_trades(&self, limit: usize) -> Vec<Trade> {
        let trades = self.trades_buffer.read().await;
        trades.iter().rev().take(limit).cloned().collect()
    }

    /// Get data freshness in milliseconds for a timeframe.
    pub async fn get_data_freshness_ms(&self, tf: Timeframe) -> Option<u64> {
        let buffers = self.kline_buffers.read().await;
        if let Some(candles) = buffers.get(&tf) {
            if let Some(last) = candles.last() {
                let now = chrono::Utc::now().timestamp();
                let age_secs = (now - last.timestamp).max(0) as u64;
                return Some(age_secs * 1000);
            }
        }
        None
    }

    /// Check if we have data for a timeframe.
    pub async fn has_data(&self, tf: Timeframe) -> bool {
        self.kline_buffers.read().await.contains_key(&tf)
    }

    /// Get statistics about buffered data.
    pub async fn get_buffer_stats(&self) -> BufferStats {
        let klines = self.kline_buffers.read().await;
        let trades = self.trades_buffer.read().await;
        let depth = self.depth_buffer.read().await;

        BufferStats {
            kline_counts: klines.iter().map(|(tf, v)| (*tf, v.len())).collect(),
            trade_count: trades.len(),
            has_depth: depth.is_some(),
        }
    }

    /// Get message statistics for monitoring.
    pub fn get_message_stats(&self) -> MessageStats {
        MessageStats {
            total_messages: self.stats.messages_received.load(Ordering::Relaxed),
            kline_messages: self.stats.kline_messages.load(Ordering::Relaxed),
            depth_messages: self.stats.depth_messages.load(Ordering::Relaxed),
            trade_messages: self.stats.trade_messages.load(Ordering::Relaxed),
            errors: self.stats.errors.load(Ordering::Relaxed),
        }
    }

    /// Check if the WebSocket is connected and receiving data.
    pub async fn is_healthy(&self) -> bool {
        // Consider healthy if we've received at least one message
        self.stats.messages_received.load(Ordering::Relaxed) > 0
    }

    /// Log current buffer and message statistics.
    pub async fn log_status(&self) {
        let buffer_stats = self.get_buffer_stats().await;
        let msg_stats = self.get_message_stats();

        info!(
            total_messages = msg_stats.total_messages,
            klines = msg_stats.kline_messages,
            depth = msg_stats.depth_messages,
            trades = msg_stats.trade_messages,
            errors = msg_stats.errors,
            "WebSocket message statistics"
        );

        info!(
            kline_buffers = ?buffer_stats.kline_counts,
            trade_count = buffer_stats.trade_count,
            has_depth = buffer_stats.has_depth,
            "WebSocket buffer statistics"
        );
    }
}

/// Statistics about WebSocket messages received.
#[derive(Debug, Clone)]
pub struct MessageStats {
    pub total_messages: u64,
    pub kline_messages: u64,
    pub depth_messages: u64,
    pub trade_messages: u64,
    pub errors: u64,
}

/// Statistics about the WebSocket buffer state.
#[derive(Debug, Clone)]
pub struct BufferStats {
    pub kline_counts: HashMap<Timeframe, usize>,
    pub trade_count: usize,
    pub has_depth: bool,
}

// ============================================================================
// Binance WebSocket Message Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct StreamWrapper {
    stream: String,
    data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct KlineMessage {
    k: KlineData,
}

#[derive(Debug, Deserialize)]
struct KlineData {
    /// Kline start time (ms)
    t: i64,
    /// Interval
    i: String,
    /// Open price
    o: String,
    /// High price
    h: String,
    /// Low price
    l: String,
    /// Close price
    c: String,
    /// Volume
    v: String,
}

#[derive(Debug, Deserialize)]
struct DepthMessage {
    bids: Vec<Vec<String>>,
    asks: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct AggTradeMessage {
    /// Price
    p: String,
    /// Quantity
    q: String,
    /// Trade time (ms)
    #[serde(rename = "T")]
    trade_time: i64,
    /// Is buyer maker
    m: bool,
}

/// Parse a decimal from a string value.
fn parse_decimal(s: &str) -> Result<Decimal> {
    s.parse::<Decimal>()
        .with_context(|| format!("failed to parse decimal: {s}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_decimal() {
        assert_eq!(parse_decimal("123.456").unwrap(), Decimal::new(123456, 3));
        assert_eq!(parse_decimal("0").unwrap(), Decimal::ZERO);
        assert!(parse_decimal("invalid").is_err());
    }

    #[test]
    fn test_build_stream_url() {
        let config = WebSocketConfig {
            enabled: true,
            binance_ws_url: "wss://stream.binance.com:9443".to_string(),
            reconnect_delay_ms: 5000,
            max_reconnect_attempts: 10,
            ping_interval_seconds: 30,
            subscriptions: crate::config::types::WebSocketSubscriptions {
                klines: vec![Timeframe::M1, Timeframe::M5],
                depth: true,
                trades: true,
            },
        };

        let manager = WebSocketManager::new(
            config,
            "BNBUSDT".to_string(),
            CancellationToken::new(),
        );

        let url = manager.build_stream_url();
        assert!(url.contains("bnbusdt@kline_1m"));
        assert!(url.contains("bnbusdt@kline_5m"));
        assert!(url.contains("bnbusdt@depth20@100ms"));
        assert!(url.contains("bnbusdt@aggTrade"));
    }
}
