//! Configuration for the mempool decoder binary.
//!
//! Loads from environment variables with sensible defaults.
//! Optionally loads a JSON config file if `DECODER_CONFIG_PATH` is set.

use anyhow::{Context, Result};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;

/// Runtime configuration for the mempool decoder.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// BSC WebSocket URL for pending transaction subscription.
    pub websocket_url: String,
    /// Redis connection URL.
    pub redis_url: String,
    /// Redis key for the aggregate signal.
    pub aggregate_channel: String,
    /// How often to publish aggregate signal to Redis (seconds).
    pub publish_interval_seconds: u64,
    /// LRU dedup cache capacity (number of tx hashes).
    pub dedup_cache_size: usize,
    /// Delay between WebSocket reconnect attempts (seconds).
    pub reconnect_delay_seconds: u64,
    /// Maximum reconnect attempts before giving up.
    pub max_reconnect_attempts: u32,
    /// USD threshold for whale classification.
    pub whale_threshold_usd: Decimal,
    /// Poison score threshold — swaps above this are filtered out.
    pub poison_threshold: f64,
    /// Volatile tokens to track (e.g. ["WBNB", "BTCB", "ETH"]).
    pub monitored_volatile: Vec<String>,
    /// Stablecoin symbols for reference.
    pub monitored_stable: Vec<String>,
}

/// Optional JSON config overlay.
#[derive(Debug, Deserialize)]
struct JsonConfig {
    websocket_url: Option<String>,
    redis_url: Option<String>,
    aggregate_channel: Option<String>,
    publish_interval_seconds: Option<u64>,
    dedup_cache_size: Option<usize>,
    reconnect_delay_seconds: Option<u64>,
    max_reconnect_attempts: Option<u32>,
    whale_threshold_usd: Option<Decimal>,
    poison_threshold: Option<f64>,
    monitored_volatile: Option<Vec<String>>,
    monitored_stable: Option<Vec<String>>,
}

impl DecoderConfig {
    /// Load configuration from environment variables with defaults.
    ///
    /// If `DECODER_CONFIG_PATH` is set, loads a JSON file first and overlays
    /// environment variables on top.
    pub fn from_env() -> Result<Self> {
        // Optionally load JSON base config.
        let json_cfg = match std::env::var("DECODER_CONFIG_PATH").ok() {
            Some(path) if !path.is_empty() => {
                let contents = std::fs::read_to_string(&path)
                    .with_context(|| format!("failed to read config file: {path}"))?;
                Some(
                    serde_json::from_str::<JsonConfig>(&contents)
                        .with_context(|| format!("failed to parse config file: {path}"))?,
                )
            }
            _ => None,
        };

        let ws_fallback = "wss://bsc-ws-node.nariox.org:443".to_string();

        // Environment variable for WebSocket URL: check BSC_RPC_URL_WS first.
        let websocket_url = std::env::var("BSC_RPC_URL_WS")
            .ok()
            .filter(|v| !v.is_empty())
            .or_else(|| json_cfg.as_ref().and_then(|c| c.websocket_url.clone()))
            .unwrap_or(ws_fallback);

        let redis_url = std::env::var("REDIS_URL")
            .ok()
            .filter(|v| !v.is_empty())
            .or_else(|| json_cfg.as_ref().and_then(|c| c.redis_url.clone()))
            .unwrap_or_else(|| "redis://localhost:6379".to_string());

        let aggregate_channel = std::env::var("DECODER_AGGREGATE_CHANNEL")
            .ok()
            .filter(|v| !v.is_empty())
            .or_else(|| json_cfg.as_ref().and_then(|c| c.aggregate_channel.clone()))
            .unwrap_or_else(|| "mempool:aggregate_signal".to_string());

        let publish_interval_seconds = env_parse("DECODER_PUBLISH_INTERVAL")
            .or_else(|| json_cfg.as_ref().and_then(|c| c.publish_interval_seconds))
            .unwrap_or(5);

        let dedup_cache_size = env_parse("DECODER_DEDUP_CACHE_SIZE")
            .or_else(|| json_cfg.as_ref().and_then(|c| c.dedup_cache_size))
            .unwrap_or(100_000);

        let reconnect_delay_seconds = env_parse("DECODER_RECONNECT_DELAY")
            .or_else(|| json_cfg.as_ref().and_then(|c| c.reconnect_delay_seconds))
            .unwrap_or(5);

        let max_reconnect_attempts = env_parse("DECODER_MAX_RECONNECT")
            .or_else(|| json_cfg.as_ref().and_then(|c| c.max_reconnect_attempts))
            .unwrap_or(10);

        let whale_threshold_usd = env_parse("DECODER_WHALE_THRESHOLD")
            .or_else(|| json_cfg.as_ref().and_then(|c| c.whale_threshold_usd))
            .unwrap_or(dec!(10_000));

        let poison_threshold = env_parse("DECODER_POISON_THRESHOLD")
            .or_else(|| json_cfg.as_ref().and_then(|c| c.poison_threshold))
            .unwrap_or(0.7);

        let monitored_volatile = json_cfg
            .as_ref()
            .and_then(|c| c.monitored_volatile.clone())
            .unwrap_or_else(|| vec!["WBNB".into(), "BTCB".into(), "ETH".into()]);

        let monitored_stable = json_cfg
            .as_ref()
            .and_then(|c| c.monitored_stable.clone())
            .unwrap_or_else(|| vec!["USDT".into(), "USDC".into(), "FDUSD".into()]);

        Ok(Self {
            websocket_url,
            redis_url,
            aggregate_channel,
            publish_interval_seconds,
            dedup_cache_size,
            reconnect_delay_seconds,
            max_reconnect_attempts,
            whale_threshold_usd,
            poison_threshold,
            monitored_volatile,
            monitored_stable,
        })
    }
}

/// Parse an environment variable into a type that implements `FromStr`.
fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.is_empty())
        .and_then(|v| v.parse().ok())
}
