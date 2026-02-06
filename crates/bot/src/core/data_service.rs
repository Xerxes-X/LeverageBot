//! Multi-source market data service for BSC Leverage Bot.
//!
//! Fetches, caches, and normalises market data from multiple sources for the
//! signal pipeline.  Chainlink alone is insufficient — it provides only the
//! latest spot price with 27-60 s resolution, no volume data, and no OHLCV
//! history.  The signal architecture requires order book depth, recent trades,
//! funding rates, exchange flow proxies, and liquidation level data.
//!
//! Data Sources:
//!   - Binance Spot API: klines, depth, aggTrades, ticker/price
//!   - Binance Futures API: fundingRate
//!   - Aave V3 Subgraph: liquidation level distribution
//!   - Redis (optional): mempool aggregate signal from Phase 10 decoder
//!
//! Caching (in-memory HashMap with per-data-type TTL):
//!   - OHLCV 1h: 60 s, other: 30 s
//!   - Order book: 5 s
//!   - Recent trades: 10 s
//!   - Funding rate: 300 s
//!   - Liquidation levels: 300 s
//!   - Exchange flows: 120 s
//!   - Current price: 5 s

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde_json::Value;
use std::sync::Arc;
use tracing::{debug, trace, warn};

use crate::config::{BinanceRateLimitConfig, DataSourceConfig};
use crate::errors::BotError;
use crate::execution::aave_client::AaveClient;
use crate::types::{
    ExchangeFlows, LiquidationLevel, MempoolSignal, OrderBookSnapshot, Trade, OHLCV,
};

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

const BINANCE_SPOT_BASE: &str = "https://api.binance.com";
const BINANCE_FUTURES_BASE: &str = "https://fapi.binance.com";

const TTL_OHLCV_1H: Duration = Duration::from_secs(60);
const TTL_OHLCV_OTHER: Duration = Duration::from_secs(30);
const TTL_ORDER_BOOK: Duration = Duration::from_secs(5);
const TTL_RECENT_TRADES: Duration = Duration::from_secs(10);
const TTL_FUNDING_RATE: Duration = Duration::from_secs(300);
const TTL_CURRENT_PRICE: Duration = Duration::from_secs(5);
const TTL_LIQUIDATION_LEVELS: Duration = Duration::from_secs(300);
const TTL_EXCHANGE_FLOWS: Duration = Duration::from_secs(120);
const TTL_OPEN_INTEREST: Duration = Duration::from_secs(60);
const TTL_LONG_SHORT_RATIO: Duration = Duration::from_secs(60);

// ═══════════════════════════════════════════════════════════════════════════
// Cache
// ═══════════════════════════════════════════════════════════════════════════

/// A single cache entry with expiration.
#[derive(Clone)]
struct CacheEntry<T: Clone> {
    data: T,
    expires_at: Instant,
}

impl<T: Clone> CacheEntry<T> {
    fn new(data: T, ttl: Duration) -> Self {
        Self {
            data,
            expires_at: Instant::now() + ttl,
        }
    }

    fn is_valid(&self) -> bool {
        Instant::now() < self.expires_at
    }
}

/// Per-data-type in-memory cache.
struct DataCache {
    ohlcv: HashMap<String, CacheEntry<Vec<OHLCV>>>,
    order_book: HashMap<String, CacheEntry<OrderBookSnapshot>>,
    trades: HashMap<String, CacheEntry<Vec<Trade>>>,
    funding_rate: HashMap<String, CacheEntry<Decimal>>,
    current_price: HashMap<String, CacheEntry<Decimal>>,
    liquidation_levels: HashMap<String, CacheEntry<Vec<LiquidationLevel>>>,
    exchange_flows: HashMap<String, CacheEntry<ExchangeFlows>>,
    open_interest: HashMap<String, CacheEntry<Decimal>>,
    long_short_ratio: HashMap<String, CacheEntry<Decimal>>,
}

impl DataCache {
    fn new() -> Self {
        Self {
            ohlcv: HashMap::new(),
            order_book: HashMap::new(),
            trades: HashMap::new(),
            funding_rate: HashMap::new(),
            current_price: HashMap::new(),
            liquidation_levels: HashMap::new(),
            exchange_flows: HashMap::new(),
            open_interest: HashMap::new(),
            long_short_ratio: HashMap::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DataService
// ═══════════════════════════════════════════════════════════════════════════

/// Async multi-source market data fetcher with per-data-type caching.
///
/// All public methods return typed structs from `crate::types`.  Network
/// failures are logged but returned as errors — the signal engine degrades
/// gracefully when individual data sources are unavailable.
pub struct DataService {
    client: reqwest::Client,
    aave_client: Arc<AaveClient>,
    redis_client: Option<redis::Client>,
    cache: Mutex<DataCache>,
    config: DataSourceConfig,
    #[allow(dead_code)]
    rate_limits: BinanceRateLimitConfig,
}

impl DataService {
    /// Create a new `DataService`.
    ///
    /// # Arguments
    /// * `config` — data source configuration (symbol, interval, etc.)
    /// * `aave_client` — shared Aave client for on-chain queries
    /// * `redis_client` — optional Redis connection for mempool signals
    /// * `rate_limits` — Binance rate limit parameters
    pub fn new(
        config: DataSourceConfig,
        aave_client: Arc<AaveClient>,
        redis_client: Option<redis::Client>,
        rate_limits: BinanceRateLimitConfig,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("reqwest client should build");

        Self {
            client,
            aave_client,
            redis_client,
            cache: Mutex::new(DataCache::new()),
            config,
            rate_limits,
        }
    }

    // -----------------------------------------------------------------------
    // Private: HTTP + cache helpers
    // -----------------------------------------------------------------------

    /// Issue a GET request to a Binance endpoint and return the parsed JSON.
    async fn binance_get(&self, base: &str, path: &str, params: &[(&str, &str)]) -> Result<Value> {
        let url = format!("{base}{path}");
        let resp = self
            .client
            .get(&url)
            .query(params)
            .send()
            .await
            .with_context(|| format!("GET {url}"))?;

        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            warn!("Rate limited by {url}");
            return Err(BotError::DataUnavailable {
                name: url.clone(),
            }
            .into());
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            warn!("HTTP {status} from {url}: {body}");
            return Err(BotError::DataUnavailable { name: url }.into());
        }

        resp.json::<Value>()
            .await
            .with_context(|| format!("parse JSON from {url}"))
    }

    /// Current unix timestamp in seconds.
    fn now_unix() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }

    // -----------------------------------------------------------------------
    // Core OHLCV (Binance klines)
    // -----------------------------------------------------------------------

    /// Fetch OHLCV candles from Binance Spot API.
    ///
    /// Binance `/api/v3/klines` returns `[[open_time, O, H, L, C, V, …], …]`.
    pub async fn get_ohlcv(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<OHLCV>> {
        let cache_key = format!("ohlcv:{symbol}:{interval}:{limit}");

        // Check cache.
        {
            let cache = self.cache.lock().expect("cache lock poisoned");
            if let Some(entry) = cache.ohlcv.get(&cache_key) {
                if entry.is_valid() {
                    trace!(
                        cache_key = %cache_key,
                        candles = entry.data.len(),
                        "OHLCV cache HIT"
                    );
                    return Ok(entry.data.clone());
                }
            }
        }

        trace!(
            cache_key = %cache_key,
            "OHLCV cache MISS - fetching from API"
        );

        let start = Instant::now();
        let limit_str = limit.to_string();
        let data = self
            .binance_get(
                BINANCE_SPOT_BASE,
                "/api/v3/klines",
                &[("symbol", symbol), ("interval", interval), ("limit", &limit_str)],
            )
            .await?;

        let latency = start.elapsed();

        let arr = data.as_array().ok_or_else(|| {
            BotError::DataUnavailable {
                name: "klines response not an array".into(),
            }
        })?;

        let mut candles = Vec::with_capacity(arr.len());
        for k in arr {
            let items = match k.as_array() {
                Some(a) if a.len() >= 6 => a,
                _ => continue,
            };
            let timestamp = items[0].as_i64().unwrap_or(0) / 1000;
            let open = parse_decimal_str(&items[1]);
            let high = parse_decimal_str(&items[2]);
            let low = parse_decimal_str(&items[3]);
            let close = parse_decimal_str(&items[4]);
            let volume = parse_decimal_str(&items[5]);

            candles.push(OHLCV {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
            });
        }

        let ttl = if interval == "1h" {
            TTL_OHLCV_1H
        } else {
            TTL_OHLCV_OTHER
        };

        debug!(
            symbol = symbol,
            interval = interval,
            candles = candles.len(),
            latency_ms = latency.as_millis() as u64,
            ttl_secs = ttl.as_secs(),
            latest_close = %candles.last().map(|c| c.close).unwrap_or_default(),
            "OHLCV fetched and cached"
        );

        {
            let mut cache = self.cache.lock().expect("cache lock poisoned");
            cache
                .ohlcv
                .insert(cache_key, CacheEntry::new(candles.clone(), ttl));
        }

        Ok(candles)
    }

    // -----------------------------------------------------------------------
    // Current price
    // -----------------------------------------------------------------------

    /// Get the current price from Binance ticker.
    pub async fn get_current_price(&self, symbol: &str) -> Result<Decimal> {
        let cache_key = format!("price:{symbol}");

        {
            let cache = self.cache.lock().expect("cache lock poisoned");
            if let Some(entry) = cache.current_price.get(&cache_key) {
                if entry.is_valid() {
                    trace!(symbol = symbol, price = %entry.data, "price cache HIT");
                    return Ok(entry.data);
                }
            }
        }

        trace!(symbol = symbol, "price cache MISS - fetching from API");

        let start = Instant::now();
        let data = self
            .binance_get(
                BINANCE_SPOT_BASE,
                "/api/v3/ticker/price",
                &[("symbol", symbol)],
            )
            .await?;

        let latency = start.elapsed();

        let price = data
            .get("price")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or(Decimal::ZERO);

        debug!(
            symbol = symbol,
            price = %price,
            latency_ms = latency.as_millis() as u64,
            "price fetched and cached"
        );

        {
            let mut cache = self.cache.lock().expect("cache lock poisoned");
            cache
                .current_price
                .insert(cache_key, CacheEntry::new(price, TTL_CURRENT_PRICE));
        }

        Ok(price)
    }

    // -----------------------------------------------------------------------
    // Order book (for OBI — Kolm et al. 2023)
    // -----------------------------------------------------------------------

    /// Fetch order book depth from Binance.
    pub async fn get_order_book(
        &self,
        symbol: &str,
        limit: u32,
    ) -> Result<OrderBookSnapshot> {
        let cache_key = format!("depth:{symbol}:{limit}");

        {
            let cache = self.cache.lock().expect("cache lock poisoned");
            if let Some(entry) = cache.order_book.get(&cache_key) {
                if entry.is_valid() {
                    trace!(symbol = symbol, "order book cache HIT");
                    return Ok(entry.data.clone());
                }
            }
        }

        trace!(symbol = symbol, "order book cache MISS - fetching from API");

        let start = Instant::now();
        let limit_str = limit.to_string();
        let data = self
            .binance_get(
                BINANCE_SPOT_BASE,
                "/api/v3/depth",
                &[("symbol", symbol), ("limit", &limit_str)],
            )
            .await?;

        let latency = start.elapsed();

        let bids = parse_price_qty_array(data.get("bids"));
        let asks = parse_price_qty_array(data.get("asks"));

        let snapshot = OrderBookSnapshot {
            bids,
            asks,
            timestamp: Self::now_unix(),
        };

        debug!(
            symbol = symbol,
            bid_levels = snapshot.bids.len(),
            ask_levels = snapshot.asks.len(),
            latency_ms = latency.as_millis() as u64,
            "order book fetched and cached"
        );

        {
            let mut cache = self.cache.lock().expect("cache lock poisoned");
            cache
                .order_book
                .insert(cache_key, CacheEntry::new(snapshot.clone(), TTL_ORDER_BOOK));
        }

        Ok(snapshot)
    }

    // -----------------------------------------------------------------------
    // Recent trades (for VPIN — Easley et al. 2012)
    // -----------------------------------------------------------------------

    /// Fetch aggregated recent trades from Binance.
    pub async fn get_recent_trades(
        &self,
        symbol: &str,
        limit: u32,
    ) -> Result<Vec<Trade>> {
        let cache_key = format!("trades:{symbol}:{limit}");

        {
            let cache = self.cache.lock().expect("cache lock poisoned");
            if let Some(entry) = cache.trades.get(&cache_key) {
                if entry.is_valid() {
                    trace!(symbol = symbol, "trades cache HIT");
                    return Ok(entry.data.clone());
                }
            }
        }

        trace!(symbol = symbol, "trades cache MISS - fetching from API");

        let start = Instant::now();
        let limit_str = limit.to_string();
        let data = self
            .binance_get(
                BINANCE_SPOT_BASE,
                "/api/v3/aggTrades",
                &[("symbol", symbol), ("limit", &limit_str)],
            )
            .await?;

        let latency = start.elapsed();

        let arr = data.as_array().ok_or_else(|| BotError::DataUnavailable {
            name: "aggTrades response not an array".into(),
        })?;

        let mut trades = Vec::with_capacity(arr.len());
        for t in arr {
            let price = t
                .get("p")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<Decimal>().ok())
                .unwrap_or(Decimal::ZERO);
            let quantity = t
                .get("q")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<Decimal>().ok())
                .unwrap_or(Decimal::ZERO);
            let timestamp = t
                .get("T")
                .and_then(|v| v.as_i64())
                .unwrap_or(0)
                / 1000;
            let is_buyer_maker = t
                .get("m")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            trades.push(Trade {
                price,
                quantity,
                timestamp,
                is_buyer_maker,
            });
        }

        let buy_count = trades.iter().filter(|t| !t.is_buyer_maker).count();
        let total_volume: Decimal = trades.iter().map(|t| t.quantity).sum();

        debug!(
            symbol = symbol,
            trade_count = trades.len(),
            buy_count = buy_count,
            sell_count = trades.len() - buy_count,
            total_volume = %total_volume,
            latency_ms = latency.as_millis() as u64,
            "trades fetched and cached"
        );

        {
            let mut cache = self.cache.lock().expect("cache lock poisoned");
            cache
                .trades
                .insert(cache_key, CacheEntry::new(trades.clone(), TTL_RECENT_TRADES));
        }

        Ok(trades)
    }

    // -----------------------------------------------------------------------
    // Funding rate (Binance Futures)
    // -----------------------------------------------------------------------

    /// Get latest funding rate from Binance perpetual futures.
    ///
    /// Extreme positive funding → contrarian short; extreme negative → long.
    pub async fn get_funding_rate(&self, symbol: &str) -> Result<Option<Decimal>> {
        let cache_key = format!("funding:{symbol}");

        {
            let cache = self.cache.lock().expect("cache lock poisoned");
            if let Some(entry) = cache.funding_rate.get(&cache_key) {
                if entry.is_valid() {
                    return Ok(Some(entry.data));
                }
            }
        }

        let data = self
            .binance_get(
                BINANCE_FUTURES_BASE,
                "/fapi/v1/fundingRate",
                &[("symbol", symbol), ("limit", "1")],
            )
            .await?;

        let arr = match data.as_array() {
            Some(a) if !a.is_empty() => a,
            _ => return Ok(None),
        };

        let rate = arr[0]
            .get("fundingRate")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok());

        if let Some(r) = rate {
            let mut cache = self.cache.lock().expect("cache lock poisoned");
            cache
                .funding_rate
                .insert(cache_key, CacheEntry::new(r, TTL_FUNDING_RATE));
        }

        Ok(rate)
    }

    // -----------------------------------------------------------------------
    // Open Interest (Binance Futures)
    // -----------------------------------------------------------------------

    /// Get open interest from Binance perpetual futures.
    ///
    /// Open interest represents the total number of outstanding derivative
    /// contracts (not yet settled). Rising OI with rising price = strong trend.
    /// Falling OI = positions being closed, potential reversal.
    ///
    /// Easley et al. (2012): OI provides information about trader conviction.
    pub async fn get_open_interest(&self, symbol: &str) -> Result<Decimal> {
        let cache_key = format!("oi:{symbol}");

        {
            let cache = self.cache.lock().expect("cache lock poisoned");
            if let Some(entry) = cache.open_interest.get(&cache_key) {
                if entry.is_valid() {
                    return Ok(entry.data);
                }
            }
        }

        let data = self
            .binance_get(
                BINANCE_FUTURES_BASE,
                "/fapi/v1/openInterest",
                &[("symbol", symbol)],
            )
            .await?;

        let oi = data
            .get("openInterest")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or(Decimal::ZERO);

        {
            let mut cache = self.cache.lock().expect("cache lock poisoned");
            cache
                .open_interest
                .insert(cache_key, CacheEntry::new(oi, TTL_OPEN_INTEREST));
        }

        Ok(oi)
    }

    // -----------------------------------------------------------------------
    // Long/Short Ratio (Binance Futures)
    // -----------------------------------------------------------------------

    /// Get global long/short account ratio from Binance perpetual futures.
    ///
    /// Ratio > 1 = more longs than shorts (crowded long, contrarian short).
    /// Ratio < 1 = more shorts than longs (crowded short, contrarian long).
    ///
    /// Extreme readings (> 2.0 or < 0.5) suggest potential mean reversion.
    pub async fn get_long_short_ratio(&self, symbol: &str) -> Result<Decimal> {
        let cache_key = format!("lsr:{symbol}");

        {
            let cache = self.cache.lock().expect("cache lock poisoned");
            if let Some(entry) = cache.long_short_ratio.get(&cache_key) {
                if entry.is_valid() {
                    return Ok(entry.data);
                }
            }
        }

        let data = self
            .binance_get(
                BINANCE_FUTURES_BASE,
                "/fapi/v1/globalLongShortAccountRatio",
                &[("symbol", symbol), ("period", "5m"), ("limit", "1")],
            )
            .await?;

        let arr = match data.as_array() {
            Some(a) if !a.is_empty() => a,
            _ => return Ok(Decimal::ONE), // Default to neutral if unavailable
        };

        let ratio = arr[0]
            .get("longShortRatio")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or(Decimal::ONE);

        {
            let mut cache = self.cache.lock().expect("cache lock poisoned");
            cache
                .long_short_ratio
                .insert(cache_key, CacheEntry::new(ratio, TTL_LONG_SHORT_RATIO));
        }

        Ok(ratio)
    }

    // -----------------------------------------------------------------------
    // Liquidation levels (Aave V3 subgraph)
    // -----------------------------------------------------------------------

    /// Query Aave V3 subgraph for position health-factor distribution and
    /// compute liquidation price levels.
    ///
    /// Perez et al. (FC 2021): small price variations (3%) can make > $10 M
    /// in DeFi positions liquidatable, creating cascading sell pressure.
    pub async fn get_liquidation_levels(
        &self,
        asset: &str,
        subgraph_url: &str,
    ) -> Result<Vec<LiquidationLevel>> {
        let cache_key = format!("liq_levels:{asset}");

        {
            let cache = self.cache.lock().expect("cache lock poisoned");
            if let Some(entry) = cache.liquidation_levels.get(&cache_key) {
                if entry.is_valid() {
                    return Ok(entry.data.clone());
                }
            }
        }

        if subgraph_url.is_empty() {
            debug!("Aave subgraph URL not configured, skipping liquidation levels");
            return Ok(Vec::new());
        }

        let query = r#"{
            users(
                where: { borrowedReservesCount_gt: 0 }
                first: 100
                orderBy: id
            ) {
                id
                reserves(where: { currentATokenBalance_gt: "0" }) {
                    currentATokenBalance
                    reserve {
                        symbol
                        price { priceInEth }
                        decimals
                        reserveLiquidationThreshold
                    }
                }
                borrows: reserves(where: { currentVariableDebt_gt: "0" }) {
                    currentVariableDebt
                    reserve {
                        symbol
                        price { priceInEth }
                        decimals
                    }
                }
            }
        }"#;

        let body = serde_json::json!({ "query": query });

        let resp = self
            .client
            .post(subgraph_url)
            .json(&body)
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .with_context(|| "Aave subgraph POST")?;

        if !resp.status().is_success() {
            warn!("Aave subgraph returned {}", resp.status());
            return Ok(Vec::new());
        }

        let result: Value = resp.json().await.with_context(|| "parse subgraph JSON")?;

        let users = match result
            .get("data")
            .and_then(|d| d.get("users"))
            .and_then(|u| u.as_array())
        {
            Some(u) => u,
            None => return Ok(Vec::new()),
        };

        // Bucket liquidation prices.
        let mut levels: HashMap<i64, (Decimal, u32)> = HashMap::new();

        for user in users {
            let hf = match estimate_user_hf(user) {
                Some(h) if h > Decimal::ZERO && h < dec!(2) => h,
                _ => continue,
            };

            let collateral_usd = estimate_user_collateral_usd(user);
            // Bucket to nearest $10.
            let bucket = (hf.to_i64().unwrap_or(0) / 10) * 10;

            let entry = levels.entry(bucket).or_insert((Decimal::ZERO, 0));
            entry.0 += collateral_usd;
            entry.1 += 1;
        }

        let mut result_list: Vec<LiquidationLevel> = levels
            .into_iter()
            .map(|(bucket, (collateral, count))| LiquidationLevel {
                price: Decimal::from(bucket),
                total_collateral_at_risk_usd: collateral,
                position_count: count,
            })
            .collect();

        // Sort by collateral at risk descending.
        result_list.sort_by(|a, b| {
            b.total_collateral_at_risk_usd
                .cmp(&a.total_collateral_at_risk_usd)
        });

        {
            let mut cache = self.cache.lock().expect("cache lock poisoned");
            cache.liquidation_levels.insert(
                cache_key,
                CacheEntry::new(result_list.clone(), TTL_LIQUIDATION_LEVELS),
            );
        }

        Ok(result_list)
    }

    // -----------------------------------------------------------------------
    // Exchange flows (Chi et al. 2024)
    // -----------------------------------------------------------------------

    /// Estimate exchange flows using Binance 24 h ticker stats as a proxy.
    ///
    /// Chi et al. (2024): net inflows to exchanges positively predict
    /// BTC/ETH returns (buying power arriving); outflows indicate capital
    /// withdrawal (bearish).
    pub async fn get_exchange_flows(
        &self,
        token: &str,
        window_minutes: u32,
    ) -> Result<ExchangeFlows> {
        let cache_key = format!("flows:{token}:{window_minutes}");

        {
            let cache = self.cache.lock().expect("cache lock poisoned");
            if let Some(entry) = cache.exchange_flows.get(&cache_key) {
                if entry.is_valid() {
                    return Ok(entry.data.clone());
                }
            }
        }

        let symbol = if token == "USDT" {
            "BTCUSDT".to_string()
        } else {
            format!("{token}USDT")
        };

        let data = self
            .binance_get(
                BINANCE_SPOT_BASE,
                "/api/v3/ticker/24hr",
                &[("symbol", &symbol)],
            )
            .await?;

        let quote_volume = data
            .get("quoteVolume")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or(Decimal::ZERO);

        let price_change_pct = data
            .get("priceChangePercent")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or(Decimal::ZERO);

        let hourly_volume = quote_volume / dec!(24);
        let window_dec = Decimal::from(window_minutes) / dec!(60);

        let (inflow, outflow) = if price_change_pct > Decimal::ZERO {
            let inf = hourly_volume * window_dec;
            (inf, inf * dec!(0.7))
        } else {
            let out = hourly_volume * window_dec;
            (out * dec!(0.7), out)
        };

        let flows = ExchangeFlows {
            inflow_usd: inflow,
            outflow_usd: outflow,
            avg_hourly_flow: hourly_volume,
            data_age_seconds: 0,
        };

        {
            let mut cache = self.cache.lock().expect("cache lock poisoned");
            cache
                .exchange_flows
                .insert(cache_key, CacheEntry::new(flows.clone(), TTL_EXCHANGE_FLOWS));
        }

        Ok(flows)
    }

    // -----------------------------------------------------------------------
    // Mempool signal (Redis — Phase 10 decoder)
    // -----------------------------------------------------------------------

    /// Retrieve the latest mempool aggregate signal from Redis.
    ///
    /// Returns `Ok(None)` if Redis is not configured or the key does not exist.
    pub async fn get_mempool_signal(&self, channel: &str) -> Result<Option<MempoolSignal>> {
        let redis = match &self.redis_client {
            Some(r) => r,
            None => return Ok(None),
        };

        let mut conn = redis
            .get_multiplexed_async_connection()
            .await
            .with_context(|| "redis connect for mempool signal")?;

        let raw: Option<String> = redis::cmd("GET")
            .arg(channel)
            .query_async(&mut conn)
            .await
            .with_context(|| "redis GET mempool signal")?;

        match raw {
            Some(json_str) => {
                let signal: MempoolSignal = serde_json::from_str(&json_str)
                    .with_context(|| "parse MempoolSignal JSON")?;
                Ok(Some(signal))
            }
            None => Ok(None),
        }
    }

    // -----------------------------------------------------------------------
    // Convenience: log returns for GARCH
    // -----------------------------------------------------------------------

    /// Compute log returns from recent OHLCV close prices.
    pub async fn get_recent_returns(
        &self,
        symbol: &str,
        periods: u32,
    ) -> Result<Vec<Decimal>> {
        let candles = self.get_ohlcv(symbol, "1h", periods + 1).await?;
        if candles.len() < 2 {
            return Ok(Vec::new());
        }

        let mut returns = Vec::with_capacity(candles.len() - 1);
        for i in 1..candles.len() {
            let prev = candles[i - 1].close.to_f64().unwrap_or(0.0);
            let cur = candles[i].close.to_f64().unwrap_or(0.0);
            if prev > 0.0 && cur > 0.0 {
                let ratio = cur / prev;
                if ratio > 0.0 {
                    if let Some(d) = Decimal::from_f64(ratio.ln()) {
                        returns.push(d);
                    }
                }
            }
        }

        Ok(returns)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Free helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a `serde_json::Value` that may be a number-as-string into `Decimal`.
fn parse_decimal_str(v: &Value) -> Decimal {
    v.as_str()
        .and_then(|s| s.parse::<Decimal>().ok())
        .or_else(|| v.as_f64().and_then(Decimal::from_f64))
        .unwrap_or(Decimal::ZERO)
}

/// Parse `[[price_str, qty_str], …]` arrays from Binance depth response.
fn parse_price_qty_array(v: Option<&Value>) -> Vec<(Decimal, Decimal)> {
    let arr = match v.and_then(|a| a.as_array()) {
        Some(a) => a,
        None => return Vec::new(),
    };

    arr.iter()
        .filter_map(|entry| {
            let pair = entry.as_array()?;
            if pair.len() < 2 {
                return None;
            }
            let price = pair[0].as_str()?.parse::<Decimal>().ok()?;
            let qty = pair[1].as_str()?.parse::<Decimal>().ok()?;
            Some((price, qty))
        })
        .collect()
}

/// Estimate a user's health factor from subgraph data.
fn estimate_user_hf(user: &Value) -> Option<Decimal> {
    let reserves = user.get("reserves")?.as_array()?;
    let borrows = user.get("borrows")?.as_array()?;
    if reserves.is_empty() || borrows.is_empty() {
        return None;
    }

    let mut total_collateral_eth = Decimal::ZERO;
    let mut total_debt_eth = Decimal::ZERO;
    let mut weighted_lt = Decimal::ZERO;

    for r in reserves {
        let reserve = r.get("reserve")?;
        let balance: Decimal = r
            .get("currentATokenBalance")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(Decimal::ZERO);
        let decimals: u32 = reserve
            .get("decimals")
            .and_then(|v| v.as_u64())
            .unwrap_or(18) as u32;
        let price_eth: Decimal = reserve
            .get("price")
            .and_then(|p| p.get("priceInEth"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(Decimal::ZERO);
        let lt: Decimal = reserve
            .get("reserveLiquidationThreshold")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(Decimal::ZERO);

        let divisor = Decimal::from(10u64.pow(decimals));
        let value_eth = balance / divisor * price_eth;
        total_collateral_eth += value_eth;
        weighted_lt += value_eth * lt / dec!(10000);
    }

    for b in borrows {
        let reserve = b.get("reserve")?;
        let debt: Decimal = b
            .get("currentVariableDebt")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(Decimal::ZERO);
        let decimals: u32 = reserve
            .get("decimals")
            .and_then(|v| v.as_u64())
            .unwrap_or(18) as u32;
        let price_eth: Decimal = reserve
            .get("price")
            .and_then(|p| p.get("priceInEth"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(Decimal::ZERO);

        let divisor = Decimal::from(10u64.pow(decimals));
        let value_eth = debt / divisor * price_eth;
        total_debt_eth += value_eth;
    }

    if total_collateral_eth <= Decimal::ZERO || total_debt_eth <= Decimal::ZERO {
        return None;
    }

    let avg_lt = weighted_lt / total_collateral_eth;
    let hf = (total_collateral_eth * avg_lt) / total_debt_eth;
    Some(hf)
}

/// Estimate total collateral USD from subgraph data (1 ETH ≈ $2 000).
fn estimate_user_collateral_usd(user: &Value) -> Decimal {
    let reserves = match user.get("reserves").and_then(|v| v.as_array()) {
        Some(r) => r,
        None => return Decimal::ZERO,
    };

    let mut total = Decimal::ZERO;
    for r in reserves {
        let reserve = match r.get("reserve") {
            Some(rv) => rv,
            None => continue,
        };
        let balance: Decimal = r
            .get("currentATokenBalance")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(Decimal::ZERO);
        let decimals: u32 = reserve
            .get("decimals")
            .and_then(|v| v.as_u64())
            .unwrap_or(18) as u32;
        let price_eth: Decimal = reserve
            .get("price")
            .and_then(|p| p.get("priceInEth"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(Decimal::ZERO);

        let divisor = Decimal::from(10u64.pow(decimals));
        total += balance / divisor * price_eth * dec!(2000);
    }

    total
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_decimal_str_string() {
        let v = serde_json::json!("123.456");
        assert_eq!(parse_decimal_str(&v), "123.456".parse::<Decimal>().unwrap());
    }

    #[test]
    fn test_parse_decimal_str_number() {
        let v = serde_json::json!(42.5);
        let d = parse_decimal_str(&v);
        assert!(d > Decimal::ZERO);
    }

    #[test]
    fn test_parse_decimal_str_null() {
        let v = serde_json::json!(null);
        assert_eq!(parse_decimal_str(&v), Decimal::ZERO);
    }

    #[test]
    fn test_parse_price_qty_array() {
        let v = serde_json::json!([["100.5", "2.0"], ["99.0", "3.5"]]);
        let result = parse_price_qty_array(Some(&v));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "100.5".parse::<Decimal>().unwrap());
        assert_eq!(result[1].1, "3.5".parse::<Decimal>().unwrap());
    }

    #[test]
    fn test_parse_price_qty_array_none() {
        assert!(parse_price_qty_array(None).is_empty());
    }

    #[test]
    fn test_parse_ohlcv_response() {
        // Simulate Binance klines response parsing.
        let kline = serde_json::json!([
            1700000000000i64, "600.0", "605.0", "595.0", "602.0", "1000.0",
            1700003600000i64, "500000.0", 100, "600.0", "50.0", "0"
        ]);
        let items = kline.as_array().unwrap();
        let ts = items[0].as_i64().unwrap() / 1000;
        let open = parse_decimal_str(&items[1]);
        let high = parse_decimal_str(&items[2]);

        assert_eq!(ts, 1700000000);
        assert_eq!(open, "600.0".parse::<Decimal>().unwrap());
        assert_eq!(high, "605.0".parse::<Decimal>().unwrap());
    }

    #[test]
    fn test_parse_aggtrades_response() {
        let trade_json = serde_json::json!({
            "p": "600.50",
            "q": "2.5",
            "T": 1700000000000i64,
            "m": true
        });
        let price = trade_json
            .get("p")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap();
        let is_buyer_maker = trade_json.get("m").and_then(|v| v.as_bool()).unwrap();

        assert_eq!(price, "600.50".parse::<Decimal>().unwrap());
        assert!(is_buyer_maker);
    }
}
