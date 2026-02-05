//! Background task that maintains a cache of token prices from Binance.
//!
//! Refreshes every 60 seconds to keep USD estimations reasonably accurate
//! without hitting rate limits.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use rust_decimal::Decimal;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

/// Thread-safe cache of token prices (e.g. "BNBUSDT" → 620.50).
#[derive(Debug, Clone)]
pub struct PriceCache {
    prices: Arc<Mutex<HashMap<String, Decimal>>>,
}

impl PriceCache {
    /// Create a new empty price cache.
    pub fn new() -> Self {
        Self {
            prices: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get the cached price for a symbol (e.g. "BNBUSDT").
    pub fn get_price(&self, symbol: &str) -> Option<Decimal> {
        self.prices
            .lock()
            .expect("price cache lock poisoned")
            .get(symbol)
            .copied()
    }

    /// Get a snapshot of all cached prices.
    pub fn get_all(&self) -> HashMap<String, Decimal> {
        self.prices
            .lock()
            .expect("price cache lock poisoned")
            .clone()
    }

    /// Update the cache with new prices.
    fn update(&self, new_prices: HashMap<String, Decimal>) {
        let mut cache = self.prices.lock().expect("price cache lock poisoned");
        for (symbol, price) in new_prices {
            cache.insert(symbol, price);
        }
    }
}

/// Symbols to fetch from Binance.
const SYMBOLS: &[&str] = &["BNBUSDT", "BTCUSDT", "ETHUSDT"];

/// Background task that refreshes token prices from Binance every 60 seconds.
pub async fn run_price_updater(
    cache: PriceCache,
    shutdown: CancellationToken,
) {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("failed to build HTTP client");

    // Fetch immediately on startup, then every 60s.
    loop {
        match fetch_prices(&client).await {
            Ok(prices) => {
                debug!(
                    count = prices.len(),
                    "updated price cache"
                );
                cache.update(prices);
            }
            Err(e) => {
                warn!(error = %e, "failed to fetch prices from Binance");
            }
        }

        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_secs(60)) => {}
            _ = shutdown.cancelled() => {
                debug!("price updater shutting down");
                return;
            }
        }
    }
}

/// Fetch current prices from Binance ticker API.
async fn fetch_prices(client: &reqwest::Client) -> Result<HashMap<String, Decimal>> {
    // Build the symbols query parameter: ["BNBUSDT","BTCUSDT","ETHUSDT"]
    let symbols_json: String = format!(
        "[{}]",
        SYMBOLS
            .iter()
            .map(|s| format!("\"{s}\""))
            .collect::<Vec<_>>()
            .join(",")
    );

    let resp = client
        .get("https://api.binance.com/api/v3/ticker/price")
        .query(&[("symbols", &symbols_json)])
        .send()
        .await?;

    let body: serde_json::Value = resp.json().await?;

    let mut prices = HashMap::new();

    if let Some(arr) = body.as_array() {
        for item in arr {
            let symbol = item["symbol"].as_str().unwrap_or_default();
            let price_str = item["price"].as_str().unwrap_or("0");
            if let Ok(price) = price_str.parse::<Decimal>() {
                prices.insert(symbol.to_string(), price);
            }
        }
    }

    Ok(prices)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_price_cache_update_and_get() {
        let cache = PriceCache::new();
        assert!(cache.get_price("BNBUSDT").is_none());

        let mut prices = HashMap::new();
        prices.insert("BNBUSDT".to_string(), dec!(620.50));
        prices.insert("BTCUSDT".to_string(), dec!(98000));
        cache.update(prices);

        assert_eq!(cache.get_price("BNBUSDT"), Some(dec!(620.50)));
        assert_eq!(cache.get_price("BTCUSDT"), Some(dec!(98000)));
        assert!(cache.get_price("DOGEUSD").is_none());
    }

    #[test]
    fn test_get_all_snapshot() {
        let cache = PriceCache::new();
        let mut prices = HashMap::new();
        prices.insert("BNBUSDT".to_string(), dec!(600));
        cache.update(prices);

        let all = cache.get_all();
        assert_eq!(all.len(), 1);
        assert_eq!(all["BNBUSDT"], dec!(600));
    }
}
