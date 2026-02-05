//! DEX aggregator client — parallel fan-out with best-quote selection.
//!
//! Ported from Python `execution/aggregator_client.py`. Queries 1inch,
//! OpenOcean, and ParaSwap in parallel, selects the best quote by output
//! amount, validates router addresses against a whitelist, checks DEX-Oracle
//! price divergence, and caches results with a configurable TTL.
//!
//! Academic basis: Angeris et al. (2022) proved CFMM routing is convex;
//! Diamandis et al. (FC 2023) showed efficient routing scales linearly.

use std::collections::HashSet;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use alloy::primitives::{Address, U256};
use anyhow::{Context, Result};
use lru::LruCache;
use reqwest::Client;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tracing::{debug, warn};

use crate::config::{AggregatorConfig, AggregatorProviderConfig};
use crate::errors::BotError;
use crate::types::SwapQuote;

/// Cache entry: quote + insertion time for TTL expiry.
struct CacheEntry {
    quote: SwapQuote,
    inserted_at: Instant,
}

/// Per-provider rate-limit state.
struct RateLimitState {
    /// Minimum interval between requests.
    interval: Duration,
    /// When the last request was sent.
    last_request: Option<Instant>,
}

/// DEX aggregator fan-out client.
///
/// Queries multiple DEX aggregator APIs in parallel, selects the best quote,
/// validates the router address against a whitelist, and enforces per-provider
/// rate limits. Results are cached with a configurable TTL.
pub struct AggregatorClient {
    http: Client,
    config: AggregatorConfig,
    approved_routers: HashSet<String>,
    cache: Mutex<LruCache<String, CacheEntry>>,
    cache_ttl: Duration,
    rate_limits: Mutex<Vec<(String, RateLimitState)>>,
}

impl AggregatorClient {
    /// Build from config. Collects all approved routers into a single set.
    pub fn new(config: &AggregatorConfig, cache_ttl_seconds: u64) -> Self {
        let mut approved = HashSet::new();
        for provider in &config.providers {
            for router in &provider.approved_routers {
                approved.insert(router.to_lowercase());
            }
        }

        let rate_limits: Vec<(String, RateLimitState)> = config
            .providers
            .iter()
            .map(|p| {
                let rps = p.rate_limit_rps.max(1);
                (
                    p.name.clone(),
                    RateLimitState {
                        interval: Duration::from_secs_f64(1.0 / rps as f64),
                        last_request: None,
                    },
                )
            })
            .collect();

        Self {
            http: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
            config: config.clone(),
            approved_routers: approved,
            cache: Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(64).expect("cache size 64 is non-zero"),
            )),
            cache_ttl: Duration::from_secs(cache_ttl_seconds),
            rate_limits: Mutex::new(rate_limits),
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Query all enabled providers in parallel, return the best valid quote.
    ///
    /// Checks the cache first. On cache miss, fans out to all enabled providers,
    /// filters by router whitelist, and returns the quote with the highest
    /// `to_amount`. Caches the result on success.
    pub async fn get_best_quote(
        &self,
        from_token: Address,
        to_token: Address,
        amount: U256,
        max_slippage_bps: u32,
    ) -> Result<SwapQuote, BotError> {
        // Check cache
        let cache_key = Self::cache_key(from_token, to_token, amount);
        if let Some(cached) = self.get_cached(&cache_key) {
            debug!(provider = %cached.provider, "returning cached quote");
            return Ok(cached);
        }

        // Fan out to all enabled providers
        let enabled: Vec<&AggregatorProviderConfig> = self
            .config
            .providers
            .iter()
            .filter(|p| p.enabled)
            .collect();

        if enabled.is_empty() {
            return Err(BotError::AggregatorUnavailable);
        }

        let mut futures = Vec::with_capacity(enabled.len());
        for provider in &enabled {
            futures.push(self.query_provider(provider, from_token, to_token, amount, max_slippage_bps));
        }

        let results = futures::future::join_all(futures).await;

        // Collect successful quotes with valid routers
        let mut valid: Vec<SwapQuote> = results
            .into_iter()
            .filter_map(|r| match r {
                Ok(q) => {
                    if self.validate_router(&q.router_address) {
                        Some(q)
                    } else {
                        warn!(
                            provider = %q.provider,
                            router = %q.router_address,
                            "quote rejected: router not in whitelist"
                        );
                        None
                    }
                }
                Err(e) => {
                    warn!(error = %e, "provider query failed");
                    None
                }
            })
            .collect();

        if valid.is_empty() {
            return Err(BotError::AggregatorUnavailable);
        }

        // Select best quote (highest to_amount)
        valid.sort_by(|a, b| b.to_amount.cmp(&a.to_amount));
        let best = valid.into_iter().next().expect("valid is non-empty after is_empty check");

        // Cache the result
        self.set_cached(cache_key, best.clone());
        debug!(
            provider = %best.provider,
            to_amount = %best.to_amount,
            "selected best quote"
        );

        Ok(best)
    }

    /// Check DEX-Oracle price divergence.
    ///
    /// Returns `Ok(())` if within tolerance, `Err(PriceDivergence)` otherwise.
    /// Per Deng et al. (ICSE 2024): ad-hoc oracle controls are insufficient.
    pub fn check_dex_oracle_divergence(
        &self,
        quote: &SwapQuote,
        oracle_from_usd: Decimal,
        oracle_to_usd: Decimal,
        from_decimals: u8,
        to_decimals: u8,
        max_divergence_pct: Decimal,
    ) -> Result<(), BotError> {
        if oracle_from_usd <= Decimal::ZERO || oracle_to_usd <= Decimal::ZERO {
            return Err(BotError::PriceDivergence {
                divergence_pct: 100.0,
                max_pct: max_divergence_pct.to_f64().unwrap_or(1.0),
            });
        }

        // Oracle exchange rate: how many to_tokens per from_token
        let oracle_rate = oracle_from_usd / oracle_to_usd;

        // DEX exchange rate: normalize by decimals
        let from_divisor = Decimal::from(10u64.pow(from_decimals as u32));
        let to_divisor = Decimal::from(10u64.pow(to_decimals as u32));
        let from_human = quote.from_amount / from_divisor;
        let to_human = quote.to_amount / to_divisor;

        if from_human <= Decimal::ZERO {
            return Err(BotError::PriceDivergence {
                divergence_pct: 100.0,
                max_pct: max_divergence_pct.to_f64().unwrap_or(1.0),
            });
        }

        let dex_rate = to_human / from_human;
        let divergence = ((dex_rate - oracle_rate).abs() / oracle_rate) * dec!(100);

        if divergence > max_divergence_pct {
            return Err(BotError::PriceDivergence {
                divergence_pct: divergence.to_f64().unwrap_or(0.0),
                max_pct: max_divergence_pct.to_f64().unwrap_or(1.0),
            });
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Provider dispatch
    // -----------------------------------------------------------------------

    async fn query_provider(
        &self,
        provider: &AggregatorProviderConfig,
        from_token: Address,
        to_token: Address,
        amount: U256,
        max_slippage_bps: u32,
    ) -> Result<SwapQuote> {
        self.enforce_rate_limit(&provider.name).await;

        match provider.name.as_str() {
            "1inch" => self.query_1inch(provider, from_token, to_token, amount, max_slippage_bps).await,
            "openocean" => self.query_openocean(provider, from_token, to_token, amount, max_slippage_bps).await,
            "paraswap" => self.query_paraswap(provider, from_token, to_token, amount, max_slippage_bps).await,
            other => anyhow::bail!("unknown aggregator provider: {other}"),
        }
    }

    // -----------------------------------------------------------------------
    // 1inch Classic API v6
    // -----------------------------------------------------------------------

    async fn query_1inch(
        &self,
        provider: &AggregatorProviderConfig,
        from_token: Address,
        to_token: Address,
        amount: U256,
        max_slippage_bps: u32,
    ) -> Result<SwapQuote> {
        let url = format!("{}/swap", provider.base_url);
        let slippage_pct = Decimal::from(max_slippage_bps) / dec!(100);

        let mut req = self
            .http
            .get(&url)
            .timeout(Duration::from_secs(provider.timeout_seconds))
            .query(&[
                ("src", format!("{from_token:#x}")),
                ("dst", format!("{to_token:#x}")),
                ("amount", amount.to_string()),
                ("from", "0x0000000000000000000000000000000000000000".into()),
                ("slippage", slippage_pct.to_string()),
                ("disableEstimate", "true".into()),
            ]);

        // Add API key if configured
        let api_key = if !provider.api_key_env.is_empty() {
            std::env::var(&provider.api_key_env).ok()
        } else {
            None
        };
        if let Some(key) = &api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let resp: serde_json::Value = req
            .send()
            .await
            .context("1inch HTTP request failed")?
            .error_for_status()
            .context("1inch HTTP error status")?
            .json()
            .await
            .context("1inch JSON parse failed")?;

        let dst_amount_str = resp["dstAmount"]
            .as_str()
            .context("missing dstAmount")?;
        let to_amount: Decimal = dst_amount_str.parse().context("invalid dstAmount")?;
        let to_amount_min = to_amount * Decimal::from(10_000 - max_slippage_bps)
            / dec!(10_000);

        let calldata_hex = resp["tx"]["data"]
            .as_str()
            .context("missing tx.data")?;
        let calldata = hex::decode(calldata_hex.strip_prefix("0x").unwrap_or(calldata_hex))
            .context("invalid tx.data hex")?;

        let router = resp["tx"]["to"]
            .as_str()
            .context("missing tx.to")?
            .to_string();

        let gas = resp["tx"]["gas"]
            .as_u64()
            .or_else(|| resp["tx"]["gas"].as_str().and_then(|s| s.parse().ok()))
            .unwrap_or(300_000);

        Ok(SwapQuote {
            provider: "1inch".into(),
            from_token: format!("{from_token:#x}"),
            to_token: format!("{to_token:#x}"),
            from_amount: Decimal::from_str_exact(&amount.to_string())
                .unwrap_or_default(),
            to_amount,
            to_amount_min,
            calldata,
            router_address: router,
            gas_estimate: gas,
            price_impact: Decimal::ZERO,
        })
    }

    // -----------------------------------------------------------------------
    // OpenOcean API v4
    // -----------------------------------------------------------------------

    async fn query_openocean(
        &self,
        provider: &AggregatorProviderConfig,
        from_token: Address,
        to_token: Address,
        amount: U256,
        max_slippage_bps: u32,
    ) -> Result<SwapQuote> {
        let url = format!("{}/swap_quote", provider.base_url);
        let slippage_pct = Decimal::from(max_slippage_bps) / dec!(100);

        let resp: serde_json::Value = self
            .http
            .get(&url)
            .timeout(Duration::from_secs(provider.timeout_seconds))
            .query(&[
                ("inTokenAddress", format!("{from_token:#x}")),
                ("outTokenAddress", format!("{to_token:#x}")),
                ("amount", amount.to_string()),
                ("gasPrice", "5".into()),
                ("slippage", slippage_pct.to_string()),
            ])
            .send()
            .await
            .context("OpenOcean HTTP request failed")?
            .error_for_status()
            .context("OpenOcean HTTP error status")?
            .json()
            .await
            .context("OpenOcean JSON parse failed")?;

        let data = &resp["data"];
        let out_amount_str = data["outAmount"]
            .as_str()
            .or_else(|| data["outAmount"].as_u64().map(|_| ""))
            .context("missing data.outAmount")?;
        let to_amount: Decimal = if out_amount_str.is_empty() {
            Decimal::from(data["outAmount"].as_u64().context("invalid outAmount")?)
        } else {
            out_amount_str.parse().context("invalid outAmount")?
        };

        let min_out_str = data["minOutAmount"].as_str().unwrap_or("0");
        let to_amount_min: Decimal = min_out_str.parse().unwrap_or_else(|_| {
            to_amount * Decimal::from(10_000 - max_slippage_bps) / dec!(10_000)
        });
        // Fall back to calculated if minOutAmount is zero or negative
        let to_amount_min = if to_amount_min <= Decimal::ZERO {
            to_amount * Decimal::from(10_000 - max_slippage_bps) / dec!(10_000)
        } else {
            to_amount_min
        };

        let calldata_hex = data["data"]
            .as_str()
            .context("missing data.data")?;
        let calldata = hex::decode(calldata_hex.strip_prefix("0x").unwrap_or(calldata_hex))
            .context("invalid data.data hex")?;

        let router = data["to"]
            .as_str()
            .context("missing data.to")?
            .to_string();

        let gas = data["estimatedGas"]
            .as_u64()
            .or_else(|| data["estimatedGas"].as_str().and_then(|s| s.parse().ok()))
            .unwrap_or(300_000);

        Ok(SwapQuote {
            provider: "openocean".into(),
            from_token: format!("{from_token:#x}"),
            to_token: format!("{to_token:#x}"),
            from_amount: Decimal::from_str_exact(&amount.to_string())
                .unwrap_or_default(),
            to_amount,
            to_amount_min,
            calldata,
            router_address: router,
            gas_estimate: gas,
            price_impact: Decimal::ZERO,
        })
    }

    // -----------------------------------------------------------------------
    // ParaSwap API v5 (two-step: prices + build-tx)
    // -----------------------------------------------------------------------

    async fn query_paraswap(
        &self,
        provider: &AggregatorProviderConfig,
        from_token: Address,
        to_token: Address,
        amount: U256,
        max_slippage_bps: u32,
    ) -> Result<SwapQuote> {
        // Step 1: Get price route
        let prices_url = format!("{}/prices", provider.base_url);
        let price_resp: serde_json::Value = self
            .http
            .get(&prices_url)
            .timeout(Duration::from_secs(provider.timeout_seconds))
            .query(&[
                ("srcToken", format!("{from_token:#x}")),
                ("destToken", format!("{to_token:#x}")),
                ("amount", amount.to_string()),
                ("srcDecimals", "18".into()),
                ("destDecimals", "18".into()),
                ("side", "SELL".into()),
                ("network", "56".into()),
            ])
            .send()
            .await
            .context("ParaSwap prices HTTP request failed")?
            .error_for_status()
            .context("ParaSwap prices HTTP error status")?
            .json()
            .await
            .context("ParaSwap prices JSON parse failed")?;

        let price_route = &price_resp["priceRoute"];
        let dest_amount_str = price_route["destAmount"]
            .as_str()
            .context("missing priceRoute.destAmount")?;
        let dest_amount: Decimal = dest_amount_str.parse().context("invalid destAmount")?;
        let gas_cost = price_route["gasCost"]
            .as_u64()
            .or_else(|| price_route["gasCost"].as_str().and_then(|s| s.parse().ok()))
            .unwrap_or(300_000);

        // Step 2: Build transaction
        let tx_url = format!("{}/transactions/56", provider.base_url);
        let tx_body = serde_json::json!({
            "srcToken": format!("{from_token:#x}"),
            "destToken": format!("{to_token:#x}"),
            "srcAmount": amount.to_string(),
            "destAmount": dest_amount_str,
            "priceRoute": price_route,
            "userAddress": "0x0000000000000000000000000000000000000000",
            "slippage": max_slippage_bps,
        });

        let tx_resp: serde_json::Value = self
            .http
            .post(&tx_url)
            .timeout(Duration::from_secs(provider.timeout_seconds))
            .json(&tx_body)
            .send()
            .await
            .context("ParaSwap tx-build HTTP request failed")?
            .error_for_status()
            .context("ParaSwap tx-build HTTP error status")?
            .json()
            .await
            .context("ParaSwap tx-build JSON parse failed")?;

        let calldata_hex = tx_resp["data"]
            .as_str()
            .context("missing data in tx response")?;
        let calldata = hex::decode(calldata_hex.strip_prefix("0x").unwrap_or(calldata_hex))
            .context("invalid tx data hex")?;

        let router = tx_resp["to"]
            .as_str()
            .context("missing to in tx response")?
            .to_string();

        let gas = tx_resp["gas"]
            .as_u64()
            .or_else(|| tx_resp["gas"].as_str().and_then(|s| s.parse().ok()))
            .unwrap_or(gas_cost);

        let to_amount_min = dest_amount * Decimal::from(10_000 - max_slippage_bps)
            / dec!(10_000);

        Ok(SwapQuote {
            provider: "paraswap".into(),
            from_token: format!("{from_token:#x}"),
            to_token: format!("{to_token:#x}"),
            from_amount: Decimal::from_str_exact(&amount.to_string())
                .unwrap_or_default(),
            to_amount: dest_amount,
            to_amount_min,
            calldata,
            router_address: router,
            gas_estimate: gas,
            price_impact: Decimal::ZERO,
        })
    }

    // -----------------------------------------------------------------------
    // Rate limiting
    // -----------------------------------------------------------------------

    async fn enforce_rate_limit(&self, provider_name: &str) {
        let sleep_duration = {
            let mut limits = self.rate_limits.lock().expect("rate_limits lock poisoned");
            let entry = limits.iter_mut().find(|(name, _)| name == provider_name);
            if let Some((_, state)) = entry {
                if let Some(last) = state.last_request {
                    let elapsed = last.elapsed();
                    if elapsed < state.interval {
                        Some(state.interval - elapsed)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some(d) = sleep_duration {
            tokio::time::sleep(d).await;
        }

        // Update last request time
        if let Ok(mut limits) = self.rate_limits.lock() {
            if let Some((_, state)) = limits.iter_mut().find(|(name, _)| name == provider_name) {
                state.last_request = Some(Instant::now());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Router validation
    // -----------------------------------------------------------------------

    fn validate_router(&self, router_address: &str) -> bool {
        if self.approved_routers.is_empty() {
            return true;
        }
        self.approved_routers.contains(&router_address.to_lowercase())
    }

    // -----------------------------------------------------------------------
    // Quote cache
    // -----------------------------------------------------------------------

    fn cache_key(from_token: Address, to_token: Address, amount: U256) -> String {
        format!("{:#x}:{:#x}:{}", from_token, to_token, amount)
    }

    fn get_cached(&self, key: &str) -> Option<SwapQuote> {
        let mut cache = self.cache.lock().ok()?;
        let entry = cache.get(key)?;
        if entry.inserted_at.elapsed() > self.cache_ttl {
            cache.pop(key);
            return None;
        }
        Some(entry.quote.clone())
    }

    fn set_cached(&self, key: String, quote: SwapQuote) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.put(
                key,
                CacheEntry {
                    quote,
                    inserted_at: Instant::now(),
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy::primitives::address;

    fn test_config() -> AggregatorConfig {
        AggregatorConfig {
            providers: vec![],
            max_slippage_bps: 50,
            max_price_impact_percent: dec!(1.0),
        }
    }

    fn make_quote(provider: &str, to_amount: Decimal, router: &str) -> SwapQuote {
        SwapQuote {
            provider: provider.into(),
            from_token: "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c".into(),
            to_token: "0x55d398326f99059ff775485246999027b3197955".into(),
            from_amount: dec!(1_000_000_000_000_000_000), // 1e18
            to_amount,
            to_amount_min: to_amount * dec!(0.995),
            calldata: vec![0x01, 0x02],
            router_address: router.into(),
            gas_estimate: 300_000,
            price_impact: Decimal::ZERO,
        }
    }

    // -----------------------------------------------------------------------
    // Router validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_router_validation_pass() {
        let mut config = test_config();
        config.providers.push(AggregatorProviderConfig {
            name: "1inch".into(),
            enabled: true,
            priority: 1,
            base_url: String::new(),
            api_key_env: String::new(),
            rate_limit_rps: 1,
            timeout_seconds: 5,
            approved_routers: vec!["0x111111125421cA6dc452d289314280a0f8842A65".into()],
            params: Default::default(),
        });
        let client = AggregatorClient::new(&config, 10);
        assert!(client.validate_router("0x111111125421ca6dc452d289314280a0f8842a65"));
    }

    #[test]
    fn test_router_validation_case_insensitive() {
        let mut config = test_config();
        config.providers.push(AggregatorProviderConfig {
            name: "1inch".into(),
            enabled: true,
            priority: 1,
            base_url: String::new(),
            api_key_env: String::new(),
            rate_limit_rps: 1,
            timeout_seconds: 5,
            approved_routers: vec!["0xABCDEF".into()],
            params: Default::default(),
        });
        let client = AggregatorClient::new(&config, 10);
        assert!(client.validate_router("0xabcdef"));
        assert!(client.validate_router("0xABCDEF"));
    }

    #[test]
    fn test_router_validation_reject_unknown() {
        let mut config = test_config();
        config.providers.push(AggregatorProviderConfig {
            name: "1inch".into(),
            enabled: true,
            priority: 1,
            base_url: String::new(),
            api_key_env: String::new(),
            rate_limit_rps: 1,
            timeout_seconds: 5,
            approved_routers: vec!["0x111111125421cA6dc452d289314280a0f8842A65".into()],
            params: Default::default(),
        });
        let client = AggregatorClient::new(&config, 10);
        assert!(!client.validate_router("0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"));
    }

    // -----------------------------------------------------------------------
    // Divergence check
    // -----------------------------------------------------------------------

    #[test]
    fn test_divergence_within_threshold() {
        let client = AggregatorClient::new(&test_config(), 10);
        let quote = make_quote("test", dec!(600_000_000_000_000_000_000), "0x1234");
        // from_amount=1e18, to_amount=600e18 → DEX rate = 600 (WBNB→USDT)
        // Oracle: WBNB=$600, USDT=$1 → oracle_rate = 600
        let result = client.check_dex_oracle_divergence(
            &quote,
            dec!(600),  // oracle_from_usd
            dec!(1),    // oracle_to_usd
            18,         // from_decimals
            18,         // to_decimals
            dec!(1.0),  // max divergence %
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_divergence_exceeds_threshold() {
        let client = AggregatorClient::new(&test_config(), 10);
        // DEX gives 580 USDT per WBNB, oracle says 600
        let quote = make_quote("test", dec!(580_000_000_000_000_000_000), "0x1234");
        let result = client.check_dex_oracle_divergence(
            &quote,
            dec!(600),
            dec!(1),
            18,
            18,
            dec!(1.0), // 1% max
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("divergence"));
    }

    #[test]
    fn test_divergence_zero_oracle_price() {
        let client = AggregatorClient::new(&test_config(), 10);
        let quote = make_quote("test", dec!(600_000_000_000_000_000_000), "0x1234");
        let result = client.check_dex_oracle_divergence(
            &quote,
            Decimal::ZERO, // zero oracle price
            dec!(1),
            18,
            18,
            dec!(1.0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_divergence_different_decimals() {
        let client = AggregatorClient::new(&test_config(), 10);
        // from: 18 decimals (1e18), to: 6 decimals
        // from_amount = 1e18 (1 token), to_amount = 600e6 (600 USDT in 6-decimal)
        // from_human = 1, to_human = 600, dex_rate = 600
        let quote = SwapQuote {
            provider: "test".into(),
            from_token: "0xfrom".into(),
            to_token: "0xto".into(),
            from_amount: dec!(1_000_000_000_000_000_000), // 1e18
            to_amount: dec!(600_000_000),                   // 600e6
            to_amount_min: dec!(597_000_000),
            calldata: vec![],
            router_address: "0x1234".into(),
            gas_estimate: 300_000,
            price_impact: Decimal::ZERO,
        };
        let result = client.check_dex_oracle_divergence(
            &quote,
            dec!(600),
            dec!(1),
            18, // from_decimals
            6,  // to_decimals
            dec!(1.0),
        );
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // Quote cache
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_hit() {
        let client = AggregatorClient::new(&test_config(), 10);
        let from = address!("bb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c");
        let to = address!("55d398326f99059fF775485246999027B3197955");
        let amount = U256::from(1_000_000_000_000_000_000u128);

        let key = AggregatorClient::cache_key(from, to, amount);
        let quote = make_quote("1inch", dec!(600), "0x1234");
        client.set_cached(key.clone(), quote.clone());

        let cached = client.get_cached(&key);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().provider, "1inch");
    }

    #[test]
    fn test_cache_miss_after_expiry() {
        let client = AggregatorClient::new(&test_config(), 0); // 0s TTL = instant expiry
        let from = address!("bb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c");
        let to = address!("55d398326f99059fF775485246999027B3197955");
        let amount = U256::from(1_000_000_000_000_000_000u128);

        let key = AggregatorClient::cache_key(from, to, amount);
        let quote = make_quote("1inch", dec!(600), "0x1234");
        client.set_cached(key.clone(), quote);

        // Sleep 1ms to ensure TTL expired
        std::thread::sleep(Duration::from_millis(1));
        assert!(client.get_cached(&key).is_none());
    }

    #[test]
    fn test_cache_different_keys() {
        let client = AggregatorClient::new(&test_config(), 10);
        let from = address!("bb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c");
        let to = address!("55d398326f99059fF775485246999027B3197955");

        let key1 = AggregatorClient::cache_key(from, to, U256::from(1u64));
        let key2 = AggregatorClient::cache_key(from, to, U256::from(2u64));

        client.set_cached(key1.clone(), make_quote("a", dec!(100), "0x1"));
        client.set_cached(key2.clone(), make_quote("b", dec!(200), "0x2"));

        assert_eq!(client.get_cached(&key1).unwrap().provider, "a");
        assert_eq!(client.get_cached(&key2).unwrap().provider, "b");
    }

    // -----------------------------------------------------------------------
    // get_best_quote — no providers enabled
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_no_enabled_providers_returns_unavailable() {
        let config = test_config(); // empty providers
        let client = AggregatorClient::new(&config, 10);
        let result = client
            .get_best_quote(
                address!("bb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"),
                address!("55d398326f99059fF775485246999027B3197955"),
                U256::from(1_000_000_000_000_000_000u128),
                50,
            )
            .await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BotError::AggregatorUnavailable));
    }
}
