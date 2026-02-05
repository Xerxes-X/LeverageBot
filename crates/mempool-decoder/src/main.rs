//! Mempool Decoder — standalone BSC pending transaction decoder and aggregator.
//!
//! Subscribes to BSC WebSocket `newPendingTransactions`, decodes DEX swap
//! calldata from 12 monitored routers (26 selectors), classifies buy/sell
//! direction, aggregates rolling statistics, and publishes to Redis for
//! the LeverageBot to consume.

mod aggregator;
mod classifier;
mod config;
mod constants;
mod decoder;
mod poison;
mod price_cache;
mod publisher;
mod types;
mod websocket;

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

use crate::aggregator::SwapAggregator;
use crate::classifier::{classify_direction, estimate_usd_value, get_volatile_token, volatile_symbol};
use crate::config::DecoderConfig;
use crate::poison::compute_poison_score;
use crate::price_cache::PriceCache;
use crate::types::{DecodedSwap, SwapDirection};

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file (ignore if missing).
    let _ = dotenvy::dotenv();

    // Initialize tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(true)
        .init();

    // Load configuration.
    let config = DecoderConfig::from_env().context("failed to load decoder config")?;

    info!(
        ws_url = %config.websocket_url,
        redis_channel = %config.aggregate_channel,
        publish_interval = config.publish_interval_seconds,
        dedup_size = config.dedup_cache_size,
        volatile_tokens = ?config.monitored_volatile,
        "mempool decoder starting"
    );

    let shutdown = CancellationToken::new();

    // -----------------------------------------------------------------------
    // Components
    // -----------------------------------------------------------------------

    let price_cache = PriceCache::new();
    let agg = Arc::new(Mutex::new(SwapAggregator::new(
        config.whale_threshold_usd,
        &config.monitored_volatile,
    )));

    // Channel for decoded swaps: websocket → processing loop.
    let (swap_tx, mut swap_rx) = mpsc::channel::<DecodedSwap>(1024);

    // -----------------------------------------------------------------------
    // Spawn background tasks
    // -----------------------------------------------------------------------

    // 1. Price updater — fetches Binance prices every 60s.
    let price_cache_clone = price_cache.clone();
    let shutdown_clone = shutdown.clone();
    let price_handle = tokio::spawn(async move {
        price_cache::run_price_updater(price_cache_clone, shutdown_clone).await;
    });

    // 2. WebSocket listener — connects, subscribes, decodes, sends to channel.
    let ws_url = config.websocket_url.clone();
    let dedup_size = config.dedup_cache_size;
    let reconnect_delay = config.reconnect_delay_seconds;
    let max_reconnect = config.max_reconnect_attempts;
    let shutdown_clone = shutdown.clone();
    let ws_handle = tokio::spawn(async move {
        websocket::run_websocket(
            &ws_url,
            swap_tx,
            dedup_size,
            reconnect_delay,
            max_reconnect,
            shutdown_clone,
        )
        .await;
    });

    // 3. Redis publisher — publishes aggregate signal every N seconds.
    let agg_clone = agg.clone();
    let redis_url = config.redis_url.clone();
    let channel = config.aggregate_channel.clone();
    let interval = config.publish_interval_seconds;
    let shutdown_clone = shutdown.clone();
    let pub_handle = tokio::spawn(async move {
        publisher::run_publisher(agg_clone, &redis_url, &channel, interval, shutdown_clone).await;
    });

    // 4. Processing loop — classify, estimate USD, score poison, record.
    let agg_clone = agg.clone();
    let poison_threshold = config.poison_threshold;
    let shutdown_clone = shutdown.clone();
    let process_handle = tokio::spawn(async move {
        run_processing_loop(
            &mut swap_rx,
            &price_cache,
            &agg_clone,
            poison_threshold,
            &shutdown_clone,
        )
        .await;
    });

    info!("all tasks running — press Ctrl+C to shutdown");

    // -----------------------------------------------------------------------
    // Wait for shutdown
    // -----------------------------------------------------------------------

    tokio::signal::ctrl_c()
        .await
        .context("failed to listen for Ctrl+C")?;

    info!("shutdown signal received, stopping gracefully...");
    shutdown.cancel();

    // Wait for all tasks to finish.
    let (price_res, ws_res, pub_res, proc_res) =
        tokio::join!(price_handle, ws_handle, pub_handle, process_handle);

    if let Err(e) = price_res {
        error!(error = %e, "price updater task panicked");
    }
    if let Err(e) = ws_res {
        error!(error = %e, "WebSocket listener task panicked");
    }
    if let Err(e) = pub_res {
        error!(error = %e, "Redis publisher task panicked");
    }
    if let Err(e) = proc_res {
        error!(error = %e, "processing loop task panicked");
    }

    info!("shutdown complete");
    Ok(())
}

/// Main processing loop: receives decoded swaps, classifies them, estimates
/// USD value, computes poison score, and records in the aggregator.
async fn run_processing_loop(
    swap_rx: &mut mpsc::Receiver<DecodedSwap>,
    price_cache: &PriceCache,
    aggregator: &Arc<Mutex<SwapAggregator>>,
    poison_threshold: f64,
    shutdown: &CancellationToken,
) {
    let mut processed = 0u64;
    let mut filtered_direction = 0u64;
    let mut filtered_poison = 0u64;

    loop {
        tokio::select! {
            swap = swap_rx.recv() => {
                let mut swap = match swap {
                    Some(s) => s,
                    None => {
                        info!("swap channel closed, stopping processing loop");
                        return;
                    }
                };

                // 1. Classify direction.
                let direction = classify_direction(swap.token_in, swap.token_out);
                if direction == SwapDirection::Skip {
                    filtered_direction += 1;
                    continue;
                }
                swap.direction = direction;

                // 2. Identify the volatile token symbol.
                let volatile_addr = match get_volatile_token(&swap) {
                    Some(addr) => addr,
                    None => continue,
                };
                let symbol = match volatile_symbol(volatile_addr) {
                    Some(s) => s,
                    None => continue,
                };

                // 3. Estimate USD value.
                let prices = price_cache.get_all();
                swap.usd_value = estimate_usd_value(&swap, &prices);

                // 4. Compute poison score and filter.
                swap.poison_score = compute_poison_score(&swap, None);
                if swap.poison_score >= poison_threshold {
                    filtered_poison += 1;
                    continue;
                }

                // 5. Record in aggregator.
                {
                    let mut agg = aggregator.lock().expect("aggregator lock poisoned");
                    agg.record_swap(symbol, direction, swap.usd_value);
                }

                processed += 1;

                if processed % 1000 == 0 {
                    info!(
                        processed = processed,
                        filtered_direction = filtered_direction,
                        filtered_poison = filtered_poison,
                        "processing stats"
                    );
                }
            }
            _ = shutdown.cancelled() => {
                info!(
                    processed = processed,
                    filtered_direction = filtered_direction,
                    filtered_poison = filtered_poison,
                    "processing loop shutting down"
                );
                return;
            }
        }
    }
}
