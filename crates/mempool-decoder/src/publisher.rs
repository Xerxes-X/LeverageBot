//! Redis publisher that periodically writes the aggregate mempool signal.
//!
//! Uses `SET key json EX 30` (not PUBLISH) so the bot can poll with GET.
//! The 30-second TTL ensures stale data auto-expires if the decoder stops.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use redis::AsyncCommands;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::aggregator::SwapAggregator;

/// Run the Redis publisher loop.
///
/// Every `interval_secs`, locks the aggregator, prunes expired swaps,
/// builds the aggregate signal, serializes to JSON, and writes to Redis
/// with a 30-second TTL.
pub async fn run_publisher(
    aggregator: Arc<Mutex<SwapAggregator>>,
    redis_url: &str,
    channel: &str,
    interval_secs: u64,
    shutdown: CancellationToken,
) {
    let client = match redis::Client::open(redis_url) {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "failed to create Redis client");
            return;
        }
    };

    let mut conn = match client.get_multiplexed_async_connection().await {
        Ok(c) => {
            info!("Redis publisher connected");
            c
        }
        Err(e) => {
            error!(error = %e, "failed to connect to Redis");
            return;
        }
    };

    let interval = std::time::Duration::from_secs(interval_secs);
    let key = channel.to_string();

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = shutdown.cancelled() => {
                debug!("publisher shutting down");
                return;
            }
        }

        // Lock, prune, and build signal.
        let signal = {
            let mut agg = aggregator.lock().expect("aggregator lock poisoned");
            agg.prune();
            agg.build_signal()
        };

        // Serialize to JSON.
        let json = match serde_json::to_string(&signal) {
            Ok(j) => j,
            Err(e) => {
                warn!(error = %e, "failed to serialize signal");
                continue;
            }
        };

        // SET with 30-second TTL.
        let result: Result<(), redis::RedisError> = conn
            .set_ex(&key, &json, 30)
            .await;

        match result {
            Ok(()) => {
                debug!(
                    pairs = signal.pairs.len(),
                    json_bytes = json.len(),
                    "published aggregate signal"
                );
            }
            Err(e) => {
                warn!(error = %e, "failed to publish signal to Redis");
                // Attempt to reconnect on next iteration.
                match client.get_multiplexed_async_connection().await {
                    Ok(new_conn) => {
                        conn = new_conn;
                        debug!("reconnected to Redis");
                    }
                    Err(re) => {
                        warn!(error = %re, "Redis reconnect failed");
                    }
                }
            }
        }
    }
}
