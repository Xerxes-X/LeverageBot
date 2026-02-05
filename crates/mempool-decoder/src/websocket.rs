//! WebSocket connection to a BSC node for pending transaction subscription.
//!
//! Connects to a BSC WebSocket endpoint, subscribes to `newPendingTransactions`
//! with full transaction objects, and forwards decoded swaps through an mpsc channel.
//! Includes LRU-based deduplication and automatic reconnection with backoff.

use std::num::NonZeroUsize;

use alloy::primitives::{Address, B256, U256};
use futures::stream::StreamExt;
use lru::LruCache;
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::decoder;
use crate::types::DecodedSwap;

/// Run the WebSocket listener loop with reconnection support.
pub async fn run_websocket(
    url: &str,
    swap_tx: mpsc::Sender<DecodedSwap>,
    dedup_size: usize,
    reconnect_delay_secs: u64,
    max_reconnect_attempts: u32,
    shutdown: CancellationToken,
) {
    let cache_size = NonZeroUsize::new(dedup_size).expect("dedup_size must be non-zero");
    let mut dedup_cache: LruCache<B256, ()> = LruCache::new(cache_size);
    let mut attempt = 0u32;

    loop {
        if shutdown.is_cancelled() {
            info!("WebSocket listener shutdown requested");
            return;
        }

        info!(url = url, attempt = attempt + 1, "connecting to BSC WebSocket");

        match connect_and_listen(url, &swap_tx, &mut dedup_cache, &shutdown).await {
            Ok(()) => {
                // Clean shutdown via cancellation.
                info!("WebSocket listener stopped cleanly");
                return;
            }
            Err(e) => {
                attempt += 1;
                if attempt >= max_reconnect_attempts {
                    error!(
                        error = %e,
                        attempts = attempt,
                        "max reconnect attempts reached, giving up"
                    );
                    return;
                }

                let delay = reconnect_delay_secs * (attempt as u64).min(6); // cap backoff
                warn!(
                    error = %e,
                    attempt = attempt,
                    delay_secs = delay,
                    "WebSocket disconnected, reconnecting"
                );

                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(delay)) => {}
                    _ = shutdown.cancelled() => return,
                }
            }
        }
    }
}

/// Connect to the WebSocket, subscribe, and process messages until
/// disconnection or shutdown.
async fn connect_and_listen(
    url: &str,
    swap_tx: &mpsc::Sender<DecodedSwap>,
    dedup_cache: &mut LruCache<B256, ()>,
    shutdown: &CancellationToken,
) -> Result<(), anyhow::Error> {
    let (ws_stream, _response) = connect_async(url).await?;
    let (mut write, mut read) = ws_stream.split();

    // Subscribe to newPendingTransactions with full tx objects.
    let subscribe_msg = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_subscribe",
        "params": ["newPendingTransactions", true]
    });

    use futures::SinkExt;
    write
        .send(Message::Text(subscribe_msg.to_string().into()))
        .await?;

    info!("subscribed to newPendingTransactions");

    // Track stats.
    let mut total_received = 0u64;
    let mut total_decoded = 0u64;
    let mut total_deduped = 0u64;

    loop {
        tokio::select! {
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        total_received += 1;

                        if let Some(swap) = process_message(&text, dedup_cache, &mut total_deduped) {
                            total_decoded += 1;

                            if swap_tx.send(swap).await.is_err() {
                                debug!("swap channel closed, stopping listener");
                                return Ok(());
                            }
                        }

                        // Periodic stats logging.
                        if total_received % 10_000 == 0 {
                            info!(
                                received = total_received,
                                decoded = total_decoded,
                                deduped = total_deduped,
                                "WebSocket stats"
                            );
                        }
                    }
                    Some(Ok(Message::Ping(data))) => {
                        use futures::SinkExt;
                        let _ = write.send(Message::Pong(data)).await;
                    }
                    Some(Ok(Message::Close(_))) => {
                        info!("WebSocket closed by server");
                        return Err(anyhow::anyhow!("WebSocket closed by server"));
                    }
                    Some(Err(e)) => {
                        return Err(e.into());
                    }
                    None => {
                        return Err(anyhow::anyhow!("WebSocket stream ended"));
                    }
                    _ => {} // Binary, Pong, Frame — ignore.
                }
            }
            _ = shutdown.cancelled() => {
                info!("shutdown requested, closing WebSocket");
                return Ok(());
            }
        }
    }
}

/// Process a single WebSocket message and attempt to decode it as a swap.
fn process_message(
    text: &str,
    dedup_cache: &mut LruCache<B256, ()>,
    dedup_count: &mut u64,
) -> Option<DecodedSwap> {
    let msg: serde_json::Value = serde_json::from_str(text).ok()?;

    // The subscription notification has shape:
    // { "jsonrpc": "2.0", "method": "eth_subscribe", "params": { "result": { ... tx ... } } }
    let tx = msg.get("params")?.get("result")?;

    // Extract transaction fields.
    let hash_str = tx.get("hash")?.as_str()?;
    let to_str = tx.get("to")?.as_str()?;
    let input_str = tx.get("input")?.as_str()?;

    // Parse tx hash.
    let tx_hash: B256 = hash_str.parse().ok()?;

    // Deduplication check.
    if dedup_cache.contains(&tx_hash) {
        *dedup_count += 1;
        return None;
    }
    dedup_cache.put(tx_hash, ());

    // Parse addresses and values.
    let to: Address = to_str.parse().ok()?;

    let calldata = hex::decode(input_str.strip_prefix("0x").unwrap_or(input_str)).ok()?;

    let value_str = tx.get("value").and_then(|v| v.as_str()).unwrap_or("0x0");
    let value = U256::from_str_radix(
        value_str.strip_prefix("0x").unwrap_or(value_str),
        16,
    )
    .unwrap_or(U256::ZERO);

    let gas_price_str = tx
        .get("gasPrice")
        .and_then(|v| v.as_str())
        .unwrap_or("0x0");
    let gas_price = u128::from_str_radix(
        gas_price_str.strip_prefix("0x").unwrap_or(gas_price_str),
        16,
    )
    .unwrap_or(0);

    // Attempt to decode the transaction.
    decoder::decode_transaction(to, &calldata, value, tx_hash, gas_price)
}
