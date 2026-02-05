//! Rolling window aggregator for mempool swap statistics.
//!
//! Maintains sliding windows (1m, 5m, 15m) per volatile token and computes
//! aggregate signal metrics: buy/sell volumes, direction score, whale counts,
//! volume acceleration.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::types::{MempoolSignal, MempoolTokenSignal, SwapDirection};

// ---------------------------------------------------------------------------
// Window durations
// ---------------------------------------------------------------------------

const WINDOW_1M: Duration = Duration::from_secs(60);
const WINDOW_5M: Duration = Duration::from_secs(300);
const WINDOW_15M: Duration = Duration::from_secs(900);
const WINDOW_30M: Duration = Duration::from_secs(1800);

/// Interval at which 5m volume snapshots are taken for volume acceleration.
const VOLUME_SNAPSHOT_INTERVAL: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single swap record for the rolling windows.
#[derive(Debug, Clone)]
struct SwapRecord {
    timestamp: Instant,
    direction: SwapDirection,
    usd_value: Decimal,
    is_whale: bool,
}

/// Rolling window data for a single volatile token.
#[derive(Debug)]
struct TokenWindows {
    /// All swaps within 1 minute.
    swaps: VecDeque<SwapRecord>,
    /// Historical 5m volume snapshots for acceleration calculation.
    trailing_5m_volumes: VecDeque<(Instant, Decimal)>,
    /// Total swaps ever seen (not windowed).
    total_swaps: u32,
    /// Last time a volume snapshot was recorded.
    last_snapshot: Instant,
}

impl TokenWindows {
    fn new() -> Self {
        Self {
            swaps: VecDeque::new(),
            trailing_5m_volumes: VecDeque::new(),
            total_swaps: 0,
            last_snapshot: Instant::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// SwapAggregator
// ---------------------------------------------------------------------------

/// Aggregates decoded swaps into rolling window statistics per volatile token.
pub struct SwapAggregator {
    /// Per-token rolling windows.
    windows: HashMap<String, TokenWindows>,
    /// USD threshold for whale classification.
    whale_threshold: Decimal,
}

impl SwapAggregator {
    /// Create a new aggregator for the given volatile tokens.
    pub fn new(whale_threshold: Decimal, volatile_tokens: &[String]) -> Self {
        let mut windows = HashMap::new();
        for token in volatile_tokens {
            windows.insert(token.clone(), TokenWindows::new());
        }
        Self {
            windows,
            whale_threshold,
        }
    }

    /// Record a decoded swap into the aggregator.
    pub fn record_swap(&mut self, symbol: &str, direction: SwapDirection, usd_value: Decimal) {
        let windows = match self.windows.get_mut(symbol) {
            Some(w) => w,
            None => return, // Unknown token — ignore.
        };

        let is_whale = usd_value >= self.whale_threshold;
        let record = SwapRecord {
            timestamp: Instant::now(),
            direction,
            usd_value,
            is_whale,
        };

        windows.swaps.push_back(record);
        windows.total_swaps += 1;
    }

    /// Remove expired entries from all windows.
    pub fn prune(&mut self) {
        let now = Instant::now();

        for windows in self.windows.values_mut() {
            // Remove swaps older than 15m (the largest window).
            while let Some(front) = windows.swaps.front() {
                if now.duration_since(front.timestamp) > WINDOW_15M {
                    windows.swaps.pop_front();
                } else {
                    break;
                }
            }

            // Remove trailing volume snapshots older than 30m.
            while let Some(&(ts, _)) = windows.trailing_5m_volumes.front() {
                if now.duration_since(ts) > WINDOW_30M {
                    windows.trailing_5m_volumes.pop_front();
                } else {
                    break;
                }
            }

            // Take a 5m volume snapshot if enough time has elapsed.
            if now.duration_since(windows.last_snapshot) >= VOLUME_SNAPSHOT_INTERVAL {
                let vol_5m = sum_volume(&windows.swaps, now, WINDOW_5M);
                windows.trailing_5m_volumes.push_back((now, vol_5m));
                windows.last_snapshot = now;
            }
        }
    }

    /// Compute the aggregate signal for a single token.
    pub fn compute_signal(&self, symbol: &str) -> MempoolTokenSignal {
        let windows = match self.windows.get(symbol) {
            Some(w) => w,
            None => return empty_signal(),
        };

        let now = Instant::now();

        // Buy/sell volumes for each window.
        let (buy_1m, sell_1m) = buy_sell_volumes(&windows.swaps, now, WINDOW_1M);
        let (buy_5m, sell_5m) = buy_sell_volumes(&windows.swaps, now, WINDOW_5M);
        let (buy_15m, sell_15m) = buy_sell_volumes(&windows.swaps, now, WINDOW_15M);

        // Buy/sell ratio for 5m window.
        let total_5m = buy_5m + sell_5m;
        let buy_sell_ratio_5m = if total_5m > Decimal::ZERO {
            buy_5m / total_5m
        } else {
            dec!(0.5)
        };

        // Direction score: (ratio - 0.5) * 2, range [-1.0, 1.0].
        let direction_score_5m = (buy_sell_ratio_5m - dec!(0.5)) * dec!(2);

        // Transaction counts for 5m.
        let (tx_buy_5m, tx_sell_5m) = tx_counts(&windows.swaps, now, WINDOW_5M);

        // Whale counts for 15m.
        let (whale_buy_15m, whale_sell_15m) = whale_counts(&windows.swaps, now, WINDOW_15M);

        // Volume acceleration: current 5m / trailing 30m average 5m.
        let volume_acceleration = compute_volume_acceleration(
            total_5m,
            &windows.trailing_5m_volumes,
        );

        MempoolTokenSignal {
            buy_volume_1m_usd: buy_1m,
            sell_volume_1m_usd: sell_1m,
            buy_volume_5m_usd: buy_5m,
            sell_volume_5m_usd: sell_5m,
            buy_volume_15m_usd: buy_15m,
            sell_volume_15m_usd: sell_15m,
            direction_score_5m,
            buy_sell_ratio_5m,
            tx_count_buy_5m: tx_buy_5m,
            tx_count_sell_5m: tx_sell_5m,
            whale_buy_count_15m: whale_buy_15m,
            whale_sell_count_15m: whale_sell_15m,
            volume_acceleration,
            total_swaps_seen: windows.total_swaps,
        }
    }

    /// Build the full aggregate signal across all monitored tokens.
    pub fn build_signal(&self) -> MempoolSignal {
        let mut pairs = HashMap::new();

        for symbol in self.windows.keys() {
            pairs.insert(symbol.clone(), self.compute_signal(symbol));
        }

        MempoolSignal {
            pairs,
            timestamp: chrono::Utc::now().timestamp(),
            data_age_seconds: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn buy_sell_volumes(
    swaps: &VecDeque<SwapRecord>,
    now: Instant,
    window: Duration,
) -> (Decimal, Decimal) {
    let cutoff = now - window;
    let mut buy = Decimal::ZERO;
    let mut sell = Decimal::ZERO;

    for record in swaps.iter().rev() {
        if record.timestamp < cutoff {
            break;
        }
        match record.direction {
            SwapDirection::Buy => buy += record.usd_value,
            SwapDirection::Sell => sell += record.usd_value,
            SwapDirection::Skip => {}
        }
    }

    (buy, sell)
}

fn sum_volume(
    swaps: &VecDeque<SwapRecord>,
    now: Instant,
    window: Duration,
) -> Decimal {
    let cutoff = now - window;
    let mut total = Decimal::ZERO;

    for record in swaps.iter().rev() {
        if record.timestamp < cutoff {
            break;
        }
        total += record.usd_value;
    }

    total
}

fn tx_counts(
    swaps: &VecDeque<SwapRecord>,
    now: Instant,
    window: Duration,
) -> (u32, u32) {
    let cutoff = now - window;
    let mut buys = 0u32;
    let mut sells = 0u32;

    for record in swaps.iter().rev() {
        if record.timestamp < cutoff {
            break;
        }
        match record.direction {
            SwapDirection::Buy => buys += 1,
            SwapDirection::Sell => sells += 1,
            SwapDirection::Skip => {}
        }
    }

    (buys, sells)
}

fn whale_counts(
    swaps: &VecDeque<SwapRecord>,
    now: Instant,
    window: Duration,
) -> (u32, u32) {
    let cutoff = now - window;
    let mut whale_buys = 0u32;
    let mut whale_sells = 0u32;

    for record in swaps.iter().rev() {
        if record.timestamp < cutoff {
            break;
        }
        if record.is_whale {
            match record.direction {
                SwapDirection::Buy => whale_buys += 1,
                SwapDirection::Sell => whale_sells += 1,
                SwapDirection::Skip => {}
            }
        }
    }

    (whale_buys, whale_sells)
}

fn compute_volume_acceleration(
    current_5m: Decimal,
    trailing: &VecDeque<(Instant, Decimal)>,
) -> Decimal {
    if trailing.is_empty() {
        return dec!(1.0);
    }

    let sum: Decimal = trailing.iter().map(|(_, v)| v).sum();
    let count = Decimal::from(trailing.len() as u32);
    let avg = sum / count;

    if avg > Decimal::ZERO {
        current_5m / avg
    } else {
        dec!(1.0)
    }
}

fn empty_signal() -> MempoolTokenSignal {
    MempoolTokenSignal {
        buy_volume_1m_usd: Decimal::ZERO,
        sell_volume_1m_usd: Decimal::ZERO,
        buy_volume_5m_usd: Decimal::ZERO,
        sell_volume_5m_usd: Decimal::ZERO,
        buy_volume_15m_usd: Decimal::ZERO,
        sell_volume_15m_usd: Decimal::ZERO,
        direction_score_5m: Decimal::ZERO,
        buy_sell_ratio_5m: dec!(0.5),
        tx_count_buy_5m: 0,
        tx_count_sell_5m: 0,
        whale_buy_count_15m: 0,
        whale_sell_count_15m: 0,
        volume_acceleration: dec!(1.0),
        total_swaps_seen: 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_aggregator() -> SwapAggregator {
        SwapAggregator::new(
            dec!(10_000),
            &["WBNB".to_string(), "BTCB".to_string(), "ETH".to_string()],
        )
    }

    #[test]
    fn test_empty_signal() {
        let agg = make_aggregator();
        let sig = agg.compute_signal("WBNB");
        assert_eq!(sig.buy_volume_5m_usd, Decimal::ZERO);
        assert_eq!(sig.sell_volume_5m_usd, Decimal::ZERO);
        assert_eq!(sig.buy_sell_ratio_5m, dec!(0.5));
        assert_eq!(sig.direction_score_5m, Decimal::ZERO);
        assert_eq!(sig.total_swaps_seen, 0);
    }

    #[test]
    fn test_record_and_compute() {
        let mut agg = make_aggregator();
        agg.record_swap("WBNB", SwapDirection::Buy, dec!(5000));
        agg.record_swap("WBNB", SwapDirection::Buy, dec!(3000));
        agg.record_swap("WBNB", SwapDirection::Sell, dec!(2000));

        let sig = agg.compute_signal("WBNB");
        assert_eq!(sig.buy_volume_5m_usd, dec!(8000));
        assert_eq!(sig.sell_volume_5m_usd, dec!(2000));
        assert_eq!(sig.tx_count_buy_5m, 2);
        assert_eq!(sig.tx_count_sell_5m, 1);
        assert_eq!(sig.total_swaps_seen, 3);
        // ratio = 8000/10000 = 0.8, direction = (0.8 - 0.5) * 2 = 0.6
        assert_eq!(sig.buy_sell_ratio_5m, dec!(0.8));
        assert_eq!(sig.direction_score_5m, dec!(0.6));
    }

    #[test]
    fn test_whale_detection() {
        let mut agg = make_aggregator();
        agg.record_swap("BTCB", SwapDirection::Buy, dec!(15000)); // whale
        agg.record_swap("BTCB", SwapDirection::Buy, dec!(500));   // not whale
        agg.record_swap("BTCB", SwapDirection::Sell, dec!(20000)); // whale

        let sig = agg.compute_signal("BTCB");
        assert_eq!(sig.whale_buy_count_15m, 1);
        assert_eq!(sig.whale_sell_count_15m, 1);
    }

    #[test]
    fn test_unknown_token_ignored() {
        let mut agg = make_aggregator();
        agg.record_swap("DOGE", SwapDirection::Buy, dec!(1000));
        // Should not panic; DOGE is not a monitored token.
        let sig = agg.compute_signal("DOGE");
        assert_eq!(sig.total_swaps_seen, 0);
    }

    #[test]
    fn test_build_signal_contains_all_tokens() {
        let agg = make_aggregator();
        let signal = agg.build_signal();
        assert!(signal.pairs.contains_key("WBNB"));
        assert!(signal.pairs.contains_key("BTCB"));
        assert!(signal.pairs.contains_key("ETH"));
        assert_eq!(signal.pairs.len(), 3);
    }
}
