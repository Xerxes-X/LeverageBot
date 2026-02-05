//! Pure computation module for technical indicators and statistical measures.
//!
//! No I/O, no side effects. Takes OHLCV arrays / trade data and returns
//! indicator values. All computations use `Decimal` for precision; statistical
//! routines (Hurst, GARCH, realized vol) work internally in `f64` for
//! numerical stability and convert back to `Decimal` at the boundary.
//!
//! Indicators implemented:
//! - Standard: EMA, RSI (Wilder's smoothing), MACD, Bollinger Bands, ATR
//! - Regime / Statistical: Hurst exponent (R/S method), realized volatility,
//!   GARCH(1,1) one-step-ahead forecast
//! - Microstructure: VPIN (Easley et al. 2012), OBI (Kolm et al. 2023)
//!
//! References:
//!     Bollerslev (1986), "Generalized Autoregressive Conditional Heteroskedasticity".
//!     Hansen & Lunde (2005), "A Forecast Comparison of Volatility Models".
//!     Easley, Lopez de Prado & O'Hara (2012), "Flow Toxicity and Liquidity
//!         in a High-Frequency World", Review of Financial Studies.
//!     Kolm, Turiel & Westray (2023), "Deep Order Flow Imbalance",
//!         Journal of Financial Economics.
//!     Wilder (1978), "New Concepts in Technical Trading Systems".

use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use rust_decimal::MathematicalOps;
use rust_decimal_macros::dec;

use crate::config::IndicatorParams;
use crate::types::{IndicatorSnapshot, Trade, OHLCV};

// ═══════════════════════════════════════════════════════════════════════════
// Standard Technical Indicators
// ═══════════════════════════════════════════════════════════════════════════

/// Exponential Moving Average.
///
/// Multiplier `k = 2 / (period + 1)`. First value seeded with SMA of the
/// first `period` prices. Returns empty `Vec` if insufficient data.
pub fn ema(prices: &[Decimal], period: usize) -> Vec<Decimal> {
    if prices.len() < period || period == 0 {
        return Vec::new();
    }

    let k = dec!(2) / Decimal::from(period as u64 + 1);
    let one_minus_k = dec!(1) - k;

    // Seed with SMA of first `period` values.
    let sma: Decimal = prices[..period].iter().copied().sum::<Decimal>()
        / Decimal::from(period as u64);

    let mut result = Vec::with_capacity(prices.len() - period + 1);
    result.push(sma);

    for &price in &prices[period..] {
        let prev = *result.last().expect("result is seeded with SMA");
        let ema_val = price * k + prev * one_minus_k;
        result.push(ema_val);
    }

    result
}

/// Relative Strength Index (Wilder's smoothing).
///
/// Uses smoothing factor `1/period` (not the standard EMA `2/(period+1)`).
/// Returns 50 if insufficient data.
pub fn rsi(prices: &[Decimal], period: usize) -> Decimal {
    if prices.len() < period + 1 || period == 0 {
        return dec!(50);
    }

    let period_d = Decimal::from(period as u64);
    let period_minus_1 = Decimal::from(period as u64 - 1);

    // Price changes.
    let changes: Vec<Decimal> = prices.windows(2).map(|w| w[1] - w[0]).collect();

    // Initial average gain/loss from first `period` changes.
    let mut avg_gain = changes[..period]
        .iter()
        .map(|&c| if c > Decimal::ZERO { c } else { Decimal::ZERO })
        .sum::<Decimal>()
        / period_d;

    let mut avg_loss = changes[..period]
        .iter()
        .map(|&c| if c < Decimal::ZERO { -c } else { Decimal::ZERO })
        .sum::<Decimal>()
        / period_d;

    // Wilder's smoothing for remaining changes.
    for &c in &changes[period..] {
        if c > Decimal::ZERO {
            avg_gain = (avg_gain * period_minus_1 + c) / period_d;
            avg_loss = (avg_loss * period_minus_1) / period_d;
        } else {
            avg_gain = (avg_gain * period_minus_1) / period_d;
            avg_loss = (avg_loss * period_minus_1 + c.abs()) / period_d;
        }
    }

    if avg_loss == Decimal::ZERO {
        return dec!(100);
    }

    let rs = avg_gain / avg_loss;
    dec!(100) - (dec!(100) / (dec!(1) + rs))
}

/// Moving Average Convergence Divergence.
///
/// Returns `(macd_line, signal_line, histogram)`.
/// Returns `(0, 0, 0)` if insufficient data.
pub fn macd(
    prices: &[Decimal],
    fast: usize,
    slow: usize,
    signal: usize,
) -> (Decimal, Decimal, Decimal) {
    if prices.len() < slow + signal {
        return (Decimal::ZERO, Decimal::ZERO, Decimal::ZERO);
    }

    let fast_ema = ema(prices, fast);
    let slow_ema = ema(prices, slow);

    if fast_ema.is_empty() || slow_ema.is_empty() {
        return (Decimal::ZERO, Decimal::ZERO, Decimal::ZERO);
    }

    // Align: MACD line = fast_ema - slow_ema, from the slow-start onward.
    let offset = slow - fast;
    let macd_values: Vec<Decimal> = (0..slow_ema.len())
        .map(|i| fast_ema[i + offset] - slow_ema[i])
        .collect();

    let signal_ema = ema(&macd_values, signal);
    if signal_ema.is_empty() {
        return (Decimal::ZERO, Decimal::ZERO, Decimal::ZERO);
    }

    let macd_line = *macd_values.last().expect("macd_values non-empty after ema");
    let signal_line = *signal_ema.last().expect("signal_ema non-empty after is_empty check");
    let histogram = macd_line - signal_line;

    (macd_line, signal_line, histogram)
}

/// Bollinger Bands (SMA-based with population standard deviation).
///
/// Returns `(upper, middle, lower)`.  Falls back to `(price, price, price)`
/// if insufficient data.
pub fn bollinger_bands(
    prices: &[Decimal],
    period: usize,
    std_mult: Decimal,
) -> (Decimal, Decimal, Decimal) {
    let fallback = prices.last().copied().unwrap_or(Decimal::ZERO);
    if prices.len() < period || period == 0 {
        return (fallback, fallback, fallback);
    }

    let window = &prices[prices.len() - period..];
    let period_d = Decimal::from(period as u64);
    let middle: Decimal = window.iter().copied().sum::<Decimal>() / period_d;

    // Population variance.
    let variance: Decimal = window
        .iter()
        .map(|&p| {
            let diff = p - middle;
            diff * diff
        })
        .sum::<Decimal>()
        / period_d;

    let std_dev = variance.sqrt().unwrap_or(Decimal::ZERO);
    let upper = middle + std_mult * std_dev;
    let lower = middle - std_mult * std_dev;

    (upper, middle, lower)
}

/// Average True Range (Wilder's smoothing).
///
/// `TR = max(H-L, |H-prevC|, |L-prevC|)`.  Returns `Decimal::ZERO` on
/// mismatched or insufficient data.
pub fn atr(
    highs: &[Decimal],
    lows: &[Decimal],
    closes: &[Decimal],
    period: usize,
) -> Decimal {
    let n = highs.len();
    if n < period + 1 || lows.len() != n || closes.len() != n || period == 0 {
        return Decimal::ZERO;
    }

    // Compute true ranges.
    let true_ranges: Vec<Decimal> = (1..n)
        .map(|i| {
            let hl = highs[i] - lows[i];
            let hc = (highs[i] - closes[i - 1]).abs();
            let lc = (lows[i] - closes[i - 1]).abs();
            hl.max(hc).max(lc)
        })
        .collect();

    let period_d = Decimal::from(period as u64);
    let period_m1 = Decimal::from(period as u64 - 1);

    // First ATR = simple average of first `period` TRs.
    let mut atr_val: Decimal =
        true_ranges[..period].iter().copied().sum::<Decimal>() / period_d;

    // Wilder's smoothing for remaining TRs.
    for &tr in &true_ranges[period..] {
        atr_val = (atr_val * period_m1 + tr) / period_d;
    }

    atr_val
}

// ═══════════════════════════════════════════════════════════════════════════
// Regime & Statistical Indicators
// ═══════════════════════════════════════════════════════════════════════════

/// Rescaled range (R/S) Hurst exponent.
///
/// H > 0.55 → persistent / trending (momentum preferred)
/// H < 0.45 → anti-persistent / mean-reverting
/// 0.45 ≤ H ≤ 0.55 → random walk
///
/// Returns `0.5` when there is insufficient data (< 20 prices).
pub fn hurst_exponent(prices: &[Decimal], max_lag: usize) -> Decimal {
    if prices.len() < 20 {
        return dec!(0.5);
    }

    // Convert to f64 log returns for numerical stability.
    let mut log_returns: Vec<f64> = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        let prev = prices[i - 1].to_f64().unwrap_or(0.0);
        let cur = prices[i].to_f64().unwrap_or(0.0);
        if prev > 0.0 && cur > 0.0 {
            let ratio = cur / prev;
            if ratio > 0.0 {
                log_returns.push(ratio.ln());
            }
        }
    }

    if log_returns.len() < 20 {
        return dec!(0.5);
    }

    let n = log_returns.len();
    let upper_lag = max_lag.min(n / 2);
    if upper_lag < 2 {
        return dec!(0.5);
    }

    let mut log_rs_values: Vec<f64> = Vec::new();
    let mut log_n_values: Vec<f64> = Vec::new();

    for lag in 2..=upper_lag {
        let mut rs_list: Vec<f64> = Vec::new();

        // Non-overlapping sub-periods of size `lag`.
        let mut start = 0;
        while start + lag <= n {
            let subseries = &log_returns[start..start + lag];

            let mean_val: f64 = subseries.iter().sum::<f64>() / lag as f64;

            // Cumulative deviation from mean.
            let mut cum_dev: Vec<f64> = Vec::with_capacity(lag);
            let mut running = 0.0_f64;
            for &val in subseries {
                running += val - mean_val;
                cum_dev.push(running);
            }

            let r = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);

            // Standard deviation (population).
            let var: f64 =
                subseries.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / lag as f64;
            let s = if var > 0.0 { var.sqrt() } else { 1e-10 };

            if s > 0.0 {
                rs_list.push(r / s);
            }

            start += lag;
        }

        if !rs_list.is_empty() {
            let avg_rs: f64 = rs_list.iter().sum::<f64>() / rs_list.len() as f64;
            if avg_rs > 0.0 {
                log_rs_values.push(avg_rs.ln());
                log_n_values.push((lag as f64).ln());
            }
        }
    }

    if log_rs_values.len() < 2 {
        return dec!(0.5);
    }

    // Linear regression: log(R/S) = H * log(n) + c → slope = H.
    let n_pts = log_rs_values.len() as f64;
    let mean_x: f64 = log_n_values.iter().sum::<f64>() / n_pts;
    let mean_y: f64 = log_rs_values.iter().sum::<f64>() / n_pts;

    let numerator: f64 = log_n_values
        .iter()
        .zip(log_rs_values.iter())
        .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
        .sum();
    let denominator: f64 = log_n_values.iter().map(|&x| (x - mean_x).powi(2)).sum();

    if denominator == 0.0 {
        return dec!(0.5);
    }

    let h = (numerator / denominator).clamp(0.0, 1.0);

    // Round to 6 decimal places and convert back.
    Decimal::from_f64((h * 1_000_000.0).round() / 1_000_000.0).unwrap_or(dec!(0.5))
}

/// Annualized realized volatility from log returns.
///
/// Assumes hourly candles (annualizing factor = √8760).
/// Returns `Decimal::ZERO` if fewer than 2 closes.
pub fn realized_volatility(closes: &[Decimal], window: usize) -> Decimal {
    if closes.len() < 2 {
        return Decimal::ZERO;
    }

    let recent = if closes.len() >= window {
        &closes[closes.len() - window..]
    } else {
        closes
    };

    let mut log_returns: Vec<f64> = Vec::with_capacity(recent.len() - 1);
    for i in 1..recent.len() {
        let prev = recent[i - 1].to_f64().unwrap_or(0.0);
        let cur = recent[i].to_f64().unwrap_or(0.0);
        if prev > 0.0 && cur > 0.0 {
            let ratio = cur / prev;
            if ratio > 0.0 {
                log_returns.push(ratio.ln());
            }
        }
    }

    if log_returns.len() < 2 {
        return Decimal::ZERO;
    }

    let n = log_returns.len() as f64;
    let mean_ret: f64 = log_returns.iter().sum::<f64>() / n;
    let variance: f64 =
        log_returns.iter().map(|&r| (r - mean_ret).powi(2)).sum::<f64>() / (n - 1.0);

    // Annualize: hourly → yearly (8760 hours/year).
    let annualized = (variance * 8760.0).sqrt();

    Decimal::from_f64((annualized * 1e8).round() / 1e8).unwrap_or(Decimal::ZERO)
}

/// GARCH(1,1) one-step-ahead volatility forecast.
///
/// `σ²_{t+1} = ω + α · r²_t + β · σ²_t`
///
/// Stationarity constraint: `α + β < 1` (enforced by scaling).
/// Returns `Decimal::ZERO` if fewer than 2 returns.
pub fn garch_volatility(
    returns: &[Decimal],
    omega: Decimal,
    alpha: Decimal,
    beta: Decimal,
) -> Decimal {
    if returns.len() < 2 {
        return Decimal::ZERO;
    }

    // Enforce stationarity.
    let (alpha, beta) = if alpha + beta >= dec!(1) {
        let total = alpha + beta;
        (alpha / total * dec!(0.99), beta / total * dec!(0.99))
    } else {
        (alpha, beta)
    };

    // Initialize variance with sample variance (f64).
    let float_returns: Vec<f64> = returns.iter().map(|r| r.to_f64().unwrap_or(0.0)).collect();
    let mean_ret: f64 = float_returns.iter().sum::<f64>() / float_returns.len() as f64;
    let sample_var: f64 =
        float_returns.iter().map(|&r| (r - mean_ret).powi(2)).sum::<f64>() / float_returns.len() as f64;

    let mut sigma_sq = Decimal::from_f64(sample_var.max(1e-10)).unwrap_or(dec!(0.0000000001));

    // Forward recursion.
    for &ret in returns {
        let r_sq = ret * ret;
        sigma_sq = omega + alpha * r_sq + beta * sigma_sq;
    }

    // Return standard deviation.
    sigma_sq.sqrt().unwrap_or(Decimal::ZERO)
}

// ═══════════════════════════════════════════════════════════════════════════
// Microstructure Indicators
// ═══════════════════════════════════════════════════════════════════════════

/// Volume-Synchronized Probability of Informed Trading.
///
/// Easley et al. (2012): VPIN = mean(|V_buy − V_sell| / V_total) over
/// `window` equal-volume buckets.  Returns a value in `[0, 1]`.
pub fn vpin(trades: &[Trade], bucket_size: Decimal, window: usize) -> Decimal {
    if trades.is_empty() || bucket_size <= Decimal::ZERO {
        return Decimal::ZERO;
    }

    let mut buckets: Vec<(Decimal, Decimal)> = Vec::new(); // (buy_vol, sell_vol)
    let mut current_buy = Decimal::ZERO;
    let mut current_sell = Decimal::ZERO;
    let mut current_total = Decimal::ZERO;

    for trade in trades {
        let vol = trade.quantity;
        // Tick rule: is_buyer_maker == true → sell-initiated.
        if trade.is_buyer_maker {
            current_sell += vol;
        } else {
            current_buy += vol;
        }
        current_total += vol;

        // Fill buckets.
        while current_total >= bucket_size {
            let ratio = if current_total > Decimal::ZERO {
                bucket_size / current_total
            } else {
                dec!(1)
            };
            let bucket_buy = current_buy * ratio;
            let bucket_sell = current_sell * ratio;
            buckets.push((bucket_buy, bucket_sell));

            // Carry over overflow.
            current_buy = (current_buy - bucket_buy).max(Decimal::ZERO);
            current_sell = (current_sell - bucket_sell).max(Decimal::ZERO);
            current_total = current_buy + current_sell;
        }
    }

    if buckets.is_empty() {
        return Decimal::ZERO;
    }

    // Use the most recent `window` buckets.
    let start = if buckets.len() > window {
        buckets.len() - window
    } else {
        0
    };
    let recent = &buckets[start..];

    let mut vpin_sum = Decimal::ZERO;
    let mut valid = 0u32;

    for &(buy_vol, sell_vol) in recent {
        let total = buy_vol + sell_vol;
        if total > Decimal::ZERO {
            let imbalance = (buy_vol - sell_vol).abs() / total;
            vpin_sum += imbalance;
            valid += 1;
        }
    }

    if valid == 0 {
        return Decimal::ZERO;
    }

    vpin_sum / Decimal::from(valid)
}

/// Order Book Imbalance.
///
/// `OBI = (bid_vol − ask_vol) / (bid_vol + ask_vol)` in `[-1, 1]`.
/// Positive = buy pressure, negative = sell pressure.
pub fn order_book_imbalance(
    bids: &[(Decimal, Decimal)],
    asks: &[(Decimal, Decimal)],
) -> Decimal {
    let bid_vol: Decimal = bids.iter().map(|(_, qty)| qty).sum();
    let ask_vol: Decimal = asks.iter().map(|(_, qty)| qty).sum();

    let total = bid_vol + ask_vol;
    if total == Decimal::ZERO {
        return Decimal::ZERO;
    }

    (bid_vol - ask_vol) / total
}

// ═══════════════════════════════════════════════════════════════════════════
// Composite
// ═══════════════════════════════════════════════════════════════════════════

/// Compute all indicators from OHLCV data and optional microstructure data.
///
/// Aggregates standard technical indicators, regime statistics, and
/// microstructure signals into a single [`IndicatorSnapshot`].
pub fn compute_all(
    candles: &[OHLCV],
    trades: Option<&[Trade]>,
    ob_bids: Option<&[(Decimal, Decimal)]>,
    ob_asks: Option<&[(Decimal, Decimal)]>,
    config: &IndicatorParams,
) -> IndicatorSnapshot {
    let closes: Vec<Decimal> = candles.iter().map(|c| c.close).collect();
    let highs: Vec<Decimal> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<Decimal> = candles.iter().map(|c| c.low).collect();
    let volumes: Vec<Decimal> = candles.iter().map(|c| c.volume).collect();

    let fallback_price = closes.last().copied().unwrap_or(Decimal::ZERO);

    // EMA.
    let ema_fast_vals = ema(&closes, config.ema_fast as usize);
    let ema_slow_vals = ema(&closes, config.ema_slow as usize);
    let ema_trend_vals = ema(&closes, config.ema_trend as usize);

    let ema_20 = ema_fast_vals.last().copied().unwrap_or(fallback_price);
    let ema_50 = ema_slow_vals.last().copied().unwrap_or(fallback_price);
    let ema_200 = ema_trend_vals.last().copied().unwrap_or(fallback_price);

    // RSI.
    let rsi_val = rsi(&closes, config.rsi_period as usize);

    // MACD.
    let (macd_line, macd_sig, macd_hist) = macd(
        &closes,
        config.macd_fast as usize,
        config.macd_slow as usize,
        config.macd_signal as usize,
    );

    // Bollinger Bands.
    let bb_std = Decimal::try_from(config.bb_std).unwrap_or(dec!(2));
    let (bb_upper, bb_middle, bb_lower) =
        bollinger_bands(&closes, config.bb_period as usize, bb_std);

    // ATR.
    let atr_period = config.atr_period as usize;
    let atr_val = atr(&highs, &lows, &closes, atr_period);

    // ATR ratio: current ATR / 50-period average ATR.
    let atr_ratio = if candles.len() >= atr_period + 50 {
        let mut atr_history: Vec<Decimal> = Vec::with_capacity(50);
        for i in 0..50 {
            let end = candles.len() - 49 + i;
            let start = end.saturating_sub(atr_period + 1);
            let val = atr(&highs[start..end], &lows[start..end], &closes[start..end], atr_period);
            if val > Decimal::ZERO {
                atr_history.push(val);
            }
        }
        if !atr_history.is_empty() {
            let avg: Decimal = atr_history.iter().copied().sum::<Decimal>()
                / Decimal::from(atr_history.len() as u64);
            if avg > Decimal::ZERO {
                atr_val / avg
            } else {
                dec!(1)
            }
        } else {
            dec!(1)
        }
    } else {
        dec!(1)
    };

    // Volume 20-period average.
    let vol_window = volumes.len().min(20);
    let volume_20_avg = if vol_window > 0 {
        volumes[volumes.len() - vol_window..]
            .iter()
            .copied()
            .sum::<Decimal>()
            / Decimal::from(vol_window as u64)
    } else {
        Decimal::ZERO
    };

    // Hurst exponent.
    let hurst_val = hurst_exponent(&closes, config.hurst_max_lag as usize);

    // VPIN.
    let vpin_val = if let Some(t) = trades {
        if t.is_empty() {
            Decimal::ZERO
        } else {
            let total_vol: Decimal = t.iter().map(|tr| tr.quantity).sum();
            let divisor = if config.vpin_bucket_divisor > 0 {
                Decimal::from(config.vpin_bucket_divisor)
            } else {
                dec!(1)
            };
            let bucket_sz = total_vol / divisor;
            vpin(t, bucket_sz, config.vpin_window as usize)
        }
    } else {
        Decimal::ZERO
    };

    // OBI.
    let obi_val = match (ob_bids, ob_asks) {
        (Some(bids), Some(asks)) => order_book_imbalance(bids, asks),
        _ => Decimal::ZERO,
    };

    // Recent prices (last 200).
    let recent_start = if closes.len() > 200 {
        closes.len() - 200
    } else {
        0
    };
    let recent_prices = closes[recent_start..].to_vec();

    IndicatorSnapshot {
        price: fallback_price,
        ema_20,
        ema_50,
        ema_200,
        rsi_14: rsi_val,
        macd_line,
        macd_signal: macd_sig,
        macd_histogram: macd_hist,
        bb_upper,
        bb_middle,
        bb_lower,
        atr_14: atr_val,
        atr_ratio,
        volume: volumes.last().copied().unwrap_or(Decimal::ZERO),
        volume_20_avg,
        hurst: hurst_val,
        vpin: vpin_val,
        obi: obi_val,
        recent_prices,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Decimal {
        s.parse().unwrap()
    }

    // -- EMA ---------------------------------------------------------------

    #[test]
    fn test_ema_basic() {
        let prices: Vec<Decimal> = (1..=10).map(|i| Decimal::from(i)).collect();
        let result = ema(&prices, 3);
        // First value = SMA of [1,2,3] = 2
        assert_eq!(result[0], dec!(2));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 8); // 10 - 3 + 1
    }

    #[test]
    fn test_ema_insufficient_data() {
        let prices = vec![dec!(1), dec!(2)];
        assert!(ema(&prices, 5).is_empty());
    }

    #[test]
    fn test_ema_period_zero() {
        let prices = vec![dec!(1), dec!(2), dec!(3)];
        assert!(ema(&prices, 0).is_empty());
    }

    // -- RSI ---------------------------------------------------------------

    #[test]
    fn test_rsi_all_gains() {
        // Monotonically increasing -> RSI should be 100.
        let prices: Vec<Decimal> = (1..=20).map(|i| Decimal::from(i)).collect();
        assert_eq!(rsi(&prices, 14), dec!(100));
    }

    #[test]
    fn test_rsi_all_losses() {
        // Monotonically decreasing -> RSI should be 0.
        let prices: Vec<Decimal> = (0..20).rev().map(|i| Decimal::from(i + 1)).collect();
        let val = rsi(&prices, 14);
        assert!(val < dec!(1), "expected near-zero RSI, got {val}");
    }

    #[test]
    fn test_rsi_insufficient_data() {
        let prices = vec![dec!(10), dec!(11)];
        assert_eq!(rsi(&prices, 14), dec!(50));
    }

    // -- MACD --------------------------------------------------------------

    #[test]
    fn test_macd_insufficient_data() {
        let prices: Vec<Decimal> = (1..=10).map(|i| Decimal::from(i)).collect();
        let (m, s, h) = macd(&prices, 12, 26, 9);
        assert_eq!(m, Decimal::ZERO);
        assert_eq!(s, Decimal::ZERO);
        assert_eq!(h, Decimal::ZERO);
    }

    #[test]
    fn test_macd_flat_prices() {
        // All same price -> MACD should be zero.
        let prices: Vec<Decimal> = vec![dec!(100); 50];
        let (m, s, h) = macd(&prices, 12, 26, 9);
        assert_eq!(m, Decimal::ZERO);
        assert_eq!(s, Decimal::ZERO);
        assert_eq!(h, Decimal::ZERO);
    }

    // -- Bollinger Bands ---------------------------------------------------

    #[test]
    fn test_bb_flat_prices() {
        let prices = vec![dec!(100); 20];
        let (upper, middle, lower) = bollinger_bands(&prices, 20, dec!(2));
        assert_eq!(upper, dec!(100));
        assert_eq!(middle, dec!(100));
        assert_eq!(lower, dec!(100));
    }

    #[test]
    fn test_bb_insufficient_data() {
        let prices = vec![dec!(50), dec!(51)];
        let (u, m, l) = bollinger_bands(&prices, 20, dec!(2));
        assert_eq!(u, dec!(51));
        assert_eq!(m, dec!(51));
        assert_eq!(l, dec!(51));
    }

    // -- ATR ---------------------------------------------------------------

    #[test]
    fn test_atr_mismatched_lengths() {
        let highs = vec![dec!(10), dec!(11)];
        let lows = vec![dec!(9)];
        let closes = vec![dec!(10), dec!(10)];
        assert_eq!(atr(&highs, &lows, &closes, 14), Decimal::ZERO);
    }

    #[test]
    fn test_atr_basic() {
        // 16 bars, ATR period 14 -> should produce a value.
        let highs: Vec<Decimal> = (0..16).map(|i| Decimal::from(102 + i % 3)).collect();
        let lows: Vec<Decimal> = (0..16).map(|i| Decimal::from(98 - i % 3)).collect();
        let closes: Vec<Decimal> = (0..16).map(|_| dec!(100)).collect();
        let val = atr(&highs, &lows, &closes, 14);
        assert!(val > Decimal::ZERO);
    }

    // -- Hurst Exponent ----------------------------------------------------

    #[test]
    fn test_hurst_insufficient_data() {
        let prices = vec![dec!(100); 5];
        assert_eq!(hurst_exponent(&prices, 20), dec!(0.5));
    }

    #[test]
    fn test_hurst_trending_series() {
        // Monotonically increasing prices should give H > 0.5.
        let prices: Vec<Decimal> = (0..200)
            .map(|i| Decimal::from(100) + Decimal::from(i))
            .collect();
        let h = hurst_exponent(&prices, 20);
        assert!(
            h > dec!(0.5),
            "trending series should have H > 0.5, got {h}"
        );
    }

    // -- GARCH Volatility --------------------------------------------------

    #[test]
    fn test_garch_insufficient_data() {
        let returns = vec![dec!(0.01)];
        assert_eq!(
            garch_volatility(&returns, dec!(0.00001), dec!(0.1), dec!(0.85)),
            Decimal::ZERO
        );
    }

    #[test]
    fn test_garch_stationarity_enforcement() {
        // alpha + beta = 1.0 -> should be scaled to 0.99.
        let returns: Vec<Decimal> = vec![dec!(0.01), dec!(-0.02), dec!(0.015), dec!(-0.005)];
        let vol = garch_volatility(&returns, dec!(0.00001), dec!(0.5), dec!(0.5));
        assert!(vol > Decimal::ZERO, "GARCH should produce positive vol");
    }

    // -- VPIN --------------------------------------------------------------

    #[test]
    fn test_vpin_empty_trades() {
        assert_eq!(vpin(&[], dec!(100), 50), Decimal::ZERO);
    }

    #[test]
    fn test_vpin_zero_bucket_size() {
        let trades = vec![Trade {
            price: dec!(100),
            quantity: dec!(10),
            timestamp: 0,
            is_buyer_maker: false,
        }];
        assert_eq!(vpin(&trades, Decimal::ZERO, 50), Decimal::ZERO);
    }

    // -- OBI ---------------------------------------------------------------

    #[test]
    fn test_obi_balanced() {
        let bids = vec![(dec!(100), dec!(10))];
        let asks = vec![(dec!(101), dec!(10))];
        assert_eq!(order_book_imbalance(&bids, &asks), Decimal::ZERO);
    }

    #[test]
    fn test_obi_buy_pressure() {
        let bids = vec![(dec!(100), dec!(20))];
        let asks = vec![(dec!(101), dec!(10))];
        let obi = order_book_imbalance(&bids, &asks);
        assert!(obi > Decimal::ZERO, "should show buy pressure");
    }

    #[test]
    fn test_obi_sell_pressure() {
        let bids = vec![(dec!(100), dec!(5))];
        let asks = vec![(dec!(101), dec!(15))];
        let obi = order_book_imbalance(&bids, &asks);
        assert!(obi < Decimal::ZERO, "should show sell pressure");
    }

    #[test]
    fn test_obi_empty() {
        assert_eq!(
            order_book_imbalance(&[], &[]),
            Decimal::ZERO
        );
    }

    // -- Realized Volatility -----------------------------------------------

    #[test]
    fn test_realized_vol_flat() {
        let closes = vec![dec!(100); 30];
        // All same price -> zero volatility.
        assert_eq!(realized_volatility(&closes, 24), Decimal::ZERO);
    }

    #[test]
    fn test_realized_vol_insufficient() {
        let closes = vec![dec!(100)];
        assert_eq!(realized_volatility(&closes, 24), Decimal::ZERO);
    }
}
