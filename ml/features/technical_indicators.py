"""
Technical indicators for feature engineering.

Based on research:
- arXiv 2407.11786: Technical indicators enhance XGBoost performance
- MDPI 2025: EMA, RSI, MACD, BB effective for crypto prediction
"""

import numpy as np
import pandas as pd
from typing import Tuple


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average (EMA).

    Args:
        prices: Price series
        period: EMA period (e.g., 12, 26, 50)

    Returns:
        EMA series
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) using Wilder's smoothing.

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss

    Args:
        prices: Price series
        period: RSI period (default: 14)

    Returns:
        RSI series (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Wilder's smoothing (exponential moving average with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence (MACD).

    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)
    Histogram = MACD - Signal

    Args:
        prices: Price series
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        Tuple of (MACD, Signal, Histogram)
    """
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)

    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal

    return macd, signal, histogram


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands with position indicator.

    Middle Band = SMA(period)
    Upper Band = Middle + (num_std * std)
    Lower Band = Middle - (num_std * std)
    BB Position = (Price - Lower) / (Upper - Lower)

    Args:
        prices: Price series
        period: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        Tuple of (upper, middle, lower, position)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    # BB Position: 0 = at lower band, 0.5 = at middle, 1 = at upper band
    position = (prices - lower) / ((upper - lower) + 1e-10)

    return upper, middle, lower, position


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Average True Range (ATR) - volatility indicator.

    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = EMA(True Range, period)

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period (default: 14)

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


def calculate_hurst_exponent(prices: pd.Series, lags: int = 100) -> float:
    """
    Hurst exponent for regime detection (trending vs. mean-reverting).

    H > 0.55: Trending market
    H < 0.45: Mean-reverting market
    0.45 <= H <= 0.55: Random walk

    Based on: Maraj-Mervar & Aybar (FracTime 2025)

    Args:
        prices: Price series
        lags: Number of lags for R/S analysis (default: 100)

    Returns:
        Hurst exponent (0 to 1)
    """
    if len(prices) < lags:
        return 0.5  # Default to random walk

    log_returns = np.log(prices / prices.shift(1)).dropna()

    # R/S analysis
    tau = []
    rs = []

    for lag in range(2, min(lags, len(log_returns))):
        # Split into chunks
        n_chunks = len(log_returns) // lag
        if n_chunks < 1:
            continue

        for i in range(n_chunks):
            chunk = log_returns.iloc[i*lag:(i+1)*lag].values

            # Mean-adjusted cumulative sum
            mean_chunk = chunk.mean()
            cumsum = np.cumsum(chunk - mean_chunk)

            # Range and standard deviation
            R = cumsum.max() - cumsum.min()
            S = chunk.std() + 1e-10

            rs.append(R / S)
        tau.append(lag)

    # Hurst = slope of log(R/S) vs log(lag)
    if len(rs) > 0 and len(tau) > 0:
        log_rs = np.log(rs)
        log_tau = np.log(tau * len(rs) // len(tau))  # Repeat tau to match rs length
        hurst = np.polyfit(log_tau[:len(log_rs)], log_rs, 1)[0]
        return max(0.0, min(1.0, hurst))  # Clamp to [0, 1]
    else:
        return 0.5
