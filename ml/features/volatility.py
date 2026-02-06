"""
Volatility estimation features.

Based on research:
- GARCH(1,1) for conditional volatility forecasting
- Parkinson volatility (high-low range estimator)
- Realized volatility from high-frequency data
"""

import numpy as np
import pandas as pd
from typing import Tuple


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 15
) -> pd.Series:
    """
    Realized Volatility - standard deviation of returns.

    RV = sqrt(sum(returns^2))

    Annualized for crypto (365 * 24 * 60 minutes).

    Args:
        returns: Log returns or simple returns
        window: Window size in minutes (default: 15m)

    Returns:
        Realized volatility series (annualized)
    """
    # Sum of squared returns
    rv = returns.rolling(window=window).std()

    # Annualize (assuming 1-minute data, 365 * 24 * 60 = 525,600 minutes per year)
    annualization_factor = np.sqrt(525_600 / window)
    rv_annualized = rv * annualization_factor

    return rv_annualized


def calculate_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Parkinson Volatility Estimator.

    More efficient than close-to-close volatility.
    Uses high-low range information.

    Parkinson = sqrt((1 / (4 * ln(2))) * (ln(high/low))^2)

    Args:
        high: High price series
        low: Low price series
        window: Rolling window

    Returns:
        Parkinson volatility series
    """
    log_hl = np.log(high / (low + 1e-10))
    log_hl_squared = log_hl ** 2

    # Rolling mean
    mean_log_hl_sq = log_hl_squared.rolling(window=window).mean()

    # Parkinson estimator
    parkinson = np.sqrt(mean_log_hl_sq / (4 * np.log(2)))

    return parkinson


def calculate_garch_volatility(
    returns: pd.Series,
    omega: float = 0.00001,
    alpha: float = 0.1,
    beta: float = 0.85
) -> pd.Series:
    """
    GARCH(1,1) Conditional Volatility.

    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

    where epsilon_t = returns_t

    Based on GARCH model for financial time series.
    Used for position sizing (Fractional Kelly with GARCH volatility).

    Args:
        returns: Log returns series
        omega: Constant term (default: 0.00001)
        alpha: ARCH coefficient (default: 0.1)
        beta: GARCH coefficient (default: 0.85)

    Returns:
        Conditional volatility series (sigma_t)
    """
    # Initialize
    variance = pd.Series(index=returns.index, dtype=float)
    variance.iloc[0] = returns.var()  # Initial variance

    # GARCH recursion
    for t in range(1, len(returns)):
        prev_variance = variance.iloc[t-1]
        prev_return_sq = returns.iloc[t-1] ** 2

        variance.iloc[t] = omega + alpha * prev_return_sq + beta * prev_variance

    # Volatility = sqrt(variance)
    volatility = np.sqrt(variance)

    return volatility


def calculate_ewma_volatility(
    returns: pd.Series,
    lambda_param: float = 0.94
) -> pd.Series:
    """
    Exponentially Weighted Moving Average (EWMA) Volatility.

    RiskMetrics approach (JP Morgan).
    sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * return_{t-1}^2

    Args:
        returns: Log returns series
        lambda_param: Decay parameter (default: 0.94 for daily, 0.99 for intraday)

    Returns:
        EWMA volatility series
    """
    # Squared returns
    returns_sq = returns ** 2

    # EWMA variance
    variance = returns_sq.ewm(alpha=(1 - lambda_param), adjust=False).mean()

    # Volatility
    volatility = np.sqrt(variance)

    return volatility


def calculate_rolling_max_drawdown(
    prices: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Rolling Maximum Drawdown.

    Drawdown = (price - running_max) / running_max

    Args:
        prices: Price series
        window: Rolling window (default: 60 minutes = 1 hour)

    Returns:
        Rolling max drawdown series (negative values)
    """
    rolling_max = prices.rolling(window=window, min_periods=1).max()
    drawdown = (prices - rolling_max) / (rolling_max + 1e-10)

    # Maximum drawdown (most negative)
    max_dd = drawdown.rolling(window=window, min_periods=1).min()

    return max_dd


def calculate_time_since_high(
    prices: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Time Since Rolling High.

    Number of periods since the rolling maximum.

    Args:
        prices: Price series
        window: Rolling window

    Returns:
        Periods since high (0 to window)
    """
    rolling_max = prices.rolling(window=window, min_periods=1).max()

    # Find when price equals rolling max
    at_high = (prices == rolling_max).astype(int)

    # Cumulative count since last high
    time_since = at_high.groupby((at_high != at_high.shift()).cumsum()).cumcount()

    return time_since


def calculate_volatility_ratio(
    short_vol: pd.Series,
    long_vol: pd.Series
) -> pd.Series:
    """
    Volatility Ratio - short-term vol / long-term vol.

    Ratio > 1: Increasing volatility
    Ratio < 1: Decreasing volatility

    Args:
        short_vol: Short-term volatility (e.g., 15-minute)
        long_vol: Long-term volatility (e.g., 1-hour)

    Returns:
        Volatility ratio series
    """
    ratio = short_vol / (long_vol + 1e-10)
    return ratio


def calculate_garman_klass_volatility(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Garman-Klass Volatility Estimator.

    More efficient than Parkinson, uses OHLC data.

    GK = sqrt((1/2) * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2)

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series
        window: Rolling window

    Returns:
        Garman-Klass volatility series
    """
    log_hl = np.log(high / (low + 1e-10))
    log_co = np.log(close / (open_price + 1e-10))

    term1 = 0.5 * (log_hl ** 2)
    term2 = (2 * np.log(2) - 1) * (log_co ** 2)

    gk_sq = term1 - term2

    # Rolling mean
    gk = np.sqrt(gk_sq.rolling(window=window).mean())

    return gk
