"""
Market microstructure features.

Based on research:
- Kolm et al. (2023): Order book imbalance accounts for 73% of prediction performance
- Abad & Yagüe (2025): VPIN significantly predicts price jumps
- Easley et al. (2012): Volume-Synchronized Probability of Informed Trading
"""

import numpy as np
import pandas as pd
from typing import Optional


def calculate_order_book_imbalance(
    bid_volume: pd.Series,
    ask_volume: pd.Series
) -> pd.Series:
    """
    Order Book Imbalance (OBI).

    OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)

    Positive OBI: More buying pressure
    Negative OBI: More selling pressure

    Based on Kolm et al. (2023) - 73% of prediction performance.

    Args:
        bid_volume: Total bid volume (all levels or top N levels)
        ask_volume: Total ask volume (all levels or top N levels)

    Returns:
        OBI series (-1 to 1)
    """
    total_volume = bid_volume + ask_volume
    obi = (bid_volume - ask_volume) / (total_volume + 1e-10)
    return obi


def calculate_vpin(
    volume: pd.Series,
    price_change: pd.Series,
    bucket_size: int = 50,
    window: int = 20
) -> pd.Series:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).

    VPIN measures order flow toxicity and predicts price jumps.

    Based on:
    - Easley et al. (2012): "Flow Toxicity and Liquidity in a HFT World"
    - Abad & Yagüe (2025): VPIN significantly predicts price jumps

    Args:
        volume: Trade volume series
        price_change: Price change series (close - open or tick-to-tick)
        bucket_size: Number of trades per volume bucket
        window: Rolling window for VPIN calculation

    Returns:
        VPIN series (0 to 1)
    """
    # Classify trades as buy or sell based on price change
    buy_volume = volume.where(price_change > 0, 0)
    sell_volume = volume.where(price_change < 0, 0)

    # Volume buckets (simplified - use rolling window instead of fixed buckets)
    rolling_buy = buy_volume.rolling(window=window).sum()
    rolling_sell = sell_volume.rolling(window=window).sum()
    rolling_total = volume.rolling(window=window).sum()

    # VPIN = |buy_volume - sell_volume| / total_volume
    vpin = abs(rolling_buy - rolling_sell) / (rolling_total + 1e-10)

    return vpin.clip(0, 1)


def calculate_effective_spread(
    trade_price: pd.Series,
    mid_price: pd.Series
) -> pd.Series:
    """
    Effective Spread - measure of transaction costs.

    Effective Spread = 2 * |trade_price - mid_price|

    Args:
        trade_price: Actual trade execution price
        mid_price: Mid-point of best bid-ask

    Returns:
        Effective spread series
    """
    return 2 * abs(trade_price - mid_price)


def calculate_price_impact(
    volume: pd.Series,
    price_change: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Price Impact - how much volume moves the price.

    Price Impact = price_change / volume

    Rolling standardized version for better scaling.

    Args:
        volume: Trade volume
        price_change: Price change (can be returns or absolute change)
        window: Rolling window for standardization

    Returns:
        Price impact series
    """
    impact = price_change / (volume + 1e-10)

    # Standardize (z-score) for better comparison across time
    mean = impact.rolling(window=window).mean()
    std = impact.rolling(window=window).std()

    standardized_impact = (impact - mean) / (std + 1e-10)

    return standardized_impact


def calculate_trade_flow_imbalance(
    volume: pd.Series,
    price_change: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Trade Flow Imbalance - net signed volume.

    Signed Volume = volume * sign(price_change)
    TFI = sum(signed_volume) / sum(abs(volume))

    Args:
        volume: Trade volume
        price_change: Price change to determine trade direction
        window: Rolling window

    Returns:
        Trade flow imbalance series (-1 to 1)
    """
    # Sign of price change indicates buy (+1) or sell (-1)
    sign = np.sign(price_change)

    # Signed volume
    signed_volume = volume * sign

    # Rolling sum
    sum_signed = signed_volume.rolling(window=window).sum()
    sum_abs = volume.rolling(window=window).sum()

    tfi = sum_signed / (sum_abs + 1e-10)

    return tfi.clip(-1, 1)


def calculate_bid_ask_spread(
    best_bid: pd.Series,
    best_ask: pd.Series,
    mid_price: Optional[pd.Series] = None
) -> pd.Series:
    """
    Bid-Ask Spread (absolute or relative).

    Absolute Spread = ask - bid
    Relative Spread = (ask - bid) / mid_price

    Args:
        best_bid: Best bid price
        best_ask: Best ask price
        mid_price: Optional mid-price for relative spread

    Returns:
        Spread series (relative if mid_price provided, else absolute)
    """
    spread = best_ask - best_bid

    if mid_price is not None:
        # Relative spread (basis points)
        spread = spread / (mid_price + 1e-10)

    return spread


def calculate_depth_imbalance(
    bid_depth: pd.Series,
    ask_depth: pd.Series,
    level: int = 5
) -> pd.Series:
    """
    Order Book Depth Imbalance at specific level.

    Depth Imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)

    Args:
        bid_depth: Cumulative bid volume up to level
        ask_depth: Cumulative ask volume up to level
        level: Order book level (e.g., 5 for top 5 levels)

    Returns:
        Depth imbalance series (-1 to 1)
    """
    total_depth = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / (total_depth + 1e-10)

    return imbalance.clip(-1, 1)
