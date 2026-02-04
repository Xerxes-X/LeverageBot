"""
Technical indicator and statistical computation module for BSC Leverage Bot.

Pure computation — no I/O, no side effects. Takes OHLCV arrays and returns
indicator values. All computations use Decimal for precision consistency.

Indicators implemented:
- Standard: EMA, RSI (Wilder's smoothing), MACD, Bollinger Bands, ATR
- Regime/Statistical: Hurst exponent (R/S method), realized volatility,
  GARCH(1,1) one-step-ahead forecast
- Microstructure: VPIN (volume-synchronized probability of informed trading),
  OBI (order book imbalance)

References:
    Bollerslev (1986), "Generalized Autoregressive Conditional Heteroskedasticity",
        Journal of Econometrics.
    Hansen & Lunde (2005), "A Forecast Comparison of Volatility Models",
        Journal of Applied Econometrics.
    Easley, Lopez de Prado & O'Hara (2012), "Flow Toxicity and Liquidity
        in a High-Frequency World", Review of Financial Studies.
    Kolm, Turiel & Westray (2023), "Deep Order Flow Imbalance",
        Journal of Financial Economics.
    Maraj-Mervar & Aybar (2025), "Regime-Adaptive Strategies via Hurst Exponent",
        Fractals and Time Series.
    Wilder (1978), "New Concepts in Technical Trading Systems".
    Abad & Yague (2025), "VPIN as a Predictor of Price Jumps in Crypto",
        ScienceDirect.

Usage:
    from core.indicators import Indicators
    from shared.types import OHLCV, Trade

    candles = [...]  # List[OHLCV]
    ema_values = Indicators.ema([c.close for c in candles], period=20)
    hurst = Indicators.hurst_exponent([c.close for c in candles])
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.types import OHLCV, IndicatorSnapshot, Trade


class Indicators:
    """Static methods for technical indicator and statistical computation."""

    # ------------------------------------------------------------------
    # Standard Technical Indicators
    # ------------------------------------------------------------------

    @staticmethod
    def ema(prices: list[Decimal], period: int) -> list[Decimal]:
        """
        Exponential Moving Average.

        Uses standard multiplier k = 2 / (period + 1). First value is
        initialized with SMA of the first `period` prices.

        Args:
            prices: List of price values (oldest first).
            period: EMA lookback period.

        Returns:
            List of EMA values (same length as input after warm-up).
            Returns empty list if insufficient data.
        """
        if len(prices) < period or period <= 0:
            return []

        k = Decimal("2") / Decimal(str(period + 1))
        one_minus_k = Decimal("1") - k

        # Seed with SMA of first `period` values
        sma = sum(prices[:period]) / Decimal(str(period))
        result = [sma]

        for price in prices[period:]:
            ema_val = price * k + result[-1] * one_minus_k
            result.append(ema_val)

        return result

    @staticmethod
    def rsi(prices: list[Decimal], period: int = 14) -> Decimal:
        """
        Relative Strength Index using Wilder's smoothing method.

        Wilder (1978): uses exponential moving average of gains and losses
        with smoothing factor 1/period (NOT the standard EMA 2/(period+1)).

        Args:
            prices: List of price values (oldest first). Needs period+1 values minimum.
            period: RSI lookback period (default 14).

        Returns:
            RSI value between 0 and 100.

        Raises:
            ValueError: If insufficient data.
        """
        if len(prices) < period + 1:
            raise ValueError(f"RSI requires at least {period + 1} prices, got {len(prices)}")

        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Initial average gain/loss from first `period` changes
        gains = [max(c, Decimal("0")) for c in changes[:period]]
        losses = [max(-c, Decimal("0")) for c in changes[:period]]

        avg_gain = sum(gains) / Decimal(str(period))
        avg_loss = sum(losses) / Decimal(str(period))

        # Wilder's smoothing for remaining changes
        for c in changes[period:]:
            if c > 0:
                avg_gain = (avg_gain * Decimal(str(period - 1)) + c) / Decimal(str(period))
                avg_loss = (avg_loss * Decimal(str(period - 1))) / Decimal(str(period))
            else:
                avg_gain = (avg_gain * Decimal(str(period - 1))) / Decimal(str(period))
                avg_loss = (avg_loss * Decimal(str(period - 1)) + abs(c)) / Decimal(str(period))

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi_val = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))
        return rsi_val

    @staticmethod
    def macd(
        prices: list[Decimal],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[Decimal, Decimal, Decimal]:
        """
        Moving Average Convergence Divergence.

        Args:
            prices: List of price values (oldest first).
            fast: Fast EMA period (default 12).
            slow: Slow EMA period (default 26).
            signal: Signal line EMA period (default 9).

        Returns:
            Tuple of (macd_line, signal_line, histogram).

        Raises:
            ValueError: If insufficient data for computation.
        """
        if len(prices) < slow + signal:
            raise ValueError(f"MACD requires at least {slow + signal} prices, got {len(prices)}")

        fast_ema = Indicators.ema(prices, fast)
        slow_ema = Indicators.ema(prices, slow)

        if not fast_ema or not slow_ema:
            raise ValueError("Insufficient data for MACD EMA computation")

        # Align: fast EMA starts at index (fast-1), slow at (slow-1)
        # MACD line = fast_ema - slow_ema (aligned from the slow start)
        offset = slow - fast
        macd_values = []
        for i in range(len(slow_ema)):
            macd_val = fast_ema[i + offset] - slow_ema[i]
            macd_values.append(macd_val)

        # Signal line = EMA of MACD line
        signal_ema = Indicators.ema(macd_values, signal)

        if not signal_ema:
            raise ValueError("Insufficient data for MACD signal line")

        macd_line = macd_values[-1]
        signal_line = signal_ema[-1]
        histogram = macd_line - signal_line

        return (macd_line, signal_line, histogram)

    @staticmethod
    def bollinger_bands(
        prices: list[Decimal],
        period: int = 20,
        std_mult: Decimal = Decimal("2.0"),
    ) -> tuple[Decimal, Decimal, Decimal]:
        """
        Bollinger Bands (SMA-based with standard deviation bands).

        Args:
            prices: List of price values (oldest first).
            period: SMA lookback period (default 20).
            std_mult: Standard deviation multiplier (default 2.0).

        Returns:
            Tuple of (upper_band, middle_band, lower_band).

        Raises:
            ValueError: If insufficient data.
        """
        if len(prices) < period:
            raise ValueError(
                f"Bollinger Bands requires at least {period} prices, got {len(prices)}"
            )

        window = prices[-period:]
        middle = sum(window) / Decimal(str(period))

        # Population standard deviation over the window
        variance = sum((p - middle) ** 2 for p in window) / Decimal(str(period))
        # Use math.sqrt via float conversion for Decimal sqrt
        std_dev = Decimal(str(math.sqrt(float(variance))))

        upper = middle + std_mult * std_dev
        lower = middle - std_mult * std_dev

        return (upper, middle, lower)

    @staticmethod
    def atr(
        highs: list[Decimal],
        lows: list[Decimal],
        closes: list[Decimal],
        period: int = 14,
    ) -> Decimal:
        """
        Average True Range.

        TR = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = Wilder's smoothed average of TR over `period` bars.

        Args:
            highs: List of high prices.
            lows: List of low prices.
            closes: List of close prices.
            period: ATR period (default 14).

        Returns:
            Current ATR value.

        Raises:
            ValueError: If insufficient or mismatched data.
        """
        n = len(highs)
        if n < period + 1 or len(lows) != n or len(closes) != n:
            raise ValueError(f"ATR requires at least {period + 1} bars with matching arrays")

        # Compute true ranges
        true_ranges = []
        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            true_ranges.append(max(hl, hc, lc))

        # First ATR = simple average of first `period` TRs
        atr_val = sum(true_ranges[:period]) / Decimal(str(period))

        # Wilder's smoothing for remaining TRs
        for tr in true_ranges[period:]:
            atr_val = (atr_val * Decimal(str(period - 1)) + tr) / Decimal(str(period))

        return atr_val

    # ------------------------------------------------------------------
    # Regime & Statistical Indicators
    # ------------------------------------------------------------------

    @staticmethod
    def hurst_exponent(prices: list[Decimal], max_lag: int = 20) -> Decimal:
        """
        Rescaled range (R/S) Hurst exponent estimation.

        Classifies market regime:
            H > 0.55 -> persistent / trending (momentum preferred)
            H < 0.45 -> anti-persistent / mean-reverting
            0.45 <= H <= 0.55 -> random walk

        Maraj-Mervar & Aybar (FracTime 2025): regime-adaptive strategies
        achieve Sharpe 2.10 vs 0.85 for static strategies on crypto markets.

        Implementation follows the classical R/S analysis method
        (Mandelbrot & Wallis, 1969; Hurst, 1951).

        Args:
            prices: List of price values (minimum 100 data points recommended).
            max_lag: Maximum sub-period size for R/S computation (default 20).

        Returns:
            Hurst exponent H in [0, 1].
        """
        if len(prices) < 20:
            return Decimal("0.5")  # Insufficient data -> random walk assumption

        # Convert to log returns
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i] <= 0 or prices[i - 1] <= 0:
                continue
            ratio = float(prices[i]) / float(prices[i - 1])
            if ratio > 0:
                log_returns.append(math.log(ratio))

        if len(log_returns) < 20:
            return Decimal("0.5")

        n = len(log_returns)

        # R/S analysis across multiple sub-period sizes
        # Use lag sizes from 2 to max_lag (or n//2 if smaller)
        lag_sizes = range(2, min(max_lag + 1, n // 2 + 1))
        if len(list(lag_sizes)) < 2:
            return Decimal("0.5")

        log_rs_values = []
        log_n_values = []

        for lag in range(2, min(max_lag + 1, n // 2 + 1)):
            # Split series into non-overlapping sub-periods of size `lag`
            rs_list = []
            for start in range(0, n - lag + 1, lag):
                subseries = log_returns[start : start + lag]
                if len(subseries) < lag:
                    continue

                mean_val = sum(subseries) / len(subseries)

                # Cumulative deviation from mean
                cumdev = []
                running = 0.0
                for val in subseries:
                    running += val - mean_val
                    cumdev.append(running)

                # Range
                r = max(cumdev) - min(cumdev)

                # Standard deviation
                var = sum((x - mean_val) ** 2 for x in subseries) / len(subseries)
                s = math.sqrt(var) if var > 0 else 1e-10

                if s > 0:
                    rs_list.append(r / s)

            if rs_list:
                avg_rs = sum(rs_list) / len(rs_list)
                if avg_rs > 0:
                    log_rs_values.append(math.log(avg_rs))
                    log_n_values.append(math.log(lag))

        if len(log_rs_values) < 2:
            return Decimal("0.5")

        # Linear regression: log(R/S) = H * log(n) + c
        # Slope = Hurst exponent
        n_points = len(log_rs_values)
        mean_x = sum(log_n_values) / n_points
        mean_y = sum(log_rs_values) / n_points

        numerator = sum(
            (log_n_values[i] - mean_x) * (log_rs_values[i] - mean_y) for i in range(n_points)
        )
        denominator = sum((x - mean_x) ** 2 for x in log_n_values)

        if denominator == 0:
            return Decimal("0.5")

        h = numerator / denominator

        # Clamp to valid range [0, 1]
        h = max(0.0, min(1.0, h))

        return Decimal(str(round(h, 6)))

    @staticmethod
    def realized_volatility(closes: list[Decimal], window: int = 24) -> Decimal:
        """
        Annualized realized volatility from log returns.

        Used as GARCH seed and for BTC-BNB volatility spillover computation.

        Args:
            closes: List of close prices.
            window: Number of recent bars to use (default 24 for hourly candles).

        Returns:
            Annualized realized volatility as a Decimal.
        """
        if len(closes) < 2:
            return Decimal("0")

        recent = closes[-window:] if len(closes) >= window else closes

        log_returns = []
        for i in range(1, len(recent)):
            if recent[i] > 0 and recent[i - 1] > 0:
                ratio = float(recent[i]) / float(recent[i - 1])
                if ratio > 0:
                    log_returns.append(math.log(ratio))

        if len(log_returns) < 2:
            return Decimal("0")

        # Variance of log returns
        mean_ret = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_ret) ** 2 for r in log_returns) / (len(log_returns) - 1)

        # Annualize: assuming hourly bars, ~8760 hours/year
        annualized_vol = math.sqrt(variance * 8760)

        return Decimal(str(round(annualized_vol, 8)))

    @staticmethod
    def garch_volatility(
        returns: list[Decimal],
        omega: Decimal = Decimal("0.00001"),
        alpha: Decimal = Decimal("0.1"),
        beta: Decimal = Decimal("0.85"),
    ) -> Decimal:
        """
        GARCH(1,1) one-step-ahead volatility forecast.

        sigma^2_{t+1} = omega + alpha * r^2_t + beta * sigma^2_t

        Bollerslev (1986): foundational heteroskedasticity model.
        Hansen & Lunde (2005): GARCH(1,1) is difficult to beat for
        standard volatility forecasting.

        Stationarity constraint: alpha + beta < 1.

        Args:
            returns: List of log returns (or simple returns).
            omega: Long-run variance weight (default 0.00001).
            alpha: ARCH coefficient — reaction to recent shock (default 0.1).
            beta: GARCH coefficient — persistence (default 0.85).

        Returns:
            One-step-ahead volatility forecast (standard deviation).
        """
        if len(returns) < 2:
            return Decimal("0")

        # Validate stationarity: alpha + beta < 1
        if alpha + beta >= Decimal("1"):
            # Enforce stationarity by scaling down
            total = alpha + beta
            alpha = alpha / total * Decimal("0.99")
            beta = beta / total * Decimal("0.99")

        # Initialize variance with sample variance of returns
        float_returns = [float(r) for r in returns]
        mean_ret = sum(float_returns) / len(float_returns)
        sample_var = sum((r - mean_ret) ** 2 for r in float_returns) / len(float_returns)

        sigma_sq = Decimal(str(max(sample_var, 1e-10)))

        # Iterate GARCH(1,1) recursion
        for ret in returns:
            r_sq = ret**2
            sigma_sq = omega + alpha * r_sq + beta * sigma_sq

        # One-step-ahead: already computed as the last sigma_sq
        # Return standard deviation (sqrt of variance)
        vol = Decimal(str(math.sqrt(max(float(sigma_sq), 0))))

        return vol

    # ------------------------------------------------------------------
    # Microstructure Indicators
    # ------------------------------------------------------------------

    @staticmethod
    def vpin(
        trades: list[Trade],
        bucket_size: Decimal,
        window: int = 50,
    ) -> Decimal:
        """
        Volume-Synchronized Probability of Informed Trading.

        Easley, Lopez de Prado & O'Hara (2012): VPIN measures flow
        toxicity as mean(|V_buy - V_sell| / V_total) over N volume
        buckets. Buy/sell classification via tick rule.

        Abad & Yague (2025): VPIN significantly predicts crypto price jumps.

        Args:
            trades: List of Trade objects (price, quantity, is_buyer_maker).
            bucket_size: Volume per bucket (e.g., avg_daily_volume / 50).
            window: Number of recent buckets for VPIN computation (default 50).

        Returns:
            VPIN value in [0, 1]. Higher values indicate more informed trading.
        """
        if not trades or bucket_size <= 0:
            return Decimal("0")

        # Classify trades and fill volume buckets
        buckets: list[tuple[Decimal, Decimal]] = []  # (buy_vol, sell_vol)
        current_buy = Decimal("0")
        current_sell = Decimal("0")
        current_total = Decimal("0")

        for trade in trades:
            vol = trade.quantity
            # Tick rule: is_buyer_maker=True means the trade was sell-initiated
            # (the buyer was the passive maker, so the seller aggressed)
            if trade.is_buyer_maker:
                current_sell += vol
            else:
                current_buy += vol

            current_total += vol

            # When bucket is full, record and start new one
            while current_total >= bucket_size:
                _overflow = current_total - bucket_size

                # Pro-rate the overflow back
                if current_total > 0:
                    ratio = bucket_size / (current_total)
                    bucket_buy = current_buy * ratio
                    bucket_sell = current_sell * ratio
                else:
                    bucket_buy = current_buy
                    bucket_sell = current_sell

                buckets.append((bucket_buy, bucket_sell))

                # Carry over the overflow
                remaining_buy = current_buy - bucket_buy
                remaining_sell = current_sell - bucket_sell
                current_buy = max(remaining_buy, Decimal("0"))
                current_sell = max(remaining_sell, Decimal("0"))
                current_total = current_buy + current_sell

        if not buckets:
            return Decimal("0")

        # Use the most recent `window` buckets
        recent_buckets = buckets[-window:]

        # VPIN = mean(|V_buy - V_sell| / V_total) over window
        vpin_sum = Decimal("0")
        valid_buckets = 0

        for buy_vol, sell_vol in recent_buckets:
            total = buy_vol + sell_vol
            if total > 0:
                imbalance = abs(buy_vol - sell_vol) / total
                vpin_sum += imbalance
                valid_buckets += 1

        if valid_buckets == 0:
            return Decimal("0")

        return vpin_sum / Decimal(str(valid_buckets))

    @staticmethod
    def order_book_imbalance(
        bids: list[tuple[Decimal, Decimal]],
        asks: list[tuple[Decimal, Decimal]],
    ) -> Decimal:
        """
        Order Book Imbalance (OBI).

        OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        Kolm, Turiel & Westray (2023): OBI accounts for 73% of
        short-term price prediction performance.

        Args:
            bids: List of (price, quantity) tuples, sorted price descending.
            asks: List of (price, quantity) tuples, sorted price ascending.

        Returns:
            OBI value in [-1, 1]. Positive = buy pressure, negative = sell pressure.
        """
        bid_volume = sum(qty for _, qty in bids) if bids else Decimal("0")
        ask_volume = sum(qty for _, qty in asks) if asks else Decimal("0")

        total = bid_volume + ask_volume
        if total == 0:
            return Decimal("0")

        return Decimal(bid_volume - ask_volume) / Decimal(total)

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------

    @staticmethod
    def compute_all(
        candles: list[OHLCV],
        trades: list[Trade] | None = None,
        order_book_bids: list[tuple[Decimal, Decimal]] | None = None,
        order_book_asks: list[tuple[Decimal, Decimal]] | None = None,
        ema_fast: int = 20,
        ema_slow: int = 50,
        ema_trend: int = 200,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: Decimal = Decimal("2.0"),
        atr_period: int = 14,
        hurst_max_lag: int = 20,
        vpin_bucket_divisor: int = 50,
        vpin_window: int = 50,
    ) -> IndicatorSnapshot:
        """
        Compute all indicators from OHLCV data and optional microstructure data.

        Aggregates standard technical indicators, regime statistics, and
        microstructure signals into a single IndicatorSnapshot.

        Args:
            candles: List of OHLCV bars (oldest first).
            trades: Optional list of Trade objects for VPIN computation.
            order_book_bids: Optional bid side for OBI.
            order_book_asks: Optional ask side for OBI.
            (remaining args): Indicator configuration parameters.

        Returns:
            IndicatorSnapshot containing all computed values.
        """
        from shared.types import IndicatorSnapshot as IndSnap

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]

        # EMA
        ema_fast_vals = Indicators.ema(closes, ema_fast)
        ema_slow_vals = Indicators.ema(closes, ema_slow)
        ema_trend_vals = Indicators.ema(closes, ema_trend)

        ema_20 = ema_fast_vals[-1] if ema_fast_vals else closes[-1]
        ema_50 = ema_slow_vals[-1] if ema_slow_vals else closes[-1]
        ema_200 = ema_trend_vals[-1] if ema_trend_vals else closes[-1]

        # RSI
        try:
            rsi_val = Indicators.rsi(closes, rsi_period)
        except ValueError:
            rsi_val = Decimal("50")

        # MACD
        try:
            macd_line, macd_sig, macd_hist = Indicators.macd(
                closes, macd_fast, macd_slow, macd_signal
            )
        except ValueError:
            macd_line = macd_sig = macd_hist = Decimal("0")

        # Bollinger Bands
        try:
            bb_upper, bb_middle, bb_lower = Indicators.bollinger_bands(closes, bb_period, bb_std)
        except ValueError:
            bb_upper = bb_middle = bb_lower = closes[-1]

        # ATR
        try:
            atr_val = Indicators.atr(highs, lows, closes, atr_period)
        except ValueError:
            atr_val = Decimal("0")

        # ATR ratio: current ATR / 50-period average ATR
        # Compute rolling ATR over last 50 periods for ratio
        if len(candles) >= atr_period + 50:
            atr_history = []
            for i in range(50):
                end = len(candles) - 49 + i
                start = max(0, end - atr_period - 1)
                try:
                    h_atr = Indicators.atr(
                        highs[start:end], lows[start:end], closes[start:end], atr_period
                    )
                    atr_history.append(h_atr)
                except ValueError:
                    pass
            atr_50_avg = (
                sum(atr_history) / Decimal(str(len(atr_history))) if atr_history else atr_val
            )
            atr_ratio = atr_val / atr_50_avg if atr_50_avg > 0 else Decimal("1.0")
        else:
            atr_ratio = Decimal("1.0")

        # Volume average
        vol_window = min(20, len(volumes))
        volume_20_avg = (
            sum(volumes[-vol_window:]) / Decimal(str(vol_window))
            if vol_window > 0
            else Decimal("0")
        )

        # Hurst exponent
        hurst = Indicators.hurst_exponent(closes, hurst_max_lag)

        # VPIN
        if trades:
            avg_daily_vol = sum(t.quantity for t in trades)
            bucket_size = (
                avg_daily_vol / Decimal(str(vpin_bucket_divisor))
                if vpin_bucket_divisor > 0
                else Decimal("1")
            )
            vpin_val = Indicators.vpin(trades, bucket_size, vpin_window)
        else:
            vpin_val = Decimal("0")

        # OBI
        if order_book_bids is not None and order_book_asks is not None:
            obi_val = Indicators.order_book_imbalance(order_book_bids, order_book_asks)
        else:
            obi_val = Decimal("0")

        # Recent prices for downstream Hurst re-computation
        recent_prices = closes[-200:] if len(closes) >= 200 else closes[:]

        return IndSnap(
            price=closes[-1],
            ema_20=ema_20,
            ema_50=ema_50,
            ema_200=ema_200,
            rsi_14=rsi_val,
            macd_line=macd_line,
            macd_signal=macd_sig,
            macd_histogram=macd_hist,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            atr_14=atr_val,
            atr_ratio=atr_ratio,
            volume=volumes[-1] if volumes else Decimal("0"),
            volume_20_avg=volume_20_avg,
            hurst=hurst,
            vpin=vpin_val,
            obi=obi_val,
            recent_prices=recent_prices,
        )
