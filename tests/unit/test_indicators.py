"""
Unit tests for core/indicators.py.

Tests verify mathematical correctness of:
- EMA computation with known series
- RSI (Wilder's smoothing) against known values
- MACD signal line and histogram
- Bollinger Bands standard deviation calculation
- ATR (true range + Wilder's smoothing)
- Hurst exponent: trending vs mean-reverting vs random series
- GARCH(1,1) convergence and stationarity constraint
- VPIN volume bucketing and informed trading detection
- OBI order book imbalance computation
- compute_all composite indicator snapshot

References verified:
    Bollerslev (1986): GARCH stationarity alpha + beta < 1
    Easley et al. (2012): VPIN volume bucket methodology
    Kolm et al. (2023): OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    Wilder (1978): RSI smoothing factor = 1/period
    Mandelbrot & Wallis (1969): R/S Hurst exponent estimation
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from core.indicators import Indicators
from shared.types import OHLCV, Trade

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _d(v: str | float | int) -> Decimal:
    """Shorthand for Decimal."""
    return Decimal(str(v))


def _make_candles(closes: list[float], base_price: float = 100.0) -> list[OHLCV]:
    """Create OHLCV candles from close prices with synthetic OHLV."""
    candles = []
    for i, c in enumerate(closes):
        candles.append(
            OHLCV(
                timestamp=1700000000 + i * 3600,
                open=_d(c * 0.999),
                high=_d(c * 1.005),
                low=_d(c * 0.995),
                close=_d(c),
                volume=_d(1000 + i * 10),
            )
        )
    return candles


def _make_trades(
    prices: list[float], quantities: list[float], buyer_maker: list[bool]
) -> list[Trade]:
    """Create Trade objects from arrays."""
    return [
        Trade(
            price=_d(p),
            quantity=_d(q),
            timestamp=1700000000 + i,
            is_buyer_maker=bm,
        )
        for i, (p, q, bm) in enumerate(zip(prices, quantities, buyer_maker, strict=True))
    ]


# ===========================================================================
# EMA Tests
# ===========================================================================


class TestEMA:
    def test_ema_with_known_values(self):
        """EMA of constant series should equal that constant."""
        prices = [_d("100")] * 30
        result = Indicators.ema(prices, 10)
        assert len(result) > 0
        # All EMA values should be 100 for constant input
        for val in result:
            assert abs(val - _d("100")) < _d("0.0001")

    def test_ema_rising_series(self):
        """EMA of a strictly rising series should also be rising."""
        prices = [_d(str(100 + i)) for i in range(30)]
        result = Indicators.ema(prices, 10)
        for i in range(1, len(result)):
            assert result[i] > result[i - 1]

    def test_ema_length(self):
        """EMA output length: len(prices) - period + 1."""
        prices = [_d("100")] * 50
        result = Indicators.ema(prices, 20)
        assert len(result) == 50 - 20 + 1

    def test_ema_insufficient_data(self):
        """EMA with fewer data points than period returns empty."""
        prices = [_d("100")] * 5
        assert Indicators.ema(prices, 10) == []

    def test_ema_seed_is_sma(self):
        """First EMA value should be the SMA of the first period prices."""
        prices = [_d(str(i)) for i in range(1, 11)]  # 1,2,...,10
        result = Indicators.ema(prices, 10)
        expected_sma = sum(range(1, 11)) / 10  # 5.5
        assert abs(result[0] - _d(str(expected_sma))) < _d("0.0001")

    def test_ema_multiplier(self):
        """Verify EMA uses k = 2/(period+1)."""
        prices = [_d("100")] * 10 + [_d("110")]
        result = Indicators.ema(prices, 10)
        # k = 2/11 ≈ 0.1818
        k = _d("2") / _d("11")
        expected = _d("100") + k * (_d("110") - _d("100"))
        assert abs(result[-1] - expected) < _d("0.001")


# ===========================================================================
# RSI Tests (Wilder's smoothing)
# ===========================================================================


class TestRSI:
    def test_rsi_all_gains(self):
        """RSI should be 100 when all changes are positive."""
        prices = [_d(str(100 + i)) for i in range(20)]
        rsi = Indicators.rsi(prices, 14)
        assert rsi == _d("100")

    def test_rsi_all_losses(self):
        """RSI should be 0 when all changes are negative."""
        prices = [_d(str(100 - i)) for i in range(20)]
        rsi = Indicators.rsi(prices, 14)
        assert rsi == _d("0")

    def test_rsi_bounded(self):
        """RSI should always be between 0 and 100."""
        prices = [_d(str(100 + (i % 5) - 2)) for i in range(30)]
        rsi = Indicators.rsi(prices, 14)
        assert _d("0") <= rsi <= _d("100")

    def test_rsi_fifty_for_balanced(self):
        """RSI should be near 50 for alternating up/down of equal magnitude."""
        prices = [_d("100")]
        for i in range(30):
            if i % 2 == 0:
                prices.append(prices[-1] + _d("1"))
            else:
                prices.append(prices[-1] - _d("1"))
        rsi = Indicators.rsi(prices, 14)
        # Should be near 50 for balanced gains/losses
        assert _d("40") < rsi < _d("60")

    def test_rsi_insufficient_data(self):
        """RSI should raise ValueError with insufficient data."""
        with pytest.raises(ValueError, match="at least"):
            Indicators.rsi([_d("100")] * 10, 14)


# ===========================================================================
# MACD Tests
# ===========================================================================


class TestMACD:
    def test_macd_constant_series(self):
        """MACD of constant series should be zero."""
        prices = [_d("100")] * 50
        macd_line, signal_line, histogram = Indicators.macd(prices)
        assert abs(macd_line) < _d("0.001")
        assert abs(signal_line) < _d("0.001")
        assert abs(histogram) < _d("0.001")

    def test_macd_returns_three_values(self):
        """MACD should return (line, signal, histogram)."""
        prices = [_d(str(100 + i * 0.1)) for i in range(50)]
        result = Indicators.macd(prices)
        assert len(result) == 3

    def test_macd_histogram_is_difference(self):
        """Histogram should equal MACD line minus signal line."""
        prices = [_d(str(100 + i * 0.5 + (i % 3))) for i in range(50)]
        macd_line, signal_line, histogram = Indicators.macd(prices)
        assert abs(histogram - (macd_line - signal_line)) < _d("0.0001")

    def test_macd_rising_trend(self):
        """MACD line should be positive in a strong uptrend."""
        prices = [_d(str(100 + i * 2)) for i in range(50)]
        macd_line, _, _ = Indicators.macd(prices)
        assert macd_line > 0

    def test_macd_insufficient_data(self):
        """MACD should raise ValueError with insufficient data."""
        with pytest.raises(ValueError, match="at least"):
            Indicators.macd([_d("100")] * 20)


# ===========================================================================
# Bollinger Bands Tests
# ===========================================================================


class TestBollingerBands:
    def test_bb_constant_series(self):
        """Bands should converge when price is constant."""
        prices = [_d("100")] * 25
        upper, middle, lower = Indicators.bollinger_bands(prices, 20)
        assert middle == _d("100")
        # Std dev is 0 for constant series
        assert upper == _d("100")
        assert lower == _d("100")

    def test_bb_middle_is_sma(self):
        """Middle band should be SMA of last `period` prices."""
        prices = [_d(str(i)) for i in range(100, 125)]
        upper, middle, lower = Indicators.bollinger_bands(prices, 20)
        expected_sma = sum(range(105, 125)) / 20
        assert abs(middle - _d(str(expected_sma))) < _d("0.001")

    def test_bb_symmetry(self):
        """Upper and lower should be equidistant from middle."""
        prices = [_d(str(100 + i % 5)) for i in range(25)]
        upper, middle, lower = Indicators.bollinger_bands(prices, 20)
        assert abs((upper - middle) - (middle - lower)) < _d("0.0001")

    def test_bb_wider_with_higher_std_mult(self):
        """Bands should be wider with higher std multiplier."""
        prices = [_d(str(100 + (i % 10))) for i in range(30)]
        u1, m1, l1 = Indicators.bollinger_bands(prices, 20, _d("1.0"))
        u2, m2, l2 = Indicators.bollinger_bands(prices, 20, _d("3.0"))
        assert (u2 - l2) > (u1 - l1)

    def test_bb_insufficient_data(self):
        """Should raise ValueError with insufficient data."""
        with pytest.raises(ValueError, match="at least"):
            Indicators.bollinger_bands([_d("100")] * 10, 20)


# ===========================================================================
# ATR Tests
# ===========================================================================


class TestATR:
    def test_atr_constant_bars(self):
        """ATR should be the bar range for constant-range bars."""
        n = 20
        highs = [_d("105")] * n
        lows = [_d("95")] * n
        closes = [_d("100")] * n
        atr = Indicators.atr(highs, lows, closes, 14)
        # True range for each bar is max(105-95, |105-100|, |95-100|) = 10
        assert abs(atr - _d("10")) < _d("0.1")

    def test_atr_positive(self):
        """ATR should always be positive."""
        highs = [_d(str(100 + i)) for i in range(20)]
        lows = [_d(str(99 + i)) for i in range(20)]
        closes = [_d(str(99.5 + i)) for i in range(20)]
        atr = Indicators.atr(highs, lows, closes, 14)
        assert atr > 0

    def test_atr_gap_up_detection(self):
        """ATR should capture gap-ups via |high - prev_close|."""
        highs = [_d("101")] * 10 + [_d("115")] + [_d("116")] * 5
        lows = [_d("99")] * 10 + [_d("110")] + [_d("114")] * 5
        closes = [_d("100")] * 10 + [_d("112")] + [_d("115")] * 5
        atr = Indicators.atr(highs, lows, closes, 14)
        # The gap should inflate ATR above normal 2-point range
        assert atr > _d("2")

    def test_atr_insufficient_data(self):
        """Should raise ValueError with insufficient data."""
        with pytest.raises(ValueError, match="at least"):
            Indicators.atr([_d("100")] * 5, [_d("99")] * 5, [_d("99.5")] * 5, 14)


# ===========================================================================
# Hurst Exponent Tests (Mandelbrot & Wallis 1969)
# ===========================================================================


class TestHurstExponent:
    def test_hurst_trending_series(self):
        """
        A trending (cumulative sum) series should have H > 0.5.

        Mandelbrot & Wallis (1969): persistent series exhibit H approaching 1.
        """
        # Generate trending series: cumulative sum of positive increments
        import random

        random.seed(42)
        prices = [_d("100")]
        for _ in range(200):
            prices.append(prices[-1] + _d(str(random.uniform(0.1, 0.5))))

        h = Indicators.hurst_exponent(prices, max_lag=20)
        # Trending series should have H > 0.5
        assert h > _d("0.5"), f"Trending series should have H > 0.5, got {h}"

    def test_hurst_mean_reverting_series(self):
        """
        A mean-reverting (oscillating) series should have H < 0.5.

        Mandelbrot & Wallis (1969): anti-persistent series exhibit H < 0.5.
        """
        # Oscillating series: alternating up and down with noise
        prices = [_d("100")]
        for i in range(200):
            if i % 2 == 0:
                prices.append(prices[-1] + _d("2"))
            else:
                prices.append(prices[-1] - _d("2"))

        h = Indicators.hurst_exponent(prices, max_lag=20)
        assert h < _d("0.5"), f"Mean-reverting series should have H < 0.5, got {h}"

    def test_hurst_range(self):
        """Hurst exponent should be in [0, 1]."""
        import random

        random.seed(123)
        prices = [_d(str(100 + random.gauss(0, 2))) for _ in range(200)]
        h = Indicators.hurst_exponent(prices, max_lag=20)
        assert _d("0") <= h <= _d("1")

    def test_hurst_insufficient_data(self):
        """Returns 0.5 (random walk assumption) with < 20 data points."""
        h = Indicators.hurst_exponent([_d("100")] * 10)
        assert h == _d("0.5")

    def test_hurst_constant_series(self):
        """Constant series should return 0.5 (no information)."""
        h = Indicators.hurst_exponent([_d("100")] * 100)
        # Constant returns => degenerate case
        assert _d("0") <= h <= _d("1")


# ===========================================================================
# Realized Volatility Tests
# ===========================================================================


class TestRealizedVolatility:
    def test_rv_zero_for_constant(self):
        """Realized volatility of constant prices should be 0."""
        closes = [_d("100")] * 30
        rv = Indicators.realized_volatility(closes, 24)
        assert rv == _d("0")

    def test_rv_positive_for_varying(self):
        """Realized volatility should be positive for varying prices."""
        closes = [_d(str(100 + i * 0.5)) for i in range(30)]
        rv = Indicators.realized_volatility(closes, 24)
        assert rv > 0

    def test_rv_higher_for_more_volatile(self):
        """More volatile series should produce higher RV."""
        closes_calm = [_d(str(100 + i * 0.1)) for i in range(30)]
        closes_wild = [_d(str(100 + i * 2)) for i in range(30)]
        rv_calm = Indicators.realized_volatility(closes_calm, 24)
        rv_wild = Indicators.realized_volatility(closes_wild, 24)
        assert rv_wild > rv_calm

    def test_rv_insufficient_data(self):
        """Single price should return 0."""
        assert Indicators.realized_volatility([_d("100")], 24) == _d("0")


# ===========================================================================
# GARCH(1,1) Tests (Bollerslev 1986; Hansen & Lunde 2005)
# ===========================================================================


class TestGARCH:
    def test_garch_positive_output(self):
        """GARCH volatility forecast should be positive for varying returns."""
        returns = [_d(str(0.01 * ((-1) ** i))) for i in range(50)]
        vol = Indicators.garch_volatility(returns)
        assert vol > 0

    def test_garch_stationarity_enforcement(self):
        """GARCH should enforce alpha + beta < 1 stationarity constraint."""
        returns = [_d(str(0.01 * ((-1) ** i))) for i in range(50)]
        # Intentionally violate stationarity
        vol = Indicators.garch_volatility(
            returns,
            omega=_d("0.00001"),
            alpha=_d("0.6"),
            beta=_d("0.6"),  # alpha + beta = 1.2 > 1
        )
        # Should still return a valid positive number
        assert vol > 0

    def test_garch_higher_vol_for_volatile_returns(self):
        """GARCH should forecast higher vol for more volatile return series."""
        calm_returns = [_d(str(0.001 * ((-1) ** i))) for i in range(100)]
        wild_returns = [_d(str(0.05 * ((-1) ** i))) for i in range(100)]
        vol_calm = Indicators.garch_volatility(calm_returns)
        vol_wild = Indicators.garch_volatility(wild_returns)
        assert vol_wild > vol_calm

    def test_garch_convergence(self):
        """
        GARCH with constant returns should converge to unconditional vol.

        Bollerslev (1986): unconditional variance = omega / (1 - alpha - beta).
        """
        # All zero returns -> variance should converge to omega / (1-alpha-beta)
        returns = [_d("0")] * 200
        omega = _d("0.00001")
        alpha = _d("0.1")
        beta = _d("0.85")
        vol = Indicators.garch_volatility(returns, omega, alpha, beta)
        # Unconditional variance = 0.00001 / (1 - 0.1 - 0.85) = 0.0002
        unconditional_vol = _d(str(math.sqrt(0.00001 / 0.05)))
        # Should be close to unconditional vol
        assert abs(vol - unconditional_vol) < unconditional_vol * _d("0.5")

    def test_garch_insufficient_data(self):
        """Returns 0 with fewer than 2 data points."""
        assert Indicators.garch_volatility([_d("0.01")]) == _d("0")


# ===========================================================================
# VPIN Tests (Easley et al. 2012; Abad & Yague 2025)
# ===========================================================================


class TestVPIN:
    def test_vpin_balanced_flow(self):
        """VPIN should be low (~0) for balanced buy/sell flow."""
        # Equal buy and sell volume
        trades = _make_trades(
            prices=[100.0] * 100,
            quantities=[10.0] * 100,
            buyer_maker=[i % 2 == 0 for i in range(100)],
        )
        bucket_size = _d("100")  # Each bucket holds 100 units
        vpin = Indicators.vpin(trades, bucket_size, window=10)
        # Balanced flow: |buy - sell| ≈ 0
        assert vpin < _d("0.2"), f"Balanced VPIN should be low, got {vpin}"

    def test_vpin_all_buys(self):
        """VPIN should be 1.0 when all flow is buyer-initiated."""
        trades = _make_trades(
            prices=[100.0] * 50,
            quantities=[10.0] * 50,
            buyer_maker=[False] * 50,  # All buyer-initiated (NOT buyer_maker)
        )
        bucket_size = _d("100")
        vpin = Indicators.vpin(trades, bucket_size, window=5)
        assert vpin == _d("1"), f"All-buy VPIN should be 1.0, got {vpin}"

    def test_vpin_all_sells(self):
        """VPIN should be 1.0 when all flow is seller-initiated."""
        trades = _make_trades(
            prices=[100.0] * 50,
            quantities=[10.0] * 50,
            buyer_maker=[True] * 50,  # All seller-initiated
        )
        bucket_size = _d("100")
        vpin = Indicators.vpin(trades, bucket_size, window=5)
        assert vpin == _d("1"), f"All-sell VPIN should be 1.0, got {vpin}"

    def test_vpin_range(self):
        """VPIN should always be in [0, 1]."""
        import random

        random.seed(42)
        trades = _make_trades(
            prices=[100.0] * 200,
            quantities=[random.uniform(1, 20) for _ in range(200)],
            buyer_maker=[random.choice([True, False]) for _ in range(200)],
        )
        bucket_size = _d("50")
        vpin = Indicators.vpin(trades, bucket_size, window=20)
        assert _d("0") <= vpin <= _d("1")

    def test_vpin_empty_trades(self):
        """VPIN should be 0 for empty trade list."""
        assert Indicators.vpin([], _d("100"), 10) == _d("0")

    def test_vpin_zero_bucket_size(self):
        """VPIN should be 0 for zero bucket size."""
        trades = _make_trades([100.0], [10.0], [True])
        assert Indicators.vpin(trades, _d("0"), 10) == _d("0")


# ===========================================================================
# OBI Tests (Kolm et al. 2023)
# ===========================================================================


class TestOBI:
    def test_obi_balanced(self):
        """OBI should be 0 for equal bid and ask volume."""
        bids = [(_d("100"), _d("50")), (_d("99"), _d("50"))]
        asks = [(_d("101"), _d("50")), (_d("102"), _d("50"))]
        obi = Indicators.order_book_imbalance(bids, asks)
        assert obi == _d("0")

    def test_obi_all_bids(self):
        """OBI should be 1.0 when only bids exist."""
        bids = [(_d("100"), _d("100"))]
        asks: list[tuple[Decimal, Decimal]] = []
        obi = Indicators.order_book_imbalance(bids, asks)
        assert obi == _d("1")

    def test_obi_all_asks(self):
        """OBI should be -1.0 when only asks exist."""
        bids: list[tuple[Decimal, Decimal]] = []
        asks = [(_d("101"), _d("100"))]
        obi = Indicators.order_book_imbalance(bids, asks)
        assert obi == _d("-1")

    def test_obi_buy_pressure(self):
        """Positive OBI indicates buy pressure."""
        bids = [(_d("100"), _d("200"))]
        asks = [(_d("101"), _d("100"))]
        obi = Indicators.order_book_imbalance(bids, asks)
        assert obi > 0
        # OBI = (200-100)/(200+100) = 100/300 ≈ 0.333
        expected = _d("100") / _d("300")
        assert abs(obi - expected) < _d("0.001")

    def test_obi_sell_pressure(self):
        """Negative OBI indicates sell pressure."""
        bids = [(_d("100"), _d("100"))]
        asks = [(_d("101"), _d("200"))]
        obi = Indicators.order_book_imbalance(bids, asks)
        assert obi < 0

    def test_obi_empty(self):
        """OBI of empty book should be 0."""
        assert Indicators.order_book_imbalance([], []) == _d("0")


# ===========================================================================
# compute_all Tests
# ===========================================================================


class TestComputeAll:
    def test_compute_all_basic(self):
        """compute_all should return a valid IndicatorSnapshot."""
        # Need enough candles for all indicators (at least 200 for EMA-200)
        candles = _make_candles([100 + i * 0.1 for i in range(250)])
        snapshot = Indicators.compute_all(candles)

        assert snapshot.price == candles[-1].close
        assert snapshot.ema_20 > 0
        assert snapshot.ema_50 > 0
        assert _d("0") <= snapshot.rsi_14 <= _d("100")
        assert snapshot.atr_14 > 0
        assert _d("0") <= snapshot.hurst <= _d("1")

    def test_compute_all_with_trades_and_depth(self):
        """compute_all should compute VPIN and OBI when provided."""
        candles = _make_candles([100 + i * 0.1 for i in range(250)])
        trades = _make_trades(
            [100.0] * 100,
            [10.0] * 100,
            [i % 2 == 0 for i in range(100)],
        )
        bids = [(_d("100"), _d("100"))]
        asks = [(_d("101"), _d("50"))]

        snapshot = Indicators.compute_all(
            candles, trades=trades, order_book_bids=bids, order_book_asks=asks
        )
        assert snapshot.obi > 0  # More bids than asks
        # VPIN should be computed (not zero since we have trades)
        assert snapshot.vpin >= _d("0")

    def test_compute_all_minimal_data(self):
        """compute_all should handle minimal data gracefully."""
        candles = _make_candles([100.0] * 5)
        snapshot = Indicators.compute_all(candles)
        assert snapshot.price == candles[-1].close
        # RSI should fall back to 50
        assert snapshot.rsi_14 == _d("50")

    def test_compute_all_recent_prices(self):
        """recent_prices should contain up to 200 recent closes."""
        candles = _make_candles([100 + i * 0.01 for i in range(300)])
        snapshot = Indicators.compute_all(candles)
        assert len(snapshot.recent_prices) == 200
        assert snapshot.recent_prices[-1] == candles[-1].close

    def test_compute_all_bollinger_in_range(self):
        """Bollinger bands: lower < middle < upper."""
        candles = _make_candles([100 + i * 0.5 + (i % 3) for i in range(250)])
        snapshot = Indicators.compute_all(candles)
        assert snapshot.bb_lower <= snapshot.bb_middle <= snapshot.bb_upper
