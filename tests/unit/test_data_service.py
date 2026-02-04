"""
Unit tests for core/data_service.py.

Tests verify:
- Binance kline/OHLCV parsing and normalization
- Order book depth parsing for OBI computation
- aggTrades parsing for VPIN computation
- Funding rate fetching and parsing
- Liquidation level computation from subgraph data
- Exchange flow estimation from ticker data
- Per-data-type cache TTL behavior
- Rate limit tracking
- Graceful degradation on network failure

Mock strategy: `_get_json` is patched to return known API responses.
No real API requests are made.
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.types import (
    OHLCV,
    ExchangeFlows,
    OrderBookSnapshot,
    PendingSwapVolume,
    Trade,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _d(v) -> Decimal:
    return Decimal(str(v))


# Mock config to avoid loading real JSON files
MOCK_SIGNALS_CONFIG = {
    "data_source": {
        "symbol": "BNBUSDT",
        "interval": "1h",
        "history_candles": 200,
    },
    "signal_sources": {
        "tier_2": {
            "liquidation_heatmap": {
                "aave_subgraph_url": "",
            }
        }
    },
}

MOCK_RATE_LIMITS = {
    "binance": {
        "spot_max_requests_per_minute": 1200,
        "futures_max_requests_per_minute": 500,
    }
}


@pytest.fixture
def mock_config():
    """Patch ConfigLoader to return mock configs."""
    mock_loader = MagicMock()
    mock_loader.get_signals_config.return_value = MOCK_SIGNALS_CONFIG
    mock_loader.get_rate_limit_config.return_value = MOCK_RATE_LIMITS
    with patch("core.data_service.get_config", return_value=mock_loader):
        yield mock_loader


@pytest.fixture
def data_service(mock_config):
    """Create PriceDataService with mocked config."""
    from core.data_service import PriceDataService

    service = PriceDataService(session=None)
    return service


# ---------------------------------------------------------------------------
# Sample Binance API responses
# ---------------------------------------------------------------------------

SAMPLE_KLINES = [
    [
        1700000000000,  # open_time (ms)
        "310.50",  # open
        "315.20",  # high
        "308.10",  # low
        "312.80",  # close
        "45230.5",  # volume
        1700003599999,  # close_time
        "14123456.78",  # quote volume
        1234,  # trades
        "22615.25",  # taker buy base
        "7061728.39",  # taker buy quote
        "0",  # ignore
    ],
    [
        1700003600000,
        "312.80",
        "318.50",
        "311.00",
        "316.40",
        "52100.3",
        1700007199999,
        "16432190.12",
        1567,
        "26050.15",
        "8216095.06",
        "0",
    ],
]

SAMPLE_DEPTH = {
    "lastUpdateId": 123456789,
    "bids": [
        ["312.50", "100.5"],
        ["312.00", "200.3"],
        ["311.50", "150.0"],
    ],
    "asks": [
        ["313.00", "80.2"],
        ["313.50", "120.8"],
        ["314.00", "90.5"],
    ],
}

SAMPLE_AGG_TRADES = [
    {
        "a": 123456,
        "p": "312.50",
        "q": "1.5",
        "f": 100,
        "l": 100,
        "T": 1700000000000,
        "m": False,  # buyer was taker (buy-initiated)
    },
    {
        "a": 123457,
        "p": "312.40",
        "q": "2.0",
        "f": 101,
        "l": 101,
        "T": 1700000001000,
        "m": True,  # buyer was maker (sell-initiated)
    },
    {
        "a": 123458,
        "p": "312.60",
        "q": "0.8",
        "f": 102,
        "l": 102,
        "T": 1700000002000,
        "m": False,
    },
]

SAMPLE_FUNDING_RATE = [
    {
        "symbol": "BNBUSDT",
        "fundingRate": "0.00032",
        "fundingTime": 1700000000000,
    }
]

SAMPLE_OPEN_INTEREST = {
    "symbol": "BNBUSDT",
    "openInterest": "1234567.89",
}

SAMPLE_TICKER_24H = {
    "symbol": "BTCUSDT",
    "priceChange": "500.00",
    "priceChangePercent": "1.25",
    "quoteVolume": "2400000000.00",
    "volume": "60000.00",
    "lastPrice": "40500.00",
}

SAMPLE_PRICE_TICKER = {
    "symbol": "BNBUSDT",
    "price": "312.50",
}


# ===========================================================================
# OHLCV / Klines Tests
# ===========================================================================


class TestGetOHLCV:
    async def test_parse_binance_klines(self, data_service):
        """Should correctly parse Binance kline response into OHLCV."""
        data_service._get_json = AsyncMock(return_value=SAMPLE_KLINES)

        candles = await data_service.get_ohlcv("BNBUSDT", "1h", 2)

        assert len(candles) == 2
        assert isinstance(candles[0], OHLCV)

        # First candle
        assert candles[0].timestamp == 1700000000  # ms -> s
        assert candles[0].open == _d("310.50")
        assert candles[0].high == _d("315.20")
        assert candles[0].low == _d("308.10")
        assert candles[0].close == _d("312.80")
        assert candles[0].volume == _d("45230.5")

        # Second candle
        assert candles[1].close == _d("316.40")

    async def test_ohlcv_caching(self, data_service):
        """Second call should return cached data without HTTP request."""
        mock_json = AsyncMock(return_value=SAMPLE_KLINES)
        data_service._get_json = mock_json

        # First call - makes request
        result1 = await data_service.get_ohlcv("BNBUSDT", "1h", 2)
        # Second call - should use cache
        result2 = await data_service.get_ohlcv("BNBUSDT", "1h", 2)

        assert result1 == result2
        # Only one call to _get_json
        assert mock_json.call_count == 1

    async def test_ohlcv_empty_response(self, data_service):
        """Should return empty list on empty API response."""
        data_service._get_json = AsyncMock(return_value=[])

        candles = await data_service.get_ohlcv("BNBUSDT", "1h", 200)
        assert candles == []

    async def test_ohlcv_none_response(self, data_service):
        """Should return empty list on None response (HTTP error)."""
        data_service._get_json = AsyncMock(return_value=None)

        candles = await data_service.get_ohlcv("BNBUSDT", "1h", 200)
        assert candles == []

    async def test_ohlcv_malformed_kline_skipped(self, data_service):
        """Should skip malformed klines without crashing."""
        malformed = [
            [1700000000000, "310.50", "315.20"],  # Too few fields
            SAMPLE_KLINES[0],  # Valid
        ]
        data_service._get_json = AsyncMock(return_value=malformed)

        candles = await data_service.get_ohlcv("BNBUSDT", "1h", 2)
        assert len(candles) == 1  # Only the valid one


# ===========================================================================
# Order Book Tests
# ===========================================================================


class TestGetOrderBook:
    async def test_parse_depth(self, data_service):
        """Should correctly parse Binance depth into OrderBookSnapshot."""
        data_service._get_json = AsyncMock(return_value=SAMPLE_DEPTH)

        snapshot = await data_service.get_order_book("BNBUSDT", 20)

        assert isinstance(snapshot, OrderBookSnapshot)
        assert len(snapshot.bids) == 3
        assert len(snapshot.asks) == 3

        # Bids: price, quantity as Decimal
        assert snapshot.bids[0] == (_d("312.50"), _d("100.5"))
        assert snapshot.bids[1] == (_d("312.00"), _d("200.3"))

        # Asks
        assert snapshot.asks[0] == (_d("313.00"), _d("80.2"))

    async def test_depth_caching(self, data_service):
        """Order book should be cached (5s TTL)."""
        mock_json = AsyncMock(return_value=SAMPLE_DEPTH)
        data_service._get_json = mock_json

        r1 = await data_service.get_order_book("BNBUSDT", 20)
        r2 = await data_service.get_order_book("BNBUSDT", 20)

        assert r1.bids == r2.bids
        assert mock_json.call_count == 1

    async def test_depth_empty_on_error(self, data_service):
        """Should return empty snapshot on error."""
        data_service._get_json = AsyncMock(return_value=None)

        snapshot = await data_service.get_order_book("BNBUSDT", 20)
        assert snapshot.bids == []
        assert snapshot.asks == []


# ===========================================================================
# Recent Trades (aggTrades) Tests
# ===========================================================================


class TestGetRecentTrades:
    async def test_parse_agg_trades(self, data_service):
        """Should correctly parse Binance aggTrades into Trade objects."""
        data_service._get_json = AsyncMock(return_value=SAMPLE_AGG_TRADES)

        trades = await data_service.get_recent_trades("BNBUSDT", 1000)

        assert len(trades) == 3
        assert isinstance(trades[0], Trade)

        # First trade: buyer was taker (buy-initiated)
        assert trades[0].price == _d("312.50")
        assert trades[0].quantity == _d("1.5")
        assert trades[0].timestamp == 1700000000
        assert trades[0].is_buyer_maker is False

        # Second trade: buyer was maker (sell-initiated)
        assert trades[1].is_buyer_maker is True

    async def test_trades_caching(self, data_service):
        """Trades should be cached (10s TTL)."""
        mock_json = AsyncMock(return_value=SAMPLE_AGG_TRADES)
        data_service._get_json = mock_json

        r1 = await data_service.get_recent_trades("BNBUSDT", 1000)
        r2 = await data_service.get_recent_trades("BNBUSDT", 1000)

        assert len(r1) == len(r2)
        assert mock_json.call_count == 1

    async def test_trades_empty_on_error(self, data_service):
        """Should return empty list on error."""
        data_service._get_json = AsyncMock(return_value=None)

        trades = await data_service.get_recent_trades("BNBUSDT", 1000)
        assert trades == []


# ===========================================================================
# Funding Rate Tests (Aloosh & Bekaert 2022)
# ===========================================================================


class TestGetFundingRate:
    async def test_parse_funding_rate(self, data_service):
        """Should correctly parse Binance funding rate."""
        data_service._get_json = AsyncMock(return_value=SAMPLE_FUNDING_RATE)

        rate = await data_service.get_funding_rate("BNBUSDT")

        assert rate == _d("0.00032")

    async def test_funding_rate_caching(self, data_service):
        """Funding rate should be cached (300s TTL)."""
        mock_json = AsyncMock(return_value=SAMPLE_FUNDING_RATE)
        data_service._get_json = mock_json

        r1 = await data_service.get_funding_rate("BNBUSDT")
        r2 = await data_service.get_funding_rate("BNBUSDT")

        assert r1 == r2
        assert mock_json.call_count == 1

    async def test_funding_rate_none_on_error(self, data_service):
        """Should return None on error."""
        data_service._get_json = AsyncMock(return_value=None)

        rate = await data_service.get_funding_rate("BNBUSDT")
        assert rate is None

    async def test_funding_rate_none_on_empty(self, data_service):
        """Should return None on empty response."""
        data_service._get_json = AsyncMock(return_value=[])

        rate = await data_service.get_funding_rate("BNBUSDT")
        assert rate is None


# ===========================================================================
# Open Interest Tests
# ===========================================================================


class TestGetOpenInterest:
    async def test_parse_open_interest(self, data_service):
        """Should correctly parse open interest."""
        data_service._get_json = AsyncMock(return_value=SAMPLE_OPEN_INTEREST)

        oi = await data_service.get_open_interest("BNBUSDT")
        assert oi == _d("1234567.89")

    async def test_oi_none_on_error(self, data_service):
        """Should return None on error."""
        data_service._get_json = AsyncMock(return_value=None)

        oi = await data_service.get_open_interest("BNBUSDT")
        assert oi is None


# ===========================================================================
# Current Price Tests
# ===========================================================================


class TestGetCurrentPrice:
    async def test_parse_price(self, data_service):
        """Should correctly parse Binance price ticker."""
        data_service._get_json = AsyncMock(return_value=SAMPLE_PRICE_TICKER)

        price = await data_service.get_current_price("BNBUSDT")
        assert price == _d("312.50")

    async def test_price_zero_on_error(self, data_service):
        """Should return 0 on error."""
        data_service._get_json = AsyncMock(return_value=None)

        price = await data_service.get_current_price("BNBUSDT")
        assert price == _d("0")


# ===========================================================================
# Exchange Flows Tests (Chi et al. 2024)
# ===========================================================================


class TestGetExchangeFlows:
    async def test_parse_exchange_flows(self, data_service):
        """Should estimate exchange flows from 24h ticker data."""
        data_service._get_json = AsyncMock(return_value=SAMPLE_TICKER_24H)

        flows = await data_service.get_exchange_flows("USDT", 60)

        assert isinstance(flows, ExchangeFlows)
        assert flows.avg_hourly_flow > 0
        # Positive price change -> net inflow
        assert flows.inflow_usd > 0

    async def test_flows_zero_on_error(self, data_service):
        """Should return zero flows on error."""
        data_service._get_json = AsyncMock(return_value=None)

        flows = await data_service.get_exchange_flows("USDT", 60)
        assert flows.inflow_usd == _d("0")
        assert flows.outflow_usd == _d("0")


# ===========================================================================
# Liquidation Levels Tests
# ===========================================================================


class TestGetLiquidationLevels:
    async def test_empty_without_subgraph_url(self, data_service):
        """Should return empty list when subgraph URL is not configured."""
        levels = await data_service.get_liquidation_levels("WBNB")
        assert levels == []


# ===========================================================================
# Pending Swap Volume (Mempool) Tests
# ===========================================================================


class TestGetPendingSwapVolume:
    async def test_default_neutral_without_w3(self, data_service):
        """Should return neutral values when no Web3 instance provided."""
        pending = await data_service.get_pending_swap_volume(15)

        assert isinstance(pending, PendingSwapVolume)
        assert pending.net_buy_ratio == _d("0.5")
        assert pending.window_seconds == 15 * 60


# ===========================================================================
# Cache TTL Tests
# ===========================================================================


class TestCacheBehavior:
    async def test_cache_invalidation(self, data_service):
        """Cache entries should expire after TTL."""
        from core.data_service import _CacheEntry

        entry = _CacheEntry(data="test", ttl_seconds=0.01)
        assert entry.is_valid

        await asyncio.sleep(0.02)
        assert not entry.is_valid

    def test_cache_set_and_get(self, data_service):
        """Should store and retrieve from cache."""
        data_service._set_cached("test_key", "test_value", 60)
        assert data_service._get_cached("test_key") == "test_value"

    def test_cache_miss(self, data_service):
        """Should return None for missing cache key."""
        assert data_service._get_cached("nonexistent") is None


# ===========================================================================
# Rate Limit Tests
# ===========================================================================


class TestRateLimiting:
    def test_spot_rate_limit_tracking(self, data_service):
        """Should track spot API requests per minute."""
        # Should be within limit initially
        assert data_service._check_spot_rate_limit() is True

        # Simulate hitting the limit
        data_service._spot_requests_this_minute = 1200
        assert data_service._check_spot_rate_limit() is False

    def test_futures_rate_limit_tracking(self, data_service):
        """Should track futures API requests per minute."""
        assert data_service._check_futures_rate_limit() is True

        data_service._futures_requests_this_minute = 500
        assert data_service._check_futures_rate_limit() is False

    def test_rate_limit_resets_after_minute(self, data_service):
        """Rate limit counter should reset after 60 seconds."""
        data_service._spot_requests_this_minute = 1200
        # Simulate time passing
        data_service._spot_minute_start = time.monotonic() - 61

        assert data_service._check_spot_rate_limit() is True
        assert data_service._spot_requests_this_minute == 1


# ===========================================================================
# Recent Returns Tests
# ===========================================================================


class TestGetRecentReturns:
    async def test_returns_from_ohlcv(self, data_service):
        """Should compute log returns from OHLCV data."""
        klines = [
            [
                1700000000000 + i * 3600000,
                str(100 + i),
                str(101 + i),
                str(99 + i),
                str(100 + i),
                "1000",
                1700003599999 + i * 3600000,
                "100000",
                100,
                "500",
                "50000",
                "0",
            ]
            for i in range(10)
        ]
        data_service._get_json = AsyncMock(return_value=klines)

        returns = await data_service.get_recent_returns("BNBUSDT", 10)
        assert len(returns) == 9  # n-1 returns from n prices
        # All returns should be positive (rising prices)
        for r in returns:
            assert r > 0
