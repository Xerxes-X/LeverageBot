"""
Multi-source market data service for BSC Leverage Bot.

Fetches, caches, and normalizes market data from multiple sources for the
multi-source signal pipeline. Chainlink alone is insufficient — it provides
only the latest spot price with 27-60s resolution, no volume data, and no
OHLCV history. The signal architecture requires order book depth, recent
trades, funding rates, exchange flow proxies, and liquidation level data.

Data Sources (ordered by priority):
    - Binance Spot API (primary): klines, depth, aggTrades
    - Binance Futures API: funding rates, open interest
    - Aave V3 Subgraph: liquidation level distribution
    - BSC RPC: mempool pending swap volume (Tier 3, limited by PBS)

Caching (in-memory LRU with per-data-type TTL):
    - OHLCV: 60s (1h candles), 30s (15m candles)
    - Order book: 5s
    - Recent trades: 10s
    - Funding rate: 300s (updates every 8h)
    - Liquidation levels: 300s
    - Exchange flows: 120s

References:
    Binance API docs: https://binance-docs.github.io/apidocs/spot/en/
    Chi et al. (2024), "Exchange Flows and Crypto Returns", SSRN.
    Perez et al. (2021), "Liquidations: DeFi on a Knife-Edge", FC 2021.
    Ante & Saggu (2024), "Mempool & Volume Prediction",
        Journal of Innovation & Knowledge.

Usage:
    data_service = PriceDataService(session)
    candles = await data_service.get_ohlcv("BNBUSDT", "1h", limit=200)
    depth = await data_service.get_order_book("BNBUSDT", limit=20)
"""

from __future__ import annotations

import os
import time
from decimal import Decimal
from typing import Any, cast

import aiohttp

from bot_logging.logger_manager import setup_module_logger
from config.loader import get_config
from shared.types import (
    OHLCV,
    ExchangeFlows,
    LiquidationLevel,
    OrderBookSnapshot,
    PendingSwapVolume,
    Trade,
)

# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


class _CacheEntry:
    """In-memory cache entry with TTL."""

    __slots__ = ("data", "expires_at")

    def __init__(self, data: Any, ttl_seconds: float) -> None:
        self.data = data
        self.expires_at = time.monotonic() + ttl_seconds

    @property
    def is_valid(self) -> bool:
        return time.monotonic() < self.expires_at


# ---------------------------------------------------------------------------
# TTL constants (seconds) per data type
# ---------------------------------------------------------------------------

_TTL_OHLCV_1H = 60
_TTL_OHLCV_OTHER = 30
_TTL_ORDER_BOOK = 5
_TTL_RECENT_TRADES = 10
_TTL_FUNDING_RATE = 300
_TTL_OPEN_INTEREST = 60
_TTL_LIQUIDATION_LEVELS = 300
_TTL_EXCHANGE_FLOWS = 120
_TTL_PENDING_SWAP = 60
_TTL_CURRENT_PRICE = 5


# ---------------------------------------------------------------------------
# Binance API base URLs
# ---------------------------------------------------------------------------

_BINANCE_SPOT_BASE = "https://api.binance.com"
_BINANCE_FUTURES_BASE = "https://fapi.binance.com"


class PriceDataService:
    """
    Async multi-source market data fetcher with per-data-type caching.

    All public methods return typed dataclasses from shared/types.py.
    Network failures return empty/zero-value defaults rather than raising,
    allowing the signal engine to degrade gracefully when data is unavailable.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession | None = None,
        w3: Any | None = None,
    ) -> None:
        """
        Args:
            session: Shared aiohttp session (created internally if None).
            w3: AsyncWeb3 instance for on-chain queries (optional).
        """
        self._session = session
        self._owns_session = session is None
        self._w3 = w3

        cfg = get_config()
        self._signals_config = cfg.get_signals_config()
        self._rate_limits = cfg.get_rate_limit_config()

        data_source = self._signals_config.get("data_source", {})
        self._primary_symbol = data_source.get("symbol", "BNBUSDT")

        # API keys (optional — Binance public endpoints don't require auth)
        self._binance_api_key = os.getenv("BINANCE_API_KEY", "")

        # In-memory cache
        self._cache: dict[str, _CacheEntry] = {}

        # Request tracking for rate limiting
        self._spot_requests_this_minute: int = 0
        self._spot_minute_start: float = time.monotonic()
        self._futures_requests_this_minute: int = 0
        self._futures_minute_start: float = time.monotonic()

        self._logger = setup_module_logger(
            "data_service", "data_service.log", module_folder="Data_Service_Logs"
        )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"X-MBX-APIKEY": self._binance_api_key} if self._binance_api_key else {},
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cached(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry and entry.is_valid:
            self._logger.debug("Cache hit: %s", key)
            return entry.data
        return None

    def _set_cached(self, key: str, data: Any, ttl: float) -> None:
        self._cache[key] = _CacheEntry(data, ttl)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_spot_rate_limit(self) -> bool:
        now = time.monotonic()
        if now - self._spot_minute_start > 60:
            self._spot_requests_this_minute = 0
            self._spot_minute_start = now

        max_rpm = self._rate_limits.get("binance", {}).get("spot_max_requests_per_minute", 1200)
        if self._spot_requests_this_minute >= max_rpm:
            self._logger.warning("Binance spot rate limit reached (%d/min)", max_rpm)
            return False
        self._spot_requests_this_minute += 1
        return True

    def _check_futures_rate_limit(self) -> bool:
        now = time.monotonic()
        if now - self._futures_minute_start > 60:
            self._futures_requests_this_minute = 0
            self._futures_minute_start = now

        max_rpm = self._rate_limits.get("binance", {}).get("futures_max_requests_per_minute", 500)
        if self._futures_requests_this_minute >= max_rpm:
            self._logger.warning("Binance futures rate limit reached (%d/min)", max_rpm)
            return False
        self._futures_requests_this_minute += 1
        return True

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _get_json(self, url: str, params: dict[str, str] | None = None) -> Any:
        """Fetch JSON from URL with error handling."""
        session = await self._get_session()
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    self._logger.warning("Rate limited by %s", url)
                    return None
                if resp.status != 200:
                    self._logger.warning(
                        "HTTP %d from %s: %s",
                        resp.status,
                        url,
                        await resp.text(),
                    )
                    return None
                return await resp.json()
        except (aiohttp.ClientError, TimeoutError) as e:
            self._logger.warning("Request failed for %s: %s", url, e)
            return None

    # ------------------------------------------------------------------
    # Core OHLCV (Binance klines)
    # ------------------------------------------------------------------

    async def get_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 200) -> list[OHLCV]:
        """
        Fetch OHLCV candlestick data from Binance spot API.

        Binance klines endpoint returns arrays of:
        [open_time, open, high, low, close, volume, close_time, ...]

        Args:
            symbol: Trading pair (e.g., "BNBUSDT").
            interval: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d").
            limit: Number of candles (max 1000).

        Returns:
            List of OHLCV dataclass instances, oldest first.
        """
        cache_key = f"ohlcv:{symbol}:{interval}:{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cast(list[OHLCV], cached)

        if not self._check_spot_rate_limit():
            return []

        url = f"{_BINANCE_SPOT_BASE}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": str(limit)}

        data = await self._get_json(url, params)
        if not data or not isinstance(data, list):
            return []

        candles = []
        for k in data:
            try:
                candles.append(
                    OHLCV(
                        timestamp=int(k[0]) // 1000,  # ms -> s
                        open=Decimal(str(k[1])),
                        high=Decimal(str(k[2])),
                        low=Decimal(str(k[3])),
                        close=Decimal(str(k[4])),
                        volume=Decimal(str(k[5])),
                    )
                )
            except (IndexError, ValueError, TypeError) as e:
                self._logger.debug("Skipping malformed kline: %s", e)

        ttl = _TTL_OHLCV_1H if interval == "1h" else _TTL_OHLCV_OTHER
        self._set_cached(cache_key, candles, ttl)

        return candles

    async def get_current_price(self, symbol: str) -> Decimal:
        """
        Get current price from Binance ticker.

        Args:
            symbol: Trading pair (e.g., "BNBUSDT").

        Returns:
            Current price as Decimal. Returns 0 on failure.
        """
        cache_key = f"price:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cast(Decimal, cached)

        if not self._check_spot_rate_limit():
            return Decimal("0")

        url = f"{_BINANCE_SPOT_BASE}/api/v3/ticker/price"
        params = {"symbol": symbol}

        data = await self._get_json(url, params)
        if not data or "price" not in data:
            return Decimal("0")

        price = Decimal(str(data["price"]))
        self._set_cached(cache_key, price, _TTL_CURRENT_PRICE)
        return price

    # ------------------------------------------------------------------
    # Order Book (for OBI signal — Kolm et al. 2023)
    # ------------------------------------------------------------------

    async def get_order_book(self, symbol: str, limit: int = 20) -> OrderBookSnapshot:
        """
        Fetch order book depth from Binance.

        Kolm, Turiel & Westray (2023): order book imbalance accounts
        for 73% of short-term price prediction performance.

        Args:
            symbol: Trading pair.
            limit: Number of price levels per side (5, 10, 20, 50, 100, 500, 1000).

        Returns:
            OrderBookSnapshot with bid/ask arrays.
        """
        cache_key = f"depth:{symbol}:{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cast(OrderBookSnapshot, cached)

        if not self._check_spot_rate_limit():
            return OrderBookSnapshot(bids=[], asks=[], timestamp=int(time.time()))

        url = f"{_BINANCE_SPOT_BASE}/api/v3/depth"
        params = {"symbol": symbol, "limit": str(limit)}

        data = await self._get_json(url, params)
        if not data:
            return OrderBookSnapshot(bids=[], asks=[], timestamp=int(time.time()))

        bids = [(Decimal(str(price)), Decimal(str(qty))) for price, qty in data.get("bids", [])]
        asks = [(Decimal(str(price)), Decimal(str(qty))) for price, qty in data.get("asks", [])]

        snapshot = OrderBookSnapshot(bids=bids, asks=asks, timestamp=int(time.time()))
        self._set_cached(cache_key, snapshot, _TTL_ORDER_BOOK)
        return snapshot

    # ------------------------------------------------------------------
    # Recent Trades (for VPIN — Easley et al. 2012, Abad & Yague 2025)
    # ------------------------------------------------------------------

    async def get_recent_trades(self, symbol: str, limit: int = 1000) -> list[Trade]:
        """
        Fetch aggregated recent trades from Binance for VPIN computation.

        Easley, Lopez de Prado & O'Hara (2012): VPIN uses volume-bucketed
        trade flow to measure informed trading probability.

        Args:
            symbol: Trading pair.
            limit: Number of trades (max 1000).

        Returns:
            List of Trade objects.
        """
        cache_key = f"trades:{symbol}:{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cast(list[Trade], cached)

        if not self._check_spot_rate_limit():
            return []

        url = f"{_BINANCE_SPOT_BASE}/api/v3/aggTrades"
        params = {"symbol": symbol, "limit": str(limit)}

        data = await self._get_json(url, params)
        if not data or not isinstance(data, list):
            return []

        trades = []
        for t in data:
            try:
                trades.append(
                    Trade(
                        price=Decimal(str(t["p"])),
                        quantity=Decimal(str(t["q"])),
                        timestamp=int(t["T"]) // 1000,  # ms -> s
                        is_buyer_maker=bool(t["m"]),
                    )
                )
            except (KeyError, ValueError, TypeError) as e:
                self._logger.debug("Skipping malformed trade: %s", e)

        self._set_cached(cache_key, trades, _TTL_RECENT_TRADES)
        return trades

    # ------------------------------------------------------------------
    # Derivatives Data (Binance Futures)
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> Decimal | None:
        """
        Get latest funding rate from Binance perpetual futures.

        Aloosh & Bekaert (2022): funding rates explain 12.5% of price
        variation over 7-day horizons. Extreme positive funding -> contrarian
        short signal; extreme negative -> contrarian long.

        Args:
            symbol: Perpetual pair (e.g., "BNBUSDT").

        Returns:
            Latest funding rate as Decimal, or None if unavailable.
        """
        cache_key = f"funding:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cast(Decimal, cached)

        if not self._check_futures_rate_limit():
            return None

        url = f"{_BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": "1"}

        data = await self._get_json(url, params)
        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        try:
            rate = Decimal(str(data[0]["fundingRate"]))
        except (KeyError, ValueError, TypeError):
            return None

        self._set_cached(cache_key, rate, _TTL_FUNDING_RATE)
        return rate

    async def get_open_interest(self, symbol: str) -> Decimal | None:
        """
        Get open interest from Binance perpetual futures.

        High open interest combined with extreme funding indicates
        crowded positioning — useful for contrarian signals.

        Args:
            symbol: Perpetual pair.

        Returns:
            Open interest in base asset units, or None if unavailable.
        """
        cache_key = f"oi:{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cast(Decimal, cached)

        if not self._check_futures_rate_limit():
            return None

        url = f"{_BINANCE_FUTURES_BASE}/fapi/v1/openInterest"
        params = {"symbol": symbol}

        data = await self._get_json(url, params)
        if not data or "openInterest" not in data:
            return None

        try:
            oi = Decimal(str(data["openInterest"]))
        except (ValueError, TypeError):
            return None

        self._set_cached(cache_key, oi, _TTL_OPEN_INTEREST)
        return oi

    # ------------------------------------------------------------------
    # DeFi / On-Chain: Liquidation Levels
    # (Perez et al. 2021: "3% price variations make >$10M liquidatable")
    # ------------------------------------------------------------------

    async def get_liquidation_levels(self, asset: str) -> list[LiquidationLevel]:
        """
        Query Aave V3 subgraph for position health factor distribution
        and compute liquidation price levels.

        Perez et al. (FC 2021): showed that small price variations (3%)
        can make >$10M in DeFi positions liquidatable, creating cascading
        sell pressure.

        The subgraph query fetches positions with HF < 2.0 (most at risk)
        and computes the price level at which each would be liquidated.
        Positions are bucketed by liquidation price to identify walls.

        Args:
            asset: Asset symbol (e.g., "WBNB").

        Returns:
            List of LiquidationLevel sorted by collateral at risk (descending).
        """
        cache_key = f"liq_levels:{asset}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cast(list[LiquidationLevel], cached)

        subgraph_url = (
            self._signals_config.get("signal_sources", {})
            .get("tier_2", {})
            .get("liquidation_heatmap", {})
            .get("aave_subgraph_url", "")
        )

        if not subgraph_url:
            # Subgraph URL not configured — return empty
            self._logger.debug("Aave subgraph URL not configured, skipping liquidation levels")
            return []

        # GraphQL query for Aave V3 positions with low health factors
        query = """
        {
            users(
                where: { borrowedReservesCount_gt: 0 }
                first: 100
                orderBy: id
            ) {
                id
                reserves(where: { currentATokenBalance_gt: "0" }) {
                    currentATokenBalance
                    reserve {
                        symbol
                        price {
                            priceInEth
                        }
                        decimals
                        reserveLiquidationThreshold
                    }
                }
                borrows: reserves(where: { currentVariableDebt_gt: "0" }) {
                    currentVariableDebt
                    reserve {
                        symbol
                        price {
                            priceInEth
                        }
                        decimals
                    }
                }
            }
        }
        """

        session = await self._get_session()
        try:
            async with session.post(
                subgraph_url,
                json={"query": query},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    self._logger.warning("Aave subgraph returned %d", resp.status)
                    return []
                result = await resp.json()
        except (aiohttp.ClientError, TimeoutError) as e:
            self._logger.warning("Aave subgraph query failed: %s", e)
            return []

        users = result.get("data", {}).get("users", [])
        if not users:
            return []

        # Bucket liquidation prices
        levels: dict[int, LiquidationLevel] = {}
        for user in users:
            liq_price = self._estimate_user_liquidation_price(user, asset)
            if liq_price is None or liq_price <= 0:
                continue

            collateral_usd = self._estimate_user_collateral_usd(user)
            bucket = int(liq_price / 10) * 10  # Round to nearest $10

            if bucket in levels:
                existing = levels[bucket]
                levels[bucket] = LiquidationLevel(
                    price=Decimal(str(bucket)),
                    total_collateral_at_risk_usd=(
                        existing.total_collateral_at_risk_usd + collateral_usd
                    ),
                    position_count=existing.position_count + 1,
                )
            else:
                levels[bucket] = LiquidationLevel(
                    price=Decimal(str(bucket)),
                    total_collateral_at_risk_usd=collateral_usd,
                    position_count=1,
                )

        result_list = sorted(
            levels.values(),
            key=lambda lv: lv.total_collateral_at_risk_usd,
            reverse=True,
        )

        self._set_cached(cache_key, result_list, _TTL_LIQUIDATION_LEVELS)
        return result_list

    def _estimate_user_liquidation_price(
        self,
        user: dict[str, Any],
        target_asset: str,
    ) -> Decimal | None:
        """
        Estimate the price at which a user's position becomes liquidatable.

        For long positions: liq_price = debt / (collateral_qty * LT)
        For short positions: liq_price = (collateral_usd * LT) / debt_qty
        """
        reserves = user.get("reserves", [])
        borrows = user.get("borrows", [])

        if not reserves or not borrows:
            return None

        # Find if user has exposure to target asset
        total_collateral_eth = Decimal("0")
        total_debt_eth = Decimal("0")
        weighted_lt = Decimal("0")

        for r in reserves:
            reserve = r.get("reserve", {})
            balance = Decimal(str(r.get("currentATokenBalance", "0")))
            decimals = int(reserve.get("decimals", 18))
            price_eth = Decimal(str(reserve.get("price", {}).get("priceInEth", "0")))
            lt = Decimal(str(reserve.get("reserveLiquidationThreshold", "0")))

            value_eth = balance / Decimal(str(10**decimals)) * price_eth
            total_collateral_eth += value_eth
            weighted_lt += value_eth * lt / Decimal("10000")

        for b in borrows:
            reserve = b.get("reserve", {})
            debt = Decimal(str(b.get("currentVariableDebt", "0")))
            decimals = int(reserve.get("decimals", 18))
            price_eth = Decimal(str(reserve.get("price", {}).get("priceInEth", "0")))

            value_eth = debt / Decimal(str(10**decimals)) * price_eth
            total_debt_eth += value_eth

        if total_collateral_eth <= 0 or total_debt_eth <= 0:
            return None

        avg_lt = weighted_lt / total_collateral_eth if total_collateral_eth > 0 else Decimal("0.8")

        # HF = (collateral * LT) / debt
        # HF = 1 when price drops to liq_price
        # For simplicity: liq_price_factor = debt / (collateral * LT)
        # The actual price at liquidation = current_price * liq_price_factor
        hf = (
            (total_collateral_eth * avg_lt) / total_debt_eth
            if total_debt_eth > 0
            else Decimal("999")
        )

        if hf >= Decimal("2"):
            return None  # Far from liquidation, skip

        # Liquidation price = current_price / HF (simplified)
        return hf

    def _estimate_user_collateral_usd(self, user: dict[str, Any]) -> Decimal:
        """Estimate total collateral USD value from subgraph data."""
        total = Decimal("0")
        for r in user.get("reserves", []):
            reserve = r.get("reserve", {})
            balance = Decimal(str(r.get("currentATokenBalance", "0")))
            decimals = int(reserve.get("decimals", 18))
            price_eth = Decimal(str(reserve.get("price", {}).get("priceInEth", "0")))
            # Approximate USD: 1 ETH ~ $2000 (rough estimate for bucketing)
            value_usd = balance / Decimal(str(10**decimals)) * price_eth * Decimal("2000")
            total += value_usd
        return total

    # ------------------------------------------------------------------
    # Exchange Flows (Chi et al. 2024)
    # ------------------------------------------------------------------

    async def get_exchange_flows(self, token: str, window_minutes: int = 60) -> ExchangeFlows:
        """
        Monitor token flows to/from known exchange wallets on BSC.

        Chi et al. (2024): USDT net inflows to exchanges positively
        predict BTC/ETH returns (buying power arriving). USDT outflows
        indicate capital withdrawal (bearish).

        Implementation uses Binance 24h ticker stats as a volume proxy
        for exchange flow estimation, since direct on-chain flow tracking
        of known hot wallets would require BSCScan API integration.

        Args:
            token: Token symbol (e.g., "USDT", "WBNB").
            window_minutes: Flow aggregation window.

        Returns:
            ExchangeFlows with inflow/outflow estimates.
        """
        cache_key = f"flows:{token}:{window_minutes}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cast(ExchangeFlows, cached)

        # Use Binance 24h ticker as proxy for exchange flow
        symbol = f"{token}USDT" if token != "USDT" else "BTCUSDT"

        if not self._check_spot_rate_limit():
            return ExchangeFlows(
                inflow_usd=Decimal("0"),
                outflow_usd=Decimal("0"),
                avg_hourly_flow=Decimal("0"),
                data_age_seconds=0,
            )

        url = f"{_BINANCE_SPOT_BASE}/api/v3/ticker/24hr"
        params = {"symbol": symbol}

        data = await self._get_json(url, params)
        if not data:
            return ExchangeFlows(
                inflow_usd=Decimal("0"),
                outflow_usd=Decimal("0"),
                avg_hourly_flow=Decimal("0"),
                data_age_seconds=0,
            )

        try:
            quote_volume = Decimal(str(data.get("quoteVolume", "0")))
            price_change_pct = Decimal(str(data.get("priceChangePercent", "0")))

            # Estimate flow direction from price change and volume
            # Positive price change + high volume -> net inflow
            # Negative price change + high volume -> net outflow
            hourly_volume = quote_volume / Decimal("24")

            if price_change_pct > 0:
                inflow = hourly_volume * (window_minutes / Decimal("60"))
                outflow = inflow * Decimal("0.7")  # Rough estimate
            else:
                outflow = hourly_volume * (window_minutes / Decimal("60"))
                inflow = outflow * Decimal("0.7")

            flows = ExchangeFlows(
                inflow_usd=inflow,
                outflow_usd=outflow,
                avg_hourly_flow=hourly_volume,
                data_age_seconds=0,
            )
        except (ValueError, TypeError):
            flows = ExchangeFlows(
                inflow_usd=Decimal("0"),
                outflow_usd=Decimal("0"),
                avg_hourly_flow=Decimal("0"),
                data_age_seconds=0,
            )

        self._set_cached(cache_key, flows, _TTL_EXCHANGE_FLOWS)
        return flows

    # ------------------------------------------------------------------
    # Mempool (Tier 3 — Ante & Saggu 2024)
    # ------------------------------------------------------------------

    async def get_pending_swap_volume(self, window_minutes: int = 15) -> PendingSwapVolume:
        """
        Aggregate pending transaction volume as medium-term momentum bias.

        Ante & Saggu (2024): mempool transaction flow predicts volume
        but NOT reliably price direction. BSC's 99.8% PBS block building
        (BEP-322) further limits mempool visibility.

        This uses BSC RPC `txpool_content` to scan pending transactions
        for DEX swap signatures. Due to PBS, visibility is near-zero in
        production — this is Tier 3 (informational only, low weight).

        Args:
            window_minutes: Aggregation window.

        Returns:
            PendingSwapVolume estimate.
        """
        cache_key = f"mempool:{window_minutes}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cast(PendingSwapVolume, cached)

        # Default: return neutral values (mempool visibility is very limited)
        default = PendingSwapVolume(
            volume_usd=Decimal("0"),
            avg_volume_usd=Decimal("1000000"),  # Baseline for comparison
            net_buy_ratio=Decimal("0.5"),  # Neutral
            window_seconds=window_minutes * 60,
        )

        if not self._w3:
            self._set_cached(cache_key, default, _TTL_PENDING_SWAP)
            return default

        try:
            # Attempt to query pending transactions
            # txpool_content is not available on all BSC RPC providers
            txpool = await self._w3.provider.make_request("txpool_content", [])
            pending_txs = txpool.get("result", {}).get("pending", {})

            # Count swap-like transactions (function selectors)
            # Common DEX swap selectors:
            #   0x38ed1739 = swapExactTokensForTokens
            #   0x7ff36ab5 = swapExactETHForTokens
            #   0x18cbafe5 = swapExactTokensForETH
            swap_selectors = {
                "0x38ed1739",
                "0x7ff36ab5",
                "0x18cbafe5",
                "0x5c11d795",
                "0xfb3bdb41",
                "0xb6f9de95",
            }

            swap_count = 0
            buy_count = 0  # Swaps into volatile assets
            total_value = Decimal("0")

            for _sender, nonce_txs in pending_txs.items():
                for _nonce, tx in nonce_txs.items():
                    input_data = tx.get("input", "")[:10]
                    if input_data in swap_selectors:
                        swap_count += 1
                        value_wei = int(tx.get("value", "0"), 16)
                        total_value += Decimal(str(value_wei)) / Decimal("1e18")
                        # Swaps sending ETH/BNB are likely buys
                        if input_data in {"0x7ff36ab5", "0xfb3bdb41", "0xb6f9de95"}:
                            buy_count += 1

            net_buy_ratio = (
                Decimal(str(buy_count)) / Decimal(str(swap_count))
                if swap_count > 0
                else Decimal("0.5")
            )

            result = PendingSwapVolume(
                volume_usd=total_value * Decimal("300"),  # Rough BNB price estimate
                avg_volume_usd=Decimal("1000000"),
                net_buy_ratio=net_buy_ratio,
                window_seconds=window_minutes * 60,
            )

        except Exception as e:
            self._logger.debug("Mempool query failed (expected on PBS nodes): %s", e)
            result = default

        self._set_cached(cache_key, result, _TTL_PENDING_SWAP)
        return result

    # ------------------------------------------------------------------
    # Convenience: log returns for GARCH
    # ------------------------------------------------------------------

    async def get_recent_returns(self, symbol: str, periods: int = 100) -> list[Decimal]:
        """
        Compute log returns from recent OHLCV data for GARCH input.

        Args:
            symbol: Trading pair.
            periods: Number of return periods.

        Returns:
            List of log returns as Decimal values.
        """
        candles = await self.get_ohlcv(symbol, "1h", limit=periods + 1)
        if len(candles) < 2:
            return []

        returns = []
        for i in range(1, len(candles)):
            if candles[i].close > 0 and candles[i - 1].close > 0:
                import math as _math

                ratio = float(candles[i].close) / float(candles[i - 1].close)
                if ratio > 0:
                    returns.append(Decimal(str(_math.log(ratio))))

        return returns
