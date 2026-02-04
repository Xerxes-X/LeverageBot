"""
DEX aggregator fan-out client for BSC Leverage Bot.

Queries 1inch, OpenOcean, and ParaSwap in parallel, selects the best
quote by output amount, and validates against Chainlink oracle prices
to reject quotes with excessive DEX-Oracle divergence.

Usage:
    client = AggregatorClient(aave_client)
    quote = await client.get_best_quote(from_token, to_token, amount, 18, 18)
    await client.close()
"""

from __future__ import annotations

import asyncio
import os
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

import aiohttp

from bot_logging.logger_manager import setup_module_logger
from config.loader import get_config
from shared.constants import DEFAULT_MAX_SLIPPAGE_BPS
from shared.types import SwapQuote

if TYPE_CHECKING:
    from collections.abc import Callable

    from execution.aave_client import AaveClient

_ZERO_ADDRESS = "0x" + "00" * 20


class AggregatorClientError(Exception):
    """Raised when no aggregator can provide a valid quote."""


class AggregatorClient:
    """
    Async DEX aggregator fan-out client.

    Queries 1inch, OpenOcean, and ParaSwap in parallel.  Selects the
    best quote by ``to_amount``.  Rejects quotes diverging more than
    ``max_dex_oracle_divergence_pct`` from Chainlink oracle prices.
    """

    def __init__(self, aave_client: AaveClient) -> None:
        self._aave_client = aave_client

        cfg = get_config()
        agg_cfg = cfg.get_aggregator_config()
        positions_cfg = cfg.get_positions_config()
        timing_cfg = cfg.get_timing_config()

        # Provider configs (only enabled)
        self._providers: list[dict[str, Any]] = [
            p for p in agg_cfg.get("providers", []) if p.get("enabled", True)
        ]

        # Approved router whitelist per provider
        self._approved_routers: dict[str, set[str]] = {}
        for p in self._providers:
            self._approved_routers[p["name"]] = {
                addr.lower() for addr in p.get("approved_routers", [])
            }

        # Slippage (bps)
        self._max_slippage_bps: int = agg_cfg.get(
            "max_slippage_bps",
            positions_cfg.get("max_slippage_bps", DEFAULT_MAX_SLIPPAGE_BPS),
        )

        # Oracle divergence threshold
        self._max_divergence_pct = Decimal(
            str(positions_cfg.get("max_dex_oracle_divergence_pct", "1.0"))
        )

        # Timeout per request
        agg_timing = timing_cfg.get("aggregator", {})
        self._timeout: float = agg_timing.get("quote_timeout_seconds", 5)

        # Quote cache
        self._cache: dict[str, tuple[float, SwapQuote]] = {}
        self._cache_ttl: float = agg_timing.get("quote_cache_ttl_seconds", 10)

        # Per-provider rate limiting
        self._rate_limits: dict[str, float] = {}
        self._rate_limit_state: dict[str, float] = {}
        for p in self._providers:
            rps = p.get("rate_limit_rps", 1)
            self._rate_limits[p["name"]] = 1.0 / rps if rps > 0 else 1.0
            self._rate_limit_state[p["name"]] = 0.0

        # API keys from environment
        self._api_keys: dict[str, str] = {}
        for p in self._providers:
            env_key = p.get("api_key_env", "")
            if env_key:
                self._api_keys[p["name"]] = os.environ.get(env_key, "")

        # Provider base URLs
        self._base_urls: dict[str, str] = {p["name"]: p["base_url"] for p in self._providers}

        # Provider dispatch table
        self._fetchers: dict[str, Callable[..., Any]] = {
            "1inch": self._fetch_1inch,
            "openocean": self._fetch_openocean,
            "paraswap": self._fetch_paraswap,
        }

        # Lazy-init aiohttp session
        self._session: aiohttp.ClientSession | None = None

        self._logger = setup_module_logger(
            "aggregator", "aggregator.log", module_folder="Aggregator_Logs"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_best_quote(
        self,
        from_token: str,
        to_token: str,
        from_amount: int,
        from_decimals: int,
        to_decimals: int,
    ) -> SwapQuote:
        """
        Fan out to all enabled providers, pick best quote, validate
        against oracle.

        Raises ``AggregatorClientError`` if no valid quote is available.
        """
        cache_key = self._get_cache_key(from_token, to_token, from_amount)
        cached = self._check_cache(cache_key)
        if cached is not None:
            self._logger.debug("Cache hit for %s", cache_key)
            return cached

        # Build coroutines for enabled providers
        coroutines = []
        for provider in self._providers:
            name = provider["name"]
            fetcher = self._fetchers.get(name)
            if fetcher is None:
                continue
            if name == "paraswap":
                coroutines.append(
                    fetcher(from_token, to_token, from_amount, from_decimals, to_decimals)
                )
            else:
                coroutines.append(fetcher(from_token, to_token, from_amount))

        if not coroutines:
            raise AggregatorClientError("No enabled providers configured")

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Filter valid quotes
        valid_quotes: list[SwapQuote] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                self._logger.warning("Provider %s failed: %s", self._providers[i]["name"], result)
                continue
            if result is None:
                continue
            quote_result = cast(SwapQuote, result)
            # Validate router
            provider_name = self._providers[i]["name"]
            if not self._validate_router(provider_name, quote_result.router_address):
                self._logger.warning(
                    "Provider %s returned unapproved router %s",
                    provider_name,
                    quote_result.router_address,
                )
                continue
            valid_quotes.append(quote_result)

        if not valid_quotes:
            raise AggregatorClientError("All providers failed to return a valid quote")

        # Sort by to_amount descending (best quote first)
        valid_quotes.sort(key=lambda q: q.to_amount, reverse=True)

        # Validate oracle divergence (best-first)
        for quote in valid_quotes:
            is_valid = await self._validate_oracle_divergence(quote, from_decimals, to_decimals)
            if is_valid:
                self._set_cache(cache_key, quote)
                self._logger.info(
                    "Best quote: provider=%s to_amount=%s router=%s",
                    quote.provider,
                    quote.to_amount,
                    quote.router_address,
                )
                return quote
            self._logger.warning(
                "Quote from %s rejected: oracle divergence exceeds %.1f%%",
                quote.provider,
                self._max_divergence_pct,
            )

        raise AggregatorClientError(
            f"All quotes exceed max oracle divergence ({self._max_divergence_pct}%)"
        )

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Provider-specific fetch methods
    # ------------------------------------------------------------------

    async def _fetch_1inch(
        self, from_token: str, to_token: str, from_amount: int
    ) -> SwapQuote | None:
        """Fetch swap quote from 1inch v6.0 API."""
        try:
            base_url = self._base_urls.get("1inch", "")
            slippage = self._max_slippage_bps / 100  # 1inch uses percentage

            params = {
                "src": from_token,
                "dst": to_token,
                "amount": str(from_amount),
                "from": _ZERO_ADDRESS,
                "slippage": str(slippage),
                "disableEstimate": "true",
            }

            headers = {}
            api_key = self._api_keys.get("1inch", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            data = await self._rate_limited_get(
                "1inch", f"{base_url}/swap", headers=headers, params=params
            )

            to_amount = Decimal(str(data["dstAmount"]))
            to_amount_min = to_amount * (10000 - self._max_slippage_bps) // 10000
            tx = data.get("tx", {})

            return SwapQuote(
                provider="1inch",
                from_token=from_token,
                to_token=to_token,
                from_amount=Decimal(str(from_amount)),
                to_amount=to_amount,
                to_amount_min=to_amount_min,
                calldata=bytes.fromhex(tx.get("data", "0x")[2:]),
                router_address=tx.get("to", ""),
                gas_estimate=int(tx.get("gas", 0)),
                price_impact=Decimal("0"),
            )
        except Exception as exc:
            self._logger.warning("1inch fetch failed: %s", exc)
            return None

    async def _fetch_openocean(
        self, from_token: str, to_token: str, from_amount: int
    ) -> SwapQuote | None:
        """Fetch swap quote from OpenOcean v3 API."""
        try:
            base_url = self._base_urls.get("openocean", "")
            slippage = self._max_slippage_bps / 100

            params = {
                "inTokenAddress": from_token,
                "outTokenAddress": to_token,
                "amount": str(from_amount),
                "gasPrice": "5",
                "slippage": str(slippage),
            }

            resp_data = await self._rate_limited_get(
                "openocean", f"{base_url}/swap_quote", params=params
            )

            data = resp_data.get("data", {})
            to_amount = Decimal(str(data["outAmount"]))
            to_amount_min = Decimal(str(data.get("minOutAmount", "0")))
            if to_amount_min <= 0:
                to_amount_min = to_amount * (10000 - self._max_slippage_bps) // 10000

            return SwapQuote(
                provider="openocean",
                from_token=from_token,
                to_token=to_token,
                from_amount=Decimal(str(from_amount)),
                to_amount=to_amount,
                to_amount_min=to_amount_min,
                calldata=bytes.fromhex(data.get("data", "0x")[2:]),
                router_address=data.get("to", ""),
                gas_estimate=int(data.get("estimatedGas", 0)),
                price_impact=Decimal("0"),
            )
        except Exception as exc:
            self._logger.warning("OpenOcean fetch failed: %s", exc)
            return None

    async def _fetch_paraswap(
        self,
        from_token: str,
        to_token: str,
        from_amount: int,
        from_decimals: int,
        to_decimals: int,
    ) -> SwapQuote | None:
        """Fetch swap quote from ParaSwap v5 API (two-step: prices + transactions)."""
        try:
            base_url = self._base_urls.get("paraswap", "")
            slippage = self._max_slippage_bps  # ParaSwap uses bps

            # Step 1: Get price route
            price_params = {
                "srcToken": from_token,
                "destToken": to_token,
                "amount": str(from_amount),
                "srcDecimals": str(from_decimals),
                "destDecimals": str(to_decimals),
                "side": "SELL",
                "network": "56",
            }

            price_data = await self._rate_limited_get(
                "paraswap", f"{base_url}/prices", params=price_params
            )

            price_route = price_data.get("priceRoute", {})
            dest_amount = Decimal(str(price_route.get("destAmount", "0")))
            gas_cost = int(price_route.get("gasCost", 0))

            # Step 2: Build transaction
            tx_body = {
                "srcToken": from_token,
                "destToken": to_token,
                "srcAmount": str(from_amount),
                "destAmount": str(dest_amount),
                "priceRoute": price_route,
                "userAddress": _ZERO_ADDRESS,
                "slippage": slippage,
            }

            tx_data = await self._rate_limited_post(
                "paraswap", f"{base_url}/transactions/56", json_data=tx_body
            )

            to_amount_min = dest_amount * (10000 - self._max_slippage_bps) // 10000

            return SwapQuote(
                provider="paraswap",
                from_token=from_token,
                to_token=to_token,
                from_amount=Decimal(str(from_amount)),
                to_amount=dest_amount,
                to_amount_min=to_amount_min,
                calldata=bytes.fromhex(tx_data.get("data", "0x")[2:]),
                router_address=tx_data.get("to", ""),
                gas_estimate=int(tx_data.get("gas", gas_cost)),
                price_impact=Decimal("0"),
            )
        except Exception as exc:
            self._logger.warning("ParaSwap fetch failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # HTTP helpers with rate limiting
    # ------------------------------------------------------------------

    async def _rate_limited_get(
        self,
        provider_name: str,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """GET with per-provider rate limiting and timeout."""
        await self._enforce_rate_limit(provider_name)

        session = await self._ensure_session()
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with session.get(url, headers=headers, params=params, timeout=timeout) as resp:
            resp.raise_for_status()
            return cast(dict[str, Any], await resp.json())

    async def _rate_limited_post(
        self,
        provider_name: str,
        url: str,
        json_data: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST with per-provider rate limiting and timeout."""
        await self._enforce_rate_limit(provider_name)

        session = await self._ensure_session()
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with session.post(url, json=json_data, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            return cast(dict[str, Any], await resp.json())

    async def _enforce_rate_limit(self, provider_name: str) -> None:
        """Sleep if necessary to respect the per-provider rate limit."""
        min_interval = self._rate_limits.get(provider_name, 1.0)
        last_time = self._rate_limit_state.get(provider_name, 0.0)
        elapsed = time.monotonic() - last_time

        if last_time > 0 and elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

        self._rate_limit_state[provider_name] = time.monotonic()

    # ------------------------------------------------------------------
    # Oracle divergence validation
    # ------------------------------------------------------------------

    async def _validate_oracle_divergence(
        self, quote: SwapQuote, from_decimals: int, to_decimals: int
    ) -> bool:
        """
        Compare DEX implied price to Chainlink oracle price.

        Returns True if divergence is within ``_max_divergence_pct``.
        """
        oracle_prices = await self._aave_client.get_assets_prices(
            [quote.from_token, quote.to_token]
        )
        oracle_from_usd = oracle_prices[0]
        oracle_to_usd = oracle_prices[1]

        if oracle_from_usd <= 0 or oracle_to_usd <= 0:
            self._logger.error(
                "Oracle returned zero/negative price (from=%s, to=%s), rejecting",
                oracle_from_usd,
                oracle_to_usd,
            )
            return False

        # Oracle exchange rate: how many to_tokens per from_token
        oracle_rate = oracle_from_usd / oracle_to_usd

        # DEX exchange rate from the quote
        from_human = quote.from_amount / Decimal(10**from_decimals)
        to_human = quote.to_amount / Decimal(10**to_decimals)

        if from_human <= 0:
            return False

        dex_rate = to_human / from_human
        divergence_pct = abs(dex_rate - oracle_rate) / oracle_rate * Decimal("100")

        self._logger.debug(
            "Oracle divergence: dex_rate=%s oracle_rate=%s divergence=%.4f%%",
            dex_rate,
            oracle_rate,
            divergence_pct,
        )

        return divergence_pct <= self._max_divergence_pct

    # ------------------------------------------------------------------
    # Router validation
    # ------------------------------------------------------------------

    def _validate_router(self, provider_name: str, router_address: str) -> bool:
        """Check router address against the approved whitelist for the provider."""
        approved = self._approved_routers.get(provider_name, set())
        if not approved:
            return True  # No whitelist configured — allow all
        return router_address.lower() in approved

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cache_key(self, from_token: str, to_token: str, from_amount: int) -> str:
        """Build deterministic cache key."""
        return f"{from_token.lower()}:{to_token.lower()}:{from_amount}"

    def _check_cache(self, key: str) -> SwapQuote | None:
        """Return cached quote if TTL not expired, else None."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        stored_time, quote = entry
        if time.monotonic() - stored_time > self._cache_ttl:
            del self._cache[key]
            return None
        return quote

    def _set_cache(self, key: str, quote: SwapQuote) -> None:
        """Store quote in cache with current monotonic timestamp."""
        self._cache[key] = (time.monotonic(), quote)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazy-init aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
