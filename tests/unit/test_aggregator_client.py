"""
Unit tests for execution/aggregator_client.py.

All tests mock HTTP responses via aioresponses and AaveClient via
unittest.mock.  Tests verify parallel fan-out, best-quote selection,
oracle divergence rejection, rate limiting, caching, router validation,
provider response parsing, and session lifecycle.
"""

from __future__ import annotations

import re
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aioresponses import aioresponses

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

SAMPLE_FROM_TOKEN = "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"  # WBNB
SAMPLE_TO_TOKEN = "0x55d398326f99059fF775485246999027B3197955"  # USDT
SAMPLE_FROM_AMOUNT = 1 * 10**18  # 1 WBNB in wei
SAMPLE_FROM_DECIMALS = 18
SAMPLE_TO_DECIMALS = 18

BNB_PRICE_USD = Decimal("612.34")
USDT_PRICE_USD = Decimal("1.0")

SAMPLE_TO_AMOUNT = 612 * 10**18  # ~612 USDT
SAMPLE_TO_AMOUNT_BETTER = 613 * 10**18
SAMPLE_TO_AMOUNT_BEST = 614 * 10**18

# Router addresses (matching config/aggregator.json)
ROUTER_1INCH = "0x111111125421cA6dc452d289314280a0f8842A65"
ROUTER_OPENOCEAN = "0x6352a56caadC4F1E25CD6c75970Fa768A3304e64"
ROUTER_PARASWAP = "0xDEF171Fe48CF0115B1d80b88dc8eAB59176FEe57"

# Provider base URLs
URL_1INCH = "https://api.1inch.dev/swap/v6.0/56"
URL_OPENOCEAN = "https://open-api.openocean.finance/v3/56"
URL_PARASWAP = "https://apiv5.paraswap.io"

# aioresponses needs regex patterns to match URLs with query params
RE_1INCH_SWAP = re.compile(r"https://api\.1inch\.dev/swap/v6\.0/56/swap")
RE_OPENOCEAN_SWAP = re.compile(r"https://open-api\.openocean\.finance/v3/56/swap_quote")
RE_PARASWAP_PRICES = re.compile(r"https://apiv5\.paraswap\.io/prices")
RE_PARASWAP_TX = re.compile(r"https://apiv5\.paraswap\.io/transactions/56")

SAMPLE_CALLDATA = "abcdef1234567890"

# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

STANDARD_AGG_CONFIG = {
    "providers": [
        {
            "name": "1inch",
            "enabled": True,
            "priority": 1,
            "base_url": URL_1INCH,
            "api_key_env": "",
            "rate_limit_rps": 1,
            "timeout_seconds": 5,
            "approved_routers": [ROUTER_1INCH],
            "params": {"disableEstimate": True},
        },
        {
            "name": "openocean",
            "enabled": True,
            "priority": 2,
            "base_url": URL_OPENOCEAN,
            "api_key_env": "",
            "rate_limit_rps": 2,
            "timeout_seconds": 5,
            "approved_routers": [ROUTER_OPENOCEAN],
            "params": {},
        },
        {
            "name": "paraswap",
            "enabled": True,
            "priority": 3,
            "base_url": URL_PARASWAP,
            "api_key_env": "",
            "rate_limit_rps": 2,
            "timeout_seconds": 5,
            "approved_routers": [ROUTER_PARASWAP],
            "params": {},
        },
    ],
    "max_slippage_bps": 50,
    "max_price_impact_percent": "1.0",
}

STANDARD_POS_CONFIG = {
    "max_dex_oracle_divergence_pct": "1.0",
    "max_slippage_bps": 50,
}

STANDARD_TIMING_CONFIG = {
    "aggregator": {
        "quote_timeout_seconds": 5,
        "quote_cache_ttl_seconds": 10,
    },
}


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def _1inch_response(to_amount: int = SAMPLE_TO_AMOUNT, router: str = ROUTER_1INCH) -> dict:
    return {
        "dstAmount": str(to_amount),
        "tx": {
            "data": f"0x{SAMPLE_CALLDATA}",
            "to": router,
            "gas": 200000,
        },
    }


def _openocean_response(
    to_amount: int = SAMPLE_TO_AMOUNT, router: str = ROUTER_OPENOCEAN
) -> dict:
    return {
        "data": {
            "outAmount": str(to_amount),
            "minOutAmount": str(int(to_amount * 9950 / 10000)),
            "data": f"0x{SAMPLE_CALLDATA}",
            "to": router,
            "estimatedGas": 180000,
        },
    }


def _paraswap_price_response(to_amount: int = SAMPLE_TO_AMOUNT) -> dict:
    return {
        "priceRoute": {
            "destAmount": str(to_amount),
            "gasCost": 250000,
        },
    }


def _paraswap_tx_response(router: str = ROUTER_PARASWAP) -> dict:
    return {
        "data": f"0x{SAMPLE_CALLDATA}",
        "to": router,
        "gas": 250000,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_aave_client():
    client = MagicMock()
    client.get_assets_prices = AsyncMock(
        return_value=[BNB_PRICE_USD, USDT_PRICE_USD]
    )
    return client


def _make_client(mock_aave_client, agg_config=None, pos_config=None, timing_config=None):
    """Create an AggregatorClient with patched config and logger."""
    if agg_config is None:
        agg_config = STANDARD_AGG_CONFIG
    if pos_config is None:
        pos_config = STANDARD_POS_CONFIG
    if timing_config is None:
        timing_config = STANDARD_TIMING_CONFIG

    with (
        patch("execution.aggregator_client.get_config") as mock_cfg,
        patch("execution.aggregator_client.setup_module_logger") as mock_logger,
    ):
        mock_loader = MagicMock()
        mock_loader.get_aggregator_config.return_value = agg_config
        mock_loader.get_positions_config.return_value = pos_config
        mock_loader.get_timing_config.return_value = timing_config
        mock_cfg.return_value = mock_loader
        mock_logger.return_value = MagicMock()

        from execution.aggregator_client import AggregatorClient

        return AggregatorClient(mock_aave_client)


@pytest.fixture
def aggregator_client(mock_aave_client):
    return _make_client(mock_aave_client)


def _register_all_providers(
    mocked,
    to_amount_1inch=SAMPLE_TO_AMOUNT,
    to_amount_openocean=SAMPLE_TO_AMOUNT_BETTER,
    to_amount_paraswap=SAMPLE_TO_AMOUNT_BEST,
):
    """Register mock responses for all 3 providers."""
    mocked.get(
        RE_1INCH_SWAP,
        payload=_1inch_response(to_amount_1inch),
    )
    mocked.get(
        RE_OPENOCEAN_SWAP,
        payload=_openocean_response(to_amount_openocean),
    )
    mocked.get(
        RE_PARASWAP_PRICES,
        payload=_paraswap_price_response(to_amount_paraswap),
    )
    mocked.post(
        RE_PARASWAP_TX,
        payload=_paraswap_tx_response(),
    )


# ---------------------------------------------------------------------------
# A. Best quote selection tests
# ---------------------------------------------------------------------------


class TestBestQuoteSelection:

    async def test_selects_highest_to_amount(self, aggregator_client):
        with aioresponses() as mocked:
            _register_all_providers(mocked)

            quote = await aggregator_client.get_best_quote(
                SAMPLE_FROM_TOKEN,
                SAMPLE_TO_TOKEN,
                SAMPLE_FROM_AMOUNT,
                SAMPLE_FROM_DECIMALS,
                SAMPLE_TO_DECIMALS,
            )

        assert quote.provider == "paraswap"
        assert quote.to_amount == Decimal(str(SAMPLE_TO_AMOUNT_BEST))
        await aggregator_client.close()

    async def test_returns_only_valid_quote_when_others_fail(self, aggregator_client):
        with aioresponses() as mocked:
            # 1inch and paraswap fail, openocean succeeds
            mocked.get(RE_1INCH_SWAP, status=500)
            mocked.get(
                RE_OPENOCEAN_SWAP,
                payload=_openocean_response(),
            )
            mocked.get(RE_PARASWAP_PRICES, status=500)

            quote = await aggregator_client.get_best_quote(
                SAMPLE_FROM_TOKEN,
                SAMPLE_TO_TOKEN,
                SAMPLE_FROM_AMOUNT,
                SAMPLE_FROM_DECIMALS,
                SAMPLE_TO_DECIMALS,
            )

        assert quote.provider == "openocean"
        await aggregator_client.close()

    async def test_all_providers_fail_raises_error(self, aggregator_client):
        from execution.aggregator_client import AggregatorClientError

        with aioresponses() as mocked:
            mocked.get(RE_1INCH_SWAP, status=500)
            mocked.get(RE_OPENOCEAN_SWAP, status=500)
            mocked.get(RE_PARASWAP_PRICES, status=500)

            with pytest.raises(AggregatorClientError, match="All providers failed"):
                await aggregator_client.get_best_quote(
                    SAMPLE_FROM_TOKEN,
                    SAMPLE_TO_TOKEN,
                    SAMPLE_FROM_AMOUNT,
                    SAMPLE_FROM_DECIMALS,
                    SAMPLE_TO_DECIMALS,
                )
        await aggregator_client.close()

    async def test_skips_disabled_providers(self, mock_aave_client):
        config = {
            **STANDARD_AGG_CONFIG,
            "providers": [
                {**STANDARD_AGG_CONFIG["providers"][0], "enabled": False},
                STANDARD_AGG_CONFIG["providers"][1],
                {**STANDARD_AGG_CONFIG["providers"][2], "enabled": False},
            ],
        }
        client = _make_client(mock_aave_client, agg_config=config)

        with aioresponses() as mocked:
            # Only openocean should be queried
            mocked.get(
                RE_OPENOCEAN_SWAP,
                payload=_openocean_response(),
            )

            quote = await client.get_best_quote(
                SAMPLE_FROM_TOKEN,
                SAMPLE_TO_TOKEN,
                SAMPLE_FROM_AMOUNT,
                SAMPLE_FROM_DECIMALS,
                SAMPLE_TO_DECIMALS,
            )

        assert quote.provider == "openocean"
        await client.close()

    async def test_returns_cached_quote_within_ttl(self, aggregator_client):
        with aioresponses() as mocked:
            _register_all_providers(mocked)

            # First call — hits HTTP
            quote1 = await aggregator_client.get_best_quote(
                SAMPLE_FROM_TOKEN,
                SAMPLE_TO_TOKEN,
                SAMPLE_FROM_AMOUNT,
                SAMPLE_FROM_DECIMALS,
                SAMPLE_TO_DECIMALS,
            )

        # Second call — should use cache (no mock responses registered)
        quote2 = await aggregator_client.get_best_quote(
            SAMPLE_FROM_TOKEN,
            SAMPLE_TO_TOKEN,
            SAMPLE_FROM_AMOUNT,
            SAMPLE_FROM_DECIMALS,
            SAMPLE_TO_DECIMALS,
        )

        assert quote1.provider == quote2.provider
        assert quote1.to_amount == quote2.to_amount
        await aggregator_client.close()


# ---------------------------------------------------------------------------
# B. Oracle divergence tests
# ---------------------------------------------------------------------------


class TestOracleDivergence:

    async def test_accepts_quote_within_divergence(self, aggregator_client):
        # Default oracle prices give rate 612.34, quote is ~612 USDT/BNB → ~0.05% off
        with aioresponses() as mocked:
            _register_all_providers(
                mocked,
                to_amount_1inch=SAMPLE_TO_AMOUNT,
                to_amount_openocean=SAMPLE_TO_AMOUNT,
                to_amount_paraswap=SAMPLE_TO_AMOUNT,
            )

            quote = await aggregator_client.get_best_quote(
                SAMPLE_FROM_TOKEN,
                SAMPLE_TO_TOKEN,
                SAMPLE_FROM_AMOUNT,
                SAMPLE_FROM_DECIMALS,
                SAMPLE_TO_DECIMALS,
            )

        assert quote is not None
        await aggregator_client.close()

    async def test_rejects_quote_exceeding_divergence(self, aggregator_client):
        from execution.aggregator_client import AggregatorClientError

        # All providers return ~700 USDT for 1 BNB (oracle says 612.34) → ~14% off
        bad_amount = 700 * 10**18
        with aioresponses() as mocked:
            _register_all_providers(
                mocked,
                to_amount_1inch=bad_amount,
                to_amount_openocean=bad_amount,
                to_amount_paraswap=bad_amount,
            )

            with pytest.raises(AggregatorClientError, match="oracle divergence"):
                await aggregator_client.get_best_quote(
                    SAMPLE_FROM_TOKEN,
                    SAMPLE_TO_TOKEN,
                    SAMPLE_FROM_AMOUNT,
                    SAMPLE_FROM_DECIMALS,
                    SAMPLE_TO_DECIMALS,
                )
        await aggregator_client.close()

    async def test_falls_through_to_second_best(self, aggregator_client):
        # Best quote (paraswap) returns 700 USDT → diverges
        # Second best (openocean) returns 612 USDT → passes
        with aioresponses() as mocked:
            _register_all_providers(
                mocked,
                to_amount_1inch=611 * 10**18,
                to_amount_openocean=612 * 10**18,
                to_amount_paraswap=700 * 10**18,  # will diverge
            )

            quote = await aggregator_client.get_best_quote(
                SAMPLE_FROM_TOKEN,
                SAMPLE_TO_TOKEN,
                SAMPLE_FROM_AMOUNT,
                SAMPLE_FROM_DECIMALS,
                SAMPLE_TO_DECIMALS,
            )

        # Should skip paraswap (diverges) and pick openocean
        assert quote.provider == "openocean"
        await aggregator_client.close()

    async def test_rejects_when_oracle_returns_zero(self, mock_aave_client):
        mock_aave_client.get_assets_prices = AsyncMock(
            return_value=[Decimal("0"), USDT_PRICE_USD]
        )
        client = _make_client(mock_aave_client)

        from execution.aggregator_client import AggregatorClientError

        with aioresponses() as mocked:
            _register_all_providers(mocked)

            with pytest.raises(AggregatorClientError, match="oracle divergence"):
                await client.get_best_quote(
                    SAMPLE_FROM_TOKEN,
                    SAMPLE_TO_TOKEN,
                    SAMPLE_FROM_AMOUNT,
                    SAMPLE_FROM_DECIMALS,
                    SAMPLE_TO_DECIMALS,
                )
        await client.close()


# ---------------------------------------------------------------------------
# C. Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimiting:

    async def test_respects_rate_limit_delay(self, aggregator_client):
        with aioresponses() as mocked:
            # Register responses for two calls
            mocked.get(RE_1INCH_SWAP, payload=_1inch_response(), repeat=True)
            mocked.get(RE_OPENOCEAN_SWAP, payload=_openocean_response(), repeat=True)
            mocked.get(RE_PARASWAP_PRICES, payload=_paraswap_price_response(), repeat=True)
            mocked.post(RE_PARASWAP_TX, payload=_paraswap_tx_response(), repeat=True)

            # First call
            await aggregator_client.get_best_quote(
                SAMPLE_FROM_TOKEN,
                SAMPLE_TO_TOKEN,
                SAMPLE_FROM_AMOUNT,
                SAMPLE_FROM_DECIMALS,
                SAMPLE_TO_DECIMALS,
            )

            # Clear cache to force second HTTP call
            aggregator_client._cache.clear()

            # Second call should still succeed (rate limiter sleeps if needed)
            quote = await aggregator_client.get_best_quote(
                SAMPLE_FROM_TOKEN,
                SAMPLE_TO_TOKEN,
                SAMPLE_FROM_AMOUNT,
                SAMPLE_FROM_DECIMALS,
                SAMPLE_TO_DECIMALS,
            )

        assert quote is not None
        await aggregator_client.close()

    async def test_independent_provider_limits(self, aggregator_client):
        # Verify that rate limit state is tracked per provider
        assert "1inch" in aggregator_client._rate_limit_state
        assert "openocean" in aggregator_client._rate_limit_state
        # 1inch is 1 RPS (1.0s interval), openocean is 2 RPS (0.5s interval)
        assert aggregator_client._rate_limits["1inch"] == 1.0
        assert aggregator_client._rate_limits["openocean"] == 0.5


# ---------------------------------------------------------------------------
# D. Timeout handling tests
# ---------------------------------------------------------------------------


class TestTimeoutHandling:

    async def test_provider_timeout_returns_none(self, aggregator_client):
        with aioresponses() as mocked:
            # 1inch times out, others succeed
            mocked.get(RE_1INCH_SWAP, exception=TimeoutError("timeout"))
            mocked.get(RE_OPENOCEAN_SWAP, payload=_openocean_response())
            mocked.get(RE_PARASWAP_PRICES, payload=_paraswap_price_response())
            mocked.post(RE_PARASWAP_TX, payload=_paraswap_tx_response())

            quote = await aggregator_client.get_best_quote(
                SAMPLE_FROM_TOKEN,
                SAMPLE_TO_TOKEN,
                SAMPLE_FROM_AMOUNT,
                SAMPLE_FROM_DECIMALS,
                SAMPLE_TO_DECIMALS,
            )

        assert quote is not None
        assert quote.provider != "1inch"
        await aggregator_client.close()

    async def test_all_providers_timeout_raises_error(self, aggregator_client):
        from execution.aggregator_client import AggregatorClientError

        with aioresponses() as mocked:
            mocked.get(RE_1INCH_SWAP, exception=TimeoutError("timeout"))
            mocked.get(RE_OPENOCEAN_SWAP, exception=TimeoutError("timeout"))
            mocked.get(RE_PARASWAP_PRICES, exception=TimeoutError("timeout"))

            with pytest.raises(AggregatorClientError, match="All providers failed"):
                await aggregator_client.get_best_quote(
                    SAMPLE_FROM_TOKEN,
                    SAMPLE_TO_TOKEN,
                    SAMPLE_FROM_AMOUNT,
                    SAMPLE_FROM_DECIMALS,
                    SAMPLE_TO_DECIMALS,
                )
        await aggregator_client.close()


# ---------------------------------------------------------------------------
# E. Cache behavior tests
# ---------------------------------------------------------------------------


class TestCacheBehavior:

    async def test_cache_hit_skips_http(self, aggregator_client):
        from shared.types import SwapQuote

        # Pre-populate cache
        cached_quote = SwapQuote(
            provider="cached",
            from_token=SAMPLE_FROM_TOKEN,
            to_token=SAMPLE_TO_TOKEN,
            from_amount=Decimal(str(SAMPLE_FROM_AMOUNT)),
            to_amount=Decimal("612000000000000000000"),
            to_amount_min=Decimal("609000000000000000000"),
            calldata=b"\xab\xcd",
            router_address=ROUTER_1INCH,
            gas_estimate=200000,
            price_impact=Decimal("0"),
        )
        cache_key = aggregator_client._get_cache_key(
            SAMPLE_FROM_TOKEN, SAMPLE_TO_TOKEN, SAMPLE_FROM_AMOUNT
        )
        aggregator_client._set_cache(cache_key, cached_quote)

        # No HTTP mocks — would fail if HTTP was attempted
        quote = await aggregator_client.get_best_quote(
            SAMPLE_FROM_TOKEN,
            SAMPLE_TO_TOKEN,
            SAMPLE_FROM_AMOUNT,
            SAMPLE_FROM_DECIMALS,
            SAMPLE_TO_DECIMALS,
        )

        assert quote.provider == "cached"
        await aggregator_client.close()

    async def test_cache_expired_refetches(self, aggregator_client):
        from shared.types import SwapQuote

        # Pre-populate cache with expired entry
        cached_quote = SwapQuote(
            provider="expired",
            from_token=SAMPLE_FROM_TOKEN,
            to_token=SAMPLE_TO_TOKEN,
            from_amount=Decimal(str(SAMPLE_FROM_AMOUNT)),
            to_amount=Decimal("612000000000000000000"),
            to_amount_min=Decimal("609000000000000000000"),
            calldata=b"\xab\xcd",
            router_address=ROUTER_1INCH,
            gas_estimate=200000,
            price_impact=Decimal("0"),
        )
        cache_key = aggregator_client._get_cache_key(
            SAMPLE_FROM_TOKEN, SAMPLE_TO_TOKEN, SAMPLE_FROM_AMOUNT
        )
        # Set cache with old timestamp (expired)
        aggregator_client._cache[cache_key] = (time.monotonic() - 999, cached_quote)

        with aioresponses() as mocked:
            _register_all_providers(mocked)

            quote = await aggregator_client.get_best_quote(
                SAMPLE_FROM_TOKEN,
                SAMPLE_TO_TOKEN,
                SAMPLE_FROM_AMOUNT,
                SAMPLE_FROM_DECIMALS,
                SAMPLE_TO_DECIMALS,
            )

        # Should have fetched fresh quotes
        assert quote.provider != "expired"
        await aggregator_client.close()

    async def test_different_params_different_keys(self, aggregator_client):
        key1 = aggregator_client._get_cache_key(SAMPLE_FROM_TOKEN, SAMPLE_TO_TOKEN, 100)
        key2 = aggregator_client._get_cache_key(SAMPLE_FROM_TOKEN, SAMPLE_TO_TOKEN, 200)
        assert key1 != key2


# ---------------------------------------------------------------------------
# F. Router validation tests
# ---------------------------------------------------------------------------


class TestRouterValidation:

    def test_approved_router_accepted(self, aggregator_client):
        assert aggregator_client._validate_router("1inch", ROUTER_1INCH) is True
        assert aggregator_client._validate_router("openocean", ROUTER_OPENOCEAN) is True
        assert aggregator_client._validate_router("paraswap", ROUTER_PARASWAP) is True

    async def test_unapproved_router_rejected(self, aggregator_client):
        from execution.aggregator_client import AggregatorClientError

        fake_router = "0x0000000000000000000000000000000000000BAD"
        with aioresponses() as mocked:
            # All providers return unapproved routers
            mocked.get(
                RE_1INCH_SWAP,
                payload=_1inch_response(router=fake_router),
            )
            mocked.get(
                RE_OPENOCEAN_SWAP,
                payload=_openocean_response(router=fake_router),
            )
            mocked.get(
                RE_PARASWAP_PRICES,
                payload=_paraswap_price_response(),
            )
            mocked.post(
                RE_PARASWAP_TX,
                payload={**_paraswap_tx_response(), "to": fake_router},
            )

            with pytest.raises(AggregatorClientError, match="All providers failed"):
                await aggregator_client.get_best_quote(
                    SAMPLE_FROM_TOKEN,
                    SAMPLE_TO_TOKEN,
                    SAMPLE_FROM_AMOUNT,
                    SAMPLE_FROM_DECIMALS,
                    SAMPLE_TO_DECIMALS,
                )
        await aggregator_client.close()


# ---------------------------------------------------------------------------
# G. Provider parsing tests
# ---------------------------------------------------------------------------


class TestProviderParsing:

    async def test_parse_1inch_response(self, aggregator_client):
        with aioresponses() as mocked:
            mocked.get(RE_1INCH_SWAP, payload=_1inch_response())
            quote = await aggregator_client._fetch_1inch(
                SAMPLE_FROM_TOKEN, SAMPLE_TO_TOKEN, SAMPLE_FROM_AMOUNT
            )

        assert quote is not None
        assert quote.provider == "1inch"
        assert quote.to_amount == Decimal(str(SAMPLE_TO_AMOUNT))
        assert quote.router_address == ROUTER_1INCH
        assert quote.gas_estimate == 200000
        assert quote.calldata == bytes.fromhex(SAMPLE_CALLDATA)
        await aggregator_client.close()

    async def test_parse_openocean_response(self, aggregator_client):
        with aioresponses() as mocked:
            mocked.get(RE_OPENOCEAN_SWAP, payload=_openocean_response())
            quote = await aggregator_client._fetch_openocean(
                SAMPLE_FROM_TOKEN, SAMPLE_TO_TOKEN, SAMPLE_FROM_AMOUNT
            )

        assert quote is not None
        assert quote.provider == "openocean"
        assert quote.to_amount == Decimal(str(SAMPLE_TO_AMOUNT))
        assert quote.router_address == ROUTER_OPENOCEAN
        assert quote.gas_estimate == 180000
        assert quote.to_amount_min > 0
        await aggregator_client.close()

    async def test_parse_paraswap_response(self, aggregator_client):
        with aioresponses() as mocked:
            mocked.get(RE_PARASWAP_PRICES, payload=_paraswap_price_response())
            mocked.post(
                RE_PARASWAP_TX,
                payload=_paraswap_tx_response(),
            )
            quote = await aggregator_client._fetch_paraswap(
                SAMPLE_FROM_TOKEN,
                SAMPLE_TO_TOKEN,
                SAMPLE_FROM_AMOUNT,
                SAMPLE_FROM_DECIMALS,
                SAMPLE_TO_DECIMALS,
            )

        assert quote is not None
        assert quote.provider == "paraswap"
        assert quote.to_amount == Decimal(str(SAMPLE_TO_AMOUNT))
        assert quote.router_address == ROUTER_PARASWAP
        assert quote.gas_estimate == 250000
        await aggregator_client.close()


# ---------------------------------------------------------------------------
# H. Session lifecycle tests
# ---------------------------------------------------------------------------


class TestSessionLifecycle:

    async def test_close_closes_session(self, aggregator_client):
        # Trigger session creation by making a request
        with aioresponses() as mocked:
            _register_all_providers(mocked)
            await aggregator_client.get_best_quote(
                SAMPLE_FROM_TOKEN,
                SAMPLE_TO_TOKEN,
                SAMPLE_FROM_AMOUNT,
                SAMPLE_FROM_DECIMALS,
                SAMPLE_TO_DECIMALS,
            )

        assert aggregator_client._session is not None
        await aggregator_client.close()
        assert aggregator_client._session is None
