# Mempool Enhancement Plan — Order Flow Signal Integration

## Overview

Upgrade the LeverageBot's Tier 3 mempool signal from a stub returning neutral defaults to a production-grade **aggregate order flow analysis** system. The system monitors pending BSC swap transactions in real-time, computes directional bias from aggregate buy/sell volume, and feeds this as a signal component into the existing 5-layer signal engine.

This is **not** frontrunning or sandwich attacking. The bot does not race individual transactions or attempt to get ahead of specific pending swaps. Instead, it observes the aggregate directional bias of pending DEX swaps — if 70% of pending volume is buy-side, that is a bullish order flow signal. The bot then takes a leveraged position in the direction the market is naturally moving, submitted independently of any specific pending transaction.

### Academic Basis

- **Kolm, Turiel & Westray (2023, Journal of Financial Economics)**: Order book imbalance accounts for 73% of short-term price prediction. Pending mempool swaps are effectively an advance look at order flow before it reaches the order book.
- **Easley, Lopez de Prado & O'Hara (2012, Review of Financial Studies)**: VPIN measures the probability of informed trading from executed trades. Mempool data provides VPIN-equivalent signal *before* execution — a strictly earlier indicator.
- **Ante & Saggu (2024, Journal of Innovation & Knowledge)**: Mempool transaction flow predicts volume (not individual price direction). Aggregate volume bias *is* directional: if 70% of pending swap volume is buy-side, that is a bullish volume signal.
- **Chi et al. (2024, SSRN)**: Exchange flows predict returns. Pending DEX swaps on-chain are the DeFi equivalent of exchange flow data.
- **Shen, Urquhart & Wang (2019, Economics Letters)**: Tweet *volume* (not polarity) predicts BTC price. The same principle applies: aggregate mempool swap volume in a direction predicts short-term price movement.
- **Cont, Kukanov & Stoikov (2014, Journal of Financial Econometrics)**: The price impact of order book events — theoretical foundation for order flow as a price predictor.

### Distinction from MEV Extraction

| MEV Extraction (not what this does) | Order Flow Signal (what this does) |
|---|---|
| Targets a specific pending transaction | Observes aggregate directional bias |
| Must execute before the target tx | Executes whenever signal is strong enough |
| Profits at the expense of the target trader | Profits from correctly predicting market direction |
| Requires builder-level access on BSC | Works with partial public mempool visibility |
| Legally risky (potential market manipulation) | Standard order flow analysis (legal and common in TradFi) |

---

## Existing Infrastructure

### ArbitrageTestBot Mempool Listener (Verified Working)

The `/home/rahim/ArbitrageTestBot` project contains a fully functional mempool monitoring system that has been tested on BSC mainnet:

**Confirmed capabilities (from production logs — 2,428+ swaps decoded):**

- WebSocket connection to BSC (`wss://bsc-ws-node.nariox.org:443` or configurable `BSC_RPC_URL_WS`)
- Subscription via `eth_subscribe` with `newPendingTransactions` (full transaction objects)
- Decodes swap calldata from **12 DEX routers** (PancakeSwap V2/V3/Universal, Biswap, ApeSwap, 1inch, ParaSwap, KyberSwap, OpenOcean, Firebird)
- Recognizes **26 swap function selectors** across V2, V3, SmartRouter, Universal Router, and aggregator formats
- Extracts: `token_in`, `token_out`, `amount_in`, `amount_out_min`, `path`, `deadline`, `gas_price`, `is_v3`
- Publishes decoded swaps to Redis channel `mempool:pending_swaps`
- Processing latency: 1-2ms per transaction
- Includes poison detection (sandwich attack identification), deduplication, and shadow metrics

**Key files:**

| File | Lines | Purpose |
|------|-------|---------|
| `mempool_listener.py` | 2,637 | Main listener, decoder, publisher |
| `mempool_shadow_metrics.py` | 1,425 | Metrics and validation |
| `mempool_state_predictor.py` | 930 | State prediction engine |
| `node_relay.py` | 300+ | Unified WebSocket connection |
| `config/mempool.json` | 204 | Configuration |

### Current LeverageBot Tier 3 Stub

The LeverageBot's `core/data_service.py` has a `get_pending_swap_volume()` method (lines 776-864) that calls `txpool_content` via RPC. This is the wrong approach — `txpool_content` polls the local node's mempool snapshot, which on BSC is nearly empty due to PBS. The ArbitrageTestBot's approach of subscribing to `newPendingTransactions` via WebSocket captures transactions as they propagate through the P2P network, before builders absorb them.

---

## Architecture

### Data Flow

```
BSC P2P Network
    │
    ▼
WebSocket (newPendingTransactions)
    │
    ▼
┌─────────────────────────────────────┐
│  Mempool Decoder (Rust)             │
│  ─────────────────────────          │
│  • Receive full tx objects          │
│  • Match to_address against         │
│    12 monitored DEX routers         │
│  • Match function selector          │
│    against 26 known swap sigs       │
│  • ABI-decode swap parameters:      │
│    token_in, token_out, amount_in,  │
│    amount_out_min, path             │
│  • Classify direction:              │
│    buy (stable→volatile) or         │
│    sell (volatile→stable)           │
│  • Deduplication (seen tx cache)    │
│  • Poison detection (sandwich       │
│    pattern scoring)                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Rolling Aggregator (Rust)          │
│  ─────────────────────────          │
│  Per token pair (e.g. WBNB/USDT):  │
│  • Sliding windows: 1m, 5m, 15m    │
│  • Net buy volume (USD)             │
│  • Net sell volume (USD)            │
│  • Buy/sell ratio                   │
│  • Volume-weighted direction score  │
│  • Large trade detection (whale)    │
│  • Transaction count (buys/sells)   │
│  • Volume acceleration              │
│    (current vs trailing average)    │
└──────────────┬──────────────────────┘
               │
               ▼
         Redis / IPC
      (aggregated signal)
               │
               ▼
┌─────────────────────────────────────┐
│  LeverageBot SignalEngine (Python)  │
│  ─────────────────────────────────  │
│  • Consumes aggregate mempool       │
│    signal from Redis                │
│  • Integrates as signal component   │
│    in ensemble scoring              │
│  • Weight: configurable (0.10-0.20) │
│  • Regime-adjusted like all signals │
└─────────────────────────────────────┘
```

### Why Rust for the Decoder

The mempool decoder is the latency-critical component. On BSC with 750ms block times, pending transactions propagate rapidly. The decoder must:

- Maintain a persistent WebSocket connection
- Parse and decode calldata for every incoming transaction
- Classify and aggregate in real-time
- Handle burst traffic (BSC produces ~2,000 transactions per block)

The ArbitrageTestBot's Python implementation achieves 1-2ms per transaction, which is sufficient. However, a Rust implementation provides:

- **Deterministic latency**: No GC pauses or asyncio event loop contention
- **Higher throughput**: Handle burst traffic without backpressure
- **Standalone process**: Decoupled from the Python bot's event loop — the decoder runs independently and publishes aggregated results
- **Shared with ArbitrageTestBot**: A single Rust decoder can feed both bots via Redis

The LeverageBot's Python signal engine does **not** need to be rewritten in Rust. It consumes pre-aggregated signals at its existing 60-second refresh rate. Only the decoder and aggregator are latency-sensitive.

---

## Component Specifications

### 1. Rust Mempool Decoder (`mempool-decoder/`)

**Responsibilities:**
- Maintain WebSocket connection to BSC node (`wss://` endpoint)
- Subscribe to `newPendingTransactions` with full transaction objects
- For each incoming transaction:
  - Check `to` address against monitored router set (12 routers)
  - Extract first 4 bytes of calldata (function selector)
  - Match against known swap selectors (26 selectors)
  - ABI-decode parameters based on selector type (V2, V3, SmartRouter, Universal Router, aggregator)
  - Extract: `token_in`, `token_out`, `amount_in`, `amount_out_min`, `path`
  - Classify as buy or sell based on token roles (stable→volatile = buy, volatile→stable = sell)
  - Estimate USD value using cached token prices
- Deduplication via LRU hash set (capacity: 100,000 tx hashes)
- Poison detection scoring (high gas + short deadline + exact output = suspicious)

**Monitored Routers (from ArbitrageTestBot config):**

| Router | Address |
|--------|---------|
| PancakeSwap V2 | `0x10ED43C718714eb63d5aA57B78B54704E256024E` |
| PancakeSwap V3 Swap | `0x1b81D678ffb9C0263b24A97847620C99d213eB14` |
| PancakeSwap V3 Smart | `0x13f4EA83D0bd40E75C8222255bc855a974568Dd4` |
| PancakeSwap Universal | `0x1a0a18ac4BECDDbd6389559687d1a73d8927e416` |
| PancakeSwap Universal 2 | `0xd9c500dff816a1da21a48a732d3498bf09dc9aeb` |
| Biswap | `0x3a6d8cA21D1CF76F653A67577FA0D27453350dD8` |
| ApeSwap | `0xcF0feBd3f17CEf5b47b0cD257aCf6025c5BFf3b7` |
| 1inch V4 | `0x1111111254fb6c44bac0bed2854e76f90643097d` |
| ParaSwap Augustus V5 | `0xDEF171Fe48CF0115B1d80b88dc8eAB59176FEe57` |
| KyberSwap Meta V2 | `0x6131B5fae19EA4f9D964eAc0408E4408b66337b5` |
| OpenOcean V2 | `0x6352a56caadc4f1e25cd6c75970fa768a3304e64` |
| Firebird | `0x92e4f29be975c1b1eb72e77de24dccf11432a5bd` |

**Swap Selectors (26 total):**

V2 (9): `0x7ff36ab5`, `0x18cbafe5`, `0x38ed1739`, `0x8803dbee`, `0xfb3bdb41`, `0x4a25d94a`, `0xb6f9de95`, `0x791ac947`, `0x5c11d795`

V3 Standard (4): `0x414bf389`, `0xc04b8d59`, `0xdb3e2198`, `0xf28c0498`

V3 SmartRouter (6): `0x04e45aaf`, `0xb858183f`, `0x5023b4df`, `0x09b81346`, `0x472b43f3`, `0x42712a67`

Universal Router (3): `0x3593564c`, `0x24856996`, `0x24856bc3`

Aggregator (4): `0x12aa3caf`, `0x0502b1c5`, `0xe449022e`, `0x5f575529`

**Output:** Decoded `PendingSwap` structs published to Redis channel `mempool:decoded_swaps`

**Rust crates:**
- `tokio-tungstenite` — async WebSocket
- `ethabi` / `alloy-primitives` — ABI decoding
- `redis` — Redis pub/sub
- `lru` — deduplication cache
- `serde` / `serde_json` — serialization

### 2. Rust Rolling Aggregator

Can be in the same binary as the decoder or a separate module. Consumes decoded swaps and maintains rolling statistics.

**Per token pair (WBNB/USDT, WBNB/USDC, BTCB/USDT, ETH/USDT):**

| Metric | Window | Description |
|--------|--------|-------------|
| `buy_volume_usd` | 1m, 5m, 15m | Total USD volume of buy-side swaps |
| `sell_volume_usd` | 1m, 5m, 15m | Total USD volume of sell-side swaps |
| `net_flow_usd` | 1m, 5m, 15m | `buy_volume - sell_volume` |
| `buy_sell_ratio` | 1m, 5m, 15m | `buy_volume / (buy_volume + sell_volume)` [0.0-1.0] |
| `direction_score` | 5m | `(buy_sell_ratio - 0.5) * 2` [-1.0 to 1.0] |
| `tx_count_buy` | 5m | Number of buy transactions |
| `tx_count_sell` | 5m | Number of sell transactions |
| `whale_buy_count` | 15m | Buys > $10,000 USD |
| `whale_sell_count` | 15m | Sells > $10,000 USD |
| `volume_acceleration` | 5m | `current_5m_volume / trailing_30m_avg_5m_volume` |
| `poison_filtered_count` | 5m | Transactions excluded by poison detection |

**Direction classification:**
- `token_in` is a stablecoin (USDT, USDC, FDUSD, BUSD) AND `token_out` is volatile (WBNB, BTCB, ETH) → **BUY**
- `token_in` is volatile AND `token_out` is stablecoin → **SELL**
- Both volatile or both stable → **SKIP** (not directionally informative for leverage)

**USD estimation:**
- For BNB-denominated swaps: multiply by cached BNB/USD price (refreshed every 60s from Binance ticker)
- For token-denominated swaps: use `amount_in` with known decimals and cached price
- Rough estimates are acceptable — this is for aggregate signal, not precise accounting

**Output:** Aggregated `MempoolSignal` published to Redis channel `mempool:aggregate_signal` every 5 seconds:

```json
{
  "timestamp": 1738000000,
  "pairs": {
    "WBNB": {
      "buy_volume_1m_usd": 45200.50,
      "sell_volume_1m_usd": 31800.20,
      "buy_volume_5m_usd": 198500.00,
      "sell_volume_5m_usd": 165200.00,
      "buy_volume_15m_usd": 580000.00,
      "sell_volume_15m_usd": 520000.00,
      "direction_score_5m": 0.18,
      "buy_sell_ratio_5m": 0.546,
      "tx_count_buy_5m": 42,
      "tx_count_sell_5m": 35,
      "whale_buy_count_15m": 3,
      "whale_sell_count_15m": 1,
      "volume_acceleration": 1.35,
      "total_swaps_seen": 77
    },
    "BTCB": { ... },
    "ETH": { ... }
  }
}
```

### 3. Python Signal Component (`core/signal_engine.py`)

Replace the current `_compute_aggregate_mempool_flow()` method with a Redis-backed implementation that consumes the Rust aggregator's output.

**Signal logic:**

```
1. Read latest MempoolSignal from Redis (mempool:aggregate_signal)
2. For the target symbol (e.g. WBNB):
   a. Extract direction_score_5m and volume_acceleration
   b. If volume_acceleration < 1.2 → zero strength (normal flow, no edge)
   c. If abs(direction_score_5m) < 0.1 → zero strength (balanced flow)
   d. Direction: LONG if direction_score > 0, SHORT if < 0
   e. Strength: min(abs(direction_score_5m) * 2, 1.0)
   f. Confidence modifiers:
      - Whale alignment (whales agree with direction) → +0.15
      - Volume acceleration > 2.0 → +0.10
      - 1m and 5m direction agree → +0.10
      - Poison-filtered ratio > 20% → -0.20 (high sandwich activity = noisy signal)
   g. Final confidence = base_confidence * modifiers, capped at 0.8
3. Emit SignalComponent(source="mempool_order_flow", tier=2, ...)
```

**Promotion to Tier 2:** With real data instead of the stub, mempool order flow becomes a meaningful signal. Suggested weight: **0.12-0.15** (between exchange_flows at 0.08 and order_book_imbalance at 0.30). The signal captures the same phenomenon as OBI (directional order flow) but from a different data source (pending transactions vs. current order book), providing genuine diversification.

### 4. Python Data Service (`core/data_service.py`)

Replace `get_pending_swap_volume()` with a Redis consumer method:

```python
async def get_mempool_signal(self, symbol: str) -> MempoolSignal:
    """
    Read latest aggregate mempool signal from Redis.

    The Rust mempool-decoder publishes aggregated order flow
    statistics every 5 seconds. This method reads the latest
    snapshot for the given symbol.

    Returns neutral defaults if Redis is unavailable or data
    is stale (>30 seconds old).
    """
```

### 5. Configuration Updates

**`config/signals.json` — Update Tier 3 → Tier 2 promotion:**

```json
"signal_sources": {
    "tier_2": {
        "btc_volatility_spillover": { "enabled": true, "weight": "0.10" },
        "liquidation_heatmap": { "enabled": true, "weight": "0.10" },
        "exchange_flows": { "enabled": true, "weight": "0.08" },
        "funding_rate": { "enabled": true, "weight": "0.07" },
        "mempool_order_flow": {
            "enabled": true,
            "weight": "0.12",
            "redis_channel": "mempool:aggregate_signal",
            "min_volume_acceleration": 1.2,
            "min_direction_score": 0.1,
            "whale_threshold_usd": 10000,
            "stale_data_max_seconds": 30,
            "windows": ["1m", "5m", "15m"]
        }
    },
    "tier_3": {
        "social_sentiment_volume": { "enabled": false, "weight": "0.03", "note": "Requires external sentiment API." }
    }
}
```

**`config/mempool.json` (NEW in LeverageBot):**

```json
{
    "enabled": true,
    "redis": {
        "url": "redis://localhost:6379",
        "decoded_swaps_channel": "mempool:decoded_swaps",
        "aggregate_signal_channel": "mempool:aggregate_signal",
        "aggregate_publish_interval_seconds": 5
    },
    "monitored_tokens": {
        "volatile": ["WBNB", "BTCB", "ETH"],
        "stable": ["USDT", "USDC", "FDUSD"]
    },
    "aggregation": {
        "windows_seconds": [60, 300, 900],
        "whale_threshold_usd": 10000,
        "poison_filter_enabled": true,
        "poison_suspicion_threshold": 0.7
    },
    "decoder": {
        "websocket_url_env": "BSC_RPC_URL_WS",
        "websocket_fallback": "wss://bsc-ws-node.nariox.org:443",
        "dedup_cache_size": 100000,
        "reconnect_delay_seconds": 5,
        "max_reconnect_attempts": 10
    }
}
```

**`.env.example` — Add:**

```bash
# Mempool decoder
BSC_RPC_URL_WS=wss://bsc-ws-node.nariox.org:443
REDIS_URL=redis://localhost:6379
```

### 6. Shared Types (`shared/types.py`)

Replace `PendingSwapVolume` with:

```python
@dataclass(frozen=True)
class MempoolTokenSignal:
    buy_volume_1m_usd: Decimal
    sell_volume_1m_usd: Decimal
    buy_volume_5m_usd: Decimal
    sell_volume_5m_usd: Decimal
    buy_volume_15m_usd: Decimal
    sell_volume_15m_usd: Decimal
    direction_score_5m: Decimal      # [-1.0, 1.0]
    buy_sell_ratio_5m: Decimal       # [0.0, 1.0]
    tx_count_buy_5m: int
    tx_count_sell_5m: int
    whale_buy_count_15m: int
    whale_sell_count_15m: int
    volume_acceleration: Decimal     # current vs trailing avg
    total_swaps_seen: int
    timestamp: int

@dataclass(frozen=True)
class MempoolSignal:
    pairs: dict[str, MempoolTokenSignal]
    timestamp: int
    data_age_seconds: int
```

---

## Rust Project Structure

```
mempool-decoder/
├── Cargo.toml
├── src/
│   ├── main.rs                  # Entry point, WebSocket loop
│   ├── config.rs                # Load routers, selectors, token lists
│   ├── decoder/
│   │   ├── mod.rs
│   │   ├── v2.rs                # V2 swap ABI decoding
│   │   ├── v3.rs                # V3 + SmartRouter decoding
│   │   ├── universal.rs         # Universal Router command parsing
│   │   └── aggregator.rs        # 1inch, ParaSwap, etc.
│   ├── classifier.rs            # Buy/sell classification, USD estimation
│   ├── aggregator.rs            # Rolling window statistics
│   ├── poison.rs                # Sandwich detection scoring
│   ├── dedup.rs                 # LRU tx hash deduplication
│   └── redis_publisher.rs       # Publish to Redis channels
└── tests/
    ├── test_v2_decode.rs
    ├── test_v3_decode.rs
    ├── test_universal_decode.rs
    ├── test_aggregation.rs
    └── test_classifier.rs
```

---

## Mempool Visibility on BSC — Realistic Expectations

### What you will see

BSC's BEP-322 PBS means 99.8% of blocks are built by specialized builders, and many wallets route through private RPCs. However, the ArbitrageTestBot's logs confirm that pending transactions *are* visible via `newPendingTransactions` subscription. The transactions you see are those broadcast through the P2P network before builders absorb them.

The visible sample is biased toward:
- Retail users using standard RPCs (not private MEV protection)
- Smaller transactions (large/sophisticated traders use private channels)
- PancakeSwap V2 (most popular retail DEX on BSC)

### What you will NOT see

- Transactions submitted via private RPCs (48 Club, BlockRazor, Merkle)
- Transactions from wallets with built-in MEV protection (Binance Wallet, Trust Wallet, OKX)
- Builder-internal transactions
- Validator-internal transactions

### Why partial visibility is sufficient for order flow analysis

For aggregate directional signal generation, you do not need to see every transaction. You need a **statistically representative sample** that reflects the prevailing market sentiment. Retail flow is:

- **Directional and momentum-driven** — retail tends to buy during uptrends and sell during downtrends (herding behavior is well-documented in behavioral finance)
- **A leading indicator for further retail flow** — if visible retail is buying heavily, invisible retail is likely doing the same
- **Complementary to existing signals** — OBI measures the current order book; mempool flow measures what is *about to hit* the order book

The signal should be treated as a **supplementary directional bias** (weight 0.12), not a high-confidence standalone indicator. The ensemble scoring system already handles uncertain/noisy signals correctly by requiring agreement across multiple sources before acting.

---

## Build Phases

### Phase 10A: Rust Mempool Decoder

1. Initialize Rust project (`cargo init mempool-decoder`)
2. Implement WebSocket connection + `newPendingTransactions` subscription
3. Implement router matching (12 routers) and selector matching (26 selectors)
4. Implement V2 swap ABI decoding
5. Implement V3 / SmartRouter / Universal Router decoding
6. Implement aggregator (1inch, ParaSwap) decoding
7. Implement buy/sell classification and USD estimation
8. Implement deduplication (LRU hash set)
9. Implement poison detection scoring
10. Implement Redis publishing
11. Write unit tests for each decoder variant using real BSC transaction calldata (captured from ArbitrageTestBot logs)
12. Integration test: connect to BSC mainnet WebSocket, verify decoded swaps match ArbitrageTestBot output

### Phase 10B: Rust Rolling Aggregator

1. Implement sliding window data structure (1m, 5m, 15m)
2. Implement per-token-pair aggregation (WBNB, BTCB, ETH)
3. Implement direction scoring and volume acceleration
4. Implement whale detection
5. Implement Redis aggregate signal publishing (every 5 seconds)
6. Write unit tests with synthetic swap streams
7. Integration test: feed real decoded swaps, verify aggregate output

### Phase 10C: Python Integration

1. Add `redis` (aioredis) to LeverageBot dependencies
2. Add `MempoolTokenSignal` and `MempoolSignal` types to `shared/types.py`
3. Create `config/mempool.json`
4. Replace `get_pending_swap_volume()` in `core/data_service.py` with `get_mempool_signal()` (Redis consumer)
5. Replace `_compute_aggregate_mempool_flow()` in `core/signal_engine.py` with real signal component consuming Redis data
6. Promote mempool signal from Tier 3 to Tier 2 in `config/signals.json`
7. Update `main.py` to initialize Redis connection
8. Update `.env.example` with `BSC_RPC_URL_WS` and `REDIS_URL`
9. Write `tests/unit/test_mempool_signal.py` (mock Redis, verify signal computation)
10. Run `mypy --strict`, `ruff`, `black`, full test suite

### Phase 10D: End-to-End Validation

1. Run Rust decoder against BSC mainnet WebSocket
2. Verify decoded swap output matches ArbitrageTestBot for same transactions
3. Run Rust aggregator and verify aggregate signal in Redis
4. Run LeverageBot in dry-run mode consuming live mempool signal
5. Log signal component values and verify they contribute meaningfully to ensemble scoring
6. Compare signal engine output with and without mempool signal enabled

---

## Dependencies

### Rust (mempool-decoder)

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
tokio-tungstenite = { version = "0.24", features = ["native-tls"] }
alloy-primitives = "0.8"
alloy-sol-types = "0.8"
redis = { version = "0.27", features = ["tokio-comp"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
lru = "0.12"
tracing = "0.1"
tracing-subscriber = "0.3"
```

### Python (LeverageBot additions)

```toml
# Add to pyproject.toml [project.dependencies]
redis = ">=5.0,<6"
```

---

## References

### Order Flow and Price Prediction
1. Kolm, Turiel & Westray — "Deep Order Flow Imbalance" (J. Financial Economics, 2023) — OBI accounts for 73% of prediction
2. Easley, Lopez de Prado & O'Hara — "Flow Toxicity and Liquidity" (Review of Financial Studies, 2012) — VPIN framework
3. Cont, Kukanov & Stoikov — "The Price Impact of Order Book Events" (J. Financial Econometrics, 2014) — Order flow theory
4. Abad & Yague — "VPIN Predicts Crypto Price Jumps" (ScienceDirect, 2025)
5. Chi, White & Lee — "Cryptocurrency Exchange Flows and Returns" (SSRN, 2024) — Exchange flows predict returns

### Mempool and Volume Prediction
6. Ante & Saggu — "Mempool Transaction Flow and Volume Prediction" (J. Innovation & Knowledge, 2024) — Mempool predicts volume
7. Shen, Urquhart & Wang — "Does Twitter Predict Bitcoin?" (Economics Letters, 2019) — Volume > polarity for prediction

### MEV and BSC Infrastructure
8. Daian et al. — "Flash Boys 2.0" (IEEE S&P, 2020) — [arXiv:1904.05234](https://arxiv.org/abs/1904.05234)
9. BEP-322 — Builder API Specification for BNB Smart Chain — [GitHub](https://github.com/bnb-chain/BEPs/blob/master/BEPs/BEP322.md)
10. BNB Chain — "MEV Demystified" (2024) — [Blog](https://www.bnbchain.org/en/blog/mev-demystified-exploring-the-mev-landscape-in-the-bnb-chain-ecosystem)
11. Qin, Zhou & Gervais — "Quantifying Blockchain Extractable Value" (IEEE S&P, 2022) — [arXiv:2101.05511](https://arxiv.org/abs/2101.05511)
12. Torres et al. — "Frontrunner Jones and the Raiders of the Dark Forest" (USENIX Security, 2021)

### Behavioral Finance and Herding
13. Bouri, Gupta & Roubaud — "Herding behaviour in cryptocurrencies" (Finance Research Letters, 2019) — Retail herding patterns
14. Vidal-Tomas, Ibanez & Farinos — "Herding in the cryptocurrency market" (International Review of Financial Analysis, 2019)
