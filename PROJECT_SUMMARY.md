# BSC Leverage Bot — Project Summary

## What It Does

The LeverageBot is an automated leveraged trading bot for Binance Smart Chain (BSC) that opens and manages leveraged long and short positions on **Aave V3** using flash loans.

- **Longs**: Flash-borrows a stablecoin (e.g. USDT), swaps it to a volatile asset (e.g. WBNB), supplies that as collateral on Aave, and keeps the debt open — profiting if the volatile asset rises.
- **Shorts**: Flash-borrows a volatile asset, swaps it to a stablecoin, supplies the stablecoin as collateral, and keeps the volatile debt open — profiting if the volatile asset falls.
- **Monitors** positions continuously, automatically deleveraging or closing when health factor drops.
- **Generates signals** from a 5-layer ensemble of technical indicators, microstructure data (order book imbalance, VPIN), regime detection (Hurst exponent), and on-chain data (liquidation heatmaps, funding rates).

---

## Architecture

### Three Concurrent Tasks

The bot runs as a single Python asyncio process with three concurrent tasks communicating via a shared `asyncio.Queue`:

```
┌─────────────────┐     HealthStatus     ┌──────────────┐
│  HealthMonitor   │ ──────────────────→ │              │
│  (tiered polling)│                      │   Strategy    │
└─────────────────┘                      │  (decisions)  │
                                          │              │
┌─────────────────┐     TradeSignal      │              │
│  SignalEngine    │ ──────────────────→ │              │
│  (5-layer)      │                      └──────┬───────┘
└─────────────────┘                             │
                                                ↓
                                       ┌────────────────┐
                                       │ PositionManager │
                                       │ (flash loans,   │
                                       │  swaps, Aave)   │
                                       └────────────────┘
```

No multiprocessing, Redis, or external IPC — position management is not CPU-bound and a single event loop handles all I/O concurrently.

---

## Directory Structure

```
LeverageBot/
├── main.py                              # Asyncio entry point (python main.py)
├── pyproject.toml                       # Dependencies & tool config
├── .env.example                         # Environment variable template
│
├── config/                              # Configuration
│   ├── loader.py                        # Singleton ConfigLoader with @lru_cache
│   ├── validate.py                      # Schema validation at startup
│   ├── app.json                         # Logging & general settings
│   ├── aave.json                        # Aave V3 risk params & supported assets
│   ├── aggregator.json                  # DEX provider configs (1inch, OpenOcean, ParaSwap)
│   ├── signals.json                     # Signal engine & indicator parameters
│   ├── positions.json                   # Position limits & risk thresholds
│   ├── timing.json                      # Polling intervals & timeouts
│   ├── rate_limits.json                 # API rate limiting
│   ├── chains/
│   │   └── 56.json                      # BSC chain config (contracts, tokens, feeds)
│   └── abis/                            # Smart contract ABIs
│       ├── aave_v3_pool.json
│       ├── aave_v3_data_provider.json
│       ├── aave_v3_oracle.json
│       ├── leverage_executor.json       # Custom flash loan receiver (not yet deployed)
│       ├── erc20.json
│       ├── chainlink_aggregator_v3.json
│       └── multicall3.json
│
├── core/                                # Core trading logic
│   ├── signal_engine.py                 # 5-layer multi-source signal generation
│   ├── health_monitor.py                # Tiered HF polling with Chainlink checks
│   ├── strategy.py                      # Decision engine (entry, exit, deleverage)
│   ├── position_manager.py              # Position lifecycle (open, close, increase)
│   ├── data_service.py                  # Market data fetcher (Binance, Aave subgraph)
│   ├── indicators.py                    # Technical indicators & statistics
│   ├── safety.py                        # Kill switches & safety gates
│   └── pnl_tracker.py                  # SQLite P&L and position history
│
├── execution/                           # On-chain interaction layer
│   ├── aave_client.py                   # Aave V3 read + calldata encoding
│   ├── aggregator_client.py             # DEX aggregator fan-out (parallel quotes)
│   └── tx_submitter.py                  # Tx signing, simulation, MEV-protected submit
│
├── shared/                              # Shared types & constants
│   ├── types.py                         # Dataclasses & enums
│   ├── constants.py                     # Contract addresses, numeric constants
│   └── serialization_utils.py           # JSON encoder for Decimal/HexBytes
│
├── bot_logging/
│   └── logger_manager.py               # Per-module log file organization
│
├── tests/
│   ├── conftest.py                      # Shared pytest fixtures
│   ├── unit/                            # 358 unit tests across 13 files
│   └── integration/                     # Anvil fork tests (3, skipped without fork)
│
├── data/
│   └── positions.db                     # SQLite database (created at runtime)
│
├── logs/                                # Runtime logs (one folder per module)
│
└── contracts/
    └── foundry.toml                     # Foundry scaffold for LeverageExecutor
```

---

## Component Details

### 1. HealthMonitor (`core/health_monitor.py`)

Continuously polls Aave V3 for the wallet's health factor at dynamic intervals based on risk tier:

| Tier | Health Factor | Poll Interval |
|------|--------------|---------------|
| SAFE | > 2.0 | 15 seconds |
| WATCH | 1.5 – 2.0 | 5 seconds |
| WARNING | 1.3 – 1.5 | 2 seconds |
| CRITICAL | < 1.3 | 1 second |

Additional capabilities:
- Checks Chainlink oracle freshness (rejects stale feeds based on heartbeat)
- Predicts HF degradation over time via interest rate accrual (Taylor series approximation)
- Triggers global pause after 5 consecutive RPC failures

### 2. SignalEngine (`core/signal_engine.py`)

5-layer multi-source signal pipeline:

**Layer 1 — Regime Detection**
- Hurst exponent classifies market as trending (>0.55), mean-reverting (<0.45), ranging, or volatile

**Layer 2 — Directional Signals (3 tiers)**

| Tier | Weight | Sources |
|------|--------|---------|
| Tier 1 | 75% | Technical indicators (EMA, RSI, MACD, BB), order book imbalance, VPIN |
| Tier 2 | 25% | BTC volatility spillover, liquidation heatmap, exchange flows, funding rates |
| Tier 3 | Disabled | Mempool flow (limited on BSC), social sentiment (requires external API) |

**Layer 3 — Ensemble Scoring**
- Weighted aggregation with regime-dependent multipliers (e.g. trending regime boosts momentum signals 1.2x, reduces mean-reversion 0.5x)

**Layer 4 — Position Sizing**
- Fractional Kelly criterion (25% of full Kelly) with GARCH(1,1) volatility adjustment
- Size reduced further during high volatility (>4%) or drawdown periods (>10%)

**Layer 5 — Alpha Decay Monitoring**
- Compares rolling 30-day accuracy vs. historical 180-day accuracy
- If recent accuracy < 70% of historical, raises the minimum confidence threshold by 10%

### 3. Strategy (`core/strategy.py`)

Consumes both HealthStatus and TradeSignal events from the shared queue:

**Health-driven actions:**
- CRITICAL tier: immediately closes position
- WARNING tier: deleverages to target HF 1.8

**Signal-driven entry (all must pass):**
1. Ensemble confidence >= 70% (dynamically adjusted by alpha decay)
2. Maximum 3 signals per day not exceeded
3. Borrow rate < 15% APR and borrow cost < 0.5% of notional
4. Stress test passes (simulates -5% to -30% price drops, HF must stay > 1.1)
5. Liquidation cascade check (if >$50M TVL at risk, applies extra -3% price drop)
6. Position size within configured limits

**Exit rules:**
- Take-profit at +5% gain
- Stop-loss at -3% loss
- Trailing stop at 2% below peak
- Max hold duration of 168 hours (1 week)

### 4. PositionManager (`core/position_manager.py`)

Orchestrates the flash loan lifecycle:

1. Gets best swap quote from 3 DEX aggregators in parallel
2. Validates quote against Chainlink oracle (rejects >1% divergence)
3. Encodes Aave V3 flash loan calldata
4. Runs all safety gate checks (leverage, size, cooldown, gas price)
5. Simulates transaction via `eth_call`
6. Submits via MEV-protected RPC
7. Records position in SQLite via PnLTracker

Also handles closing (flash loan with mode=0 to repay), deleveraging (partial close to reach target HF), and stress testing before every entry.

### 5. Execution Layer (`execution/`)

**`aave_client.py`** — Async reads from Aave V3 contracts (getUserAccountData, getReserveData, getAssetPrice) and sync ABI calldata encoding for flash loans, supply, withdraw, and repay operations.

**`aggregator_client.py`** — Parallel fan-out to 1inch, OpenOcean, and ParaSwap. Selects best quote by output amount. Enforces router address whitelist and oracle divergence checks. Caches quotes for 10 seconds.

**`tx_submitter.py`** — Signs transactions with the executor private key, simulates via `eth_call` before submission, submits to MEV-protected RPC (`rpc.48.club`) to avoid sandwich attacks. Manages nonce state with async locking and periodic on-chain refresh.

### 6. Safety (`core/safety.py`)

Centralized safety gates checked before every action:

- Global pause flag (programmatic or via `PAUSE` sentinel file in project root)
- Leverage ratio cap (default 3.0x)
- Position size cap (default $10,000)
- Gas price cap (default 10 gwei)
- Cooldown between actions (30 seconds)
- Daily transaction limit (50/day)
- All checks default to "deny" on error — the system refuses to trade rather than trading incorrectly

### 7. PnL Tracker (`core/pnl_tracker.py`)

SQLite database (`data/positions.db`) with three tables:

- **positions**: Full lifecycle (open/close timestamps, tokens, amounts, realized P&L)
- **position_snapshots**: Periodic snapshots (HF, unrealized P&L, borrow rate)
- **transactions**: Every on-chain tx (hash, gas cost, success/revert)

Provides rolling trading statistics (win rate, Sharpe ratio, max drawdown) used by the alpha decay monitoring system.

### 8. Data Service (`core/data_service.py`)

Multi-source market data with caching and graceful degradation:

| Data | Source | Cache TTL |
|------|--------|-----------|
| OHLCV candles | Binance Spot `/klines` | 60s |
| Order book | Binance Spot `/depth` | 5s |
| Recent trades | Binance Spot `/aggTrades` | 10s |
| Funding rate | Binance Futures `/fundingRate` | 300s |
| Liquidation levels | Aave V3 subgraph | 120s |
| Exchange flows | Binance (proxy via trade volume) | 60s |

Network failures return empty/zero-value defaults rather than crashing.

### 9. Indicators (`core/indicators.py`)

Pure computation module (no I/O). All calculations use `Decimal` for precision:

- **Standard**: EMA, RSI (Wilder's smoothing), MACD, Bollinger Bands, ATR
- **Statistical**: Hurst exponent (R/S analysis), GARCH(1,1) volatility forecast
- **Microstructure**: VPIN (volume-synchronized probability of informed trading), order book imbalance

---

## How to Run

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env — see "What's Missing" section below

# 4. Run the bot (dry-run by default)
python main.py

# 5. Run tests
pytest tests/unit/ -v
```

The entrypoint is `main.py` at the project root. It runs in **dry-run mode by default** (`EXECUTOR_DRY_RUN=true`), meaning the full signal/health/strategy pipeline executes but no transactions are submitted on-chain.

### Startup Sequence

1. Loads `.env` via python-dotenv
2. Validates all config files (exits on failure)
3. Connects to BSC RPC via AsyncWeb3
4. Instantiates all components in dependency order
5. Launches 3 concurrent asyncio tasks
6. Waits for SIGINT/SIGTERM, then gracefully shuts down (cooperative stop, task cancellation, session cleanup)

---

## Environment Variables (`.env`)

```bash
# BSC RPC
BSC_RPC_URL_HTTP=https://bsc-dataseed1.binance.org/
BSC_RPC_URL_HTTP_FALLBACK=https://bsc-dataseed2.binance.org/
BSC_RPC_URL_MEV_PROTECTED=https://rpc.48.club

# Execution (REQUIRED for live trading)
LEVERAGE_EXECUTOR_ADDRESS=               # Deployed LeverageExecutor contract
EXECUTOR_PRIVATE_KEY=                    # Private key for signing transactions
EXECUTOR_DRY_RUN=true                    # Set to false for live execution

# Aave V3
AAVE_V3_POOL_ADDRESS=0x6807dc923806fE8Fd134338EABCA509979a7e0cB

# Position Management (overrides config/positions.json)
MAX_LEVERAGE_RATIO=3.0
MIN_HEALTH_FACTOR=1.5
MAX_FLASH_LOAN_USD=5000
MAX_GAS_PRICE_GWEI=10

# API Keys
ONEINCH_API_KEY=                         # Required for 1inch aggregator
BINANCE_API_KEY=                         # Optional (public endpoints used by default)
BINANCE_API_SECRET=                      # Optional

# User Wallet (REQUIRED)
USER_WALLET_ADDRESS=                     # BSC address to monitor and manage positions for
```

---

## Key Configuration Files

### `config/positions.json` — Risk Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dry_run` | `true` | No real transactions when enabled |
| `max_flash_loan_usd` | 5,000 | Max single position size |
| `max_position_usd` | 10,000 | Total portfolio limit |
| `max_leverage_ratio` | 3.0x | Maximum leverage |
| `min_health_factor` | 1.5 | Minimum acceptable HF |
| `deleverage_threshold` | 1.4 | Auto-deleverage when HF drops to this |
| `close_threshold` | 1.25 | Emergency close when HF drops to this |
| `target_hf_after_deleverage` | 1.8 | Rebalance target after deleveraging |
| `max_gas_price_gwei` | 10 | Skip transaction if gas exceeds this |
| `max_slippage_bps` | 50 | 0.5% max slippage on swaps |
| `max_transactions_per_24h` | 50 | Daily transaction rate limit |

### `config/signals.json` — Signal Engine

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `blended` | Signal mode: `blended`, `momentum`, or `mean_reversion` |
| `min_confidence` | 0.7 | 70% ensemble confidence required to trade |
| `max_signals_per_day` | 3 | Maximum entries per day |
| `kelly_fraction` | 0.25 | Use 25% of full Kelly for position sizing |
| `take_profit_percent` | 5.0% | Close at +5% gain |
| `stop_loss_percent` | 3.0% | Close at -3% loss |
| `trailing_stop_percent` | 2.0% | Trail at 2% below peak |
| `max_hold_hours` | 168 | Maximum position duration (1 week) |

### `config/aave.json` — Aave V3 Asset Config

| Asset | LTV | Liquidation Threshold | Notes |
|-------|-----|----------------------|-------|
| WBNB | 75% | 80% | Primary long collateral |
| USDC | 77% | 80% | Preferred short collateral |
| USDT | 75% | 78% | Isolation mode enabled |
| BTCB | 70% | 75% | — |
| ETH | 80% | 82.5% | Highest LTV |

### `config/aggregator.json` — DEX Routing

| Provider | Priority | Rate Limit | Router |
|----------|----------|------------|--------|
| 1inch | 1 | 1 req/s | `0x1111111254...` |
| OpenOcean | 2 | 2 req/s | `0x6352a56caa...` |
| ParaSwap | 3 | 2 req/s | `0xDEF171Fe48...` |

---

## Data Flow: Opening a Long Position

```
1. SignalEngine fetches 200 hourly WBNB/USDT candles from Binance
2. Computes indicators: EMA(20/50/200), RSI, MACD, BB, ATR, Hurst, GARCH
3. Fetches order book (20 levels) → computes OBI
4. Fetches recent trades (1000) → computes VPIN
5. Classifies regime via Hurst exponent (e.g. "trending")
6. Generates SignalComponents from each enabled source
7. Ensemble scoring: weighted average = 0.82 confidence, direction = LONG
8. Fractional Kelly sizing: $2,400 recommended
9. TradeSignal placed in asyncio.Queue

10. Strategy consumes signal from queue
11. Checks: confidence (0.82 >= 0.70) ✓
12. Checks: signals today (1 < 3) ✓
13. Fetches borrow rate for USDT: 4.2% APR (< 15%) ✓
14. Stress test: simulates -5% to -30% BNB drops, HF stays > 1.1 ✓
15. Cascade check: TVL at risk < $50M ✓

16. PositionManager.open_position(LONG, debt=USDT, collateral=WBNB, $2,400)
17. AggregatorClient fans out to 1inch + OpenOcean + ParaSwap in parallel
18. Best quote selected (highest WBNB output), validated vs Chainlink oracle
19. AaveClient encodes flashLoan calldata (mode=2, keep variable debt)
20. SafetyState.can_open_position() → approved
21. TxSubmitter simulates via eth_call → success
22. TxSubmitter signs and submits via MEV-protected RPC
23. Waits for confirmation (max 60s)
24. PnLTracker records position in SQLite

25. HealthMonitor now polls this position's HF continuously
```

---

## What's Missing for Full Functionality

### Blockers (Must Have)

#### 1. LeverageExecutor Smart Contract

The bot relies on a custom Solidity contract that acts as the Aave V3 flash loan receiver. This contract must implement `IFlashLoanReceiver.executeOperation()` to atomically: receive flash loan → swap via DEX router → supply collateral to Aave → leave debt open.

- The ABI exists (`config/abis/leverage_executor.json`, 635 lines)
- A Foundry project is scaffolded in `contracts/`
- **The contract has not been written or deployed**
- `config/chains/56.json` has `"leverage_executor": ""` (empty)

**Action required**: Write the Solidity contract, test on Anvil fork, deploy to BSC mainnet, and set `LEVERAGE_EXECUTOR_ADDRESS` in `.env`.

#### 2. Wallet with Private Key

Two environment variables must be configured:

| Variable | Purpose |
|---|---|
| `USER_WALLET_ADDRESS` | BSC wallet the bot monitors and manages positions for. The bot exits immediately if this is not set. |
| `EXECUTOR_PRIVATE_KEY` | Private key for signing transactions. This wallet must hold BNB for gas and be authorized to call the LeverageExecutor contract. |

#### 3. Initial Collateral on Aave

The wallet needs collateral already supplied to Aave V3 (or sufficient tokens to supply). The flash loan mechanism borrows against the user's Aave position — without existing collateral, the first flash loan with `mode=2` (keep debt) will revert because there is nothing backing the new debt.

#### 4. Token Approvals

The wallet must grant ERC-20 approvals to:
- **Aave V3 Pool** (`0x6807dc923806fE8Fd134338EABCA509979a7e0cB`) — to spend tokens for supply/repay
- **LeverageExecutor contract** — to move tokens on behalf of the wallet
- **DEX router addresses** — for swap execution (addresses listed in `config/aggregator.json`)

#### 5. Disable Dry-Run

Set `EXECUTOR_DRY_RUN=false` in `.env`. While dry-run is enabled (the default), the Strategy logs what it would do but never submits transactions.

### Recommended (Should Have)

#### 6. 1inch API Key

Set `ONEINCH_API_KEY` in `.env`. Without it, 1inch quotes will fail and only OpenOcean and ParaSwap will be available for swap routing.

#### 7. Private RPC Endpoint

The default public BSC RPCs (`bsc-dataseed1.binance.org`) have rate limits and reliability issues. For production use, a paid RPC provider (QuickNode, Ankr, Chainstack, etc.) is strongly recommended — especially since the health monitor polls every 1–2 seconds during critical tier.

### Optional

#### 8. Binance API Keys

`BINANCE_API_KEY` and `BINANCE_API_SECRET` are optional. The bot uses Binance public endpoints for market data. Authenticated access provides higher rate limits.

---

## Readiness Checklist

| Item | Status | Priority |
|---|---|---|
| LeverageExecutor contract deployed | Not done | **Blocker** |
| `USER_WALLET_ADDRESS` configured | Not done | **Blocker** |
| `EXECUTOR_PRIVATE_KEY` configured | Not done | **Blocker** |
| Initial collateral supplied to Aave | Not done | **Blocker** |
| Token approvals granted | Not done | **Blocker** |
| `EXECUTOR_DRY_RUN=false` | Default `true` | **Required for live** |
| `ONEINCH_API_KEY` set | Not done | Recommended |
| Private RPC endpoint | Using public default | Recommended |
| Binance API keys | Not done | Optional |

---

## Test Suite

**358 unit tests passing** across 13 test files. All static analysis clean:

| Tool | Result |
|------|--------|
| mypy --strict | 0 errors (23 source files) |
| ruff | All checks passed |
| black | All files formatted |
| pytest | 358 passed |

3 integration tests exist in `tests/integration/test_aave_fork.py` but require an Anvil fork node (`--fork-url`) and are skipped without it.

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with integration tests (requires Anvil)
ANVIL_FORK_URL=https://your-rpc pytest tests/ -v

# Static analysis
mypy --strict main.py core/ execution/ config/ shared/ bot_logging/
ruff check .
black --check .
```

---

## Dependencies

```
Python >=3.10, <3.12

Runtime:
  web3 >=6.12        # AsyncWeb3, contract interaction
  aiohttp >=3.9      # Async HTTP (Binance, DEX APIs)
  python-dotenv >=1.0 # .env file loading
  eth-abi >=4.2       # ABI encoding/decoding
  eth-account >=0.10  # Account management, tx signing

Dev:
  pytest, pytest-asyncio, pytest-mock
  aioresponses        # Mock HTTP responses
  mypy, ruff, black   # Static analysis & formatting
```

No external database drivers — SQLite via Python's `sqlite3` stdlib.
