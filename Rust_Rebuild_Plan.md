# BSC Leverage Bot — Comprehensive Rust Rebuild Plan

## Executive Summary

This document specifies a complete rebuild of the BSC Leverage Bot from Python (asyncio + web3.py) to Rust (tokio + Alloy). The rebuild preserves the existing 5-layer signal architecture, Aave V3 flash loan execution model, and all risk management logic while gaining Rust's compile-time safety guarantees, deterministic latency, and superior throughput.

### Why Rebuild in Rust

| Dimension | Python (Current) | Rust (Target) |
|-----------|-----------------|---------------|
| Latency determinism | GIL contention, GC pauses (10-50ms), asyncio event loop jitter | No GC, zero-cost abstractions, predictable latency (<1ms jitter) |
| Memory safety | Runtime errors (None, type mismatches) caught only during execution | Compile-time ownership, borrowing, and lifetime checks eliminate use-after-free, data races, null pointer dereferences (Jung et al., "Safe Systems Programming in Rust," ACM CACM; RustBelt, POPL 2018 — first machine-checked safety proof for Rust's type system) |
| Numerical precision | `Decimal` via Python stdlib — correct but slow (pure Python arbitrary precision) | `rust_decimal` (96-bit integer + scale) for off-chain math; `alloy_primitives::U256` (ruint-backed, 35-60% faster than ethers-rs) for on-chain values (Paradigm, "Introducing Alloy v1.0," 2025) |
| ABI encoding | Runtime JSON ABI parsing via web3.py (~100µs per encode) | Compile-time ABI via `sol!` macro — up to 10x faster static encoding (Paradigm, 2025) |
| Concurrency model | Single-threaded asyncio; CPU-bound tasks (GARCH, Hurst, VPIN) block the event loop | Multi-threaded Tokio runtime; CPU-bound tasks run on `spawn_blocking` pool without blocking I/O tasks |
| Error handling | Runtime exceptions; missed `except` clauses cause crashes | `Result<T, E>` algebraic types; the compiler forces handling every error path (`thiserror` for typed errors, `anyhow` for contextual propagation — Palmieri, "Zero to Production in Rust") |
| Dependency safety | pip ecosystem has no compile-time guarantees; transitive dependency conflicts common | Cargo's semver resolver prevents diamond dependency conflicts; `cargo audit` checks for known vulnerabilities |
| Type system | Optional type hints (mypy --strict); enforcement is opt-in and incomplete | Types are mandatory and checked at compile time; generic traits enable zero-cost polymorphism |
| Performance (CPU) | Interpreted; 10-100x slower than compiled for numerical tasks (JetBrains, "Rust vs Python," 2025; benchmarks show Rust completes CPU-intensive tasks up to 10x faster) | Compiled to native code; LLVM optimizations; SIMD-capable |
| Performance (I/O) | Comparable (asyncio + aiohttp are efficient for I/O-bound work) | Comparable (tokio + reqwest are efficient for I/O-bound work); advantage in burst handling |

**Academic justification for language choice**: Berger & Zorn (2006) established that memory safety bugs account for ~70% of critical software vulnerabilities. The White House ONCD (2024) formally recommended memory-safe languages for critical infrastructure including financial systems. Rust is the only systems language that eliminates these classes of bugs at compile time without garbage collection overhead (Jung et al., ACM CACM). For a trading bot managing leveraged DeFi positions — where a single bug can cause liquidation — compile-time safety is not a performance optimization but a risk management requirement.

---

## Architecture Decision Summary

The Rust rebuild follows the same logical architecture as the Python implementation but leverages Rust's type system, ownership model, and async runtime for stronger guarantees:

| Component | Python Pattern | Rust Pattern |
|-----------|---------------|-------------|
| Async runtime | `asyncio` single-thread event loop | `tokio` multi-threaded work-stealing runtime |
| Inter-task communication | `asyncio.Queue` (untyped) | `tokio::sync::mpsc` bounded channels (typed, backpressure) |
| Shared state | Mutable references passed between components | `Arc<T>` for shared immutable state; `Arc<Mutex<T>>` or actor pattern for mutable state (Ryhl, "Actors with Tokio") |
| Contract interaction | `web3.py` with runtime JSON ABI parsing | Alloy `sol!` macro with compile-time ABI generation |
| Provider | `AsyncWeb3` with `HTTPProvider` | `alloy::provider::ProviderBuilder` with HTTP/WS transports |
| Transaction signing | `eth_account.Account.sign_transaction()` | `alloy::signer::local::PrivateKeySigner` |
| HTTP client | `aiohttp.ClientSession` | `reqwest::Client` with `reqwest-middleware` (retry + rate-limit) |
| Configuration | JSON files + `@lru_cache` loader | `config` crate (layered JSON + env overrides) + `serde` deserialization into typed structs |
| Logging | Python `logging` with per-module file handlers | `tracing` + `tracing-subscriber` with per-module `Layer` configuration |
| Database | `sqlite3` stdlib (sync, wrapped in async) | `sqlx` with compile-time checked queries (native async SQLite) |
| Decimal math | Python `decimal.Decimal` (arbitrary precision) | `rust_decimal::Decimal` (128-bit, 28-29 significant digits) |
| EVM math | `int` with manual WAD/RAY scaling | `alloy_primitives::U256` with type-safe WAD/RAY newtypes |
| Error handling | `try/except` with string messages | `thiserror` enums for typed errors; `anyhow::Context` for propagation |
| Testing | `pytest` + `pytest-asyncio` + `aioresponses` | `#[tokio::test]` + `wiremock` + `mockall` + `proptest` + Anvil fork tests |

### Graceful Shutdown Pattern

The Python bot uses `asyncio.Event` for shutdown signaling. The Rust equivalent uses `tokio_util::sync::CancellationToken` — a hierarchical cancellation primitive that propagates shutdown across all spawned tasks (documented in "Rust Tokio Task Cancellation Patterns," cybernetist.com, 2024):

```rust
let shutdown = CancellationToken::new();
let health_shutdown = shutdown.child_token();
let signal_shutdown = shutdown.child_token();
let strategy_shutdown = shutdown.child_token();

// Each task checks: tokio::select! { _ = shutdown.cancelled() => break, ... }
// Main: signal::ctrl_c().await => shutdown.cancel();
```

### Backpressure via Bounded Channels

The Python bot uses unbounded `asyncio.Queue`. The Rust rebuild uses bounded `tokio::sync::mpsc::channel` to implement backpressure — if the Strategy consumer falls behind, the HealthMonitor and SignalEngine producers block rather than consuming unbounded memory (Biriukov, "Async Rust with Tokio I/O Streams: Backpressure, Concurrency, and Ergonomics"):

```rust
let (health_tx, signal_rx) = tokio::sync::mpsc::channel::<SignalEvent>(64);
// Producers: health_tx.send(event).await blocks if 64 events buffered
// Consumer: signal_rx.recv().await returns None when all producers dropped
```

---

## Technology Stack

### Core Dependencies

```toml
[dependencies]
# === Blockchain / EVM ===
alloy = { version = "1.5", features = ["full", "node-bindings"] }

# === Async Runtime ===
tokio = { version = "1.49", features = ["full"] }
tokio-util = { version = "0.7", features = ["rt"] }  # CancellationToken
tokio-tungstenite = { version = "0.28", features = ["native-tls"] }

# === HTTP / Networking ===
reqwest = { version = "0.12", features = ["json", "gzip", "rustls-tls"] }
reqwest-middleware = "0.5"
reqwest-retry = "0.9"

# === Numerical / Financial ===
rust_decimal = { version = "1.39", features = ["maths", "serde-str"] }
rust_decimal_macros = "1.39"

# === Data / Persistence ===
sqlx = { version = "0.8", features = ["runtime-tokio", "sqlite", "migrate"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
redis = { version = "1.0", features = ["aio", "tokio-comp"] }

# === Technical Analysis ===
ta = "0.4"

# === Statistics / Numerics ===
statrs = "0.18"
ndarray = "0.17"
ndarray-stats = "0.7"

# === Configuration ===
config = "0.15"
dotenvy = "0.15"

# === Logging / Tracing ===
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
tracing-appender = "0.2"

# === Error Handling ===
anyhow = "1"
thiserror = "2"

# === Utilities ===
chrono = { version = "0.4", features = ["serde"] }
lru = "0.12"

[dev-dependencies]
proptest = "1.9"
wiremock = "0.6"
mockall = "0.13"
```

### Crate Selection Rationale

| Crate | Replaces (Python) | Why This Crate |
|-------|-------------------|---------------|
| `alloy 1.5` | `web3.py 6.x` | Paradigm's official ethers-rs successor; `sol!` macro for compile-time ABI; 35-60% faster U256 via ruint; BSC-compatible out of the box (Paradigm, "Introducing Alloy v1.0," May 2025) |
| `rust_decimal 1.39` | `decimal.Decimal` | 128-bit decimal with `maths` feature (pow, ln, exp); sufficient for all off-chain financial math (28-29 significant digits); `serde-str` feature serializes as strings in JSON (prevents floating-point corruption) |
| `sqlx 0.8` | `sqlite3` stdlib | Compile-time checked SQL queries catch schema mismatches at build time; native async with Tokio; migration support; note: do not combine with `rusqlite` in same binary (semver hazard from duplicate `libsqlite3-sys`) |
| `ta 0.4` | Custom `indicators.py` | Streaming `Next` trait API matches polling-based signal updates; covers EMA, RSI, MACD, Bollinger Bands; standard TA parameters prevent overfitting per Hudson & Urquhart (2019) |
| `statrs 0.18` | Custom GARCH/Hurst | Statistical distributions (Normal, Student-t) for GARCH likelihood estimation and confidence intervals |
| `ndarray 0.17` | Python lists + manual iteration | NumPy-equivalent N-dimensional arrays; foundation for Hurst R/S analysis and GARCH parameter estimation |
| `config 0.15` | Custom `config/loader.py` | Layered configuration (JSON + env overrides) matching 12-factor app principles; deserializes directly into typed Rust structs |
| `tracing 0.1` | Python `logging` | Async-aware structured logging with span-based tracing; `#[instrument]` macro for automatic function-level tracing; JSON output via `tracing-subscriber` |
| `reqwest 0.12` + `reqwest-middleware` | `aiohttp.ClientSession` | Connection pooling, JSON deserialization, gzip; middleware chain for retry (exponential backoff) and rate limiting per DEX aggregator |
| `redis 1.0` | N/A (new in Rust) | Async pub/sub for mempool decoder IPC; split sink/stream for concurrent subscribe + publish |
| `wiremock 0.6` | `aioresponses` | HTTP mock server for black-box testing of Binance API, 1inch, OpenOcean, ParaSwap clients |
| `mockall 0.13` | `pytest-mock` | Auto-generate mocks from traits; used for mocking provider, signer, and data service interfaces |
| `proptest 1.9` | N/A | Property-based testing for edge cases in health factor math, Kelly sizing, GARCH estimation; generates random inputs and finds minimal failing cases |

---

## Revised File Structure

```
leverage-bot/
├── Cargo.toml                              # Workspace root
├── Cargo.lock
├── .env.example
├── .gitignore
├── rust-toolchain.toml                     # Pin Rust version (MSRV 1.88+ for Alloy)
│
├── crates/
│   ├── bot/                                # Main bot binary
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs                     # Tokio entrypoint, task orchestration
│   │       ├── config/
│   │       │   ├── mod.rs                  # Config loading + validation
│   │       │   ├── types.rs                # Typed config structs (BotConfig, SignalConfig, etc.)
│   │       │   └── validate.rs             # Schema validation
│   │       ├── core/
│   │       │   ├── mod.rs
│   │       │   ├── health_monitor.rs       # Tiered HF polling + oracle freshness
│   │       │   ├── strategy.rs             # Entry/exit logic + risk engine
│   │       │   ├── signal_engine.rs        # 5-layer signal pipeline
│   │       │   ├── indicators.rs           # EMA, RSI, MACD, BB, ATR, Hurst, GARCH, VPIN, OBI
│   │       │   ├── data_service.rs         # Multi-source market data (Binance, Chainlink, Aave)
│   │       │   ├── position_manager.rs     # Position lifecycle orchestration
│   │       │   ├── pnl_tracker.rs          # SQLite P&L tracking
│   │       │   └── safety.rs              # Kill switches, dry-run, global pause
│   │       ├── execution/
│   │       │   ├── mod.rs
│   │       │   ├── aave_client.rs          # Aave V3 read + calldata encoding
│   │       │   ├── aggregator_client.rs    # DEX aggregator fan-out
│   │       │   └── tx_submitter.rs         # Signing, simulation, MEV-protected submission
│   │       ├── types/
│   │       │   ├── mod.rs                  # Re-exports
│   │       │   ├── position.rs             # PositionDirection, PositionState, PositionAction
│   │       │   ├── signal.rs               # TradeSignal, SignalComponent, MarketRegime
│   │       │   ├── market_data.rs          # OHLCV, OrderBookSnapshot, Trade, ExchangeFlows
│   │       │   ├── aave.rs                 # UserAccountData, ReserveData, BorrowRateInfo
│   │       │   ├── aggregator.rs           # SwapQuote, AggregatorProvider
│   │       │   ├── health.rs               # HFTier, HealthStatus
│   │       │   ├── pnl.rs                  # RealizedPnL, TradingStats, StrategyHealthReport
│   │       │   └── mempool.rs              # MempoolSignal, MempoolTokenSignal
│   │       ├── constants.rs                # WAD, RAY, addresses, flash loan modes
│   │       └── errors.rs                   # Typed error enums (thiserror)
│   │
│   └── mempool-decoder/                    # Standalone mempool decoder binary
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs                     # WebSocket loop, entry point
│           ├── config.rs                   # Router addresses, selectors, token lists
│           ├── decoder/
│           │   ├── mod.rs
│           │   ├── v2.rs                   # V2 swap ABI decoding (9 selectors)
│           │   ├── v3.rs                   # V3 + SmartRouter decoding (10 selectors)
│           │   ├── universal.rs            # Universal Router command parsing (3 selectors)
│           │   └── aggregator.rs           # 1inch, ParaSwap, etc. (4 selectors)
│           ├── classifier.rs               # Buy/sell classification, USD estimation
│           ├── aggregator.rs               # Rolling window statistics (1m, 5m, 15m)
│           ├── poison.rs                   # Sandwich detection scoring
│           ├── dedup.rs                    # LRU tx hash deduplication
│           └── redis_publisher.rs          # Publish to Redis channels
│
├── config/                                 # Runtime configuration (JSON)
│   ├── app.json
│   ├── aave.json
│   ├── aggregator.json
│   ├── signals.json
│   ├── positions.json
│   ├── timing.json
│   ├── rate_limits.json
│   ├── mempool.json
│   └── chains/
│       └── 56.json                         # BSC chain config
│
├── contracts/                              # Solidity (Foundry — unchanged)
│   ├── foundry.toml
│   ├── src/
│   │   ├── LeverageExecutor.sol
│   │   └── interfaces/
│   │       ├── IFlashLoanReceiver.sol
│   │       └── IAaveV3Pool.sol
│   └── test/
│       └── LeverageExecutor.t.sol
│
├── migrations/                             # SQLx migrations
│   └── 001_initial.sql                     # positions, position_snapshots, transactions
│
└── tests/                                  # Integration tests
    ├── common/
    │   └── mod.rs                          # Shared test fixtures
    ├── aave_fork_test.rs                   # Anvil BSC fork: long + short lifecycle
    ├── aggregator_integration_test.rs      # Live aggregator quote validation
    └── signal_pipeline_test.rs             # End-to-end signal generation
```

### Workspace Structure Rationale

The project uses a Cargo workspace with two binary crates:

1. **`crates/bot/`** — The main leverage bot. Single binary, multi-threaded Tokio runtime.
2. **`crates/mempool-decoder/`** — Standalone mempool decoder. Runs independently, communicates via Redis.

This mirrors the Python architecture where the mempool decoder is a separate process. The workspace allows shared dependencies (Alloy, serde, etc.) to be compiled once and linked into both binaries.

---

## Contract Interface Definitions

The `sol!` macro generates compile-time type-safe bindings for all on-chain interactions. These replace the runtime JSON ABI parsing in Python's web3.py.

### Aave V3 Pool Interface

```rust
use alloy::sol;

sol! {
    #[sol(rpc)]
    interface IPool {
        function flashLoan(
            address receiverAddress,
            address[] calldata assets,
            uint256[] calldata amounts,
            uint256[] calldata interestRateModes,
            address onBehalfOf,
            bytes calldata params,
            uint16 referralCode
        ) external;

        function getUserAccountData(address user) external view returns (
            uint256 totalCollateralBase,
            uint256 totalDebtBase,
            uint256 availableBorrowsBase,
            uint256 currentLiquidationThreshold,
            uint256 ltv,
            uint256 healthFactor
        );

        function supply(
            address asset, uint256 amount, address onBehalfOf, uint16 referralCode
        ) external;

        function withdraw(
            address asset, uint256 amount, address to
        ) external returns (uint256);

        function repay(
            address asset, uint256 amount, uint256 interestRateMode, address onBehalfOf
        ) external returns (uint256);

        function getReserveData(address asset) external view returns (
            uint256 configuration,
            uint128 liquidityIndex,
            uint128 currentLiquidityRate,
            uint128 variableBorrowIndex,
            uint128 currentVariableBorrowRate,
            uint128 currentStableBorrowRate,
            uint40 lastUpdateTimestamp,
            uint16 id,
            address aTokenAddress,
            address stableDebtTokenAddress,
            address variableDebtTokenAddress,
            address interestRateStrategyAddress,
            uint128 accruedToTreasury,
            uint128 unbacked,
            uint128 isolationModeTotalDebt
        );
    }
}

sol! {
    #[sol(rpc)]
    interface IPoolDataProvider {
        function getReserveConfigurationData(address asset) external view returns (
            uint256 decimals,
            uint256 ltv,
            uint256 liquidationThreshold,
            uint256 liquidationBonus,
            uint256 reserveFactor,
            bool usageAsCollateralEnabled,
            bool borrowingEnabled,
            bool stableBorrowRateEnabled,
            bool isActive,
            bool isFrozen
        );
    }
}
```

### Chainlink Aggregator Interface

```rust
sol! {
    #[sol(rpc)]
    interface IAggregatorV3 {
        function latestRoundData() external view returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );

        function decimals() external view returns (uint8);
    }
}
```

### LeverageExecutor Interface

```rust
sol! {
    #[sol(rpc)]
    interface ILeverageExecutor {
        function openLeveragePosition(
            address debtAsset,
            uint256 flashAmount,
            address collateralAsset,
            address swapRouter,
            bytes calldata swapCalldata,
            uint256 minCollateralOut
        ) external;

        function closeLeveragePosition(
            address debtAsset,
            uint256 debtAmount,
            address collateralAsset,
            uint256 collateralToWithdraw,
            address swapRouter,
            bytes calldata swapCalldata,
            uint256 minDebtTokenOut
        ) external;

        function deleveragePosition(
            address debtAsset,
            uint256 repayAmount,
            address collateralAsset,
            uint256 collateralToWithdraw,
            address swapRouter,
            bytes calldata swapCalldata,
            uint256 minDebtTokenOut
        ) external;

        function setRouterApproval(address router, bool approved) external;
        function rescueTokens(address token, uint256 amount) external;
    }
}
```

### ERC-20 Interface

```rust
sol! {
    #[sol(rpc)]
    interface IERC20 {
        function balanceOf(address account) external view returns (uint256);
        function approve(address spender, uint256 amount) external returns (bool);
        function allowance(address owner, address spender) external view returns (uint256);
        function decimals() external view returns (uint8);
    }
}
```

---

## Type System Design

Rust's type system provides guarantees that Python's optional typing cannot. The following types encode domain invariants at the type level.

### Newtypes for On-Chain Fixed-Point

```rust
use alloy_primitives::U256;
use std::ops::{Add, Sub, Mul, Div};

/// WAD-scaled value (18 decimals). Used for health factors, prices, amounts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Wad(pub U256);

impl Wad {
    pub const ONE: Wad = Wad(U256::from_limbs([1_000_000_000_000_000_000, 0, 0, 0]));
    pub const ZERO: Wad = Wad(U256::ZERO);

    /// Convert to rust_decimal for off-chain display/computation
    pub fn to_decimal(&self) -> rust_decimal::Decimal {
        // self.0 / 1e18
        let raw = self.0.to_string();
        rust_decimal::Decimal::from_str_exact(&raw)
            .unwrap_or_default() / rust_decimal_macros::dec!(1_000_000_000_000_000_000)
    }
}

/// RAY-scaled value (27 decimals). Used for Aave interest rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ray(pub U256);
```

This prevents accidentally mixing WAD and RAY values — a common bug in DeFi that the Python implementation guards against only via comments.

### Signal Types (Layer 2 of Signal Architecture)

```rust
use rust_decimal::Decimal;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionDirection {
    Long,   // Collateral=volatile, Debt=stable
    Short,  // Collateral=stable, Debt=volatile
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    Trending,       // Hurst > 0.55
    MeanReverting,  // Hurst < 0.45
    Ranging,        // 0.45 <= Hurst <= 0.55, ATR ratio < 1.0
    Volatile,       // ATR ratio > 3.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalComponent {
    pub source: String,
    pub tier: u8,
    pub direction: PositionDirection,
    pub strength: Decimal,      // -1.0 to 1.0
    pub weight: Decimal,
    pub confidence: Decimal,    // 0.0 to 1.0
    pub data_age_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeSignal {
    pub direction: PositionDirection,
    pub confidence: Decimal,
    pub strategy_mode: String,
    pub regime: MarketRegime,
    pub components: Vec<SignalComponent>,
    pub recommended_size_usd: Decimal,
    pub hurst_exponent: Decimal,
    pub garch_volatility: Decimal,
    pub timestamp: i64,
}
```

### Health Monitoring Types

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HFTier {
    Safe,       // HF > 2.0 -> poll every 15s
    Watch,      // 1.5-2.0  -> poll every 5s
    Warning,    // 1.3-1.5  -> poll every 2s
    Critical,   // < 1.3    -> poll every 1s
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub health_factor: Decimal,
    pub tier: HFTier,
    pub collateral_usd: Decimal,
    pub debt_usd: Decimal,
    pub borrow_rate_apr: Decimal,
    pub oracle_fresh: bool,
    pub predicted_hf_10m: Decimal,
    pub timestamp: i64,
}
```

### Unified Event Type for Channel Communication

```rust
/// Events consumed by the Strategy task from the shared channel.
/// Replaces Python's untyped asyncio.Queue.
#[derive(Debug, Clone)]
pub enum SignalEvent {
    Health(HealthStatus),
    Trade(TradeSignal),
    Shutdown,
}
```

### Error Types

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BotError {
    // Execution errors
    #[error("Transaction simulation failed: {reason}")]
    SimulationFailed { reason: String },

    #[error("Transaction reverted: {reason} (tx: {tx_hash})")]
    TxReverted { tx_hash: String, reason: String },

    #[error("Transaction timed out after {timeout_seconds}s (tx: {tx_hash})")]
    TxTimeout { tx_hash: String, timeout_seconds: u64 },

    // Aggregator errors
    #[error("All aggregator providers failed")]
    AggregatorUnavailable,

    #[error("DEX-Oracle price divergence: {divergence_pct:.2}% (max {max_pct:.2}%)")]
    PriceDivergence { divergence_pct: f64, max_pct: f64 },

    // Safety errors
    #[error("Safety gate blocked: {reason}")]
    SafetyBlocked { reason: String },

    #[error("Oracle stale: {age_seconds}s old (max {max_seconds}s)")]
    OracleStale { age_seconds: u64, max_seconds: u64 },

    // Data errors
    #[error("Data source unavailable: {source}")]
    DataUnavailable { source: String },

    // Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    // Forwarded errors
    #[error(transparent)]
    Alloy(#[from] alloy::transports::TransportError),

    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),

    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),

    #[error(transparent)]
    Redis(#[from] redis::RedisError),
}
```

---

## Phase-by-Phase Rebuild Specification

Each phase corresponds to the original Python implementation plan phases but adapted for Rust idioms.

### Phase 0: Project Scaffolding and Foundation

**Objective**: Set up the Cargo workspace, toolchain, CI, and foundational infrastructure.

**Tasks**:

1. **Initialize Cargo workspace**
   - Create workspace `Cargo.toml` with `crates/bot` and `crates/mempool-decoder` members
   - Pin Rust toolchain via `rust-toolchain.toml` (minimum 1.88 for Alloy 1.5)
   - Configure `clippy.toml` with DeFi-specific lints (arithmetic overflow, unwrap usage)

2. **Copy configuration files** from existing Python project
   - All JSON config files (`config/*.json`, `config/chains/56.json`) remain unchanged
   - `.env.example` remains unchanged
   - Foundry contracts (`contracts/`) remain unchanged

3. **Set up SQLx migrations**
   - Create `migrations/001_initial.sql` with the three tables (positions, position_snapshots, transactions) matching the existing SQLite schema
   - Configure `sqlx-cli` for offline query checking

4. **Define error types** (`src/errors.rs`)
   - Typed error enum using `thiserror` (see Error Types section above)

5. **Define constants** (`src/constants.rs`)
   - WAD (1e18), RAY (1e27), USD_DECIMALS (1e8), SECONDS_PER_YEAR
   - All contract addresses from `config/chains/56.json`
   - Flash loan modes (0 = repay, 2 = keep debt)

6. **Define all type modules** (`src/types/*.rs`)
   - All domain types with `Serialize`/`Deserialize` derives
   - Newtype wrappers (Wad, Ray) for on-chain fixed-point values

7. **Configure tracing**
   - Per-module log files matching the existing folder structure (Health_Monitor_Logs/, Strategy_Logs/, etc.)
   - JSON formatter for structured logging
   - `EnvFilter` for runtime log level control

**Deliverables**: Compiling workspace with all types defined, empty module stubs, configuration loading, and logging infrastructure.

**Tests**: Compilation check; config deserialization round-trip test; constant value assertions.

---

### Phase 1: Configuration Layer

**Objective**: Type-safe configuration loading with validation, replacing Python's `config/loader.py`.

**Specification**:

```rust
// src/config/types.rs
#[derive(Debug, Deserialize, Clone)]
pub struct BotConfig {
    pub chain: ChainConfig,
    pub aave: AaveConfig,
    pub aggregator: AggregatorConfig,
    pub signals: SignalConfig,
    pub positions: PositionConfig,
    pub timing: TimingConfig,
    pub rate_limits: RateLimitConfig,
    pub app: AppConfig,
    pub mempool: Option<MempoolConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChainConfig {
    pub chain_id: u64,
    pub chain_name: String,
    pub block_time_seconds: f64,
    pub rpc: RpcConfig,
    pub contracts: ContractsConfig,
    pub chainlink_feeds: HashMap<String, ChainlinkFeedConfig>,
    pub tokens: HashMap<String, TokenConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SignalConfig {
    pub enabled: bool,
    pub mode: String,  // "momentum", "mean_reversion", "blended", "manual"
    pub data_source: DataSourceConfig,
    pub indicators: IndicatorParams,
    pub signal_sources: SignalSourcesConfig,
    pub entry_rules: EntryRulesConfig,
    pub position_sizing: PositionSizingConfig,
    pub alpha_decay_monitoring: AlphaDecayConfig,
    pub exit_rules: ExitRulesConfig,
}

// ... (all other config structs matching the JSON schema)
```

**Loading pattern** (using `config` crate):

```rust
pub fn load_config() -> Result<BotConfig, BotError> {
    dotenvy::dotenv().ok();

    let chain: ChainConfig = load_json("config/chains/56.json")?;
    let aave: AaveConfig = load_json("config/aave.json")?;
    let signals: SignalConfig = load_json("config/signals.json")?;
    // ... etc.

    // Environment variable overrides
    let config = config::Config::builder()
        .set_override_option("positions.dry_run",
            std::env::var("EXECUTOR_DRY_RUN").ok().map(|v| v == "true"))?
        .set_override_option("positions.max_flash_loan_usd",
            std::env::var("MAX_FLASH_LOAN_USD").ok())?
        // ... etc.
        .build()?;

    validate_config(&bot_config)?;
    Ok(bot_config)
}
```

**Validation** (`src/config/validate.rs`):
- All contract addresses are valid checksummed addresses
- `max_leverage_ratio` > 1.0 and <= 5.0
- `min_health_factor` >= 1.1
- Signal source weights sum to approximately 1.0 per tier
- Required environment variables present when `dry_run = false`

**Tests**:
- Valid config loads successfully
- Missing required fields cause typed errors
- Env var overrides work correctly
- Invalid values (negative leverage, HF < 1.0) rejected
- Partial config with defaults filled in

**Academic justification**: The `config` crate implements the 12-Factor App methodology (Wiggins, 2012), specifically Factor III (Store config in the environment). The `serde` derive macro ensures that config struct fields and JSON keys are statically verified at compile time — a typo in a config key becomes a compile error, not a runtime surprise.

---

### Phase 2: Aave V3 Client

**Objective**: Implement the Aave V3 read + calldata encoding layer using Alloy's type-safe contract bindings.

**Key difference from Python**: The Python `aave_client.py` uses runtime JSON ABI parsing via web3.py. The Rust version uses the `sol!` macro (see Contract Interface Definitions above) for compile-time ABI generation — encoding errors become compile errors.

**Specification**:

```rust
pub struct AaveClient {
    pool: IPool::IPoolInstance<Http<Client>, RootProvider<Http<Client>>>,
    data_provider: IPoolDataProvider::IPoolDataProviderInstance<...>,
    oracle: IAggregatorV3::IAggregatorV3Instance<...>,
    config: AaveConfig,
}

impl AaveClient {
    /// Construct with Alloy provider
    pub fn new(provider: RootProvider<Http<Client>>, config: &AaveConfig, chain: &ChainConfig) -> Self;

    // --- Async Read Operations ---
    pub async fn get_user_account_data(&self, user: Address) -> Result<UserAccountData>;
    pub async fn get_reserve_data(&self, asset: Address) -> Result<ReserveData>;
    pub async fn get_asset_price(&self, feed_address: Address) -> Result<Decimal>;
    pub async fn get_flash_loan_premium(&self) -> Result<Decimal>;

    // --- Calldata Encoding (sync, no I/O) ---
    pub fn encode_flash_loan(&self, params: FlashLoanParams) -> Bytes;
    pub fn encode_supply(&self, asset: Address, amount: U256, on_behalf_of: Address) -> Bytes;
    pub fn encode_withdraw(&self, asset: Address, amount: U256, to: Address) -> Bytes;
    pub fn encode_repay(&self, asset: Address, amount: U256, rate_mode: U256, on_behalf_of: Address) -> Bytes;
}
```

**Health factor conversion** (on-chain WAD to off-chain Decimal):

```rust
pub fn wad_to_decimal(wad: U256) -> Decimal {
    let raw_str = wad.to_string();
    Decimal::from_str(&raw_str).unwrap_or_default()
        / dec!(1_000_000_000_000_000_000)
}
```

**Oracle freshness validation** (ported from Python, per Deng et al., ICSE 2024, "Safeguarding DeFi Smart Contracts against Oracle Deviations"):

```rust
pub async fn check_oracle_freshness(
    &self,
    feed: &IAggregatorV3::IAggregatorV3Instance<...>,
    max_staleness_seconds: u64,
) -> Result<bool> {
    let data = feed.latestRoundData().call().await?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let age = now - data.updatedAt.to::<u64>();

    if age > max_staleness_seconds {
        tracing::error!(age, max_staleness_seconds, "Oracle data stale");
        return Ok(false);
    }
    if data.answeredInRound < data.roundId {
        tracing::warn!(
            answered_in_round = data.answeredInRound.to::<u64>(),
            round_id = data.roundId.to::<u64>(),
            "Oracle round incomplete"
        );
    }
    Ok(true)
}
```

**Tests** (using `mockall` for provider trait mocking):
- `get_user_account_data` correctly converts WAD health factor
- `get_reserve_data` extracts isolation mode flag
- `encode_flash_loan` produces correct calldata (verified against known-good encoding)
- Oracle freshness rejects stale data (age > max)
- RAY-to-Decimal conversion accuracy for interest rates

---

### Phase 3: Safety and Health Monitor

**Objective**: Implement the safety gate system and tiered health factor monitoring.

#### Safety Module

```rust
pub struct SafetyState {
    dry_run: bool,
    max_position_usd: Decimal,
    max_leverage_ratio: Decimal,
    min_health_factor: Decimal,
    max_gas_price_gwei: u64,
    global_pause: AtomicBool,
    last_action_time: Mutex<Instant>,
    cooldown: Duration,
    daily_tx_count: AtomicU32,
    max_tx_per_24h: u32,
}

impl SafetyState {
    pub fn can_open_position(&self, amount_usd: Decimal, leverage: Decimal) -> Result<(), BotError>;
    pub fn can_submit_tx(&self, gas_price_gwei: u64) -> Result<(), BotError>;
    pub fn trigger_global_pause(&self);
    pub fn check_pause_sentinel(&self) -> bool;
    pub fn record_action(&self);
}
```

**Default-to-safe**: `SafetyState::default()` creates a state with `dry_run=true`, `max_position_usd=0`. On any internal error, the safety module returns `Err(SafetyBlocked)` rather than allowing the action.

#### Health Monitor

```rust
pub struct HealthMonitor {
    aave_client: Arc<AaveClient>,
    safety: Arc<SafetyState>,
    config: TimingConfig,
    event_tx: mpsc::Sender<SignalEvent>,
    shutdown: CancellationToken,
    consecutive_failures: AtomicU32,
}

impl HealthMonitor {
    pub async fn run(&self) -> Result<()> {
        loop {
            tokio::select! {
                _ = self.shutdown.cancelled() => break,
                _ = tokio::time::sleep(self.current_interval()) => {
                    match self.poll_once().await {
                        Ok(status) => {
                            self.consecutive_failures.store(0, Ordering::Relaxed);
                            let _ = self.event_tx.send(SignalEvent::Health(status)).await;
                        }
                        Err(e) => {
                            let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
                            if failures >= 5 {
                                tracing::error!("5 consecutive RPC failures, pausing");
                                self.safety.trigger_global_pause();
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
```

**Compound interest HF prediction** (matches Aave V3's `MathUtils.calculateCompoundedInterest()` 3-term Taylor series):

```rust
pub fn predict_hf_at(&self, seconds_ahead: u64, position: &PositionState) -> Decimal {
    let rate_per_second = position.borrow_rate_ray.to_decimal()
        / dec!(1_000_000_000_000_000_000_000_000_000)  // RAY
        / dec!(31_536_000);  // SECONDS_PER_YEAR
    let dt = Decimal::from(seconds_ahead);

    // Aave's 3-term Taylor approximation
    let compound = dec!(1)
        + rate_per_second * dt
        + (rate_per_second * dt).powi(2) / dec!(2);

    let projected_debt = position.debt_usd * compound;
    (position.collateral_usd * position.liquidation_threshold) / projected_debt
}
```

**Tests**:
- Tier transitions at correct HF thresholds
- Poll interval changes with tier
- Oracle freshness validation rejects stale data and triggers global pause
- Compound interest prediction matches Aave MathUtils output for known inputs
- 5 consecutive failures trigger global pause
- Graceful shutdown via CancellationToken
- Safety gates: default-to-deny on error, cooldown enforcement, gas price cap

**Property-based tests** (proptest):
```rust
proptest! {
    #[test]
    fn health_factor_always_positive(
        collateral_usd in 100u64..1_000_000u64,
        debt_usd in 1u64..500_000u64,
        lt_bps in 5000u32..9500u32,
    ) {
        let hf = compute_health_factor(
            Decimal::from(collateral_usd),
            Decimal::from(debt_usd),
            Decimal::from(lt_bps) / dec!(10000),
        );
        prop_assert!(hf > Decimal::ZERO);
    }
}
```

---

### Phase 4: DEX Aggregator Client

**Objective**: Implement parallel fan-out to 1inch, OpenOcean, and ParaSwap with best-quote selection, rate limiting, and DEX-Oracle divergence checking.

**Specification**:

```rust
pub struct AggregatorClient {
    http_client: ClientWithMiddleware,
    providers: Vec<AggregatorProvider>,
    approved_routers: HashSet<Address>,
    config: AggregatorConfig,
}

impl AggregatorClient {
    /// Query all enabled providers in parallel, return best quote.
    /// Academic basis: Angeris et al. (2022) proved CFMM routing is convex;
    /// aggregators implement practical solvers. Diamandis et al. (FC 2023)
    /// showed efficient routing scales linearly.
    pub async fn get_best_quote(
        &self,
        from_token: Address,
        to_token: Address,
        amount: U256,
        max_slippage_bps: u32,
    ) -> Result<SwapQuote> {
        let futures = self.providers.iter()
            .filter(|p| p.enabled)
            .map(|p| self.query_provider(p, from_token, to_token, amount, max_slippage_bps));

        let results = futures::future::join_all(futures).await;
        let valid: Vec<SwapQuote> = results.into_iter().filter_map(|r| r.ok()).collect();

        if valid.is_empty() {
            return Err(BotError::AggregatorUnavailable);
        }

        let best = valid.into_iter().max_by_key(|q| q.to_amount).unwrap();
        Ok(best)
    }

    /// Reject swap if DEX price diverges >1% from Chainlink oracle.
    /// Deng et al. (ICSE 2024) found ad-hoc oracle controls insufficient.
    pub fn check_dex_oracle_divergence(
        &self,
        quote: &SwapQuote,
        chainlink_price: Decimal,
        max_divergence_pct: Decimal,
    ) -> Result<()> {
        let implied_dex_price = Decimal::from_str(&quote.from_amount.to_string())?
            / Decimal::from_str(&quote.to_amount.to_string())?;
        let divergence = ((implied_dex_price - chainlink_price).abs() / chainlink_price)
            * dec!(100);

        if divergence > max_divergence_pct {
            return Err(BotError::PriceDivergence {
                divergence_pct: divergence.to_f64().unwrap_or(0.0),
                max_pct: max_divergence_pct.to_f64().unwrap_or(0.0),
            });
        }
        Ok(())
    }
}
```

**Rate limiting per provider** (using `tower::limit::RateLimit` or `reqwest-ratelimit`):

```rust
// Each provider gets its own rate-limited HTTP client
fn build_provider_client(rate_limit_rps: u32) -> ClientWithMiddleware {
    let retry_policy = ExponentialBackoff::builder()
        .build_with_max_retries(3);

    ClientBuilder::new(reqwest::Client::new())
        .with(RetryTransientMiddleware::new_with_policy(retry_policy))
        // Rate limit enforced per provider
        .build()
}
```

**Provider-specific implementations** (1inch API v6, OpenOcean API v4, ParaSwap/Velora API v5):

```rust
async fn query_1inch(&self, params: &SwapParams) -> Result<SwapQuote> {
    // 1inch Classic API v6 — NOT Fusion (intent-based, incompatible with flash loans)
    // Official docs: portal.1inch.dev/documentation
    let url = format!("{}/swap", self.config.base_url);
    let resp = self.http_client.get(&url)
        .query(&[
            ("src", params.from_token.to_string()),
            ("dst", params.to_token.to_string()),
            ("amount", params.amount.to_string()),
            ("from", params.executor_address.to_string()),
            ("slippage", (params.max_slippage_bps as f64 / 100.0).to_string()),
            ("disableEstimate", "true".to_string()),
        ])
        .header("Authorization", format!("Bearer {}", self.api_key))
        .send().await?;
    // Parse and validate router address against whitelist
    // ...
}
```

**Tests** (using `wiremock`):
- Parallel fan-out returns best quote (highest output amount)
- Individual provider failure doesn't crash (graceful degradation)
- All providers failing returns `AggregatorUnavailable`
- DEX-Oracle divergence > 1% rejected
- Rate limiting enforced (requests delayed when exceeding RPS)
- Router address validated against whitelist
- Quote caching (10s TTL)

**References**: 1inch Classic API v6 documentation (portal.1inch.dev); OpenOcean API v4 documentation (apis.openocean.finance); ParaSwap/Velora API v5 documentation (developers.velora.xyz); Angeris et al. (2022), ACM EC, arXiv:2204.05238; Diamandis et al. (2023), FC, arXiv:2302.04938.

---

### Phase 5: Transaction Submitter

**Objective**: Implement transaction signing, simulation, MEV-protected submission, and nonce management using Alloy.

**Specification**:

```rust
pub struct TxSubmitter {
    provider: Arc<RootProvider<Http<Client>>>,
    mev_provider: Arc<RootProvider<Http<Client>>>,  // 48 Club Privacy RPC
    signer: PrivateKeySigner,
    nonce: Mutex<Option<u64>>,
    config: TransactionConfig,
}

impl TxSubmitter {
    /// Simulate transaction via eth_call (15s timeout)
    pub async fn simulate(&self, tx: &TransactionRequest) -> Result<Bytes> {
        tokio::time::timeout(
            Duration::from_secs(self.config.simulation_timeout_seconds),
            self.provider.call(tx).await
        ).await.map_err(|_| BotError::SimulationFailed {
            reason: "Simulation timed out".into()
        })?
    }

    /// Sign and submit via MEV-protected RPC (48 Club Privacy RPC: rpc.48.club)
    /// 48 Club docs: docs.48.club/privacy-rpc
    pub async fn submit_and_wait(&self, tx: TransactionRequest) -> Result<TransactionReceipt> {
        let nonce = self.get_next_nonce().await?;
        let tx = tx.nonce(nonce);

        let pending = self.mev_provider
            .send_transaction(tx)
            .await?;

        let receipt = tokio::time::timeout(
            Duration::from_secs(self.config.confirmation_timeout_seconds),
            pending.get_receipt()
        ).await.map_err(|_| BotError::TxTimeout {
            tx_hash: pending.tx_hash().to_string(),
            timeout_seconds: self.config.confirmation_timeout_seconds,
        })??;

        if !receipt.status() {
            return Err(BotError::TxReverted {
                tx_hash: receipt.transaction_hash.to_string(),
                reason: Self::decode_revert_reason(&receipt),
            });
        }

        Ok(receipt)
    }

    /// Thread-safe nonce management with async locking
    async fn get_next_nonce(&self) -> Result<u64> {
        let mut nonce_guard = self.nonce.lock().await;
        let nonce = match *nonce_guard {
            Some(n) => n,
            None => {
                let n = self.provider
                    .get_transaction_count(self.signer.address())
                    .pending()
                    .await?;
                n
            }
        };
        *nonce_guard = Some(nonce + 1);
        Ok(nonce)
    }

    /// Replace stuck transaction with 12.5% gas bump at same nonce
    pub async fn replace_stuck_tx(&self, nonce: u64) -> Result<B256>;

    /// Decode revert reason from receipt
    pub fn decode_revert_reason(receipt: &TransactionReceipt) -> String;
}
```

**Tests**:
- Nonce management: sequential nonces under concurrent access
- Simulation failure returns typed error
- MEV-protected submission routes to correct provider
- Stuck tx replacement uses same nonce + 12.5% gas bump
- Revert reason decoding for known error selectors
- Timeout handling for both simulation and confirmation

---

### Phase 6: Smart Contract (Foundry — Unchanged)

The Solidity smart contract and Foundry tests remain identical to the Python implementation plan. The `LeverageExecutor.sol` contract is chain-agnostic — it does not care whether the caller is Python or Rust.

**Tasks**:
1. `contracts/foundry.toml` — install OpenZeppelin
2. `contracts/src/interfaces/IFlashLoanReceiver.sol` and `IAaveV3Pool.sol`
3. `contracts/src/LeverageExecutor.sol` — direction-agnostic flash loan receiver
4. `contracts/test/LeverageExecutor.t.sol` — Foundry fork tests for both LONG and SHORT flows
5. Generate ABI JSON from compiled contract
6. Define Rust `sol!` bindings from the generated ABI

**Reference**: Aave V3 flash loan docs (aave.com/docs/aave-v3/guides/flash-loans); Cyfrin Updraft Aave V3 course.

---

### Phase 7: Position Manager, Strategy, and P&L Tracker

#### Position Manager

**Specification**:

```rust
pub struct PositionManager {
    aave_client: Arc<AaveClient>,
    aggregator_client: Arc<AggregatorClient>,
    tx_submitter: Arc<TxSubmitter>,
    pnl_tracker: Arc<PnLTracker>,
    safety: Arc<SafetyState>,
    config: PositionConfig,
}

impl PositionManager {
    /// Open position — direction-agnostic (same contract handles long + short)
    /// LONG: open_position(Long, USDT, 5000, WBNB) -> flash loan USDT, swap to WBNB, supply WBNB
    /// SHORT: open_position(Short, WBNB, 10, USDC) -> flash loan WBNB, swap to USDC, supply USDC
    pub async fn open_position(
        &self,
        direction: PositionDirection,
        debt_token: Address,
        amount: U256,
        collateral_token: Address,
    ) -> Result<PositionState>;

    /// Close position — flash loan mode=0 (must repay in same tx)
    pub async fn close_position(&self) -> Result<RealizedPnL>;

    /// Partial deleverage to target health factor
    pub async fn partial_deleverage(&self, target_hf: Decimal) -> Result<PositionState>;

    /// Isolation mode check for short positions (USDT may be restricted)
    /// Prefer USDC for short collateral: higher LTV (77%), higher LT (80%), no isolation restrictions
    async fn check_isolation_mode(&self, collateral_token: Address) -> Result<()>;
}
```

**Open position flow** (ported from Python):
1. If SHORT: check isolation mode for collateral token
2. `aggregator_client.get_best_quote(debt_token -> collateral_token)` — parallel fan-out
3. `aggregator_client.check_dex_oracle_divergence(quote, chainlink_price)` — reject if >1%
4. Validate: `to_amount_min >= expected * (1 - slippage)`
5. `aave_client.encode_flash_loan(...)` — mode=2 (keep debt)
6. `safety.can_submit_tx(gas_price)?`
7. `tx_submitter.simulate(tx)?`
8. If `!dry_run`: `tx_submitter.submit_and_wait(tx)?`
9. `aave_client.get_user_account_data()` — verify post-execution HF
10. `pnl_tracker.record_open(position, tx_hash, gas_cost)`

#### Strategy (Risk Engine — Layer 5)

```rust
pub struct Strategy {
    position_manager: Arc<PositionManager>,
    aave_client: Arc<AaveClient>,
    pnl_tracker: Arc<PnLTracker>,
    safety: Arc<SafetyState>,
    event_rx: mpsc::Receiver<SignalEvent>,
    config: StrategyConfig,
    shutdown: CancellationToken,
}

impl Strategy {
    pub async fn run(&mut self) -> Result<()> {
        loop {
            tokio::select! {
                _ = self.shutdown.cancelled() => break,
                event = self.event_rx.recv() => {
                    match event {
                        Some(SignalEvent::Health(status)) => self.handle_health(status).await?,
                        Some(SignalEvent::Trade(signal)) => self.handle_trade_signal(signal).await?,
                        Some(SignalEvent::Shutdown) | None => break,
                    }
                }
            }
        }
        Ok(())
    }
}
```

**Direction-aware stress test** (analytical, no AMM simulation — Perez et al., FC 2021; OECD 2023):

```rust
/// For LONG: HF = (collateral * (1 + drop) * LT) / debt
/// For SHORT: HF = (collateral * LT) / (debt * (1 + drop))
/// The short formula is the inverse — HF is convex in price.
pub fn stress_test(
    &self,
    direction: PositionDirection,
    collateral_usd: Decimal,
    debt_usd: Decimal,
    lt: Decimal,
    price_drops: &[Decimal],
) -> Vec<Decimal> {
    price_drops.iter().map(|drop| {
        match direction {
            PositionDirection::Long =>
                (collateral_usd * (dec!(1) + drop) * lt) / debt_usd,
            PositionDirection::Short =>
                (collateral_usd * lt) / (debt_usd * (dec!(1) + drop)),
        }
    }).collect()
}
```

**Liquidation cascade multiplier** (Perez et al., FC 2021: "3% price variations make >$10M liquidatable"; OECD 2023: "liquidations boost price volatility during stress"):

```rust
pub fn stress_test_with_cascade(
    &self,
    direction: PositionDirection,
    collateral_usd: Decimal,
    debt_usd: Decimal,
    lt: Decimal,
    price_drops: &[Decimal],
    market_total_supply_usd: Decimal,
) -> Vec<Decimal> {
    price_drops.iter().map(|drop| {
        let estimated_liquidatable = self.estimate_market_liquidations(*drop, market_total_supply_usd);
        let effective_drop = if estimated_liquidatable > dec!(50_000_000) {
            *drop + dec!(-0.03)  // 3% additional cascade
        } else {
            *drop
        };
        self.compute_hf_at_drop(direction, collateral_usd, debt_usd, lt, effective_drop)
    }).collect()
}
```

**Fractional Kelly position sizing** (MacLean et al., 2010, Quantitative Finance):

```rust
pub fn compute_kelly_fraction(&self, edge: Decimal, volatility: Decimal) -> Decimal {
    if edge <= dec!(0) {
        return dec!(0);
    }
    let variance = volatility.powi(2);
    let full_kelly = if variance > dec!(0) { edge / variance } else { dec!(0) };
    // 25% fractional Kelly — standard risk reduction
    let fractional = full_kelly * dec!(0.25);
    fractional.min(self.config.max_leverage_ratio)
}
```

**Close factor risk check** (Aave V3 `LiquidationLogic.sol` — positions below $2,000 face 100% close factor):

```rust
pub fn check_close_factor_risk(
    &self,
    collateral_usd: Decimal,
    debt_usd: Decimal,
    price_drops: &[Decimal],
) -> bool {
    for drop in price_drops {
        let projected_collateral = collateral_usd * (dec!(1) + drop);
        if projected_collateral < dec!(2000) || debt_usd < dec!(2000) {
            tracing::warn!(
                drop = %drop,
                projected_collateral = %projected_collateral,
                "Position hits 100% close factor threshold"
            );
            return false;
        }
    }
    true
}
```

**Alpha decay monitoring** (Cong et al., 2024, Annual Review of Financial Economics: crypto strategy alpha decays with ~12-month half-life):

```rust
pub fn check_strategy_health(&self) -> StrategyHealthReport {
    let recent = self.pnl_tracker.get_rolling_stats(30);
    let historical = self.pnl_tracker.get_rolling_stats(180);

    let mut report = StrategyHealthReport::default();

    if let (Some(r), Some(h)) = (recent, historical) {
        if h.win_rate > dec!(0) {
            let accuracy_ratio = r.win_rate / h.win_rate;
            if accuracy_ratio < dec!(0.7) {
                report.alpha_decay_detected = true;
                report.accuracy_ratio = accuracy_ratio;
            }
        }
        if h.sharpe_ratio > dec!(0) {
            let sharpe_ratio = r.sharpe_ratio / h.sharpe_ratio;
            if sharpe_ratio < dec!(0.5) {
                report.alpha_decay_detected = true;
            }
        }
    }
    report
}
```

#### P&L Tracker

```rust
pub struct PnLTracker {
    pool: sqlx::SqlitePool,
}

impl PnLTracker {
    pub async fn new(db_path: &str) -> Result<Self> {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)  // SQLite is single-writer
            .connect(&format!("sqlite:{db_path}?mode=rwc"))
            .await?;
        sqlx::migrate!("./migrations").run(&pool).await?;
        Ok(Self { pool })
    }

    // Compile-time checked queries via sqlx::query!
    pub async fn record_open(&self, position: &PositionState, tx_hash: &str, gas_cost: Decimal) -> Result<i64> {
        let id = sqlx::query!(
            r#"INSERT INTO positions (direction, open_timestamp, debt_token, collateral_token,
               initial_debt_amount, initial_collateral_amount, open_tx_hash, open_borrow_rate_apr, total_gas_costs_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
            position.direction.as_str(),
            position.open_timestamp,
            position.debt_token,
            position.collateral_token,
            position.initial_debt_amount.to_string(),
            position.initial_collateral_amount.to_string(),
            tx_hash,
            position.borrow_rate_apr.to_string(),
            gas_cost.to_string(),
        )
        .execute(&self.pool)
        .await?
        .last_insert_rowid();
        Ok(id)
    }

    pub async fn get_rolling_stats(&self, window_days: u32) -> Option<TradingStats>;
    pub async fn record_close(&self, position_id: i64, tx_hash: &str, gas_cost: Decimal, tokens_received: Decimal) -> Result<RealizedPnL>;
    pub async fn snapshot(&self, position_id: i64, status: &HealthStatus) -> Result<()>;
}
```

**Tests**:
- Position open/close lifecycle with both LONG and SHORT
- Direction-aware stress test: long HF linear, short HF convex
- Cascade multiplier produces different results than without
- Close factor risk check rejects positions near $2,000 threshold
- Borrow rate cost check rejects when APR too high
- Fractional Kelly sizing: no edge -> zero size; high volatility -> reduced size
- Alpha decay detection: accuracy ratio < 0.7 raises threshold
- Isolation mode rejection for USDT shorts
- P&L tracker: record_open, record_close, unrealized P&L for long vs short
- Deleverage amount formula produces valid HF improvement
- SQLx migration creates correct schema
- Property-based: stress_test HF monotonically decreases with larger drops

---

### Phase 7.5: Signal Engine, Data Service, and Indicators

This is the largest phase — implementing the 5-layer signal architecture.

#### Indicators Module (Pure Computation, No I/O)

```rust
pub struct Indicators;

impl Indicators {
    // --- Standard Technical Indicators ---
    // (use `ta` crate for streaming computation)

    pub fn ema(prices: &[Decimal], period: usize) -> Vec<Decimal>;
    pub fn rsi(prices: &[Decimal], period: usize) -> Decimal;  // Wilder's smoothing
    pub fn macd(prices: &[Decimal], fast: usize, slow: usize, signal: usize) -> (Decimal, Decimal, Decimal);
    pub fn bollinger_bands(prices: &[Decimal], period: usize, std_mult: Decimal) -> (Decimal, Decimal, Decimal);
    pub fn atr(highs: &[Decimal], lows: &[Decimal], closes: &[Decimal], period: usize) -> Decimal;

    // --- Regime & Statistical Indicators ---

    /// R/S Hurst exponent. H > 0.55 -> trending; H < 0.45 -> mean-reverting.
    /// Maraj-Mervar & Aybar (FracTime 2025): regime-adaptive Sharpe 2.10 vs 0.85 static.
    pub fn hurst_exponent(prices: &[Decimal], max_lag: usize) -> Decimal {
        // Implement R/S analysis using ndarray for regression
        // No production crate exists; manual implementation ~100 lines
    }

    /// GARCH(1,1) one-step-ahead volatility forecast.
    /// sigma^2_{t+1} = omega + alpha * epsilon^2_t + beta * sigma^2_t
    /// Hansen & Lunde (2005): GARCH(1,1) difficult to beat.
    /// Bollerslev (1986): original GARCH specification.
    pub fn garch_volatility(
        returns: &[Decimal],
        omega: Decimal,   // 0.00001
        alpha: Decimal,   // 0.1
        beta: Decimal,    // 0.85
    ) -> Decimal {
        // Forward recursion: ~30 lines
        // No production crate exists; manual implementation required
    }

    /// VPIN — Volume-Synchronized Probability of Informed Trading
    /// Easley et al. (2012, Review of Financial Studies): VPIN framework.
    /// Abad & Yague (2025, ScienceDirect): VPIN predicts crypto price jumps.
    pub fn vpin(trades: &[Trade], bucket_size: Decimal, window: usize) -> Decimal;

    /// OBI — Order Book Imbalance
    /// Kolm et al. (2023, J. Financial Economics): 73% of prediction performance.
    pub fn order_book_imbalance(
        bids: &[(Decimal, Decimal)],
        asks: &[(Decimal, Decimal)],
    ) -> Decimal {
        let bid_vol: Decimal = bids.iter().map(|(_, qty)| qty).sum();
        let ask_vol: Decimal = asks.iter().map(|(_, qty)| qty).sum();
        let total = bid_vol + ask_vol;
        if total == dec!(0) { return dec!(0); }
        (bid_vol - ask_vol) / total  // [-1, 1]
    }

    /// Realized volatility from log returns.
    pub fn realized_volatility(closes: &[Decimal], window: usize) -> Decimal;

    /// Compute all indicators from OHLCV data.
    pub fn compute_all(candles: &[OHLCV], config: &IndicatorParams) -> IndicatorSnapshot;
}
```

**GARCH(1,1) implementation** (no production Rust crate exists; ~30 lines of forward recursion):

```rust
pub fn garch_volatility(
    returns: &[Decimal],
    omega: Decimal,
    alpha: Decimal,
    beta: Decimal,
) -> Decimal {
    if returns.is_empty() { return dec!(0); }

    // Initialize with realized variance
    let mean_return: Decimal = returns.iter().sum::<Decimal>() / Decimal::from(returns.len());
    let initial_variance: Decimal = returns.iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<Decimal>() / Decimal::from(returns.len());

    let mut sigma_sq = initial_variance;

    // Forward recursion
    for r in returns {
        let epsilon_sq = r.powi(2);
        sigma_sq = omega + alpha * epsilon_sq + beta * sigma_sq;
    }

    // One-step-ahead forecast
    sigma_sq.sqrt().unwrap_or(dec!(0))
}
```

**Hurst exponent R/S analysis** (no production crate; ~80 lines):

```rust
pub fn hurst_exponent(prices: &[Decimal], max_lag: usize) -> Decimal {
    if prices.len() < 100 { return dec!(0.5); } // Insufficient data

    // 1. Compute log returns
    let log_returns: Vec<f64> = prices.windows(2)
        .map(|w| (w[1].to_f64().unwrap() / w[0].to_f64().unwrap()).ln())
        .collect();

    // 2. For each window size n from 2 to max_lag:
    let mut log_rs = Vec::new();
    let mut log_n = Vec::new();

    for n in (2..=max_lag.min(log_returns.len() / 4)).step_by(1) {
        let mut rs_values = Vec::new();
        for chunk in log_returns.chunks(n) {
            if chunk.len() < n { continue; }
            let mean = chunk.iter().sum::<f64>() / n as f64;
            let deviations: Vec<f64> = chunk.iter().map(|r| r - mean).collect();
            let cumdev: Vec<f64> = deviations.iter()
                .scan(0.0, |acc, &d| { *acc += d; Some(*acc) })
                .collect();
            let range = cumdev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                      - cumdev.iter().cloned().fold(f64::INFINITY, f64::min);
            let std_dev = (chunk.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
            if std_dev > 0.0 {
                rs_values.push(range / std_dev);
            }
        }
        if !rs_values.is_empty() {
            let avg_rs = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
            log_rs.push(avg_rs.ln());
            log_n.push((n as f64).ln());
        }
    }

    // 3. Linear regression: H = slope of log(R/S) vs log(n)
    if log_rs.len() < 3 { return dec!(0.5); }
    let h = linear_regression_slope(&log_n, &log_rs);
    Decimal::from_f64(h).unwrap_or(dec!(0.5))
}
```

#### Data Service (Multi-Source Market Data)

```rust
pub struct DataService {
    http_client: ClientWithMiddleware,
    aave_client: Arc<AaveClient>,
    cache: Mutex<DataCache>,
    config: DataSourceConfig,
}

struct DataCache {
    ohlcv: HashMap<String, (Instant, Vec<OHLCV>)>,
    order_book: HashMap<String, (Instant, OrderBookSnapshot)>,
    trades: HashMap<String, (Instant, Vec<Trade>)>,
    funding_rate: HashMap<String, (Instant, Decimal)>,
    liquidation_levels: HashMap<String, (Instant, Vec<LiquidationLevel>)>,
    exchange_flows: HashMap<String, (Instant, ExchangeFlows)>,
}

impl DataService {
    /// Binance Spot API: GET /api/v3/klines
    pub async fn get_ohlcv(&self, symbol: &str, interval: &str, limit: usize) -> Result<Vec<OHLCV>>;

    /// Binance Spot API: GET /api/v3/depth (for OBI)
    pub async fn get_order_book(&self, symbol: &str, limit: usize) -> Result<OrderBookSnapshot>;

    /// Binance Spot API: GET /api/v3/aggTrades (for VPIN)
    pub async fn get_recent_trades(&self, symbol: &str, limit: usize) -> Result<Vec<Trade>>;

    /// Binance Futures API: GET /fapi/v1/fundingRate
    pub async fn get_funding_rate(&self, symbol: &str) -> Result<Option<Decimal>>;

    /// Aave V3 subgraph: position HF distribution for liquidation heatmap
    pub async fn get_liquidation_levels(&self, asset: &str) -> Result<Vec<LiquidationLevel>>;

    /// Exchange flow proxy (BSC hot wallet monitoring / DefiLlama)
    pub async fn get_exchange_flows(&self, token: &str, window_minutes: u32) -> Result<ExchangeFlows>;

    /// Redis-backed mempool signal (from Rust decoder)
    pub async fn get_mempool_signal(&self, symbol: &str) -> Result<Option<MempoolSignal>>;

    /// Current price from Binance ticker
    pub async fn get_current_price(&self, symbol: &str) -> Result<Decimal>;
}
```

**Cache TTLs** (matching Python implementation):

| Data Type | Cache TTL | Source |
|-----------|----------|--------|
| OHLCV (1h) | 60s | Binance Spot /klines |
| OHLCV (15m) | 30s | Binance Spot /klines |
| Order book | 5s | Binance Spot /depth |
| Recent trades | 10s | Binance Spot /aggTrades |
| Funding rate | 300s | Binance Futures /fundingRate |
| Liquidation levels | 120s | Aave V3 subgraph |
| Exchange flows | 120s | BSC hot wallet / DefiLlama |

#### Signal Engine (5-Layer Architecture)

```rust
pub struct SignalEngine {
    data_service: Arc<DataService>,
    pnl_tracker: Arc<PnLTracker>,
    event_tx: mpsc::Sender<SignalEvent>,
    config: SignalConfig,
    shutdown: CancellationToken,
}

impl SignalEngine {
    pub async fn run(&self) -> Result<()> {
        loop {
            tokio::select! {
                _ = self.shutdown.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(
                    self.config.data_source.refresh_interval_seconds
                )) => {
                    match self.evaluate_once().await {
                        Ok(Some(signal)) => {
                            let _ = self.event_tx.send(SignalEvent::Trade(signal)).await;
                        }
                        Ok(None) => {} // Confidence below threshold
                        Err(e) => tracing::warn!("Signal evaluation failed: {e}"),
                    }
                }
            }
        }
        Ok(())
    }

    async fn evaluate_once(&self) -> Result<Option<TradeSignal>> {
        // Layer 1: Regime Detection
        let candles = self.data_service.get_ohlcv("BNBUSDT", "1h", 200).await?;
        let indicators = Indicators::compute_all(&candles, &self.config.indicators);
        let regime = self.detect_regime(&indicators);

        // Layer 2: Multi-Source Signal Collection (parallel)
        let components = self.collect_all_signals(&indicators, regime).await;

        // Layer 3: Ensemble Confidence Scoring
        let (direction, confidence) = self.compute_ensemble_confidence(&components, regime);
        if confidence < self.get_effective_confidence_threshold() {
            return Ok(None);
        }

        // Layer 4: Position Sizing (Fractional Kelly)
        let volatility = Indicators::garch_volatility(
            &self.get_recent_returns(&candles),
            self.config.indicators.garch_omega,
            self.config.indicators.garch_alpha,
            self.config.indicators.garch_beta,
        );
        let edge = self.get_rolling_edge();
        let kelly_f = self.compute_kelly_fraction(edge, volatility);
        let account_equity = self.get_account_equity().await?;
        let size_usd = (account_equity * kelly_f).min(self.config.position_sizing.max_position_usd());

        // Layer 5 check: alpha decay
        let alpha_decayed = self.check_alpha_decay();

        Ok(Some(TradeSignal {
            direction,
            confidence,
            strategy_mode: self.config.mode.clone(),
            regime,
            components,
            recommended_size_usd: size_usd,
            hurst_exponent: indicators.hurst,
            garch_volatility: volatility,
            timestamp: chrono::Utc::now().timestamp(),
        }))
    }

    /// Collect all signal sources in parallel using tokio::join!
    async fn collect_all_signals(
        &self,
        indicators: &IndicatorSnapshot,
        regime: MarketRegime,
    ) -> Vec<SignalComponent> {
        // Tier 1 (parallel)
        let (tech, obi, vpin_result) = tokio::join!(
            self.compute_technical_signals(indicators),
            self.compute_order_book_imbalance(),
            self.compute_vpin(),
        );

        // Tier 2 (parallel)
        let (btc_spill, liq_heat, exch_flows, funding) = tokio::join!(
            self.compute_btc_volatility_spillover(),
            self.compute_liquidation_heatmap(),
            self.compute_exchange_flows(),
            self.compute_funding_rate_signal(),
        );

        // Tier 3 (if enabled)
        let mempool = if self.config.signal_sources.tier_2_or_3_mempool_enabled() {
            self.compute_mempool_flow_signal().await
        } else {
            None
        };

        let mut components = vec![tech, obi, vpin_result, btc_spill, liq_heat, exch_flows, funding];
        components.extend(mempool);
        components.into_iter().filter_map(|r| r.ok()).collect()
    }

    /// Regime-weighted ensemble scoring
    /// Lo (2004, J. Portfolio Management): Adaptive Market Hypothesis
    /// Timmermann & Granger (2004): regime-switching models
    fn compute_ensemble_confidence(
        &self,
        components: &[SignalComponent],
        regime: MarketRegime,
    ) -> (PositionDirection, Decimal) {
        let mut bull_score = dec!(0);
        let mut bear_score = dec!(0);
        let mut total_weight = dec!(0);

        for c in components {
            if c.data_age_seconds > self.config.entry_rules.max_signal_age_seconds {
                continue;
            }
            let regime_mult = self.regime_weight_multiplier(&c.source, regime);
            let weighted = c.strength.abs() * c.weight * c.confidence * regime_mult;
            match c.direction {
                PositionDirection::Long => bull_score += weighted,
                PositionDirection::Short => bear_score += weighted,
            }
            total_weight += c.weight * regime_mult;
        }

        if total_weight == dec!(0) {
            return (PositionDirection::Long, dec!(0));
        }

        let net_score = (bull_score - bear_score) / total_weight;
        let direction = if net_score >= dec!(0) {
            PositionDirection::Long
        } else {
            PositionDirection::Short
        };

        // Agreement bonus: >70% consensus -> +15%
        let active: Vec<_> = components.iter()
            .filter(|c| c.strength.abs() > dec!(0.1))
            .collect();
        let mut confidence = net_score.abs();
        if !active.is_empty() {
            let majority_frac = active.iter()
                .filter(|c| c.direction == direction)
                .count() as f64 / active.len() as f64;
            if majority_frac > 0.7 {
                confidence *= dec!(1.15);
            }
        }

        (direction, confidence.min(dec!(1)))
    }
}
```

**Tests**:
- 5-layer pipeline end-to-end with mock data
- Hurst exponent: known persistent series yields H > 0.55; random walk yields ~0.5
- GARCH convergence and stationarity (alpha + beta < 1)
- VPIN volume bucketing matches Easley et al. (2012) algorithm
- OBI symmetry: equal bid/ask -> 0; all bids -> 1
- Ensemble confidence: agreement bonus triggers at 70% consensus
- Regime-adaptive weights: trending boosts momentum 1.2x
- Kelly sizing: zero edge -> zero size; capped at max leverage
- Alpha decay detection raises confidence threshold
- Mempool signal integration from mock Redis
- Each signal component individually tested

---

### Phase 8: Main Entrypoint and Integration

**Specification** (`src/main.rs`):

```rust
use tokio::signal;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .init();

    // Load configuration
    let config = load_config()?;
    tracing::info!(
        dry_run = config.positions.dry_run,
        chain = config.chain.chain_name,
        mode = config.signals.mode,
        "Starting BSC Leverage Bot"
    );

    // Initialize providers
    let provider = ProviderBuilder::new()
        .with_recommended_fillers()
        .on_http(config.chain.rpc.http_url.parse()?);
    let mev_provider = ProviderBuilder::new()
        .on_http(config.chain.rpc.mev_protected_url.parse()?);

    // Initialize components (dependency injection order)
    let safety = Arc::new(SafetyState::from_config(&config.positions));
    let aave_client = Arc::new(AaveClient::new(provider.clone(), &config.aave, &config.chain));
    let aggregator_client = Arc::new(AggregatorClient::new(&config.aggregator)?);
    let tx_submitter = Arc::new(TxSubmitter::new(provider.clone(), mev_provider, &config)?);
    let pnl_tracker = Arc::new(PnLTracker::new("data/positions.db").await?);
    let position_manager = Arc::new(PositionManager::new(
        aave_client.clone(), aggregator_client.clone(),
        tx_submitter.clone(), pnl_tracker.clone(),
        safety.clone(), &config.positions,
    ));
    let data_service = Arc::new(DataService::new(&config)?);

    // Shared channel and shutdown token
    let (event_tx, event_rx) = tokio::sync::mpsc::channel::<SignalEvent>(64);
    let shutdown = CancellationToken::new();

    let health_monitor = HealthMonitor::new(
        aave_client.clone(), safety.clone(), &config.timing,
        event_tx.clone(), shutdown.child_token(),
    );
    let signal_engine = SignalEngine::new(
        data_service.clone(), pnl_tracker.clone(),
        event_tx.clone(), &config.signals, shutdown.child_token(),
    );
    let mut strategy = Strategy::new(
        position_manager.clone(), aave_client.clone(),
        pnl_tracker.clone(), safety.clone(),
        event_rx, &config, shutdown.child_token(),
    );

    // Launch 3 concurrent tasks
    let health_handle = tokio::spawn(async move { health_monitor.run().await });
    let signal_handle = tokio::spawn(async move { signal_engine.run().await });
    let strategy_handle = tokio::spawn(async move { strategy.run().await });

    // Wait for shutdown signal
    signal::ctrl_c().await?;
    tracing::info!("Shutdown signal received, stopping gracefully...");
    shutdown.cancel();

    // Wait for all tasks to complete
    let _ = tokio::join!(health_handle, signal_handle, strategy_handle);
    tracing::info!("Shutdown complete");

    Ok(())
}
```

**Integration tests** (using Anvil BSC fork):

```rust
#[tokio::test]
async fn test_long_position_lifecycle_on_bsc_fork() {
    let anvil = Anvil::new()
        .fork("https://bsc-dataseed1.binance.org")
        .spawn();
    let provider = ProviderBuilder::new()
        .on_http(anvil.endpoint().parse().unwrap());

    // Deploy LeverageExecutor, approve tokens, open long, check HF, close
    // ...
}

#[tokio::test]
async fn test_short_position_lifecycle_on_bsc_fork() {
    // Same as above but with short direction
    // ...
}
```

**Tests**:
- All three tasks start and stop cleanly via CancellationToken
- Startup banner logs correct configuration
- Graceful shutdown on SIGINT/SIGTERM
- Anvil fork integration: full LONG lifecycle (open -> monitor -> close)
- Anvil fork integration: full SHORT lifecycle (open -> monitor -> close)

---

### Phase 9: Hardening

**Objective**: Production-readiness verification.

**Tasks**:

1. **Static analysis**
   - `cargo clippy -- -D warnings` with custom lint configuration
   - `cargo audit` for known vulnerabilities in dependencies
   - `cargo deny check` for license compliance and duplicate dependencies

2. **Type safety verification**
   - All `unwrap()` calls audited — replace with `?` or `expect()` with context
   - No `unsafe` blocks (unless explicitly justified)
   - All numeric conversions use checked arithmetic

3. **Structured logging audit**
   - Every module uses `tracing` with appropriate levels
   - Error paths include full context via `anyhow::Context`
   - Sensitive data (private keys, full calldata) never logged

4. **End-to-end dry-run testing**
   - Run against BSC mainnet with `dry_run=true`
   - Verify signal pipeline generates correct signals
   - Verify health monitoring polls at correct intervals
   - Verify aggregator quotes are fetched and validated
   - Verify transaction simulation succeeds (eth_call)

5. **Performance profiling**
   - Measure signal engine cycle time (target: <5s per evaluation)
   - Measure indicator computation time (target: <100ms for 200 candles)
   - Measure memory usage under sustained operation

6. **Error recovery verification**
   - RPC failures trigger retry with backoff
   - Aggregator failures fall through gracefully
   - SQLite write failures don't crash the bot
   - Nonce management recovers from stuck transactions

---

### Phase 10: Mempool Order Flow Enhancement (Rust — Already Native)

In the Python plan, the mempool decoder was the only Rust component. In the full Rust rebuild, it becomes a second binary in the same Cargo workspace.

#### Phase 10A: Mempool Decoder (`crates/mempool-decoder/`)

Specification unchanged from `Mempool_Enhancement_Plan.md`. Key crates:

```toml
[dependencies]
tokio = { version = "1.49", features = ["full"] }
tokio-tungstenite = { version = "0.28", features = ["native-tls"] }
alloy-primitives = "1.5"
alloy-sol-types = "1.5"
redis = { version = "1.0", features = ["aio", "tokio-comp"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
lru = "0.12"
tracing = "0.1"
tracing-subscriber = "0.3"
```

**Responsibilities**:
- WebSocket connection to BSC node (`newPendingTransactions`)
- Router matching (12 routers) and selector matching (26 selectors)
- V2, V3, SmartRouter, Universal Router, aggregator ABI decoding
- Buy/sell classification (stable->volatile = BUY, volatile->stable = SELL)
- LRU deduplication (100k capacity)
- Poison detection (sandwich scoring)
- Redis publishing (`mempool:decoded_swaps`)

#### Phase 10B: Rolling Aggregator

In-process module within the mempool decoder binary.

**Per token pair (WBNB, BTCB, ETH)**:
- Sliding windows: 1m, 5m, 15m
- Metrics: buy/sell volume USD, net flow, direction score, tx counts, whale detection, volume acceleration
- Publishes to `mempool:aggregate_signal` every 5 seconds

#### Phase 10C: Bot Integration

The main bot's `DataService` consumes mempool signals from Redis:

```rust
pub async fn get_mempool_signal(&self, symbol: &str) -> Result<Option<MempoolSignal>> {
    let mut conn = self.redis_client.get_async_connection().await?;
    let raw: Option<String> = conn.get("mempool:aggregate_signal").await?;
    match raw {
        Some(json) => {
            let signal: MempoolSignal = serde_json::from_str(&json)?;
            let age = chrono::Utc::now().timestamp() - signal.timestamp;
            if age > 30 { return Ok(None); } // Stale
            Ok(Some(signal))
        }
        None => Ok(None),
    }
}
```

Signal promoted from Tier 3 to Tier 2 with weight 0.12 (as specified in `Mempool_Enhancement_Plan.md`).

#### Phase 10D: End-to-End Validation

1. Run mempool decoder against BSC mainnet WebSocket
2. Verify decoded output matches ArbitrageTestBot for same transactions
3. Run bot in dry-run mode consuming live mempool signal
4. Compare signal engine output with and without mempool signal

---

## Testing Strategy

### Unit Tests (per module, fast, no network)

| Module | Test Focus | Framework |
|--------|-----------|-----------|
| `config` | Deserialization, validation, env overrides | `#[test]` + serde |
| `aave_client` | WAD/RAY conversion, calldata encoding, oracle freshness | `mockall` for provider |
| `aggregator_client` | Parallel fan-out, best-quote selection, divergence check | `wiremock` for HTTP mocking |
| `tx_submitter` | Nonce management, simulation, revert decoding | `mockall` for provider |
| `health_monitor` | Tier transitions, compound interest prediction | `mockall` + `tokio::time::pause` |
| `safety` | Kill switches, cooldown, pause sentinel | `#[test]` |
| `strategy` | Stress test (long vs short), cascade, Kelly, alpha decay | `#[test]` + `proptest` |
| `signal_engine` | 5-layer pipeline, ensemble confidence, regime weights | `wiremock` + `mockall` |
| `indicators` | Hurst, GARCH, VPIN, OBI, EMA, RSI, MACD, BB | `#[test]` + `proptest` |
| `data_service` | Binance API parsing, cache TTL, graceful degradation | `wiremock` |
| `pnl_tracker` | SQLite lifecycle, P&L computation, rolling stats | `sqlx` test database |
| `position_manager` | Open/close/deleverage flows, isolation mode | `mockall` |

### Property-Based Tests (proptest)

| Property | Module |
|----------|--------|
| Health factor always positive for valid inputs | `indicators`, `strategy` |
| Stress test HF monotonically decreases with larger drops | `strategy` |
| Short stress test produces steeper HF decline than long for same drop | `strategy` |
| Kelly fraction always >= 0 and <= max_leverage | `signal_engine` |
| GARCH variance always positive with valid parameters | `indicators` |
| OBI in range [-1, 1] for any bid/ask input | `indicators` |

### Integration Tests (Anvil BSC fork)

| Test | Coverage |
|------|----------|
| Long position lifecycle | open -> monitor HF -> deleverage -> close |
| Short position lifecycle | open -> monitor HF -> deleverage -> close |
| Aggregator live quotes | Query real DEX aggregators, validate against Chainlink |
| Signal pipeline end-to-end | Fetch real Binance data, compute all indicators, generate signal |

### Smart Contract Tests (Foundry — unchanged)

| Test | Coverage |
|------|----------|
| `LeverageExecutor.t.sol` | Long flow (USDT->WBNB), short flow (WBNB->USDT), access control, router whitelist, slippage |

---

## Migration Strategy

The rebuild should proceed incrementally to maintain a working system at each step:

1. **Phase 0-1**: Scaffolding + config. Bot doesn't run yet.
2. **Phase 2-5**: Core infrastructure (Aave, aggregator, tx, safety). Each module independently testable.
3. **Phase 6**: Smart contract (unchanged — already Solidity).
4. **Phase 7**: Position manager + strategy + P&L. Bot can open/close positions.
5. **Phase 7.5**: Signal engine + indicators. Bot can generate entry signals.
6. **Phase 8**: Main entrypoint. Bot is fully functional.
7. **Phase 9**: Hardening. Bot is production-ready.
8. **Phase 10**: Mempool integration. Full feature parity with Python + mempool.

The Python bot can continue running in production during the rebuild. Switch over only after Phase 9 passes end-to-end dry-run testing against BSC mainnet.

---

## References

### Rust Language and Safety

1. Jung, R., Jourdan, J.-H., Krebbers, R., & Dreyer, D. — "Safe Systems Programming in Rust" (ACM Communications, 2021) — Establishes Rust as the first industry language combining safety and low-level control
2. Jung, R. et al. — "RustBelt: Securing the Foundations of the Rust Programming Language" (POPL 2018) — First machine-checked safety proof for Rust's type system
3. Jung, R. — "Stacked Borrows: An Aliasing Model for Rust" (MPI-SWS) — Formal aliasing model underlying the borrow checker
4. White House ONCD (2024) — "Back to the Building Blocks: A Path Toward Secure and Measurable Software" — Recommends memory-safe languages for critical infrastructure
5. Berger, E. & Zorn, B. (2006) — "DieHard: Probabilistic Memory Safety for Unsafe Languages" (PLDI) — ~70% of critical vulnerabilities are memory safety bugs

### DeFi and Trading Systems

6. Angeris, G. et al. — "Optimal Routing for Constant Function Market Makers" (ACM EC 2022, arXiv:2204.05238) — CFMM routing is convex; aggregators implement practical solvers
7. Diamandis, T. et al. — "An Efficient Algorithm for Optimal Routing Through Constant Function Market Makers" (FC 2023, arXiv:2302.04938) — Efficient routing scales linearly
8. Heimbach, L. & Huang, X. — "DeFi Leverage" (BIS Working Paper No. 1171, 2024) — Comprehensive analysis of leveraged DeFi positions
9. Perez, D. et al. — "Liquidations: DeFi on a Knife-Edge" (FC 2021) — "3% price variations make >$10M liquidatable"
10. OECD — "DeFi Risks: Interconnectedness, Leverage, and Opacity" (2023) — "Liquidations boost price volatility during stress"
11. Deng, J. et al. — "Safeguarding DeFi Smart Contracts against Oracle Deviations" (ICSE 2024) — "Existing ad-hoc control mechanisms are often insufficient"

### Signal Architecture

12. Kolm, P., Turiel, J., & Westray, N. — "Deep Order Flow Imbalance" (J. Financial Economics, 2023) — OBI accounts for 73% of prediction
13. Easley, D., Lopez de Prado, M., & O'Hara, M. — "Flow Toxicity and Liquidity" (Review of Financial Studies, 2012) — VPIN framework
14. Abad, D. & Yague, J. — "VPIN Predicts Crypto Price Jumps" (ScienceDirect, 2025) — Crypto-specific VPIN validation
15. Maraj-Mervar, B. & Aybar, S. — Regime-adaptive strategies via Hurst exponent (FracTime, 2025) — Sharpe 2.10 vs 0.85 static
16. MacLean, L. et al. — "Good and Bad Properties of the Kelly Criterion" (Quantitative Finance, 2010) — Fractional Kelly for controlled growth
17. Cong, L. et al. — Crypto trading strategy alpha decay (Annual Review of Financial Economics, 2024) — ~12-month half-life
18. Hansen, P. & Lunde, A. — "A Forecast Comparison of Volatility Models" (J. Applied Econometrics, 2005) — GARCH(1,1) difficult to beat
19. Bollerslev, T. — "Generalized Autoregressive Conditional Heteroscedasticity" (J. Econometrics, 1986) — Original GARCH specification
20. Hudson, R. & Urquhart, A. — "Technical Trading and Cryptocurrencies" (Annals of Operations Research, 2019) — ~15,000 rules show significant predictability
21. Aloosh, A. & Bekaert, G. — "Funding Rate Arbitrage" (SSRN, 2022) — Funding rates explain 12.5% of price variation
22. Chi, J., White, R., & Lee, W. — "Cryptocurrency Exchange Flows and Returns" (SSRN, 2024) — USDT inflows predict returns
23. Ante, L. & Saggu, A. — "Mempool Transaction Flow and Volume Prediction" (J. Innovation & Knowledge, 2024)
24. Shen, D., Urquhart, A., & Wang, P. — "Does Twitter Predict Bitcoin?" (Economics Letters, 2019) — Volume > polarity
25. Cont, R., Kukanov, A., & Stoikov, S. — "The Price Impact of Order Book Events" (J. Financial Econometrics, 2014)
26. Lo, A. — "The Adaptive Markets Hypothesis" (J. Portfolio Management, 2004)

### Rust Ecosystem

27. Paradigm — "Introducing Alloy v1.0" (May 2025) — 60% faster U256, 10x faster ABI encoding
28. Ryhl, A. — "Actors with Tokio" (ryhl.io, 2024) — Actor pattern for async Rust trading systems
29. Biriukov, N. — "Async Rust with Tokio I/O Streams: Backpressure" (biriukov.dev) — Bounded channels for flow control
30. Palmieri, L. — "Zero to Production in Rust" — Error handling patterns with thiserror + anyhow

### BSC Infrastructure

31. BEP-322 — "Builder API Specification for BNB Smart Chain" (github.com/bnb-chain/BEPs) — PBS specification
32. 48 Club — "Privacy RPC Documentation" (docs.48.club/privacy-rpc) — MEV-protected transaction submission
33. BNB Chain — "MEV Demystified" (2024) — BSC MEV landscape overview

### Aave V3

34. Aave — "Flash Loans Guide" (aave.com/docs/aave-v3/guides/flash-loans) — Flash loan mechanism
35. Aave — "Interest Rate Strategy" (aave.com/docs/aave-v3/smart-contracts/interest-rate-strategy) — Kinked rate model
36. Chaos Labs — "Aave V3 Risk Parameter Methodology" (chaoslabs.xyz) — Formal methodology for LTV/LT parameters
37. Aave V3 — `GenericLogic.sol` (github.com/aave/aave-v3-core) — Authoritative health factor computation
