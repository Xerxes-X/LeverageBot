# BSC Leverage Bot — Revised Implementation Plan

## Architecture Decision Summary

The original plan was **over-engineered** — it copied 60-70% of the arbitrage bot's architecture (9 pool managers, custom AMM math, split optimizer, event streaming, bundle submission, 30+ Redis channels, 15-20 multiprocesses) that is fundamentally unnecessary for a leverage bot. This revised plan is purpose-built for position management.

| Dimension | Arbitrage Bot | Leverage Bot (This Plan) |
|---|---|---|
| Core operation | Detect price discrepancies across venues | Open/close/adjust a lending position (long or short) |
| Latency sensitivity | Sub-millisecond | Seconds to minutes |
| State tracking | Hundreds of pools across 9 protocols | Own Aave V3 position (1 address) |
| Swap routing | Custom split optimizer | DEX aggregator fan-out (1inch, OpenOcean, ParaSwap — best quote wins) |
| Event streaming | Mandatory (real-time graph updates) | Polling sufficient (WebSocket not required for aggregator quotes — all are REST-only) |
| MEV posture | Perpetrator | Victim (use MEV-protected RPC — 48 Club Privacy RPC) |
| Execution frequency | Continuous | Rare (open/adjust/close) |
| Position direction | N/A | Long AND Short (same contract, parameterized token roles) |
| Entry signals | Continuous scanning | 5-layer signal architecture: Regime Detection → Multi-Source Directional Signals → Ensemble Confidence → Position Sizing → Risk Management |
| Signal sources | Price feeds only | Tier 1: Technical indicators (EMA, RSI, MACD, BB), order book imbalance, VPIN. Tier 2: BTC volatility spillover, liquidation heatmaps, exchange flows, funding rates, **mempool order flow** (Rust decoder via Redis). Tier 3: Social sentiment (disabled) |
| Position sizing | Fixed | Fractional Kelly Criterion with GARCH(1,1) volatility estimate |
| Architecture | 15-20 multiprocesses, Redis pub/sub | Single asyncio process, in-memory queues + Redis for mempool decoder IPC (Phase 10) |

**Academic justification**: Angeris et al., "Optimal Routing for CFMMs" (ACM EC 2022, arXiv:2204.05238) proved CFMM routing is a convex optimization problem. DEX aggregators implement practical solvers for this — 1inch Pathfinder explores 400+ BSC liquidity sources. Building custom routing is strictly worse. Confirmed by Diamandis et al. (FC 2023, arXiv:2302.04938) who showed the efficient routing algorithm scales linearly in the number of pools.

**Mempool monitoring decision (revised)**: The original plan rejected mempool monitoring due to BSC's PBS adoption reducing public mempool visibility. However, live testing via the ArbitrageTestBot's mempool listener (2,428+ swaps decoded from a single session via `newPendingTransactions` WebSocket subscription) confirmed that a meaningful sample of pending swap transactions **is** visible on BSC — primarily retail flow from standard RPC users. While this is a biased sample (private RPC users and MEV-protected wallets are invisible), it is sufficient for **aggregate order flow analysis** — observing the directional bias of pending swaps as a supplementary signal, not racing individual transactions. Mempool order flow has been promoted from Tier 3 (stub) to **Tier 2** with weight 0.12, implemented via a standalone Rust decoder + rolling aggregator publishing to Redis. The Rust decoder handles the latency-critical path (WebSocket + ABI decoding at 1-2ms per tx), while the Python signal engine consumes pre-aggregated results at its existing 60-second refresh rate. Academic basis: Kolm et al. (2023) found OBI accounts for 73% of short-term prediction — pending mempool swaps provide an advance look at order flow before it hits the book; Ante & Saggu (2024) confirmed mempool predicts volume, and aggregate volume bias is directional; Chi et al. (2024) showed exchange flows predict returns. See `Mempool_Enhancement_Plan.md` for full specification. The bot's "Victim" MEV posture remains — transaction submission still uses MEV-protected RPC (48 Club Privacy RPC).

**Signal architecture decision**: Phase 3 research established that no single signal source reliably predicts crypto price movements. Kolm et al. (2023) found order book imbalance accounts for 73% of prediction performance. VPIN significantly predicts price jumps (Abad & Yagüe, ScienceDirect 2025). BTC is the net volatility transmitter to BNB (DCC-GARCH spillover literature). Liquidation heatmaps from Aave V3 health factor distributions identify price levels where cascading liquidations create self-reinforcing moves. Exchange flows — particularly USDT net inflows — positively predict BTC/ETH returns (Chi et al. 2024). The bot integrates these into a 5-layer architecture: Regime Detection (Hurst exponent) → Multi-Source Directional Signals (ensemble of Tier 1/2/3 sources) → Confidence Scoring → Position Sizing (Fractional Kelly with GARCH volatility) → Risk Management. Alpha decay monitoring ensures strategies are rotated as effectiveness degrades (~12-month half-life per Cong et al. 2024).

---

## What the Bot Actually Does

### Signal-Driven Entry (5-Layer Architecture)

0. **Signal engine** continuously evaluates a multi-source signal pipeline organized into 5 layers:

   **Layer 1 — Regime Detection**: Hurst exponent classifies the current market as trending (H>0.55), mean-reverting (H<0.45), or random-walk (0.45≤H≤0.55). Only signals aligned with the detected regime pass through. Regime-adaptive strategies achieve Sharpe 2.10 vs 0.85 for static strategies (Maraj-Mervar & Aybar, FracTime 2025).

   **Layer 2 — Multi-Source Directional Signals** (weighted ensemble):
   - **Tier 1** (highest reliability): Technical indicators (EMA, RSI, MACD, BB), order book imbalance from Binance (73% of prediction performance — Kolm et al. 2023), VPIN from confirmed on-chain trades (predicts price jumps — Abad & Yagüe 2025)
   - **Tier 2** (supplementary): BTC→BNB volatility spillover (BTC is net transmitter), Aave V3 liquidation heatmaps (identify cascade price levels), exchange flows (USDT inflows → bullish — Chi et al. 2024), Binance perp funding rates (12.5% price variation over 7 days — Aloosh & Bekaert 2022), **mempool order flow** (aggregate directional bias from pending DEX swaps — Rust decoder via Redis, weight 0.12 — see `Mempool_Enhancement_Plan.md`)
   - **Tier 3** (low weight, informational): Social sentiment volume proxy (tweet count > polarity — Shen et al. 2019; disabled — requires external API)

   **Layer 3 — Ensemble Confidence Scoring**: Each signal source contributes a weighted score. Combined confidence must exceed the configurable threshold (default: 0.7). Based on MDPI (2025): confidence-threshold filtering achieves 82.68% accuracy at 12% market coverage.

   **Layer 4 — Position Sizing**: Fractional Kelly Criterion (f* = edge/odds, scaled to 25% Kelly) with GARCH(1,1) volatility estimate. Prevents oversizing during low-volatility regimes and undersizing during confirmed trends. MacLean et al. (2010) proved fractional Kelly maximizes long-run growth while controlling drawdown.

   **Layer 5 — Risk Management**: Strategy applies borrow rate cost check, direction-aware stress test, cascade modeling, close factor risk, and safety kill switches before triggering position entry.

   When a signal passes all 5 layers, a `TradeSignal` is emitted to the strategy layer specifying direction (LONG or SHORT), confidence, contributing signals, and recommended position size.

### Long Position Flow (borrow stable, buy volatile — profit from price increase)

1. Call `Pool.flashLoan()` with **mode=2** (variable debt) for **stablecoin** (USDT/USDC)
2. In `executeOperation()`: swap debt token (USDT) → collateral token (WBNB) via DEX aggregator calldata
3. Supply collateral (WBNB) to Aave V3 (flash loan becomes variable-rate USDT debt — no repayment in same tx)
4. Monitor health factor via tiered polling, adjust or unwind when needed
5. Close position: flash loan **mode=0** to repay USDT debt, withdraw WBNB collateral, swap WBNB → USDT, repay flash loan

**Health factor for longs**: `HF = (WBNB_value * LT_WBNB) / USDT_debt` — linear in collateral price. A 20% price drop reduces HF by ~20%.

### Short Position Flow (borrow volatile, sell for stable — profit from price decrease)

1. Call `Pool.flashLoan()` with **mode=2** (variable debt) for **volatile asset** (WBNB)
2. In `executeOperation()`: swap debt token (WBNB) → collateral token (USDT/USDC) via DEX aggregator calldata
3. Supply collateral (USDT/USDC) to Aave V3 (flash loan becomes variable-rate WBNB debt — no repayment in same tx)
4. Monitor health factor via tiered polling, adjust or unwind when needed
5. Close position: flash loan **mode=0** to repay WBNB debt, withdraw USDT collateral, swap USDT → WBNB, repay flash loan

**Health factor for shorts**: `HF = (USDT_value * LT_USDT) / (WBNB_debt * price)` — **inversely proportional** to debt asset price (convex function). A 20% price increase reduces HF by ~16.7%; a 50% increase reduces HF by ~33.3%. Shorts face accelerating danger during price pumps.

**The same `LeverageExecutor.sol` contract handles both directions** — the `openLeveragePosition(debtAsset, flashAmount, collateralAsset, ...)` function is direction-agnostic. The caller determines direction by which token addresses fill which parameter slots.

### Key Asymmetries Between Long and Short

| Risk Factor | Long Position | Short Position |
|------------|---------------|----------------|
| Cascading liquidation risk | **HIGH** — selling volatile collateral into falling market creates feedback loop (OECD 2023) | **LOW** — selling stablecoin does not cascade |
| Self-reinforcing leverage | Decreases as prices rise (self-correcting) | Increases as prices rise (self-reinforcing danger) |
| Tail risk | Bounded at zero | Theoretically unbounded (debt asset can rise indefinitely) |
| Borrow rate | Stablecoins typically 3–9.5% APR | Volatile assets variable, can spike on utilization |
| Collateral LT (WBNB/USDT) | 80% (WBNB as collateral) | 78% (USDT) or 80% (USDC — preferred for shorts) |
| Isolation Mode risk | None for WBNB | **USDT may be isolated** — check on-chain before opening; prefer USDC |

**References**: Heimbach & Huang (2024), "DeFi Leverage," BIS Working Paper No. 1171; Perez et al. (2021), "Liquidations: DeFi on a Knife-Edge," FC 2021

---

## Revised File Structure

```
LeverageBot/
├── main.py                                  (NEW)  -- asyncio entrypoint
├── pyproject.toml                           (NEW)  -- project metadata + deps
│
├── config/
│   ├── __init__.py                          (KEEP)
│   ├── loader.py                            (KEEP, MODIFY) -- remove arb methods, add leverage methods
│   ├── validate.py                          (NEW)  -- config schema validation
│   ├── app.json                             (KEEP, MODIFY) -- update module_folders
│   ├── timing.json                          (KEEP, MODIFY) -- health monitor tiers + tx timeouts
│   ├── rate_limits.json                     (KEEP, MODIFY) -- aggregator API rate limits
│   ├── aave.json                            (NEW)  -- Aave V3 risk params, assets, flash loan premium
│   ├── positions.json                       (NEW)  -- position limits, safety thresholds
│   ├── aggregator.json                      (NEW)  -- 1inch/OpenOcean/ParaSwap endpoints + routers
│   ├── signals.json                         (NEW)  -- signal engine config: indicators, thresholds, data source
│   ├── mempool.json                         (NEW)  -- mempool decoder config: Redis, tokens, aggregation windows
│   ├── chains/
│   │   └── 56.json                          (KEEP, MODIFY) -- remove DEX block, add Aave+Chainlink
│   ├── abis/
│   │   ├── aave_v3_pool.json                (NEW)
│   │   ├── aave_v3_data_provider.json       (NEW)
│   │   ├── aave_v3_oracle.json              (NEW)
│   │   ├── chainlink_aggregator_v3.json     (NEW)
│   │   ├── erc20.json                       (NEW)
│   │   ├── leverage_executor.json           (NEW) -- generated by Foundry
│   │   └── multicall3.json                  (NEW)
│   ├── redis_channels.json                  (DELETE)
│   └── websocket.json                       (DELETE)
│
├── core/
│   ├── __init__.py                          (KEEP)
│   ├── health_monitor.py                    (NEW)  -- tiered HF polling + Chainlink events + oracle freshness
│   ├── strategy.py                          (NEW)  -- entry/exit/sizing logic, stress testing (long + short)
│   ├── signal_engine.py                     (NEW)  -- technical indicator signal generation + confidence scoring
│   ├── indicators.py                        (NEW)  -- EMA, RSI, MACD, Bollinger Bands, ATR computation
│   ├── data_service.py                      (NEW)  -- historical OHLCV from Binance API + fallbacks
│   ├── position_manager.py                  (NEW)  -- open/close/deleverage orchestration (long + short)
│   ├── pnl_tracker.py                       (NEW)  -- P&L tracking, position history, interest accounting
│   └── safety.py                            (NEW)  -- kill switches, dry-run, global pause
│
├── execution/
│   ├── __init__.py                          (KEEP)
│   ├── aggregator_client.py                 (NEW)  -- 1inch/OpenOcean/ParaSwap fan-out + DEX-Oracle divergence check
│   ├── aave_client.py                       (NEW)  -- Aave V3 Web3 read/encode wrapper
│   └── tx_submitter.py                      (NEW)  -- signing, submission, receipt, revert decode, nonce mgmt
│
├── shared/
│   ├── __init__.py                          (KEEP)
│   ├── serialization_utils.py               (KEEP) -- DecimalEncoder, no changes
│   ├── types.py                             (NEW)  -- dataclasses: Position, Quote, AccountData, HFTier, TradeSignal, OHLCV
│   └── constants.py                         (NEW)  -- WAD, RAY, addresses, flash loan modes
│
├── bot_logging/
│   ├── __init__.py                          (KEEP)
│   └── logger_manager.py                    (KEEP, MODIFY) -- update _MODULE_FOLDERS
│
├── contracts/
│   ├── foundry.toml                         (NEW)
│   ├── src/
│   │   ├── LeverageExecutor.sol             (NEW)  -- IFlashLoanReceiver + aggregator swap (both long + short)
│   │   └── interfaces/
│   │       ├── IFlashLoanReceiver.sol       (NEW)
│   │       └── IAaveV3Pool.sol              (NEW)
│   └── test/
│       └── LeverageExecutor.t.sol           (NEW)  -- Foundry fork tests (long + short flows)
│
├── tests/
│   ├── conftest.py                          (NEW)  -- shared pytest fixtures
│   ├── unit/
│   │   ├── test_health_monitor.py           (NEW)
│   │   ├── test_position_manager.py         (NEW)
│   │   ├── test_aggregator_client.py        (NEW)
│   │   ├── test_strategy.py                 (NEW)
│   │   ├── test_safety.py                   (NEW)
│   │   ├── test_aave_client.py              (NEW)
│   │   ├── test_config_loader.py            (NEW)
│   │   ├── test_signal_engine.py            (NEW)  -- 5-layer pipeline, ensemble confidence, regime weights, Kelly sizing, alpha decay
│   │   ├── test_indicators.py              (NEW)  -- indicator math + Hurst, GARCH, VPIN, OBI verification
│   │   ├── test_data_service.py             (NEW)  -- multi-source data fetch/cache tests (klines, depth, aggTrades, flows)
│   │   └── test_pnl_tracker.py              (NEW)  -- P&L calculation, interest accrual
│   └── integration/
│       └── test_aave_fork.py                (NEW)  -- Anvil BSC fork tests (long + short)
│
├── data/
│   └── positions.db                         (RUNTIME) -- SQLite position history (auto-created)
│
├── mempool-decoder/                         (NEW — Phase 10) -- Rust mempool decoder + aggregator
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs                          -- WebSocket loop, entry point
│   │   ├── config.rs                        -- Router addresses, selectors, token lists
│   │   ├── decoder/
│   │   │   ├── mod.rs
│   │   │   ├── v2.rs                        -- V2 swap ABI decoding (9 selectors)
│   │   │   ├── v3.rs                        -- V3 + SmartRouter decoding (10 selectors)
│   │   │   ├── universal.rs                 -- Universal Router command parsing (3 selectors)
│   │   │   └── aggregator.rs               -- 1inch, ParaSwap, etc. (4 selectors)
│   │   ├── classifier.rs                   -- Buy/sell classification, USD estimation
│   │   ├── aggregator.rs                   -- Rolling window statistics (1m, 5m, 15m)
│   │   ├── poison.rs                        -- Sandwich detection scoring
│   │   ├── dedup.rs                         -- LRU tx hash deduplication
│   │   └── redis_publisher.rs              -- Publish to Redis channels
│   └── tests/
│
├── .env.example                             (KEEP, MODIFY)
├── .gitignore                               (KEEP, MODIFY)
│
├── data/                                    (DELETE entire directory — old arb data)
├── gas/                                     (DELETE entire directory)
├── verification/                            (DELETE entire directory)
└── shared/math/                             (DELETE entire directory)
```

**Total: ~60 files. ~5,800-6,900 lines Python + ~2,000-2,500 lines Rust.**

---

## Script-by-Script Specification

### 1. `main.py` — Asyncio Entrypoint (~140-180 lines)

**Purpose**: Single-process asyncio runner. No multiprocessing, no CPU pinning, no forkserver, no Redis.

**Responsibilities**:
- Load `.env` and validate configuration
- Initialize single `AsyncWeb3` provider (shared)
- Initialize shared instances: `SafetyState`, `AaveClient`, `AggregatorClient`, `TxSubmitter`, `PriceDataService`, `PnLTracker`, `PositionManager`, `HealthMonitor`, `SignalEngine`, `Strategy`
- Launch **3 concurrent asyncio tasks**: `health_monitor.run()`, `signal_engine.run(queue)`, and `strategy.run(queue)`
- Both `health_monitor` and `signal_engine` push to the same `signal_queue`; strategy consumes both `HealthStatus` and `TradeSignal` types
- Handle `SIGTERM`/`SIGINT` via `asyncio.Event` for cooperative cancellation
- Log startup banner (dry_run status, addresses, config summary, signal mode, indicator params)

**Error Handling**: Unhandled exceptions in tasks caught by done callback; triggers graceful shutdown.

---

### 2. `core/health_monitor.py` — Health Factor Polling + Oracle Validation (~300-370 lines)

**Purpose**: Continuously poll Aave V3 health factor at tier-appropriate intervals, validate oracle freshness, and predict HF drift using compound interest.

**Key Classes**:
```python
class HFTier(Enum):
    SAFE = "safe"           # HF > 2.0 → poll every 15s
    WATCH = "watch"         # 1.5-2.0  → poll every 5s
    WARNING = "warning"     # 1.3-1.5  → poll every 2s
    CRITICAL = "critical"   # < 1.3    → poll every 1s + Chainlink WS events

class HealthMonitor:
    async def run(self) -> None              # Main loop (runs indefinitely)
    async def _poll_once(self) -> HealthStatus
    def _determine_tier(self, hf) -> HFTier
    def _compute_poll_interval(self, tier) -> float
    async def _subscribe_chainlink_events(self) -> None  # Only when CRITICAL
    def predict_hf_at(self, seconds_ahead) -> Decimal    # Compound interest drift (see below)
    async def check_oracle_freshness(self, feed_address, max_staleness_seconds=60) -> bool
    async def get_borrow_rate(self, asset) -> BorrowRateInfo  # Current rate + utilization
```

**Oracle Freshness Validation** (CRITICAL — prevents stale price decisions):

Every poll cycle must validate that Chainlink data is fresh before trusting the health factor:

```python
async def check_oracle_freshness(self, feed_address: str, max_staleness_seconds: int = 60) -> bool:
    """Returns False and triggers global pause if oracle data is stale."""
    (roundId, answer, startedAt, updatedAt, answeredInRound) = \
        chainlink_feed.functions.latestRoundData().call()
    age = int(time.time()) - updatedAt
    if age > max_staleness_seconds:
        logger.critical(f"Oracle stale: {age}s old (max {max_staleness_seconds}s)")
        self.safety.trigger_global_pause()
        return False
    if answeredInRound < roundId:
        logger.warning(f"Oracle round incomplete: answeredInRound={answeredInRound} < roundId={roundId}")
    return True
```

**Justification**: Deng et al. (ICSE 2024), "Safeguarding DeFi Smart Contracts against Oracle Deviations," found that "existing ad-hoc control mechanisms are often insufficient to protect DeFi protocols against oracle deviation." Explicit staleness checks are mandatory.

**Compound Interest HF Prediction** (replaces linear approximation):

Aave V3 compounds interest per second using a Taylor series in `MathUtils.calculateCompoundedInterest()`. For prediction horizons >10 minutes, linear approximation diverges meaningfully from reality.

```python
def predict_hf_at(self, seconds_ahead: int, position: PositionState) -> Decimal:
    """Predict health factor at t+seconds_ahead using compound interest model."""
    rate_per_second = position.borrow_rate_ray / RAY / SECONDS_PER_YEAR
    # Aave's 3-term Taylor approximation (matches on-chain MathUtils.sol)
    dt = Decimal(seconds_ahead)
    compound = 1 + rate_per_second * dt + (rate_per_second * dt) ** 2 / 2
    projected_debt = position.debt_usd * compound
    # Collateral does not change between oracle updates (supply APY accrual is negligible)
    return (position.collateral_usd * position.liquidation_threshold) / projected_debt
```

**Why polling is sufficient**: Chainlink oracle updates on BSC happen every 27-60s (heartbeat) or on 0.1% deviation. Between updates, HF only drifts from interest accrual (negligible at per-second scale). HF=1.5 buffer means ~33% collateral drop before liquidation. Oracle lag gives 10-30s reaction window.

**Chainlink `AnswerUpdated` subscription**: Subscribe to events on the relevant price feed only when tier=CRITICAL. This provides oracle price change detection without mempool monitoring.

**Output**: `HealthStatus` objects pushed to `asyncio.Queue` for Strategy consumption.

---

### 3. `core/strategy.py` — Entry/Exit Logic + Risk Engine + Position Sizing (~500-580 lines)

**Purpose**: Decision engine — evaluates trade signals, manages position lifecycle, applies risk filters including direction-aware stress testing, borrow rate cost analysis, liquidation cascade modeling, GARCH-informed position sizing via fractional Kelly criterion, and alpha decay monitoring.

**Key Functions**:
```python
class Strategy:
    async def run(self, signal_queue) -> None           # Consume HealthStatus AND TradeSignal
    async def evaluate_entry(self, signal: TradeSignal) -> Tuple[bool, str]
    async def evaluate_open(self, direction, debt_token, collateral_token, amount) -> bool
    def stress_test(self, direction, collateral_usd, debt_usd, liq_threshold_bps, price_drops) -> List[Decimal]
    def stress_test_with_cascade(self, direction, collateral_usd, debt_usd, liq_threshold_bps, price_drops, market_total_supply_usd) -> List[Decimal]
    def compute_deleverage_amount(self, current_hf, target_hf, collateral_usd, debt_usd, liq_threshold_bps) -> Decimal
    async def check_borrow_rate_acceptable(self, asset, projected_hold_hours) -> Tuple[bool, Decimal]
    async def check_close_factor_risk(self, collateral_usd, debt_usd) -> bool
    def validate_position_size(self, signal: TradeSignal) -> Decimal  # Apply Kelly + risk limits
    def compute_garch_volatility(self) -> Decimal                      # GARCH(1,1) for sizing
    def check_strategy_health(self) -> StrategyHealthReport             # Alpha decay + performance
    async def handle_signal(self, status: HealthStatus) -> None
    async def handle_trade_signal(self, signal: TradeSignal) -> None
```

**Direction-Aware Stress Test Formulas** (analytical, no AMM simulation):

For **LONG** positions (volatile collateral, stable debt):
```
HF_long = (collateral_usd * (1 + price_change) * liquidation_threshold) / debt_usd
```

For **SHORT** positions (stable collateral, volatile debt):
```
HF_short = (collateral_usd * liquidation_threshold) / (debt_usd * (1 + price_change))
```

The short formula is the **inverse** — HF is a convex function of price. Small price increases cause moderate HF drops; large increases cause accelerating drops. This asymmetry must be modeled separately.

**Liquidation Cascade Multiplier** (accounts for systemic feedback loops):

Static stress tests assume a price drop happens once and stops. In reality, liquidations create sell pressure that causes further price drops (Perez et al., FC 2021: "3% price variations can make >$10M liquidatable"; OECD 2023: "liquidations boost price volatility during stress").

```python
def stress_test_with_cascade(self, direction, collateral_usd, debt_usd,
                              liq_threshold_bps, price_drops, market_total_supply_usd) -> List[Decimal]:
    """Apply cascade multiplier: if initial drop triggers >$50M in market-wide
    liquidations, assume additional 2-5% cascade-induced decline."""
    results = []
    for drop in price_drops:
        base_hf = self._compute_hf_at_drop(direction, collateral_usd, debt_usd, liq_threshold_bps, drop)
        # Estimate market-wide liquidatable value at this price drop
        estimated_liquidatable = self._estimate_market_liquidations(drop, market_total_supply_usd)
        if estimated_liquidatable > Decimal("50_000_000"):
            cascade_additional = Decimal("-0.03")  # 3% additional from cascade selling
            base_hf = self._compute_hf_at_drop(direction, collateral_usd, debt_usd,
                                                 liq_threshold_bps, drop + cascade_additional)
        results.append(base_hf)
    return results
```

**Close Factor Risk Check** (Aave V3 `LiquidationLogic.sol`):

| Condition | Max Liquidation per Call |
|-----------|------------------------|
| HF > 0.95 AND collateral >= $2,000 AND debt >= $2,000 | 50% of debt |
| HF <= 0.95 | **100% of debt** |
| Collateral or debt < $2,000 | **100% of debt** |

At the bot's $5K-$10K range with 3x leverage, a significant price drop can push values below $2,000, enabling **complete single-call liquidation** with no partial-liquidation second chance. The 5% liquidation bonus means the bot loses more than just the price decline.

```python
async def check_close_factor_risk(self, collateral_usd: Decimal, debt_usd: Decimal) -> bool:
    """Warn if position size risks 100% close factor liquidation."""
    for drop in self.stress_test_drops:
        projected_collateral = collateral_usd * (1 + drop)
        if projected_collateral < Decimal("2000") or debt_usd < Decimal("2000"):
            logger.warning(f"At {drop:.0%} drop, position hits 100% close factor threshold")
            return False  # Reject — too risky
    return True
```

**Borrow Rate Cost Check** (prevents entering when holding cost exceeds expected return):

Aave V3's kinked interest rate model can spike from ~5% to 100%+ APR when utilization crosses the optimal point. The bot must factor borrowing cost into entry decisions.

```python
async def check_borrow_rate_acceptable(self, asset: str, projected_hold_hours: float) -> Tuple[bool, Decimal]:
    """Reject entry if projected borrow cost exceeds threshold."""
    reserve_data = await self.aave_client.get_reserve_data(asset)
    current_rate_apr = reserve_data.variable_borrow_rate / RAY * Decimal("100")
    utilization = reserve_data.utilization_rate
    # Calculate projected cost
    projected_cost_pct = current_rate_apr * Decimal(projected_hold_hours) / Decimal("8760")
    # Reject if cost exceeds threshold (default: 0.5% for expected hold period)
    max_cost = self.config.get("max_borrow_cost_pct", Decimal("0.5"))
    acceptable = projected_cost_pct <= max_cost
    if not acceptable:
        logger.warning(f"Borrow rate too high: {current_rate_apr:.2f}% APR, "
                       f"projected cost {projected_cost_pct:.3f}% over {projected_hold_hours}h")
    return (acceptable, current_rate_apr)
```

**References**: "Optimal Risk-Aware Interest Rates for DeFi Lending Protocols" (arXiv:2502.19862); Aave V3 Interest Rate Strategy documentation; Perez et al. (2021), FC 2021; Aave V3.3 Close Factor improvements.

**Position Sizing via Fractional Kelly Criterion** (prevents catastrophic oversizing):

The signal engine provides a recommended position size via fractional Kelly (Layer 4 of the signal architecture). The strategy validates and constrains this size through additional risk checks:

```python
def validate_position_size(self, signal: TradeSignal) -> Decimal:
    """Apply risk constraints to signal engine's recommended position size."""
    raw_size = signal.recommended_size_usd

    # 1. GARCH volatility adjustment — reduce size in high-volatility regimes
    garch_vol = self.compute_garch_volatility()
    if garch_vol > self.config.high_vol_threshold:
        vol_scalar = self.config.high_vol_threshold / garch_vol
        raw_size *= vol_scalar
        logger.info(f"Position reduced by {(1-vol_scalar)*100:.0f}% for high volatility ({garch_vol:.4f})")

    # 2. Hard limits
    raw_size = min(raw_size, Decimal(str(self.config.max_position_usd)))
    raw_size = min(raw_size, self._available_equity() * self.config.max_leverage_ratio)

    # 3. Drawdown-based reduction (Vince optimal-f approach)
    if self.pnl_tracker.current_drawdown_pct > Decimal("0.1"):
        # Reduce sizing proportionally to drawdown beyond 10%
        dd_scalar = max(Decimal("0.25"), 1 - self.pnl_tracker.current_drawdown_pct)
        raw_size *= dd_scalar
        logger.warning(f"Position reduced to {dd_scalar*100:.0f}% due to drawdown")

    return raw_size

def compute_garch_volatility(self) -> Decimal:
    """GARCH(1,1) one-step-ahead volatility forecast for position sizing.
    Hansen & Lunde (2005): GARCH(1,1) difficult to beat for standard
    volatility forecasting. Hybrid GARCH+LSTM is better (Kim & Won 2018)
    but GARCH(1,1) is sufficient and avoids ML complexity."""
    returns = self.data_service.get_recent_returns("BNBUSDT", periods=100)
    return Indicators.garch_volatility(returns)
```

**Alpha Decay Monitoring and Strategy Health**:

Cong et al. (2024) documented that crypto trading strategies decay with ~12-month half-life — the crypto carry trade went from Sharpe 6.45 (2018-2021) to negative (2022+). The strategy continuously monitors its own performance to detect degradation.

```python
def check_strategy_health(self) -> StrategyHealthReport:
    """Detect alpha decay and recommend parameter refresh or strategy rotation."""
    stats = self.pnl_tracker.get_rolling_stats(window_days=30)
    historical = self.pnl_tracker.get_rolling_stats(window_days=180)

    report = StrategyHealthReport()

    # Accuracy decay check
    if historical.win_rate > 0:
        accuracy_ratio = stats.win_rate / historical.win_rate
        if accuracy_ratio < Decimal("0.7"):
            report.alpha_decay_detected = True
            report.recommendations.append(
                f"Win rate decayed to {accuracy_ratio:.0%} of 6-month average. "
                "Consider parameter refresh or regime filter adjustment."
            )

    # Sharpe ratio degradation
    if historical.sharpe_ratio > 0:
        sharpe_ratio = stats.sharpe_ratio / historical.sharpe_ratio
        if sharpe_ratio < Decimal("0.5"):
            report.alpha_decay_detected = True
            report.recommendations.append(
                f"Sharpe ratio at {sharpe_ratio:.0%} of historical. Strategy may be crowded."
            )

    # If alpha decay detected, increase confidence threshold by 10%
    if report.alpha_decay_detected:
        self.dynamic_confidence_threshold = min(
            self.config.min_confidence * Decimal("1.1"),
            Decimal("0.9")
        )
        logger.warning(f"Alpha decay: raising confidence threshold to {self.dynamic_confidence_threshold}")

    return report
```

**Deleverage Amount Formula** (derived from HF constraint — valid for both long and short):
```
repay_amount = (D - C * LT / h_t) / (1 + f - LT / h_t)
```
Where D=debt, C=collateral, LT=liquidation threshold, h_t=target HF, f=flash premium. The swap direction reverses for shorts (withdraw stablecoin collateral, swap to volatile asset, repay volatile debt).

---

### 4. `core/position_manager.py` — Position Lifecycle (~450-520 lines)

**Purpose**: Orchestrates each position action by composing aggregator quotes, Aave calldata, and tx submission. Handles both **long and short** positions using the same contract with parameterized token roles.

**Key Functions**:
```python
class PositionManager:
    async def open_position(self, direction: PositionDirection, debt_token: str,
                            amount: Decimal, collateral_token: str) -> PositionState
    async def increase_position(self, additional_amount: Decimal) -> PositionState
    async def partial_deleverage(self, target_hf: Decimal) -> PositionState
    async def close_position(self) -> PositionState
    async def _check_isolation_mode(self, collateral_token: str) -> Tuple[bool, str]
```

**Open Position Flow** (direction-aware):
1. **Isolation Mode check** (SHORT only): If collateral is USDT, verify it is not restricted by Aave Isolation Mode. USDT in isolation imposes debt ceilings and restricts borrowable assets. **Prefer USDC for short collateral** (LTV 77% vs 75%, LT 80% vs 78%, fewer restrictions).
2. `aggregator_client.get_best_quote(debt_token → collateral_token)` → best SwapQuote from parallel fan-out
3. `aggregator_client.check_dex_oracle_divergence(quote, chainlink_price)` → reject if >1% divergence
4. Validate quote: `to_amount_min >= expected * (1 - slippage)`
5. `aave_client.encode_flash_loan(debt_token, amount, mode=2, params)` → tx calldata
6. `safety.can_submit_tx(gas_price)` → check kill switches
7. `tx_submitter.simulate(tx)` → eth_call dry run
8. If `dry_run=false`: `tx_submitter.submit_and_wait(tx)` → receipt
9. `aave_client.get_user_account_data()` → verify HF post-execution
10. `pnl_tracker.record_open(position, tx_hash, gas_cost)` → persist to SQLite

**LONG open example**: `open_position(LONG, USDT, 5000, WBNB)` → flash loan 5000 USDT, swap to WBNB, supply WBNB
**SHORT open example**: `open_position(SHORT, WBNB, 10, USDT)` → flash loan 10 WBNB, swap to USDT, supply USDT

**Close Position Flow** (direction-aware):
1. Get full debt amount from Aave (USDT for long, WBNB for short)
2. Flash loan debt in mode=0 (must repay in same tx)
3. In `executeOperation`: repay debt → withdraw collateral → swap collateral→debt → repay flash loan
4. Remainder sent back to wallet
5. `pnl_tracker.record_close(position_id, tx_hash, gas_cost, tokens_received)` → compute realized P&L

**LONG close**: flash loan USDT (mode=0) → repay USDT debt → withdraw WBNB → swap WBNB→USDT → repay flash loan
**SHORT close**: flash loan WBNB (mode=0) → repay WBNB debt → withdraw USDT → swap USDT→WBNB → repay flash loan

**Isolation Mode Check** (for short positions using stablecoin collateral):
```python
async def _check_isolation_mode(self, collateral_token: str) -> Tuple[bool, str]:
    """Check if collateral asset is restricted by Isolation Mode on Aave V3 BSC."""
    reserve_data = await self.aave_client.get_reserve_data(collateral_token)
    if reserve_data.isolation_mode_enabled:
        if reserve_data.debt_ceiling > 0:
            current_debt = reserve_data.current_isolated_debt
            if current_debt >= reserve_data.debt_ceiling:
                return (False, f"{collateral_token} isolation debt ceiling reached")
        return (True, f"{collateral_token} is isolated but within ceiling")
    return (True, f"{collateral_token} is not in isolation mode")
```

**Error Handling**: Flash loans are atomic — on-chain revert means no state change. Off-chain failures (aggregator timeout, RPC failure) are retried with backoff. All gas costs logged to `pnl_tracker`.

---

### 5. `core/safety.py` — Kill Switches (~150-180 lines)

**Purpose**: Centralized safety controls checked before every action.

```python
class SafetyState:
    dry_run: bool                     # Default: true
    max_position_usd: Decimal
    max_leverage_ratio: Decimal
    min_health_factor: Decimal
    max_gas_price_gwei: int
    global_pause: bool
    cooldown_seconds: float
    max_tx_per_24h: int

    def can_open_position(self, amount_usd, leverage) -> Tuple[bool, str]
    def can_submit_tx(self, gas_price_gwei) -> Tuple[bool, str]
    def trigger_emergency_close(self) -> None
    def check_pause_sentinel(self) -> bool   # Check PAUSE file in project root
```

**Default-to-safe**: If config missing or corrupt, defaults to `dry_run=true`, `max_position=0`.

---

### 6. `execution/aggregator_client.py` — DEX Aggregator Integration (~400-470 lines)

**Purpose**: Unified interface to 3 DEX aggregators with **parallel fan-out** (best quote wins), rate limiting, and DEX-Oracle price divergence detection.

**Providers** (all queried in parallel — best `to_amount` wins):
1. **1inch Classic API v6** — 1 RPS free, 400+ BSC liquidity sources, `disableEstimate: true` for flash loan context
   - **CRITICAL**: 1inch Fusion mode is NOT compatible with flash loan callbacks (intent-based, async). Must use Classic/Aggregation API only.
   - Pathfinder splits trades across 5-20 micro-steps across venues
2. **OpenOcean V4** — 2 RPS free, 0% fee, <150ms response, 1000+ sources
3. **ParaSwap V5 (Velora)** — 2 RPS free, Hopper algorithm, 50% positive slippage fee

**Key Parameters for All Providers**:
- `fromAddress` = LeverageExecutor contract address (NOT user wallet)
- `disableEstimate: true` (tokens arrive via flash loan, not in wallet)

```python
@dataclass(frozen=True)
class SwapQuote:
    provider: str
    from_token: str
    to_token: str
    from_amount: Decimal
    to_amount: Decimal
    to_amount_min: Decimal
    calldata: bytes           # Raw calldata for aggregator router
    router_address: str       # Address to call() with calldata
    gas_estimate: int
    price_impact: Decimal

class AggregatorClient:
    async def get_best_quote(self, from_token, to_token, amount, max_slippage_bps=50) -> SwapQuote
    async def check_dex_oracle_divergence(self, quote: SwapQuote, chainlink_price: Decimal, max_divergence_pct=1.0) -> bool
```

**Parallel Fan-Out** (replaces sequential failover):

```python
async def get_best_quote(self, from_token, to_token, amount, max_slippage_bps=50) -> SwapQuote:
    """Query all enabled aggregators in parallel, return best quote."""
    quotes = await asyncio.gather(
        self._quote_1inch(from_token, to_token, amount, max_slippage_bps),
        self._quote_openocean(from_token, to_token, amount, max_slippage_bps),
        self._quote_paraswap(from_token, to_token, amount, max_slippage_bps),
        return_exceptions=True
    )
    valid = [q for q in quotes if isinstance(q, SwapQuote)]
    if not valid:
        raise AggregatorUnavailableError("All aggregator providers failed")
    best = max(valid, key=lambda q: q.to_amount)
    logger.info(f"Best quote: {best.provider} ({best.to_amount} vs "
                f"{', '.join(f'{q.provider}={q.to_amount}' for q in valid if q != best)})")
    return best
```

**DEX-Oracle Price Divergence Check**:

When the bot swaps via DEX, it gets the DEX price. When Aave values collateral, it uses the Chainlink price. During flash crashes or liquidity events, these can diverge — the bot might receive less collateral than Aave values it at, creating an inflated HF that corrects when the oracle catches up.

```python
async def check_dex_oracle_divergence(self, quote: SwapQuote,
                                       chainlink_price: Decimal,
                                       max_divergence_pct: Decimal = Decimal("1.0")) -> bool:
    """Reject swap if DEX price diverges >1% from Chainlink oracle price."""
    implied_dex_price = quote.from_amount / quote.to_amount  # price per collateral token
    divergence = abs(implied_dex_price - chainlink_price) / chainlink_price * 100
    if divergence > max_divergence_pct:
        logger.warning(f"DEX-Oracle divergence: {divergence:.2f}% (max {max_divergence_pct}%)")
        return False
    return True
```

**WebSocket Not Required**: All three aggregators are REST-only for swap quotes (confirmed via official documentation). Quote freshness is a non-issue because: (1) quotes are fetched on-demand at swap time, not continuously; (2) flash loan atomicity means quote and execution happen in the same block; (3) `minAmountOut` on-chain enforcement causes reverts (not losses) if conditions deteriorate. See Angeris et al. (2022) and Diamandis et al. (2023).

**Failover**: If all parallel requests fail → raise `AggregatorUnavailableError`. Individual provider HTTP 429/500/timeout errors are caught silently in `gather(return_exceptions=True)`.

---

### 7. `execution/aave_client.py` — Aave V3 Web3 Wrapper (~300-350 lines)

**Purpose**: Thin read + encode wrapper for Aave V3 contracts. No transaction submission.

**Read Operations**:
```python
class AaveClient:
    async def get_user_account_data(self, user) -> UserAccountData
    async def get_reserve_data(self, asset) -> ReserveData
    async def get_flash_loan_premium(self) -> Decimal
    async def get_asset_price(self, asset) -> Decimal
```

**Encode Operations** (calldata only):
```python
    def encode_flash_loan(self, receiver, assets, amounts, modes, on_behalf_of, params, referral) -> bytes
    def encode_supply(self, asset, amount, on_behalf_of, referral) -> bytes
    def encode_withdraw(self, asset, amount, to) -> bytes
    def encode_repay(self, asset, amount, rate_mode, on_behalf_of) -> bytes
```

**Key Aave V3 Details**:
- `getUserAccountData()` returns values in USD base currency (8 decimal places)
- Health factor: WAD (1e18 = HF of 1.0)
- Flash loan mode 2: debt remains open (no repayment needed)
- Flash loan mode 0: must repay in same tx (premium = 0.05%)

---

### 8. `execution/tx_submitter.py` — Transaction Signing & Submission + Nonce Management (~280-340 lines)

**Purpose**: Sign, simulate, submit, confirm, decode transactions with robust nonce management and stuck transaction recovery.

```python
class TxSubmitter:
    async def simulate(self, tx) -> bytes            # eth_call, returns output or raises
    async def submit(self, tx) -> str                # Sign + send, returns tx hash
    async def wait_for_receipt(self, tx_hash, timeout=60) -> dict
    async def submit_and_wait(self, tx) -> dict      # Combined
    async def get_gas_price(self) -> Tuple[int, int] # (maxFee, priorityFee) in Wei
    async def _get_next_nonce(self) -> int            # Thread-safe local nonce counter
    async def _replace_stuck_tx(self, nonce, gas_bump_pct=12.5) -> str
    async def _recover_nonce_on_startup(self) -> None
    @staticmethod
    def decode_revert_reason(revert_data) -> str
```

**Nonce Management** (prevents stuck/dropped transactions on BSC):

| Strategy | Implementation |
|----------|---------------|
| Local nonce counter | In-memory counter initialized from `getTransactionCount('pending')` on startup |
| Atomic increment | `asyncio.Lock` prevents race conditions between health monitor and strategy tasks |
| Replacement logic | If TX pending >30s, re-send with same nonce + 12.5% higher gas price |
| Gap detection | If `getTransactionCount('latest')` != local counter - pending count, re-sync |
| Crash recovery | On startup, query `pending` vs `latest` nonce counts to detect abandoned transactions |

```python
class TxSubmitter:
    def __init__(self, ...):
        self._nonce: Optional[int] = None
        self._nonce_lock = asyncio.Lock()

    async def _get_next_nonce(self) -> int:
        async with self._nonce_lock:
            if self._nonce is None:
                self._nonce = await self.w3.eth.get_transaction_count(self.address, 'pending')
            nonce = self._nonce
            self._nonce += 1
            return nonce

    async def _replace_stuck_tx(self, nonce: int, gas_bump_pct: float = 12.5) -> str:
        """Re-submit a stuck transaction with higher gas at the same nonce."""
        current_gas = await self.get_gas_price()
        bumped_gas = int(current_gas[0] * (1 + gas_bump_pct / 100))
        # Build replacement tx with same nonce, higher gas
        ...
```

**Failed TX Handling**: On BSC, failed transactions consume gas up to the revert point. With ~5 Gwei gas and 500K-1M gas for a flash loan + swap + supply operation, a failed TX costs ~$0.75-$1.50. After revert:
1. Decode revert reason from receipt
2. Distinguish retryable failures (stale quote, gas too low) from permanent failures (missing approval, router not whitelisted)
3. Do NOT immediately retry with same parameters — price/state has likely changed
4. Log to P&L tracker as a gas cost

**No bundle submission needed**. Normal tx submission via MEV-protected RPC (48 Club Privacy RPC: `https://rpc.48.club`).

---

### 9. `contracts/src/LeverageExecutor.sol` — Smart Contract (~300-350 lines Solidity)

**Purpose**: On-chain contract that receives Aave flash loans and executes aggregator swaps atomically.

**Key Design**: Single contract, NO protocol-specific adapters. Accepts arbitrary calldata for approved routers.

```solidity
contract LeverageExecutor is IFlashLoanReceiver, Ownable, Pausable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    address public immutable AAVE_POOL;
    mapping(address => bool) public approvedRouters;

    // Entry points (called by bot)
    function openLeveragePosition(
        address debtAsset, uint256 flashAmount, address collateralAsset,
        address swapRouter, bytes calldata swapCalldata, uint256 minCollateralOut
    ) external onlyOwner whenNotPaused nonReentrant;

    function closeLeveragePosition(
        address debtAsset, uint256 debtAmount, address collateralAsset,
        uint256 collateralToWithdraw, address swapRouter,
        bytes calldata swapCalldata, uint256 minDebtTokenOut
    ) external onlyOwner whenNotPaused nonReentrant;

    function deleveragePosition(
        address debtAsset, uint256 repayAmount, address collateralAsset,
        uint256 collateralToWithdraw, address swapRouter,
        bytes calldata swapCalldata, uint256 minDebtTokenOut
    ) external onlyOwner whenNotPaused nonReentrant;

    // Aave callback
    function executeOperation(...) external override returns (bool);

    // Admin
    function setRouterApproval(address router, bool approved) external onlyOwner;
    function rescueTokens(address token, uint256 amount) external onlyOwner;
}
```

**executeOperation for open (mode=2)** — identical logic for both LONG and SHORT:
1. Validate `msg.sender == AAVE_POOL` and `initiator == address(this)`
2. Decode params: `(swapRouter, swapCalldata, collateralAsset, minCollateralOut)`
3. Validate `approvedRouters[swapRouter]`
4. `forceApprove()` debt token to swapRouter (handles USDT non-standard approve)
5. `swapRouter.call(swapCalldata)` — swaps debt → collateral
6. Verify collateral received >= `minCollateralOut`
7. Supply collateral to Aave
8. Return true (mode=2: debt stays open)

**Long vs Short**: The contract is **direction-agnostic**. For a long, `debtAsset=USDT`, `collateralAsset=WBNB`. For a short, `debtAsset=WBNB`, `collateralAsset=USDT`. The same `openLeveragePosition`, `closeLeveragePosition`, and `deleveragePosition` functions handle both. The caller (Python bot) determines direction by which token addresses fill which parameter slots. This is confirmed by reference implementations (Cyfrin Updraft `LongShort.sol`, DeFi Saver recipe system).

**Security**: `forceApprove()` for USDT, immutable AAVE_POOL, router whitelist, minAmountOut slippage protection, onlyOwner, ReentrancyGuard, Pausable.

---

### 10. `core/signal_engine.py` — 5-Layer Entry Signal Generation (~400-480 lines)

**Purpose**: Generates confidence-scored trade entry/exit signals from a multi-source signal pipeline. This module addresses the critical gap identified in the audit: the plan previously defined position management (HOW) but not signal generation (WHEN to enter). Phase 3 research established that no single signal source is reliable — ensemble methods with confidence thresholds are required.

**Key Design**: The signal engine implements a 5-layer architecture derived from peer-reviewed findings: Regime Detection → Multi-Source Directional Signals → Ensemble Confidence → Position Sizing → Risk Management. Each layer filters or modulates the next. Conservative defaults and mandatory confidence thresholds prevent low-quality entries.

```python
class SignalEngine:
    async def run(self, signal_queue: asyncio.Queue) -> None   # Main loop: fetch data, compute, emit
    async def _evaluate_once(self) -> Optional[TradeSignal]

    # Layer 1: Regime Detection
    def _detect_regime(self, indicators: IndicatorSnapshot) -> MarketRegime
    def _compute_hurst_exponent(self, prices: List[Decimal], max_lag: int = 20) -> Decimal

    # Layer 2: Multi-Source Signal Collection
    def _compute_technical_signals(self, indicators: IndicatorSnapshot) -> SignalComponent
    async def _compute_order_book_imbalance(self) -> SignalComponent
    async def _compute_vpin(self) -> SignalComponent
    async def _compute_btc_volatility_spillover(self) -> SignalComponent
    async def _compute_liquidation_heatmap(self) -> SignalComponent
    async def _compute_exchange_flows(self) -> SignalComponent
    async def _compute_funding_rate_signal(self) -> SignalComponent
    async def _compute_aggregate_mempool_flow(self) -> SignalComponent

    # Layer 3: Ensemble Confidence
    def _compute_ensemble_confidence(self, components: List[SignalComponent], regime: MarketRegime) -> Decimal
    def _check_entry_rules(self, direction: PositionDirection, indicators: IndicatorSnapshot) -> bool

    # Layer 4: Position Sizing
    def _compute_kelly_fraction(self, confidence: Decimal, volatility: Decimal) -> Decimal
    def _compute_position_size(self, kelly_f: Decimal, account_equity: Decimal) -> Decimal

    # Layer 5: Handed off to Strategy (risk management)

    # Monitoring
    def _check_alpha_decay(self, signal_history: List[TradeSignal]) -> bool
```

**Layer 1 — Regime Detection via Hurst Exponent**:

The Hurst exponent (H) classifies market behavior. Maraj-Mervar & Aybar (FracTime 2025) demonstrated regime-adaptive strategies achieve Sharpe 2.10 vs 0.85 for static strategies on crypto markets.

```python
def _compute_hurst_exponent(self, prices: List[Decimal], max_lag: int = 20) -> Decimal:
    """Rescaled range (R/S) method for Hurst exponent estimation."""
    log_returns = [ln(prices[i] / prices[i-1]) for i in range(1, len(prices))]
    # R/S analysis across multiple sub-period sizes
    # H > 0.55 → trending (momentum strategies preferred)
    # H < 0.45 → mean-reverting (mean-reversion strategies preferred)
    # 0.45 ≤ H ≤ 0.55 → random walk (reduce position sizing, raise confidence threshold)
    ...
    return hurst_exponent

def _detect_regime(self, indicators: IndicatorSnapshot) -> MarketRegime:
    h = self._compute_hurst_exponent(indicators.recent_prices)
    if h > Decimal("0.55") and indicators.atr_ratio > Decimal("1.0"):
        return MarketRegime.TRENDING
    elif h < Decimal("0.45"):
        return MarketRegime.MEAN_REVERTING
    elif indicators.atr_ratio > Decimal("3.0"):
        return MarketRegime.VOLATILE
    else:
        return MarketRegime.RANGING
```

**Layer 2 — Multi-Source Signal Components** (tiered by reliability):

**Tier 1 signals** (highest weight, most evidence):

| Signal | Weight | Academic Basis |
|--------|--------|----------------|
| Technical indicators (EMA, RSI, MACD, BB) | 0.25 | Hudson & Urquhart (2019): ~15,000 rules show significant predictability |
| Order book imbalance | 0.30 | Kolm et al. (2023): 73% of prediction performance from OBI |
| VPIN (Volume-Synchronized Probability of Informed Trading) | 0.20 | Abad & Yagüe (2025): VPIN significantly predicts price jumps |

**Tier 2 signals** (supplementary, moderate weight):

| Signal | Weight | Academic Basis |
|--------|--------|----------------|
| BTC→BNB volatility spillover | 0.10 | DCC-GARCH literature: BTC is net volatility transmitter |
| Liquidation heatmap | 0.10 | Perez et al. (2021): 3% variations make >$10M liquidatable |
| Exchange flows (USDT inflows) | 0.08 | Chi et al. (2024): USDT net inflows positively predict returns |
| Funding rate | 0.07 | Aloosh & Bekaert (2022): 12.5% price variation over 7 days |

**Tier 3 signals** (informational, low weight):

| Signal | Weight | Academic Basis |
|--------|--------|----------------|
| Aggregate mempool flow (5-30 min windows) | 0.05 | Ante & Saggu (2024): mempool predicts volume, not direction |
| Social sentiment volume proxy | 0.03 | Shen et al. (2019): tweet volume > polarity for BTC |

```python
@dataclass(frozen=True)
class SignalComponent:
    source: str              # e.g., "order_book_imbalance", "vpin", "technical"
    tier: int                # 1, 2, or 3
    direction: PositionDirection  # LONG or SHORT
    strength: Decimal        # -1.0 to 1.0 (negative = bearish, positive = bullish)
    weight: Decimal          # tier-dependent weight
    confidence: Decimal      # 0.0-1.0 self-assessed confidence
    data_age_seconds: int    # freshness of underlying data
```

**Order Book Imbalance** (Tier 1):

Kolm et al. (2023) demonstrated OBI is the dominant feature for short-term price prediction. The bot queries Binance's order book depth API and computes:

```python
async def _compute_order_book_imbalance(self) -> SignalComponent:
    """Compute order book imbalance from Binance spot order book."""
    depth = await self.data_service.get_order_book("BNBUSDT", limit=20)
    bid_volume = sum(qty for price, qty in depth.bids)
    ask_volume = sum(qty for price, qty in depth.asks)
    obi = (bid_volume - ask_volume) / (bid_volume + ask_volume)  # [-1, 1]
    # OBI > 0.3 → strong buy pressure; OBI < -0.3 → strong sell pressure
    direction = PositionDirection.LONG if obi > 0 else PositionDirection.SHORT
    return SignalComponent(
        source="order_book_imbalance", tier=1,
        direction=direction, strength=obi,
        weight=Decimal("0.30"), confidence=abs(obi),
        data_age_seconds=0
    )
```

**VPIN — Volume-Synchronized Probability of Informed Trading** (Tier 1):

VPIN measures the imbalance between buy- and sell-initiated volume using volume bucketing (Easley et al. 2012). Abad & Yagüe (2025) confirmed VPIN significantly predicts crypto price jumps. Computed from confirmed on-chain trades, not mempool.

```python
async def _compute_vpin(self) -> SignalComponent:
    """Compute VPIN from recent trade flow (Binance aggTrades endpoint)."""
    trades = await self.data_service.get_recent_trades("BNBUSDT", limit=1000)
    # Volume bucket size = avg_daily_volume / 50
    bucket_size = self.avg_daily_volume / 50
    buckets = self._volume_bucket(trades, bucket_size)
    # VPIN = mean(|V_buy - V_sell| / V_total) over last N buckets
    vpin = mean(abs(b.buy_vol - b.sell_vol) / b.total_vol for b in buckets[-self.vpin_window:])
    # High VPIN (>0.7) → informed trading detected → potential price jump imminent
    # Direction inferred from net buy/sell imbalance in recent buckets
    net_direction = sum(b.buy_vol - b.sell_vol for b in buckets[-5:])
    direction = PositionDirection.LONG if net_direction > 0 else PositionDirection.SHORT
    return SignalComponent(
        source="vpin", tier=1,
        direction=direction, strength=Decimal(str(vpin)),
        weight=Decimal("0.20"), confidence=min(vpin * Decimal("1.3"), Decimal("1.0")),
        data_age_seconds=int(time.time() - trades[-1].timestamp)
    )
```

**BTC Volatility Spillover** (Tier 2):

BTC is the dominant volatility transmitter in crypto markets. DCC-GARCH analysis shows BTC volatility shocks propagate to altcoins within 1-4 hours. Monitoring BTC's realized volatility provides early warning for BNB volatility regime shifts.

```python
async def _compute_btc_volatility_spillover(self) -> SignalComponent:
    """Monitor BTC realized volatility as leading indicator for BNB."""
    btc_candles = await self.data_service.get_ohlcv("BTCUSDT", "1h", limit=24)
    bnb_candles = await self.data_service.get_ohlcv("BNBUSDT", "1h", limit=24)
    btc_rv = self._realized_volatility(btc_candles)
    bnb_rv = self._realized_volatility(bnb_candles)
    # If BTC volatility is spiking but BNB hasn't yet → expect spillover
    spillover_ratio = btc_rv / bnb_rv if bnb_rv > 0 else Decimal("1.0")
    # Spillover ratio > 1.5 → BNB volatility likely to increase
    # Use BTC direction as leading indicator for BNB
    btc_direction = PositionDirection.LONG if btc_candles[-1].close > btc_candles[-4].close \
                    else PositionDirection.SHORT
    return SignalComponent(
        source="btc_volatility_spillover", tier=2,
        direction=btc_direction,
        strength=min(spillover_ratio / Decimal("3.0"), Decimal("1.0")),
        weight=Decimal("0.10"), confidence=min(spillover_ratio / Decimal("2.0"), Decimal("1.0")),
        data_age_seconds=int(time.time() - btc_candles[-1].timestamp)
    )
```

**Liquidation Heatmap** (Tier 2):

Monitoring Aave V3 health factor distributions identifies price levels where cascading liquidations would occur. These act as magnets — prices approaching liquidation walls experience amplified moves from cascade selling (Perez et al. FC 2021). The bot can also query Binance Futures liquidation data for cross-venue awareness.

```python
async def _compute_liquidation_heatmap(self) -> SignalComponent:
    """Identify nearby liquidation price levels from Aave V3 position data."""
    # Query Aave V3 for aggregate position data (via subgraph or direct contract reads)
    # Identify price levels where large positions would be liquidated
    # If current price is approaching a liquidation cluster from above → SHORT bias
    # If current price is approaching a liquidation cluster from below → LONG bias (bounce)
    liquidation_levels = await self.data_service.get_liquidation_levels("WBNB")
    current_price = await self.data_service.get_current_price("BNBUSDT")
    nearest_above = min((l for l in liquidation_levels if l.price > current_price),
                         key=lambda l: l.price - current_price, default=None)
    nearest_below = min((l for l in liquidation_levels if l.price < current_price),
                         key=lambda l: current_price - l.price, default=None)
    # Proximity-weighted signal: closer liquidation walls have stronger influence
    ...
    return SignalComponent(source="liquidation_heatmap", tier=2, ...)
```

**Exchange Flows** (Tier 2):

Chi et al. (2024) found USDT net inflows to exchanges positively predict BTC/ETH returns, while ETH net inflows negatively predict returns (selling pressure). The bot monitors major BSC bridge/exchange contract flows as a directional signal.

```python
async def _compute_exchange_flows(self) -> SignalComponent:
    """Monitor USDT/WBNB flows to/from major exchange hot wallets on BSC."""
    flows = await self.data_service.get_exchange_flows("USDT", window_minutes=60)
    net_inflow = flows.inflow_usd - flows.outflow_usd
    # USDT net inflows → bullish (buying power arriving)
    # USDT net outflows → bearish (capital leaving)
    direction = PositionDirection.LONG if net_inflow > 0 else PositionDirection.SHORT
    strength = min(abs(net_inflow) / flows.avg_hourly_flow, Decimal("1.0"))
    return SignalComponent(
        source="exchange_flows", tier=2,
        direction=direction, strength=strength,
        weight=Decimal("0.08"), confidence=strength * Decimal("0.8"),
        data_age_seconds=flows.data_age_seconds
    )
```

**Funding Rate** (Tier 2):

Aloosh & Bekaert (2022) found funding rates explain 12.5% of price variation over 7-day horizons, with predictive power decaying thereafter. Extremely negative funding → contrarian long signal; extremely positive → contrarian short signal.

```python
async def _compute_funding_rate_signal(self) -> SignalComponent:
    """Use Binance perp funding rate as contrarian signal."""
    funding = await self.data_service.get_funding_rate("BNBUSDT")
    # Funding rate > 0.05% → overleveraged longs → contrarian SHORT
    # Funding rate < -0.05% → overleveraged shorts → contrarian LONG
    if abs(funding) < Decimal("0.0005"):
        return SignalComponent(source="funding_rate", tier=2,
                               direction=PositionDirection.LONG, strength=Decimal("0"),
                               weight=Decimal("0.07"), confidence=Decimal("0"), data_age_seconds=0)
    direction = PositionDirection.SHORT if funding > 0 else PositionDirection.LONG
    strength = min(abs(funding) / Decimal("0.001"), Decimal("1.0"))
    return SignalComponent(
        source="funding_rate", tier=2,
        direction=direction, strength=strength,
        weight=Decimal("0.07"), confidence=strength * Decimal("0.7"),
        data_age_seconds=0
    )
```

**Aggregate Mempool Flow** (Tier 3):

NOT used for real-time entry. Ante & Saggu (2024) showed mempool transaction flow predicts volume but not reliably price direction. BSC's 99.8% PBS block building further limits visibility. Aggregated over 5-30 minute windows as a weak momentum bias only.

```python
async def _compute_aggregate_mempool_flow(self) -> SignalComponent:
    """Aggregate pending transaction volume as medium-term momentum bias."""
    # Query BSC mempool via txpool_content (limited visibility due to PBS)
    # Aggregate large pending DEX swap volumes over 5-30 min window
    # High pending swap volume → increased volatility expected (not direction)
    pending = await self.data_service.get_pending_swap_volume(window_minutes=15)
    if pending.volume_usd < pending.avg_volume_usd * Decimal("1.5"):
        return SignalComponent(source="mempool_flow", tier=3,
                               direction=PositionDirection.LONG, strength=Decimal("0"),
                               weight=Decimal("0.05"), confidence=Decimal("0"), data_age_seconds=0)
    # Infer weak direction from net buy/sell ratio of pending swaps
    direction = PositionDirection.LONG if pending.net_buy_ratio > Decimal("0.55") \
                else PositionDirection.SHORT
    return SignalComponent(
        source="mempool_flow", tier=3,
        direction=direction,
        strength=Decimal(str(pending.net_buy_ratio - Decimal("0.5"))) * 2,
        weight=Decimal("0.05"),
        confidence=Decimal("0.3"),  # Low confidence — informational only
        data_age_seconds=pending.window_seconds
    )
```

**Layer 3 — Ensemble Confidence Scoring**:

All signal components are combined into a single confidence score using regime-weighted aggregation:

```python
def _compute_ensemble_confidence(self, components: List[SignalComponent],
                                  regime: MarketRegime) -> Decimal:
    """Weighted ensemble of all signal sources, regime-adjusted."""
    # Separate bullish and bearish components
    bull_score = Decimal("0")
    bear_score = Decimal("0")
    total_weight = Decimal("0")
    for c in components:
        if c.data_age_seconds > self.max_signal_age_seconds:
            continue  # Skip stale signals
        regime_mult = self._regime_weight_multiplier(c.source, regime)
        weighted = c.strength * c.weight * c.confidence * regime_mult
        if c.direction == PositionDirection.LONG:
            bull_score += weighted
        else:
            bear_score += weighted
        total_weight += c.weight * regime_mult
    if total_weight == 0:
        return Decimal("0")
    # Net directional confidence
    net_score = (bull_score - bear_score) / total_weight
    # Agreement bonus: if >70% of components agree on direction, boost confidence
    directions = [c.direction for c in components if c.strength > Decimal("0.1")]
    if directions:
        majority_pct = max(
            sum(1 for d in directions if d == PositionDirection.LONG),
            sum(1 for d in directions if d == PositionDirection.SHORT)
        ) / len(directions)
        if majority_pct > Decimal("0.7"):
            net_score *= Decimal("1.15")  # 15% agreement bonus
    return min(abs(net_score), Decimal("1.0"))
```

**Regime-Weight Multipliers**: In TRENDING regimes, momentum-aligned signals (EMA, order book imbalance) get 1.2x weight, while mean-reversion signals get 0.5x. In MEAN_REVERTING regimes, the inverse applies. In VOLATILE regimes, all weights reduce to 0.7x (raise confidence threshold). In RANGING regimes, all reduce to 0.8x. Based on Adaptive Market Hypothesis evidence (Lo, 2004; Timmermann & Granger, 2004).

**Configurable Strategy Modes** (via `config/signals.json`):

| Mode | Entry Rule | Academic Basis |
|------|-----------|----------------|
| `momentum` | Regime=TRENDING + Tier 1 consensus bullish + confidence > threshold | Hudson & Urquhart (2019): ~15,000 rules show significant predictability |
| `mean_reversion` | Regime=MEAN_REVERTING + RSI extremes + BB squeeze/expansion + OBI divergence | Kosc et al. (2019): short-term contrarian > momentum in cross-section |
| `blended` | Weighted combination across all modes, regime-adapted | Beluska & Vojtko (2024): 50/50 blend Sharpe 1.71 |
| `manual` | External signal via SIGNAL file in project root or REST endpoint | DeFi Saver model: human-initiated entry |

**Layer 4 — Position Sizing (Fractional Kelly Criterion)**:

```python
def _compute_kelly_fraction(self, confidence: Decimal, volatility: Decimal) -> Decimal:
    """Fractional Kelly for position sizing. MacLean et al. (2010)."""
    # Edge estimate from historical signal accuracy (rolling 30-day window)
    edge = self._get_rolling_edge()  # win_rate * avg_win - loss_rate * avg_loss
    if edge <= 0:
        return Decimal("0")  # No edge → no position
    # Kelly fraction: f* = edge / variance
    variance = volatility ** 2
    full_kelly = edge / variance if variance > 0 else Decimal("0")
    # Apply fractional Kelly (25% of full Kelly) — standard risk reduction
    fractional = full_kelly * Decimal("0.25")
    # Cap at max leverage ratio from config
    return min(fractional, self.config.max_leverage_ratio)

def _compute_position_size(self, kelly_f: Decimal, account_equity: Decimal) -> Decimal:
    """Convert Kelly fraction to USD position size."""
    position_usd = account_equity * kelly_f
    # Apply hard limits from config
    return min(position_usd, self.config.max_position_usd)
```

**Alpha Decay Monitoring**:

Cong et al. (2024) documented that crypto trading strategies decay with ~12-month half-life. The carry trade went from Sharpe 6.45 to negative post-2021. The bot tracks rolling signal accuracy and degrades confidence when performance deteriorates.

```python
def _check_alpha_decay(self, signal_history: List[TradeSignal]) -> bool:
    """Detect strategy decay via rolling accuracy degradation."""
    if len(signal_history) < 30:
        return False
    recent_accuracy = self._compute_accuracy(signal_history[-30:])
    historical_accuracy = self._compute_accuracy(signal_history[-180:-30])
    if historical_accuracy > 0:
        decay_ratio = recent_accuracy / historical_accuracy
        if decay_ratio < Decimal("0.7"):
            logger.warning(f"Alpha decay detected: accuracy dropped to {decay_ratio:.0%} of historical")
            return True  # Recommend strategy rotation or parameter refresh
    return False
```

**What this module intentionally avoids** (overengineering guard):
- No custom deep learning models (XGBoost/RF consistently outperform for crypto — Hafid et al. 2024; 234-paper survey shows marginal gains with enormous overfitting risk)
- No sub-minute timeframes (bot's execution latency makes them impractical)
- No social media NLP (tweet volume is used as a Tier 3 proxy, but full NLP/sentiment analysis is not built — too noisy per Shen et al. 2019)
- No real-time mempool entry triggers (BSC PBS + bot latency makes them infeasible)

**References**: Kolm et al. (2023), J. Financial Economics; Abad & Yagüe (2025), ScienceDirect; Easley et al. (2012), J. Financial Markets (VPIN); Ante & Saggu (2024), J. Innovation & Knowledge; Aloosh & Bekaert (2022), SSRN; Chi et al. (2024), SSRN; Maraj-Mervar & Aybar (2025), FracTime; MacLean et al. (2010), Quantitative Finance; Cong et al. (2024), Annual Review of Financial Economics; Lo (2004), J. Portfolio Management (AMH); MDPI (2025), Applied Sciences; Perez et al. (2021), FC 2021

---

### 11. `core/indicators.py` — Technical Indicator & Statistical Computation (~280-350 lines)

**Purpose**: Pure computation module for technical indicators, regime statistics, and volatility models. No I/O, no side effects — takes OHLCV arrays, returns indicator values.

```python
class Indicators:
    # --- Standard Technical Indicators ---
    @staticmethod
    def ema(prices: List[Decimal], period: int) -> List[Decimal]

    @staticmethod
    def rsi(prices: List[Decimal], period: int = 14) -> Decimal

    @staticmethod
    def macd(prices: List[Decimal], fast: int = 12, slow: int = 26,
             signal: int = 9) -> Tuple[Decimal, Decimal, Decimal]  # (macd_line, signal_line, histogram)

    @staticmethod
    def bollinger_bands(prices: List[Decimal], period: int = 20,
                        std_mult: Decimal = Decimal("2.0")) -> Tuple[Decimal, Decimal, Decimal]  # (upper, middle, lower)

    @staticmethod
    def atr(highs: List[Decimal], lows: List[Decimal], closes: List[Decimal],
            period: int = 14) -> Decimal

    @staticmethod
    def obv(closes: List[Decimal], volumes: List[Decimal]) -> Decimal

    # --- Regime & Statistical Indicators (Phase 3 additions) ---
    @staticmethod
    def hurst_exponent(prices: List[Decimal], max_lag: int = 20) -> Decimal
        """Rescaled range (R/S) Hurst exponent.
        H > 0.55 → persistent/trending; H < 0.45 → anti-persistent/mean-reverting.
        Maraj-Mervar & Aybar (FracTime 2025): regime-adaptive Sharpe 2.10."""

    @staticmethod
    def realized_volatility(closes: List[Decimal], window: int = 24) -> Decimal
        """Annualized realized volatility from log returns. Used for GARCH seed
        and BTC-BNB volatility spillover computation."""

    @staticmethod
    def garch_volatility(returns: List[Decimal], omega: Decimal = Decimal("0.00001"),
                          alpha: Decimal = Decimal("0.1"), beta: Decimal = Decimal("0.85")) -> Decimal
        """GARCH(1,1) one-step-ahead volatility forecast.
        σ²_{t+1} = ω + α·r²_t + β·σ²_t
        Bollerslev (1986); Hansen & Lunde (2005): GARCH(1,1) difficult to beat
        for standard volatility forecasting. Used for Kelly position sizing."""

    @staticmethod
    def vpin(trades: List[Trade], bucket_size: Decimal, window: int = 50) -> Decimal
        """Volume-Synchronized Probability of Informed Trading.
        Easley et al. (2012): VPIN = mean(|V_buy - V_sell| / V_total) over N buckets.
        Buy/sell classification via tick rule (compare trade price to midpoint)."""

    @staticmethod
    def order_book_imbalance(bids: List[Tuple[Decimal, Decimal]],
                              asks: List[Tuple[Decimal, Decimal]]) -> Decimal
        """OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume).
        Kolm et al. (2023): 73% of prediction performance."""

    # --- Composite ---
    @staticmethod
    def compute_all(candles: List[OHLCV], config: SignalConfig) -> IndicatorSnapshot
```

**Implementation Notes**:
- All computations use `Decimal` for precision consistency with the rest of the bot
- EMA uses the standard multiplier `k = 2 / (period + 1)`
- RSI uses Wilder's smoothing method (exponential moving average of gains/losses)
- Hurst exponent uses R/S analysis with lag sizes from 2 to `max_lag`; requires minimum 100 data points
- GARCH(1,1) uses standard constraint α + β < 1 for stationarity; initialized with realized volatility
- VPIN uses tick rule for buy/sell classification — if trade price ≥ midpoint, it's a buy; otherwise sell
- Standard parameters used throughout — no parameter optimization (prevents overfitting)

---

### 12. `core/data_service.py` — Multi-Source Market Data (~350-420 lines)

**Purpose**: Fetches, caches, and normalizes market data from multiple sources for the multi-source signal pipeline. Chainlink alone is insufficient — it provides only the latest spot price with 27-60s resolution, no volume data, and no OHLCV history. The expanded signal architecture (Phase 3) requires order book depth, recent trades, funding rates, exchange flow proxies, and liquidation level data.

```python
@dataclass(frozen=True)
class OHLCV:
    timestamp: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

@dataclass(frozen=True)
class OrderBookSnapshot:
    bids: List[Tuple[Decimal, Decimal]]  # (price, quantity) sorted desc
    asks: List[Tuple[Decimal, Decimal]]  # (price, quantity) sorted asc
    timestamp: int

@dataclass(frozen=True)
class ExchangeFlows:
    inflow_usd: Decimal
    outflow_usd: Decimal
    avg_hourly_flow: Decimal
    data_age_seconds: int

@dataclass(frozen=True)
class PendingSwapVolume:
    volume_usd: Decimal
    avg_volume_usd: Decimal
    net_buy_ratio: Decimal  # 0.0-1.0
    window_seconds: int

@dataclass(frozen=True)
class LiquidationLevel:
    price: Decimal
    total_collateral_at_risk_usd: Decimal
    position_count: int

class PriceDataService:
    # --- Core OHLCV ---
    async def get_ohlcv(self, symbol: str, interval: str = "1h",
                        limit: int = 200) -> List[OHLCV]
    async def get_current_price(self, symbol: str) -> Decimal

    # --- Order Book (for OBI signal) ---
    async def get_order_book(self, symbol: str, limit: int = 20) -> OrderBookSnapshot

    # --- Recent Trades (for VPIN computation) ---
    async def get_recent_trades(self, symbol: str, limit: int = 1000) -> List[Trade]

    # --- Derivatives Data ---
    async def get_funding_rate(self, symbol: str) -> Optional[Decimal]
    async def get_open_interest(self, symbol: str) -> Optional[Decimal]

    # --- DeFi / On-Chain ---
    async def get_aave_rates(self, asset: str) -> AaveRates
    async def get_liquidation_levels(self, asset: str) -> List[LiquidationLevel]
    async def get_exchange_flows(self, token: str, window_minutes: int = 60) -> ExchangeFlows

    # --- Mempool (Tier 3, limited visibility) ---
    async def get_pending_swap_volume(self, window_minutes: int = 15) -> PendingSwapVolume
```

**Data Sources** (ordered by priority):

| Source | Endpoint | Data Provided | Cost | Rate Limit |
|--------|----------|--------------|------|------------|
| **Binance Spot API** (primary) | `GET /api/v3/klines` | OHLCV for BNB/USDT at any timeframe | Free | 1200 req/min |
| **Binance Spot API** | `GET /api/v3/depth` | Order book depth (for OBI) | Free | 1200 req/min |
| **Binance Spot API** | `GET /api/v3/aggTrades` | Aggregated recent trades (for VPIN) | Free | 1200 req/min |
| **Binance Futures API** | `GET /fapi/v1/fundingRate` | BNB perpetual funding rates | Free | 500 req/min |
| **Binance Futures API** | `GET /fapi/v1/openInterest` | Open interest (leverage proxy) | Free | 500 req/min |
| **GeckoTerminal** (fallback) | `GET /api/v2/networks/bsc/ohlcv` | DEX-specific OHLCV | Free, no key | 30 req/min |
| **Chainlink** (current price only) | On-chain `latestRoundData()` | Latest spot price | RPC cost only | N/A |
| **Aave V3 Pool** (rates + positions) | On-chain `getReserveData()` | Borrow/supply APY, utilization | RPC cost only | N/A |
| **Aave V3 Subgraph** (liquidation levels) | GraphQL query | Position health factor distribution | Free | 1000 req/day |
| **BSC RPC** (mempool, Tier 3) | `txpool_content` | Pending transactions | RPC cost only | N/A |

**Liquidation Level Computation**:

The bot queries Aave V3's subgraph (or directly scans user positions via `getUserAccountData`) to build a distribution of health factors and their associated collateral values. Price levels where HF would cross 1.0 for large aggregate collateral represent liquidation walls.

```python
async def get_liquidation_levels(self, asset: str) -> List[LiquidationLevel]:
    """Query Aave V3 subgraph for position HF distribution, compute liquidation prices."""
    # Query positions with HF < 2.0 (most at risk)
    positions = await self._query_aave_subgraph(asset, max_hf=2.0)
    levels = {}
    for pos in positions:
        # Compute price at which this position would be liquidated
        # For longs: liq_price = (debt / (collateral_qty * LT))
        # For shorts: liq_price = (collateral_usd * LT) / debt_qty
        liq_price = self._compute_liquidation_price(pos)
        bucket = round(liq_price, -1)  # Round to nearest $10
        if bucket not in levels:
            levels[bucket] = LiquidationLevel(price=bucket, total_collateral_at_risk_usd=Decimal("0"), position_count=0)
        levels[bucket].total_collateral_at_risk_usd += pos.collateral_usd
        levels[bucket].position_count += 1
    return sorted(levels.values(), key=lambda l: l.total_collateral_at_risk_usd, reverse=True)
```

**Exchange Flow Proxy**:

Direct exchange flow APIs are typically paid. The bot monitors known exchange hot wallet addresses on BSC for large USDT/WBNB transfers as a free proxy. Known Binance BSC hot wallets are pre-configured. Alternatively, DefiLlama's free API provides aggregate flow data.

```python
async def get_exchange_flows(self, token: str, window_minutes: int = 60) -> ExchangeFlows:
    """Monitor token flows to/from known exchange wallets on BSC."""
    # Option A: Query BSCScan API for recent transfers to/from known exchange addresses
    # Option B: Query DefiLlama bridge/flow API (free, aggregated)
    # Returns net inflow/outflow for the specified window
    ...
```

**Caching**: In-memory LRU cache with TTL based on data type:
- OHLCV: 5min cache for 1h candles, 1min for 15m candles
- Order book: 5s cache (fast-moving, but OBI is computed on-demand)
- Recent trades: 10s cache (for VPIN, batched computation)
- Funding rate: 5min cache (updates every 8 hours)
- Liquidation levels: 5min cache (positions change slowly)
- Exchange flows: 2min cache (aggregated over longer windows anyway)

**Why Binance API as primary**: BNB is the primary collateral asset and Binance is the highest-liquidity venue for BNB/USDT. The API provides all required data types (klines, depth, aggTrades, funding) at generous free rate limits. The bot's total API usage (~20-30 req/min across all endpoints) stays well within limits.

---

### 13. `core/pnl_tracker.py` — P&L Tracking and Position History (~200-250 lines)

**Purpose**: Tracks position lifecycle, computes realized/unrealized P&L including accrued interest, and maintains persistent history in SQLite.

```python
class PnLTracker:
    def __init__(self, db_path: str = "data/positions.db"):
        self.db = sqlite3.connect(db_path)
        self._create_tables()

    async def record_open(self, position: PositionState, tx_hash: str, gas_cost_usd: Decimal) -> int
    async def record_close(self, position_id: int, tx_hash: str, gas_cost_usd: Decimal,
                           tokens_received: Decimal) -> RealizedPnL
    async def record_deleverage(self, position_id: int, tx_hash: str, gas_cost_usd: Decimal) -> None
    async def snapshot(self, position_id: int, position: PositionState) -> None
    async def get_unrealized_pnl(self, position: PositionState) -> Decimal
    async def get_accrued_interest(self, position: PositionState) -> Decimal
    async def get_position_history(self, limit: int = 50) -> List[PositionRecord]
    async def get_summary_stats(self) -> TradingStats
```

**SQLite Schema**:

```sql
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    direction TEXT NOT NULL,          -- 'LONG' or 'SHORT'
    open_timestamp INTEGER NOT NULL,
    close_timestamp INTEGER,
    debt_token TEXT NOT NULL,
    collateral_token TEXT NOT NULL,
    initial_debt_amount TEXT NOT NULL,
    initial_collateral_amount TEXT NOT NULL,
    flash_loan_premium_paid TEXT,
    close_debt_amount TEXT,
    close_collateral_amount TEXT,
    realized_pnl_usd TEXT,
    total_gas_costs_usd TEXT,
    open_tx_hash TEXT NOT NULL,
    close_tx_hash TEXT,
    open_borrow_rate_apr TEXT,
    avg_borrow_rate_apr TEXT,
    close_reason TEXT                -- 'signal', 'deleverage', 'emergency', 'manual'
);

CREATE TABLE position_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER NOT NULL REFERENCES positions(id),
    timestamp INTEGER NOT NULL,
    collateral_value_usd TEXT NOT NULL,
    debt_value_usd TEXT NOT NULL,
    health_factor TEXT NOT NULL,
    borrow_rate_apr TEXT NOT NULL,
    unrealized_pnl_usd TEXT NOT NULL
);

CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER REFERENCES positions(id),
    tx_hash TEXT NOT NULL UNIQUE,
    timestamp INTEGER NOT NULL,
    tx_type TEXT NOT NULL,           -- 'open', 'close', 'deleverage', 'increase'
    gas_used INTEGER,
    gas_price_gwei TEXT,
    gas_cost_usd TEXT,
    success BOOLEAN NOT NULL,
    revert_reason TEXT
);
```

**Unrealized P&L Calculation**:
```python
async def get_unrealized_pnl(self, position: PositionState) -> Decimal:
    # For LONG: profit = (current_collateral_value - initial_collateral_value) - accrued_interest
    # For SHORT: profit = (initial_debt_value - current_debt_value) - accrued_interest
    accrued = await self.get_accrued_interest(position)
    if position.direction == PositionDirection.LONG:
        return (position.collateral_usd - position.initial_collateral_usd) - accrued
    else:
        return (position.initial_debt_usd - position.debt_usd) - accrued
```

**Accrued Interest** (from Aave V3's `variableDebtToken`):
```python
async def get_accrued_interest(self, position: PositionState) -> Decimal:
    scaled_debt = await variable_debt_token.scaledBalanceOf(executor_address)
    prev_index = await variable_debt_token.getPreviousIndex(executor_address)
    initial_borrowed = scaled_debt * prev_index / RAY
    current_debt = await variable_debt_token.balanceOf(executor_address)
    return current_debt - initial_borrowed
```

---

### 14. `shared/types.py` — Shared Data Types (~250-300 lines)

Centralized dataclasses:

```python
class PositionDirection(Enum):
    LONG = "long"    # Collateral=volatile, Debt=stable
    SHORT = "short"  # Collateral=stable, Debt=volatile

class MarketRegime(Enum):
    TRENDING = "trending"         # Hurst > 0.55, ATR ratio 1.0-3.0x
    MEAN_REVERTING = "mean_reverting"  # Hurst < 0.45
    RANGING = "ranging"           # Hurst 0.45-0.55, ATR ratio < 1.0
    VOLATILE = "volatile"         # ATR ratio > 3.0

@dataclass(frozen=True)
class SignalComponent:
    source: str              # e.g., "order_book_imbalance", "vpin", "technical"
    tier: int                # 1, 2, or 3
    direction: PositionDirection
    strength: Decimal        # -1.0 to 1.0
    weight: Decimal          # tier-dependent weight
    confidence: Decimal      # 0.0-1.0 self-assessed confidence
    data_age_seconds: int    # freshness of underlying data

@dataclass(frozen=True)
class TradeSignal:
    direction: PositionDirection
    confidence: Decimal          # 0.0-1.0 (ensemble confidence from all sources)
    strategy_mode: str           # 'momentum', 'mean_reversion', 'blended', 'manual'
    indicators: IndicatorSnapshot
    regime: MarketRegime
    components: List[SignalComponent]  # Contributing signal sources
    recommended_size_usd: Decimal     # Kelly-derived position size
    hurst_exponent: Decimal           # Current Hurst H value
    garch_volatility: Decimal         # GARCH(1,1) forecast
    timestamp: int

@dataclass(frozen=True)
class IndicatorSnapshot:
    price: Decimal
    ema_20: Decimal
    ema_50: Decimal
    ema_200: Decimal
    rsi_14: Decimal
    macd_line: Decimal
    macd_signal: Decimal
    macd_histogram: Decimal
    bb_upper: Decimal
    bb_middle: Decimal
    bb_lower: Decimal
    atr_14: Decimal
    atr_ratio: Decimal           # ATR(14) / ATR_50_avg
    volume: Decimal
    volume_20_avg: Decimal
    hurst: Decimal               # Hurst exponent
    vpin: Decimal                # VPIN value
    obi: Decimal                 # Order book imbalance [-1, 1]
    recent_prices: List[Decimal] # For Hurst computation (last 200 closes)

@dataclass(frozen=True)
class OHLCV:
    timestamp: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

@dataclass(frozen=True)
class BorrowRateInfo:
    variable_rate_apr: Decimal
    utilization_rate: Decimal
    optimal_utilization: Decimal
    rate_at_kink: Decimal         # Rate if utilization were at optimal point

@dataclass
class RealizedPnL:
    gross_pnl_usd: Decimal
    accrued_interest_usd: Decimal
    gas_costs_usd: Decimal
    flash_loan_premiums_usd: Decimal
    net_pnl_usd: Decimal

@dataclass
class TradingStats:
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl_usd: Decimal
    avg_pnl_per_trade_usd: Decimal
    win_rate: Decimal
    sharpe_ratio: Decimal
    avg_hold_duration_hours: Decimal
    current_drawdown_pct: Decimal
    max_drawdown_pct: Decimal

@dataclass
class StrategyHealthReport:
    alpha_decay_detected: bool = False
    accuracy_ratio: Decimal = Decimal("1.0")   # recent / historical win rate
    sharpe_ratio: Decimal = Decimal("1.0")     # recent / historical Sharpe
    recommendations: List[str] = field(default_factory=list)
    dynamic_confidence_threshold: Optional[Decimal] = None

@dataclass(frozen=True)
class Trade:
    """Individual trade from exchange (for VPIN computation)."""
    price: Decimal
    quantity: Decimal
    timestamp: int
    is_buyer_maker: bool  # True = sell-initiated (maker was buyer)
```

Also includes existing types: `HFTier`, `PositionAction`, `SwapQuote`, `UserAccountData`, `ReserveData`, `PositionState` (updated with `direction: PositionDirection`), `HealthStatus`, `SafetyCheck`, `OrderBookSnapshot`, `ExchangeFlows`, `PendingSwapVolume`, `LiquidationLevel`.

### 15. `shared/constants.py` — Shared Constants (~60-70 lines)

`WAD` (1e18), `RAY` (1e27), `USD_DECIMALS` (1e8), `SECONDS_PER_YEAR` (31536000), Aave V3 addresses, Chainlink feed addresses, flash loan modes, default safety values, `CLOSE_FACTOR_THRESHOLD_USD` (Decimal("2000")).

---

## Key Contract Addresses

### Aave V3 on BSC
- Pool: `0x6807dc923806fE8Fd134338EABCA509979a7e0cB`
- PoolAddressesProvider: `0xff75B6da14FfbbfD355Daf7a2731456b3562Ba6D`
- AaveOracle: `0x39bc1bfDa2130d6Bb6DBEfd366939b4c7aa7C697`
- Pool Data Provider: `0xc90Df74A7c16245c5F5C5870327Ceb38Fe5d5328`
- ACL Manager: `0x2D97F8FA96886Fd923c065F5457F9DDd494e3877`

### Chainlink BSC Feeds
- BNB/USD: `0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE` (27s heartbeat, 0.1% deviation)
- BTC/USD: `0x264990fbd0A4796A3E3d8E37C4d5F87a3aCa5Ebf` (60s heartbeat, 0.05% deviation)
- ETH/USD: `0x9ef1B8c0E4F7dc8bF5719Ea496883DC6401d5b2e` (60s heartbeat, 0.05% deviation)

### BSC Infrastructure
- Multicall3: `0xcA11bde05977b3631167028862bE2a173976CA11`

---

## Config File Schemas

### `config/chains/56.json` (MODIFIED — remove entire `dex` block)
```json
{
    "chain_id": 56,
    "chain_name": "BSC Mainnet",
    "block_time_seconds": 0.75,
    "native_token": "BNB",
    "wrapped_native": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
    "rpc": {
        "http_url": "https://bsc-dataseed1.binance.org/",
        "http_url_fallback": "https://bsc-dataseed2.binance.org/",
        "mev_protected_url": "https://rpc.48.club"
    },
    "contracts": {
        "multicall3": "0xcA11bde05977b3631167028862bE2a173976CA11",
        "aave_v3_pool": "0x6807dc923806fE8Fd134338EABCA509979a7e0cB",
        "aave_v3_pool_addresses_provider": "0xff75B6da14FfbbfD355Daf7a2731456b3562Ba6D",
        "aave_v3_oracle": "0x39bc1bfDa2130d6Bb6DBEfd366939b4c7aa7C697",
        "aave_v3_data_provider": "0xc90Df74A7c16245c5F5C5870327Ceb38Fe5d5328",
        "leverage_executor": ""
    },
    "chainlink_feeds": {
        "BNB_USD": { "address": "0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE", "heartbeat_seconds": 27, "deviation_threshold_percent": 0.1, "decimals": 8 },
        "BTC_USD": { "address": "0x264990fbd0A4796A3E3d8E37C4d5F87a3aCa5Ebf", "heartbeat_seconds": 60, "deviation_threshold_percent": 0.05, "decimals": 8 },
        "ETH_USD": { "address": "0x9ef1B8c0E4F7dc8bF5719Ea496883DC6401d5b2e", "heartbeat_seconds": 60, "deviation_threshold_percent": 0.05, "decimals": 8 }
    },
    "tokens": {
        "WBNB": { "address": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c", "decimals": 18 },
        "USDT": { "address": "0x55d398326f99059fF775485246999027B3197955", "decimals": 18 },
        "USDC": { "address": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d", "decimals": 18 },
        "BTCB": { "address": "0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c", "decimals": 18 },
        "ETH":  { "address": "0x2170Ed0880ac9A755fd29B2688956BD959F933F8", "decimals": 18 },
        "FDUSD": { "address": "0xc5f0f7b66764F6ec8C8Dff7BA683102295E16409", "decimals": 18 },
        "CAKE": { "address": "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82", "decimals": 18 }
    },
    "multicall": { "batch_size": 25, "batch_delay_seconds": 0.1 }
}
```

### `config/aave.json` (NEW)
```json
{
    "flash_loan_premium_bps": 5,
    "flash_loan_premium": "0.0005",
    "referral_code": 0,
    "supported_assets": {
        "WBNB": { "ltv_bps": 7500, "liquidation_threshold_bps": 8000, "liquidation_bonus_bps": 500, "can_be_collateral": true, "can_be_borrowed": true, "isolation_mode": false },
        "USDT": { "ltv_bps": 7500, "liquidation_threshold_bps": 7800, "liquidation_bonus_bps": 500, "can_be_collateral": true, "can_be_borrowed": true, "isolation_mode": true, "note": "USDT may be restricted to Isolation Mode — verify on-chain before using as short collateral. Prefer USDC." },
        "USDC": { "ltv_bps": 7700, "liquidation_threshold_bps": 8000, "liquidation_bonus_bps": 500, "can_be_collateral": true, "can_be_borrowed": true, "isolation_mode": false, "note": "Preferred stablecoin for short collateral: higher LTV (77%), higher LT (80%), no isolation restrictions." },
        "BTCB": { "ltv_bps": 7000, "liquidation_threshold_bps": 7500, "liquidation_bonus_bps": 1000, "can_be_collateral": true, "can_be_borrowed": true, "isolation_mode": false },
        "ETH":  { "ltv_bps": 8000, "liquidation_threshold_bps": 8250, "liquidation_bonus_bps": 500, "can_be_collateral": true, "can_be_borrowed": true, "isolation_mode": false }
    },
    "e_mode": {
        "enabled": false,
        "category_id": 0,
        "note": "E-Mode disabled: WBNB and stablecoins are in different categories. E-Mode only helps correlated pairs (stablecoin-stablecoin at 97% LTV, or BNB-BNB_LSD at 90%+ LTV). Aave V3.2 Liquid E-Modes may expand options for future BNB liquid staking derivative strategies."
    },
    "interest_rate_model": {
        "note": "Aave V3 uses a kinked interest rate model. Below optimal utilization: R = R_base + (U/U_optimal) * R_slope1. Above: R = R_base + R_slope1 + ((U-U_optimal)/(1-U_optimal)) * R_slope2. Slope2 can be 60-300% APR — the bot MUST monitor utilization to avoid rate spikes.",
        "monitor_utilization": true,
        "max_acceptable_borrow_apr": "15.0",
        "rate_spike_pause_threshold_apr": "50.0"
    }
}
```

### `config/positions.json` (NEW)
```json
{
    "dry_run": true,
    "max_flash_loan_usd": 5000,
    "max_position_usd": 10000,
    "max_leverage_ratio": "3.0",
    "min_health_factor": "1.5",
    "deleverage_threshold": "1.4",
    "close_threshold": "1.25",
    "target_hf_after_deleverage": "1.8",
    "max_gas_price_gwei": 10,
    "max_slippage_bps": 50,
    "cooldown_between_actions_seconds": 30,
    "max_transactions_per_24h": 50,
    "stress_test_price_drops": ["-0.05", "-0.10", "-0.15", "-0.20", "-0.30"],
    "min_stress_test_hf": "1.1",
    "cascade_liquidation_threshold_usd": 50000000,
    "cascade_additional_drop": "-0.03",
    "close_factor_warning_threshold_usd": 2000,
    "max_borrow_cost_pct": "0.5",
    "max_acceptable_borrow_apr": "15.0",
    "preferred_short_collateral": "USDC",
    "max_dex_oracle_divergence_pct": "1.0",
    "oracle_max_staleness_seconds": 60
}
```

### `config/aggregator.json` (NEW)
```json
{
    "providers": [
        {
            "name": "1inch",
            "enabled": true,
            "priority": 1,
            "base_url": "https://api.1inch.dev/swap/v6.0/56",
            "api_key_env": "ONEINCH_API_KEY",
            "rate_limit_rps": 1,
            "timeout_seconds": 5,
            "approved_routers": ["0x111111125421cA6dc452d289314280a0f8842A65"],
            "params": { "disableEstimate": true }
        },
        {
            "name": "openocean",
            "enabled": true,
            "priority": 2,
            "base_url": "https://open-api.openocean.finance/v3/56",
            "api_key_env": "",
            "rate_limit_rps": 2,
            "timeout_seconds": 5,
            "approved_routers": ["0x6352a56caadC4F1E25CD6c75970Fa768A3304e64"],
            "params": {}
        },
        {
            "name": "paraswap",
            "enabled": true,
            "priority": 3,
            "base_url": "https://apiv5.paraswap.io",
            "api_key_env": "",
            "rate_limit_rps": 2,
            "timeout_seconds": 5,
            "approved_routers": ["0xDEF171Fe48CF0115B1d80b88dc8eAB59176FEe57"],
            "params": {}
        }
    ],
    "max_slippage_bps": 50,
    "max_price_impact_percent": "1.0"
}
```

### `config/signals.json` (NEW)
```json
{
    "enabled": true,
    "mode": "blended",
    "data_source": {
        "primary": "binance",
        "symbol": "BNBUSDT",
        "interval": "1h",
        "history_candles": 200,
        "refresh_interval_seconds": 60,
        "fallback": "geckoterminal"
    },
    "indicators": {
        "ema_fast": 20,
        "ema_slow": 50,
        "ema_trend": 200,
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2.0,
        "atr_period": 14,
        "hurst_max_lag": 20,
        "hurst_min_data_points": 100,
        "vpin_bucket_divisor": 50,
        "vpin_window": 50,
        "garch_omega": "0.00001",
        "garch_alpha": "0.1",
        "garch_beta": "0.85"
    },
    "signal_sources": {
        "tier_1": {
            "technical_indicators": { "enabled": true, "weight": "0.25" },
            "order_book_imbalance": { "enabled": true, "weight": "0.30", "depth_levels": 20 },
            "vpin": { "enabled": true, "weight": "0.20", "trade_lookback": 1000 }
        },
        "tier_2": {
            "btc_volatility_spillover": { "enabled": true, "weight": "0.10", "btc_symbol": "BTCUSDT", "lookback_hours": 24 },
            "liquidation_heatmap": { "enabled": true, "weight": "0.10", "aave_subgraph_url": "", "max_hf_query": "2.0" },
            "exchange_flows": { "enabled": true, "weight": "0.08", "flow_window_minutes": 60 },
            "funding_rate": { "enabled": true, "weight": "0.07", "extreme_threshold": "0.0005" }
        },
        "tier_3": {
            "aggregate_mempool_flow": { "enabled": false, "weight": "0.05", "window_minutes": 15, "note": "Disabled by default — limited visibility due to BSC PBS. Enable for experimentation." },
            "social_sentiment_volume": { "enabled": false, "weight": "0.03", "note": "Disabled by default — requires external sentiment API." }
        }
    },
    "entry_rules": {
        "min_confidence": "0.7",
        "require_trend_alignment": true,
        "require_volume_confirmation": false,
        "max_signals_per_day": 3,
        "regime_filter": {
            "enabled": true,
            "trending_hurst_threshold": "0.55",
            "mean_reverting_hurst_threshold": "0.45",
            "min_atr_ratio": "1.0",
            "max_atr_ratio": "3.0"
        },
        "regime_weight_multipliers": {
            "trending": { "momentum_signals": "1.2", "mean_reversion_signals": "0.5" },
            "mean_reverting": { "momentum_signals": "0.5", "mean_reversion_signals": "1.2" },
            "volatile": { "all_signals": "0.7" },
            "ranging": { "all_signals": "0.8" }
        },
        "agreement_bonus_threshold": "0.7",
        "agreement_bonus_multiplier": "1.15",
        "max_signal_age_seconds": 120
    },
    "position_sizing": {
        "method": "fractional_kelly",
        "kelly_fraction": "0.25",
        "high_vol_threshold": "0.04",
        "drawdown_reduction_start": "0.10",
        "min_position_usd": "100",
        "rolling_edge_window_days": 30
    },
    "alpha_decay_monitoring": {
        "enabled": true,
        "accuracy_decay_threshold": "0.7",
        "sharpe_decay_threshold": "0.5",
        "confidence_boost_on_decay": "1.1",
        "rolling_window_days": 30,
        "historical_window_days": 180,
        "note": "Cong et al. (2024): crypto strategy alpha decays with ~12-month half-life"
    },
    "exit_rules": {
        "take_profit_percent": "5.0",
        "stop_loss_percent": "3.0",
        "trailing_stop_percent": "2.0",
        "max_hold_hours": 168
    },
    "short_signals": {
        "enabled": true,
        "preferred_collateral": "USDC"
    }
}
```

**Academic justification for indicator parameters**: Standard TA parameters (EMA 20/50/200, RSI 14, MACD 12/26/9) prevent overfitting — Hudson & Urquhart (2019) tested ~15,000 rules and found standard configurations show "significant predictability." The confidence threshold (0.7) is from MDPI (2025) — 82.68% accuracy at 12% coverage. OBI weight (0.30) reflects Kolm et al. (2023) — 73% of prediction performance. GARCH(1,1) parameters (α=0.1, β=0.85) follow Hansen & Lunde (2005) benchmark. VPIN bucket divisor (50) follows Easley et al. (2012) recommendation. Fractional Kelly (25%) follows MacLean et al. (2010) for controlled growth with bounded drawdown. Hurst thresholds (0.45/0.55) follow Maraj-Mervar & Aybar (2025) FracTime framework.

### `config/timing.json` (MODIFIED)
```json
{
    "health_monitoring": {
        "safe_interval_seconds": 15,
        "watch_interval_seconds": 5,
        "warning_interval_seconds": 2,
        "critical_interval_seconds": 1,
        "stale_data_threshold_failures": 5
    },
    "aggregator": {
        "quote_timeout_seconds": 5,
        "quote_cache_ttl_seconds": 10
    },
    "transaction": {
        "confirmation_timeout_seconds": 60,
        "simulation_timeout_seconds": 15,
        "nonce_refresh_interval_seconds": 30
    },
    "web3_connection": {
        "max_retries": 5,
        "retry_delay_seconds": 2.0
    },
    "error_recovery": {
        "rpc_retry_base_delay_seconds": 1,
        "rpc_retry_max_delay_seconds": 30
    }
}
```

### `config/app.json` (MODIFIED)
```json
{
    "precision": { "decimal_precision": 78 },
    "timezone": { "app_timezone": "UTC" },
    "logging": {
        "log_dir": "logs",
        "module_folders": {
            "health_monitor": "Health_Monitor_Logs",
            "strategy": "Strategy_Logs",
            "signal_engine": "Signal_Engine_Logs",
            "data_service": "Data_Service_Logs",
            "position_manager": "Position_Manager_Logs",
            "pnl_tracker": "PnL_Tracker_Logs",
            "aggregator": "Aggregator_Logs",
            "aave_client": "Aave_Client_Logs",
            "tx_submitter": "TX_Submitter_Logs",
            "safety": "Safety_Logs",
            "deep_dive": "Deep_Dive_Logs"
        }
    }
}
```

### `.env.example` (MODIFIED)
```
# BSC RPC
BSC_RPC_URL_HTTP=https://bsc-dataseed1.binance.org/
BSC_RPC_URL_HTTP_FALLBACK=https://bsc-dataseed2.binance.org/
BSC_RPC_URL_MEV_PROTECTED=https://rpc.48.club

# Execution
LEVERAGE_EXECUTOR_ADDRESS=
EXECUTOR_PRIVATE_KEY=
EXECUTOR_DRY_RUN=true

# Aave V3
AAVE_V3_POOL_ADDRESS=0x6807dc923806fE8Fd134338EABCA509979a7e0cB

# Position Management
MAX_LEVERAGE_RATIO=3.0
MIN_HEALTH_FACTOR=1.5
MAX_FLASH_LOAN_USD=5000
MAX_GAS_PRICE_GWEI=10

# Aggregator
ONEINCH_API_KEY=

# Data Service (for signal engine)
BINANCE_API_KEY=
BINANCE_API_SECRET=

# User Wallet
USER_WALLET_ADDRESS=
```

---

## Data Flow

```
main.py
  │
  ├── health_monitor.run()          ◄── asyncio task (runs forever)
  │     │
  │     ├── aave_client.get_user_account_data()  ◄── RPC poll (tiered interval)
  │     ├── check_oracle_freshness()             ◄── validate Chainlink updatedAt
  │     ├── get_borrow_rate()                    ◄── monitor utilization + rate
  │     │
  │     └── signal_queue.put(HealthStatus)
  │
  ├── signal_engine.run()           ◄── asyncio task (5-layer pipeline, runs forever)
  │     │
  │     │  ┌─── LAYER 1: Regime Detection ───────────────────────────────────┐
  │     ├── data_service.get_ohlcv("BNBUSDT")   ◄── Binance klines (cached)
  │     ├── indicators.hurst_exponent(prices)    ◄── R/S analysis → H value
  │     ├── _detect_regime(indicators)           ◄── TRENDING / MEAN_REVERTING / RANGING / VOLATILE
  │     │  └─────────────────────────────────────────────────────────────────┘
  │     │
  │     │  ┌─── LAYER 2: Multi-Source Signal Collection ─────────────────────┐
  │     ├── Tier 1 (parallel):
  │     │     ├── indicators.compute_all(candles)           ◄── EMA, RSI, MACD, BB, ATR
  │     │     ├── _compute_order_book_imbalance()           ◄── Binance depth API → OBI
  │     │     └── _compute_vpin()                           ◄── Binance aggTrades → VPIN
  │     │
  │     ├── Tier 2 (parallel):
  │     │     ├── _compute_btc_volatility_spillover()       ◄── BTC vs BNB realized vol
  │     │     ├── _compute_liquidation_heatmap()            ◄── Aave V3 HF distribution
  │     │     ├── _compute_exchange_flows()                 ◄── USDT flow to/from exchanges
  │     │     └── _compute_funding_rate_signal()            ◄── Binance perp funding rate
  │     │
  │     ├── Tier 3 (if enabled):
  │     │     ├── _compute_aggregate_mempool_flow()         ◄── BSC txpool (limited visibility)
  │     │     └── (social sentiment volume — disabled default)
  │     │  └─────────────────────────────────────────────────────────────────┘
  │     │
  │     │  ┌─── LAYER 3: Ensemble Confidence ────────────────────────────────┐
  │     ├── _compute_ensemble_confidence(components, regime) ◄── weighted + regime-adjusted
  │     │     ├── regime_weight_multipliers applied          ◄── boost aligned, suppress misaligned
  │     │     └── agreement_bonus (>70% consensus → +15%)   ◄── multi-source confirmation
  │     │  └─────────────────────────────────────────────────────────────────┘
  │     │
  │     │  ┌─── LAYER 4: Position Sizing ────────────────────────────────────┐
  │     ├── indicators.garch_volatility(returns)             ◄── GARCH(1,1) σ² forecast
  │     ├── _compute_kelly_fraction(confidence, volatility)  ◄── f* = edge/σ² × 0.25
  │     ├── _compute_position_size(kelly_f, equity)          ◄── USD amount, capped
  │     │  └─────────────────────────────────────────────────────────────────┘
  │     │
  │     ├── _check_alpha_decay(signal_history)               ◄── rolling accuracy check
  │     │
  │     └── [if confidence > threshold AND regime OK AND no alpha decay]
  │           signal_queue.put(TradeSignal)                   ◄── direction + confidence + size + components
  │
  └── strategy.run(signal_queue)    ◄── asyncio task (Layer 5: Risk Management, runs forever)
        │
        ├── signal_queue.get()  →  HealthStatus OR TradeSignal
        │
        ├── [if TradeSignal with confidence > min_confidence]
        │     │
        │     ├── validate_position_size(signal)              ◄── GARCH vol adjustment, drawdown reduction
        │     ├── check_borrow_rate_acceptable()              ◄── reject if APR too high
        │     ├── stress_test(direction, ...)                  ◄── direction-aware formula (long vs short)
        │     ├── stress_test_with_cascade(...)                ◄── cascade multiplier if >$50M liquidatable
        │     ├── check_close_factor_risk(...)                 ◄── warn if position risks 100% close factor
        │     ├── check_strategy_health()                      ◄── alpha decay → raise threshold
        │     ├── safety.can_open_position()                   ◄── kill switches, cooldown, gas check
        │     │
        │     └── position_manager.open_position(direction, debt_token, validated_amount, collateral_token)
        │           │
        │           ├── [if SHORT] _check_isolation_mode(collateral_token)
        │           ├── aggregator_client.get_best_quote()   ◄── parallel fan-out to 1inch/OO/PS
        │           │     └── check_dex_oracle_divergence()  ◄── reject if >1% divergence
        │           │     └── SwapQuote { calldata, router, to_amount_min }
        │           │
        │           ├── aave_client.encode_flash_loan()      ◄── local encoding
        │           │     └── tx calldata (to: LeverageExecutor)
        │           │
        │           ├── tx_submitter.simulate(tx)            ◄── eth_call
        │           │
        │           ├── [if !dry_run]
        │           │     tx_submitter.submit_and_wait(tx)    ◄── MEV-protected RPC
        │           │
        │           ├── aave_client.get_user_account_data()  ◄── verify post-tx state
        │           │
        │           └── pnl_tracker.record_open(position)    ◄── persist to SQLite
        │
        ├── [if HealthStatus with HF crossed deleverage/close threshold]
        │     │
        │     └── position_manager.deleverage() / close()
        │           └── pnl_tracker.record_close/deleverage()
        │
        └── [else: log status, snapshot to pnl_tracker, continue]
```

---

## Modifications to Existing Files

### `config/loader.py` — Remove/Add Methods

**Remove** (arb-era methods that have no corresponding config files in revised architecture):
- `get_protocol_config()` — protocols/ directory deleted
- `get_redis_channels()` — redis_channels.json deleted
- `get_websocket_config()` — websocket.json deleted
- `get_gas_config()` — gas.json not needed
- `get_cache_config()` — no caching layer
- `get_mev_config()` — no MEV bundle submission
- `get_optimizer_config()` — no split optimizer
- `get_multiprocessing_config()` — no multiprocessing
- `get_verification_config()` — no verification pipeline
- Module-level functions: `get_channel()`, `get_signal()`, `get_key()`

**Add**:
- `get_aave_config()` → loads `config/aave.json`
- `get_aggregator_config()` → loads `config/aggregator.json`
- `get_signals_config()` → loads `config/signals.json`

**Keep** (already exist and still needed):
- `get_positions_config()` — already exists, loads `config/positions.json`
- `get_chain_config()`, `get_app_config()`, `get_timing_config()`, `get_rate_limit_config()`
- `get_abi()`, `get_config_file()`, `clear_cache()`, `get_config()`

### `bot_logging/logger_manager.py` — Update Module Folders

Update the `_MODULE_FOLDERS` default dict to match the new `app.json` module_folders (remove event_streamer, pool_managers, venue_aggregator, split_optimizer, leverage_engine, liquidation_monitor, verifier, executor, bundle_submitter, gas_oracle; add health_monitor, aggregator, aave_client, tx_submitter, strategy, signal_engine, data_service, pnl_tracker).

---

## Build Phases

### Phase 0: Cleanup and Foundation
- Delete dead files (`data/`, `gas/`, `verification/`, `shared/math/`, `config/redis_channels.json`, `config/websocket.json`)
- Create `pyproject.toml` with dependencies (add `python-binance` for data_service)
- Create empty `__init__.py` files for new packages
- Update `.env.example` and `.gitignore`

### Phase 1: Config Layer
- Modify `config/loader.py` (remove arb methods, add leverage methods including `get_signals_config()`)
- Create `config/aave.json`, `config/positions.json`, `config/aggregator.json`, `config/signals.json`
- Modify `config/chains/56.json` (fix block_time to 0.75s, set mev_protected_url to 48 Club)
- Modify `config/timing.json`, `config/rate_limits.json`, `config/app.json`
- Create `config/validate.py`
- Create ABI files in `config/abis/`
- Modify `bot_logging/logger_manager.py` (update module folders)
- Create `shared/types.py` (include `PositionDirection`, `TradeSignal`, `OHLCV`, `IndicatorSnapshot`, `MarketRegime`, `BorrowRateInfo`, `RealizedPnL`, `TradingStats`) and `shared/constants.py`

### Phase 2: Aave Client
- Create ABI JSON files for Aave contracts
- Implement `execution/aave_client.py` (include `get_reserve_data()` for isolation mode + borrow rate checks)
- Write `tests/unit/test_aave_client.py`

### Phase 3: Safety and Health Monitor
- Implement `core/safety.py`
- Implement `core/health_monitor.py` (include oracle freshness validation, compound interest HF prediction, borrow rate monitoring)
- Write `tests/unit/test_safety.py` and `tests/unit/test_health_monitor.py`

### Phase 4: Aggregator Client
- Implement `execution/aggregator_client.py` (parallel fan-out to all 3 providers, best-quote selection, DEX-Oracle divergence check)
- Write `tests/unit/test_aggregator_client.py`

### Phase 5: Transaction Submitter
- Implement `execution/tx_submitter.py` (include nonce management with asyncio.Lock, stuck tx replacement, crash recovery)
- Write unit tests for revert decoding, nonce management, and tx replacement

### Phase 6: Smart Contract
- Create `contracts/foundry.toml`, install OpenZeppelin
- Create interfaces: `IFlashLoanReceiver.sol`, `IAaveV3Pool.sol`
- Implement `LeverageExecutor.sol` (verify both long and short flows work with same functions)
- Write `contracts/test/LeverageExecutor.t.sol` (Foundry fork tests — **test both long AND short position flows**)
- Generate ABI → `config/abis/leverage_executor.json`

### Phase 7: Position Manager and Strategy
- Implement `core/position_manager.py` (direction-aware: long + short, isolation mode check for shorts)
- Implement `core/strategy.py` (direction-aware stress tests, cascade modeling, close factor risk check, borrow rate cost check)
- Implement `core/pnl_tracker.py` (SQLite schema creation, P&L computation, interest accrual tracking)
- Write `tests/unit/test_position_manager.py`, `tests/unit/test_strategy.py`, `tests/unit/test_pnl_tracker.py`

### Phase 7.5: Signal Engine, Data Service, and Multi-Source Pipeline
- Implement `core/data_service.py` (Binance klines, depth, aggTrades, funding rate; Aave subgraph for liquidation levels; exchange flow proxy; mempool aggregation; caching per data type)
- Implement `core/indicators.py` (EMA, RSI, MACD, BB, ATR + Hurst exponent, GARCH(1,1), VPIN, OBI)
- Implement `core/signal_engine.py` (5-layer architecture: Hurst regime detection → multi-source signal collection with Tier 1/2/3 → ensemble confidence with regime-adaptive weights → fractional Kelly position sizing → alpha decay monitoring)
- Write `tests/unit/test_data_service.py` (order book, aggTrades, liquidation levels, exchange flows mocks)
- Write `tests/unit/test_indicators.py` (Hurst vs known series, GARCH convergence, VPIN computation, OBI)
- Write `tests/unit/test_signal_engine.py` (ensemble confidence scoring, regime multipliers, Kelly sizing, alpha decay detection, agreement bonus)

### Phase 8: Main Entrypoint and Integration
- Implement `main.py` (launch **3 asyncio tasks**: health_monitor, signal_engine, strategy)
- Wire signal_queue to accept both `HealthStatus` and `TradeSignal`
- Write `tests/integration/test_aave_fork.py` (test both long and short position lifecycle)
- Write `tests/conftest.py` and `tests/unit/test_config_loader.py`

### Phase 9: Hardening
- `mypy --strict`, `ruff`, `black` on all files
- Structured logging in every module
- Review all error paths
- Verify oracle freshness checks work end-to-end
- Verify borrow rate monitoring prevents high-cost entries
- Verify stress test with cascade produces different results than without
- End-to-end dry-run testing against BSC mainnet (both long and short signal → position lifecycle)

### Phase 10: Mempool Order Flow Enhancement (Rust + Python)

See `Mempool_Enhancement_Plan.md` for full specification.

**Phase 10A: Rust Mempool Decoder**
- Initialize Rust project (`cargo init mempool-decoder`)
- Implement WebSocket connection + `newPendingTransactions` subscription
- Implement router matching (12 routers) and selector matching (26 selectors)
- Implement V2, V3, SmartRouter, Universal Router, and aggregator ABI decoding
- Implement buy/sell classification and USD estimation
- Implement deduplication (LRU hash set, 100k capacity) and poison detection scoring
- Implement Redis publishing (`mempool:decoded_swaps`)
- Write unit tests using real BSC calldata captured from ArbitrageTestBot logs

**Phase 10B: Rust Rolling Aggregator**
- Implement sliding window data structure (1m, 5m, 15m)
- Implement per-token-pair aggregation (WBNB, BTCB, ETH): net flow, direction score, volume acceleration, whale detection
- Implement Redis aggregate signal publishing every 5 seconds (`mempool:aggregate_signal`)
- Write unit tests with synthetic swap streams

**Phase 10C: Python Integration**
- Add `redis` dependency to LeverageBot
- Add `MempoolTokenSignal` and `MempoolSignal` types to `shared/types.py`
- Create `config/mempool.json`
- Replace `get_pending_swap_volume()` in `core/data_service.py` with Redis-backed `get_mempool_signal()`
- Replace `_compute_aggregate_mempool_flow()` in `core/signal_engine.py` with real signal component
- Promote mempool signal from Tier 3 to Tier 2 in `config/signals.json` (weight 0.12)
- Update `main.py` to initialize Redis connection
- Write `tests/unit/test_mempool_signal.py`
- Run `mypy --strict`, `ruff`, `black`, full test suite

**Phase 10D: End-to-End Validation**
- Run Rust decoder against BSC mainnet WebSocket
- Verify decoded output matches ArbitrageTestBot for same transactions
- Run LeverageBot in dry-run mode consuming live mempool signal
- Compare signal engine output with and without mempool signal enabled

---

## Testing Strategy

### Unit Tests (mocked, fast, no network)
| Test File | Tests |
|-----------|-------|
| `test_health_monitor.py` | Tier transitions, poll intervals, stale data detection, **oracle freshness validation**, **compound interest HF prediction**, borrow rate monitoring |
| `test_position_manager.py` | Open/close/deleverage flows with mocked deps; **both long AND short flows**; calldata verification; **isolation mode rejection for USDT shorts** |
| `test_aggregator_client.py` | Mock HTTP for all 3 providers; **parallel fan-out best-quote selection**; rate limiting; minimum output; **DEX-Oracle divergence rejection** |
| `test_strategy.py` | **Direction-aware stress test (long formula vs short formula)**; **cascade multiplier**; **close factor risk check**; **borrow rate cost check**; **validate_position_size with GARCH adjustment**; **drawdown-based position reduction**; **check_strategy_health and alpha decay**; deleverage amount formula; tier-based dispatch |
| `test_safety.py` | Kill switches; dry-run; default-to-safe; cooldown; pause sentinel |
| `test_aave_client.py` | WAD/RAY conversion; data parsing; calldata encoding; **reserve data with isolation mode fields** |
| `test_config_loader.py` | Config loading; env var overrides; missing file handling; **signals.json loading** |
| `test_signal_engine.py` | **5-layer pipeline end-to-end**; **ensemble confidence scoring with regime weights**; **Hurst regime detection (trending/mean-reverting/ranging/volatile)**; **OBI signal component**; **VPIN signal component**; **BTC volatility spillover**; **funding rate contrarian signal**; **agreement bonus (>70% consensus)**; **fractional Kelly position sizing**; **alpha decay detection and threshold adjustment**; **regime-adaptive weight multipliers** |
| `test_indicators.py` | **EMA computation vs known values**; **RSI Wilder's method**; **MACD line/signal/histogram**; **Bollinger Band width**; **ATR calculation**; **Hurst exponent vs known persistent/antipersistent series**; **GARCH(1,1) convergence and stationarity**; **VPIN volume bucketing and classification**; **OBI symmetry** |
| `test_data_service.py` | **Mock Binance klines, depth, aggTrades responses**; **OHLCV parsing**; **OrderBookSnapshot parsing**; **cache TTL per data type**; **fallback to GeckoTerminal**; **liquidation level computation from mock Aave subgraph**; **exchange flow proxy** |
| `test_pnl_tracker.py` | **SQLite table creation**; **record_open/record_close lifecycle**; **unrealized P&L for long vs short**; **accrued interest calculation**; **summary stats** |
| `test_mempool_signal.py` | **Mock Redis consumer**; **direction score computation from aggregated data**; **volume acceleration threshold**; **whale alignment bonus**; **poison ratio penalty**; **stale data handling (>30s returns neutral)**; **integration with ensemble scoring** |

### Integration Tests
| Test File | Tests |
|-----------|-------|
| `test_aave_fork.py` | Full stack against Anvil BSC fork: **open long**, check HF, deleverage, close; **open short**, check HF, deleverage, close |

### Smart Contract Tests (Foundry)
| Test File | Tests |
|-----------|-------|
| `LeverageExecutor.t.sol` | All functions vs BSC fork; access control; router whitelist; slippage; **long flow** (USDT→WBNB); **short flow** (WBNB→USDT); close both directions |

### Verification
Run `pytest tests/` — all pass. Run `forge test --fork-url $BSC_RPC_URL_HTTP -vvv` — all pass. Run `python main.py` in dry-run mode — polls HF, computes signals, logs tiers + indicators + confidence, no transactions submitted.

---

## What Was Removed and Why

| Removed Component | Lines Saved | Replacement | Academic Justification |
|---|---|---|---|
| 9 pool managers | ~15,000+ | Aggregator APIs (parallel fan-out) | Angeris et al. (2022): routing is convex optimization; aggregators solve it. Diamandis et al. (2023): efficient algorithm scales linearly. |
| 5 AMM math libraries | ~3,000+ | None | Bot does not simulate swaps |
| Event streamer (DEX) | ~850 | Polling | Chainlink heartbeat 27-60s; polling sufficient |
| Split optimizer | ~1,500+ | Aggregator APIs | 1inch Pathfinder splits across 5-20 micro-steps across 400+ sources |
| Venue aggregator | ~500+ | None | No venues to aggregate |
| Bundle submitter | ~500+ | Standard tx submission + MEV-protected RPC | No MEV competition (Daian et al., 2020); 48 Club Privacy RPC for protection |
| Redis pub/sub (30+ ch) | ~200+ | asyncio.Queue | Single process; no IPC needed |
| Multiprocess (15-20) | ~500+ | Single asyncio | Position management is not CPU-bound |
| Mempool monitoring (real-time MEV) | N/A (never built) | Aggregate order flow signal via Rust decoder (Phase 10) | Real-time MEV exploitation rejected (BSC 99.8% PBS, 750ms blocks). Replaced with aggregate directional bias analysis — not frontrunning, but order flow signal from pending swaps. |

## What Was Added (Post-Audit) and Why

| Added Component | Lines | Purpose | Academic Justification |
|---|---|---|---|
| **Phase 2 (Audit Response):** | | | |
| Signal engine + indicators + data service | ~550-700 | Entry signal generation (WHEN to enter) | Hudson & Urquhart (2019): technical rules have significant predictability in crypto; MDPI (2025): confidence threshold at 12% coverage maximizes profitability |
| Short position support | ~100 (modifications) | Inverse leverage positions | Heimbach & Huang (2024): long/short leverage negatively correlated; same contract handles both |
| Direction-aware stress tests | ~60 (modifications) | Correct HF formula for shorts | Perez et al. (2021): 3% price changes can trigger >$10M liquidations |
| Cascade multiplier | ~30 | Model liquidation feedback loops | OECD (2023): liquidations boost volatility; Klages-Mundt & Minca (2019): deleveraging spirals |
| Oracle freshness check | ~20 | Prevent stale price decisions | Deng et al. (ICSE 2024): ad-hoc oracle checks are insufficient |
| Borrow rate monitoring | ~40 | Prevent high-cost position entry | arXiv:2502.19862: rate spikes during stress trap leveraged positions |
| P&L tracker | ~200-250 | Position history and profitability | BIS WP 1171: wallet-level P&L methodology |
| DEX-Oracle divergence | ~20 | Prevent inflated HF from price divergence | Chainlink vs DEX price divergence during flash crashes |
| Nonce management | ~40 | Prevent stuck/dropped transactions | BSC 0.75s block time requires robust nonce handling |
| Close factor modeling | ~20 | Warn about 100% liquidation risk at small sizes | Aave V3.3: <$2K positions liquidatable in single call |
| **Phase 3 (Deep Investigation — Signal Intelligence):** | | | |
| 5-layer signal architecture | ~200 (expansion) | Replace simple TA with multi-source ensemble | Kolm et al. (2023): OBI 73% of prediction; ensemble methods outperform single-source |
| Order book imbalance (OBI) | ~40 | Dominant short-term price predictor | Kolm et al. (2023): OBI accounts for 73% of prediction performance |
| VPIN computation | ~50 | Detect informed trading / price jump probability | Easley et al. (2012): VPIN framework; Abad & Yagüe (2025): significant crypto prediction |
| BTC volatility spillover | ~30 | Early warning for BNB volatility shifts | DCC-GARCH literature: BTC is net volatility transmitter to altcoins |
| Liquidation heatmap | ~40 | Identify cascade-risk price levels | Perez et al. (2021): 3% moves can create >$10M liquidations; liquidation walls as price magnets |
| Exchange flow monitoring | ~30 | Directional bias from capital flows | Chi et al. (2024): USDT inflows positively predict returns |
| Funding rate signal | ~25 | Contrarian signal from leverage imbalance | Aloosh & Bekaert (2022): 12.5% price variation explained over 7 days |
| Aggregate mempool flow (Tier 3 stub) | ~25 | Medium-term volume/momentum bias | Ante & Saggu (2024): mempool predicts volume, not direction |
| **Phase 10 (Mempool Order Flow Enhancement):** | | | |
| Rust mempool decoder | ~1,500 (Rust) | WebSocket listener, ABI decoder for 26 swap selectors across 12 routers, dedup, poison detection | ArbitrageTestBot confirmed 2,428+ swaps visible on BSC; Kolm et al. (2023): order flow = 73% of prediction |
| Rust rolling aggregator | ~500 (Rust) | Sliding window stats (1m/5m/15m), direction scoring, whale detection, volume acceleration | Chi et al. (2024): exchange flows predict returns; aggregate volume bias is directional |
| Python mempool signal integration | ~100 | Redis consumer, signal component, Tier 2 promotion (weight 0.12) | Cont et al. (2014): order flow price impact theory |
| `config/mempool.json` | ~40 | Mempool decoder/aggregator configuration | — |
| `MempoolSignal` types | ~30 | Typed data structures for aggregated mempool data | — |
| Hurst exponent regime detection | ~30 | Adaptive strategy selection by market regime | Maraj-Mervar & Aybar (2025): Sharpe 2.10 vs 0.85 for static strategies |
| GARCH(1,1) volatility model | ~25 | Volatility-adjusted position sizing | Hansen & Lunde (2005): GARCH(1,1) difficult to beat; Bollerslev (1986) |
| Fractional Kelly position sizing | ~40 | Optimal growth rate with bounded drawdown | MacLean et al. (2010): fractional Kelly maximizes long-run growth |
| Alpha decay monitoring | ~35 | Detect strategy degradation, auto-adjust | Cong et al. (2024): crypto alpha decays ~12-month half-life |
| Regime-adaptive weight multipliers | ~20 | Boost/suppress signals based on regime | Lo (2004): AMH — efficiency is time-varying |
| Expanded data service endpoints | ~150 (expansion) | Support OBI, VPIN, flows, liquidations | Multiple sources needed for multi-source architecture |
| Expanded types (SignalComponent, etc.) | ~80 (expansion) | Data structures for signal pipeline | Required for type-safe signal composition |

---

## References

### DeFi Leverage and Liquidation
1. Perez, Werner, Xu, Livshits — "Liquidations: DeFi on a Knife-Edge" (FC 2021, LNCS vol. 12675) — [arXiv:2009.13235](https://arxiv.org/abs/2009.13235)
2. Heimbach & Huang — "DeFi Leverage" (BIS Working Paper No. 1171, 2024) — [PDF](https://www.bis.org/publ/work1171.pdf)
3. Klages-Mundt & Minca — "(In)Stability for the Blockchain: Deleveraging Spirals" (2019/2020) — [arXiv:1906.02152](https://arxiv.org/abs/1906.02152)
4. OECD — "DeFi Liquidations: Volatility and Liquidity" (2023)
5. "Locked In, Levered Up: Risk, Return, and Ruin in DeFi Lending" (British Accounting Review, 2025)
6. Lehar et al. — "Systemic Fragility in Decentralized Markets"
7. Tovanich et al. — "Contagion in Decentralized Lending Protocols" (DeFi '23)

### AMM Routing and Swap Execution
8. Angeris, Chitra, Evans, Boyd — "Optimal Routing for CFMMs" (ACM EC 2022) — [arXiv:2204.05238](https://arxiv.org/abs/2204.05238)
9. Diamandis, Resnick, Chitra, Angeris — "Efficient Algorithm for Optimal Routing Through CFMMs" (FC 2023) — [arXiv:2302.04938](https://arxiv.org/abs/2302.04938)
10. Milionis et al. — "Automated Market Making and Loss-Versus-Rebalancing" (LVR paper)

### MEV and Mempool
11. Daian et al. — "Flash Boys 2.0" (IEEE S&P 2020) — [arXiv:1904.05234](https://arxiv.org/abs/1904.05234)
12. Qin, Zhou, Gervais — "Quantifying Blockchain Extractable Value" (IEEE S&P 2022) — [arXiv:2101.05511](https://arxiv.org/abs/2101.05511)
13. Cartea & Sanchez-Betancourt — "Detecting Toxic Flow" (2023) — [arXiv:2312.05827](https://arxiv.org/abs/2312.05827)
14. BIS Bulletin No. 58 — "Miners as Intermediaries: Extractable Value and Market Manipulation"

### Trading Signals and Technical Analysis
15. Fang et al. — "Cryptocurrency Trading: A Comprehensive Survey" (Financial Innovation, 2022) — [Springer](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-021-00321-6)
16. Hudson & Urquhart — "Technical Trading and Cryptocurrencies" (Annals of Operations Research, 2019, Vol. 297) — [Springer](https://link.springer.com/article/10.1007/s10479-019-03357-1)
17. Hafid et al. — "Predicting Bitcoin Market Trends with Enhanced Technical Indicator Integration" (2024) — [arXiv:2410.06935](https://arxiv.org/html/2410.06935v1)
18. Beluska & Vojtko — "Revisiting Trend-following and Mean-Reversion in Bitcoin" (SSRN, 2024) — [PDF](https://papers.ssrn.com/sol3/Delivery.cfm/4955617.pdf?abstractid=4955617)
19. MDPI — "Confidence-Threshold Framework for Crypto Trading" (Applied Sciences 15(20):11145, 2025)

### Order Book and Microstructure (Phase 3)
20. Kolm, Turiel, Westray — "Deep Order Flow Imbalance: Extracting Alpha from the Limit Order Book" (Journal of Financial Economics, 2023) — Order book imbalance accounts for 73% of prediction performance
21. Easley, López de Prado, O'Hara — "Flow Toxicity and Liquidity in a High-Frequency World" (Review of Financial Studies, 2012) — VPIN framework
22. Abad & Yagüe — "VPIN as Predictor of Price Jumps in Cryptocurrency Markets" (ScienceDirect, 2025) — VPIN significantly predicts crypto price jumps
23. Cont, Kukanov, Stoikov — "The Price Impact of Order Book Events" (J. Financial Econometrics, 2014) — Theoretical foundation for order book imbalance as price predictor

### Volatility and Regime Detection (Phase 3)
24. Bollerslev — "Generalized Autoregressive Conditional Heteroskedasticity" (J. Econometrics, 1986) — GARCH(1,1) model
25. Hansen & Lunde — "A Forecast Comparison of Volatility Models: Does Anything Beat a GARCH(1,1)?" (J. Applied Econometrics, 2005) — Benchmark showing GARCH(1,1) is difficult to beat
26. Kim & Won — "Forecasting Stock Volatility Using GARCH-LSTM Hybrid" (Expert Systems with Applications, 2018) — Hybrid model outperforms standalone GARCH
27. Maraj-Mervar & Aybar — "Regime-Adaptive Trading via Hurst Exponent" (FracTime, 2025) — Sharpe 2.10 vs 0.85 for static strategies
28. Lo — "The Adaptive Markets Hypothesis" (J. Portfolio Management, 2004) — Market efficiency is time-varying; regime detection is essential
29. Timmermann & Granger — "Efficient Market Hypothesis and Forecasting" (International J. Forecasting, 2004) — Predictability varies across regimes

### Crypto Volatility Spillover (Phase 3)
30. Diebold & Yilmaz — "Better to Give than to Receive: Predictive Directional Measurement of Volatility Spillovers" (International J. Forecasting, 2012) — Spillover index methodology
31. DCC-GARCH volatility spillover literature — BTC as net volatility transmitter to altcoins; BNB as net receiver with 1-4 hour propagation lag

### Exchange Flows and On-Chain Signals (Phase 3)
32. Chi, White, Lee — "Cryptocurrency Exchange Flows and Returns" (SSRN, 2024) — USDT net inflows positively predict BTC/ETH returns; ETH net inflows negatively predict returns
33. Aloosh & Bekaert — "The Role of Funding Rates in Cryptocurrency Markets" (SSRN, 2022) — Funding rates explain 12.5% of price variation over 7-day horizons
34. Shen, Urquhart, Wang — "Does Twitter Predict Bitcoin?" (Economics Letters, 2019, Vol. 174) — Tweet volume more predictive than polarity for BTC returns

### Mempool, MEV, and Order Flow (Phase 3 + Phase 10)
35. Ante & Saggu — "Mempool Transaction Flow and Price Prediction" (Journal of Innovation & Knowledge, 2024) — Mempool predicts volume but NOT reliably price direction
36. Wahrstätter et al. — "Blockchain Censorship" (WWW 2023) — PBS adoption and mempool visibility constraints
54. BEP-322 — Builder API Specification for BNB Smart Chain — [GitHub](https://github.com/bnb-chain/BEPs/blob/master/BEPs/BEP322.md)
55. BNB Chain — "MEV Demystified" (2024) — [Blog](https://www.bnbchain.org/en/blog/mev-demystified-exploring-the-mev-landscape-in-the-bnb-chain-ecosystem) — 99.8% PBS adoption, Good Will Alliance >95% sandwich reduction
56. Qin, Zhou & Gervais — "Quantifying Blockchain Extractable Value" (IEEE S&P, 2022) — [arXiv:2101.05511](https://arxiv.org/abs/2101.05511) — $540M BEV over 32 months
57. Torres et al. — "Frontrunner Jones and the Raiders of the Dark Forest" (USENIX Security, 2021) — 200K frontrunning attacks, $18.4M profit
58. Cont, Kukanov & Stoikov — "The Price Impact of Order Book Events" (J. Financial Econometrics, 2014) — Theoretical foundation for order flow as price predictor
59. Bouri, Gupta & Roubaud — "Herding behaviour in cryptocurrencies" (Finance Research Letters, 2019) — Retail herding patterns support visible mempool flow as directional signal
60. Weintraub et al. — "A Flash(bot) in the Pan" (ACM IMC, 2022) — [arXiv:2206.04185](https://arxiv.org/abs/2206.04185) — >99.9% builder participation, MEV centralization
61. Oz et al. — "Who Wins Ethereum Block Building Auctions and Why?" (AFT, 2024) — [arXiv:2407.13931](https://arxiv.org/abs/2407.13931) — 3 builders produce 80% of blocks

### Position Sizing and Risk Management (Phase 3)
37. MacLean, Thorp, Ziemba — "Good and Bad Properties of the Kelly Criterion" (Quantitative Finance, 2010) — Fractional Kelly (25%) maximizes long-run growth with controlled drawdown
38. Kelly — "A New Interpretation of Information Rate" (Bell System Technical J., 1956) — Original Kelly criterion
39. Cong, Li, Tang, Yang — "Crypto Wash Trading and Alpha Decay" (Annual Review of Financial Economics, 2024) — Trading strategy alpha decays with ~12-month half-life; carry trade Sharpe from 6.45 to negative

### Alpha Decay and Strategy Lifecycle (Phase 3)
40. McLean & Pontiff — "Does Academic Research Destroy Stock Return Predictability?" (J. Finance, 2016) — Factor returns decay 32% post-publication; relevant to crypto alpha
41. Kosc, Sakowski, Ślepaczuk — "Momentum and Contrarian Effects on the Cryptocurrency Market" (Physica A, 2019) — Short-term contrarian outperforms momentum in cross-section

### Interest Rates and Oracle Security
42. "Optimal Risk-Aware Interest Rates for DeFi Lending Protocols" (2025) — [arXiv:2502.19862](https://arxiv.org/html/2502.19862v1)
43. "From Rules to Rewards: RL for Interest Rate Adjustment in DeFi Lending" (2025) — [arXiv:2506.00505](https://arxiv.org/html/2506.00505v1)
44. Deng et al. — "Safeguarding DeFi Smart Contracts against Oracle Deviations" (ICSE 2024)
45. "Oracles in DeFi: Attack Costs, Profits and Mitigation Measures" (Frontiers, 2023)

### Protocol Documentation
46. Aave V3 Technical Documentation — [aave.com/docs/aave-v3](https://aave.com/docs/aave-v3)
47. Aave V3 Interest Rate Strategy — [Smart Contracts Docs](https://aave.com/docs/aave-v3/smart-contracts/interest-rate-strategy)
48. Aave V3 Flash Loan Guide — [Flash Loans](https://aave.com/docs/aave-v3/guides/flash-loans)
49. Aave V3.3 Close Factor — [GitHub](https://github.com/aave-dao/aave-v3-origin/blob/main/docs/3.3/Aave-v3.3-features.md)
50. Aave V3 E-Mode — [Docs](https://aave.com/help/borrowing/e-mode)
51. Chaos Labs Risk Parameter Methodology — [PDF](https://chaoslabs.xyz/resources/chaos_aave_risk_param_methodology.pdf)
52. Chainlink Data Feeds BSC — Oracle heartbeat intervals, deviation thresholds
53. Chainlink SVR Documentation — [Docs](https://docs.chain.link/data-feeds/svr-feeds)
54. BNB Chain Maxwell Hard Fork (0.75s blocks) — [Blog](https://www.bnbchain.org/en/blog/bnb-chain-announces-maxwell-hardfork)
55. BNB Chain BEP-322 (PBS) — [GitHub](https://github.com/bnb-chain/BEPs/blob/master/BEPs/BEP322.md)
56. 48 Club Privacy RPC — [Docs](https://docs.48.club/privacy-rpc)
57. DeFi Saver Automation — Production leverage management architecture
58. 1inch Aggregation Protocol / Pathfinder — [Docs](https://1inch.network/aggregation-protocol/)
59. OpenOcean API V4 — [Docs](https://apis.openocean.finance/developer/apis/swap-api/api-v4)
60. ParaSwap API V5 — [Docs](https://developers.paraswap.network/api/master/api-v5)
