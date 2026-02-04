# BSC Leverage Bot Implementation Plan -- Comprehensive Audit Report

**Audit Date**: 2026-02-03
**Documents Audited**: `BSC_Leverage_Bot_Implementation_Plan.md`, `vast-humming-rocket.md` (identical content)
**Methodology**: Cross-referencing plan specifications against peer-reviewed academic literature, official protocol documentation, and production DeFi system architectures.

---

## Executive Summary

The implementation plan is architecturally sound for a **long-only, single-pair position management bot**. The decision to delegate swap routing to DEX aggregators is well-justified (Angeris et al., 2022). The tiered health factor polling, flash loan mechanics, and safety controls are correctly designed.

However, the audit identifies **7 critical gaps**, **6 important gaps**, and **4 enhancement opportunities** across five audit dimensions. The most significant finding is that the plan lacks entry signal generation entirely -- it manages positions but has no mechanism to decide WHEN to enter them. The second most significant finding is that the plan only covers long positions while the smart contract is already capable of both long and short.

### Severity Classification

| Severity | Count | Definition |
|----------|-------|------------|
| **CRITICAL** | 7 | Missing functionality that would cause financial loss or prevent the bot from operating as intended |
| **IMPORTANT** | 6 | Gaps that reduce robustness, accuracy, or operational visibility |
| **ENHANCEMENT** | 4 | Opportunities to improve performance or prepare for future expansion |

---

## Audit Dimension 1: Long and Short Position Support

### Finding 1.1 -- CRITICAL: Plan Only Describes Long Positions

The plan exclusively describes the long position flow:

```
LONG (lines 24-28):
  Flash loan USDT (mode=2) -> Swap USDT->WBNB -> Supply WBNB as collateral
  Result: Collateral=volatile (WBNB), Debt=stable (USDT)
```

The short position flow is the precise inverse:

```
SHORT (missing from plan):
  Flash loan WBNB (mode=2) -> Swap WBNB->USDT -> Supply USDT as collateral
  Result: Collateral=stable (USDT), Debt=volatile (WBNB)
```

**The smart contract (`LeverageExecutor.sol`) already supports both directions** -- its `openLeveragePosition(debtAsset, flashAmount, collateralAsset, ...)` signature is direction-agnostic. The gap is entirely in the Python bot code.

**What needs to change for short support:**

| Component | Change Required |
|-----------|----------------|
| `shared/types.py` | Add `PositionDirection` enum (`LONG`, `SHORT`) to `PositionState` |
| `core/strategy.py` | Stress test formula must use inverse formula for shorts (see 1.2) |
| `core/position_manager.py` | `open_position()` hardcodes `USDT->WBNB` swap direction; must accept configurable token roles |
| `config/aave.json` | Must verify USDT/USDC Isolation Mode status on-chain before opening shorts |

**References:**
- Aave V3 Flash Loan Documentation: https://aave.com/docs/aave-v3/guides/flash-loans
- Heimbach & Huang (2024), "DeFi Leverage," BIS Working Paper No. 1171 -- documents negative correlation between long/short wallet leverage
- DeFi Saver short position reference: https://help.defisaver.com/protocols/aave/how-can-i-long-or-short-assets-using-aave
- Cyfrin Updraft Aave V3 Course: unified `LongShort.sol` contract for both directions

### Finding 1.2 -- CRITICAL: Stress Test Formula Is Wrong for Short Positions

The plan's stress test (line 185-187):

```
HF_at_drop = (collateral_usd * (1 + price_change) * liquidation_threshold) / debt_usd
```

This formula assumes the **collateral** is the volatile asset (correct for longs, wrong for shorts).

**Correct formula for shorts** (collateral is stable, debt is volatile):

```
HF_short = (collateral_usd * liquidation_threshold) / (debt_usd * (1 + price_change))
```

The health factor response is **nonlinear** for shorts: `HF(P) = K/P` (convex function). A 20% price increase reduces HF by 16.7%, but a 50% increase reduces HF by 33.3%. This convexity means shorts face accelerating danger during price pumps.

**The deleverage formula** (line 189-190) is algebraically valid for both directions, but the **swap direction must reverse**: short deleverage withdraws USDT collateral, swaps USDT->WBNB, repays WBNB debt.

**References:**
- Perez et al. (2021), "Liquidations: DeFi on a Knife-Edge," FC 2021, LNCS vol. 12675

### Finding 1.3 -- IMPORTANT: USDT Isolation Mode Risk for Short Positions

USDT may be restricted to **Isolation Mode** on Aave V3 BSC, which imposes:
- A debt ceiling (maximum borrowing against USDT collateral)
- Restriction to borrowing only governance-approved assets
- Cannot add other collateral types alongside isolated USDT

**USDC is generally preferable as short collateral**: LTV=77% vs 75%, LT=80% vs 78%, and fewer governance restrictions.

The bot must call `getReserveData()` to check isolation mode status before opening any short position with USDT collateral.

**References:**
- Aave V3 Isolation Mode: https://aave.com/docs/aave-v3/overview
- Aave Risk Parameters: https://docs.aave.com/risk/asset-risk/risk-parameters

### Finding 1.4 -- IMPORTANT: Health Factor Asymmetry Between Long and Short

| Risk Factor | Long Position | Short Position |
|------------|---------------|----------------|
| Cascading liquidation risk | **HIGH** -- selling volatile collateral into falling market creates feedback loop | **LOW** -- selling stablecoin does not cascade |
| Self-reinforcing leverage | Decreases as prices rise (self-correcting) | Increases as prices rise (self-reinforcing danger) |
| Tail risk | Bounded at zero (collateral can't go negative) | Theoretically unbounded (debt asset can rise indefinitely) |
| Interest rate | Stablecoin borrow rates typically 3-8% APR | Volatile asset borrow rates variable, can spike |

**References:**
- OECD (2023), "DeFi Liquidations: Volatility and Liquidity"
- Klages-Mundt & Minca (2019/2020), "(In)Stability for the Blockchain: Deleveraging Spirals and Stablecoin Attacks"

---

## Audit Dimension 2: Optimal Exchange Path and Token Splitting

### Finding 2.1 -- CONFIRMED CORRECT: DEX Aggregator Delegation Is Optimal

The plan's decision to delegate swap routing to aggregator APIs rather than building custom routing is **strongly supported** by the academic literature.

Angeris et al. (ACM EC 2022, arXiv:2204.05238) proved that CFMM routing is a convex optimization problem. Diamandis et al. (FC 2023, arXiv:2302.04938) developed an efficient algorithm that scales linearly in the number of pools. These are exactly the algorithms that 1inch Pathfinder, OpenOcean, and ParaSwap implement internally -- exploring 400+ liquidity sources and splitting trades across 5-20 micro-steps.

Building custom routing would mean:
- Implementing convex optimization solvers for hundreds of BSC pools
- Maintaining pool state for every relevant DEX (PancakeSwap V2/V3, Biswap, DODO, Wombat, etc.)
- Continuous engineering to track new DEX deployments and liquidity migrations
- Strictly worse coverage than aggregators (your 7-8 DEXes vs their 400+ sources)

**References:**
- Angeris, Chitra, Evans, Boyd (2022), "Optimal Routing for CFMMs," arXiv:2204.05238
- Diamandis, Resnick, Chitra, Angeris (2023), "An Efficient Algorithm for Optimal Routing Through CFMMs," arXiv:2302.04938
- CFMMRouter.jl reference implementation: https://github.com/bcc-research/CFMMRouter.jl

### Finding 2.2 -- CONFIRMED CORRECT: WebSocket Is Not Required for Aggregator Quotes

**None of the three DEX aggregators offer WebSocket streaming of swap quotes.** The entire industry operates on REST request-response:

| Aggregator | Quote Interface | Response Time | WebSocket Available? |
|-----------|----------------|---------------|---------------------|
| 1inch | REST (Classic/Aggregation API v6) | <400ms | Only for Fusion resolver orders (irrelevant) |
| OpenOcean | REST (V4 API) | <150ms | Undocumented for quotes |
| ParaSwap | REST (V5 API) | <300ms | None |

The bot's execution pattern (swap only during position open/close/deleverage) means quotes are needed **on-demand**, not continuously. Quote staleness is a non-issue because:

1. Flash loan atomicity ensures the swap executes in the same block as the quote-based `minAmountOut` is evaluated
2. `minAmountOut` on-chain enforcement causes reverts (not losses) if conditions deteriorate
3. The quote-to-execution gap (1-3 BSC blocks) causes negligible price drift for liquid pairs (0.01-0.05%)

**References:**
- 1inch Developer Portal: https://business.1inch.com/portal/documentation/
- OpenOcean API V4: https://apis.openocean.finance/developer/apis/swap-api/api-v4
- Milionis et al., "Automated Market Making and Loss-Versus-Rebalancing" (LVR paper)

### Finding 2.3 -- ENHANCEMENT: Multi-Aggregator Fan-Out for Best Execution

The plan configures three aggregators with failover (lines 500-537). An improvement would be to **fan out quote requests to all three in parallel** and select the best quote, rather than sequential failover:

```python
async def get_best_quote(self, from_token, to_token, amount):
    quotes = await asyncio.gather(
        self._quote_1inch(from_token, to_token, amount),
        self._quote_openocean(from_token, to_token, amount),
        self._quote_paraswap(from_token, to_token, amount),
        return_exceptions=True
    )
    valid = [q for q in quotes if not isinstance(q, Exception)]
    return max(valid, key=lambda q: q.to_amount)
```

This provides best execution rather than defaulting to 1inch. 1inch has ~60% DEX aggregator market share but is not always the cheapest for every pair/size combination.

---

## Audit Dimension 3: Entry/Exit Signal Detection

### Finding 3.1 -- CRITICAL: No Entry Signal Generation Exists

This is the **most significant gap** in the plan. The `strategy.py` module (lines 168-191) defines position management logic (stress test, deleverage calculation, health factor response) but **contains no mechanism for deciding WHEN to enter a position**.

The data flow diagram (lines 622-656) confirms this: `strategy.run(signal_queue)` consumes only `HealthStatus` objects from `HealthMonitor`. These contain health factor data for an already-open position. There is no input pathway for market data that would drive initial entry decisions.

**How production systems handle this:** DeFi Saver, Instadapp, and Contango all require human-initiated position entry. They automate position *maintenance* (deleverage, boost, close), not position *opening*. This is a deliberate design choice -- automated entry into leveraged positions carries substantial risk.

**If automated entry is desired, the plan needs three new modules:**

| New Module | Lines (est.) | Purpose |
|-----------|-------------|---------|
| `core/data_service.py` | ~200-250 | Fetch historical OHLCV from Binance API + fallbacks |
| `core/indicators.py` | ~150-200 | Compute EMA, RSI, MACD, Bollinger Bands, ATR |
| `core/signal_engine.py` | ~200-250 | Generate confidence-scored entry/exit signals |
| `config/signals.json` | ~40 | Signal configuration (strategy mode, indicator params, thresholds) |

**Revised data flow with signal engine:**

```
main.py
  |
  +-- health_monitor.run()         <-- existing (unchanged)
  |     +-- signal_queue.put(HealthStatus)
  |
  +-- signal_engine.run()          <-- NEW asyncio task
  |     +-- data_service.get_ohlcv()       <-- Binance API (periodic)
  |     +-- indicators.compute_all()       <-- EMA, RSI, MACD, BB
  |     +-- generate_signal()              <-- Confidence-scored
  |     +-- signal_queue.put(TradeSignal)  <-- NEW signal type
  |
  +-- strategy.run(signal_queue)   <-- modified to handle both signal types
```

**References:**
- Fang et al. (2022), "Cryptocurrency Trading: A Comprehensive Survey," Financial Innovation (Springer) -- survey of 146 papers
- Hudson & Urquhart (2019), "Technical Trading and Cryptocurrencies," Annals of Operations Research, Vol. 297, pp. 191-220 -- 15,000 technical trading rules show significant predictability

### Finding 3.2 -- CRITICAL: Chainlink Alone Is Insufficient for Signal Generation

The plan references Chainlink feeds for price data, but Chainlink provides only:
- Latest spot price (single value)
- 27-60 second update frequency
- No volume data
- No OHLCV history
- No derived indicators

For any technical indicator computation, the bot needs at minimum:
- 200+ historical candles (for EMA(200))
- Volume data (for OBV, volume confirmation)
- High/Low data (for Bollinger Bands, ATR, support/resistance)

**Recommended data source:** Binance API (`python-binance` library). BNB is the primary collateral asset and Binance is the highest-liquidity venue for BNB/USDT. The klines endpoint is free (1200 req/min) and provides OHLCV at any timeframe.

### Finding 3.3 -- IMPORTANT: Recommended Initial Strategy Based on Academic Evidence

Based on Hudson & Urquhart (2019) showing significant predictability from technical rules in crypto, and the MDPI (2025) confidence-threshold framework achieving 82.68% direction accuracy at 12% market coverage, the recommended starting strategy is:

**Momentum + Confidence Threshold + Regime Filter:**

- **Long entry:** EMA(20) > EMA(50) AND price > EMA(200) AND RSI(14) in [50,70] AND MACD histogram > 0 AND confidence > 0.7
- **Short entry:** EMA(20) < EMA(50) AND price < EMA(200) AND RSI(14) in [30,50] AND MACD histogram < 0 AND confidence > 0.7
- **Regime filter:** ATR(14) within 1-3x its 50-period average (avoid extremely quiet or violent markets)

Key academic caution: "Strategies that appear effective in backtesting often fail in practical use" (Preprints.org survey of 234 papers, 2024). Use standard indicator parameters (EMA 20/50/200, RSI 14, MACD 12/26/9) to avoid overfitting.

**References:**
- Hudson & Urquhart (2019), Annals of Operations Research
- Hafid et al. (2024), "Predicting Bitcoin Market Trends with Enhanced Technical Indicator Integration," arXiv:2410.06935
- MDPI (2025), "Confidence-Threshold Framework," Applied Sciences 15(20):11145
- Beluska & Vojtko (2024), SSRN:4955617 -- 50/50 momentum/mean-reversion blend delivered Sharpe 1.71

---

## Audit Dimension 4: Mempool Monitoring for Price Prediction

### Finding 4.1 -- NOT RECOMMENDED: Mempool Monitoring Should Not Be Added

After thorough analysis, mempool monitoring is **not viable** for this leverage bot's use case. The conclusion is supported across all dimensions:

**BSC mempool visibility is declining rapidly:**
- 98% of BSC blocks now use PBS (BEP-322 Builder API) -- transactions flow through builders, not the public mempool
- BSC's Maxwell hard fork (June 2025) reduced block time to 0.75 seconds
- The Good Will Alliance achieved >90% reduction in sandwich attacks through builder-level filtering
- BSC's 2025 roadmap includes "directed mempools" that further reduce public visibility

**Latency budget is categorically insufficient:**

```
Bot's execution pipeline:
  Mempool detection:      ~50-400ms
  Strategy evaluation:    ~10-50ms
  Aggregator quote:       ~500-5000ms
  Tx construction:        ~50ms
  eth_call simulation:    ~200-500ms
  Tx submission:          ~100ms
  Block inclusion:        ~750-1500ms
  TOTAL:                  1.66-7.8 seconds (2-10 blocks)

A whale's swap is included in: ~0.75 seconds (1 block)
```

The price impact occurs before the bot can react. MEV sandwich bots operate in ~60ms (same block as target) -- the leverage bot cannot compete.

**Monitoring Chainlink `transmit()` transactions is also impractical:**
- The plan already subscribes to `AnswerUpdated` events in CRITICAL tier, which provides the same information post-confirmation
- Oracle front-running is a well-known MEV vector dominated by professional searchers
- Chainlink SVR (Smart Value Recapture) was created specifically to recapture this value

**Adding mempool monitoring would conflict with the plan's simplicity goals:**
- ~1,000-1,800 additional lines of code (+25-50%)
- External dependency on bloXroute ($300-1,250/month)
- WebSocket connection contradicting the polling-is-sufficient architecture
- Contradicts the "Victim" MEV posture (using mempool offensively while submitting via privacy RPC)

**References:**
- Daian et al. (2020), "Flash Boys 2.0," IEEE S&P 2020, arXiv:1904.05234
- Qin, Zhou & Gervais (2022), "Quantifying Blockchain Extractable Value," IEEE S&P 2022, arXiv:2101.05511
- BNB Chain Maxwell Hard Fork: https://www.bnbchain.org/en/blog/bnb-chain-announces-maxwell-hardfork
- BNB Chain BEP-322 Builder API: https://github.com/bnb-chain/BEPs/blob/master/BEPs/BEP322.md
- EigenPhi BSC Mempool Benchmark: https://eigenphi.substack.com/p/benchmarking-bnb-chain-mempool-tx-services

### Finding 4.2 -- CONFIRMED CORRECT: MEV-Protected RPC Is the Right Approach

The plan's "Victim" MEV posture (line 14) and MEV-protected RPC configuration are correct. The bot should:

1. Submit all transactions via 48 Club Privacy RPC (`https://rpc.48.club`) or equivalent
2. Rely on `minCollateralOut`/`minDebtTokenOut` for on-chain slippage protection
3. Not add offensive mempool capabilities

**References:**
- 48 Club Privacy RPC: https://docs.48.club/privacy-rpc
- BNB Chain MEV FAQ: https://www.bnbchain.org/en/blog/bnb-chain-mev-faqs

---

## Audit Dimension 5: Overlooked Components

### Finding 5.1 -- CRITICAL: No Borrow Rate Monitoring

The plan makes **no mention** of monitoring or factoring Aave V3 variable borrow rates into position decisions. This is a significant profitability risk.

Aave V3 uses a kinked interest rate model. When borrowing stablecoins (USDT/USDC) on BSC:
- Below optimal utilization (~80-90%): borrow rate ~5-9.5% APR
- Above optimal utilization: rate can spike to **60-300% APR**

A leveraged position opened at 5% APR that suddenly faces 100% APR due to utilization crossing the kink becomes unprofitable almost immediately.

**Required addition to `strategy.py`:**

```python
class BorrowRateMonitor:
    async def get_current_rate(self, asset) -> Decimal
    async def get_utilization(self, asset) -> Decimal
    def rate_headroom_to_kink(self, current_util, optimal_util) -> Decimal
    def projected_cost(self, debt_usd, rate_apr, duration_hours) -> Decimal
```

Entry decisions should reject positions where `projected_borrow_cost > expected_price_appreciation`.

**References:**
- Aave V3 Interest Rate Strategy: https://aave.com/docs/aave-v3/smart-contracts/interest-rate-strategy
- "Optimal Risk-Aware Interest Rates for DeFi Lending Protocols," arXiv:2502.19862
- "From Rules to Rewards: RL for Interest Rate Adjustment in DeFi Lending," arXiv:2506.00505

### Finding 5.2 -- CRITICAL: No Oracle Freshness Check

The plan subscribes to Chainlink events in CRITICAL tier but **never validates that the oracle data is fresh**. If Chainlink nodes fail to update within the heartbeat period (27s for BNB/USD), the on-chain price becomes stale and all health factor calculations are based on outdated data.

**Required addition to `health_monitor.py`:**

```python
async def check_oracle_freshness(self, feed_address, max_staleness_seconds=60):
    (roundId, answer, startedAt, updatedAt, answeredInRound) = \
        chainlink_feed.functions.latestRoundData().call()
    age = current_timestamp - updatedAt
    if age > max_staleness_seconds:
        logger.critical(f"Oracle stale: {age}s old (max {max_staleness_seconds}s)")
        self.safety.trigger_global_pause()
        return False
    return True
```

**References:**
- Deng et al. (2024), "Safeguarding DeFi Smart Contracts against Oracle Deviations," ICSE 2024
- Chainlink latestRoundData documentation

### Finding 5.3 -- CRITICAL: Liquidation Close Factor Creates Hidden Risk at Small Positions

Aave V3's close factor mechanics (from `LiquidationLogic.sol`):

| Condition | Max Liquidation per Call |
|-----------|------------------------|
| HF > 0.95 AND collateral >= $2,000 AND debt >= $2,000 | 50% of debt |
| HF <= 0.95 | **100% of debt** |
| Collateral or debt < $2,000 | **100% of debt** |

At the plan's $5,000-$10,000 position range with 3x leverage, a significant price drop can push collateral or debt below $2,000, enabling **complete single-call liquidation** with no partial-liquidation second chance. Combined with the 5% liquidation bonus, this represents a severe loss scenario that the stress test does not model.

**Recommendation:** Add a minimum position size floor and model the 100% close factor threshold in stress tests.

**References:**
- Aave V3.3 Close Factor: https://github.com/aave-dao/aave-v3-origin/blob/main/docs/3.3/Aave-v3.3-features.md
- Aave V3 Liquidation Documentation: https://docs.aave.com/developers/guides/liquidations

### Finding 5.4 -- IMPORTANT: No P&L Tracking or Position History

The bot opens and closes positions but **does not track profitability**. There is no mechanism to compute:

- Unrealized P&L (current collateral value - current debt value - initial capital)
- Realized P&L (tokens received after close - initial capital - gas - flash loan premiums)
- Accrued interest (from `variableDebtToken.scaledBalanceOf()` and `getPreviousIndex()`)

**Recommendation:** Add SQLite-based position tracking with three tables: `positions` (lifecycle), `position_snapshots` (periodic valuation), `transactions` (gas costs, success/failure).

**References:**
- BIS Working Paper No. 1171 (Heimbach & Huang, 2024) -- methodology for wallet-level P&L tracking

### Finding 5.5 -- IMPORTANT: Stress Test Ignores Liquidation Cascades

The plan's stress test formula is static -- it assumes a price drop happens once and stops. In reality, liquidation cascades create a feedback loop:

1. Price drops X% -> positions liquidated -> collateral sold -> additional Y% price drop -> more liquidations

Perez et al. (2021) showed that 3% price variations can make >$10M liquidatable. The OECD (2023) documented that "liquidations contribute to boosting price volatility during periods of stress."

**Recommendation:** Add a cascade multiplier to stress tests. If the initial price drop triggers >$50M in market-wide liquidations (queryable from Aave's total supply/borrow data), assume an additional 2-5% cascade-induced decline.

**References:**
- Perez et al. (2021), FC 2021
- OECD (2023), "DeFi Liquidations: Volatility and Liquidity"
- Lehar et al., "Systemic Fragility in Decentralized Markets"
- Klages-Mundt & Minca (2019), "(In)Stability for the Blockchain"

### Finding 5.6 -- IMPORTANT: Interest Accrual Model in HF Prediction Is Too Simplistic

`HealthMonitor.predict_hf_at(seconds_ahead)` uses "linear drift from borrow rate" (line 157). Aave V3 compounds interest **per second** using a Taylor series approximation in `MathUtils.calculateCompoundedInterest()`. For prediction horizons beyond ~10 minutes, the linear approximation diverges meaningfully from compound interest.

**References:**
- Aave V3 MathUtils.sol: https://github.com/aave/aave-v3-core/blob/master/contracts/protocol/libraries/math/MathUtils.sol

### Finding 5.7 -- IMPORTANT: BSC Block Time Configuration Is Outdated

`config/chains/56.json` specifies `"block_time_seconds": 0.45`. After the Maxwell hard fork (June 2025), BSC's block time is **0.75 seconds** (real-world average ~0.8s), not 0.45s. This affects polling interval calculations and transaction timeout estimates.

**References:**
- BNB Chain Maxwell Hardfork Announcement (June 2025)

### Finding 5.8 -- ENHANCEMENT: E-Mode Analysis

The plan correctly disables E-Mode (`"enabled": false, "category_id": 0`) since WBNB and USDT are in different categories. However, E-Mode becomes relevant if:

- BNB liquid staking derivatives (stkBNB, BNBx) are added to a BNB-correlated E-Mode category (90%+ LTV possible)
- Stablecoin-stablecoin strategies are added (97% LTV for USDT/USDC pairs)

Aave V3.2 "Liquid E-Modes" allow assets in multiple categories simultaneously, expanding future options.

**References:**
- Aave E-Mode: https://aave.com/help/borrowing/e-mode
- LlamaRisk: Aave V3.2 Liquid E-Modes analysis

### Finding 5.9 -- ENHANCEMENT: DEX-Oracle Price Divergence Check

When the bot swaps via DEX aggregator, it gets the DEX price. When Aave values collateral, it uses the Chainlink price. During flash crashes or liquidity events, these can diverge. The bot might receive less collateral than Aave values it at, creating an inflated HF that corrects when the oracle catches up.

**Recommendation:** After receiving an aggregator quote, compare the implied swap price against the Chainlink price. If divergence exceeds 1%, log a warning and consider widening slippage tolerance or aborting.

### Finding 5.10 -- ENHANCEMENT: Nonce Management Robustness

The plan's `tx_submitter.py` does not detail nonce management beyond the timing config (`nonce_refresh_interval_seconds: 30`). For production reliability:

- Maintain a local nonce counter initialized from `getTransactionCount('pending')` on startup
- Use `asyncio.Lock` to prevent race conditions
- Implement replacement logic: if TX pending >30s, re-send with same nonce + 12.5% higher gas
- On startup, detect abandoned pending transactions by comparing `pending` vs `latest` nonce counts

---

## Summary: Priority-Ordered Action Items

### Critical (Must Address Before Production)

| # | Finding | Plan Section Affected | New/Modified |
|---|---------|----------------------|--------------|
| 1 | Add entry signal generation (`signal_engine.py`, `indicators.py`, `data_service.py`) | `core/strategy.py` data flow | 3 new files + config |
| 2 | Add short position support to Python code | `shared/types.py`, `core/strategy.py`, `core/position_manager.py` | Modified |
| 3 | Fix stress test formula for short positions (inverse HF formula) | `core/strategy.py` line 185 | Modified |
| 4 | Add borrow rate monitoring and cost-based entry/exit filters | `core/strategy.py` | New functionality |
| 5 | Add oracle freshness validation (`latestRoundData().updatedAt`) | `core/health_monitor.py` | New functionality |
| 6 | Model 100% close factor at small position sizes in stress tests | `core/strategy.py` | Modified |
| 7 | Add historical price data source (Binance API) for indicator computation | New `core/data_service.py` | New file |

### Important (Should Address in Phase 1)

| # | Finding | Plan Section Affected | New/Modified |
|---|---------|----------------------|--------------|
| 8 | Add P&L tracking and position history (SQLite) | New module | New file |
| 9 | Add liquidation cascade multiplier to stress tests | `core/strategy.py` | Modified |
| 10 | Fix compound interest model in `predict_hf_at()` | `core/health_monitor.py` | Modified |
| 11 | Update BSC block time to 0.75s | `config/chains/56.json` | Modified |
| 12 | Check USDT Isolation Mode on-chain before short positions | `core/position_manager.py` | Modified |
| 13 | Robust nonce management with locking and replacement | `execution/tx_submitter.py` | Modified |

### Enhancements (Phase 2+)

| # | Finding | Recommendation |
|---|---------|---------------|
| 14 | Multi-aggregator fan-out for best execution | Parallel quote all 3 aggregators, pick best |
| 15 | DEX-Oracle price divergence detection | Compare swap price vs Chainlink before submission |
| 16 | E-Mode support preparation | Document category constraints; prepare for BNB LSDs |
| 17 | Borrow rate spike alerting | Monitor utilization approaching kink point |

---

## Revised File Count and Line Estimates

| Category | Original Plan | After Audit | Delta |
|----------|--------------|-------------|-------|
| Files | ~40 | ~48 | +8 |
| Lines | ~3,600-4,350 | ~4,800-5,700 | +1,200-1,350 |
| New configs | 3 | 4 | +1 (`signals.json`) |
| New core modules | 4 | 7 | +3 (`signal_engine`, `indicators`, `data_service`) |

---

## Academic References (Consolidated)

### DeFi Leverage and Liquidation
1. Perez, Werner, Xu, Livshits -- "Liquidations: DeFi on a Knife-Edge" (FC 2021, LNCS vol. 12675) [arXiv:2009.13235](https://arxiv.org/abs/2009.13235)
2. Heimbach & Huang -- "DeFi Leverage" (BIS Working Paper No. 1171, 2024) [PDF](https://www.bis.org/publ/work1171.pdf)
3. Klages-Mundt & Minca -- "(In)Stability for the Blockchain" (Crypto Economics Systems, 2019/2020) [arXiv:1906.02152](https://arxiv.org/abs/1906.02152)
4. OECD -- "DeFi Liquidations: Volatility and Liquidity" (2023) [PDF](https://www.oecd.org/content/dam/oecd/en/publications/reports/2023/07/defi-liquidations_89cba79d/0524faaf-en.pdf)
5. "Locked In, Levered Up: Risk, Return, and Ruin in DeFi Lending" (British Accounting Review, 2025)
6. Lehar et al. -- "Systemic Fragility in Decentralized Markets" [PDF](https://www.snb.ch/dam/jcr:12bc6015-1616-4f49-a89f-83ecbe2708a2/sem_2023_05_26_lehar.n.pdf)
7. Tovanich et al. -- "Contagion in Decentralized Lending Protocols" (DeFi '23) [PDF](https://hal.science/hal-04221228v1/document)

### AMM Routing and Swap Execution
8. Angeris, Chitra, Evans, Boyd -- "Optimal Routing for CFMMs" (ACM EC 2022) [arXiv:2204.05238](https://arxiv.org/abs/2204.05238)
9. Diamandis, Resnick, Chitra, Angeris -- "Efficient Algorithm for Optimal Routing Through CFMMs" (FC 2023) [arXiv:2302.04938](https://arxiv.org/abs/2302.04938)
10. Milionis et al. -- "Automated Market Making and Loss-Versus-Rebalancing" [PDF](https://anthonyleezhang.github.io/pdfs/lvr.pdf)

### MEV and Mempool
11. Daian et al. -- "Flash Boys 2.0" (IEEE S&P 2020) [arXiv:1904.05234](https://arxiv.org/abs/1904.05234)
12. Qin, Zhou, Gervais -- "Quantifying Blockchain Extractable Value" (IEEE S&P 2022) [arXiv:2101.05511](https://arxiv.org/abs/2101.05511)
13. Cartea & Sanchez-Betancourt -- "Detecting Toxic Flow" (2023) [arXiv:2312.05827](https://arxiv.org/abs/2312.05827)
14. BIS Bulletin No. 58 -- "Miners as Intermediaries: Extractable Value and Market Manipulation" [PDF](https://www.bis.org/publ/bisbull58.pdf)

### Trading Signals and Technical Analysis
15. Fang et al. -- "Cryptocurrency Trading: A Comprehensive Survey" (Financial Innovation, 2022) [Springer](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-021-00321-6)
16. Hudson & Urquhart -- "Technical Trading and Cryptocurrencies" (Annals of Operations Research, 2019, Vol. 297, pp. 191-220) [Springer](https://link.springer.com/article/10.1007/s10479-019-03357-1)
17. Hafid et al. -- "Predicting Bitcoin Market Trends with Enhanced Technical Indicator Integration" (2024) [arXiv:2410.06935](https://arxiv.org/html/2410.06935v1)
18. Beluska & Vojtko -- "Revisiting Trend-following and Mean-Reversion in Bitcoin" (SSRN, 2024) [PDF](https://papers.ssrn.com/sol3/Delivery.cfm/4955617.pdf?abstractid=4955617)

### Interest Rates and Oracle Security
19. "Optimal Risk-Aware Interest Rates for DeFi Lending Protocols" (2025) [arXiv:2502.19862](https://arxiv.org/html/2502.19862v1)
20. Deng et al. -- "Safeguarding DeFi Smart Contracts against Oracle Deviations" (ICSE 2024) [PDF](https://www.cs.toronto.edu/~fanl/papers/oracle-icse24.pdf)
21. "Oracles in DeFi: Attack Costs, Profits and Mitigation Measures" (Frontiers, 2023) [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9857405/)

### Protocol Documentation
22. Aave V3 Technical Documentation: https://aave.com/docs/aave-v3
23. Aave V3 Risk Parameters (Chaos Labs): https://chaoslabs.xyz/resources/chaos_aave_risk_param_methodology.pdf
24. Chainlink Data Feeds BSC: https://docs.chain.link/data-feeds
25. BNB Chain BEP-322 (PBS): https://github.com/bnb-chain/BEPs/blob/master/BEPs/BEP322.md
26. 1inch Aggregation Protocol: https://1inch.network/aggregation-protocol/
