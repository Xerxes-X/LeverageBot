# Aave V3 Flash Loan Liquidation Strategy

## Table of Contents

1. [Strategy Overview](#1-strategy-overview)
2. [Core Concepts](#2-core-concepts)
3. [Step-by-Step Execution Flow](#3-step-by-step-execution-flow)
4. [On-Chain Component (Smart Contract)](#4-on-chain-component-smart-contract)
5. [Off-Chain Component (Monitoring Bot)](#5-off-chain-component-monitoring-bot)
6. [Profitability Calculation](#6-profitability-calculation)
7. [Aave V3 Deployed Addresses](#7-aave-v3-deployed-addresses)
8. [Risk Analysis](#8-risk-analysis)
9. [Competitive Landscape](#9-competitive-landscape)
10. [References](#10-references)

---

## 1. Strategy Overview

This strategy uses **Aave V3 mode-0 flash loans** to atomically liquidate underwater borrowing
positions on Aave V3. The liquidator requires **zero upfront capital** -- all funds are borrowed
and repaid within a single transaction. Profit comes from the **liquidation bonus** (5-10%
depending on the asset), minus flash loan fees, swap costs, and gas.

### Atomic Execution Guarantee

Due to the atomicity of EVM transactions (Qin et al., Financial Cryptography 2021), the entire
operation either succeeds completely or reverts with no state changes. The only cost of a failed
attempt is the gas fee for the reverted transaction. This eliminates capital risk entirely.

### Target Profit Range

- **Per liquidation**: $1 - $500+ depending on position size and asset
- **Minimum viable liquidation**: ~$500 debt on BSC ($0.05 gas), ~$5,000 debt on Ethereum ($2-5 gas)
- **Sweet spot**: $2,000 - $50,000 debt positions with volatile collateral (WETH, WBTC, WBNB)

---

## 2. Core Concepts

### 2.1 Health Factor

The health factor determines whether a position is eligible for liquidation:

```
                  Σ(collateral_i × price_i × liquidationThreshold_i)
Health Factor = ──────────────────────────────────────────────────────
                              Σ(debt_j × price_j)
```

- **HF >= 1.0**: Position is healthy, cannot be liquidated
- **0.95 < HF < 1.0**: Position is liquidatable; up to **50%** of debt can be repaid per call
- **HF <= 0.95**: Position is liquidatable; up to **100%** of debt can be repaid per call

The threshold `CLOSE_FACTOR_HF_THRESHOLD = 0.95` is defined in Aave V3's `LiquidationLogic.sol`.
The default close factor (50%) applies when both the collateral and debt exceed
`MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD = 2000e8` ($2,000 in base currency) AND health factor > 0.95.

Source: [LiquidationLogic.sol](https://github.com/aave-dao/aave-v3-origin/blob/main/src/contracts/protocol/libraries/logic/LiquidationLogic.sol)

### 2.2 Liquidation Bonus

When a liquidator repays a borrower's debt, they receive the equivalent value in the borrower's
collateral **plus a bonus**. This bonus compensates for market risk and incentivizes liquidations.

```
collateral_received = (debt_repaid × debt_price × liquidation_bonus) / collateral_price
```

From `LiquidationLogic.sol`:

```solidity
// Base collateral to cover the debt
vars.baseCollateral = (debtAssetPrice * debtToCover * collateralAssetUnit)
                    / (vars.collateralAssetPrice * debtAssetUnit);

// Apply liquidation bonus (e.g., 10500 = 105% = 5% bonus)
vars.maxCollateralToLiquidate = vars.baseCollateral.percentMul(liquidationBonus);
```

The `liquidationBonus` is stored as a percentage with 4 decimal places:
- `10500` = 105% = **5% bonus** (stablecoins, ETH)
- `11000` = 110% = **10% bonus** (higher-risk assets)

### 2.3 Typical Liquidation Bonus by Asset Class

| Asset Type | Liquidation Bonus | Examples |
|------------|-------------------|----------|
| Stablecoins | 4.5 - 5% | USDC, USDT, DAI |
| Major L1 tokens | 5 - 6% | ETH, WBTC, WBNB |
| DeFi / Mid-cap | 7.5 - 10% | LINK, AAVE, CRV |
| Long-tail / High-risk | 10 - 15% | Varies by governance |

Exact values per asset are queried on-chain via:

```solidity
IPoolDataProvider(DATA_PROVIDER).getReserveConfigurationData(asset)
// Returns: (..., liquidationThreshold, liquidationBonus, ...)
```

### 2.4 Flash Loan Mode 0 Mechanics

Mode 0 = **standard flash loan**: borrow and repay within the same transaction.

```
1. Bot calls Pool.flashLoanSimple(receiver, debtAsset, amount, params, 0)
2. Pool transfers `amount` of debtAsset to receiver contract
3. Pool calls receiver.executeOperation(asset, amount, premium, initiator, params)
4. Inside executeOperation:
   a. Approve Pool to spend debtAsset (for liquidationCall)
   b. Call Pool.liquidationCall(collateral, debt, user, debtToCover, false)
   c. Receive discounted collateral
   d. Swap collateral back to debtAsset via DEX
   e. Approve Pool to pull back amount + premium
5. Pool pulls amount + premium via transferFrom
6. Transaction completes (or reverts if insufficient balance)
```

**Flash loan premium**: 0.05% on Aave V3 (was 0.09%, reduced via governance).
On a $10,000 flash loan, the fee is **$5**.

---

## 3. Step-by-Step Execution Flow

### Complete Liquidation Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OFF-CHAIN (Rust Bot)                        │
│                                                                    │
│  Step 1: Monitor    Continuously track borrower health factors     │
│  Step 2: Detect     Identify positions where HF < 1.0             │
│  Step 3: Evaluate   Calculate profit: bonus - fees - gas - slippage│
│  Step 4: Simulate   eth_call to verify tx succeeds and is profit- │
│                     able at current block state                    │
│  Step 5: Submit     Send tx via private RPC to avoid frontrunning  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ON-CHAIN (Smart Contract)                      │
│                                                                    │
│  Step 6: Flash Borrow   Borrow debt token from Aave V3 (mode 0)   │
│  Step 7: Liquidate       Call liquidationCall() on Aave Pool       │
│  Step 8: Receive         Get discounted collateral (+ bonus)       │
│  Step 9: Swap            Convert collateral → debt token via DEX   │
│  Step 10: Repay          Return flash loan + premium to Aave       │
│  Step 11: Profit         Remaining tokens sent to bot owner        │
│                                                                    │
│  If any step fails → entire transaction reverts (no loss)          │
└─────────────────────────────────────────────────────────────────────┘
```

### Detailed Step Breakdown

#### Step 1 - Monitor Health Factors

The bot maintains an in-memory index of all active Aave V3 borrowing positions. Positions are
tracked by subscribing to protocol events emitted by the Aave Pool contract:

| Event | Trigger |
|-------|---------|
| `Supply(reserve, user, onBehalfOf, amount, referralCode)` | Collateral added |
| `Withdraw(reserve, user, to, amount)` | Collateral removed |
| `Borrow(reserve, onBehalfOf, user, amount, interestRateMode, borrowRate, referralCode)` | Debt increased |
| `Repay(reserve, user, repayer, amount, useATokens)` | Debt decreased |
| `LiquidationCall(collateralAsset, debtAsset, user, debtToCover, ...)` | Position liquidated by someone |
| `ReserveDataUpdated(reserve, liquidityRate, stableBorrowRate, variableBorrowRate, liquidityIndex, variableBorrowIndex)` | Interest rate / index changed |

Additionally, the bot must track **price feed updates** from the Aave Oracle contract, since
health factors change with asset prices. On BSC, Aave V3 uses Chainlink oracles -- subscribe to
`AnswerUpdated(int256 current, uint256 roundId, uint256 updatedAt)` on each oracle contract.

**Periodic validation**: Every N blocks (e.g., every 10 blocks), call `getUserAccountData(user)`
on the Pool contract for the top-N most at-risk positions to validate locally computed health
factors against on-chain state.

```solidity
// Pool.getUserAccountData(address user) returns:
(
    uint256 totalCollateralBase,      // Total collateral in base currency (USD, 8 decimals)
    uint256 totalDebtBase,            // Total debt in base currency
    uint256 availableBorrowsBase,     // Remaining borrow capacity
    uint256 currentLiquidationThreshold, // Weighted avg liquidation threshold
    uint256 ltv,                      // Weighted avg LTV
    uint256 healthFactor              // 1e18 = 1.0
)
```

#### Step 2 - Detect Liquidatable Positions

A position is liquidatable when `healthFactor < 1e18` (i.e., < 1.0).

The bot should maintain a **priority queue** sorted by health factor (ascending). When a price
update occurs, recompute health factors for all positions holding that collateral/debt asset and
re-sort.

For each liquidatable position, determine:
- Which collateral asset(s) the borrower holds (may be multiple)
- Which debt asset(s) the borrower owes (may be multiple)
- The optimal (collateral, debt) pair to liquidate (highest bonus, deepest DEX liquidity)

```solidity
// For each reserve the borrower interacts with:
IPoolDataProvider(DATA_PROVIDER).getUserReserveData(asset, user)
// Returns: currentATokenBalance, currentStableDebt, currentVariableDebt, ...
```

#### Step 3 - Evaluate Profitability

```
gross_profit = collateral_received_value - debt_repaid_value
             = debt_repaid × (liquidation_bonus - 1.0)

net_profit   = gross_profit - flash_loan_fee - swap_cost - gas_cost

Where:
  flash_loan_fee = debt_repaid × 0.0005          (0.05% Aave V3 premium)
  swap_cost      = collateral_value × swap_fee    (DEX fee tier, typically 0.01-0.3%)
  swap_slippage  = estimated from DEX pool depth
  gas_cost       = estimated_gas × gas_price      (~400K-600K gas for full flow)
```

**Example** (BSC, liquidating $5,000 USDT debt, WBNB collateral, 10% bonus):

```
debt_repaid        = $5,000 USDT
collateral_received = $5,000 × 1.10 / WBNB_price = $5,500 worth of WBNB
gross_profit       = $500

flash_loan_fee     = $5,000 × 0.0005    = $2.50
swap_cost          = $5,500 × 0.0025    = $13.75  (PancakeSwap V3 0.25% tier)
gas_cost           = 500,000 × 3 gwei × BNB_price / 1e9 ≈ $0.45

net_profit         = $500 - $2.50 - $13.75 - $0.45 = $483.30
```

**Example** (BSC, small $500 USDT debt, WBNB collateral, 10% bonus):

```
debt_repaid        = $500 USDT
collateral_received = $550 worth of WBNB
gross_profit       = $50

flash_loan_fee     = $0.25
swap_cost          = $1.38
gas_cost           = $0.45

net_profit         = $50 - $0.25 - $1.38 - $0.45 = $47.92
```

#### Step 4 - Simulate Transaction

Before submitting, simulate the exact transaction via `eth_call`:

```rust
let call_data = contract.encode_execute(asset, amount, params);
let result = provider.call(
    TransactionRequest::new()
        .to(contract_address)
        .data(call_data)
        .from(bot_eoa),
    BlockId::latest()
).await?;
```

This returns the exact outcome at the current block state. If the simulation reverts or shows
insufficient profit, skip this opportunity.

#### Step 5 - Submit Transaction

Submit via **private/MEV-protected RPC** to prevent frontrunning by other liquidation bots:

| Chain | Private RPC Options |
|-------|---------------------|
| BSC | 48 Club (`https://rpc-bsc.48.club`), Blocker builder, BNB48 Privacy RPC |
| Ethereum | Flashbots Protect (`https://rpc.flashbots.net`), MEV-Share, MEV Blocker |

Set gas price slightly above market to ensure inclusion. On BSC, priority fee competition is
less intense than Ethereum.

#### Steps 6-11 - On-Chain Execution (Inside Smart Contract)

See Section 4 for the complete smart contract implementation.

---

## 4. On-Chain Component (Smart Contract)

### 4.1 Contract Architecture

```
┌─────────────────────────────────────────────────┐
│           FlashLoanLiquidator.sol                │
│                                                  │
│  execute()                                       │
│    └→ Pool.flashLoanSimple()                     │
│         └→ executeOperation() [callback]         │
│              ├→ Pool.liquidationCall()            │
│              ├→ DEXRouter.swap(collateral → debt) │
│              ├→ IERC20.approve(Pool, repayment)   │
│              └→ return true                       │
│                                                  │
│  After flash loan settles:                       │
│    └→ sweep remaining profit to owner            │
└─────────────────────────────────────────────────┘
```

### 4.2 Solidity Implementation

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IPool} from "@aave/v3-core/contracts/interfaces/IPool.sol";
import {IPoolAddressesProvider} from "@aave/v3-core/contracts/interfaces/IPoolAddressesProvider.sol";
import {IFlashLoanSimpleReceiver} from "@aave/v3-core/contracts/flashloan-simple/IFlashLoanSimpleReceiver.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

contract FlashLoanLiquidator is IFlashLoanSimpleReceiver {
    using SafeERC20 for IERC20;

    IPoolAddressesProvider public immutable override ADDRESSES_PROVIDER;
    IPool public immutable override POOL;
    address public immutable owner;

    struct LiquidationParams {
        address collateralAsset;   // The collateral to seize
        address borrower;          // The underwater borrower
        uint256 debtToCover;       // Amount of debt to repay
        address swapRouter;        // DEX router for collateral → debt swap
        bytes swapCalldata;        // Encoded swap call (built off-chain)
        uint256 minProfit;         // Minimum profit or revert (slippage guard)
    }

    constructor(address addressesProvider) {
        ADDRESSES_PROVIDER = IPoolAddressesProvider(addressesProvider);
        POOL = IPool(IPoolAddressesProvider(addressesProvider).getPool());
        owner = msg.sender;
    }

    /// -----------------------------------------------------------------------
    /// Entry point: called by the bot's EOA
    /// -----------------------------------------------------------------------

    /// @notice Initiate a flash-loan-powered liquidation
    /// @param debtAsset   The debt token to flash borrow (e.g., USDT)
    /// @param flashAmount The amount to flash borrow (>= debtToCover)
    /// @param params      ABI-encoded LiquidationParams
    function execute(
        address debtAsset,
        uint256 flashAmount,
        bytes calldata params
    ) external {
        require(msg.sender == owner, "unauthorized");

        POOL.flashLoanSimple(
            address(this),   // receiver
            debtAsset,       // asset to borrow
            flashAmount,     // amount
            params,          // forwarded to executeOperation
            0                // referral code
        );

        // After flash loan completes, sweep all profit to owner
        uint256 profit = IERC20(debtAsset).balanceOf(address(this));
        if (profit > 0) {
            IERC20(debtAsset).safeTransfer(owner, profit);
        }
    }

    /// -----------------------------------------------------------------------
    /// Flash loan callback: called by Aave Pool
    /// -----------------------------------------------------------------------

    /// @notice Aave V3 flash loan callback
    /// @param asset     The borrowed asset
    /// @param amount    The borrowed amount
    /// @param premium   The flash loan fee (amount × 0.05%)
    /// @param initiator Must be this contract
    /// @param params    ABI-encoded LiquidationParams
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        // Security: only the Pool can call this, and only we can initiate
        require(msg.sender == address(POOL), "caller must be Aave Pool");
        require(initiator == address(this), "initiator must be self");

        LiquidationParams memory liq = abi.decode(params, (LiquidationParams));

        // ---- Step 7: Execute the liquidation ----
        // Approve Pool to pull debtToCover of the debt asset
        IERC20(asset).safeIncreaseAllowance(address(POOL), liq.debtToCover);

        // Call liquidationCall: repay borrower's debt, receive their collateral
        POOL.liquidationCall(
            liq.collateralAsset,    // collateral to seize
            asset,                  // debt asset we're repaying
            liq.borrower,           // the underwater borrower
            liq.debtToCover,        // how much debt to repay
            false                   // receive underlying tokens, not aTokens
        );

        // ---- Step 9: Swap collateral back to debt asset ----
        uint256 collateralBalance = IERC20(liq.collateralAsset).balanceOf(address(this));

        // Approve the DEX router to spend our collateral
        IERC20(liq.collateralAsset).safeIncreaseAllowance(liq.swapRouter, collateralBalance);

        // Execute the swap (calldata is constructed off-chain by the bot)
        (bool swapOk, ) = liq.swapRouter.call(liq.swapCalldata);
        require(swapOk, "collateral swap failed");

        // ---- Step 10: Approve flash loan repayment ----
        uint256 amountOwed = amount + premium;
        IERC20(asset).safeIncreaseAllowance(address(POOL), amountOwed);

        // ---- Step 11: Verify profit ----
        uint256 remainingBalance = IERC20(asset).balanceOf(address(this));
        require(remainingBalance >= amountOwed + liq.minProfit, "insufficient profit");

        return true;
    }

    /// -----------------------------------------------------------------------
    /// Admin: recover stuck tokens (safety measure)
    /// -----------------------------------------------------------------------

    function withdrawToken(address token) external {
        require(msg.sender == owner, "unauthorized");
        IERC20(token).safeTransfer(owner, IERC20(token).balanceOf(address(this)));
    }
}
```

### 4.3 Key Design Decisions

**Why `flashLoanSimple` instead of `flashLoan`?**
`flashLoanSimple` is gas-optimized for single-asset borrows. Since we only need to borrow the
debt token, the simpler variant saves ~20K gas.

**Why `receiveAToken = false`?**
We need the underlying collateral token (e.g., WBNB) to swap on a DEX. Receiving aTokens would
require an extra `withdraw()` step.

**Why build swap calldata off-chain?**
The optimal swap route changes block-to-block. The off-chain bot queries DEX aggregator APIs
(0x, 1inch, PancakeSwap routing) for the best path and encodes the calldata before submission.

**Why `minProfit` guard?**
Between simulation and on-chain execution, prices can move. The `minProfit` parameter ensures the
transaction reverts if slippage erodes the profit below an acceptable threshold, protecting
against losses from DEX price impact.

---

## 5. Off-Chain Component (Monitoring Bot)

### 5.1 Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Liquidation Bot (Rust)                         │
│                                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐                │
│  │  Event       │    │  Position    │    │  Health     │                │
│  │  Listener    │───→│  Index       │───→│  Factor     │                │
│  │  (WSS)       │    │  (HashMap)   │    │  Calculator │                │
│  └─────────────┘    └──────────────┘    └──────┬──────┘                │
│                                                 │                       │
│  ┌─────────────┐    ┌──────────────┐    ┌──────▼──────┐                │
│  │  Price       │    │  Profitab-   │    │  Priority   │                │
│  │  Feed        │───→│  ility       │←───│  Queue      │                │
│  │  (Oracle)    │    │  Engine      │    │  (by HF)    │                │
│  └─────────────┘    └──────┬───────┘    └─────────────┘                │
│                            │                                            │
│                     ┌──────▼───────┐    ┌─────────────┐                │
│                     │  Simulator   │───→│  Submitter   │                │
│                     │  (eth_call)  │    │  (Private    │                │
│                     │              │    │   RPC)       │                │
│                     └──────────────┘    └─────────────┘                │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Position Indexing

The bot must maintain a local index of every active borrowing position on Aave V3. This avoids
expensive on-chain calls for every health factor check.

**Data structure per user:**

```rust
struct UserPosition {
    /// Collateral: asset_address → aToken balance (in underlying units)
    collaterals: HashMap<Address, U256>,
    /// Debts: asset_address → (stable_debt + variable_debt) in underlying units
    debts: HashMap<Address, U256>,
    /// Precomputed health factor (updated on any change)
    health_factor: f64,
    /// Last block this position was updated
    last_updated_block: u64,
}
```

**Bootstrap**: On startup, query historical events from a recent checkpoint (or the Aave subgraph)
to build the initial index. Then maintain it incrementally via real-time event subscriptions.

### 5.3 Health Factor Computation

The bot computes health factors locally using:

```rust
fn compute_health_factor(
    user: &UserPosition,
    prices: &HashMap<Address, f64>,        // From oracle feed
    reserve_configs: &HashMap<Address, ReserveConfig>,  // LT, LB, decimals
) -> f64 {
    let mut weighted_collateral = 0.0;
    let mut total_debt = 0.0;

    for (asset, balance) in &user.collaterals {
        let config = &reserve_configs[asset];
        let value = to_usd(*balance, prices[asset], config.decimals);
        weighted_collateral += value * (config.liquidation_threshold as f64 / 10000.0);
    }

    for (asset, debt) in &user.debts {
        let config = &reserve_configs[asset];
        total_debt += to_usd(*debt, prices[asset], config.decimals);
    }

    if total_debt == 0.0 {
        return f64::MAX; // No debt = infinite health factor
    }

    weighted_collateral / total_debt
}
```

### 5.4 Event Subscription

Subscribe to the Aave Pool contract via WebSocket for real-time updates:

```rust
// Pseudo-code for event subscription
let pool_filter = Filter::new()
    .address(AAVE_POOL_ADDRESS)
    .events(vec![
        "Supply(address,address,address,uint256,uint16)",
        "Withdraw(address,address,address,uint256)",
        "Borrow(address,address,address,uint256,uint8,uint256,uint16)",
        "Repay(address,address,address,uint256,bool)",
        "LiquidationCall(address,address,address,uint256,uint256,address,bool)",
    ]);

let oracle_filter = Filter::new()
    .address(oracle_addresses) // All Chainlink feeds used by Aave
    .event("AnswerUpdated(int256,uint256,uint256)");
```

### 5.5 Swap Route Construction

For each liquidation opportunity, query a DEX for the optimal swap route:

**On BSC:**
- PancakeSwap V3 (primary, lowest fees for major pairs)
- PancakeSwap V2 (deeper liquidity for some pairs)
- Direct pool swaps (skip router overhead for simple pairs)

**On Ethereum:**
- Uniswap V3/V4 direct pool calls
- 0x API or 1inch API for multi-hop routing

The bot must encode the exact swap calldata that the contract will execute:

```rust
// Example: encode a PancakeSwap V3 exactInputSingle call
let swap_calldata = pancake_router.encode(
    "exactInputSingle",
    (ExactInputSingleParams {
        tokenIn: collateral_asset,
        tokenOut: debt_asset,
        fee: 2500,  // 0.25% tier
        recipient: liquidator_contract,
        deadline: block_timestamp + 120,
        amountIn: collateral_amount,
        amountOutMinimum: min_debt_received, // slippage protection
        sqrtPriceLimitX96: 0,
    },)
);
```

---

## 6. Profitability Calculation

### 6.1 Formula

```
Net Profit = Collateral Value Received
           - Debt Repaid
           - Flash Loan Premium
           - DEX Swap Cost
           - Gas Cost

Where:
  Collateral Value Received = debtToCover × (debtPrice / collateralPrice) × liquidationBonus
  Flash Loan Premium        = flashAmount × FLASHLOAN_PREMIUM_TOTAL  (0.05% on Aave V3)
  DEX Swap Cost             = collateralValue × dexFeeRate + priceImpact
  Gas Cost                  = gasUsed × gasPrice × nativeTokenPrice
```

### 6.2 Worked Examples

#### Example A: Large liquidation on BSC (WBNB collateral, USDT debt)

| Parameter | Value |
|-----------|-------|
| Borrower's USDT debt | $10,000 |
| Health factor | 0.97 (close factor = 50%) |
| Max debt to cover | $5,000 |
| WBNB liquidation bonus | 10% (11000) |
| Collateral received | $5,500 worth of WBNB |
| Flash loan amount | 5,000 USDT |
| Flash loan premium (0.05%) | 2.50 USDT |
| PancakeSwap V3 fee (0.25%) | 13.75 USDT |
| Swap slippage (~0.05%) | 2.75 USDT |
| Gas cost (~500K gas × 3 gwei) | ~$0.45 |
| **Net profit** | **~$480.55** |

#### Example B: Small liquidation on BSC (WETH collateral, USDC debt)

| Parameter | Value |
|-----------|-------|
| Borrower's USDC debt | $1,000 |
| Health factor | 0.92 (close factor = 100%) |
| Max debt to cover | $1,000 |
| WETH liquidation bonus | 5% (10500) |
| Collateral received | $1,050 worth of WETH |
| Flash loan amount | 1,000 USDC |
| Flash loan premium (0.05%) | 0.50 USDC |
| PancakeSwap V3 fee (0.05%) | 0.525 USDC |
| Swap slippage (~0.1%) | 1.05 USDC |
| Gas cost | ~$0.45 |
| **Net profit** | **~$47.48** |

#### Example C: Marginal liquidation (barely profitable)

| Parameter | Value |
|-----------|-------|
| Borrower's USDC debt | $200 |
| Health factor | 0.98 (close factor = 50%) |
| Max debt to cover | $100 |
| Liquidation bonus | 5% |
| Gross bonus | $5.00 |
| Flash loan premium | $0.05 |
| Swap cost | $0.15 |
| Gas cost (BSC) | $0.45 |
| **Net profit** | **~$4.35** |

On Ethereum mainnet, Example C would be unprofitable due to ~$2-5 gas costs.

### 6.3 Minimum Viable Position Sizes

| Chain | Gas Cost | Min Debt for $1 Profit (5% bonus) | Min Debt for $10 Profit |
|-------|----------|-----------------------------------|------------------------|
| BSC | ~$0.05 | ~$15 | ~$250 |
| Ethereum | ~$3.00 | ~$80 | ~$300 |
| Arbitrum | ~$0.10 | ~$5 | ~$250 |
| Base | ~$0.05 | ~$5 | ~$250 |

*Assumes 0.05% flash fee + 0.25% DEX fee. Actual minimums depend on pool liquidity depth.*

---

## 7. Aave V3 Deployed Addresses

### 7.1 BNB Chain (Primary Target)

| Contract | Address |
|----------|---------|
| PoolAddressesProvider | `0xff75B6da14FfbbfD355Daf7a2731456b3562Ba6D` |
| Pool | `0x6807dc923806fE8Fd134338EABCA509979a7e0cB` |
| PoolDataProvider | `0xc90Df74A7c16245c5F5C5870327Ceb38Fe5d5328` |
| Oracle | `0x39bc1bfDa2130d6Bb6DBEfd366939b4c7aa7C697` |
| ACLManager | `0x2D97F8FA96886Fd923c065F5457F9DDd494e3877` |

**Key Assets on BNB Chain:**

| Asset | Underlying | aToken | Variable Debt Token |
|-------|-----------|--------|---------------------|
| WBNB | `0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c` | `0x9B00a094...` | `0x0E76414d...` |
| USDC | `0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d` | `0x00901a07...` | `0xcDBBEd56...` |
| USDT | `0x55d398326f99059fF775485246999027B3197955` | `0xa9251ca9...` | `0xF8bb2Be5...` |

Source: [aave-address-book/AaveV3BNB.sol](https://github.com/bgd-labs/aave-address-book/blob/main/src/AaveV3BNB.sol)

### 7.2 Ethereum Mainnet

| Contract | Address |
|----------|---------|
| Pool | `0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2` |

Source: [Etherscan - Aave Pool V3](https://etherscan.io/address/0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2)

### 7.3 Querying Liquidation Bonus On-Chain

```solidity
// Call on PoolDataProvider to get asset-specific liquidation parameters
(
    uint256 decimals,
    uint256 ltv,
    uint256 liquidationThreshold,
    uint256 liquidationBonus,       // e.g., 10500 = 105% = 5% bonus
    uint256 reserveFactor,
    bool usageAsCollateralEnabled,
    bool borrowingEnabled,
    bool stableBorrowRateEnabled,
    bool isActive,
    bool isFrozen
) = IPoolDataProvider(DATA_PROVIDER).getReserveConfigurationData(assetAddress);
```

---

## 8. Risk Analysis

### 8.1 Capital Risk: Zero

The flash loan guarantees atomicity. If the liquidation fails, the swap fails, or profit is
insufficient, the entire transaction reverts. The only loss is the gas fee for the failed tx.

### 8.2 Gas Waste Risk: Low-Medium

Failed simulations that pass `eth_call` but fail on-chain (due to state changes between
simulation and inclusion) waste gas. Mitigation:
- Use private RPCs that don't charge for reverted transactions (Flashbots Protect)
- Set tight `minProfit` thresholds
- Check mempool for competing liquidation transactions

### 8.3 Smart Contract Risk: Low

The contract is simple (< 100 lines of logic), stateless, and only callable by the owner.
Primary attack surface is the external swap call -- mitigated by the `minProfit` check that
reverts if the swap returns less than expected.

### 8.4 Oracle Manipulation Risk: Minimal

Aave V3 uses Chainlink oracles with multiple data sources and deviation thresholds. The bot
relies on the same oracles as Aave, so the health factor computation is consistent.

### 8.5 Regulatory Risk

Liquidation is an explicitly designed protocol mechanism -- Aave relies on third-party
liquidators to maintain solvency. Performing liquidations is a public good for protocol health,
not an exploit. The Aave documentation explicitly encourages third-party liquidators.

---

## 9. Competitive Landscape

### 9.1 Who Competes

Liquidation is competitive. Multiple bots monitor the same positions and race to liquidate first.
Competitors range from:
- **Professional MEV searchers** using Flashbots bundles and builder APIs
- **Protocol-integrated bots** run by lending protocol teams
- **Individual operators** running custom bots

### 9.2 Competitive Advantages

| Advantage | How to Achieve |
|-----------|----------------|
| **Latency** | Co-locate node infrastructure near validators/builders |
| **Gas optimization** | Minimize contract gas usage (inline assembly, skip router) |
| **Coverage breadth** | Monitor all chains where Aave V3 is deployed, not just Ethereum |
| **Niche assets** | Focus on long-tail assets where fewer bots operate |
| **Partial liquidations** | Optimize `debtToCover` amount for maximum profit-per-gas |
| **Price prediction** | Use pending mempool transactions to anticipate price movements |

### 9.3 BSC vs Ethereum Competition

BSC is **less competitive** for liquidation bots because:
- Lower gas costs reduce the barrier but also reduce the urgency to optimize
- Fewer professional MEV searchers operate on BSC compared to Ethereum
- The BSC builder/proposer ecosystem is less mature (no Flashbots equivalent at scale)
- Aave V3 on BSC has lower TVL, meaning smaller but more frequent opportunities

This makes BSC the recommended starting chain for a new liquidation bot.

### 9.4 Strategies to Win

1. **Multi-collateral liquidation**: When a borrower has multiple collateral types, choose the
   one with the highest liquidation bonus AND deepest DEX liquidity
2. **Debt amount optimization**: Don't always liquidate the maximum allowed -- sometimes a
   smaller liquidation with less price impact yields more profit per gas spent
3. **Pre-liquidation positioning**: When a position's health factor is approaching 1.0 but hasn't
   crossed yet, prepare the transaction and submit the instant it becomes liquidatable
4. **Batch liquidation**: If multiple positions become liquidatable in the same block (e.g.,
   during a price crash), batch multiple liquidations into a single transaction

---

## 10. References

### Protocol Documentation

- [Aave V3 Flash Loans](https://aave.com/docs/aave-v3/guides/flash-loans) - Official flash loan specification
- [Aave V3 Pool Contract](https://aave.com/docs/aave-v3/smart-contracts/pool) - `liquidationCall` and `flashLoanSimple` function signatures
- [Aave V3 View Contracts](https://aave.com/docs/aave-v3/smart-contracts/view-contracts) - `getReserveConfigurationData`, `getUserReserveData`
- [Aave V3 LiquidationLogic.sol](https://github.com/aave-dao/aave-v3-origin/blob/main/src/contracts/protocol/libraries/logic/LiquidationLogic.sol) - On-chain liquidation logic (close factor, bonus calculation, dust prevention)
- [Aave V3 Parameters Dashboard](https://aave.com/docs/resources/parameters) - Per-asset risk parameters
- [Aave Address Book](https://github.com/bgd-labs/aave-address-book) - Deployed contract addresses per chain
- [Aave Liquidation Guide](https://github.com/galacticcouncil/aave-docs-v3/blob/main/guides/liquidations.md) - Official liquidation walkthrough

### Governance

- [AIP-196: Add DeFi Saver to FlashBorrowers](https://governance.aave.com/t/arfc-add-defi-saver-to-flashborrowers-on-aave-v3/12410) - Flash borrower whitelist mechanism
- [AIP-235: FlashBorrowers Whitelist Part II](https://governance-v2.aave.com/governance/proposal/235/) - Extended whitelist

### Academic & Research Papers

- Qin, Zhou, Livshits, Gervais. ["Attacking the DeFi Ecosystem with Flash Loans for Fun and Profit."](https://link.springer.com/chapter/10.1007/978-3-662-64322-8_1) *Financial Cryptography and Data Security 2021*. Springer. - Seminal peer-reviewed paper on flash loan atomicity and DeFi exploits.
- Chandler, Stiles, Blinken. ["DeFi Flash Loans: What 'Atomicity' Makes Possible."](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4116909) *SSRN Working Paper*. - Analysis of why flash loans cannot exist in traditional finance.
- ["Strengthening DeFi Security: A Static Analysis Approach to Flash Loan Vulnerabilities."](https://arxiv.org/html/2411.01230v2) *arXiv 2024*. - Vulnerability detection framework.
- ["Protecting DeFi Platforms against Non-Price Flash Loan Attacks."](https://arxiv.org/pdf/2503.01944) *arXiv 2025*.
- Cornelli, Gambacorta. ["Why DeFi lending? Evidence from Aave V2."](https://www.bis.org/publ/work1183.pdf) *BIS Working Papers*. - Empirical analysis of Aave lending behavior.
- Pangea Foundation. ["Aave's Liquidators."](https://blog.pangea.foundation/aaves-liquidators/) - Empirical study of Aave liquidator behavior and profitability.
- Pyth Network. ["Value Leakage and Fragmentation in Liquidations."](https://www.pyth.network/blog/value-leakage-and-fragmentation-in-liquidations) - Analysis of MEV extraction in liquidation markets.

### Technical Guides

- [Performing Liquidations on Aave (Amberdata)](https://blog.amberdata.io/performing-liquidations-on-the-aave-defi-lending-protocol) - Practical liquidation guide
- [Aave V3 Flash Leverage (Cyfrin Updraft)](https://updraft.cyfrin.io/courses/aave-v3/app/flash-leverage) - Educational flash loan course
- [Aave V3 Health Factor Architecture (Cyfrin)](https://updraft.cyfrin.io/courses/aave-v3/contract-architecture/health-factor) - Health factor deep dive
- [QuickNode: Flash Loans on Aave](https://www.quicknode.com/guides/defi/lending-protocols/how-to-make-a-flash-loan-using-aave) - Hands-on flash loan tutorial
- [Aave V3 Liquidation Tracker (QuickNode)](https://www.quicknode.com/sample-app-library/ethereum-aave-liquidation-tracker) - Sample monitoring application
- [MixBytes: Modern DeFi Lending - Aave V3](https://mixbytes.io/blog/modern-defi-lending-protocols-how-its-made-aave-v3) - Technical architecture breakdown

### DEX / Swap Infrastructure

- [0x Protocol Settler](https://github.com/0xProject/0x-settler) - Settlement contract for DEX aggregation
- [0x Swap API](https://0x.org/docs/0x-swap-api/introduction) - DEX aggregation and routing API
- [PancakeSwap V3 Docs](https://docs.pancakeswap.finance/) - BSC-native DEX

### Infrastructure

- [Safe Smart Account Architecture](https://safe.mirror.xyz/t76RZPgEKdRmWNIbEzi75onWPeZrBrwbLRejuj-iPpQ) - Proxy wallet architecture
- [Flashbots Protect RPC](https://docs.flashbots.net/) - MEV protection on Ethereum
- [48 Club BSC RPC](https://rpc-bsc.48.club) - Private RPC for BSC
