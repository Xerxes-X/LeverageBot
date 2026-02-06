# Paper Trading Setup Guide

## Overview

This guide configures the LeverageBot for **comprehensive paper trading** with:
- ✅ Real market data from Binance and Aave V3
- ✅ Real signal detection and trade opportunities
- ✅ Simulated $500 starting portfolio
- ✅ Full position tracking with take profit / stop loss
- ✅ Real-time P&L monitoring
- ✅ Performance metrics (Sharpe, Sortino, Calmar ratios)
- ❌ NO actual blockchain transactions
- ❌ NO real money at risk

This allows you to thoroughly test the bot's strategy and risk management before live trading.

---

## Quick Start

###Step 1: Configure Environment

Edit `.env` file:

```bash
# REQUIRED: Set dry run mode to true
EXECUTOR_DRY_RUN=true

# REQUIRED: BSC RPC endpoints (for reading Aave data)
BSC_RPC_URL_HTTP=https://bsc-dataseed1.binance.org/
BSC_RPC_URL_MEV_PROTECTED=https://rpc.48.club

# OPTIONAL: Leave empty for paper trading (bot generates random addresses)
USER_WALLET_ADDRESS=
EXECUTOR_PRIVATE_KEY=
LEVERAGE_EXECUTOR_ADDRESS=

# OPTIONAL: DEX aggregator API keys (not required for testing)
ONEINCH_API_KEY=
```

### Step 2: Configure Position Sizing

Edit `config/positions.json`:

```json
{
  "dry_run": true,
  "max_flash_loan_usd": 1500,        // Max $1,500 per trade (3x of $500)
  "max_position_usd": 1500,
  "max_leverage_ratio": "3.0",       // Maximum 3x leverage
  "min_health_factor": "1.5",         // Safety buffer above liquidation
  "max_slippage_bps": 50,            // 0.5% max slippage
  "cooldown_between_actions_seconds": 300  // 5 min between trades
}
```

### Step 3: Enable Multi-Timeframe Mode (Optional but Recommended)

Edit `config/signals.json`:

```json
{
  "enabled": true,
  "mode": "hybrid",

  "multi_timeframe": {
    "enabled": true,
    "trading_style": "MidFrequency",
    "timeframes": [
      {"timeframe": "M1", "enabled": true, "weight": "0.05"},
      {"timeframe": "M5", "enabled": true, "weight": "0.08"},
      {"timeframe": "M15", "enabled": true, "weight": "0.10"},
      {"timeframe": "M30", "enabled": true, "weight": "0.12"},
      {"timeframe": "H1", "enabled": true, "weight": "0.20"},
      {"timeframe": "H2", "enabled": true, "weight": "0.12"},
      {"timeframe": "H4", "enabled": true, "weight": "0.18"},
      {"timeframe": "H6", "enabled": true, "weight": "0.15"}
    ],
    "aggregation": {
      "weight_mode": "fixed",
      "min_timeframe_agreement": "0.50",
      "require_higher_tf_alignment": true,
      "direction_timeframes": ["H4", "H6"]
    }
  },

  "websocket": {
    "enabled": true,
    "binance_ws_url": "wss://stream.binance.com:9443",
    "reconnect_delay_ms": 5000,
    "max_reconnect_attempts": 10,
    "ping_interval_seconds": 30,
    "subscriptions": {
      "klines": ["M1", "M5", "M15", "M30"],
      "depth": true,
      "trades": true
    }
  }
}
```

### Step 4: Run the Bot

```bash
# Build in release mode
cargo build --release

# Run with INFO level logging
RUST_LOG=info ./target/release/leverage-bot

# Or with more detailed DEBUG logging
RUST_LOG=debug ./target/release/leverage-bot
```

---

## What Happens in Paper Trading Mode

### 1. Startup Validation (First 30 seconds)

The bot will:
```
╔═══════════════════════════════════════════════════════════════╗
║   LEVERAGE BOT v0.1.0 - DRY RUN MODE - BSC                    ║
╚═══════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════
          STARTUP VALIDATION - DATA SOURCE CHECK
═══════════════════════════════════════════════════════════════
✓ OHLCV data validated
✓ Current price validated
✓ Order book validated
✓ Recent trades validated
✓ Funding rate validated
✓ Open Interest validated
✓ Long/Short Ratio validated
───────────────────────────────────────────────────────────────
RESULT: 7/7 data sources available - ALL CRITICAL CHECKS PASSED
═══════════════════════════════════════════════════════════════
```

### 2. Signal Generation

The bot monitors market conditions every 30 seconds (MidFrequency) or 5 seconds (Scalping):

```
Multi-TF Signal Detected:
- Direction: LONG
- Confidence: 0.68
- Agreeing Timeframes: 6/8
- Entry Price: 623.45 USD
- Strategy: MidFrequency
```

### 3. Position Opening (Simulated)

When a high-confidence signal is detected:

```
╔══════════════════════════════════════════════════════════════════╗
║              PAPER TRADE: POSITION OPENED                        ║
╠══════════════════════════════════════════════════════════════════╣
║ Direction: Long
║ Entry Price: 623.45 USD
║ Position Size: 500.00 USD
║ Leverage: 3.00x
║ ─────────────────────────────────────────────────────────────── ║
║ Risk Management:
║   Take Profit: 660.85 USD (+6.00%)
║   Stop Loss: 610.98 USD (-2.00%)
║   Liquidation Price: 529.93 USD
║   Health Factor: 2.150
║ ─────────────────────────────────────────────────────────────── ║
║ Portfolio:
║   Available Cash: 0.00 USD
║   Total Equity: 500.00 USD
║   Total Positions: 1
╚══════════════════════════════════════════════════════════════════╝
```

**What Actually Happens:**
- ✅ Bot fetches real quotes from 1inch/OpenOcean/ParaSwap
- ✅ Bot simulates the transaction via `eth_call` (gas estimation)
- ✅ Bot calculates realistic slippage (0.3% base + market impact)
- ✅ Bot tracks the position in SQLite database
- ✅ Bot creates periodic snapshots (every 5 minutes)
- ❌ NO blockchain transaction is submitted
- ❌ NO gas costs incurred

### 4. Real-Time Position Monitoring

Every 30 seconds, the bot:
- Fetches current BNBUSDT price from Binance
- Calculates mark-to-market unrealized P&L
- Checks if take profit reached
- Checks if stop loss triggered
- Monitors health factor for liquidation risk
- Snapshots position state to database

```
Position Update:
- Current Price: 635.20 USD
- Unrealized P&L: +58.75 USD (+11.75%)
- Health Factor: 2.18
- Time Held: 2.3 hours
```

### 5. Position Closing (Take Profit Example)

```
╔══════════════════════════════════════════════════════════════════╗
║              PAPER TRADE: POSITION CLOSED                        ║
╠══════════════════════════════════════════════════════════════════╣
║ Direction: Long
║ Entry Price: 623.45 USD
║ Exit Price: 661.20 USD
║ ─────────────────────────────────────────────────────────────── ║
║ Performance:
║   Realized P&L: +30.00 USD (+6.00%)
║   Hold Duration: 3.5 hours
║   Exit Reason: take_profit
║ ─────────────────────────────────────────────────────────────── ║
║ Portfolio:
║   Available Cash: 530.00 USD
║   Total Equity: 530.00 USD
║   Cumulative P&L: +30.00 USD
║   ROE: +6.00%
║   Current Drawdown: 0.00%
║   Max Drawdown: 0.00%
╚══════════════════════════════════════════════════════════════════╝
```

### 6. Performance Statistics

After 10+ trades, check `data/positions.db`:

```sql
-- View all closed positions
SELECT * FROM positions WHERE close_timestamp IS NOT NULL;

-- View position snapshots (tick-by-tick)
SELECT * FROM position_snapshots ORDER BY timestamp DESC;
```

Or request a performance report in the logs.

---

## Key Features of Paper Trading System

### 1. Realistic Slippage Simulation

The bot simulates realistic trading conditions:
- **Base Slippage**: 0.3% (30 basis points)
- **Market Impact**: Increases linearly with order size
- **Formula**: `Total Slippage = 0.3% + 0.1% × (Size / $100k)`

Example:
- $500 order: 0.3% slippage
- $5,000 order: 0.35% slippage
- $50,000 order: 0.8% slippage

### 2. Take Profit / Stop Loss

**Default Settings** (configurable in paper_trading.rs):
- Stop Loss: 2% below entry (LONG) or above entry (SHORT)
- Take Profit: 6% above entry (LONG) or below entry (SHORT)
- Risk-Reward Ratio: 1:3

**Automatic Triggers:**
- ✅ TP hit → Position closed with profit
- ✅ SL hit → Position closed with loss
- ✅ HF < 1.5 → Position closed (liquidation risk)

### 3. Position Sizing

Based on $500 starting capital:
- **1x Leverage**: $500 max position
- **2x Leverage**: $1,000 max position
- **3x Leverage**: $1,500 max position (configured max)

The bot enforces:
- `max_flash_loan_usd`: $1,500 (prevents over-leveraging)
- `max_position_usd`: $1,500
- `min_health_factor`: 1.5 (safety buffer)

### 4. Performance Metrics

The bot calculates academic-grade risk-adjusted metrics:

**Sharpe Ratio** (risk-adjusted return):
```
Sharpe = Mean(Returns) / StdDev(Returns)
```
- > 1.0 = Good
- > 2.0 = Very Good
- > 3.0 = Excellent

**Sortino Ratio** (downside risk):
```
Sortino = Mean(Returns) / Downside_Deviation
```
- Only penalizes negative volatility
- Better for leveraged strategies
- Higher is better

**Calmar Ratio** (drawdown-adjusted):
```
Calmar = Average_Return / Max_Drawdown
```
- Measures worst-case scenario
- Critical for leverage trading
- > 1.0 = acceptable
- > 3.0 = excellent

**Maximum Drawdown**:
```
Max_DD = (Peak_Equity - Trough_Equity) / Peak_Equity
```
- Measures largest capital loss from peak
- Essential for position sizing
- < 20% = good risk management
- > 50% = very risky

### 5. Real-Time Data Sources

**Market Data (Binance):**
- OHLCV klines (8 timeframes)
- Current price (5s TTL)
- Order book depth (5s TTL)
- Recent trades (10s TTL)
- Funding rate (5min TTL)
- Open interest (1min TTL)
- Long/Short ratio (1min TTL)

**On-Chain Data (Aave V3 BSC):**
- User account state
- Borrow rates (real-time APR)
- Health factor calculation
- Liquidation thresholds

**All data is REAL** - only the transactions are simulated.

---

## Monitoring Your Paper Trading

### Method 1: Real-Time Logs

```bash
# Follow logs in real-time
tail -f logs/bot-$(date +%Y-%m-%d).json | jq '.fields'

# Filter for position events
tail -f logs/bot-$(date +%Y-%m-%d).json | jq 'select(.fields.message | contains("PAPER TRADE"))'
```

### Method 2: Database Queries

```bash
# Install sqlite3
sudo apt install sqlite3

# Query positions
sqlite3 data/positions.db "SELECT * FROM positions ORDER BY id DESC LIMIT 10;"

# Query snapshots
sqlite3 data/positions.db "SELECT * FROM position_snapshots ORDER BY timestamp DESC LIMIT 20;"

# Calculate win rate
sqlite3 data/positions.db "
  SELECT
    COUNT(*) as total_trades,
    SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as winning_trades,
    ROUND(AVG(CASE WHEN realized_pnl_usd > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate_pct
  FROM positions
  WHERE close_timestamp IS NOT NULL;
"
```

### Method 3: Performance Report

The bot generates comprehensive reports showing:
- Portfolio summary (starting capital, current equity, ROE)
- Trading statistics (total trades, win rate, avg P&L)
- Risk metrics (Sharpe, Sortino, Calmar, max drawdown)

---

## Testing Checklist

Before going live, verify:

- [ ] Bot connects to all data sources ✓
- [ ] Signals are generated with reasonable frequency
- [ ] Positions open with correct direction (LONG/SHORT)
- [ ] Take profit triggers correctly
- [ ] Stop loss triggers correctly
- [ ] Health factor monitoring works
- [ ] P&L calculations are accurate
- [ ] Performance metrics make sense
- [ ] No crashes or errors over 24+ hours
- [ ] Win rate > 50% (ideally 60%+)
- [ ] Sharpe ratio > 1.0
- [ ] Max drawdown < 20%

**Recommended Testing Duration:**
- Minimum: 7 days (1 week of data)
- Ideal: 30 days (1 month)
- Conservative: 90 days (1 quarter)

---

## Transitioning to Live Trading

**When paper trading shows consistent profitability:**

1. **Review Results:**
   - Sharpe ratio > 1.5
   - Win rate > 55%
   - Max drawdown < 15%
   - Consistent performance across different market conditions

2. **Set Up Live Environment:**
   ```bash
   # .env changes
   EXECUTOR_DRY_RUN=false
   USER_WALLET_ADDRESS=0xYourRealWalletAddress
   EXECUTOR_PRIVATE_KEY=0xYourRealPrivateKey
   LEVERAGE_EXECUTOR_ADDRESS=0xDeployedContractAddress
   ONEINCH_API_KEY=your_real_api_key
   ```

3. **Start Small:**
   - Begin with $100-500 real capital
   - Use lower leverage (1-2x instead of 3x)
   - Monitor closely for first week

4. **Scale Gradually:**
   - Increase capital only after proven success
   - Never risk more than 10% of portfolio per trade
   - Maintain emergency exit plan

---

## Troubleshooting

### "No signals generated after 1 hour"

**Possible causes:**
- Market is ranging (no clear trend)
- Confidence threshold too high
- Timeframe alignment too strict

**Solutions:**
```json
// config/signals.json - reduce entry requirements
{
  "entry_rules": {
    "min_confidence": "0.60",  // Lower from 0.70
    "require_trend_alignment": false  // Disable if needed
  }
}
```

### "All positions hit stop loss"

**Possible causes:**
- Stop loss too tight (2% may be too small for volatile assets)
- Strategy needs optimization
- Market conditions unfavorable

**Solutions:**
```bash
# Increase stop loss in paper_trading.rs
default_stop_loss_pct: dec!(3.0),  // 3% instead of 2%
default_take_profit_pct: dec!(9.0), // 9% to maintain 1:3 R:R
```

### "Health factor keeps triggering liquidation risk"

**Possible causes:**
- Leverage too high
- Volatile asset (BNB has high volatility)
- Insufficient collateral buffer

**Solutions:**
```json
// config/positions.json
{
  "max_leverage_ratio": "2.0",  // Reduce from 3.0
  "min_health_factor": "1.8"     // Increase from 1.5
}
```

---

## Academic References

This paper trading system implements best practices from:

1. **Event-Driven Backtesting** (QuantStart, 2024)
   - FIFO event queue
   - Prevents lookahead bias

2. **Slippage Modeling** (QuantJourney, 2024)
   - Multi-factor slippage model
   - Market impact calculation

3. **Risk-Adjusted Metrics** (Sharpe 1966, Sortino 1980, Calmar 1991)
   - Sharpe ratio (total volatility)
   - Sortino ratio (downside risk)
   - Calmar ratio (drawdown risk)

4. **Aave V3 Liquidation Mechanics** (Aave Documentation, 2024)
   - Health factor calculation
   - Liquidation threshold enforcement

5. **Position Sizing** (Ralph Vince, "The Mathematics of Money Management", 1992)
   - Kelly criterion
   - Fixed fractional position sizing

---

## Support

For issues or questions:
1. Check logs in `logs/bot-YYYY-MM-DD.json`
2. Query database: `sqlite3 data/positions.db`
3. Review this guide's troubleshooting section
4. Check GitHub issues: https://github.com/anthropics/claude-code/issues

**Remember:** Paper trading is for testing only. Past performance does not guarantee future results. Always start with small amounts in live trading and never risk more than you can afford to lose.
