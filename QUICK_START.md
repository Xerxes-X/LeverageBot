# Quick Start - Paper Trading Mode

## Setup (5 Minutes)

### 1. Configure `.env`
```bash
EXECUTOR_DRY_RUN=true
BSC_RPC_URL_HTTP=https://bsc-dataseed1.binance.org/
BSC_RPC_URL_MEV_PROTECTED=https://rpc.48.club

# Leave empty for testing
USER_WALLET_ADDRESS=
EXECUTOR_PRIVATE_KEY=
LEVERAGE_EXECUTOR_ADDRESS=
```

### 2. Verify `config/positions.json`
```json
{
  "dry_run": true,
  "max_flash_loan_usd": 1500,
  "max_leverage_ratio": "3.0",
  "min_health_factor": "1.5"
}
```

### 3. Run the Bot
```bash
cargo build --release
RUST_LOG=info ./target/release/leverage-bot
```

---

## What to Expect

### Startup (First 30 seconds)
```
╔═══════════════════════════════════════════════════════════════╗
║   LEVERAGE BOT v0.1.0 - DRY RUN MODE - BSC                    ║
╚═══════════════════════════════════════════════════════════════╝

✓ OHLCV data validated
✓ Current price validated
✓ Order book validated
ALL CRITICAL CHECKS PASSED
```

### When Signal Detected
```
╔══════════════════════════════════════════════════════════════════╗
║              PAPER TRADE: POSITION OPENED                        ║
╠══════════════════════════════════════════════════════════════════╣
║ Direction: Long
║ Entry Price: 623.45 USD
║ Position Size: 500.00 USD
║ Leverage: 3.00x
║ ─────────────────────────────────────────────────────────────── ║
║ Take Profit: 660.85 USD (+6.00%)
║ Stop Loss: 610.98 USD (-2.00%)
║ Health Factor: 2.150
╚══════════════════════════════════════════════════════════════════╝
```

### Real-Time Monitoring (Every 30s)
```
Position Update:
- Current Price: 635.20 USD
- Unrealized P&L: +58.75 USD (+11.75%)
- Health Factor: 2.18
```

### Position Close
```
╔══════════════════════════════════════════════════════════════════╗
║              PAPER TRADE: POSITION CLOSED                        ║
╠══════════════════════════════════════════════════════════════════╣
║ Realized P&L: +30.00 USD (+6.00%)
║ Exit Reason: take_profit
║ Portfolio Equity: 530.00 USD
║ ROE: +6.00%
╚══════════════════════════════════════════════════════════════════╝
```

---

## What Gets Tracked

### Real Market Data ✓
- Live BNBUSDT price from Binance
- Order book depth (20 levels)
- Recent trades (100)
- Funding rates
- Open interest
- Long/Short ratios

### Simulated with $500 ✓
- Position entries (LONG/SHORT)
- Take profit triggers (+6%)
- Stop loss triggers (-2%)
- Liquidation risk monitoring
- Mark-to-market P&L
- Portfolio equity

### Performance Metrics ✓
- Win rate (%)
- Sharpe ratio (risk-adjusted return)
- Sortino ratio (downside risk)
- Calmar ratio (drawdown-adjusted)
- Maximum drawdown (%)
- Average hold time (hours)

### NOT Executed ✗
- Blockchain transactions
- Gas costs
- Real flash loans
- Actual DEX swaps

---

## Monitoring

### Watch Logs
```bash
# Real-time logs
tail -f logs/bot-$(date +%Y-%m-%d).json | jq '.fields.message'

# Filter position events
tail -f logs/bot-*.json | jq 'select(.fields.message | contains("PAPER TRADE"))'
```

### Query Database
```bash
sqlite3 data/positions.db "SELECT * FROM positions ORDER BY id DESC LIMIT 5;"
```

### Check Performance
```bash
sqlite3 data/positions.db "
  SELECT
    COUNT(*) as trades,
    SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
    ROUND(AVG(realized_pnl_usd), 2) as avg_pnl
  FROM positions
  WHERE close_timestamp IS NOT NULL;
"
```

---

## Expected Results (After 7-30 Days)

### Good Performance
- ✅ Win rate: 55-70%
- ✅ Sharpe ratio: > 1.5
- ✅ Max drawdown: < 20%
- ✅ Positive cumulative P&L

### Red Flags
- ❌ Win rate < 50%
- ❌ Sharpe ratio < 0.5
- ❌ Max drawdown > 30%
- ❌ Frequent stop losses

---

## Files to Review

1. **PAPER_TRADING_SETUP.md** - Comprehensive guide (20 pages)
2. **IMPLEMENTATION_SUMMARY.md** - Technical details (15 pages)
3. **logs/bot-YYYY-MM-DD.json** - Daily logs
4. **data/positions.db** - All trade history

---

## Troubleshooting

**No signals after 1 hour?**
```json
// config/signals.json - lower threshold
{
  "entry_rules": {
    "min_confidence": "0.60"  // Was 0.70
  }
}
```

**Too many stop losses?**
```
// Edit paper_trading.rs line 68-69
default_stop_loss_pct: dec!(3.0),    // Was 2.0
default_take_profit_pct: dec!(9.0),  // Was 6.0 (maintains 1:3 R:R)
```

**Health factor warnings?**
```json
// config/positions.json
{
  "max_leverage_ratio": "2.0",  // Was 3.0
  "min_health_factor": "1.8"     // Was 1.5
}
```

---

## Next Steps

1. ✅ Run for 7-30 days
2. ✅ Collect 20+ trades
3. ✅ Review metrics
4. ✅ Optimize if needed
5. ⚠️  Consider live trading (start with $100-500)

**Remember**: Paper trading is risk-free testing. Live trading involves real financial risk. Never risk more than you can afford to lose.

---

## Support

- Logs: `logs/bot-YYYY-MM-DD.json`
- Database: `sqlite3 data/positions.db`
- Documentation: `PAPER_TRADING_SETUP.md`
- Issues: https://github.com/anthropics/claude-code/issues
