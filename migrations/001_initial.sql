-- Initial schema for BSC Leverage Bot P&L tracking.
-- All financial values stored as TEXT for Decimal precision.

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    direction TEXT NOT NULL,
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
    close_reason TEXT
);

CREATE TABLE IF NOT EXISTS position_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER NOT NULL REFERENCES positions(id),
    timestamp INTEGER NOT NULL,
    collateral_value_usd TEXT NOT NULL,
    debt_value_usd TEXT NOT NULL,
    health_factor TEXT NOT NULL,
    borrow_rate_apr TEXT NOT NULL,
    unrealized_pnl_usd TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER REFERENCES positions(id),
    tx_hash TEXT NOT NULL UNIQUE,
    timestamp INTEGER NOT NULL,
    tx_type TEXT NOT NULL,
    gas_used INTEGER,
    gas_price_gwei TEXT,
    gas_cost_usd TEXT,
    success BOOLEAN NOT NULL,
    revert_reason TEXT
);
