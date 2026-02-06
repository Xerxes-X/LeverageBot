#!/usr/bin/env python3
"""
TimescaleDB Schema Setup for LeverageBot ML Pipeline

Creates hypertables for:
- OHLCV data (1-minute candles)
- ML features (candlestick + volume features)
- Labeled trades (training data)
- Model performance tracking

Usage:
    python scripts/setup_timescaledb.py
"""

import sys
from sqlalchemy import create_engine, text
import os

# Database connection
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASS = os.getenv('POSTGRES_PASSWORD', 'password')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'leverage_bot')

engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

def setup_ohlcv_table():
    """Create hypertable for OHLCV data"""
    print("Setting up OHLCV hypertable...")

    with engine.connect() as conn:
        # Create table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                open NUMERIC(20, 8),
                high NUMERIC(20, 8),
                low NUMERIC(20, 8),
                close NUMERIC(20, 8),
                volume NUMERIC(30, 8)
            );
        """))
        conn.commit()

        # Convert to hypertable (only if not already)
        try:
            conn.execute(text("SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);"))
            conn.commit()
            print("  ✓ Created hypertable 'ohlcv'")
        except Exception as e:
            print(f"  ℹ Hypertable already exists: {e}")

        # Create index
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv (symbol, time DESC);"))
        conn.commit()
        print("  ✓ Created index on (symbol, time)")

        # Create continuous aggregate for 5-minute EMA
        try:
            conn.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS ema_5m
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('5 minutes', time) AS bucket,
                    symbol,
                    last(close, time) AS close
                FROM ohlcv
                GROUP BY bucket, symbol;
            """))
            conn.commit()
            print("  ✓ Created continuous aggregate 'ema_5m'")

            # Add refresh policy
            conn.execute(text("""
                SELECT add_continuous_aggregate_policy('ema_5m',
                    start_offset => INTERVAL '1 hour',
                    end_offset => INTERVAL '1 minute',
                    schedule_interval => INTERVAL '1 minute',
                    if_not_exists => TRUE);
            """))
            conn.commit()
            print("  ✓ Added continuous aggregate refresh policy")
        except Exception as e:
            print(f"  ℹ Continuous aggregate already exists: {e}")

def setup_ml_features_table():
    """Create hypertable for ML features"""
    print("\nSetting up ML features hypertable...")

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ml_features (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                -- Candlestick features (12)
                body_size NUMERIC,
                body_position NUMERIC,
                upper_wick_ratio NUMERIC,
                lower_wick_ratio NUMERIC,
                is_doji INTEGER,
                is_hammer INTEGER,
                is_shooting_star INTEGER,
                candles_above_ema20 NUMERIC,
                candles_below_ema20 NUMERIC,
                ema_slope NUMERIC,
                trigram_id INTEGER,
                -- Volume features (6)
                volume_ratio NUMERIC,
                buying_pressure NUMERIC,
                selling_pressure NUMERIC,
                volume_trend NUMERIC,
                volume_volatility NUMERIC,
                volume_price_correlation NUMERIC
            );
        """))
        conn.commit()

        try:
            conn.execute(text("SELECT create_hypertable('ml_features', 'time', if_not_exists => TRUE);"))
            conn.commit()
            print("  ✓ Created hypertable 'ml_features'")
        except Exception as e:
            print(f"  ℹ Hypertable already exists: {e}")

        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ml_features_symbol_time ON ml_features (symbol, time DESC);"))
        conn.commit()
        print("  ✓ Created index on (symbol, time)")

def setup_labeled_trades_table():
    """Create table for labeled trades (training data)"""
    print("\nSetting up labeled_trades table...")

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS labeled_trades (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price NUMERIC(20, 8),
                exit_price NUMERIC(20, 8),
                label INTEGER,  -- 1 (Win), -1 (Loss), 0 (Hold)
                pnl_pct NUMERIC,
                regime TEXT,    -- 'trending', 'mean_reverting', 'random_walk'
                PRIMARY KEY (timestamp, symbol, direction)
            );
        """))
        conn.commit()
        print("  ✓ Created table 'labeled_trades'")

        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_labeled_trades_timestamp ON labeled_trades (timestamp DESC);"))
        conn.commit()
        print("  ✓ Created index on timestamp")

def setup_model_performance_table():
    """Create table for model performance tracking"""
    print("\nSetting up model_performance table...")

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_performance (
                date DATE NOT NULL,
                model_version TEXT NOT NULL,
                ic NUMERIC,  -- Information Coefficient
                sharpe NUMERIC,
                win_rate NUMERIC,
                trades_count INTEGER,
                PRIMARY KEY (date, model_version)
            );
        """))
        conn.commit()
        print("  ✓ Created table 'model_performance'")

def verify_setup():
    """Verify all tables and hypertables were created"""
    print("\nVerifying setup...")

    with engine.connect() as conn:
        # Check hypertables
        result = conn.execute(text("""
            SELECT hypertable_name, num_dimensions
            FROM timescaledb_information.hypertables;
        """))

        hypertables = result.fetchall()
        print(f"\n  Hypertables created: {len(hypertables)}")
        for ht in hypertables:
            print(f"    - {ht[0]} ({ht[1]} dimensions)")

        # Check continuous aggregates
        result = conn.execute(text("""
            SELECT view_name, refresh_interval
            FROM timescaledb_information.continuous_aggregates;
        """))

        caggs = result.fetchall()
        print(f"\n  Continuous aggregates: {len(caggs)}")
        for ca in caggs:
            print(f"    - {ca[0]} (refresh: {ca[1]})")

        # Check regular tables
        result = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('labeled_trades', 'model_performance');
        """))

        tables = result.fetchall()
        print(f"\n  Regular tables: {len(tables)}")
        for tbl in tables:
            print(f"    - {tbl[0]}")

def main():
    print("=" * 60)
    print("LeverageBot ML Pipeline - TimescaleDB Setup")
    print("=" * 60)

    try:
        # Verify TimescaleDB extension is enabled
        with engine.connect() as conn:
            result = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"))
            version = result.fetchone()

            if version is None:
                print("\n❌ ERROR: TimescaleDB extension not enabled!")
                print("Run: sudo -u postgres psql -d leverage_bot -c \"CREATE EXTENSION timescaledb;\"")
                sys.exit(1)
            else:
                print(f"\n✓ TimescaleDB version: {version[0]}\n")

        # Create all tables
        setup_ohlcv_table()
        setup_ml_features_table()
        setup_labeled_trades_table()
        setup_model_performance_table()

        # Verify
        verify_setup()

        print("\n" + "=" * 60)
        print("✅ TimescaleDB setup complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run: python scripts/test_postgres_connection.py")
        print("  2. Import historical data: python scripts/import_binance_ohlcv.py")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
