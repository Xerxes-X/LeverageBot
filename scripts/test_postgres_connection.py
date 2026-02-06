#!/usr/bin/env python3
"""
Test PostgreSQL + TimescaleDB Connection

Verifies PostgreSQL database is accessible and TimescaleDB extension is enabled.

Usage:
    python scripts/test_postgres_connection.py
"""

import sys
import os

try:
    from sqlalchemy import create_engine, text
except ImportError:
    print("❌ ERROR: sqlalchemy not installed!")
    print("Run: pip install sqlalchemy==2.0.36 psycopg2-binary==2.9.10")
    sys.exit(1)

# Database connection settings
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASS = os.getenv('POSTGRES_PASSWORD', 'password')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'leverage_bot')

def test_connection():
    """Test basic PostgreSQL connection"""
    print("Testing PostgreSQL connection...")

    try:
        engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"  ✓ PostgreSQL version: {version.split(',')[0]}")

        return engine

    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check if PostgreSQL is running:")
        print(f"     sudo systemctl status postgresql")
        print(f"  2. Verify database exists:")
        print(f"     sudo -u postgres psql -c '\\l'")
        print(f"  3. Create database if missing:")
        print(f"     sudo -u postgres createdb {DB_NAME}")
        print(f"  4. Check connection string:")
        print(f"     postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        sys.exit(1)

def test_timescaledb(engine):
    """Test TimescaleDB extension"""
    print("\nTesting TimescaleDB extension...")

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"))
            version = result.fetchone()

            if version is None:
                print(f"  ❌ TimescaleDB extension NOT enabled!")
                print(f"\nRun:")
                print(f"  sudo -u postgres psql -d {DB_NAME} -c \"CREATE EXTENSION timescaledb;\"")
                return False
            else:
                print(f"  ✓ TimescaleDB version: {version[0]}")

        return True

    except Exception as e:
        print(f"  ❌ TimescaleDB check failed: {e}")
        return False

def test_hypertables(engine):
    """Test existing hypertables"""
    print("\nChecking hypertables...")

    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT hypertable_name, num_dimensions, num_chunks
                FROM timescaledb_information.hypertables;
            """))

            hypertables = result.fetchall()

            if len(hypertables) == 0:
                print(f"  ℹ No hypertables found (run setup_timescaledb.py first)")
            else:
                print(f"  ✓ Found {len(hypertables)} hypertables:")
                for ht in hypertables:
                    print(f"    - {ht[0]} ({ht[1]} dimensions, {ht[2]} chunks)")

        return True

    except Exception as e:
        print(f"  ❌ Hypertable check failed: {e}")
        return False

def test_data(engine):
    """Test data in OHLCV table"""
    print("\nChecking OHLCV data...")

    try:
        with engine.connect() as conn:
            # Check if table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name = 'ohlcv'
                );
            """))

            if not result.fetchone()[0]:
                print(f"  ℹ OHLCV table does not exist yet (run setup_timescaledb.py)")
                return True

            # Count rows
            result = conn.execute(text("SELECT COUNT(*) FROM ohlcv;"))
            count = result.fetchone()[0]
            print(f"  ✓ OHLCV rows: {count:,}")

            if count > 0:
                # Get date range
                result = conn.execute(text("""
                    SELECT MIN(time), MAX(time), COUNT(DISTINCT symbol)
                    FROM ohlcv;
                """))
                min_time, max_time, symbols = result.fetchone()
                print(f"  ✓ Date range: {min_time} to {max_time}")
                print(f"  ✓ Symbols: {symbols}")

        return True

    except Exception as e:
        print(f"  ❌ Data check failed: {e}")
        return False

def main():
    print("=" * 60)
    print("LeverageBot ML Pipeline - PostgreSQL Connection Test")
    print("=" * 60)

    engine = test_connection()
    timescaledb_ok = test_timescaledb(engine)

    if not timescaledb_ok:
        print("\n❌ TimescaleDB extension test FAILED!")
        print("=" * 60)
        sys.exit(1)

    hypertables_ok = test_hypertables(engine)
    data_ok = test_data(engine)

    print("\n" + "=" * 60)
    if timescaledb_ok and hypertables_ok and data_ok:
        print("✅ PostgreSQL + TimescaleDB connection test PASSED!")
        print("=" * 60)
        print("\nDatabase is ready for ML pipeline.")
        print("\nNext steps:")
        if hypertables_ok:
            print("  1. Import historical OHLCV data: python scripts/import_binance_ohlcv.py")
            print("  2. Generate labeled trades: python scripts/generate_labeled_trades.py")
        else:
            print("  1. Run schema setup: python scripts/setup_timescaledb.py")
            print("  2. Import historical data: python scripts/import_binance_ohlcv.py")
    else:
        print("❌ PostgreSQL connection test FAILED!")
        print("=" * 60)
        print("\nFix the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
