# LeverageBot ML Pipeline Scripts

This directory contains Python helper scripts for setting up and managing the ML infrastructure (MCP servers, TimescaleDB, Chroma).

## Setup Scripts

### `setup_timescaledb.py`
Creates TimescaleDB schema for ML pipeline:
- Creates hypertables for OHLCV data
- Creates hypertables for ML features
- Creates tables for labeled trades and model performance
- Sets up continuous aggregates for EMA calculation

**Usage:**
```bash
python scripts/setup_timescaledb.py
```

**Prerequisites:**
- PostgreSQL 14+ installed
- TimescaleDB extension installed
- Database `leverage_bot` created

---

## Testing Scripts

### `test_postgres_connection.py`
Tests PostgreSQL + TimescaleDB connectivity:
- Verifies database connection
- Checks TimescaleDB extension is enabled
- Lists existing hypertables
- Shows OHLCV data statistics

**Usage:**
```bash
python scripts/test_postgres_connection.py
```

### `test_chroma_connection.py`
Tests Chroma MCP server connectivity:
- Tests REST API connection (MCP server)
- Tests persistent client connection
- Creates/queries test collection
- Verifies vector database is operational

**Usage:**
```bash
python scripts/test_chroma_connection.py
```

---

## Multi-Computer Sync Scripts

### `export_mcp_data.py`
Exports MCP data for syncing to second computer:
- Exports PostgreSQL tables (OHLCV, features, labeled trades) to SQL dump
- Exports Chroma collections to JSON
- Creates sync manifest with timestamps

**Usage:**
```bash
# On main computer
python scripts/export_mcp_data.py

# Commit to Git
git add mcp_data/exports/*
git commit -m "MCP data export - $(date)"
git push origin master
```

**Output:**
- `mcp_data/exports/postgres_export_<timestamp>.sql`
- `mcp_data/exports/chroma_export_<timestamp>.json`
- `mcp_data/exports/sync_manifest.json`
- Symlinks: `postgres_latest.sql`, `chroma_latest.json`

### `import_mcp_data.py`
Imports MCP data from main computer:
- Imports PostgreSQL data from SQL dump
- Imports Chroma collections from JSON
- Verifies import success

**Usage:**
```bash
# On second computer
git pull origin master
python scripts/import_mcp_data.py
```

---

## Data Collection Scripts (To Be Created)

### `import_binance_ohlcv.py` (TODO)
Fetches historical OHLCV data from Binance and stores in TimescaleDB:
- Fetches 2 years of 1-minute candles
- Stores in OHLCV hypertable
- Validates data integrity

### `generate_labeled_trades.py` (TODO)
Runs historical simulation to generate labeled trades:
- Simulates SignalEngine logic on historical data
- Labels trades as Win/Loss/Hold based on outcomes
- Stores in `labeled_trades` table

### `feature_engineering.py` (TODO)
Extracts ML features from OHLCV data:
- Computes candlestick morphology features (12 features)
- Computes volume features (6 features)
- Stores in `ml_features` hypertable

---

## Training Scripts (To Be Created)

### `train_xgboost.py` (TODO)
Trains XGBoost model with Walk-Forward CV:
- Loads features from TimescaleDB
- Performs Bayesian hyperparameter optimization
- Computes Deflated Sharpe Ratio and PBO
- Exports to ONNX with INT8 quantization

### `train_gaf_cnn.py` (TODO - Phase 2)
Trains GAF-CNN model for candlestick pattern recognition:
- Encodes 20-candle windows as GAF images
- Trains PyTorch CNN
- Stores pattern embeddings in Chroma
- Exports to ONNX

---

## Environment Variables

All scripts support the following environment variables:

### PostgreSQL
```bash
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your_password
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=leverage_bot
```

### Chroma
```bash
export CHROMA_HOST=localhost
export CHROMA_PORT=8000
export CHROMA_PERSIST_DIR="$HOME/LeverageBot/mcp_data/chroma_db"
```

---

## Quick Start

### First Time Setup (Main Computer)

1. Install MCP servers:
   ```bash
   npm install -g @chroma-core/chroma-mcp
   sudo apt install timescaledb-postgresql-14
   ```

2. Create database:
   ```bash
   sudo -u postgres createdb leverage_bot
   sudo -u postgres psql -d leverage_bot -c "CREATE EXTENSION timescaledb;"
   ```

3. Start Chroma:
   ```bash
   chroma-mcp start --persist-directory ~/LeverageBot/mcp_data/chroma_db --port 8000 &
   ```

4. Run setup:
   ```bash
   python scripts/setup_timescaledb.py
   python scripts/test_postgres_connection.py
   python scripts/test_chroma_connection.py
   ```

### Sync to Second Computer

On main computer:
```bash
python scripts/export_mcp_data.py
git add mcp_data/exports/*
git commit -m "MCP data sync"
git push origin master
```

On second computer:
```bash
git pull origin master
python scripts/import_mcp_data.py
python scripts/test_postgres_connection.py
python scripts/test_chroma_connection.py
```

---

## Troubleshooting

### PostgreSQL Connection Failed
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Start PostgreSQL
sudo systemctl start postgresql

# Verify database exists
sudo -u postgres psql -c '\l'
```

### TimescaleDB Extension Not Found
```bash
# Install TimescaleDB
sudo apt install timescaledb-postgresql-14

# Enable extension
sudo -u postgres psql -d leverage_bot -c "CREATE EXTENSION timescaledb;"
```

### Chroma Server Not Running
```bash
# Start Chroma MCP server
chroma-mcp start --persist-directory ~/LeverageBot/mcp_data/chroma_db --port 8000 &

# Verify server is running
curl http://localhost:8000/api/v1/heartbeat
```

### Permission Denied
```bash
# Make scripts executable
chmod +x scripts/*.py

# Or run with python explicitly
python scripts/setup_timescaledb.py
```

---

## Dependencies

Install all required Python packages:

```bash
pip install \
    chromadb==0.5.0 \
    sqlalchemy==2.0.36 \
    psycopg2-binary==2.9.10 \
    pandas==2.2.3 \
    numpy==2.2.1 \
    xgboost==2.1.3 \
    scikit-learn==1.6.1 \
    scikit-optimize==0.10.2 \
    torch==2.5.1 \
    onnx==1.18.0 \
    onnxruntime==1.21.0 \
    skl2onnx==1.18.0 \
    ta==0.11.0 \
    pyts==0.13.0 \
    mlfinlab==2.1.0
```

---

## References

- **MCP Infrastructure Investigation**: `../ML_INFRASTRUCTURE_INVESTIGATION.md`
- **ML Implementation Guide**: `../ML_IMPLEMENTATION_GUIDE.md`
- **Chroma Documentation**: https://docs.trychroma.com/
- **TimescaleDB Documentation**: https://docs.timescale.com/
