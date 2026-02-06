# Quick Start Guide - Phase 1 ML

Get the ML system running in 5 minutes.

---

## Prerequisites

- Python 3.10 or 3.11
- 8GB RAM minimum
- Internet connection (for data download)

---

## Option 1: Development Mode (Recommended for Testing)

### Step 1: Start Services

```bash
cd ml/
chmod +x scripts/start_services.sh
./scripts/start_services.sh
```

This will:
- Create virtual environment
- Install dependencies
- Start ML service on http://localhost:8000

### Step 2: Download Data & Train Model

In a new terminal:

```bash
cd ml/
source ml_env/bin/activate

# Download 60 days of WBNB/USDT 1-minute data (~86,400 candles)
python scripts/download_data.py --symbol WBNBUSDT --interval 1m --days 60

# Train XGBoost model
python scripts/train_xgboost.py --config configs/xgboost_baseline.yaml
```

Training takes ~5-10 minutes. You'll see:
```
Train set: 69,120 samples
Test set: 17,280 samples
Training XGBoost model...
Best iteration: 245
Best score: 0.6125

=== Model Performance ===
accuracy: 0.6050
win_rate: 0.6050
sharpe_ratio: 1.75
```

### Step 3: Test Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "price": 625.50,
    "volume": 145230.5,
    "bid_volume": 72500.0,
    "ask_volume": 68200.0
  }'
```

Expected response:
```json
{
  "direction": "LONG",
  "confidence": 0.72,
  "raw_probability": 0.86,
  "should_trade": true,
  "recommended_size": 25000.0,
  "model_version": "xgboost_phase1_v1",
  "latency_ms": 6.5
}
```

---

## Option 2: Docker Deployment

### Step 1: Build & Run

```bash
cd ml/

# Build image
docker build -t leveragebot-ml:latest .

# Run container (requires pre-trained model)
docker run -d \
  --name leveragebot-ml \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/xgboost_phase1_v1.pkl \
  leveragebot-ml:latest
```

### Step 2: Check Health

```bash
curl http://localhost:8000/health
```

---

## Option 3: Docker Compose (Full Stack)

Includes ML service + MLflow + Prometheus + Grafana:

```bash
cd ml/
docker-compose up -d
```

Access services:
- ML API: http://localhost:8000
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

---

## Integrate with Rust Bot

### Step 1: Update Bot Configuration

Edit `config/app.json`:

```json
{
  "ml_service": {
    "enabled": true,
    "base_url": "http://localhost:8000",
    "timeout_ms": 100,
    "confidence_threshold": 0.55,
    "weight": 0.4
  }
}
```

### Step 2: Build Bot with ML Client

```bash
cd crates/bot/
cargo build --release
```

### Step 3: Run Bot (Dry-Run Mode)

```bash
EXECUTOR_PRIVATE_KEY=<random_key> \
SAFETY_DRY_RUN=true \
cargo run --release
```

Look for log messages:
```
INFO  ML service healthy - model: xgboost_phase1_v1
INFO  ML signal: LONG (confidence: 0.72, latency: 6.5ms)
```

---

## Troubleshooting

### Issue: "Model not found"

**Solution:** Train a model first:
```bash
python scripts/download_data.py --symbol WBNBUSDT --interval 1m --days 60
python scripts/train_xgboost.py --config configs/xgboost_baseline.yaml
```

### Issue: "Connection refused"

**Solution:** Ensure ML service is running:
```bash
curl http://localhost:8000/health
```

### Issue: "Latency too high"

**Solutions:**
1. Use fewer features (feature selection)
2. Run on faster hardware
3. Increase timeout in config
4. Check system load

---

## Performance Targets

After training, verify you meet Phase 1 targets:

| Metric | Target | How to Check |
|--------|--------|-------------|
| Win Rate | 58-62% | Training output: `win_rate` |
| Sharpe Ratio | 1.5-2.0 | Training output: `sharpe_ratio` |
| Inference Latency | <10ms | API response: `latency_ms` |

---

## Next Steps

1. **Backtest:** `python scripts/backtest.py --model models/xgboost_phase1_v1.pkl --data data/test/`
2. **Monitor:** Access API docs at http://localhost:8000/docs
3. **Optimize:** Tune hyperparameters or add features
4. **Deploy:** Move to production with Docker

---

**Estimated Time:** 15-20 minutes (including data download and training)

**Need Help?** See `INTEGRATION_GUIDE.md` for detailed integration steps.
