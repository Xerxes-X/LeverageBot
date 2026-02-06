# ML Service Integration Guide

This guide shows how to integrate the ML prediction service with the existing Rust bot.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Rust Bot (main.rs)                       │
│                                                             │
│  ┌──────────────┐  HTTP   ┌────────────────────────────┐  │
│  │ SignalEngine │ ←────→  │ MLClient (ml_client.rs)    │  │
│  │  (existing)  │         │                            │  │
│  └──────┬───────┘         └────────────┬───────────────┘  │
│         │                               │                  │
│         │ SignalEvent                   │                  │
│         ↓                               │                  │
│  ┌──────────────┐                       │                  │
│  │   Strategy   │                       │ HTTP POST        │
│  └──────────────┘                       │                  │
└─────────────────────────────────────────┼──────────────────┘
                                          │
                                          ↓
                          ┌───────────────────────────────────┐
                          │  Python ML Service (FastAPI)     │
                          │  Port: 8000                      │
                          │  Endpoint: POST /predict         │
                          └───────────────────────────────────┘
```

---

## Step 1: Add ML Client to Bot

### 1.1 Update `crates/bot/src/lib.rs`

Add the ML client module:

```rust
// In crates/bot/src/lib.rs

pub mod ml_client;  // Add this line

pub mod config;
pub mod types;
pub mod errors;
// ... existing modules
```

### 1.2 Create ML Configuration

Add to `config/app.json`:

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

### 1.3 Update Config Struct

In `crates/bot/src/config/types.rs`, add:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct MLServiceConfig {
    pub enabled: bool,
    pub base_url: String,
    pub timeout_ms: u64,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
    #[serde(default = "default_ml_weight")]
    pub weight: f64,
}

fn default_confidence_threshold() -> f64 { 0.55 }
fn default_ml_weight() -> f64 { 0.4 }

// Add to AppConfig
#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    // ... existing fields
    pub ml_service: Option<MLServiceConfig>,
}
```

---

## Step 2: Integrate with Signal Engine

### 2.1 Update `core/signal_engine.rs`

Add ML client to SignalEngine:

```rust
use crate::ml_client::{MLClient, PredictionRequest};

pub struct SignalEngine {
    // ... existing fields
    ml_client: Option<Arc<MLClient>>,
    ml_config: Option<MLServiceConfig>,
}

impl SignalEngine {
    pub fn new(
        // ... existing params
        ml_client: Option<Arc<MLClient>>,
        ml_config: Option<MLServiceConfig>,
    ) -> Self {
        Self {
            // ... existing fields
            ml_client,
            ml_config,
        }
    }

    async fn gather_tier2_signals(&self) -> Vec<(String, f64, f64)> {
        let mut signals = Vec::new();

        // ... existing tier 2 signal gathering ...

        // Tier 2.5: ML predictions (if enabled)
        if let Some(ml_client) = &self.ml_client {
            if let Some(ml_config) = &self.ml_config {
                match self.get_ml_signal(ml_client, ml_config).await {
                    Ok((source, score, weight)) => {
                        signals.push((source, score, weight));
                    }
                    Err(e) => {
                        warn!("ML signal failed: {}", e);
                        // Continue with other signals (graceful degradation)
                    }
                }
            }
        }

        signals
    }

    async fn get_ml_signal(
        &self,
        ml_client: &MLClient,
        ml_config: &MLServiceConfig
    ) -> Result<(String, f64, f64), BotError> {
        // Prepare market data
        let latest_price = self.data_service.get_latest_price(&self.asset).await?;
        let ohlcv = self.data_service.get_ohlcv(&self.asset, 1).await?;

        // Get order book data (if available)
        let (bid_volume, ask_volume, best_bid, best_ask) =
            match self.data_service.get_order_book(&self.asset).await {
                Ok(ob) => {
                    let bid_vol = ob.bids.iter().map(|b| b.quantity).sum();
                    let ask_vol = ob.asks.iter().map(|a| a.quantity).sum();
                    let best_bid = ob.bids.first().map(|b| b.price);
                    let best_ask = ob.asks.first().map(|a| a.price);
                    (Some(bid_vol), Some(ask_vol), best_bid, best_ask)
                }
                Err(_) => (None, None, None, None)
            };

        // Get BTC price for correlation
        let btc_price = self.data_service.get_latest_price("BTCUSDT").await.ok();

        // Create prediction request
        let request = PredictionRequest {
            price: latest_price,
            open: ohlcv.open,
            high: ohlcv.high,
            low: ohlcv.low,
            volume: ohlcv.volume,
            bid_volume,
            ask_volume,
            best_bid,
            best_ask,
            bid_depth_5: None,  // TODO: Implement depth aggregation
            ask_depth_5: None,
            btc_price,
            funding_rate: None, // TODO: Get from data service
            symbol: Some(self.asset.to_string()),
        };

        // Get prediction with fallback
        let prediction = ml_client.predict_with_fallback(request).await
            .ok_or_else(|| BotError::DataError("ML prediction failed".to_string()))?;

        // Only use if confidence exceeds threshold
        if prediction.confidence < ml_config.confidence_threshold {
            return Err(BotError::DataError("ML confidence too low".to_string()));
        }

        // Convert to signal score (-1 to +1)
        let score = if prediction.direction == "LONG" {
            prediction.confidence
        } else {
            -prediction.confidence
        };

        info!(
            "ML signal: {} (confidence: {:.2}, latency: {:.1}ms)",
            prediction.direction,
            prediction.confidence,
            prediction.latency_ms
        );

        Ok(("ml_xgboost".to_string(), score, ml_config.weight))
    }
}
```

---

## Step 3: Update Main Entrypoint

### 3.1 Initialize ML Client in `main.rs`

```rust
use crate::ml_client::MLClient;

#[tokio::main]
async fn main() -> Result<()> {
    // ... existing initialization ...

    // Initialize ML client (if enabled)
    let ml_client = if let Some(ml_config) = &config.ml_service {
        if ml_config.enabled {
            info!("Initializing ML service client: {}", ml_config.base_url);

            let client = MLClient::new(
                ml_config.base_url.clone(),
                ml_config.timeout_ms,
                true
            )?;

            // Health check
            match client.health_check().await {
                Ok(health) => {
                    info!(
                        "ML service healthy - model: {}, uptime: {:.0}s, avg latency: {:.1}ms",
                        health.model_version,
                        health.uptime_seconds,
                        health.avg_latency_ms
                    );
                    Some(Arc::new(client))
                }
                Err(e) => {
                    if ml_config.enabled {
                        error!("ML service health check failed: {}", e);
                        return Err(e.into());
                    } else {
                        warn!("ML service health check failed (continuing without ML): {}", e);
                        None
                    }
                }
            }
        } else {
            info!("ML service disabled in config");
            None
        }
    } else {
        info!("ML service not configured");
        None
    };

    // ... existing component initialization ...

    // Create signal engine with ML client
    let signal_engine = SignalEngine::new(
        // ... existing params
        ml_client.clone(),
        config.ml_service.clone(),
    );

    // ... rest of main ...
}
```

---

## Step 4: Start ML Service

### 4.1 Install Dependencies

```bash
cd ml/
python -m venv ml_env
source ml_env/bin/activate
pip install -r requirements.txt
```

### 4.2 Train Model (if not already done)

```bash
python scripts/download_data.py --symbol WBNBUSDT --interval 1m --days 60
python scripts/train_xgboost.py --config configs/xgboost_baseline.yaml
```

### 4.3 Start Service

```bash
# Development
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Production (4 workers)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4.4 Verify Service

```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "price": 625.50,
    "volume": 145230.5,
    "bid_volume": 72500.0,
    "ask_volume": 68200.0
  }'
```

---

## Step 5: Docker Deployment (Optional)

### 5.1 Build Docker Image

```bash
cd ml/
docker build -t leveragebot-ml:latest .
```

### 5.2 Run Container

```bash
docker run -d \
  --name leveragebot-ml \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/xgboost_phase1_v1.pkl \
  leveragebot-ml:latest
```

### 5.3 Docker Compose (Full Stack)

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-service:
    build: ./ml
    ports:
      - "8000:8000"
    volumes:
      - ./ml/models:/app/models:ro
    environment:
      - MODEL_PATH=/app/models/xgboost_phase1_v1.pkl
    restart: unless-stopped

  rust-bot:
    build: .
    depends_on:
      - ml-service
    environment:
      - ML_SERVICE_URL=http://ml-service:8000
    restart: unless-stopped
```

---

## Configuration Examples

### Dry-Run Mode (ML Only)

Test ML predictions without executing trades:

```json
{
  "ml_service": {
    "enabled": true,
    "base_url": "http://localhost:8000",
    "timeout_ms": 100,
    "confidence_threshold": 0.60,
    "weight": 1.0
  },
  "safety": {
    "dry_run": true
  }
}
```

### Production Mode (ML + Other Signals)

Ensemble approach with multiple signal sources:

```json
{
  "ml_service": {
    "enabled": true,
    "base_url": "http://ml-service:8000",
    "timeout_ms": 100,
    "confidence_threshold": 0.55,
    "weight": 0.4
  },
  "signal_sources": {
    "tier1": {
      "technical_indicators": { "weight": 0.25 },
      "order_book_imbalance": { "weight": 0.20 }
    },
    "tier2": {
      "ml_xgboost": { "weight": 0.40 },
      "btc_volatility": { "weight": 0.10 },
      "funding_rates": { "weight": 0.05 }
    }
  }
}
```

---

## Monitoring & Debugging

### Check ML Service Logs

```bash
# Service logs
tail -f ml/logs/service.log

# Prediction latency
curl http://localhost:8000/metrics | jq '.latency_ms'
```

### Bot Integration Logs

Look for these log messages:

```
✅ Success:
INFO  ML service healthy - model: xgboost_phase1_v1, uptime: 3600s
INFO  ML signal: LONG (confidence: 0.72, latency: 6.5ms)

⚠️  Warnings:
WARN  ML prediction failed (fallback mode): timeout after 100ms
WARN  ML confidence too low: 0.45 < 0.55

❌ Errors:
ERROR ML service health check failed: connection refused
ERROR ML service returned 503: Model not loaded
```

### Performance Monitoring

```bash
# Grafana metrics (if configured)
- ml_prediction_latency_ms (p50, p95, p99)
- ml_prediction_count
- ml_confidence_distribution
- ml_service_uptime_seconds
```

---

## Troubleshooting

### Issue: ML Service Timeout

**Symptoms:** `ML service timeout after 100ms`

**Solutions:**
1. Increase timeout in config: `"timeout_ms": 200`
2. Check ML service load: `curl http://localhost:8000/metrics`
3. Verify model is loaded: `curl http://localhost:8000/health`

### Issue: Low Confidence

**Symptoms:** `ML confidence too low: 0.45 < 0.55`

**Solutions:**
1. Lower threshold in config: `"confidence_threshold": 0.50`
2. Check market conditions (volatile markets → lower confidence)
3. Retrain model with more recent data

### Issue: Service Unavailable

**Symptoms:** `ML service connection failed`

**Solutions:**
1. Verify service is running: `curl http://localhost:8000/health`
2. Check network connectivity
3. Review service logs: `docker logs leveragebot-ml`
4. Bot will continue with fallback (other signals only)

---

## Performance Optimization

### Reduce Latency

1. **Use faster hardware:** ML inference is CPU-bound
2. **Colocate services:** Run ML service on same machine as bot
3. **Enable caching:** Feature transformer caches stateful indicators
4. **Batch predictions:** If latency allows, batch multiple predictions

### Improve Accuracy

1. **Retrain regularly:** Weekly or when performance degrades
2. **Add more features:** Order book depth, funding rates
3. **Hyperparameter tuning:** Use Optuna (see training script)
4. **Ensemble models:** Combine XGBoost with GAF-CNN (Phase 2)

---

## Next Steps

1. **Monitor Phase 1 Performance:**
   - Track win rate (target: 58-62%)
   - Track Sharpe ratio (target: 1.5-2.0)
   - Monitor latency (target: <10ms p95)

2. **Iterate Based on Results:**
   - If win rate < 58%: Retrain with more data or better features
   - If latency > 10ms: Optimize feature engineering
   - If service unstable: Add monitoring and auto-restart

3. **Prepare for Phase 2:**
   - GAF-CNN implementation (image-based patterns)
   - Ensemble XGBoost + GAF-CNN
   - Target: 65-70% win rate, 2.0-2.5 Sharpe

---

**Status:** Ready for integration testing

**Estimated Integration Time:** 2-4 hours

**Risk Level:** Low (graceful fallback if ML service fails)
