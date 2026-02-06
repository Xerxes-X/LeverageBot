# LeverageBot ML System

Machine learning components for the BSC Leverage Trading Bot.

## Phase 1: XGBoost Baseline (Weeks 1-4)

**Targets:**
- Win Rate: 58-62%
- Sharpe Ratio: 1.5-2.0
- Inference Latency: <10ms

## Quick Start

### 1. Setup Environment

```bash
cd ml/
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Historical Data

```bash
python scripts/download_data.py --symbol WBNBUSDT --interval 1m --days 60
```

### 3. Train Model

```bash
python scripts/train_xgboost.py --config configs/xgboost_baseline.yaml
```

### 4. Run Backtesting

```bash
python scripts/backtest.py --model models/xgboost_phase1_v1.pkl --data data/test/
```

### 5. Start Prediction Service

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Directory Structure

```
ml/
├── api/              # FastAPI prediction service
├── configs/          # Model and training configurations
├── data/             # Training and test data
│   ├── raw/          # Raw market data from Binance
│   ├── processed/    # Engineered features
│   ├── train/        # Training datasets
│   └── test/         # Test/validation datasets
├── experiments/      # MLflow experiment tracking
├── features/         # Feature engineering modules
├── logs/             # Application logs
├── models/           # Trained model artifacts
├── notebooks/        # Jupyter notebooks for exploration
├── scripts/          # Training, backtesting, deployment scripts
└── tests/            # Unit and integration tests
```

## Key Files

- `ML_PHASE_1_IMPLEMENTATION_PLAN.md` - Comprehensive implementation guide
- `requirements.txt` - Python dependencies
- `configs/xgboost_baseline.yaml` - Model hyperparameters

## Performance Monitoring

Access MLflow UI:
```bash
mlflow ui --backend-store-uri experiments/
```

## Testing

```bash
pytest tests/ -v --cov=. --cov-report=html
```

## API Documentation

Once the service is running, visit:
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## Integration with Rust Bot

The Rust bot (`crates/bot`) calls the ML service via HTTP:

```rust
let ml_client = MLClient::new("http://localhost:8000".to_string());
let prediction = ml_client.get_prediction(&market_data).await?;
```

## Academic References

See `ML_PHASE_1_IMPLEMENTATION_PLAN.md` for full list of peer-reviewed papers.

## Support

For issues or questions, refer to the main project documentation.
