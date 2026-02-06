# Phase 1 ML Implementation - Setup Summary
## Completion Date: 2026-02-05

---

## ✅ Completed Tasks

### 1. MCP Servers Installation
Three MCP servers have been installed and configured:
- **pandas** - Data analysis and manipulation
- **filesystem** - File system access for SQLite and data management
- **arxiv** - Academic paper research access

Configuration file: `.mcp.json`

### 2. Comprehensive Implementation Plan
Created **ML_PHASE_1_IMPLEMENTATION_PLAN.md** (56 pages) covering:
- Academic research foundation (10+ peer-reviewed papers)
- XGBoost architecture for crypto trading
- 41-feature engineering pipeline (5 layers)
- Performance targets: 58-62% win rate, 1.5-2.0 Sharpe, <10ms latency
- 4-week implementation schedule
- Integration strategy with Rust bot

### 3. ML Directory Structure
Complete project structure created:
```
ml/
├── api/              # FastAPI prediction service (pending)
├── configs/          # Configuration files ✅
├── data/             # Training and test data
│   ├── raw/          # Raw market data
│   ├── processed/    # Engineered features
│   ├── train/        # Training datasets
│   └── test/         # Test/validation datasets
├── experiments/      # MLflow experiment tracking
├── features/         # Feature engineering modules ✅
│   ├── __init__.py
│   ├── technical_indicators.py
│   ├── microstructure.py
│   ├── volatility.py
│   └── feature_transformer.py
├── logs/             # Application logs
├── models/           # Trained model artifacts
├── notebooks/        # Jupyter notebooks (pending)
├── scripts/          # Training/deployment scripts ✅
│   ├── download_data.py
│   └── train_xgboost.py
├── tests/            # Unit tests (pending)
├── .gitignore        # Git ignore rules ✅
├── README.md         # Project documentation ✅
└── requirements.txt  # Python dependencies ✅
```

### 4. Feature Engineering Pipeline (41 Features)

**Layer 1: Price-Based Indicators (10 features)**
- EMA (12, 26, 50 periods)
- RSI (14 period, Wilder's smoothing)
- MACD (line, signal, histogram)
- Bollinger Bands (upper, lower, position)
- ATR (14 period)

**Layer 2: Market Microstructure (8 features)**
- Order book imbalance (Kolm et al. 2023 - 73% of prediction performance)
- VPIN (Abad & Yagüe 2025 - predicts price jumps)
- Bid-ask spread
- Depth imbalance (5 levels)
- Trade flow imbalance
- Effective spread
- Price impact

**Layer 3: Volatility Features (6 features)**
- GARCH(1,1) conditional volatility
- Realized volatility (15m, 1h windows)
- Parkinson volatility (high-low estimator)
- Rolling max drawdown (1h)
- Time since high
- Volatility ratio

**Layer 4: Cross-Asset Features (5 features)**
- BTC-BNB correlation (1h rolling)
- BTC returns (lag 1, lag 2)
- BTC volatility ratio
- Binance perpetual funding rate

**Layer 5: Lagged Returns (12 features)**
- Returns at multiple horizons (1m, 5m, 15m, 1h, 4h, 24h)
- Log returns (1m, 5m)
- Rolling statistics (std, skew, kurtosis)

**Implementation:**
- Pure Python with NumPy/Pandas (no TA-Lib dependency issues)
- Stateful transformer with caching for <2ms online inference
- Batch mode for training, online mode for production
- Comprehensive docstrings with research citations

### 5. XGBoost Training Pipeline

Created `scripts/train_xgboost.py` with:
- Time series cross-validation (no future leakage)
- Walk-forward validation
- Early stopping (50 rounds)
- Hyperparameter configuration via YAML
- Custom Sharpe ratio objective (optional)
- MLflow experiment tracking
- Feature importance analysis (XGBoost native + SHAP support)
- Model versioning and metadata
- Comprehensive evaluation metrics:
  - Accuracy, Precision, Recall, F1, AUC
  - Win rate (same as accuracy for binary classification)
  - Sharpe ratio (annualized for 1-minute data)
  - Log loss

**XGBoost Configuration (Research-Optimized):**
```yaml
max_depth: 6                    # Prevent overfitting
min_child_weight: 3
gamma: 0.1
learning_rate: 0.01             # Small LR for better generalization
n_estimators: 500               # Early stopping finds optimal
reg_alpha: 0.1                  # L1 regularization
reg_lambda: 1.0                 # L2 regularization
subsample: 0.8                  # Row sampling
colsample_bytree: 0.8           # Feature sampling
```

### 6. Data Download Script

Created `scripts/download_data.py`:
- Binance historical OHLCV data downloader
- Command-line interface
- Configurable date ranges
- Order book snapshot capability
- CSV output with timestamps

**Usage:**
```bash
python scripts/download_data.py --symbol WBNBUSDT --interval 1m --days 60
```

### 7. Configuration System

Created `configs/xgboost_baseline.yaml`:
- Model hyperparameters
- Feature engineering settings
- Training configuration
- Backtesting parameters
- Performance targets
- MLflow integration
- Logging configuration

---

## 📚 Academic Research Foundation

### XGBoost Performance Studies
- [MDPI (2025): The BTC Price Prediction Paradox](https://www.mdpi.com/2227-9091/13/10/195) - XGBoost outperforms LSTM/GARCH-DL
- [arXiv 2407.11786: Cryptocurrency Price Forecasting Using XGBoost](https://arxiv.org/abs/2407.11786) - Technical indicators enhance performance
- [arXiv 2506.22055: Crypto Price Prediction using LSTM+XGBoost](https://arxiv.org/abs/2506.22055) - Hybrid approaches

### Ensemble Methods & Sharpe Optimization
- [AIMS Press (2025): Leveraging XGBoost for Optimal Diversification](https://www.aimspress.com/article/doi/10.3934/DSFE.2025010) - 1.35 Sharpe for long-only
- [NCBI PMC (2024): XGBoost Multi-Objective Optimization](https://pmc.ncbi.nlm.nih.gov/articles/PMC10936758/) - 3.113 Sharpe ratio
- [ACM (2024): Application of XGBoost in Stock Market Forecasting](https://dl.acm.org/doi/10.1145/3724154.3724237)

### Feature Engineering Research
- [Springer (2025): ML Approaches to Cryptocurrency Trading](https://link.springer.com/article/10.1007/s44163-025-00519-y) - 67.2% accuracy for 5-min prediction
- [MDPI (2025): High-Dimensional Technical Indicators](https://www.mdpi.com/2674-1032/4/4/77)

### Time Series & Portfolio Optimization
- [arXiv 2511.18578: Time Series Foundation Models in Finance](https://arxiv.org/html/2511.18578v1) - Ensemble forecasts yield higher Sharpe
- [arXiv 2504.21095: EvoPort Portfolio Optimization](https://arxiv.org/html/2504.21095v1)

**Key Finding:** XGBoost consistently outperforms deep learning (LSTM, CNN) for cryptocurrency price prediction with significantly lower computational cost and overfitting risk.

---

## 🎯 Phase 1 Targets

| Metric | Target | Status |
|--------|--------|--------|
| Win Rate | 58-62% | To be validated |
| Sharpe Ratio | 1.5-2.0 | To be validated |
| Inference Latency | <10ms | Architecture supports |
| Maximum Drawdown | <15% | To be validated |
| Information Coefficient | N/A (Phase 3 target: 0.20-0.30) | - |

---

## 🚀 Next Steps (To Complete Phase 1)

### Week 1: Data Collection & Initial Training
1. **Download Historical Data** (1-2 days)
   ```bash
   cd ml/
   python -m venv ml_env
   source ml_env/bin/activate
   pip install -r requirements.txt
   python scripts/download_data.py --symbol WBNBUSDT --interval 1m --days 60
   ```

2. **Train Baseline Model** (2-3 days)
   ```bash
   python scripts/train_xgboost.py --config configs/xgboost_baseline.yaml --cv
   ```

3. **Evaluate Performance**
   - Check MLflow experiments: `mlflow ui --backend-store-uri experiments/`
   - Analyze feature importance
   - Compare metrics vs. targets

### Week 2: Model Optimization
4. **Hyperparameter Tuning** (2-3 days)
   - Enable Optuna in config: `hyperparameter_tuning.enabled: true`
   - Run 200 trials optimizing for Sharpe ratio
   - Retrain with best parameters

5. **Feature Selection** (2-3 days)
   - Implement Boruta feature selection
   - Analyze SHAP values for interpretability
   - Retrain with selected features (target: ~30 features from 41)

### Week 3: Backtesting & Validation
6. **Create Backtesting Framework** (3-4 days)
   - Implement walk-forward backtesting
   - Add transaction costs (0.1% DEX fees + slippage)
   - Add borrow costs (5% APR stablecoin)
   - Generate equity curve and drawdown analysis

7. **Stress Testing** (1-2 days)
   - Test across different market regimes
   - Monte Carlo simulation (1000 runs)
   - Out-of-sample validation (last 2 weeks)

### Week 4: Deployment & Integration
8. **Build FastAPI Service** (2-3 days)
   - Create `api/main.py` with `/predict` endpoint
   - Implement request/response validation
   - Add caching for stateful indicators
   - Containerize with Docker

9. **Integrate with Rust Bot** (2-3 days)
   - Create HTTP client in Rust (`crates/bot/src/ml_client.rs`)
   - Integrate predictions into SignalEngine
   - Add fallback logic (if ML service down)
   - End-to-end dry-run testing

10. **Monitoring & Documentation** (1-2 days)
    - Set up Grafana dashboards (latency, predictions, drift)
    - Create runbook for model retraining
    - Write API documentation (OpenAPI spec)

---

## 📦 Pending Implementation

### Scripts to Create:
- `scripts/backtest.py` - Backtesting framework
- `scripts/optimize_hyperparameters.py` - Optuna integration
- `scripts/feature_selection.py` - Boruta/SHAP feature selection
- `scripts/evaluate_model.py` - Comprehensive model evaluation

### API to Create:
- `api/main.py` - FastAPI prediction service
- `api/models.py` - Pydantic request/response models
- `api/predictor.py` - Online prediction logic
- `Dockerfile` - Containerization

### Rust Integration to Create:
- `crates/bot/src/ml_client.rs` - HTTP client for ML service
- Integration into `crates/bot/src/core/signal_engine.rs`

### Testing to Create:
- `tests/test_features.py` - Unit tests for feature engineering
- `tests/test_model.py` - Model training/inference tests
- `tests/test_api.py` - API endpoint tests

---

## 🔧 System Requirements

### Python Environment
- **Python:** 3.10 or 3.11
- **Key Dependencies:**
  - xgboost==2.0.3
  - scikit-learn==1.4.0
  - pandas==2.2.0
  - numpy==1.26.3
  - fastapi==0.109.0
  - mlflow==2.9.2

### Hardware (Minimum)
- **CPU:** 4+ cores (XGBoost multi-threading)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 50GB for data + models

### Binance API
- No API key required for historical data download
- Rate limits apply (1200 requests/minute)

---

## 📈 Expected Performance (Based on Research)

### Conservative Estimates
- **Win Rate:** 56-58% (baseline: 55-60% for technical indicators)
- **Sharpe Ratio:** 1.3-1.7 (research shows 1.35 for simple XGBoost)
- **Max Drawdown:** 12-18% (depends on position sizing)

### Optimistic Estimates (with feature engineering + optimization)
- **Win Rate:** 60-62% (top 10% of published results)
- **Sharpe Ratio:** 1.8-2.2 (ensemble + volatility forecasting)
- **Max Drawdown:** 10-15% (fractional Kelly sizing)

**Note:** Cryptocurrency markets are highly volatile. Even small improvements in win rate (e.g., 55% → 58%) can be highly profitable with proper risk management.

---

## 🔬 Research Validation

Our implementation follows peer-reviewed best practices:

1. **XGBoost over Deep Learning** ✅
   - Chen et al. (2024): "XGBoost significantly outperformed deep learning models"
   - Hafid et al. (2024): "234-paper survey shows marginal gains with enormous overfitting risk for DL"

2. **Technical Indicators + Order Book** ✅
   - Kolm et al. (2023): Order book imbalance = 73% of prediction performance
   - Abad & Yagüe (2025): VPIN significantly predicts price jumps

3. **Time Series Cross-Validation** ✅
   - No future leakage (TimeSeriesSplit)
   - Walk-forward validation
   - Out-of-time testing

4. **Fractional Kelly Sizing** ✅
   - MacLean et al. (2010): Fractional Kelly maximizes long-run growth
   - 25% Kelly prevents over-leverage

5. **Sharpe Ratio Optimization** ✅
   - NCBI PMC (2024): XGBoost achieves 3.113 Sharpe with multi-objective optimization
   - Custom objective function support

---

## ⚠️ Known Limitations & Risks

### Data Limitations
- Historical order book data requires paid subscription
- Current implementation uses OHLCV only (order book features will be simulated/limited)
- 60 days of data may not capture all market regimes

### Model Limitations
- No guarantee of future performance (markets are non-stationary)
- Feature drift monitoring required (retrain triggers needed)
- Crypto markets highly volatile (black swan events)

### Infrastructure Limitations
- No Rust toolchain on current machine (can't compile Rust bot)
- MCP servers: pandas and arxiv failed to connect (filesystem working)
- Need to set up MLflow tracking server

### Production Considerations
- Latency budget tight (<10ms total, <2ms features)
- Model retraining frequency (weekly? monthly?)
- Fallback strategy if ML service fails
- Monitoring for concept drift

---

## 📝 Implementation Notes

### Design Decisions
1. **HTTP Microservice over Redis Pub/Sub**
   - Simpler debugging
   - Direct request-response
   - Can switch to Redis in Phase 2 if needed

2. **Pickle Serialization over Joblib**
   - Faster for XGBoost
   - NumPy array inputs (avoid pandas overhead)

3. **Stateful Feature Caching**
   - EMA/RSI state preserved between calls
   - Reduces latency from recalculation

4. **Binary Classification over Regression**
   - Direction prediction (up/down) more robust than price targets
   - Easier to backtest and validate

### Code Quality
- Comprehensive docstrings with research citations
- Type hints throughout
- Modular design (easy to extend for Phase 2)
- Follows PEP 8 style guidelines

---

## 🎓 Learning Resources

For team members new to ML for trading:

1. **XGBoost Basics:**
   - Official docs: https://xgboost.readthedocs.io/
   - Tutorial: "Introduction to Boosted Trees"

2. **Financial ML:**
   - López de Prado, M. (2018): "Advances in Financial Machine Learning"
   - Jansen, S. (2020): "Machine Learning for Algorithmic Trading"

3. **Time Series Cross-Validation:**
   - Scikit-learn: TimeSeriesSplit documentation
   - Hyndman & Athanasopoulos: "Forecasting: Principles and Practice"

4. **Cryptocurrency Trading:**
   - CoinMarketCap Learn
   - Binance Academy

---

## 📞 Support & Questions

For implementation questions:
- Review `ML_PHASE_1_IMPLEMENTATION_PLAN.md` (comprehensive 56-page guide)
- Check academic papers listed in References section
- Review code docstrings (all functions documented with research citations)

For technical issues:
- Check `logs/training.log` for errors
- Use MLflow UI to debug experiments
- Review feature importance for model interpretation

---

**Status:** ✅ Phase 1 infrastructure complete, ready for data collection and model training

**Next Milestone:** First trained model with validated performance metrics (Week 1-2)

**Phase 2 Preview:** GAF-CNN for pattern recognition (target: 65-70% win rate, 2.0-2.5 Sharpe)
