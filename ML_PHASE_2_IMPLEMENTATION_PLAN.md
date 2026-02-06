# Phase 2 Implementation Plan: GAF-CNN

**Date:** 2026-02-05
**Status:** In Progress
**Estimated Timeline:** 3-5 days

---

## 🎯 Objectives

### Primary Goals

1. **Implement GAF-CNN model** for cryptocurrency time series prediction
2. **Achieve >90% pattern recognition** accuracy (from research benchmarks)
3. **Improve win rate to 65-70%** (from current 50.84%)
4. **Maintain <50ms inference latency**
5. **Ensemble with Phase 1 XGBoost** for robust predictions

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Pattern Recognition | >90% | CNN validation accuracy |
| Win Rate (Ensemble) | 65-70% | Test set accuracy |
| Sharpe (Ensemble) | 2.0-2.5 | Backtest Sharpe ratio |
| Inference Time | <50ms | Average prediction latency |
| Model Size | <500MB | Saved model file size |

---

## 🏗️ Architecture Design

### Option A: Multi-Resolution Ensemble (Selected)

**Rationale:** Best performance in literature, captures patterns at multiple time scales

**Components:**

1. **GAF Transformer**
   - Input: Time series window (OHLCV data)
   - Output: GAF image (GASF + GADF channels)
   - Window sizes: 15m, 30m, 60m
   - Image size: 64×64 pixels (efficient for CNNs)

2. **CNN Models (3 models)**
   - **CNN-15m:** Captures intraday patterns
   - **CNN-30m:** Captures session patterns
   - **CNN-60m:** Captures hourly trends
   - Each: ResNet18 architecture with transfer learning

3. **Ensemble Layer**
   - Combines: 3 CNN predictions + 1 XGBoost prediction
   - Weights: Optimized on validation set
   - Method: Weighted average or meta-learner

4. **Integration with Phase 1**
   - Phase 1 XGBoost: Feature-based prediction
   - Phase 2 GAF-CNN: Pattern-based prediction
   - Final: Weighted ensemble

---

## 📊 Data Pipeline

### Step 1: Data Preparation

**Input:** Same 180-day dataset as Phase 1
- BNBUSDT: 259,201 samples ✅
- BTCUSDT: 259,201 samples ✅

**Preprocessing:**
1. Extract rolling windows: 15m, 30m, 60m
2. Normalize each window to [-1, 1]
3. Generate GASF + GADF images
4. Create labels (same percentile-based as Phase 1)
5. Train/test split: 80/20 (time-series aware)

**Output:**
- Training: ~165K windows × 3 resolutions = ~495K GAF images
- Testing: ~41K windows × 3 resolutions = ~123K GAF images

### Step 2: GAF Transformation

**For each window:**

```python
def generate_gaf(window, size=64):
    # Normalize to [-1, 1]
    normalized = 2 * (window - window.min()) / (window.max() - window.min() + 1e-8) - 1

    # Angular encoding
    phi = np.arccos(normalized)

    # GASF: cos(phi_i + phi_j)
    gasf = np.cos(phi[:, None] + phi[None, :])

    # GADF: sin(phi_i - phi_j)
    gadf = np.sin(phi[:, None] - phi[None, :])

    # Resize to target size
    gasf_resized = resize(gasf, (size, size))
    gadf_resized = resize(gadf, (size, size))

    # Stack as 2-channel image
    gaf_image = np.stack([gasf_resized, gadf_resized], axis=-1)

    return gaf_image
```

**Optimization:**
- Cache GAF images to disk (avoid regenerating)
- Use data generators for memory efficiency
- Parallel processing with multiprocessing

### Step 3: Data Augmentation

**Techniques:**
- Time shifting: ±5 minutes
- Scaling: ×0.95 to ×1.05
- Noise addition: Gaussian σ=0.01
- Horizontal flip: Mirror temporal patterns

**Goal:** Increase effective dataset size by 3-5×

---

## 🧠 CNN Architecture

### Model: ResNet18 with Transfer Learning

**Base:** Pre-trained ResNet18 from ImageNet
**Modifications:**
- Input layer: 2 channels (GASF + GADF) vs 3 (RGB)
- Output layer: 1 neuron with sigmoid (binary classification)
- Freeze early layers, fine-tune later layers

**Architecture:**

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18

class GAF_CNN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Load pre-trained ResNet18
        self.resnet = resnet18(pretrained=pretrained)

        # Modify first conv layer for 2-channel input
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify final layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)
```

**Hyperparameters:**
- Optimizer: AdamW
- Learning rate: 1e-4 (with cosine annealing)
- Batch size: 64
- Epochs: 50 (with early stopping)
- Loss: BCEWithLogitsLoss
- Weight decay: 1e-5

---

## 🎓 Training Strategy

### Phase 1: Individual CNN Training

**For each window size (15m, 30m, 60m):**

1. **Data Loading**
   - Load GAF images for specific window size
   - Balanced sampling (50/50 UP/DOWN)
   - Data augmentation on-the-fly

2. **Transfer Learning**
   - Freeze layers 1-2 (early features)
   - Fine-tune layers 3-4 (domain-specific features)
   - Train final classifier layers

3. **Training**
   - Train for 50 epochs
   - Early stopping: patience=10
   - Save best model (validation loss)
   - Track: loss, accuracy, AUC, Sharpe

4. **Validation**
   - Evaluate on hold-out validation set (10% of train)
   - Metrics: accuracy, precision, recall, AUC, Sharpe
   - Ensure >90% pattern recognition accuracy

**Expected Training Time:** 2-3 hours per model on GPU

### Phase 2: Ensemble Optimization

**Step 1: Individual Model Evaluation**
- Evaluate each CNN on test set
- Evaluate Phase 1 XGBoost on test set
- Calculate individual Sharpe ratios

**Step 2: Weight Optimization**

```python
def optimize_weights(predictions, labels, returns):
    """Find optimal ensemble weights"""

    def objective(weights):
        # Normalize weights
        weights = weights / weights.sum()

        # Ensemble prediction
        ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))

        # Calculate Sharpe
        sharpe = calculate_sharpe(labels, ensemble_pred, returns)

        return -sharpe  # Minimize negative Sharpe

    # Optimize
    from scipy.optimize import minimize
    initial_weights = np.ones(4) / 4
    bounds = [(0, 1) for _ in range(4)]
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

    return result.x
```

**Expected Weights:**
- XGBoost: 0.3-0.4 (feature-based)
- CNN-15m: 0.15-0.20 (short-term patterns)
- CNN-30m: 0.20-0.25 (medium-term patterns)
- CNN-60m: 0.20-0.25 (long-term patterns)

### Phase 3: Meta-Learning (Optional)

**Alternative to weighted average:**

Train a meta-learner (LightGBM) that takes as input:
- 4 base model predictions
- Confidence scores
- Market regime indicators

**Advantages:**
- Non-linear combination
- Adaptive weighting based on market conditions
- Potential for better performance

---

## 📈 Evaluation Metrics

### Primary Metrics

1. **Pattern Recognition Accuracy**
   - Target: >90%
   - Measures: How well CNN identifies patterns in GAF images

2. **Win Rate**
   - Target: 65-70%
   - Measures: % of correct directional predictions

3. **Sharpe Ratio**
   - Target: 2.0-2.5 (ensemble)
   - Measures: Risk-adjusted returns

4. **Information Coefficient**
   - Target: 0.02-0.05
   - Measures: Correlation between predictions and returns

### Secondary Metrics

5. **Precision & Recall**
6. **AUC-ROC**
7. **Max Drawdown**
8. **Calmar Ratio**
9. **Inference Latency**

### Benchmarks

**Compare Against:**
- Phase 1 XGBoost alone (Sharpe 4.11, Win 50.84%)
- Random baseline (Sharpe 0, Win 50%)
- Buy-and-hold (Sharpe ~0.5-1.0)
- Literature benchmarks (90-93% pattern accuracy)

---

## 🚀 Implementation Steps

### Day 1: GAF Transformation Pipeline

**Tasks:**
1. ✅ Research complete (Task 15)
2. ⏳ Implement GAF transformation functions
   - GASF generation
   - GADF generation
   - Image resizing and normalization
   - Caching mechanism

3. ⏳ Create data generators
   - PyTorch Dataset class
   - DataLoader with batching
   - On-the-fly augmentation

4. ⏳ Generate GAF images for all windows
   - 15m windows
   - 30m windows
   - 60m windows
   - Save to disk

**Deliverables:**
- `ml/gaf/gaf_transformer.py`
- `ml/gaf/gaf_dataset.py`
- `data/gaf/bnb_15m/`, `data/gaf/bnb_30m/`, `data/gaf/bnb_60m/`

### Day 2: CNN Architecture & Setup

**Tasks:**
1. ⏳ Implement CNN architecture
   - ResNet18 with transfer learning
   - 2-channel input modification
   - Custom classifier head

2. ⏳ Setup training infrastructure
   - Training loop with early stopping
   - Learning rate scheduling
   - Metric tracking (MLflow)
   - Model checkpointing

3. ⏳ Create training scripts
   - `train_gaf_cnn_15m.py`
   - `train_gaf_cnn_30m.py`
   - `train_gaf_cnn_60m.py`

**Deliverables:**
- `ml/models/gaf_cnn.py`
- `ml/scripts/train_gaf_cnn.py`
- `ml/configs/gaf_cnn_config.yaml`

### Day 3: Model Training

**Tasks:**
1. ⏳ Train CNN-15m
   - 50 epochs with early stopping
   - Track validation metrics
   - Save best model

2. ⏳ Train CNN-30m
   - Same process

3. ⏳ Train CNN-60m
   - Same process

4. ⏳ Evaluate individual models
   - Test set performance
   - Pattern recognition accuracy
   - Sharpe ratio

**Deliverables:**
- `ml/models/gaf_cnn_15m_v1.pth`
- `ml/models/gaf_cnn_30m_v1.pth`
- `ml/models/gaf_cnn_60m_v1.pth`
- Training logs and metrics

### Day 4: Ensemble Integration

**Tasks:**
1. ⏳ Load Phase 1 XGBoost model
2. ⏳ Generate predictions from all 4 models
3. ⏳ Optimize ensemble weights
   - Grid search or Bayesian optimization
   - Maximize Sharpe ratio

4. ⏳ Implement ensemble predictor
   - Weighted average
   - Or meta-learner

5. ⏳ Evaluate ensemble performance
   - Test set metrics
   - Compare vs Phase 1 alone
   - Validate improvements

**Deliverables:**
- `ml/models/phase2_ensemble_v1.pkl`
- `ml/scripts/ensemble_predictor.py`
- Performance comparison report

### Day 5: Validation & Documentation

**Tasks:**
1. ⏳ Walk-forward validation
2. ⏳ Out-of-sample testing
3. ⏳ Benchmark against literature
4. ⏳ Create deployment plan
5. ⏳ Update API for Phase 2
6. ⏳ Integration testing
7. ⏳ Final documentation

**Deliverables:**
- `ML_PHASE_2_RESULTS.md`
- Updated `api/main.py` with Phase 2 endpoint
- `PHASE_2_DEPLOYMENT_GUIDE.md`

---

## 🔧 Technical Requirements

### Software Dependencies

**New Requirements:**
```
# Deep Learning
torch==2.1.0
torchvision==0.16.0
pillow==10.1.0

# Image Processing
opencv-python==4.8.1
scikit-image==0.22.0

# Optimization
scipy==1.12.0  # Already installed

# Visualization
tensorboard==2.15.0
```

### Hardware Requirements

**Minimum:**
- CPU: 8 cores
- RAM: 16 GB
- Storage: 50 GB (for GAF images)

**Recommended:**
- GPU: NVIDIA with 8+ GB VRAM (e.g., RTX 3070)
- RAM: 32 GB
- Storage: 100 GB SSD

**Note:** Training on CPU is possible but slower (6-8 hours vs 2-3 hours)

---

## 📁 File Structure

```
ml/
├── gaf/
│   ├── __init__.py
│   ├── gaf_transformer.py      # GAF transformation functions
│   ├── gaf_dataset.py          # PyTorch Dataset for GAF images
│   └── augmentation.py         # Data augmentation utilities
│
├── models/
│   ├── gaf_cnn.py              # CNN architecture definition
│   ├── gaf_cnn_15m_v1.pth      # Trained 15m model
│   ├── gaf_cnn_30m_v1.pth      # Trained 30m model
│   ├── gaf_cnn_60m_v1.pth      # Trained 60m model
│   └── phase2_ensemble_v1.pkl  # Ensemble weights/meta-learner
│
├── scripts/
│   ├── generate_gaf_images.py  # Pre-generate GAF images
│   ├── train_gaf_cnn.py        # Training script
│   ├── optimize_ensemble.py    # Ensemble weight optimization
│   └── evaluate_phase2.py      # Evaluation script
│
├── api/
│   ├── main.py                 # Updated with Phase 2 endpoint
│   └── gaf_predictor.py        # Phase 2 prediction logic
│
└── data/
    └── gaf/
        ├── bnb_15m/            # 15-minute GAF images
        ├── bnb_30m/            # 30-minute GAF images
        └── bnb_60m/            # 60-minute GAF images
```

---

## 🎯 Risk Mitigation

### Potential Issues & Solutions

**1. GAF image generation is slow**
- Solution: Pre-generate all images, cache to disk
- Parallel processing with multiprocessing
- Expected: 30-60 minutes for all images

**2. CNN training takes too long**
- Solution: Transfer learning from ImageNet
- Reduce image size to 48×48 if needed
- Use mixed precision training (FP16)

**3. Ensemble doesn't improve over Phase 1**
- Solution: Try different ensemble strategies
- Adjust ensemble weights
- Consider when to use Phase 2 vs Phase 1 alone
- Worst case: Phase 1 XGBoost (Sharpe 4.11) still excellent

**4. Pattern recognition <90%**
- Solution: Increase training data (more augmentation)
- Try deeper architecture (ResNet34/50)
- Adjust window sizes
- Tune hyperparameters

**5. Inference latency >50ms**
- Solution: Model quantization
- Reduce image size
- Use TorchScript for optimization
- Cache GAF transformations

---

## 📊 Success Metrics Summary

### Must Have (P0)

- ✅ GAF transformation pipeline working
- ✅ CNN models training successfully
- ✅ Ensemble integration functional
- ✅ Inference latency <50ms

### Should Have (P1)

- ✅ Pattern recognition >90%
- ✅ Win rate 65-70%
- ✅ Sharpe ratio 2.0-2.5
- ✅ Outperforms Phase 1 alone

### Nice to Have (P2)

- Meta-learning ensemble
- Real-time GAF generation
- Model interpretability (CAM visualization)
- A/B testing framework

---

## 🚀 Deployment Strategy

### Phase 2 API Endpoint

```python
@app.post("/predict/phase2")
async def predict_phase2(request: PredictionRequest):
    """Phase 2 GAF-CNN + XGBoost ensemble prediction"""

    # 1. Get Phase 1 prediction
    xgb_pred = xgboost_model.predict(features)

    # 2. Generate GAF images
    gaf_15m = generate_gaf(window_15m)
    gaf_30m = generate_gaf(window_30m)
    gaf_60m = generate_gaf(window_60m)

    # 3. Get CNN predictions
    cnn_15m_pred = cnn_15m_model(gaf_15m)
    cnn_30m_pred = cnn_30m_model(gaf_30m)
    cnn_60m_pred = cnn_60m_model(gaf_60m)

    # 4. Ensemble
    ensemble_pred = (
        w_xgb * xgb_pred +
        w_15m * cnn_15m_pred +
        w_30m * cnn_30m_pred +
        w_60m * cnn_60m_pred
    )

    return PredictionResponse(
        direction="LONG" if ensemble_pred > 0.5 else "SHORT",
        confidence=abs(ensemble_pred - 0.5) * 2,
        raw_probability=ensemble_pred,
        model_contributions={
            "xgboost": xgb_pred,
            "cnn_15m": cnn_15m_pred,
            "cnn_30m": cnn_30m_pred,
            "cnn_60m": cnn_60m_pred
        }
    )
```

### Gradual Rollout

**Stage 1: Validation**
- Deploy Phase 2 alongside Phase 1
- A/B test with 10% traffic
- Monitor performance metrics

**Stage 2: Expansion**
- Increase to 50% traffic if metrics good
- Continue monitoring

**Stage 3: Full Deployment**
- Switch to Phase 2 as primary if validated
- Keep Phase 1 as fallback

---

## 📚 References

- Research summary: `ML_PHASE_2_RESEARCH_SUMMARY.md`
- Phase 1 results: `ML_OPTIMIZATION_COMPLETE.md`
- Integration guide: `INTEGRATION_GUIDE.md`

---

**Status:** Implementation plan complete, ready to begin coding

**Next Step:** Implement GAF transformation pipeline (Day 1 tasks)
