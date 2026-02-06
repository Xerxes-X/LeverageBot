# Phase 2 GAF-CNN: Performance Analysis & Next Steps

**Date**: 2026-02-06 01:00am
**Status**: ⚠️ BELOW TARGET - Requires Investigation

---

## 🔴 Critical Findings

### Ensemble Performance

| Metric | Individual Avg | Ensemble | Target | Gap |
|--------|----------------|----------|--------|-----|
| **Accuracy** | 51.59% | **47.33%** | 90% | **-42.67%** |
| Precision | - | 43.22% | - | - |
| Recall | - | 43.51% | - | - |
| F1 Score | - | 43.37% | - | - |

**Problem**: Ensemble performs **WORSE** than individual models (-8.3%)

**Expected**: Ensemble should be +10-15% better, not worse!

---

## 🔍 Root Cause Analysis

### 1. Individual Model Performance Issues

**All 3 models peaked at Epoch 1**, then degraded:

| Model | Best Epoch | Val Accuracy | Notes |
|-------|------------|--------------|-------|
| 15m | 1 | 52.80% | No improvement after epoch 1 |
| 30m | 1 | 51.72% | No improvement after epoch 1 |
| 60m | 1 | 50.25% | No improvement after epoch 1 |

**This suggests**:
- Models are not learning meaningful patterns
- Overfitting immediately (epoch 1 is best)
- Possible data quality or labeling issues
- Architecture may not be suitable for this data

### 2. ~50% Accuracy = Random Guessing

**Binary classification baseline**: 50% (coin flip)

All models are performing **at or below random chance**:
- 15m: 52.80% (slightly better than random)
- 30m: 51.72% (barely better than random)
- 60m: 50.25% (essentially random)

**This indicates**: The models have learned almost nothing!

### 3. Confusion Matrix Analysis

```
                Predicted
                DOWN    UP
  Actual DOWN   5630   5492   (50.6% correct)
  Actual UP     5426   4180   (43.5% correct)
```

- **DOWN prediction**: Barely better than random (50.6%)
- **UP prediction**: Worse than random (43.5%)
- Model is biased toward predicting DOWN

---

## 🤔 Possible Causes

### A. Data Issues (Most Likely)

1. **Label Quality**
   - Are UP/DOWN labels correctly defined?
   - Check label generation logic in GAF preprocessing
   - Verify label distribution (should be ~50/50)

2. **Data Leakage**
   - Are train/val splits truly independent?
   - Check for look-ahead bias in feature generation

3. **GAF Image Quality**
   - Verify GAF images actually contain visual patterns
   - Check normalization (values should be [-1, 1])
   - Inspect sample images visually

### B. Architecture Issues

1. **Transfer Learning Mismatch**
   - ImageNet pretrained weights may not suit GAF images
   - GAF images are fundamentally different from natural images
   - **Fix**: Try training from scratch (no pretrained weights)

2. **Input Channels**
   - Using 2 channels (GASF + GADF)
   - Pretrained weights are for 3 channels (RGB)
   - Channel averaging may lose information

3. **Model Capacity**
   - ResNet18 may be too complex (11M params for 50% accuracy!)
   - Simpler architecture might work better

### C. Training Issues

1. **Learning Rate Too High**
   - lr=1e-4 might be too aggressive
   - Models overfit immediately (epoch 1 best)
   - **Fix**: Try lr=1e-5 or 1e-6

2. **Early Stopping Too Aggressive**
   - Patience=10 stops training quickly
   - Models need more time to converge
   - **Fix**: Increase patience to 20-30

3. **Optimizer**
   - AdamW might not be optimal
   - **Fix**: Try SGD with momentum

---

## 🛠️ Recommended Actions (Priority Order)

### Immediate (Debug Phase)

**1. Verify Data Quality** ⭐ TOP PRIORITY
```python
# Check label distribution
print("Label counts:", np.bincount(labels))

# Visualize sample GAF images
import matplotlib.pyplot as plt
plt.imshow(gaf_images[0, :, :, 0])  # GASF channel
plt.imshow(gaf_images[0, :, :, 1])  # GADF channel

# Check for NaN/Inf
print("NaN count:", np.isnan(gaf_images).sum())
print("Inf count:", np.isinf(gaf_images).sum())

# Verify value ranges
print("Min:", gaf_images.min(), "Max:", gaf_images.max())
```

**2. Inspect Sample Predictions**
- Load a few validation samples
- Get predictions from each model
- Check if predictions make sense given the input

**3. Baseline Comparison**
- Train simple logistic regression on flattened GAF images
- If it performs better than CNN, architecture is the problem

### Short-term (Fixes)

**4. Retrain with Modified Hyperparameters**
```python
# Try these changes:
- learning_rate = 1e-5 (10x lower)
- pretrained = False (train from scratch)
- patience = 30 (more epochs before stopping)
- freeze_early_layers = False (train all layers)
```

**5. Simplify Architecture**
- Try smaller model (ResNet10 or custom simple CNN)
- Reduce dropout (0.5 → 0.2)
- Remove batch norm

**6. Data Augmentation**
```python
# For GAF images:
- Random rotation (±5 degrees)
- Random scaling (0.95-1.05)
- Gaussian noise (σ=0.01)
```

### Long-term (Alternative Approaches)

**7. Different Architecture**
- Try Vision Transformer (ViT) instead of CNN
- Use 1D CNN on raw time series (skip GAF transformation)
- Recurrent CNN (combining temporal + spatial)

**8. Feature Engineering**
- Add technical indicators as additional channels
- Combine GAF with other image representations
- Multi-task learning (predict both direction + magnitude)

**9. Ensemble Improvements**
- Stacked ensemble with meta-learner
- Different ensemble weights (try inverse accuracy weighting)
- Add Phase 1 (XGBoost) to ensemble

---

## 📊 Phase 1 vs Phase 2 Comparison

| Metric | Phase 1 (XGBoost) | Phase 2 (GAF-CNN) | Difference |
|--------|-------------------|-------------------|------------|
| Win Rate | 63.4% | ~47% (ensemble) | **-16.4%** ❌ |
| Sharpe | 4.11 | Not tested | - |
| Training Time | ~30 min | 2.4 hours | +1.9 hours |
| Complexity | Medium | High | - |

**Conclusion**: Phase 1 significantly outperforms Phase 2 currently!

---

## 🎯 Revised Expectations

### Original Plan
- Individual models: 50-65%
- Ensemble: 90-93%
- Phase 1 + Phase 2: 68-72% win rate

### Current Reality
- Individual models: 50-53% ✅ (met low end)
- Ensemble: 47% ❌ (worse than individuals!)
- Phase 1 + Phase 2: Would likely decrease performance

### Adjusted Goals

**Option A: Debug & Retrain**
- Target: 70% individual, 80-85% ensemble
- Timeline: 1-2 days of experimentation
- Risk: May still not reach 90%

**Option B: Use Phase 1 Only**
- Keep XGBoost (63.4% win rate, 4.11 Sharpe)
- Skip Phase 2 GAF-CNN entirely
- Deploy with proven model

**Option C: Hybrid Approach**
- Use XGBoost as primary
- Add simple technical indicators
- Skip complex deep learning

---

## 💡 Key Insights

1. **Deep learning isn't always better** - XGBoost (Phase 1) significantly outperforms CNN (Phase 2)

2. **~50% accuracy = failure** - Model hasn't learned anything meaningful

3. **Ensemble made it worse** - Indicates models are not complementary (all making same mistakes)

4. **Early overfitting** - All models peaked at epoch 1 then degraded

5. **Data quality critical** - Must verify labels and GAF images are correct

---

## 🚀 Recommended Next Step

**I recommend**: Investigate data quality issues before retraining

**Questions to answer**:
1. Are the labels (UP/DOWN) correct?
2. Do GAF images visually show patterns?
3. Is there data leakage in preprocessing?
4. Should we use Phase 1 (XGBoost) instead?

**Your call**:
- **Option A**: Debug and retrain (1-2 days)
- **Option B**: Deploy Phase 1 (XGBoost) only
- **Option C**: Simplify Phase 2 approach

What would you like to do?
