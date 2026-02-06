# Phase 2 GAF-CNN - Quick Reference

**Last Updated**: 2026-02-06 01:10am
**Status**: ⏸️ Paused - Awaiting Option A continuation on another machine

---

## 📊 Current State (One-Page Summary)

### Models Trained
- ✅ 15m CNN: 52.80% val accuracy (124 MB)
- ✅ 30m CNN: 51.72% val accuracy (124 MB)
- ✅ 60m CNN: 50.25% val accuracy (124 MB)
- ❌ Ensemble: 47.33% accuracy (WORSE than individuals!)

### Performance vs Targets
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Ensemble Accuracy | 47% | 90% | ❌ -43% gap |
| Phase 1 (XGBoost) | 63.4% | - | ✅ BEST |

### Problem
**Models are guessing randomly** (~50% = coin flip)
- All peaked at Epoch 1 (immediate overfitting)
- No meaningful pattern learning
- Ensemble combines failures → worse performance

---

## 📁 Key Files Location

**All files in**: `/home/rom/LeverageBot/ml/`

**MUST READ before continuing**:
1. `OPTION_A_CONTINUATION_GUIDE.md` ← **START HERE**
2. `ML_PHASE_2_ANALYSIS.md` ← Root cause analysis
3. `ML_PHASE_2_TRAINING_COMPLETE.md` ← Training results

**Data**:
- Source: `data/BTCUSDT_1m_2024.csv`
- GAF images: `data/gaf/bnb_{15,30,60}m/` (39 GB total)

**Models**:
- Trained: `models/gaf_cnn_{15,30,60}m_v1.pth`
- Ensemble: `models/ensemble.py`

**Scripts**:
- GAF generation: `scripts/generate_gaf_fixed.py` ✅ WORKING
- Training: `scripts/train_gaf_cnn_precomputed.py` ⚠️ NEEDS FIXES
- Validation: `scripts/validate_ensemble.py`

---

## 🚀 How to Continue (Quick Start)

### 1. Setup
```bash
cd /home/rom/LeverageBot/ml
source ml_env/bin/activate
```

### 2. Read Documentation
```bash
# Read full continuation guide
cat OPTION_A_CONTINUATION_GUIDE.md | less

# Read problem analysis
cat ML_PHASE_2_ANALYSIS.md | less
```

### 3. Start Debugging

**Phase 1: Data Quality** (PRIORITY)
```bash
# Check labels
python -c "
import numpy as np
labels = np.load('data/gaf/bnb_15m/train_labels.npy')
print(f'UP: {np.sum(labels==1):,} ({np.mean(labels==1)*100:.1f}%)')
print(f'DOWN: {np.sum(labels==0):,} ({np.mean(labels==0)*100:.1f}%)')
"

# Check images
python -c "
import numpy as np
images = np.load('data/gaf/bnb_15m/train_images.npy', mmap_mode='r')
print(f'Shape: {images.shape}')
print(f'Range: [{images[:1000].min():.2f}, {images[:1000].max():.2f}]')
print(f'NaN: {np.isnan(images[:1000]).sum()}')
"
```

**Phase 2: Retrain with Fixes**
```bash
# Try lower learning rate + no pretrained weights
python scripts/train_gaf_cnn_precomputed.py \
    --window_size 15 \
    --learning_rate 1e-5 \
    --pretrained False \
    --freeze_early False \
    --patience 30
```

---

## 🎯 Decision Tree

```
Is ensemble accuracy > 70%?
├─ YES → Deploy Phase 2
└─ NO
   └─ Is individual model > 65%?
      ├─ YES → Tune ensemble weights
      └─ NO
         └─ Did data quality check pass?
            ├─ YES → Try different architecture
            └─ NO → Fix data, regenerate GAF, retrain
```

**If still < 60% after all fixes**:
→ **Use Phase 1 (XGBoost) only** - already proven at 63.4% win rate

---

## ⚙️ Training Parameters to Try

**Current (Failed)**:
- LR: 1e-4, Pretrained: True, Patience: 10
- Result: 50-53% accuracy

**Option A (Conservative)**:
- LR: 1e-5, Pretrained: False, Patience: 30
- Expected: Slower but more stable learning

**Option B (Aggressive)**:
- LR: 5e-5, Pretrained: True, Patience: 20
- Expected: Faster convergence

**Option C (Minimal)**:
- LR: 1e-6, Pretrained: False, Patience: 50
- Expected: Very slow but thorough

---

## 🔍 Debug Checklist

**Before retraining**:
- [ ] Labels balanced ~50/50
- [ ] GAF images show visible patterns
- [ ] No NaN/Inf in data
- [ ] Baseline logistic regression > 55%

**After training**:
- [ ] Validation improves over epochs (not just epoch 1)
- [ ] Training accuracy > validation (normal gap)
- [ ] Model file size ~124 MB
- [ ] Ensemble > individuals

---

## 📊 Success Metrics

| Phase | Metric | Minimum | Good | Excellent |
|-------|--------|---------|------|-----------|
| Debug | Baseline LR accuracy | 55% | 60% | 65% |
| Retrain | Individual CNN | 65% | 70% | 75% |
| Ensemble | Multi-resolution | 70% | 80% | 85% |
| Final | vs Phase 1 (63.4%) | Match | +5% | +10% |

---

## 💾 Disk Space

**Current usage**:
- GAF images: 39 GB (15m + 30m + 60m)
- Models: 372 MB (3 × 124 MB)
- Source data: ~500 MB
- **Total**: ~40 GB

**If retraining**:
- Keep old models (rename to _v1_old)
- New models: +372 MB per iteration
- Plan for ~50 GB total

---

## 🔧 Emergency Commands

**Reset everything**:
```bash
# Delete failed models (keeps data)
rm models/gaf_cnn_*m_v1.pth
rm models/gaf_cnn_*m_v1_metadata.pkl
```

**Regenerate GAF** (if data corrupt):
```bash
# Takes ~15 min per window
python scripts/generate_gaf_fixed.py --window_size 15
python scripts/generate_gaf_fixed.py --window_size 30
python scripts/generate_gaf_fixed.py --window_size 60
```

**Check processes**:
```bash
# See if training is running
ps aux | grep python | grep train_gaf

# Kill if needed
pkill -f train_gaf_cnn
```

---

## 📞 When to Give Up

**Abandon Phase 2 if**:
1. Data quality checks fail repeatedly
2. Best individual model < 60% after 5+ attempts
3. Ensemble still worse than individuals
4. Time invested > 40 hours

**Then**:
→ **Deploy Phase 1 (XGBoost)** - proven 63.4% win rate, 4.11 Sharpe
→ Focus on production deployment instead of R&D

---

## ✅ Final Reminder

**Goal**: Debug Phase 2 to achieve 70%+ ensemble accuracy

**Fallback**: Phase 1 (XGBoost) is already working at 63.4%

**Priority**: Data quality first, architecture second, hyperparameters third

**Timeline**: 15-30 hours of experimentation expected

**See**: `OPTION_A_CONTINUATION_GUIDE.md` for detailed steps

---

**Status**: Ready to continue on another machine ✅
