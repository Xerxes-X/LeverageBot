# Phase 2: GAF-CNN Training Progress

**Status**: ✅ In Progress - All pipelines running successfully
**Date**: 2026-02-05
**Approach**: Multi-resolution ensemble (15m, 30m, 60m windows)

---

## 🎯 Phase 2 Targets

| Metric | Target | Phase 1 Baseline |
|--------|--------|------------------|
| Win Rate | 65-70% | 63.4% |
| Pattern Recognition Accuracy | >90% | N/A (XGBoost) |
| Sharpe Ratio | 2.0-2.5 | 4.11 |
| Sortino Ratio | 3.0-3.5 | 5.61 |

---

## 📊 Current Status (as of 21:10 Feb 5)

### GAF Image Generation

| Window | Status | Progress | Data Size | ETA |
|--------|--------|----------|-----------|-----|
| **15m** | ✅ **COMPLETE** | 100% | 13 GB (train + val) | Done |
| **30m** | 🔄 Running | ~91% (Chunk 19/21) | Generating... | ~5 min |
| **60m** | 🔄 Running | ~33% (Chunk 7/21) | Generating... | ~20 min |

**GAF Specification**:
- Image size: 64×64 pixels
- Channels: 2 (GASF + GADF)
- Train/val split: 90/10 (chronological)
- Total samples per window: ~207,000 images
- Memory per dataset: ~13 GB

### CNN Training

| Window | Status | Progress | Best Val Acc | ETA |
|--------|--------|----------|--------------|-----|
| **15m** | 🔄 **Training** | Epoch 1/50 (~8 it/s) | N/A (just started) | 2-3 hours |
| **30m** | ⏳ Queued | Waiting for GAF generation | - | Start in ~5 min |
| **60m** | ⏳ Queued | Waiting for GAF generation | - | Start in ~20 min |

**Training Configuration**:
- Model: ResNet18 (transfer learning from ImageNet)
- Frozen layers: layer1, layer2 (early feature extraction)
- Trainable params: ~10.6M / 11.2M total
- Batch size: 64
- Learning rate: 1e-4 with cosine annealing
- Optimizer: AdamW (weight_decay=1e-5)
- Early stopping: patience=10
- Device: CPU (no GPU required)
- Expected time per model: **2-3 hours on CPU**

---

## 🔧 Technical Implementation

### Memory Optimization (Fixed!)

**Issue Solved**: Original GAF generation failed with 0-byte output files due to OOM during list-to-array conversion.

**Root Cause**:
```
List of 207K images: 12.65 GB
+ Converting to numpy array: 12.65 GB
= Peak memory: ~25 GB (near 31 GB system limit)
```

**Solution**: Pre-allocate numpy array with `np.empty()` and fill directly
- Memory usage stays at ~13 GB (no temporary list)
- All 207K images generated successfully
- Saved as separate files to avoid pickle protocol 4 issue (>4GB objects)

### File Structure

```
ml/data/gaf/
├── bnb_15m/
│   ├── train_images.npy  (12 GB - 186,598 images)
│   ├── train_labels.npy  (1.5 MB)
│   ├── val_images.npy    (1.3 GB - 20,734 images)
│   ├── val_labels.npy    (163 KB)
│   └── metadata.pkl
├── bnb_30m/  (generating...)
└── bnb_60m/  (generating...)
```

### Training Pipeline

**Script**: `scripts/train_gaf_cnn_precomputed.py`

**Architecture**:
- ResNet18 backbone (pretrained on ImageNet)
- Modified first conv layer: 3 channels → 2 channels (GASF + GADF)
- Weight initialization: Average RGB channels from pretrained weights
- Custom classifier head:
  - Dropout(0.5)
  - Linear(512 → 128) + ReLU + BatchNorm
  - Dropout(0.3)
  - Linear(128 → 1) [binary classification]

**Advantages of Pre-computed Approach**:
- ✅ Faster training (no on-the-fly GAF generation)
- ✅ Consistent images across epochs
- ✅ Larger batch sizes possible (64 vs 32)
- ✅ Better for debugging/reproducibility

---

## 📈 Expected Timeline

**Tonight (Feb 5)**:
- ✅ 21:04 - 15m GAF generation complete
- ✅ 21:08 - 15m CNN training started
- 🔄 21:12 - 30m GAF generation complete (EST)
- 🔄 21:12 - 30m CNN training start (EST)
- 🔄 21:28 - 60m GAF generation complete (EST)
- 🔄 21:28 - 60m CNN training start (EST)

**Tomorrow Morning (Feb 6)**:
- ✅ ~00:00 - All 3 CNN models trained
- 🎯 Build multi-resolution ensemble (Task #20)
- 🎯 Validate Phase 2 performance (Task #21)

**Total time**: ~6 hours end-to-end (generation + training)

---

## 🔍 Monitoring

### Check GAF Generation Progress
```bash
# 30m progress
tail -5 logs/gaf_30m.log

# 60m progress
tail -5 logs/gaf_60m.log
```

### Check CNN Training Progress
```bash
# 15m training (currently running)
tail -20 logs/cnn_15m.log

# When started:
tail -20 logs/cnn_30m.log
tail -20 logs/cnn_60m.log
```

### Check All Running Processes
```bash
ps aux | grep -E "(generate_gaf|train_gaf_cnn)" | grep -v grep
```

### System Resources
- **CPU**: ~300% usage across 3 processes (normal for parallel work)
- **Memory**: ~17.7 GB / 31 GB (57%) - safe headroom
- **Disk**: ~13 GB per GAF dataset = 39 GB total + models

---

## 🎓 Literature Baseline

Based on peer-reviewed research (5+ papers):
- **GAF-CNN pattern recognition**: 90-93% accuracy
- **Multi-resolution ensemble**: +2-5% improvement over single window
- **Transfer learning (ImageNet → Finance)**: Faster convergence, better generalization

**Our target**: 90%+ pattern recognition accuracy, which is achievable based on literature.

---

## 🚀 Next Steps (After Training Completes)

### Task #20: Create Ensemble
- Load all 3 trained models (15m, 30m, 60m)
- Implement voting/averaging mechanism
- Test ensemble on validation set
- Compare ensemble vs individual models

### Task #21: Validate Phase 2
- Generate predictions on holdout test set
- Calculate win rate, Sharpe, Sortino
- Compare to Phase 1 (XGBoost) baseline
- Analyze pattern recognition accuracy (confusion matrix)
- Identify regime-specific performance

### Phase 1 + Phase 2 Integration
- Combine XGBoost (Phase 1) + GAF-CNN (Phase 2)
- Two-stage ensemble:
  - XGBoost: Market conditions, technical signals
  - GAF-CNN: Pattern recognition, microstructure
- Expected improvement: 5-10% win rate boost

---

## 📝 Notes

1. **CPU Training**: Pre-computed images make CPU training viable (2-3 hours vs 6-9 hours on-the-fly)
2. **Memory Management**: Pre-allocation strategy solved all OOM issues
3. **Pickle Protocol**: Saving arrays separately (not in dict) avoids protocol 4 requirement
4. **No GPU Needed**: CPU sufficient for both generation and training
5. **Background Execution**: All processes run in background with logs for monitoring

---

## ✅ Milestones

- [x] Research GAF-CNN architecture (Task #15)
- [x] Design implementation plan (Task #16)
- [x] Implement GAF transformation (Task #17)
- [x] Build CNN architecture (Task #18)
- [x] Solve memory optimization issues
- [x] Generate 15m GAF images
- [x] Start 15m CNN training
- [🔄] Generate 30m, 60m GAF images (in progress)
- [ ] Train 30m, 60m CNN models (queued)
- [ ] Build ensemble (Task #20)
- [ ] Validate Phase 2 (Task #21)
