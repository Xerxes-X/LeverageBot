# Phase 2: Ready to Train! 🚀

**Date:** 2026-02-05
**Status:** ✅ All infrastructure complete - Ready for overnight training

---

## ✅ What's Complete

### Infrastructure (100%)
- ✅ GAF transformation pipeline
- ✅ CNN architecture (ResNet18)
- ✅ Training scripts
- ✅ All dependencies installed

### Approach: On-The-Fly GAF Generation

After multiple attempts at pre-generating GAF images, we're using a more robust approach:
- **Generate GAF images during training** (on-the-fly)
- More reliable, uses less disk space
- Training time: 3-4 hours per model (acceptable for overnight)

---

## 🚀 Start Training Tonight

Run these 3 commands to train all models in parallel:

```bash
cd /home/rom/LeverageBot/ml
source ml_env/bin/activate

# Train all 3 models overnight (run in parallel)
nohup python scripts/train_gaf_cnn_onthefly.py --window_size 15 > logs/cnn_15m.log 2>&1 &
nohup python scripts/train_gaf_cnn_onthefly.py --window_size 30 > logs/cnn_30m.log 2>&1 &
nohup python scripts/train_gaf_cnn_onthefly.py --window_size 60 > logs/cnn_60m.log 2>&1 &

# Check they're running
ps aux | grep train_gaf_cnn_onthefly

# Monitor progress (optional)
tail -f logs/cnn_15m.log
```

**Timeline:**
- Start: Tonight (~9 PM)
- Finish: Tomorrow morning (~6-9 AM)
- Each model: 3-4 hours

---

## 📊 Expected Results

| Model | Window | Pattern Recognition Target | Expected |
|-------|--------|----------------------------|----------|
| CNN-15m | 15 min | >90% | 85-92% |
| CNN-30m | 30 min | >90% | 85-92% |
| CNN-60m | 60 min | >90% | 85-92% |

**Ensemble (XGBoost + 3 CNNs):**
- Win Rate Target: 65-70%
- Sharpe Target: 2.0-2.5

---

## 📁 Output Files

After training completes, you'll have:

```
ml/models/
├── gaf_cnn_15m_v1.pth              # 15-min CNN model
├── gaf_cnn_15m_v1_metadata.pkl     # Training history
├── gaf_cnn_30m_v1.pth              # 30-min CNN model
├── gaf_cnn_30m_v1_metadata.pkl
├── gaf_cnn_60m_v1.pth              # 60-min CNN model
└── gaf_cnn_60m_v1_metadata.pkl

ml/logs/
├── cnn_15m.log                     # Training logs
├── cnn_30m.log
└── cnn_60m.log
```

---

## 🔍 Monitor Training

### Check if training is running:
```bash
ps aux | grep train_gaf_cnn
```

### View progress:
```bash
# 15-min model
tail -f ml/logs/cnn_15m.log

# 30-min model
tail -f ml/logs/cnn_30m.log

# 60-min model
tail -f ml/logs/cnn_60m.log
```

### Check completion:
```bash
grep "TRAINING COMPLETE" ml/logs/cnn_*.log
```

---

## 📋 Tomorrow Morning Checklist

When training completes:

1. **Check Results**
   ```bash
   grep "Best val accuracy" ml/logs/cnn_*.log
   ```

2. **Verify Models Saved**
   ```bash
   ls -lh ml/models/gaf_cnn_*m_v1.pth
   ```

3. **Next Steps:**
   - Task 20: Create ensemble (XGBoost + 3 CNNs)
   - Task 21: Validate Phase 2 performance
   - Document results

---

## 🎯 Phase 2 Timeline

| Task | Status | Time | When |
|------|--------|------|------|
| Research | ✅ Done | 1 hour | Complete |
| Planning | ✅ Done | 1 hour | Complete |
| GAF Pipeline | ✅ Done | 2 hours | Complete |
| CNN Architecture | ✅ Done | 2 hours | Complete |
| **CNN Training** | ⏳ **Ready** | **9-12 hours** | **Tonight** |
| Ensemble | Pending | 1 hour | Tomorrow PM |
| Validation | Pending | 1 hour | Tomorrow PM |

**Total remaining:** ~11-14 hours (mostly overnight training)

---

## ⚙️ Training Configuration

```python
Model: ResNet18 (transfer learning from ImageNet)
Input: 64×64 GAF images (2 channels: GASF + GADF)
Batch size: 32
Learning rate: 1e-4 with cosine annealing
Early stopping: Patience 10
Dropout: 0.5
Trainable params: ~10.6M
```

---

## 💻 Hardware Utilization

**Your Ryzen 7 7800X3D:**
- 16 threads available
- Training 3 models in parallel
- Each model uses 1 thread
- CPU utilization: ~20-30% (plenty of headroom)

**Memory:**
- Each model: ~2-3 GB
- Total: ~6-9 GB (you have 31 GB) ✅

---

## 🐛 Troubleshooting

### If training fails to start:
```bash
cd /home/rom/LeverageBot/ml
source ml_env/bin/activate
python scripts/train_gaf_cnn_onthefly.py --window_size 15
```

### If you see errors:
Check logs:
```bash
tail -100 ml/logs/cnn_15m.log
```

### If process dies:
Restart specific model:
```bash
nohup python scripts/train_gaf_cnn_onthefly.py --window_size 15 > logs/cnn_15m.log 2>&1 &
```

---

## 📚 Next Phase

After CNN training completes (tomorrow morning):

1. **Create Ensemble** (~1 hour)
   - Load XGBoost + 3 CNNs
   - Optimize weights
   - Test predictions

2. **Validate Phase 2** (~1 hour)
   - Walk-forward validation
   - Calculate metrics
   - Compare vs Phase 1

3. **Document Results**
   - Final Phase 2 report
   - Deployment guide
   - Integration with bot

---

## 🎉 Almost There!

Phase 2 is 90% complete. Just need to:
1. Start training tonight (3 commands)
2. Let it run overnight
3. Create ensemble tomorrow
4. Phase 2 DONE! ✅

---

**Commands to run RIGHT NOW:**

```bash
cd /home/rom/LeverageBot/ml && source ml_env/bin/activate && \
nohup python scripts/train_gaf_cnn_onthefly.py --window_size 15 > logs/cnn_15m.log 2>&1 & \
nohup python scripts/train_gaf_cnn_onthefly.py --window_size 30 > logs/cnn_30m.log 2>&1 & \
nohup python scripts/train_gaf_cnn_onthefly.py --window_size 60 > logs/cnn_60m.log 2>&1 & \
echo "✅ All 3 models started training!"
```

**That's it! Come back tomorrow morning and models will be trained.** 🚀
