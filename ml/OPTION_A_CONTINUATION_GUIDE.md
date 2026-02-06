# Option A: Debug & Retrain - Continuation Guide

**Purpose**: Complete guide to resume Phase 2 GAF-CNN debugging and retraining on another machine
**Created**: 2026-02-06 01:05am
**Status**: Ready for continuation

---

## 📋 Executive Summary

### What Was Completed

✅ **Phase 2 GAF-CNN Training (All 3 Models)**
- 15m, 30m, 60m window models trained successfully
- All models achieved ~50-53% validation accuracy (random performance)
- Multi-resolution ensemble created and validated
- **Problem Identified**: Ensemble performs worse (47%) than individuals (52%)
- **Root Cause**: Models haven't learned meaningful patterns

### Current Situation

**Models Trained**: 3 CNN models (15m, 30m, 60m) - 372 MB total
**Performance**: 47-53% accuracy (target: 90%)
**Comparison**: Phase 1 XGBoost (63.4% win rate) significantly outperforms Phase 2 CNN (47%)
**Issue**: Models are essentially guessing randomly, ensemble makes it worse

### Goal of Option A

**Debug why models aren't learning**, then **retrain with fixes** to achieve:
- Individual model accuracy: 70%+ (vs current 50-53%)
- Ensemble accuracy: 80-85%+ (vs current 47%)
- Meet or exceed Phase 1 performance (63.4% win rate)

---

## 🗂️ Project Structure

### Directory Layout

```
/home/rom/LeverageBot/ml/
├── data/
│   ├── BTCUSDT_1m_2024.csv                    # Source data (1-min OHLCV)
│   └── gaf/
│       ├── bnb_15m/                           # 15m GAF images (13 GB)
│       │   ├── train_images.npy               # 186,598 images
│       │   ├── train_labels.npy               # Binary labels
│       │   ├── val_images.npy                 # 20,734 images
│       │   ├── val_labels.npy
│       │   └── metadata.pkl
│       ├── bnb_30m/                           # 30m GAF images (13 GB)
│       └── bnb_60m/                           # 60m GAF images (13 GB)
│
├── models/
│   ├── gaf_cnn_15m_v1.pth                     # Trained 15m model (124 MB)
│   ├── gaf_cnn_15m_v1_metadata.pkl
│   ├── gaf_cnn_30m_v1.pth                     # Trained 30m model (124 MB)
│   ├── gaf_cnn_30m_v1_metadata.pkl
│   ├── gaf_cnn_60m_v1.pth                     # Trained 60m model (124 MB)
│   ├── gaf_cnn_60m_v1_metadata.pkl
│   ├── ensemble.py                             # Ensemble class
│   ├── gaf_cnn.py                              # CNN architecture
│   └── ensemble_validation_results.pkl         # Validation results
│
├── gaf/
│   ├── gaf_transformer.py                      # GAF transformation functions
│   └── gaf_dataset.py                          # PyTorch dataset classes
│
├── scripts/
│   ├── generate_gaf_fixed.py                   # GAF image generation (WORKING)
│   ├── train_gaf_cnn_precomputed.py           # Training script (CURRENT)
│   └── validate_ensemble.py                    # Ensemble validation
│
├── logs/                                       # Training logs
├── ml_env/                                     # Python virtual environment
│
└── Documentation:
    ├── ML_PHASE_2_ANALYSIS.md                  # Performance analysis (READ FIRST!)
    ├── ML_PHASE_2_TRAINING_COMPLETE.md        # Training results summary
    ├── ML_TRAINING_STATUS.md                   # Training progress log
    └── OPTION_A_CONTINUATION_GUIDE.md         # THIS FILE
```

---

## 🔍 Phase 1: Data Quality Investigation

### Step 1.1: Verify Label Generation

**Why**: ~50% accuracy suggests labels might be random or incorrect

**Check label file**:
```python
import numpy as np

# Load labels
labels_15m = np.load('data/gaf/bnb_15m/train_labels.npy')

# Check distribution
print("Label distribution:")
print(f"  UP (1):   {np.sum(labels_15m == 1):,} ({np.mean(labels_15m == 1)*100:.1f}%)")
print(f"  DOWN (0): {np.sum(labels_15m == 0):,} ({np.mean(labels_15m == 0)*100:.1f}%)")

# Check for NaN or invalid values
print(f"NaN count: {np.isnan(labels_15m).sum()}")
print(f"Invalid values: {np.sum((labels_15m != 0) & (labels_15m != 1))}")
```

**Expected**: ~50/50 distribution (balanced)
**If imbalanced**: Label generation logic may be wrong

**Action if failed**: Review label generation in `scripts/generate_gaf_fixed.py` lines 50-75

### Step 1.2: Inspect GAF Image Quality

**Why**: If images don't contain patterns, CNN can't learn

**Visual inspection**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Load sample images
images = np.load('data/gaf/bnb_15m/train_images.npy', mmap_mode='r')

# Plot first 4 images
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    # GASF channel
    axes[0, i].imshow(images[i, :, :, 0], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, i].set_title(f'Sample {i} - GASF')
    axes[0, i].axis('off')

    # GADF channel
    axes[1, i].imshow(images[i, :, :, 1], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, i].set_title(f'Sample {i} - GADF')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('gaf_visual_inspection.png', dpi=150)
print("✅ Saved to gaf_visual_inspection.png")
```

**Expected**: Clear patterns/structures visible in images
**If blank/noise**: GAF transformation is broken

**Check value ranges**:
```python
print(f"Min value: {images[:1000].min()}")  # Should be ~-1
print(f"Max value: {images[:1000].max()}")  # Should be ~1
print(f"Mean: {images[:1000].mean()}")      # Should be ~0
print(f"NaN count: {np.isnan(images[:1000]).sum()}")  # Should be 0
```

**Action if failed**: Review GAF transformation in `gaf/gaf_transformer.py`

### Step 1.3: Check for Data Leakage

**Why**: If train/val aren't independent, validation is meaningless

**Verify chronological split**:
```python
# Check that validation comes AFTER training
with open('data/gaf/bnb_15m/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print("Split info:", metadata)

# Load source data and check timestamps
import pandas as pd
df = pd.read_csv('data/BTCUSDT_1m_2024.csv')
print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
```

**Expected**: Train data is earlier than validation data (chronological)
**If overlapping**: Rebuild GAF with proper time-based split

### Step 1.4: Baseline Model Test

**Why**: If simple model outperforms CNN, architecture is the problem

**Train logistic regression**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
train_images = np.load('data/gaf/bnb_15m/train_images.npy')
train_labels = np.load('data/gaf/bnb_15m/train_labels.npy')
val_images = np.load('data/gaf/bnb_15m/val_images.npy')
val_labels = np.load('data/gaf/bnb_15m/val_labels.npy')

# Flatten images
train_flat = train_images.reshape(len(train_images), -1)
val_flat = val_images.reshape(len(val_images), -1)

# Train simple logistic regression
print("Training baseline logistic regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(train_flat[:10000], train_labels[:10000])  # Subset for speed

# Evaluate
val_preds = lr.predict(val_flat)
acc = accuracy_score(val_labels, val_preds)

print(f"Baseline accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"CNN accuracy:      0.5280 (52.80%)")

if acc > 0.55:
    print("✅ Data contains learnable patterns (architecture issue)")
else:
    print("❌ Even simple model fails (data quality issue)")
```

---

## 🛠️ Phase 2: Architecture Fixes

### Step 2.1: Train Without Pretrained Weights

**Why**: ImageNet weights may not suit GAF images

**Modify training script** (`scripts/train_gaf_cnn_precomputed.py` line 84):
```python
# OLD:
model = GAF_CNN(pretrained=args.pretrained, dropout=args.dropout, freeze_early_layers=args.freeze_early)

# NEW:
model = GAF_CNN(pretrained=False, dropout=args.dropout, freeze_early_layers=False)
```

**Retrain**:
```bash
source ml_env/bin/activate
python scripts/train_gaf_cnn_precomputed.py --window_size 15 --pretrained False --freeze_early False
```

**Expected**: Should train longer before overfitting

### Step 2.2: Lower Learning Rate

**Why**: Current lr=1e-4 may be too high (causes immediate overfitting)

**Modify** (`scripts/train_gaf_cnn_precomputed.py` line 96):
```python
# OLD:
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# NEW (add CLI arg):
parser.add_argument('--learning_rate', type=float, default=1e-5)  # 10x lower
```

**Retrain**:
```bash
python scripts/train_gaf_cnn_precomputed.py --window_size 15 --learning_rate 1e-5
```

**Expected**: Slower but more stable learning

### Step 2.3: Increase Early Stopping Patience

**Why**: Current patience=10 may be too aggressive

**Modify** (`scripts/train_gaf_cnn_precomputed.py` line 154):
```python
# OLD:
parser.add_argument('--patience', type=int, default=10)

# NEW:
parser.add_argument('--patience', type=int, default=30)
```

**Retrain**:
```bash
python scripts/train_gaf_cnn_precomputed.py --window_size 15 --patience 30
```

**Expected**: Trains for 30+ epochs

### Step 2.4: Reduce Model Complexity

**Why**: ResNet18 (11M params) may be overkill for this task

**Create simpler architecture** (`models/simple_cnn.py`):
```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    """Simpler CNN architecture for GAF images."""

    def __init__(self, dropout=0.3):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # -> 1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

**Modify training script** to use SimpleCNN instead of GAF_CNN

**Expected**: Faster training, less overfitting

---

## 🔬 Phase 3: Hyperparameter Tuning

### Step 3.1: Grid Search Configuration

**Test these combinations**:

| Experiment | LR | Dropout | Pretrained | Batch Size | Patience |
|------------|-----|---------|------------|------------|----------|
| 1 (Baseline) | 1e-4 | 0.5 | True | 64 | 10 |
| 2 | 1e-5 | 0.5 | False | 64 | 30 |
| 3 | 1e-5 | 0.3 | False | 32 | 30 |
| 4 | 1e-6 | 0.2 | False | 64 | 50 |
| 5 | 5e-5 | 0.3 | True | 64 | 30 |

**Script** (`scripts/hyperparameter_search.py`):
```python
import subprocess

configs = [
    {'lr': 1e-4, 'dropout': 0.5, 'pretrained': True, 'batch': 64, 'patience': 10},
    {'lr': 1e-5, 'dropout': 0.5, 'pretrained': False, 'batch': 64, 'patience': 30},
    {'lr': 1e-5, 'dropout': 0.3, 'pretrained': False, 'batch': 32, 'patience': 30},
    {'lr': 1e-6, 'dropout': 0.2, 'pretrained': False, 'batch': 64, 'patience': 50},
    {'lr': 5e-5, 'dropout': 0.3, 'pretrained': True, 'batch': 64, 'patience': 30},
]

for i, cfg in enumerate(configs):
    print(f"Running experiment {i+1}/5...")

    cmd = [
        'python', 'scripts/train_gaf_cnn_precomputed.py',
        '--window_size', '15',
        '--learning_rate', str(cfg['lr']),
        '--dropout', str(cfg['dropout']),
        '--pretrained', str(cfg['pretrained']),
        '--batch_size', str(cfg['batch']),
        '--patience', str(cfg['patience'])
    ]

    subprocess.run(cmd)

    # Rename output to preserve
    subprocess.run(['mv', 'models/gaf_cnn_15m_v1.pth', f'models/gaf_cnn_15m_exp{i+1}.pth'])

print("✅ All experiments complete")
```

**Run**:
```bash
source ml_env/bin/activate
python scripts/hyperparameter_search.py
```

**Expected**: One configuration should achieve >60% validation accuracy

### Step 3.2: Data Augmentation

**Add augmentation** to `gaf/gaf_dataset.py`:
```python
import torchvision.transforms as T

class GAFAugmentation:
    def __init__(self):
        self.transforms = T.Compose([
            T.RandomRotation(5),  # ±5 degrees
            T.RandomAffine(0, scale=(0.95, 1.05)),  # ±5% scale
            T.GaussianNoise(0, 0.01),  # Add noise
        ])

    def __call__(self, img):
        return self.transforms(img)
```

**Modify GAFDataset** (line 96):
```python
if self.transform is not None and self.mode == 'train':
    image = self.transform(image)
# ADD:
elif self.mode == 'train':
    image = GAFAugmentation()(image)
```

**Expected**: Better generalization, less overfitting

---

## 📊 Phase 4: Alternative Approaches

### Option 4A: Different Image Representation

Instead of GAF, try:

1. **Recurrence Plots**
2. **Markov Transition Fields**
3. **Raw spectrograms**
4. **Multiple Technical Indicators as Image Channels**

### Option 4B: 1D CNN (Skip Image Transformation)

**Train on raw time series**:
```python
class Conv1D_Model(nn.Module):
    def __init__(self, seq_len=15, features=5):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
```

**Advantage**: No image transformation needed, simpler preprocessing

### Option 4C: Combine with Phase 1

**Hybrid ensemble**:
- Phase 1 (XGBoost): Technical indicators
- Phase 2 (CNN): Pattern recognition
- Meta-learner: Combine both

**Expected**: Best of both worlds

---

## ✅ Success Criteria

### Minimum Viable Performance

| Metric | Current | Minimum Target | Stretch Goal |
|--------|---------|----------------|--------------|
| Individual Model Accuracy | 50-53% | **65%** | 75% |
| Ensemble Accuracy | 47% | **70%** | 85% |
| Phase 1 + 2 Win Rate | - | **65%** | 72% |

**Deployment decision**:
- If ensemble < 60%: Use Phase 1 (XGBoost) only
- If ensemble 60-70%: Consider hybrid approach
- If ensemble > 70%: Deploy Phase 2 ensemble

---

## 🗂️ Files to Review Before Starting

**Priority 1 (Must Read)**:
1. `ML_PHASE_2_ANALYSIS.md` - Problem analysis and root causes
2. `ML_PHASE_2_TRAINING_COMPLETE.md` - Training results summary
3. `scripts/generate_gaf_fixed.py` - Label generation logic (lines 50-75)
4. `gaf/gaf_transformer.py` - GAF transformation implementation

**Priority 2 (Reference)**:
5. `models/gaf_cnn.py` - Current CNN architecture
6. `scripts/train_gaf_cnn_precomputed.py` - Training script
7. `models/ensemble.py` - Ensemble implementation

**Priority 3 (Context)**:
8. `ML_TRAINING_STATUS.md` - Training progress log
9. `scripts/validate_ensemble.py` - Validation methodology

---

## 🚀 Quick Start Commands

### Setup Environment

```bash
cd /home/rom/LeverageBot/ml
source ml_env/bin/activate

# Verify dependencies
python -c "import torch, torchvision, numpy, sklearn; print('✅ All imports OK')"

# Check GPU availability (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Run Data Quality Checks

```bash
# Label distribution
python -c "
import numpy as np
labels = np.load('data/gaf/bnb_15m/train_labels.npy')
print(f'UP: {np.mean(labels==1)*100:.1f}%, DOWN: {np.mean(labels==0)*100:.1f}%')
"

# Image value ranges
python -c "
import numpy as np
images = np.load('data/gaf/bnb_15m/train_images.npy', mmap_mode='r')
print(f'Min: {images[:1000].min():.2f}, Max: {images[:1000].max():.2f}')
"
```

### Train Single Model (Test)

```bash
# Quick test (5 epochs, 15m window)
python scripts/train_gaf_cnn_precomputed.py \
    --window_size 15 \
    --epochs 5 \
    --learning_rate 1e-5 \
    --pretrained False
```

### Validate Results

```bash
python scripts/validate_ensemble.py
```

---

## 📝 Debugging Checklist

Before retraining, verify:

- [ ] Labels are balanced (~50/50 UP/DOWN)
- [ ] GAF images contain visible patterns
- [ ] No NaN or Inf values in data
- [ ] Train/val split is chronological (no leakage)
- [ ] Baseline logistic regression achieves >55% accuracy
- [ ] GPU available (optional, for faster training)

After each training run, check:

- [ ] Validation accuracy improves over epochs (not just epoch 1)
- [ ] Training doesn't overfit immediately
- [ ] Model saved properly (check file size ~124 MB)
- [ ] Ensemble accuracy > individual models
- [ ] Compare to Phase 1 (XGBoost) performance

---

## 💡 Key Insights to Remember

1. **Current models are essentially guessing** - ~50% = random
2. **Ensemble made it worse** - Indicates fundamental problem, not just need for tuning
3. **All models peaked at epoch 1** - Strong signal of overfitting or data issues
4. **Phase 1 is still better** - XGBoost 63.4% vs CNN ensemble 47%
5. **Data quality is most likely culprit** - Check labels and GAF images first

---

## 📞 Contact Points (If Stuck)

**Key questions to ask**:
1. Do the GAF images visually show different patterns for UP vs DOWN?
2. What does baseline logistic regression achieve?
3. Does training from scratch (no pretrained weights) help?
4. Should we abandon Phase 2 and use Phase 1 only?

**Alternative approaches if debugging fails**:
- Use Phase 1 (XGBoost) exclusively - already proven to work
- Simplify to technical indicators + simple neural network
- Try traditional ML instead of deep learning

---

## ⏱️ Estimated Timeline

| Phase | Task | Time Estimate |
|-------|------|---------------|
| 1 | Data quality investigation | 2-4 hours |
| 2 | Architecture fixes | 3-5 hours |
| 3 | Hyperparameter tuning | 6-12 hours |
| 4 | Alternative approaches | 4-8 hours |
| **Total** | | **15-29 hours** |

**Note**: This is experimentation time, not including model training time (~45 min per model)

---

## ✅ Final Checklist Before Deployment

- [ ] Ensemble accuracy > 70%
- [ ] Outperforms or matches Phase 1 (63.4% win rate)
- [ ] Tested on out-of-sample data
- [ ] No data leakage verified
- [ ] Model inference time acceptable (<100ms)
- [ ] Memory requirements reasonable (<2 GB)

If all checks pass → Proceed to deployment
If not → Recommend using Phase 1 (XGBoost) instead

---

**Good luck with the debugging! The data is all there, just needs proper investigation and fixes.**
