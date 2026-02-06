#!/bin/bash
# Train all 3 GAF-CNN models sequentially (safe - no system freezing!)
# Total time: ~9-10 hours

set -e  # Exit on any error

cd /home/rom/LeverageBot/ml
source ml_env/bin/activate

echo "=================================================================="
echo "GAF-CNN Sequential Training Pipeline"
echo "=================================================================="
echo ""
echo "⚠️  This will take ~9-10 hours total"
echo "✅ Safe memory usage: Max 14 GB per model (no freezing!)"
echo ""
echo "Training schedule:"
echo "  1. 15m model: ~2-3 hours"
echo "  2. 30m model: ~2-3 hours"
echo "  3. 60m model: ~2-3 hours"
echo ""
echo "Press Ctrl+C in the next 5 seconds to cancel..."
sleep 5
echo ""

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check memory before each training
check_memory() {
    free_mem=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$free_mem" -lt 15 ]; then
        log "⚠️  WARNING: Only ${free_mem}GB free RAM (need 15GB+)"
        log "   Waiting 30 seconds for memory to clear..."
        sleep 30
    fi
}

# Step 1: Train 15m CNN
log "=================================================================="
log "Step 1/3: Training 15m CNN Model"
log "=================================================================="
check_memory
python scripts/train_gaf_cnn_precomputed.py --window_size 15 2>&1 | tee logs/cnn_15m_final.log
if [ $? -eq 0 ]; then
    log "✅ 15m CNN training complete"
else
    log "❌ 15m CNN training failed - stopping pipeline"
    exit 1
fi
echo ""

# Step 2: Train 30m CNN
log "=================================================================="
log "Step 2/3: Training 30m CNN Model"
log "=================================================================="
check_memory
python scripts/train_gaf_cnn_precomputed.py --window_size 30 2>&1 | tee logs/cnn_30m_final.log
if [ $? -eq 0 ]; then
    log "✅ 30m CNN training complete"
else
    log "❌ 30m CNN training failed - stopping pipeline"
    exit 1
fi
echo ""

# Step 3: Train 60m CNN
log "=================================================================="
log "Step 3/3: Training 60m CNN Model"
log "=================================================================="
check_memory
python scripts/train_gaf_cnn_precomputed.py --window_size 60 2>&1 | tee logs/cnn_60m_final.log
if [ $? -eq 0 ]; then
    log "✅ 60m CNN training complete"
else
    log "❌ 60m CNN training failed - stopping pipeline"
    exit 1
fi
echo ""

# Summary
log "=================================================================="
log "✅✅✅ ALL TRAINING COMPLETE!"
log "=================================================================="
log ""
log "Trained models:"
ls -lh models/gaf_cnn_*m_v1.pth
log ""
log "Model metadata:"
ls -lh models/gaf_cnn_*m_v1_metadata.pkl
log ""
log "Training logs:"
log "  - logs/cnn_15m_final.log"
log "  - logs/cnn_30m_final.log"
log "  - logs/cnn_60m_final.log"
log ""
log "Next steps:"
log "  - Task #20: Create multi-resolution ensemble"
log "  - Task #21: Validate Phase 2 performance"
log "  - Integration with Phase 1 (XGBoost)"
log ""
log "Pipeline completed at: $(date)"
