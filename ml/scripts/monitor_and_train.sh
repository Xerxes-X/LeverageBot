#!/bin/bash
# Monitor GAF generation and auto-start training when ready

cd /home/rom/LeverageBot/ml
source ml_env/bin/activate

echo "==================================================================="
echo "GAF-CNN Training Monitor"
echo "==================================================================="
echo ""

# Function to check if GAF generation is complete
check_gaf_complete() {
    local window=$1
    local dir="data/gaf/bnb_${window}m"

    # Check if all 4 required files exist and are non-empty
    if [ -f "$dir/train_images.npy" ] && [ -s "$dir/train_images.npy" ] && \
       [ -f "$dir/train_labels.npy" ] && [ -s "$dir/train_labels.npy" ] && \
       [ -f "$dir/val_images.npy" ] && [ -s "$dir/val_images.npy" ] && \
       [ -f "$dir/val_labels.npy" ] && [ -s "$dir/val_labels.npy" ]; then
        return 0  # Complete
    else
        return 1  # Not complete
    fi
}

# Function to check if training is already running
check_training_running() {
    local window=$1
    if ps aux | grep -q "[t]rain_gaf_cnn_precomputed.py --window_size $window"; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

# Function to start training
start_training() {
    local window=$1
    echo "✅ Starting ${window}m CNN training..."
    nohup python scripts/train_gaf_cnn_precomputed.py --window_size $window > logs/cnn_${window}m.log 2>&1 &
    echo "   PID: $!"
    echo "   Log: logs/cnn_${window}m.log"
    echo ""
}

# Monitor loop
while true; do
    echo "[$(date '+%H:%M:%S')] Checking status..."

    # Check 30m
    if check_gaf_complete 30; then
        if ! check_training_running 30; then
            echo "🎯 30m GAF generation complete!"
            start_training 30
        else
            echo "✅ 30m training already running"
        fi
    else
        echo "⏳ 30m GAF still generating..."
    fi

    # Check 60m
    if check_gaf_complete 60; then
        if ! check_training_running 60; then
            echo "🎯 60m GAF generation complete!"
            start_training 60
        else
            echo "✅ 60m training already running"
        fi
    else
        echo "⏳ 60m GAF still generating..."
    fi

    # Check if all trainings are done
    if check_training_running 15 || check_training_running 30 || check_training_running 60; then
        echo "🔄 Training in progress..."
    else
        # Check if all GAF generations are complete
        if check_gaf_complete 30 && check_gaf_complete 60; then
            echo ""
            echo "==================================================================="
            echo "✅✅ ALL TRAINING COMPLETE!"
            echo "==================================================================="
            echo ""
            echo "Trained models:"
            ls -lh models/gaf_cnn_*m_v1.pth 2>/dev/null
            echo ""
            echo "Next steps:"
            echo "  - Task #20: Create ensemble (scripts/create_ensemble.py)"
            echo "  - Task #21: Validate Phase 2 performance"
            echo ""
            break
        fi
    fi

    echo ""
    sleep 60  # Check every minute
done

echo "Monitor script complete."
