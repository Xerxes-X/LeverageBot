#!/bin/bash
# Start all ML services (development mode)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ML_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ML_DIR"

echo "=========================================="
echo "LeverageBot ML Services Startup"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "ml_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ml_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ml_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if model exists
MODEL_PATH="models/xgboost_phase1_v1.pkl"
if [ ! -f "$MODEL_PATH" ]; then
    echo ""
    echo "⚠️  Warning: Model not found at $MODEL_PATH"
    echo ""
    echo "To train a model:"
    echo "  1. Download data: python scripts/download_data.py --symbol WBNBUSDT --interval 1m --days 60"
    echo "  2. Train model: python scripts/train_xgboost.py --config configs/xgboost_baseline.yaml"
    echo ""
    read -p "Continue without model? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create logs directory
mkdir -p logs

# Start MLflow (optional)
START_MLFLOW=${START_MLFLOW:-false}
if [ "$START_MLFLOW" = "true" ]; then
    echo ""
    echo "Starting MLflow tracking server..."
    mlflow server \
        --host 0.0.0.0 \
        --port 5000 \
        --backend-store-uri experiments/ \
        > logs/mlflow.log 2>&1 &

    MLFLOW_PID=$!
    echo "MLflow started (PID: $MLFLOW_PID)"
    echo "  URL: http://localhost:5000"
fi

# Start ML service
echo ""
echo "Starting ML prediction service..."
echo "  URL: http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"
echo ""

if [ -f "$MODEL_PATH" ]; then
    export MODEL_PATH="$MODEL_PATH"
fi

# Start with auto-reload in development
uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info

# Cleanup on exit
trap "echo 'Shutting down...'; [ ! -z '$MLFLOW_PID' ] && kill $MLFLOW_PID 2>/dev/null; exit" INT TERM
