"""
FastAPI ML Prediction Service.

High-performance inference service for XGBoost price direction predictions.
Target latency: <10ms (p95)

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import pickle
import time
import yaml
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import logging
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfo
)
from features.feature_transformer import FeatureTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LeverageBot ML Service",
    description="XGBoost price direction prediction service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class ServiceState:
    """Global service state."""

    def __init__(self):
        self.model = None
        self.transformer = None
        self.model_metadata = {}
        self.feature_names = []
        self.start_time = time.time()
        self.total_predictions = 0
        self.latency_history = deque(maxlen=1000)  # Keep last 1000 latencies

        # Configuration
        self.confidence_threshold = 0.55
        self.kelly_fraction = 0.25
        self.max_position_size = 0.5

    def load_model(self, model_path: str, metadata_path: str):
        """Load model and metadata."""
        logger.info(f"Loading model from {model_path}...")

        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.model_metadata = yaml.safe_load(f)
        else:
            logger.warning(f"Metadata not found: {metadata_path}")
            self.model_metadata = {
                'model_name': 'xgboost_phase1',
                'model_version': '1.0',
                'training_date': 'unknown'
            }

        # Load feature transformer
        transformer_path = model_path.replace('.pkl', '_transformer_state.pkl')
        if os.path.exists(transformer_path):
            self.transformer = FeatureTransformer()
            self.transformer.load_state(transformer_path)
            self.feature_names = self.transformer.get_feature_names()
        else:
            logger.warning(f"Transformer state not found: {transformer_path}")
            self.transformer = FeatureTransformer()
            self.feature_names = []

        logger.info(f"Model loaded successfully: {self.model_metadata.get('model_name', 'unknown')}")
        logger.info(f"Features: {len(self.feature_names)}")

    def calculate_position_size(self, confidence: float, capital: float = 100000) -> float:
        """Calculate recommended position size based on confidence."""
        # Map confidence to edge
        edge = max(0, (confidence - 0.5) * 2)

        # Fractional Kelly
        kelly_size = edge * self.kelly_fraction

        # Apply to capital
        position_size = capital * kelly_size

        # Cap at max
        max_size = capital * self.max_position_size
        return min(position_size, max_size)

    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time

    def get_avg_latency(self) -> float:
        """Get average latency from recent predictions."""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)

    def record_prediction(self, latency_ms: float):
        """Record prediction metrics."""
        self.total_predictions += 1
        self.latency_history.append(latency_ms)


# Global state instance
state = ServiceState()


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("Starting LeverageBot ML Service...")

    # Load model
    model_path = os.getenv('MODEL_PATH', 'models/xgboost_phase1_v1.pkl')
    metadata_path = model_path.replace('.pkl', '_metadata.yaml')

    if os.path.exists(model_path):
        state.load_model(model_path, metadata_path)
    else:
        logger.error(f"Model not found: {model_path}")
        logger.warning("Service starting without model - predictions will fail")

    logger.info("Service ready!")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "service": "LeverageBot ML Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    status = "healthy" if state.model is not None else "unhealthy"

    return HealthResponse(
        status=status,
        model_loaded=state.model is not None,
        model_version=state.model_metadata.get('model_version', 'unknown'),
        uptime_seconds=state.get_uptime(),
        total_predictions=state.total_predictions,
        avg_latency_ms=state.get_avg_latency()
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info():
    """Get model information."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metrics = state.model_metadata.get('metrics', {})

    return ModelInfo(
        name=state.model_metadata.get('model_name', 'unknown'),
        version=state.model_metadata.get('model_version', 'unknown'),
        trained_date=state.model_metadata.get('training_date', 'unknown'),
        training_samples=metrics.get('n_samples', 0),
        n_features=len(state.feature_names),
        feature_names=state.feature_names[:10] + ['...'] if len(state.feature_names) > 10 else state.feature_names,
        metrics=metrics
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Get price direction prediction.

    Returns prediction with confidence, direction, and recommended position size.
    Target latency: <10ms (p95)
    """
    start_time = time.perf_counter()

    # Check model loaded
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to features
        # For now, use a simplified approach - in production, maintain full state
        market_data = {
            'close': request.price,
            'open': request.open or request.price,
            'high': request.high or request.price * 1.01,
            'low': request.low or request.price * 0.99,
            'volume': request.volume,
            'bid_volume': request.bid_volume or request.volume * 0.5,
            'ask_volume': request.ask_volume or request.volume * 0.5,
            'best_bid': request.best_bid or request.price * 0.999,
            'best_ask': request.best_ask or request.price * 1.001,
            'bid_depth_5': request.bid_depth_5 or request.volume,
            'ask_depth_5': request.ask_depth_5 or request.volume,
        }

        # Create mini DataFrame for feature engineering
        df = pd.DataFrame([market_data])

        # Add BTC data if available
        if request.btc_price:
            df['btc_close'] = request.btc_price

        if request.funding_rate:
            df['funding_rate'] = request.funding_rate

        # Engineer features
        # Note: This is simplified - in production, maintain rolling state
        features_df = state.transformer.transform(df)

        if features_df.empty or features_df.isna().any().any():
            # Fallback: use simplified features
            logger.warning("Feature engineering produced NaN, using simplified approach")
            features = np.zeros(len(state.feature_names))
            # Set some basic features
            features[0] = request.price  # EMA proxy
        else:
            features = features_df.iloc[0].values

        # Reshape for prediction
        features = features.reshape(1, -1)

        # Model inference
        pred_proba = state.model.predict_proba(features)[0, 1]

        # Calculate confidence and direction
        confidence = abs(pred_proba - 0.5) * 2  # Map [0.5, 1.0] to [0, 1]
        direction = 'LONG' if pred_proba > 0.5 else 'SHORT'

        # Should trade?
        should_trade = confidence >= state.confidence_threshold

        # Recommended position size
        recommended_size = None
        if should_trade:
            recommended_size = state.calculate_position_size(confidence)

        # Get top contributing features (simplified)
        feature_importance = state.model.feature_importances_
        top_indices = np.argsort(feature_importance)[-5:][::-1]
        top_features = [
            {
                'name': state.feature_names[i] if i < len(state.feature_names) else f'feature_{i}',
                'value': float(features[0, i]) if i < features.shape[1] else 0.0
            }
            for i in top_indices
        ]

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics
        state.record_prediction(latency_ms)

        # Create response
        response = PredictionResponse(
            direction=direction,
            confidence=confidence,
            raw_probability=pred_proba,
            should_trade=should_trade,
            recommended_size=recommended_size,
            model_version=state.model_metadata.get('model_version', '1.0'),
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            top_features=top_features
        )

        # Log slow predictions
        if latency_ms > 10:
            logger.warning(f"Slow prediction: {latency_ms:.2f}ms")

        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(requests: List[PredictionRequest]):
    """
    Batch prediction endpoint.

    More efficient for multiple predictions.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []

    for req in requests:
        try:
            pred = await predict(req)
            results.append(pred)
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            results.append(None)

    return {
        'predictions': results,
        'total': len(requests),
        'successful': len([r for r in results if r is not None])
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus-style metrics endpoint.

    Returns service metrics for monitoring.
    """
    if not state.latency_history:
        p50 = p95 = p99 = 0
    else:
        latencies = sorted(state.latency_history)
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

    return {
        'service_uptime_seconds': state.get_uptime(),
        'total_predictions': state.total_predictions,
        'predictions_per_second': state.total_predictions / max(state.get_uptime(), 1),
        'latency_ms': {
            'mean': state.get_avg_latency(),
            'p50': p50,
            'p95': p95,
            'p99': p99
        },
        'model_loaded': state.model is not None,
        'model_version': state.model_metadata.get('model_version', 'unknown')
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv('DEBUG') else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn

    # For development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
