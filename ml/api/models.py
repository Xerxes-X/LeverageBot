"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for price prediction."""

    # Current market data
    price: float = Field(..., gt=0, description="Current price (close)")
    open: Optional[float] = Field(None, gt=0, description="Open price")
    high: Optional[float] = Field(None, gt=0, description="High price")
    low: Optional[float] = Field(None, gt=0, description="Low price")
    volume: float = Field(..., ge=0, description="Trading volume")

    # Order book data (optional but recommended)
    bid_volume: Optional[float] = Field(None, ge=0, description="Total bid volume")
    ask_volume: Optional[float] = Field(None, ge=0, description="Total ask volume")
    best_bid: Optional[float] = Field(None, gt=0, description="Best bid price")
    best_ask: Optional[float] = Field(None, gt=0, description="Best ask price")
    bid_depth_5: Optional[float] = Field(None, ge=0, description="Bid depth (5 levels)")
    ask_depth_5: Optional[float] = Field(None, ge=0, description="Ask depth (5 levels)")

    # Cross-asset data (optional)
    btc_price: Optional[float] = Field(None, gt=0, description="BTC price (for correlation)")
    funding_rate: Optional[float] = Field(None, description="Perpetual funding rate")

    # Metadata
    timestamp: Optional[datetime] = Field(None, description="Data timestamp")
    symbol: Optional[str] = Field("WBNBUSDT", description="Trading pair")

    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        """Set default timestamp if not provided."""
        return v or datetime.now()

    class Config:
        schema_extra = {
            "example": {
                "price": 625.50,
                "open": 624.20,
                "high": 626.80,
                "low": 623.90,
                "volume": 145230.5,
                "bid_volume": 72500.0,
                "ask_volume": 68200.0,
                "best_bid": 625.45,
                "best_ask": 625.55,
                "bid_depth_5": 150000.0,
                "ask_depth_5": 145000.0,
                "btc_price": 43250.00,
                "funding_rate": 0.0001,
                "symbol": "WBNBUSDT"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for price prediction."""

    # Prediction results
    direction: str = Field(..., description="Predicted direction: LONG or SHORT")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    raw_probability: float = Field(..., ge=0, le=1, description="Raw model probability (up)")

    # Recommended action
    should_trade: bool = Field(..., description="Whether confidence exceeds threshold")
    recommended_size: Optional[float] = Field(None, description="Recommended position size (USD)")

    # Metadata
    model_version: str = Field(..., description="Model version")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: datetime = Field(..., description="Prediction timestamp")

    # Feature importance (top 5)
    top_features: Optional[List[dict]] = Field(None, description="Top contributing features")

    class Config:
        schema_extra = {
            "example": {
                "direction": "LONG",
                "confidence": 0.72,
                "raw_probability": 0.86,
                "should_trade": True,
                "recommended_size": 25000.0,
                "model_version": "xgboost_phase1_v1",
                "latency_ms": 6.5,
                "timestamp": "2026-02-05T10:30:45.123456",
                "top_features": [
                    {"name": "order_book_imbalance", "value": 0.45},
                    {"name": "rsi_14", "value": 42.3},
                    {"name": "ema_12", "value": 624.5}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Loaded model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    total_predictions: int = Field(..., description="Total predictions served")
    avg_latency_ms: float = Field(..., description="Average latency (ms)")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "xgboost_phase1_v1",
                "uptime_seconds": 3600.5,
                "total_predictions": 1250,
                "avg_latency_ms": 6.8
            }
        }


class ModelInfo(BaseModel):
    """Model information response."""

    name: str
    version: str
    trained_date: str
    training_samples: int
    n_features: int
    feature_names: List[str]
    metrics: dict

    class Config:
        schema_extra = {
            "example": {
                "name": "xgboost_phase1_baseline",
                "version": "1.0",
                "trained_date": "2026-02-05T10:00:00",
                "training_samples": 86400,
                "n_features": 41,
                "feature_names": ["ema_12", "rsi_14", "macd", "..."],
                "metrics": {
                    "win_rate": 0.605,
                    "sharpe_ratio": 1.85,
                    "accuracy": 0.605,
                    "auc": 0.682
                }
            }
        }
