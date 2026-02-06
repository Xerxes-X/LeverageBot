/*
 * ML Client - HTTP client for ML prediction service integration
 *
 * Connects Rust trading bot to Python FastAPI ML service.
 * Target latency: <10ms total (includes network + inference)
 *
 * Based on Phase 1 ML implementation plan.
 */

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use crate::errors::BotError;

/// ML prediction request
#[derive(Debug, Serialize)]
pub struct PredictionRequest {
    // Current market data
    pub price: f64,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub volume: f64,

    // Order book data (optional but recommended)
    pub bid_volume: Option<f64>,
    pub ask_volume: Option<f64>,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub bid_depth_5: Option<f64>,
    pub ask_depth_5: Option<f64>,

    // Cross-asset data
    pub btc_price: Option<f64>,
    pub funding_rate: Option<f64>,

    // Metadata
    pub symbol: Option<String>,
}

/// ML prediction response
#[derive(Debug, Deserialize)]
pub struct PredictionResponse {
    // Prediction results
    pub direction: String,           // "LONG" or "SHORT"
    pub confidence: f64,              // 0.0 to 1.0
    pub raw_probability: f64,         // Model output probability

    // Recommended action
    pub should_trade: bool,
    pub recommended_size: Option<f64>, // USD

    // Metadata
    pub model_version: String,
    pub latency_ms: f64,
    pub timestamp: String,

    // Feature importance (top 5)
    #[serde(default)]
    pub top_features: Vec<TopFeature>,
}

#[derive(Debug, Deserialize)]
pub struct TopFeature {
    pub name: String,
    pub value: f64,
}

/// Health check response
#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
    pub model_version: String,
    pub uptime_seconds: f64,
    pub total_predictions: u64,
    pub avg_latency_ms: f64,
}

/// ML service client
pub struct MLClient {
    client: Client,
    base_url: String,
    timeout: Duration,
    enabled: bool,
}

impl MLClient {
    /// Create new ML client
    ///
    /// # Arguments
    /// * `base_url` - ML service URL (e.g., "http://localhost:8000")
    /// * `timeout_ms` - Request timeout in milliseconds (default: 100ms)
    /// * `enabled` - Whether ML service is enabled (fallback if false)
    pub fn new(base_url: String, timeout_ms: u64, enabled: bool) -> Result<Self, BotError> {
        let timeout = Duration::from_millis(timeout_ms);

        let client = Client::builder()
            .timeout(timeout)
            .pool_idle_timeout(Some(Duration::from_secs(10)))
            .pool_max_idle_per_host(10)
            .build()
            .map_err(|e| BotError::ConfigError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            base_url,
            timeout,
            enabled,
        })
    }

    /// Check if ML service is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get prediction from ML service
    ///
    /// # Arguments
    /// * `request` - Market data for prediction
    ///
    /// # Returns
    /// * `Ok(PredictionResponse)` - Prediction result
    /// * `Err(BotError)` - If request fails or times out
    pub async fn predict(&self, request: PredictionRequest) -> Result<PredictionResponse, BotError> {
        if !self.enabled {
            return Err(BotError::ConfigError("ML service disabled".to_string()));
        }

        let start = Instant::now();
        let url = format!("{}/predict", self.base_url);

        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    BotError::NetworkError(format!("ML service timeout after {}ms", self.timeout.as_millis()))
                } else if e.is_connect() {
                    BotError::NetworkError(format!("ML service connection failed: {}", e))
                } else {
                    BotError::NetworkError(format!("ML service request failed: {}", e))
                }
            })?;

        // Check status
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|_| "unknown error".to_string());
            return Err(BotError::NetworkError(format!(
                "ML service returned {}: {}",
                status, body
            )));
        }

        // Parse response
        let prediction = response
            .json::<PredictionResponse>()
            .await
            .map_err(|e| BotError::NetworkError(format!("Failed to parse ML response: {}", e)))?;

        let elapsed = start.elapsed().as_millis();

        // Log slow predictions
        if elapsed > 10 {
            log::warn!("Slow ML prediction: {}ms (service reports {}ms)", elapsed, prediction.latency_ms);
        }

        log::debug!(
            "ML prediction: {} (confidence: {:.2}, total latency: {}ms)",
            prediction.direction,
            prediction.confidence,
            elapsed
        );

        Ok(prediction)
    }

    /// Health check - verify ML service is available
    pub async fn health_check(&self) -> Result<HealthResponse, BotError> {
        if !self.enabled {
            return Err(BotError::ConfigError("ML service disabled".to_string()));
        }

        let url = format!("{}/health", self.base_url);

        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| BotError::NetworkError(format!("Health check failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(BotError::NetworkError(format!(
                "Health check returned status: {}",
                response.status()
            )));
        }

        let health = response
            .json::<HealthResponse>()
            .await
            .map_err(|e| BotError::NetworkError(format!("Failed to parse health response: {}", e)))?;

        Ok(health)
    }

    /// Get prediction with fallback
    ///
    /// If ML service fails, returns None instead of error.
    /// Allows bot to continue with existing signal sources.
    pub async fn predict_with_fallback(&self, request: PredictionRequest) -> Option<PredictionResponse> {
        match self.predict(request).await {
            Ok(pred) => Some(pred),
            Err(e) => {
                log::warn!("ML prediction failed (fallback mode): {}", e);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ml_client_disabled() {
        let client = MLClient::new(
            "http://localhost:8000".to_string(),
            100,
            false
        ).unwrap();

        assert!(!client.is_enabled());

        let request = PredictionRequest {
            price: 625.0,
            open: None,
            high: None,
            low: None,
            volume: 10000.0,
            bid_volume: None,
            ask_volume: None,
            best_bid: None,
            best_ask: None,
            bid_depth_5: None,
            ask_depth_5: None,
            btc_price: None,
            funding_rate: None,
            symbol: Some("WBNBUSDT".to_string()),
        };

        let result = client.predict(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_prediction_request_serialization() {
        let request = PredictionRequest {
            price: 625.50,
            open: Some(624.20),
            high: Some(626.80),
            low: Some(623.90),
            volume: 145230.5,
            bid_volume: Some(72500.0),
            ask_volume: Some(68200.0),
            best_bid: Some(625.45),
            best_ask: Some(625.55),
            bid_depth_5: Some(150000.0),
            ask_depth_5: Some(145000.0),
            btc_price: Some(43250.00),
            funding_rate: Some(0.0001),
            symbol: Some("WBNBUSDT".to_string()),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("625.5"));
        assert!(json.contains("WBNBUSDT"));
    }
}
