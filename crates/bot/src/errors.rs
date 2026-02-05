use thiserror::Error;

/// Typed error hierarchy for the leverage bot.
///
/// Library-internal errors use specific variants; application code wraps with
/// `anyhow::Context` for propagation.
#[derive(Error, Debug)]
pub enum BotError {
    // -- Execution ----------------------------------------------------------
    #[error("transaction simulation failed: {reason}")]
    SimulationFailed { reason: String },

    #[error("transaction reverted: {reason} (tx: {tx_hash})")]
    TxReverted { tx_hash: String, reason: String },

    #[error("transaction timed out after {timeout_seconds}s (tx: {tx_hash})")]
    TxTimeout { tx_hash: String, timeout_seconds: u64 },

    // -- Aggregator ---------------------------------------------------------
    #[error("all aggregator providers failed")]
    AggregatorUnavailable,

    #[error("DEX-Oracle price divergence: {divergence_pct:.2}% (max {max_pct:.2}%)")]
    PriceDivergence { divergence_pct: f64, max_pct: f64 },

    // -- Position -----------------------------------------------------------
    #[error("position error: {reason}")]
    PositionError { reason: String },

    // -- Safety -------------------------------------------------------------
    #[error("safety gate blocked: {reason}")]
    SafetyBlocked { reason: String },

    #[error("oracle stale: {age_seconds}s old (max {max_seconds}s)")]
    OracleStale {
        age_seconds: u64,
        max_seconds: u64,
    },

    // -- Data ---------------------------------------------------------------
    #[error("data source unavailable: {name}")]
    DataUnavailable { name: String },

    // -- Aave ---------------------------------------------------------------
    #[error("Aave error: {reason}")]
    AaveError { reason: String },

    // -- Database -----------------------------------------------------------
    #[error("database error: {reason}")]
    DatabaseError { reason: String },

    // -- Configuration ------------------------------------------------------
    #[error("configuration error: {0}")]
    Config(String),

    // -- Forwarded errors ---------------------------------------------------
    #[error(transparent)]
    Alloy(#[from] alloy::transports::TransportError),

    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),

    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),

    #[error(transparent)]
    Redis(#[from] redis::RedisError),

    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
