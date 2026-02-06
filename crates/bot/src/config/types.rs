use rust_decimal::Decimal;
use serde::Deserialize;
use std::collections::HashMap;

use crate::types::timeframe::{Timeframe, TradingStyle};

// ---------------------------------------------------------------------------
// Top-level aggregate
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct BotConfig {
    pub app: AppConfig,
    pub chain: ChainConfig,
    pub aave: AaveConfig,
    pub aggregator: AggregatorConfig,
    pub signals: SignalConfig,
    pub positions: PositionConfig,
    pub timing: TimingConfig,
    pub rate_limits: RateLimitConfig,
    pub mempool: Option<MempoolConfig>,
}

// ---------------------------------------------------------------------------
// app.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub precision: PrecisionConfig,
    pub timezone: TimezoneConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PrecisionConfig {
    pub decimal_precision: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TimezoneConfig {
    pub app_timezone: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoggingConfig {
    pub log_dir: String,
    pub module_folders: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// chains/56.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ChainConfig {
    pub chain_id: u64,
    pub chain_name: String,
    pub block_time_seconds: f64,
    pub native_token: String,
    pub wrapped_native: String,
    pub rpc: RpcConfig,
    pub contracts: ContractsConfig,
    pub chainlink_feeds: HashMap<String, ChainlinkFeedConfig>,
    pub tokens: HashMap<String, TokenConfig>,
    pub multicall: MulticallConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RpcConfig {
    pub http_url: String,
    pub http_url_fallback: String,
    pub mev_protected_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContractsConfig {
    pub multicall3: String,
    pub aave_v3_pool: String,
    pub aave_v3_pool_addresses_provider: String,
    pub aave_v3_oracle: String,
    pub aave_v3_data_provider: String,
    pub leverage_executor: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChainlinkFeedConfig {
    pub address: String,
    pub heartbeat_seconds: u64,
    pub deviation_threshold_percent: f64,
    pub decimals: u8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenConfig {
    pub address: String,
    pub decimals: u8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MulticallConfig {
    pub batch_size: u32,
    pub batch_delay_seconds: f64,
}

// ---------------------------------------------------------------------------
// aave.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct AaveConfig {
    pub flash_loan_premium_bps: u32,
    #[serde(with = "rust_decimal::serde::str")]
    pub flash_loan_premium: Decimal,
    pub referral_code: u16,
    pub supported_assets: HashMap<String, AaveAssetConfig>,
    pub e_mode: EModeConfig,
    pub interest_rate_model: InterestRateModelConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AaveAssetConfig {
    pub ltv_bps: u32,
    pub liquidation_threshold_bps: u32,
    pub liquidation_bonus_bps: u32,
    pub can_be_collateral: bool,
    pub can_be_borrowed: bool,
    pub isolation_mode: bool,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EModeConfig {
    pub enabled: bool,
    pub category_id: u8,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InterestRateModelConfig {
    pub note: Option<String>,
    pub monitor_utilization: bool,
    #[serde(with = "rust_decimal::serde::str")]
    pub max_acceptable_borrow_apr: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub rate_spike_pause_threshold_apr: Decimal,
}

// ---------------------------------------------------------------------------
// aggregator.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct AggregatorConfig {
    pub providers: Vec<AggregatorProviderConfig>,
    pub max_slippage_bps: u32,
    #[serde(with = "rust_decimal::serde::str")]
    pub max_price_impact_percent: Decimal,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AggregatorProviderConfig {
    pub name: String,
    pub enabled: bool,
    pub priority: u32,
    pub base_url: String,
    pub api_key_env: String,
    pub rate_limit_rps: u32,
    pub timeout_seconds: u64,
    pub approved_routers: Vec<String>,
    pub params: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// signals.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct SignalConfig {
    pub enabled: bool,
    pub mode: String,
    pub data_source: DataSourceConfig,
    pub indicators: IndicatorParams,
    pub signal_sources: SignalSourcesConfig,
    pub entry_rules: EntryRulesConfig,
    pub position_sizing: PositionSizingConfig,
    pub alpha_decay_monitoring: AlphaDecayConfig,
    pub exit_rules: ExitRulesConfig,
    pub short_signals: ShortSignalsConfig,
    /// Multi-timeframe configuration (optional, disabled by default).
    #[serde(default)]
    pub multi_timeframe: Option<MultiTimeframeConfig>,
    /// WebSocket streaming configuration (optional).
    #[serde(default)]
    pub websocket: Option<WebSocketConfig>,
    /// Extended indicators (MFI, OI, L/S ratio).
    #[serde(default)]
    pub extended_indicators: Option<ExtendedIndicatorConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataSourceConfig {
    pub primary: String,
    pub symbol: String,
    pub interval: String,
    pub history_candles: u32,
    pub refresh_interval_seconds: u64,
    pub fallback: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IndicatorParams {
    pub ema_fast: u32,
    pub ema_slow: u32,
    pub ema_trend: u32,
    pub rsi_period: u32,
    pub macd_fast: u32,
    pub macd_slow: u32,
    pub macd_signal: u32,
    pub bb_period: u32,
    pub bb_std: f64,
    pub atr_period: u32,
    pub hurst_max_lag: u32,
    pub hurst_min_data_points: u32,
    pub vpin_bucket_divisor: u32,
    pub vpin_window: u32,
    #[serde(with = "rust_decimal::serde::str")]
    pub garch_omega: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub garch_alpha: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub garch_beta: Decimal,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SignalSourcesConfig {
    pub tier_1: HashMap<String, serde_json::Value>,
    pub tier_2: HashMap<String, serde_json::Value>,
    pub tier_3: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EntryRulesConfig {
    #[serde(with = "rust_decimal::serde::str")]
    pub min_confidence: Decimal,
    pub require_trend_alignment: bool,
    pub require_volume_confirmation: bool,
    pub max_signals_per_day: u32,
    pub regime_filter: RegimeFilterConfig,
    pub regime_weight_multipliers: HashMap<String, HashMap<String, String>>,
    #[serde(with = "rust_decimal::serde::str")]
    pub agreement_bonus_threshold: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub agreement_bonus_multiplier: Decimal,
    pub max_signal_age_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegimeFilterConfig {
    pub enabled: bool,
    #[serde(with = "rust_decimal::serde::str")]
    pub trending_hurst_threshold: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub mean_reverting_hurst_threshold: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub min_atr_ratio: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub max_atr_ratio: Decimal,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PositionSizingConfig {
    pub method: String,
    #[serde(with = "rust_decimal::serde::str")]
    pub kelly_fraction: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub high_vol_threshold: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub drawdown_reduction_start: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub min_position_usd: Decimal,
    pub rolling_edge_window_days: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AlphaDecayConfig {
    pub enabled: bool,
    #[serde(with = "rust_decimal::serde::str")]
    pub accuracy_decay_threshold: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub sharpe_decay_threshold: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub confidence_boost_on_decay: Decimal,
    pub rolling_window_days: u32,
    pub historical_window_days: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExitRulesConfig {
    #[serde(with = "rust_decimal::serde::str")]
    pub take_profit_percent: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub stop_loss_percent: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub trailing_stop_percent: Decimal,
    pub max_hold_hours: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ShortSignalsConfig {
    pub enabled: bool,
    pub preferred_collateral: String,
}

// ---------------------------------------------------------------------------
// Multi-Timeframe Configuration
// ---------------------------------------------------------------------------

/// Configuration for multi-timeframe signal analysis.
#[derive(Debug, Clone, Deserialize)]
pub struct MultiTimeframeConfig {
    /// Enable multi-timeframe analysis.
    pub enabled: bool,
    /// Trading style determines primary evaluation timeframes.
    #[serde(default)]
    pub trading_style: TradingStyle,
    /// Per-timeframe configuration.
    pub timeframes: Vec<TimeframeConfigEntry>,
    /// Signal aggregation settings.
    pub aggregation: TimeframeAggregationConfig,
}

impl Default for MultiTimeframeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            trading_style: TradingStyle::default(),
            timeframes: vec![],
            aggregation: TimeframeAggregationConfig::default(),
        }
    }
}

/// Configuration for a specific timeframe.
#[derive(Debug, Clone, Deserialize)]
pub struct TimeframeConfigEntry {
    /// The timeframe (parsed from string like "1m", "4h").
    pub timeframe: Timeframe,
    /// Whether this timeframe is enabled.
    pub enabled: bool,
    /// Weight in the aggregation (0.0 to 1.0).
    #[serde(with = "rust_decimal::serde::str")]
    pub weight: Decimal,
    /// Number of historical candles to fetch.
    pub history_candles: u32,
    /// Optional indicator parameter overrides.
    #[serde(default)]
    pub indicator_overrides: Option<IndicatorParams>,
}

/// Configuration for timeframe signal aggregation.
#[derive(Debug, Clone, Deserialize)]
pub struct TimeframeAggregationConfig {
    /// Weight mode: "fixed", "adaptive", "regime_adjusted".
    #[serde(default = "default_weight_mode")]
    pub weight_mode: String,
    /// Minimum fraction of timeframes that must agree on direction.
    #[serde(with = "rust_decimal::serde::str", default = "default_min_agreement")]
    pub min_timeframe_agreement: Decimal,
    /// Whether higher timeframes must align with the signal direction.
    #[serde(default = "default_true")]
    pub require_higher_tf_alignment: bool,
    /// Timeframes used for direction confirmation (e.g., ["h4", "h6"]).
    #[serde(default)]
    pub direction_timeframes: Vec<Timeframe>,
}

fn default_weight_mode() -> String {
    "fixed".to_string()
}

fn default_min_agreement() -> Decimal {
    Decimal::new(5, 1) // 0.5
}

fn default_true() -> bool {
    true
}

impl Default for TimeframeAggregationConfig {
    fn default() -> Self {
        Self {
            weight_mode: default_weight_mode(),
            min_timeframe_agreement: default_min_agreement(),
            require_higher_tf_alignment: true,
            direction_timeframes: vec![Timeframe::H4, Timeframe::H6],
        }
    }
}

/// Per-timeframe indicator parameters (scaled from base).
#[derive(Debug, Clone, Deserialize)]
pub struct TimeframeIndicatorParams {
    pub timeframe: Timeframe,
    pub ema_fast: u32,
    pub ema_slow: u32,
    pub ema_trend: u32,
    pub rsi_period: u32,
    pub macd_fast: u32,
    pub macd_slow: u32,
    pub macd_signal: u32,
    pub bb_period: u32,
    pub bb_std: f64,
    pub atr_period: u32,
}

impl TimeframeIndicatorParams {
    /// Create scaled parameters for a timeframe from base 1H parameters.
    ///
    /// Shorter timeframes use more periods to cover similar time windows.
    #[must_use]
    pub fn from_base(base: &IndicatorParams, tf: Timeframe) -> Self {
        let scale = tf.ema_scale_factor();

        // For very short timeframes, don't scale below minimum useful periods
        let min_period = |scaled: f64, min: u32| -> u32 {
            (scaled.round() as u32).max(min)
        };

        Self {
            timeframe: tf,
            ema_fast: min_period(base.ema_fast as f64 * scale, 3),
            ema_slow: min_period(base.ema_slow as f64 * scale, 5),
            ema_trend: min_period(base.ema_trend as f64 * scale, 20),
            rsi_period: base.rsi_period, // RSI period typically stays fixed
            macd_fast: min_period(base.macd_fast as f64 * scale, 5),
            macd_slow: min_period(base.macd_slow as f64 * scale, 10),
            macd_signal: base.macd_signal, // Signal period stays fixed
            bb_period: min_period(base.bb_period as f64 * scale, 10),
            bb_std: base.bb_std,
            atr_period: base.atr_period, // ATR period typically stays fixed
        }
    }

    /// Convert scaled timeframe params back to full IndicatorParams.
    ///
    /// Non-scaled parameters (hurst, vpin, garch) use sensible defaults.
    #[must_use]
    pub fn to_indicator_params(&self) -> IndicatorParams {
        IndicatorParams {
            ema_fast: self.ema_fast,
            ema_slow: self.ema_slow,
            ema_trend: self.ema_trend,
            rsi_period: self.rsi_period,
            macd_fast: self.macd_fast,
            macd_slow: self.macd_slow,
            macd_signal: self.macd_signal,
            bb_period: self.bb_period,
            bb_std: self.bb_std,
            atr_period: self.atr_period,
            // These parameters don't scale with timeframe
            hurst_max_lag: 20,
            hurst_min_data_points: 100,
            vpin_bucket_divisor: 50,
            vpin_window: 50,
            garch_omega: rust_decimal_macros::dec!(0.00001),
            garch_alpha: rust_decimal_macros::dec!(0.1),
            garch_beta: rust_decimal_macros::dec!(0.85),
        }
    }
}

// ---------------------------------------------------------------------------
// WebSocket Configuration
// ---------------------------------------------------------------------------

/// WebSocket streaming configuration for real-time market data.
#[derive(Debug, Clone, Deserialize)]
pub struct WebSocketConfig {
    /// Enable WebSocket streaming.
    pub enabled: bool,
    /// Binance WebSocket base URL.
    #[serde(default = "default_binance_ws_url")]
    pub binance_ws_url: String,
    /// Reconnection delay in milliseconds.
    #[serde(default = "default_reconnect_delay")]
    pub reconnect_delay_ms: u64,
    /// Maximum reconnection attempts before giving up.
    #[serde(default = "default_max_reconnect")]
    pub max_reconnect_attempts: u32,
    /// Ping interval to keep connection alive (seconds).
    #[serde(default = "default_ping_interval")]
    pub ping_interval_seconds: u64,
    /// Stream subscriptions.
    pub subscriptions: WebSocketSubscriptions,
}

fn default_binance_ws_url() -> String {
    "wss://stream.binance.com:9443".to_string()
}

fn default_reconnect_delay() -> u64 {
    5000
}

fn default_max_reconnect() -> u32 {
    10
}

fn default_ping_interval() -> u64 {
    30
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            binance_ws_url: default_binance_ws_url(),
            reconnect_delay_ms: default_reconnect_delay(),
            max_reconnect_attempts: default_max_reconnect(),
            ping_interval_seconds: default_ping_interval(),
            subscriptions: WebSocketSubscriptions::default(),
        }
    }
}

/// WebSocket stream subscriptions.
#[derive(Debug, Clone, Deserialize)]
pub struct WebSocketSubscriptions {
    /// Timeframes to stream klines for.
    #[serde(default)]
    pub klines: Vec<Timeframe>,
    /// Subscribe to order book depth.
    #[serde(default)]
    pub depth: bool,
    /// Subscribe to aggregate trades.
    #[serde(default)]
    pub trades: bool,
}

impl Default for WebSocketSubscriptions {
    fn default() -> Self {
        Self {
            klines: vec![Timeframe::M1, Timeframe::M5, Timeframe::M15, Timeframe::M30],
            depth: true,
            trades: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Extended Indicators Configuration
// ---------------------------------------------------------------------------

/// Configuration for additional indicators (MFI, Open Interest, L/S Ratio).
#[derive(Debug, Clone, Deserialize)]
pub struct ExtendedIndicatorConfig {
    /// Money Flow Index configuration.
    #[serde(default)]
    pub mfi: MfiConfig,
    /// Open Interest configuration.
    #[serde(default)]
    pub open_interest: OpenInterestConfig,
    /// Long/Short Ratio configuration.
    #[serde(default)]
    pub long_short_ratio: LongShortRatioConfig,
}

impl Default for ExtendedIndicatorConfig {
    fn default() -> Self {
        Self {
            mfi: MfiConfig::default(),
            open_interest: OpenInterestConfig::default(),
            long_short_ratio: LongShortRatioConfig::default(),
        }
    }
}

/// Money Flow Index configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct MfiConfig {
    pub enabled: bool,
    pub period: u32,
    pub overbought_threshold: f64,
    pub oversold_threshold: f64,
    #[serde(with = "rust_decimal::serde::str")]
    pub weight: Decimal,
}

impl Default for MfiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            period: 14,
            overbought_threshold: 80.0,
            oversold_threshold: 20.0,
            weight: Decimal::new(8, 2), // 0.08
        }
    }
}

/// Open Interest configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenInterestConfig {
    pub enabled: bool,
    /// Polling interval in seconds.
    pub poll_interval_seconds: u64,
    #[serde(with = "rust_decimal::serde::str")]
    pub weight: Decimal,
}

impl Default for OpenInterestConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_seconds: 60,
            weight: Decimal::new(6, 2), // 0.06
        }
    }
}

/// Long/Short Ratio configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct LongShortRatioConfig {
    pub enabled: bool,
    /// Polling interval in seconds.
    pub poll_interval_seconds: u64,
    /// Extreme ratio threshold for contrarian signal.
    #[serde(with = "rust_decimal::serde::str")]
    pub extreme_long_threshold: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub extreme_short_threshold: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub weight: Decimal,
}

impl Default for LongShortRatioConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_seconds: 60,
            extreme_long_threshold: Decimal::new(2, 0),   // 2.0
            extreme_short_threshold: Decimal::new(5, 1),  // 0.5
            weight: Decimal::new(7, 2), // 0.07
        }
    }
}

// ---------------------------------------------------------------------------
// positions.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct PositionConfig {
    pub dry_run: bool,
    pub max_flash_loan_usd: u64,
    pub max_position_usd: u64,
    #[serde(with = "rust_decimal::serde::str")]
    pub max_leverage_ratio: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub min_health_factor: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub deleverage_threshold: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub close_threshold: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub target_hf_after_deleverage: Decimal,
    pub max_gas_price_gwei: u64,
    pub max_slippage_bps: u32,
    pub cooldown_between_actions_seconds: u64,
    pub max_transactions_per_24h: u32,
    pub stress_test_price_drops: Vec<String>,
    #[serde(with = "rust_decimal::serde::str")]
    pub min_stress_test_hf: Decimal,
    pub cascade_liquidation_threshold_usd: u64,
    #[serde(with = "rust_decimal::serde::str")]
    pub cascade_additional_drop: Decimal,
    pub close_factor_warning_threshold_usd: u64,
    #[serde(with = "rust_decimal::serde::str")]
    pub max_borrow_cost_pct: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub max_acceptable_borrow_apr: Decimal,
    pub preferred_short_collateral: String,
    #[serde(with = "rust_decimal::serde::str")]
    pub max_dex_oracle_divergence_pct: Decimal,
    pub oracle_max_staleness_seconds: u64,
}

// ---------------------------------------------------------------------------
// timing.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct TimingConfig {
    pub health_monitoring: HealthMonitoringTiming,
    pub aggregator: AggregatorTiming,
    pub transaction: TransactionTiming,
    pub web3_connection: Web3ConnectionConfig,
    pub error_recovery: ErrorRecoveryConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HealthMonitoringTiming {
    pub safe_interval_seconds: u64,
    pub watch_interval_seconds: u64,
    pub warning_interval_seconds: u64,
    pub critical_interval_seconds: u64,
    pub stale_data_threshold_failures: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AggregatorTiming {
    pub quote_timeout_seconds: u64,
    pub quote_cache_ttl_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TransactionTiming {
    pub confirmation_timeout_seconds: u64,
    pub simulation_timeout_seconds: u64,
    pub nonce_refresh_interval_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Web3ConnectionConfig {
    pub max_retries: u32,
    pub retry_delay_seconds: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorRecoveryConfig {
    pub rpc_retry_base_delay_seconds: u64,
    pub rpc_retry_max_delay_seconds: u64,
}

// ---------------------------------------------------------------------------
// rate_limits.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct RateLimitConfig {
    pub rpc: RpcRateLimitConfig,
    pub aggregator: HashMap<String, AggregatorRateLimit>,
    pub binance: BinanceRateLimitConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RpcRateLimitConfig {
    pub max_requests_per_second: u32,
    pub burst_limit: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AggregatorRateLimit {
    pub max_requests_per_second: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BinanceRateLimitConfig {
    pub spot_max_requests_per_minute: u32,
    pub futures_max_requests_per_minute: u32,
}

// ---------------------------------------------------------------------------
// mempool.json (optional)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct MempoolConfig {
    pub enabled: bool,
    pub redis: MempoolRedisConfig,
    pub monitored_tokens: MempoolTokensConfig,
    pub aggregation: MempoolAggregationConfig,
    pub decoder: MempoolDecoderConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MempoolRedisConfig {
    pub url: String,
    pub decoded_swaps_channel: String,
    pub aggregate_signal_channel: String,
    pub aggregate_publish_interval_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MempoolTokensConfig {
    pub volatile: Vec<String>,
    pub stable: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MempoolAggregationConfig {
    pub windows_seconds: Vec<u64>,
    pub whale_threshold_usd: u64,
    pub poison_filter_enabled: bool,
    pub poison_suspicion_threshold: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MempoolDecoderConfig {
    pub websocket_url_env: String,
    pub websocket_fallback: String,
    pub dedup_cache_size: u64,
    pub reconnect_delay_seconds: u64,
    pub max_reconnect_attempts: u32,
}
