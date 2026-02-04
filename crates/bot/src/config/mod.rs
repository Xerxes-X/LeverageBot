pub mod types;
pub mod validate;

pub use types::*;

use anyhow::{Context, Result};
use rust_decimal::Decimal;
use std::path::Path;
use std::str::FromStr;
use tracing::info;

/// Load and merge all config JSON files into a single [`BotConfig`],
/// then apply environment variable overrides and validate.
///
/// Expected directory layout:
/// ```text
/// config/
///   app.json
///   chains/56.json
///   aave.json
///   aggregator.json
///   signals.json
///   positions.json
///   timing.json
///   rate_limits.json
///   mempool.json   (optional)
/// ```
///
/// # Environment variable overrides
///
/// The following env vars override the corresponding JSON config values:
///
/// | Env Var                       | Config Field                           |
/// |-------------------------------|----------------------------------------|
/// | `EXECUTOR_DRY_RUN`            | `positions.dry_run`                    |
/// | `MAX_FLASH_LOAN_USD`          | `positions.max_flash_loan_usd`         |
/// | `MAX_LEVERAGE_RATIO`          | `positions.max_leverage_ratio`         |
/// | `MIN_HEALTH_FACTOR`           | `positions.min_health_factor`          |
/// | `MAX_GAS_PRICE_GWEI`          | `positions.max_gas_price_gwei`         |
/// | `BSC_RPC_URL_HTTP`            | `chain.rpc.http_url`                   |
/// | `BSC_RPC_URL_HTTP_FALLBACK`   | `chain.rpc.http_url_fallback`          |
/// | `BSC_RPC_URL_MEV_PROTECTED`   | `chain.rpc.mev_protected_url`          |
/// | `LEVERAGE_EXECUTOR_ADDRESS`   | `chain.contracts.leverage_executor`    |
pub fn load_config(config_dir: &Path) -> Result<BotConfig> {
    let read = |name: &str| -> Result<String> {
        let path = config_dir.join(name);
        std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read config file: {}", path.display()))
    };

    let app: AppConfig =
        serde_json::from_str(&read("app.json")?).context("parsing app.json")?;

    let chain: ChainConfig =
        serde_json::from_str(&read("chains/56.json")?).context("parsing chains/56.json")?;

    let aave: AaveConfig =
        serde_json::from_str(&read("aave.json")?).context("parsing aave.json")?;

    let aggregator: AggregatorConfig =
        serde_json::from_str(&read("aggregator.json")?).context("parsing aggregator.json")?;

    let signals: SignalConfig =
        serde_json::from_str(&read("signals.json")?).context("parsing signals.json")?;

    let positions: PositionConfig =
        serde_json::from_str(&read("positions.json")?).context("parsing positions.json")?;

    let timing: TimingConfig =
        serde_json::from_str(&read("timing.json")?).context("parsing timing.json")?;

    let rate_limits: RateLimitConfig =
        serde_json::from_str(&read("rate_limits.json")?).context("parsing rate_limits.json")?;

    // Mempool config is optional.
    let mempool: Option<MempoolConfig> = match read("mempool.json") {
        Ok(contents) => {
            Some(serde_json::from_str(&contents).context("parsing mempool.json")?)
        }
        Err(_) => None,
    };

    let mut config = BotConfig {
        app,
        chain,
        aave,
        aggregator,
        signals,
        positions,
        timing,
        rate_limits,
        mempool,
    };

    apply_env_overrides(&mut config);
    validate::validate_config(&config)?;

    Ok(config)
}

// ---------------------------------------------------------------------------
// Environment variable overrides
// ---------------------------------------------------------------------------

/// Apply environment variable overrides to the loaded config.
///
/// Only non-empty env vars take effect. Parse failures are logged and skipped
/// (the JSON default remains).
fn apply_env_overrides(config: &mut BotConfig) {
    // -- Positions -----------------------------------------------------------
    if let Some(val) = env_bool("EXECUTOR_DRY_RUN") {
        info!(dry_run = val, "env override: EXECUTOR_DRY_RUN");
        config.positions.dry_run = val;
    }

    if let Some(val) = env_parse::<u64>("MAX_FLASH_LOAN_USD") {
        info!(val, "env override: MAX_FLASH_LOAN_USD");
        config.positions.max_flash_loan_usd = val;
    }

    if let Some(val) = env_decimal("MAX_LEVERAGE_RATIO") {
        info!(%val, "env override: MAX_LEVERAGE_RATIO");
        config.positions.max_leverage_ratio = val;
    }

    if let Some(val) = env_decimal("MIN_HEALTH_FACTOR") {
        info!(%val, "env override: MIN_HEALTH_FACTOR");
        config.positions.min_health_factor = val;
    }

    if let Some(val) = env_parse::<u64>("MAX_GAS_PRICE_GWEI") {
        info!(val, "env override: MAX_GAS_PRICE_GWEI");
        config.positions.max_gas_price_gwei = val;
    }

    // -- RPC URLs ------------------------------------------------------------
    if let Some(val) = env_string("BSC_RPC_URL_HTTP") {
        info!("env override: BSC_RPC_URL_HTTP");
        config.chain.rpc.http_url = val;
    }

    if let Some(val) = env_string("BSC_RPC_URL_HTTP_FALLBACK") {
        info!("env override: BSC_RPC_URL_HTTP_FALLBACK");
        config.chain.rpc.http_url_fallback = val;
    }

    if let Some(val) = env_string("BSC_RPC_URL_MEV_PROTECTED") {
        info!("env override: BSC_RPC_URL_MEV_PROTECTED");
        config.chain.rpc.mev_protected_url = val;
    }

    // -- Contracts -----------------------------------------------------------
    if let Some(val) = env_string("LEVERAGE_EXECUTOR_ADDRESS") {
        info!("env override: LEVERAGE_EXECUTOR_ADDRESS");
        config.chain.contracts.leverage_executor = val;
    }
}

/// Read a non-empty env var as a `String`.
fn env_string(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

/// Read a non-empty env var as a bool (`true`, `1`, `yes` → true).
fn env_bool(key: &str) -> Option<bool> {
    env_string(key).map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes"))
}

/// Read a non-empty env var and parse it as `T`.
fn env_parse<T: FromStr>(key: &str) -> Option<T> {
    env_string(key).and_then(|v| v.parse().ok())
}

/// Read a non-empty env var and parse it as `Decimal`.
fn env_decimal(key: &str) -> Option<Decimal> {
    env_string(key).and_then(|v| Decimal::from_str(&v).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::path::PathBuf;

    fn project_config_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("config")
    }

    // -----------------------------------------------------------------------
    // Helper: write a minimal set of config JSON files to a temp dir.
    // -----------------------------------------------------------------------

    fn write_test_configs(dir: &Path) {
        std::fs::create_dir_all(dir.join("chains")).unwrap();

        std::fs::write(
            dir.join("app.json"),
            r#"{
                "precision": { "decimal_precision": 78 },
                "timezone": { "app_timezone": "UTC" },
                "logging": { "log_dir": "logs", "module_folders": {} }
            }"#,
        )
        .unwrap();

        std::fs::write(
            dir.join("chains/56.json"),
            r#"{
                "chain_id": 56,
                "chain_name": "BSC Mainnet",
                "block_time_seconds": 0.75,
                "native_token": "BNB",
                "wrapped_native": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
                "rpc": {
                    "http_url": "https://bsc-dataseed1.binance.org/",
                    "http_url_fallback": "https://bsc-dataseed2.binance.org/",
                    "mev_protected_url": "https://rpc.48.club"
                },
                "contracts": {
                    "multicall3": "0xcA11bde05977b3631167028862bE2a173976CA11",
                    "aave_v3_pool": "0x6807dc923806fE8Fd134338EABCA509979a7e0cB",
                    "aave_v3_pool_addresses_provider": "0xff75B6da14FfbbfD355Daf7a2731456b3562Ba6D",
                    "aave_v3_oracle": "0x39bc1bfDa2130d6Bb6DBEfd366939b4c7aa7C697",
                    "aave_v3_data_provider": "0xc90Df74A7c16245c5F5C5870327Ceb38Fe5d5328",
                    "leverage_executor": ""
                },
                "chainlink_feeds": {
                    "BNB_USD": { "address": "0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE", "heartbeat_seconds": 27, "deviation_threshold_percent": 0.1, "decimals": 8 }
                },
                "tokens": {
                    "WBNB": { "address": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c", "decimals": 18 },
                    "USDT": { "address": "0x55d398326f99059fF775485246999027B3197955", "decimals": 18 }
                },
                "multicall": { "batch_size": 25, "batch_delay_seconds": 0.1 }
            }"#,
        )
        .unwrap();

        std::fs::write(
            dir.join("aave.json"),
            r#"{
                "flash_loan_premium_bps": 5,
                "flash_loan_premium": "0.0005",
                "referral_code": 0,
                "supported_assets": {
                    "WBNB": { "ltv_bps": 7500, "liquidation_threshold_bps": 8000, "liquidation_bonus_bps": 500, "can_be_collateral": true, "can_be_borrowed": true, "isolation_mode": false }
                },
                "e_mode": { "enabled": false, "category_id": 0 },
                "interest_rate_model": { "monitor_utilization": true, "max_acceptable_borrow_apr": "15.0", "rate_spike_pause_threshold_apr": "50.0" }
            }"#,
        )
        .unwrap();

        std::fs::write(
            dir.join("aggregator.json"),
            r#"{
                "providers": [
                    {
                        "name": "1inch",
                        "enabled": true,
                        "priority": 1,
                        "base_url": "https://api.1inch.dev/swap/v6.0/56",
                        "api_key_env": "ONEINCH_API_KEY",
                        "rate_limit_rps": 1,
                        "timeout_seconds": 5,
                        "approved_routers": ["0x111111125421cA6dc452d289314280a0f8842A65"],
                        "params": {}
                    }
                ],
                "max_slippage_bps": 50,
                "max_price_impact_percent": "1.0"
            }"#,
        )
        .unwrap();

        std::fs::write(
            dir.join("signals.json"),
            r#"{
                "enabled": true,
                "mode": "blended",
                "data_source": { "primary": "binance", "symbol": "BNBUSDT", "interval": "1h", "history_candles": 200, "refresh_interval_seconds": 60, "fallback": "geckoterminal" },
                "indicators": { "ema_fast": 20, "ema_slow": 50, "ema_trend": 200, "rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "bb_period": 20, "bb_std": 2.0, "atr_period": 14, "hurst_max_lag": 20, "hurst_min_data_points": 100, "vpin_bucket_divisor": 50, "vpin_window": 50, "garch_omega": "0.00001", "garch_alpha": "0.1", "garch_beta": "0.85" },
                "signal_sources": { "tier_1": {}, "tier_2": {}, "tier_3": {} },
                "entry_rules": {
                    "min_confidence": "0.7",
                    "require_trend_alignment": true,
                    "require_volume_confirmation": false,
                    "max_signals_per_day": 3,
                    "regime_filter": { "enabled": true, "trending_hurst_threshold": "0.55", "mean_reverting_hurst_threshold": "0.45", "min_atr_ratio": "1.0", "max_atr_ratio": "3.0" },
                    "regime_weight_multipliers": {},
                    "agreement_bonus_threshold": "0.7",
                    "agreement_bonus_multiplier": "1.15",
                    "max_signal_age_seconds": 120
                },
                "position_sizing": { "method": "fractional_kelly", "kelly_fraction": "0.25", "high_vol_threshold": "0.04", "drawdown_reduction_start": "0.10", "min_position_usd": "100", "rolling_edge_window_days": 30 },
                "alpha_decay_monitoring": { "enabled": true, "accuracy_decay_threshold": "0.7", "sharpe_decay_threshold": "0.5", "confidence_boost_on_decay": "1.1", "rolling_window_days": 30, "historical_window_days": 180 },
                "exit_rules": { "take_profit_percent": "5.0", "stop_loss_percent": "3.0", "trailing_stop_percent": "2.0", "max_hold_hours": 168 },
                "short_signals": { "enabled": true, "preferred_collateral": "USDC" }
            }"#,
        )
        .unwrap();

        std::fs::write(
            dir.join("positions.json"),
            r#"{
                "dry_run": true,
                "max_flash_loan_usd": 5000,
                "max_position_usd": 10000,
                "max_leverage_ratio": "3.0",
                "min_health_factor": "1.5",
                "deleverage_threshold": "1.4",
                "close_threshold": "1.25",
                "target_hf_after_deleverage": "1.8",
                "max_gas_price_gwei": 10,
                "max_slippage_bps": 50,
                "cooldown_between_actions_seconds": 30,
                "max_transactions_per_24h": 50,
                "stress_test_price_drops": ["-0.05", "-0.10"],
                "min_stress_test_hf": "1.1",
                "cascade_liquidation_threshold_usd": 50000000,
                "cascade_additional_drop": "-0.03",
                "close_factor_warning_threshold_usd": 2000,
                "max_borrow_cost_pct": "0.5",
                "max_acceptable_borrow_apr": "15.0",
                "preferred_short_collateral": "USDC",
                "max_dex_oracle_divergence_pct": "1.0",
                "oracle_max_staleness_seconds": 60
            }"#,
        )
        .unwrap();

        std::fs::write(
            dir.join("timing.json"),
            r#"{
                "health_monitoring": { "safe_interval_seconds": 15, "watch_interval_seconds": 5, "warning_interval_seconds": 2, "critical_interval_seconds": 1, "stale_data_threshold_failures": 5 },
                "aggregator": { "quote_timeout_seconds": 5, "quote_cache_ttl_seconds": 10 },
                "transaction": { "confirmation_timeout_seconds": 60, "simulation_timeout_seconds": 15, "nonce_refresh_interval_seconds": 30 },
                "web3_connection": { "max_retries": 5, "retry_delay_seconds": 2.0 },
                "error_recovery": { "rpc_retry_base_delay_seconds": 1, "rpc_retry_max_delay_seconds": 30 }
            }"#,
        )
        .unwrap();

        std::fs::write(
            dir.join("rate_limits.json"),
            r#"{
                "rpc": { "max_requests_per_second": 50, "burst_limit": 100 },
                "aggregator": { "1inch": { "max_requests_per_second": 1 } },
                "binance": { "spot_max_requests_per_minute": 1200, "futures_max_requests_per_minute": 500 }
            }"#,
        )
        .unwrap();
    }

    // -----------------------------------------------------------------------
    // Env cleanup helper — prevents parallel test interference.
    // -----------------------------------------------------------------------

    /// Remove all bot-related env vars so tests don't interfere with each other.
    fn clean_bot_env() {
        for key in [
            "EXECUTOR_DRY_RUN",
            "MAX_FLASH_LOAN_USD",
            "MAX_LEVERAGE_RATIO",
            "MIN_HEALTH_FACTOR",
            "MAX_GAS_PRICE_GWEI",
            "BSC_RPC_URL_HTTP",
            "BSC_RPC_URL_HTTP_FALLBACK",
            "BSC_RPC_URL_MEV_PROTECTED",
            "LEVERAGE_EXECUTOR_ADDRESS",
            "EXECUTOR_PRIVATE_KEY",
            "USER_WALLET_ADDRESS",
        ] {
            std::env::remove_var(key);
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    #[serial]
    fn test_load_real_configs() {
        clean_bot_env();
        let dir = project_config_dir();
        if !dir.exists() {
            eprintln!("skipping — config dir not found at {}", dir.display());
            return;
        }
        // Force dry_run=true so live mode checks don't block the test.
        std::env::set_var("EXECUTOR_DRY_RUN", "true");
        let config = load_config(&dir).expect("config should load and validate");
        assert_eq!(config.chain.chain_id, 56);
        assert!(config.positions.dry_run);
        assert!(config.signals.enabled);
        clean_bot_env();
    }

    #[test]
    #[serial]
    fn test_load_test_configs() {
        clean_bot_env();
        let tmp = tempfile::tempdir().unwrap();
        write_test_configs(tmp.path());
        let config = load_config(tmp.path()).expect("test config should load");
        assert_eq!(config.chain.chain_id, 56);
        assert!(config.positions.dry_run);
        assert_eq!(config.positions.max_flash_loan_usd, 5000);
        clean_bot_env();
    }

    #[test]
    #[serial]
    fn test_missing_config_file_errors() {
        clean_bot_env();
        let tmp = tempfile::tempdir().unwrap();
        let err = load_config(tmp.path()).unwrap_err();
        assert!(
            err.to_string().contains("failed to read config file"),
            "expected file-not-found error, got: {err}"
        );
        clean_bot_env();
    }

    #[test]
    #[serial]
    fn test_env_override_dry_run_to_false() {
        clean_bot_env();
        let tmp = tempfile::tempdir().unwrap();
        write_test_configs(tmp.path());

        // Set dry_run=false AND provide required live mode env vars.
        std::env::set_var("EXECUTOR_DRY_RUN", "false");
        std::env::set_var("EXECUTOR_PRIVATE_KEY", "0xdeadbeef");
        std::env::set_var("USER_WALLET_ADDRESS", "0x1234567890abcdef1234567890abcdef12345678");
        std::env::set_var(
            "LEVERAGE_EXECUTOR_ADDRESS",
            "0xABCDEF1234567890abcdef1234567890ABCDEF12",
        );

        let config = load_config(tmp.path()).unwrap();
        assert!(!config.positions.dry_run);
        assert_eq!(
            config.chain.contracts.leverage_executor,
            "0xABCDEF1234567890abcdef1234567890ABCDEF12"
        );
        clean_bot_env();
    }

    #[test]
    #[serial]
    fn test_env_override_max_flash_loan() {
        clean_bot_env();
        let tmp = tempfile::tempdir().unwrap();
        write_test_configs(tmp.path());

        std::env::set_var("MAX_FLASH_LOAN_USD", "9999");
        let config = load_config(tmp.path()).unwrap();
        assert_eq!(config.positions.max_flash_loan_usd, 9999);
        clean_bot_env();
    }

    #[test]
    #[serial]
    fn test_env_override_rpc_url() {
        clean_bot_env();
        let tmp = tempfile::tempdir().unwrap();
        write_test_configs(tmp.path());

        std::env::set_var("BSC_RPC_URL_HTTP", "https://custom-rpc.example.com");
        let config = load_config(tmp.path()).unwrap();
        assert_eq!(config.chain.rpc.http_url, "https://custom-rpc.example.com");
        clean_bot_env();
    }

    #[test]
    #[serial]
    fn test_env_override_leverage_ratio() {
        clean_bot_env();
        let tmp = tempfile::tempdir().unwrap();
        write_test_configs(tmp.path());

        std::env::set_var("MAX_LEVERAGE_RATIO", "4.5");
        let config = load_config(tmp.path()).unwrap();
        assert_eq!(
            config.positions.max_leverage_ratio,
            Decimal::from_str("4.5").unwrap()
        );
        clean_bot_env();
    }

    #[test]
    #[serial]
    fn test_env_override_empty_string_ignored() {
        clean_bot_env();
        let tmp = tempfile::tempdir().unwrap();
        write_test_configs(tmp.path());

        std::env::set_var("MAX_FLASH_LOAN_USD", "");
        let config = load_config(tmp.path()).unwrap();
        assert_eq!(config.positions.max_flash_loan_usd, 5000);
        clean_bot_env();
    }

    #[test]
    #[serial]
    fn test_env_override_invalid_parse_ignored() {
        clean_bot_env();
        let tmp = tempfile::tempdir().unwrap();
        write_test_configs(tmp.path());

        std::env::set_var("MAX_FLASH_LOAN_USD", "not_a_number");
        let config = load_config(tmp.path()).unwrap();
        assert_eq!(config.positions.max_flash_loan_usd, 5000);
        clean_bot_env();
    }

    #[test]
    #[serial]
    fn test_live_mode_rejects_missing_private_key() {
        clean_bot_env();
        let tmp = tempfile::tempdir().unwrap();
        write_test_configs(tmp.path());

        // Set dry_run=false but omit EXECUTOR_PRIVATE_KEY.
        std::env::set_var("EXECUTOR_DRY_RUN", "false");
        std::env::set_var("USER_WALLET_ADDRESS", "0x1234567890abcdef1234567890abcdef12345678");
        std::env::set_var(
            "LEVERAGE_EXECUTOR_ADDRESS",
            "0xABCDEF1234567890abcdef1234567890ABCDEF12",
        );

        let err = load_config(tmp.path()).unwrap_err();
        assert!(
            err.to_string().contains("EXECUTOR_PRIVATE_KEY"),
            "expected missing-key error, got: {err}"
        );
        clean_bot_env();
    }

    #[test]
    #[serial]
    fn test_live_mode_rejects_missing_executor_contract() {
        clean_bot_env();
        let tmp = tempfile::tempdir().unwrap();
        write_test_configs(tmp.path());

        // Set dry_run=false, provide keys but no executor address.
        std::env::set_var("EXECUTOR_DRY_RUN", "false");
        std::env::set_var("EXECUTOR_PRIVATE_KEY", "0xdeadbeef");
        std::env::set_var("USER_WALLET_ADDRESS", "0x1234567890abcdef1234567890abcdef12345678");
        // leverage_executor is "" in test config JSON.

        let err = load_config(tmp.path()).unwrap_err();
        assert!(
            err.to_string().contains("leverage_executor"),
            "expected missing-executor error, got: {err}"
        );
        clean_bot_env();
    }
}
