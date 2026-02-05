use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;

use alloy::primitives::Address;
use alloy::providers::RootProvider;
use alloy::signers::local::PrivateKeySigner;
use alloy::transports::http::reqwest::Url;
use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use leverage_bot::config;
use leverage_bot::core::data_service::DataService;
use leverage_bot::core::health_monitor::HealthMonitor;
use leverage_bot::core::mtf_data_aggregator::MultiTfDataAggregator;
use leverage_bot::core::mtf_signal_engine::MultiTfSignalEngine;
use leverage_bot::core::pnl_tracker::PnLTracker;
use leverage_bot::core::position_manager::PositionManager;
use leverage_bot::core::safety::SafetyState;
use leverage_bot::core::signal_engine::SignalEngine;
use leverage_bot::core::startup_validator::{MultiTfStartupValidator, StartupValidator};
use leverage_bot::core::strategy::Strategy;
use leverage_bot::core::websocket_manager::WebSocketManager;
use leverage_bot::execution::aave_client::AaveClient;
use leverage_bot::execution::aggregator_client::AggregatorClient;
use leverage_bot::execution::tx_submitter::TxSubmitter;
use leverage_bot::logging;
use leverage_bot::types::SignalEvent;

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file (ignore if missing).
    let _ = dotenvy::dotenv();

    // Determine config directory — default to `./config`.
    let config_dir = std::env::var("BOT_CONFIG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("config"));

    // Load and validate configuration.
    let config = config::load_config(&config_dir)?;

    // Initialize tracing — hold the guard for the process lifetime.
    let _guard = logging::init_tracing(&config.app.logging)?;

    // Print startup banner
    log_startup_banner(&config);

    // Log detailed configuration summary
    log_configuration_summary(&config);

    // -----------------------------------------------------------------------
    // Signer and addresses
    // -----------------------------------------------------------------------

    let (signer, user_address) = init_signer_and_address(config.positions.dry_run)?;

    let executor_address: Address = if config.chain.contracts.leverage_executor.is_empty() {
        if !config.positions.dry_run {
            anyhow::bail!("leverage_executor address is required in live mode");
        }
        warn!("no leverage_executor address set — using zero address (dry run)");
        Address::ZERO
    } else {
        config
            .chain
            .contracts
            .leverage_executor
            .parse()
            .context("failed to parse leverage_executor address")?
    };

    info!(
        signer = %signer.address(),
        user = %user_address,
        executor = %executor_address,
        "addresses initialized"
    );

    // -----------------------------------------------------------------------
    // Blockchain providers
    // -----------------------------------------------------------------------

    let rpc_url: Url = config
        .chain
        .rpc
        .http_url
        .parse()
        .context("failed to parse RPC URL")?;
    let provider = RootProvider::new_http(rpc_url);

    let mev_url: Url = config
        .chain
        .rpc
        .mev_protected_url
        .parse()
        .context("failed to parse MEV-protected RPC URL")?;
    let mev_provider = RootProvider::new_http(mev_url);

    info!("blockchain providers initialized");

    // -----------------------------------------------------------------------
    // Component construction (dependency injection order)
    // -----------------------------------------------------------------------

    // 1. Safety gate
    let safety = Arc::new(SafetyState::from_config(&config.positions));

    // 2. Aave V3 client
    let aave_client = Arc::new(AaveClient::new(
        provider.clone(),
        &config.aave,
        &config.chain,
    ));

    // 3. DEX aggregator client
    let aggregator_client = Arc::new(AggregatorClient::new(
        &config.aggregator,
        config.timing.aggregator.quote_cache_ttl_seconds,
    ));

    // 4. Transaction submitter (signing + MEV-protected submission)
    let tx_submitter = Arc::new(TxSubmitter::new(
        provider.clone(),
        mev_provider,
        signer,
        &config.timing.transaction,
        config.chain.chain_id,
    ));

    // 5. P&L tracker (async — creates SQLite DB + runs migrations)
    std::fs::create_dir_all("data").context("failed to create data directory")?;
    let pnl_tracker = Arc::new(
        PnLTracker::new("data/positions.db")
            .await
            .context("failed to initialize PnL tracker")?,
    );

    // 6. Redis client (optional — for mempool signal consumption)
    let redis_client = init_redis_client(&config)?;

    // 6b. Spawn mempool decoder if mempool is enabled
    let mempool_decoder_process = spawn_mempool_decoder_if_enabled(&config)?;

    // 7. Data service (Binance, Aave subgraph, Redis mempool)
    let data_service = Arc::new(DataService::new(
        config.signals.data_source.clone(),
        aave_client.clone(),
        redis_client,
        config.rate_limits.binance.clone(),
    ));

    // -----------------------------------------------------------------------
    // Shared event channel and shutdown token (created early for multi-TF)
    // -----------------------------------------------------------------------

    let (event_tx, event_rx) = mpsc::channel::<SignalEvent>(64);
    let shutdown = CancellationToken::new();

    // 7b. Multi-timeframe components (optional)
    let mtf_enabled = config
        .signals
        .multi_timeframe
        .as_ref()
        .map(|mtf| mtf.enabled)
        .unwrap_or(false);

    let (ws_manager, mtf_aggregator) = if mtf_enabled {
        info!("multi-timeframe mode enabled");

        // Initialize WebSocket manager if websocket streaming is enabled
        let ws_mgr = config
            .signals
            .websocket
            .as_ref()
            .filter(|ws| ws.enabled)
            .map(|ws_config| {
                info!(
                    ws_url = %ws_config.binance_ws_url,
                    "initializing WebSocket manager"
                );
                Arc::new(WebSocketManager::new(
                    ws_config.clone(),
                    config.signals.data_source.symbol.clone(),
                    shutdown.clone(),
                ))
            });

        // Initialize multi-TF data aggregator
        let mtf_config = config.signals.multi_timeframe.clone().unwrap();
        let aggregator = Arc::new(MultiTfDataAggregator::new(
            data_service.clone(),
            ws_mgr.clone(),
            mtf_config,
            config.signals.data_source.symbol.clone(),
        ));

        (ws_mgr, Some(aggregator))
    } else {
        info!("multi-timeframe mode disabled — using legacy signal engine");
        (None, None)
    };

    // 8. Position manager
    let position_manager = Arc::new(PositionManager::new(
        aave_client.clone(),
        aggregator_client.clone(),
        tx_submitter.clone(),
        pnl_tracker.clone(),
        safety.clone(),
        executor_address,
        user_address,
        config.positions.clone(),
        config.aave.clone(),
        config.chain.tokens.clone(),
    ));

    info!("all components initialized");

    // -----------------------------------------------------------------------
    // Startup validation - verify all data sources are accessible
    // -----------------------------------------------------------------------

    let validator = StartupValidator::new(data_service.clone(), config.signals.clone());
    let validation_result = validator.validate_all().await;

    if !validation_result.all_critical_passed {
        error!("Critical data sources unavailable - bot cannot start safely");
        if !config.positions.dry_run {
            anyhow::bail!("Startup validation failed - critical data sources unavailable");
        }
        warn!("Continuing in dry-run mode despite validation failures");
    }

    // Validate multi-timeframe sources if enabled
    if let Some(ref aggregator) = mtf_aggregator {
        let mtf_validator = MultiTfStartupValidator::new(
            aggregator.clone(),
            ws_manager.clone(),
        );
        let tf_results = mtf_validator.validate_timeframes().await;

        let failed_tfs: Vec<_> = tf_results
            .iter()
            .filter(|(_, &ok)| !ok)
            .map(|(tf, _)| format!("{:?}", tf))
            .collect();

        if !failed_tfs.is_empty() {
            warn!(
                failed = ?failed_tfs,
                "Some timeframes failed validation - signals may be degraded"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Runtime actors
    // -----------------------------------------------------------------------

    let health_monitor = HealthMonitor::new(
        aave_client.clone(),
        safety.clone(),
        config.timing.clone(),
        user_address,
        event_tx.clone(),
        shutdown.clone(),
    );

    let mut strategy = Strategy::new(
        position_manager.clone(),
        aave_client.clone(),
        pnl_tracker.clone(),
        safety.clone(),
        event_rx,
        &config.positions,
        &config.signals,
        shutdown.clone(),
    );

    // -----------------------------------------------------------------------
    // Launch concurrent tasks
    // -----------------------------------------------------------------------

    info!("spawning runtime tasks");

    let health_handle = tokio::spawn(async move {
        if let Err(e) = health_monitor.run().await {
            error!(error = %e, "health monitor exited with error");
        }
    });

    // Spawn WebSocket manager task if enabled
    let ws_handle = if let Some(ws_mgr) = ws_manager {
        let ws_mgr_clone = ws_mgr.clone();
        Some(tokio::spawn(async move {
            if let Err(e) = ws_mgr_clone.run().await {
                error!(error = %e, "websocket manager exited with error");
            }
        }))
    } else {
        None
    };

    // Spawn the appropriate signal engine based on config
    let signal_handle = if let Some(aggregator) = mtf_aggregator {
        // Multi-timeframe signal engine
        let mtf_config = config.signals.multi_timeframe.clone().unwrap();
        let mtf_signal_engine = MultiTfSignalEngine::new(
            aggregator,
            aave_client.clone(),
            pnl_tracker.clone(),
            event_tx,
            config.signals.clone(),
            mtf_config,
            user_address,
            shutdown.clone(),
        );

        tokio::spawn(async move {
            if let Err(e) = mtf_signal_engine.run().await {
                error!(error = %e, "multi-tf signal engine exited with error");
            }
        })
    } else {
        // Legacy single-timeframe signal engine
        let signal_engine = SignalEngine::new(
            data_service.clone(),
            aave_client.clone(),
            pnl_tracker.clone(),
            event_tx,
            config.signals.clone(),
            user_address,
            shutdown.clone(),
        );

        tokio::spawn(async move {
            if let Err(e) = signal_engine.run().await {
                error!(error = %e, "signal engine exited with error");
            }
        })
    };

    let strategy_handle = tokio::spawn(async move {
        if let Err(e) = strategy.run().await {
            error!(error = %e, "strategy engine exited with error");
        }
    });

    info!("all tasks running — press Ctrl+C to shutdown");

    // -----------------------------------------------------------------------
    // Wait for shutdown signal
    // -----------------------------------------------------------------------

    tokio::signal::ctrl_c()
        .await
        .context("failed to listen for Ctrl+C")?;

    info!("shutdown signal received, stopping gracefully...");
    shutdown.cancel();

    // Terminate mempool decoder if it was spawned
    if let Some(mut child) = mempool_decoder_process {
        info!("terminating mempool decoder process");
        if let Err(e) = child.kill() {
            warn!(error = %e, "failed to kill mempool decoder process");
        }
        // Wait for it to exit to avoid zombie process
        let _ = child.wait();
    }

    // Wait for all tasks to finish.
    let (health_res, signal_res, strategy_res) =
        tokio::join!(health_handle, signal_handle, strategy_handle);

    // Also wait for WebSocket handle if it was spawned
    if let Some(ws_h) = ws_handle {
        if let Err(e) = ws_h.await {
            error!(error = %e, "websocket manager task panicked");
        }
    }

    if let Err(e) = health_res {
        error!(error = %e, "health monitor task panicked");
    }
    if let Err(e) = signal_res {
        error!(error = %e, "signal engine task panicked");
    }
    if let Err(e) = strategy_res {
        error!(error = %e, "strategy engine task panicked");
    }

    info!("shutdown complete");
    Ok(())
}

// ---------------------------------------------------------------------------
// Initialization helpers
// ---------------------------------------------------------------------------

/// Initialize the transaction signer and user address from environment variables.
///
/// In dry-run mode, generates a random ephemeral signer if `EXECUTOR_PRIVATE_KEY`
/// is not set, and derives the user address from the signer if
/// `USER_WALLET_ADDRESS` is not set.
fn init_signer_and_address(dry_run: bool) -> Result<(PrivateKeySigner, Address)> {
    let signer = match std::env::var("EXECUTOR_PRIVATE_KEY")
        .ok()
        .filter(|v| !v.is_empty())
    {
        Some(key) => {
            let key = key.strip_prefix("0x").unwrap_or(&key);
            key.parse::<PrivateKeySigner>()
                .context("failed to parse EXECUTOR_PRIVATE_KEY")?
        }
        None => {
            if !dry_run {
                anyhow::bail!("EXECUTOR_PRIVATE_KEY is required in live mode");
            }
            info!("no private key set — generating ephemeral signer (dry run)");
            PrivateKeySigner::random()
        }
    };

    let user_address = match std::env::var("USER_WALLET_ADDRESS")
        .ok()
        .filter(|v| !v.is_empty())
    {
        Some(addr) => addr
            .parse::<Address>()
            .context("failed to parse USER_WALLET_ADDRESS")?,
        None => {
            if !dry_run {
                anyhow::bail!("USER_WALLET_ADDRESS is required in live mode");
            }
            let addr = signer.address();
            info!(%addr, "no user address set — using signer address (dry run)");
            addr
        }
    };

    Ok((signer, user_address))
}

/// Initialize the Redis client from mempool config, if enabled.
fn init_redis_client(config: &config::BotConfig) -> Result<Option<redis::Client>> {
    match &config.mempool {
        Some(mempool) if mempool.enabled => {
            let client = redis::Client::open(mempool.redis.url.as_str())
                .context("failed to create Redis client")?;
            info!("Redis client initialized for mempool");
            Ok(Some(client))
        }
        _ => {
            info!("mempool disabled — skipping Redis client");
            Ok(None)
        }
    }
}

/// Spawn the mempool decoder binary as a child process if mempool is enabled.
///
/// The decoder binary is expected to be in the same directory as the main bot binary,
/// or in the target/release directory during development.
fn spawn_mempool_decoder_if_enabled(config: &config::BotConfig) -> Result<Option<Child>> {
    // Check if mempool is enabled
    let mempool_config = match &config.mempool {
        Some(m) if m.enabled => m,
        _ => {
            info!("mempool disabled — not spawning mempool decoder");
            return Ok(None);
        }
    };

    // Find the mempool-decoder binary
    let decoder_path = find_mempool_decoder_binary()?;

    // Get WebSocket URL: try env var first, then fallback from config
    let ws_url = std::env::var(&mempool_config.decoder.websocket_url_env)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| mempool_config.decoder.websocket_fallback.clone());

    info!(
        path = %decoder_path.display(),
        redis_url = %mempool_config.redis.url,
        ws_url = %ws_url,
        "spawning mempool decoder process"
    );

    // Build environment variables for the decoder
    let rust_log = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());

    // Spawn the process
    let child = Command::new(&decoder_path)
        .env("RUST_LOG", rust_log)
        .env("REDIS_URL", &mempool_config.redis.url)
        .env("BSC_RPC_URL_WS", &ws_url)
        .env(
            "DECODER_AGGREGATE_CHANNEL",
            &mempool_config.redis.aggregate_signal_channel,
        )
        .env(
            "DECODER_PUBLISH_INTERVAL",
            mempool_config.redis.aggregate_publish_interval_seconds.to_string(),
        )
        .env(
            "DECODER_DEDUP_CACHE_SIZE",
            mempool_config.decoder.dedup_cache_size.to_string(),
        )
        .env(
            "DECODER_RECONNECT_DELAY",
            mempool_config.decoder.reconnect_delay_seconds.to_string(),
        )
        .env(
            "DECODER_MAX_RECONNECT",
            mempool_config.decoder.max_reconnect_attempts.to_string(),
        )
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("failed to spawn mempool-decoder process")?;

    info!(pid = child.id(), "mempool decoder started");

    Ok(Some(child))
}

/// Find the mempool-decoder binary path.
///
/// Searches in order:
/// 1. Same directory as current executable
/// 2. ./target/release/mempool-decoder
/// 3. PATH
fn find_mempool_decoder_binary() -> Result<PathBuf> {
    // Try same directory as current executable
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(exe_dir) = current_exe.parent() {
            let decoder_path = exe_dir.join("mempool-decoder");
            if decoder_path.exists() {
                return Ok(decoder_path);
            }
        }
    }

    // Try target/release directory (development)
    let dev_path = PathBuf::from("target/release/mempool-decoder");
    if dev_path.exists() {
        return Ok(dev_path);
    }

    // Try current directory
    let cwd_path = PathBuf::from("mempool-decoder");
    if cwd_path.exists() {
        return Ok(cwd_path);
    }

    // Fallback: assume it's in PATH
    Ok(PathBuf::from("mempool-decoder"))
}

// ---------------------------------------------------------------------------
// Logging helpers
// ---------------------------------------------------------------------------

/// Print the startup banner with version and build info.
fn log_startup_banner(config: &config::BotConfig) {
    let version = env!("CARGO_PKG_VERSION");
    let mode = if config.positions.dry_run {
        "DRY RUN"
    } else {
        "LIVE"
    };

    info!("╔═══════════════════════════════════════════════════════════════╗");
    info!("║                                                               ║");
    info!("║   ██╗     ███████╗██╗   ██╗███████╗██████╗  █████╗  ██████╗  ║");
    info!("║   ██║     ██╔════╝██║   ██║██╔════╝██╔══██╗██╔══██╗██╔════╝  ║");
    info!("║   ██║     █████╗  ██║   ██║█████╗  ██████╔╝███████║██║  ███╗ ║");
    info!("║   ██║     ██╔══╝  ╚██╗ ██╔╝██╔══╝  ██╔══██╗██╔══██║██║   ██║ ║");
    info!("║   ███████╗███████╗ ╚████╔╝ ███████╗██████╔╝██║  ██║╚██████╔╝ ║");
    info!("║   ╚══════╝╚══════╝  ╚═══╝  ╚══════╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ║");
    info!("║                                                               ║");
    info!("║                  BSC Aave V3 Leverage Bot                     ║");
    info!("║                                                               ║");
    info!("╠═══════════════════════════════════════════════════════════════╣");
    info!(
        "║  Version: {:<10}  Mode: {:<10}  Chain: {:<13}║",
        version, mode, config.chain.chain_name
    );
    info!("╚═══════════════════════════════════════════════════════════════╝");
}

/// Log a detailed configuration summary.
fn log_configuration_summary(config: &config::BotConfig) {
    info!("═══════════════════════════════════════════════════════════════");
    info!("                   CONFIGURATION SUMMARY");
    info!("═══════════════════════════════════════════════════════════════");

    // Chain configuration
    info!("───────────────────────────────────────────────────────────────");
    info!("CHAIN CONFIGURATION:");
    info!(
        chain_id = config.chain.chain_id,
        chain_name = %config.chain.chain_name,
        rpc_url = %config.chain.rpc.http_url,
        mev_url = %config.chain.rpc.mev_protected_url,
        "Chain settings"
    );

    // Position configuration
    info!("───────────────────────────────────────────────────────────────");
    info!("POSITION CONFIGURATION:");
    info!(
        dry_run = config.positions.dry_run,
        max_leverage_ratio = %config.positions.max_leverage_ratio,
        max_flash_loan_usd = config.positions.max_flash_loan_usd,
        max_position_usd = config.positions.max_position_usd,
        min_health_factor = %config.positions.min_health_factor,
        max_slippage_bps = config.positions.max_slippage_bps,
        "Position parameters"
    );

    // Signal configuration
    info!("───────────────────────────────────────────────────────────────");
    info!("SIGNAL CONFIGURATION:");
    info!(
        enabled = config.signals.enabled,
        mode = %config.signals.mode,
        symbol = %config.signals.data_source.symbol,
        interval = %config.signals.data_source.interval,
        history_candles = config.signals.data_source.history_candles,
        refresh_interval_secs = config.signals.data_source.refresh_interval_seconds,
        "Signal parameters"
    );
    info!(
        min_confidence = %config.signals.entry_rules.min_confidence,
        require_trend_alignment = config.signals.entry_rules.require_trend_alignment,
        require_volume_confirmation = config.signals.entry_rules.require_volume_confirmation,
        max_signals_per_day = config.signals.entry_rules.max_signals_per_day,
        "Entry rules"
    );

    // Multi-timeframe configuration
    let mtf_enabled = config
        .signals
        .multi_timeframe
        .as_ref()
        .map(|mtf| mtf.enabled)
        .unwrap_or(false);

    if mtf_enabled {
        let mtf = config.signals.multi_timeframe.as_ref().unwrap();
        info!("───────────────────────────────────────────────────────────────");
        info!("MULTI-TIMEFRAME CONFIGURATION:");
        info!(
            enabled = mtf.enabled,
            trading_style = ?mtf.trading_style,
            "Multi-TF mode"
        );

        let enabled_tfs: Vec<_> = mtf
            .timeframes
            .iter()
            .filter(|tf| tf.enabled)
            .map(|tf| format!("{:?}", tf.timeframe))
            .collect();
        info!(
            timeframes = ?enabled_tfs,
            count = enabled_tfs.len(),
            "Enabled timeframes"
        );

        info!(
            weight_mode = %mtf.aggregation.weight_mode,
            min_agreement = %mtf.aggregation.min_timeframe_agreement,
            require_htf_alignment = mtf.aggregation.require_higher_tf_alignment,
            "Aggregation settings"
        );
    }

    // WebSocket configuration
    if let Some(ws) = &config.signals.websocket {
        if ws.enabled {
            info!("───────────────────────────────────────────────────────────────");
            info!("WEBSOCKET CONFIGURATION:");
            info!(
                enabled = ws.enabled,
                ws_url = %ws.binance_ws_url,
                reconnect_delay_ms = ws.reconnect_delay_ms,
                max_reconnect_attempts = ws.max_reconnect_attempts,
                "WebSocket settings"
            );
            info!(
                kline_streams = ws.subscriptions.klines.len(),
                depth_enabled = ws.subscriptions.depth,
                trades_enabled = ws.subscriptions.trades,
                "Stream subscriptions"
            );
        }
    }

    // Aggregator configuration
    info!("───────────────────────────────────────────────────────────────");
    info!("DEX AGGREGATOR CONFIGURATION:");
    let provider_names: Vec<_> = config
        .aggregator
        .providers
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    info!(
        providers = ?provider_names,
        count = provider_names.len(),
        max_slippage_bps = config.aggregator.max_slippage_bps,
        max_price_impact_pct = %config.aggregator.max_price_impact_percent,
        "Aggregator settings"
    );

    // Aave configuration
    info!("───────────────────────────────────────────────────────────────");
    info!("AAVE V3 CONFIGURATION:");
    info!(
        flash_loan_premium_bps = config.aave.flash_loan_premium_bps,
        referral_code = config.aave.referral_code,
        supported_assets = config.aave.supported_assets.len(),
        "Aave settings"
    );

    // Mempool configuration
    let mempool_enabled = config.mempool.as_ref().is_some_and(|m| m.enabled);
    info!("───────────────────────────────────────────────────────────────");
    info!("MEMPOOL CONFIGURATION:");
    info!(
        enabled = mempool_enabled,
        "Mempool monitoring"
    );

    // Timing configuration
    info!("───────────────────────────────────────────────────────────────");
    info!("TIMING CONFIGURATION:");
    info!(
        health_safe_interval_secs = config.timing.health_monitoring.safe_interval_seconds,
        health_critical_interval_secs = config.timing.health_monitoring.critical_interval_seconds,
        tx_confirm_timeout_secs = config.timing.transaction.confirmation_timeout_seconds,
        tx_simulation_timeout_secs = config.timing.transaction.simulation_timeout_seconds,
        "Timing parameters"
    );

    info!("═══════════════════════════════════════════════════════════════");
}
