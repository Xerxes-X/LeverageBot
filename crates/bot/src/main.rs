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
use leverage_bot::core::pnl_tracker::PnLTracker;
use leverage_bot::core::position_manager::PositionManager;
use leverage_bot::core::safety::SafetyState;
use leverage_bot::core::signal_engine::SignalEngine;
use leverage_bot::core::strategy::Strategy;
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

    info!(
        chain_id = config.chain.chain_id,
        chain_name = %config.chain.chain_name,
        dry_run = config.positions.dry_run,
        "BSC Leverage Bot starting"
    );

    info!(
        providers = config.aggregator.providers.len(),
        signals_enabled = config.signals.enabled,
        mempool_enabled = config.mempool.as_ref().is_some_and(|m| m.enabled),
        "configuration loaded successfully"
    );

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
    // Shared event channel and shutdown token
    // -----------------------------------------------------------------------

    let (event_tx, event_rx) = mpsc::channel::<SignalEvent>(64);
    let shutdown = CancellationToken::new();

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

    let signal_engine = SignalEngine::new(
        data_service.clone(),
        aave_client.clone(),
        pnl_tracker.clone(),
        event_tx, // last sender — no more clones needed
        config.signals.clone(),
        user_address,
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

    let signal_handle = tokio::spawn(async move {
        if let Err(e) = signal_engine.run().await {
            error!(error = %e, "signal engine exited with error");
        }
    });

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
