use std::path::PathBuf;

use anyhow::Result;
use tracing::info;

use leverage_bot::config;
use leverage_bot::logging;

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

    // TODO: Phase 1+ — spawn runtime tasks.
    info!("Phase 0 complete — scaffolding only. Exiting.");

    Ok(())
}
