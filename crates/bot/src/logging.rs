use anyhow::Result;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::config::LoggingConfig;

/// Initialise the global tracing subscriber.
///
/// Returns a [`WorkerGuard`] that **must** be held for the lifetime of the
/// process — dropping it flushes and closes the log file writer.
pub fn init_tracing(logging: &LoggingConfig) -> Result<WorkerGuard> {
    // Create log directory if it doesn't exist.
    std::fs::create_dir_all(&logging.log_dir)?;

    let file_appender = tracing_appender::rolling::daily(&logging.log_dir, "bot.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("leverage_bot=info,warn"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(
            fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false)
                .json(),
        )
        .with(
            fmt::layer()
                .with_writer(std::io::stderr)
                .with_target(true)
                .compact(),
        )
        .init();

    Ok(guard)
}
