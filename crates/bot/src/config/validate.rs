use anyhow::{bail, Result};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use super::types::BotConfig;

/// Validate invariants across the merged config that serde alone cannot enforce.
///
/// This replaces the Python `config/validate.py` validators with stronger
/// compile-time + runtime checks. Called automatically by [`super::load_config`].
pub fn validate_config(config: &BotConfig) -> Result<()> {
    let mut errors: Vec<String> = Vec::new();

    validate_chain_config(config, &mut errors);
    validate_positions_config(config, &mut errors);
    validate_aave_config(config, &mut errors);
    validate_aggregator_config(config, &mut errors);
    validate_signals_config(config, &mut errors);
    validate_timing_config(config, &mut errors);
    validate_live_mode_requirements(config, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        let msg = format!(
            "Configuration validation failed ({} error{}):\n  - {}",
            errors.len(),
            if errors.len() == 1 { "" } else { "s" },
            errors.join("\n  - ")
        );
        bail!("{msg}");
    }
}

// ---------------------------------------------------------------------------
// Chain config
// ---------------------------------------------------------------------------

fn validate_chain_config(config: &BotConfig, errors: &mut Vec<String>) {
    let chain = &config.chain;

    if chain.chain_id != 56 {
        errors.push(format!(
            "chain: only BSC (chain_id=56) is supported, got {}",
            chain.chain_id
        ));
    }

    if chain.rpc.http_url.is_empty() {
        errors.push("chain.rpc: http_url is empty".into());
    }

    // Validate contract addresses.
    let contract_addrs = [
        ("aave_v3_pool", &chain.contracts.aave_v3_pool),
        (
            "aave_v3_pool_addresses_provider",
            &chain.contracts.aave_v3_pool_addresses_provider,
        ),
        ("aave_v3_oracle", &chain.contracts.aave_v3_oracle),
        ("aave_v3_data_provider", &chain.contracts.aave_v3_data_provider),
        ("multicall3", &chain.contracts.multicall3),
    ];

    for (name, addr) in &contract_addrs {
        if let Err(e) = validate_address(addr) {
            errors.push(format!("chain.contracts.{name}: {e}"));
        }
    }

    // leverage_executor can be empty (not yet deployed), but if set must be valid.
    if !chain.contracts.leverage_executor.is_empty() {
        if let Err(e) = validate_address(&chain.contracts.leverage_executor) {
            errors.push(format!("chain.contracts.leverage_executor: {e}"));
        }
    }

    // Must have at least one chainlink feed.
    if chain.chainlink_feeds.is_empty() {
        errors.push("chain.chainlink_feeds: must have at least one feed".into());
    }

    for (name, feed) in &chain.chainlink_feeds {
        if let Err(e) = validate_address(&feed.address) {
            errors.push(format!("chain.chainlink_feeds.{name}.address: {e}"));
        }
    }

    // Must have at least one token.
    if chain.tokens.is_empty() {
        errors.push("chain.tokens: must have at least one token".into());
    }

    for (name, token) in &chain.tokens {
        if let Err(e) = validate_address(&token.address) {
            errors.push(format!("chain.tokens.{name}.address: {e}"));
        }
    }
}

// ---------------------------------------------------------------------------
// Positions config
// ---------------------------------------------------------------------------

fn validate_positions_config(config: &BotConfig, errors: &mut Vec<String>) {
    let pos = &config.positions;

    // Health factor thresholds must be ordered:
    // close_threshold < deleverage_threshold < min_health_factor
    if pos.close_threshold >= pos.deleverage_threshold {
        errors.push(format!(
            "positions: close_threshold ({}) must be < deleverage_threshold ({})",
            pos.close_threshold, pos.deleverage_threshold
        ));
    }
    if pos.deleverage_threshold >= pos.min_health_factor {
        errors.push(format!(
            "positions: deleverage_threshold ({}) must be < min_health_factor ({})",
            pos.deleverage_threshold, pos.min_health_factor
        ));
    }

    // All health thresholds must be > 1.0 (liquidation).
    if pos.min_health_factor <= dec!(1) {
        errors.push(format!(
            "positions: min_health_factor ({}) must be > 1.0",
            pos.min_health_factor
        ));
    }
    if pos.close_threshold <= dec!(1) {
        errors.push(format!(
            "positions: close_threshold ({}) must be > 1.0",
            pos.close_threshold
        ));
    }

    // Leverage ratio: must be > 1.0 and <= 5.0 (safety cap).
    if pos.max_leverage_ratio <= dec!(1) || pos.max_leverage_ratio > dec!(5) {
        errors.push(format!(
            "positions: max_leverage_ratio ({}) must be in (1.0, 5.0]",
            pos.max_leverage_ratio
        ));
    }

    // Slippage sanity.
    if pos.max_slippage_bps > 500 {
        errors.push(format!(
            "positions: max_slippage_bps ({}) exceeds 500 (5%)",
            pos.max_slippage_bps
        ));
    }

    // Gas price sanity.
    if pos.max_gas_price_gwei == 0 {
        errors.push("positions: max_gas_price_gwei must be > 0".into());
    }

    // Target HF after deleverage should be >= min_health_factor.
    if pos.target_hf_after_deleverage < pos.min_health_factor {
        errors.push(format!(
            "positions: target_hf_after_deleverage ({}) should be >= min_health_factor ({})",
            pos.target_hf_after_deleverage, pos.min_health_factor
        ));
    }
}

// ---------------------------------------------------------------------------
// Aave config
// ---------------------------------------------------------------------------

fn validate_aave_config(config: &BotConfig, errors: &mut Vec<String>) {
    let aave = &config.aave;

    if aave.supported_assets.is_empty() {
        errors.push("aave: supported_assets must not be empty".into());
    }

    for (name, asset) in &aave.supported_assets {
        // LTV must be < liquidation threshold.
        if asset.ltv_bps >= asset.liquidation_threshold_bps {
            errors.push(format!(
                "aave.supported_assets.{name}: ltv_bps ({}) must be < liquidation_threshold_bps ({})",
                asset.ltv_bps, asset.liquidation_threshold_bps
            ));
        }
        // Thresholds must be <= 10000 (100%).
        if asset.liquidation_threshold_bps > 10000 {
            errors.push(format!(
                "aave.supported_assets.{name}: liquidation_threshold_bps ({}) exceeds 10000",
                asset.liquidation_threshold_bps
            ));
        }
    }

    // Flash loan premium sanity.
    if aave.flash_loan_premium < Decimal::ZERO || aave.flash_loan_premium > dec!(0.01) {
        errors.push(format!(
            "aave: flash_loan_premium ({}) should be in [0, 0.01]",
            aave.flash_loan_premium
        ));
    }
}

// ---------------------------------------------------------------------------
// Aggregator config
// ---------------------------------------------------------------------------

fn validate_aggregator_config(config: &BotConfig, errors: &mut Vec<String>) {
    let agg = &config.aggregator;

    if agg.providers.is_empty() {
        errors.push("aggregator: providers list is empty".into());
        return;
    }

    let enabled_count = agg.providers.iter().filter(|p| p.enabled).count();
    if enabled_count == 0 {
        errors.push("aggregator: at least one provider must be enabled".into());
    }

    for provider in &agg.providers {
        if provider.base_url.is_empty() {
            errors.push(format!(
                "aggregator.providers.{}: base_url is empty",
                provider.name
            ));
        }
        if provider.approved_routers.is_empty() {
            errors.push(format!(
                "aggregator.providers.{}: approved_routers is empty",
                provider.name
            ));
        }
        for (i, router) in provider.approved_routers.iter().enumerate() {
            if let Err(e) = validate_address(router) {
                errors.push(format!(
                    "aggregator.providers.{}.approved_routers[{i}]: {e}",
                    provider.name
                ));
            }
        }
    }

    // Price impact must be positive.
    if agg.max_price_impact_percent <= Decimal::ZERO {
        errors.push(format!(
            "aggregator: max_price_impact_percent ({}) must be > 0",
            agg.max_price_impact_percent
        ));
    }
}

// ---------------------------------------------------------------------------
// Signals config
// ---------------------------------------------------------------------------

fn validate_signals_config(config: &BotConfig, errors: &mut Vec<String>) {
    let sig = &config.signals;

    // Mode must be one of the known modes.
    let valid_modes = ["momentum", "mean_reversion", "blended", "manual"];
    if !valid_modes.contains(&sig.mode.as_str()) {
        errors.push(format!(
            "signals: mode '{}' is not one of {:?}",
            sig.mode, valid_modes
        ));
    }

    // Confidence range.
    if sig.entry_rules.min_confidence < Decimal::ZERO
        || sig.entry_rules.min_confidence > dec!(1)
    {
        errors.push(format!(
            "signals.entry_rules: min_confidence ({}) must be in [0, 1]",
            sig.entry_rules.min_confidence
        ));
    }

    // EMA ordering: fast < slow < trend.
    let ind = &sig.indicators;
    if ind.ema_fast >= ind.ema_slow {
        errors.push(format!(
            "signals.indicators: ema_fast ({}) must be < ema_slow ({})",
            ind.ema_fast, ind.ema_slow
        ));
    }
    if ind.ema_slow >= ind.ema_trend {
        errors.push(format!(
            "signals.indicators: ema_slow ({}) must be < ema_trend ({})",
            ind.ema_slow, ind.ema_trend
        ));
    }

    // MACD: fast < slow.
    if ind.macd_fast >= ind.macd_slow {
        errors.push(format!(
            "signals.indicators: macd_fast ({}) must be < macd_slow ({})",
            ind.macd_fast, ind.macd_slow
        ));
    }

    // GARCH parameters: alpha + beta < 1 for stationarity.
    let garch_sum = ind.garch_alpha + ind.garch_beta;
    if garch_sum >= dec!(1) {
        errors.push(format!(
            "signals.indicators: garch_alpha + garch_beta ({}) must be < 1 for stationarity",
            garch_sum
        ));
    }

    // Hurst regime thresholds: mean_reverting < trending.
    let rf = &sig.entry_rules.regime_filter;
    if rf.enabled && rf.mean_reverting_hurst_threshold >= rf.trending_hurst_threshold {
        errors.push(format!(
            "signals.entry_rules.regime_filter: mean_reverting_hurst_threshold ({}) must be < trending_hurst_threshold ({})",
            rf.mean_reverting_hurst_threshold, rf.trending_hurst_threshold
        ));
    }

    // Exit rules: stop_loss must be positive, take_profit must be positive.
    if sig.exit_rules.stop_loss_percent <= Decimal::ZERO {
        errors.push(format!(
            "signals.exit_rules: stop_loss_percent ({}) must be > 0",
            sig.exit_rules.stop_loss_percent
        ));
    }
    if sig.exit_rules.take_profit_percent <= Decimal::ZERO {
        errors.push(format!(
            "signals.exit_rules: take_profit_percent ({}) must be > 0",
            sig.exit_rules.take_profit_percent
        ));
    }
}

// ---------------------------------------------------------------------------
// Timing config
// ---------------------------------------------------------------------------

fn validate_timing_config(config: &BotConfig, errors: &mut Vec<String>) {
    let hm = &config.timing.health_monitoring;

    // Monitoring intervals should decrease as severity increases.
    if hm.safe_interval_seconds <= hm.watch_interval_seconds {
        errors.push(format!(
            "timing.health_monitoring: safe_interval ({}) should be > watch_interval ({})",
            hm.safe_interval_seconds, hm.watch_interval_seconds
        ));
    }
    if hm.watch_interval_seconds <= hm.warning_interval_seconds {
        errors.push(format!(
            "timing.health_monitoring: watch_interval ({}) should be > warning_interval ({})",
            hm.watch_interval_seconds, hm.warning_interval_seconds
        ));
    }
    if hm.warning_interval_seconds <= hm.critical_interval_seconds {
        errors.push(format!(
            "timing.health_monitoring: warning_interval ({}) should be > critical_interval ({})",
            hm.warning_interval_seconds, hm.critical_interval_seconds
        ));
    }
}

// ---------------------------------------------------------------------------
// Live mode requirements
// ---------------------------------------------------------------------------

fn validate_live_mode_requirements(config: &BotConfig, errors: &mut Vec<String>) {
    if config.positions.dry_run {
        return; // Skip live-mode checks in dry-run.
    }

    // In live mode, critical env vars must be set.
    if std::env::var("EXECUTOR_PRIVATE_KEY")
        .ok()
        .filter(|v| !v.is_empty())
        .is_none()
    {
        errors.push("live mode: EXECUTOR_PRIVATE_KEY env var is required when dry_run=false".into());
    }

    if std::env::var("USER_WALLET_ADDRESS")
        .ok()
        .filter(|v| !v.is_empty())
        .is_none()
    {
        errors.push("live mode: USER_WALLET_ADDRESS env var is required when dry_run=false".into());
    }

    // Leverage executor contract must be deployed.
    if config.chain.contracts.leverage_executor.is_empty() {
        errors.push(
            "live mode: chain.contracts.leverage_executor must be set when dry_run=false".into(),
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate an Ethereum-style address string: must be 0x-prefixed and 42 chars
/// of hex.
fn validate_address(addr: &str) -> Result<(), String> {
    if addr.is_empty() {
        return Err("address is empty".into());
    }
    if !addr.starts_with("0x") && !addr.starts_with("0X") {
        return Err(format!("address '{addr}' must start with 0x"));
    }
    if addr.len() != 42 {
        return Err(format!(
            "address '{addr}' has length {} (expected 42)",
            addr.len()
        ));
    }
    // Verify remaining chars are hex.
    if !addr[2..].chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("address '{addr}' contains non-hex characters"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_address_valid() {
        assert!(validate_address("0xcA11bde05977b3631167028862bE2a173976CA11").is_ok());
        assert!(validate_address("0x6807dc923806fE8Fd134338EABCA509979a7e0cB").is_ok());
    }

    #[test]
    fn test_validate_address_empty() {
        assert!(validate_address("").is_err());
    }

    #[test]
    fn test_validate_address_no_prefix() {
        let err = validate_address("cA11bde05977b3631167028862bE2a173976CA11").unwrap_err();
        assert!(err.contains("must start with 0x"));
    }

    #[test]
    fn test_validate_address_wrong_length() {
        let err = validate_address("0xcA11bde05977b3631167028862bE2a17").unwrap_err();
        assert!(err.contains("length"));
    }

    #[test]
    fn test_validate_address_non_hex() {
        let err = validate_address("0xZZ11bde05977b3631167028862bE2a173976CA11").unwrap_err();
        assert!(err.contains("non-hex"));
    }
}
