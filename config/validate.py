"""
Configuration schema validation for BSC Leverage Bot.

Validates that all required config files exist and contain required keys.
Run at startup to fail fast on misconfiguration.
"""

from typing import Any

from config.loader import get_config


class ConfigValidationError(ValueError):
    """Raised when a required config key is missing or invalid."""

    pass


def _check_keys(config: dict[str, Any], required_keys: list[str], config_name: str) -> list[str]:
    """Check that all required keys exist in a config dict. Returns list of missing keys."""
    missing = []
    for key in required_keys:
        parts = key.split(".")
        current = config
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                missing.append(key)
                break
            current = current[part]
    return missing


def validate_chain_config(config: dict[str, Any]) -> list[str]:
    """Validate chains/56.json has required fields."""
    return _check_keys(
        config,
        [
            "chain_id",
            "block_time_seconds",
            "rpc.http_url",
            "contracts.aave_v3_pool",
            "contracts.aave_v3_data_provider",
            "chainlink_feeds",
            "tokens",
        ],
        "chains/56.json",
    )


def validate_aave_config(config: dict[str, Any]) -> list[str]:
    """Validate aave.json has required fields."""
    return _check_keys(
        config,
        [
            "flash_loan_premium_bps",
            "supported_assets",
        ],
        "aave.json",
    )


def validate_positions_config(config: dict[str, Any]) -> list[str]:
    """Validate positions.json has required fields."""
    return _check_keys(
        config,
        [
            "dry_run",
            "max_flash_loan_usd",
            "max_position_usd",
            "max_leverage_ratio",
            "min_health_factor",
            "deleverage_threshold",
            "close_threshold",
            "max_gas_price_gwei",
        ],
        "positions.json",
    )


def validate_aggregator_config(config: dict[str, Any]) -> list[str]:
    """Validate aggregator.json has required fields."""
    errors = _check_keys(config, ["providers"], "aggregator.json")
    if not errors:
        providers = config.get("providers", [])
        if not isinstance(providers, list) or len(providers) == 0:
            errors.append("providers: must be a non-empty list")
        else:
            enabled = [p for p in providers if p.get("enabled", False)]
            if len(enabled) == 0:
                errors.append("providers: at least one provider must be enabled")
    return errors


def validate_signals_config(config: dict[str, Any]) -> list[str]:
    """Validate signals.json has required fields."""
    return _check_keys(
        config,
        [
            "enabled",
            "mode",
            "data_source",
            "indicators",
        ],
        "signals.json",
    )


def validate_all_configs() -> None:
    """
    Validate all config files. Raises ConfigValidationError with details
    if any required keys are missing.
    """
    loader = get_config()
    all_errors: dict[str, list[str]] = {}

    validators = {
        "chains/56.json": (loader.get_chain_config, validate_chain_config),
        "aave.json": (loader.get_aave_config, validate_aave_config),
        "positions.json": (loader.get_positions_config, validate_positions_config),
        "aggregator.json": (loader.get_aggregator_config, validate_aggregator_config),
        "signals.json": (loader.get_signals_config, validate_signals_config),
    }

    for config_name, (loader_fn, validator_fn) in validators.items():
        config = loader_fn()
        if not config:
            all_errors[config_name] = ["Config file is empty or not found"]
            continue
        errors = validator_fn(config)
        if errors:
            all_errors[config_name] = errors

    if all_errors:
        lines = ["Configuration validation failed:"]
        for config_name, errors in all_errors.items():
            lines.append(f"\n  {config_name}:")
            for error in errors:
                lines.append(f"    - missing: {error}")
        raise ConfigValidationError("\n".join(lines))
