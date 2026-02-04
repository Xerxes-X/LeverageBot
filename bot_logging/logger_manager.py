"""
Centralized logging for BSC Leverage Bot.

Adapted from ArbitrageTestBot/logger_manager.py (~1,408 lines).
Provides standardized logging with JSON and human-readable formatters,
per-module log files, and structured deep-dive logging.

Usage:
    from bot_logging.logger_manager import setup_module_logger, create_module_log_directories

    create_module_log_directories()
    logger = setup_module_logger('my_logger', 'my_module.log', module_folder='Health_Monitor_Logs')
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Resolve project root
_PROJECT_ROOT = Path(__file__).parent.parent

# Load logging config
try:
    from config.loader import get_config

    _app_config = get_config().get_app_config()
except ImportError:
    _app_config = {}

_LOG_DIR = str(_PROJECT_ROOT / _app_config.get("logging", {}).get("log_dir", "logs"))
_MODULE_FOLDERS = _app_config.get("logging", {}).get(
    "module_folders",
    {
        "health_monitor": "Health_Monitor_Logs",
        "strategy": "Strategy_Logs",
        "signal_engine": "Signal_Engine_Logs",
        "data_service": "Data_Service_Logs",
        "position_manager": "Position_Manager_Logs",
        "pnl_tracker": "PnL_Tracker_Logs",
        "aggregator": "Aggregator_Logs",
        "aave_client": "Aave_Client_Logs",
        "tx_submitter": "TX_Submitter_Logs",
        "safety": "Safety_Logs",
        "deep_dive": "Deep_Dive_Logs",
    },
)


# ============================================================================
# FORMATTERS
# ============================================================================


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter with correlation ID support."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Include extra fields if present
        for key in (
            "trace_id",
            "correlation_id",
            "block_number",
            "pool_address",
            "event_type",
            "channel",
            "error",
        ):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Pretty-printed log formatter for console and human-readable files."""

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)


class RawMessageFormatter(logging.Formatter):
    """Pass-through formatter for pre-formatted log messages (e.g., Redis payloads)."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


# ============================================================================
# LOGGER FACTORY
# ============================================================================

_logger_cache: dict[str, logging.Logger] = {}


def create_module_log_directories() -> dict[str, str]:
    """
    Create organized log directory structure.

    Returns dict mapping folder key to absolute path.
    """
    created = {}
    os.makedirs(_LOG_DIR, exist_ok=True)
    for key, folder_name in _MODULE_FOLDERS.items():
        folder_path = os.path.join(_LOG_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        created[key] = folder_path
    return created


def setup_module_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    module_folder: str | None = None,
    use_json_formatter: bool = False,
    use_raw_formatter: bool = False,
) -> logging.Logger:
    """
    Create a module-specific logger with file and optional console handlers.

    Args:
        name: Logger name (should be unique per module/component).
        log_file: Log filename (placed inside module_folder if specified).
        level: Logging level (default INFO).
        module_folder: Subfolder within logs/ directory (e.g., 'Event_Streamer_Logs').
        use_json_formatter: Use structured JSON format (default False = human-readable).
        use_raw_formatter: Use raw pass-through format (for Redis payload logging).

    Returns:
        Configured logging.Logger instance.
    """
    # Return cached logger if already created
    cache_key = f"{name}:{module_folder}:{log_file}"
    if cache_key in _logger_cache:
        return _logger_cache[cache_key]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        _logger_cache[cache_key] = logger
        return logger

    # Determine log file path
    if module_folder:
        log_path = os.path.join(_LOG_DIR, module_folder, log_file)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    else:
        log_path = os.path.join(_LOG_DIR, log_file)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Select formatter
    formatter: logging.Formatter
    if use_raw_formatter:
        formatter = RawMessageFormatter()
    elif use_json_formatter:
        formatter = JSONFormatter()
    else:
        formatter = HumanReadableFormatter()

    # File handler
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _logger_cache[cache_key] = logger
    return logger


def setup_deep_dive_logger() -> logging.Logger:
    """Create the centralized deep-dive data tracing logger."""
    return setup_module_logger(
        "deep_dive",
        "deep_dive_trace.log",
        module_folder="Deep_Dive_Logs",
        use_json_formatter=True,
    )


_deep_dive_logger: logging.Logger | None = None


def get_deep_dive_logger() -> logging.Logger:
    """Get or create the deep-dive logger (lazy singleton)."""
    global _deep_dive_logger
    if _deep_dive_logger is None:
        _deep_dive_logger = setup_deep_dive_logger()
    return _deep_dive_logger


# ============================================================================
# STRUCTURED LOGGING HELPERS (Deep-dive tracing)
# ============================================================================


def log_data_entry(
    trace_id: str,
    source_module: str,
    what: str,
    why: str,
    data_type: str,
    data: Any,
    previous_stage: str | None = None,
    parent_trace_ids: list[Any] | None = None,
) -> None:
    """Log a data entry event to the deep-dive trace log."""
    logger = get_deep_dive_logger()
    logger.info(
        json.dumps(
            {
                "event": "DATA_ENTRY",
                "trace_id": trace_id,
                "source_module": source_module,
                "what": what,
                "why": why,
                "data_type": data_type,
                "data": data,
                "previous_stage": previous_stage,
                "parent_trace_ids": parent_trace_ids,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            default=str,
        )
    )


def log_data_processing(
    trace_id: str,
    source_module: str,
    what: str,
    why: str,
    data_type: str,
    input_data: Any,
    output_data: Any,
    intermediate_states: list[Any] | None = None,
) -> None:
    """Log a data processing event to the deep-dive trace log."""
    logger = get_deep_dive_logger()
    logger.info(
        json.dumps(
            {
                "event": "DATA_PROCESSING",
                "trace_id": trace_id,
                "source_module": source_module,
                "what": what,
                "why": why,
                "data_type": data_type,
                "input_data": input_data,
                "output_data": output_data,
                "intermediate_states": intermediate_states,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            default=str,
        )
    )


def log_data_output(
    trace_id: str,
    source_module: str,
    what: str,
    why: str,
    data_type: str,
    data: Any,
    next_stage: str,
) -> None:
    """Log a data output event to the deep-dive trace log."""
    logger = get_deep_dive_logger()
    logger.info(
        json.dumps(
            {
                "event": "DATA_OUTPUT",
                "trace_id": trace_id,
                "source_module": source_module,
                "what": what,
                "why": why,
                "data_type": data_type,
                "data": data,
                "next_stage": next_stage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            default=str,
        )
    )
