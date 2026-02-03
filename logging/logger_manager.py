"""
Centralized logging for BSC Leverage Bot.

Adapted from ArbitrageTestBot/logger_manager.py (~1,408 lines).
Provides standardized logging with JSON and human-readable formatters,
per-module log files, and structured deep-dive logging.

Usage:
    from logging.logger_manager import setup_module_logger, create_module_log_directories

    create_module_log_directories()
    logger = setup_module_logger('my_logger', 'my_module.log', module_folder='Event_Streamer_Logs')
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Resolve project root
_PROJECT_ROOT = Path(__file__).parent.parent

# Load logging config
try:
    from config.loader import get_config
    _app_config = get_config().get_app_config()
except ImportError:
    _app_config = {}

_LOG_DIR = str(_PROJECT_ROOT / _app_config.get("logging", {}).get("log_dir", "logs"))
_MODULE_FOLDERS = _app_config.get("logging", {}).get("module_folders", {
    "event_streamer": "Event_Streamer_Logs",
    "pool_managers": "Pool_Manager_Logs",
    "aave_state": "Aave_State_Logs",
    "venue_aggregator": "Venue_Aggregator_Logs",
    "split_optimizer": "Split_Optimizer_Logs",
    "leverage_engine": "Leverage_Engine_Logs",
    "position_manager": "Position_Manager_Logs",
    "liquidation_monitor": "Liquidation_Monitor_Logs",
    "verifier": "Verifier_Logs",
    "executor": "Executor_Logs",
    "bundle_submitter": "Bundle_Submitter_Logs",
    "gas_oracle": "Gas_Oracle_Logs",
    "safety": "Safety_Logs",
    "deep_dive": "Deep_Dive_Logs",
})


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
        for key in ("trace_id", "correlation_id", "block_number", "pool_address",
                     "event_type", "channel", "error"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Pretty-printed log formatter for console and human-readable files."""

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)


class RawMessageFormatter(logging.Formatter):
    """Pass-through formatter for pre-formatted log messages (e.g., Redis payloads)."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


# ============================================================================
# LOGGER FACTORY
# ============================================================================

_logger_cache: Dict[str, logging.Logger] = {}


def create_module_log_directories() -> Dict[str, str]:
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
    module_folder: Optional[str] = None,
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


_deep_dive_logger: Optional[logging.Logger] = None


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
    previous_stage: Optional[str] = None,
    parent_trace_ids: Optional[list] = None,
) -> None:
    """Log a data entry event to the deep-dive trace log."""
    logger = get_deep_dive_logger()
    logger.info(json.dumps({
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
    }, default=str))


def log_data_processing(
    trace_id: str,
    source_module: str,
    what: str,
    why: str,
    data_type: str,
    input_data: Any,
    output_data: Any,
    intermediate_states: Optional[list] = None,
) -> None:
    """Log a data processing event to the deep-dive trace log."""
    logger = get_deep_dive_logger()
    logger.info(json.dumps({
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
    }, default=str))


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
    logger.info(json.dumps({
        "event": "DATA_OUTPUT",
        "trace_id": trace_id,
        "source_module": source_module,
        "what": what,
        "why": why,
        "data_type": data_type,
        "data": data,
        "next_stage": next_stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, default=str))
