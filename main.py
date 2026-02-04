"""
BSC Leverage Bot — Main Entrypoint.

Single-process asyncio runner that orchestrates three concurrent tasks:
    1. HealthMonitor  — tiered HF polling, oracle freshness checks
    2. SignalEngine   — 5-layer multi-source signal pipeline
    3. Strategy       — risk-filtered trade execution

All components communicate through a shared in-memory asyncio.Queue.
No multiprocessing, Redis, or external IPC — position management is not
CPU-bound and a single event loop handles all I/O concurrently.

References:
    Daian et al. (2020), "Flash Boys 2.0", IEEE S&P — MEV-protected RPC
        justification (no MEV competition for this bot).
    Heimbach & Huang (2024) — Long/short leverage negatively correlated;
        single contract handles both directions.

Usage:
    python main.py          # dry-run by default (set EXECUTOR_DRY_RUN=false)
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from typing import Any

from dotenv import load_dotenv

from bot_logging.logger_manager import setup_module_logger
from config.loader import get_config, get_env_var
from config.validate import ConfigValidationError, validate_all_configs

# ---------------------------------------------------------------------------
# Module logger (logged to logs/ root, no sub-folder)
# ---------------------------------------------------------------------------
_logger = setup_module_logger("main", "Main_Logs")


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------


def _log_banner(
    dry_run: bool,
    rpc_url: str,
    user_address: str,
    executor_address: str,
    signal_mode: str,
    max_leverage: str,
    min_hf: str,
) -> None:
    """Log a concise startup summary."""
    _logger.info("=" * 60)
    _logger.info("BSC Leverage Bot starting")
    _logger.info("=" * 60)
    _logger.info("  dry_run         : %s", dry_run)
    _logger.info(
        "  rpc             : %s...%s", rpc_url[:25], rpc_url[-6:] if len(rpc_url) > 31 else ""
    )
    _logger.info("  user_address    : %s", user_address)
    _logger.info("  executor        : %s", executor_address or "(not set)")
    _logger.info("  signal_mode     : %s", signal_mode)
    _logger.info("  max_leverage    : %sx", max_leverage)
    _logger.info("  min_health_factor: %s", min_hf)
    _logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Task done callback — detect unhandled exceptions
# ---------------------------------------------------------------------------


def _task_done_callback(
    task: asyncio.Task[None],
    shutdown_event: asyncio.Event,
) -> None:
    """Called when any of the three core tasks finishes (normally or with error)."""
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        _logger.info("Task %s cancelled", task.get_name())
        return

    if exc is not None:
        _logger.critical(
            "Task %s failed with unhandled exception: %s",
            task.get_name(),
            exc,
            exc_info=exc,
        )
        shutdown_event.set()


# ---------------------------------------------------------------------------
# Main async entry
# ---------------------------------------------------------------------------


async def _run() -> None:
    """Wire all components and launch the three concurrent tasks."""
    # ------------------------------------------------------------------
    # 1. Load environment and validate configuration
    # ------------------------------------------------------------------
    load_dotenv()

    try:
        validate_all_configs()
    except ConfigValidationError as exc:
        _logger.critical("Config validation failed:\n%s", exc)
        sys.exit(1)

    cfg = get_config()
    positions_cfg = cfg.get_positions_config()
    signals_cfg = cfg.get_signals_config()

    # Environment variables
    rpc_url: str = os.getenv("BSC_RPC_URL_HTTP", "https://bsc-dataseed1.binance.org/")
    user_address: str = os.getenv("USER_WALLET_ADDRESS", "")
    executor_address: str = os.getenv("LEVERAGE_EXECUTOR_ADDRESS", "")
    private_key: str = os.getenv("EXECUTOR_PRIVATE_KEY", "")
    dry_run: bool = get_env_var("EXECUTOR_DRY_RUN", True, bool)

    if not user_address:
        _logger.critical("USER_WALLET_ADDRESS not set in environment")
        sys.exit(1)

    signal_mode = signals_cfg.get("mode", "blended")
    max_leverage = str(positions_cfg.get("max_leverage_ratio", "3.0"))
    min_hf = str(positions_cfg.get("min_health_factor", "1.5"))

    _log_banner(dry_run, rpc_url, user_address, executor_address, signal_mode, max_leverage, min_hf)

    # ------------------------------------------------------------------
    # 2. Initialize AsyncWeb3 provider (shared across all components)
    # ------------------------------------------------------------------
    from web3 import AsyncWeb3
    from web3.providers import AsyncHTTPProvider

    w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url))
    connected = await w3.is_connected()
    if not connected:
        _logger.critical("Cannot connect to BSC RPC at %s", rpc_url)
        sys.exit(1)
    chain_id = await w3.eth.chain_id
    _logger.info("Connected to chain %d via %s", chain_id, rpc_url[:40])

    # ------------------------------------------------------------------
    # 3. Initialize shared instances (dependency order)
    # ------------------------------------------------------------------
    import aiohttp

    from core.data_service import PriceDataService
    from core.health_monitor import HealthMonitor
    from core.pnl_tracker import PnLTracker
    from core.position_manager import PositionManager
    from core.safety import SafetyState
    from core.signal_engine import SignalEngine
    from core.strategy import Strategy
    from execution.aave_client import AaveClient
    from execution.aggregator_client import AggregatorClient
    from execution.tx_submitter import TxSubmitter

    safety = SafetyState()
    aave_client = AaveClient(w3)
    aggregator_client = AggregatorClient(aave_client)
    tx_submitter = TxSubmitter(w3, safety, private_key, user_address)
    pnl_tracker = PnLTracker()
    position_manager = PositionManager(
        aave_client=aave_client,
        aggregator_client=aggregator_client,
        tx_submitter=tx_submitter,
        safety=safety,
        pnl_tracker=pnl_tracker,
        executor_address=executor_address,
        user_address=user_address,
    )

    # Shared queue: HealthMonitor and SignalEngine produce; Strategy consumes
    signal_queue: asyncio.Queue[Any] = asyncio.Queue[Any]()

    http_session = aiohttp.ClientSession()
    data_service = PriceDataService(session=http_session, w3=w3)
    health_monitor = HealthMonitor(aave_client, safety, user_address, signal_queue)
    signal_engine = SignalEngine(data_service, aave_client=aave_client, pnl_tracker=pnl_tracker)
    strategy = Strategy(position_manager, aave_client, pnl_tracker, safety, signal_queue)

    # ------------------------------------------------------------------
    # 4. Signal handling for graceful shutdown
    # ------------------------------------------------------------------
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal(sig: signal.Signals) -> None:
        _logger.info("Received %s — initiating graceful shutdown", sig.name)
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal, sig)

    # ------------------------------------------------------------------
    # 5. Launch 3 concurrent tasks
    # ------------------------------------------------------------------
    task_health = asyncio.create_task(health_monitor.run(), name="health_monitor")
    task_signal = asyncio.create_task(signal_engine.run(signal_queue), name="signal_engine")
    task_strategy = asyncio.create_task(strategy.run(), name="strategy")

    tasks = [task_health, task_signal, task_strategy]

    for t in tasks:
        t.add_done_callback(lambda done_task: _task_done_callback(done_task, shutdown_event))

    _logger.info("All tasks launched: health_monitor, signal_engine, strategy")

    # ------------------------------------------------------------------
    # 6. Wait for shutdown signal, then cancel tasks
    # ------------------------------------------------------------------
    try:
        await shutdown_event.wait()
    finally:
        _logger.info("Shutting down — cancelling tasks")

        # Signal cooperative stop
        health_monitor.stop()
        signal_engine.stop()
        strategy.stop()

        # Cancel and wait
        for t in tasks:
            if not t.done():
                t.cancel()

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for t, result in zip(tasks, results, strict=False):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                _logger.error("Task %s exited with error: %s", t.get_name(), result)

        # Cleanup resources
        await http_session.close()
        _logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        _logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
