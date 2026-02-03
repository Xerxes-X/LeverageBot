"""
BSC Event Streamer - Real-Time Blockchain Event Listener

Adapted from ArbitrageTestBot/dex_price_streamer.py (~4,522 lines).
Reuse: ~70% (WebSocket subscription, event parsing, Redis publishing, DecimalEncoder)

Purpose:
    Connect to BSC node via WebSocket, subscribe to real-time log events from all
    target DEX pools AND Aave V3 lending pool, parse raw events, publish to
    protocol-specific Redis channels.

Supported Protocols:
    - V2 events (PancakeSwap V2, BiSwap, SushiSwap): Swap, Sync
    - V3 events (PancakeSwap V3, Uniswap V3, Thena): Swap, Mint, Burn
    - Ellipsis/Curve events: TokenExchange, TokenExchangeUnderlying
    - Wombat events: Swap
    - DODO events: DODOSwap
    - Aave V3 events: Supply, Borrow, Repay, FlashLoan, LiquidationCall, ReserveDataUpdated

BSC-Specific:
    - 0.45s block time (Fermi hardfork) = ~133 blocks/minute
    - Higher event throughput requires efficient batch publishing
    - baseFeePerGas available in block headers since Fermi

References:
    - ArbitrageTestBot/dex_price_streamer.py
    - Aave V3 event signatures from ABI
    - BNB Chain Fermi hardfork documentation
"""

import asyncio
import json
import logging
import os
import random
import signal
import socket
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, getcontext, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Tuple

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from dotenv import load_dotenv
from eth_abi.abi import decode as abi_decode
from web3 import Web3

try:
    from hexbytes import HexBytes
except ImportError:
    HexBytes = None

try:
    import websockets
except ImportError:
    websockets = None

try:
    import psutil
except ImportError:
    psutil = None

# web3 7.x compatibility: ExtraDataToPOAMiddleware (renamed from geth_poa_middleware)
try:
    from web3.middleware import ExtraDataToPOAMiddleware as geth_poa_middleware
except ImportError:
    from web3.middleware.geth_poa import geth_poa_middleware

# --- Project imports ---
# Adjust sys.path for project root imports
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from config.loader import ConfigLoader, get_config, get_channel, get_env_var
from shared.serialization_utils import DecimalEncoder
from logging.logger_manager import (
    setup_module_logger,
    create_module_log_directories,
    log_data_entry,
    log_data_output,
)


# ============================================================================
# ENVIRONMENT & CONFIGURATION
# ============================================================================

load_dotenv()

# Initialize config
_config = get_config()
_chain_config = _config.get_chain_config(56)
_app_config = _config.get_app_config()
_timing_config = _config.get_timing_config()
_ws_config = _config.get_websocket_config()
_rate_limit_config = _config.get_rate_limit_config()

# Decimal precision (128 covers both V3 Q128.128 and V2 uint256)
DECIMAL_PRECISION = _app_config.get("precision", {}).get("decimal_precision", 128)
getcontext().prec = DECIMAL_PRECISION

# RPC URLs
BSC_HTTP_URL = get_env_var(
    "BSC_RPC_URL_HTTP",
    _chain_config.get("rpc", {}).get("http_url", "https://bsc-dataseed1.binance.org/"),
    str,
)
BSC_WS_URL = get_env_var(
    "BSC_RPC_URL_WS",
    _chain_config.get("rpc", {}).get("ws_url", "wss://bsc-ws-node.nariox.org:443"),
    str,
)

# Redis
_redis_channels_config = _config.get_redis_channels()
REDIS_URL = get_env_var(
    "REDIS_URL",
    _redis_channels_config.get("redis_url", "redis://localhost:6379/0"),
    str,
)

# Node Relay mode
USE_NODE_RELAY = os.getenv("USE_NODE_RELAY", "true").lower() == "true"
CHANNEL_NODE_BLOCKCHAIN_EVENTS = get_channel("node_blockchain_events")

# ============================================================================
# REDIS CHANNEL NAMES
# ============================================================================

CHANNEL_RAW_V2_EVENTS = get_channel("raw_v2_events")
CHANNEL_RAW_V3_EVENTS = get_channel("raw_v3_events")
CHANNEL_RAW_CURVE_EVENTS = get_channel("raw_curve_events")
CHANNEL_RAW_WOMBAT_EVENTS = get_channel("raw_wombat_events")
CHANNEL_RAW_DODO_EVENTS = get_channel("raw_dodo_events")
CHANNEL_RAW_AAVE_EVENTS = get_channel("raw_aave_events")
CHANNEL_GAS_PRICE_UPDATES = get_channel("gas_price_updates")
CHANNEL_READINESS_SIGNALS = get_channel("readiness_signals")
CHANNEL_SYSTEM_CONTROL = get_channel("system_control")
SIGNAL_EVENT_STREAMER_READY = _config.get_signal_value("event_streamer_ready")

# ============================================================================
# WEBSOCKET CONFIGURATION
# ============================================================================

_ws_conn_config = _ws_config.get("connection", {})
MAX_WS_CONNECTION_ATTEMPTS = _ws_conn_config.get("max_connection_attempts", 10)
WS_PING_INTERVAL = _ws_conn_config.get("ping_interval_seconds", 20)
WS_PING_TIMEOUT = _ws_conn_config.get("ping_timeout_seconds", 30)
WS_CLOSE_TIMEOUT = _ws_conn_config.get("close_timeout_seconds", 10)

_ws_timeout_config = _ws_config.get("timeouts", {})
WS_SUBSCRIPTION_TIMEOUT = _ws_timeout_config.get("subscription_response_timeout_seconds", 15.0)
WS_MESSAGE_TIMEOUT = _ws_timeout_config.get("message_receive_timeout_seconds", 60.0)

_ws_reconnect_config = _ws_config.get("reconnection", {})
WS_RECONNECT_BASE_DELAY = _ws_reconnect_config.get("base_delay_seconds", 2)
WS_RECONNECT_MAX_DELAY = _ws_reconnect_config.get("max_delay_seconds", 60)
WS_JITTER_MAX = _ws_reconnect_config.get("jitter_max_seconds", 1.0)

# Error recovery
_error_config = _timing_config.get("error_recovery", {})
INNER_LOOP_ERROR_DELAY = _error_config.get("inner_loop_error_delay_seconds", 1)

# ============================================================================
# LOGGING SETUP
# ============================================================================

create_module_log_directories()

stream_logger = setup_module_logger(
    "bsc_event_streamer", "stream_info.log", module_folder="Event_Streamer_Logs"
)
redis_logger = setup_module_logger(
    "redis_publisher", "redis_publish.log", module_folder="Event_Streamer_Logs"
)
ws_logger = setup_module_logger(
    "websocket", "websocket.log", module_folder="Event_Streamer_Logs"
)
event_logger = setup_module_logger(
    "event_decoder", "event_decode.log", module_folder="Event_Streamer_Logs"
)

# Per-channel logger cache
_channel_logger_cache: Dict[str, logging.Logger] = {}


def _get_channel_logger(channel: str) -> logging.Logger:
    if channel not in _channel_logger_cache:
        safe = channel.replace(":", "_")
        _channel_logger_cache[channel] = setup_module_logger(
            f"redis_{safe}", f"redis_{safe}.log", module_folder="Event_Streamer_Logs"
        )
    return _channel_logger_cache[channel]


# ============================================================================
# WEB3 SETUP
# ============================================================================

if not BSC_HTTP_URL:
    raise ValueError("BSC_RPC_URL_HTTP not set in environment or config.")
if not BSC_WS_URL:
    raise ValueError("BSC_RPC_URL_WS not set in environment or config.")

web3_http = Web3(Web3.HTTPProvider(BSC_HTTP_URL))
web3_http.middleware_onion.inject(geth_poa_middleware, layer=0)


# ============================================================================
# EVENT TOPIC SIGNATURES
# ============================================================================
# Pre-computed Keccak256 hashes of event signatures for O(1) event routing.
# These are the canonical topic0 values used by eth_subscribe log filters.

class EventTopics:
    """Canonical event topic hashes for all monitored protocols."""

    # --- Uniswap V2 / PancakeSwap V2 / BiSwap / SushiSwap ---
    # Swap(address,uint256,uint256,uint256,uint256,address)
    V2_SWAP = Web3.keccak(text="Swap(address,uint256,uint256,uint256,uint256,address)").hex()
    # Sync(uint112,uint112)
    V2_SYNC = Web3.keccak(text="Sync(uint112,uint112)").hex()

    # --- Uniswap V3 / PancakeSwap V3 / Thena FUSION ---
    # Swap(address,address,int256,int256,uint160,uint128,int24)
    V3_SWAP = Web3.keccak(text="Swap(address,address,int256,int256,uint160,uint128,int24)").hex()
    # Mint(address,address,int24,int24,uint128,uint256,uint256)
    V3_MINT = Web3.keccak(text="Mint(address,address,int24,int24,uint128,uint256,uint256)").hex()
    # Burn(address,int24,int24,uint128,uint256,uint256)
    V3_BURN = Web3.keccak(text="Burn(address,int24,int24,uint128,uint256,uint256)").hex()

    # --- Ellipsis / Curve StableSwap ---
    # TokenExchange(address,int128,uint256,int128,uint256)
    CURVE_TOKEN_EXCHANGE = Web3.keccak(
        text="TokenExchange(address,int128,uint256,int128,uint256)"
    ).hex()
    # TokenExchangeUnderlying(address,int128,uint256,int128,uint256)
    CURVE_TOKEN_EXCHANGE_UNDERLYING = Web3.keccak(
        text="TokenExchangeUnderlying(address,int128,uint256,int128,uint256)"
    ).hex()

    # --- Wombat Exchange ---
    # Swap(address,address,address,uint256,uint256,address)
    WOMBAT_SWAP = Web3.keccak(
        text="Swap(address,address,address,uint256,uint256,address)"
    ).hex()

    # --- DODO ---
    # DODOSwap(address,address,uint256,uint256,address,address)
    DODO_SWAP = Web3.keccak(
        text="DODOSwap(address,address,uint256,uint256,address,address)"
    ).hex()

    # --- Aave V3 ---
    # Supply(address,address,address,uint256,uint16)
    AAVE_SUPPLY = Web3.keccak(text="Supply(address,address,address,uint256,uint16)").hex()
    # Borrow(address,address,address,uint256,uint8,uint256,uint16)
    AAVE_BORROW = Web3.keccak(
        text="Borrow(address,address,address,uint256,uint8,uint256,uint16)"
    ).hex()
    # Repay(address,address,address,uint256,bool)
    AAVE_REPAY = Web3.keccak(text="Repay(address,address,address,uint256,bool)").hex()
    # FlashLoan(address,address,address,uint256,uint8,uint256,uint16)
    AAVE_FLASH_LOAN = Web3.keccak(
        text="FlashLoan(address,address,address,uint256,uint8,uint256,uint16)"
    ).hex()
    # LiquidationCall(address,address,address,uint256,uint256,address,bool)
    AAVE_LIQUIDATION_CALL = Web3.keccak(
        text="LiquidationCall(address,address,address,uint256,uint256,address,bool)"
    ).hex()
    # ReserveDataUpdated(address,uint256,uint256,uint256,uint256,uint256)
    AAVE_RESERVE_DATA_UPDATED = Web3.keccak(
        text="ReserveDataUpdated(address,uint256,uint256,uint256,uint256,uint256)"
    ).hex()


# Mapping from topic hash -> (event_name, protocol, abi_types_indexed, abi_types_data)
# This enables O(1) event identification and routing.
EVENT_TOPIC_MAP: Dict[str, Tuple[str, str, List[str], List[str]]] = {
    # V2 events
    EventTopics.V2_SWAP: (
        "V2_SWAP", "v2",
        ["address"],  # indexed: sender
        ["uint256", "uint256", "uint256", "uint256"],  # amount0In, amount1In, amount0Out, amount1Out
        # Note: 'to' is indexed as topic[2] in V2 Swap
    ),
    EventTopics.V2_SYNC: (
        "V2_SYNC", "v2",
        [],  # no indexed params
        ["uint112", "uint112"],  # reserve0, reserve1
    ),
    # V3 events
    EventTopics.V3_SWAP: (
        "V3_SWAP", "v3",
        ["address", "address"],  # indexed: sender, recipient
        ["int256", "int256", "uint160", "uint128", "int24"],  # amount0, amount1, sqrtPriceX96, liquidity, tick
    ),
    EventTopics.V3_MINT: (
        "V3_MINT", "v3",
        ["address"],  # indexed: owner  (NOTE: second 'address' in sig is non-indexed 'sender' param)
        ["address", "int24", "int24", "uint128", "uint256", "uint256"],  # sender, tickLower, tickUpper, amount, amount0, amount1
    ),
    EventTopics.V3_BURN: (
        "V3_BURN", "v3",
        ["address"],  # indexed: owner
        ["int24", "int24", "uint128", "uint256", "uint256"],  # tickLower, tickUpper, amount, amount0, amount1
    ),
    # Curve/Ellipsis events
    EventTopics.CURVE_TOKEN_EXCHANGE: (
        "CURVE_TOKEN_EXCHANGE", "curve",
        ["address"],  # indexed: buyer
        ["int128", "uint256", "int128", "uint256"],  # sold_id, tokens_sold, bought_id, tokens_bought
    ),
    EventTopics.CURVE_TOKEN_EXCHANGE_UNDERLYING: (
        "CURVE_TOKEN_EXCHANGE_UNDERLYING", "curve",
        ["address"],  # indexed: buyer
        ["int128", "uint256", "int128", "uint256"],  # sold_id, tokens_sold, bought_id, tokens_bought
    ),
    # Wombat events
    EventTopics.WOMBAT_SWAP: (
        "WOMBAT_SWAP", "wombat",
        ["address", "address", "address"],  # indexed: sender, fromToken, toToken
        ["uint256", "uint256"],  # fromAmount, toAmount
        # Note: 'to' is indexed as topic[4] but Wombat uses 3 indexed
    ),
    # DODO events
    EventTopics.DODO_SWAP: (
        "DODO_SWAP", "dodo",
        ["address", "address"],  # indexed: fromToken, toToken
        ["uint256", "uint256"],  # fromAmount, toAmount
        # Note: sender and receiver are also in data for some DODO versions
    ),
    # Aave V3 events
    EventTopics.AAVE_SUPPLY: (
        "AAVE_SUPPLY", "aave",
        ["address", "address"],  # indexed: reserve, onBehalfOf
        ["address", "uint256", "uint16"],  # user, amount, referralCode
    ),
    EventTopics.AAVE_BORROW: (
        "AAVE_BORROW", "aave",
        ["address", "address"],  # indexed: reserve, onBehalfOf
        ["address", "uint256", "uint8", "uint256", "uint16"],  # user, amount, interestRateMode, borrowRate, referralCode
    ),
    EventTopics.AAVE_REPAY: (
        "AAVE_REPAY", "aave",
        ["address", "address"],  # indexed: reserve, user
        ["address", "uint256", "bool"],  # repayer, amount, useATokens
    ),
    EventTopics.AAVE_FLASH_LOAN: (
        "AAVE_FLASH_LOAN", "aave",
        ["address", "address"],  # indexed: target, asset
        ["address", "uint256", "uint8", "uint256", "uint16"],  # initiator, amount, interestRateMode, premium, referralCode
    ),
    EventTopics.AAVE_LIQUIDATION_CALL: (
        "AAVE_LIQUIDATION_CALL", "aave",
        ["address", "address", "address"],  # indexed: collateralAsset, debtAsset, user
        ["uint256", "uint256", "address", "bool"],  # debtToCover, liquidatedCollateral, liquidator, receiveAToken
    ),
    EventTopics.AAVE_RESERVE_DATA_UPDATED: (
        "AAVE_RESERVE_DATA_UPDATED", "aave",
        ["address"],  # indexed: reserve
        ["uint256", "uint256", "uint256", "uint256", "uint256"],  # liquidityRate, stableBorrowRate, variableBorrowRate, liquidityIndex, variableBorrowIndex
    ),
}

# Build topic hash -> protocol routing map for channel selection
PROTOCOL_TO_CHANNEL: Dict[str, str] = {
    "v2": CHANNEL_RAW_V2_EVENTS,
    "v3": CHANNEL_RAW_V3_EVENTS,
    "curve": CHANNEL_RAW_CURVE_EVENTS,
    "wombat": CHANNEL_RAW_WOMBAT_EVENTS,
    "dodo": CHANNEL_RAW_DODO_EVENTS,
    "aave": CHANNEL_RAW_AAVE_EVENTS,
}

# All topic hashes for the eth_subscribe filter
ALL_MONITORED_TOPICS: List[str] = list(EVENT_TOPIC_MAP.keys())


# ============================================================================
# MONITORED CONTRACTS
# ============================================================================

class MonitoredContracts:
    """
    Registry of all contract addresses the streamer subscribes to.

    On startup, this is populated from config and/or database with all
    pool addresses for each protocol, plus the Aave V3 Pool contract.
    """

    def __init__(self):
        # Map: address_lower -> protocol_name
        self._address_to_protocol: Dict[str, str] = {}
        # Map: address_lower -> metadata dict
        self._address_to_metadata: Dict[str, Dict[str, Any]] = {}
        # Set of all monitored addresses (for O(1) membership test)
        self._monitored_set: Set[str] = set()

    def register(self, address: str, protocol: str, metadata: Optional[Dict] = None) -> None:
        """Register a contract address for monitoring."""
        addr = address.lower()
        self._address_to_protocol[addr] = protocol
        self._address_to_metadata[addr] = metadata or {}
        self._monitored_set.add(addr)

    def is_monitored(self, address: str) -> bool:
        """Check if address is being monitored (O(1))."""
        return address.lower() in self._monitored_set

    def get_protocol(self, address: str) -> Optional[str]:
        """Get protocol name for a monitored address."""
        return self._address_to_protocol.get(address.lower())

    def get_metadata(self, address: str) -> Dict[str, Any]:
        """Get metadata for a monitored address."""
        return self._address_to_metadata.get(address.lower(), {})

    def get_all_addresses(self) -> List[str]:
        """Get list of all monitored addresses."""
        return list(self._monitored_set)

    def count(self) -> int:
        """Total number of monitored contracts."""
        return len(self._monitored_set)

    def count_by_protocol(self) -> Dict[str, int]:
        """Count contracts per protocol."""
        counts: Dict[str, int] = {}
        for proto in self._address_to_protocol.values():
            counts[proto] = counts.get(proto, 0) + 1
        return counts


# Global registry instance
contracts = MonitoredContracts()


# ============================================================================
# GLOBAL STATE
# ============================================================================

# Redis client (initialized async in worker)
redis_client: Optional[redis.Redis] = None

# Shutdown coordination
shutdown_event = asyncio.Event()

# Block synchronization
_startup_block_number: Optional[int] = None
_latest_processed_block: int = 0

# Block timestamp cache (O(1) lookup, capped at 500 entries)
_block_timestamp_cache: OrderedDict = OrderedDict()
MAX_TIMESTAMP_CACHE_SIZE = 500

# Trace ID counter (monotonic, used as simple trace_id for low overhead)
_trace_counter: int = 0


def _generate_trace_id() -> str:
    """Generate a lightweight trace ID: timestamp_ms-BSE-counter."""
    global _trace_counter
    _trace_counter += 1
    return f"{int(time.time() * 1000)}-BSE-{_trace_counter:08d}"


# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================

def _handle_shutdown_signal(sig, frame):
    """Signal handler for graceful shutdown."""
    print(f"[BSC_EVENT_STREAMER] Received signal {signal.Signals(sig).name}. Initiating shutdown...")
    stream_logger.warning(f"Received shutdown signal ({signal.Signals(sig).name}).")
    shutdown_event.set()


signal.signal(signal.SIGINT, _handle_shutdown_signal)
signal.signal(signal.SIGTERM, _handle_shutdown_signal)


# ============================================================================
# CPU BINDING
# ============================================================================

def bind_to_core(core_id: int) -> None:
    """Pin the current process to a specific CPU core for cache locality."""
    if psutil is None:
        print(f"[CPU_BIND_WARN] psutil not installed. Cannot bind to core {core_id}.")
        return
    try:
        p = psutil.Process()
        available = p.cpu_affinity()
        if available is not None and core_id in available:
            p.cpu_affinity([core_id])
            print(f"[CPU_BIND] Process {p.pid} bound to core {core_id}.")
        else:
            print(f"[CPU_BIND_WARN] Core {core_id} not available. Available: {available}")
    except Exception as e:
        print(f"[CPU_BIND_ERROR] Could not bind to core {core_id}: {e}")


# ============================================================================
# EVENT DECODING
# ============================================================================

def decode_event_from_log(log_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Decode a raw Ethereum log entry into a structured event dict.

    Uses the pre-computed EVENT_TOPIC_MAP for O(1) event identification.
    Decodes indexed topics and ABI-encoded data fields.

    Args:
        log_data: Raw log dict with 'topics', 'data', 'address', 'blockNumber', 'logIndex'

    Returns:
        Decoded event dict or None if event not recognized/decodable.
    """
    topics = log_data.get("topics", [])
    if not topics:
        return None

    # Normalize topic0 to hex string
    topic0 = topics[0]
    if isinstance(topic0, bytes):
        topic0 = "0x" + topic0.hex()
    elif isinstance(topic0, str) and not topic0.startswith("0x"):
        topic0 = "0x" + topic0
    topic0 = topic0.lower()

    # O(1) lookup
    event_info = EVENT_TOPIC_MAP.get(topic0)
    if event_info is None:
        return None

    event_name, protocol, indexed_types, data_types = event_info

    # Extract contract address
    address = log_data.get("address", "")
    if isinstance(address, bytes):
        address = "0x" + address.hex()
    address = address.lower()

    # Extract block number
    block_number = log_data.get("blockNumber", 0)
    if isinstance(block_number, str):
        block_number = int(block_number, 16) if block_number.startswith("0x") else int(block_number)

    # Extract log index
    log_index = log_data.get("logIndex", 0)
    if isinstance(log_index, str):
        log_index = int(log_index, 16) if log_index.startswith("0x") else int(log_index)

    # Extract transaction hash
    tx_hash = log_data.get("transactionHash", "")
    if isinstance(tx_hash, bytes):
        tx_hash = "0x" + tx_hash.hex()

    # Decode indexed parameters from topics[1:]
    indexed_values = []
    for i, idx_type in enumerate(indexed_types):
        if i + 1 < len(topics):
            raw_topic = topics[i + 1]
            if isinstance(raw_topic, str):
                raw_topic = bytes.fromhex(raw_topic.replace("0x", ""))
            try:
                decoded = abi_decode([idx_type], raw_topic)
                indexed_values.append(decoded[0])
            except Exception:
                indexed_values.append(raw_topic)
        else:
            indexed_values.append(None)

    # Decode non-indexed data
    raw_data = log_data.get("data", "0x")
    if isinstance(raw_data, str):
        raw_data = bytes.fromhex(raw_data.replace("0x", ""))

    data_values = []
    if data_types and len(raw_data) > 0:
        try:
            data_values = list(abi_decode(data_types, raw_data))
        except Exception as e:
            event_logger.warning(
                f"Failed to decode data for {event_name} at {address} block {block_number}: {e}"
            )
            return None

    return {
        "event_name": event_name,
        "protocol": protocol,
        "address": address,
        "block_number": block_number,
        "log_index": log_index,
        "tx_hash": tx_hash,
        "indexed_values": indexed_values,
        "data_values": data_values,
        "raw_topics": topics,
        "raw_data": log_data.get("data", "0x"),
    }


# ============================================================================
# EVENT TO PAYLOAD BUILDERS
# ============================================================================
# Each builder converts a decoded event into a Redis-publishable payload dict.
# All numeric values are converted to strings to preserve precision.

def _build_v2_swap_payload(event: Dict, metadata: Dict) -> Dict[str, Any]:
    """Build payload for V2 Swap event."""
    # data_values: [amount0In, amount1In, amount0Out, amount1Out]
    dv = event["data_values"]
    return {
        "event_type": "V2_SWAP",
        "pool_address": event["address"],
        "block_number": event["block_number"],
        "log_index": event["log_index"],
        "tx_hash": event["tx_hash"],
        "sender": _addr_str(event["indexed_values"][0]) if event["indexed_values"] else "",
        "amount0_in": str(dv[0]) if len(dv) > 0 else "0",
        "amount1_in": str(dv[1]) if len(dv) > 1 else "0",
        "amount0_out": str(dv[2]) if len(dv) > 2 else "0",
        "amount1_out": str(dv[3]) if len(dv) > 3 else "0",
        "metadata": metadata,
    }


def _build_v2_sync_payload(event: Dict, metadata: Dict) -> Dict[str, Any]:
    """Build payload for V2 Sync event."""
    # data_values: [reserve0, reserve1]
    dv = event["data_values"]
    return {
        "event_type": "V2_SYNC",
        "pool_address": event["address"],
        "block_number": event["block_number"],
        "log_index": event["log_index"],
        "tx_hash": event["tx_hash"],
        "reserve0": str(dv[0]) if len(dv) > 0 else "0",
        "reserve1": str(dv[1]) if len(dv) > 1 else "0",
        "metadata": metadata,
    }


def _build_v3_swap_payload(event: Dict, metadata: Dict) -> Dict[str, Any]:
    """Build payload for V3 Swap event."""
    # data_values: [amount0, amount1, sqrtPriceX96, liquidity, tick]
    dv = event["data_values"]
    return {
        "event_type": "V3_SWAP",
        "pool_address": event["address"],
        "block_number": event["block_number"],
        "log_index": event["log_index"],
        "tx_hash": event["tx_hash"],
        "sender": _addr_str(event["indexed_values"][0]) if len(event["indexed_values"]) > 0 else "",
        "recipient": _addr_str(event["indexed_values"][1]) if len(event["indexed_values"]) > 1 else "",
        "amount0": str(dv[0]) if len(dv) > 0 else "0",
        "amount1": str(dv[1]) if len(dv) > 1 else "0",
        "sqrt_price_x96": str(dv[2]) if len(dv) > 2 else "0",
        "liquidity": str(dv[3]) if len(dv) > 3 else "0",
        "tick": str(dv[4]) if len(dv) > 4 else "0",
        "metadata": metadata,
    }


def _build_v3_mint_payload(event: Dict, metadata: Dict) -> Dict[str, Any]:
    """Build payload for V3 Mint event."""
    # data_values: [sender, tickLower, tickUpper, amount, amount0, amount1]
    dv = event["data_values"]
    return {
        "event_type": "V3_MINT",
        "pool_address": event["address"],
        "block_number": event["block_number"],
        "log_index": event["log_index"],
        "tx_hash": event["tx_hash"],
        "owner": _addr_str(event["indexed_values"][0]) if event["indexed_values"] else "",
        "sender": _addr_str(dv[0]) if len(dv) > 0 else "",
        "tick_lower": str(dv[1]) if len(dv) > 1 else "0",
        "tick_upper": str(dv[2]) if len(dv) > 2 else "0",
        "liquidity_delta": str(dv[3]) if len(dv) > 3 else "0",
        "amount0": str(dv[4]) if len(dv) > 4 else "0",
        "amount1": str(dv[5]) if len(dv) > 5 else "0",
        "metadata": metadata,
    }


def _build_v3_burn_payload(event: Dict, metadata: Dict) -> Dict[str, Any]:
    """Build payload for V3 Burn event."""
    # data_values: [tickLower, tickUpper, amount, amount0, amount1]
    dv = event["data_values"]
    return {
        "event_type": "V3_BURN",
        "pool_address": event["address"],
        "block_number": event["block_number"],
        "log_index": event["log_index"],
        "tx_hash": event["tx_hash"],
        "owner": _addr_str(event["indexed_values"][0]) if event["indexed_values"] else "",
        "tick_lower": str(dv[0]) if len(dv) > 0 else "0",
        "tick_upper": str(dv[1]) if len(dv) > 1 else "0",
        "liquidity_delta": str(-abs(int(dv[2]))) if len(dv) > 2 else "0",  # Negative for burns
        "amount0": str(dv[3]) if len(dv) > 3 else "0",
        "amount1": str(dv[4]) if len(dv) > 4 else "0",
        "metadata": metadata,
    }


def _build_curve_exchange_payload(event: Dict, metadata: Dict) -> Dict[str, Any]:
    """Build payload for Curve/Ellipsis TokenExchange event."""
    # data_values: [sold_id, tokens_sold, bought_id, tokens_bought]
    dv = event["data_values"]
    return {
        "event_type": event["event_name"],  # CURVE_TOKEN_EXCHANGE or CURVE_TOKEN_EXCHANGE_UNDERLYING
        "pool_address": event["address"],
        "block_number": event["block_number"],
        "log_index": event["log_index"],
        "tx_hash": event["tx_hash"],
        "buyer": _addr_str(event["indexed_values"][0]) if event["indexed_values"] else "",
        "sold_id": str(dv[0]) if len(dv) > 0 else "0",
        "tokens_sold": str(dv[1]) if len(dv) > 1 else "0",
        "bought_id": str(dv[2]) if len(dv) > 2 else "0",
        "tokens_bought": str(dv[3]) if len(dv) > 3 else "0",
        "metadata": metadata,
    }


def _build_wombat_swap_payload(event: Dict, metadata: Dict) -> Dict[str, Any]:
    """Build payload for Wombat Swap event."""
    # indexed: [sender, fromToken, toToken], data: [fromAmount, toAmount]
    iv = event["indexed_values"]
    dv = event["data_values"]
    return {
        "event_type": "WOMBAT_SWAP",
        "pool_address": event["address"],
        "block_number": event["block_number"],
        "log_index": event["log_index"],
        "tx_hash": event["tx_hash"],
        "sender": _addr_str(iv[0]) if len(iv) > 0 else "",
        "from_token": _addr_str(iv[1]) if len(iv) > 1 else "",
        "to_token": _addr_str(iv[2]) if len(iv) > 2 else "",
        "from_amount": str(dv[0]) if len(dv) > 0 else "0",
        "to_amount": str(dv[1]) if len(dv) > 1 else "0",
        "metadata": metadata,
    }


def _build_dodo_swap_payload(event: Dict, metadata: Dict) -> Dict[str, Any]:
    """Build payload for DODO DODOSwap event."""
    # indexed: [fromToken, toToken], data: [fromAmount, toAmount]
    iv = event["indexed_values"]
    dv = event["data_values"]
    return {
        "event_type": "DODO_SWAP",
        "pool_address": event["address"],
        "block_number": event["block_number"],
        "log_index": event["log_index"],
        "tx_hash": event["tx_hash"],
        "from_token": _addr_str(iv[0]) if len(iv) > 0 else "",
        "to_token": _addr_str(iv[1]) if len(iv) > 1 else "",
        "from_amount": str(dv[0]) if len(dv) > 0 else "0",
        "to_amount": str(dv[1]) if len(dv) > 1 else "0",
        "metadata": metadata,
    }


def _build_aave_event_payload(event: Dict, metadata: Dict) -> Dict[str, Any]:
    """
    Build payload for any Aave V3 event.

    Since Aave events have varied structures, we include all decoded values
    generically and let the aave_state_manager interpret them.
    """
    iv = event["indexed_values"]
    dv = event["data_values"]
    return {
        "event_type": event["event_name"],
        "pool_address": event["address"],
        "block_number": event["block_number"],
        "log_index": event["log_index"],
        "tx_hash": event["tx_hash"],
        "indexed_values": [_format_value(v) for v in iv],
        "data_values": [_format_value(v) for v in dv],
        "metadata": metadata,
    }


def _addr_str(value: Any) -> str:
    """Convert an address value to checksum string."""
    if isinstance(value, bytes):
        return Web3.to_checksum_address("0x" + value.hex()[-40:])
    if isinstance(value, int):
        return Web3.to_checksum_address("0x" + hex(value)[2:].zfill(40))
    if isinstance(value, str):
        return Web3.to_checksum_address(value) if len(value) >= 40 else value
    return str(value)


def _format_value(value: Any) -> str:
    """Format a decoded ABI value to string for JSON serialization."""
    if isinstance(value, bytes):
        return "0x" + value.hex()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


# Dispatch table: event_name -> builder function
_PAYLOAD_BUILDERS: Dict[str, Any] = {
    "V2_SWAP": _build_v2_swap_payload,
    "V2_SYNC": _build_v2_sync_payload,
    "V3_SWAP": _build_v3_swap_payload,
    "V3_MINT": _build_v3_mint_payload,
    "V3_BURN": _build_v3_burn_payload,
    "CURVE_TOKEN_EXCHANGE": _build_curve_exchange_payload,
    "CURVE_TOKEN_EXCHANGE_UNDERLYING": _build_curve_exchange_payload,
    "WOMBAT_SWAP": _build_wombat_swap_payload,
    "DODO_SWAP": _build_dodo_swap_payload,
    "AAVE_SUPPLY": _build_aave_event_payload,
    "AAVE_BORROW": _build_aave_event_payload,
    "AAVE_REPAY": _build_aave_event_payload,
    "AAVE_FLASH_LOAN": _build_aave_event_payload,
    "AAVE_LIQUIDATION_CALL": _build_aave_event_payload,
    "AAVE_RESERVE_DATA_UPDATED": _build_aave_event_payload,
}


# ============================================================================
# REDIS PUBLISHING
# ============================================================================

async def publish_event(channel: str, payload: Dict[str, Any], trace_id: str) -> None:
    """
    Publish a decoded event to the appropriate Redis channel.

    All payloads are JSON-serialized with DecimalEncoder to handle large integers
    and Decimal values safely.

    Args:
        channel: Redis channel to publish to.
        payload: Event payload dict.
        trace_id: Trace ID for correlation.
    """
    global redis_client
    if redis_client is None:
        return

    # Add trace metadata
    payload["_trace_id"] = trace_id
    payload["_timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["_source"] = "bsc_event_streamer"

    try:
        payload_str = json.dumps(payload, cls=DecimalEncoder)
        await redis_client.publish(channel, payload_str)
        _get_channel_logger(channel).info(payload_str)
    except Exception as e:
        redis_logger.error(f"Failed to publish to {channel}: {e}")


async def publish_gas_from_block(block_number: int) -> None:
    """
    Extract and publish gas price information from a block header.

    BSC has baseFeePerGas since the Fermi hardfork (0.45s blocks).
    This provides the gas oracle with per-block gas pricing data.

    Args:
        block_number: Block number to fetch gas data from.
    """
    global redis_client
    if redis_client is None:
        return

    try:
        block = await asyncio.to_thread(
            web3_http.eth.get_block, block_number
        )
        gas_data = {
            "block_number": block_number,
            "timestamp": block.get("timestamp", 0),
            "gas_used": block.get("gasUsed", 0),
            "gas_limit": block.get("gasLimit", 0),
            "base_fee_per_gas": str(block.get("baseFeePerGas", 0)),
            "_source": "bsc_event_streamer",
            "_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        payload_str = json.dumps(gas_data, cls=DecimalEncoder)
        await redis_client.publish(CHANNEL_GAS_PRICE_UPDATES, payload_str)
    except Exception as e:
        stream_logger.warning(f"Failed to fetch gas data for block {block_number}: {e}")


# ============================================================================
# WEB3 CONNECTION MANAGEMENT
# ============================================================================

async def ensure_web3_connected() -> int:
    """
    Verify HTTP provider is connected and return current block number.

    Retries with exponential backoff per timing config.

    Returns:
        Current block number on BSC.
    """
    max_retries = _timing_config.get("web3_connection", {}).get("max_retries", 5)
    retry_delay = _timing_config.get("web3_connection", {}).get("retry_delay_seconds", 2.0)

    for attempt in range(max_retries):
        try:
            block_num = await asyncio.to_thread(web3_http.eth.get_block_number)
            stream_logger.info(f"Web3 HTTP connected. Current block: {block_num}")
            return block_num
        except Exception as e:
            stream_logger.warning(
                f"Web3 HTTP connection attempt {attempt + 1}/{max_retries} failed: {e}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))

    raise ConnectionError("Failed to connect to BSC HTTP provider after retries.")


# ============================================================================
# BLOCK TIMESTAMP CACHE
# ============================================================================

async def get_block_timestamp(block_number: int) -> str:
    """
    Get ISO timestamp for a block number (cached).

    Uses an OrderedDict cache capped at MAX_TIMESTAMP_CACHE_SIZE entries.

    Args:
        block_number: BSC block number.

    Returns:
        ISO format timestamp string.
    """
    if block_number in _block_timestamp_cache:
        return _block_timestamp_cache[block_number]

    try:
        block = await asyncio.to_thread(web3_http.eth.get_block, block_number)
        ts = block.get("timestamp", 0)
        iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        # Cache with LRU eviction
        _block_timestamp_cache[block_number] = iso
        if len(_block_timestamp_cache) > MAX_TIMESTAMP_CACHE_SIZE:
            _block_timestamp_cache.popitem(last=False)

        return iso
    except Exception as e:
        stream_logger.warning(f"Failed to get timestamp for block {block_number}: {e}")
        return datetime.now(timezone.utc).isoformat()


# ============================================================================
# CONTRACT REGISTRY LOADER
# ============================================================================

def load_monitored_contracts_from_config() -> None:
    """
    Load monitored contract addresses from chain config.

    Registers:
    - All DEX pool addresses from config/chains/56.json
    - Aave V3 Pool contract address
    - Additional pools can be added via config/pools/*.json files

    For a production deployment, this would also load from a database of
    discovered pools (similar to ArbitrageTestBot's token_pairs.db).
    """
    # Register Aave V3 Pool
    aave_pool = _chain_config.get("contracts", {}).get("aave_v3_pool", "")
    if aave_pool:
        contracts.register(aave_pool, "aave", {"name": "Aave V3 Pool"})
        stream_logger.info(f"Registered Aave V3 Pool: {aave_pool}")

    # Load additional pool lists from config/pools/ directory (if exists)
    pools_dir = _config._config_dir / "pools"
    if pools_dir.exists():
        for pool_file in pools_dir.glob("*.json"):
            try:
                with open(pool_file) as f:
                    pool_list = json.load(f)
                protocol = pool_file.stem  # e.g., "v2_pools" -> "v2_pools"
                if isinstance(pool_list, list):
                    for pool in pool_list:
                        addr = pool.get("address", "")
                        proto = pool.get("protocol", protocol)
                        if addr:
                            contracts.register(addr, proto, pool)
                stream_logger.info(f"Loaded {len(pool_list)} pools from {pool_file.name}")
            except Exception as e:
                stream_logger.warning(f"Failed to load pool file {pool_file}: {e}")

    stream_logger.info(
        f"Total monitored contracts: {contracts.count()} "
        f"(by protocol: {contracts.count_by_protocol()})"
    )


# ============================================================================
# MAIN EVENT PROCESSING
# ============================================================================

async def process_raw_log(log_data: Dict[str, Any]) -> None:
    """
    Process a single raw log event from the blockchain.

    Pipeline:
    1. Decode event using O(1) topic hash lookup
    2. Check if contract is monitored
    3. Build protocol-specific payload
    4. Publish to appropriate Redis channel

    Args:
        log_data: Raw log dict from WebSocket or Node Relay.
    """
    global _latest_processed_block

    # Decode the event
    event = decode_event_from_log(log_data)
    if event is None:
        return  # Unrecognized event topic

    # Check if we're monitoring this contract
    address = event["address"]
    # For DEX events: check contract registry
    # For Aave events: always process (single known contract)
    if event["protocol"] != "aave" and not contracts.is_monitored(address):
        return

    # Stale event filtering: skip events from before startup
    if _startup_block_number is not None and event["block_number"] <= _startup_block_number:
        return

    # Update latest processed block
    if event["block_number"] > _latest_processed_block:
        # Publish gas data for new blocks (one per block)
        asyncio.create_task(publish_gas_from_block(event["block_number"]))
        _latest_processed_block = event["block_number"]

    # Get metadata for this contract
    metadata = contracts.get_metadata(address)

    # Build event-specific payload
    builder = _PAYLOAD_BUILDERS.get(event["event_name"])
    if builder is None:
        event_logger.warning(f"No payload builder for event: {event['event_name']}")
        return

    payload = builder(event, metadata)

    # Determine target Redis channel
    channel = PROTOCOL_TO_CHANNEL.get(event["protocol"])
    if channel is None:
        event_logger.warning(f"No channel for protocol: {event['protocol']}")
        return

    # Publish
    trace_id = _generate_trace_id()
    await publish_event(channel, payload, trace_id)


# ============================================================================
# NODE RELAY MODE (Subscribe to Redis instead of WebSocket)
# ============================================================================

async def _run_node_relay_mode() -> None:
    """
    Subscribe to Redis channel for blockchain events published by node_relay.py.

    This mode consolidates all WebSocket connections into node_relay.py,
    reducing connection count to 1 and avoiding RPC rate limiting.
    """
    global redis_client

    stream_logger.info("[NODE_RELAY] Starting Node Relay mode - subscribing to Redis channel")

    retry_count = 0
    max_retries = MAX_WS_CONNECTION_ATTEMPTS

    while not shutdown_event.is_set() and retry_count < max_retries:
        try:
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(CHANNEL_NODE_BLOCKCHAIN_EVENTS)
            stream_logger.info(f"[NODE_RELAY] Subscribed to {CHANNEL_NODE_BLOCKCHAIN_EVENTS}")
            retry_count = 0  # Reset on successful connection

            async for message in pubsub.listen():
                if shutdown_event.is_set():
                    break

                if message["type"] != "message":
                    continue

                try:
                    raw = message["data"]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    log_data = json.loads(raw)

                    # node_relay publishes individual log entries
                    # or batches - handle both
                    if isinstance(log_data, list):
                        for entry in log_data:
                            await process_raw_log(entry)
                    else:
                        await process_raw_log(log_data)

                except json.JSONDecodeError as e:
                    event_logger.warning(f"[NODE_RELAY] Invalid JSON: {e}")
                except Exception as e:
                    event_logger.error(f"[NODE_RELAY] Error processing message: {e}")
                    if INNER_LOOP_ERROR_DELAY > 0:
                        await asyncio.sleep(INNER_LOOP_ERROR_DELAY)

        except asyncio.CancelledError:
            stream_logger.info("[NODE_RELAY] Cancelled, shutting down.")
            break
        except RedisConnectionError as e:
            retry_count += 1
            delay = min(WS_RECONNECT_BASE_DELAY * (2 ** retry_count), WS_RECONNECT_MAX_DELAY)
            jitter = random.uniform(0, WS_JITTER_MAX)
            stream_logger.warning(
                f"[NODE_RELAY] Redis connection lost: {e}. "
                f"Retry {retry_count}/{max_retries} in {delay + jitter:.1f}s"
            )
            await asyncio.sleep(delay + jitter)
        except Exception as e:
            retry_count += 1
            delay = min(WS_RECONNECT_BASE_DELAY * (2 ** retry_count), WS_RECONNECT_MAX_DELAY)
            stream_logger.error(
                f"[NODE_RELAY] Unexpected error: {e}. "
                f"Retry {retry_count}/{max_retries} in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
        finally:
            try:
                await pubsub.unsubscribe(CHANNEL_NODE_BLOCKCHAIN_EVENTS)
                await pubsub.close()
            except Exception:
                pass


# ============================================================================
# WEBSOCKET MODE (Direct BSC node connection)
# ============================================================================

async def _run_websocket_mode() -> None:
    """
    Connect directly to BSC node via WebSocket and subscribe to log events.

    Uses eth_subscribe("logs", ...) with topic filters for all monitored
    event types across all registered contract addresses.

    Implements reconnection with exponential backoff and jitter.
    """
    if websockets is None:
        raise ImportError("websockets package required for WebSocket mode. Install: pip install websockets")

    retry_count = 0

    while not shutdown_event.is_set() and retry_count < MAX_WS_CONNECTION_ATTEMPTS:
        try:
            stream_logger.info(f"[WEBSOCKET] Connecting to {BSC_WS_URL}...")

            async with websockets.connect(
                BSC_WS_URL,
                ping_interval=WS_PING_INTERVAL,
                ping_timeout=WS_PING_TIMEOUT,
                close_timeout=WS_CLOSE_TIMEOUT,
                max_size=10 * 1024 * 1024,  # 10MB max message
            ) as ws:
                # Set TCP_NODELAY for lower latency (disable Nagle's algorithm)
                if hasattr(ws, "transport") and ws.transport is not None:
                    sock = ws.transport.get_extra_info("socket")
                    if sock is not None:
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                stream_logger.info("[WEBSOCKET] Connected. Sending eth_subscribe...")

                # Build subscription payload
                subscription_params = _build_subscription_params()
                await ws.send(json.dumps(subscription_params))

                # Wait for subscription confirmation
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=WS_SUBSCRIPTION_TIMEOUT)
                    response_data = json.loads(response)
                    if "error" in response_data:
                        stream_logger.error(f"[WEBSOCKET] Subscription error: {response_data['error']}")
                        retry_count += 1
                        continue
                    sub_id = response_data.get("result", "unknown")
                    stream_logger.info(f"[WEBSOCKET] Subscribed. Subscription ID: {sub_id}")
                except asyncio.TimeoutError:
                    stream_logger.error("[WEBSOCKET] Subscription response timed out.")
                    retry_count += 1
                    continue

                retry_count = 0  # Reset on successful subscription

                # Main message loop
                while not shutdown_event.is_set():
                    try:
                        raw_message = await asyncio.wait_for(
                            ws.recv(), timeout=WS_MESSAGE_TIMEOUT
                        )
                        message = json.loads(raw_message)

                        # eth_subscribe notifications have method "eth_subscription"
                        if message.get("method") != "eth_subscription":
                            continue

                        params = message.get("params", {})
                        log_data = params.get("result", {})

                        if not log_data:
                            continue

                        await process_raw_log(log_data)

                    except asyncio.TimeoutError:
                        # No message received within timeout - check connection
                        stream_logger.debug("[WEBSOCKET] No message received, checking connection...")
                        try:
                            pong_waiter = await ws.ping()
                            await asyncio.wait_for(pong_waiter, timeout=WS_PING_TIMEOUT)
                        except Exception:
                            stream_logger.warning("[WEBSOCKET] Ping failed, reconnecting...")
                            break

                    except websockets.exceptions.ConnectionClosed as e:
                        stream_logger.warning(f"[WEBSOCKET] Connection closed: {e}")
                        break

                    except json.JSONDecodeError as e:
                        event_logger.warning(f"[WEBSOCKET] Invalid JSON message: {e}")
                        continue

                    except Exception as e:
                        event_logger.error(f"[WEBSOCKET] Error in message loop: {e}")
                        if INNER_LOOP_ERROR_DELAY > 0:
                            await asyncio.sleep(INNER_LOOP_ERROR_DELAY)

        except asyncio.CancelledError:
            stream_logger.info("[WEBSOCKET] Cancelled, shutting down.")
            break
        except (ConnectionRefusedError, OSError) as e:
            retry_count += 1
            delay = min(
                WS_RECONNECT_BASE_DELAY * (2 ** retry_count) + random.uniform(0, WS_JITTER_MAX),
                WS_RECONNECT_MAX_DELAY,
            )
            stream_logger.warning(
                f"[WEBSOCKET] Connection refused: {e}. "
                f"Retry {retry_count}/{MAX_WS_CONNECTION_ATTEMPTS} in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
        except Exception as e:
            retry_count += 1
            delay = min(
                WS_RECONNECT_BASE_DELAY * (2 ** retry_count) + random.uniform(0, WS_JITTER_MAX),
                WS_RECONNECT_MAX_DELAY,
            )
            stream_logger.error(
                f"[WEBSOCKET] Unexpected error: {e}. "
                f"Retry {retry_count}/{MAX_WS_CONNECTION_ATTEMPTS} in {delay:.1f}s\n"
                f"{traceback.format_exc()}"
            )
            await asyncio.sleep(delay)

    if retry_count >= MAX_WS_CONNECTION_ATTEMPTS:
        stream_logger.critical(
            f"[WEBSOCKET] Exhausted {MAX_WS_CONNECTION_ATTEMPTS} connection attempts. Exiting."
        )


def _build_subscription_params() -> Dict[str, Any]:
    """
    Build the eth_subscribe JSON-RPC payload for log event subscription.

    Constructs a filter that watches all monitored contract addresses for
    all monitored event topic hashes.

    Returns:
        JSON-RPC request dict for eth_subscribe("logs", filter).
    """
    # Collect all monitored addresses
    addresses = contracts.get_all_addresses()

    # Use checksum addresses for the subscription
    checksum_addresses = []
    for addr in addresses:
        try:
            checksum_addresses.append(Web3.to_checksum_address(addr))
        except Exception:
            checksum_addresses.append(addr)

    # Build topics filter: topic[0] is the event signature
    # We want to match ANY of our monitored event topics
    topics_filter = [ALL_MONITORED_TOPICS]

    subscription = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_subscribe",
        "params": [
            "logs",
            {
                "address": checksum_addresses,
                "topics": topics_filter,
            },
        ],
    }

    stream_logger.info(
        f"[SUBSCRIPTION] Built filter: {len(checksum_addresses)} addresses, "
        f"{len(ALL_MONITORED_TOPICS)} event topics"
    )

    return subscription


# ============================================================================
# MAIN LISTENER DISPATCH
# ============================================================================

async def main_listener_loop() -> None:
    """
    Dispatch to either Node Relay or WebSocket mode based on configuration.
    """
    if USE_NODE_RELAY:
        stream_logger.info("Starting in NODE RELAY mode (subscribing to Redis)")
        await _run_node_relay_mode()
    else:
        stream_logger.info("Starting in WEBSOCKET mode (direct BSC node connection)")
        await _run_websocket_mode()


# ============================================================================
# WORKER ENTRY POINT
# ============================================================================

def bsc_event_streamer_worker(core_id: Optional[int] = None) -> None:
    """
    Main entry point when launched as a multiprocess worker from main.py.

    1. Binds to CPU core (if specified)
    2. Initializes Redis connection
    3. Loads monitored contracts from config
    4. Captures startup block number
    5. Starts main listener loop
    6. Signals readiness once connected and receiving events

    Args:
        core_id: CPU core to pin this process to (optional).
    """
    if core_id is not None:
        bind_to_core(core_id)

    async def _async_main():
        global redis_client, _startup_block_number

        # Initialize Redis
        try:
            redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
            await redis_client.ping()
            stream_logger.info(f"[INIT] Redis connected: {REDIS_URL}")
        except Exception as e:
            stream_logger.critical(f"[INIT] Redis connection failed: {e}")
            return

        # Load monitored contracts
        load_monitored_contracts_from_config()

        if contracts.count() == 0:
            stream_logger.warning(
                "[INIT] No contracts registered. The streamer will only capture "
                "events from contracts registered at runtime via config/pools/ directory."
            )

        # Capture startup block number BEFORE starting listener
        # This is CRITICAL for stale event filtering
        try:
            _startup_block_number = await ensure_web3_connected()
            stream_logger.info(f"[INIT] Startup block: {_startup_block_number}")
        except ConnectionError as e:
            stream_logger.critical(f"[INIT] Web3 connection failed: {e}")
            return

        # Signal readiness
        try:
            await redis_client.rpush(CHANNEL_READINESS_SIGNALS, SIGNAL_EVENT_STREAMER_READY)
            stream_logger.info(f"[INIT] Readiness signal sent: {SIGNAL_EVENT_STREAMER_READY}")
        except Exception as e:
            stream_logger.warning(f"[INIT] Failed to send readiness signal: {e}")

        # Start main listener loop
        try:
            await main_listener_loop()
        except asyncio.CancelledError:
            stream_logger.info("[SHUTDOWN] Main listener cancelled.")
        except Exception as e:
            stream_logger.critical(f"[FATAL] Unhandled exception in listener: {e}\n{traceback.format_exc()}")
        finally:
            # Cleanup
            if redis_client:
                try:
                    await redis_client.close()
                except Exception:
                    pass
            stream_logger.info("[SHUTDOWN] BSC Event Streamer shut down.")

    # Run the async main function
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        print("[BSC_EVENT_STREAMER] Interrupted by user.")


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"[BSC_EVENT_STREAMER] Starting (PID: {os.getpid()})...")
    print(f"[BSC_EVENT_STREAMER] Mode: {'NODE_RELAY' if USE_NODE_RELAY else 'WEBSOCKET'}")
    print(f"[BSC_EVENT_STREAMER] BSC HTTP: {BSC_HTTP_URL}")
    print(f"[BSC_EVENT_STREAMER] BSC WS: {BSC_WS_URL}")
    print(f"[BSC_EVENT_STREAMER] Redis: {REDIS_URL}")

    bsc_event_streamer_worker()
