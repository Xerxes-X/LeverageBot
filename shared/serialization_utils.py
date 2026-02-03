"""
Serialization utilities for BSC Leverage Bot.

Adapted from ArbitrageTestBot/serialization_utils.py (~269 lines).
Provides JSON encoding for Decimal, HexBytes, large integers, and web3 types.

Usage:
    from shared.serialization_utils import DecimalEncoder
    json.dumps(data, cls=DecimalEncoder)
"""

import json
from decimal import Decimal
from json import JSONEncoder
from typing import Any

try:
    from hexbytes import HexBytes
except ImportError:
    HexBytes = None


class DecimalEncoder(JSONEncoder):
    """
    Custom JSON encoder handling Decimal, HexBytes, large integers, and web3.py types.

    Sources:
    - RFC 7159 section 6 (JSON number limits)
    - IEEE 754-2008 (double precision safe integer limit: 2^53 - 1)
    """

    # IEEE 754 double precision safe integer limit
    _MAX_SAFE_INTEGER = 2**53 - 1

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        # Handle FastDecimal from gmpy2 wrapper
        if hasattr(obj, "_value") and type(obj).__name__ == "FastDecimal":
            return str(obj)
        # Handle HexBytes from web3.py (addresses, tx hashes, raw bytes)
        if HexBytes is not None and isinstance(obj, (HexBytes, bytes)):
            return obj.hex() if hasattr(obj, "hex") else obj.decode("utf-8", errors="replace")
        if isinstance(obj, bytes):
            return obj.hex()
        # Handle web3.py AttributeDict (common in transaction/block responses)
        if hasattr(obj, "__iter__") and hasattr(obj, "keys"):
            return dict(obj)
        return super().default(obj)

    def encode(self, obj: Any) -> str:
        """Override encode to convert large integers to strings before JSON serialization."""
        return super().encode(self._convert_large_ints(obj))

    def _convert_large_ints(self, obj: Any) -> Any:
        """
        Recursively convert integers exceeding IEEE 754 safe limits to strings.

        This preserves precision for values like sqrtPriceX96 (uint160), liquidity (uint128),
        and other EVM uint256 values that exceed JavaScript Number.MAX_SAFE_INTEGER.
        """
        if isinstance(obj, dict):
            return {k: self._convert_large_ints(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_large_ints(item) for item in obj]
        elif isinstance(obj, int) and (obj > self._MAX_SAFE_INTEGER or obj < -self._MAX_SAFE_INTEGER):
            return str(obj)
        return obj
