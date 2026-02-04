"""
Shared data types for BSC Leverage Bot.

Centralized dataclasses and enums used across all modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PositionDirection(Enum):
    LONG = "long"  # Collateral=volatile, Debt=stable
    SHORT = "short"  # Collateral=stable, Debt=volatile


class MarketRegime(Enum):
    TRENDING = "trending"  # Hurst > 0.55, ATR ratio 1.0-3.0x
    MEAN_REVERTING = "mean_reverting"  # Hurst < 0.45
    RANGING = "ranging"  # Hurst 0.45-0.55, ATR ratio < 1.0
    VOLATILE = "volatile"  # ATR ratio > 3.0


class HFTier(Enum):
    SAFE = "safe"  # HF > 2.0 -> poll every 15s
    WATCH = "watch"  # 1.5-2.0  -> poll every 5s
    WARNING = "warning"  # 1.3-1.5  -> poll every 2s
    CRITICAL = "critical"  # < 1.3    -> poll every 1s + Chainlink events


class PositionAction(Enum):
    OPEN = "open"
    CLOSE = "close"
    DELEVERAGE = "deleverage"
    INCREASE = "increase"


# ---------------------------------------------------------------------------
# Market Data Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OHLCV:
    timestamp: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True)
class Trade:
    """Individual trade from exchange (for VPIN computation)."""

    price: Decimal
    quantity: Decimal
    timestamp: int
    is_buyer_maker: bool  # True = sell-initiated (maker was buyer)


@dataclass(frozen=True)
class OrderBookSnapshot:
    bids: list[tuple[Decimal, Decimal]]  # (price, quantity) sorted desc
    asks: list[tuple[Decimal, Decimal]]  # (price, quantity) sorted asc
    timestamp: int


# ---------------------------------------------------------------------------
# Signal Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndicatorSnapshot:
    price: Decimal
    ema_20: Decimal
    ema_50: Decimal
    ema_200: Decimal
    rsi_14: Decimal
    macd_line: Decimal
    macd_signal: Decimal
    macd_histogram: Decimal
    bb_upper: Decimal
    bb_middle: Decimal
    bb_lower: Decimal
    atr_14: Decimal
    atr_ratio: Decimal  # ATR(14) / ATR_50_avg
    volume: Decimal
    volume_20_avg: Decimal
    hurst: Decimal  # Hurst exponent
    vpin: Decimal  # VPIN value
    obi: Decimal  # Order book imbalance [-1, 1]
    recent_prices: list[Decimal]  # For Hurst computation (last 200 closes)


@dataclass(frozen=True)
class SignalComponent:
    source: str  # e.g., "order_book_imbalance", "vpin", "technical"
    tier: int  # 1, 2, or 3
    direction: PositionDirection
    strength: Decimal  # -1.0 to 1.0
    weight: Decimal  # tier-dependent weight
    confidence: Decimal  # 0.0-1.0 self-assessed confidence
    data_age_seconds: int  # freshness of underlying data


@dataclass(frozen=True)
class TradeSignal:
    direction: PositionDirection
    confidence: Decimal  # 0.0-1.0 (ensemble confidence from all sources)
    strategy_mode: str  # 'momentum', 'mean_reversion', 'blended', 'manual'
    indicators: IndicatorSnapshot
    regime: MarketRegime
    components: list[SignalComponent]  # Contributing signal sources
    recommended_size_usd: Decimal  # Kelly-derived position size
    hurst_exponent: Decimal  # Current Hurst H value
    garch_volatility: Decimal  # GARCH(1,1) forecast
    timestamp: int


# ---------------------------------------------------------------------------
# Aave / Position Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserAccountData:
    total_collateral_usd: Decimal
    total_debt_usd: Decimal
    available_borrow_usd: Decimal
    current_liquidation_threshold: Decimal
    ltv: Decimal
    health_factor: Decimal


@dataclass(frozen=True)
class ReserveData:
    variable_borrow_rate: Decimal
    utilization_rate: Decimal
    isolation_mode_enabled: bool
    debt_ceiling: Decimal
    current_isolated_debt: Decimal


@dataclass
class PositionState:
    direction: PositionDirection
    debt_token: str
    collateral_token: str
    debt_usd: Decimal
    collateral_usd: Decimal
    initial_debt_usd: Decimal
    initial_collateral_usd: Decimal
    health_factor: Decimal
    borrow_rate_ray: Decimal
    liquidation_threshold: Decimal


@dataclass(frozen=True)
class HealthStatus:
    health_factor: Decimal
    tier: HFTier
    collateral_usd: Decimal
    debt_usd: Decimal
    timestamp: int


@dataclass(frozen=True)
class BorrowRateInfo:
    variable_rate_apr: Decimal
    utilization_rate: Decimal
    optimal_utilization: Decimal
    rate_at_kink: Decimal  # Rate if utilization were at optimal point


# ---------------------------------------------------------------------------
# Swap / Execution Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwapQuote:
    provider: str
    from_token: str
    to_token: str
    from_amount: Decimal
    to_amount: Decimal
    to_amount_min: Decimal
    calldata: bytes  # Raw calldata for aggregator router
    router_address: str  # Address to call() with calldata
    gas_estimate: int
    price_impact: Decimal


@dataclass(frozen=True)
class SafetyCheck:
    can_proceed: bool
    reason: str


# ---------------------------------------------------------------------------
# P&L Types
# ---------------------------------------------------------------------------


@dataclass
class RealizedPnL:
    gross_pnl_usd: Decimal
    accrued_interest_usd: Decimal
    gas_costs_usd: Decimal
    flash_loan_premiums_usd: Decimal
    net_pnl_usd: Decimal


@dataclass
class TradingStats:
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl_usd: Decimal
    avg_pnl_per_trade_usd: Decimal
    win_rate: Decimal
    sharpe_ratio: Decimal
    avg_hold_duration_hours: Decimal
    current_drawdown_pct: Decimal
    max_drawdown_pct: Decimal


# ---------------------------------------------------------------------------
# Strategy Health Types
# ---------------------------------------------------------------------------


@dataclass
class StrategyHealthReport:
    alpha_decay_detected: bool = False
    accuracy_ratio: Decimal = Decimal("1.0")  # recent / historical win rate
    sharpe_ratio: Decimal = Decimal("1.0")  # recent / historical Sharpe
    recommendations: list[str] = field(default_factory=list)
    dynamic_confidence_threshold: Decimal | None = None


# ---------------------------------------------------------------------------
# Data Service Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExchangeFlows:
    inflow_usd: Decimal
    outflow_usd: Decimal
    avg_hourly_flow: Decimal
    data_age_seconds: int


@dataclass(frozen=True)
class PendingSwapVolume:
    volume_usd: Decimal
    avg_volume_usd: Decimal
    net_buy_ratio: Decimal  # 0.0-1.0
    window_seconds: int


@dataclass(frozen=True)
class LiquidationLevel:
    price: Decimal
    total_collateral_at_risk_usd: Decimal
    position_count: int
