"""
P&L tracking and position history for BSC Leverage Bot.

Tracks position lifecycle, computes realized/unrealized P&L including
accrued interest, and maintains persistent history in SQLite.

Usage:
    from core.pnl_tracker import PnLTracker

    tracker = PnLTracker()
    position_id = await tracker.record_open(position, tx_hash, gas_cost)
    pnl = await tracker.record_close(position_id, tx_hash, gas_cost, tokens_received)
"""

from __future__ import annotations

import sqlite3
import time
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bot_logging.logger_manager import setup_module_logger
from shared.types import (
    PositionDirection,
    PositionState,
    RealizedPnL,
    TradingStats,
)

if TYPE_CHECKING:
    pass

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_DB_DIR = _PROJECT_ROOT / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "positions.db"

_SECONDS_PER_HOUR = 3600


class PnLTracker:
    """
    SQLite-backed P&L tracker for position lifecycle management.

    Tracks opens, closes, deleverages, and periodic snapshots.
    Computes realized/unrealized P&L including accrued interest costs.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(_DEFAULT_DB_PATH)

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db_path = db_path
        self._db = sqlite3.connect(db_path)
        self._db.row_factory = sqlite3.Row
        self._create_tables()

        self._logger = setup_module_logger(
            "pnl_tracker", "pnl_tracker.log", module_folder="PnL_Tracker_Logs"
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create SQLite tables if they don't exist."""
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                direction TEXT NOT NULL,
                open_timestamp INTEGER NOT NULL,
                close_timestamp INTEGER,
                debt_token TEXT NOT NULL,
                collateral_token TEXT NOT NULL,
                initial_debt_amount TEXT NOT NULL,
                initial_collateral_amount TEXT NOT NULL,
                flash_loan_premium_paid TEXT,
                close_debt_amount TEXT,
                close_collateral_amount TEXT,
                realized_pnl_usd TEXT,
                total_gas_costs_usd TEXT,
                open_tx_hash TEXT NOT NULL,
                close_tx_hash TEXT,
                open_borrow_rate_apr TEXT,
                avg_borrow_rate_apr TEXT,
                close_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS position_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id INTEGER NOT NULL REFERENCES positions(id),
                timestamp INTEGER NOT NULL,
                collateral_value_usd TEXT NOT NULL,
                debt_value_usd TEXT NOT NULL,
                health_factor TEXT NOT NULL,
                borrow_rate_apr TEXT NOT NULL,
                unrealized_pnl_usd TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id INTEGER REFERENCES positions(id),
                tx_hash TEXT NOT NULL UNIQUE,
                timestamp INTEGER NOT NULL,
                tx_type TEXT NOT NULL,
                gas_used INTEGER,
                gas_price_gwei TEXT,
                gas_cost_usd TEXT,
                success BOOLEAN NOT NULL,
                revert_reason TEXT
            );
        """)
        self._db.commit()

    # ------------------------------------------------------------------
    # Record operations
    # ------------------------------------------------------------------

    async def record_open(
        self,
        position: PositionState,
        tx_hash: str,
        gas_cost_usd: Decimal,
    ) -> int:
        """
        Record a new position opening.

        Returns the auto-generated position ID.
        """
        now = int(time.time())
        cursor = self._db.execute(
            """INSERT INTO positions
               (direction, open_timestamp, debt_token, collateral_token,
                initial_debt_amount, initial_collateral_amount,
                total_gas_costs_usd, open_tx_hash, open_borrow_rate_apr)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                position.direction.value,
                now,
                position.debt_token,
                position.collateral_token,
                str(position.initial_debt_usd),
                str(position.initial_collateral_usd),
                str(gas_cost_usd),
                tx_hash,
                str(position.borrow_rate_ray),
            ),
        )
        position_id = cursor.lastrowid
        if position_id is None:
            raise RuntimeError("Failed to obtain position ID after INSERT")

        # Record the transaction
        self._db.execute(
            """INSERT INTO transactions
               (position_id, tx_hash, timestamp, tx_type, gas_cost_usd, success)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (position_id, tx_hash, now, "open", str(gas_cost_usd), True),
        )
        self._db.commit()

        self._logger.info(
            "Position opened: id=%d direction=%s debt=%s collateral=%s tx=%s",
            position_id,
            position.direction.value,
            position.debt_token,
            position.collateral_token,
            tx_hash,
        )
        return position_id

    async def record_close(
        self,
        position_id: int,
        tx_hash: str,
        gas_cost_usd: Decimal,
        tokens_received: Decimal,
        close_reason: str = "signal",
    ) -> RealizedPnL:
        """
        Record position close and compute realized P&L.

        Returns a RealizedPnL with breakdown of costs and profit.
        """
        now = int(time.time())

        # Fetch the open position data
        row = self._db.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
        if row is None:
            raise ValueError(f"Position {position_id} not found")

        _initial_debt = Decimal(row["initial_debt_amount"])
        _initial_collateral = Decimal(row["initial_collateral_amount"])
        direction = row["direction"]
        prev_gas = Decimal(row["total_gas_costs_usd"] or "0")
        total_gas = prev_gas + gas_cost_usd

        # Compute realized P&L
        # For LONG: profit = tokens_received - initial_debt (you borrowed stables, profit is excess)
        # For SHORT: profit = tokens_received - initial_debt
        # (you borrowed volatile, profit is excess)
        # In both cases, tokens_received is what you get back after repaying flash loan
        gross_pnl = tokens_received
        accrued_interest = Decimal("0")
        flash_premiums = Decimal("0")

        if direction == PositionDirection.LONG.value:
            # Long: collateral appreciated, tokens_received is the surplus
            gross_pnl = tokens_received
        else:
            # Short: debt depreciated, tokens_received is the surplus
            gross_pnl = tokens_received

        net_pnl = gross_pnl - total_gas - flash_premiums - accrued_interest

        # Update position record
        self._db.execute(
            """UPDATE positions SET
               close_timestamp = ?, close_tx_hash = ?, close_reason = ?,
               realized_pnl_usd = ?, total_gas_costs_usd = ?
               WHERE id = ?""",
            (now, tx_hash, close_reason, str(net_pnl), str(total_gas), position_id),
        )

        # Record the transaction
        self._db.execute(
            """INSERT INTO transactions
               (position_id, tx_hash, timestamp, tx_type, gas_cost_usd, success)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (position_id, tx_hash, now, "close", str(gas_cost_usd), True),
        )
        self._db.commit()

        result = RealizedPnL(
            gross_pnl_usd=gross_pnl,
            accrued_interest_usd=accrued_interest,
            gas_costs_usd=total_gas,
            flash_loan_premiums_usd=flash_premiums,
            net_pnl_usd=net_pnl,
        )

        self._logger.info(
            "Position closed: id=%d net_pnl=$%s reason=%s tx=%s",
            position_id,
            net_pnl,
            close_reason,
            tx_hash,
        )
        return result

    async def record_deleverage(
        self,
        position_id: int,
        tx_hash: str,
        gas_cost_usd: Decimal,
    ) -> None:
        """Record a deleverage event for an existing position."""
        now = int(time.time())

        # Update total gas costs
        row = self._db.execute(
            "SELECT total_gas_costs_usd FROM positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Position {position_id} not found")

        prev_gas = Decimal(row["total_gas_costs_usd"] or "0")
        total_gas = prev_gas + gas_cost_usd
        self._db.execute(
            "UPDATE positions SET total_gas_costs_usd = ? WHERE id = ?",
            (str(total_gas), position_id),
        )

        # Record the transaction
        self._db.execute(
            """INSERT INTO transactions
               (position_id, tx_hash, timestamp, tx_type, gas_cost_usd, success)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (position_id, tx_hash, now, "deleverage", str(gas_cost_usd), True),
        )
        self._db.commit()

        self._logger.info(
            "Deleverage recorded: position_id=%d gas=$%s tx=%s",
            position_id,
            gas_cost_usd,
            tx_hash,
        )

    async def snapshot(
        self,
        position_id: int,
        position: PositionState,
    ) -> None:
        """Take a periodic snapshot of an open position's state."""
        unrealized = await self.get_unrealized_pnl(position)
        now = int(time.time())

        self._db.execute(
            """INSERT INTO position_snapshots
               (position_id, timestamp, collateral_value_usd, debt_value_usd,
                health_factor, borrow_rate_apr, unrealized_pnl_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                position_id,
                now,
                str(position.collateral_usd),
                str(position.debt_usd),
                str(position.health_factor),
                str(position.borrow_rate_ray),
                str(unrealized),
            ),
        )
        self._db.commit()

    # ------------------------------------------------------------------
    # P&L computation
    # ------------------------------------------------------------------

    async def get_unrealized_pnl(self, position: PositionState) -> Decimal:
        """
        Compute unrealized P&L for an open position.

        For LONG: profit = (current_collateral_value - initial_collateral_value) - accrued_interest
        For SHORT: profit = (initial_debt_value - current_debt_value) - accrued_interest
        """
        accrued = await self.get_accrued_interest(position)
        if position.direction == PositionDirection.LONG:
            return (position.collateral_usd - position.initial_collateral_usd) - accrued
        else:
            return (position.initial_debt_usd - position.debt_usd) - accrued

    async def get_accrued_interest(self, position: PositionState) -> Decimal:
        """
        Estimate accrued interest based on position state.

        Uses the difference between current debt and initial debt as a proxy
        for interest accrual when on-chain query isn't available.
        """
        if position.debt_usd <= position.initial_debt_usd:
            return Decimal("0")
        return position.debt_usd - position.initial_debt_usd

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    async def get_position_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get closed position history, most recent first."""
        rows = self._db.execute(
            """SELECT * FROM positions
               WHERE close_timestamp IS NOT NULL
               ORDER BY close_timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    async def get_open_position(self) -> dict[str, Any] | None:
        """Get the current open position, if any."""
        row = self._db.execute("""SELECT * FROM positions
               WHERE close_timestamp IS NULL
               ORDER BY open_timestamp DESC LIMIT 1""").fetchone()
        return dict(row) if row else None

    async def get_summary_stats(self) -> TradingStats:
        """Compute aggregate trading statistics from closed positions."""
        return await self.get_rolling_stats(window_days=None)

    async def get_rolling_stats(self, window_days: int | None = 30) -> TradingStats:
        """
        Compute trading statistics for a rolling window.

        Args:
            window_days: Number of days to look back. None = all time.
        """
        if window_days is not None:
            cutoff = int(time.time()) - (window_days * 86400)
            rows = self._db.execute(
                """SELECT * FROM positions
                   WHERE close_timestamp IS NOT NULL
                     AND close_timestamp >= ?
                   ORDER BY close_timestamp DESC""",
                (cutoff,),
            ).fetchall()
        else:
            rows = self._db.execute("""SELECT * FROM positions
                   WHERE close_timestamp IS NOT NULL
                   ORDER BY close_timestamp DESC""").fetchall()

        total_trades = len(rows)
        if total_trades == 0:
            return TradingStats(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl_usd=Decimal("0"),
                avg_pnl_per_trade_usd=Decimal("0"),
                win_rate=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                avg_hold_duration_hours=Decimal("0"),
                current_drawdown_pct=Decimal("0"),
                max_drawdown_pct=Decimal("0"),
            )

        pnls: list[Decimal] = []
        hold_durations: list[int] = []
        winning = 0
        losing = 0

        for row in rows:
            pnl = Decimal(row["realized_pnl_usd"] or "0")
            pnls.append(pnl)
            if pnl > 0:
                winning += 1
            elif pnl < 0:
                losing += 1

            duration = (row["close_timestamp"] or 0) - row["open_timestamp"]
            hold_durations.append(duration)

        total_pnl = sum(pnls, Decimal("0"))
        avg_pnl = total_pnl / Decimal(total_trades)

        win_rate = Decimal(winning) / Decimal(total_trades) if total_trades > 0 else Decimal("0")

        avg_hold_hours = (
            Decimal(sum(hold_durations)) / Decimal(len(hold_durations)) / _SECONDS_PER_HOUR
            if hold_durations
            else Decimal("0")
        )

        # Sharpe ratio (daily returns approximation)
        sharpe = self._compute_sharpe(pnls)

        # Drawdown
        current_dd, max_dd = self._compute_drawdowns(pnls)

        return TradingStats(
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            total_pnl_usd=total_pnl,
            avg_pnl_per_trade_usd=avg_pnl,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            avg_hold_duration_hours=avg_hold_hours,
            current_drawdown_pct=current_dd,
            max_drawdown_pct=max_dd,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_drawdown_pct(self) -> Decimal:
        """Quick access to current drawdown percentage from all-time data."""
        rows = self._db.execute("""SELECT realized_pnl_usd FROM positions
               WHERE close_timestamp IS NOT NULL
               ORDER BY close_timestamp ASC""").fetchall()
        pnls = [Decimal(r["realized_pnl_usd"] or "0") for r in rows]
        if not pnls:
            return Decimal("0")
        current_dd, _ = self._compute_drawdowns(pnls)
        return current_dd

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sharpe(pnls: list[Decimal]) -> Decimal:
        """Compute Sharpe ratio from a list of P&L values."""
        if len(pnls) < 2:
            return Decimal("0")

        mean_pnl: Decimal = sum(pnls, Decimal("0")) / Decimal(len(pnls))
        variance: Decimal = sum(((p - mean_pnl) ** 2 for p in pnls), Decimal("0")) / Decimal(
            len(pnls) - 1
        )

        if variance <= 0:
            return Decimal("0")

        # Use Newton's method for sqrt since Decimal doesn't have sqrt
        std_dev = PnLTracker._decimal_sqrt(variance)
        if std_dev <= 0:
            return Decimal("0")

        return mean_pnl / std_dev

    @staticmethod
    def _decimal_sqrt(value: Decimal) -> Decimal:
        """Compute square root of a Decimal using Newton's method."""
        if value <= 0:
            return Decimal("0")
        # Initial guess
        x = value
        for _ in range(50):
            x_new = (x + value / x) / Decimal("2")
            if abs(x_new - x) < Decimal("1e-18"):
                break
            x = x_new
        return x

    @staticmethod
    def _compute_drawdowns(pnls: list[Decimal]) -> tuple[Decimal, Decimal]:
        """
        Compute current and max drawdown from cumulative P&L series.

        Returns (current_drawdown_pct, max_drawdown_pct) as positive decimals
        (e.g. 0.15 = 15% drawdown).
        """
        if not pnls:
            return Decimal("0"), Decimal("0")

        cumulative = Decimal("0")
        peak = Decimal("0")
        max_drawdown = Decimal("0")

        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            if peak > 0:
                dd = (peak - cumulative) / peak
                if dd > max_drawdown:
                    max_drawdown = dd

        current_dd = Decimal("0")
        if peak > 0:
            current_dd = (peak - cumulative) / peak
            current_dd = max(current_dd, Decimal("0"))

        return current_dd, max_drawdown

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the SQLite database connection."""
        if self._db:
            self._db.close()
            self._logger.debug("PnLTracker database closed")
