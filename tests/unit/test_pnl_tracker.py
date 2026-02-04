"""
Unit tests for core/pnl_tracker.py.

Tests verify SQLite table creation, record_open/record_close lifecycle,
unrealized P&L for long vs short, accrued interest calculation, and
summary statistics computation.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from shared.types import PositionDirection, PositionState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_position(
    direction: PositionDirection = PositionDirection.LONG,
    debt_usd: Decimal = Decimal("5000"),
    collateral_usd: Decimal = Decimal("5200"),
    initial_debt_usd: Decimal = Decimal("5000"),
    initial_collateral_usd: Decimal = Decimal("5200"),
    health_factor: Decimal = Decimal("1.8"),
    borrow_rate_ray: Decimal = Decimal("50000000000000000000000000"),
    liquidation_threshold: Decimal = Decimal("0.80"),
    debt_token: str = "USDT",
    collateral_token: str = "WBNB",
) -> PositionState:
    return PositionState(
        direction=direction,
        debt_token=debt_token,
        collateral_token=collateral_token,
        debt_usd=debt_usd,
        collateral_usd=collateral_usd,
        initial_debt_usd=initial_debt_usd,
        initial_collateral_usd=initial_collateral_usd,
        health_factor=health_factor,
        borrow_rate_ray=borrow_rate_ray,
        liquidation_threshold=liquidation_threshold,
    )


def _make_tracker(db_path: str):
    """Create a PnLTracker with patched logger."""
    with patch("core.pnl_tracker.setup_module_logger") as mock_logger:
        mock_logger.return_value = MagicMock()
        from core.pnl_tracker import PnLTracker

        return PnLTracker(db_path=db_path)


@pytest.fixture
def db_path():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    with contextlib.suppress(OSError):
        os.unlink(path)


@pytest.fixture
def tracker(db_path):
    """Create a PnLTracker with a temp database."""
    t = _make_tracker(db_path)
    yield t
    t.close()


# ---------------------------------------------------------------------------
# Table creation tests
# ---------------------------------------------------------------------------


class TestTableCreation:

    def test_positions_table_exists(self, tracker):
        cursor = tracker._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='positions'"
        )
        assert cursor.fetchone() is not None

    def test_snapshots_table_exists(self, tracker):
        cursor = tracker._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='position_snapshots'"
        )
        assert cursor.fetchone() is not None

    def test_transactions_table_exists(self, tracker):
        cursor = tracker._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'"
        )
        assert cursor.fetchone() is not None

    def test_idempotent_table_creation(self, db_path):
        """Creating tracker twice on same DB should not fail."""
        t1 = _make_tracker(db_path)
        t2 = _make_tracker(db_path)
        t1.close()
        t2.close()


# ---------------------------------------------------------------------------
# Record open/close lifecycle
# ---------------------------------------------------------------------------


class TestRecordLifecycle:

    @pytest.mark.asyncio
    async def test_record_open_returns_id(self, tracker):
        pos = _make_position()
        pos_id = await tracker.record_open(pos, "0xabc123", Decimal("0.50"))
        assert isinstance(pos_id, int)
        assert pos_id > 0

    @pytest.mark.asyncio
    async def test_record_open_persists_to_db(self, tracker):
        pos = _make_position()
        pos_id = await tracker.record_open(pos, "0xabc123", Decimal("0.50"))

        row = tracker._db.execute("SELECT * FROM positions WHERE id = ?", (pos_id,)).fetchone()
        assert row is not None
        assert row["direction"] == "long"
        assert row["debt_token"] == "USDT"
        assert row["collateral_token"] == "WBNB"
        assert row["open_tx_hash"] == "0xabc123"

    @pytest.mark.asyncio
    async def test_record_open_creates_transaction(self, tracker):
        pos = _make_position()
        pos_id = await tracker.record_open(pos, "0xabc123", Decimal("0.50"))

        tx = tracker._db.execute(
            "SELECT * FROM transactions WHERE position_id = ?", (pos_id,)
        ).fetchone()
        assert tx is not None
        assert tx["tx_type"] == "open"
        assert tx["success"] == 1

    @pytest.mark.asyncio
    async def test_record_close_computes_pnl(self, tracker):
        pos = _make_position()
        pos_id = await tracker.record_open(pos, "0xopen", Decimal("0.50"))

        pnl = await tracker.record_close(
            pos_id, "0xclose", Decimal("0.30"), Decimal("200"), "signal"
        )

        assert pnl.gross_pnl_usd == Decimal("200")
        assert pnl.gas_costs_usd == Decimal("0.80")  # 0.50 + 0.30
        assert pnl.net_pnl_usd == Decimal("200") - Decimal("0.80")

    @pytest.mark.asyncio
    async def test_record_close_persists_close_data(self, tracker):
        pos = _make_position()
        pos_id = await tracker.record_open(pos, "0xopen", Decimal("0.50"))

        await tracker.record_close(pos_id, "0xclose", Decimal("0.30"), Decimal("100"), "signal")

        row = tracker._db.execute("SELECT * FROM positions WHERE id = ?", (pos_id,)).fetchone()
        assert row["close_timestamp"] is not None
        assert row["close_tx_hash"] == "0xclose"
        assert row["close_reason"] == "signal"

    @pytest.mark.asyncio
    async def test_record_close_nonexistent_raises(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            await tracker.record_close(999, "0xhash", Decimal("0"), Decimal("0"))


# ---------------------------------------------------------------------------
# Deleverage recording
# ---------------------------------------------------------------------------


class TestRecordDeleverage:

    @pytest.mark.asyncio
    async def test_record_deleverage_accumulates_gas(self, tracker):
        pos = _make_position()
        pos_id = await tracker.record_open(pos, "0xopen", Decimal("0.50"))

        await tracker.record_deleverage(pos_id, "0xdelev", Decimal("0.20"))

        row = tracker._db.execute(
            "SELECT total_gas_costs_usd FROM positions WHERE id = ?", (pos_id,)
        ).fetchone()
        assert Decimal(row["total_gas_costs_usd"]) == Decimal("0.70")

    @pytest.mark.asyncio
    async def test_record_deleverage_creates_transaction(self, tracker):
        pos = _make_position()
        pos_id = await tracker.record_open(pos, "0xopen", Decimal("0.50"))

        await tracker.record_deleverage(pos_id, "0xdelev", Decimal("0.20"))

        txs = tracker._db.execute(
            "SELECT * FROM transactions WHERE position_id = ? AND tx_type = 'deleverage'",
            (pos_id,),
        ).fetchall()
        assert len(txs) == 1


# ---------------------------------------------------------------------------
# Unrealized P&L
# ---------------------------------------------------------------------------


class TestUnrealizedPnL:

    @pytest.mark.asyncio
    async def test_long_unrealized_pnl_positive(self, tracker):
        """Long: collateral appreciated."""
        pos = _make_position(
            direction=PositionDirection.LONG,
            collateral_usd=Decimal("5500"),
            initial_collateral_usd=Decimal("5200"),
            debt_usd=Decimal("5000"),
            initial_debt_usd=Decimal("5000"),
        )
        pnl = await tracker.get_unrealized_pnl(pos)
        # 5500 - 5200 - 0 (no accrued interest since debt == initial)
        assert pnl == Decimal("300")

    @pytest.mark.asyncio
    async def test_long_unrealized_pnl_negative(self, tracker):
        """Long: collateral depreciated."""
        pos = _make_position(
            direction=PositionDirection.LONG,
            collateral_usd=Decimal("4800"),
            initial_collateral_usd=Decimal("5200"),
            debt_usd=Decimal("5000"),
            initial_debt_usd=Decimal("5000"),
        )
        pnl = await tracker.get_unrealized_pnl(pos)
        assert pnl == Decimal("-400")

    @pytest.mark.asyncio
    async def test_short_unrealized_pnl_positive(self, tracker):
        """Short: debt (volatile) decreased in value."""
        pos = _make_position(
            direction=PositionDirection.SHORT,
            collateral_usd=Decimal("5000"),
            initial_collateral_usd=Decimal("5000"),
            debt_usd=Decimal("4500"),
            initial_debt_usd=Decimal("5000"),
            debt_token="WBNB",
            collateral_token="USDC",
        )
        pnl = await tracker.get_unrealized_pnl(pos)
        # (5000 - 4500) - 0 = 500
        assert pnl == Decimal("500")

    @pytest.mark.asyncio
    async def test_short_unrealized_pnl_negative(self, tracker):
        """Short: debt (volatile) increased in value."""
        pos = _make_position(
            direction=PositionDirection.SHORT,
            collateral_usd=Decimal("5000"),
            initial_collateral_usd=Decimal("5000"),
            debt_usd=Decimal("5500"),
            initial_debt_usd=Decimal("5000"),
            debt_token="WBNB",
            collateral_token="USDC",
        )
        pnl = await tracker.get_unrealized_pnl(pos)
        # (5000 - 5500) - 500 interest = -1000
        assert pnl == Decimal("-1000")

    @pytest.mark.asyncio
    async def test_unrealized_pnl_with_accrued_interest(self, tracker):
        """Accrued interest reduces P&L."""
        pos = _make_position(
            direction=PositionDirection.LONG,
            collateral_usd=Decimal("5500"),
            initial_collateral_usd=Decimal("5200"),
            debt_usd=Decimal("5050"),  # 50 interest accrued
            initial_debt_usd=Decimal("5000"),
        )
        pnl = await tracker.get_unrealized_pnl(pos)
        # (5500 - 5200) - 50 interest = 250
        assert pnl == Decimal("250")


# ---------------------------------------------------------------------------
# Accrued interest
# ---------------------------------------------------------------------------


class TestAccruedInterest:

    @pytest.mark.asyncio
    async def test_no_interest_when_debt_unchanged(self, tracker):
        pos = _make_position(
            debt_usd=Decimal("5000"),
            initial_debt_usd=Decimal("5000"),
        )
        interest = await tracker.get_accrued_interest(pos)
        assert interest == Decimal("0")

    @pytest.mark.asyncio
    async def test_positive_interest_when_debt_increased(self, tracker):
        pos = _make_position(
            debt_usd=Decimal("5100"),
            initial_debt_usd=Decimal("5000"),
        )
        interest = await tracker.get_accrued_interest(pos)
        assert interest == Decimal("100")


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


class TestSummaryStats:

    @pytest.mark.asyncio
    async def test_empty_stats(self, tracker):
        stats = await tracker.get_summary_stats()
        assert stats.total_trades == 0
        assert stats.win_rate == Decimal("0")

    @pytest.mark.asyncio
    async def test_stats_after_trades(self, tracker):
        # Create and close 3 positions: 2 winners, 1 loser
        for i, (pnl_amount, direction) in enumerate(
            [
                (Decimal("100"), PositionDirection.LONG),
                (Decimal("50"), PositionDirection.LONG),
                (Decimal("-30"), PositionDirection.SHORT),
            ]
        ):
            pos = _make_position(direction=direction)
            pos_id = await tracker.record_open(pos, f"0xopen{i}", Decimal("0.10"))
            _net = pnl_amount - Decimal("0.20")  # total gas
            await tracker.record_close(pos_id, f"0xclose{i}", Decimal("0.10"), pnl_amount, "signal")

        stats = await tracker.get_summary_stats()
        assert stats.total_trades == 3
        assert stats.winning_trades == 2
        assert stats.losing_trades == 1

    @pytest.mark.asyncio
    async def test_win_rate_calculation(self, tracker):
        # 2 winners out of 4
        for i in range(4):
            pos = _make_position()
            pos_id = await tracker.record_open(pos, f"0xopen{i}", Decimal("0.10"))
            pnl_amount = Decimal("100") if i < 2 else Decimal("-50")
            await tracker.record_close(pos_id, f"0xclose{i}", Decimal("0.10"), pnl_amount, "signal")

        stats = await tracker.get_summary_stats()
        assert stats.win_rate == Decimal("0.5")


# ---------------------------------------------------------------------------
# Rolling stats
# ---------------------------------------------------------------------------


class TestRollingStats:

    @pytest.mark.asyncio
    async def test_rolling_30_day_window(self, tracker):
        pos = _make_position()
        pos_id = await tracker.record_open(pos, "0xopen", Decimal("0.10"))
        await tracker.record_close(pos_id, "0xclose", Decimal("0.10"), Decimal("50"))

        stats = await tracker.get_rolling_stats(window_days=30)
        assert stats.total_trades == 1


# ---------------------------------------------------------------------------
# Drawdown computation
# ---------------------------------------------------------------------------


class TestDrawdowns:

    def test_no_drawdown_on_monotonic_gains(self):
        from core.pnl_tracker import PnLTracker

        pnls = [Decimal("100"), Decimal("50"), Decimal("200")]
        current_dd, max_dd = PnLTracker._compute_drawdowns(pnls)
        assert current_dd == Decimal("0")
        assert max_dd == Decimal("0")

    def test_drawdown_after_loss(self):
        from core.pnl_tracker import PnLTracker

        pnls = [Decimal("100"), Decimal("-50")]
        current_dd, max_dd = PnLTracker._compute_drawdowns(pnls)
        assert current_dd == Decimal("0.5")  # 50% drawdown from peak of 100
        assert max_dd == Decimal("0.5")

    def test_empty_pnls(self):
        from core.pnl_tracker import PnLTracker

        current_dd, max_dd = PnLTracker._compute_drawdowns([])
        assert current_dd == Decimal("0")
        assert max_dd == Decimal("0")


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:

    @pytest.mark.asyncio
    async def test_snapshot_persists(self, tracker):
        pos = _make_position()
        pos_id = await tracker.record_open(pos, "0xopen", Decimal("0.10"))

        await tracker.snapshot(pos_id, pos)

        rows = tracker._db.execute(
            "SELECT * FROM position_snapshots WHERE position_id = ?", (pos_id,)
        ).fetchall()
        assert len(rows) == 1
        assert Decimal(rows[0]["health_factor"]) == Decimal("1.8")


# ---------------------------------------------------------------------------
# Position history
# ---------------------------------------------------------------------------


class TestPositionHistory:

    @pytest.mark.asyncio
    async def test_get_position_history(self, tracker):
        pos = _make_position()
        pos_id = await tracker.record_open(pos, "0xopen", Decimal("0.10"))
        await tracker.record_close(pos_id, "0xclose", Decimal("0.10"), Decimal("50"))

        history = await tracker.get_position_history(limit=10)
        assert len(history) == 1
        assert history[0]["close_reason"] == "signal"

    @pytest.mark.asyncio
    async def test_get_open_position(self, tracker):
        pos = _make_position()
        await tracker.record_open(pos, "0xopen", Decimal("0.10"))

        open_pos = await tracker.get_open_position()
        assert open_pos is not None
        assert open_pos["close_timestamp"] is None
