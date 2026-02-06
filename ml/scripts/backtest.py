"""
Backtesting framework for XGBoost trading strategy.

Implements walk-forward backtesting with:
- Realistic transaction costs (DEX fees + slippage)
- Borrow costs (stablecoin APR)
- Fractional Kelly position sizing
- Risk management filters
- Comprehensive performance metrics

Based on research:
- MacLean et al. (2010): Fractional Kelly for optimal growth
- López de Prado (2018): Walk-forward validation for financial ML

Usage:
    python backtest.py --model models/xgboost_phase1_v1.pkl --data data/test/
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import yaml
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_transformer import FeatureTransformer


class Position:
    """Represents an open trading position."""

    def __init__(
        self,
        direction: str,
        entry_price: float,
        position_size: float,
        timestamp: datetime,
        collateral_asset: str = 'WBNB',
        debt_asset: str = 'USDT'
    ):
        self.direction = direction  # 'LONG' or 'SHORT'
        self.entry_price = entry_price
        self.position_size = position_size
        self.timestamp = timestamp
        self.collateral_asset = collateral_asset
        self.debt_asset = debt_asset
        self.exit_price = None
        self.exit_timestamp = None
        self.pnl = None
        self.return_pct = None

    def close(self, exit_price: float, exit_timestamp: datetime) -> float:
        """Close position and calculate PnL."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp

        if self.direction == 'LONG':
            # Long: Profit if price goes up
            self.return_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # SHORT
            # Short: Profit if price goes down
            self.return_pct = (self.entry_price - exit_price) / self.entry_price

        self.pnl = self.position_size * self.return_pct
        return self.pnl

    def get_holding_period_hours(self) -> float:
        """Calculate holding period in hours."""
        if self.exit_timestamp:
            return (self.exit_timestamp - self.timestamp).total_seconds() / 3600
        return 0.0


class Backtester:
    """
    Walk-forward backtesting engine with realistic transaction costs.
    """

    def __init__(
        self,
        model_path: str,
        initial_capital: float = 100_000,
        position_size_method: str = 'fractional_kelly',
        kelly_fraction: float = 0.25,
        trading_fee_bps: float = 10,  # 0.1% (DEX aggregator + slippage)
        borrow_rate_annual: float = 0.05,  # 5% APR
        max_position_size: float = 0.5,  # 50% of capital
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        confidence_threshold: float = 0.55
    ):
        """
        Initialize backtester.

        Args:
            model_path: Path to trained XGBoost model
            initial_capital: Starting capital in USD
            position_size_method: 'fixed', 'fractional_kelly', 'kelly'
            kelly_fraction: Fraction of Kelly for position sizing (0.25 = 25% Kelly)
            trading_fee_bps: Trading fee in basis points (10 = 0.1%)
            borrow_rate_annual: Annual borrow rate (0.05 = 5%)
            max_position_size: Maximum position size as fraction of capital
            stop_loss_pct: Stop loss percentage (None = no stop loss)
            take_profit_pct: Take profit percentage (None = no TP)
            confidence_threshold: Minimum prediction confidence to enter trade
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_method = position_size_method
        self.kelly_fraction = kelly_fraction
        self.trading_fee_bps = trading_fee_bps
        self.borrow_rate_annual = borrow_rate_annual
        self.borrow_rate_hourly = borrow_rate_annual / 365 / 24
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.confidence_threshold = confidence_threshold

        # Load model
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load feature transformer
        transformer_path = model_path.replace('.pkl', '_transformer_state.pkl')
        if os.path.exists(transformer_path):
            self.transformer = FeatureTransformer()
            self.transformer.load_state(transformer_path)
        else:
            print("Warning: Feature transformer state not found, using new transformer")
            self.transformer = FeatureTransformer()

        # Trading state
        self.current_position: Optional[Position] = None
        self.closed_positions: List[Position] = []
        self.equity_curve: List[Dict] = []
        self.trades: List[Dict] = []

    def calculate_position_size(
        self,
        confidence: float,
        current_capital: float
    ) -> float:
        """
        Calculate position size based on confidence and Kelly criterion.

        Args:
            confidence: Model confidence (0 to 1)
            current_capital: Current portfolio capital

        Returns:
            Position size in USD
        """
        if self.position_size_method == 'fixed':
            # Fixed 10% of capital
            size = current_capital * 0.1

        elif self.position_size_method == 'fractional_kelly':
            # Fractional Kelly: f* = (p - q) / b, where p = win prob, q = 1-p, b = odds
            # Simplified: edge / odds, then apply kelly_fraction

            # Map confidence to edge
            # confidence 0.5 = no edge, 1.0 = maximum edge
            edge = (confidence - 0.5) * 2  # Map [0.5, 1.0] to [0, 1.0]
            edge = max(0, edge)  # No negative edge

            # Assume 1:1 odds (1% gain vs 1% loss)
            kelly_fraction_size = edge  # For 1:1 odds, Kelly = edge

            # Apply fractional Kelly
            size = current_capital * kelly_fraction_size * self.kelly_fraction

        elif self.position_size_method == 'kelly':
            # Full Kelly (not recommended - high risk)
            edge = (confidence - 0.5) * 2
            edge = max(0, edge)
            size = current_capital * edge

        else:
            raise ValueError(f"Unknown position size method: {self.position_size_method}")

        # Cap at max position size
        max_size = current_capital * self.max_position_size
        size = min(size, max_size)

        return size

    def calculate_transaction_cost(self, position_size: float) -> float:
        """Calculate transaction cost (entry + exit)."""
        # Entry cost + Exit cost
        total_cost = (position_size * self.trading_fee_bps / 10000) * 2
        return total_cost

    def calculate_borrow_cost(
        self,
        position_size: float,
        holding_period_hours: float
    ) -> float:
        """Calculate borrow cost based on holding period."""
        borrow_cost = position_size * self.borrow_rate_hourly * holding_period_hours
        return borrow_cost

    def check_risk_management(
        self,
        position: Position,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Check if position should be closed due to risk management rules.

        Returns:
            (should_close, reason)
        """
        if position.direction == 'LONG':
            price_change_pct = (current_price - position.entry_price) / position.entry_price
        else:  # SHORT
            price_change_pct = (position.entry_price - current_price) / position.entry_price

        # Stop loss check
        if self.stop_loss_pct and price_change_pct < -self.stop_loss_pct:
            return True, 'stop_loss'

        # Take profit check
        if self.take_profit_pct and price_change_pct > self.take_profit_pct:
            return True, 'take_profit'

        return False, ''

    def run(self, test_data: pd.DataFrame) -> Dict:
        """
        Run backtest on test data.

        Args:
            test_data: DataFrame with OHLCV data and timestamps

        Returns:
            Dict with performance metrics and results
        """
        print(f"\nRunning backtest on {len(test_data):,} samples...")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Position sizing: {self.position_size_method} (Kelly fraction: {self.kelly_fraction})")
        print(f"Transaction cost: {self.trading_fee_bps} bps")
        print(f"Borrow rate: {self.borrow_rate_annual*100:.2f}% APR")

        # Engineer features
        print("Engineering features...")
        features_df = self.transformer.transform(test_data)

        # Align indices
        features_df = features_df.dropna()
        test_data_aligned = test_data.loc[features_df.index]

        # Iterate through data
        for idx, (timestamp, row) in enumerate(test_data_aligned.iterrows()):
            current_price = row['close']
            features = features_df.iloc[idx:idx+1]

            # Get model prediction
            pred_proba = self.model.predict_proba(features)[0, 1]
            confidence = abs(pred_proba - 0.5) * 2  # Map [0.5, 1.0] to [0, 1]
            direction = 'LONG' if pred_proba > 0.5 else 'SHORT'

            # Check existing position
            if self.current_position:
                # Check risk management
                should_close, reason = self.check_risk_management(
                    self.current_position,
                    current_price
                )

                # Check if signal reversed (basic exit strategy)
                signal_reversed = (
                    (self.current_position.direction == 'LONG' and direction == 'SHORT' and confidence > 0.6) or
                    (self.current_position.direction == 'SHORT' and direction == 'LONG' and confidence > 0.6)
                )

                if should_close or signal_reversed:
                    # Close position
                    pnl = self.current_position.close(current_price, timestamp)

                    # Calculate costs
                    transaction_cost = self.calculate_transaction_cost(self.current_position.position_size)
                    holding_hours = self.current_position.get_holding_period_hours()
                    borrow_cost = self.calculate_borrow_cost(self.current_position.position_size, holding_hours)

                    # Net PnL after costs
                    net_pnl = pnl - transaction_cost - borrow_cost
                    self.capital += net_pnl

                    # Record trade
                    self.trades.append({
                        'entry_timestamp': self.current_position.timestamp,
                        'exit_timestamp': timestamp,
                        'direction': self.current_position.direction,
                        'entry_price': self.current_position.entry_price,
                        'exit_price': current_price,
                        'position_size': self.current_position.position_size,
                        'gross_pnl': pnl,
                        'transaction_cost': transaction_cost,
                        'borrow_cost': borrow_cost,
                        'net_pnl': net_pnl,
                        'return_pct': self.current_position.return_pct,
                        'holding_hours': holding_hours,
                        'exit_reason': reason if reason else 'signal_reversal'
                    })

                    self.closed_positions.append(self.current_position)
                    self.current_position = None

            # Consider opening new position
            if self.current_position is None and confidence > self.confidence_threshold:
                # Calculate position size
                position_size = self.calculate_position_size(confidence, self.capital)

                # Check if we have enough capital
                if position_size > 0 and position_size < self.capital:
                    # Open position
                    self.current_position = Position(
                        direction=direction,
                        entry_price=current_price,
                        position_size=position_size,
                        timestamp=timestamp
                    )

            # Record equity
            current_equity = self.capital
            if self.current_position:
                # Mark-to-market
                if self.current_position.direction == 'LONG':
                    unrealized_pnl = self.current_position.position_size * (
                        (current_price - self.current_position.entry_price) / self.current_position.entry_price
                    )
                else:
                    unrealized_pnl = self.current_position.position_size * (
                        (self.current_position.entry_price - current_price) / self.current_position.entry_price
                    )
                current_equity += unrealized_pnl

            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'capital': self.capital,
                'has_position': self.current_position is not None
            })

        # Close any remaining position
        if self.current_position:
            final_price = test_data_aligned.iloc[-1]['close']
            final_timestamp = test_data_aligned.index[-1]
            pnl = self.current_position.close(final_price, final_timestamp)

            transaction_cost = self.calculate_transaction_cost(self.current_position.position_size)
            holding_hours = self.current_position.get_holding_period_hours()
            borrow_cost = self.calculate_borrow_cost(self.current_position.position_size, holding_hours)

            net_pnl = pnl - transaction_cost - borrow_cost
            self.capital += net_pnl

            self.trades.append({
                'entry_timestamp': self.current_position.timestamp,
                'exit_timestamp': final_timestamp,
                'direction': self.current_position.direction,
                'entry_price': self.current_position.entry_price,
                'exit_price': final_price,
                'position_size': self.current_position.position_size,
                'gross_pnl': pnl,
                'transaction_cost': transaction_cost,
                'borrow_cost': borrow_cost,
                'net_pnl': net_pnl,
                'return_pct': self.current_position.return_pct,
                'holding_hours': holding_hours,
                'exit_reason': 'backtest_end'
            })

            self.closed_positions.append(self.current_position)

        # Calculate performance metrics
        metrics = self.calculate_metrics()

        return {
            'metrics': metrics,
            'trades': pd.DataFrame(self.trades),
            'equity_curve': pd.DataFrame(self.equity_curve)
        }

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'n_trades': 0
            }

        trades_df = pd.DataFrame(self.trades)

        # Basic metrics
        total_return = self.capital - self.initial_capital
        total_return_pct = (self.capital / self.initial_capital - 1) * 100
        n_trades = len(trades_df)

        # Win rate
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0

        # Average trade metrics
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        losing_trades = trades_df[trades_df['net_pnl'] <= 0]
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0

        # Profit factor
        gross_profit = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Sharpe ratio
        returns = trades_df['return_pct'].values
        sharpe_ratio = self._calculate_sharpe(returns)

        # Sortino ratio (downside deviation)
        sortino_ratio = self._calculate_sortino(returns)

        # Calmar ratio
        max_drawdown = self._calculate_max_drawdown()
        calmar_ratio = (total_return_pct / 100) / abs(max_drawdown) if max_drawdown != 0 else 0

        # Holding period
        avg_holding_hours = trades_df['holding_hours'].mean()

        # Costs
        total_transaction_costs = trades_df['transaction_cost'].sum()
        total_borrow_costs = trades_df['borrow_cost'].sum()

        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'avg_holding_hours': avg_holding_hours,
            'total_transaction_costs': total_transaction_costs,
            'total_borrow_costs': total_borrow_costs,
            'total_costs': total_transaction_costs + total_borrow_costs
        }

        return metrics

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0.0

        # Annualize (assuming average 1-hour holding period)
        # 8760 hours per year
        sharpe = (mean_return / std_return) * np.sqrt(8760)

        return sharpe

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean()
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return np.inf

        downside_std = downside_returns.std()

        if downside_std == 0:
            return 0.0

        sortino = (mean_return / downside_std) * np.sqrt(8760)

        return sortino

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage."""
        if not self.equity_curve:
            return 0.0

        equity = np.array([x['equity'] for x in self.equity_curve])
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min()

        return max_dd * 100  # Return as percentage

    def plot_results(self, output_path: str = 'backtest_results.html'):
        """Create interactive plots of backtest results."""
        if not self.equity_curve or not self.trades:
            print("No data to plot")
            return

        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Equity Curve', 'Drawdown', 'Trade Returns'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25]
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # Mark trades
        for _, trade in trades_df.iterrows():
            color = 'green' if trade['net_pnl'] > 0 else 'red'
            fig.add_trace(
                go.Scatter(
                    x=[trade['exit_timestamp']],
                    y=[equity_df[equity_df['timestamp'] == trade['exit_timestamp']]['equity'].values[0]],
                    mode='markers',
                    marker=dict(color=color, size=8, symbol='diamond'),
                    showlegend=False,
                    hovertext=f"{trade['direction']}: ${trade['net_pnl']:.2f}"
                ),
                row=1, col=1
            )

        # Drawdown
        equity = equity_df['equity'].values
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax * 100

        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=drawdown,
                name='Drawdown %',
                fill='tozeroy',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )

        # Trade returns
        fig.add_trace(
            go.Bar(
                x=trades_df['exit_timestamp'],
                y=trades_df['return_pct'] * 100,
                name='Trade Return %',
                marker_color=['green' if x > 0 else 'red' for x in trades_df['return_pct']]
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=3, col=1)

        fig.update_layout(
            title='Backtest Results',
            height=1000,
            showlegend=True
        )

        # Save
        fig.write_html(output_path)
        print(f"Results plotted to {output_path}")

    def print_report(self, metrics: Dict):
        """Print formatted backtest report."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)

        print(f"\n📊 Portfolio Performance")
        print(f"  Initial Capital:     ${metrics['initial_capital']:,.2f}")
        print(f"  Final Capital:       ${metrics['final_capital']:,.2f}")
        print(f"  Total Return:        ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")

        print(f"\n📈 Trading Statistics")
        print(f"  Number of Trades:    {metrics['n_trades']}")
        print(f"  Win Rate:            {metrics['win_rate']*100:.2f}%")
        print(f"  Average Win:         ${metrics['avg_win']:,.2f}")
        print(f"  Average Loss:        ${metrics['avg_loss']:,.2f}")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")

        print(f"\n📉 Risk Metrics")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:.2f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:.2f}")

        print(f"\n⏱️  Timing")
        print(f"  Avg Holding Period:  {metrics['avg_holding_hours']:.1f} hours")

        print(f"\n💰 Costs")
        print(f"  Transaction Costs:   ${metrics['total_transaction_costs']:,.2f}")
        print(f"  Borrow Costs:        ${metrics['total_borrow_costs']:,.2f}")
        print(f"  Total Costs:         ${metrics['total_costs']:,.2f}")

        # Phase 1 targets check
        print(f"\n🎯 Phase 1 Targets")
        target_win_rate = 0.58
        target_sharpe = 1.5

        win_rate_check = "✅" if metrics['win_rate'] >= target_win_rate else "❌"
        sharpe_check = "✅" if metrics['sharpe_ratio'] >= target_sharpe else "❌"

        print(f"  Win Rate Target:     58-62%  {win_rate_check} Actual: {metrics['win_rate']*100:.2f}%")
        print(f"  Sharpe Target:       1.5-2.0 {sharpe_check} Actual: {metrics['sharpe_ratio']:.2f}")

        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Backtest XGBoost trading strategy')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--kelly-fraction', type=float, default=0.25, help='Kelly fraction (0.25 = 25%% Kelly)')
    parser.add_argument('--confidence-threshold', type=float, default=0.55, help='Min confidence to trade')
    parser.add_argument('--output', type=str, default='backtest_results.html', help='Output plot path')

    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {args.data}...")
    test_data = pd.read_csv(args.data)
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
    test_data = test_data.sort_values('timestamp').reset_index(drop=True)

    print(f"Loaded {len(test_data):,} samples")
    print(f"Date range: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")

    # Create backtester
    backtester = Backtester(
        model_path=args.model,
        initial_capital=args.capital,
        kelly_fraction=args.kelly_fraction,
        confidence_threshold=args.confidence_threshold
    )

    # Run backtest
    results = backtester.run(test_data)

    # Print report
    backtester.print_report(results['metrics'])

    # Plot results
    backtester.plot_results(args.output)

    # Save detailed results
    trades_output = args.output.replace('.html', '_trades.csv')
    results['trades'].to_csv(trades_output, index=False)
    print(f"Trade details saved to {trades_output}")

    equity_output = args.output.replace('.html', '_equity.csv')
    results['equity_curve'].to_csv(equity_output, index=False)
    print(f"Equity curve saved to {equity_output}")

    print("\nBacktest complete!")


if __name__ == '__main__':
    main()
