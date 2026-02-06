"""
Feature Transformer - Unified feature engineering pipeline.

Transforms raw OHLCV + order book data into ML-ready features.
Designed for <2ms latency (production requirement: total <10ms including model inference).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import yaml

from .technical_indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_hurst_exponent
)
from .microstructure import (
    calculate_order_book_imbalance,
    calculate_vpin,
    calculate_trade_flow_imbalance,
    calculate_bid_ask_spread,
    calculate_depth_imbalance
)
from .volatility import (
    calculate_realized_volatility,
    calculate_parkinson_volatility,
    calculate_garch_volatility,
    calculate_rolling_max_drawdown,
    calculate_time_since_high,
    calculate_volatility_ratio
)


class FeatureTransformer:
    """
    Transform raw market data into engineered features.

    Supports two modes:
    1. Batch mode: Transform entire DataFrame (training)
    2. Online mode: Transform single sample with cached state (inference)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature transformer.

        Args:
            config: Feature configuration dict or path to YAML config
        """
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)

        self.config = config.get('features', {})

        # Stateful indicators cache (for online mode)
        self.ema_cache = {}
        self.rsi_cache = {}
        self.garch_cache = {'variance': None}

        self.feature_names = []

    def fit(self, df: pd.DataFrame) -> 'FeatureTransformer':
        """
        Fit transformer on training data (calculate feature names).

        Args:
            df: Training data with OHLCV columns

        Returns:
            self
        """
        features_df = self.transform(df)
        self.feature_names = features_df.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame into features (batch mode).

        Args:
            df: DataFrame with columns: open, high, low, close, volume,
                bid_volume, ask_volume, etc.

        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=df.index)

        # Layer 1: Price-based indicators
        features = self._add_price_features(features, df)

        # Layer 2: Market microstructure
        features = self._add_microstructure_features(features, df)

        # Layer 3: Volatility
        features = self._add_volatility_features(features, df)

        # Layer 4: Cross-asset features (if available)
        if 'btc_close' in df.columns:
            features = self._add_cross_asset_features(features, df)

        # Layer 5: Lagged returns
        features = self._add_lagged_returns(features, df)

        return features

    def _add_price_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add Layer 1: Price-based technical indicators."""
        close = df['close']

        # EMAs
        features['ema_12'] = calculate_ema(close, 12)
        features['ema_26'] = calculate_ema(close, 26)
        features['ema_50'] = calculate_ema(close, 50)

        # RSI
        features['rsi_14'] = calculate_rsi(close, 14)

        # MACD
        macd, signal, histogram = calculate_macd(close)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower, bb_position = calculate_bollinger_bands(close)
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_position'] = bb_position

        # ATR (if high/low available)
        if all(col in df.columns for col in ['high', 'low']):
            features['atr_14'] = calculate_atr(df['high'], df['low'], close, 14)

        return features

    def _add_microstructure_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add Layer 2: Market microstructure features."""
        # Order book imbalance
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            features['order_book_imbalance'] = calculate_order_book_imbalance(
                df['bid_volume'], df['ask_volume']
            )

            # Depth imbalance (if depth data available)
            if 'bid_depth_5' in df.columns and 'ask_depth_5' in df.columns:
                features['depth_imbalance_5'] = calculate_depth_imbalance(
                    df['bid_depth_5'], df['ask_depth_5']
                )

        # Bid-ask spread
        if 'best_bid' in df.columns and 'best_ask' in df.columns:
            mid_price = (df['best_bid'] + df['best_ask']) / 2
            features['bid_ask_spread'] = calculate_bid_ask_spread(
                df['best_bid'], df['best_ask'], mid_price
            )

        # VPIN (simplified - use volume and price change)
        if 'volume' in df.columns:
            price_change = df['close'] - df['open']
            features['vpin'] = calculate_vpin(df['volume'], price_change, window=20)

            # Trade flow imbalance
            features['trade_flow_imbalance'] = calculate_trade_flow_imbalance(
                df['volume'], price_change, window=20
            )

        return features

    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add Layer 3: Volatility features."""
        close = df['close']
        returns = close.pct_change()

        # Realized volatility (15-minute window)
        features['realized_volatility_15m'] = calculate_realized_volatility(returns, window=15)

        # Realized volatility (1-hour window)
        features['realized_volatility_1h'] = calculate_realized_volatility(returns, window=60)

        # Parkinson volatility (if high/low available)
        if 'high' in df.columns and 'low' in df.columns:
            features['parkinson_volatility'] = calculate_parkinson_volatility(
                df['high'], df['low'], window=20
            )

        # GARCH(1,1) volatility
        features['conditional_volatility_garch'] = calculate_garch_volatility(returns)

        # Volatility ratio
        features['volatility_ratio'] = calculate_volatility_ratio(
            features['realized_volatility_15m'],
            features['realized_volatility_1h']
        )

        # Rolling max drawdown
        features['rolling_max_drawdown_1h'] = calculate_rolling_max_drawdown(close, window=60)

        # Time since high
        features['time_since_high'] = calculate_time_since_high(close, window=60)

        return features

    def _add_cross_asset_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add Layer 4: Cross-asset features (BTC spillover)."""
        if 'btc_close' not in df.columns:
            return features

        btc_close = df['btc_close']
        bnb_close = df['close']

        # BTC returns
        btc_returns = btc_close.pct_change()
        features['btc_returns_lag1'] = btc_returns.shift(1)
        features['btc_returns_lag2'] = btc_returns.shift(2)

        # BTC-BNB correlation (1-hour rolling)
        bnb_returns = bnb_close.pct_change()
        features['btc_bnb_correlation_1h'] = bnb_returns.rolling(window=60).corr(btc_returns)

        # BTC volatility ratio
        btc_vol = calculate_realized_volatility(btc_returns, window=60)
        bnb_vol = calculate_realized_volatility(bnb_returns, window=60)
        features['btc_volatility_ratio'] = btc_vol / (bnb_vol + 1e-10)

        # Funding rate (if available)
        if 'funding_rate' in df.columns:
            features['binance_perp_funding_rate'] = df['funding_rate']

        return features

    def _add_lagged_returns(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add Layer 5: Lagged returns and momentum."""
        close = df['close']
        returns = close.pct_change()
        log_returns = np.log(close / close.shift(1))

        # Simple returns at different horizons
        features['returns_1m'] = returns
        features['returns_5m'] = close.pct_change(5)
        features['returns_15m'] = close.pct_change(15)
        features['returns_1h'] = close.pct_change(60)
        features['returns_4h'] = close.pct_change(240)
        features['returns_24h'] = close.pct_change(1440)

        # Log returns
        features['log_returns_1m'] = log_returns
        features['log_returns_5m'] = np.log(close / close.shift(5))

        # Rolling statistics of returns
        features['returns_std_15m'] = returns.rolling(window=15).std()
        features['returns_std_1h'] = returns.rolling(window=60).std()
        features['returns_skew_1h'] = returns.rolling(window=60).skew()
        features['returns_kurt_1h'] = returns.rolling(window=60).kurt()

        return features

    def transform_online(self, market_data: Dict, use_cache: bool = True) -> np.ndarray:
        """
        Transform single sample with cached state (online mode for inference).

        Target: <2ms latency

        Args:
            market_data: Dict with keys: price, volume, bid_volume, ask_volume, etc.
            use_cache: Use cached stateful indicators (EMA, RSI)

        Returns:
            Feature vector (1D numpy array)
        """
        # This is a simplified version for demonstration
        # In production, you would maintain full state and update incrementally
        features = {}

        price = market_data['price']

        # Layer 1: Price features (simplified with cache)
        if use_cache and 'ema_12' in self.ema_cache:
            # Update EMA incrementally (exponential smoothing)
            alpha_12 = 2 / (12 + 1)
            features['ema_12'] = alpha_12 * price + (1 - alpha_12) * self.ema_cache['ema_12']
            self.ema_cache['ema_12'] = features['ema_12']
        else:
            features['ema_12'] = price
            self.ema_cache['ema_12'] = price

        # Similar for other indicators...
        # (Full implementation would cache all stateful indicators)

        # Layer 2: Microstructure (stateless)
        if 'bid_volume' in market_data and 'ask_volume' in market_data:
            bid = market_data['bid_volume']
            ask = market_data['ask_volume']
            features['order_book_imbalance'] = (bid - ask) / (bid + ask + 1e-10)

        # Layer 3-5: Other features...
        # (Implementation similar to batch mode but with cached state)

        # Convert to array matching training feature order
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])

        return feature_vector

    @classmethod
    def from_config(cls, config_path: str) -> 'FeatureTransformer':
        """
        Create transformer from YAML config file.

        Args:
            config_path: Path to config YAML

        Returns:
            FeatureTransformer instance
        """
        return cls(config=config_path)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names

    def save_state(self, path: str):
        """Save transformer state (cache) for online mode."""
        import pickle
        state = {
            'ema_cache': self.ema_cache,
            'rsi_cache': self.rsi_cache,
            'garch_cache': self.garch_cache,
            'feature_names': self.feature_names
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, path: str):
        """Load transformer state (cache) for online mode."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.ema_cache = state['ema_cache']
        self.rsi_cache = state['rsi_cache']
        self.garch_cache = state['garch_cache']
        self.feature_names = state['feature_names']
