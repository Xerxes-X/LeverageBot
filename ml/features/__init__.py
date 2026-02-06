"""
Feature engineering module for LeverageBot ML system.

Based on peer-reviewed research:
- Kolm et al. (2023): Order book imbalance (73% of prediction performance)
- Abad & Yagüe (2025): VPIN for price jump prediction
- MDPI (2025): Technical indicators + XGBoost for crypto prediction
"""

from .technical_indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr
)

from .microstructure import (
    calculate_order_book_imbalance,
    calculate_vpin,
    calculate_effective_spread,
    calculate_price_impact
)

from .volatility import (
    calculate_garch_volatility,
    calculate_realized_volatility,
    calculate_parkinson_volatility
)

from .feature_transformer import FeatureTransformer

__all__ = [
    'calculate_ema',
    'calculate_rsi',
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_atr',
    'calculate_order_book_imbalance',
    'calculate_vpin',
    'calculate_effective_spread',
    'calculate_price_impact',
    'calculate_garch_volatility',
    'calculate_realized_volatility',
    'calculate_parkinson_volatility',
    'FeatureTransformer'
]
