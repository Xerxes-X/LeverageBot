"""
Boruta feature selection for comprehensive XGBoost model.

Identifies statistically important features vs. random noise.
Removes features that don't provide predictive value.
"""

import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from datetime import datetime
import sys
import os

# Import feature engineering from comprehensive script
sys.path.append('scripts')

def load_data():
    """Load preprocessed data with comprehensive features."""
    print("Loading data...")

    # Load BNB and BTC data
    bnb_df = pd.read_csv('data/raw/BNBUSDT_1m_180d.csv')
    bnb_df['timestamp'] = pd.to_datetime(bnb_df['timestamp'])

    btc_df = pd.read_csv('data/raw/BTCUSDT_1m_180d.csv')
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df = btc_df.rename(columns={
        'open': 'btc_open',
        'high': 'btc_high',
        'low': 'btc_low',
        'close': 'btc_close',
        'volume': 'btc_volume'
    })

    df = pd.merge(bnb_df, btc_df, on='timestamp', how='inner')

    print(f"Loaded {len(df):,} samples")
    return df

def engineer_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive feature engineering (same as training script)."""
    features = pd.DataFrame(index=df.index)

    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    btc_close = df['btc_close']
    btc_volume = df['btc_volume']

    # Layer 1: Price indicators
    features['ema_5'] = close.ewm(span=5, adjust=False).mean()
    features['ema_10'] = close.ewm(span=10, adjust=False).mean()
    features['ema_20'] = close.ewm(span=20, adjust=False).mean()
    features['ema_50'] = close.ewm(span=50, adjust=False).mean()
    features['ema_5_10_cross'] = features['ema_5'] - features['ema_10']
    features['ema_10_20_cross'] = features['ema_10'] - features['ema_20']

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    features['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    features['macd'] = ema_12 - ema_26
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']

    # Bollinger Bands
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    features['bb_position'] = (close - sma_20) / (std_20 + 1e-10)

    # Layer 2: Microstructure
    features['volume_sma_10'] = volume.rolling(window=10).mean()
    features['volume_ratio'] = volume / (features['volume_sma_10'] + 1e-10)
    features['volume_weighted_price'] = (close * volume) / (volume + 1e-10)
    features['high_low_spread'] = (high - low) / (close + 1e-10)
    features['close_open_change'] = (close - open_price) / (open_price + 1e-10)
    features['intrabar_momentum'] = (close - open_price) / (high - low + 1e-10)

    # Layer 3: Volatility
    returns = close.pct_change()
    features['vol_5m'] = returns.rolling(5).std()
    features['vol_15m'] = returns.rolling(15).std()
    features['vol_60m'] = returns.rolling(60).std()

    # Parkinson volatility
    hl_ratio = np.log(high / low)
    features['parkinson_vol'] = np.sqrt((hl_ratio ** 2) / (4 * np.log(2)))
    features['parkinson_vol_sma'] = features['parkinson_vol'].rolling(20).mean()

    # ATR
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr_14'] = true_range.rolling(14).mean()

    # Drawdown
    cummax = close.cummax()
    drawdown = (close - cummax) / cummax
    features['drawdown'] = drawdown
    features['time_since_high'] = (cummax != close).astype(int).groupby((cummax == close).cumsum()).cumsum()

    # Layer 4: BTC cross-asset features
    btc_returns = btc_close.pct_change()
    features['btc_close_norm'] = (btc_close - btc_close.rolling(100).mean()) / (btc_close.rolling(100).std() + 1e-10)
    features['btc_returns_1m'] = btc_returns
    features['btc_returns_15m'] = btc_close.pct_change(15)
    features['btc_returns_60m'] = btc_close.pct_change(60)
    features['btc_vol_15m'] = btc_returns.rolling(15).std()
    features['btc_vol_60m'] = btc_returns.rolling(60).std()

    # BTC-BNB correlation
    features['btc_bnb_corr_30m'] = returns.rolling(30).corr(btc_returns)
    features['btc_bnb_corr_60m'] = returns.rolling(60).corr(btc_returns)

    # Cross-asset momentum
    features['bnb_btc_momentum_diff'] = returns.rolling(20).mean() - btc_returns.rolling(20).mean()
    features['btc_volume_ratio'] = btc_volume / (btc_volume.rolling(10).mean() + 1e-10)

    # Layer 5: Momentum & lagged returns
    features['momentum_5'] = close.pct_change(5)
    features['momentum_10'] = close.pct_change(10)
    features['momentum_20'] = close.pct_change(20)
    features['momentum_60'] = close.pct_change(60)

    # Lagged returns
    features['return_lag1'] = returns.shift(1)
    features['return_lag2'] = returns.shift(2)
    features['return_lag5'] = returns.shift(5)
    features['return_lag10'] = returns.shift(10)

    # Rolling statistics
    features['return_mean_10'] = returns.rolling(10).mean()
    features['return_std_10'] = returns.rolling(10).std()
    features['return_skew_20'] = returns.rolling(20).skew()
    features['return_kurt_20'] = returns.rolling(20).kurt()

    return features

def create_labels(df, horizon=15):
    """Balanced percentile-based labels."""
    future_price = df['close'].shift(-horizon)
    future_return = (future_price - df['close']) / df['close']

    p60 = future_return.quantile(0.60)
    p40 = future_return.quantile(0.40)

    labels = pd.Series(index=df.index, dtype=float)
    labels[future_return >= p60] = 1
    labels[future_return <= p40] = 0

    return labels

def main():
    # Load data
    df = load_data()

    # Engineer features
    print("\nEngineering comprehensive features...")
    features_df = engineer_comprehensive_features(df)

    # Create labels
    print("Creating balanced labels...")
    labels = create_labels(df)

    # Combine and drop NaN
    features_df['label'] = labels
    features_df = features_df.dropna()

    print(f"\nDataset: {len(features_df):,} samples")
    print(f"Features: {features_df.shape[1] - 1}")
    print(f"Class balance: {labels.value_counts(normalize=True).to_dict()}")

    # Use only training set for feature selection (80%)
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_idx]

    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label'].astype(int)

    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"Starting Boruta feature selection...")
    print("This will take 10-15 minutes...\n")

    # Initialize Random Forest for Boruta
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        n_jobs=-1,
        random_state=42
    )

    # Initialize Boruta
    boruta_selector = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        max_iter=100,
        verbose=2,
        random_state=42
    )

    # Run Boruta
    print("Running Boruta algorithm...")
    boruta_selector.fit(X_train.values, y_train.values)

    # Get results
    feature_names = X_train.columns.tolist()
    selected_features = [f for f, s in zip(feature_names, boruta_selector.support_) if s]
    tentative_features = [f for f, s in zip(feature_names, boruta_selector.support_weak_) if s]
    rejected_features = [f for f, s in zip(feature_names, boruta_selector.support_) if not s]

    # Get feature importance rankings
    feature_ranks = dict(zip(feature_names, boruta_selector.ranking_))

    print("\n" + "="*70)
    print("BORUTA FEATURE SELECTION RESULTS")
    print("="*70)

    print(f"\n✅ Selected features ({len(selected_features)}):")
    for feat in sorted(selected_features):
        print(f"  - {feat} (rank {feature_ranks[feat]})")

    if tentative_features:
        print(f"\n⚠️  Tentative features ({len(tentative_features)}):")
        for feat in sorted(tentative_features):
            print(f"  - {feat} (rank {feature_ranks[feat]})")

    print(f"\n❌ Rejected features ({len(rejected_features)}):")
    for feat in sorted(rejected_features):
        print(f"  - {feat} (rank {feature_ranks[feat]})")

    # Combine selected + tentative for final feature set
    final_features = selected_features + tentative_features

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Original features: {len(feature_names)}")
    print(f"Selected (confirmed): {len(selected_features)}")
    print(f"Selected (tentative): {len(tentative_features)}")
    print(f"Final feature set: {len(final_features)}")
    print(f"Rejected: {len(rejected_features)}")
    print(f"Reduction: {len(rejected_features)/len(feature_names)*100:.1f}%")

    # Save results
    os.makedirs('models', exist_ok=True)

    results = {
        'selection_date': datetime.now().isoformat(),
        'original_features': len(feature_names),
        'selected_confirmed': selected_features,
        'selected_tentative': tentative_features,
        'rejected': rejected_features,
        'final_features': final_features,
        'feature_ranks': feature_ranks,
        'method': 'Boruta with Random Forest',
        'parameters': {
            'n_estimators': 100,
            'max_depth': 7,
            'max_iter': 100
        }
    }

    with open('models/selected_features.yaml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

    print(f"\n✅ Results saved to models/selected_features.yaml")

    # Save selected feature names for training
    with open('models/feature_names_selected.txt', 'w') as f:
        for feat in final_features:
            f.write(f"{feat}\n")

    print(f"✅ Feature names saved to models/feature_names_selected.txt")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Retrain XGBoost with selected features:")
    print("   python scripts/train_with_selected_features.py")
    print("\n2. Compare performance:")
    print("   - Original (46 features): Sharpe 4.11, Win rate 50.84%")
    print(f"   - Selected ({len(final_features)} features): TBD")
    print("\n3. If performance improves, use for ensemble training")

    return results

if __name__ == '__main__':
    results = main()
