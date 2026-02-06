"""
Comprehensive XGBoost training with enhanced features and BTC correlation.

Option B: Comprehensive Optimization
- 180 days of data
- BTC cross-asset features
- Enhanced feature engineering (41+ features)
- Improved labeling strategy
"""

import numpy as np
import pandas as pd
import pickle
import yaml
import os
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

def load_data(bnb_path: str, btc_path: str) -> pd.DataFrame:
    """Load BNB and BTC data, merge on timestamp."""
    print(f"Loading BNB data from {bnb_path}...")
    bnb_df = pd.read_csv(bnb_path)
    bnb_df['timestamp'] = pd.to_datetime(bnb_df['timestamp'])
    bnb_df = bnb_df.sort_values('timestamp').reset_index(drop=True)

    print(f"Loading BTC data from {btc_path}...")
    btc_df = pd.read_csv(btc_path)
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df = btc_df.sort_values('timestamp').reset_index(drop=True)

    # Merge on timestamp
    btc_df = btc_df.rename(columns={
        'open': 'btc_open',
        'high': 'btc_high',
        'low': 'btc_low',
        'close': 'btc_close',
        'volume': 'btc_volume'
    })

    df = pd.merge(bnb_df, btc_df, on='timestamp', how='inner')

    print(f"Loaded {len(df):,} samples with BTC correlation")
    return df

def engineer_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer comprehensive feature set with BTC correlation.
    Target: 41+ features
    """
    print("Engineering comprehensive features...")
    features = pd.DataFrame(index=df.index)

    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']

    btc_close = df['btc_close']
    btc_volume = df['btc_volume']

    # ====== Layer 1: Price-based indicators (12 features) ======

    # EMAs
    features['ema_5'] = close.ewm(span=5, adjust=False).mean()
    features['ema_10'] = close.ewm(span=10, adjust=False).mean()
    features['ema_20'] = close.ewm(span=20, adjust=False).mean()
    features['ema_50'] = close.ewm(span=50, adjust=False).mean()

    # EMA crossovers
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

    # ====== Layer 2: Market microstructure proxies (6 features) ======

    # Volume-based (order book proxy)
    features['volume_sma_10'] = volume.rolling(window=10).mean()
    features['volume_ratio'] = volume / (features['volume_sma_10'] + 1e-10)
    features['volume_spike'] = (features['volume_ratio'] > 2.0).astype(int)

    # Price-volume relationship
    returns = close.pct_change()
    features['pv_correlation'] = returns.rolling(20).corr(volume.pct_change())

    # Trade flow proxy (price movement direction)
    features['price_direction'] = np.sign(close.diff())
    features['volume_weighted_price'] = (close * volume).rolling(10).sum() / (volume.rolling(10).sum() + 1e-10)

    # ====== Layer 3: Volatility features (8 features) ======

    # Realized volatility
    features['returns_std_5m'] = returns.rolling(window=5).std()
    features['returns_std_15m'] = returns.rolling(window=15).std()
    features['returns_std_60m'] = returns.rolling(window=60).std()

    # ATR
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    features['atr_14'] = true_range.ewm(span=14, adjust=False).mean()
    features['atr_ratio'] = features['atr_14'] / close

    # Volatility ratio
    features['vol_ratio'] = features['returns_std_15m'] / (features['returns_std_60m'] + 1e-10)

    # Parkinson volatility (high-low)
    log_hl = np.log(high / (low + 1e-10))
    features['parkinson_vol'] = np.sqrt(log_hl.rolling(20).mean() / (4 * np.log(2)))

    # ====== Layer 4: Cross-asset features (BTC spillover - 10 features) ======

    # BTC returns
    btc_returns = btc_close.pct_change()
    features['btc_returns'] = btc_returns
    features['btc_returns_lag1'] = btc_returns.shift(1)
    features['btc_returns_lag2'] = btc_returns.shift(2)

    # BTC-BNB correlation (rolling)
    features['btc_bnb_corr_30m'] = returns.rolling(30).corr(btc_returns)
    features['btc_bnb_corr_60m'] = returns.rolling(60).corr(btc_returns)

    # BTC volatility
    features['btc_vol_15m'] = btc_returns.rolling(15).std()
    features['btc_vol_60m'] = btc_returns.rolling(60).std()

    # BTC-BNB volatility ratio
    features['vol_spillover'] = features['btc_vol_15m'] / (features['returns_std_15m'] + 1e-10)

    # Relative strength (BNB vs BTC)
    features['bnb_btc_strength'] = returns.rolling(30).mean() / (btc_returns.rolling(30).mean() + 1e-10)

    # Volume correlation
    features['vol_corr'] = volume.pct_change().rolling(20).corr(btc_volume.pct_change())

    # ====== Layer 5: Momentum & Lagged features (12 features) ======

    # Multi-timeframe returns
    features['returns_1m'] = returns
    features['returns_5m'] = close.pct_change(5)
    features['returns_15m'] = close.pct_change(15)
    features['returns_30m'] = close.pct_change(30)
    features['returns_60m'] = close.pct_change(60)

    # Log returns
    features['log_returns_1m'] = np.log(close / close.shift(1))
    features['log_returns_5m'] = np.log(close / close.shift(5))

    # Rolling statistics
    features['returns_skew_30m'] = returns.rolling(30).skew()
    features['returns_kurt_30m'] = returns.rolling(30).kurt()

    # Price position
    features['dist_from_high_60'] = (close - close.rolling(60).max()) / close
    features['dist_from_low_60'] = (close - close.rolling(60).min()) / close

    # Momentum indicators
    features['price_momentum'] = close / close.shift(20) - 1

    print(f"Engineered {len(features.columns)} features")

    return features

def create_balanced_labels(df: pd.DataFrame, horizon: int = 15) -> pd.Series:
    """
    Create balanced labels using percentile-based approach.
    Top 40% = UP (1), Bottom 40% = DOWN (0), Middle 20% = filtered out
    """
    future_price = df['close'].shift(-horizon)
    future_return = (future_price - df['close']) / df['close']

    # Calculate percentiles
    p60 = future_return.quantile(0.60)
    p40 = future_return.quantile(0.40)

    # Label: 1 if top 40%, 0 if bottom 40%, NaN if middle 20%
    labels = pd.Series(index=df.index, dtype=float)
    labels[future_return >= p60] = 1
    labels[future_return <= p40] = 0

    return labels

def optimize_threshold(y_true, y_pred_proba, returns):
    """Find optimal prediction threshold for max Sharpe."""
    best_sharpe = -np.inf
    best_threshold = 0.5

    for threshold in np.arange(0.40, 0.65, 0.01):
        signals = (y_pred_proba > threshold).astype(int)
        strategy_returns = signals * returns
        sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-8)) * np.sqrt(525_600)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = threshold

    return best_threshold, best_sharpe

def main():
    # Load data with BTC correlation
    df = load_data(
        'data/raw/BNBUSDT_1m_180d.csv',
        'data/raw/BTCUSDT_1m_180d.csv'
    )

    # Engineer comprehensive features
    features_df = engineer_comprehensive_features(df)

    # Create balanced labels (removes middle 20%)
    labels = create_balanced_labels(df, horizon=15)

    # Combine
    features_df['label'] = labels

    # Drop NaN (includes middle 20% filtered labels)
    features_df = features_df.dropna()

    print(f"\nFinal dataset: {len(features_df):,} samples")
    label_counts = features_df['label'].value_counts()
    print(f"Class distribution: Up={label_counts.get(1.0, 0)} ({label_counts.get(1.0, 0)/len(features_df)*100:.1f}%), Down={label_counts.get(0.0, 0)} ({label_counts.get(0.0, 0)/len(features_df)*100:.1f}%)")

    # Split data (80/20)
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]

    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    print(f"Train set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    # Train model with better hyperparameters
    print("\nTraining XGBoost model with enhanced configuration...")

    # Split training into train/val
    val_split = 0.2
    val_idx = int(len(X_train) * (1 - val_split))

    X_train_final = X_train.iloc[:val_idx]
    y_train_final = y_train.iloc[:val_idx]
    X_val = X_train.iloc[val_idx:]
    y_val = y_train.iloc[val_idx:]

    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.03,
        n_estimators=500,
        min_child_weight=3,
        gamma=0.15,
        reg_alpha=0.2,
        reg_lambda=1.5,
        subsample=0.8,
        colsample_bytree=0.75,
        objective='binary:logistic',
        eval_metric=['logloss', 'auc'],
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_train_final,
        y_train_final,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=50
    )

    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score:.4f}")

    # Get test returns for Sharpe calculation
    test_returns = df['close'].pct_change().iloc[split_idx:split_idx+len(X_test)].reset_index(drop=True)

    # Optimize threshold on validation set
    val_returns = df['close'].pct_change().iloc[val_idx:val_idx+len(X_val)].reset_index(drop=True)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    optimal_threshold, val_sharpe = optimize_threshold(y_val, y_val_pred_proba, val_returns)

    print(f"\nOptimal threshold: {optimal_threshold:.3f} (val Sharpe: {val_sharpe:.2f})")

    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET (COMPREHENSIVE)")
    print("="*60)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > optimal_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Win Rate:  {accuracy:.2%}")

    # Calculate Sharpe ratio
    signals = (y_pred_proba > optimal_threshold).astype(int)
    strategy_returns = signals * test_returns
    sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-8)) * np.sqrt(525_600)

    print(f"Sharpe Ratio: {sharpe:.2f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n=== Top 15 Features ===")
    print(feature_importance.head(15).to_string(index=False))

    # Check Phase 1 targets
    print("\n" + "="*60)
    print("PHASE 1 TARGET ACHIEVEMENT")
    print("="*60)
    print(f"Win Rate Target:   58-62%")
    print(f"Actual Win Rate:   {accuracy:.2%} {'✅' if 0.58 <= accuracy <= 0.62 else '⚠️' if accuracy >= 0.55 else '❌'}")
    print(f"\nSharpe Target:     1.5-2.0")
    print(f"Actual Sharpe:     {sharpe:.2f} {'✅' if 1.5 <= sharpe <= 2.0 else '⚠️' if sharpe >= 1.0 else '❌'}")

    if accuracy >= 0.58 and sharpe >= 1.5:
        print("\n✅ PHASE 1 TARGETS ACHIEVED!")
        print("   Model ready for production deployment")
    elif accuracy >= 0.55 and sharpe >= 1.0:
        print("\n⚠️  Close to targets - minor optimization needed")
        print("   Consider: hyperparameter tuning or ensemble methods")
    else:
        print("\n⚠️  Progress made but needs more optimization")
        print("   Next: Optuna hyperparameter tuning + feature selection")

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/xgboost_comprehensive_v1.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save feature names
    with open('models/feature_names_comprehensive.txt', 'w') as f:
        f.write('\n'.join(X_train.columns))

    # Save metadata
    metadata = {
        'model_name': 'xgboost_comprehensive',
        'model_version': '1.0',
        'training_date': datetime.now().isoformat(),
        'n_features': len(X_train.columns),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'data_days': 180,
        'btc_correlation': True,
        'optimal_threshold': float(optimal_threshold),
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'win_rate': float(accuracy),
            'sharpe_ratio': float(sharpe)
        }
    }

    with open('models/xgboost_comprehensive_v1_metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)

    print(f"\nModel saved to {model_path}")
    print("Comprehensive training complete!")

if __name__ == '__main__':
    main()
