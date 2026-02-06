"""
Simplified XGBoost training script for Phase 1.

Uses only essential features to avoid excessive NaN values.
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

def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Loaded {len(df):,} samples")
    return df

def engineer_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer simplified features with minimal lookback.
    Avoids excessive NaN values.
    """
    print("Engineering features...")
    features = pd.DataFrame(index=df.index)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Price-based features (short lookback)
    features['ema_5'] = close.ewm(span=5, adjust=False).mean()
    features['ema_10'] = close.ewm(span=10, adjust=False).mean()
    features['ema_20'] = close.ewm(span=20, adjust=False).mean()

    # RSI (14 period)
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
    features['bb_upper'] = sma_20 + (2 * std_20)
    features['bb_lower'] = sma_20 - (2 * std_20)
    features['bb_position'] = (close - features['bb_lower']) / ((features['bb_upper'] - features['bb_lower']) + 1e-10)

    # Volatility (short-term)
    returns = close.pct_change()
    features['returns_std_5m'] = returns.rolling(window=5).std()
    features['returns_std_15m'] = returns.rolling(window=15).std()

    # Volume features
    features['volume_sma_10'] = volume.rolling(window=10).mean()
    features['volume_ratio'] = volume / (features['volume_sma_10'] + 1e-10)

    # Price momentum
    features['returns_5m'] = close.pct_change(5)
    features['returns_15m'] = close.pct_change(15)
    features['returns_30m'] = close.pct_change(30)

    # ATR
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    features['atr_14'] = true_range.ewm(span=14, adjust=False).mean()

    print(f"Engineered {len(features.columns)} features")

    return features

def create_labels(df: pd.DataFrame, horizon: int = 15) -> pd.Series:
    """Create binary labels for price direction."""
    future_price = df['close'].shift(-horizon)
    future_return = (future_price - df['close']) / df['close']
    labels = (future_return > 0).astype(int)
    return labels

def main():
    # Load data
    df = load_data('data/raw/BNBUSDT_1m_60d.csv')

    # Engineer features
    features_df = engineer_simple_features(df)

    # Create labels
    labels = create_labels(df, horizon=15)

    # Combine
    features_df['label'] = labels

    # Drop NaN (first 30 rows should be enough)
    features_df = features_df.dropna()

    print(f"\nFinal dataset: {len(features_df):,} samples")
    print(f"Class distribution: Up={labels.sum()}, Down={len(labels)-labels.sum()}")

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

    # Train model
    print("\nTraining XGBoost model...")

    # Split training into train/val
    val_split = 0.2
    val_idx = int(len(X_train) * (1 - val_split))

    X_train_final = X_train.iloc[:val_idx]
    y_train_final = y_train.iloc[:val_idx]
    X_val = X_train.iloc[val_idx:]
    y_val = y_train.iloc[val_idx:]

    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.01,
        n_estimators=500,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
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

    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Win Rate:  {accuracy:.2%}")

    # Calculate Sharpe ratio (simplified)
    test_returns = df['close'].pct_change().iloc[split_idx:split_idx+len(X_test)].reset_index(drop=True)
    signals = (y_pred_proba > 0.5).astype(int)
    strategy_returns = signals * test_returns
    sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-8)) * np.sqrt(525_600)

    print(f"Sharpe Ratio: {sharpe:.2f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n=== Top 10 Features ===")
    print(feature_importance.head(10).to_string(index=False))

    # Check Phase 1 targets
    print("\n" + "="*60)
    print("PHASE 1 TARGET ACHIEVEMENT")
    print("="*60)
    print(f"Win Rate Target:   58-62%")
    print(f"Actual Win Rate:   {accuracy:.2%} {'✅' if 0.58 <= accuracy <= 0.62 else '⚠️'}")
    print(f"\nSharpe Target:     1.5-2.0")
    print(f"Actual Sharpe:     {sharpe:.2f} {'✅' if 1.5 <= sharpe <= 2.0 else '⚠️'}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/xgboost_phase1_simple.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save feature names
    with open('models/feature_names.txt', 'w') as f:
        f.write('\n'.join(X_train.columns))

    # Save metadata
    metadata = {
        'model_name': 'xgboost_phase1_simple',
        'model_version': '1.0',
        'training_date': datetime.now().isoformat(),
        'n_features': len(X_train.columns),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
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

    with open('models/xgboost_phase1_simple_metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)

    print(f"\nModel saved to {model_path}")
    print("Training complete!")

if __name__ == '__main__':
    main()
