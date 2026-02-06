"""
Train XGBoost with feature-selected subset (20 features).

Compares performance against comprehensive model (46 features).
Uses same hyperparameters for fair comparison.
"""

import numpy as np
import pandas as pd
import pickle
import yaml
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

def load_data():
    """Load BNB and BTC data."""
    print("Loading data...")

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
    """Engineer all features (will be filtered later)."""
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

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    features['rsi_14'] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    features['macd'] = ema_12 - ema_26
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']

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

    hl_ratio = np.log(high / low)
    features['parkinson_vol'] = np.sqrt((hl_ratio ** 2) / (4 * np.log(2)))
    features['parkinson_vol_sma'] = features['parkinson_vol'].rolling(20).mean()

    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr_14'] = true_range.rolling(14).mean()

    cummax = close.cummax()
    drawdown = (close - cummax) / cummax
    features['drawdown'] = drawdown
    features['time_since_high'] = (cummax != close).astype(int).groupby((cummax == close).cumsum()).cumsum()

    # Layer 4: BTC cross-asset
    btc_returns = btc_close.pct_change()
    features['btc_close_norm'] = (btc_close - btc_close.rolling(100).mean()) / (btc_close.rolling(100).std() + 1e-10)
    features['btc_returns_1m'] = btc_returns
    features['btc_returns_15m'] = btc_close.pct_change(15)
    features['btc_returns_60m'] = btc_close.pct_change(60)
    features['btc_vol_15m'] = btc_returns.rolling(15).std()
    features['btc_vol_60m'] = btc_returns.rolling(60).std()
    features['btc_bnb_corr_30m'] = returns.rolling(30).corr(btc_returns)
    features['btc_bnb_corr_60m'] = returns.rolling(60).corr(btc_returns)
    features['bnb_btc_momentum_diff'] = returns.rolling(20).mean() - btc_returns.rolling(20).mean()
    features['btc_volume_ratio'] = btc_volume / (btc_volume.rolling(10).mean() + 1e-10)

    # Layer 5: Momentum
    features['momentum_5'] = close.pct_change(5)
    features['momentum_10'] = close.pct_change(10)
    features['momentum_20'] = close.pct_change(20)
    features['momentum_60'] = close.pct_change(60)
    features['return_lag1'] = returns.shift(1)
    features['return_lag2'] = returns.shift(2)
    features['return_lag5'] = returns.shift(5)
    features['return_lag10'] = returns.shift(10)
    features['return_mean_10'] = returns.rolling(10).mean()
    features['return_std_10'] = returns.rolling(10).std()
    features['return_skew_20'] = returns.rolling(20).skew()
    features['return_kurt_20'] = returns.rolling(20).kurt()

    return features

def create_labels(df, horizon=15):
    """Percentile-based balanced labels."""
    future_price = df['close'].shift(-horizon)
    future_return = (future_price - df['close']) / df['close']

    p60 = future_return.quantile(0.60)
    p40 = future_return.quantile(0.40)

    labels = pd.Series(index=df.index, dtype=float)
    labels[future_return >= p60] = 1
    labels[future_return <= p40] = 0

    return labels

def calculate_sharpe(y_true, y_pred_proba, returns):
    """Calculate Sharpe ratio from predictions."""
    signals = (y_pred_proba > 0.5).astype(int)
    signals = np.where(signals == 1, 1, -1)
    strategy_returns = signals * returns.reset_index(drop=True)

    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()

    if std_return == 0:
        return 0.0

    sharpe = (mean_return / std_return) * np.sqrt(525_600)
    return sharpe

def main():
    # Load selected features
    print("Loading selected features...")
    with open('models/feature_names_selected.txt', 'r') as f:
        selected_features = [line.strip() for line in f]

    print(f"Selected features: {len(selected_features)}")

    # Load data
    df = load_data()

    # Engineer all features
    print("\nEngineering comprehensive features...")
    features_df = engineer_comprehensive_features(df)

    # Create labels
    labels = create_labels(df)

    # Combine and clean
    features_df['label'] = labels
    features_df = features_df.dropna()

    # Filter to selected features only
    X_all = features_df[selected_features]
    y_all = features_df['label'].astype(int)

    print(f"\nDataset: {len(X_all):,} samples")
    print(f"Features: {len(selected_features)}")
    print(f"Class balance: UP={sum(y_all==1)/len(y_all):.1%}, DOWN={sum(y_all==0)/len(y_all):.1%}")

    # Split
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]

    X_train = train_df[selected_features]
    y_train = train_df['label'].astype(int)
    X_test = test_df[selected_features]
    y_test = test_df['label'].astype(int)

    # Get returns for Sharpe calculation
    df_clean = df.loc[features_df.index]
    returns_test = df_clean['close'].pct_change().iloc[split_idx:]

    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Test: {len(X_test):,} samples")

    # Train XGBoost with same hyperparameters as comprehensive model
    print("\n" + "="*70)
    print("TRAINING XGBOOST WITH SELECTED FEATURES")
    print("="*70)

    params = {
        'max_depth': 6,
        'learning_rate': 0.03,
        'n_estimators': 500,
        'min_child_weight': 3,
        'gamma': 0.15,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'subsample': 0.8,
        'colsample_bytree': 0.75,
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    sharpe = calculate_sharpe(y_test, y_pred_proba, returns_test)

    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best AUC: {model.best_score:.4f}")

    print("\n" + "="*70)
    print("TEST SET PERFORMANCE (Selected Features)")
    print("="*70)
    print(f"Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:     {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:        {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:      {f1:.4f}")
    print(f"AUC:           {auc:.4f}")
    print(f"Sharpe Ratio:  {sharpe:.2f}")

    # Load comprehensive model results for comparison
    print("\n" + "="*70)
    print("COMPARISON: Selected (20) vs Comprehensive (46)")
    print("="*70)
    print(f"{'Metric':<20} {'Selected (20)':<20} {'Comprehensive (46)':<20} {'Change'}")
    print("-" * 70)
    print(f"{'Features':<20} {len(selected_features):<20} {'46':<20} {'-26 (-57%)'}")
    print(f"{'Win Rate':<20} {f'{accuracy*100:.2f}%':<20} {'50.84%':<20} {f'{(accuracy-0.5084)*100:+.2f}%'}")
    print(f"{'Sharpe Ratio':<20} {f'{sharpe:.2f}':<20} {'4.11':<20} {f'{sharpe-4.11:+.2f}'}")
    print(f"{'AUC':<20} {f'{auc:.4f}':<20} {'0.5074':<20} {f'{auc-0.5074:+.4f}'}")

    # Feature importance from selected model
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + "="*70)
    print("TOP 10 FEATURE IMPORTANCES (Selected Model)")
    print("="*70)
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")

    # Save model
    import os
    os.makedirs('models', exist_ok=True)

    model_path = 'models/xgboost_selected_features_v1.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n✅ Model saved to {model_path}")

    # Save metadata
    metadata = {
        'model_type': 'XGBoost Selected Features',
        'training_date': datetime.now().isoformat(),
        'num_features': len(selected_features),
        'feature_names': selected_features,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'hyperparameters': params,
        'performance': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'sharpe_ratio': float(sharpe)
        },
        'best_iteration': int(model.best_iteration),
        'best_auc': float(model.best_score)
    }

    with open('models/xgboost_selected_features_v1_metadata.yaml', 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"✅ Metadata saved to models/xgboost_selected_features_v1_metadata.yaml")

    # Save feature importance
    feature_importance.to_csv('models/selected_features_importance.csv', index=False)
    print(f"✅ Feature importance saved to models/selected_features_importance.csv")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if sharpe > 4.11:
        print("✅ Selected features model OUTPERFORMS comprehensive model!")
        print(f"   Sharpe improved from 4.11 to {sharpe:.2f} ({(sharpe-4.11)/4.11*100:+.1f}%)")
        print("   Recommendation: Use selected features model for ensemble")
    elif sharpe > 3.5:
        print("✅ Selected features model performs similarly to comprehensive")
        print(f"   Sharpe: {sharpe:.2f} vs 4.11 (slight decrease but fewer features)")
        print("   Recommendation: Use selected features for faster inference")
    else:
        print("⚠️  Selected features model underperforms comprehensive model")
        print(f"   Sharpe: {sharpe:.2f} vs 4.11")
        print("   Recommendation: Stick with comprehensive model (46 features)")

    return metadata

if __name__ == '__main__':
    results = main()
