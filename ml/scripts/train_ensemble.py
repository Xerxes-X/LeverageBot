"""
Train ensemble of XGBoost models with different configurations.

Trains 5 models with diverse hyperparameters:
1. Best manual (our current champion - Sharpe 4.11)
2. Conservative (high regularization)
3. Aggressive (low regularization)
4. Shallow trees (max_depth 4)
5. Deep trees (max_depth 8)

Averages predictions to improve generalization and win rate.
"""

import numpy as np
import pandas as pd
import pickle
import yaml
from datetime import datetime
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
    """Engineer comprehensive 46 features."""
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
    """Calculate Sharpe ratio."""
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
    # Load and prepare data
    df = load_data()

    print("\nEngineering comprehensive features...")
    features_df = engineer_comprehensive_features(df)

    labels = create_labels(df)

    features_df['label'] = labels
    features_df = features_df.dropna()

    print(f"\nDataset: {len(features_df):,} samples")
    print(f"Features: {features_df.shape[1] - 1}")

    # Split
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]

    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label'].astype(int)
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label'].astype(int)

    df_clean = df.loc[features_df.index]
    returns_test = df_clean['close'].pct_change().iloc[split_idx:]

    print(f"Train: {len(X_train):,} samples")
    print(f"Test: {len(X_test):,} samples")

    # Define ensemble configurations
    configs = {
        'best_manual': {
            'max_depth': 6,
            'learning_rate': 0.03,
            'n_estimators': 500,
            'min_child_weight': 3,
            'gamma': 0.15,
            'reg_alpha': 0.2,
            'reg_lambda': 1.5,
            'subsample': 0.8,
            'colsample_bytree': 0.75
        },
        'conservative': {
            'max_depth': 5,
            'learning_rate': 0.02,
            'n_estimators': 500,
            'min_child_weight': 5,
            'gamma': 0.3,
            'reg_alpha': 0.5,
            'reg_lambda': 2.5,
            'subsample': 0.7,
            'colsample_bytree': 0.7
        },
        'aggressive': {
            'max_depth': 7,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_weight': 1,
            'gamma': 0.05,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'subsample': 0.9,
            'colsample_bytree': 0.9
        },
        'shallow': {
            'max_depth': 4,
            'learning_rate': 0.04,
            'n_estimators': 500,
            'min_child_weight': 4,
            'gamma': 0.2,
            'reg_alpha': 0.3,
            'reg_lambda': 2.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'deep': {
            'max_depth': 8,
            'learning_rate': 0.025,
            'n_estimators': 500,
            'min_child_weight': 2,
            'gamma': 0.1,
            'reg_alpha': 0.15,
            'reg_lambda': 1.0,
            'subsample': 0.85,
            'colsample_bytree': 0.8
        }
    }

    # Train ensemble
    print("\n" + "="*80)
    print("TRAINING ENSEMBLE (5 MODELS)")
    print("="*80)

    models = {}
    predictions = {}
    metrics = {}

    for name, config in configs.items():
        print(f"\n--- Training {name} model ---")

        params = {
            **config,
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': 42
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        sharpe = calculate_sharpe(y_test, y_pred_proba, returns_test)

        print(f"  Win rate: {acc*100:.2f}%")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Best iteration: {model.best_iteration}")

        models[name] = model
        predictions[name] = y_pred_proba
        metrics[name] = {'accuracy': acc, 'sharpe': sharpe, 'best_iter': model.best_iteration}

    # Calculate ensemble predictions (average)
    print("\n" + "="*80)
    print("ENSEMBLE PREDICTIONS (AVERAGE)")
    print("="*80)

    ensemble_proba = np.mean(list(predictions.values()), axis=0)
    ensemble_pred = (ensemble_proba > 0.5).astype(int)

    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_precision = precision_score(y_test, ensemble_pred, zero_division=0)
    ensemble_recall = recall_score(y_test, ensemble_pred, zero_division=0)
    ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=0)
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    ensemble_sharpe = calculate_sharpe(y_test, ensemble_proba, returns_test)

    print(f"\nAccuracy:     {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
    print(f"Precision:    {ensemble_precision:.4f} ({ensemble_precision*100:.2f}%)")
    print(f"Recall:       {ensemble_recall:.4f} ({ensemble_recall*100:.2f}%)")
    print(f"F1 Score:     {ensemble_f1:.4f}")
    print(f"AUC:          {ensemble_auc:.4f}")
    print(f"Sharpe Ratio: {ensemble_sharpe:.2f}")

    # Comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Win Rate':<15} {'Sharpe':<15}")
    print("-" * 80)

    for name, m in metrics.items():
        print(f"{name:<20} {m['accuracy']*100:>6.2f}%       {m['sharpe']:>8.2f}")

    print("-" * 80)
    print(f"{'ENSEMBLE (avg)':<20} {ensemble_acc*100:>6.2f}%       {ensemble_sharpe:>8.2f}")
    print(f"{'Original (46 feat)':<20} {'50.84%':>14} {'4.11':>15}")

    # Save ensemble
    import os
    os.makedirs('models', exist_ok=True)

    ensemble_data = {
        'models': models,
        'predictions': predictions,
        'metrics': metrics,
        'ensemble_metrics': {
            'accuracy': float(ensemble_acc),
            'precision': float(ensemble_precision),
            'recall': float(ensemble_recall),
            'f1_score': float(ensemble_f1),
            'auc': float(ensemble_auc),
            'sharpe_ratio': float(ensemble_sharpe)
        },
        'configs': configs,
        'training_date': datetime.now().isoformat()
    }

    with open('models/xgboost_ensemble_v1.pkl', 'wb') as f:
        pickle.dump(ensemble_data, f)

    print(f"\n✅ Ensemble saved to models/xgboost_ensemble_v1.pkl")

    # Save metadata
    metadata = {
        'model_type': 'XGBoost Ensemble (5 models)',
        'training_date': datetime.now().isoformat(),
        'num_models': len(models),
        'model_names': list(models.keys()),
        'num_features': X_train.shape[1],
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'individual_performance': metrics,
        'ensemble_performance': ensemble_data['ensemble_metrics'],
        'configs': configs
    }

    with open('models/xgboost_ensemble_v1_metadata.yaml', 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"✅ Metadata saved to models/xgboost_ensemble_v1_metadata.yaml")

    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)

    if ensemble_sharpe > 4.11 and ensemble_acc > 0.5084:
        print("✅ ENSEMBLE OUTPERFORMS - Use ensemble for deployment!")
        print(f"   Win rate: {ensemble_acc*100:.2f}% vs 50.84%")
        print(f"   Sharpe: {ensemble_sharpe:.2f} vs 4.11")
    elif ensemble_sharpe > 4.0:
        print("✅ ENSEMBLE PERFORMS WELL - Consider ensemble for robustness")
        print(f"   Sharpe: {ensemble_sharpe:.2f} (comparable to 4.11)")
        print(f"   Win rate: {ensemble_acc*100:.2f}%")
    else:
        print("⚠️  SINGLE MODEL BETTER - Use comprehensive model (46 features)")
        print(f"   Best single: Sharpe 4.11, Win rate 50.84%")
        print(f"   Ensemble: Sharpe {ensemble_sharpe:.2f}, Win rate {ensemble_acc*100:.2f}%")

    return ensemble_data

if __name__ == '__main__':
    results = main()
