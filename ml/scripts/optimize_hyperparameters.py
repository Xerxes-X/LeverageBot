"""
Hyperparameter optimization using Optuna.

Searches for optimal XGBoost parameters to maximize Sharpe ratio + accuracy.
Uses 200 trials as specified in comprehensive optimization plan.
"""

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import pickle
import yaml
from datetime import datetime

# Import from comprehensive training script
import sys
sys.path.append('scripts')

def load_data():
    """Load preprocessed data with features."""
    print("Loading data...")

    # Load BNB and BTC data
    bnb_df = pd.read_csv('data/raw/BNBUSDT_1m_180d.csv')
    bnb_df['timestamp'] = pd.to_datetime(bnb_df['timestamp'])

    btc_df = pd.read_csv('data/raw/BTCUSDT_1m_180d.csv')
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df = btc_df.rename(columns={'close': 'btc_close', 'volume': 'btc_volume'})

    df = pd.merge(bnb_df, btc_df[['timestamp', 'btc_close', 'btc_volume']], on='timestamp', how='inner')

    print(f"Loaded {len(df):,} samples")
    return df

def engineer_features(df):
    """Quick feature engineering (simplified for speed)."""
    features = pd.DataFrame(index=df.index)

    close = df['close']
    volume = df['volume']
    btc_close = df['btc_close']

    # Essential features only for optimization speed
    features['ema_5'] = close.ewm(span=5, adjust=False).mean()
    features['ema_10'] = close.ewm(span=10, adjust=False).mean()
    features['ema_20'] = close.ewm(span=20, adjust=False).mean()
    features['ema_50'] = close.ewm(span=50, adjust=False).mean()

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

    # Volatility
    returns = close.pct_change()
    features['vol_15m'] = returns.rolling(15).std()
    features['vol_60m'] = returns.rolling(60).std()

    # Volume
    features['volume_ratio'] = volume / volume.rolling(10).mean()

    # BTC correlation
    btc_returns = btc_close.pct_change()
    features['btc_corr'] = returns.rolling(30).corr(btc_returns)
    features['btc_vol'] = btc_returns.rolling(15).std()

    # Momentum
    features['momentum_20'] = close / close.shift(20) - 1

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

def objective(trial, X, y, returns):
    """
    Optuna objective function.

    Optimizes for combined metric: 0.5 * accuracy + 0.5 * normalized_sharpe
    """
    # Hyperparameter search space
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42
    }

    # Time series cross-validation (3 folds for speed)
    tscv = TimeSeriesSplit(n_splits=3)

    accuracies = []
    sharpes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        returns_val = returns.iloc[val_idx]

        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_fold, y_train_fold, verbose=False)

        # Predict
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Accuracy
        acc = accuracy_score(y_val_fold, y_pred)
        accuracies.append(acc)

        # Sharpe
        signals = (y_pred_proba > 0.5).astype(int)
        strategy_returns = signals * returns_val.reset_index(drop=True)
        sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-8)) * np.sqrt(525_600)
        sharpes.append(sharpe)

    # Average metrics
    avg_accuracy = np.mean(accuracies)
    avg_sharpe = np.mean(sharpes)

    # Normalize Sharpe (target range 0-5, normalize to 0-1)
    normalized_sharpe = min(avg_sharpe / 5.0, 1.0) if avg_sharpe > 0 else 0

    # Combined objective: balance accuracy and Sharpe
    combined_metric = 0.5 * avg_accuracy + 0.5 * normalized_sharpe

    # Log intermediate results
    trial.set_user_attr('accuracy', avg_accuracy)
    trial.set_user_attr('sharpe', avg_sharpe)

    return combined_metric

def main():
    # Load and prepare data
    df = load_data()

    # Engineer features
    print("Engineering features...")
    features_df = engineer_features(df)

    # Create labels
    labels = create_labels(df)

    # Combine
    features_df['label'] = labels
    features_df = features_df.dropna()

    print(f"Dataset: {len(features_df):,} samples")

    # Split
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_idx]

    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']

    # Get returns for Sharpe calculation
    train_returns = df['close'].pct_change()
    train_returns = train_returns.loc[train_df.index]

    print(f"Train set: {len(X_train):,} samples")
    print(f"\nStarting Optuna optimization with 50 trials...")
    print("This will take 5-10 minutes...\n")

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, train_returns),
        n_trials=50,
        show_progress_bar=True,
        n_jobs=1  # XGBoost already uses all cores
    )

    # Results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)

    best_trial = study.best_trial

    print(f"\nBest trial #{best_trial.number}:")
    print(f"  Combined metric: {best_trial.value:.4f}")
    print(f"  Accuracy: {best_trial.user_attrs['accuracy']:.4f} ({best_trial.user_attrs['accuracy']*100:.2f}%)")
    print(f"  Sharpe ratio: {best_trial.user_attrs['sharpe']:.2f}")

    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Save best hyperparameters
    os.makedirs('models', exist_ok=True)

    best_params = best_trial.params.copy()
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = ['logloss', 'auc']
    best_params['tree_method'] = 'hist'
    best_params['n_jobs'] = -1
    best_params['random_state'] = 42

    with open('models/best_hyperparameters.yaml', 'w') as f:
        yaml.dump({
            'hyperparameters': best_params,
            'optimization_date': datetime.now().isoformat(),
            'n_trials': 50,
            'best_accuracy': float(best_trial.user_attrs['accuracy']),
            'best_sharpe': float(best_trial.user_attrs['sharpe']),
            'combined_metric': float(best_trial.value)
        }, f)

    print(f"\nBest hyperparameters saved to models/best_hyperparameters.yaml")

    # Save study for analysis
    with open('models/optuna_study.pkl', 'wb') as f:
        pickle.dump(study, f)

    print(f"Optuna study saved to models/optuna_study.pkl")
    print("\nOptimization complete!")
    print("Next step: python scripts/train_with_best_params.py")

if __name__ == '__main__':
    import os
    main()
