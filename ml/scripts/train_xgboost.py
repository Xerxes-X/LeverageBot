"""
Train XGBoost model for price direction prediction.

Based on peer-reviewed research:
- MDPI 2025: XGBoost outperforms deep learning for BTC prediction
- arXiv 2407.11786: Technical indicators + XGBoost for crypto
- AIMS Press 2025: XGBoost achieves 1.35 Sharpe ratio for long-only

Usage:
    python train_xgboost.py --config configs/xgboost_baseline.yaml
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import yaml
import os
import sys
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report
)
import xgboost as xgb
import mlflow
import mlflow.xgboost

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_transformer import FeatureTransformer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Loaded {len(df):,} samples")
    return df


def create_labels(df: pd.DataFrame, horizon: int = 15, threshold: float = 0.001) -> pd.Series:
    """
    Create binary labels for price direction.

    Args:
        df: DataFrame with 'close' column
        horizon: Forward-looking horizon in periods (e.g., 15 minutes)
        threshold: Minimum return to be considered "up" (e.g., 0.001 = 0.1%)

    Returns:
        Binary labels (1 = up, 0 = down)
    """
    future_price = df['close'].shift(-horizon)
    future_return = (future_price - df['close']) / df['close']

    # Binary classification: 1 if return > threshold, else 0
    labels = (future_return > threshold).astype(int)

    return labels


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Engineer features from raw OHLCV data.

    Args:
        df: Raw OHLCV DataFrame
        config: Feature configuration

    Returns:
        DataFrame with engineered features
    """
    print("Engineering features...")

    # Create feature transformer
    transformer = FeatureTransformer(config)

    # Transform data
    transformer.fit(df)
    features_df = transformer.transform(df)

    # Save transformer for later use
    transformer.save_state('models/feature_transformer_state.pkl')

    # Save feature names
    with open('models/feature_names.txt', 'w') as f:
        f.write('\n'.join(transformer.get_feature_names()))

    print(f"Engineered {len(transformer.get_feature_names())} features")

    return features_df, transformer


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict
) -> xgb.XGBClassifier:
    """
    Train XGBoost model with early stopping.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Model configuration

    Returns:
        Trained XGBoost model
    """
    print("Training XGBoost model...")

    params = config['model']['params'].copy()
    early_stopping = config['model'].get('early_stopping_rounds', 50)

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping,
        verbose=True
    )

    print(f"Best iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score:.4f}")

    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dict of evaluation metrics
    """
    print("\nEvaluating model...")

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'logloss': log_loss(y_test, y_pred_proba),
        'win_rate': accuracy_score(y_test, y_pred)  # Same as accuracy for binary
    }

    # Print results
    print("\n=== Model Performance ===")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    return metrics


def calculate_sharpe_ratio(
    predictions: np.ndarray,
    labels: np.ndarray,
    returns: pd.Series,
    threshold: float = 0.5
) -> float:
    """
    Calculate Sharpe ratio of trading strategy.

    Strategy: Long if prediction > threshold, else flat.

    Args:
        predictions: Model probability predictions
        labels: True labels (unused, for reference)
        returns: Actual returns
        threshold: Prediction threshold for trading

    Returns:
        Annualized Sharpe ratio
    """
    # Trading signals (1 = long, 0 = no position)
    signals = (predictions > threshold).astype(int)

    # Strategy returns
    strategy_returns = signals * returns

    # Sharpe ratio (annualized for 1-minute data)
    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()

    if std_return == 0:
        return 0.0

    # Annualization factor: 525,600 minutes per year
    sharpe = (mean_return / std_return) * np.sqrt(525_600)

    return sharpe


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    config: dict,
    n_splits: int = 5
) -> dict:
    """
    Perform time series cross-validation.

    Args:
        X: Features
        y: Labels
        returns: Actual returns (for Sharpe calculation)
        config: Model configuration
        n_splits: Number of CV splits

    Returns:
        Dict with CV results
    """
    print(f"\nPerforming {n_splits}-fold time series cross-validation...")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_metrics = {
        'accuracy': [],
        'auc': [],
        'sharpe': []
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}/{n_splits}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        returns_val = returns.iloc[val_idx]

        # Train model
        model = train_model(X_train, y_train, X_val, y_val, config)

        # Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        sharpe = calculate_sharpe_ratio(y_pred_proba, y_val, returns_val)

        cv_metrics['accuracy'].append(acc)
        cv_metrics['auc'].append(auc)
        cv_metrics['sharpe'].append(sharpe)

        print(f"Fold {fold} - Accuracy: {acc:.4f}, AUC: {auc:.4f}, Sharpe: {sharpe:.4f}")

    # Summary
    print("\n=== Cross-Validation Summary ===")
    for metric_name, values in cv_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric_name}: {mean:.4f} ± {std:.4f}")

    return cv_metrics


def save_model(model: xgb.XGBClassifier, config: dict, metrics: dict, output_path: str):
    """Save trained model and metadata."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    # Save metadata
    metadata = {
        'model_name': config['model']['name'],
        'model_version': config['model']['version'],
        'training_date': datetime.now().isoformat(),
        'metrics': metrics,
        'config': config
    }

    metadata_path = output_path.replace('.pkl', '_metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)

    print(f"\nModel saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser.add_argument('--config', type=str, default='configs/xgboost_baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to training data (overrides config)')
    parser.add_argument('--cv', action='store_true',
                        help='Perform cross-validation')
    parser.add_argument('--output', type=str, default='models/xgboost_phase1_v1.pkl',
                        help='Output path for trained model')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set up MLflow
    mlflow_config = config.get('mlflow', {})
    mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'experiments/'))
    mlflow.set_experiment(mlflow_config.get('experiment_name', 'xgboost_phase1'))

    with mlflow.start_run():
        # Log config
        mlflow.log_params({
            'model_name': config['model']['name'],
            'max_depth': config['model']['params']['max_depth'],
            'learning_rate': config['model']['params']['learning_rate'],
            'n_estimators': config['model']['params']['n_estimators']
        })

        # Load data
        data_path = args.data or f"data/raw/{config['data']['symbol']}_{config['data']['interval']}_{config['data']['lookback_days']}d.csv"
        df = load_data(data_path)

        # Create labels
        label_config = config['data']['label']
        horizon = int(label_config['horizon'].replace('m', ''))  # Extract minutes
        labels = create_labels(df, horizon=horizon, threshold=label_config['threshold'])

        # Engineer features
        features_df, transformer = engineer_features(df, config)

        # Combine features and labels
        features_df['label'] = labels
        features_df = features_df.dropna()  # Remove rows with NaN

        print(f"\nFinal dataset: {len(features_df):,} samples")
        print(f"Class distribution: {features_df['label'].value_counts().to_dict()}")

        # Split data
        split_idx = int(len(features_df) * config['data']['train_test_split'])

        train_df = features_df.iloc[:split_idx]
        test_df = features_df.iloc[split_idx:]

        X_train = train_df.drop('label', axis=1)
        y_train = train_df['label']
        X_test = test_df.drop('label', axis=1)
        y_test = test_df['label']

        print(f"Train set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")

        # Cross-validation (optional)
        if args.cv:
            # Calculate returns for Sharpe ratio
            returns = df['close'].pct_change().iloc[:len(features_df)]
            returns = returns[features_df.index]  # Align with features

            cv_results = cross_validate(
                X_train.iloc[:len(X_train)//2],  # Use first half for CV
                y_train.iloc[:len(y_train)//2],
                returns.iloc[:len(returns)//2],
                config,
                n_splits=config['training']['cv_splits']
            )
            mlflow.log_metrics({
                'cv_accuracy_mean': np.mean(cv_results['accuracy']),
                'cv_auc_mean': np.mean(cv_results['auc']),
                'cv_sharpe_mean': np.mean(cv_results['sharpe'])
            })

        # Train final model
        val_split = config['data']['validation_split']
        val_idx = int(len(X_train) * (1 - val_split))

        X_train_final = X_train.iloc[:val_idx]
        y_train_final = y_train.iloc[:val_idx]
        X_val = X_train.iloc[val_idx:]
        y_val = y_train.iloc[val_idx:]

        model = train_model(X_train_final, y_train_final, X_val, y_val, config)

        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test)

        # Calculate Sharpe ratio on test set
        test_returns = df['close'].pct_change().iloc[split_idx:split_idx+len(X_test)]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        sharpe = calculate_sharpe_ratio(y_pred_proba, y_test, test_returns.reset_index(drop=True))
        metrics['sharpe_ratio'] = sharpe

        print(f"\nTest Sharpe Ratio: {sharpe:.4f}")

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n=== Top 10 Features ===")
        print(feature_importance.head(10).to_string(index=False))

        # Save feature importance
        feature_importance.to_csv('models/feature_importance.csv', index=False)
        mlflow.log_artifact('models/feature_importance.csv')

        # Save model
        save_model(model, config, metrics, args.output)
        mlflow.xgboost.log_model(model, "model")

        # Check targets
        targets = config.get('targets', {})
        print("\n=== Target Achievement ===")
        print(f"Win Rate Target: {targets.get('win_rate_min', 0.58):.2%} - {targets.get('win_rate_max', 0.62):.2%}")
        print(f"Actual Win Rate: {metrics['win_rate']:.2%}")
        print(f"Sharpe Target: {targets.get('sharpe_ratio_min', 1.5):.2f} - {targets.get('sharpe_ratio_max', 2.0):.2f}")
        print(f"Actual Sharpe: {sharpe:.2f}")

        if metrics['win_rate'] >= targets.get('win_rate_min', 0.58) and \
           sharpe >= targets.get('sharpe_ratio_min', 1.5):
            print("\n✅ Phase 1 targets achieved!")
        else:
            print("\n⚠️  Phase 1 targets not yet met. Consider:")
            print("  - Hyperparameter tuning with Optuna")
            print("  - Feature selection (Boruta, SHAP)")
            print("  - Collecting more training data")

    print("\nTraining complete!")
    print(f"Model: {args.output}")
    print(f"Next step: python scripts/backtest.py --model {args.output}")


if __name__ == '__main__':
    main()
