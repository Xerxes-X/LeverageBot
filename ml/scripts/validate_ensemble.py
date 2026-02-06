"""
Validate Multi-Resolution Ensemble Performance

Evaluates the ensemble on validation data and compares to individual models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

from models.ensemble import MultiResolutionEnsemble
from models.gaf_cnn import GAF_CNN


def load_validation_data(window_size: str):
    """Load validation GAF images and labels."""
    data_dir = Path(f'data/gaf/bnb_{window_size}m')

    print(f"Loading {window_size}m validation data...")
    val_images = np.load(data_dir / 'val_images.npy')
    val_labels = np.load(data_dir / 'val_labels.npy')

    print(f"  ✅ {len(val_images):,} images, {len(val_labels):,} labels")

    return val_images, val_labels


def evaluate_individual_model(window_size: str):
    """Evaluate a single model on its validation set."""
    from sklearn.metrics import accuracy_score

    # Load model
    model_path = f'models/gaf_cnn_{window_size}m_v1.pth'
    model = GAF_CNN(pretrained=False, dropout=0.5)

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load validation data
    val_images, val_labels = load_validation_data(window_size)

    # Convert to tensors
    val_images_tensor = torch.from_numpy(val_images).float().permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    val_labels_tensor = torch.from_numpy(val_labels).float()

    # Predict in batches
    batch_size = 64
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(val_images_tensor), batch_size):
            batch = val_images_tensor[i:i+batch_size]
            logits = model(batch)
            probas = torch.sigmoid(logits).squeeze()
            preds = (probas >= 0.5).float()
            all_preds.append(preds)

    all_preds = torch.cat(all_preds).numpy()

    # Compute accuracy
    accuracy = accuracy_score(val_labels, all_preds)

    return accuracy


def main():
    print("="*80)
    print("PHASE 2: MULTI-RESOLUTION ENSEMBLE VALIDATION")
    print("="*80)
    print("")

    # 1. Evaluate individual models
    print("Step 1: Evaluating Individual Models")
    print("-" * 40)

    individual_results = {}

    for window in ['15', '30', '60']:
        acc = evaluate_individual_model(window)
        individual_results[window] = acc
        print(f"  {window}m model accuracy: {acc:.4f} ({acc*100:.2f}%)")

    print("")

    # 2. Load validation data for all windows
    print("Step 2: Loading Validation Data for Ensemble")
    print("-" * 40)

    val_data = {}

    for window in ['15', '30', '60']:
        images, labels = load_validation_data(window)
        val_data[window] = (images, labels)

    # Ensure all have same labels (they should - same data, different windows)
    labels_15 = val_data['15'][1]
    labels_30 = val_data['30'][1]
    labels_60 = val_data['60'][1]

    # Use the shortest sequence (should all be ~20K but just in case)
    min_len = min(len(labels_15), len(labels_30), len(labels_60))

    print(f"\n  Using {min_len:,} validation samples (shortest sequence)")
    print("")

    # 3. Create and evaluate ensemble
    print("Step 3: Evaluating Multi-Resolution Ensemble")
    print("-" * 40)

    ensemble = MultiResolutionEnsemble(model_dir='models', device='cpu')

    # Prepare data
    images_15m = torch.from_numpy(val_data['15'][0][:min_len]).float().permute(0, 3, 1, 2)
    images_30m = torch.from_numpy(val_data['30'][0][:min_len]).float().permute(0, 3, 1, 2)
    images_60m = torch.from_numpy(val_data['60'][0][:min_len]).float().permute(0, 3, 1, 2)
    labels = torch.from_numpy(labels_15[:min_len]).float()

    # Evaluate
    print("\n  Computing ensemble predictions...")
    metrics = ensemble.evaluate(images_15m, images_30m, images_60m, labels)

    # 4. Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print("")

    print("Individual Models:")
    print("-" * 40)
    for window, acc in individual_results.items():
        print(f"  {window}m:  {acc:.4f} ({acc*100:.2f}%)")

    avg_individual = np.mean(list(individual_results.values()))
    print(f"  Avg:  {avg_individual:.4f} ({avg_individual*100:.2f}%)")
    print("")

    print("Multi-Resolution Ensemble:")
    print("-" * 40)
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print("")

    # Improvement
    improvement = metrics['accuracy'] - avg_individual
    improvement_pct = (improvement / avg_individual) * 100

    print(f"  Improvement over avg individual: +{improvement:.4f} ({improvement_pct:+.1f}%)")
    print("")

    # Confusion matrix
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print("Confusion Matrix:")
        print("-" * 40)
        print(f"                Predicted")
        print(f"                DOWN    UP")
        print(f"  Actual DOWN  {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"  Actual UP    {cm[1,0]:6d}  {cm[1,1]:6d}")
        print("")

        if 'true_positives' in metrics:
            print(f"  True Positives:  {metrics['true_positives']:,}")
            print(f"  True Negatives:  {metrics['true_negatives']:,}")
            print(f"  False Positives: {metrics['false_positives']:,}")
            print(f"  False Negatives: {metrics['false_negatives']:,}")
            print("")

    # Target comparison
    print("Phase 2 Targets:")
    print("-" * 40)
    target_acc = 0.90
    print(f"  Target Pattern Recognition: {target_acc:.0%}")
    print(f"  Ensemble Actual:            {metrics['accuracy']:.0%}")

    if metrics['accuracy'] >= target_acc:
        print(f"  ✅ TARGET MET! ({metrics['accuracy']*100:.2f}% >= {target_acc*100:.0f}%)")
    else:
        gap = target_acc - metrics['accuracy']
        print(f"  ⚠️  Gap: {gap:.2%} below target")
        print(f"     Note: Further improvements possible with:")
        print(f"       - Hyperparameter tuning")
        print(f"       - Longer training (remove early stopping)")
        print(f"       - Data augmentation")
        print(f"       - Stacked ensemble with meta-learner")

    print("")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'individual_models': individual_results,
        'ensemble_metrics': metrics,
        'ensemble_weights': ensemble.weights,
        'validation_samples': min_len,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }

    results_file = 'models/ensemble_validation_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"✅ Results saved to: {results_file}")
    print("")

    print("="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
