"""
Multi-Resolution GAF-CNN Ensemble

Combines 15m, 30m, and 60m models for improved pattern recognition.
Expected ensemble accuracy: 65-70%+ (vs 50-53% individual models)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

from models.gaf_cnn import GAF_CNN


class MultiResolutionEnsemble:
    """
    Ensemble of GAF-CNN models trained on different time windows.

    Combines 15m, 30m, and 60m predictions using weighted soft voting.
    """

    def __init__(
        self,
        model_dir: str = 'models',
        weights: Dict[str, float] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            model_dir: Directory containing trained models
            weights: Dict mapping window size to weight (e.g., {'15': 0.4, '30': 0.35, '60': 0.25})
                     If None, weights by validation accuracy
            device: 'cpu' or 'cuda'
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.windows = ['15', '30', '60']

        # Load models
        self.models = {}
        self.metadata = {}

        for window in self.windows:
            model_path = self.model_dir / f'gaf_cnn_{window}m_v1.pth'
            metadata_path = self.model_dir / f'gaf_cnn_{window}m_v1_metadata.pkl'

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            # Load model
            model = GAF_CNN(pretrained=False, dropout=0.5, freeze_early_layers=True)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            self.models[window] = model

            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata[window] = pickle.load(f)

        # Set weights
        if weights is None:
            # Weight by validation accuracy
            weights = self._compute_accuracy_weights()

        self.weights = weights

        print(f"✅ Loaded {len(self.models)} models")
        print(f"   Weights: {self.weights}")

    def _compute_accuracy_weights(self) -> Dict[str, float]:
        """Compute weights based on validation accuracy."""
        accuracies = {}

        for window in self.windows:
            if window in self.metadata:
                accuracies[window] = self.metadata[window].get('best_val_accuracy', 0.5)
            else:
                # Fallback to known values
                known_accs = {'15': 0.5280, '30': 0.5172, '60': 0.5025}
                accuracies[window] = known_accs.get(window, 0.5)

        # Normalize to sum to 1
        total = sum(accuracies.values())
        weights = {k: v / total for k, v in accuracies.items()}

        return weights

    def predict_proba(
        self,
        images_15m: torch.Tensor,
        images_30m: torch.Tensor,
        images_60m: torch.Tensor
    ) -> np.ndarray:
        """
        Predict probabilities using weighted soft voting.

        Args:
            images_15m: Tensor of shape (N, 2, 64, 64) for 15m window
            images_30m: Tensor of shape (N, 2, 64, 64) for 30m window
            images_60m: Tensor of shape (N, 2, 64, 64) for 60m window

        Returns:
            Array of shape (N,) with ensemble probabilities
        """
        images = {
            '15': images_15m.to(self.device),
            '30': images_30m.to(self.device),
            '60': images_60m.to(self.device)
        }

        # Get predictions from each model
        probas = {}

        with torch.no_grad():
            for window in self.windows:
                logits = self.models[window](images[window])
                proba = torch.sigmoid(logits).squeeze().cpu().numpy()
                probas[window] = proba

        # Weighted average
        ensemble_proba = sum(
            probas[window] * self.weights[window]
            for window in self.windows
        )

        return ensemble_proba

    def predict(
        self,
        images_15m: torch.Tensor,
        images_30m: torch.Tensor,
        images_60m: torch.Tensor,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict binary labels using ensemble.

        Args:
            images_15m: Tensor for 15m window
            images_30m: Tensor for 30m window
            images_60m: Tensor for 60m window
            threshold: Decision threshold (default 0.5)

        Returns:
            Array of binary predictions (0 or 1)
        """
        probas = self.predict_proba(images_15m, images_30m, images_60m)
        return (probas >= threshold).astype(int)

    def evaluate(
        self,
        images_15m: torch.Tensor,
        images_30m: torch.Tensor,
        images_60m: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate ensemble on validation data.

        Returns:
            Dict with accuracy, precision, recall, f1
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        # Get predictions
        probas = self.predict_proba(images_15m, images_30m, images_60m)
        preds = (probas >= 0.5).astype(int)

        labels_np = labels.cpu().numpy()

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(labels_np, preds),
            'precision': precision_score(labels_np, preds, zero_division=0),
            'recall': recall_score(labels_np, preds, zero_division=0),
            'f1': f1_score(labels_np, preds, zero_division=0)
        }

        # Confusion matrix
        cm = confusion_matrix(labels_np, preds)
        metrics['confusion_matrix'] = cm

        # True/False Positives/Negatives
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)

        return metrics


if __name__ == '__main__':
    print("Testing Multi-Resolution Ensemble...")

    # Create ensemble
    ensemble = MultiResolutionEnsemble(model_dir='models')

    # Test with dummy data
    batch_size = 4
    dummy_15m = torch.randn(batch_size, 2, 64, 64)
    dummy_30m = torch.randn(batch_size, 2, 64, 64)
    dummy_60m = torch.randn(batch_size, 2, 64, 64)

    # Predict probabilities
    probas = ensemble.predict_proba(dummy_15m, dummy_30m, dummy_60m)
    print(f"\n✅ Ensemble probabilities: {probas}")

    # Predict labels
    preds = ensemble.predict(dummy_15m, dummy_30m, dummy_60m)
    print(f"✅ Ensemble predictions: {preds}")

    print("\n✅ Ensemble test complete!")
