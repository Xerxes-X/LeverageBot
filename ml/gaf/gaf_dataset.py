"""
PyTorch Dataset for GAF images.

Provides efficient data loading for training CNN models.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd


class GAFDataset(Dataset):
    """
    PyTorch Dataset for GAF images.

    Supports:
    - On-the-fly GAF generation
    - Pre-computed GAF images from disk
    - Data augmentation
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        gaf_images: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        transform=None,
        mode='train'
    ):
        """
        Args:
            data_path: Path to saved GAF images (*.npy or directory)
            gaf_images: Pre-loaded GAF images of shape (N, H, W, C)
            labels: Labels of shape (N,)
            transform: Data augmentation transforms
            mode: 'train', 'val', or 'test'
        """
        self.transform = transform
        self.mode = mode

        if gaf_images is not None:
            # Use pre-loaded images
            self.gaf_images = gaf_images
            self.labels = labels
        elif data_path is not None:
            # Load from disk
            self.load_from_disk(data_path)
        else:
            raise ValueError("Must provide either data_path or gaf_images")

        # Validate
        assert len(self.gaf_images) == len(self.labels), \
            f"Mismatch: {len(self.gaf_images)} images vs {len(self.labels)} labels"

    def load_from_disk(self, data_path: str):
        """Load GAF images and labels from disk."""
        data_path = Path(data_path)

        if data_path.suffix == '.npy':
            # Single .npy file
            data = np.load(data_path, allow_pickle=True).item()
            self.gaf_images = data['images']
            self.labels = data['labels']
        elif data_path.is_dir():
            # Directory with separate files
            self.gaf_images = np.load(data_path / 'images.npy')
            self.labels = np.load(data_path / 'labels.npy')
        else:
            raise ValueError(f"Invalid data_path: {data_path}")

    def __len__(self) -> int:
        return len(self.gaf_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (image, label) where:
            - image: Tensor of shape (C, H, W) for CNN
            - label: Tensor of shape (1,) for binary classification
        """
        # Get image and label
        image = self.gaf_images[idx]  # Shape: (H, W, C) or (H, W)
        label = self.labels[idx]

        # Ensure 3D (H, W, C)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        # Apply transforms (augmentation)
        if self.transform is not None and self.mode == 'train':
            image = self.transform(image)

        # Convert to tensor and permute to (C, H, W)
        image = torch.from_numpy(image).float().permute(2, 0, 1)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

    def get_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = None,
        num_workers: int = 4
    ) -> DataLoader:
        """
        Create DataLoader for this dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle (None = auto based on mode)
            num_workers: Number of worker processes

        Returns:
            PyTorch DataLoader
        """
        if shuffle is None:
            shuffle = (self.mode == 'train')

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )


class GAFDatasetFromDF(Dataset):
    """
    PyTorch Dataset that generates GAF images on-the-fly from DataFrame.

    Useful for: - Development/experimentation
    - When GAF images haven't been pre-generated
    - Dynamic window sizes

    Note: Slower than pre-computed images (use for small datasets only)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        window_size: int,
        size: int = 64,
        feature: str = 'close',
        mode: str = 'both',
        transform=None,
        dataset_mode='train'
    ):
        """
        Args:
            df: DataFrame with OHLCV data
            labels: Labels for each window
            window_size: Window size in minutes
            size: Target GAF image size
            feature: Feature to extract from df
            mode: 'gasf', 'gadf', or 'both'
            transform: Data augmentation
            dataset_mode: 'train', 'val', or 'test'
        """
        from .gaf_transformer import extract_windows

        self.windows = extract_windows(df, window_size, feature)
        self.labels = labels
        self.size = size
        self.gaf_mode = mode
        self.transform = transform
        self.dataset_mode = dataset_mode

        assert len(self.windows) == len(self.labels), \
            f"Mismatch: {len(self.windows)} windows vs {len(self.labels)} labels"

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate GAF image on-the-fly and return with label."""
        from .gaf_transformer import generate_gaf

        # Generate GAF
        window = self.windows[idx]
        gaf = generate_gaf(window, size=self.size, mode=self.gaf_mode)

        # Ensure 3D (H, W, C)
        if gaf.ndim == 2:
            gaf = gaf[:, :, np.newaxis]

        # Apply transforms
        if self.transform is not None and self.dataset_mode == 'train':
            gaf = self.transform(gaf)

        # Convert to tensor (C, H, W)
        gaf_tensor = torch.from_numpy(gaf).float().permute(2, 0, 1)

        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return gaf_tensor, label

    def get_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = None,
        num_workers: int = 0  # 0 for on-the-fly generation
    ) -> DataLoader:
        """Create DataLoader."""
        if shuffle is None:
            shuffle = (self.dataset_mode == 'train')

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )


# Utility functions

def save_gaf_dataset(gaf_images: np.ndarray, labels: np.ndarray, save_path: str):
    """
    Save GAF images and labels to disk.

    Args:
        gaf_images: Array of GAF images (N, H, W, C)
        labels: Array of labels (N,)
        save_path: Path to save (*.npy)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'images': gaf_images,
        'labels': labels
    }

    np.save(save_path, data)
    print(f"✅ Saved {len(gaf_images)} GAF images to {save_path}")


def load_gaf_dataset(load_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load GAF images and labels from disk.

    Args:
        load_path: Path to load (*.npy)

    Returns:
        Tuple of (gaf_images, labels)
    """
    data = np.load(load_path, allow_pickle=True).item()
    return data['images'], data['labels']


def create_train_val_split(
    gaf_images: np.ndarray,
    labels: np.ndarray,
    val_split: float = 0.1,
    time_series_split: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Split dataset into train and validation sets.

    Args:
        gaf_images: GAF images
        labels: Labels
        val_split: Validation set fraction
        time_series_split: If True, split chronologically (recommended)

    Returns:
        ((train_images, train_labels), (val_images, val_labels))
    """
    n = len(gaf_images)
    split_idx = int(n * (1 - val_split))

    if time_series_split:
        # Chronological split (no shuffle)
        train_images = gaf_images[:split_idx]
        train_labels = labels[:split_idx]
        val_images = gaf_images[split_idx:]
        val_labels = labels[split_idx:]
    else:
        # Random split
        indices = np.random.permutation(n)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_images = gaf_images[train_indices]
        train_labels = labels[train_indices]
        val_images = gaf_images[val_indices]
        val_labels = labels[val_indices]

    return (train_images, train_labels), (val_images, val_labels)


if __name__ == '__main__':
    # Test dataset
    print("Testing GAFDataset...")

    # Create synthetic data
    n_samples = 100
    img_size = 64
    gaf_images = np.random.randn(n_samples, img_size, img_size, 2).astype(np.float32)
    labels = np.random.randint(0, 2, n_samples).astype(np.float32)

    # Create dataset
    dataset = GAFDataset(gaf_images=gaf_images, labels=labels, mode='train')
    print(f"✅ Dataset created: {len(dataset)} samples")

    # Create dataloader
    dataloader = dataset.get_dataloader(batch_size=16)
    print(f"✅ DataLoader created: {len(dataloader)} batches")

    # Test batch
    for images, labels in dataloader:
        print(f"✅ Batch shape: {images.shape}, Labels: {labels.shape}")
        break

    print("✅ GAFDataset test complete!")
