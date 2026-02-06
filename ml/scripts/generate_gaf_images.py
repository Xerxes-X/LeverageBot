"""
Pre-generate GAF images for all window sizes.

This script:
1. Loads BNBUSDT data (180 days)
2. Creates labels (same percentile-based as Phase 1)
3. Generates GAF images for 15m, 30m, 60m windows
4. Saves to disk for fast loading during training

Estimated time: 30-60 minutes
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaf.gaf_transformer import generate_gaf, extract_windows, batch_generate_gaf
from gaf.gaf_dataset import save_gaf_dataset, create_train_val_split


def load_data():
    """Load BNBUSDT data."""
    print("Loading data...")

    df = pd.read_csv('data/raw/BNBUSDT_1m_180d.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Loaded {len(df):,} samples")
    return df


def create_labels(df, window_size, horizon=15):
    """
    Create percentile-based labels (same as Phase 1).

    Args:
        df: DataFrame with close prices
        window_size: Window size (for offset)
        horizon: Forward-looking horizon in minutes

    Returns:
        Array of labels (1=UP, 0=DOWN, NaN=neutral)
    """
    # Calculate future returns
    future_price = df['close'].shift(-horizon - window_size + 1)
    current_price = df['close']
    future_return = (future_price - current_price) / current_price

    # Percentile-based labeling
    p60 = future_return.quantile(0.60)
    p40 = future_return.quantile(0.40)

    labels = pd.Series(index=df.index, dtype=float)
    labels[future_return >= p60] = 1.0  # UP
    labels[future_return <= p40] = 0.0  # DOWN
    # Middle 20% remains NaN (filtered out)

    return labels.values


def generate_gaf_for_window_size(df, window_size, size=64, feature='close'):
    """
    Generate all GAF images for a specific window size.

    Args:
        df: DataFrame with OHLCV data
        window_size: Window size in minutes
        size: Target GAF image size
        feature: Feature to extract

    Returns:
        Tuple of (gaf_images, labels, valid_indices)
    """
    print(f"\n{'='*70}")
    print(f"Generating GAF images for {window_size}-minute windows")
    print(f"{'='*70}")

    # Extract windows
    print("Extracting windows...")
    windows = extract_windows(df, window_size, feature)
    print(f"Extracted {len(windows):,} windows")

    # Create labels
    print("Creating labels...")
    all_labels = create_labels(df, window_size)

    # Align labels with windows
    # Each window starts at index i, so label for window i is at index i + window_size - 1
    window_labels = []
    for i in range(len(windows)):
        label_idx = i + window_size - 1
        if label_idx < len(all_labels):
            window_labels.append(all_labels[label_idx])
        else:
            window_labels.append(np.nan)

    window_labels = np.array(window_labels)

    # Filter out NaN labels (neutral/missing)
    valid_mask = ~np.isnan(window_labels)
    valid_windows = [windows[i] for i in range(len(windows)) if valid_mask[i]]
    valid_labels = window_labels[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    print(f"Valid windows (after filtering): {len(valid_windows):,}")
    print(f"Class balance: UP={np.sum(valid_labels==1)/len(valid_labels):.1%}, "
          f"DOWN={np.sum(valid_labels==0)/len(valid_labels):.1%}")

    # Generate GAF images in parallel
    print(f"Generating GAF images (size={size}×{size})...")
    print("This will take 10-20 minutes...")

    gaf_images = batch_generate_gaf(
        valid_windows,
        size=size,
        mode='both',  # GASF + GADF
        n_jobs=-1
    )

    print(f"✅ Generated {len(gaf_images):,} GAF images")
    print(f"   Shape: {gaf_images.shape}")
    print(f"   Memory: {gaf_images.nbytes / 1024**2:.1f} MB")

    return gaf_images, valid_labels, valid_indices


def main():
    # Configuration
    WINDOW_SIZES = [15, 30, 60]  # minutes
    GAF_SIZE = 64  # 64×64 pixel images
    VAL_SPLIT = 0.1  # 10% validation set

    # Create output directories
    os.makedirs('data/gaf', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load data
    df = load_data()

    # Process each window size
    for window_size in WINDOW_SIZES:
        print(f"\n{'#'*70}")
        print(f"# PROCESSING {window_size}-MINUTE WINDOWS")
        print(f"{'#'*70}")

        # Generate GAF images
        gaf_images, labels, valid_indices = generate_gaf_for_window_size(
            df, window_size, size=GAF_SIZE
        )

        # Train/validation split (chronological)
        print(f"\nSplitting into train/val ({1-VAL_SPLIT:.0%}/{VAL_SPLIT:.0%})...")
        (train_images, train_labels), (val_images, val_labels) = create_train_val_split(
            gaf_images, labels, val_split=VAL_SPLIT, time_series_split=True
        )

        print(f"Train: {len(train_images):,} samples")
        print(f"Val:   {len(val_images):,} samples")

        # Save datasets
        output_dir = f'data/gaf/bnb_{window_size}m'
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving to {output_dir}/...")

        # Save train
        save_gaf_dataset(
            train_images,
            train_labels,
            f'{output_dir}/train.npy'
        )

        # Save validation
        save_gaf_dataset(
            val_images,
            val_labels,
            f'{output_dir}/val.npy'
        )

        # Save metadata
        metadata = {
            'window_size': window_size,
            'gaf_size': GAF_SIZE,
            'feature': 'close',
            'mode': 'both',  # GASF + GADF
            'train_samples': len(train_images),
            'val_samples': len(val_images),
            'train_class_balance': {
                'UP': float(np.sum(train_labels==1) / len(train_labels)),
                'DOWN': float(np.sum(train_labels==0) / len(train_labels))
            },
            'val_class_balance': {
                'UP': float(np.sum(val_labels==1) / len(val_labels)),
                'DOWN': float(np.sum(val_labels==0) / len(val_labels))
            },
            'valid_indices': valid_indices.tolist(),
            'generation_date': datetime.now().isoformat()
        }

        with open(f'{output_dir}/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print(f"✅ Metadata saved to {output_dir}/metadata.pkl")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for window_size in WINDOW_SIZES:
        output_dir = f'data/gaf/bnb_{window_size}m'

        with open(f'{output_dir}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        print(f"\n{window_size}-minute windows:")
        print(f"  Train: {metadata['train_samples']:,} samples")
        print(f"  Val:   {metadata['val_samples']:,} samples")
        print(f"  Balance: {metadata['train_class_balance']['UP']:.1%} UP, "
              f"{metadata['train_class_balance']['DOWN']:.1%} DOWN")

    print(f"\n{'='*70}")
    print("✅ GAF IMAGE GENERATION COMPLETE!")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Review generated images:")
    print("   python scripts/visualize_gaf.py")
    print("\n2. Train CNN models:")
    print("   python scripts/train_gaf_cnn.py --window_size 15")
    print("   python scripts/train_gaf_cnn.py --window_size 30")
    print("   python scripts/train_gaf_cnn.py --window_size 60")


if __name__ == '__main__':
    import time
    start_time = time.time()

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed/60:.1f} minutes")
