"""
Generate GAF images for ONE window size with progress bar.
Run separately for each window size to avoid memory/parallel issues.
"""

import numpy as np
import pandas as pd
import sys
import os
import pickle
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaf.gaf_transformer import generate_gaf, extract_windows
from gaf.gaf_dataset import save_gaf_dataset, create_train_val_split


def create_labels(df, window_size, horizon=15):
    future_price = df['close'].shift(-horizon - window_size + 1)
    current_price = df['close']
    future_return = (future_price - current_price) / current_price

    p60 = future_return.quantile(0.60)
    p40 = future_return.quantile(0.40)

    labels = pd.Series(index=df.index, dtype=float)
    labels[future_return >= p60] = 1.0
    labels[future_return <= p40] = 0.0
    return labels.values


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, required=True, choices=[15, 30, 60])
    args = parser.parse_args()

    window_size = args.window_size
    print(f"\n{'#'*70}\n# Generating GAF for {window_size}-minute windows\n{'#'*70}\n")

    # Load data
    print("Loading data...")
    df = pd.read_csv('data/raw/BNBUSDT_1m_180d.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"✅ Loaded {len(df):,} samples\n")

    # Extract windows
    print("Extracting windows...")
    windows = extract_windows(df, window_size, 'close')
    print(f"✅ Extracted {len(windows):,} windows\n")

    # Create labels
    print("Creating labels...")
    all_labels = create_labels(df, window_size)
    window_labels = []
    for i in range(len(windows)):
        label_idx = i + window_size - 1
        if label_idx < len(all_labels):
            window_labels.append(all_labels[label_idx])
        else:
            window_labels.append(np.nan)

    window_labels = np.array(window_labels)
    valid_mask = ~np.isnan(window_labels)
    valid_windows = [windows[i] for i in range(len(windows)) if valid_mask[i]]
    valid_labels = window_labels[valid_mask]

    print(f"✅ Valid windows: {len(valid_windows):,}")
    print(f"   UP: {np.sum(valid_labels==1)/len(valid_labels):.1%}, DOWN: {np.sum(valid_labels==0)/len(valid_labels):.1%}\n")

    # Generate GAF images
    print(f"Generating {len(valid_windows):,} GAF images (64×64)...")
    print("Estimated time: 10-20 minutes\n")

    gaf_images = []
    for window in tqdm(valid_windows, desc="GAF Progress"):
        gaf = generate_gaf(window, size=64, mode='both')
        gaf_images.append(gaf)

    gaf_images = np.array(gaf_images)
    print(f"\n✅ Generated {len(gaf_images):,} images | Shape: {gaf_images.shape} | Size: {gaf_images.nbytes/1024**2:.1f} MB\n")

    # Split and save
    print("Splitting train/val (90/10)...")
    (train_images, train_labels), (val_images, val_labels) = create_train_val_split(
        gaf_images, valid_labels, val_split=0.1, time_series_split=True
    )
    print(f"✅ Train: {len(train_images):,} | Val: {len(val_images):,}\n")

    output_dir = f'data/gaf/bnb_{window_size}m'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving to {output_dir}/...")
    save_gaf_dataset(train_images, train_labels, f'{output_dir}/train.npy')
    save_gaf_dataset(val_images, val_labels, f'{output_dir}/val.npy')

    metadata = {
        'window_size': window_size,
        'gaf_size': 64,
        'train_samples': len(train_images),
        'val_samples': len(val_images),
        'generation_date': datetime.now().isoformat()
    }
    with open(f'{output_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n{'='*70}")
    print(f"✅ COMPLETE! {window_size}m GAF images saved")
    print(f"{'='*70}\n")
