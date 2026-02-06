"""
Memory-optimized GAF generation - pre-allocates array to avoid OOM.

Key fix: Pre-allocate numpy array instead of building list then converting.
"""
import numpy as np
import pandas as pd
import sys, os, pickle
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaf.gaf_transformer import generate_gaf, extract_windows

def create_labels(df, window_size, horizon=15):
    future_price = df['close'].shift(-horizon - window_size + 1)
    current_price = df['close']
    future_return = (future_price - current_price) / current_price
    p60, p40 = future_return.quantile(0.60), future_return.quantile(0.40)
    labels = pd.Series(index=df.index, dtype=float)
    labels[future_return >= p60] = 1.0
    labels[future_return <= p40] = 0.0
    return labels.values

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, required=True, choices=[15, 30, 60])
    args = parser.parse_args()

    ws = args.window_size
    GAF_SIZE = 64
    CHANNELS = 2

    print(f"\n{'#'*70}\n# GAF Generation: {ws}-minute (MEMORY-OPTIMIZED)\n{'#'*70}\n")

    # Load data
    df = pd.read_csv('data/raw/BNBUSDT_1m_180d.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Extract windows and labels
    windows = extract_windows(df, ws, 'close')
    all_labels = create_labels(df, ws)

    window_labels = [all_labels[i + ws - 1] if (i + ws - 1) < len(all_labels) else np.nan
                     for i in range(len(windows))]
    window_labels = np.array(window_labels)
    valid_mask = ~np.isnan(window_labels)
    valid_windows = [windows[i] for i in range(len(windows)) if valid_mask[i]]
    valid_labels = window_labels[valid_mask]

    n_valid = len(valid_windows)
    print(f"✅ {n_valid:,} valid windows | UP: {np.sum(valid_labels==1)/len(valid_labels):.1%}\n")

    # PRE-ALLOCATE numpy array (KEY FIX - avoids OOM)
    print(f"Pre-allocating array: ({n_valid}, {GAF_SIZE}, {GAF_SIZE}, {CHANNELS})")
    mem_gb = (n_valid * GAF_SIZE * GAF_SIZE * CHANNELS * 8) / (1024**3)
    print(f"Memory required: {mem_gb:.2f} GB\n")

    gaf_images = np.empty((n_valid, GAF_SIZE, GAF_SIZE, CHANNELS), dtype=np.float64)
    print(f"✅ Array allocated\n")

    # Generate and fill directly into pre-allocated array
    CHUNK_SIZE = 10000
    n_chunks = (n_valid + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"Generating GAF in {n_chunks} chunks of {CHUNK_SIZE}...\n")

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, n_valid)
        chunk_windows = valid_windows[start_idx:end_idx]

        print(f"Chunk {chunk_idx+1}/{n_chunks}: {len(chunk_windows):,} images")

        # Generate and store directly
        for i, window in enumerate(tqdm(chunk_windows, desc=f"Chunk {chunk_idx+1}")):
            gaf = generate_gaf(window, size=GAF_SIZE, mode='both')
            gaf_images[start_idx + i] = gaf

        print(f"  ✅ Chunk {chunk_idx+1} complete | Progress: {end_idx:,}/{n_valid:,} ({end_idx/n_valid*100:.1f}%)\n")

    print(f"✅ All GAF generated | Shape: {gaf_images.shape}\n")

    # Split train/val
    split_idx = int(n_valid * 0.9)
    print(f"Splitting train/val (90/10)...")
    train_images = gaf_images[:split_idx]
    train_labels = valid_labels[:split_idx]
    val_images = gaf_images[split_idx:]
    val_labels = valid_labels[split_idx:]
    print(f"✅ Train: {len(train_images):,} | Val: {len(val_images):,}\n")

    # Save
    output_dir = f'data/gaf/bnb_{ws}m'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving to {output_dir}/...")
    # Save arrays separately (avoids pickle protocol issue with >4GB objects)
    np.save(f'{output_dir}/train_images.npy', train_images)
    print(f"  ✅ train_images.npy saved ({train_images.nbytes/1024**2:.1f} MB)")

    np.save(f'{output_dir}/train_labels.npy', train_labels)
    print(f"  ✅ train_labels.npy saved")

    np.save(f'{output_dir}/val_images.npy', val_images)
    print(f"  ✅ val_images.npy saved ({val_images.nbytes/1024**2:.1f} MB)")

    np.save(f'{output_dir}/val_labels.npy', val_labels)
    print(f"  ✅ val_labels.npy saved")

    metadata = {
        'window_size': ws,
        'gaf_size': GAF_SIZE,
        'train_samples': len(train_images),
        'val_samples': len(val_images),
        'generation_date': datetime.now().isoformat()
    }
    with open(f'{output_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  ✅ metadata.pkl saved\n")

    print(f"{'='*70}\n✅ COMPLETE! {ws}m GAF images saved to {output_dir}/\n{'='*70}\n")
