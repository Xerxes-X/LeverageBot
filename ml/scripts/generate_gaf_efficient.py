"""Memory-efficient GAF generation - saves in chunks to avoid OOM"""
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
    print(f"\n{'#'*70}\n# GAF Generation: {ws}-minute windows\n{'#'*70}\n")

    # Load & prepare
    df = pd.read_csv('data/raw/BNBUSDT_1m_180d.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    windows = extract_windows(df, ws, 'close')
    all_labels = create_labels(df, ws)

    window_labels = [all_labels[i + ws - 1] if (i + ws - 1) < len(all_labels) else np.nan
                     for i in range(len(windows))]
    window_labels = np.array(window_labels)
    valid_mask = ~np.isnan(window_labels)
    valid_windows = [windows[i] for i in range(len(windows)) if valid_mask[i]]
    valid_labels = window_labels[valid_mask]

    print(f"✅ {len(valid_windows):,} valid windows | UP: {np.sum(valid_labels==1)/len(valid_labels):.1%}\n")

    # Generate GAF in chunks to avoid memory issues
    CHUNK_SIZE = 10000
    n_chunks = (len(valid_windows) + CHUNK_SIZE - 1) // CHUNK_SIZE

    all_gaf = []
    print(f"Generating GAF in {n_chunks} chunks of {CHUNK_SIZE}...")

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, len(valid_windows))
        chunk_windows = valid_windows[start_idx:end_idx]

        print(f"\nChunk {chunk_idx+1}/{n_chunks}: processing {len(chunk_windows):,} images...")
        chunk_gaf = [generate_gaf(w, size=64, mode='both') for w in tqdm(chunk_windows, desc=f"Chunk {chunk_idx+1}")]
        all_gaf.extend(chunk_gaf)

        print(f"  ✅ Chunk complete | Total so far: {len(all_gaf):,}")

    gaf_images = np.array(all_gaf)
    print(f"\n✅ All GAF generated | Shape: {gaf_images.shape} | Size: {gaf_images.nbytes/1024**2:.1f} MB\n")

    # Split & save
    split_idx = int(len(gaf_images) * 0.9)
    train_images, train_labels = gaf_images[:split_idx], valid_labels[:split_idx]
    val_images, val_labels = gaf_images[split_idx:], valid_labels[split_idx:]

    output_dir = f'data/gaf/bnb_{ws}m'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving to {output_dir}/...")
    np.save(f'{output_dir}/train.npy', {'images': train_images, 'labels': train_labels})
    np.save(f'{output_dir}/val.npy', {'images': val_images, 'labels': val_labels})

    metadata = {'window_size': ws, 'gaf_size': 64, 'train_samples': len(train_images),
                'val_samples': len(val_images), 'generation_date': datetime.now().isoformat()}
    with open(f'{output_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n{'='*70}\n✅ COMPLETE! {ws}m saved | Train: {len(train_images):,} | Val: {len(val_images):,}\n{'='*70}\n")
