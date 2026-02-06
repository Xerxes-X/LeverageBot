"""
Train GAF-CNN with ON-THE-FLY GAF generation.

This avoids pre-generation issues and generates GAF images during training.
Training time: 3-4 hours per model (acceptable for overnight training).

Usage:
    python train_gaf_cnn_onthefly.py --window_size 15
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from tqdm import tqdm

from gaf.gaf_dataset import GAFDatasetFromDF
from models.gaf_cnn import GAF_CNN


def calculate_metrics(outputs, labels):
    preds = (torch.sigmoid(outputs) > 0.5).float()
    accuracy = (preds == labels).float().mean()
    return accuracy.item()


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_accuracy, n_batches = 0, 0, 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += calculate_metrics(outputs, labels)
        n_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{calculate_metrics(outputs, labels):.4f}'})

    return total_loss / n_batches, total_accuracy / n_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_accuracy, n_batches = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_accuracy += calculate_metrics(outputs, labels)
            n_batches += 1

    return total_loss / n_batches, total_accuracy / n_batches


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


def main(args):
    print("="*80)
    print(f"TRAINING GAF-CNN: {args.window_size}-min (ON-THE-FLY GENERATION)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cpu':
        print("⚠️  Training on CPU (3-4 hours per model)")
        print("   GAF images generated on-the-fly during training\n")

    # Load data
    print("Loading data...")
    df = pd.read_csv('data/raw/BNBUSDT_1m_180d.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"✅ Loaded {len(df):,} samples\n")

    # Create labels
    print("Creating labels...")
    labels = create_labels(df, args.window_size)

    # Align labels with windows
    from gaf.gaf_transformer import extract_windows
    windows = extract_windows(df, args.window_size, 'close')
    window_labels = []
    for i in range(len(windows)):
        label_idx = i + args.window_size - 1
        if label_idx < len(labels):
            window_labels.append(labels[label_idx])
        else:
            window_labels.append(np.nan)

    window_labels = np.array(window_labels)
    valid_mask = ~np.isnan(window_labels)
    valid_labels = window_labels[valid_mask]

    # Filter df to valid windows
    df_valid = df.iloc[:len(windows)][valid_mask].reset_index(drop=True)
    valid_labels = valid_labels[:len(df_valid)]

    print(f"✅ Valid windows: {len(df_valid):,}")
    print(f"   UP: {np.sum(valid_labels==1)/len(valid_labels):.1%}, DOWN: {np.sum(valid_labels==0)/len(valid_labels):.1%}\n")

    # Split train/val
    split_idx = int(len(df_valid) * 0.9)
    train_df = df_valid.iloc[:split_idx]
    train_labels = valid_labels[:split_idx]
    val_df = df_valid.iloc[split_idx:]
    val_labels = valid_labels[split_idx:]

    print(f"Train: {len(train_df):,} samples")
    print(f"Val:   {len(val_df):,} samples\n")

    # Create datasets with on-the-fly GAF generation
    print("Creating datasets (GAF generated on-the-fly)...")
    train_dataset = GAFDatasetFromDF(
        df=train_df,
        labels=train_labels,
        window_size=args.window_size,
        size=64,
        feature='close',
        mode='both',
        dataset_mode='train'
    )

    val_dataset = GAFDatasetFromDF(
        df=val_df,
        labels=val_labels,
        window_size=args.window_size,
        size=64,
        feature='close',
        mode='both',
        dataset_mode='val'
    )

    train_loader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = val_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"✅ Datasets created\n")

    # Create model
    print("Creating model...")
    model = GAF_CNN(pretrained=args.pretrained, dropout=args.dropout, freeze_early_layers=args.freeze_early)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model: ResNet18 | Params: {trainable_params:,} trainable / {total_params:,} total\n")

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)

    print(f"Starting training...")
    print(f"Epochs: {args.epochs} | LR: {args.learning_rate} | Batch: {args.batch_size} | Patience: {args.patience}\n")

    best_val_loss = float('inf')
    best_val_accuracy = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    import time
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")

        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} Summary ({epoch_time/60:.1f} min):")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            patience_counter = 0

            model_path = f'models/gaf_cnn_{args.window_size}m_v1.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }, model_path)

            print(f"  ✅ New best model saved! (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⚠️  Early stopping (patience={args.patience})")
                break

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Best val accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")

    # Save metadata
    metadata = {
        'window_size': args.window_size,
        'model_type': 'GAF_CNN_OnTheFly',
        'training_date': datetime.now().isoformat(),
        'device': str(device),
        'epochs_trained': epoch,
        'best_val_accuracy': float(best_val_accuracy),
        'training_time_hours': total_time / 3600,
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'dropout': args.dropout,
        },
        'history': history
    }

    with open(f'models/gaf_cnn_{args.window_size}m_v1_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n✅ Model: models/gaf_cnn_{args.window_size}m_v1.pth")
    print(f"✅ Metadata: models/gaf_cnn_{args.window_size}m_v1_metadata.pkl")

    if best_val_accuracy >= 0.90:
        print(f"\n✅✅ EXCELLENT - {best_val_accuracy*100:.2f}% exceeds 90% target!")
    elif best_val_accuracy >= 0.85:
        print(f"\n✅ GOOD - {best_val_accuracy*100:.2f}% close to 90% target")
    else:
        print(f"\n⚠️  ACCEPTABLE - {best_val_accuracy*100:.2f}% below target but usable")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, required=True, choices=[15, 30, 60])
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--freeze_early', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)  # Smaller for on-the-fly
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()
    main(args)
