"""
Train GAF-CNN model for specific window size.

Usage:
    python train_gaf_cnn.py --window_size 15
    python train_gaf_cnn.py --window_size 30
    python train_gaf_cnn.py --window_size 60
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
from datetime import datetime
import yaml
import pickle

from gaf.gaf_dataset import GAFDataset
from models.gaf_cnn import GAF_CNN


def calculate_metrics(outputs, labels):
    """Calculate accuracy and other metrics."""
    preds = (torch.sigmoid(outputs) > 0.5).float()
    accuracy = (preds == labels).float().mean()
    return accuracy.item()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_accuracy = 0
    n_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_accuracy += calculate_metrics(outputs, labels)
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_accuracy = total_accuracy / n_batches

    return avg_loss, avg_accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    n_batches = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_accuracy += calculate_metrics(outputs, labels)
            n_batches += 1

            # Store predictions
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / n_batches
    avg_accuracy = total_accuracy / n_batches

    return avg_loss, avg_accuracy, np.array(all_preds), np.array(all_labels)


def main(args):
    print("="*80)
    print(f"TRAINING GAF-CNN FOR {args.window_size}-MINUTE WINDOWS")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cpu':
        print("⚠️  Training on CPU (will take 2-3 hours per model)")
        print("   This is normal - GPU not required")

    # Load datasets
    print(f"\nLoading GAF datasets...")
    data_dir = f'data/gaf/bnb_{args.window_size}m'

    train_dataset = GAFDataset(
        data_path=f'{data_dir}/train.npy',
        mode='train'
    )

    val_dataset = GAFDataset(
        data_path=f'{data_dir}/val.npy',
        mode='val'
    )

    print(f"✅ Train: {len(train_dataset):,} samples")
    print(f"✅ Val:   {len(val_dataset):,} samples")

    # Create dataloaders
    train_loader = train_dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = val_dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches:   {len(val_loader)}")

    # Create model
    print(f"\nCreating model...")
    model = GAF_CNN(
        pretrained=args.pretrained,
        dropout=args.dropout,
        freeze_early_layers=args.freeze_early
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created (ResNet18)")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )

    # Training loop
    print(f"\nStarting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Early stopping patience: {args.patience}")
    print()

    best_val_loss = float('inf')
    best_val_accuracy = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        # Step scheduler
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:2d}/{args.epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            patience_counter = 0

            # Save best model
            model_path = f'models/gaf_cnn_{args.window_size}m_v1.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }, model_path)

            print(f"   ✅ New best model saved! Val Acc: {val_acc:.4f}")

        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⚠️  Early stopping triggered (patience={args.patience})")
                break

    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")

    # Save metadata
    metadata = {
        'window_size': args.window_size,
        'model_type': 'GAF_CNN (ResNet18)',
        'training_date': datetime.now().isoformat(),
        'device': str(device),
        'epochs_trained': epoch,
        'best_val_loss': float(best_val_loss),
        'best_val_accuracy': float(best_val_accuracy),
        'total_training_time_minutes': total_time / 60,
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'dropout': args.dropout,
            'weight_decay': args.weight_decay,
            'freeze_early_layers': args.freeze_early,
            'pretrained': args.pretrained
        },
        'history': history
    }

    metadata_path = f'models/gaf_cnn_{args.window_size}m_v1_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n✅ Metadata saved to {metadata_path}")
    print(f"✅ Model saved to models/gaf_cnn_{args.window_size}m_v1.pth")

    # Pattern recognition assessment
    print(f"\n{'='*80}")
    print(f"PATTERN RECOGNITION ASSESSMENT")
    print(f"{'='*80}")
    print(f"Validation accuracy: {best_val_accuracy*100:.2f}%")
    if best_val_accuracy >= 0.90:
        print(f"✅✅ EXCELLENT - Exceeds 90% target!")
    elif best_val_accuracy >= 0.85:
        print(f"✅ GOOD - Close to 90% target")
    elif best_val_accuracy >= 0.80:
        print(f"⚠️  ACCEPTABLE - Below 90% target but usable")
    else:
        print(f"❌ POOR - Well below 90% target")

    print(f"\nNext steps:")
    print(f"1. Train remaining window sizes (if not done)")
    print(f"2. Create ensemble with XGBoost:")
    print(f"   python scripts/create_ensemble.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAF-CNN model')

    # Data
    parser.add_argument('--window_size', type=int, required=True,
                        choices=[15, 30, 60],
                        help='Window size in minutes')

    # Model
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='Use ImageNet pre-trained weights')
    parser.add_argument('--freeze_early', type=bool, default=True,
                        help='Freeze early layers for transfer learning')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')

    args = parser.parse_args()

    main(args)
