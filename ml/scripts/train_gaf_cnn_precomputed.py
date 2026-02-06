"""
Train GAF-CNN using PRE-COMPUTED GAF images.

Faster than on-the-fly generation since images are already created.
Training time: ~2-3 hours per model on CPU.

Usage:
    python train_gaf_cnn_precomputed.py --window_size 15
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm

from gaf.gaf_dataset import GAFDataset
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


def main(args):
    print("="*80)
    print(f"TRAINING GAF-CNN: {args.window_size}-min (PRE-COMPUTED IMAGES)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cpu':
        print("⚠️  Training on CPU (~2-3 hours with pre-computed images)")
        print("   Images loaded from disk (no GAF generation needed)\\n")

    # Load pre-computed GAF images
    data_dir = f'data/gaf/bnb_{args.window_size}m'
    print(f"Loading pre-computed images from {data_dir}/...")

    train_images = np.load(f'{data_dir}/train_images.npy')
    train_labels = np.load(f'{data_dir}/train_labels.npy')
    val_images = np.load(f'{data_dir}/val_images.npy')
    val_labels = np.load(f'{data_dir}/val_labels.npy')

    print(f"✅ Train: {len(train_images):,} images ({train_images.nbytes/1024**3:.2f} GB)")
    print(f"✅ Val:   {len(val_images):,} images ({val_images.nbytes/1024**3:.2f} GB)")
    print(f"   Image shape: {train_images.shape[1:]} (64x64x2 channels)")
    print(f"   UP: {np.sum(train_labels==1)/len(train_labels):.1%}, DOWN: {np.sum(train_labels==0)/len(train_labels):.1%}\\n")

    # Create datasets
    print("Creating PyTorch datasets...")
    train_dataset = GAFDataset(
        gaf_images=train_images,
        labels=train_labels,
        mode='train'
    )

    val_dataset = GAFDataset(
        gaf_images=val_images,
        labels=val_labels,
        mode='val'
    )

    train_loader = train_dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # 0 for simplicity on CPU
    )
    val_loader = val_dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"✅ Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}\\n")

    # Create model
    print("Creating model...")
    model = GAF_CNN(pretrained=args.pretrained, dropout=args.dropout, freeze_early_layers=args.freeze_early)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model: ResNet18 | Params: {trainable_params:,} trainable / {total_params:,} total\\n")

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)

    print(f"Starting training...")
    print(f"Epochs: {args.epochs} | LR: {args.learning_rate} | Batch: {args.batch_size} | Patience: {args.patience}\\n")

    best_val_loss = float('inf')
    best_val_accuracy = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    import time
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")

        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"\\nEpoch {epoch} Summary ({epoch_time/60:.1f} min):")
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
                print(f"\\n⚠️  Early stopping (patience={args.patience})")
                break

    total_time = time.time() - start_time
    print(f"\\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Best val accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")

    # Save metadata
    metadata = {
        'window_size': args.window_size,
        'model_type': 'GAF_CNN_PreComputed',
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

    print(f"\\n✅ Model: models/gaf_cnn_{args.window_size}m_v1.pth")
    print(f"✅ Metadata: models/gaf_cnn_{args.window_size}m_v1_metadata.pkl")

    if best_val_accuracy >= 0.90:
        print(f"\\n✅✅ EXCELLENT - {best_val_accuracy*100:.2f}% exceeds 90% target!")
    elif best_val_accuracy >= 0.85:
        print(f"\\n✅ GOOD - {best_val_accuracy*100:.2f}% close to 90% target")
    else:
        print(f"\\n⚠️  ACCEPTABLE - {best_val_accuracy*100:.2f}% below target but usable")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, required=True, choices=[15, 30, 60])
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--freeze_early', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)  # Larger batch since no on-the-fly generation
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()
    main(args)
