"""
GAF-CNN Model Architecture for Phase 2.

ResNet18-based CNN with transfer learning for GAF image classification.
Modified for 2-channel input (GASF + GADF).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional


class GAF_CNN(nn.Module):
    """
    ResNet18-based CNN for GAF image classification.

    Features:
    - Pre-trained on ImageNet with transfer learning
    - Modified conv1 for 2-channel input (GASF + GADF)
    - Custom classifier head with dropout
    - Binary classification (UP/DOWN)
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.5,
        freeze_early_layers: bool = True
    ):
        """
        Args:
            pretrained: Use ImageNet pre-trained weights
            num_classes: Number of output classes (1 for binary)
            dropout: Dropout probability
            freeze_early_layers: Freeze first 2 layers for transfer learning
        """
        super(GAF_CNN, self).__init__()

        # Load pre-trained ResNet18
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.resnet = resnet18(weights=weights)
        else:
            self.resnet = resnet18(weights=None)

        # Modify first conv layer for 2-channel input (GASF + GADF)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # Modified: Conv2d(2, 64, kernel_size=7, stride=2, padding=3)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            2,  # 2 channels (GASF + GADF)
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Initialize new conv1 weights
        if pretrained:
            # Average the RGB weights to initialize 2-channel weights
            with torch.no_grad():
                # Original weights: (64, 3, 7, 7)
                # New weights: (64, 2, 7, 7)
                self.resnet.conv1.weight[:, 0, :, :] = original_conv1.weight[:, 0:2, :, :].mean(dim=1)
                self.resnet.conv1.weight[:, 1, :, :] = original_conv1.weight[:, 1:3, :, :].mean(dim=1)

        # Freeze early layers for transfer learning
        if freeze_early_layers:
            # Freeze layer1 and layer2 (early feature extractors)
            for param in self.resnet.layer1.parameters():
                param.requires_grad = False
            for param in self.resnet.layer2.parameters():
                param.requires_grad = False

        # Modify classifier head
        num_features = self.resnet.fc.in_features  # 512 for ResNet18
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.6),  # Less dropout in second layer
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 2, height, width)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.resnet(x)

    def unfreeze_all_layers(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (before final classification).

        Args:
            x: Input tensor of shape (batch_size, 2, height, width)

        Returns:
            Embeddings of shape (batch_size, 512)
        """
        # Forward pass through all layers except fc
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class LightweightGAF_CNN(nn.Module):
    """
    Lightweight CNN for faster training (alternative to ResNet18).

    Use this if ResNet18 is too slow on CPU.
    """

    def __init__(self, num_classes: int = 1, dropout: float = 0.4):
        super(LightweightGAF_CNN, self).__init__()

        # Conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(
    model_type: str = 'resnet18',
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create CNN model.

    Args:
        model_type: 'resnet18' or 'lightweight'
        pretrained: Use pre-trained weights (only for resnet18)
        **kwargs: Additional arguments for model

    Returns:
        CNN model
    """
    if model_type == 'resnet18':
        return GAF_CNN(pretrained=pretrained, **kwargs)
    elif model_type == 'lightweight':
        return LightweightGAF_CNN(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == '__main__':
    # Test model
    print("Testing GAF-CNN model...")

    # Create model
    model = GAF_CNN(pretrained=True)
    print(f"✅ Model created")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 2, 64, 64)
    output = model(dummy_input)
    print(f"✅ Forward pass successful")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")

    # Test embeddings
    embeddings = model.get_embeddings(dummy_input)
    print(f"✅ Embeddings extraction successful")
    print(f"   Embeddings shape: {embeddings.shape}")

    print("\n✅ GAF-CNN model test complete!")
