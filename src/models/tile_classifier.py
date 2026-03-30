"""
Tile-level ResNet50 classifier for Multiple Myeloma histopathology.

Classical baseline: PyTorch ResNet50 (pretrained ImageNet) with custom head.
- Per-tile classification with mean-pooling to slide-level predictions
- Binary and multi-class support
- Mixed-precision training with AMP
- Cosine annealing learning rate schedule
- Early stopping with gradient clipping
- Checkpoint management
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models


class TileClassifier(nn.Module):
    """ResNet50-based tile classifier with custom classification head.

    Attributes:
        backbone: Pretrained ResNet50 feature extractor
        num_classes: Number of output classes (2 for binary, >2 for multi-class)
        feature_dim: Dimension of ResNet50 features (2048)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        """Initialize tile classifier.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to load ImageNet pretrained weights
            dropout: Dropout rate in classification head
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 2048

        # Load pretrained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.resnet50(weights=weights)

        # Remove original classification head
        self.backbone.fc = nn.Identity()

        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Classification logits (B, num_classes)
        """
        features = self.backbone(x)  # (B, 2048)
        logits = self.head(features)  # (B, num_classes)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract ResNet50 features without classification head.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Feature tensor (B, 2048)
        """
        return self.backbone(x)


class TileDataset(Dataset):
    """Simple PyTorch Dataset for tile-level data.

    Assumes tiles are stored as PNG/JPG files with structure:
    data_root/class_name/slide_id/tile_id.png
    """

    def __init__(
        self,
        tile_paths: List[str],
        labels: List[int],
        transform=None,
    ):
        """Initialize dataset.

        Args:
            tile_paths: List of tile image paths
            labels: List of corresponding labels
            transform: Optional torchvision transforms
        """
        self.tile_paths = tile_paths
        self.labels = labels
        self.transform = transform

        assert len(tile_paths) == len(labels), "Mismatch between paths and labels"

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get tile, label, and path.

        Args:
            idx: Index

        Returns:
            (image_tensor, label, tile_path)
        """
        from PIL import Image

        img_path = self.tile_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, img_path


class TileClassifierTrainer:
    """Trainer for tile-level classification.

    Handles:
    - Mixed-precision training
    - Gradient clipping
    - Learning rate scheduling (cosine annealing)
    - Early stopping
    - Checkpoint management
    """

    def __init__(
        self,
        model: TileClassifier,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 50,
        patience: int = 10,
        checkpoint_dir: Optional[str] = None,
    ):
        """Initialize trainer.

        Args:
            model: TileClassifier instance
            device: torch.device (cuda or cpu)
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            max_epochs: Maximum training epochs
            patience: Early stopping patience (epochs)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=1e-6,
        )

        # Mixed precision
        self.scaler = GradScaler()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training DataLoader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for images, labels, _ in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(device_type=self.device.type):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            # Backward with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model.

        Args:
            val_loader: Validation DataLoader

        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast(device_type=self.device.type):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, List]:
        """Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            verbose: Whether to print progress

        Returns:
            Dictionary with training history
        """
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Learning rate scheduling
            self.scheduler.step()

            # Logging
            if verbose:
                print(
                    f"Epoch {epoch + 1:3d}/{self.max_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )

            # Early stopping + checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint.

        Args:
            epoch: Current epoch number
        """
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.best_val_loss = checkpoint["best_val_loss"]


def predict_slide_level(
    model: TileClassifier,
    tile_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, List[str]]:
    """Predict slide-level label from tile predictions.

    Mean-pools tile predictions to get slide-level prediction.

    Args:
        model: Trained TileClassifier
        tile_loader: DataLoader with tiles from a single slide
        device: torch.device

    Returns:
        (slide_label, tile_predictions, tile_paths)
    """
    model.eval()
    tile_probs_list = []
    tile_paths_list = []

    with torch.no_grad():
        for images, _, tile_paths in tile_loader:
            images = images.to(device)

            with autocast(device_type=device.type):
                logits = model(images)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            tile_probs_list.append(probs)
            tile_paths_list.extend(tile_paths)

    # Mean pool tile predictions
    all_tile_probs = np.concatenate(tile_probs_list, axis=0)
    slide_probs = all_tile_probs.mean(axis=0)
    slide_label = slide_probs.argmax()

    return float(slide_label), all_tile_probs, tile_paths_list
