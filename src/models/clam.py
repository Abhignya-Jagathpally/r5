"""
CLAM: Clustering-constrained Attention Multiple Instance Learning.

Implements:
- CLAM-SB (single-branch): standard attention pooling
- CLAM-MB (multi-branch): multi-head attention with instance clustering
- Instance-level pseudo-labeling for interpretability
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


class AttentionLayer(nn.Module):
    """Gated attention mechanism (Ilse et al. 2018, Lu et al. 2021).

    Computes: a = softmax(W · (tanh(V·h) ⊙ sigmoid(U·h)))
    Linear in N (number of instances), not quadratic.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float = 0.0,
    ):
        """Initialize gated attention layer.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (output dim for value projection)
            attention_dim: Internal attention dimension
            dropout: Dropout on attention scores
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Gated attention paths (Ilse et al. 2018)
        self.V = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
        )
        self.U = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.W = nn.Linear(attention_dim, 1)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Value transformation — LayerNorm instead of BatchNorm1d so that
        # single-slide inference (batch_size=1) works correctly.
        self.value = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass per Ilse et al. 2018 / Lu et al. 2021 CLAM paper.

        Args:
            embeddings: (num_tiles, input_dim)

        Returns:
            (aggregated, attention_weights)
              aggregated: (hidden_dim,) — slide-level representation
              attention_weights: (num_tiles,) — per-tile importance
        """
        # Gated attention: a = softmax(W · (tanh(V·h) ⊙ sigmoid(U·h)))
        v = self.V(embeddings)          # (num_tiles, attention_dim)
        u = self.U(embeddings)          # (num_tiles, attention_dim)
        scores = self.W(v * u)          # (num_tiles, 1)  — element-wise gate
        scores = self.attn_dropout(scores)
        attention_weights = torch.softmax(scores.squeeze(-1), dim=0)  # (num_tiles,)

        # Value projection
        h = self.value(embeddings)      # (num_tiles, hidden_dim)

        # Attention-weighted aggregation (O(N), not O(N²))
        aggregated = torch.sum(h * attention_weights.unsqueeze(1), dim=0)  # (hidden_dim,)

        return aggregated, attention_weights


class CLAM_SB(nn.Module):
    """CLAM Single-Branch model.

    Single attention head with instance-level clustering constraint.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.25,
    ):
        """Initialize CLAM-SB.

        Args:
            input_dim: Feature dimension
            hidden_dim: Hidden dimension
            attention_dim: Attention dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Attention mechanism
        self.attention = AttentionLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: (num_tiles, input_dim)
            return_attention: Whether to return attention weights

        Returns:
            If return_attention=False: logits (num_classes,)
            If return_attention=True: (logits, attention_weights)
        """
        aggregated, attention_weights = self.attention(embeddings)

        logits = self.classifier(aggregated)

        if return_attention:
            return logits, attention_weights
        else:
            return logits


class CLAM_MB(nn.Module):
    """CLAM Multi-Branch model.

    Multiple attention heads for interpretability.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        num_classes: int = 2,
        num_heads: int = 3,
        dropout: float = 0.25,
        inst_cluster: bool = True,
    ):
        """Initialize CLAM-MB.

        Args:
            input_dim: Feature dimension
            hidden_dim: Hidden dimension
            attention_dim: Attention dimension
            num_classes: Number of output classes
            num_heads: Number of attention heads
            dropout: Dropout rate
            inst_cluster: Whether to use instance clustering
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.inst_cluster = inst_cluster

        # Multiple attention heads
        self.attention_heads = nn.ModuleList([
            AttentionLayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                attention_dim=attention_dim,
            )
            for _ in range(num_heads)
        ])

        # Classifier
        classifier_input_dim = hidden_dim * num_heads
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # TODO: instance-level clustering loss weight (0.3) is from the CLAM paper
        # defaults but hasn't been validated on MM tissue specifically
        # Instance-level classifier for pseudo-labels
        if inst_cluster:
            self.instance_classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(
        self,
        embeddings: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: (num_tiles, input_dim)
            return_attention: Whether to return attention weights

        Returns:
            If return_attention=False: logits (num_classes,)
            If return_attention=True: (logits, attention_weights_dict)
        """
        # Multi-head attention
        aggregated_list = []
        attention_dict = {}

        for i, attention_head in enumerate(self.attention_heads):
            agg, weights = attention_head(embeddings)
            aggregated_list.append(agg)
            attention_dict[f"head_{i}"] = weights

        # Concatenate all head outputs
        aggregated = torch.cat(aggregated_list, dim=0)

        # Classification
        logits = self.classifier(aggregated)

        if return_attention:
            return logits, attention_dict
        else:
            return logits

    def get_instance_predictions(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Get instance-level predictions for pseudo-labeling.

        Args:
            embeddings: (num_tiles, input_dim)

        Returns:
            Instance logits (num_tiles, num_classes)
        """
        if not self.inst_cluster:
            raise RuntimeError("Instance classifier not enabled")

        return self.instance_classifier(embeddings)


class CLAMTrainer:
    """Trainer for CLAM models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 15,
        inst_lambda: float = 0.0,
        checkpoint_dir: Optional[str] = None,
    ):
        """Initialize trainer.

        Args:
            model: CLAM model (SB or MB)
            device: torch.device
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            max_epochs: Maximum epochs
            patience: Early stopping patience
            inst_lambda: Weight for instance clustering loss
            checkpoint_dir: Checkpoint directory
        """
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.inst_lambda = inst_lambda

        from pathlib import Path
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=1e-6,
        )

        # Mixed precision
        self.scaler = GradScaler()

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch.

        Args:
            train_loader: DataLoader yielding (embeddings, label)

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for embeddings, labels in train_loader:
            batch_size = embeddings.shape[0]
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            total_batch_loss = 0.0

            for b in range(batch_size):
                bag_embeddings = embeddings[b].to(self.device)
                bag_label = labels[b:b+1]

                with autocast(device_type=self.device.type):
                    logits = self.model(bag_embeddings)
                    logits = logits.unsqueeze(0)
                    loss = self.criterion(logits, bag_label)

                    # Instance clustering loss (optional, per Lu et al. 2021)
                    if self.inst_lambda > 0.0 and hasattr(self.model, "get_instance_predictions"):
                        inst_logits = self.model.get_instance_predictions(bag_embeddings)
                        # Soft pseudo-labels from attention weights (not hard bag label)
                        # High-attention tiles get the bag label; low-attention tiles
                        # get the opposite label. This avoids oversupplying supervision.
                        _, attn_weights = self.model(bag_embeddings, return_attention=True)
                        attn_weights = attn_weights.detach()
                        attn_median = attn_weights.median()
                        bag_label_val = bag_label.item()
                        pseudo_labels = torch.where(
                            attn_weights > attn_median,
                            torch.tensor(bag_label_val, device=self.device),
                            torch.tensor(1 - bag_label_val, device=self.device),
                        ).long()
                        inst_loss = self.criterion(inst_logits, pseudo_labels)
                        loss = loss + self.inst_lambda * inst_loss

                total_batch_loss += loss

            (total_batch_loss / batch_size).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += (total_batch_loss / batch_size).item()
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
            for embeddings, labels in val_loader:
                batch_size = embeddings.shape[0]
                labels = labels.to(self.device)

                batch_loss = 0.0

                for b in range(batch_size):
                    bag_embeddings = embeddings[b].to(self.device)
                    bag_label = labels[b:b+1]

                    with autocast(device_type=self.device.type):
                        logits = self.model(bag_embeddings)
                        logits = logits.unsqueeze(0)
                        loss = self.criterion(logits, bag_label)

                    batch_loss += loss.item()

                    pred = logits.argmax(dim=1)
                    correct += (pred == bag_label).sum().item()
                    total += 1

                total_loss += batch_loss

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, any]:
        """Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            verbose: Print progress

        Returns:
            Training history
        """
        for epoch in range(self.max_epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            self.scheduler.step()

            if verbose:
                print(
                    f"Epoch {epoch + 1:3d}/{self.max_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )

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
        """Save model checkpoint to disk."""
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
        """Restore model state from checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.best_val_loss = checkpoint["best_val_loss"]
