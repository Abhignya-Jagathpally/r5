"""
Multimodal fusion strategies for combining imaging, radiomics, and genomic features.

Implements multiple fusion architectures:
- Early fusion: Concatenate features before classification
- Late fusion: Separate classifiers per modality, combine predictions
- Cross-attention fusion: Bidirectional attention between modalities
- Gated fusion: Learn per-sample modality weights

All fusion methods output interpretable attention weights showing modality contributions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultimodalFusion(ABC, nn.Module):
    """
    Abstract base class for multimodal fusion methods.

    All implementations must:
    - Accept multiple modality feature vectors
    - Return fused features and per-modality attention weights
    - Support optional modality masks (e.g., missing modalities)
    """

    def __init__(self, modality_dims: Dict[str, int], num_classes: int):
        """
        Initialize fusion module.

        Args:
            modality_dims: Dictionary mapping modality names to feature dimensions
                Example: {'imaging': 1536, 'radiomics': 128, 'genomic': 256}
            num_classes: Number of output classes
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.num_classes = num_classes
        self.modality_names = list(modality_dims.keys())

    @abstractmethod
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            features: Dictionary mapping modality names to feature tensors
                Shape for each modality: (B, feature_dim)
            modality_mask: Optional dictionary of boolean masks indicating
                valid modalities (True = available, False = missing)
                Shape: (B,) for each modality

        Returns:
            - logits: Classification logits of shape (B, num_classes)
            - outputs: Dictionary with:
                - 'fused_features': Fused representation (B, fused_dim)
                - 'modality_weights': Attention weights per modality (B, num_modalities)
                - Additional modality-specific outputs
        """
        pass


class EarlyFusion(MultimodalFusion):
    """
    Early fusion: Concatenate all modality features, then classify.

    Simple baseline that treats all modalities equally and learns
    their joint representation from concatenation.
    """

    def __init__(self, modality_dims: Dict[str, int], num_classes: int):
        """Initialize early fusion."""
        super().__init__(modality_dims, num_classes)

        # Compute total fused dimension
        total_dim = sum(modality_dims.values())

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: concatenate and classify.

        Args:
            features: Dictionary of feature tensors
            modality_mask: Optional masks for missing modalities

        Returns:
            - logits: Classification logits
            - outputs: Fused features and uniform weights
        """
        # Concatenate features in consistent order
        feature_list = [features[name] for name in self.modality_names]
        fused = torch.cat(feature_list, dim=1)  # (B, total_dim)

        logits = self.classifier(fused)

        # Uniform weights (early fusion treats all equally)
        B = fused.shape[0]
        num_modalities = len(self.modality_names)
        weights = torch.ones(B, num_modalities, device=fused.device) / num_modalities

        return logits, {
            "fused_features": fused,
            "modality_weights": weights,
        }


class LateFusion(MultimodalFusion):
    """
    Late fusion: Separate classifiers per modality, then combine.

    Trains independent classifiers for each modality, then combines
    predictions through learned or fixed aggregation.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        num_classes: int,
        learned_weights: bool = True,
    ):
        """
        Initialize late fusion.

        Args:
            modality_dims: Dictionary of modality dimensions
            num_classes: Number of output classes
            learned_weights: If True, learn combination weights; else use equal weights
        """
        super().__init__(modality_dims, num_classes)

        self.num_modalities = len(modality_dims)
        self.learned_weights = learned_weights

        # Separate classifier per modality
        self.classifiers = nn.ModuleDict()
        for name, dim in modality_dims.items():
            self.classifiers[name] = nn.Sequential(
                nn.Linear(dim, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, num_classes),
            )

        # Learned combination weights
        if learned_weights:
            self.weight_network = nn.Sequential(
                nn.Linear(num_classes * self.num_modalities, 64),
                nn.Tanh(),
                nn.Linear(64, self.num_modalities),
            )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: independent classifiers then fusion.

        Args:
            features: Dictionary of feature tensors
            modality_mask: Optional masks for missing modalities

        Returns:
            - logits: Combined classification logits
            - outputs: Per-modality predictions and weights
        """
        B = features[self.modality_names[0]].shape[0]

        # Get predictions from each modality classifier
        modality_logits = {}
        logits_list = []

        for name in self.modality_names:
            feat = features[name]
            logits = self.classifiers[name](feat)  # (B, num_classes)
            modality_logits[name] = logits
            logits_list.append(logits)

        logits_tensor = torch.stack(logits_list, dim=1)  # (B, num_modalities, num_classes)

        # Compute combination weights
        if self.learned_weights:
            # Concatenate all predictions
            concat_logits = torch.cat(logits_list, dim=1)  # (B, num_modalities * num_classes)
            weights = F.softmax(self.weight_network(concat_logits), dim=1)  # (B, num_modalities)
        else:
            weights = torch.ones(B, self.num_modalities, device=logits_list[0].device) / self.num_modalities

        # Apply modality mask if provided
        if modality_mask is not None:
            mask_list = [
                modality_mask[name].float() if name in modality_mask else torch.ones(B)
                for name in self.modality_names
            ]
            mask_tensor = torch.stack(mask_list, dim=1)  # (B, num_modalities)
            weights = weights * mask_tensor
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-5)

        # Weighted aggregation
        weights_expanded = weights.unsqueeze(2)  # (B, num_modalities, 1)
        fused_logits = (logits_tensor * weights_expanded).sum(dim=1)  # (B, num_classes)

        return fused_logits, {
            "fused_features": fused_logits,
            "modality_weights": weights,
            "modality_logits": modality_logits,
        }


class CrossAttentionFusion(MultimodalFusion):
    """
    Cross-attention fusion: Bidirectional attention between modalities.

    Each modality attends to all other modalities, then results are
    concatenated for final classification.

    Uses multi-head cross-attention for learning complex modality interactions.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        num_classes: int,
        num_heads: int = 4,
        hidden_dim: int = 256,
    ):
        """
        Initialize cross-attention fusion.

        Args:
            modality_dims: Dictionary of modality dimensions
            num_classes: Number of output classes
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension for cross-attention
        """
        super().__init__(modality_dims, num_classes)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Project each modality to hidden dimension
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, hidden_dim) for name, dim in modality_dims.items()}
        )

        # Cross-attention modules
        self.cross_attentions = nn.ModuleDict()
        for name in self.modality_names:
            # This modality attends to all others
            other_names = [n for n in self.modality_names if n != name]
            for other_name in other_names:
                key = f"{name}_to_{other_name}"
                self.cross_attentions[key] = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    batch_first=True,
                    dropout=0.1,
                )

        # Final fusion classifier
        fused_dim = hidden_dim * len(self.modality_names)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: cross-attention then classify.

        Args:
            features: Dictionary of feature tensors
            modality_mask: Optional masks for missing modalities

        Returns:
            - logits: Classification logits
            - outputs: Attended features and attention weights
        """
        B = features[self.modality_names[0]].shape[0]

        # Project all features to hidden dimension
        projected = {}
        for name in self.modality_names:
            projected[name] = self.projections[name](features[name]).unsqueeze(
                1
            )  # (B, 1, hidden_dim)

        # Cross-attention
        attended_features = {}
        attention_weights = {}

        for name in self.modality_names:
            other_names = [n for n in self.modality_names if n != name]
            attended_list = [projected[name]]

            for other_name in other_names:
                key = f"{name}_to_{other_name}"
                query = projected[name]
                key_val = projected[other_name]

                # Self-attention on this modality
                attn_out, attn_weights = self.cross_attentions[key](
                    query, key_val, key_val
                )
                attended_list.append(attn_out)
                attention_weights[key] = attn_weights

            # Concatenate original and attended features
            attended_features[name] = torch.cat(attended_list, dim=2).squeeze(
                1
            )  # (B, hidden_dim * num_others+1)

        # Concatenate all modalities
        all_attended = torch.cat(
            [attended_features[name] for name in self.modality_names], dim=1
        )  # (B, hidden_dim * num_modalities)

        logits = self.classifier(all_attended)

        # Compute per-modality weights from attention
        modality_weights = []
        for name in self.modality_names:
            other_names = [n for n in self.modality_names if n != name]
            key = f"{name}_to_{other_names[0]}" if other_names else None
            if key and key in attention_weights:
                weight = attention_weights[key].mean(dim=1).max(dim=0)[0].detach()
            else:
                weight = torch.ones(1, device=logits.device) / len(self.modality_names)
            modality_weights.append(weight)

        modality_weights = torch.stack(modality_weights, dim=0).unsqueeze(0)
        modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-5)

        return logits, {
            "fused_features": all_attended,
            "modality_weights": modality_weights.squeeze(0),
            "attention_weights": attention_weights,
        }


class GatedFusion(MultimodalFusion):
    """
    Gated fusion: Learn per-sample importance weights for each modality.

    Uses a gating network to learn which modality to trust for each sample.
    Useful for capturing modality relevance to specific phenotypes or outcomes.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        num_classes: int,
        hidden_dim: int = 128,
    ):
        """
        Initialize gated fusion.

        Args:
            modality_dims: Dictionary of modality dimensions
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for gating network
        """
        super().__init__(modality_dims, num_classes)

        self.hidden_dim = hidden_dim
        total_dim = sum(modality_dims.values())

        # Gating network: learns per-sample importance weights
        self.gating_network = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(modality_dims)),
        )

        # Feature projection and classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: compute gates and apply to features.

        Args:
            features: Dictionary of feature tensors
            modality_mask: Optional masks for missing modalities

        Returns:
            - logits: Classification logits
            - outputs: Gated features and weights
        """
        B = features[self.modality_names[0]].shape[0]

        # Concatenate all features
        feature_list = [features[name] for name in self.modality_names]
        concatenated = torch.cat(feature_list, dim=1)  # (B, total_dim)

        # Compute gating weights
        gate_scores = self.gating_network(concatenated)  # (B, num_modalities)
        gate_weights = F.softmax(gate_scores, dim=1)  # (B, num_modalities)

        # Apply modality mask if provided
        if modality_mask is not None:
            mask_list = [
                modality_mask[name].float() if name in modality_mask else torch.ones(B)
                for name in self.modality_names
            ]
            mask_tensor = torch.stack(mask_list, dim=1)  # (B, num_modalities)
            gate_weights = gate_weights * mask_tensor
            gate_weights = gate_weights / (gate_weights.sum(dim=1, keepdim=True) + 1e-5)

        # Apply gates to individual features
        gated_features = []
        for i, name in enumerate(self.modality_names):
            gated_feat = features[name] * gate_weights[:, i].unsqueeze(1)
            gated_features.append(gated_feat)

        gated_concatenated = torch.cat(gated_features, dim=1)  # (B, total_dim)

        logits = self.classifier(gated_concatenated)

        return logits, {
            "fused_features": gated_concatenated,
            "modality_weights": gate_weights,
            "gate_scores": gate_scores,
        }


def get_fusion_module(
    fusion_type: str,
    modality_dims: Dict[str, int],
    num_classes: int,
    **kwargs,
) -> MultimodalFusion:
    """
    Factory function to get the appropriate fusion module.

    Args:
        fusion_type: Type of fusion ('early', 'late', 'cross_attention', 'gated')
        modality_dims: Dictionary mapping modality names to dimensions
        num_classes: Number of output classes
        **kwargs: Additional arguments specific to fusion type

    Returns:
        Configured fusion module

    Example:
        fusion = get_fusion_module(
            'cross_attention',
            {'imaging': 1536, 'radiomics': 128},
            num_classes=2,
            num_heads=4
        )
    """
    fusion_type = fusion_type.lower().strip()

    if fusion_type == "early":
        return EarlyFusion(modality_dims, num_classes)

    elif fusion_type == "late":
        learned_weights = kwargs.get("learned_weights", True)
        return LateFusion(modality_dims, num_classes, learned_weights=learned_weights)

    elif fusion_type == "cross_attention":
        num_heads = kwargs.get("num_heads", 4)
        hidden_dim = kwargs.get("hidden_dim", 256)
        return CrossAttentionFusion(
            modality_dims,
            num_classes,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
        )

    elif fusion_type == "gated":
        hidden_dim = kwargs.get("hidden_dim", 128)
        return GatedFusion(modality_dims, num_classes, hidden_dim=hidden_dim)

    else:
        raise ValueError(
            f"Unknown fusion type: {fusion_type}. "
            "Supported: 'early', 'late', 'cross_attention', 'gated'"
        )
