"""
Modern Multiple Instance Learning (MIL) heads for weakly supervised learning.

Implements state-of-the-art MIL aggregation strategies that sit on top of
foundation model embeddings to learn slide-level predictions from tile-level
(or region-level) features.

Architectures:
- TransMIL: Transformer-based MIL with positional encoding
- DTFD-MIL: Double-tier feature distillation
- HIPT-style: Hierarchical region-level then slide-level aggregation
"""

from typing import Optional, Tuple, Dict, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding using tile (x, y) coordinates.

    Maps spatial coordinates to a high-dimensional learned representation.
    """

    def __init__(self, hidden_dim: int, max_coord: float = 10000.0):
        """
        Initialize positional encoding.

        Args:
            hidden_dim: Dimension of output encoding
            max_coord: Maximum coordinate value for normalization
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_coord = max_coord
        self.fc = nn.Linear(2, hidden_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates.

        Args:
            coords: Coordinate tensor of shape (B, N, 2) with (x, y) values

        Returns:
            Positional encoding of shape (B, N, hidden_dim)
        """
        # Normalize coordinates
        coords_norm = coords / self.max_coord

        # Learn embeddings
        encoding = self.fc(coords_norm)

        return encoding


class TransMIL(nn.Module):
    """
    Transformer-based Multiple Instance Learning head.

    Uses self-attention over patch embeddings with learnable [CLS] token.
    Incorporates spatial information through positional encoding from
    tile (x, y) coordinates.

    Reference: "TransMIL: Transformer based Correlated Multiple Instance Learning
    for Whole Slide Image Classification" (NIPS 2021)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
        use_coords: bool = True,
    ):
        """
        Initialize TransMIL.

        Args:
            input_dim: Dimension of input embeddings (e.g., 1536 for UNI2-h)
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            num_classes: Number of output classes
            use_coords: If True, incorporate tile coordinates via positional encoding
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_coords = use_coords

        # Project input to hidden dimension
        self.embedding_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        if use_coords:
            self.pos_encoding = PositionalEncoding(hidden_dim)
            self.coord_projection = nn.Linear(hidden_dim, hidden_dim)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            embeddings: Patch embeddings of shape (B, N, input_dim)
                where B is batch size, N is number of patches
            coords: Optional spatial coordinates of shape (B, N, 2)
            mask: Optional attention mask of shape (B, N)

        Returns:
            - logits: Classification logits of shape (B, num_classes)
            - aux: Dictionary with attention weights and intermediate features
        """
        B, N, D = embeddings.shape

        # Project embeddings to hidden dimension
        x = self.embedding_projection(embeddings)  # (B, N, hidden_dim)

        # Add positional encoding if coordinates provided
        if self.use_coords and coords is not None:
            pos_enc = self.pos_encoding(coords)  # (B, N, hidden_dim)
            pos_proj = self.coord_projection(pos_enc)
            x = x + pos_proj

        # Expand [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)

        # Concatenate [CLS] with patches
        x = torch.cat([cls, x], dim=1)  # (B, N+1, hidden_dim)

        # Create attention mask if provided
        attn_mask = None
        if mask is not None:
            # Append True for [CLS] token
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            mask_with_cls = torch.cat([cls_mask, mask], dim=1)
            # Convert to attention mask (True = ignore)
            attn_mask = ~mask_with_cls.unsqueeze(1).expand(B, N + 1, N + 1)

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Extract [CLS] representation
        cls_out = x[:, 0, :]  # (B, hidden_dim)

        # Classify
        logits = self.classifier(cls_out)  # (B, num_classes)

        # Prepare auxiliary outputs for interpretability
        aux = {
            "cls_embedding": cls_out,
            "patch_embeddings": x[:, 1:, :],  # Exclude [CLS]
        }

        return logits, aux


class DTFDMIL(nn.Module):
    """
    Double-Tier Feature Distillation Multiple Instance Learning.

    Uses pseudo-bag generation and two-tier attention aggregation:
    - Tier 1: Attention aggregation within pseudo-bags
    - Tier 2: Attention aggregation across pseudo-bags

    Reference: "DTFD-MIL: Double-Tier Feature Distillation Multiple Instance
    Learning for Histopathology Whole Slide Image Classification"
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_pseudo_bags: int = 8,
        tier1_hidden: int = 256,
        tier2_hidden: int = 128,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        """
        Initialize DTFD-MIL.

        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for feature processing
            num_pseudo_bags: Number of pseudo-bags for tier 1
            tier1_hidden: Hidden dimension for tier 1 attention
            tier2_hidden: Hidden dimension for tier 2 attention
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_pseudo_bags = num_pseudo_bags
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Feature projection
        self.feature_projection = nn.Linear(input_dim, hidden_dim)

        # Pseudo-bag generation (clustering centers)
        self.pseudo_bag_centers = nn.Parameter(torch.randn(num_pseudo_bags, hidden_dim))

        # Tier 1: Attention within pseudo-bags
        self.tier1_attention = nn.Sequential(
            nn.Linear(hidden_dim, tier1_hidden),
            nn.Tanh(),
            nn.Linear(tier1_hidden, 1),
        )

        # Tier 2: Attention across pseudo-bags
        self.tier2_attention = nn.Sequential(
            nn.Linear(hidden_dim, tier2_hidden),
            nn.Tanh(),
            nn.Linear(tier2_hidden, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            embeddings: Patch embeddings of shape (B, N, input_dim)
            mask: Optional mask of shape (B, N)

        Returns:
            - logits: Classification logits of shape (B, num_classes)
            - aux: Dictionary with attention weights
        """
        B, N, D = embeddings.shape

        # Project features
        x = self.feature_projection(embeddings)  # (B, N, hidden_dim)

        # Assign instances to pseudo-bags via soft assignment
        pseudo_centers = self.pseudo_bag_centers  # (K, hidden_dim)
        # Compute similarity to each pseudo-bag center
        similarities = torch.einsum("bnh,kh->bnk", x, pseudo_centers)  # (B, N, K)
        assignments = F.softmax(similarities, dim=2)  # (B, N, K)

        # Tier 1: Aggregate within each pseudo-bag
        tier1_scores = self.tier1_attention(x)  # (B, N, 1)
        tier1_weights = torch.softmax(tier1_scores, dim=1)  # (B, N, 1)

        # Weighted aggregation within pseudo-bags
        # Shape: (B, K, hidden_dim)
        pseudo_bag_features = torch.einsum(
            "bnk,bnh,bn1->bkh", assignments, x, tier1_weights
        )

        # Normalize by assignment counts
        assignment_counts = assignments.sum(dim=1, keepdim=True) + 1e-5
        pseudo_bag_features = pseudo_bag_features / assignment_counts

        # Tier 2: Aggregate across pseudo-bags
        tier2_scores = self.tier2_attention(pseudo_bag_features)  # (B, K, 1)
        tier2_weights = torch.softmax(tier2_scores, dim=1)  # (B, K, 1)

        # Final slide representation
        slide_representation = (
            pseudo_bag_features * tier2_weights
        ).sum(dim=1)  # (B, hidden_dim)

        # Classify
        logits = self.classifier(slide_representation)  # (B, num_classes)

        aux = {
            "tier1_weights": tier1_weights,
            "tier2_weights": tier2_weights,
            "pseudo_bag_features": pseudo_bag_features,
            "assignments": assignments,
        }

        return logits, aux


class HIRCLSMILHead(nn.Module):
    """
    Hierarchical Region-based Class Learning Slide encoder.

    Two-level hierarchy:
    1. Region level: Aggregate nearby patches into regions
    2. Slide level: Aggregate region features to slide level

    Uses attention-based aggregation at each level.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_regions: int = 16,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        """
        Initialize HIPT-style hierarchical head.

        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for processing
            num_regions: Number of spatial regions to partition slide into
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_regions = num_regions
        self.num_classes = num_classes

        # Region encoder
        self.region_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Region-level attention
        self.region_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Slide-level encoder
        self.slide_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Slide-level attention
        self.slide_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            embeddings: Patch embeddings of shape (B, N, input_dim)
            coords: Optional spatial coordinates of shape (B, N, 2)
            mask: Optional mask of shape (B, N)

        Returns:
            - logits: Classification logits of shape (B, num_classes)
            - aux: Dictionary with regional and slide-level features
        """
        B, N, D = embeddings.shape

        # Region assignment
        if coords is not None:
            # Assign patches to regions based on coordinates
            coords_min = coords.min(dim=1, keepdim=True)[0]
            coords_max = coords.max(dim=1, keepdim=True)[0]
            coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-5)

            # Simple grid-based assignment
            grid_size = int(self.num_regions ** 0.5)
            region_ids = (
                (coords_norm[:, :, 0] * grid_size).long() * grid_size
                + (coords_norm[:, :, 1] * grid_size).long()
            )
            region_ids = torch.clamp(region_ids, 0, self.num_regions - 1)
        else:
            # Fallback: assign sequentially
            region_ids = torch.arange(N, device=embeddings.device).unsqueeze(0)
            region_ids = region_ids % self.num_regions

        # Region level aggregation
        region_features = []
        region_weights = []

        for r in range(self.num_regions):
            mask_r = region_ids == r
            if mask_r.sum() == 0:
                # Empty region - use zero vector
                region_feat = torch.zeros(B, self.hidden_dim, device=embeddings.device)
                region_weight = torch.zeros(B, 1, device=embeddings.device)
            else:
                # Encode patches in region
                region_embeddings = embeddings[mask_r].reshape(B, -1, D)
                region_emb = self.region_encoder(region_embeddings)  # (B, n_r, hidden_dim)

                # Attention pooling within region
                attn_scores = self.region_attention(region_emb)  # (B, n_r, 1)
                attn_weights = F.softmax(attn_scores, dim=1)

                region_feat = (region_emb * attn_weights).sum(dim=1)  # (B, hidden_dim)
                region_weight = attn_weights.sum(dim=1)  # (B, 1)

            region_features.append(region_feat)
            region_weights.append(region_weight)

        region_features = torch.stack(region_features, dim=1)  # (B, num_regions, hidden_dim)
        region_weights = torch.cat(region_weights, dim=1)  # (B, num_regions)

        # Slide level aggregation
        slide_emb = self.slide_encoder(region_features)  # (B, num_regions, hidden_dim)
        slide_attn = self.slide_attention(slide_emb)  # (B, num_regions, 1)
        slide_weights = F.softmax(slide_attn, dim=1)  # (B, num_regions, 1)

        slide_representation = (slide_emb * slide_weights).sum(dim=1)  # (B, hidden_dim)

        # Classify
        logits = self.classifier(slide_representation)  # (B, num_classes)

        aux = {
            "region_features": region_features,
            "region_weights": region_weights,
            "slide_weights": slide_weights.squeeze(-1),
        }

        return logits, aux
