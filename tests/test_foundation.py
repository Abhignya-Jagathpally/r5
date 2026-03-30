"""
Comprehensive tests for foundation model integration.

Tests:
- Feature extractor factory and abstract interface
- MIL heads (TransMIL, DTFD-MIL, HIPT)
- Multimodal fusion strategies
- Explainability utilities
- Configuration loading and validation
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from typing import Dict

# Import modules to test
from src.models.foundation.feature_extractor import get_extractor
from src.models.foundation.mil_heads import TransMIL, DTFDMIL, HIRCLSMILHead
from src.models.foundation.multimodal_fusion import (
    EarlyFusion,
    LateFusion,
    CrossAttentionFusion,
    GatedFusion,
    get_fusion_module,
)
from src.models.foundation.explainability import AttentionExplainer, GradCAM


class TestFeatureExtractor:
    """Test feature extractor factory and implementations."""

    def test_get_extractor_resnet50(self):
        """Test ResNet50 extractor initialization."""
        extractor = get_extractor("resnet50", device="cpu")
        assert extractor is not None
        assert extractor.embedding_dim == 2048
        assert extractor.model_name == "resnet50_imagenet"

    def test_get_extractor_uni2h(self):
        """Test UNI2H extractor initialization."""
        extractor = get_extractor("uni2h", device="cpu")
        assert extractor is not None
        assert extractor.embedding_dim == 1536
        assert extractor.model_name == "uni2h"

    def test_get_extractor_titan(self):
        """Test TITAN extractor initialization."""
        extractor = get_extractor("titan", device="cpu")
        assert extractor is not None
        assert extractor.embedding_dim == 768
        assert extractor.model_name == "titan"

    def test_get_extractor_invalid(self):
        """Test invalid extractor raises error."""
        with pytest.raises(ValueError):
            get_extractor("invalid_backbone", device="cpu")

    def test_custom_embedding_dim(self):
        """Test custom embedding dimension."""
        extractor = get_extractor("resnet50", embedding_dim=1024, device="cpu")
        assert extractor.embedding_dim == 1024

    def test_extractor_config(self):
        """Test extractor configuration export."""
        extractor = get_extractor("uni2h", batch_size=32, device="cpu")
        config = extractor.model_config
        assert isinstance(config, dict)
        assert "model_name" in config
        assert "embedding_dim" in config
        assert "batch_size" in config


class TestTransMIL:
    """Test TransMIL MIL head."""

    def test_transmil_initialization(self):
        """Test TransMIL initialization."""
        model = TransMIL(
            input_dim=1536,
            hidden_dim=512,
            num_heads=8,
            num_layers=2,
            num_classes=2,
        )
        assert model is not None
        assert model.input_dim == 1536
        assert model.num_classes == 2

    def test_transmil_forward(self):
        """Test TransMIL forward pass."""
        batch_size = 4
        num_patches = 100
        input_dim = 1536

        model = TransMIL(
            input_dim=input_dim,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            num_classes=2,
        )

        # Random embeddings
        embeddings = torch.randn(batch_size, num_patches, input_dim)

        logits, aux = model(embeddings)

        assert logits.shape == (batch_size, 2)
        assert "cls_embedding" in aux
        assert "patch_embeddings" in aux
        assert aux["cls_embedding"].shape == (batch_size, 256)

    def test_transmil_with_coords(self):
        """Test TransMIL with spatial coordinates."""
        batch_size = 4
        num_patches = 100
        input_dim = 1536

        model = TransMIL(
            input_dim=input_dim,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            num_classes=2,
            use_coords=True,
        )

        embeddings = torch.randn(batch_size, num_patches, input_dim)
        coords = torch.randn(batch_size, num_patches, 2) * 1000  # (x, y) coordinates

        logits, aux = model(embeddings, coords=coords)

        assert logits.shape == (batch_size, 2)

    def test_transmil_with_mask(self):
        """Test TransMIL with attention mask."""
        batch_size = 4
        num_patches = 100
        input_dim = 1536

        model = TransMIL(
            input_dim=input_dim,
            hidden_dim=256,
            num_heads=4,
            num_classes=2,
        )

        embeddings = torch.randn(batch_size, num_patches, input_dim)
        # Some patches are valid, some are masked out
        mask = torch.rand(batch_size, num_patches) > 0.2

        logits, aux = model(embeddings, mask=mask)

        assert logits.shape == (batch_size, 2)


class TestDTFDMIL:
    """Test DTFD-MIL head."""

    def test_dtfdmil_initialization(self):
        """Test DTFD-MIL initialization."""
        model = DTFDMIL(
            input_dim=1536,
            hidden_dim=256,
            num_pseudo_bags=8,
            num_classes=2,
        )
        assert model is not None
        assert model.num_pseudo_bags == 8

    def test_dtfdmil_forward(self):
        """Test DTFD-MIL forward pass."""
        batch_size = 4
        num_patches = 100
        input_dim = 1536

        model = DTFDMIL(
            input_dim=input_dim,
            hidden_dim=256,
            num_pseudo_bags=8,
            num_classes=2,
        )

        embeddings = torch.randn(batch_size, num_patches, input_dim)

        logits, aux = model(embeddings)

        assert logits.shape == (batch_size, 2)
        assert "tier1_weights" in aux
        assert "tier2_weights" in aux
        assert "assignments" in aux


class TestHIRCLSMIL:
    """Test HIPT-style hierarchical MIL head."""

    def test_hirclsmil_initialization(self):
        """Test HIPT initialization."""
        model = HIRCLSMILHead(
            input_dim=1536,
            hidden_dim=512,
            num_regions=16,
            num_classes=2,
        )
        assert model is not None
        assert model.num_regions == 16

    def test_hirclsmil_forward(self):
        """Test HIPT forward pass."""
        batch_size = 4
        num_patches = 100
        input_dim = 1536

        model = HIRCLSMILHead(
            input_dim=input_dim,
            hidden_dim=256,
            num_regions=16,
            num_classes=2,
        )

        embeddings = torch.randn(batch_size, num_patches, input_dim)

        logits, aux = model(embeddings)

        assert logits.shape == (batch_size, 2)
        assert "region_features" in aux
        assert "region_weights" in aux

    def test_hirclsmil_with_coords(self):
        """Test HIPT with spatial coordinates."""
        batch_size = 4
        num_patches = 100
        input_dim = 1536

        model = HIRCLSMILHead(
            input_dim=input_dim,
            hidden_dim=256,
            num_regions=4,  # 2x2 grid
            num_classes=2,
        )

        embeddings = torch.randn(batch_size, num_patches, input_dim)
        coords = torch.randn(batch_size, num_patches, 2) * 1000

        logits, aux = model(embeddings, coords=coords)

        assert logits.shape == (batch_size, 2)


class TestMultimodalFusion:
    """Test multimodal fusion strategies."""

    def test_early_fusion(self):
        """Test early fusion."""
        modality_dims = {"imaging": 1536, "radiomics": 128}
        fusion = EarlyFusion(modality_dims, num_classes=2)

        features = {
            "imaging": torch.randn(8, 1536),
            "radiomics": torch.randn(8, 128),
        }

        logits, outputs = fusion(features)

        assert logits.shape == (8, 2)
        assert "fused_features" in outputs
        assert "modality_weights" in outputs
        assert outputs["modality_weights"].shape == (8, 2)

    def test_late_fusion(self):
        """Test late fusion."""
        modality_dims = {"imaging": 1536, "radiomics": 128}
        fusion = LateFusion(modality_dims, num_classes=2, learned_weights=True)

        features = {
            "imaging": torch.randn(8, 1536),
            "radiomics": torch.randn(8, 128),
        }

        logits, outputs = fusion(features)

        assert logits.shape == (8, 2)
        assert "modality_weights" in outputs
        assert outputs["modality_weights"].shape == (8, 2)

    def test_cross_attention_fusion(self):
        """Test cross-attention fusion."""
        modality_dims = {"imaging": 1536, "radiomics": 128}
        fusion = CrossAttentionFusion(
            modality_dims,
            num_classes=2,
            num_heads=4,
            hidden_dim=256,
        )

        features = {
            "imaging": torch.randn(8, 1536),
            "radiomics": torch.randn(8, 128),
        }

        logits, outputs = fusion(features)

        assert logits.shape == (8, 2)
        assert "modality_weights" in outputs

    def test_gated_fusion(self):
        """Test gated fusion."""
        modality_dims = {"imaging": 1536, "radiomics": 128}
        fusion = GatedFusion(modality_dims, num_classes=2, hidden_dim=128)

        features = {
            "imaging": torch.randn(8, 1536),
            "radiomics": torch.randn(8, 128),
        }

        logits, outputs = fusion(features)

        assert logits.shape == (8, 2)
        assert "gate_scores" in outputs
        assert "modality_weights" in outputs

    def test_get_fusion_module_factory(self):
        """Test fusion module factory."""
        modality_dims = {"imaging": 1536, "radiomics": 128}

        for fusion_type in ["early", "late", "cross_attention", "gated"]:
            fusion = get_fusion_module(
                fusion_type,
                modality_dims=modality_dims,
                num_classes=2,
            )
            assert fusion is not None

    def test_get_fusion_module_invalid(self):
        """Test invalid fusion type raises error."""
        modality_dims = {"imaging": 1536}
        with pytest.raises(ValueError):
            get_fusion_module(
                "invalid_fusion",
                modality_dims=modality_dims,
                num_classes=2,
            )

    def test_fusion_with_missing_modality(self):
        """Test fusion with missing modalities."""
        modality_dims = {"imaging": 1536, "radiomics": 128, "genomic": 256}
        fusion = GatedFusion(modality_dims, num_classes=2)

        features = {
            "imaging": torch.randn(8, 1536),
            "radiomics": torch.randn(8, 128),
            "genomic": torch.randn(8, 256),
        }

        # Mask out genomic modality (not available)
        modality_mask = {
            "imaging": torch.ones(8, dtype=torch.bool),
            "radiomics": torch.ones(8, dtype=torch.bool),
            "genomic": torch.zeros(8, dtype=torch.bool),  # Missing
        }

        logits, outputs = fusion(features, modality_mask=modality_mask)

        assert logits.shape == (8, 2)
        # Check that genomic gets near-zero weight
        weights = outputs["modality_weights"]
        assert weights[:, 2].mean() < weights[:, 0].mean()


class TestExplainability:
    """Test explainability utilities."""

    def test_attention_explainer_initialization(self):
        """Test AttentionExplainer initialization."""
        explainer = AttentionExplainer(
            tile_size=256,
            magnification=20,
            top_k=10,
        )
        assert explainer is not None
        assert explainer.top_k == 10

    def test_attention_explainer_heatmap(self):
        """Test heatmap generation."""
        explainer = AttentionExplainer()

        num_tiles = 50
        attention_weights = np.random.rand(num_tiles)
        coords = np.random.rand(num_tiles, 2) * 10000

        heatmap = explainer.generate_heatmap(
            attention_weights,
            coords,
            output_size=(512, 512),
        )

        assert heatmap.shape == (512, 512)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_attention_explainer_top_tiles(self):
        """Test top-K tile extraction."""
        explainer = AttentionExplainer(top_k=5)

        num_tiles = 20
        attention_weights = np.arange(num_tiles, dtype=np.float32)
        tile_paths = [f"/path/tile_{i}.jpg" for i in range(num_tiles)]

        top_tiles = explainer.get_top_tiles(attention_weights, tile_paths, k=5)

        assert len(top_tiles) == 5
        # Should be sorted by weight (descending)
        assert top_tiles[0][1] > top_tiles[-1][1]

    def test_attention_explainer_report(self):
        """Test report generation."""
        explainer = AttentionExplainer()

        num_tiles = 50
        attention_weights = np.random.rand(num_tiles)
        coords = np.random.rand(num_tiles, 2) * 10000
        tile_paths = [f"/path/tile_{i}.jpg" for i in range(num_tiles)]

        report = explainer.generate_report(
            slide_id="slide_001",
            prediction=0.85,
            confidence=0.92,
            attention_weights=attention_weights,
            coords=coords,
            tile_paths=tile_paths,
        )

        assert report["slide_id"] == "slide_001"
        assert report["prediction"] == 0.85
        assert "heatmap" in report
        assert "top_tiles" in report

    def test_attention_explainer_modality_importance(self):
        """Test modality importance computation."""
        explainer = AttentionExplainer()

        modality_weights = {
            "imaging": np.array([0.6, 0.5, 0.7]),
            "radiomics": np.array([0.3, 0.4, 0.2]),
            "genomic": np.array([0.1, 0.1, 0.1]),
        }

        importance = explainer.compute_modality_importance(modality_weights)

        assert isinstance(importance, dict)
        assert "imaging" in importance
        assert "radiomics" in importance
        # Sum should be approximately 1
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_gradcam_initialization(self):
        """Test GradCAM initialization."""
        dummy_model = torch.nn.Linear(10, 2)
        gradcam = GradCAM(dummy_model)
        assert gradcam is not None


class TestEndToEndPipeline:
    """Test end-to-end pipelines."""

    def test_uni2h_to_transmil_to_early_fusion(self):
        """Test full pipeline: UNI2H -> TransMIL -> Early Fusion."""
        # Note: This is a mock test that doesn't actually load UNI2H
        # In real tests, you would need real data

        batch_size = 2
        num_patches = 50

        # Mock UNI2H embeddings
        embeddings = torch.randn(batch_size, num_patches, 1536)
        coords = torch.randn(batch_size, num_patches, 2) * 1000

        # TransMIL
        mil_head = TransMIL(
            input_dim=1536,
            hidden_dim=256,
            num_heads=4,
            num_layers=1,
            num_classes=2,
        )

        imaging_logits, mil_aux = mil_head(embeddings, coords=coords)

        # Extract slide representation
        slide_representation = mil_aux["cls_embedding"]  # (batch_size, 256)

        # Mock radiomics
        radiomics_features = torch.randn(batch_size, 128)

        # Early fusion
        modality_dims = {"imaging": 256, "radiomics": 128}
        fusion = EarlyFusion(modality_dims, num_classes=2)

        final_features = {
            "imaging": slide_representation,
            "radiomics": radiomics_features,
        }

        final_logits, fusion_aux = fusion(final_features)

        assert final_logits.shape == (batch_size, 2)
        assert fusion_aux["modality_weights"].shape == (batch_size, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
