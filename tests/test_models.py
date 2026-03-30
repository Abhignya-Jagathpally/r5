"""
Unit tests for classical baseline models.

Tests:
- ABMIL forward pass and attention mechanism
- CLAM forward pass and instance clustering
- Tile classifier
- MIL dataset loading
- Loss functions
- Mean pool baseline
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import zarr

from src.models.abmil import ABMIL, GatedAttention
from src.models.clam import CLAM_SB, CLAM_MB, AttentionLayer
from src.models.losses import (
    CoxPartialLikelihoodLoss,
    FocalLoss,
    SmoothTopKLoss,
    InstanceClusteringLoss,
)
from src.models.mean_pool_baseline import MeanPoolBaseline
from src.models.mil_dataset import MILDataset, SplitsManager
from src.models.tile_classifier import TileClassifier


class TestGatedAttention:
    """Test GatedAttention mechanism."""

    def test_forward_shapes(self):
        """Test output shapes."""
        batch_size = 10
        input_dim = 2048
        hidden_dim = 256
        attention_dim = 128

        attention = GatedAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            gated=True,
        )

        # Random embeddings
        embeddings = torch.randn(batch_size, input_dim)

        aggregated, weights = attention(embeddings)

        assert aggregated.shape == (input_dim,)
        assert weights.shape == (batch_size,)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        batch_size = 20
        input_dim = 2048
        hidden_dim = 256
        attention_dim = 128

        attention = GatedAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            gated=False,
        )

        embeddings = torch.randn(batch_size, input_dim)
        _, weights = attention(embeddings)

        assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.all(weights >= 0)

    def test_gated_vs_ungated(self):
        """Test difference between gated and ungated attention."""
        input_dim = 2048
        hidden_dim = 256
        attention_dim = 128
        batch_size = 15

        attention_gated = GatedAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            gated=True,
        )

        attention_ungated = GatedAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            gated=False,
        )

        embeddings = torch.randn(batch_size, input_dim)

        agg_gated, w_gated = attention_gated(embeddings)
        agg_ungated, w_ungated = attention_ungated(embeddings)

        # Outputs should be different (not exactly equal)
        assert not torch.allclose(agg_gated, agg_ungated)
        assert not torch.allclose(w_gated, w_ungated)


class TestABMIL:
    """Test ABMIL model."""

    def test_forward_pass(self):
        """Test ABMIL forward pass."""
        model = ABMIL(
            input_dim=2048,
            hidden_dim=256,
            attention_dim=128,
            num_classes=2,
            dropout=0.25,
        )

        num_instances = 100
        embeddings = torch.randn(num_instances, 2048)

        logits = model(embeddings)

        assert logits.shape == (2,)
        assert logits.dtype == torch.float32

    def test_forward_with_attention(self):
        """Test ABMIL with attention output."""
        model = ABMIL(
            input_dim=2048,
            hidden_dim=256,
            attention_dim=128,
            num_classes=2,
        )

        num_instances = 50
        embeddings = torch.randn(num_instances, 2048)

        logits, attention_weights = model(embeddings, return_attention=True)

        assert logits.shape == (2,)
        assert attention_weights.shape == (num_instances,)
        assert torch.isclose(attention_weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_eval_mode(self):
        """Test model in eval mode."""
        model = ABMIL(num_classes=2)
        model.eval()

        embeddings = torch.randn(30, 2048)

        with torch.no_grad():
            logits = model(embeddings)

        assert logits.shape == (2,)


class TestCLAM:
    """Test CLAM models."""

    def test_clam_sb_forward(self):
        """Test CLAM-SB forward pass."""
        model = CLAM_SB(
            input_dim=2048,
            hidden_dim=256,
            num_classes=2,
        )

        embeddings = torch.randn(100, 2048)
        logits = model(embeddings)

        assert logits.shape == (2,)

    def test_clam_mb_forward(self):
        """Test CLAM-MB forward pass."""
        model = CLAM_MB(
            input_dim=2048,
            hidden_dim=256,
            num_classes=2,
            num_heads=3,
            inst_cluster=True,
        )

        embeddings = torch.randn(100, 2048)
        logits = model(embeddings)

        # Multi-head concatenates outputs
        assert logits.shape == (2,)

    def test_clam_mb_attention_dict(self):
        """Test CLAM-MB returns attention for each head."""
        num_heads = 3
        model = CLAM_MB(
            input_dim=2048,
            hidden_dim=256,
            num_classes=2,
            num_heads=num_heads,
            inst_cluster=False,
        )

        embeddings = torch.randn(50, 2048)
        logits, attention_dict = model(embeddings, return_attention=True)

        assert len(attention_dict) == num_heads
        for key, weights in attention_dict.items():
            assert weights.shape == (50,)
            assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_clam_mb_instance_predictions(self):
        """Test instance-level predictions for CLAM-MB."""
        model = CLAM_MB(
            input_dim=2048,
            hidden_dim=256,
            num_classes=2,
            inst_cluster=True,
        )

        embeddings = torch.randn(100, 2048)
        inst_logits = model.get_instance_predictions(embeddings)

        assert inst_logits.shape == (100, 2)


class TestTileClassifier:
    """Test tile-level classifier."""

    def test_initialization(self):
        """Test model initialization."""
        model = TileClassifier(num_classes=2, pretrained=False)

        assert model.num_classes == 2
        assert model.feature_dim == 2048

    def test_forward_pass(self):
        """Test forward pass."""
        model = TileClassifier(num_classes=2, pretrained=False)
        model.eval()

        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            logits = model(images)

        assert logits.shape == (batch_size, 2)

    def test_feature_extraction(self):
        """Test feature extraction without classification head."""
        model = TileClassifier(num_classes=2, pretrained=False)
        model.eval()

        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            features = model.extract_features(images)

        assert features.shape == (batch_size, 2048)


class TestLossFunctions:
    """Test loss functions."""

    def test_cox_loss(self):
        """Test Cox partial likelihood loss."""
        loss_fn = CoxPartialLikelihoodLoss()

        # Synthetic data
        risk_scores = torch.randn(10)
        event_times = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        event_indicators = torch.tensor([1, 0, 1, 1, 0, 1, 1, 0, 1, 1])

        loss = loss_fn(risk_scores, event_times, event_indicators)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_focal_loss(self):
        """Test focal loss."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))

        loss = loss_fn(logits, targets)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_instance_clustering_loss(self):
        """Test instance clustering loss."""
        loss_fn = InstanceClusteringLoss()

        instance_logits = torch.randn(50, 2)
        pseudo_labels = torch.randint(0, 2, (50,))

        loss = loss_fn(instance_logits, pseudo_labels)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_smooth_topk_loss(self):
        """Test smooth top-k loss."""
        loss_fn = SmoothTopKLoss(k=5, temperature=1.0)

        instance_scores = torch.randn(20, 2)
        bag_label = 0

        loss = loss_fn(instance_scores, bag_label)

        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestMILDataset:
    """Test MIL dataset loading."""

    def test_mil_dataset_creation(self):
        """Test MIL dataset creation with mock Zarr store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "embeddings.zarr"
            store = zarr.open(zarr_path, mode="w")

            # Create mock data
            num_slides = 3
            num_tiles_per_slide = 10
            embedding_dim = 2048

            slide_ids = []
            labels = []

            for slide_idx in range(num_slides):
                slide_id = f"slide_{slide_idx}"
                slide_ids.append(slide_id)
                labels.append(slide_idx % 2)

                # Create slide group
                slide_group = store.create_group(slide_id)

                # Create tile embeddings
                for tile_idx in range(num_tiles_per_slide):
                    embedding = np.random.randn(embedding_dim).astype(np.float32)
                    slide_group.array(
                        f"tile_{tile_idx}_{tile_idx}.npy",
                        embedding,
                    )

            # Create dataset
            dataset = MILDataset(
                zarr_path=str(zarr_path),
                slide_ids=slide_ids,
                labels=labels,
                max_patches=20,
                sampling_strategy="all",
            )

            assert len(dataset) == num_slides

            # Get one sample
            embeddings, coordinates, label, slide_id = dataset[0]

            assert embeddings.shape[0] > 0
            assert embeddings.shape[1] == embedding_dim
            assert coordinates.shape[0] == embeddings.shape[0]
            assert label in [0, 1]

    def test_splits_manager(self):
        """Test splits manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits_path = Path(tmpdir) / "splits.csv"

            # Create splits CSV
            with open(splits_path, "w") as f:
                f.write("slide_id,label,split\n")
                for i in range(20):
                    split = "train" if i < 10 else ("val" if i < 15 else "test")
                    f.write(f"slide_{i},0,{split}\n")

            manager = SplitsManager(str(splits_path))

            train_ids, train_labels = manager.get_split("train")
            assert len(train_ids) == 10

            val_ids, val_labels = manager.get_split("val")
            assert len(val_ids) == 5

            test_ids, test_labels = manager.get_split("test")
            assert len(test_ids) == 5


class TestMeanPoolBaseline:
    """Test mean pool baseline."""

    def test_mean_pool_fit_and_predict(self):
        """Test mean pool baseline fit and predict."""
        baseline = MeanPoolBaseline(
            embedding_dim=2048,
            classifiers=["logistic_regression", "random_forest"],
        )

        # Synthetic data
        X_train = np.random.randn(20, 2048)
        y_train = np.random.randint(0, 2, 20)

        X_test = np.random.randn(5, 2048)
        y_test = np.random.randint(0, 2, 5)

        # Fit
        baseline.fit(X_train, y_train, verbose=False)

        # Predict
        preds = baseline.predict(X_test, classifier="logistic_regression")
        assert preds.shape == (5,)
        assert all(p in [0, 1] for p in preds)

        # Predict proba
        probs = baseline.predict_proba(X_test, classifier="logistic_regression")
        assert probs.shape == (5, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_mean_pool_evaluate(self):
        """Test mean pool evaluation."""
        baseline = MeanPoolBaseline(classifiers=["logistic_regression"])

        X_train = np.random.randn(30, 2048)
        y_train = np.random.randint(0, 2, 30)

        X_test = np.random.randn(10, 2048)
        y_test = np.random.randint(0, 2, 10)

        baseline.fit(X_train, y_train, verbose=False)
        metrics = baseline.evaluate(X_test, y_test, classifier="logistic_regression")

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "auroc" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1"] <= 1


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_abmil_training_step(self):
        """Test ABMIL training step."""
        model = ABMIL(
            input_dim=2048,
            hidden_dim=256,
            num_classes=2,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Single training step
        embeddings = torch.randn(50, 2048)
        label = torch.tensor([1])

        logits = model(embeddings).unsqueeze(0)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_clam_with_instance_loss(self):
        """Test CLAM with instance clustering loss."""
        model = CLAM_MB(
            input_dim=2048,
            hidden_dim=256,
            num_classes=2,
            num_heads=2,
            inst_cluster=True,
        )

        embeddings = torch.randn(100, 2048)
        bag_label = torch.tensor([0])

        # Forward pass
        logits = model(embeddings)

        # Instance predictions
        inst_logits = model.get_instance_predictions(embeddings)

        assert logits.shape == (2,)
        assert inst_logits.shape == (100, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
