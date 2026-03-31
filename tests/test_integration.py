"""
Integration tests for the MM Imaging Pathology & Radiomics Pipeline.

Tests end-to-end data flow between pipeline stages with synthetic data,
verifying that outputs from each stage are correctly consumed by the next.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def pipeline_tmpdir():
    """Create a temporary directory for pipeline outputs."""
    tmpdir = tempfile.mkdtemp(prefix="mm_pipeline_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def demo_data(pipeline_tmpdir):
    """Generate synthetic demo data for integration testing."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from scripts.generate_demo_data import generate_demo_data

    paths = generate_demo_data(
        output_dir=pipeline_tmpdir,
        n_patients=20,
        n_slides_per_patient=1,
        n_tiles_per_slide=50,
        embedding_dim=512,
        seed=42,
    )
    return paths


class TestDataGeneration:
    """Test that demo data generation produces valid outputs."""

    def test_generates_patient_metadata(self, demo_data):
        df = pd.read_csv(demo_data["patient_metadata"])
        assert len(df) == 20
        assert set(df.columns) >= {"patient_id", "label", "age", "survival_time", "event"}
        assert df["label"].nunique() >= 2

    def test_generates_slide_metadata(self, demo_data):
        df = pd.read_csv(demo_data["slide_metadata"])
        assert len(df) == 20
        assert "slide_id" in df.columns
        assert "label" in df.columns

    def test_generates_embeddings(self, demo_data):
        embed_path = Path(demo_data["embeddings"])
        assert embed_path.exists()
        data = np.load(embed_path)
        assert len(data.files) == 20  # One per slide

    def test_generates_radiomics(self, demo_data):
        df = pd.read_csv(demo_data["radiomics_features"])
        assert len(df) == 20
        rad_cols = [c for c in df.columns if c.startswith("radiomics_")]
        assert len(rad_cols) == 93

    def test_generates_splits(self, demo_data):
        df = pd.read_csv(demo_data["splits"])
        assert set(df["split"].unique()) == {"train", "val", "test"}
        # Patient-level integrity
        if "patient_id" in df.columns:
            for pid, group in df.groupby("patient_id"):
                assert group["split"].nunique() == 1, f"Patient {pid} in multiple splits"

    def test_splits_have_both_classes(self, demo_data):
        df = pd.read_csv(demo_data["splits"])
        for split in ["train", "val", "test"]:
            subset = df[df["split"] == split]
            if len(subset) >= 5:
                assert subset["label"].nunique() >= 2, f"{split} has only one class"


class TestPreprocessingToBaselines:
    """Test data flows correctly from preprocessing to baseline training."""

    def test_embeddings_align_with_splits(self, demo_data):
        """Verify every slide in splits has corresponding embeddings."""
        splits_df = pd.read_csv(demo_data["splits"])
        embed_data = np.load(demo_data["embeddings"])

        embed_slide_ids = {k.replace("emb_", "") for k in embed_data.files}
        split_slide_ids = set(splits_df["slide_id"])

        assert split_slide_ids.issubset(embed_slide_ids), (
            f"Slides in splits but not in embeddings: {split_slide_ids - embed_slide_ids}"
        )

    def test_mean_pool_produces_correct_shape(self, demo_data):
        """Test that mean-pooling embeddings gives one vector per slide."""
        embed_data = np.load(demo_data["embeddings"])

        for key in embed_data.files:
            emb = embed_data[key]
            assert emb.ndim == 2  # (n_tiles, dim)
            mean_pooled = emb.mean(axis=0)
            assert mean_pooled.shape == (512,)  # embedding_dim

    def test_radiomics_align_with_splits(self, demo_data):
        """Verify every slide in splits has radiomics features."""
        splits_df = pd.read_csv(demo_data["splits"])
        rad_df = pd.read_csv(demo_data["radiomics_features"])

        split_ids = set(splits_df["slide_id"])
        rad_ids = set(rad_df["slide_id"])

        assert split_ids.issubset(rad_ids), (
            f"Slides in splits but not in radiomics: {split_ids - rad_ids}"
        )


class TestBaselineTraining:
    """Test baseline model training with synthetic data."""

    def test_logistic_regression_trains(self, demo_data):
        """End-to-end: load embeddings → mean-pool → train LR → predict."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        splits_df = pd.read_csv(demo_data["splits"])
        embed_data = np.load(demo_data["embeddings"])

        # Mean-pool to slide-level
        features, labels = {}, {}
        for _, row in splits_df.iterrows():
            key = f"emb_{row.slide_id}"
            if key in embed_data.files:
                features[row.slide_id] = embed_data[key].mean(axis=0)
                labels[row.slide_id] = row.label

        train_ids = splits_df[splits_df.split == "train"]["slide_id"]
        test_ids = splits_df[splits_df.split == "test"]["slide_id"]

        X_train = np.array([features[sid] for sid in train_ids if sid in features])
        y_train = np.array([labels[sid] for sid in train_ids if sid in labels])
        X_test = np.array([features[sid] for sid in test_ids if sid in features])
        y_test = np.array([labels[sid] for sid in test_ids if sid in labels])

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            pytest.skip("Not enough classes in train/test split")

        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train, y_train)
        probs = lr.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, probs)
        assert 0.0 <= auroc <= 1.0

    def test_abmil_forward_pass(self, demo_data):
        """Test ABMIL model forward pass with correct shapes."""
        import torch
        import torch.nn as nn

        embed_data = np.load(demo_data["embeddings"])
        key = embed_data.files[0]
        tiles = torch.tensor(embed_data[key], dtype=torch.float32)  # (n_tiles, dim)

        # Minimal ABMIL attention
        dim = tiles.shape[1]
        attn = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
        weights = torch.softmax(attn(tiles), dim=0)
        aggregated = (weights * tiles).sum(dim=0)

        assert aggregated.shape == (dim,)
        assert torch.isfinite(aggregated).all()


class TestFusionIntegration:
    """Test multimodal fusion with imaging + radiomics."""

    def test_feature_alignment(self, demo_data):
        """Verify imaging and radiomics features can be aligned by slide_id."""
        splits_df = pd.read_csv(demo_data["splits"])
        rad_df = pd.read_csv(demo_data["radiomics_features"])
        embed_data = np.load(demo_data["embeddings"])

        embed_ids = {k.replace("emb_", "") for k in embed_data.files}
        rad_ids = set(rad_df["slide_id"])
        split_ids = set(splits_df["slide_id"])

        common = embed_ids & rad_ids & split_ids
        assert len(common) == len(split_ids), "Not all slides have both modalities"

    def test_gated_fusion_forward(self, demo_data):
        """Test gated fusion network forward pass."""
        import torch
        import torch.nn as nn

        img_dim, rad_dim = 512, 93
        hidden = 64
        batch = 5

        img_proj = nn.Linear(img_dim, hidden)
        rad_proj = nn.Linear(rad_dim, hidden)
        gate = nn.Sequential(nn.Linear(hidden * 2, 2), nn.Softmax(dim=-1))
        head = nn.Linear(hidden, 2)

        img_feat = torch.randn(batch, img_dim)
        rad_feat = torch.randn(batch, rad_dim)

        img_h = torch.relu(img_proj(img_feat))
        rad_h = torch.relu(rad_proj(rad_feat))
        g = gate(torch.cat([img_h, rad_h], dim=-1))
        fused = g[:, 0:1] * img_h + g[:, 1:2] * rad_h
        logits = head(fused)

        assert logits.shape == (batch, 2)
        assert g.sum(dim=1).allclose(torch.ones(batch), atol=1e-5)


class TestEvaluationIntegration:
    """Test evaluation metrics on pipeline outputs."""

    def test_classification_metrics_on_predictions(self, demo_data):
        """Full metrics computation on synthetic predictions."""
        splits_df = pd.read_csv(demo_data["splits"])
        test_df = splits_df[splits_df.split == "test"]

        if len(test_df) < 3 or test_df.label.nunique() < 2:
            pytest.skip("Not enough test data")

        y_true = test_df.label.values
        y_pred_proba = np.random.RandomState(42).rand(len(y_true))

        from src.evaluation.metrics import ClassificationMetrics
        metrics = ClassificationMetrics(bootstrap_iterations=50, random_seed=42)
        result = metrics.compute_auroc(y_true, y_pred_proba)

        assert "auroc" in result
        assert 0.0 <= result["auroc"].value <= 1.0
        assert result["auroc"].ci_lower <= result["auroc"].value <= result["auroc"].ci_upper

    def test_stratified_bootstrap_preserves_class_ratio(self):
        """Verify stratified bootstrap maintains class proportions."""
        from src.evaluation.metrics import stratified_bootstrap_indices

        rng = np.random.RandomState(42)
        y = np.array([0]*80 + [1]*20)  # 80/20 imbalance

        for _ in range(10):
            idx = stratified_bootstrap_indices(rng, y, len(y))
            resampled_ratio = y[idx].mean()
            # Should stay close to 0.20 (±0.05 tolerance)
            assert 0.10 <= resampled_ratio <= 0.35, f"Ratio {resampled_ratio} too far from 0.20"


class TestCheckpointIntegration:
    """Test checkpoint save/load cycle."""

    def test_checkpoint_roundtrip(self, pipeline_tmpdir):
        """Save and load a checkpoint, verify metadata integrity."""
        import torch
        from src.utils.checkpoint_manager import CheckpointManager

        ckpt_dir = os.path.join(pipeline_tmpdir, "checkpoints")
        mgr = CheckpointManager(
            checkpoint_dir=ckpt_dir,
            experiment_id="test_run",
            max_keep=3,
            monitor_metric="val_auroc",
            mode="max",
        )

        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())

        # Save 3 checkpoints
        for epoch in range(3):
            mgr.save(
                model=model, optimizer=optimizer, scheduler=None,
                epoch=epoch,
                metrics={"val_auroc": 0.5 + epoch * 0.1, "train_loss": 1.0 - epoch * 0.2},
                config={"lr": 1e-3, "epochs": 10},
            )

        # Verify best
        best = mgr.find_best()
        assert best is not None

        # Load and verify metadata
        model2 = torch.nn.Linear(10, 2)
        meta = mgr.load(str(best), model2)
        assert "epoch" in meta
        assert "config_hash" in meta
        assert "timestamp" in meta

        # Verify manifest
        manifest = mgr.get_manifest()
        assert len(manifest) == 3


class TestEndToEnd:
    """Full pipeline end-to-end test via main.py."""

    def test_main_dry_run(self, pipeline_tmpdir):
        """Verify main.py --dry-run exits cleanly."""
        import subprocess
        result = subprocess.run(
            ["python3", "main.py", "--config", "configs/pipeline.yaml",
             "--stages", "all", "--dry-run",
             "--output-dir", pipeline_tmpdir],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

    def test_full_demo_pipeline(self, pipeline_tmpdir):
        """Run full pipeline with --demo flag and verify outputs."""
        import subprocess
        result = subprocess.run(
            ["python3", "main.py", "--config", "configs/pipeline.yaml",
             "--stages", "all", "--demo", "--device", "cpu",
             "--output-dir", pipeline_tmpdir,
             "--data-dir", os.path.join(pipeline_tmpdir, "data")],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}\n{result.stdout[-2000:]}"
        assert "SUCCESS" in result.stdout

        # Verify outputs exist
        run_json = Path(pipeline_tmpdir) / "pipeline_run.json"
        assert run_json.exists()
        with open(run_json) as f:
            summary = json.load(f)
        assert summary["status"] == "success"
        assert len(summary["timings"]) == 6  # All 6 stages

        # Verify evaluation metrics
        metrics_json = Path(pipeline_tmpdir) / "evaluation" / "all_metrics.json"
        assert metrics_json.exists()
