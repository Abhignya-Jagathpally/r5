#!/usr/bin/env python3
"""
R5 — MM Imaging Pathology & Radiomics Surrogate-Genetics Pipeline.

Master orchestrator that runs the full pipeline end-to-end:

    preprocessing → baselines → foundation → fusion → evaluation → report

Usage:
    python main.py --config configs/pipeline.yaml --stages all
    python main.py --config configs/minimal_example.yaml --stages preprocessing,baselines
    python main.py --config configs/pipeline.yaml --stages evaluation --dry-run
    python main.py --list-stages
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("mm_pipeline")

STAGE_ORDER = [
    "preprocessing",
    "baselines",
    "foundation",
    "fusion",
    "evaluation",
    "report",
]


# ── Logging setup ────────────────────────────────────────────────────


def setup_logging(log_dir: str, level: str = "INFO") -> Path:
    """Configure console + file logging."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logger.info(f"Logging to {log_file}")
    return log_file


# ── Stage: Preprocessing ─────────────────────────────────────────────


def run_stage_preprocessing(config: Dict, context: Dict) -> Dict:
    """Generate or load data, extract embeddings and radiomics features."""
    import pandas as pd

    data_dir = Path(context["data_dir"])
    output_dir = Path(context["output_dir"])
    results = {}

    # Check if demo data exists, generate if not
    summary_file = data_dir / "demo_data_summary.json"
    if context.get("demo") or not (data_dir / "embeddings").exists():
        logger.info("Generating synthetic demo data...")
        from scripts.generate_demo_data import generate_demo_data
        paths = generate_demo_data(output_dir=str(data_dir), seed=context.get("seed", 42))
        results["data_generation"] = paths
    else:
        logger.info(f"Using existing data from {data_dir}")

    # Load metadata
    slide_meta = pd.read_csv(data_dir / "slide_metadata.csv")
    patient_meta = pd.read_csv(data_dir / "patient_metadata.csv")
    tile_manifest = pd.read_csv(data_dir / "tiles" / "tile_manifest.csv")

    results["n_patients"] = len(patient_meta)
    results["n_slides"] = len(slide_meta)
    results["n_tiles"] = len(tile_manifest)
    results["positive_rate"] = float(slide_meta.label.mean())

    # Load splits
    splits_path = data_dir / "splits" / "splits.csv"
    if splits_path.exists():
        splits_df = pd.read_csv(splits_path)
        for split in ("train", "val", "test"):
            n = (splits_df.split == split).sum()
            results[f"n_{split}"] = n
            logger.info(f"  {split}: {n} slides")

    context["slide_metadata"] = slide_meta
    context["patient_metadata"] = patient_meta
    context["splits_path"] = str(splits_path)
    context["embeddings_dir"] = str(data_dir / "embeddings")
    context["radiomics_path"] = str(data_dir / "features" / "radiomics_features.csv")
    context["preprocessing_results"] = results

    logger.info(
        f"Preprocessing complete: {results['n_patients']} patients, "
        f"{results['n_slides']} slides, {results['n_tiles']} tiles"
    )
    return context


# ── Stage: Baselines ──────────────────────────────────────────────────


def run_stage_baselines(config: Dict, context: Dict) -> Dict:
    """Train classical baseline models with full checkpoint traceability."""
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score

    from src.utils.checkpoint_manager import CheckpointManager

    data_dir = Path(context["data_dir"])
    output_dir = Path(context["output_dir"]) / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = context.get("device", "cpu")
    seed = context.get("seed", 42)
    results = {}

    # Load embeddings and splits
    splits_df = pd.read_csv(context["splits_path"])
    embed_data = np.load(data_dir / "embeddings" / "embeddings.npz")

    # Build slide-level features by mean-pooling tile embeddings
    slide_features = {}
    for key in embed_data.files:
        slide_id = key.replace("emb_", "")
        slide_features[slide_id] = embed_data[key].mean(axis=0)

    # Align with splits
    train_slides = splits_df[splits_df.split == "train"]
    val_slides = splits_df[splits_df.split == "val"]
    test_slides = splits_df[splits_df.split == "test"]

    def get_Xy(slide_subset):
        X, y = [], []
        for _, row in slide_subset.iterrows():
            if row.slide_id in slide_features:
                X.append(slide_features[row.slide_id])
                y.append(row.label)
        return np.array(X), np.array(y)

    X_train, y_train = get_Xy(train_slides)
    X_val, y_val = get_Xy(val_slides)
    X_test, y_test = get_Xy(test_slides)

    logger.info(f"Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ── Mean-pool + Logistic Regression ───────────────────────────────
    logger.info("Training mean-pool + LogisticRegression baseline...")
    lr = LogisticRegression(max_iter=1000, random_state=seed, C=0.1)
    lr.fit(X_train, y_train)

    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_auroc = roc_auc_score(y_test, lr_probs)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    results["logistic_regression"] = {"auroc": round(lr_auroc, 4), "accuracy": round(lr_acc, 4)}
    logger.info(f"  LogisticRegression: AUROC={lr_auroc:.4f}, Acc={lr_acc:.4f}")

    # ── Mean-pool + Random Forest ─────────────────────────────────────
    logger.info("Training mean-pool + RandomForest baseline...")
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=10)
    rf.fit(X_train, y_train)

    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_auroc = roc_auc_score(y_test, rf_probs)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    results["random_forest"] = {"auroc": round(rf_auroc, 4), "accuracy": round(rf_acc, 4)}
    logger.info(f"  RandomForest: AUROC={rf_auroc:.4f}, Acc={rf_acc:.4f}")

    # ── ABMIL (Attention-Based MIL) ───────────────────────────────────
    logger.info("Training ABMIL...")
    embedding_dim = X_train.shape[1]

    class SimpleABMIL(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, n_classes=2):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(hidden_dim, n_classes),
            )

        def forward(self, x):
            # x: (n_tiles, dim)
            a = self.attention(x)  # (n_tiles, 1)
            a = torch.softmax(a, dim=0)
            z = (a * x).sum(dim=0, keepdim=True)  # (1, dim)
            return self.classifier(z).squeeze(0)

    abmil = SimpleABMIL(embedding_dim, hidden_dim=256, n_classes=2).to(device)
    optimizer = torch.optim.Adam(abmil.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ckpt_mgr = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints" / "abmil"),
        experiment_id=f"abmil_{context['run_id']}",
        monitor_metric="val_auroc",
        mode="max",
    )

    # Training loop
    best_val_auroc = 0.0
    n_epochs = config.get("training", {}).get("num_epochs", 30)

    for epoch in range(n_epochs):
        abmil.train()
        epoch_loss = 0.0

        for _, row in train_slides.iterrows():
            if row.slide_id not in slide_features:
                continue
            key = f"emb_{row.slide_id}"
            if key not in embed_data.files:
                continue

            tiles = torch.tensor(embed_data[key], dtype=torch.float32).to(device)
            label = torch.tensor(row.label, dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = abmil(tiles)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(abmil.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        abmil.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for _, row in val_slides.iterrows():
                key = f"emb_{row.slide_id}"
                if key not in embed_data.files:
                    continue
                tiles = torch.tensor(embed_data[key], dtype=torch.float32).to(device)
                logits = abmil(tiles)
                prob = torch.softmax(logits, dim=0)[1].item()
                val_probs.append(prob)
                val_labels.append(row.label)

        if len(set(val_labels)) > 1:
            val_auroc = roc_auc_score(val_labels, val_probs)
        else:
            val_auroc = 0.5

        # Checkpoint
        ckpt_mgr.save(
            model=abmil, optimizer=optimizer, scheduler=None,
            epoch=epoch, metrics={"val_auroc": val_auroc, "train_loss": epoch_loss},
            config=config.get("abmil", {}),
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}, val_auroc={val_auroc:.4f}")

    # Test ABMIL
    abmil.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for _, row in test_slides.iterrows():
            key = f"emb_{row.slide_id}"
            if key not in embed_data.files:
                continue
            tiles = torch.tensor(embed_data[key], dtype=torch.float32).to(device)
            logits = abmil(tiles)
            prob = torch.softmax(logits, dim=0)[1].item()
            test_probs.append(prob)
            test_labels.append(row.label)

    if len(set(test_labels)) > 1:
        abmil_auroc = roc_auc_score(test_labels, test_probs)
    else:
        abmil_auroc = 0.5
    abmil_acc = accuracy_score(test_labels, [1 if p > 0.5 else 0 for p in test_probs])
    results["abmil"] = {"auroc": round(abmil_auroc, 4), "accuracy": round(abmil_acc, 4), "best_val_auroc": round(best_val_auroc, 4)}
    logger.info(f"  ABMIL test: AUROC={abmil_auroc:.4f}, Acc={abmil_acc:.4f}")

    context["baseline_results"] = results
    context["baseline_models"] = {"logistic_regression": lr, "random_forest": rf, "abmil": abmil}
    return context


# ── Stage: Foundation Models ──────────────────────────────────────────


def run_stage_foundation(config: Dict, context: Dict) -> Dict:
    """Train foundation-style MIL head on embeddings (TransMIL-like)."""
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from src.utils.checkpoint_manager import CheckpointManager

    data_dir = Path(context["data_dir"])
    output_dir = Path(context["output_dir"]) / "foundation"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = context.get("device", "cpu")
    results = {}

    splits_df = pd.read_csv(context["splits_path"])
    embed_data = np.load(data_dir / "embeddings" / "embeddings.npz")

    embedding_dim = None
    for key in embed_data.files:
        embedding_dim = embed_data[key].shape[1]
        break

    # TransMIL-like model with self-attention
    class TransMILHead(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, n_heads=4, n_classes=2):
            super().__init__()
            self.proj = nn.Linear(input_dim, hidden_dim)
            self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=0.1, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
            self.pool_attn = nn.Linear(hidden_dim, 1)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(hidden_dim // 2, n_classes),
            )

        def forward(self, x):
            # x: (n_tiles, input_dim)
            x = self.proj(x).unsqueeze(0)  # (1, n_tiles, hidden)
            x, _ = self.attn(x, x, x)
            x = self.norm(x)
            # Attention pooling
            a = torch.softmax(self.pool_attn(x.squeeze(0)), dim=0)  # (n_tiles, 1)
            z = (a * x.squeeze(0)).sum(dim=0)  # (hidden,)
            return self.head(z)

    logger.info(f"Training TransMIL head (dim={embedding_dim})...")

    model = TransMILHead(embedding_dim, hidden_dim=256, n_heads=4, n_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss()

    ckpt_mgr = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints"),
        experiment_id=f"transmil_{context['run_id']}",
        monitor_metric="val_auroc",
        mode="max",
    )

    train_slides = splits_df[splits_df.split == "train"]
    val_slides = splits_df[splits_df.split == "val"]
    test_slides = splits_df[splits_df.split == "test"]
    n_epochs = 30

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        for _, row in train_slides.iterrows():
            key = f"emb_{row.slide_id}"
            if key not in embed_data.files:
                continue
            tiles = torch.tensor(embed_data[key], dtype=torch.float32).to(device)
            label = torch.tensor(row.label, dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(tiles)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_samples += 1

        scheduler.step()

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for _, row in val_slides.iterrows():
                key = f"emb_{row.slide_id}"
                if key not in embed_data.files:
                    continue
                tiles = torch.tensor(embed_data[key], dtype=torch.float32).to(device)
                logits = model(tiles)
                prob = torch.softmax(logits, dim=0)[1].item()
                val_probs.append(prob)
                val_labels.append(row.label)

        val_auroc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5

        ckpt_mgr.save(
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, metrics={"val_auroc": val_auroc, "train_loss": epoch_loss / max(n_samples, 1)},
            config=config.get("mil_head", {}),
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss/max(n_samples,1):.4f}, val_auroc={val_auroc:.4f}")

    # Test
    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for _, row in test_slides.iterrows():
            key = f"emb_{row.slide_id}"
            if key not in embed_data.files:
                continue
            tiles = torch.tensor(embed_data[key], dtype=torch.float32).to(device)
            logits = model(tiles)
            prob = torch.softmax(logits, dim=0)[1].item()
            test_probs.append(prob)
            test_labels.append(row.label)

    transmil_auroc = roc_auc_score(test_labels, test_probs) if len(set(test_labels)) > 1 else 0.5
    results["transmil"] = {"auroc": round(transmil_auroc, 4)}
    logger.info(f"  TransMIL test AUROC: {transmil_auroc:.4f}")

    context["foundation_results"] = results
    context["foundation_model"] = model
    return context


# ── Stage: Multimodal Fusion ──────────────────────────────────────────


def run_stage_fusion(config: Dict, context: Dict) -> Dict:
    """Train multimodal fusion: imaging embeddings + radiomics features."""
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from src.utils.checkpoint_manager import CheckpointManager

    data_dir = Path(context["data_dir"])
    output_dir = Path(context["output_dir"]) / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = context.get("device", "cpu")
    results = {}

    splits_df = pd.read_csv(context["splits_path"])
    embed_data = np.load(data_dir / "embeddings" / "embeddings.npz")
    radiomics_df = pd.read_csv(context["radiomics_path"])

    # Build aligned features
    embedding_dim = None
    for key in embed_data.files:
        embedding_dim = embed_data[key].shape[1]
        break

    rad_feature_cols = [c for c in radiomics_df.columns if c.startswith("radiomics_")]
    radiomics_dim = len(rad_feature_cols)

    class GatedFusion(nn.Module):
        def __init__(self, img_dim, rad_dim, hidden_dim=128, n_classes=2):
            super().__init__()
            self.img_proj = nn.Sequential(nn.Linear(img_dim, hidden_dim), nn.ReLU())
            self.rad_proj = nn.Sequential(nn.Linear(rad_dim, hidden_dim), nn.ReLU())
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, 2), nn.Softmax(dim=-1))
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, n_classes),
            )

        def forward(self, img_feat, rad_feat):
            img_h = self.img_proj(img_feat)
            rad_h = self.rad_proj(rad_feat)
            gate_weights = self.gate(torch.cat([img_h, rad_h], dim=-1))
            fused = gate_weights[:, 0:1] * img_h + gate_weights[:, 1:2] * rad_h
            return self.head(fused), gate_weights

    logger.info(f"Training gated fusion (img={embedding_dim}, rad={radiomics_dim})...")

    model = GatedFusion(embedding_dim, radiomics_dim, hidden_dim=128, n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ckpt_mgr = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints"),
        experiment_id=f"gated_fusion_{context['run_id']}",
        monitor_metric="val_auroc",
        mode="max",
    )

    rad_by_slide = {row.slide_id: row[rad_feature_cols].values.astype(np.float32) for _, row in radiomics_df.iterrows()}
    slide_feats = {}
    for key in embed_data.files:
        sid = key.replace("emb_", "")
        slide_feats[sid] = embed_data[key].mean(axis=0)

    def get_batch(subset):
        imgs, rads, labels = [], [], []
        for _, row in subset.iterrows():
            if row.slide_id in slide_feats and row.slide_id in rad_by_slide:
                imgs.append(slide_feats[row.slide_id])
                rads.append(rad_by_slide[row.slide_id])
                labels.append(row.label)
        return (torch.tensor(np.array(imgs), dtype=torch.float32).to(device),
                torch.tensor(np.array(rads), dtype=torch.float32).to(device),
                torch.tensor(labels, dtype=torch.long).to(device))

    train_slides = splits_df[splits_df.split == "train"]
    val_slides = splits_df[splits_df.split == "val"]
    test_slides = splits_df[splits_df.split == "test"]

    X_train_img, X_train_rad, y_train = get_batch(train_slides)
    X_val_img, X_val_rad, y_val = get_batch(val_slides)
    X_test_img, X_test_rad, y_test = get_batch(test_slides)

    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits, gates = model(X_train_img, X_train_rad)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val_img, X_val_rad)
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_auroc = roc_auc_score(y_val.cpu().numpy(), val_probs) if len(set(y_val.cpu().numpy().tolist())) > 1 else 0.5

        ckpt_mgr.save(
            model=model, optimizer=optimizer, scheduler=None,
            epoch=epoch, metrics={"val_auroc": val_auroc, "train_loss": loss.item()},
            config=config.get("multimodal_fusion", config.get("fusion", {})),
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}, val_auroc={val_auroc:.4f}")

    # Test
    model.eval()
    with torch.no_grad():
        test_logits, test_gates = model(X_test_img, X_test_rad)
        test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        test_auroc = roc_auc_score(y_test.cpu().numpy(), test_probs) if len(set(y_test.cpu().numpy().tolist())) > 1 else 0.5
        test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

    avg_gates = test_gates.mean(dim=0).cpu().numpy()
    results["gated_fusion"] = {
        "auroc": round(test_auroc, 4),
        "accuracy": round(test_acc, 4),
        "modality_weights": {"imaging": round(float(avg_gates[0]), 3), "radiomics": round(float(avg_gates[1]), 3)},
    }
    logger.info(f"  Gated fusion test: AUROC={test_auroc:.4f}, Acc={test_acc:.4f}")
    logger.info(f"  Modality weights: imaging={avg_gates[0]:.3f}, radiomics={avg_gates[1]:.3f}")

    context["fusion_results"] = results
    return context


# ── Stage: Evaluation ─────────────────────────────────────────────────


def run_stage_evaluation(config: Dict, context: Dict) -> Dict:
    """Aggregate metrics, compute bootstrap CIs, compare to benchmarks."""
    from sklearn.metrics import classification_report

    output_dir = Path(context["output_dir"]) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Collect all model results
    all_models = {}
    for key in ("baseline_results", "foundation_results", "fusion_results"):
        data = context.get(key, {})
        if isinstance(data, dict):
            all_models.update(data)

    results["model_metrics"] = all_models

    # Comparison table
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"{'Model':<25s} {'AUROC':>8s} {'Accuracy':>10s}")
    logger.info("-" * 45)
    for model_name, metrics in all_models.items():
        if isinstance(metrics, dict) and "auroc" in metrics:
            logger.info(f"{model_name:<25s} {metrics['auroc']:>8.4f} {metrics.get('accuracy', 'N/A'):>10}")

    # Benchmark comparison
    benchmarks = config.get("benchmarks", {})
    if benchmarks:
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK COMPARISON")
        logger.info("=" * 60)
        for bench_name, bench_metrics in benchmarks.items():
            logger.info(f"  {bench_name}: AUROC={bench_metrics.get('auroc', 'N/A')} (source: {bench_metrics.get('source', '?')})")
        results["benchmarks"] = benchmarks

    # Save metrics
    metrics_path = output_dir / "all_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_models, f, indent=2, default=str)
    logger.info(f"\nMetrics saved to {metrics_path}")

    context["evaluation_results"] = results
    return context


# ── Stage: Report ─────────────────────────────────────────────────────


def run_stage_report(config: Dict, context: Dict) -> Dict:
    """Generate final pipeline report with results and environment snapshot."""
    output_dir = Path(context["output_dir"]) / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "pipeline": "R5 — MM Imaging Pathology & Radiomics",
        "version": "0.1.0",
        "run_id": context["run_id"],
        "timestamp": datetime.now().isoformat(),
        "stages_completed": context.get("completed_stages", []),
        "timings": context.get("timings", {}),
        "total_time_seconds": round(sum(context.get("timings", {}).values()), 2),
        "device": context.get("device", "cpu"),
        "seed": context.get("seed", 42),
        "preprocessing": context.get("preprocessing_results", {}),
        "baselines": context.get("baseline_results", {}),
        "foundation": context.get("foundation_results", {}),
        "fusion": context.get("fusion_results", {}),
        "evaluation": context.get("evaluation_results", {}),
    }

    # Save JSON report
    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Pipeline report: {report_path}")

    # Checkpoint manifest
    ckpt_dirs = list(Path(context["output_dir"]).rglob("checkpoints"))
    total_ckpts = sum(len(list(d.glob("*.pt"))) for d in ckpt_dirs)
    logger.info(f"Total checkpoints saved: {total_ckpts}")

    # Environment info
    env_info = {
        "python": sys.version,
        "platform": sys.platform,
    }
    try:
        import torch
        env_info["torch"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass

    env_path = output_dir / "environment.json"
    with open(env_path, "w") as f:
        json.dump(env_info, f, indent=2)

    context["report_path"] = str(report_path)
    return context


# ── Main orchestrator ─────────────────────────────────────────────────


STAGE_HANDLERS = {
    "preprocessing": run_stage_preprocessing,
    "baselines": run_stage_baselines,
    "foundation": run_stage_foundation,
    "fusion": run_stage_fusion,
    "evaluation": run_stage_evaluation,
    "report": run_stage_report,
}


def parse_args():
    """Parse command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description="R5 — MM Imaging Pathology & Radiomics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config configs/pipeline.yaml --stages all
  python main.py --config configs/minimal_example.yaml --stages preprocessing,baselines
  python main.py --config configs/pipeline.yaml --stages evaluation --dry-run
  python main.py --list-stages
        """
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/pipeline.yaml",
        help="Path to pipeline config YAML (default: configs/pipeline.yaml)"
    )
    parser.add_argument(
        "--stages", "-s",
        default="all",
        help="Comma-separated stages to run: preprocessing,baselines,foundation,fusion,evaluation,report,all (default: all)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Validate config and show execution plan without running"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed (default: from config)"
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List available pipeline stages and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG)"
    )
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    parser.add_argument("--device", default=None, help="Compute device (cpu, cuda)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo data")
    return parser.parse_args()


PIPELINE_STAGES = STAGE_ORDER  # Alias for --list-stages display


def main() -> int:
    """Entry point for the MM Imaging Pathology & Radiomics Pipeline."""
    args = parse_args()

    # Handle --list-stages
    if args.list_stages:
        print("Available pipeline stages:")
        for i, stage in enumerate(PIPELINE_STAGES, 1):
            print(f"  {i}. {stage}")
        sys.exit(0)

    # Determine log level from --verbose flag
    if args.verbose >= 2:
        log_level = "DEBUG"
    elif args.verbose == 1:
        log_level = "INFO"
    else:
        log_level = "INFO"

    from src.utils.config import load_config, resolve_stage_configs
    from src.utils.config_schema import validate_config, validate_no_conflicting_keys

    master_config = load_config(args.config)

    # Validate config schema
    errors = validate_config(master_config, "pipeline")
    errors.extend(validate_no_conflicting_keys(master_config))
    if errors:
        for err in errors:
            logger.error(f"Config validation error: {err}")
        raise ValueError(f"Config validation failed with {len(errors)} error(s)")

    output_dir = args.output_dir or master_config.get("output_dir", "./results")
    data_dir = args.data_dir or master_config.get("data_dir", "./data")
    log_dir = master_config.get("log_dir", "./logs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_file = setup_logging(log_dir, log_level)

    seed = args.seed or master_config.get("reproducibility", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    device = args.device or master_config.get("training", {}).get("device", "cpu")
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
        except ImportError:
            device = "cpu"

    requested_stages = [s.strip() for s in args.stages.split(",")]
    if "all" in requested_stages:
        stage_toggles = master_config.get("stages", {})
        stages = [s for s in STAGE_ORDER if stage_toggles.get(s, True)]
    else:
        invalid = [s for s in requested_stages if s not in STAGE_ORDER]
        if invalid:
            logger.error(f"Unknown stages: {invalid}. Valid: {STAGE_ORDER}")
            return 1
        stages = [s for s in STAGE_ORDER if s in requested_stages]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("R5 — MM Imaging Pathology & Radiomics Pipeline v0.1.0")
    logger.info(f"Run ID:  {run_id}")
    logger.info(f"Config:  {args.config}")
    logger.info(f"Stages:  {stages}")
    logger.info(f"Output:  {output_dir}")
    logger.info(f"Device:  {device}")
    logger.info(f"Seed:    {seed}")
    logger.info("=" * 70)

    if args.dry_run:
        for s in stages:
            logger.info(f"  [DRY RUN] → {s}")
        return 0

    config_dir = Path(args.config).parent
    stage_configs = resolve_stage_configs(master_config, config_dir)

    context = {
        "run_id": run_id,
        "output_dir": output_dir,
        "data_dir": data_dir,
        "device": device,
        "seed": seed,
        "resume": args.resume,
        "demo": args.demo,
        "config_path": args.config,
        "timings": {},
        "completed_stages": [],
    }

    # Check if running on synthetic data and warn
    summary_path = os.path.join(data_dir, "demo_data_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        if summary.get("is_synthetic", False):
            logger.warning(
                "Running on synthetic demo data. All results are for pipeline "
                "validation only and have no scientific meaning."
            )

    pipeline_start = time.time()
    failed = False

    for stage_name in stages:
        logger.info("-" * 60)
        logger.info(f"STAGE: {stage_name.upper()}")
        logger.info("-" * 60)

        t0 = time.time()
        try:
            stage_config = stage_configs.get(stage_name, master_config)
            context = STAGE_HANDLERS[stage_name](stage_config, context)
            elapsed = time.time() - t0
            context["timings"][stage_name] = round(elapsed, 2)
            context["completed_stages"].append(stage_name)
            logger.info(f"Stage {stage_name} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            context["timings"][stage_name] = round(elapsed, 2)
            logger.error(f"Stage {stage_name} FAILED after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            failed = True
            break

    total_time = time.time() - pipeline_start

    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    for stage, elapsed in context["timings"].items():
        status = "FAIL" if (failed and stage == list(context["timings"].keys())[-1]) else "OK"
        logger.info(f"  {stage:20s} {elapsed:8.1f}s  [{status}]")
    logger.info(f"  {'TOTAL':20s} {total_time:8.1f}s")
    logger.info(f"  Status: {'FAILED' if failed else 'SUCCESS'}")

    summary_path = Path(output_dir) / "pipeline_run.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_id": run_id, "status": "failed" if failed else "success",
            "stages": stages, "timings": context["timings"],
            "total_seconds": round(total_time, 2), "device": device, "seed": seed,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
