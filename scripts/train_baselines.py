"""
Training orchestration script for classical baselines.

Supports:
- Tile classifier (per-tile ResNet50)
- Mean pool baseline (simplest baseline)
- ABMIL (Attention-Based MIL)
- CLAM (Clustering-constrained Attention MIL)
- Radiomics + Survival models

Usage:
    python scripts/train_baselines.py \
        --config configs/model_baselines.yaml \
        --model abmil \
        --output_dir results/abmil
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Import models
from src.models.tile_classifier import TileClassifier, TileClassifierTrainer
from src.models.mean_pool_baseline import MeanPoolBaseline
from src.models.abmil import ABMIL, ABMILTrainer
from src.models.clam import CLAM_SB, CLAM_MB, CLAMTrainer
from src.models.radiomics_survival import (
    CoxProportionalHazards,
    RandomSurvivalForest,
    FeatureSelector,
)
from src.models.mil_dataset import SplitsManager


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to file and console.

    Args:
        output_dir: Directory for log files

    Returns:
        Logger instance
    """
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_dir / "train.log")
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to config YAML

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_mean_pool_baseline(
    config: Dict,
    output_dir: str,
    logger: logging.Logger,
) -> Dict[str, any]:
    """Train mean pooling baseline.

    Args:
        config: Configuration dictionary
        output_dir: Output directory
        logger: Logger instance

    Returns:
        Results dictionary
    """
    logger.info("Training Mean Pool Baseline")

    # Initialize baseline
    baseline = MeanPoolBaseline(
        embedding_dim=config["mean_pool"]["embedding_dim"],
        classifiers=config["mean_pool"]["classifiers"],
    )

    # Load splits
    splits_manager = SplitsManager(config["data"]["splits_csv"])
    train_slide_ids, train_labels = splits_manager.get_split("train")
    val_slide_ids, val_labels = splits_manager.get_split("val")
    test_slide_ids, test_labels = splits_manager.get_split("test")

    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    # Load embeddings
    logger.info("Loading training embeddings...")
    train_embeddings, train_valid_ids = baseline.load_embeddings_from_zarr(
        config["data"]["zarr_path"],
        train_slide_ids,
    )

    logger.info("Loading validation embeddings...")
    val_embeddings, val_valid_ids = baseline.load_embeddings_from_zarr(
        config["data"]["zarr_path"],
        val_slide_ids,
    )

    logger.info("Loading test embeddings...")
    test_embeddings, test_valid_ids = baseline.load_embeddings_from_zarr(
        config["data"]["zarr_path"],
        test_slide_ids,
    )

    # Filter labels to match valid slides
    train_indices = [i for i, sid in enumerate(train_slide_ids) if sid in train_valid_ids]
    train_labels = train_labels[train_indices]

    val_indices = [i for i, sid in enumerate(val_slide_ids) if sid in val_valid_ids]
    val_labels = val_labels[val_indices]

    test_indices = [i for i, sid in enumerate(test_slide_ids) if sid in test_valid_ids]
    test_labels = test_labels[test_indices]

    # Train
    logger.info(f"Training set size: {len(train_embeddings)}")
    train_results = baseline.fit(train_embeddings, train_labels, verbose=True)

    # Evaluate on all splits
    results = {"model": "mean_pool"}

    for clf_name in baseline.classifiers:
        logger.info(f"\n=== {clf_name} ===")

        # Train metrics
        for key, value in train_results.items():
            if clf_name in key:
                results[f"train_{key}"] = value
                logger.info(f"Train {key}: {value:.4f}")

        # Val metrics
        val_metrics = baseline.evaluate(val_embeddings, val_labels, classifier=clf_name)
        results[f"val_{clf_name}_acc"] = val_metrics["accuracy"]
        results[f"val_{clf_name}_f1"] = val_metrics["f1"]
        results[f"val_{clf_name}_auroc"] = val_metrics.get("auroc", 0.0)
        logger.info(f"Val Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val F1: {val_metrics['f1']:.4f}")
        logger.info(f"Val AUROC: {val_metrics.get('auroc', 0.0):.4f}")

        # Test metrics
        test_metrics = baseline.evaluate(test_embeddings, test_labels, classifier=clf_name)
        results[f"test_{clf_name}_acc"] = test_metrics["accuracy"]
        results[f"test_{clf_name}_f1"] = test_metrics["f1"]
        results[f"test_{clf_name}_auroc"] = test_metrics.get("auroc", 0.0)
        logger.info(f"Test Acc: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        logger.info(f"Test AUROC: {test_metrics.get('auroc', 0.0):.4f}")

    return results


def train_abmil(
    config: Dict,
    output_dir: str,
    logger: logging.Logger,
) -> Dict[str, any]:
    """Train ABMIL model.

    Args:
        config: Configuration dictionary
        output_dir: Output directory
        logger: Logger instance

    Returns:
        Results dictionary
    """
    logger.info("Training ABMIL")

    device = torch.device(config["training"]["device"])

    # Load splits and create dataloaders
    splits_manager = SplitsManager(config["data"]["splits_csv"])
    dataloaders = splits_manager.get_dataloaders(
        zarr_path=config["data"]["zarr_path"],
        batch_size=config["abmil"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        max_patches=config["data"]["max_patches"],
    )

    if "train" not in dataloaders or "val" not in dataloaders:
        raise ValueError("Train and val dataloaders required")

    # Initialize model
    model = ABMIL(
        input_dim=config["abmil"]["input_dim"],
        hidden_dim=config["abmil"]["hidden_dim"],
        attention_dim=config["abmil"]["attention_dim"],
        num_classes=config["abmil"]["num_classes"],
        dropout=config["abmil"]["dropout"],
        gated=config["abmil"].get("gated", True),
    )

    # Initialize trainer
    trainer = ABMILTrainer(
        model=model,
        device=device,
        learning_rate=config["abmil"]["learning_rate"],
        weight_decay=config["abmil"]["weight_decay"],
        max_epochs=config["abmil"]["max_epochs"],
        patience=config["abmil"]["patience"],
        checkpoint_dir=output_dir,
    )

    # Train
    logger.info(f"Training for max {config['abmil']['max_epochs']} epochs")
    history = trainer.train(
        dataloaders["train"],
        dataloaders["val"],
        verbose=config["training"]["verbose"],
    )

    results = {
        "model": "abmil",
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
    }

    # Test if available
    if "test" in dataloaders:
        logger.info("Evaluating on test set...")
        test_loss, test_acc = trainer.validate(dataloaders["test"])
        results["test_loss"] = test_loss
        results["test_acc"] = test_acc
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    return results


def train_clam(
    config: Dict,
    output_dir: str,
    logger: logging.Logger,
    model_type: str = "clam_sb",
) -> Dict[str, any]:
    """Train CLAM model.

    Args:
        config: Configuration dictionary
        output_dir: Output directory
        logger: Logger instance
        model_type: 'clam_sb' or 'clam_mb'

    Returns:
        Results dictionary
    """
    logger.info(f"Training {model_type.upper()}")

    device = torch.device(config["training"]["device"])

    # Load configuration for this model type
    model_config = config[model_type]

    # Load splits and create dataloaders
    splits_manager = SplitsManager(config["data"]["splits_csv"])
    dataloaders = splits_manager.get_dataloaders(
        zarr_path=config["data"]["zarr_path"],
        batch_size=model_config["batch_size"],
        num_workers=config["data"]["num_workers"],
        max_patches=config["data"]["max_patches"],
    )

    if "train" not in dataloaders or "val" not in dataloaders:
        raise ValueError("Train and val dataloaders required")

    # Initialize model
    if model_type == "clam_sb":
        model = CLAM_SB(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            attention_dim=model_config["attention_dim"],
            num_classes=model_config["num_classes"],
            dropout=model_config["dropout"],
        )
    else:  # clam_mb
        model = CLAM_MB(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            attention_dim=model_config["attention_dim"],
            num_classes=model_config["num_classes"],
            num_heads=model_config.get("num_heads", 3),
            dropout=model_config["dropout"],
            inst_cluster=model_config["inst_cluster"],
        )

    # Initialize trainer
    trainer = CLAMTrainer(
        model=model,
        device=device,
        learning_rate=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
        max_epochs=model_config["max_epochs"],
        patience=model_config["patience"],
        inst_lambda=model_config.get("inst_lambda", 0.0),
        checkpoint_dir=output_dir,
    )

    # Train
    logger.info(f"Training for max {model_config['max_epochs']} epochs")
    history = trainer.train(
        dataloaders["train"],
        dataloaders["val"],
        verbose=config["training"]["verbose"],
    )

    results = {
        "model": model_type,
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
    }

    # Test if available
    if "test" in dataloaders:
        logger.info("Evaluating on test set...")
        test_loss, test_acc = trainer.validate(dataloaders["test"])
        results["test_loss"] = test_loss
        results["test_acc"] = test_acc
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    return results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train classical baseline models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_baselines.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mean_pool", "abmil", "clam_sb", "clam_mb"],
        default="mean_pool",
        help="Model to train",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory",
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    logger = setup_logging(output_dir)

    logger.info(f"Training {args.model} with config {args.config}")

    # Train
    if args.model == "mean_pool":
        results = train_mean_pool_baseline(config, output_dir, logger)
    elif args.model == "abmil":
        results = train_abmil(config, output_dir, logger)
    elif args.model.startswith("clam"):
        results = train_clam(config, output_dir, logger, model_type=args.model)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
