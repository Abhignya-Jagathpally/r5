#!/usr/bin/env python3
"""
Extract features using foundation models on a directory of slides/tiles.

This script:
1. Loads the specified foundation model (UNI2-h, TITAN, or ResNet50)
2. Processes all slides/tiles in input directory
3. Saves embeddings to Zarr embedding store
4. Supports resume on interruption
5. Manages GPU memory efficiently
6. Tracks progress with tqdm

Usage:
    python extract_foundation_features.py \
        --input-dir /path/to/tiles/ \
        --output-dir /path/to/embeddings/ \
        --backbone uni2h \
        --batch-size 64 \
        --num-workers 4

Configuration:
    Uses configs/foundation_models.yaml for default hyperparameters.
    CLI arguments override config file settings.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np
import yaml
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.foundation import get_extractor
from src.data.embedding_store import ZarrEmbeddingStore

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, level: int = logging.INFO):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    # File handler
    log_file = log_dir / "extraction.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_file


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config or {}


def expand_path(path_str: str, config: Dict[str, Any]) -> Path:
    """Expand path variables like ${HOME}."""
    # Simple variable expansion
    path_str = path_str.replace("${HOME}", str(Path.home()))

    return Path(path_str).expanduser().resolve()


def extract_features(
    input_dir: Path,
    output_dir: Path,
    backbone: str = "uni2h",
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[str] = None,
    resume: bool = True,
    config: Optional[Dict[str, Any]] = None,
    hf_token: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract features from tiles/slides.

    Args:
        input_dir: Directory containing tiles/slides
        output_dir: Directory to save embeddings
        backbone: Feature extractor backbone
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        device: Device to use (cuda/cpu)
        resume: Resume from cached embeddings
        config: Configuration dictionary
        hf_token: HuggingFace authentication token

    Returns:
        Dictionary of extracted embeddings
    """
    config = config or {}

    logger.info(f"Extracting features from {input_dir} using {backbone}")
    logger.info(f"Saving to {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize feature extractor
    logger.info(f"Initializing {backbone} extractor...")
    try:
        extractor = get_extractor(
            backbone,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            hf_token=hf_token,
        )
        logger.info(f"Extractor initialized. Output dimension: {extractor.embedding_dim}")
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        raise

    # Extract embeddings
    logger.info("Starting feature extraction...")
    try:
        embeddings = extractor.extract_batch(
            input_dir,
            output_path=output_dir / "embeddings.pkl",
            verbose=True,
        )
        logger.info(f"Extracted {len(embeddings)} embeddings")

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise

    # Optionally save to Zarr store
    try:
        logger.info("Saving to Zarr embedding store...")
        embedding_store = ZarrEmbeddingStore(
            storage_path=output_dir / "embeddings.zarr",
            embedding_dim=extractor.embedding_dim,
            compressor="blosc",
        )

        for slide_id, embedding in tqdm(embeddings.items(), desc="Writing to Zarr"):
            embedding_store[slide_id] = embedding

        logger.info(f"Embeddings saved to Zarr: {output_dir / 'embeddings.zarr'}")

    except Exception as e:
        logger.warning(f"Failed to save to Zarr (continuing): {e}")

    # Print summary
    logger.info("=" * 60)
    logger.info("Feature Extraction Summary")
    logger.info("=" * 60)
    logger.info(f"Backbone: {backbone}")
    logger.info(f"Embedding dimension: {extractor.embedding_dim}")
    logger.info(f"Number of embeddings: {len(embeddings)}")
    logger.info(f"Model config: {extractor.model_config}")
    logger.info("=" * 60)

    return embeddings


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract features using foundation models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Extract UNI2-h features:
    python extract_foundation_features.py \\
        --input-dir data/tiles \\
        --output-dir data/embeddings \\
        --backbone uni2h

  Extract with custom batch size:
    python extract_foundation_features.py \\
        --input-dir data/tiles \\
        --output-dir data/embeddings \\
        --backbone uni2h \\
        --batch-size 128

  Use CPU:
    python extract_foundation_features.py \\
        --input-dir data/tiles \\
        --output-dir data/embeddings \\
        --device cpu
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing tiles/slides",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save embeddings",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="uni2h",
        choices=["resnet50", "uni2h", "titan"],
        help="Feature extractor backbone",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (auto-detect if not specified)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "foundation_models.yaml",
        help="Configuration file",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace authentication token (for UNI2-h)",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./logs/extraction"),
        help="Directory for log files",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from cached embeddings",
    )

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging(args.log_dir)
    logger.info(f"Logging to {log_file}")

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Validate paths
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Log system info
    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info("=" * 60)

    # Run extraction
    try:
        embeddings = extract_features(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            backbone=args.backbone,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            resume=not args.no_resume,
            config=config,
            hf_token=args.hf_token,
        )

        logger.info("Feature extraction completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Feature extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
