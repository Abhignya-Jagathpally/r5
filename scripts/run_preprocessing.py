#!/usr/bin/env python3
"""
End-to-end WSI preprocessing pipeline.

Orchestrates tiling, quality filtering, stain normalization, deduplication,
and embedding extraction with progress tracking and resume capability.

Usage:
    python scripts/run_preprocessing.py --config configs/data_pipeline.yaml \
        --slides data/raw/*.svs \
        --output_dir data/preprocessed
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import yaml
from tqdm import tqdm

from src.data import (
    WSITiler,
    StainNormalizer,
    TileDeduplicator,
    EmbeddingExtractor,
    EmbeddingStore,
)

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, log_level: str = "INFO"):
    """Setup logging to file and console."""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    log_file = log_dir_path / "pipeline.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return log_file


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_processed_slides(manifest_path: str) -> set:
    """Get set of already-processed slide IDs."""
    if not Path(manifest_path).exists():
        return set()

    df = pd.read_csv(manifest_path)
    return set(df["slide_id"].unique())


def run_tiling(
    slide_paths: List[str],
    slide_ids: List[str],
    config: Dict[str, Any],
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Run WSI tiling step."""
    logger.info("=" * 60)
    logger.info("STEP 1: WSI TILING")
    logger.info("=" * 60)

    tiles_dir = config["storage"]["tiles_dir"]
    manifest_path = Path(tiles_dir) / "tile_manifest.csv"

    # Check for existing tiles
    processed = get_processed_slides(str(manifest_path)) if skip_existing else set()

    slides_to_process = [
        (path, sid) for path, sid in zip(slide_paths, slide_ids) if sid not in processed
    ]

    if not slides_to_process:
        logger.info("All slides already tiled, skipping...")
        manifest_df = pd.read_csv(manifest_path)
        return manifest_df

    logger.info(f"Tiling {len(slides_to_process)}/{len(slide_paths)} slides")

    tiler = WSITiler(
        tile_size=config["tiling"]["tile_size"],
        magnification=config["tiling"]["magnification"],
        overlap=config["tiling"]["overlap"],
        min_tissue_fraction=config["tiling"]["min_tissue_fraction"],
        output_dir=tiles_dir,
    )

    slide_paths_to_process = [p for p, _ in slides_to_process]
    slide_ids_to_process = [s for _, s in slides_to_process]

    manifest_df = tiler.process_slides(
        slide_paths_to_process,
        slide_ids_to_process,
        max_workers=config["tiling"]["max_workers"],
        laplacian_threshold=config["tiling"]["laplacian_threshold"],
    )

    logger.info(f"Tiling complete: {len(manifest_df)} tiles extracted")
    return manifest_df


def run_quality_filtering(
    tiles_dir: str, config: Dict[str, Any]
) -> Dict[str, int]:
    """Run quality filtering step (integrated with stain normalization)."""
    logger.info("=" * 60)
    logger.info("STEP 2: QUALITY FILTERING")
    logger.info("=" * 60)

    normalizer = StainNormalizer(
        method="identity",  # Just for filtering
        max_background_fraction=config["quality_filter"]["max_background_fraction"],
        min_laplacian_variance=config["quality_filter"]["min_laplacian_variance"],
        pen_mark_detection=config["quality_filter"]["pen_mark_detection"],
    )

    tiles_path = Path(tiles_dir)
    tile_files = list(tiles_path.glob("*.png"))

    logger.info(f"Filtering {len(tile_files)} tiles...")

    stats = {"total": 0, "kept": 0, "filtered": 0}

    removed_files = []
    for tile_file in tqdm(tile_files, desc="Quality filtering"):
        try:
            import cv2

            image = cv2.imread(str(tile_file))
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            stats["total"] += 1

            if not normalizer.is_high_quality(image):
                stats["filtered"] += 1
                # Remove low-quality tile so downstream stages don't process it
                removed_files.append(tile_file)
                tile_file.unlink()
            else:
                stats["kept"] += 1

        except Exception as e:
            logger.warning(f"Error filtering {tile_file}: {e}")

    if removed_files:
        logger.info(f"Removed {len(removed_files)} low-quality tiles from disk")

    logger.info(
        f"Quality filtering complete: {stats['kept']}/{stats['total']} kept"
    )
    return stats


def run_stain_normalization(
    tiles_dir: str, normalized_dir: str, config: Dict[str, Any]
) -> Dict[str, int]:
    """Run stain normalization step."""
    logger.info("=" * 60)
    logger.info("STEP 3: STAIN NORMALIZATION")
    logger.info("=" * 60)

    normalizer = StainNormalizer(
        method=config["stain_norm"]["method"],
        max_background_fraction=config["quality_filter"]["max_background_fraction"],
        min_laplacian_variance=config["quality_filter"]["min_laplacian_variance"],
        pen_mark_detection=config["quality_filter"]["pen_mark_detection"],
    )

    reference = config["stain_norm"]["reference_slide"]

    stats = normalizer.process_tile_directory(
        tiles_dir, normalized_dir, reference_image=reference
    )

    logger.info(
        f"Stain normalization complete: {stats['normalized']} normalized"
    )
    return stats


def run_deduplication(
    tiles_dir: str, deduplicated_dir: str, config: Dict[str, Any]
) -> Dict[str, int]:
    """Run deduplication step."""
    logger.info("=" * 60)
    logger.info("STEP 4: DEDUPLICATION")
    logger.info("=" * 60)

    dedup = TileDeduplicator(
        hamming_threshold=config["dedup"]["hamming_threshold"],
        hash_algorithm=config["dedup"]["method"],
    )

    stats = dedup.deduplicate_directory(
        tiles_dir,
        output_directory=deduplicated_dir,
        remove_duplicates=config["dedup"]["remove_duplicates"],
    )

    if config["dedup"]["save_report"]:
        report_path = Path(deduplicated_dir) / "dedup_report.csv"
        dedup.save_report(str(report_path))
        logger.info(f"Dedup report saved to {report_path}")

    logger.info(f"Deduplication complete: {stats['num_removed']} removed")
    return stats


def run_embedding_extraction(
    tiles_dir: str,
    embeddings_dir: str,
    metadata_path: str,
    slide_metadata: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """Run embedding extraction step."""
    logger.info("=" * 60)
    logger.info("STEP 5: EMBEDDING EXTRACTION")
    logger.info("=" * 60)

    # Extract embeddings
    extractor = EmbeddingExtractor(
        backbone=config["embeddings"]["backbone"],
        embedding_dim=config["embeddings"]["embedding_dim"],
        device=config["embeddings"]["device"],
        pretrained=config["embeddings"]["pretrained"],
    )

    store = EmbeddingStore(embeddings_dir)

    # Get unique slides
    unique_slides = slide_metadata["slide_id"].unique()
    logger.info(f"Extracting embeddings for {len(unique_slides)} slides...")

    for slide_id in tqdm(unique_slides, desc="Embedding extraction"):
        try:
            # Get tiles for this slide
            slide_tiles_df = slide_metadata[slide_metadata["slide_id"] == slide_id]

            if len(slide_tiles_df) == 0:
                continue

            # Load tile images
            import cv2

            tile_images = []
            coordinates = []

            for _, row in slide_tiles_df.iterrows():
                tile_path = Path(tiles_dir) / row["tile_filename"]
                if not tile_path.exists():
                    continue

                image = cv2.imread(str(tile_path))
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tile_images.append(image)
                coordinates.append([row["x"], row["y"]])

            if not tile_images:
                logger.warning(f"No valid tiles for {slide_id}")
                continue

            # Extract embeddings
            coordinates = np.array(coordinates)
            embeddings = []

            for i in range(0, len(tile_images), config["embeddings"]["batch_size"]):
                batch = tile_images[
                    i : i + config["embeddings"]["batch_size"]
                ]
                batch_embeddings = extractor.extract_batch(batch)
                embeddings.append(batch_embeddings)

            embeddings = np.vstack(embeddings)

            # Get label from metadata
            label = slide_metadata.loc[
                slide_metadata["slide_id"] == slide_id, "label"
            ].iloc[0]

            # Store
            store.add_slide_embeddings(
                slide_id,
                embeddings,
                coordinates,
                label=label,
                split="train",
            )

        except Exception as e:
            logger.error(f"Error processing slide {slide_id}: {e}")

    # Save metadata
    store.save_metadata(metadata_path)
    logger.info("Embedding extraction complete")


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="WSI preprocessing pipeline"
    )
    parser.add_argument(
        "--config", required=True, help="Path to config YAML"
    )
    parser.add_argument(
        "--slides", nargs="+", required=True, help="Paths to WSI slides"
    )
    parser.add_argument(
        "--slide-ids", nargs="+", help="Slide identifiers (auto-generated if not provided)"
    )
    parser.add_argument(
        "--output-dir", default="data/preprocessed", help="Output directory"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        help="Pipeline steps to run (default: all from config)",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True, help="Resume from last step"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already-processed slides",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Validate config schema
    from src.utils.config_schema import validate_config, validate_no_conflicting_keys
    errors = validate_config(config, "data_pipeline")
    errors.extend(validate_no_conflicting_keys(config))
    if errors:
        for err in errors:
            print(f"Config validation error: {err}")
        raise ValueError(f"Config validation failed with {len(errors)} error(s)")

    # Setup logging
    log_dir = config["storage"].get("logs_dir", "logs")
    log_file = setup_logging(log_dir, config["logging"].get("level", "INFO"))
    logger.info(f"Logging to {log_file}")

    # Parse slide paths
    import glob

    slide_paths = []
    for pattern in args.slides:
        slide_paths.extend(glob.glob(pattern))
    slide_paths = sorted(set(slide_paths))

    if not slide_paths:
        logger.error("No slides found matching patterns")
        return 1

    # Generate slide IDs if not provided
    if args.slide_ids is None:
        slide_ids = [Path(p).stem for p in slide_paths]
    else:
        slide_ids = args.slide_ids

    logger.info(f"Processing {len(slide_paths)} slides")
    logger.info(f"Slide IDs: {slide_ids}")

    # Determine steps to run
    steps = args.steps or config["processing"].get("steps", [])

    # Prepare storage directories
    for key in ["tiles_dir", "normalized_dir", "deduplicated_dir", "embeddings_dir", "features_dir"]:
        Path(config["storage"][key]).mkdir(parents=True, exist_ok=True)

    try:
        # Run pipeline steps
        manifest_df = None

        if "tiling" in steps:
            manifest_df = run_tiling(slide_paths, slide_ids, config, args.skip_existing)

        if manifest_df is not None:
            if "quality_filter" in steps:
                run_quality_filtering(config["storage"]["tiles_dir"], config)

            if "stain_norm" in steps:
                run_stain_normalization(
                    config["storage"]["tiles_dir"],
                    config["storage"]["normalized_dir"],
                    config,
                )

            if "dedup" in steps:
                run_deduplication(
                    config["storage"]["normalized_dir"],
                    config["storage"]["deduplicated_dir"],
                    config,
                )

            if "embeddings" in steps:
                embeddings_metadata_path = (
                    Path(config["storage"]["embeddings_dir"]) / "metadata.parquet"
                )
                # manifest_df is tile-level; embedding extraction needs slide-level
                # metadata with 'slide_id', 'label', and tile info. Ensure the
                # manifest has the required columns, adding defaults if missing.
                embed_metadata = manifest_df.copy()
                if "label" not in embed_metadata.columns:
                    logger.warning(
                        "Tile manifest missing 'label' column — required by embedding "
                        "extraction. Add labels via a slide-level metadata CSV. "
                        "Defaulting to label=0 for all slides."
                    )
                    embed_metadata["label"] = 0
                if "tile_filename" not in embed_metadata.columns:
                    # Infer tile filename from available columns
                    if "filename" in embed_metadata.columns:
                        embed_metadata["tile_filename"] = embed_metadata["filename"]
                    elif "tile_path" in embed_metadata.columns:
                        embed_metadata["tile_filename"] = embed_metadata["tile_path"].apply(
                            lambda p: Path(p).name
                        )

                run_embedding_extraction(
                    config["storage"]["deduplicated_dir"],
                    config["storage"]["embeddings_dir"],
                    str(embeddings_metadata_path),
                    embed_metadata,
                    config,
                )

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import numpy as np

    sys.exit(main())
