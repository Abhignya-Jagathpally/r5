"""
WSI tiling pipeline: Read WSIs, detect tissue, extract tiles with metadata.

Supports both histopathology (SVS) and standard image formats (aspirate smears).
Includes parallel processing, tissue detection via HSV thresholding, and manifest generation.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    import openslide
except ImportError:
    openslide = None

logger = logging.getLogger(__name__)


class WSITiler:
    """
    Extract tiles from Whole Slide Images (WSI) with tissue detection.

    Supports both OpenSlide formats (SVS, etc.) and standard image formats.
    Tiles are saved as PNG with associated metadata manifest.

    Parameters
    ----------
    tile_size : int
        Size of square tiles in pixels (default: 256)
    magnification : int
        Target magnification level (default: 20x)
    overlap : int
        Pixel overlap between adjacent tiles (default: 0)
    min_tissue_fraction : float
        Minimum fraction of tile that must be tissue (0-1, default: 0.5)
    output_dir : str
        Directory to save tiles (default: data/tiles)
    """

    def __init__(
        self,
        tile_size: int = 256,
        magnification: int = 20,
        overlap: int = 0,
        min_tissue_fraction: float = 0.5,
        output_dir: str = "data/tiles",
        tissue_saturation_threshold: int = 15,
    ):
        self.tile_size = tile_size
        self.magnification = magnification
        self.overlap = overlap
        self.min_tissue_fraction = min_tissue_fraction
        self.tissue_saturation_threshold = tissue_saturation_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stride = self.tile_size - self.overlap
        self.tiles_manifest = []

        logger.info(
            f"WSITiler initialized: tile_size={tile_size}, magnification={magnification}, "
            f"overlap={overlap}, min_tissue_fraction={min_tissue_fraction}, "
            f"tissue_saturation_threshold={tissue_saturation_threshold}"
        )

    def _is_svs(self, slide_path: str) -> bool:
        """Check if file is SVS/TIFF (requires openslide)."""
        return slide_path.lower().endswith((".svs", ".tiff", ".tif")) and openslide is not None

    def _get_downsampling_factor(self, slide) -> float:
        """Get downsampling factor for target magnification."""
        try:
            properties = slide.properties
            if "openslide.objective-power" in properties:
                slide_mag = float(properties["openslide.objective-power"])
                return slide_mag / self.magnification
            return 1.0
        except Exception as e:
            logger.warning(f"Could not determine magnification, assuming 1x: {e}")
            return 1.0

    def _detect_tissue(self, tile: np.ndarray) -> float:
        """
        Detect tissue fraction in tile using HSV saturation thresholding.

        Returns fraction of pixels that appear to be tissue (0-1).
        """
        # Convert RGB to HSV
        if tile.shape[-1] == 3:
            hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        else:
            return 0.0

        # Extract saturation channel (S)
        saturation = hsv[:, :, 1]

        # TODO: tissue detection threshold was tuned on H&E stains only,
        # may need adjustment for IHC or special stains (e.g. CD138 for MM)
        # Tissue typically has higher saturation; background is low saturation
        tissue_mask = saturation > self.tissue_saturation_threshold

        tissue_fraction = np.sum(tissue_mask) / tissue_mask.size
        return tissue_fraction

    def _is_blurry(self, tile: np.ndarray, laplacian_threshold: float = 15.0) -> bool:
        """Check if tile is blurry using Laplacian variance."""
        # FIXME: Laplacian threshold 15.0 was picked from HistoQC defaults but
        # aspirate smears have different texture — probably needs per-modality tuning
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < laplacian_threshold

    def process_svs_slide(
        self,
        slide_path: str,
        slide_id: str,
        laplacian_threshold: float = 15.0,
    ) -> List[Dict[str, Any]]:
        """
        Process SVS (or OpenSlide-compatible) slide.

        Parameters
        ----------
        slide_path : str
            Path to SVS file
        slide_id : str
            Unique identifier for slide
        laplacian_threshold : float
            Threshold for blur detection

        Returns
        -------
        List[Dict]
            List of tile metadata dictionaries
        """
        if openslide is None:
            raise ImportError("openslide not available; install openslide-python")

        try:
            slide = openslide.open_slide(slide_path)
        except Exception as e:
            logger.error(f"Failed to open slide {slide_path}: {e}")
            return []

        downsample = self._get_downsampling_factor(slide)
        tile_size_at_ref = int(self.tile_size * downsample)
        stride_at_ref = int(self.stride * downsample)

        width, height = slide.dimensions
        tiles_data = []
        tile_count = 0
        skipped_count = 0

        logger.info(
            f"Processing SVS: {slide_id}, dimensions={width}x{height}, "
            f"downsample={downsample:.2f}"
        )

        for y in range(0, height - tile_size_at_ref, stride_at_ref):
            for x in range(0, width - tile_size_at_ref, stride_at_ref):
                try:
                    # Read tile at reference magnification
                    tile_pil = slide.read_region((x, y), 0, (tile_size_at_ref, tile_size_at_ref))
                    tile_rgb = np.array(tile_pil.convert("RGB"))

                    # Detect tissue
                    tissue_frac = self._detect_tissue(tile_rgb)
                    if tissue_frac < self.min_tissue_fraction:
                        skipped_count += 1
                        continue

                    # Check for blur
                    if self._is_blurry(tile_rgb, laplacian_threshold):
                        skipped_count += 1
                        continue

                    # Resize to target tile size if needed
                    if tile_rgb.shape[:2] != (self.tile_size, self.tile_size):
                        tile_rgb = cv2.resize(
                            tile_rgb, (self.tile_size, self.tile_size), interpolation=cv2.INTER_AREA
                        )

                    # Save tile
                    tile_filename = f"{slide_id}_{x}_{y}.png"
                    tile_path = self.output_dir / tile_filename
                    Image.fromarray(tile_rgb).save(tile_path)

                    tiles_data.append(
                        {
                            "slide_id": slide_id,
                            "tile_filename": tile_filename,
                            "x": x,
                            "y": y,
                            "magnification": self.magnification,
                            "tissue_fraction": tissue_frac,
                        }
                    )
                    tile_count += 1

                except Exception as e:
                    logger.warning(f"Error processing tile at ({x}, {y}): {e}")
                    skipped_count += 1

        logger.info(
            f"Processed {slide_id}: {tile_count} tiles extracted, {skipped_count} skipped"
        )
        slide.close()
        return tiles_data

    def process_standard_image(
        self,
        image_path: str,
        slide_id: str,
        laplacian_threshold: float = 15.0,
    ) -> List[Dict[str, Any]]:
        """
        Process standard image format (JPEG, PNG, etc.) - e.g., aspirate smears.

        Parameters
        ----------
        image_path : str
            Path to image file
        slide_id : str
            Unique identifier
        laplacian_threshold : float
            Threshold for blur detection

        Returns
        -------
        List[Dict]
            List of tile metadata dictionaries
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return []
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            return []

        height, width = image_rgb.shape[:2]
        tiles_data = []
        tile_count = 0
        skipped_count = 0

        logger.info(f"Processing image: {slide_id}, dimensions={width}x{height}")

        for y in range(0, height - self.tile_size, self.stride):
            for x in range(0, width - self.tile_size, self.stride):
                try:
                    tile = image_rgb[y : y + self.tile_size, x : x + self.tile_size]

                    # Ensure correct size
                    if tile.shape[:2] != (self.tile_size, self.tile_size):
                        continue

                    # Detect tissue
                    tissue_frac = self._detect_tissue(tile)
                    if tissue_frac < self.min_tissue_fraction:
                        skipped_count += 1
                        continue

                    # Check for blur
                    if self._is_blurry(tile, laplacian_threshold):
                        skipped_count += 1
                        continue

                    # Save tile
                    tile_filename = f"{slide_id}_{x}_{y}.png"
                    tile_path = self.output_dir / tile_filename
                    Image.fromarray(tile.astype(np.uint8)).save(tile_path)

                    tiles_data.append(
                        {
                            "slide_id": slide_id,
                            "tile_filename": tile_filename,
                            "x": x,
                            "y": y,
                            "magnification": self.magnification,
                            "tissue_fraction": tissue_frac,
                        }
                    )
                    tile_count += 1

                except Exception as e:
                    logger.warning(f"Error processing tile at ({x}, {y}): {e}")
                    skipped_count += 1

        logger.info(
            f"Processed {slide_id}: {tile_count} tiles extracted, {skipped_count} skipped"
        )
        return tiles_data

    def process_slide(
        self, slide_path: str, slide_id: str, laplacian_threshold: float = 15.0
    ) -> List[Dict[str, Any]]:
        """
        Process a single slide (auto-detects format).

        Parameters
        ----------
        slide_path : str
            Path to slide file
        slide_id : str
            Unique identifier for slide
        laplacian_threshold : float
            Threshold for blur detection

        Returns
        -------
        List[Dict]
            List of tile metadata dictionaries
        """
        if self._is_svs(slide_path):
            return self.process_svs_slide(slide_path, slide_id, laplacian_threshold)
        else:
            return self.process_standard_image(slide_path, slide_id, laplacian_threshold)

    def process_slides(
        self,
        slide_paths: List[str],
        slide_ids: Optional[List[str]] = None,
        max_workers: int = 4,
        laplacian_threshold: float = 15.0,
    ) -> pd.DataFrame:
        """
        Process multiple slides in parallel.

        Parameters
        ----------
        slide_paths : List[str]
            List of slide file paths
        slide_ids : Optional[List[str]]
            List of slide identifiers; auto-generated if None
        max_workers : int
            Number of parallel workers (default: 4)
        laplacian_threshold : float
            Threshold for blur detection

        Returns
        -------
        pd.DataFrame
            Manifest of all extracted tiles
        """
        if slide_ids is None:
            slide_ids = [Path(p).stem for p in slide_paths]

        assert len(slide_paths) == len(slide_ids), "slide_paths and slide_ids must have same length"

        all_tiles = []

        # TODO: ThreadPoolExecutor is fine for I/O-bound SVS reads but
        # ProcessPoolExecutor might be better for CPU-bound standard images
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_slide, slide_path, slide_id, laplacian_threshold
                ): slide_id
                for slide_path, slide_id in zip(slide_paths, slide_ids)
            }

            for future in as_completed(futures):
                slide_id = futures[future]
                try:
                    tiles = future.result()
                    all_tiles.extend(tiles)
                except Exception as e:
                    logger.error(f"Failed to process slide {slide_id}: {e}")

        manifest_df = pd.DataFrame(all_tiles)
        logger.info(f"Total tiles extracted: {len(manifest_df)}")

        # Save manifest
        manifest_path = self.output_dir / "tile_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        logger.info(f"Manifest saved to {manifest_path}")

        return manifest_df
