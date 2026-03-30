"""
Unit tests for data pipeline components.

Tests tiling, stain normalization, deduplication, and embedding storage
with synthetic images and data.
"""

import logging
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data import (
    WSITiler,
    StainNormalizer,
    Macenko,
    Reinhard,
    TileDeduplicator,
    EmbeddingStore,
    EmbeddingExtractor,
)

logger = logging.getLogger(__name__)


def create_synthetic_tile(
    tile_size: int = 256, tissue_fraction: float = 0.7
) -> np.ndarray:
    """Create a synthetic histopathology-like tile."""
    # Create background (white/light gray)
    tile = np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 240

    # Add tissue (darker regions with color variation)
    tissue_size = int(tile_size * np.sqrt(tissue_fraction))
    y_start = (tile_size - tissue_size) // 2
    x_start = (tile_size - tissue_size) // 2

    # Create tissue-like pattern with H&E colors
    tissue = np.random.rand(tissue_size, tissue_size, 3).astype(np.float32)
    tissue[:, :, 0] *= 50  # R channel
    tissue[:, :, 1] *= 100  # G channel
    tissue[:, :, 2] *= 150  # B channel
    tissue = np.clip(tissue, 0, 255).astype(np.uint8)

    tile[y_start : y_start + tissue_size, x_start : x_start + tissue_size] = tissue

    return tile


def create_synthetic_image(width: int = 1024, height: int = 1024) -> np.ndarray:
    """Create a synthetic medical image."""
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return image


class TestWSITiler:
    """Test WSI tiling functionality."""

    def test_tissue_detection(self):
        """Test tissue detection on synthetic tile."""
        tiler = WSITiler(tile_size=256)

        # Create tile with 70% tissue
        tile = create_synthetic_tile(tissue_fraction=0.7)

        tissue_frac = tiler._detect_tissue(tile)
        assert 0.5 < tissue_frac < 0.95, f"Expected ~0.7 tissue, got {tissue_frac}"

    def test_blur_detection(self):
        """Test blur detection."""
        tiler = WSITiler(tile_size=256)

        # Sharp tile (high Laplacian variance)
        sharp_tile = create_synthetic_tile()
        assert not tiler._is_blurry(sharp_tile), "Sharp tile should not be blurry"

        # Blurry tile (low variance)
        blurry_tile = create_synthetic_tile()
        blurry_tile = np.repeat(blurry_tile, 2, axis=0)
        blurry_tile = np.repeat(blurry_tile, 2, axis=1)
        blurry_tile = blurry_tile[: 256, :256]  # Crop back
        # Note: This isn't very blurry; use Gaussian blur for real test
        import cv2

        blurry_tile = cv2.GaussianBlur(blurry_tile, (15, 15), 0)
        assert tiler._is_blurry(blurry_tile), "Blurry tile should be detected"

    def test_process_standard_image(self):
        """Test processing of standard image format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic image
            image = create_synthetic_image()
            image_path = Path(tmpdir) / "test_image.png"
            Image.fromarray(image).save(image_path)

            # Process
            tiler = WSITiler(
                tile_size=256,
                overlap=0,
                min_tissue_fraction=0.0,  # Low threshold for test
                output_dir=str(Path(tmpdir) / "tiles"),
            )

            tiles_data = tiler.process_standard_image(str(image_path), "test_image")

            # Check results
            assert len(tiles_data) > 0, "Should extract at least one tile"
            assert all(t["slide_id"] == "test_image" for t in tiles_data)
            assert all("tile_filename" in t for t in tiles_data)

            # Check tiles were saved
            tiles_dir = Path(tmpdir) / "tiles"
            assert len(list(tiles_dir.glob("*.png"))) == len(tiles_data)

    def test_manifest_generation(self):
        """Test manifest CSV generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic image
            image = create_synthetic_image()
            image_path = Path(tmpdir) / "test_image.png"
            Image.fromarray(image).save(image_path)

            # Process
            tiler = WSITiler(
                tile_size=256,
                output_dir=str(Path(tmpdir) / "tiles"),
            )

            manifest_df = tiler.process_slides([str(image_path)], ["test_image"])

            # Check manifest
            assert isinstance(manifest_df, pd.DataFrame)
            assert "slide_id" in manifest_df.columns
            assert "tile_filename" in manifest_df.columns
            assert "x" in manifest_df.columns
            assert "y" in manifest_df.columns

            # Check CSV saved
            manifest_path = Path(tmpdir) / "tiles" / "tile_manifest.csv"
            assert manifest_path.exists()


class TestStainNormalization:
    """Test stain normalization functionality."""

    def test_macenko_fit_and_transform(self):
        """Test Macenko normalization."""
        reference = create_synthetic_tile()
        test_tile = create_synthetic_tile()

        normalizer = Macenko()
        normalizer.fit(reference)
        normalized = normalizer.transform(test_tile)

        # Check shape is preserved
        assert normalized.shape == test_tile.shape
        # Check output is uint8
        assert normalized.dtype == np.uint8
        # Check values in valid range
        assert normalized.min() >= 0
        assert normalized.max() <= 255

    def test_reinhard_fit_and_transform(self):
        """Test Reinhard normalization."""
        reference = create_synthetic_tile()
        test_tile = create_synthetic_tile()

        normalizer = Reinhard()
        normalizer.fit(reference)
        normalized = normalizer.transform(test_tile)

        # Check shape is preserved
        assert normalized.shape == test_tile.shape
        # Check output is uint8
        assert normalized.dtype == np.uint8
        # Check values in valid range
        assert normalized.min() >= 0
        assert normalized.max() <= 255

    def test_quality_filtering(self):
        """Test quality filtering."""
        normalizer = StainNormalizer(
            method="macenko",
            max_background_fraction=0.8,
            min_laplacian_variance=15.0,
        )

        # High-quality tile
        good_tile = create_synthetic_tile(tissue_fraction=0.7)
        assert normalizer.is_high_quality(good_tile)

        # Background tile (mostly white)
        bad_tile = np.ones((256, 256, 3), dtype=np.uint8) * 245
        assert not normalizer.is_high_quality(bad_tile)

    def test_batch_processing(self):
        """Test batch processing of tile directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            # Create synthetic tiles
            for i in range(5):
                tile = create_synthetic_tile()
                tile_path = input_dir / f"tile_{i}.png"
                Image.fromarray(tile).save(tile_path)

            # Process directory
            normalizer = StainNormalizer(method="reinhard")
            stats = normalizer.process_tile_directory(
                str(input_dir), str(output_dir)
            )

            # Check stats
            assert stats["processed"] > 0
            assert stats["normalized"] > 0

            # Check output files
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) > 0


class TestTileDeduplicator:
    """Test tile deduplication functionality."""

    def test_hash_computation(self):
        """Test hash computation."""
        tile = create_synthetic_tile()

        dedup = TileDeduplicator(hamming_threshold=8)

        # Hashing should work
        tile_image = Image.fromarray(tile)
        assert dedup._compute_hash(tile_image) is not None

    def test_duplicate_detection(self):
        """Test detection of near-duplicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiles_dir = Path(tmpdir) / "tiles"
            tiles_dir.mkdir()

            # Create original tile
            tile1 = create_synthetic_tile()
            tile1_path = tiles_dir / "tile_0.png"
            Image.fromarray(tile1).save(tile1_path)

            # Create near-duplicate (with slight variation)
            tile2 = tile1.copy()
            tile2[50:55, 50:55] = np.clip(tile2[50:55, 50:55].astype(int) + 10, 0, 255)
            tile2_path = tiles_dir / "tile_1.png"
            Image.fromarray(tile2.astype(np.uint8)).save(tile2_path)

            # Create distinct tile
            tile3 = create_synthetic_tile()
            tile3_path = tiles_dir / "tile_2.png"
            Image.fromarray(tile3).save(tile3_path)

            # Run deduplication
            dedup = TileDeduplicator(hamming_threshold=10)
            dedup.build_index(str(tiles_dir))
            clusters = dedup.find_clusters()

            # Should have at least 2 clusters (1 with duplicates, 1 with distinct)
            assert len(clusters) >= 2

    def test_dedup_statistics(self):
        """Test deduplication statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiles_dir = Path(tmpdir) / "tiles"
            tiles_dir.mkdir()

            # Create identical tiles
            tile = create_synthetic_tile()
            for i in range(3):
                tile_path = tiles_dir / f"tile_{i}.png"
                Image.fromarray(tile).save(tile_path)

            # Run deduplication
            dedup = TileDeduplicator(hamming_threshold=5)
            stats = dedup.deduplicate_directory(str(tiles_dir))

            # Check stats
            assert stats["total_tiles"] == 3
            assert stats["num_clusters"] >= 1
            assert stats["num_kept"] >= 1
            assert stats["num_removed"] == stats["total_tiles"] - stats["num_kept"]


class TestEmbeddingStore:
    """Test embedding store functionality."""

    def test_add_and_retrieve_embeddings(self):
        """Test adding and retrieving embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir)

            # Create synthetic embeddings
            embeddings = np.random.randn(100, 2048).astype(np.float32)
            coordinates = np.random.randint(0, 1000, (100, 2), dtype=np.int32)

            # Add to store
            store.add_slide_embeddings(
                "slide_001",
                embeddings,
                coordinates,
                label="MM",
                split="train",
            )

            # Retrieve
            data = store.get_slide_embeddings("slide_001")

            assert data["embeddings"].shape == embeddings.shape
            assert np.allclose(data["embeddings"], embeddings)
            assert data["coordinates"].shape == coordinates.shape

    def test_metadata_storage(self):
        """Test metadata storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir)

            # Add multiple slides
            for i in range(3):
                embeddings = np.random.randn(50, 2048).astype(np.float32)
                coordinates = np.random.randint(0, 1000, (50, 2), dtype=np.int32)
                store.add_slide_embeddings(
                    f"slide_{i:03d}",
                    embeddings,
                    coordinates,
                    label=f"label_{i}",
                    split="train",
                )

            # Save metadata
            metadata_path = Path(tmpdir) / "metadata.parquet"
            metadata_df = store.save_metadata(str(metadata_path))

            # Check metadata
            assert len(metadata_df) == 3
            assert all(c in metadata_df.columns for c in ["slide_id", "num_patches", "label"])

    def test_list_slides(self):
        """Test listing slides in store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir)

            # Add slides
            for i in range(3):
                embeddings = np.random.randn(10, 2048).astype(np.float32)
                coordinates = np.random.randint(0, 1000, (10, 2), dtype=np.int32)
                store.add_slide_embeddings(
                    f"slide_{i:03d}",
                    embeddings,
                    coordinates,
                )

            # List slides
            slides = store.list_slides()
            assert len(slides) == 3
            assert all(s.startswith("slide_") for s in slides)


class TestEmbeddingExtractor:
    """Test embedding extraction."""

    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = EmbeddingExtractor(backbone="resnet50", device="cpu")

        assert extractor.backbone == "resnet50"
        assert extractor.embedding_dim == 2048
        assert extractor.model is not None

    def test_extract_batch(self):
        """Test batch embedding extraction."""
        extractor = EmbeddingExtractor(backbone="resnet50", device="cpu")

        # Create synthetic images
        images = [create_synthetic_image(224, 224) for _ in range(4)]

        embeddings = extractor.extract_batch(images)

        assert embeddings.shape == (4, 2048)
        assert embeddings.dtype == np.float32

    def test_extract_from_directory(self):
        """Test extraction from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiles_dir = Path(tmpdir) / "tiles"
            tiles_dir.mkdir()

            # Create synthetic tiles
            for i in range(3):
                tile = create_synthetic_image()
                tile_path = tiles_dir / f"tile_{i}.png"
                Image.fromarray(tile).save(tile_path)

            # Extract embeddings
            extractor = EmbeddingExtractor(backbone="resnet50", device="cpu")
            embeddings, filenames = extractor.extract_from_directory(
                str(tiles_dir), batch_size=2
            )

            assert embeddings.shape == (3, 2048)
            assert len(filenames) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
