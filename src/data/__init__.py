"""
Data processing module for WSI tiling, stain normalization, deduplication, and embedding storage.

This module provides a comprehensive pipeline for:
- WSI tiling with tissue detection
- Stain normalization (Macenko, Reinhard)
- Tile deduplication using perceptual hashing
- Patch embedding extraction and storage
- Radiomics feature extraction
"""

from src.data.wsi_tiler import WSITiler
from src.data.stain_normalizer import StainNormalizer, Macenko, Reinhard
from src.data.deduplicator import TileDeduplicator
from src.data.embedding_store import EmbeddingStore, EmbeddingExtractor, PatchDataset
from src.data.radiomics_extractor import RadiomicsExtractor

__all__ = [
    "WSITiler",
    "StainNormalizer",
    "Macenko",
    "Reinhard",
    "TileDeduplicator",
    "EmbeddingStore",
    "EmbeddingExtractor",
    "PatchDataset",
    "RadiomicsExtractor",
]

__version__ = "0.1.0"
