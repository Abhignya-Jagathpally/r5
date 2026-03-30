"""
Patch embedding extraction and storage.

Extracts embeddings using pretrained backbones (ResNet50, UNI2-h, TITAN)
and stores them in Zarr format with lazy loading support for training.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import zarr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    Store and manage patch embeddings in Zarr format.

    Organizes embeddings hierarchically:
    - One zarr group per slide
    - Datasets per group: embeddings (N×D), coordinates (N×2), attention_scores (N×1)
    - Slide metadata in Parquet file

    Parameters
    ----------
    store_path : str
        Path to zarr store
    """

    def __init__(self, store_path: str = "data/embeddings"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.zarr_root = zarr.open_group(str(self.store_path), mode="a")
        self.metadata = []

        logger.info(f"EmbeddingStore initialized at {self.store_path}")

    def add_slide_embeddings(
        self,
        slide_id: str,
        embeddings: np.ndarray,
        coordinates: np.ndarray,
        attention_scores: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        split: Optional[str] = None,
    ) -> None:
        """
        Store embeddings for a single slide.

        Parameters
        ----------
        slide_id : str
            Unique slide identifier
        embeddings : np.ndarray
            Patch embeddings (N, D)
        coordinates : np.ndarray
            Patch coordinates (N, 2) - [x, y]
        attention_scores : Optional[np.ndarray]
            Attention scores per patch (N, 1); if None, create zeros
        label : Optional[str]
            Slide-level label (e.g., diagnosis)
        split : Optional[str]
            Data split (train/val/test)
        """
        assert embeddings.shape[0] == coordinates.shape[0], "embeddings and coordinates must match"

        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(f"coordinates must be (N, 2), got shape {coordinates.shape}")

        # Create slide group
        slide_group = self.zarr_root.create_group(slide_id, overwrite=True)

        # Store embeddings
        slide_group.create_dataset(
            "embeddings",
            data=embeddings.astype(np.float32),
            chunks=(min(256, embeddings.shape[0]), embeddings.shape[1]),
            compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=2),
        )

        # Store coordinates
        slide_group.create_dataset(
            "coordinates",
            data=coordinates.astype(np.int32),
            chunks=(min(256, coordinates.shape[0]), 2),
            compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=2),
        )

        # Store attention scores (placeholder if not provided)
        if attention_scores is None:
            attention_scores = np.ones((embeddings.shape[0], 1), dtype=np.float32)
        slide_group.create_dataset(
            "attention_scores",
            data=attention_scores.astype(np.float32),
            chunks=(min(256, attention_scores.shape[0]), 1),
            compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=2),
        )

        # Store metadata
        self.metadata.append(
            {
                "slide_id": slide_id,
                "num_patches": embeddings.shape[0],
                "embedding_dim": embeddings.shape[1],
                "label": label,
                "split": split,
            }
        )

        logger.info(f"Stored {embeddings.shape[0]} patches for slide {slide_id}")

    def get_slide_embeddings(self, slide_id: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings for a slide.

        Parameters
        ----------
        slide_id : str
            Slide identifier

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'embeddings', 'coordinates', 'attention_scores'
        """
        if slide_id not in self.zarr_root:
            raise ValueError(f"Slide {slide_id} not found in store")

        slide_group = self.zarr_root[slide_id]

        return {
            "embeddings": np.array(slide_group["embeddings"]),
            "coordinates": np.array(slide_group["coordinates"]),
            "attention_scores": np.array(slide_group["attention_scores"]),
        }

    def list_slides(self) -> List[str]:
        """
        List all slide IDs in store.

        Returns
        -------
        List[str]
            List of slide IDs
        """
        return list(self.zarr_root.group_keys())

    def save_metadata(self, output_path: str = None) -> pd.DataFrame:
        """
        Save slide metadata to Parquet.

        Parameters
        ----------
        output_path : str, optional
            Path to save Parquet; if None, saves to {store_path}/metadata.parquet

        Returns
        -------
        pd.DataFrame
            Metadata DataFrame
        """
        metadata_df = pd.DataFrame(self.metadata)

        if output_path is None:
            output_path = self.store_path / "metadata.parquet"

        metadata_df.to_parquet(output_path, engine="pyarrow")
        logger.info(f"Metadata saved to {output_path}")

        return metadata_df

    def load_metadata(self, parquet_path: str) -> pd.DataFrame:
        """
        Load metadata from Parquet.

        Parameters
        ----------
        parquet_path : str
            Path to metadata parquet

        Returns
        -------
        pd.DataFrame
            Metadata DataFrame
        """
        return pd.read_parquet(parquet_path)


class EmbeddingExtractor:
    """
    Extract embeddings from patches using pretrained backbones.

    Supports ResNet50 (built-in), and placeholders for UNI2-h and TITAN.

    Parameters
    ----------
    backbone : str
        Model backbone ('resnet50', 'uni2h', 'titan')
    embedding_dim : int
        Expected embedding dimension
    device : str
        Computation device ('cuda' or 'cpu')
    pretrained : bool
        Whether to use pretrained weights
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        embedding_dim: int = 2048,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pretrained: bool = True,
    ):
        self.backbone = backbone.lower()
        self.embedding_dim = embedding_dim
        self.device = device
        self.model = self._build_model(pretrained)
        self.transform = self._get_transforms()

        logger.info(
            f"EmbeddingExtractor initialized: backbone={backbone}, "
            f"dim={embedding_dim}, device={device}"
        )

    def _build_model(self, pretrained: bool) -> nn.Module:
        """Build and return model."""
        if self.backbone == "resnet50":
            if pretrained:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                model = models.resnet50(weights=None)

            # Remove classification head
            model = nn.Sequential(*list(model.children())[:-1])

        elif self.backbone == "uni2h":
            # Placeholder for UNI2-h
            # In production, load from huggingface or custom weights
            logger.warning("UNI2-h not yet integrated; using ResNet50 as fallback")
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model = nn.Sequential(*list(model.children())[:-1])

        elif self.backbone == "titan":
            # Placeholder for TITAN
            logger.warning("TITAN not yet integrated; using ResNet50 as fallback")
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model = nn.Sequential(*list(model.children())[:-1])

        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

        model = model.to(self.device)
        model.eval()

        return model

    def _get_transforms(self) -> transforms.Compose:
        """Get preprocessing transforms."""
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for a batch of images.

        Parameters
        ----------
        images : List[np.ndarray]
            List of images (H, W, 3) in RGB format

        Returns
        -------
        np.ndarray
            Embeddings (batch_size, embedding_dim)
        """
        # Convert to tensors and stack
        tensors = [self.transform(Image.fromarray(img.astype(np.uint8))) for img in images]
        batch = torch.stack(tensors).to(self.device)

        # Forward pass
        embeddings = self.model(batch)
        embeddings = embeddings.squeeze(-1).squeeze(-1)  # Remove spatial dims

        return embeddings.cpu().numpy()

    def extract_from_directory(
        self,
        tile_directory: str,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract embeddings for all tiles in directory.

        Parameters
        ----------
        tile_directory : str
            Directory containing tile images
        batch_size : int
            Batch size for extraction
        num_workers : int
            Number of data loading workers

        Returns
        -------
        Tuple[np.ndarray, List[str]]
            Embeddings (N, D) and list of corresponding filenames
        """
        tile_path = Path(tile_directory)
        tile_files = sorted(list(tile_path.glob("*.png")) + list(tile_path.glob("*.jpg")))

        if not tile_files:
            logger.warning(f"No tiles found in {tile_directory}")
            return np.array([]), []

        logger.info(f"Extracting embeddings for {len(tile_files)} tiles")

        dataset = SimpleImageDataset(tile_files, self.transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

        all_embeddings = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                embeddings = self.model(batch)
                embeddings = embeddings.squeeze(-1).squeeze(-1)
                all_embeddings.append(embeddings.cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {(batch_idx + 1) * batch_size}/{len(tile_files)} tiles")

        all_embeddings = np.vstack(all_embeddings)
        filenames = [f.name for f in tile_files]

        logger.info(f"Extracted {all_embeddings.shape[0]} embeddings with shape {all_embeddings.shape}")

        return all_embeddings, filenames


class SimpleImageDataset(Dataset):
    """Simple dataset for loading images from disk."""

    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class PatchDataset(Dataset):
    """
    PyTorch Dataset for lazy loading patches from Zarr store.

    Enables efficient training with minimal memory usage.

    Parameters
    ----------
    zarr_store : str
        Path to zarr store
    metadata_path : str
        Path to metadata parquet
    split : Optional[str]
        Filter by split (train/val/test); None for all
    """

    def __init__(
        self,
        zarr_store: str,
        metadata_path: str,
        split: Optional[str] = None,
        tile_directory: Optional[str] = None,
    ):
        self.zarr_root = zarr.open_group(zarr_store, mode="r")
        self.metadata = pd.read_parquet(metadata_path)
        self.tile_directory = tile_directory

        if split is not None:
            self.metadata = self.metadata[self.metadata["split"] == split]

        self.metadata = self.metadata.reset_index(drop=True)
        self.samples = []

        # Build sample list: (slide_id, patch_idx)
        for _, row in self.metadata.iterrows():
            slide_id = row["slide_id"]
            num_patches = row["num_patches"]
            for patch_idx in range(num_patches):
                self.samples.append((slide_id, patch_idx))

        logger.info(f"PatchDataset initialized with {len(self.samples)} patches")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        slide_id, patch_idx = self.samples[idx]

        # Load from zarr
        slide_group = self.zarr_root[slide_id]
        embedding = np.array(slide_group["embeddings"][patch_idx])
        coordinate = np.array(slide_group["coordinates"][patch_idx])
        attention_score = np.array(slide_group["attention_scores"][patch_idx, 0])

        # Get metadata
        slide_row = self.metadata[self.metadata["slide_id"] == slide_id].iloc[0]
        label = slide_row["label"]

        return {
            "slide_id": slide_id,
            "patch_idx": patch_idx,
            "embedding": torch.tensor(embedding, dtype=torch.float32),
            "coordinate": torch.tensor(coordinate, dtype=torch.int32),
            "attention_score": torch.tensor(attention_score, dtype=torch.float32),
            "label": label,
        }
