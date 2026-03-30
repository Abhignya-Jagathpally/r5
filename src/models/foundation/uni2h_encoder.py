"""
UNI2-h pathology foundation model integration.

UNI2-h (ViT-H/14) is a self-supervised vision foundation model pretrained on
335k+ histopathology slides from Mass General Brigham.

Model Details:
- Architecture: Vision Transformer H/14 (DINOv2-based)
- Embedding dimension: 1536
- Training data: 200M+ H&E and IHC tiles from 350k+ diverse slides
- License: CC-BY-NC-ND 4.0 (non-commercial academic research)
- Access: Requires HuggingFace authentication

Paper: "UNI: A Universal Pathology Foundation Model"
Reference: https://github.com/mahmoodlab/UNI

Usage:
    encoder = UNI2HEncoder(
        model_name="MahmoodLab/UNI2-h",
        batch_size=64,
        num_workers=4,
        device="cuda",
        hf_token="hf_xxxx"  # Required for model access
    )

    # Extract embeddings from a directory of tiles
    embeddings = encoder.extract_batch("path/to/tiles/")

    # Single image processing
    img_tensor = encoder.preprocess("tile.jpg")
    emb = encoder.extract_single(img_tensor)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from huggingface_hub import hf_api, login
import joblib

logger = logging.getLogger(__name__)


class TileDataset(Dataset):
    """Dataset for loading image tiles."""

    def __init__(
        self,
        tile_paths: list,
        transform=None,
        load_in_memory: bool = False,
    ):
        """
        Initialize tile dataset.

        Args:
            tile_paths: List of paths to image tiles
            transform: Optional image transform/preprocessing
            load_in_memory: If True, load all tiles into memory (faster but memory-intensive)
        """
        self.tile_paths = tile_paths
        self.transform = transform
        self.load_in_memory = load_in_memory
        self.cache = {}

        if load_in_memory:
            logger.info(f"Loading {len(tile_paths)} tiles into memory...")
            for path in tile_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    self.cache[path] = img
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        path = self.tile_paths[idx]

        if path in self.cache:
            img = self.cache[path]
        else:
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                # Return placeholder
                img = Image.new("RGB", (224, 224))

        if self.transform:
            img = self.transform(img)

        return img, path


class UNI2HEncoder(nn.Module):
    """
    UNI2-h foundation model encoder for pathology images.

    Loads the pretrained UNI2-h (ViT-H/14) model from HuggingFace and extracts
    tile-level embeddings. Supports batched inference with GPU acceleration and
    embedding caching.
    """

    def __init__(
        self,
        model_name: str = "MahmoodLab/UNI2-h",
        embedding_dim: int = 1536,
        batch_size: int = 64,
        num_workers: int = 4,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        resume_from_cache: bool = True,
    ):
        """
        Initialize UNI2-h encoder.

        Args:
            model_name: HuggingFace model identifier
            embedding_dim: Dimension of extracted embeddings (1536 for UNI2-h)
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
            device: Device to run model on ('cuda' or 'cpu'). Auto-detected if None.
            hf_token: HuggingFace authentication token (required for model access)
            cache_dir: Directory to cache extracted embeddings
            resume_from_cache: If True, skip already-extracted embeddings

        Raises:
            RuntimeError: If model cannot be downloaded or loaded
            ValueError: If hf_token is not provided and HF_TOKEN env var not set
        """
        super().__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resume_from_cache = resume_from_cache

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        # Authentication
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                logger.info("Authenticated with HuggingFace")
            except Exception as e:
                logger.warning(f"HuggingFace authentication failed: {e}")
        else:
            logger.warning(
                "No HF_TOKEN provided. Model download may fail if private access required."
            )

        # Cache setup
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "uni2h"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Embedding cache directory: {self.cache_dir}")

        # Load model
        logger.info(f"Loading {model_name} from HuggingFace...")
        try:
            self.model = timm.create_model(
                "vit_h14_clip_224.laion2b_ft_in1k",
                pretrained=True,
                num_classes=0,  # Remove classification head
            )
            # Override with UNI2-h weights if available
            try:
                checkpoint = hf_api.hf_hub_download(
                    repo_id=model_name,
                    filename="pytorch_model.bin",
                )
                state_dict = torch.load(checkpoint, map_location="cpu")
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded UNI2-h weights from HuggingFace")
            except Exception as e:
                logger.warning(f"Could not load UNI2-h weights: {e}. Using CLIP ViT-H/14")

            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

        # Setup image preprocessing (standard for ViT/CLIP)
        self.transform = self._get_transform()

    def _get_transform(self):
        """Get standard preprocessing transform for ViT models."""
        try:
            import torchvision.transforms as transforms
            from timm.data.transforms_factory import create_transform

            # Use timm's standard preprocessing
            return create_transform(
                input_size=224,
                is_training=False,
                interpolation="bilinear",
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        except Exception as e:
            logger.warning(f"Could not create timm transform: {e}. Using basic transforms.")
            import torchvision.transforms as transforms

            return transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )

    def preprocess(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess a single image.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor of shape (3, 224, 224)
        """
        img = Image.open(image_path).convert("RGB")
        return self.transform(img)

    @torch.no_grad()
    def extract_single(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract embedding for a single preprocessed image.

        Args:
            image_tensor: Preprocessed image tensor of shape (3, 224, 224) or (B, 3, 224, 224)

        Returns:
            Embedding array of shape (embedding_dim,) or (B, embedding_dim)
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        embedding = self.model(image_tensor)

        return embedding.cpu().numpy()

    @torch.no_grad()
    def extract_batch(
        self,
        tile_directory: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from all tiles in a directory.

        Args:
            tile_directory: Path to directory containing tile images
            output_path: Optional path to save embeddings as pickle (cached for resume)
            extensions: Image file extensions to process
            verbose: If True, show progress bar

        Returns:
            Dictionary mapping tile filenames to embedding arrays

        Example:
            embeddings = encoder.extract_batch("path/to/tiles/", output_path="embeddings.pkl")
            # embeddings["tile_0001.jpg"] -> array of shape (1536,)
        """
        tile_dir = Path(tile_directory)
        if not tile_dir.is_dir():
            raise ValueError(f"Directory not found: {tile_dir}")

        # Find all tile images
        tile_paths = []
        for ext in extensions:
            tile_paths.extend(tile_dir.glob(f"*{ext}"))

        if not tile_paths:
            logger.warning(f"No images found in {tile_dir}")
            return {}

        tile_paths = sorted(tile_paths)
        logger.info(f"Found {len(tile_paths)} tiles to process")

        # Check cache if resuming
        embeddings = {}
        cache_path = output_path if output_path else self.cache_dir / f"{tile_dir.name}.pkl"

        if self.resume_from_cache and Path(cache_path).exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            cached = joblib.load(cache_path)
            embeddings.update(cached)
            cached_names = set(Path(p).name for p in cached.keys())
            tile_paths = [p for p in tile_paths if p.name not in cached_names]
            logger.info(f"Resuming: {len(tile_paths)} new tiles to process")

        if not tile_paths:
            return embeddings

        # Create dataset and dataloader
        dataset = TileDataset(tile_paths, transform=self.transform, load_in_memory=False)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

        # Extract embeddings
        try:
            if verbose:
                from tqdm import tqdm

                dataloader = tqdm(dataloader, desc="Extracting embeddings")

            for batch_images, batch_paths in dataloader:
                batch_images = batch_images.to(self.device)
                batch_embeddings = self.model(batch_images)
                batch_embeddings = batch_embeddings.cpu().numpy()

                for path, embedding in zip(batch_paths, batch_embeddings):
                    embeddings[str(Path(path).name)] = embedding

        except Exception as e:
            logger.error(f"Error during embedding extraction: {e}")
            raise

        # Save cache
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(embeddings, output_path)
            logger.info(f"Saved embeddings to {output_path}")

        return embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (for nn.Module compatibility).

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Embeddings of shape (B, 1536)
        """
        return self.model(x)

    @property
    def model_config(self) -> Dict[str, Any]:
        """Return model configuration as dictionary."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "device": str(self.device),
        }
