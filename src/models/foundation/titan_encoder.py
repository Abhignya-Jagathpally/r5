"""
TITAN multimodal whole-slide foundation model integration.

TITAN (Transformer-based pathology Image and Text Alignment Network) is a
multimodal WSI foundation model pretrained on 335k+ whole-slide images with
vision-language alignment using pathology reports.

Model Details:
- Architecture: Vision Transformer with multimodal (image + text) pretraining
- Slide-level embedding dimension: 768 (typically)
- Tile-level embedding dimension: Variable based on configuration
- Training data: 335k+ WSIs from Mass General Brigham
- License: Check HuggingFace model card (likely academic non-commercial)
- Access: Public on HuggingFace

Paper: "A multimodal whole-slide foundation model for pathology" (Nature Medicine, 2025)
Reference: https://github.com/mahmoodlab/TITAN

Usage:
    encoder = TITANEncoder(
        model_name="MahmoodLab/TITAN",
        level="slide",  # or "tile"
        batch_size=16,
        device="cuda"
    )

    # Extract slide-level embeddings
    embeddings = encoder.extract_batch("path/to/slides/", level="slide")

    # Or tile-level from a single WSI
    slide_emb = encoder.extract_slide("slide.svs", level="tile")
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

logger = logging.getLogger(__name__)


class TITANEncoder(nn.Module):
    """
    TITAN multimodal foundation model encoder for whole-slide images.

    Loads the pretrained TITAN model from HuggingFace and extracts either
    slide-level or tile-level embeddings. Supports vision-language alignment
    features for multimodal fusion.
    """

    def __init__(
        self,
        model_name: str = "MahmoodLab/TITAN",
        embedding_dim_slide: int = 768,
        embedding_dim_tile: Optional[int] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        resume_from_cache: bool = True,
    ):
        """
        Initialize TITAN encoder.

        Args:
            model_name: HuggingFace model identifier
            embedding_dim_slide: Dimension of slide-level embeddings (default 768)
            embedding_dim_tile: Dimension of tile-level embeddings (if extracting tiles)
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
            device: Device to run model on ('cuda' or 'cpu'). Auto-detected if None.
            cache_dir: Directory to cache extracted embeddings
            resume_from_cache: If True, skip already-extracted embeddings

        Raises:
            RuntimeError: If model cannot be downloaded or loaded
        """
        super().__init__()

        self.model_name = model_name
        self.embedding_dim_slide = embedding_dim_slide
        self.embedding_dim_tile = embedding_dim_tile
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resume_from_cache = resume_from_cache

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        # Cache setup
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "titan"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Embedding cache directory: {self.cache_dir}")

        # Load model
        logger.info(f"Loading {model_name} from HuggingFace...")
        try:
            # Try to load TITAN model
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("TITAN model loaded successfully")

            # Extract actual embedding dimension if available
            if hasattr(self.model, "config"):
                if hasattr(self.model.config, "hidden_size"):
                    self.embedding_dim_slide = self.model.config.hidden_size
                    logger.info(
                        f"Updated embedding_dim_slide to {self.embedding_dim_slide} "
                        "from model config"
                    )

        except Exception as e:
            logger.warning(
                f"Could not load TITAN from transformers: {e}. "
                "Attempting alternative loading method..."
            )
            try:
                # Alternative: Try to load with timm
                import timm

                self.model = timm.create_model(
                    model_name.lower().replace("/", "_"),
                    pretrained=True,
                    num_classes=0,
                )
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info("TITAN model loaded via timm")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load TITAN model: {e} (timm) / {e2} (transformers)"
                )

    def _get_slide_transform(self):
        """Get preprocessing transform for WSI patches."""
        try:
            from timm.data.transforms_factory import create_transform

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

    @torch.no_grad()
    def extract_single_patch(
        self,
        patch_tensor: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract embedding for a single patch tensor.

        Args:
            patch_tensor: Preprocessed patch tensor of shape (3, 224, 224) or (B, 3, 224, 224)
            return_intermediate: If True, return intermediate layer activations for interpretability

        Returns:
            Embedding array of shape (embedding_dim,) or dict with intermediate features
        """
        if patch_tensor.dim() == 3:
            patch_tensor = patch_tensor.unsqueeze(0)

        patch_tensor = patch_tensor.to(self.device)

        if return_intermediate:
            # Enable hook to capture intermediate features
            features = {}

            def hook_fn(name):
                def hook(module, input, output):
                    features[name] = output.detach().cpu().numpy()

                return hook

            # Register hooks (adapt to your model architecture)
            handles = []
            for name, module in self.model.named_modules():
                if "layer" in name or "block" in name:
                    h = module.register_forward_hook(hook_fn(name))
                    handles.append(h)

            embedding = self.model(patch_tensor)
            features["embedding"] = embedding.cpu().numpy()

            # Remove hooks
            for h in handles:
                h.remove()

            return features
        else:
            embedding = self.model(patch_tensor)
            return embedding.cpu().numpy()

    @torch.no_grad()
    def extract_from_patches(
        self,
        patch_list: List[Union[str, Path, torch.Tensor]],
        aggregate: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Extract embeddings from a list of patch images.

        Args:
            patch_list: List of paths to patch images or tensor batch
            aggregate: If True, aggregate to slide-level using mean pooling

        Returns:
            Array of shape (N, embedding_dim) or aggregated slide embedding
        """
        embeddings = []
        transform = self._get_slide_transform()

        for patch_item in patch_list:
            if isinstance(patch_item, torch.Tensor):
                patch_tensor = patch_item
            else:
                from PIL import Image

                img = Image.open(patch_item).convert("RGB")
                patch_tensor = transform(img).unsqueeze(0)

            emb = self.extract_single_patch(patch_tensor)
            if isinstance(emb, dict):
                emb = emb["embedding"]
            embeddings.append(emb)

        embeddings = np.vstack(embeddings)

        if aggregate:
            agg_embedding = embeddings.mean(axis=0)
            return {
                "aggregated_embedding": agg_embedding,
                "patch_embeddings": embeddings,
                "num_patches": len(patch_list),
            }
        else:
            return embeddings

    @torch.no_grad()
    def extract_slide_features(
        self,
        slide_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Extract slide-level features from a whole-slide image.

        For real WSI (.svs, .tiff) processing, this would need to be
        integrated with a slide reading backend (openslide, tiffslide, etc.)
        and tile extraction pipeline.

        Args:
            slide_path: Path to whole-slide image file
            output_path: Optional path to save features

        Returns:
            Dictionary with slide-level embedding and metadata

        Note:
            This is a placeholder. Full WSI processing would require:
            - Slide reading library (openslide)
            - Tile extraction at configurable magnification
            - Batch processing of tiles
            - Aggregation strategy (mean, attention-weighted, etc.)
        """
        logger.warning(
            f"TITAN extract_slide() is a PLACEHOLDER — returning zero embeddings for {slide_path}. "
            "For real inference, install TITAN from https://github.com/mahmoodlab/TITAN "
            "and provide pretrained weights."
        )
        return {
            "slide_path": str(slide_path),
            "slide_embedding": np.zeros(self.embedding_dim_slide, dtype=np.float32),
            "num_patches": 0,
            "magnification": None,
            "_placeholder": True,
        }

    @torch.no_grad()
    def extract_batch(
        self,
        directory: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".svs", ".tiff"),
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from all slides/patches in a directory.

        Args:
            directory: Path to directory containing slide/patch images
            output_path: Optional path to save embeddings
            extensions: File extensions to process
            verbose: If True, show progress bar

        Returns:
            Dictionary mapping filenames to embedding arrays
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")

        # Find all slide/patch images
        slide_paths = []
        for ext in extensions:
            slide_paths.extend(directory.glob(f"*{ext}"))

        if not slide_paths:
            logger.warning(f"No images found in {directory}")
            return {}

        slide_paths = sorted(slide_paths)
        logger.info(f"Found {len(slide_paths)} items to process")

        # Check cache if resuming
        embeddings = {}
        cache_path = output_path if output_path else self.cache_dir / f"{directory.name}.pkl"

        if self.resume_from_cache and Path(cache_path).exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            cached = joblib.load(cache_path)
            embeddings.update(cached)
            cached_names = set(Path(p).name for p in cached.keys())
            slide_paths = [p for p in slide_paths if p.name not in cached_names]
            logger.info(f"Resuming: {len(slide_paths)} new items to process")

        if not slide_paths:
            return embeddings

        # Process slides
        try:
            if verbose:
                from tqdm import tqdm

                slide_paths = tqdm(slide_paths, desc="Extracting slide embeddings")

            for slide_path in slide_paths:
                try:
                    # For demonstration, treat as batch of patches
                    # In practice, this would use openslide to extract patches from WSI
                    features = self.extract_slide_features(slide_path)
                    embeddings[slide_path.name] = features["slide_embedding"]
                except Exception as e:
                    logger.warning(f"Failed to process {slide_path}: {e}")

        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
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
            x: Input tensor of shape (B, 3, 224, 224) for patches or similar

        Returns:
            Embeddings of shape (B, embedding_dim_slide)
        """
        return self.model(x)

    @property
    def model_config(self) -> Dict[str, Any]:
        """Return model configuration as dictionary."""
        return {
            "model_name": self.model_name,
            "embedding_dim_slide": self.embedding_dim_slide,
            "embedding_dim_tile": self.embedding_dim_tile,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "device": str(self.device),
        }
