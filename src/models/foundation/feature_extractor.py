"""
Unified feature extraction interface for multiple foundation models.

Provides abstract base class and concrete implementations for:
- ResNet50 (ImageNet pretrained)
- UNI2-h (pathology foundation model)
- TITAN (multimodal WSI foundation model)

Factory pattern for easy model selection.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from .uni2h_encoder import UNI2HEncoder
from .titan_encoder import TITANEncoder

logger = logging.getLogger(__name__)


class FeatureExtractor(ABC, nn.Module):
    """
    Abstract base class for feature extractors.

    All concrete implementations must provide:
    - standardized embedding extraction
    - batch processing capability
    - metadata tracking
    - configuration export
    """

    def __init__(self, embedding_dim: int, device: Optional[str] = None):
        """
        Initialize feature extractor.

        Args:
            embedding_dim: Dimension of extracted embeddings
            device: Device to run model on
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    @abstractmethod
    def extract_batch(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from a batch of images.

        Args:
            input_path: Path to directory or file
            output_path: Optional path to save embeddings
            **kwargs: Model-specific arguments

        Returns:
            Dictionary mapping identifiers to embedding arrays
        """
        pass

    @abstractmethod
    def extract_single(self, image_input) -> np.ndarray:
        """
        Extract embedding for a single image.

        Returns:
            Embedding array of shape (embedding_dim,)
        """
        pass

    @property
    @abstractmethod
    def model_config(self) -> Dict[str, Any]:
        """Return model configuration as dictionary."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier."""
        pass


class ResNet50ImageNet(FeatureExtractor):
    """
    ResNet50 pretrained on ImageNet.

    Standard computer vision backbone for comparison with specialized
    pathology models. Extracts 2048-dimensional features from the
    global average pooling layer.
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        batch_size: int = 32,
        num_workers: int = 4,
        device: Optional[str] = None,
    ):
        """
        Initialize ResNet50 extractor.

        Args:
            embedding_dim: Should be 2048 for standard ResNet50
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
            device: Device to run model on
        """
        super().__init__(embedding_dim, device)

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load pretrained ResNet50
        logger.info("Loading ResNet50 pretrained on ImageNet...")
        model = models.resnet50(pretrained=True)

        # Remove classification head
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def extract_single(self, image_input: Union[str, Path, torch.Tensor]) -> np.ndarray:
        """Extract embedding for a single image."""
        if isinstance(image_input, (str, Path)):
            from PIL import Image

            img = Image.open(image_input).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        else:
            img_tensor = image_input.unsqueeze(0).to(self.device)

        embedding = self.model(img_tensor)
        embedding = embedding.squeeze().cpu().numpy()
        return embedding

    def extract_batch(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Extract embeddings from all images in a directory."""
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image

        input_dir = Path(input_path)
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_dir.glob(f"*{ext}"))
        image_paths = sorted(image_paths)

        if not image_paths:
            logger.warning(f"No images found in {input_dir}")
            return {}

        class SimpleImageDataset(Dataset):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(img), str(self.paths[idx].name)

        dataset = SimpleImageDataset(image_paths, self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        embeddings = {}
        iterator = dataloader
        if verbose:
            from tqdm import tqdm

            iterator = tqdm(dataloader, desc="Extracting ResNet50 features")

        for batch_imgs, batch_names in iterator:
            batch_imgs = batch_imgs.to(self.device)
            batch_emb = self.model(batch_imgs)
            batch_emb = batch_emb.squeeze(-1).squeeze(-1).cpu().numpy()

            for name, emb in zip(batch_names, batch_emb):
                embeddings[name] = emb

        if output_path:
            import joblib

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(embeddings, output_path)
            logger.info(f"Saved embeddings to {output_path}")

        return embeddings

    @property
    def model_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "device": str(self.device),
        }

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "resnet50_imagenet"


class UNI2H(FeatureExtractor):
    """
    UNI2-h pathology foundation model wrapper.

    Delegates to UNI2HEncoder.
    """

    def __init__(
        self,
        embedding_dim: int = 1536,
        batch_size: int = 64,
        num_workers: int = 4,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize UNI2-h extractor."""
        super().__init__(embedding_dim, device)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder = UNI2HEncoder(
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            device=str(self.device),
            hf_token=hf_token,
            cache_dir=cache_dir,
        )

    @torch.no_grad()
    def extract_single(self, image_input: Union[str, Path, torch.Tensor]) -> np.ndarray:
        """Extract embedding for a single image."""
        if isinstance(image_input, (str, Path)):
            img_tensor = self.encoder.preprocess(image_input)
        else:
            img_tensor = image_input

        return self.encoder.extract_single(img_tensor)

    def extract_batch(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Extract embeddings from all tiles in a directory."""
        return self.encoder.extract_batch(
            input_path,
            output_path=output_path,
            extensions=extensions,
            verbose=verbose,
        )

    @property
    def model_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return self.encoder.model_config

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "uni2h"


class TITAN(FeatureExtractor):
    """
    TITAN multimodal WSI foundation model wrapper.

    Delegates to TITANEncoder.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        batch_size: int = 16,
        num_workers: int = 4,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize TITAN extractor."""
        super().__init__(embedding_dim, device)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder = TITANEncoder(
            embedding_dim_slide=embedding_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            device=str(self.device),
            cache_dir=cache_dir,
        )

    @torch.no_grad()
    def extract_single(self, image_input: Union[str, Path, torch.Tensor]) -> np.ndarray:
        """Extract embedding for a single patch."""
        if isinstance(image_input, (str, Path)):
            from PIL import Image

            img = Image.open(image_input).convert("RGB")
            transform = self.encoder._get_slide_transform()
            img_tensor = transform(img).unsqueeze(0)
        else:
            img_tensor = image_input.unsqueeze(0)

        return self.encoder.extract_single_patch(img_tensor)

    def extract_batch(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".svs", ".tiff"),
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Extract embeddings from all patches/slides in a directory."""
        return self.encoder.extract_batch(
            input_path,
            output_path=output_path,
            extensions=extensions,
            verbose=verbose,
        )

    @property
    def model_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return self.encoder.model_config

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "titan"


def get_extractor(
    backbone: str,
    embedding_dim: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[str] = None,
    **kwargs,
) -> FeatureExtractor:
    """
    Factory function to get the appropriate feature extractor.

    Args:
        backbone: Model name ('resnet50', 'uni2h', 'titan')
        embedding_dim: Override default embedding dimension
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        device: Device to run on
        **kwargs: Additional arguments passed to extractor

    Returns:
        FeatureExtractor instance

    Example:
        extractor = get_extractor('uni2h', batch_size=64, hf_token='hf_xxxx')
        embeddings = extractor.extract_batch('tiles/')
    """
    backbone = backbone.lower().strip()

    if backbone in ("uni2h", "titan"):
        logger.info(
            f"Selected {backbone} backbone. Note: this is a gated model requiring "
            "institutional access. Only resnet50 works without additional setup."
        )

    if backbone == "resnet50":
        if embedding_dim is None:
            embedding_dim = 2048
        return ResNet50ImageNet(
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

    elif backbone == "uni2h":
        if embedding_dim is None:
            embedding_dim = 1536
        return UNI2H(
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            **kwargs,
        )

    elif backbone == "titan":
        if embedding_dim is None:
            embedding_dim = 768
        return TITAN(
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown backbone: {backbone}. "
            "Supported: 'resnet50', 'uni2h', 'titan'"
        )
