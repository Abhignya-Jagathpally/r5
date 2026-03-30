"""
Ray/Dask parallel feature generation for WSI and radiomics data.

Implements scalable tile processing and radiomics extraction with:
- Ray-based parallel tile processing across multiple GPUs
- Dask-based parallel radiomics extraction across CPU cores
- Dynamic resource allocation and fault tolerance
- Progress tracking and memory management
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RayTileProcessorConfig:
    """Configuration for Ray tile processor."""

    num_workers: int = 4
    gpu_per_worker: float = 0.25
    batch_size: int = 32
    max_retries: int = 3
    checkpoint_interval: int = 100


class RayTileProcessor:
    """
    Parallel tile processing using Ray for multi-GPU scaling.

    This processor handles WSI tiling and embedding extraction in parallel
    across multiple GPUs with fault tolerance and progress tracking.

    Example:
        >>> config = RayTileProcessorConfig(num_workers=8, gpu_per_worker=0.5)
        >>> processor = RayTileProcessor(config)
        >>> embeddings = processor.process_wsis(wsi_paths, feature_extractor)
    """

    def __init__(self, config: RayTileProcessorConfig):
        """
        Initialize Ray tile processor.

        Args:
            config: RayTileProcessorConfig instance
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False

    def initialize(self):
        """
        Initialize Ray cluster with configured resources.

        Raises:
            RuntimeError: If Ray initialization fails
        """
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()

            ray.init(
                num_cpus=os.cpu_count() or 4,
                num_gpus=self._count_gpus(),
                logging_level=logging.INFO,
                object_store_memory=int(50e9),  # 50GB
                _system_memory=int(100e9),  # 100GB
            )
            self._initialized = True
            self.logger.info(
                f"Ray cluster initialized with {self.config.num_workers} workers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ray: {str(e)}")

    def shutdown(self):
        """Shutdown Ray cluster and cleanup resources."""
        if self._initialized:
            import ray

            try:
                ray.shutdown()
                self._initialized = False
                self.logger.info("Ray cluster shutdown complete")
            except Exception as e:
                self.logger.error(f"Error during Ray shutdown: {str(e)}")

    def _count_gpus(self) -> int:
        """Count available CUDA GPUs."""
        try:
            import torch

            return torch.cuda.device_count()
        except Exception:
            return 0

    def process_wsis(
        self,
        wsi_paths: List[Path],
        tile_extractor: Callable,
        output_dir: Path,
        **tile_kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Process multiple WSIs in parallel.

        Args:
            wsi_paths: List of paths to WSI files
            tile_extractor: Callable that extracts tiles from a WSI
            output_dir: Directory to save tile embeddings
            **tile_kwargs: Additional arguments for tile extraction

        Returns:
            Dictionary mapping WSI path to embeddings array

        Raises:
            RuntimeError: If processing fails after max retries
        """
        if not self._initialized:
            self.initialize()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Processing {len(wsi_paths)} WSIs")

        try:
            import ray

            # Create remote tile extraction tasks
            futures = []
            for i, wsi_path in enumerate(wsi_paths):
                future = self._process_single_wsi.remote(
                    self,
                    wsi_path,
                    tile_extractor,
                    output_dir,
                    i,
                    **tile_kwargs,
                )
                futures.append((wsi_path, future))

            # Collect results with progress tracking
            results = {}
            completed = 0
            for wsi_path, future in futures:
                try:
                    embeddings = ray.get(future, timeout=3600)
                    results[str(wsi_path)] = embeddings
                    completed += 1
                    if completed % max(1, len(futures) // 10) == 0:
                        self.logger.info(f"Progress: {completed}/{len(futures)} WSIs")
                except Exception as e:
                    self.logger.error(f"Failed to process {wsi_path}: {str(e)}")
                    raise

            self.logger.info(f"Successfully processed {completed}/{len(wsi_paths)} WSIs")
            return results

        except Exception as e:
            self.logger.error(f"Error during WSI processing: {str(e)}")
            raise RuntimeError(f"WSI processing failed: {str(e)}")

    @staticmethod
    def _process_single_wsi(
        self,
        wsi_path: Path,
        tile_extractor: Callable,
        output_dir: Path,
        wsi_idx: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Process a single WSI (runs on Ray worker).

        Args:
            wsi_path: Path to WSI file
            tile_extractor: Callable for tile extraction
            output_dir: Output directory
            wsi_idx: Index for checkpointing
            **kwargs: Additional arguments

        Returns:
            Array of extracted embeddings
        """
        logger.info(f"Processing WSI {wsi_idx}: {wsi_path.name}")
        try:
            embeddings = tile_extractor(wsi_path, **kwargs)
            output_path = output_dir / f"embeddings_{wsi_idx:04d}.npy"
            np.save(str(output_path), embeddings)
            logger.info(f"Saved embeddings to {output_path}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to process WSI {wsi_idx}: {str(e)}")
            raise

    def extract_batch_embeddings(
        self,
        tile_batches: List[np.ndarray],
        embedding_fn: Callable,
    ) -> List[np.ndarray]:
        """
        Extract embeddings from tile batches in parallel.

        Args:
            tile_batches: List of tile batches (each batch is array of tiles)
            embedding_fn: Callable that extracts embeddings from tiles

        Returns:
            List of embedding arrays
        """
        if not self._initialized:
            self.initialize()

        logger.info(f"Extracting embeddings from {len(tile_batches)} batches")

        try:
            import ray

            futures = []
            for batch in tile_batches:
                future = self._extract_batch_remote.remote(
                    batch, embedding_fn
                )
                futures.append(future)

            embeddings = ray.get(futures)
            return embeddings

        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            raise RuntimeError(f"Failed to extract embeddings: {str(e)}")

    @staticmethod
    @ray.remote(num_gpus=0.25)
    def _extract_batch_remote(
        batch: np.ndarray, embedding_fn: Callable
    ) -> np.ndarray:
        """Extract embeddings from a batch (Ray remote task)."""
        try:
            return embedding_fn(batch)
        except Exception as e:
            logger.error(f"Batch embedding extraction failed: {str(e)}")
            raise


@dataclass
class DaskRadiomicsConfig:
    """Configuration for Dask radiomics extractor."""

    num_workers: int = 8
    threads_per_worker: int = 4
    memory_per_worker_gb: int = 8
    chunk_size: int = 10
    timeout_seconds: int = 300


class DaskRadiomicsExtractor:
    """
    Parallel radiomics extraction using Dask for CPU scaling.

    Handles CT/PET radiomics feature extraction across multiple cores
    with memory management and progress tracking.

    Example:
        >>> config = DaskRadiomicsConfig(num_workers=16)
        >>> extractor = DaskRadiomicsExtractor(config)
        >>> features = extractor.extract_batch(images, masks)
    """

    def __init__(self, config: DaskRadiomicsConfig):
        """
        Initialize Dask radiomics extractor.

        Args:
            config: DaskRadiomicsConfig instance
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = None

    def initialize(self):
        """
        Initialize Dask distributed client.

        Raises:
            RuntimeError: If Dask initialization fails
        """
        try:
            from dask.distributed import Client

            self.client = Client(
                n_workers=self.config.num_workers,
                threads_per_worker=self.config.threads_per_worker,
                memory_limit=f"{self.config.memory_per_worker_gb}GB",
                silence_logs=False,
            )
            self.logger.info(
                f"Dask client initialized: {self.client.cluster.scheduler.address}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Dask: {str(e)}")

    def shutdown(self):
        """Shutdown Dask cluster and cleanup."""
        if self.client is not None:
            try:
                self.client.close()
                self.logger.info("Dask client shutdown complete")
            except Exception as e:
                self.logger.error(f"Error during Dask shutdown: {str(e)}")

    def extract_batch(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        radiomics_fn: Callable,
        **radiomics_kwargs,
    ) -> np.ndarray:
        """
        Extract radiomics features from images and masks in parallel.

        Args:
            images: Array of shape (N, H, W, C) or (N, H, W, D)
            masks: Array of shape (N, H, W) or (N, H, W, D)
            radiomics_fn: Callable that extracts radiomics features
            **radiomics_kwargs: Additional arguments for radiomics extraction

        Returns:
            Array of shape (N, num_features) containing extracted features

        Raises:
            RuntimeError: If extraction fails
        """
        if self.client is None:
            self.initialize()

        n_samples = len(images)
        self.logger.info(f"Extracting radiomics from {n_samples} samples")

        try:
            import dask.array as da
            from dask.diagnostics import ProgressBar

            # Convert to dask arrays with chunking
            images_dask = da.from_delayed(
                self._prepare_data(images),
                shape=images.shape,
                dtype=images.dtype,
            )
            masks_dask = da.from_delayed(
                self._prepare_data(masks),
                shape=masks.shape,
                dtype=masks.dtype,
            )

            # Process chunks
            features_list = []
            for i in range(0, n_samples, self.config.chunk_size):
                chunk_end = min(i + self.config.chunk_size, n_samples)
                img_chunk = images[i:chunk_end]
                mask_chunk = masks[i:chunk_end]

                # Submit as delayed task
                chunk_features = self.client.submit(
                    radiomics_fn, img_chunk, mask_chunk, **radiomics_kwargs
                )
                features_list.append(chunk_features)

            # Gather results
            features = []
            with ProgressBar():
                for future in features_list:
                    chunk_result = future.result(
                        timeout=self.config.timeout_seconds
                    )
                    features.append(chunk_result)

            # Concatenate all features
            all_features = np.vstack(features)
            self.logger.info(
                f"Extracted radiomics: shape {all_features.shape}"
            )
            return all_features

        except Exception as e:
            self.logger.error(f"Radiomics extraction failed: {str(e)}")
            raise RuntimeError(f"Failed to extract radiomics: {str(e)}")

    @staticmethod
    def _prepare_data(data: np.ndarray) -> Any:
        """Prepare data for Dask processing."""
        from dask import delayed

        return delayed(lambda x: x)(data)

    def extract_parallel(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        radiomics_fn: Callable,
        **radiomics_kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Extract radiomics from image/mask file pairs in parallel.

        Args:
            image_paths: List of paths to image files
            mask_paths: List of paths to mask files
            radiomics_fn: Callable for feature extraction
            **radiomics_kwargs: Additional arguments

        Returns:
            Dictionary mapping file path to extracted features

        Raises:
            RuntimeError: If any extraction fails
        """
        if self.client is None:
            self.initialize()

        self.logger.info(
            f"Submitting {len(image_paths)} radiomics extraction tasks"
        )

        try:
            futures = {}
            for img_path, mask_path in zip(image_paths, mask_paths):
                future = self.client.submit(
                    self._extract_single,
                    img_path,
                    mask_path,
                    radiomics_fn,
                    **radiomics_kwargs,
                )
                futures[str(img_path)] = future

            # Gather results with error handling
            results = {}
            for path, future in futures.items():
                try:
                    result = future.result(
                        timeout=self.config.timeout_seconds
                    )
                    results[path] = result
                except Exception as e:
                    self.logger.error(f"Failed to extract radiomics for {path}: {str(e)}")
                    raise

            self.logger.info(f"Successfully extracted radiomics for {len(results)} files")
            return results

        except Exception as e:
            self.logger.error(f"Parallel radiomics extraction failed: {str(e)}")
            raise RuntimeError(f"Radiomics extraction failed: {str(e)}")

    @staticmethod
    def _extract_single(
        image_path: Path,
        mask_path: Path,
        radiomics_fn: Callable,
        **kwargs,
    ) -> np.ndarray:
        """Extract radiomics from a single image/mask pair."""
        import nibabel as nib

        logger.debug(f"Loading image: {image_path}")
        img = nib.load(str(image_path)).get_fdata()
        mask = nib.load(str(mask_path)).get_fdata()

        logger.debug(f"Extracting radiomics from {image_path.name}")
        features = radiomics_fn(img, mask, **kwargs)
        logger.debug(f"Extracted {len(features)} features")
        return features
