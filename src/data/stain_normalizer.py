"""
Stain normalization for histopathology tiles.

Implements Macenko and Reinhard color normalization methods.
Includes quality filtering for background, blur, and pen marks.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image
from scipy.sparse.linalg import eigsh

logger = logging.getLogger(__name__)

try:
    from torchstain import NormalizationTransform
    TORCHSTAIN_AVAILABLE = True
except ImportError:
    TORCHSTAIN_AVAILABLE = False


class Macenko:
    """
    Macenko stain normalization.

    Estimates stain color matrix and normalizes images to a reference.
    Reference: Macenko et al. (2009) "A method for normalizing histology slides
    for quantitative analysis"
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.15):
        """
        Initialize Macenko normalizer.

        Parameters
        ----------
        alpha : float
            Percentile for robust color estimation (default: 1.0)
        beta : float
            Threshold for stain estimation (default: 0.15)
        """
        self.alpha = alpha
        self.beta = beta
        self.HERef = None
        self.maxCRef = None

    def _get_stain_matrix(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract stain color matrix from image using robust OD estimation.

        Returns
        -------
        H : np.ndarray
            Stain matrix (2, 3)
        maxC : np.ndarray
            Maximum concentrations (2,)
        """
        # Convert to OD (optical density)
        # Avoid log(0) by clipping
        image_float = image.astype(np.float32) / 255.0
        image_float = np.clip(image_float, 0.001, 1.0)
        OD = -np.log(image_float)

        # Reshape for PCA
        OD_reshaped = OD.reshape(-1, 3)

        # Estimate two main stain colors using SVD
        U, S, _ = np.linalg.svd(OD_reshaped.T @ OD_reshaped, full_matrices=False)

        # Get first two singular vectors (main stain directions)
        V = U[:, :2]

        # Project OD onto these directions
        projections = OD_reshaped @ V

        # Use percentile to find robust stain directions
        p_alpha = np.percentile(projections, self.alpha, axis=0)
        p_100_alpha = np.percentile(projections, 100 - self.alpha, axis=0)

        # Determine which direction is H vs E
        norms_alpha = np.linalg.norm(p_alpha, axis=None)
        norms_100_alpha = np.linalg.norm(p_100_alpha, axis=None)

        if norms_alpha > norms_100_alpha:
            p_alpha, p_100_alpha = p_100_alpha, p_alpha

        # Stain matrix
        H = np.array([p_100_alpha, p_alpha]).T
        H = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-8)

        # Max concentrations
        maxC = np.percentile(projections, 100 - self.beta, axis=0)

        return H, maxC

    def fit(self, image: np.ndarray) -> "Macenko":
        """
        Fit normalizer to reference image.

        Parameters
        ----------
        image : np.ndarray
            Reference image (H, W, 3)

        Returns
        -------
        self
        """
        HERef, maxCRef = self._get_stain_matrix(image)
        self.HERef = HERef
        self.maxCRef = maxCRef
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to reference stain.

        Parameters
        ----------
        image : np.ndarray
            Image to normalize (H, W, 3)

        Returns
        -------
        np.ndarray
            Normalized image
        """
        # FIXME: Macenko normalizer occasionally fails on sparse tissue regions
        # with < 10% tissue content. Current workaround: skip and log.
        # See also: https://github.com/schaugf/HEnorm_python/issues/3
        if self.HERef is None:
            self.fit(image)

        # Get stain matrix of input
        HE, maxC = self._get_stain_matrix(image)

        # Convert to OD
        image_float = image.astype(np.float32) / 255.0
        image_float = np.clip(image_float, 0.001, 1.0)
        OD = -np.log(image_float)

        # Solve for concentrations
        OD_reshaped = OD.reshape(-1, 3)
        C = np.linalg.lstsq(HE, OD_reshaped.T, rcond=None)[0]

        # Normalize concentrations
        C_normalized = C / (maxC[:, np.newaxis] + 1e-8) * (self.maxCRef[:, np.newaxis] + 1e-8)

        # Reconstruct with reference stain
        OD_normalized = self.HERef @ C_normalized

        # Convert back to RGB
        image_normalized = 255.0 * np.exp(-OD_normalized.T)
        image_normalized = image_normalized.reshape(image.shape)
        image_normalized = np.clip(image_normalized, 0, 255).astype(np.uint8)

        return image_normalized


class Reinhard:
    """
    Reinhard color normalization.

    Simpler baseline method that matches mean and standard deviation
    of color channels.
    Reference: Reinhard et al. (2001) "Color Transfer between Images"
    """

    def __init__(self):
        """Initialize Reinhard normalizer."""
        self.reference_mean = None
        self.reference_std = None

    def fit(self, image: np.ndarray) -> "Reinhard":
        """
        Fit normalizer to reference image.

        Parameters
        ----------
        image : np.ndarray
            Reference image (H, W, 3)

        Returns
        -------
        self
        """
        # Convert to LAB for more perceptually uniform color space
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

        self.reference_mean = np.mean(image_lab, axis=(0, 1))
        self.reference_std = np.std(image_lab, axis=(0, 1))

        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to reference color distribution.

        Parameters
        ----------
        image : np.ndarray
            Image to normalize (H, W, 3)

        Returns
        -------
        np.ndarray
            Normalized image
        """
        if self.reference_mean is None:
            self.fit(image)

        # Convert to LAB
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Get input statistics
        input_mean = np.mean(image_lab, axis=(0, 1))
        input_std = np.std(image_lab, axis=(0, 1))

        # Normalize: (I - mean_src) * (std_ref / std_src) + mean_ref
        image_normalized = (image_lab - input_mean) * (
            self.reference_std / (input_std + 1e-8)
        ) + self.reference_mean

        # Clip to valid LAB range: L=[0,100], A=[-128,127], B=[-128,127]
        image_normalized[:, :, 0] = np.clip(image_normalized[:, :, 0], 0, 100)
        image_normalized[:, :, 1] = np.clip(image_normalized[:, :, 1], -128, 127)
        image_normalized[:, :, 2] = np.clip(image_normalized[:, :, 2], -128, 127)
        image_normalized = image_normalized.astype(np.uint8)
        image_normalized = cv2.cvtColor(image_normalized, cv2.COLOR_LAB2RGB)

        return image_normalized


class StainNormalizer:
    """
    High-level stain normalization with quality filtering.

    Combines stain normalization with background filtering, blur detection,
    and pen mark detection.
    """

    def __init__(
        self,
        method: str = "macenko",
        max_background_fraction: float = 0.8,
        min_laplacian_variance: float = 15.0,
        pen_mark_detection: bool = True,
    ):
        """
        Initialize StainNormalizer.

        Parameters
        ----------
        method : str
            Normalization method ('macenko' or 'reinhard')
        max_background_fraction : float
            Max fraction of background (white) allowed in tile (0-1)
        min_laplacian_variance : float
            Minimum Laplacian variance for sharpness (blur detection)
        pen_mark_detection : bool
            Whether to filter tiles with pen marks
        """
        self.method = method.lower()
        self.max_background_fraction = max_background_fraction
        self.min_laplacian_variance = min_laplacian_variance
        self.pen_mark_detection = pen_mark_detection

        if self.method == "macenko":
            self.normalizer = Macenko()
        elif self.method == "reinhard":
            self.normalizer = Reinhard()
        elif self.method == "identity":
            self.normalizer = None  # No normalization, quality filtering only
        else:
            raise ValueError(f"Unknown method: {method}. Use 'macenko', 'reinhard', or 'identity'.")

        logger.info(
            f"StainNormalizer initialized: method={method}, "
            f"max_background_fraction={max_background_fraction}, "
            f"min_laplacian_variance={min_laplacian_variance}, "
            f"pen_mark_detection={pen_mark_detection}"
        )

    def _is_mostly_background(self, image: np.ndarray) -> bool:
        """Check if tile is mostly empty background."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Background is high intensity (close to white, 255)
        background_mask = gray > 240
        background_frac = np.sum(background_mask) / background_mask.size
        return background_frac > self.max_background_fraction

    def _is_blurry(self, image: np.ndarray) -> bool:
        """Laplacian variance blur check."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.min_laplacian_variance

    def _has_pen_marks(self, image: np.ndarray) -> bool:
        """Detect pen marks via HSV color ranges."""
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]

        # NOTE: pen mark thresholds were calibrated on TCIA CMB-MML slides.
        # Other scanners (Aperio vs Hamamatsu) may shift hue ranges slightly.
        # Blue pen: H ~= 100-130, high saturation
        blue_mask = (h > 100) & (h < 130) & (s > 150)

        # Green pen: H ~= 40-80, high saturation
        green_mask = (h > 40) & (h < 80) & (s > 150)

        # Red pen: H <10 or >170, high saturation
        red_mask = ((h < 10) | (h > 170)) & (s > 150)

        pen_mask = blue_mask | green_mask | red_mask
        pen_frac = np.sum(pen_mask) / pen_mask.size

        return pen_frac > 0.05  # More than 5% pen mark

    def is_high_quality(self, image: np.ndarray) -> bool:
        """
        Check if image passes quality filters.

        Returns True if image is high quality and should be kept.
        """
        # Check background
        if self._is_mostly_background(image):
            return False

        # Check blur
        if self._is_blurry(image):
            return False

        # Check pen marks
        if self.pen_mark_detection and self._has_pen_marks(image):
            return False

        return True

    def fit_to_reference(self, reference_path: str) -> "StainNormalizer":
        """
        Fit normalizer to a reference image.

        Parameters
        ----------
        reference_path : str
            Path to reference image

        Returns
        -------
        self
        """
        if self.normalizer is None:
            logger.info("Identity mode: no reference fitting needed")
            return self
        ref_image = cv2.imread(reference_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        self.normalizer.fit(ref_image)
        logger.info(f"Fitted to reference: {reference_path}")
        return self

    def fit_to_image(self, image: np.ndarray) -> "StainNormalizer":
        """
        Fit normalizer to an image array.

        Parameters
        ----------
        image : np.ndarray
            Reference image (H, W, 3)

        Returns
        -------
        self
        """
        if self.normalizer is not None:
            self.normalizer.fit(image)
        return self

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image stain. Returns image unchanged if method='identity'.

        Parameters
        ----------
        image : np.ndarray
            Image to normalize (H, W, 3)

        Returns
        -------
        np.ndarray
            Normalized image
        """
        if self.normalizer is None:
            return image
        return self.normalizer.transform(image)

    def process_tile_directory(
        self,
        input_dir: str,
        output_dir: str,
        reference_image: Optional[np.ndarray] = None,
        file_extension: str = "*.png",
    ) -> dict:
        """
        Process all tiles in a directory.

        Parameters
        ----------
        input_dir : str
            Input directory with tiles
        output_dir : str
            Output directory for normalized tiles
        reference_image : Optional[np.ndarray]
            Reference image for normalization; if None, use first tile
        file_extension : str
            File pattern to match (default: '*.png')

        Returns
        -------
        dict
            Statistics: {processed, filtered, normalized, errors}
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tile_files = sorted(input_path.glob(file_extension))

        stats = {"processed": 0, "filtered": 0, "normalized": 0, "errors": 0}

        # Fit to reference if provided
        if reference_image is not None:
            if isinstance(reference_image, str):
                self.fit_to_reference(reference_image)
            else:
                self.fit_to_image(reference_image)

        for i, tile_file in enumerate(tile_files):
            try:
                # Load tile
                image = cv2.imread(str(tile_file))
                if image is None:
                    logger.warning(f"Failed to read: {tile_file}")
                    stats["errors"] += 1
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                stats["processed"] += 1

                # Quality check
                if not self.is_high_quality(image):
                    stats["filtered"] += 1
                    logger.debug(f"Filtered (quality): {tile_file.name}")
                    continue

                # Fit to first valid tile if no reference provided
                if self.normalizer.HERef is None and self.normalizer.reference_mean is None:
                    self.fit_to_image(image)

                # Normalize
                normalized = self.normalize(image)
                stats["normalized"] += 1

                # Save
                output_file = output_path / tile_file.name
                Image.fromarray(normalized).save(output_file)

            except (OSError, ValueError, RuntimeError) as e:
                logger.error(f"Error processing {tile_file}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Processing complete: {stats['processed']} processed, "
            f"{stats['normalized']} normalized, {stats['filtered']} filtered, "
            f"{stats['errors']} errors"
        )

        return stats
