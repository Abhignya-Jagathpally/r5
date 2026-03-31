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

    def _estimate_stain_vectors(self, od_hat: np.ndarray, percentile: float = 1.0) -> np.ndarray:
        """Estimate stain vectors using angular robust estimation (Macenko 2009).

        Projects OD values onto the SVD plane and finds stain vectors at
        extreme angles, rather than using PCA directions directly.

        Parameters
        ----------
        od_hat : np.ndarray
            OD values above threshold, shape (N, 3)
        percentile : float
            Percentile for robust angle estimation (default: 1.0)

        Returns
        -------
        np.ndarray
            Stain matrix of shape (2, 3), rows are unit stain vectors
        """
        # SVD to find the plane of best fit (top 2 right singular vectors)
        _, _, Vt = np.linalg.svd(od_hat, full_matrices=False)
        plane = Vt[:2, :]  # (2, 3)

        # Project OD pixels onto this 2D plane
        projected = od_hat @ plane.T  # (N, 2)

        # Angular coordinates in the projected plane
        angles = np.arctan2(projected[:, 1], projected[:, 0])

        # Robust extreme angles via percentiles
        min_angle = np.percentile(angles, percentile)
        max_angle = np.percentile(angles, 100 - percentile)

        # Stain vectors from extreme angles, projected back to 3D
        vec1 = np.array([np.cos(min_angle), np.sin(min_angle)]) @ plane
        vec2 = np.array([np.cos(max_angle), np.sin(max_angle)]) @ plane

        # Ensure H&E ordering: hematoxylin absorbs more red (higher OD
        # in the red channel, index 0) than eosin
        if vec1[0] > vec2[0]:
            vec1, vec2 = vec2, vec1

        # Normalize to unit vectors
        vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)

        return np.array([vec1, vec2])

    def _get_stain_matrix(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract stain color matrix from image using Macenko angular estimation.

        Converts to optical density, filters background, then uses SVD
        projection with angular percentile estimation (Macenko et al. 2009)
        to robustly identify stain vectors even in sparse tissue regions.

        Returns
        -------
        H : np.ndarray
            Stain matrix (2, 3) — columns are unit stain vectors
        maxC : np.ndarray
            Maximum concentrations (2,)
        """
        # Convert to OD (optical density): OD = -log10((I + 1) / 240)
        od = -np.log10((image.astype(np.float64) + 1) / 240.0)
        od_reshaped = od.reshape(-1, 3)

        # Remove background pixels (those below OD threshold)
        od_norm = np.linalg.norm(od_reshaped, axis=1)
        od_hat = od_reshaped[od_norm > self.beta]

        if od_hat.shape[0] < 10:
            # Fall back to identity-like stain vectors if almost no tissue
            logger.warning("Too few tissue pixels for stain estimation, using defaults")
            stain_vectors = np.array([[0.6442, 0.7170, 0.2668],
                                      [0.0927, 0.9545, 0.2828]])
            H = stain_vectors.T
            maxC = np.array([1.0, 1.0])
            return H, maxC

        # Angular estimation of stain vectors (Macenko 2009)
        stain_vectors = self._estimate_stain_vectors(od_hat, percentile=self.alpha)

        # Stain matrix: columns are stain vectors, shape (3, 2)
        H = stain_vectors.T

        # Compute concentrations for max estimation
        concentrations = np.linalg.lstsq(H, od_reshaped.T, rcond=None)[0]  # (2, N)
        maxC = np.percentile(concentrations, 99, axis=1)

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
        if self.HERef is None:
            self.fit(image)

        # Get stain matrix of input
        HE, maxC = self._get_stain_matrix(image)

        # Convert to OD using same formula as _get_stain_matrix
        OD = -np.log10((image.astype(np.float64) + 1) / 240.0)
        OD_reshaped = OD.reshape(-1, 3)

        # Solve for concentrations
        C = np.linalg.lstsq(HE, OD_reshaped.T, rcond=None)[0]

        # Normalize concentrations
        C_normalized = C / (maxC[:, np.newaxis] + 1e-8) * (self.maxCRef[:, np.newaxis] + 1e-8)

        # Reconstruct with reference stain
        OD_normalized = self.HERef @ C_normalized

        # Convert back to RGB: inverse of OD = -log10((I+1)/240)
        image_normalized = 240.0 * np.power(10, -OD_normalized.T) - 1
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

            except Exception as e:
                logger.error(f"Error processing {tile_file}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Processing complete: {stats['processed']} processed, "
            f"{stats['normalized']} normalized, {stats['filtered']} filtered, "
            f"{stats['errors']} errors"
        )

        return stats
