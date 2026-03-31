"""
Radiomics feature extraction from CT/PET images.

Uses pyradiomics to extract first-order, shape, texture, and other features.
Supports DICOM reading and ROI-based extraction.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    from radiomics import featureextractor, getTestCase
except ImportError:
    featureextractor = None

logger = logging.getLogger(__name__)


class RadiomicsExtractor:
    """
    Extract radiomics features from medical images.

    Supports DICOM input and ROI-based extraction using pyradiomics.

    Parameters
    ----------
    config_path : Optional[str]
        Path to radiomics config YAML; if None, uses defaults
    bin_width : int
        Bin width for discretization (default: 25)
    resample_spacing : Tuple[float, float, float]
        Resampling spacing (default: [1.0, 1.0, 1.0])
    feature_classes : List[str]
        Classes of features to extract (default: all)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        bin_width: int = 25,
        resample_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        feature_classes: Optional[List[str]] = None,
    ):
        if featureextractor is None:
            raise ImportError("radiomics not available; install pyradiomics")

        self.bin_width = bin_width
        self.resample_spacing = resample_spacing

        if feature_classes is None:
            feature_classes = [
                "firstorder",
                "shape",
                "glcm",
                "glrlm",
                "glszm",
                "gldm",
            ]
        self.feature_classes = feature_classes

        # Build settings dictionary
        self.settings = {
            "binWidth": bin_width,
            "resamplingVoxelSize": list(resample_spacing),
            "normalize": True,
            "normalizeScale": 1.0,
            "removeOutliers": True,
            "padDistance": 10,
            "distances": [1],
            "angles": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            "force2D": False,
            "force2Ddimension": 0,
        }

        # Load custom config if provided
        if config_path is not None:
            with open(config_path, "r") as f:
                custom_settings = yaml.safe_load(f)
                self.settings.update(custom_settings)

        # Initialize extractor
        try:
            self.extractor = featureextractor.RadiomicsFeatureExtractor(**self.settings)

            # Enable only specified classes
            self.extractor.disableAllFeatures()
            for feature_class in self.feature_classes:
                feature_class_lower = feature_class.lower()
                if feature_class_lower == "firstorder":
                    self.extractor.enableFeatureClassByName("firstorder")
                elif feature_class_lower == "shape":
                    self.extractor.enableFeatureClassByName("shape")
                elif feature_class_lower == "glcm":
                    self.extractor.enableFeatureClassByName("glcm")
                elif feature_class_lower == "glrlm":
                    self.extractor.enableFeatureClassByName("glrlm")
                elif feature_class_lower == "glszm":
                    self.extractor.enableFeatureClassByName("glszm")
                elif feature_class_lower == "gldm":
                    self.extractor.enableFeatureClassByName("gldm")

        except (ValueError, RuntimeError, KeyError) as e:
            logger.error(f"Failed to initialize feature extractor: {e}")
            raise

        logger.info(
            f"RadiomicsExtractor initialized: classes={feature_classes}, "
            f"bin_width={bin_width}, spacing={resample_spacing}"
        )

    def extract_features(
        self, image_path: str, mask_path: str, sample_id: str = None
    ) -> Dict[str, float]:
        """
        Extract radiomics features from an image given a ROI mask.

        Parameters
        ----------
        image_path : str
            Path to image file (DICOM)
        mask_path : str
            Path to segmentation mask (DICOM or numpy)
        sample_id : str, optional
            Sample identifier for logging

        Returns
        -------
        Dict[str, float]
            Feature name -> value mapping
        """
        try:
            features = self.extractor.execute(image_path, mask_path)

            # Convert to standard dict and remove diagnostic info
            features_dict = {}
            for key, value in features.items():
                if isinstance(value, (int, float, np.number)):
                    features_dict[key] = float(value)

            if sample_id:
                logger.info(f"Extracted {len(features_dict)} features for {sample_id}")

            return features_dict

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return {}

    def extract_batch(
        self,
        image_mask_pairs: List[Tuple[str, str]],
        sample_ids: List[str] = None,
    ) -> pd.DataFrame:
        """
        Extract features for multiple image-mask pairs.

        Parameters
        ----------
        image_mask_pairs : List[Tuple[str, str]]
            List of (image_path, mask_path) tuples
        sample_ids : List[str], optional
            Sample identifiers

        Returns
        -------
        pd.DataFrame
            Features for all samples
        """
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(image_mask_pairs))]

        all_features = []

        logger.info(f"Extracting features for {len(image_mask_pairs)} samples")

        for i, (image_path, mask_path) in enumerate(image_mask_pairs):
            sample_id = sample_ids[i]

            features = self.extract_features(image_path, mask_path, sample_id)

            if features:
                features["sample_id"] = sample_id
                all_features.append(features)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_mask_pairs)} samples")

        features_df = pd.DataFrame(all_features)
        logger.info(f"Extracted features for {len(features_df)} samples")

        return features_df

    def extract_from_directory(
        self,
        image_dir: str,
        mask_dir: str,
        output_path: str = None,
        image_suffix: str = ".dcm",
        mask_suffix: str = "_mask.dcm",
    ) -> pd.DataFrame:
        """
        Extract features for all images in a directory.

        Parameters
        ----------
        image_dir : str
            Directory containing image files
        mask_dir : str
            Directory containing mask files
        output_path : str, optional
            Path to save features parquet
        image_suffix : str
            Image file suffix (default: '.dcm')
        mask_suffix : str
            Mask file suffix (default: '_mask.dcm')

        Returns
        -------
        pd.DataFrame
            Features DataFrame
        """
        image_path_obj = Path(image_dir)
        mask_path_obj = Path(mask_dir)

        image_files = sorted(image_path_obj.glob(f"*{image_suffix}"))

        image_mask_pairs = []
        sample_ids = []

        for image_file in image_files:
            # Construct corresponding mask filename
            base_name = image_file.stem
            mask_file = mask_path_obj / f"{base_name}{mask_suffix}"

            if mask_file.exists():
                image_mask_pairs.append((str(image_file), str(mask_file)))
                sample_ids.append(base_name)
            else:
                logger.warning(f"No mask found for {image_file}")

        if not image_mask_pairs:
            logger.warning(f"No image-mask pairs found in {image_dir}")
            return pd.DataFrame()

        # Extract features
        features_df = self.extract_batch(image_mask_pairs, sample_ids)

        # Save if requested
        if output_path is not None:
            features_df.to_parquet(output_path, engine="pyarrow")
            logger.info(f"Features saved to {output_path}")

        return features_df

    @staticmethod
    def load_config(config_path: str) -> Dict:
        """
        Load radiomics configuration from YAML.

        Parameters
        ----------
        config_path : str
            Path to YAML config file

        Returns
        -------
        Dict
            Configuration dictionary
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def save_config(self, output_path: str):
        """
        Save radiomics configuration to YAML.

        Parameters
        ----------
        output_path : str
            Path to save YAML
        """
        config = {
            "settings": self.settings,
            "feature_classes": self.feature_classes,
        }
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Config saved to {output_path}")
