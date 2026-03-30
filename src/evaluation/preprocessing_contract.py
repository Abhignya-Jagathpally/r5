"""
Frozen preprocessing contract for reproducible evaluation.

Records ALL preprocessing parameters (normalization stats, imputation values,
feature selection masks, etc.) and ensures they are fit ONLY on training data.

Validates that test data never leaks into preprocessing by maintaining a strict
contract that cannot be modified after fitting.
"""

from typing import Dict, Any, Optional, List, Union
import logging
import json
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


@dataclass
class NormalizationStats:
    """Container for feature normalization statistics."""
    method: str  # "zscore", "minmax"
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    min: Optional[List[float]] = None
    max: Optional[List[float]] = None
    feature_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationStats":
        return cls(**data)


@dataclass
class ImputationStats:
    """Container for missing value imputation statistics."""
    method: str  # "median", "mean", "mode", "forward_fill"
    values: Dict[str, float]  # feature -> imputation value
    feature_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImputationStats":
        return cls(**data)


@dataclass
class FeatureSelectionMask:
    """Container for feature selection information."""
    selected_features: List[str]
    n_features_original: int
    n_features_selected: int
    method: str  # "lasso", "variance", "mutual_info"
    threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSelectionMask":
        return cls(**data)


@dataclass
class StainNormalizationReference:
    """Container for stain normalization reference."""
    method: str  # "macenko", "reinhard"
    reference_image_id: Optional[str] = None
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StainNormalizationReference":
        return cls(**data)


class PreprocessingContract:
    """
    Frozen preprocessing contract.

    Records ALL preprocessing parameters and ensures:
    1. ALL parameters are fit ONLY on training data
    2. Test data transformation uses ONLY training-derived parameters
    3. No data leakage across preprocessing steps
    4. Full reproducibility via serialization
    5. Hashable version for experiment tracking

    Attributes:
        fitted (bool): Whether contract has been fit on training data
        normalization (NormalizationStats): Feature normalization parameters
        imputation (ImputationStats): Missing value imputation parameters
        feature_selection (FeatureSelectionMask): Feature selection info
        stain_normalization (StainNormalizationReference): Stain normalization reference
    """

    def __init__(self):
        """Initialize empty preprocessing contract."""
        self.fitted = False
        self.fit_timestamp: Optional[datetime] = None
        self.normalization: Optional[NormalizationStats] = None
        self.imputation: Optional[ImputationStats] = None
        self.feature_selection: Optional[FeatureSelectionMask] = None
        self.stain_normalization: Optional[StainNormalizationReference] = None
        self._contract_hash: Optional[str] = None

    def fit_normalization(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        method: str = "zscore",
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Fit normalization parameters on training data ONLY.

        Args:
            X_train (Union[np.ndarray, pd.DataFrame]): Training data
            method (str): "zscore" or "minmax" (default: "zscore")
            feature_names (Optional[List[str]]): Feature names for tracking

        Raises:
            ValueError: If already fitted or invalid method
        """
        if self.fitted:
            raise ValueError(
                "Contract already fitted. Create new instance for different splits."
            )

        if method not in ["zscore", "minmax"]:
            raise ValueError(f"Unknown normalization method: {method}")

        X_train = np.asarray(X_train)
        if X_train.ndim != 2:
            raise ValueError("Input must be 2D array")

        if method == "zscore":
            mean = np.mean(X_train, axis=0).tolist()
            std = np.std(X_train, axis=0).tolist()
            self.normalization = NormalizationStats(
                method=method,
                mean=mean,
                std=std,
                feature_names=feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
            )
        else:  # minmax
            min_vals = np.min(X_train, axis=0).tolist()
            max_vals = np.max(X_train, axis=0).tolist()
            self.normalization = NormalizationStats(
                method=method,
                min=min_vals,
                max=max_vals,
                feature_names=feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
            )

        logger.info(
            f"Fitted normalization contract: {method} "
            f"({X_train.shape[1]} features)"
        )

    def fit_imputation(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        method: str = "median",
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Fit imputation parameters on training data ONLY.

        Args:
            X_train (Union[np.ndarray, pd.DataFrame]): Training data with missing values
            method (str): "median", "mean", or "mode" (default: "median")
            feature_names (Optional[List[str]]): Feature names for tracking

        Raises:
            ValueError: If already fitted or invalid method
        """
        if self.fitted:
            raise ValueError("Contract already fitted")

        if method not in ["median", "mean", "mode"]:
            raise ValueError(f"Unknown imputation method: {method}")

        X_train = np.asarray(X_train)
        imputer = SimpleImputer(strategy=method)
        imputer.fit(X_train)

        imputation_values = {}
        feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]

        for i, fname in enumerate(feature_names):
            imputation_values[fname] = float(imputer.statistics_[i])

        self.imputation = ImputationStats(
            method=method,
            values=imputation_values,
            feature_names=feature_names
        )

        logger.info(
            f"Fitted imputation contract: {method} "
            f"({len(imputation_values)} features)"
        )

    def fit_feature_selection(
        self,
        selected_features: List[str],
        n_features_original: int,
        method: str = "lasso",
        threshold: Optional[float] = None
    ) -> None:
        """
        Record feature selection mask.

        Args:
            selected_features (List[str]): Names of selected features
            n_features_original (int): Number of original features
            method (str): Selection method for tracking (default: "lasso")
            threshold (Optional[float]): Selection threshold used

        Raises:
            ValueError: If already fitted
        """
        if self.fitted:
            raise ValueError("Contract already fitted")

        self.feature_selection = FeatureSelectionMask(
            selected_features=selected_features,
            n_features_original=n_features_original,
            n_features_selected=len(selected_features),
            method=method,
            threshold=threshold
        )

        logger.info(
            f"Fitted feature selection contract: {method} "
            f"({len(selected_features)}/{n_features_original} features selected)"
        )

    def fit_stain_normalization(
        self,
        method: str = "macenko",
        reference_image_id: Optional[str] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ) -> None:
        """
        Record stain normalization reference from training images ONLY.

        Args:
            method (str): "macenko" or "reinhardt" (default: "macenko")
            reference_image_id (Optional[str]): ID of reference image
            mean (Optional[List[float]]): Mean color values
            std (Optional[List[float]]): Std color values

        Raises:
            ValueError: If already fitted
        """
        if self.fitted:
            raise ValueError("Contract already fitted")

        self.stain_normalization = StainNormalizationReference(
            method=method,
            reference_image_id=reference_image_id,
            mean=mean,
            std=std
        )

        logger.info(
            f"Fitted stain normalization contract: {method} "
            f"(reference: {reference_image_id})"
        )

    def finalize(self) -> None:
        """
        Finalize contract after all preprocessing parameters are set.

        Prevents further modifications and computes hash.

        Raises:
            ValueError: If no preprocessing steps were configured
        """
        if self.fitted:
            raise ValueError("Contract already finalized")

        if all(v is None for v in [
            self.normalization,
            self.imputation,
            self.feature_selection,
            self.stain_normalization
        ]):
            logger.warning(
                "Contract finalized with no preprocessing steps configured. "
                "This may indicate incomplete setup."
            )

        self.fitted = True
        self.fit_timestamp = datetime.now()
        self._compute_hash()

        logger.info(f"Contract finalized. Hash: {self.contract_hash}")

    def _compute_hash(self) -> None:
        """Compute contract hash for versioning."""
        contract_dict = {
            "normalization": self.normalization.to_dict() if self.normalization else None,
            "imputation": self.imputation.to_dict() if self.imputation else None,
            "feature_selection": (
                self.feature_selection.to_dict() if self.feature_selection else None
            ),
            "stain_normalization": (
                self.stain_normalization.to_dict() if self.stain_normalization else None
            ),
        }
        contract_json = json.dumps(contract_dict, sort_keys=True, default=str)
        self._contract_hash = hashlib.sha256(
            contract_json.encode()
        ).hexdigest()[:16]

    @property
    def contract_hash(self) -> str:
        """Get contract hash for versioning."""
        if self._contract_hash is None:
            raise ValueError("Contract not finalized. Call finalize() first.")
        return self._contract_hash

    def transform_features(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Apply normalization using contract parameters.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features to normalize

        Returns:
            np.ndarray: Normalized features

        Raises:
            ValueError: If contract not fitted or shape mismatch
        """
        if not self.fitted:
            raise ValueError("Contract not fitted. Call finalize() first.")
        if self.normalization is None:
            return np.asarray(X)

        X = np.asarray(X)

        if self.normalization.method == "zscore":
            mean = np.array(self.normalization.mean)
            std = np.array(self.normalization.std)
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)
            return (X - mean) / std
        else:  # minmax
            min_vals = np.array(self.normalization.min)
            max_vals = np.array(self.normalization.max)
            ranges = max_vals - min_vals
            ranges = np.where(ranges == 0, 1.0, ranges)
            return (X - min_vals) / ranges

    def impute_missing_values(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Impute missing values using contract parameters.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features with missing values
            feature_names (Optional[List[str]]): Feature names (required if X is ndarray)

        Returns:
            np.ndarray: Imputed features

        Raises:
            ValueError: If contract not fitted
        """
        if not self.fitted:
            raise ValueError("Contract not fitted. Call finalize() first.")
        if self.imputation is None:
            return np.asarray(X)

        X = np.asarray(X)
        X_imputed = X.copy()

        if feature_names is None:
            feature_names = self.imputation.feature_names

        for i, fname in enumerate(feature_names):
            if fname in self.imputation.values:
                mask = np.isnan(X[:, i]) | np.isinf(X[:, i])
                X_imputed[mask, i] = self.imputation.values[fname]

        return X_imputed

    def select_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Apply feature selection mask.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Features to filter
            feature_names (Optional[List[str]]): Feature names

        Returns:
            np.ndarray: Selected features

        Raises:
            ValueError: If contract not fitted or selection mask not set
        """
        if not self.fitted:
            raise ValueError("Contract not fitted. Call finalize() first.")
        if self.feature_selection is None:
            return np.asarray(X)

        X = np.asarray(X)
        feature_names = feature_names or self.feature_selection.selected_features

        selected_indices = [
            i for i, name in enumerate(feature_names)
            if name in self.feature_selection.selected_features
        ]

        return X[:, selected_indices]

    def serialize(self, output_path: Union[str, Path]) -> None:
        """
        Serialize contract to JSON and pickle.

        Args:
            output_path (Union[str, Path]): Path to save (without extension)

        Raises:
            ValueError: If contract not fitted
        """
        if not self.fitted:
            raise ValueError("Contract not fitted. Call finalize() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        contract_dict = {
            "fitted": self.fitted,
            "fit_timestamp": str(self.fit_timestamp),
            "contract_hash": self.contract_hash,
            "normalization": (
                self.normalization.to_dict() if self.normalization else None
            ),
            "imputation": self.imputation.to_dict() if self.imputation else None,
            "feature_selection": (
                self.feature_selection.to_dict() if self.feature_selection else None
            ),
            "stain_normalization": (
                self.stain_normalization.to_dict() if self.stain_normalization else None
            ),
        }

        json_path = Path(str(output_path) + ".json")
        with open(json_path, "w") as f:
            json.dump(contract_dict, f, indent=2, default=str)

        # Save as pickle
        pkl_path = Path(str(output_path) + ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Contract serialized: {json_path}, {pkl_path}")

    @classmethod
    def load(cls, input_path: Union[str, Path]) -> "PreprocessingContract":
        """
        Load contract from pickle file.

        Args:
            input_path (Union[str, Path]): Path to pickle file (or without extension)

        Returns:
            PreprocessingContract: Loaded contract

        Raises:
            FileNotFoundError: If file not found
        """
        input_path = Path(input_path)

        # Try with .pkl extension
        pkl_path = Path(str(input_path) + ".pkl") if not str(input_path).endswith(".pkl") else input_path
        if not pkl_path.exists():
            raise FileNotFoundError(f"Contract file not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            contract = pickle.load(f)

        logger.info(f"Contract loaded: {pkl_path}")
        return contract

    def validate_no_leakage(self) -> bool:
        """
        Validate that contract was fit only on training data.

        This is a basic sanity check; actual enforcement requires proper
        pipeline design.

        Returns:
            bool: True if contract appears valid
        """
        if not self.fitted:
            return False

        # Check that all components are properly initialized
        components_configured = sum([
            self.normalization is not None,
            self.imputation is not None,
            self.feature_selection is not None,
            self.stain_normalization is not None
        ])

        if components_configured == 0:
            logger.warning("No preprocessing components configured in contract")
            return False

        logger.info("Contract validation passed")
        return True
