"""
Patient-level data splitting for clinical imaging studies.

Ensures strict patient-level separation in train/val/test splits to prevent
tile/patch leakage. Implements stratified splitting, time-aware splitting for
longitudinal studies, and k-fold cross-validation with patient boundaries.

All splits are reproducible via fixed random seeds and saved to CSV files.
"""

from typing import Dict, List, Tuple, Optional, Union
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from pathlib import Path

logger = logging.getLogger(__name__)


class PatientLevelSplitter:
    """
    Base class for patient-level data splitting.

    Ensures all tiles/patches from a single patient stay together in the same fold.
    No patient can appear in multiple splits.

    Attributes:
        seed (int): Random seed for reproducibility
        split_assignments (pd.DataFrame): DataFrame recording split assignments
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the splitter.

        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.split_assignments = None

    def _validate_no_leakage(
        self,
        data: pd.DataFrame,
        patient_col: str,
        split_col: str
    ) -> bool:
        """
        Validate that no patient appears in multiple splits.

        Args:
            data (pd.DataFrame): Data with split assignments
            patient_col (str): Column name for patient IDs
            split_col (str): Column name for split assignments

        Returns:
            bool: True if validation passes

        Raises:
            ValueError: If patient leakage is detected
        """
        patient_splits = data.groupby(patient_col)[split_col].unique()
        for patient_id, splits in patient_splits.items():
            if len(splits) > 1:
                raise ValueError(
                    f"Patient leakage detected! Patient {patient_id} appears in "
                    f"splits: {splits}"
                )
        logger.info("Patient-level validation passed: no leakage detected")
        return True

    def _get_unique_patients(
        self,
        data: pd.DataFrame,
        patient_col: str
    ) -> np.ndarray:
        """
        Extract unique patients from data.

        Args:
            data (pd.DataFrame): Input data
            patient_col (str): Column name for patient IDs

        Returns:
            np.ndarray: Array of unique patient IDs
        """
        unique_patients = data[patient_col].unique()
        logger.info(f"Found {len(unique_patients)} unique patients")
        return unique_patients

    def save_split_assignments(self, output_path: Union[str, Path]) -> None:
        """
        Save split assignments to CSV for reproducibility.

        Args:
            output_path (Union[str, Path]): Path to save CSV file
        """
        if self.split_assignments is None:
            raise ValueError("No split assignments to save. Run split first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.split_assignments.to_csv(output_path, index=False)
        logger.info(f"Split assignments saved to {output_path}")


class StratifiedPatientSplit(PatientLevelSplitter):
    """
    Stratified splitting at the patient level.

    Splits data into train/val/test with class stratification at the patient level.
    Ensures balanced class representation across splits.

    Attributes:
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of training data for validation set
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.15,
        seed: int = 42
    ):
        """
        Initialize stratified patient-level splitter.

        Args:
            test_size (float): Proportion of data for test set (default: 0.2)
            val_size (float): Proportion of training data for validation (default: 0.15)
            seed (int): Random seed for reproducibility (default: 42)
        """
        super().__init__(seed=seed)
        self.test_size = test_size
        self.val_size = val_size

    def split(
        self,
        data: pd.DataFrame,
        patient_col: str = "patient_id",
        label_col: str = "label"
    ) -> Dict[str, pd.DataFrame]:
        """
        Create stratified train/val/test split at patient level.

        Args:
            data (pd.DataFrame): Input data with patient IDs and labels
            patient_col (str): Column name for patient IDs (default: "patient_id")
            label_col (str): Column name for class labels (default: "label")

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with keys 'train', 'val', 'test'

        Raises:
            ValueError: If patient leakage would occur or column not found
        """
        # Validate inputs
        if patient_col not in data.columns:
            raise ValueError(f"Patient column '{patient_col}' not found in data")
        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found in data")

        # Get unique patients with their labels
        patient_labels = data.groupby(patient_col)[label_col].first()
        unique_patients = patient_labels.index.values
        labels = patient_labels.values

        # First split: train+val vs test (stratified by label)
        train_val_patients, test_patients = train_test_split(
            unique_patients,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.rng
        )

        # Second split: train vs val (stratified by label)
        train_val_labels = patient_labels[train_val_patients]
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=self.val_size,
            stratify=train_val_labels,
            random_state=self.rng
        )

        # Create split assignments
        split_assignments = []
        for patient in train_patients:
            split_assignments.append({
                patient_col: patient,
                "split": "train",
                label_col: patient_labels[patient]
            })
        for patient in val_patients:
            split_assignments.append({
                patient_col: patient,
                "split": "val",
                label_col: patient_labels[patient]
            })
        for patient in test_patients:
            split_assignments.append({
                patient_col: patient,
                "split": "test",
                label_col: patient_labels[patient]
            })

        self.split_assignments = pd.DataFrame(split_assignments)

        # Map split assignments to tiles/patches
        data_with_splits = data.copy()
        data_with_splits["split"] = data_with_splits[patient_col].map(
            self.split_assignments.set_index(patient_col)["split"]
        )

        # Validate no leakage
        self._validate_no_leakage(data_with_splits, patient_col, "split")

        # Log split statistics
        for split_name in ["train", "val", "test"]:
            split_data = data_with_splits[data_with_splits["split"] == split_name]
            n_patients = split_data[patient_col].nunique()
            n_samples = len(split_data)
            logger.info(
                f"{split_name.upper()}: {n_patients} patients, {n_samples} samples"
            )

        return {
            "train": data_with_splits[data_with_splits["split"] == "train"],
            "val": data_with_splits[data_with_splits["split"] == "val"],
            "test": data_with_splits[data_with_splits["split"] == "test"]
        }


class TimeAwareSplitter(PatientLevelSplitter):
    """
    Time-aware splitting for longitudinal studies.

    Splits data based on time points, with training on earlier measurements
    and testing on later ones. Respects patient boundaries.

    Useful for:
    - Progression studies: train on baseline/early, test on follow-up
    - Sequential imaging: temporal ordering prevents information leakage
    """

    def __init__(self, seed: int = 42):
        """
        Initialize time-aware splitter.

        Args:
            seed (int): Random seed for reproducibility (default: 42)
        """
        super().__init__(seed=seed)

    def split(
        self,
        data: pd.DataFrame,
        patient_col: str = "patient_id",
        time_col: str = "diagnosis_date",
        test_time_cutoff: Optional[float] = None,
        val_time_cutoff: Optional[float] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Create time-aware split respecting patient boundaries.

        If cutoffs are not provided, uses quantiles (val: 50th, test: 80th).

        Args:
            data (pd.DataFrame): Input data with patient IDs and timestamps
            patient_col (str): Column name for patient IDs (default: "patient_id")
            time_col (str): Column name for timestamps (default: "diagnosis_date")
            test_time_cutoff (Optional[float]): Cutoff for test set (datetime or quantile)
            val_time_cutoff (Optional[float]): Cutoff for validation set (datetime or quantile)

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with keys 'train', 'val', 'test'

        Raises:
            ValueError: If time column not found or invalid cutoffs
        """
        if patient_col not in data.columns:
            raise ValueError(f"Patient column '{patient_col}' not found")
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found")

        # Get unique patients with their earliest time point
        patient_times = data.groupby(patient_col)[time_col].min()
        unique_patients = patient_times.index.values

        # Determine cutoffs
        if val_time_cutoff is None:
            val_time_cutoff = patient_times.quantile(0.5)
        if test_time_cutoff is None:
            test_time_cutoff = patient_times.quantile(0.8)

        # Split patients by time
        train_patients = unique_patients[patient_times <= val_time_cutoff]
        val_patients = unique_patients[
            (patient_times > val_time_cutoff) & (patient_times <= test_time_cutoff)
        ]
        test_patients = unique_patients[patient_times > test_time_cutoff]

        # Create assignment mapping
        split_map = {}
        for p in train_patients:
            split_map[p] = "train"
        for p in val_patients:
            split_map[p] = "val"
        for p in test_patients:
            split_map[p] = "test"

        # Apply splits
        data_with_splits = data.copy()
        data_with_splits["split"] = data_with_splits[patient_col].map(split_map)

        # Validate
        self._validate_no_leakage(data_with_splits, patient_col, "split")

        # Log statistics
        for split_name in ["train", "val", "test"]:
            split_data = data_with_splits[data_with_splits["split"] == split_name]
            n_patients = split_data[patient_col].nunique()
            n_samples = len(split_data)
            min_time = split_data[time_col].min()
            max_time = split_data[time_col].max()
            logger.info(
                f"{split_name.upper()}: {n_patients} patients, {n_samples} samples, "
                f"time range: {min_time} to {max_time}"
            )

        self.split_assignments = pd.DataFrame({
            patient_col: list(split_map.keys()),
            "split": list(split_map.values()),
            f"{time_col}_first": [patient_times[p] for p in split_map.keys()]
        })

        return {
            "train": data_with_splits[data_with_splits["split"] == "train"],
            "val": data_with_splits[data_with_splits["split"] == "val"],
            "test": data_with_splits[data_with_splits["split"] == "test"]
        }


class KFoldPatientSplit(PatientLevelSplitter):
    """
    K-fold cross-validation at patient level.

    Implements stratified k-fold splitting while respecting patient boundaries.
    All tiles/patches from a patient remain in the same fold.

    Attributes:
        n_splits (int): Number of folds
    """

    def __init__(self, n_splits: int = 5, seed: int = 42):
        """
        Initialize k-fold splitter.

        Args:
            n_splits (int): Number of folds (default: 5)
            seed (int): Random seed for reproducibility (default: 42)
        """
        super().__init__(seed=seed)
        self.n_splits = n_splits
        self.folds = None

    def split(
        self,
        data: pd.DataFrame,
        patient_col: str = "patient_id",
        label_col: str = "label"
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate k-fold splits at patient level.

        Args:
            data (pd.DataFrame): Input data with patient IDs and labels
            patient_col (str): Column name for patient IDs (default: "patient_id")
            label_col (str): Column name for class labels (default: "label")

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: List of (train, val) DataFrames for each fold

        Raises:
            ValueError: If patient leakage would occur
        """
        if patient_col not in data.columns:
            raise ValueError(f"Patient column '{patient_col}' not found")
        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found")

        # Get unique patients with labels
        patient_labels = data.groupby(patient_col)[label_col].first()
        unique_patients = patient_labels.index.values
        labels = patient_labels.values

        # Create stratified k-fold splitter
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed
        )

        fold_assignments = []
        fold_splits = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(unique_patients, labels)
        ):
            train_patients = unique_patients[train_idx]
            val_patients = unique_patients[val_idx]

            # Create fold assignment
            for p in train_patients:
                fold_assignments.append({
                    patient_col: p,
                    "fold": fold_idx,
                    "split": "train",
                    label_col: patient_labels[p]
                })
            for p in val_patients:
                fold_assignments.append({
                    patient_col: p,
                    "fold": fold_idx,
                    "split": "val",
                    label_col: patient_labels[p]
                })

            # Map to data
            data_train = data[data[patient_col].isin(train_patients)].copy()
            data_val = data[data[patient_col].isin(val_patients)].copy()

            fold_splits.append((data_train, data_val))

            # Log fold statistics
            logger.info(
                f"Fold {fold_idx}: Train {data_train[patient_col].nunique()} "
                f"patients ({len(data_train)} samples), Val "
                f"{data_val[patient_col].nunique()} patients ({len(data_val)} samples)"
            )

        self.split_assignments = pd.DataFrame(fold_assignments)
        self.folds = fold_splits

        return fold_splits

    def get_fold(self, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get a specific fold.

        Args:
            fold_idx (int): Fold index

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train, val) DataFrames for fold

        Raises:
            ValueError: If fold index is invalid or splits not generated
        """
        if self.folds is None:
            raise ValueError("Splits not generated yet. Call split() first.")
        if fold_idx < 0 or fold_idx >= len(self.folds):
            raise ValueError(
                f"Fold index {fold_idx} out of range [0, {len(self.folds)-1}]"
            )
        return self.folds[fold_idx]
