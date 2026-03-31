"""
Comprehensive tests for evaluation framework.

Tests splitter for patient leakage, metrics computation on known data,
preprocessing contract reproducibility, bootstrap CI coverage, and
visualization output validity.
"""

import unittest
from pathlib import Path
import tempfile
import shutil

import numpy as np
import pandas as pd

from src.evaluation import (
    StratifiedPatientSplit,
    ClassificationMetrics,
    PreprocessingContract,
    ROCCurveVisualizer,
    MetricResult,
)


class TestStratifiedPatientSplit(unittest.TestCase):
    """Test patient-level splitter for data leakage."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        n_patients = 100
        samples_per_patient = 5

        patient_ids = np.repeat(np.arange(n_patients), samples_per_patient)
        labels = np.tile(np.array([0, 1]), len(patient_ids) // 2)
        self.test_data = pd.DataFrame({
            'patient_id': patient_ids,
            'sample_id': np.arange(len(patient_ids)),
            'label': labels,
            'feature_1': np.random.randn(len(patient_ids)),
            'feature_2': np.random.randn(len(patient_ids)),
        })

    def test_no_patient_leakage(self):
        """Test that patients don't appear in multiple splits."""
        splitter = StratifiedPatientSplit(test_size=0.2, val_size=0.15, seed=42)
        splits = splitter.split(self.test_data)

        # Collect patients in each split
        train_patients = set(splits['train']['patient_id'].unique())
        val_patients = set(splits['val']['patient_id'].unique())
        test_patients = set(splits['test']['patient_id'].unique())

        # Check no overlap
        self.assertEqual(len(train_patients & val_patients), 0,
                        "Train and val sets share patients")
        self.assertEqual(len(train_patients & test_patients), 0,
                        "Train and test sets share patients")
        self.assertEqual(len(val_patients & test_patients), 0,
                        "Val and test sets share patients")

    def test_all_patients_included(self):
        """Test that all patients are included in splits."""
        splitter = StratifiedPatientSplit(test_size=0.2, val_size=0.15, seed=42)
        splits = splitter.split(self.test_data)

        all_split_patients = set()
        for split_data in splits.values():
            all_split_patients.update(split_data['patient_id'].unique())

        original_patients = set(self.test_data['patient_id'].unique())

        self.assertEqual(all_split_patients, original_patients,
                        "Not all patients included in splits")

    def test_stratification(self):
        """Test that splits are stratified by label."""
        original_class_ratio = self.test_data['label'].value_counts(normalize=True)

        splitter = StratifiedPatientSplit(test_size=0.2, val_size=0.15, seed=42)
        splits = splitter.split(self.test_data)

        for split_name, split_data in splits.items():
            split_class_ratio = split_data['label'].value_counts(normalize=True)
            # Check ratio is preserved (allow 10% tolerance)
            for label in original_class_ratio.index:
                self.assertAlmostEqual(
                    original_class_ratio[label],
                    split_class_ratio[label],
                    delta=0.15,  # Allow some tolerance
                    msg=f"Class {label} ratio not preserved in {split_name}"
                )

    def test_split_sizes(self):
        """Test that split sizes are approximately correct."""
        n_total = len(self.test_data)
        n_patients = self.test_data['patient_id'].nunique()

        splitter = StratifiedPatientSplit(test_size=0.2, val_size=0.15, seed=42)
        splits = splitter.split(self.test_data)

        # Expected proportions (approximate, at patient level)
        train_pct = len(splits['train']) / n_total
        val_pct = len(splits['val']) / n_total
        test_pct = len(splits['test']) / n_total

        # Allow some tolerance due to integer rounding at patient level
        self.assertGreaterEqual(train_pct, 0.65, "Train set too small")
        self.assertLess(train_pct, 0.75, "Train set too large")
        self.assertGreaterEqual(test_pct, 0.18, "Test set too small")
        self.assertLess(test_pct, 0.25, "Test set too large")

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same splits."""
        splitter1 = StratifiedPatientSplit(test_size=0.2, val_size=0.15, seed=42)
        splits1 = splitter1.split(self.test_data)

        splitter2 = StratifiedPatientSplit(test_size=0.2, val_size=0.15, seed=42)
        splits2 = splitter2.split(self.test_data)

        # Check same patients in train sets
        train_patients_1 = set(splits1['train']['patient_id'].unique())
        train_patients_2 = set(splits2['train']['patient_id'].unique())
        self.assertEqual(train_patients_1, train_patients_2,
                        "Different seeds produce different splits")

    def test_save_split_assignments(self):
        """Test saving split assignments to CSV."""
        splitter = StratifiedPatientSplit(seed=42)
        splitter.split(self.test_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'splits.csv'
            splitter.save_split_assignments(output_path)

            self.assertTrue(output_path.exists(), "Split file not created")

            # Verify format
            splits_df = pd.read_csv(output_path)
            self.assertIn('patient_id', splits_df.columns)
            self.assertIn('split', splits_df.columns)
            self.assertIn('label', splits_df.columns)


class TestClassificationMetrics(unittest.TestCase):
    """Test classification metrics computation."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 200
        self.y_true = np.random.randint(0, 2, self.n_samples)
        self.y_pred = self.y_true.copy()
        self.y_pred[np.random.choice(self.n_samples, 20)] = 1 - self.y_pred[np.random.choice(self.n_samples, 20)]
        self.y_pred_proba = np.random.rand(self.n_samples)

    def test_auroc_computation(self):
        """Test AUROC computation."""
        metrics = ClassificationMetrics(bootstrap_iterations=100, random_seed=42)
        result = metrics.compute_auroc(self.y_true, self.y_pred_proba)

        self.assertIsInstance(result['auroc'], MetricResult)
        self.assertGreaterEqual(result['auroc'].value, 0.0)
        self.assertLessEqual(result['auroc'].value, 1.0)
        self.assertGreater(result['auroc'].ci_upper, result['auroc'].ci_lower)

    def test_accuracy_computation(self):
        """Test accuracy computation."""
        metrics = ClassificationMetrics(bootstrap_iterations=100, random_seed=42)
        result = metrics.compute_accuracy(self.y_true, self.y_pred)

        self.assertIsInstance(result, MetricResult)
        self.assertGreaterEqual(result.value, 0.0)
        self.assertLessEqual(result.value, 1.0)

    def test_f1_computation(self):
        """Test F1 score computation."""
        metrics = ClassificationMetrics(bootstrap_iterations=100, random_seed=42)
        result = metrics.compute_f1(self.y_true, self.y_pred, average='macro')

        self.assertIsInstance(result, MetricResult)
        self.assertGreaterEqual(result.value, 0.0)
        self.assertLessEqual(result.value, 1.0)

    def test_sensitivity_specificity(self):
        """Test sensitivity, specificity, PPV, NPV computation."""
        metrics = ClassificationMetrics(bootstrap_iterations=100, random_seed=42)
        result = metrics.compute_sensitivity_specificity_ppv_npv(
            self.y_true, self.y_pred
        )

        self.assertIn('sensitivity', result)
        self.assertIn('specificity', result)
        self.assertIn('ppv', result)
        self.assertIn('npv', result)

        for metric_name, metric_value in result.items():
            self.assertIsInstance(metric_value, MetricResult)
            self.assertGreaterEqual(metric_value.value, 0.0)
            self.assertLessEqual(metric_value.value, 1.0)

    def test_bootstrap_ci_coverage(self):
        """Test that bootstrap CIs have reasonable coverage."""
        # Create data with known properties
        np.random.seed(42)
        n_samples = 500
        y_true = np.array([0] * 250 + [1] * 250)
        y_pred_proba = np.array(
            [0.3 + 0.2 * np.random.rand() for _ in range(250)] +
            [0.7 + 0.2 * np.random.rand() for _ in range(250)]
        )

        metrics = ClassificationMetrics(bootstrap_iterations=1000, random_seed=42)
        result = metrics.compute_auroc(y_true, y_pred_proba)

        # True AUROC should be around 0.8 (well-separated classes)
        # CI should contain it
        self.assertLess(result['auroc'].ci_lower, 0.85)
        self.assertGreater(result['auroc'].ci_upper, 0.75)

    def test_all_metrics_together(self):
        """Test computing all metrics at once."""
        metrics = ClassificationMetrics(bootstrap_iterations=100, random_seed=42)
        all_metrics = metrics.compute_all_classification_metrics(
            self.y_true, self.y_pred, self.y_pred_proba
        )

        expected_keys = ['auroc', 'auprc', 'accuracy', 'balanced_accuracy',
                        'f1_macro', 'f1_weighted', 'sensitivity', 'specificity',
                        'ppv', 'npv']

        for key in expected_keys:
            self.assertIn(key, all_metrics, f"Missing metric: {key}")


class TestPreprocessingContract(unittest.TestCase):
    """Test preprocessing contract."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.X_train = np.random.randn(100, 10)
        self.X_test = np.random.randn(50, 10)

    def test_fit_normalization(self):
        """Test normalization contract fit."""
        contract = PreprocessingContract()
        contract.fit_normalization(self.X_train, method='zscore')

        self.assertIsNotNone(contract.normalization)
        self.assertEqual(contract.normalization.method, 'zscore')
        self.assertEqual(len(contract.normalization.mean), 10)

    def test_transform_normalization(self):
        """Test normalization transformation."""
        contract = PreprocessingContract()
        contract.fit_normalization(self.X_train, method='zscore')
        contract.finalize()

        X_test_normalized = contract.transform_features(self.X_test)

        # Check normalization was applied
        self.assertNotEqual(np.mean(X_test_normalized), np.mean(self.X_test))
        self.assertEqual(X_test_normalized.shape, self.X_test.shape)

    def test_fit_imputation(self):
        """Test imputation contract fit."""
        X_with_nan = self.X_train.copy()
        X_with_nan[0, 0] = np.nan

        contract = PreprocessingContract()
        contract.fit_imputation(X_with_nan, method='median')

        self.assertIsNotNone(contract.imputation)
        self.assertEqual(contract.imputation.method, 'median')

    def test_feature_selection_mask(self):
        """Test feature selection mask."""
        selected_features = [f'feature_{i}' for i in range(5)]
        contract = PreprocessingContract()
        contract.fit_feature_selection(
            selected_features=selected_features,
            n_features_original=10,
            method='lasso'
        )

        self.assertEqual(contract.feature_selection.n_features_selected, 5)

    def test_contract_hash_changes(self):
        """Test that contract hash changes with different parameters."""
        contract1 = PreprocessingContract()
        contract1.fit_normalization(self.X_train, method='zscore')
        contract1.finalize()
        hash1 = contract1.contract_hash

        contract2 = PreprocessingContract()
        contract2.fit_normalization(self.X_train, method='minmax')
        contract2.finalize()
        hash2 = contract2.contract_hash

        self.assertNotEqual(hash1, hash2, "Different contracts have same hash")

    def test_contract_serialization(self):
        """Test contract serialization and loading."""
        contract = PreprocessingContract()
        contract.fit_normalization(self.X_train, method='zscore')
        contract.fit_imputation(self.X_train, method='median')
        contract.finalize()

        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = Path(tmpdir) / 'contract'
            contract.serialize(contract_path)

            # Check files created
            self.assertTrue(Path(str(contract_path) + '.json').exists())
            self.assertTrue(Path(str(contract_path) + '.pkl').exists())

            # Load and verify
            loaded_contract = PreprocessingContract.load(contract_path)
            self.assertEqual(loaded_contract.contract_hash, contract.contract_hash)
            self.assertEqual(loaded_contract.normalization.method, 'zscore')

    def test_prevent_fit_after_finalize(self):
        """Test that fitting is prevented after finalize."""
        contract = PreprocessingContract()
        contract.fit_normalization(self.X_train)
        contract.finalize()

        with self.assertRaises(ValueError):
            contract.fit_imputation(self.X_train)

    def test_transform_before_finalize_raises(self):
        """Test that transform raises error before finalize."""
        contract = PreprocessingContract()
        contract.fit_normalization(self.X_train)

        with self.assertRaises(ValueError):
            contract.transform_features(self.X_test)


class TestVisualization(unittest.TestCase):
    """Test visualization generation."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.y_true = np.array([0] * 50 + [1] * 50)
        self.y_pred_proba = np.concatenate([
            np.random.uniform(0, 0.5, 50),
            np.random.uniform(0.5, 1.0, 50)
        ])
        self.y_pred = (self.y_pred_proba > 0.5).astype(int)

    def test_roc_curve_visualization(self):
        """Test ROC curve visualization."""
        viz = ROCCurveVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'roc_curve'
            fig, ax = viz.plot(
                self.y_true,
                self.y_pred_proba,
                output_path=str(output_path)
            )

            # Check files created
            self.assertTrue(Path(str(output_path) + '.png').exists())
            self.assertTrue(Path(str(output_path) + '.pdf').exists())

    def test_confusion_matrix_visualization(self):
        """Test confusion matrix visualization."""
        from src.evaluation import ConfusionMatrixVisualizer

        viz = ConfusionMatrixVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'cm'
            fig, ax = viz.plot(
                self.y_true,
                self.y_pred,
                output_path=str(output_path)
            )

            # Check files created
            self.assertTrue(Path(str(output_path) + '.png').exists())

    def test_calibration_visualization(self):
        """Test calibration plot visualization."""
        from src.evaluation import CalibrationVisualizer

        viz = CalibrationVisualizer()

        # Create dummy calibration curve data
        mean_pred = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        frac_pos = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'calibration'
            fig, ax = viz.plot(
                mean_pred,
                frac_pos,
                ece=0.05,
                output_path=str(output_path)
            )

            # Check files created
            self.assertTrue(Path(str(output_path) + '.png').exists())


class TestMetricsBehavior(unittest.TestCase):
    """Tests that verify metric values on known data, not just types."""

    def test_auroc_perfectly_separable(self):
        """AUROC on perfectly separable data should be 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        metrics = ClassificationMetrics(bootstrap_iterations=50, random_seed=42)
        result = metrics.compute_auroc(y_true, y_score)

        self.assertAlmostEqual(result['auroc'].value, 1.0, places=5,
                               msg="Perfectly separable data should give AUROC=1.0")

    def test_auroc_random_baseline(self):
        """AUROC on random data should be approximately 0.5."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 500)
        y_score = rng.random(500)

        metrics = ClassificationMetrics(bootstrap_iterations=50, random_seed=42)
        result = metrics.compute_auroc(y_true, y_score)

        self.assertAlmostEqual(result['auroc'].value, 0.5, delta=0.08,
                               msg="Random predictions should give AUROC near 0.5")

    def test_accuracy_all_correct(self):
        """Accuracy should be 1.0 when all predictions are correct."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        metrics = ClassificationMetrics(bootstrap_iterations=50, random_seed=42)
        result = metrics.compute_accuracy(y_true, y_pred)

        self.assertAlmostEqual(result.value, 1.0, places=5)

    def test_sensitivity_known_values(self):
        """Sensitivity should be TP / (TP + FN)."""
        # 3 true positives, 1 false negative -> sensitivity = 0.75
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0])

        metrics = ClassificationMetrics(bootstrap_iterations=50, random_seed=42)
        result = metrics.compute_sensitivity_specificity_ppv_npv(y_true, y_pred)

        self.assertAlmostEqual(result['sensitivity'].value, 0.75, places=5)
        self.assertAlmostEqual(result['specificity'].value, 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
