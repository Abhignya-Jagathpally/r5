#!/usr/bin/env python3
"""
End-to-end evaluation script for Multiple Myeloma imaging pipeline.

Loads trained models, computes metrics with CIs, generates visualizations,
compares against benchmarks, and generates comprehensive reports.

Usage:
    python run_evaluation.py --config configs/evaluation.yaml --model_path model.pkl
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json

import yaml
import numpy as np
import pandas as pd

from src.evaluation import (
    StratifiedPatientSplit,
    ClassificationMetrics,
    SurvivalMetrics,
    CalibrationMetrics,
    PreprocessingContract,
    MLflowBackend,
    WandbBackend,
    ROCCurveVisualizer,
    ConfusionMatrixVisualizer,
    CalibrationVisualizer,
    EvaluationReportGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """End-to-end evaluation pipeline."""

    def __init__(self, config_path: str):
        """
        Initialize evaluation pipeline.

        Args:
            config_path (str): Path to evaluation config YAML file
        """
        self.config = self._load_config(config_path)

        # Validate config schema
        from src.utils.config_schema import validate_config, validate_no_conflicting_keys
        errors = validate_config(self.config, "evaluation")
        errors.extend(validate_no_conflicting_keys(self.config))
        if errors:
            for err in errors:
                logger.error(f"Config validation error: {err}")
            raise ValueError(f"Config validation failed with {len(errors)} error(s)")

        self.setup_logging()
        logger.info("Evaluation pipeline initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config

    def setup_logging(self) -> None:
        """Setup logging based on config."""
        log_file = self.config['logging'].get('log_file', 'evaluation.log')
        log_level = self.config['logging'].get('level', 'INFO')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to {log_file}")

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from CSV.

        Args:
            data_path (str): Path to CSV file

        Returns:
            pd.DataFrame: Loaded data
        """
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data)} samples from {data_path}")
        return data

    def create_splits(
        self,
        data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Create train/val/test splits.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            Dict[str, pd.DataFrame]: Split data
        """
        method = self.config['splitting']['method']
        seed = self.config['splitting']['seed']

        if method == 'stratified_patient':
            splitter = StratifiedPatientSplit(
                test_size=self.config['splitting']['test_size'],
                val_size=self.config['splitting']['val_size'],
                seed=seed
            )

            splits = splitter.split(
                data,
                patient_col=self.config['splitting']['patient_column'],
                label_col=self.config['splitting']['label_column']
            )

            # Save split assignments
            split_assign_path = (
                Path(self.config['paths']['preprocessing_dir']) /
                'split_assignments.csv'
            )
            split_assign_path.parent.mkdir(parents=True, exist_ok=True)
            splitter.save_split_assignments(split_assign_path)

            logger.info(f"Created stratified patient-level splits")
            return splits
        else:
            raise NotImplementedError(f"Split method {method} not implemented")

    def load_preprocessing_contract(self, contract_path: str) -> PreprocessingContract:
        """
        Load preprocessing contract.

        Args:
            contract_path (str): Path to contract file

        Returns:
            PreprocessingContract: Loaded contract
        """
        contract = PreprocessingContract.load(contract_path)
        logger.info(f"Loaded preprocessing contract from {contract_path}")
        return contract

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Compute all metrics.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities
            task_type (str): 'classification' or 'survival'

        Returns:
            Dict[str, Any]: Dictionary of metrics with CIs
        """
        metrics = {}

        # Classification metrics
        if task_type == 'classification':
            clf_metrics = ClassificationMetrics(
                bootstrap_iterations=self.config['metrics']['bootstrap_iterations']
            )
            metrics.update(clf_metrics.compute_all_classification_metrics(
                y_true, y_pred, y_pred_proba
            ))

            # Calibration metrics
            cal_metrics = CalibrationMetrics()
            cal_result = cal_metrics.compute_all_calibration_metrics(
                y_true, y_pred_proba
            )
            metrics['ece'] = cal_result['ece']

        logger.info(f"Computed {len(metrics)} metrics")
        return metrics

    def generate_visualizations(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str,
        output_dir: str
    ) -> Dict[str, Path]:
        """
        Generate all visualizations.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Model name for titles
            output_dir (str): Directory to save figures

        Returns:
            Dict[str, Path]: Dictionary mapping figure names to paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        figures = {}

        # ROC curve
        roc_viz = ROCCurveVisualizer()
        if len(y_pred_proba.shape) > 1:
            auroc = self._compute_auroc(y_true, y_pred_proba)
        else:
            auroc = None

        roc_path = output_dir / f"{model_name}_roc_curve"
        roc_viz.plot(
            y_true, y_pred_proba,
            model_name=model_name,
            auroc=auroc,
            output_path=str(roc_path)
        )
        figures['ROC Curve'] = roc_path.with_suffix('.png')

        # Confusion matrix
        cm_viz = ConfusionMatrixVisualizer()
        cm_path = output_dir / f"{model_name}_confusion_matrix"
        cm_viz.plot(
            y_true, y_pred,
            normalize=True,
            output_path=str(cm_path)
        )
        figures['Confusion Matrix'] = cm_path.with_suffix('.png')

        # Calibration plot
        if len(y_pred_proba.shape) > 1:
            y_proba_binary = y_pred_proba[:, 1]
        else:
            y_proba_binary = y_pred_proba

        cal_metrics = CalibrationMetrics()
        cal_result = cal_metrics.compute_all_calibration_metrics(
            y_true, y_proba_binary
        )
        ece = cal_result['ece']

        cal_viz = CalibrationVisualizer()
        cal_path = output_dir / f"{model_name}_calibration"
        cal_viz.plot(
            cal_result['mean_predicted_prob'],
            cal_result['fraction_positive'],
            model_name=model_name,
            ece=ece,
            output_path=str(cal_path)
        )
        figures['Calibration'] = cal_path.with_suffix('.png')

        logger.info(f"Generated {len(figures)} visualizations")
        return figures

    def _compute_auroc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute AUROC."""
        from sklearn.metrics import roc_auc_score
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]
        return roc_auc_score(y_true, y_pred_proba)

    def setup_experiment_tracking(self) -> Optional[Any]:
        """
        Setup experiment tracking backend.

        Returns:
            Optional[Any]: Tracker object or None
        """
        backend = self.config['tracking'].get('backend')

        if backend == 'mlflow':
            tracker = MLflowBackend(
                experiment_name=self.config['tracking']['mlflow']['experiment_name'],
                tracking_uri=self.config['tracking']['mlflow']['tracking_uri']
            )
            tracker.start_run()
            logger.info("MLflow tracking started")
            return tracker

        elif backend == 'wandb':
            tracker = WandbBackend(
                experiment_name=self.config['tracking']['wandb'].get('project'),
                project=self.config['tracking']['wandb'].get('project'),
                entity=self.config['tracking']['wandb'].get('entity')
            )
            tracker.start_run()
            logger.info("Weights & Biases tracking started")
            return tracker

        return None

    def generate_report(
        self,
        experiment_name: str,
        dataset_info: Dict[str, Any],
        split_info: Dict[str, Any],
        preprocessing_info: Dict[str, Any],
        model_results: Dict[str, Dict[str, Any]],
        figures: Optional[Dict[str, Path]] = None
    ) -> Path:
        """
        Generate evaluation report.

        Args:
            experiment_name (str): Experiment name
            dataset_info (Dict[str, Any]): Dataset statistics
            split_info (Dict[str, Any]): Split information
            preprocessing_info (Dict[str, Any]): Preprocessing info
            model_results (Dict[str, Dict[str, Any]]): Per-model results
            figures (Optional[Dict[str, Path]]): Figures to include

        Returns:
            Path: Path to generated report
        """
        output_dir = self.config['reporting']['output_dir']
        generator = EvaluationReportGenerator(output_dir)

        report_path = generator.generate_full_report(
            experiment_name=experiment_name,
            dataset_info=dataset_info,
            split_info=split_info,
            preprocessing_info=preprocessing_info,
            model_results=model_results,
            benchmark_comparison=self.config.get('benchmarks'),
            figures=figures
        )

        # Also generate summary table
        if self.config['reporting'].get('generate_summary_table'):
            generator.generate_summary_table(model_results)

        # Generate CSV summary
        if self.config['reporting'].get('generate_csv_summary'):
            generator.generate_csv_summary(model_results)

        logger.info(f"Report generated: {report_path}")
        return report_path

    def run(
        self,
        data_path: str,
        y_pred_path: str,
        y_pred_proba_path: str,
        preprocessing_contract_path: Optional[str] = None,
        experiment_name: str = "MM Imaging Evaluation"
    ) -> None:
        """
        Run complete evaluation pipeline.

        Args:
            data_path (str): Path to data CSV file
            y_pred_path (str): Path to predictions (CSV)
            y_pred_proba_path (str): Path to probabilities (CSV)
            preprocessing_contract_path (Optional[str]): Path to preprocessing contract
            experiment_name (str): Experiment name
        """
        # Load data
        data = self.load_data(data_path)
        y_pred = pd.read_csv(y_pred_path).values.flatten()
        y_pred_proba = pd.read_csv(y_pred_proba_path).values

        # Create splits
        splits = self.create_splits(data)

        # Setup tracking
        tracker = self.setup_experiment_tracking()

        # Prepare results container
        model_results = {}
        all_figures = {}

        # Evaluate on test set
        test_data = splits['test']
        test_mask = test_data.index

        y_test_true = test_data[self.config['splitting']['label_column']].values
        y_test_pred = y_pred[test_mask]
        y_test_pred_proba = y_pred_proba[test_mask]

        # Compute metrics
        metrics = self.compute_metrics(
            y_test_true, y_test_pred, y_test_pred_proba
        )

        # Generate visualizations
        figures = self.generate_visualizations(
            y_test_true, y_test_pred, y_test_pred_proba,
            model_name="Model",
            output_dir=self.config['paths']['results_dir']
        )

        # Store results
        model_results['Model'] = {
            'metrics': metrics,
            'notes': f"Evaluated on {len(y_test_true)} test samples"
        }
        all_figures.update(figures)

        # Log to tracker
        if tracker:
            tracker.log_params(self.config['splitting'])
            tracker.log_metrics({
                k: v.value if hasattr(v, 'value') else v
                for k, v in metrics.items()
            })

        # Generate report
        dataset_info = {
            'n_patients': data[self.config['splitting']['patient_column']].nunique(),
            'n_samples': len(data),
            'n_classes': data[self.config['splitting']['label_column']].nunique(),
        }

        split_info = {
            'method': self.config['splitting']['method'],
            'seed': self.config['splitting']['seed'],
            'breakdown': {
                'train': {
                    'n_patients': splits['train'][self.config['splitting']['patient_column']].nunique(),
                    'n_samples': len(splits['train']),
                },
                'val': {
                    'n_patients': splits['val'][self.config['splitting']['patient_column']].nunique(),
                    'n_samples': len(splits['val']),
                },
                'test': {
                    'n_patients': splits['test'][self.config['splitting']['patient_column']].nunique(),
                    'n_samples': len(splits['test']),
                },
            }
        }

        preprocessing_info = {
            'contract_hash': 'N/A',
            'fit_timestamp': 'N/A'
        }

        if preprocessing_contract_path:
            contract = self.load_preprocessing_contract(preprocessing_contract_path)
            preprocessing_info['contract_hash'] = contract.contract_hash
            preprocessing_info['fit_timestamp'] = str(contract.fit_timestamp)

        # Generate final report
        report_path = self.generate_report(
            experiment_name=experiment_name,
            dataset_info=dataset_info,
            split_info=split_info,
            preprocessing_info=preprocessing_info,
            model_results=model_results,
            figures=all_figures
        )

        logger.info(f"Evaluation complete. Report: {report_path}")

        if tracker:
            tracker.end_run()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation for MM imaging pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to evaluation config YAML file'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data CSV file'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions CSV file'
    )
    parser.add_argument(
        '--probabilities',
        type=str,
        required=True,
        help='Path to prediction probabilities CSV file'
    )
    parser.add_argument(
        '--preprocessing-contract',
        type=str,
        default=None,
        help='Path to preprocessing contract (optional)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='MM Imaging Evaluation',
        help='Experiment name'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = EvaluationPipeline(args.config)
    pipeline.run(
        data_path=args.data,
        y_pred_path=args.predictions,
        y_pred_proba_path=args.probabilities,
        preprocessing_contract_path=args.preprocessing_contract,
        experiment_name=args.experiment_name
    )


if __name__ == '__main__':
    main()
