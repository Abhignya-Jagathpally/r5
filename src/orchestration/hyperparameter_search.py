"""
Hyperparameter search using Ray Tune for model optimization.

Implements:
- Search space definitions for multiple model architectures
- ASHA and PBT schedulers for efficient search
- Multiple search algorithms (Optuna, Random, Bayesian)
- Patient-level cross-validation within search
- MLflow/W&B integration for experiment tracking
"""

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSearchConfig:
    """Master configuration for hyperparameter search."""

    # Search budget
    max_trials: int = 50
    max_wall_clock_hours: float = 24.0
    num_samples: int = 50
    max_concurrent_trials: int = 4

    # Scheduler configuration
    scheduler: str = "asha"  # 'asha', 'pbt', or 'fifo'
    grace_period: int = 10
    reduction_factor: int = 3

    # Search algorithm
    search_algorithm: str = "optuna"  # 'optuna', 'random', 'bayesian'

    # Model-specific search spaces
    model_search_spaces: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Optimization metric
    metric: str = "auroc"
    metric_mode: str = "max"  # 'max' or 'min'

    # Experiment tracking
    tracking_backend: str = "mlflow"  # 'mlflow' or 'wandb'
    experiment_name: str = "mm_hparam_search"

    # Data and CV
    num_folds: int = 5
    seed: int = 42

    # Checkpointing
    checkpoint_freq: int = 1
    keep_checkpoints: int = 3


class HyperparameterSearcher:
    """
    Ray Tune-based hyperparameter search with multiple algorithms.

    Manages hyperparameter optimization with:
    - Configurable search spaces per model type
    - Efficient schedulers (ASHA, PBT)
    - Patient-level cross-validation
    - Experiment tracking and logging

    Example:
        >>> config = HyperparameterSearchConfig(max_trials=50)
        >>> searcher = HyperparameterSearcher(config)
        >>> best_config = searcher.search(
        ...     train_fn=train_model,
        ...     search_space=search_space,
        ...     data_loaders=(train_loader, val_loader)
        ... )
    """

    def __init__(self, config: HyperparameterSearchConfig):
        """
        Initialize hyperparameter searcher.

        Args:
            config: HyperparameterSearchConfig instance
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_search_spaces()

    def _setup_search_spaces(self):
        """Initialize default search spaces for all model types."""
        self.search_spaces = {
            "abmil": self._get_abmil_search_space(),
            "clam": self._get_clam_search_space(),
            "transmil": self._get_transmil_search_space(),
            "dsmil": self._get_dsmil_search_space(),
            "fusion": self._get_fusion_search_space(),
        }

        # Override with user-provided spaces
        self.search_spaces.update(self.config.model_search_spaces)

    @staticmethod
    def _get_abmil_search_space() -> Dict[str, Any]:
        """Attention-based MIL search space."""
        from ray import tune

        return {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "weight_decay": tune.loguniform(1e-7, 1e-3),
            "dropout_rate": tune.uniform(0.0, 0.5),
            "hidden_dim": tune.choice([256, 512, 768]),
            "num_attention_heads": tune.choice([4, 8, 16]),
            "warmup_steps": tune.randint(0, 500),
            "batch_size": tune.choice([32, 64, 128]),
        }

    @staticmethod
    def _get_clam_search_space() -> Dict[str, Any]:
        """CLAM search space."""
        from ray import tune

        return {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "weight_decay": tune.loguniform(1e-7, 1e-3),
            "dropout_rate": tune.uniform(0.0, 0.5),
            "instance_loss_weight": tune.uniform(0.0, 1.0),
            "bag_loss_weight": tune.uniform(0.0, 1.0),
            "num_classes": tune.choice([2, 3]),
            "hidden_dim": tune.choice([512, 768, 1024]),
            "batch_size": tune.choice([32, 64, 128]),
        }

    @staticmethod
    def _get_transmil_search_space() -> Dict[str, Any]:
        """TransMIL search space."""
        from ray import tune

        return {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "weight_decay": tune.loguniform(1e-7, 1e-3),
            "dropout_rate": tune.uniform(0.0, 0.5),
            "num_transformer_layers": tune.randint(1, 6),
            "transformer_hidden_dim": tune.choice([256, 512, 768]),
            "num_transformer_heads": tune.choice([4, 8, 16]),
            "batch_size": tune.choice([32, 64, 128]),
        }

    @staticmethod
    def _get_dsmil_search_space() -> Dict[str, Any]:
        """Dual-stream MIL search space."""
        from ray import tune

        return {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "weight_decay": tune.loguniform(1e-7, 1e-3),
            "dropout_rate": tune.uniform(0.0, 0.5),
            "instance_pooling": tune.choice(["max", "mean", "attention"]),
            "bag_pooling": tune.choice(["max", "mean", "attention"]),
            "hidden_dim": tune.choice([256, 512, 768]),
            "batch_size": tune.choice([32, 64, 128]),
        }

    @staticmethod
    def _get_fusion_search_space() -> Dict[str, Any]:
        """Multimodal fusion search space."""
        from ray import tune

        return {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "weight_decay": tune.loguniform(1e-7, 1e-3),
            "dropout_rate": tune.uniform(0.0, 0.5),
            "fusion_method": tune.choice(["concat", "bilinear", "gated"]),
            "fusion_hidden_dim": tune.choice([256, 512, 768]),
            "pathology_weight": tune.uniform(0.0, 1.0),
            "radiomics_weight": tune.uniform(0.0, 1.0),
            "batch_size": tune.choice([32, 64, 128]),
        }

    def search(
        self,
        model_type: str,
        train_fn: Callable,
        train_data: Any,
        val_data: Any,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter search for a model.

        Args:
            model_type: Type of model ('abmil', 'clam', etc.)
            train_fn: Training function with signature train_fn(config, train_data, val_data)
            train_data: Training dataset
            val_data: Validation dataset
            output_dir: Directory for saving results

        Returns:
            Dictionary with best config and metrics

        Raises:
            ValueError: If model_type not supported
            RuntimeError: If search fails
        """
        if model_type not in self.search_spaces:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported: {list(self.search_spaces.keys())}"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Starting hyperparameter search for {model_type} "
            f"with budget: {self.config.max_trials} trials, "
            f"{self.config.max_wall_clock_hours} hours"
        )

        try:
            from ray import tune
            from ray.tune import CLIReporter

            search_space = self.search_spaces[model_type]

            # Setup scheduler
            scheduler = self._create_scheduler()

            # Setup search algorithm
            search_alg = self._create_search_algorithm(search_space)

            # Setup reporter
            reporter = CLIReporter(
                metric_columns=[self.config.metric, "time_this_iter_s"]
            )

            # Run search
            analysis = tune.run(
                tune.with_parameters(
                    train_fn,
                    train_data=train_data,
                    val_data=val_data,
                ),
                name=f"{self.config.experiment_name}_{model_type}",
                config=search_space,
                num_samples=self.config.num_samples,
                scheduler=scheduler,
                search_alg=search_alg,
                metric=self.config.metric,
                mode=self.config.metric_mode,
                max_concurrent_trials=self.config.max_concurrent_trials,
                checkpoint_freq=self.config.checkpoint_freq,
                checkpoint_keep_num=self.config.keep_checkpoints,
                progress_reporter=reporter,
                local_dir=str(output_dir),
                verbose=1,
                stop={
                    "training_iteration": self.config.max_trials,
                    "time_total_s": self.config.max_wall_clock_hours * 3600,
                },
            )

            # Extract best config
            best_trial = analysis.get_best_trial(
                metric=self.config.metric, mode=self.config.metric_mode
            )

            best_config = best_trial.config
            best_metrics = best_trial.last_result

            self.logger.info(f"Search complete. Best {self.config.metric}: {best_metrics[self.config.metric]:.4f}")
            self.logger.info(f"Best config: {best_config}")

            return {
                "config": best_config,
                "metrics": best_metrics,
                "trial_id": best_trial.trial_id,
                "analysis": analysis,
            }

        except Exception as e:
            self.logger.error(f"Hyperparameter search failed: {str(e)}")
            raise RuntimeError(f"Search failed: {str(e)}")

    def _create_scheduler(self) -> Any:
        """Create Ray Tune scheduler."""
        from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

        if self.config.scheduler == "asha":
            return ASHAScheduler(
                metric=self.config.metric,
                mode=self.config.metric_mode,
                max_t=self.config.max_trials,
                grace_period=self.config.grace_period,
                reduction_factor=self.config.reduction_factor,
            )
        elif self.config.scheduler == "pbt":
            return PopulationBasedTraining(
                metric=self.config.metric,
                mode=self.config.metric_mode,
                population_size=self.config.num_samples,
            )
        else:
            return None  # FIFO scheduler

    def _create_search_algorithm(self, search_space: Dict[str, Any]) -> Any:
        """Create search algorithm instance."""
        from ray.tune.suggest import Searcher

        if self.config.search_algorithm == "optuna":
            try:
                from ray.tune.suggest.optuna import OptunaSearch
                return OptunaSearch(space=search_space)
            except Exception as e:
                self.logger.warning(
                    f"Optuna search unavailable ({str(e)}), falling back to random"
                )
                return None

        elif self.config.search_algorithm == "bayesian":
            try:
                from ray.tune.suggest.skopt import SkoptSearch
                return SkoptSearch(space=search_space, metric=self.config.metric, mode=self.config.metric_mode)
            except Exception as e:
                self.logger.warning(
                    f"Bayesian search unavailable ({str(e)}), falling back to random"
                )
                return None

        return None  # Random search

    def run_cv_search(
        self,
        model_type: str,
        train_fn: Callable,
        data: Any,
        labels: np.ndarray,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter search with patient-level cross-validation.

        Ensures no data leakage by conducting CV within search.

        Args:
            model_type: Type of model
            train_fn: Training function
            data: Full dataset
            labels: Patient labels (for stratification)
            output_dir: Output directory

        Returns:
            Best config with CV metrics

        Raises:
            RuntimeError: If search fails
        """
        from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

        self.logger.info(
            f"Running {self.config.num_folds}-fold CV search for {model_type}"
        )

        # Use patient-level CV if patient_ids are available
        patient_ids = getattr(data, "patient_ids", None)
        if patient_ids is None and hasattr(data, "__len__") and isinstance(data, np.ndarray) is False:
            # Try to extract from dataframe-like objects
            patient_ids = getattr(data, "patient_id", None)

        if patient_ids is not None:
            self.logger.info("Using patient-level StratifiedGroupKFold (no leakage)")
            kfold = StratifiedGroupKFold(
                n_splits=self.config.num_folds, shuffle=True, random_state=self.config.seed
            )
            split_iter = kfold.split(data, labels, groups=patient_ids)
        else:
            self.logger.warning(
                "No patient_ids available — falling back to sample-level StratifiedKFold. "
                "Risk of patient-level leakage if multiple samples per patient."
            )
            kfold = StratifiedKFold(
                n_splits=self.config.num_folds, shuffle=True, random_state=self.config.seed
            )
            split_iter = kfold.split(data, labels)

        cv_results = []
        fold_idx = 0

        try:
            for train_idx, val_idx in split_iter:
                fold_idx += 1
                self.logger.info(f"Running fold {fold_idx}/{self.config.num_folds}")

                # Extract fold data
                if hasattr(data, "__getitem__"):
                    train_data = [data[i] for i in train_idx]
                    val_data = [data[i] for i in val_idx]
                else:
                    train_data = data[train_idx]
                    val_data = data[val_idx]

                # Run search on this fold
                fold_result = self.search(
                    model_type=model_type,
                    train_fn=train_fn,
                    train_data=train_data,
                    val_data=val_data,
                    output_dir=output_dir / f"fold_{fold_idx}",
                )

                cv_results.append(fold_result)

            # Aggregate CV results
            metric_values = [
                r["metrics"][self.config.metric] for r in cv_results
            ]
            mean_metric = np.mean(metric_values)
            std_metric = np.std(metric_values)

            self.logger.info(
                f"CV Results: {self.config.metric} = {mean_metric:.4f} ± {std_metric:.4f}"
            )

            # Return best overall config
            best_fold_idx = np.argmax(metric_values)
            best_config = cv_results[best_fold_idx]["config"]

            return {
                "config": best_config,
                "mean_metric": mean_metric,
                "std_metric": std_metric,
                "cv_results": cv_results,
                "best_fold": best_fold_idx + 1,
            }

        except Exception as e:
            self.logger.error(f"CV search failed: {str(e)}")
            raise RuntimeError(f"CV search failed: {str(e)}")

    def get_search_space(self, model_type: str) -> Dict[str, Any]:
        """
        Get search space for a model type.

        Args:
            model_type: Type of model

        Returns:
            Search space dictionary

        Raises:
            ValueError: If model type not supported
        """
        if model_type not in self.search_spaces:
            raise ValueError(f"Unknown model type: {model_type}")
        return self.search_spaces[model_type]

    def list_model_types(self) -> List[str]:
        """List all supported model types."""
        return list(self.search_spaces.keys())
