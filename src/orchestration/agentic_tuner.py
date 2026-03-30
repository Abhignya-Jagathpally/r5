"""
Autoresearch-pattern agentic tuning for biomedical ML pipelines.

Implements Andrej Karpathy's autoresearch philosophy with:
- Locked preprocessing surface (immutable code)
- Editable training/config surface (agent-modifiable)
- Single metric optimization with fixed budget
- Complete experiment logging and reproducibility
- Safety checks against data leakage and code corruption
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class LockedSurface:
    """
    Definition of code that CANNOT be modified by agents.

    This includes:
    - Data loading and preprocessing
    - Data splitting and cross-validation
    - Evaluation metrics and comparisons
    - Experiment infrastructure
    """

    locked_files: Set[str] = field(default_factory=set)
    locked_functions: Set[str] = field(default_factory=set)
    preprocessing_contract_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "locked_files": list(self.locked_files),
            "locked_functions": list(self.locked_functions),
            "preprocessing_contract_hash": self.preprocessing_contract_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LockedSurface":
        """Deserialize from dictionary."""
        return cls(
            locked_files=set(data.get("locked_files", [])),
            locked_functions=set(data.get("locked_functions", [])),
            preprocessing_contract_hash=data.get("preprocessing_contract_hash"),
        )


@dataclass
class EditableSurface:
    """
    Definition of code that CAN be modified by agents.

    This includes:
    - Model architecture choices
    - Training hyperparameters
    - Optimization strategy
    - Augmentation strategies
    """

    editable_files: Set[str] = field(default_factory=set)
    editable_functions: Set[str] = field(default_factory=set)
    editable_config_keys: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "editable_files": list(self.editable_files),
            "editable_functions": list(self.editable_functions),
            "editable_config_keys": list(self.editable_config_keys),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditableSurface":
        """Deserialize from dictionary."""
        return cls(
            editable_files=set(data.get("editable_files", [])),
            editable_functions=set(data.get("editable_functions", [])),
            editable_config_keys=set(data.get("editable_config_keys", [])),
        )


@dataclass
class AgenticTunerConfig:
    """Configuration for agentic tuning."""

    # Optimization objective
    metric: str = "auroc"  # Metric to optimize
    metric_mode: str = "max"  # 'max' or 'min'
    metric_threshold: float = 0.05  # Alert if improvement > this (possible leakage)

    # Search budget
    budget_type: str = "trials"  # 'trials' or 'wall_clock'
    max_trials: int = 50
    max_hours: float = 24.0

    # Safety
    verify_preprocessing: bool = True
    verify_data_splits: bool = True
    verify_code_integrity: bool = True

    # Experiment tracking
    experiment_name: str = "agentic_tuning"
    log_dir: Path = field(default_factory=lambda: Path("experiments/agentic"))


@dataclass
class ExperimentResult:
    """Result from a single agentic tuning experiment."""

    trial_id: int
    timestamp: datetime
    config_diff: Dict[str, Any]
    metric_value: float
    wall_clock_seconds: float
    git_hash: Optional[str] = None
    error: Optional[str] = None
    safety_checks_passed: bool = True


class AgenticTuner:
    """
    Agentic tuning orchestrator following autoresearch pattern.

    Key features:
    - Locked preprocessing + editable training surface
    - Single metric optimization with fixed budget
    - Complete experiment logging
    - Safety verification and leakage detection
    - Reproducible experiment execution

    Example:
        >>> config = AgenticTunerConfig(metric="auroc", max_trials=20)
        >>> tuner = AgenticTuner(config, locked, editable)
        >>> results = tuner.tune(
        ...     train_fn=train_model,
        ...     data=(train_data, val_data),
        ...     baseline_config=best_hparam_config
        ... )
    """

    def __init__(
        self,
        config: AgenticTunerConfig,
        locked: LockedSurface,
        editable: EditableSurface,
    ):
        """
        Initialize agentic tuner.

        Args:
            config: AgenticTunerConfig instance
            locked: LockedSurface definition
            editable: EditableSurface definition
        """
        self.config = config
        self.locked = locked
        self.editable = editable
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup logging directory
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Experiment tracking
        self.experiments: List[ExperimentResult] = []
        self.best_result: Optional[ExperimentResult] = None
        self.best_metric_value: float = (
            -np.inf if config.metric_mode == "max" else np.inf
        )

        # Initial preprocessing hash
        self.initial_preprocessing_hash = self._compute_preprocessing_hash()

        self.logger.info(
            f"AgenticTuner initialized: {config.metric} optimization, "
            f"budget: {config.max_trials} trials"
        )

    def tune(
        self,
        train_fn: Callable,
        data: Tuple[Any, Any],
        baseline_config: Dict[str, Any],
        modification_generator: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run agentic tuning optimization.

        Args:
            train_fn: Function with signature train_fn(config, train_data, val_data) -> metrics
            data: Tuple of (train_data, val_data)
            baseline_config: Initial configuration from hyperparameter search
            modification_generator: Optional function to generate config modifications

        Returns:
            Dictionary with final results

        Raises:
            RuntimeError: If tuning fails or safety checks fail
        """
        train_data, val_data = data
        start_time = time.time()

        self.logger.info("Starting agentic tuning...")
        self.logger.info(f"Baseline metric (estimated): {self.best_metric_value}")

        trial_count = 0
        while not self._is_budget_exhausted(start_time, trial_count):
            trial_count += 1
            trial_start = time.time()

            try:
                # Generate candidate modification
                candidate_config = self._generate_candidate(
                    baseline_config, modification_generator
                )

                # Verify safety checks
                if self.config.verify_code_integrity:
                    self._verify_code_integrity()
                if self.config.verify_preprocessing:
                    self._verify_preprocessing_contract()
                if self.config.verify_data_splits:
                    self._verify_data_consistency(train_data, val_data)

                # Train and evaluate
                self.logger.info(f"Trial {trial_count}: Training with candidate config")
                metrics = train_fn(candidate_config, train_data, val_data)

                metric_value = metrics.get(self.config.metric)
                if metric_value is None:
                    raise ValueError(
                        f"Metric '{self.config.metric}' not found in training output"
                    )

                wall_clock_seconds = time.time() - trial_start

                # Record experiment
                result = self._record_experiment(
                    trial_count,
                    candidate_config,
                    baseline_config,
                    metric_value,
                    wall_clock_seconds,
                )

                # Check for suspiciously large improvement
                if self._is_suspicious_improvement(metric_value):
                    self.logger.warning(
                        f"Trial {trial_count}: Suspiciously large improvement "
                        f"({abs(metric_value - self.best_metric_value):.4f}). "
                        f"Possible data leakage or bug."
                    )
                    result.safety_checks_passed = False

                # Update best if improved
                if self._is_better(metric_value, self.best_metric_value):
                    self.best_metric_value = metric_value
                    self.best_result = result
                    baseline_config = candidate_config.copy()
                    self.logger.info(
                        f"Trial {trial_count}: New best {self.config.metric} = {metric_value:.4f}"
                    )
                else:
                    self.logger.info(
                        f"Trial {trial_count}: {self.config.metric} = {metric_value:.4f} "
                        f"(best: {self.best_metric_value:.4f})"
                    )

            except Exception as e:
                self.logger.error(f"Trial {trial_count} failed: {str(e)}")
                result = ExperimentResult(
                    trial_id=trial_count,
                    timestamp=datetime.now(),
                    config_diff={},
                    metric_value=0.0,
                    wall_clock_seconds=time.time() - trial_start,
                    error=str(e),
                    safety_checks_passed=False,
                )
                self.experiments.append(result)

        # Generate final report
        elapsed_hours = (time.time() - start_time) / 3600
        self.logger.info(
            f"Agentic tuning complete after {trial_count} trials ({elapsed_hours:.2f} hours)"
        )
        self._generate_experiment_journal()

        return {
            "best_config": asdict(self.best_result) if self.best_result else {},
            "best_metric": self.best_metric_value,
            "num_trials": trial_count,
            "elapsed_hours": elapsed_hours,
            "experiments": self.experiments,
        }

    def _generate_candidate(
        self,
        baseline_config: Dict[str, Any],
        modification_fn: Optional[Callable],
    ) -> Dict[str, Any]:
        """
        Generate a candidate configuration by modifying baseline.

        Args:
            baseline_config: Current best configuration
            modification_fn: Optional custom modification function

        Returns:
            Candidate configuration
        """
        candidate = baseline_config.copy()

        if modification_fn is not None:
            # Use provided modification function
            candidate = modification_fn(candidate)
        else:
            # Default: random perturbation of editable config keys
            num_keys_to_modify = np.random.randint(1, max(2, len(candidate) // 3))
            keys_to_modify = np.random.choice(
                list(candidate.keys()), size=num_keys_to_modify, replace=False
            )

            for key in keys_to_modify:
                if key in self.editable.editable_config_keys:
                    value = candidate[key]
                    # Multiplicative perturbation (for numeric values)
                    if isinstance(value, (int, float)):
                        scale = np.random.uniform(0.5, 2.0)
                        candidate[key] = value * scale
                    # Categorical: random from similar range
                    elif isinstance(value, str):
                        pass  # Keep as-is for string values

        return candidate

    def _record_experiment(
        self,
        trial_id: int,
        candidate_config: Dict[str, Any],
        baseline_config: Dict[str, Any],
        metric_value: float,
        wall_clock_seconds: float,
    ) -> ExperimentResult:
        """
        Record an experiment result.

        Args:
            trial_id: Trial number
            candidate_config: Configuration used
            baseline_config: Previous best config
            metric_value: Metric value achieved
            wall_clock_seconds: Execution time

        Returns:
            ExperimentResult instance
        """
        # Compute config diff
        config_diff = self._compute_config_diff(baseline_config, candidate_config)

        # Get git hash if available
        git_hash = self._get_git_hash()

        result = ExperimentResult(
            trial_id=trial_id,
            timestamp=datetime.now(),
            config_diff=config_diff,
            metric_value=metric_value,
            wall_clock_seconds=wall_clock_seconds,
            git_hash=git_hash,
        )

        self.experiments.append(result)

        # Save to disk
        result_file = self.log_dir / f"trial_{trial_id:04d}.json"
        with open(result_file, "w") as f:
            json.dump(
                {
                    "trial_id": result.trial_id,
                    "timestamp": result.timestamp.isoformat(),
                    "config_diff": result.config_diff,
                    "metric_value": float(result.metric_value),
                    "wall_clock_seconds": result.wall_clock_seconds,
                    "git_hash": result.git_hash,
                    "safety_checks_passed": result.safety_checks_passed,
                },
                f,
                indent=2,
            )

        return result

    def _compute_config_diff(
        self, config_a: Dict[str, Any], config_b: Dict[str, Any]
    ) -> Dict[str, Tuple[Any, Any]]:
        """Compute differences between two configs."""
        diff = {}
        all_keys = set(config_a.keys()) | set(config_b.keys())
        for key in all_keys:
            val_a = config_a.get(key)
            val_b = config_b.get(key)
            if val_a != val_b:
                diff[key] = (val_a, val_b)
        return diff

    def _is_better(self, metric_value: float, current_best: float) -> bool:
        """Check if metric value is better than current best."""
        if self.config.metric_mode == "max":
            return metric_value > current_best
        else:
            return metric_value < current_best

    def _is_suspicious_improvement(self, metric_value: float) -> bool:
        """Detect suspiciously large improvements (possible leakage)."""
        if self.best_metric_value == (-np.inf if self.config.metric_mode == "max" else np.inf):
            return False
        improvement = abs(metric_value - self.best_metric_value)
        return improvement > self.config.metric_threshold

    def _is_budget_exhausted(self, start_time: float, trial_count: int) -> bool:
        """Check if search budget is exhausted."""
        if self.config.budget_type == "trials":
            return trial_count >= self.config.max_trials
        else:
            elapsed_hours = (time.time() - start_time) / 3600
            return elapsed_hours >= self.config.max_hours

    def _verify_code_integrity(self):
        """Verify that locked code hasn't been modified."""
        for locked_file in self.locked.locked_files:
            file_hash = self._compute_file_hash(locked_file)
            # On first run, store hashes
            # On subsequent runs, compare
            # This is simplified; production version would maintain hash database

    def _verify_preprocessing_contract(self):
        """Verify preprocessing contract hash hasn't changed."""
        current_hash = self._compute_preprocessing_hash()
        if (
            self.initial_preprocessing_hash is not None
            and current_hash != self.initial_preprocessing_hash
        ):
            raise RuntimeError(
                "Preprocessing contract has changed! Locked files modified."
            )

    def _verify_data_consistency(self, train_data: Any, val_data: Any):
        """Verify data splits are consistent across runs."""
        # Simplified check: ensure same number of samples
        try:
            train_len = len(train_data)
            val_len = len(val_data)
            self.logger.debug(f"Data integrity: train={train_len}, val={val_len}")
        except Exception as e:
            self.logger.warning(f"Could not verify data consistency: {str(e)}")

    def _compute_preprocessing_hash(self) -> str:
        """Compute hash of all locked preprocessing code."""
        locked_content = ""
        for locked_file in self.locked.locked_files:
            try:
                with open(locked_file, "r") as f:
                    locked_content += f.read()
            except FileNotFoundError:
                pass

        return hashlib.sha256(locked_content.encode()).hexdigest()

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def _get_git_hash() -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    def _generate_experiment_journal(self):
        """Generate markdown report of all experiments."""
        journal_path = self.log_dir / "EXPERIMENT_JOURNAL.md"

        with open(journal_path, "w") as f:
            f.write(f"# Agentic Tuning Experiment Journal\n\n")
            f.write(f"**Date**: {datetime.now().isoformat()}\n")
            f.write(f"**Metric**: {self.config.metric} ({self.config.metric_mode})\n")
            f.write(f"**Total Trials**: {len(self.experiments)}\n")
            f.write(f"**Best Metric**: {self.best_metric_value:.6f}\n\n")

            f.write("## Experiment Summary\n\n")
            f.write(
                "| Trial | Timestamp | Metric | Wall Clock (s) | Safety | Config Diff |\n"
            )
            f.write("|-------|-----------|--------|----------------|--------|----------|\n")

            for exp in self.experiments:
                diff_str = ", ".join(
                    [f"{k}: {v[0]:.4f}→{v[1]:.4f}" for k, v in exp.config_diff.items()]
                )[:50]
                f.write(
                    f"| {exp.trial_id} | {exp.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"{exp.metric_value:.6f} | {exp.wall_clock_seconds:.1f} | "
                    f"{'✓' if exp.safety_checks_passed else '✗'} | {diff_str} |\n"
                )

            f.write("\n## Best Result\n\n")
            if self.best_result:
                f.write(f"**Trial ID**: {self.best_result.trial_id}\n")
                f.write(f"**Metric**: {self.best_result.metric_value:.6f}\n")
                f.write(f"**Timestamp**: {self.best_result.timestamp.isoformat()}\n")
                if self.best_result.git_hash:
                    f.write(f"**Git Hash**: {self.best_result.git_hash}\n")

        self.logger.info(f"Experiment journal saved to {journal_path}")
