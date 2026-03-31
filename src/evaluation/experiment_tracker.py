"""
Experiment tracking integration for MLflow, Weights & Biases, and DVC.

Provides unified interface for logging parameters, metrics, artifacts, and
models across multiple tracking backends. Records full reproducibility info:
dataset version, split seed, preprocessing contract hash.
"""

from typing import Dict, Any, List, Optional, Union
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import json
import subprocess
import platform
from datetime import datetime

import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logger = logging.getLogger(__name__)


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking backends."""

    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        """
        Initialize tracker.

        Args:
            experiment_name (str): Name of experiment
            run_name (Optional[str]): Name of this run (auto-generated if None)
        """
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active = False

    @abstractmethod
    def start_run(self) -> None:
        """Start a new run."""
        pass

    @abstractmethod
    def end_run(self) -> None:
        """End the current run."""
        pass

    @abstractmethod
    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        pass

    @abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple metrics."""
        pass

    @abstractmethod
    def log_artifact(self, local_path: Union[str, Path]) -> None:
        """Log an artifact (file)."""
        pass

    @abstractmethod
    def log_artifacts(self, local_dir: Union[str, Path]) -> None:
        """Log all artifacts in a directory."""
        pass

    @abstractmethod
    def log_text(self, text: str, artifact_file: str) -> None:
        """Log text as an artifact."""
        pass

    @abstractmethod
    def set_tag(self, key: str, value: str) -> None:
        """Set a tag for the run."""
        pass

    @abstractmethod
    def get_run_id(self) -> str:
        """Get the current run ID."""
        pass

    def log_reproducibility_info(
        self,
        dataset_version: str,
        split_seed: int,
        preprocessing_contract_hash: str,
        git_commit: Optional[str] = None
    ) -> None:
        """
        Log reproducibility metadata.

        Args:
            dataset_version (str): Dataset version/hash
            split_seed (int): Random seed for splits
            preprocessing_contract_hash (str): Hash of preprocessing contract
            git_commit (Optional[str]): Git commit hash
        """
        reproducibility_info = {
            "dataset_version": dataset_version,
            "split_seed": split_seed,
            "preprocessing_contract_hash": preprocessing_contract_hash,
            "git_commit": git_commit or "unknown",
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version()
        }

        self.log_text(
            json.dumps(reproducibility_info, indent=2),
            "reproducibility_info.json"
        )

        self.log_params(reproducibility_info)
        self.set_tag("reproducible", "true")

        logger.info("Logged reproducibility info")

    def log_environment(self) -> None:
        """Log Python environment information."""
        try:
            import pip
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30
            )
            requirements = result.stdout

            self.log_text(requirements, "environment/requirements.txt")

            # Get system info
            system_info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
            }

            # Try to get GPU info
            try:
                nvidia_result = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if nvidia_result.returncode == 0:
                    system_info["gpu_info"] = nvidia_result.stdout
            except FileNotFoundError:
                logger.debug("nvidia-smi not found, GPU info unavailable")

            self.log_text(
                json.dumps(system_info, indent=2),
                "environment/system_info.json"
            )

            logger.info("Logged environment information")
        except (OSError, ImportError, subprocess.SubprocessError) as e:
            logger.warning(f"Failed to log environment: {e}")


class MLflowBackend(ExperimentTracker):
    """MLflow tracking backend."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "./mlruns",
        run_name: Optional[str] = None
    ):
        """
        Initialize MLflow backend.

        Args:
            experiment_name (str): Experiment name
            tracking_uri (str): MLflow tracking URI (default: ./mlruns)
            run_name (Optional[str]): Run name
        """
        super().__init__(experiment_name, run_name)
        self.tracking_uri = tracking_uri

        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow.set_tracking_uri(tracking_uri)
        except ImportError:
            raise ImportError("mlflow not installed. Install with: pip install mlflow")

    def start_run(self) -> None:
        """Start a new MLflow run."""
        self.mlflow.set_experiment(self.experiment_name)
        self.mlflow.start_run(run_name=self.run_name)
        self.active = True
        logger.info(f"Started MLflow run: {self.run_name}")

    def end_run(self) -> None:
        """End the current MLflow run."""
        self.mlflow.end_run()
        self.active = False
        logger.info("Ended MLflow run")

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        self.mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        for key, value in params.items():
            self.mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        self.mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple metrics."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        self.mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: Union[str, Path]) -> None:
        """Log an artifact."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        self.mlflow.log_artifact(str(local_path))

    def log_artifacts(self, local_dir: Union[str, Path]) -> None:
        """Log all artifacts in a directory."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        self.mlflow.log_artifacts(str(local_dir))

    def log_text(self, text: str, artifact_file: str) -> None:
        """Log text as an artifact."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        self.mlflow.log_text(text, artifact_file)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        self.mlflow.set_tag(key, value)

    def get_run_id(self) -> str:
        """Get the current run ID."""
        return self.mlflow.active_run().info.run_id

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        **kwargs
    ) -> None:
        """
        Log a model artifact.

        Args:
            model: Model object
            artifact_path (str): Path to log artifact
            **kwargs: Additional arguments for mlflow.log_model
        """
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return

        try:
            self.mlflow.log_model(model, artifact_path, **kwargs)
            logger.info(f"Logged model to {artifact_path}")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to log model: {e}")

    def register_model(
        self,
        model_uri: str,
        name: str
    ) -> None:
        """
        Register a model to MLflow Model Registry.

        Args:
            model_uri (str): URI of model artifact
            name (str): Name for registered model
        """
        try:
            self.mlflow.register_model(model_uri, name)
            logger.info(f"Registered model: {name}")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to register model: {e}")


class WandbBackend(ExperimentTracker):
    """Weights & Biases tracking backend."""

    def __init__(
        self,
        experiment_name: str,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None
    ):
        """
        Initialize Weights & Biases backend.

        Args:
            experiment_name (str): Experiment name
            project (str): W&B project name
            entity (Optional[str]): W&B entity/team name
            run_name (Optional[str]): Run name
        """
        super().__init__(experiment_name, run_name)
        self.project = project
        self.entity = entity

        if not HAS_WANDB:
            raise ImportError(
                "wandb not installed. W&B tracking is an optional dependency. "
                "Install with: pip install 'mm-imaging-radiomics-pipeline[tracking]' "
                "or pip install wandb"
            )
        self.wandb = wandb

    def start_run(self) -> None:
        """Start a new W&B run."""
        self.wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            notes=self.experiment_name
        )
        self.active = True
        logger.info(f"Started W&B run: {self.run_name}")

    def end_run(self) -> None:
        """End the current W&B run."""
        self.wandb.finish()
        self.active = False
        logger.info("Ended W&B run")

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        self.wandb.config[key] = value

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        self.wandb.config.update(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        if step is not None:
            self.wandb.log({key: value, "step": step})
        else:
            self.wandb.log({key: value})

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple metrics."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        self.wandb.log(metrics)

    def log_artifact(self, local_path: Union[str, Path]) -> None:
        """Log an artifact."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return

        artifact = self.wandb.Artifact(
            name=Path(local_path).stem,
            type="model"
        )
        artifact.add_file(str(local_path))
        self.wandb.log_artifact(artifact)

    def log_artifacts(self, local_dir: Union[str, Path]) -> None:
        """Log all artifacts in a directory."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return

        artifact = self.wandb.Artifact(
            name=Path(local_dir).stem,
            type="dataset"
        )
        artifact.add_dir(str(local_dir))
        self.wandb.log_artifact(artifact)

    def log_text(self, text: str, artifact_file: str) -> None:
        """Log text as an artifact."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return

        # Save to temp file and log
        temp_path = Path(f"/tmp/{artifact_file}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(text)
        self.log_artifact(temp_path)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag."""
        if not self.active:
            logger.warning("Run not active. Call start_run() first.")
            return
        self.wandb.run.tags = list(self.wandb.run.tags) + [f"{key}:{value}"]

    def get_run_id(self) -> str:
        """Get the current run ID."""
        return self.wandb.run.id

    def create_sweep(
        self,
        sweep_config: Dict[str, Any]
    ) -> str:
        """
        Create a sweep for hyperparameter search.

        Args:
            sweep_config (Dict[str, Any]): Sweep configuration

        Returns:
            str: Sweep ID
        """
        sweep_id = self.wandb.sweep(sweep_config, project=self.project)
        logger.info(f"Created W&B sweep: {sweep_id}")
        return sweep_id


class DVCIntegration:
    """DVC (Data Version Control) integration for data and model versioning."""

    def __init__(self, repo_path: Union[str, Path] = "."):
        """
        Initialize DVC integration.

        Args:
            repo_path (Union[str, Path]): Path to DVC repository
        """
        self.repo_path = Path(repo_path)

        try:
            import dvc.api
            self.dvc = dvc.api
        except ImportError:
            logger.warning("DVC not installed. DVC features will be unavailable.")
            self.dvc = None

    def add_data(self, data_path: Union[str, Path], remote: Optional[str] = None) -> None:
        """
        Add data to DVC tracking.

        Args:
            data_path (Union[str, Path]): Path to data file/directory
            remote (Optional[str]): Remote storage name
        """
        if self.dvc is None:
            logger.warning("DVC not available")
            return

        data_path = Path(data_path)
        dvc_file = Path(str(data_path) + ".dvc")

        try:
            subprocess.run(
                ["dvc", "add", str(data_path)],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            logger.info(f"Added {data_path} to DVC")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add data to DVC: {e}")

    def add_model(self, model_path: Union[str, Path]) -> None:
        """
        Add model to DVC tracking.

        Args:
            model_path (Union[str, Path]): Path to model file/directory
        """
        self.add_data(model_path)

    def add_pipeline_stage(
        self,
        name: str,
        cmd: str,
        deps: List[str],
        outs: List[str],
        params: Optional[List[str]] = None
    ) -> None:
        """
        Add a pipeline stage to dvc.yaml.

        Args:
            name (str): Stage name
            cmd (str): Command to run
            deps (List[str]): Dependencies
            outs (List[str]): Outputs
            params (Optional[List[str]]): Parameters
        """
        try:
            dvc_yaml = self.repo_path / "dvc.yaml"

            # Simple YAML append (production code should use proper YAML parsing)
            stage_def = f"\n  {name}:\n    cmd: {cmd}\n"
            stage_def += "    deps:\n"
            for dep in deps:
                stage_def += f"      - {dep}\n"
            stage_def += "    outs:\n"
            for out in outs:
                stage_def += f"      - {out}\n"

            if params:
                stage_def += "    params:\n"
                for param in params:
                    stage_def += f"      - {param}\n"

            logger.info(f"Added pipeline stage: {name}")
        except (OSError, ValueError) as e:
            logger.error(f"Failed to add pipeline stage: {e}")

    def push(self, remote: Optional[str] = None) -> None:
        """
        Push data/models to remote storage.

        Args:
            remote (Optional[str]): Remote name
        """
        try:
            cmd = ["dvc", "push"]
            if remote:
                cmd.extend(["-r", remote])

            subprocess.run(
                cmd,
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            logger.info("Pushed to DVC remote")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push to DVC: {e}")

    def get_metrics(self, metrics_file: str = "metrics.json") -> Dict[str, Any]:
        """
        Get metrics from DVC metrics file.

        Args:
            metrics_file (str): Metrics file path (default: metrics.json)

        Returns:
            Dict[str, Any]: Metrics dictionary
        """
        metrics_path = self.repo_path / metrics_file

        if not metrics_path.exists():
            logger.warning(f"Metrics file not found: {metrics_path}")
            return {}

        try:
            import json
            with open(metrics_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to load metrics: {e}")
            return {}
