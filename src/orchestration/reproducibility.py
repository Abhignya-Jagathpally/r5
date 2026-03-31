"""
Reproducibility infrastructure for biomedical ML pipelines.

Provides:
- Environment snapshots (Python packages, system info, GPU details)
- Dockerfile/Singularity generation from current environment
- DVC pipeline definition auto-generation
- Git integration for experiment versioning
- Seed management for reproducibility
"""

import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentSnapshot:
    """Snapshot of the execution environment."""

    timestamp: str
    python_version: str
    platform_info: Dict[str, str]
    packages: Dict[str, str]
    gpu_info: Dict[str, Any]
    cuda_version: Optional[str] = None
    git_hash: Optional[str] = None
    git_branch: Optional[str] = None
    working_directory: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: Path):
        """Save snapshot to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Environment snapshot saved to {path}")

    @classmethod
    def create(cls) -> "EnvironmentSnapshot":
        """Create environment snapshot of current system."""
        return cls(
            timestamp=datetime.now().isoformat(),
            python_version=platform.python_version(),
            platform_info={
                "system": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            packages=cls._get_installed_packages(),
            gpu_info=cls._get_gpu_info(),
            cuda_version=cls._get_cuda_version(),
            git_hash=cls._get_git_hash(),
            git_branch=cls._get_git_branch(),
            working_directory=str(Path.cwd()),
        )

    @staticmethod
    def _get_installed_packages() -> Dict[str, str]:
        """Get list of installed Python packages."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
            )
            packages = {}
            for line in result.stdout.strip().split("\n"):
                if "==" in line:
                    name, version = line.split("==")
                    packages[name] = version
            return packages
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            logger.error(f"Failed to get installed packages: {str(e)}")
            return {}

    @staticmethod
    def _get_gpu_info() -> Dict[str, Any]:
        """Get GPU information."""
        try:
            import torch

            return {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "devices": [
                    {
                        "name": torch.cuda.get_device_name(i),
                        "capability": torch.cuda.get_device_capability(i),
                    }
                    for i in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else [],
            }
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Could not get GPU info: {str(e)}")
            return {"available": False}

    @staticmethod
    def _get_cuda_version() -> Optional[str]:
        """Get CUDA version."""
        try:
            import torch

            return torch.version.cuda
        except (ImportError, AttributeError):
            return None

    @staticmethod
    def _get_git_hash() -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path.cwd(),
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _get_git_branch() -> Optional[str]:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path.cwd(),
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None


class DockerfileGenerator:
    """Generate Dockerfile from current environment."""

    BASE_IMAGE = "nvidia/cuda:12.1.1-runtime-ubuntu22.04"

    def __init__(self, snapshot: EnvironmentSnapshot):
        """
        Initialize Dockerfile generator.

        Args:
            snapshot: EnvironmentSnapshot instance
        """
        self.snapshot = snapshot
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate(self, output_path: Path, source_dir: Optional[Path] = None) -> str:
        """
        Generate Dockerfile content.

        Args:
            output_path: Path to save Dockerfile
            source_dir: Optional source directory to include

        Returns:
            Dockerfile content as string
        """
        dockerfile = f"""# Generated Dockerfile from environment snapshot
# Created: {self.snapshot.timestamp}
# Python: {self.snapshot.python_version}
# Git Hash: {self.snapshot.git_hash}

FROM {self.BASE_IMAGE}

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Python packages
RUN pip install {' '.join([f'{pkg}=={ver}' for pkg, ver in self.snapshot.packages.items()])}

"""

        if source_dir:
            dockerfile += f"""
# Copy source code
WORKDIR /app
COPY {source_dir}/ /app/

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.orchestration"]
"""

        # Save Dockerfile
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(dockerfile)

        self.logger.info(f"Dockerfile generated: {output_path}")
        return dockerfile


class SingularityGenerator:
    """Generate Singularity/Apptainer definition file."""

    BASE_IMAGE = "docker://nvidia/cuda:12.1.1-runtime-ubuntu22.04"

    def __init__(self, snapshot: EnvironmentSnapshot):
        """
        Initialize Singularity generator.

        Args:
            snapshot: EnvironmentSnapshot instance
        """
        self.snapshot = snapshot
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate(
        self, output_path: Path, source_dir: Optional[Path] = None
    ) -> str:
        """
        Generate Singularity definition file.

        Args:
            output_path: Path to save definition file
            source_dir: Optional source directory to include

        Returns:
            Definition file content as string
        """
        deffile = f"""# Generated Singularity definition from environment snapshot
# Created: {self.snapshot.timestamp}
# Python: {self.snapshot.python_version}
# Git Hash: {self.snapshot.git_hash}

Bootstrap: docker
From: {self.BASE_IMAGE}

%post
    # Update and install dependencies
    apt-get update
    apt-get install -y python3.10 python3-pip git build-essential
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    # Install Python packages
    pip install {' '.join([f'{pkg}=={ver}' for pkg, ver in self.snapshot.packages.items()])}

%environment
    export PATH=/usr/local/bin:$PATH
    export PYTHONUNBUFFERED=1

%runscript
    exec python "$@"

%labels
    Author PhD Researcher 6
    Version {self.snapshot.timestamp}
    GitHash {self.snapshot.git_hash}
"""

        if source_dir:
            deffile += f"""
%files
    {source_dir}/ /app/

%workdir /app
"""

        # Save definition file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(deffile)

        self.logger.info(f"Singularity definition generated: {output_path}")
        return deffile


class DVCPipelineGenerator:
    """Generate DVC pipeline definition (dvc.yaml) from pipeline specification."""

    def __init__(self):
        """Initialize DVC pipeline generator."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stages: Dict[str, Dict[str, Any]] = {}

    def add_stage(
        self,
        name: str,
        cmd: str,
        deps: Optional[List[str]] = None,
        outs: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ):
        """
        Add a stage to the pipeline.

        Args:
            name: Stage name
            cmd: Command to execute
            deps: List of dependencies
            outs: List of outputs
            metrics: List of metric files
        """
        self.stages[name] = {
            "cmd": cmd,
            "deps": deps or [],
            "outs": outs or [],
            "metrics": metrics or [],
        }

    def generate(self, output_path: Path) -> str:
        """
        Generate dvc.yaml.

        Args:
            output_path: Path to save dvc.yaml

        Returns:
            YAML content as string
        """
        import yaml

        dvc_config = {"stages": self.stages}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(dvc_config, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"DVC pipeline generated: {output_path}")

        with open(output_path, "r") as f:
            content = f.read()
        return content


class ExperimentJournal:
    """Structured experiment journal for tracking all runs."""

    def __init__(self, log_dir: Path):
        """
        Initialize experiment journal.

        Args:
            log_dir: Directory for saving journal files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.entries: List[Dict[str, Any]] = []

    def add_entry(
        self,
        experiment_id: str,
        model_type: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        git_hash: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """
        Add entry to journal.

        Args:
            experiment_id: Unique experiment identifier
            model_type: Type of model trained
            config: Configuration dictionary
            metrics: Dictionary of metric results
            git_hash: Git commit hash
            notes: Optional notes about experiment
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": experiment_id,
            "model_type": model_type,
            "config": config,
            "metrics": metrics,
            "git_hash": git_hash,
            "notes": notes,
        }
        self.entries.append(entry)

    def save(self):
        """Save journal to disk."""
        import json

        journal_path = self.log_dir / "experiment_journal.json"
        with open(journal_path, "w") as f:
            json.dump(self.entries, f, indent=2, default=str)
        self.logger.info(f"Experiment journal saved: {journal_path}")

    def generate_markdown_report(self) -> str:
        """
        Generate markdown report from journal entries.

        Returns:
            Markdown formatted report
        """
        report = "# Experiment Journal\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n"
        report += f"Total Experiments: {len(self.entries)}\n\n"

        report += "## Experiments\n\n"
        for entry in self.entries:
            report += f"### {entry['experiment_id']}\n\n"
            report += f"- **Timestamp**: {entry['timestamp']}\n"
            report += f"- **Model Type**: {entry['model_type']}\n"
            report += f"- **Git Hash**: {entry['git_hash']}\n"
            report += f"- **Metrics**: {entry['metrics']}\n"
            if entry['notes']:
                report += f"- **Notes**: {entry['notes']}\n"
            report += "\n"

        return report


class SeedManager:
    """Manage random seeds for reproducibility."""

    @staticmethod
    def set_seed(seed: int = 42):
        """
        Set random seed for all libraries.

        Args:
            seed: Random seed value
        """
        import random

        import numpy as np

        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        np.random.seed(seed)
        random.seed(seed)

        logger.info(f"Random seed set to {seed}")

    @staticmethod
    def get_seed_config(seed: int = 42) -> Dict[str, Any]:
        """
        Get configuration dictionary with seed settings.

        Args:
            seed: Random seed value

        Returns:
            Configuration dictionary
        """
        return {
            "seed": seed,
            "numpy_seed": seed,
            "torch_seed": seed,
            "torch_cuda_seed": seed,
            "python_random_seed": seed,
        }
