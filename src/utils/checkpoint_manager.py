"""Checkpoint management with full traceability metadata.

Saves model checkpoints augmented with git hash, config hash, environment
info, and training metrics. Supports automatic pruning, best-model tracking,
and integrity verification.
"""

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints with traceability metadata.

    Args:
        checkpoint_dir: Directory to store checkpoints.
        experiment_id: Unique identifier for this experiment run.
        max_keep: Maximum checkpoints to retain (best is always kept).
        monitor_metric: Metric name to track for best model.
        mode: 'min' or 'max' — whether lower or higher metric is better.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        experiment_id: str,
        max_keep: int = 5,
        monitor_metric: str = "val_loss",
        mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.max_keep = max_keep
        self.monitor_metric = monitor_metric
        self.mode = mode
        self._best_value = float("inf") if mode == "min" else float("-inf")
        self._saved_paths: List[Path] = []

    def save(
        self,
        model,
        optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save checkpoint with full traceability metadata.

        Returns:
            Path to saved checkpoint file.
        """
        filename = f"{self.experiment_id}_epoch{epoch:04d}.pt"
        path = self.checkpoint_dir / filename

        env_info = {"python": sys.version}
        try:
            import torch as _torch
            env_info["torch"] = _torch.__version__
            env_info["cuda"] = _torch.cuda.is_available()
            env_info["cuda_version"] = _torch.version.cuda if _torch.cuda.is_available() else None
        except ImportError:
            env_info["torch"] = None
            env_info["cuda"] = False

        checkpoint = {
            # Model state
            "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else str(model),
            "optimizer_state_dict": optimizer.state_dict() if hasattr(optimizer, "state_dict") else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler and hasattr(scheduler, "state_dict") else None,
            "epoch": epoch,
            "metrics": metrics,
            # Traceability
            "config": config,
            "config_hash": self._config_hash(config),
            "git_hash": self._get_git_hash(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": env_info,
            "experiment_id": self.experiment_id,
        }
        if extra_state:
            checkpoint["extra_state"] = extra_state

        try:
            import torch as _torch
            _torch.save(checkpoint, path)
        except ImportError:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(checkpoint, f)
        self._saved_paths.append(path)
        logger.info(f"Checkpoint saved: {path} (epoch {epoch}, {self.monitor_metric}={metrics.get(self.monitor_metric, 'N/A')})")

        # Update best symlink
        current_value = metrics.get(self.monitor_metric)
        if current_value is not None and self._is_improvement(current_value):
            self._best_value = current_value
            best_link = self.checkpoint_dir / "best.pt"
            if best_link.is_symlink() or best_link.exists():
                best_link.unlink()
            best_link.symlink_to(path.name)
            logger.info(f"New best: {self.monitor_metric}={current_value:.4f}")

        # Prune old checkpoints
        self._prune()

        return path

    def load(
        self,
        checkpoint_path: str,
        model,
        optimizer=None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Load checkpoint and restore model/optimizer/scheduler state.

        Returns:
            Metadata dict (epoch, metrics, config, git_hash, etc.).
        """
        path = Path(checkpoint_path)
        try:
            import torch as _torch
            checkpoint = _torch.load(path, map_location=device, weights_only=False)
        except ImportError:
            import pickle
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        meta = {
            k: v for k, v in checkpoint.items()
            if k not in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict")
        }
        logger.info(
            f"Loaded checkpoint: {path} "
            f"(epoch {meta.get('epoch')}, git={meta.get('git_hash', '?')[:8]})"
        )
        return meta

    def find_latest(self) -> Optional[Path]:
        """Find the most recent checkpoint for this experiment."""
        pattern = f"{self.experiment_id}_epoch*.pt"
        candidates = sorted(self.checkpoint_dir.glob(pattern))
        return candidates[-1] if candidates else None

    def find_best(self) -> Optional[Path]:
        """Find the best checkpoint (via symlink or scanning)."""
        best_link = self.checkpoint_dir / "best.pt"
        if best_link.exists():
            return best_link.resolve()
        return self.find_latest()

    def get_manifest(self) -> List[Dict[str, Any]]:
        """Return metadata for all checkpoints in this experiment."""
        pattern = f"{self.experiment_id}_epoch*.pt"
        manifest = []
        for path in sorted(self.checkpoint_dir.glob(pattern)):
            try:
                try:
                    import torch as _torch
                    ckpt = _torch.load(path, map_location="cpu", weights_only=False)
                except ImportError:
                    import pickle
                    with open(path, "rb") as f:
                        ckpt = pickle.load(f)
                manifest.append({
                    "path": str(path),
                    "epoch": ckpt.get("epoch"),
                    "metrics": ckpt.get("metrics", {}),
                    "config_hash": ckpt.get("config_hash"),
                    "git_hash": ckpt.get("git_hash"),
                    "timestamp": ckpt.get("timestamp"),
                })
            except Exception as e:
                manifest.append({"path": str(path), "error": str(e)})
        return manifest

    def verify_integrity(self, checkpoint_path: str) -> bool:
        """Verify a checkpoint file is loadable and has required keys."""
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            required = ["model_state_dict", "epoch", "config_hash", "timestamp"]
            missing = [k for k in required if k not in ckpt]
            if missing:
                logger.warning(f"Checkpoint missing keys: {missing}")
                return False
            return True
        except Exception as e:
            logger.error(f"Checkpoint integrity check failed: {e}")
            return False

    # ── Private helpers ──────────────────────────────────────────────

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self._best_value
        return value > self._best_value

    def _prune(self) -> None:
        """Remove old checkpoints beyond max_keep (always keep best)."""
        pattern = f"{self.experiment_id}_epoch*.pt"
        candidates = sorted(self.checkpoint_dir.glob(pattern))

        best_link = self.checkpoint_dir / "best.pt"
        best_target = best_link.resolve() if best_link.exists() else None

        if len(candidates) <= self.max_keep:
            return

        to_remove = candidates[: len(candidates) - self.max_keep]
        for path in to_remove:
            if best_target and path.resolve() == best_target:
                continue  # Never remove best
            path.unlink()
            logger.debug(f"Pruned checkpoint: {path}")

    @staticmethod
    def _config_hash(config: Dict[str, Any]) -> str:
        """SHA-256 hash of JSON-serialized config for change detection."""
        serialized = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    @staticmethod
    def _get_git_hash() -> Optional[str]:
        """Get current git commit hash, or None if not in a repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
