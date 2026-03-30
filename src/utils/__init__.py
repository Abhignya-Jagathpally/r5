"""Utilities: checkpoint management, config loading, helpers."""
from .config import load_config, merge_configs


def __getattr__(name):
    """Lazy import CheckpointManager to avoid torch dependency at import time."""
    if name == "CheckpointManager":
        from .checkpoint_manager import CheckpointManager
        return CheckpointManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CheckpointManager", "load_config", "merge_configs"]
