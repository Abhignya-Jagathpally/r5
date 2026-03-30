"""Configuration loading and merging utilities."""

import json
import logging
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Args:
        path: Path to config file (.yaml, .yml, or .json)

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If file format is unsupported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p) as f:
        if p.suffix in (".yaml", ".yml"):
            config = yaml.safe_load(f) or {}
        elif p.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {p.suffix} (use .yaml or .json)")

    logger.info(f"Loaded config from {path} ({len(config)} top-level keys)")
    return config


def merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge overrides into base config (overrides win on conflicts).

    Args:
        base: Base configuration dictionary.
        overrides: Override values to merge in.

    Returns:
        Merged configuration (new dict, inputs unchanged).
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_stage_configs(
    master_config: Dict[str, Any], config_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Resolve per-stage configs by loading dedicated files and merging with master.

    Mapping:
        preprocessing stages → data_pipeline.yaml/json
        baselines            → model_baselines.yaml/json
        foundation           → foundation_models.yaml/json
        evaluation/report    → evaluation.yaml/json

    Args:
        master_config: Loaded master pipeline config.
        config_dir: Directory containing config files.

    Returns:
        Dict mapping stage name to its resolved config.
    """
    stage_file_map = {
        "preprocessing": "data_pipeline",
        "baselines": "model_baselines",
        "foundation": "foundation_models",
        "fusion": "foundation_models",
        "evaluation": "evaluation",
        "report": "evaluation",
    }

    resolved = {}
    for stage, config_name in stage_file_map.items():
        # Try JSON first, then YAML
        for ext in (".json", ".yaml", ".yml"):
            candidate = config_dir / f"{config_name}{ext}"
            if candidate.exists():
                stage_config = load_config(str(candidate))
                # Merge any inline overrides from master config
                inline = master_config.get(stage, {})
                if isinstance(inline, dict):
                    stage_config = merge_configs(stage_config, inline)
                resolved[stage] = stage_config
                break
        else:
            # No dedicated file — use master config inline section
            resolved[stage] = master_config.get(stage, {})

    return resolved


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
    """Validate that required keys exist in config.

    Returns:
        List of missing keys (empty if valid).
    """
    missing = [k for k in required_keys if k not in config]
    if missing:
        logger.warning(f"Missing config keys: {missing}")
    return missing
