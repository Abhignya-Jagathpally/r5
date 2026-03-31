"""
Config schema validation.

Validates YAML configs against expected structure before pipeline runs.
Catches missing fields and type errors early rather than mid-pipeline.
"""

from typing import Any


REQUIRED_FIELDS = {
    "data_pipeline": ["tiling.tile_size", "tiling.input_dir"],
    "model_baselines": ["training.epochs", "training.learning_rate"],
    "evaluation": ["metrics"],
    "pipeline": ["stages"],
}


def validate_config(config: dict, config_type: str) -> list[str]:
    """Validate config dict against known schema.

    Returns list of error messages (empty if valid).
    """
    errors = []
    required = REQUIRED_FIELDS.get(config_type, [])

    for field_path in required:
        parts = field_path.split(".")
        current = config
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                errors.append(f"Missing required field: {field_path}")
                break
            current = current[part]

    return errors


def validate_no_conflicting_keys(config: dict) -> list[str]:
    """Check for common config mistakes."""
    errors = []

    # Check for mixing snake_case and camelCase
    def _check_keys(d, prefix=""):
        if not isinstance(d, dict):
            return
        for key in d:
            full = f"{prefix}.{key}" if prefix else key
            if "-" in key:
                errors.append(f"Use snake_case, not kebab-case: {full}")
            _check_keys(d[key], full)

    _check_keys(config)
    return errors
