"""
Orchestration layer for MM imaging radiomics pipeline.

This module provides workflow orchestration, hyperparameter search,
agentic tuning, and reproducibility infrastructure for the
Multiple Myeloma Imaging Pathology & Radiomics Pipeline.

Key components:
- parallel_features.py: Ray/Dask parallel feature generation
- hyperparameter_search.py: Ray Tune-based hyperparameter optimization
- agentic_tuner.py: Autoresearch-pattern agentic pipeline tuning
- reproducibility.py: Environment snapshots and experiment tracking
"""

__version__ = "0.1.0"
__author__ = "PhD Researcher 6 - Imaging Pathology & Radiomics"

from .parallel_features import (
    RayTileProcessor,
    DaskRadiomicsExtractor,
)
from .hyperparameter_search import (
    HyperparameterSearchConfig,
    HyperparameterSearcher,
)
from .agentic_tuner import (
    AgenticTuner,
    EditableSurface,
    LockedSurface,
)
from .reproducibility import (
    EnvironmentSnapshot,
    ExperimentJournal,
)

__all__ = [
    "RayTileProcessor",
    "DaskRadiomicsExtractor",
    "HyperparameterSearchConfig",
    "HyperparameterSearcher",
    "AgenticTuner",
    "EditableSurface",
    "LockedSurface",
    "EnvironmentSnapshot",
    "ExperimentJournal",
]
