"""
Evaluation framework for Multiple Myeloma Imaging Pathology & Radiomics Pipeline.

This module provides:
- Patient-level data splitting with no tile/patch leakage
- Comprehensive metrics computation with confidence intervals
- Frozen preprocessing contracts for reproducibility
- Experiment tracking integration (MLflow, W&B, DVC)
- Publication-quality visualization
- Automated evaluation reporting

Strict clinical ML standards:
- All splits are at the patient level (no leakage)
- Train-only fitting of normalization and imputation
- Full reproducibility via frozen preprocessing contracts
- Statistical testing with proper error handling
"""

from .splitter import (
    PatientLevelSplitter,
    StratifiedPatientSplit,
    TimeAwareSplitter,
    KFoldPatientSplit,
)
from .metrics import (
    ClassificationMetrics,
    SurvivalMetrics,
    CalibrationMetrics,
    BootstrapMetrics,
)
from .preprocessing_contract import PreprocessingContract
from .experiment_tracker import ExperimentTracker, MLflowBackend, WandbBackend
from .visualization import (
    ROCCurveVisualizer,
    PrecisionRecallVisualizer,
    SurvivalCurveVisualizer,
    ConfusionMatrixVisualizer,
    CalibrationVisualizer,
    TrainingCurveVisualizer,
)
from .report_generator import EvaluationReportGenerator

__all__ = [
    # Splitters
    "PatientLevelSplitter",
    "StratifiedPatientSplit",
    "TimeAwareSplitter",
    "KFoldPatientSplit",
    # Metrics
    "ClassificationMetrics",
    "SurvivalMetrics",
    "CalibrationMetrics",
    "BootstrapMetrics",
    # Preprocessing
    "PreprocessingContract",
    # Tracking
    "ExperimentTracker",
    "MLflowBackend",
    "WandbBackend",
    # Visualization
    "ROCCurveVisualizer",
    "PrecisionRecallVisualizer",
    "SurvivalCurveVisualizer",
    "ConfusionMatrixVisualizer",
    "CalibrationVisualizer",
    "TrainingCurveVisualizer",
    # Reporting
    "EvaluationReportGenerator",
]

__version__ = "1.0.0"
