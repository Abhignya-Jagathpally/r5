"""
Multiple Myeloma Imaging Pathology & Radiomics Surrogate-Genetics Pipeline.

Classical baselines module with classical baseline first approach:
1. Tile-level ResNet50 classifier
2. Mean pooling baseline
3. Attention-Based Multiple Instance Learning (ABMIL)
4. Clustering-constrained Attention MIL (CLAM)
5. Radiomics + Survival analysis
"""

from src.models.tile_classifier import TileClassifier
from src.models.mean_pool_baseline import MeanPoolBaseline
from src.models.abmil import ABMIL
from src.models.clam import CLAM_SB, CLAM_MB
from src.models.radiomics_survival import CoxProportionalHazards, RandomSurvivalForest
from src.models.mil_dataset import MILDataset, create_mil_dataloader
from src.models.losses import (
    CrossEntropyLoss,
    CoxPartialLikelihoodLoss,
    InstanceClusteringLoss,
    FocalLoss,
    SmoothTopKLoss,
)

__all__ = [
    "TileClassifier",
    "MeanPoolBaseline",
    "ABMIL",
    "CLAM_SB",
    "CLAM_MB",
    "CoxProportionalHazards",
    "RandomSurvivalForest",
    "MILDataset",
    "create_mil_dataloader",
    "CrossEntropyLoss",
    "CoxPartialLikelihoodLoss",
    "InstanceClusteringLoss",
    "FocalLoss",
    "SmoothTopKLoss",
]
