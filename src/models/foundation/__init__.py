"""
Foundation model integration module for pathology imaging.

Provides unified interfaces for:
- UNI2-h: Pathology vision foundation model (ViT-H/14)
- TITAN: Multimodal whole-slide foundation model
- Feature extraction and caching
- MIL (Multiple Instance Learning) heads
- Multimodal fusion strategies
- Clinical explainability
"""

from .feature_extractor import (
    FeatureExtractor,
    ResNet50ImageNet,
    UNI2H,
    TITAN,
    get_extractor,
)
from .mil_heads import TransMIL, DTFDMIL, HIRCLSMILHead
from .multimodal_fusion import (
    EarlyFusion,
    LateFusion,
    CrossAttentionFusion,
    GatedFusion,
    get_fusion_module,
)
from .explainability import AttentionExplainer

__version__ = "0.1.0"
__all__ = [
    "FeatureExtractor",
    "ResNet50ImageNet",
    "UNI2H",
    "TITAN",
    "get_extractor",
    "TransMIL",
    "DTFDMIL",
    "HIRCLSMILHead",
    "EarlyFusion",
    "LateFusion",
    "CrossAttentionFusion",
    "GatedFusion",
    "get_fusion_module",
    "AttentionExplainer",
]
