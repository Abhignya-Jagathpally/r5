"""
MM Imaging Pathology & Radiomics Surrogate-Genetics Pipeline.

Subpackages:
    src.data          — WSI tiling, stain normalization, dedup, embeddings, radiomics
    src.models        — Classical baselines (ABMIL, CLAM, tile classifier, survival)
    src.models.foundation — Foundation models (UNI2-h, TITAN), MIL heads, fusion
    src.evaluation    — Patient-level splits, metrics, tracking, visualization, reporting
    src.orchestration — Parallel processing, HPO, agentic tuning, reproducibility
    src.utils         — Checkpoint management, config loading
"""

__version__ = "0.1.0"
