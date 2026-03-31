"""
Snakemake workflow for MM Imaging Pathology & Radiomics Pipeline.

This is the primary orchestration file that defines all pipeline stages:
1. Tile WSIs (tile_wsis)
2. Stain normalization (normalize_tiles)
3. Deduplication (deduplicate)
4. Embedding extraction (extract_embeddings)
5. Radiomics extraction (extract_radiomics)
6. Data split creation (create_splits)
7. Baseline model training (train_baseline_*)
8. Foundation model training (train_foundation_*)
9. Multimodal fusion (train_fusion)
10. Evaluation (evaluate)
11. Report generation (generate_report)

Configuration:
- Config file: configs/pipeline.yaml
- Resources: GPU, memory, thread allocation per rule
- Conda environments: Per-rule environments
- Wildcards: Model type variants
"""

import os
from pathlib import Path

# Configuration
configfile: "configs/pipeline.yaml"

# Define output directories
RESULTS_DIR = Path(config.get("output_dir", "results"))
DATA_DIR = Path(config.get("data_dir", "data"))
MODELS_DIR = RESULTS_DIR / "models"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
RADIOMICS_DIR = RESULTS_DIR / "radiomics"
SPLITS_DIR = RESULTS_DIR / "splits"
EVAL_DIR = RESULTS_DIR / "evaluation"

# Wildcard constraints
BASELINE_MODELS = config.get("baseline_models", ["abmil", "clam", "transmil"])
FOUNDATION_MODELS = config.get("foundation_models", ["uni", "gigapath"])

# Ensure directories exist
for d in [RESULTS_DIR, MODELS_DIR, EMBEDDINGS_DIR, RADIOMICS_DIR, SPLITS_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Pipeline stages configuration
STAGES = config.get("stages", {})


rule all:
    """
    Final rule defining all outputs.

    Specifies what the pipeline should produce end-to-end.
    """
    input:
        report=EVAL_DIR / "final_report.html",
        journal=EVAL_DIR / "evaluation_journal.md",


# ============================================================================
# RULE 1: Tile WSIs
# ============================================================================
if STAGES.get("tiling", True):
    rule tile_wsis:
        """
        Extract tiles from whole slide images.

        Input: Raw WSI files
        Output: Tile arrays per WSI
        """
        input:
            wsi_dir=DATA_DIR / "wsis",
        output:
            tiles=RESULTS_DIR / "tiles" / "{patient_id}.npz",
        params:
            tile_size=config.get("tile_size", 256),
            overlap=config.get("tile_overlap", 0),
        resources:
            mem_mb=32000,
            threads=4,
            time_min=120,
        shell:
            """
            python scripts/run_preprocessing.py \
                --config configs/data_pipeline.yaml \
                --slides {input.wsi_dir}/{wildcards.patient_id} \
                --output-dir $(dirname {output.tiles}) \
                --steps tiling
            """


# ============================================================================
# RULE 2: Normalize Tiles
# ============================================================================
if STAGES.get("stain_norm", True):
    rule normalize_tiles:
        """
        Apply stain normalization to tiles.

        Uses Macenko normalization or similar.
        """
        input:
            tiles=RESULTS_DIR / "tiles" / "{patient_id}.npz",
        output:
            normalized=RESULTS_DIR / "normalized_tiles" / "{patient_id}.npz",
        params:
            method=config.get("stain_norm_method", "macenko"),
        resources:
            mem_mb=16000,
            threads=2,
        shell:
            """
            python scripts/run_preprocessing.py \
                --config configs/data_pipeline.yaml \
                --slides {input.tiles} \
                --output-dir $(dirname {output.normalized}) \
                --steps stain_norm
            """


# ============================================================================
# RULE 3: Deduplicate Tiles
# ============================================================================
if STAGES.get("dedup", True):
    rule deduplicate:
        """
        Remove near-duplicate tiles to reduce redundancy.

        Uses perceptual hashing or SSIM.
        """
        input:
            normalized=RESULTS_DIR / "normalized_tiles" / "{patient_id}.npz",
        output:
            deduplicated=RESULTS_DIR / "deduplicated_tiles" / "{patient_id}.npz",
            stats=RESULTS_DIR / "dedup_stats" / "{patient_id}.json",
        params:
            threshold=config.get("dedup_threshold", 0.95),
        resources:
            mem_mb=8000,
            threads=2,
        shell:
            """
            python scripts/run_preprocessing.py \
                --config configs/data_pipeline.yaml \
                --slides {input.normalized} \
                --output-dir $(dirname {output.deduplicated}) \
                --steps dedup
            """


# ============================================================================
# RULE 4: Extract Embeddings
# ============================================================================
if STAGES.get("embeddings", True):
    rule extract_embeddings:
        """
        Extract patch embeddings using foundation model.

        Configurable backbone (UNI, GigaPath, others).
        Uses Ray parallel processing for multiple GPUs.
        """
        input:
            tiles=RESULTS_DIR / "deduplicated_tiles" / "{patient_id}.npz",
        output:
            embeddings=EMBEDDINGS_DIR / "{backbone}" / "{patient_id}.h5",
        params:
            backbone=config.get("embedding_backbone", "uni"),
            batch_size=config.get("embedding_batch_size", 32),
            checkpoint=config.get("embedding_checkpoint", ""),
        resources:
            mem_mb=40000,
            gpus=1,
            threads=8,
            time_min=60,
        shell:
            """
            python scripts/run_preprocessing.py \
                --config configs/data_pipeline.yaml \
                --slides {input.tiles} \
                --output-dir $(dirname {output.embeddings}) \
                --steps embeddings
            """


# ============================================================================
# RULE 5: Extract Radiomics
# ============================================================================
if STAGES.get("radiomics", True):
    rule extract_radiomics:
        """
        Extract radiomics features from CT/PET images.

        Uses pyradiomics with configurable settings.
        Parallel processing via Dask.
        """
        input:
            image=DATA_DIR / "imaging" / "{patient_id}" / "image.nii.gz",
            mask=DATA_DIR / "imaging" / "{patient_id}" / "mask.nii.gz",
        output:
            features=RADIOMICS_DIR / "{patient_id}.csv",
        params:
            features_to_extract=config.get("radiomics_features", ["shape", "glcm", "glrlm"]),
        resources:
            mem_mb=16000,
            threads=4,
            time_min=30,
        shell:
            """
            python scripts/run_preprocessing.py \
                --config configs/data_pipeline.yaml \
                --slides {input.image} \
                --output-dir $(dirname {output.features}) \
                --steps radiomics
            """


# ============================================================================
# RULE 6: Create Data Splits
# ============================================================================
if STAGES.get("create_splits", True):
    rule create_splits:
        """
        Generate patient-level train/val/test splits.

        Stratified by outcome/site to ensure balanced representation.
        """
        input:
            metadata=DATA_DIR / "metadata.csv",
        output:
            train_split=SPLITS_DIR / "train_split.json",
            val_split=SPLITS_DIR / "val_split.json",
            test_split=SPLITS_DIR / "test_split.json",
            split_info=SPLITS_DIR / "split_info.json",
        params:
            train_ratio=config.get("train_ratio", 0.7),
            val_ratio=config.get("val_ratio", 0.15),
            test_ratio=config.get("test_ratio", 0.15),
            stratify_by=config.get("stratify_by", "outcome"),
            seed=config.get("seed", 42),
        resources:
            mem_mb=2000,
            threads=1,
        shell:
            """
            python -m src.data.create_splits \
                --metadata {input.metadata} \
                --output_dir {SPLITS_DIR} \
                --train_ratio {params.train_ratio} \
                --val_ratio {params.val_ratio} \
                --test_ratio {params.test_ratio} \
                --stratify_by {params.stratify_by} \
                --seed {params.seed}
            """


# ============================================================================
# RULE 7: Train Baseline Models
# ============================================================================
rule train_baseline:
    """
    Train baseline MIL models.

    Wildcard: {model} in [abmil, clam, transmil, dsmil]
    Each baseline trained on pathology (embeddings) only.
    """
    input:
        embeddings=expand(
            EMBEDDINGS_DIR / "{backbone}" / "{{patient_id}}.h5",
            backbone=config.get("embedding_backbone", "uni"),
        ),
        train_split=SPLITS_DIR / "train_split.json",
        val_split=SPLITS_DIR / "val_split.json",
    output:
        model=MODELS_DIR / "baseline_{model}" / "best_model.pth",
        metrics=MODELS_DIR / "baseline_{model}" / "metrics.json",
        config=MODELS_DIR / "baseline_{model}" / "config.yaml",
    params:
        model_type="{model}",
        num_epochs=config.get("num_epochs", 50),
        learning_rate=config.get("learning_rate", 1e-4),
        batch_size=config.get("batch_size", 32),
    resources:
        mem_mb=32000,
        gpus=1,
        threads=8,
        time_min=180,
    shell:
        """
        python scripts/train_baselines.py \
            --config configs/model_baselines.yaml \
            --model {params.model_type} \
            --output_dir $(dirname {output.model})
        """


# ============================================================================
# RULE 8: Train Foundation Models
# ============================================================================
rule train_foundation:
    """
    Train foundation model variants.

    Wildcard: {foundation_model} in [uni_pretrained, gigapath_pretrained, ...]
    """
    input:
        embeddings=expand(
            EMBEDDINGS_DIR / "{backbone}" / "{{patient_id}}.h5",
            backbone=config.get("embedding_backbone", "uni"),
        ),
        train_split=SPLITS_DIR / "train_split.json",
        val_split=SPLITS_DIR / "val_split.json",
    output:
        model=MODELS_DIR / "{foundation_model}" / "best_model.pth",
        metrics=MODELS_DIR / "{foundation_model}" / "metrics.json",
    params:
        foundation_type="{foundation_model}",
        num_epochs=config.get("num_epochs", 50),
        learning_rate=config.get("learning_rate", 1e-4),
    resources:
        mem_mb=40000,
        gpus=1,
        threads=8,
        time_min=240,
    shell:
        """
        python scripts/extract_foundation_features.py \
            --input-dir {EMBEDDINGS_DIR} \
            --output-dir $(dirname {output.model}) \
            --backbone {params.foundation_type}
        """


# ============================================================================
# RULE 9: Train Fusion Models
# ============================================================================
if STAGES.get("fusion", True):
    rule train_fusion:
        """
        Train multimodal fusion model (pathology + radiomics).

        Integrates embeddings and radiomics features.
        """
        input:
            embeddings=expand(
                EMBEDDINGS_DIR / "{backbone}" / "{{patient_id}}.h5",
                backbone=config.get("embedding_backbone", "uni"),
            ),
            radiomics=expand(RADIOMICS_DIR / "{patient_id}.csv", patient_id="*"),
            train_split=SPLITS_DIR / "train_split.json",
            val_split=SPLITS_DIR / "val_split.json",
        output:
            model=MODELS_DIR / "fusion" / "best_model.pth",
            metrics=MODELS_DIR / "fusion" / "metrics.json",
        params:
            fusion_method=config.get("fusion_method", "concat"),
            pathology_weight=config.get("pathology_weight", 0.5),
            radiomics_weight=config.get("radiomics_weight", 0.5),
            num_epochs=config.get("num_epochs", 50),
        resources:
            mem_mb=40000,
            gpus=1,
            threads=8,
            time_min=240,
        shell:
            """
            python main.py --config configs/pipeline.json --stages fusion \
                --embeddings_dir {EMBEDDINGS_DIR} \
                --radiomics_dir {RADIOMICS_DIR} \
                --splits_dir {SPLITS_DIR} \
                --output_dir $(dirname {output.model}) \
                --fusion_method {params.fusion_method} \
                --pathology_weight {params.pathology_weight} \
                --radiomics_weight {params.radiomics_weight} \
                --num_epochs {params.num_epochs}
            """


# ============================================================================
# RULE 10: Evaluate
# ============================================================================
if STAGES.get("evaluation", True):
    rule evaluate:
        """
        Run comprehensive evaluation on test set.

        Metrics: AUROC, AUPR, accuracy, sensitivity, specificity, etc.
        """
        input:
            baseline_models=expand(
                MODELS_DIR / "baseline_{model}" / "best_model.pth",
                model=BASELINE_MODELS,
            ),
            foundation_models=expand(
                MODELS_DIR / "{foundation_model}" / "best_model.pth",
                foundation_model=FOUNDATION_MODELS,
            ),
            fusion_model=MODELS_DIR / "fusion" / "best_model.pth",
            test_split=SPLITS_DIR / "test_split.json",
            embeddings=expand(
                EMBEDDINGS_DIR / "{backbone}" / "{{patient_id}}.h5",
                backbone=config.get("embedding_backbone", "uni"),
            ),
            radiomics=expand(RADIOMICS_DIR / "{patient_id}.csv", patient_id="*"),
        output:
            results=EVAL_DIR / "evaluation_results.json",
            curves=EVAL_DIR / "roc_curves.png",
        resources:
            mem_mb=32000,
            gpus=1,
            threads=8,
            time_min=120,
        shell:
            """
            python -m src.evaluation.evaluate_all \
                --models_dir {MODELS_DIR} \
                --test_split {input.test_split} \
                --embeddings_dir {EMBEDDINGS_DIR} \
                --radiomics_dir {RADIOMICS_DIR} \
                --output_dir {EVAL_DIR}
            """


# ============================================================================
# RULE 11: Generate Report
# ============================================================================
if STAGES.get("report", True):
    rule generate_report:
        """
        Generate final HTML/PDF report with results.

        Includes: model comparisons, ROC curves, feature importance, etc.
        """
        input:
            results=EVAL_DIR / "evaluation_results.json",
            curves=EVAL_DIR / "roc_curves.png",
        output:
            report=EVAL_DIR / "final_report.html",
            journal=EVAL_DIR / "evaluation_journal.md",
        shell:
            """
            python -m src.evaluation.generate_report \
                --results {input.results} \
                --curves {input.curves} \
                --output {output.report} \
                --journal {output.journal}
            """
