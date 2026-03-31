# R5 — Imaging Pathology & Radiomics Surrogate-Genetics Pipeline for Multiple Myeloma

A complete, research-grade pipeline for slide-level pathology representation learning and radiomics-based molecular-surrogate prediction in Multiple Myeloma (MM).

## Architecture

```
Raw WSI/DICOM → Tiling → Stain Norm → Dedup → Patch Embeddings (Zarr)
                                                       ↓
                              Classical Baselines (ResNet50, ABMIL, CLAM, Cox/RSF)
                                                       ↓
                              Foundation Models (UNI2-h, TITAN) + MIL Heads (TransMIL, DTFD)
                                                       ↓
                              Multimodal Fusion (Early, Late, Cross-Attention, Gated)
                                                       ↓
                              Evaluation (Patient-level splits, Bootstrap CIs, MLflow/W&B)
                                                       ↓
                              Agentic Tuning (Locked preprocessing, editable config only)
```

## Stack

| Layer | Tools |
|---|---|
| Storage | Parquet/Arrow (tabular), Zarr (embeddings), DICOM/OME-TIFF (imaging) |
| Orchestration | Snakemake + Nextflow DSL2, Ray/Dask for parallelism |
| Experiment tracking | MLflow / Weights & Biases, DVC for data/model versioning |
| Reproducibility | Docker / Apptainer, environment snapshots, preprocessing contracts |

## Modeling Rule

1. **Classical baseline first** — ResNet50 mean-pool, ABMIL, CLAM, handcrafted radiomics + Cox/RSF
2. **Foundation model second** — UNI2-h (1536-dim), TITAN (768-dim) with TransMIL/DTFD-MIL heads
3. **Multimodal fusion last** — imaging + radiomics + optional genomics

## Evaluation Rule

- Patient-level splits only (no tile/patch leakage)
- Time-aware splits for longitudinal work
- Train-only fitting of normalization/imputation
- Frozen preprocessing contract before agentic tuning starts

## Repository Structure

```
r5/
├── src/
│   ├── data/                    # WSI tiling, stain norm, dedup, embeddings, radiomics
│   ├── models/                  # Tile classifier, ABMIL, CLAM, survival models
│   │   └── foundation/          # UNI2-h, TITAN, MIL heads, fusion, explainability
│   ├── evaluation/              # Splits, metrics, preprocessing contracts, tracking, viz
│   └── orchestration/           # Parallel processing, HPO, agentic tuning, reproducibility
├── configs/                     # YAML configs for all pipeline stages
├── scripts/                     # Entry-point scripts for each stage
├── tests/                       # Unit tests for all modules
├── docs/                        # Literature review, benchmarks, dataset catalog
├── nextflow/                    # Nextflow DSL2 pipeline
├── Snakefile                    # Snakemake workflow
├── Dockerfile                   # Production container
├── dvc.yaml                     # DVC pipeline stages
└── requirements.txt             # Python dependencies
```

## Benchmark Targets

> **Note:** These are literature reference values, not results from this pipeline.
> This pipeline has not yet been validated on the listed datasets. Results will be
> added here as experiments are completed.

| Task | Metric | Value | Source |
|---|---|---|---|
| H&E-based t(11;14) detection | AUROC | 0.85 | Published MM morphology study |
| Multimodal morphology+flow t(11;14) | AUROC | 0.892 | 2024 multimodal study |
| PET/CT radiomics risk stratification | C-index | 0.693 | NDMM radiomics study |

## Public Datasets

- **TCIA CMB-MML** — Cancer Moonshot Biobank MM (radiology + histopathology, 64+ patients)
- **TCIA SN-AM** — MM bone-marrow aspirate slides with stain variability
- **MiMM_SBILab** — 85 microscopic MM aspirate images
- **PCMMD** — 5000+ plasma cells with morphologic labels (2025)
- **SegPC-2021** — 775 segmentation images

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline via Snakemake
snakemake --cores 8 --configfile configs/pipeline.yaml

# Or step by step:
python scripts/run_preprocessing.py --config configs/data_pipeline.yaml
python scripts/train_baselines.py --model abmil --config configs/model_baselines.yaml
python scripts/extract_foundation_features.py --config configs/foundation_models.yaml
python scripts/run_evaluation.py --config configs/evaluation.yaml
```

## Documentation

- [Literature Landscape](docs/literature_landscape.md) — 40+ papers mapped with contradictions, gaps, and knowledge map
- [Benchmark Targets](docs/benchmark_targets.md) — All published metrics for comparison
- [Dataset Catalog](docs/dataset_catalog.md) — Public MM imaging datasets with access instructions
- [Orchestration Guide](docs/ORCHESTRATION.md) — Snakemake/Nextflow/agentic tuning details

## License

Research use. See individual dataset licenses for data terms.
