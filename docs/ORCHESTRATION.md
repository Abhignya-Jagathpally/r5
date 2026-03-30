# Orchestration Layer Documentation

## Overview

The orchestration layer provides production-grade workflow management for the MM Imaging Pathology & Radiomics Surrogate-Genetics Pipeline. It implements Andrej Karpathy's autoresearch philosophy with locked preprocessing and editable training surfaces.

## Architecture

### Core Components

```
src/orchestration/
├── __init__.py                  # Public API
├── parallel_features.py         # Ray/Dask parallel processing
├── hyperparameter_search.py    # Ray Tune-based search
├── agentic_tuner.py            # Autoresearch pattern tuning
└── reproducibility.py          # Environment snapshots & tracking
```

### Workflow Orchestration Files

```
/
├── Snakefile                   # Snakemake workflow definition
├── Nextflow/
│   ├── main.nf                # Nextflow DSL2 main pipeline
│   └── nextflow.config         # Nextflow configuration
├── dvc.yaml                    # DVC pipeline definition
├── Dockerfile                  # Container image definition
├── configs/
│   └── pipeline.yaml           # Master pipeline configuration
├── scripts/
│   └── run_pipeline.sh         # Master execution script
└── tests/
    └── test_orchestration.py   # Test suite
```

## Key Design Principles

### 1. Locked vs. Editable Surfaces

Following Karpathy's autoresearch pattern:

**Locked Surface** (immutable):
- Data loading and preprocessing (`src/data/*`)
- Evaluation metrics and splitting (`src/evaluation/`)
- Experiment infrastructure code

**Editable Surface** (agent-modifiable):
- Model configurations (`configs/model_*.yaml`)
- Training hyperparameters
- Optimization strategy
- Augmentation choices

**Safety Check**: AgenticTuner verifies preprocessing contract hash hasn't changed.

### 2. Single Metric Optimization

- Define one objective metric (e.g., AUROC for classification)
- Optimize with fixed budget (trials or wall-clock hours)
- Track all experiments completely
- Alert on suspiciously large improvements (possible leakage)

### 3. Complete Experiment Logging

Every run produces:
```json
{
  "trial_id": 42,
  "timestamp": "2024-03-30T14:32:45.123456",
  "config_diff": {"lr": [1e-4, 2e-4], "batch_size": [32, 64]},
  "metric_value": 0.8745,
  "wall_clock_seconds": 342.5,
  "git_hash": "abc123def456...",
  "safety_checks_passed": true
}
```

### 4. Reproducibility Infrastructure

- Environment snapshots (Python packages, GPU info, system details)
- Automatic Dockerfile/Singularity generation
- Git integration (commit hashing)
- Seed management (numpy, torch, random)
- DVC pipeline definition

## Modules

### parallel_features.py

Ray and Dask based parallel processing for feature extraction.

#### RayTileProcessor

Parallel WSI tile processing across multiple GPUs:

```python
from src.orchestration import RayTileProcessor, RayTileProcessorConfig

config = RayTileProcessorConfig(
    num_workers=8,
    gpu_per_worker=0.25,
    batch_size=32
)
processor = RayTileProcessor(config)

# Process multiple WSIs in parallel
embeddings = processor.process_wsis(
    wsi_paths=[Path("data/wsi1.svs"), Path("data/wsi2.svs")],
    tile_extractor=my_extractor_fn,
    output_dir=Path("results/embeddings")
)
```

**Features**:
- Automatic Ray cluster initialization
- Fault tolerance with retries
- Progress tracking
- Memory management for large WSIs

#### DaskRadiomicsExtractor

Parallel radiomics extraction across CPU cores:

```python
from src.orchestration import DaskRadiomicsExtractor, DaskRadiomicsConfig

config = DaskRadiomicsConfig(
    num_workers=16,
    threads_per_worker=4
)
extractor = DaskRadiomicsExtractor(config)

# Extract radiomics features in parallel
features = extractor.extract_batch(
    images=image_array,
    masks=mask_array,
    radiomics_fn=pyradiomics_extractor,
    features_to_extract=["glcm", "glrlm"]
)
```

**Features**:
- Dask distributed execution
- Timeout handling
- Result gathering with error handling

### hyperparameter_search.py

Ray Tune-based hyperparameter optimization.

#### HyperparameterSearcher

```python
from src.orchestration import HyperparameterSearcher, HyperparameterSearchConfig

config = HyperparameterSearchConfig(
    max_trials=50,
    num_samples=10,
    metric="auroc",
    metric_mode="max",
    scheduler="asha",
    search_algorithm="optuna"
)
searcher = HyperparameterSearcher(config)

# Run search for a model type
best_config = searcher.search(
    model_type="abmil",
    train_fn=train_model,
    train_data=train_loader,
    val_data=val_loader,
    output_dir=Path("results/search")
)
```

**Search Spaces**:
- ABMIL: attention MIL
- CLAM: clustering-constrained attention
- TransMIL: transformer-based MIL
- DSMIL: dual-stream MIL
- Fusion: multimodal fusion

**Schedulers**:
- ASHA: early stopping with successive halving
- PBT: population-based training
- FIFO: simple queue

**Search Algorithms**:
- Optuna: Bayesian optimization
- Random: random search
- Bayesian: Bayesian optimization via scikit-optimize

### agentic_tuner.py

Core autoresearch-pattern agentic tuning implementation.

#### AgenticTuner

```python
from src.orchestration import (
    AgenticTuner,
    AgenticTunerConfig,
    LockedSurface,
    EditableSurface
)

# Define surfaces
locked = LockedSurface(
    locked_files={"src/data/loader.py", "src/evaluation/metrics.py"},
    locked_functions={"load_data", "compute_auroc"}
)

editable = EditableSurface(
    editable_files={"configs/model_config.yaml"},
    editable_config_keys={"learning_rate", "batch_size", "hidden_dim"}
)

# Create tuner
config = AgenticTunerConfig(
    metric="auroc",
    metric_mode="max",
    budget_type="trials",
    max_trials=50,
    max_hours=24.0
)
tuner = AgenticTuner(config, locked, editable)

# Run agentic tuning
results = tuner.tune(
    train_fn=train_model,
    data=(train_data, val_data),
    baseline_config=best_hparam_config,
    modification_generator=my_modification_fn  # optional custom generator
)
```

**Safety Checks**:
- Verify preprocessing contract hash hasn't changed
- Verify data splits are identical across runs
- Check for code integrity violations
- Alert on suspiciously large improvements

**Output**:
- experiment_journal.md: Markdown report of all trials
- trial_*.json: Per-trial JSON logs
- Best configuration and metrics

### reproducibility.py

Environment snapshots and experiment tracking.

#### EnvironmentSnapshot

```python
from src.orchestration import EnvironmentSnapshot

# Create snapshot of current environment
snapshot = EnvironmentSnapshot.create()

# Save to file
snapshot.save(Path("results/environment.json"))
```

#### DockerfileGenerator

```python
from src.orchestration import DockerfileGenerator

snapshot = EnvironmentSnapshot.create()
generator = DockerfileGenerator(snapshot)

dockerfile_content = generator.generate(
    output_path=Path("Dockerfile.generated"),
    source_dir=Path(".")
)
```

#### ExperimentJournal

```python
from src.orchestration import ExperimentJournal

journal = ExperimentJournal(Path("logs"))

journal.add_entry(
    experiment_id="exp_001",
    model_type="abmil",
    config={"lr": 1e-4, "batch_size": 32},
    metrics={"auroc": 0.85, "aupr": 0.78},
    git_hash="abc123def456",
    notes="Baseline ABMIL model"
)

journal.save()
report = journal.generate_markdown_report()
```

#### SeedManager

```python
from src.orchestration import SeedManager

# Set all seeds for reproducibility
SeedManager.set_seed(42)

# Or get seed configuration
seed_config = SeedManager.get_seed_config(42)
```

## Workflow Execution

### Using Snakemake

```bash
# Full pipeline
snakemake --configfile configs/pipeline.yaml --cores 8

# Dry run
snakemake --configfile configs/pipeline.yaml --dryrun

# Run specific rule
snakemake create_splits --configfile configs/pipeline.yaml --cores 4

# Use SLURM cluster
snakemake --configfile configs/pipeline.yaml --profile slurm
```

### Using Nextflow

```bash
# Full pipeline
nextflow run nextflow/main.nf -c nextflow/nextflow.config

# With profile
nextflow run nextflow/main.nf -c nextflow/nextflow.config -profile slurm

# Resume failed runs
nextflow run nextflow/main.nf -c nextflow/nextflow.config -resume

# Generate DAG visualization
nextflow run nextflow/main.nf -c nextflow/nextflow.config -with-dag results/dag.html
```

### Using Master Script

```bash
# Make executable
chmod +x scripts/run_pipeline.sh

# Run with snakemake
./scripts/run_pipeline.sh --engine snakemake --profile local --jobs 8

# Run with nextflow
./scripts/run_pipeline.sh --engine nextflow --profile slurm

# Dry run
./scripts/run_pipeline.sh --engine snakemake --dry-run

# Verbose output
./scripts/run_pipeline.sh --engine snakemake --verbose
```

### Using DVC

```bash
# Run pipeline
dvc repro

# Run specific stage
dvc repro create_splits

# View pipeline DAG
dvc dag

# Show metrics
dvc metrics show
```

## Pipeline Stages

### 1. tile_wsis
Extract tiles from whole slide images (256x256, configurable).

**Input**: WSI files (.svs, .tiff)
**Output**: Tile arrays (.npz)

### 2. normalize_tiles
Apply stain normalization (Macenko, Reinhardt, Vahadane).

**Input**: Tiles
**Output**: Normalized tiles

### 3. deduplicate
Remove near-duplicate tiles using SSIM or perceptual hashing.

**Input**: Normalized tiles
**Output**: Deduplicated tiles + statistics

### 4. extract_embeddings
Extract patch embeddings using foundation models (UNI, GigaPath).

**Input**: Deduplicated tiles
**Output**: Embeddings (.h5)

### 5. extract_radiomics
Extract radiomics features from CT/PET images.

**Input**: Medical images (.nii.gz) + segmentations
**Output**: Radiomics features (.csv)

### 6. create_splits
Generate patient-level train/val/test splits.

**Input**: Metadata (.csv)
**Output**: Split definitions (.json)

### 7-9. train_baseline / train_foundation / train_fusion
Train various model architectures.

### 10. evaluate
Comprehensive evaluation on test set.

### 11. generate_report
Final HTML/PDF report with visualizations.

## Configuration

Master configuration in `configs/pipeline.yaml`:

```yaml
pipeline:
  name: mm_imaging_radiomics_pipeline
  version: "0.1.0"

stages:
  tiling: true
  stain_norm: true
  # ... more stages

resources:
  gpu_per_task: 1
  cpu_per_task: 8
  memory_gb: 32
  max_parallel_jobs: 4

agentic_tuning:
  enabled: true
  metric: auroc
  budget_type: trials
  max_trials: 50
  max_hours: 24.0

hyperparameter_search:
  enabled: true
  scheduler: asha
  num_samples: 50
```

## Testing

Run test suite:

```bash
pytest tests/test_orchestration.py -v

# Run specific test
pytest tests/test_orchestration.py::TestAgenticTuner::test_initialization -v

# With coverage
pytest tests/test_orchestration.py --cov=src/orchestration --cov-report=html
```

## Best Practices

1. **Separate Concerns**: Use locked surface for data/eval, editable for model/training
2. **Fixed Budgets**: Always set max trials or wall-clock limits
3. **Complete Logging**: Record every experiment with config, metrics, and git hash
4. **Seed Management**: Always set global seed at pipeline start
5. **Safety Checks**: Verify preprocessing contract and data consistency
6. **Container Reproducibility**: Use Docker/Singularity for production
7. **DVC Integration**: Track data and model artifacts with DVC

## Troubleshooting

### Ray initialization fails
```python
# Check if Ray is already running
import ray
if ray.is_initialized():
    ray.shutdown()
```

### Dask memory errors
- Reduce `memory_per_worker_gb` in DaskRadiomicsConfig
- Reduce `chunk_size` for smaller batches

### Suspicious improvement detected
- Check for data leakage in evaluation
- Verify preprocessing contract hasn't changed
- Review new code in editable surface

### Container build fails
```bash
# Check dependencies
pip install -r requirements.txt

# Build with debug output
docker build -t mm-pipeline:latest . --progress=plain
```

## References

- Andrej Karpathy's Autoresearch: [Tweet](https://twitter.com/karpathy/status/1570564532419579904)
- Ray Documentation: https://docs.ray.io/
- Snakemake Documentation: https://snakemake.readthedocs.io/
- Nextflow Documentation: https://www.nextflow.io/docs/latest/
- DVC Documentation: https://dvc.org/doc
