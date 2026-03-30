# Quick Start Guide - MM Imaging Radiomics Pipeline

## Installation

```bash
# Clone repository (already done)
cd /sessions/blissful-amazing-cray/r5

# Install dependencies
pip install -r requirements.txt

# Optional: Create conda environment
conda create -n mm_pipeline python=3.10
conda activate mm_pipeline
pip install -r requirements.txt
```

## Verify Installation

```bash
# Check Snakemake
snakemake --version

# Check Nextflow
nextflow -v

# Check Python dependencies
python -c "import ray; import dask; print('Ray and Dask OK')"
```

## Run Pipeline

### Option 1: Using Master Script (Recommended)

```bash
# Full pipeline with Snakemake on local machine
./scripts/run_pipeline.sh --engine snakemake --profile local --jobs 8

# Or with Nextflow on SLURM cluster
./scripts/run_pipeline.sh --engine nextflow --profile slurm --jobs 64

# Dry run to validate before execution
./scripts/run_pipeline.sh --engine snakemake --dry-run
```

### Option 2: Using Snakemake Directly

```bash
# Full pipeline
snakemake --configfile configs/pipeline.yaml --cores 8 --keep-going

# Run specific rule
snakemake create_splits --configfile configs/pipeline.yaml --cores 4

# With SLURM cluster
snakemake --configfile configs/pipeline.yaml --profile slurm
```

### Option 3: Using Nextflow Directly

```bash
# Full pipeline
nextflow run nextflow/main.nf -c nextflow/nextflow.config -profile local

# With SLURM
nextflow run nextflow/main.nf -c nextflow/nextflow.config -profile slurm

# Resume interrupted run
nextflow run nextflow/main.nf -c nextflow/nextflow.config -resume

# Generate pipeline DAG visualization
nextflow run nextflow/main.nf -with-dag results/dag.html
```

### Option 4: Using DVC

```bash
# Run pipeline
dvc repro

# Run specific stage
dvc repro create_splits

# View pipeline structure
dvc dag
```

## Configure Pipeline

Edit `configs/pipeline.yaml` to customize:

```yaml
# Data directories
data_dir: "./data"
output_dir: "./results"

# Enable/disable stages
stages:
  tiling: true
  stain_norm: true
  embeddings: true
  # ... more stages

# Model hyperparameters
baseline_configs:
  abmil:
    hidden_dim: 256
    learning_rate: 1.0e-4

# Agentic tuning budget
agentic_tuning:
  enabled: true
  max_trials: 50
  max_hours: 24.0
```

## Run Tests

```bash
# Run all tests
pytest tests/test_orchestration.py -v

# Run specific test class
pytest tests/test_orchestration.py::TestAgenticTuner -v

# With coverage report
pytest tests/test_orchestration.py --cov=src/orchestration --cov-report=html
```

## Use Orchestration Components

### Parallel Tile Processing

```python
from src.orchestration import RayTileProcessor, RayTileProcessorConfig

config = RayTileProcessorConfig(num_workers=8, gpu_per_worker=0.25)
processor = RayTileProcessor(config)

embeddings = processor.process_wsis(
    wsi_paths=[Path("data/wsi1.svs"), Path("data/wsi2.svs")],
    tile_extractor=my_extractor,
    output_dir=Path("results/embeddings")
)
```

### Hyperparameter Search

```python
from src.orchestration import HyperparameterSearcher, HyperparameterSearchConfig

config = HyperparameterSearchConfig(
    max_trials=50,
    metric="auroc",
    scheduler="asha"
)
searcher = HyperparameterSearcher(config)

best_config = searcher.search(
    model_type="abmil",
    train_fn=train_model,
    train_data=train_loader,
    val_data=val_loader,
    output_dir=Path("results/search")
)
```

### Agentic Tuning

```python
from src.orchestration import (
    AgenticTuner, AgenticTunerConfig,
    LockedSurface, EditableSurface
)

locked = LockedSurface(
    locked_files={"src/data/*", "src/evaluation/*"}
)
editable = EditableSurface(
    editable_config_keys={"learning_rate", "batch_size"}
)

config = AgenticTunerConfig(
    metric="auroc",
    max_trials=50
)
tuner = AgenticTuner(config, locked, editable)

results = tuner.tune(
    train_fn=train_model,
    data=(train_data, val_data),
    baseline_config=best_hparam_config
)
```

## Monitor Results

### Check Pipeline Logs

```bash
# View Snakemake execution logs
cat logs/snakemake.log

# View Nextflow execution logs
cat .nextflow.log

# View pipeline summary
ls -lh results/
```

### View Experiment Journal

```bash
# After agentic tuning
cat experiments/agentic/EXPERIMENT_JOURNAL.md

# Or access JSON logs
ls -la experiments/agentic/trial_*.json
```

### Check Metrics

```bash
# With DVC
dvc metrics show

# Or view JSON files
cat results/models/baseline_abmil/metrics.json
```

## Troubleshooting

### Ray Initialization Issues
```python
import ray
if ray.is_initialized():
    ray.shutdown()
# Then retry
```

### Memory Errors
- Reduce `num_workers` in RayTileProcessorConfig
- Reduce `memory_per_worker_gb` in DaskRadiomicsConfig
- Reduce `batch_size` in configs/pipeline.yaml

### Snakemake Rule Failures
```bash
# Rerun failed rules
snakemake --configfile configs/pipeline.yaml --rerun-incomplete

# Force rerun
snakemake --configfile configs/pipeline.yaml --force create_splits
```

### Nextflow Resume
```bash
# Resume from last successful task
nextflow run nextflow/main.nf -c nextflow/nextflow.config -resume
```

## Build Container

```bash
# Build Docker image
docker build -t mm-pipeline:latest .

# Run in container
docker run --gpus all -v $(pwd)/data:/workspace/data mm-pipeline:latest

# Build Singularity image
singularity build mm-pipeline.sif Singularity.def
```

## Output Structure

```
results/
├── tiles/                    # WSI tiles
├── normalized_tiles/         # After stain normalization
├── deduplicated_tiles/       # After deduplication
├── embeddings/
│   └── uni/                  # Foundation model embeddings
├── radiomics/                # Radiomics features
├── splits/                   # Train/val/test splits
├── models/
│   ├── baseline_abmil/       # Baseline models
│   ├── baseline_clam/
│   ├── foundation_uni/       # Foundation models
│   └── fusion/               # Fusion model
├── evaluation/
│   ├── evaluation_results.json
│   ├── roc_curves.png
│   └── final_report.html
└── experiments/
    └── agentic/              # Agentic tuning logs
        ├── EXPERIMENT_JOURNAL.md
        └── trial_*.json
```

## Next Steps

1. **Prepare Data**: Organize WSI and imaging data in data/
2. **Customize Config**: Edit configs/pipeline.yaml for your setup
3. **Test on Sample**: Run with small sample first
4. **Monitor Execution**: Watch logs in real-time
5. **Analyze Results**: Check evaluation report and metrics
6. **Fine-tune Models**: Use agentic tuning for optimization

## Documentation

- **Full Guide**: `/sessions/blissful-amazing-cray/r5/docs/ORCHESTRATION.md`
- **Implementation Details**: `/sessions/blissful-amazing-cray/r5/ORCHESTRATION_SUMMARY.md`
- **File Index**: `/sessions/blissful-amazing-cray/r5/IMPLEMENTATION_INDEX.md`

## Support

For issues or questions:
1. Check logs in results/ and logs/
2. Review ORCHESTRATION.md troubleshooting section
3. Run tests to verify installation
4. Check experiment journal for detailed trial logs
