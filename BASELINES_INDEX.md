# Classical Baselines: Complete Index

## Quick Navigation

### Getting Started
- **[BASELINES_QUICKSTART.md](BASELINES_QUICKSTART.md)** - Start here! Step-by-step setup and examples
- **[BASELINES_README.md](BASELINES_README.md)** - Comprehensive reference documentation
- **[BASELINES_SUMMARY.txt](BASELINES_SUMMARY.txt)** - Overview of all files and statistics

## Core Implementation Files

### Models

| File | Lines | Purpose |
|------|-------|---------|
| `src/models/__init__.py` | 41 | Module exports and clean API |
| `src/models/tile_classifier.py` | 408 | ResNet50 tile-level classifier |
| `src/models/mean_pool_baseline.py` | 292 | Simplest baseline (mean pooling) |
| `src/models/abmil.py` | 421 | Attention-Based Multiple Instance Learning |
| `src/models/clam.py` | 506 | Clustering-constrained Attention MIL |
| `src/models/radiomics_survival.py` | 424 | Cox PH and Random Survival Forest |
| `src/models/mil_dataset.py` | 325 | PyTorch Dataset for MIL |
| `src/models/losses.py` | 363 | Loss functions (CE, Cox, Focal, etc.) |
| **Total** | **2,780** | **Core implementation** |

### Configuration & Scripts

| File | Purpose |
|------|---------|
| `configs/model_baselines.yaml` | Complete configuration for all models |
| `scripts/train_baselines.py` | Training orchestration script |

### Testing & Documentation

| File | Purpose |
|------|---------|
| `tests/test_models.py` | 40+ unit tests covering all models |
| `BASELINES_README.md` | Detailed reference documentation |
| `BASELINES_QUICKSTART.md` | Quick-start guide with examples |
| `BASELINES_SUMMARY.txt` | File inventory and statistics |
| `BASELINES_INDEX.md` | This file |

## Model Overview

### 1. Mean Pool Baseline
**File**: `src/models/mean_pool_baseline.py`

Simplest baseline: load embeddings → mean pool → train simple classifier (LR/SVM/RF)
- ✓ Fastest inference
- ✓ No learned parameters (besides classifier)
- ✓ Sanity check baseline
- Baseline to beat others against

**Key Classes**:
- `MeanPoolBaseline` - Main baseline class
- `FeatureSelector` - For feature selection

**Training Example**:
```bash
python scripts/train_baselines.py --model mean_pool
```

### 2. Tile Classifier
**File**: `src/models/tile_classifier.py`

Per-tile ResNet50 classification with slide-level mean pooling.
- ✓ Pretrained ImageNet weights
- ✓ Mixed-precision training
- ✓ Cosine annealing learning rate
- ✓ Early stopping + checkpoints

**Key Classes**:
- `TileClassifier` - ResNet50-based classifier
- `TileClassifierTrainer` - Training orchestration
- `TileDataset` - PyTorch Dataset

**Usage**:
```python
model = TileClassifier(num_classes=2, pretrained=True)
trainer = TileClassifierTrainer(model, device)
history = trainer.train(train_loader, val_loader)
```

### 3. ABMIL
**File**: `src/models/abmil.py`

Attention-Based Multiple Instance Learning with gated attention.
- ✓ Learnable attention mechanism
- ✓ Attention weight visualization
- ✓ Bag-level training only
- ✓ Typically 3-5% better than mean pool

**Key Classes**:
- `GatedAttention` - Attention mechanism (standard + gated)
- `ABMIL` - Main model
- `ABMILTrainer` - Training orchestration

**Training Example**:
```bash
python scripts/train_baselines.py --model abmil
```

### 4. CLAM
**File**: `src/models/clam.py`

Clustering-constrained Attention MIL with single/multi-branch variants.
- ✓ CLAM-SB: single attention head
- ✓ CLAM-MB: multiple attention heads
- ✓ Instance-level pseudo-labeling
- ✓ Comparable to ABMIL, more interpretable

**Key Classes**:
- `AttentionLayer` - Single attention head
- `CLAM_SB` - Single-branch model
- `CLAM_MB` - Multi-branch model
- `CLAMTrainer` - Training orchestration

**Training Examples**:
```bash
python scripts/train_baselines.py --model clam_sb
python scripts/train_baselines.py --model clam_mb
```

### 5. Radiomics + Survival
**File**: `src/models/radiomics_survival.py`

Classical handcrafted radiomics with survival analysis.
- ✓ Cox Proportional Hazards
- ✓ Random Survival Forest
- ✓ Multiple feature selection strategies
- ✓ C-index evaluation

**Key Classes**:
- `FeatureSelector` - LASSO, mutual info, variance selection
- `CoxProportionalHazards` - Cox PH model
- `RandomSurvivalForest` - Random survival forest
- `KaplanMeierAnalysis` - Survival curve analysis

**Usage**:
```python
cox = CoxProportionalHazards()
cox.fit(X_train, T_train, E_train)
c_index = cox.concordance_index(X_test, T_test, E_test)
```

## Loss Functions

**File**: `src/models/losses.py`

All loss functions with comprehensive docstrings:

| Loss | Purpose |
|------|---------|
| `CrossEntropyLoss` | Standard classification |
| `CoxPartialLikelihoodLoss` | Survival analysis |
| `InstanceClusteringLoss` | Instance constraints |
| `FocalLoss` | Class imbalance handling |
| `SmoothTopKLoss` | Differentiable top-k for MIL |
| `WeightedFocalLoss` | Focal + class weights |
| `DiceLoss` | Alternative for imbalance |

## Dataset & Utilities

**File**: `src/models/mil_dataset.py`

- `MILDataset` - PyTorch Dataset for Zarr embeddings
- `SplitsManager` - Train/val/test split management
- `create_mil_dataloader` - DataLoader factory
- `collate_fn_mil` - Custom collate for variable-length bags

## Training Orchestration

**File**: `scripts/train_baselines.py`

Command-line interface for training all models:

```bash
# Usage
python scripts/train_baselines.py \
    --config configs/model_baselines.yaml \
    --model {mean_pool|abmil|clam_sb|clam_mb} \
    --output_dir results/

# Output
results/{model}/
├── checkpoints/
│   ├── checkpoint_epoch_001.pt
│   ├── checkpoint_epoch_002.pt
│   └── ...
├── logs/
│   └── train.log
└── results.json
```

## Configuration

**File**: `configs/model_baselines.yaml`

Comprehensive YAML configuration with:
- Model-specific hyperparameters
- Training settings (epochs, learning rate, patience)
- Data loading configuration
- Hardware settings (device, mixed precision)
- Logging configuration

Easily customizable for different datasets/hyperparameters.

## Testing

**File**: `tests/test_models.py`

Comprehensive unit tests:
- `TestGatedAttention` - Attention mechanism
- `TestABMIL` - ABMIL model
- `TestCLAM` - CLAM models
- `TestTileClassifier` - Tile classifier
- `TestLossFunctions` - All loss functions
- `TestMILDataset` - Dataset loading
- `TestMeanPoolBaseline` - Baseline model
- `TestIntegration` - End-to-end training

Run with:
```bash
pytest tests/test_models.py -v
```

## Data Format

### Zarr Embedding Store

Pre-extracted tile embeddings in Zarr format:
```
data/embeddings.zarr/
├── slide_001/
│   ├── tile_0_0.npy      # Shape: (2048,)
│   ├── tile_1_0.npy
│   └── ...
├── slide_002/
│   └── ...
```

### Splits CSV

Train/val/test split specification:
```csv
slide_id,label,split
slide_001,0,train
slide_002,1,train
slide_003,0,val
slide_004,1,test
```

## Performance Expectations

| Model | AUROC | Speed | Interpretability |
|-------|-------|-------|------------------|
| Mean Pool | ~0.85 | Fastest | Low |
| Tile Classifier | ~0.82 | Fast | Medium |
| ABMIL | ~0.88 | Medium | Medium-High |
| CLAM-SB | ~0.88 | Medium | High |
| CLAM-MB | ~0.89 | Slow | Very High |

*Note: Exact performance depends on dataset and hyperparameters*

## Code Statistics

```
Total Lines of Code:        2,780 (core models only)
Config Lines:                 119
Training Script:              417
Test Lines:                   496
Documentation:             ~2,500
Total Files:                  14

Model Classes:                  8 (TileClassifier, ABMIL, CLAM-SB, CLAM-MB, Cox, RSF, etc.)
Trainer Classes:                3 (TileClassifierTrainer, ABMILTrainer, CLAMTrainer)
Loss Functions:                 7
Unit Tests:                    40+
```

## Dependencies

**Core**:
- torch >= 2.0
- torchvision >= 0.15
- numpy, pandas
- scikit-learn, zarr, pyyaml

**Optional**:
- lifelines (Cox models)
- scikit-survival (Random Survival Forest)

Install: `pip install torch torchvision numpy pandas scikit-learn zarr pyyaml lifelines scikit-survival`

## Quick Start Checklist

- [ ] Read BASELINES_QUICKSTART.md
- [ ] Install dependencies: `pip install ...`
- [ ] Prepare data (Zarr embeddings + splits CSV)
- [ ] Run mean pool baseline (sanity check)
- [ ] Train ABMIL if baseline insufficient
- [ ] Compare CLAM-MB for interpretability
- [ ] Fine-tune hyperparameters in config
- [ ] Add foundation models only if needed

## Documentation Files

| File | Purpose | Read Time |
|------|---------|-----------|
| **BASELINES_QUICKSTART.md** | Setup + examples | 15 min |
| **BASELINES_README.md** | Reference docs | 30 min |
| **BASELINES_SUMMARY.txt** | Overview | 10 min |
| **BASELINES_INDEX.md** | This file | 5 min |

## Key Principles

1. **Classical Baseline First** - Start simple, add complexity only if needed
2. **Complete Independence** - Each model works standalone
3. **Production Quality** - Full logging, error handling, checkpoints
4. **Interpretability** - Attention weights, feature importance
5. **Extensibility** - Clean abstractions for custom models

## Next Steps

1. **Start here**: [BASELINES_QUICKSTART.md](BASELINES_QUICKSTART.md)
2. **Setup**: Follow setup instructions
3. **Sanity check**: Train mean pool baseline
4. **Improve**: Move to ABMIL if needed (need >5% improvement)
5. **Interpret**: Use CLAM for attention visualization
6. **Extend**: Add foundation models for remaining problems

## Support & References

For issues or questions, see:
- **Code**: Comprehensive docstrings in all files
- **Tests**: 40+ unit tests in tests/test_models.py
- **Examples**: Full examples in BASELINES_QUICKSTART.md
- **References**: Citations in BASELINES_README.md

---

**Last Updated**: March 30, 2026
**Implementation**: Production-quality classical baselines
**Models**: 5 (Tile Classifier, Mean Pool, ABMIL, CLAM-SB, CLAM-MB, Radiomics)
