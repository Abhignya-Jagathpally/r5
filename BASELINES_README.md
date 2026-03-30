# Classical Baselines for Multiple Myeloma Imaging Pathology Pipeline

This module implements production-quality classical baseline models for Multiple Myeloma histopathology analysis, following the **classical baseline first** principle.

## Overview

The baseline implementation includes:

1. **Tile Classifier** - ResNet50-based per-tile classification
2. **Mean Pool Baseline** - Simplest possible baseline (sanity check)
3. **ABMIL** - Attention-Based Multiple Instance Learning
4. **CLAM** - Clustering-constrained Attention MIL
5. **Radiomics + Survival** - Classical handcrafted features with Cox/RSF models

## File Structure

```
src/models/
├── __init__.py                 # Module exports
├── tile_classifier.py          # ResNet50 tile classifier with trainer
├── mean_pool_baseline.py       # Mean pooling sanity check baseline
├── abmil.py                    # Attention-Based MIL implementation
├── clam.py                     # CLAM single-branch and multi-branch
├── radiomics_survival.py       # Cox PH and Random Survival Forest
├── mil_dataset.py              # PyTorch Dataset for MIL training
└── losses.py                   # Loss functions (CE, Cox, Focal, etc.)

configs/
└── model_baselines.yaml        # Configuration for all baselines

scripts/
└── train_baselines.py          # Training orchestration script

tests/
└── test_models.py              # Comprehensive unit tests
```

## Model Descriptions

### 1. Tile Classifier (ResNet50)

**Purpose**: Per-tile classification with mean-pooling to slide-level predictions.

**Key Features**:
- Pretrained ResNet50 backbone from ImageNet
- Custom classification head with batch normalization
- Mixed-precision training (torch.cuda.amp)
- Cosine annealing learning rate schedule
- Early stopping with gradient clipping
- Binary and multi-class support

**Usage**:
```python
from src.models.tile_classifier import TileClassifier, TileClassifierTrainer

# Initialize model
model = TileClassifier(num_classes=2, pretrained=True, dropout=0.5)

# Initialize trainer
trainer = TileClassifierTrainer(
    model=model,
    device=torch.device('cuda'),
    learning_rate=1e-4,
    max_epochs=50,
    patience=10,
)

# Train
history = trainer.train(train_loader, val_loader)

# Predict slide-level from tiles
slide_label, tile_probs, tile_paths = predict_slide_level(
    model, tile_loader, device
)
```

### 2. Mean Pool Baseline

**Purpose**: Simplest possible baseline that all other models must beat.

**Workflow**:
1. Load pre-extracted tile embeddings from Zarr store
2. Mean-pool all tile embeddings per slide
3. Train simple classifiers:
   - Logistic Regression
   - SVM with RBF kernel
   - Random Forest

**Key Features**:
- Feature standardization (StandardScaler)
- Support for binary and multi-class classification
- AUROC, accuracy, F1 metrics
- Feature importance ranking

**Usage**:
```python
from src.models.mean_pool_baseline import MeanPoolBaseline

baseline = MeanPoolBaseline(
    embedding_dim=2048,
    classifiers=['logistic_regression', 'svm', 'random_forest']
)

# Load embeddings from Zarr
X_train, valid_ids = baseline.load_embeddings_from_zarr(
    'data/embeddings.zarr',
    slide_ids
)

# Fit
baseline.fit(X_train, y_train)

# Evaluate on all classifiers
results = baseline.evaluate_all(X_test, y_test)
```

### 3. ABMIL (Attention-Based Multiple Instance Learning)

**Purpose**: MIL with learnable attention mechanism to identify key instances.

**Architecture**:
```
Embeddings (num_tiles, 2048)
    ↓
Gated Attention (standard or gated variant)
    ↓
Feature Transform (2048 → hidden_dim → num_classes)
    ↓
Logits (num_classes,)
```

**Attention Mechanism**:
- Standard: `a_i = softmax(w^T tanh(Vh_i))`
- Gated: `a_i = softmax(w^T tanh(Vh_i) ⊙ sigmoid(Uh_i))`

**Key Features**:
- Bag-level training (only slide labels needed)
- Attention weight visualization
- Mixed-precision training
- Learning rate scheduling
- Early stopping

**Usage**:
```python
from src.models.abmil import ABMIL, ABMILTrainer

model = ABMIL(
    input_dim=2048,
    hidden_dim=256,
    attention_dim=128,
    num_classes=2,
    gated=True
)

trainer = ABMILTrainer(
    model=model,
    device=torch.device('cuda'),
    learning_rate=2e-4,
    max_epochs=100,
)

history = trainer.train(train_loader, val_loader)

# Get attention weights
logits, attention_weights = model(embeddings, return_attention=True)
```

### 4. CLAM (Clustering-constrained Attention MIL)

**Purpose**: MIL with instance-level clustering constraint for interpretability.

**Two Variants**:

**CLAM-SB (Single-Branch)**:
- Single attention head
- Simple attention mechanism
- Optional instance clustering loss

**CLAM-MB (Multi-Branch)**:
- Multiple attention heads for diverse pooling
- Instance-level pseudo-labeling
- Instance classifier for cluster assignment

**Key Features**:
- Instance-level predictions for interpretability
- Flexible number of attention heads
- Instance clustering loss (optional)
- Mixed-precision training

**Usage**:
```python
from src.models.clam import CLAM_SB, CLAM_MB, CLAMTrainer

# Single-branch
model_sb = CLAM_SB(
    input_dim=2048,
    hidden_dim=256,
    num_classes=2,
)

# Multi-branch
model_mb = CLAM_MB(
    input_dim=2048,
    hidden_dim=256,
    num_classes=2,
    num_heads=3,
    inst_cluster=True,
)

trainer = CLAMTrainer(
    model=model_mb,
    device=torch.device('cuda'),
    inst_lambda=0.5,  # Weight for instance clustering loss
    max_epochs=100,
)

history = trainer.train(train_loader, val_loader)

# Get multi-head attention
logits, attention_dict = model_mb(embeddings, return_attention=True)

# Instance predictions (for CLAM-MB with inst_cluster=True)
instance_logits = model_mb.get_instance_predictions(embeddings)
```

### 5. Radiomics + Survival

**Purpose**: Classical handcrafted radiomics features with survival models.

**Feature Selection**:
- Variance threshold
- LASSO-based selection
- Mutual information

**Survival Models**:
- **Cox Proportional Hazards**: Parametric survival model
- **Random Survival Forest**: Non-parametric ensemble

**Key Features**:
- C-index evaluation (0.5 = random, 1.0 = perfect)
- Feature importance ranking
- Kaplan-Meier curve analysis
- Proper handling of censored data

**Usage**:
```python
from src.models.radiomics_survival import (
    CoxProportionalHazards,
    RandomSurvivalForest,
    FeatureSelector,
    KaplanMeierAnalysis,
)

# Feature selection
selector = FeatureSelector(method='lasso', n_features=50)
selected_features = selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)

# Cox model
cox = CoxProportionalHazards(penalizer=0.0)
cox.fit(X_train_selected, T_train, E_train)
c_index = cox.concordance_index(X_test_selected, T_test, E_test)

# Random Survival Forest
rsf = RandomSurvivalForest(n_estimators=100, max_depth=5)
rsf.fit(X_train_selected, T_train, E_train)
c_index = rsf.concordance_index(X_test_selected, T_test, E_test)

# Kaplan-Meier analysis
km_analysis = KaplanMeierAnalysis()
km_analysis.fit_and_plot(T_test, E_test, label="survival")
```

## Dataset Format

### Zarr Embedding Store

Expected structure:
```
data/embeddings.zarr/
├── slide_id_1/
│   ├── tile_0_0.npy         # (embedding_dim,) float array
│   ├── tile_1_0.npy
│   └── tile_N_M.npy
├── slide_id_2/
│   ├── tile_*.npy
│   └── ...
└── ...
```

Each `.npy` file contains a single embedding vector of shape `(embedding_dim,)`.

### Splits CSV

Format:
```csv
slide_id,label,split
slide_001,0,train
slide_002,1,train
slide_003,0,val
slide_004,1,test
...
```

## Configuration

Edit `configs/model_baselines.yaml` to customize:

- Model architectures (input_dim, hidden_dim, num_classes)
- Training parameters (learning_rate, max_epochs, patience)
- Data loading (batch_size, max_patches, sampling_strategy)
- Hardware settings (device, mixed_precision)

## Training

### Train Mean Pool Baseline (Sanity Check)

```bash
python scripts/train_baselines.py \
    --config configs/model_baselines.yaml \
    --model mean_pool \
    --output_dir results/mean_pool
```

### Train ABMIL

```bash
python scripts/train_baselines.py \
    --config configs/model_baselines.yaml \
    --model abmil \
    --output_dir results/abmil
```

### Train CLAM-SB

```bash
python scripts/train_baselines.py \
    --config configs/model_baselines.yaml \
    --model clam_sb \
    --output_dir results/clam_sb
```

### Train CLAM-MB

```bash
python scripts/train_baselines.py \
    --config configs/model_baselines.yaml \
    --model clam_mb \
    --output_dir results/clam_mb
```

### Training Output

Each training run produces:
- `checkpoints/` - Model weights at each epoch
- `logs/train.log` - Detailed training log
- `results.json` - Final metrics (accuracy, F1, AUROC, etc.)

## Testing

Run all unit tests:

```bash
pytest tests/test_models.py -v
```

Run specific test class:

```bash
pytest tests/test_models.py::TestABMIL -v
```

### Test Coverage

- **GatedAttention**: Forward pass, attention weight normalization
- **ABMIL**: Forward pass, attention visualization, eval mode
- **CLAM-SB/MB**: Forward pass, multi-head attention, instance predictions
- **Tile Classifier**: Initialization, forward pass, feature extraction
- **Loss Functions**: Cox loss, Focal loss, Instance clustering loss, Smooth top-k
- **MIL Dataset**: Zarr loading, coordinate parsing, splits management
- **Mean Pool Baseline**: Fit, predict, evaluate, feature importance

## Key Implementation Details

### Mixed Precision Training

All models support PyTorch AMP for faster training:
```python
with autocast(device_type='cuda'):
    logits = model(embeddings)
    loss = criterion(logits, labels)
```

### Gradient Clipping

Prevents training instability:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Learning Rate Scheduling

Cosine annealing with warm restart:
```python
scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
```

### Early Stopping

Based on validation loss:
```python
if val_loss < best_val_loss:
    epochs_without_improvement = 0
    save_checkpoint()
else:
    epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        break
```

## Loss Functions

### Cross-Entropy Loss
Standard classification loss (sklearn-compatible).

### Cox Partial Likelihood Loss
For survival analysis:
```
Loss = -log(h_i / sum_j h_j) for each event
```

### Focal Loss
For class imbalance:
```
FL = -α(1-p_t)^γ log(p_t)
```

### Instance Clustering Loss
For MIL instance constraints.

### Smooth Top-K Loss
Differentiable approximation of top-k instance selection.

## Performance Expectations

### Mean Pool Baseline
- Should achieve ~0.85+ AUROC on typical MM datasets
- Fastest inference (no learned pooling)

### ABMIL
- Should improve over mean pool by 2-5%
- Provides attention visualization
- ~2-3x slower than mean pool

### CLAM
- Usually comparable or slightly better than ABMIL
- Multi-head provides additional interpretability
- Slightly more parameters

### Tile Classifier
- Per-tile classification accuracy varies by histology task
- Can be used for preprocessing or data quality filtering

## Dependencies

```
torch>=2.0
torchvision>=0.15
numpy>=1.20
pandas>=1.3
scikit-learn>=1.0
zarr>=2.10
pyyaml>=6.0
lifelines>=0.27  # For Cox models
scikit-survival>=0.20  # For Random Survival Forest
```

Install with:
```bash
pip install torch torchvision numpy pandas scikit-learn zarr pyyaml lifelines scikit-survival
```

## References

- **ABMIL**: Ilse et al. "Attention-based Deep Multiple Instance Learning" ICML 2018
- **CLAM**: Lu et al. "Data-Efficient and Weakly Supervised Computational Pathology on Whole-Slide Images" NeurIPS 2020
- **Cox PH**: Cox "Regression Models and Life Tables" JRSS 1972
- **RSF**: Ishwaran et al. "Random Survival Forests" Annals of Applied Statistics 2008

## Author Notes

This implementation prioritizes:
1. **Classical baselines first** - Simplest models that work
2. **Reproducibility** - Full code with comprehensive docstrings
3. **Production quality** - Error handling, logging, configuration
4. **Interpretability** - Attention visualization, feature importance
5. **Extensibility** - Clean abstractions for custom models

The "classical baseline first" philosophy means ABMIL, CLAM, and other sophisticated methods are only justified if they significantly outperform mean pooling (~5%+ improvement). When in doubt, start with mean pool baseline.
