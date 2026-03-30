# Quick Start Guide: Classical Baselines

## 1. Installation

```bash
# Install dependencies
pip install torch torchvision numpy pandas scikit-learn zarr pyyaml lifelines scikit-survival

# Navigate to repo
cd /path/to/repo
```

## 2. Prepare Data

Ensure you have:
- `data/embeddings.zarr` - Pre-extracted embeddings in Zarr format
- `data/splits.csv` - Train/val/test splits

### Create Example Zarr Store

```python
import numpy as np
import zarr
from pathlib import Path

# Create mock Zarr store for testing
zarr_path = "data/embeddings.zarr"
store = zarr.open(zarr_path, mode="w")

# Create 10 slides with 100 tiles each
for slide_idx in range(10):
    slide_id = f"slide_{slide_idx:03d}"
    slide_group = store.create_group(slide_id)

    for tile_idx in range(100):
        # Random 2048-dim embedding
        embedding = np.random.randn(2048).astype(np.float32)
        slide_group.array(f"tile_{tile_idx}_{tile_idx}.npy", embedding)
```

### Create Example Splits CSV

```python
import pandas as pd

slides = [f"slide_{i:03d}" for i in range(10)]
labels = [i % 2 for i in range(10)]
splits = ["train"] * 6 + ["val"] * 2 + ["test"] * 2

df = pd.DataFrame({
    "slide_id": slides,
    "label": labels,
    "split": splits
})
df.to_csv("data/splits.csv", index=False)
```

## 3. Train Models

### Option A: Command Line

```bash
# Mean pool baseline (sanity check)
python scripts/train_baselines.py \
    --config configs/model_baselines.yaml \
    --model mean_pool \
    --output_dir results/mean_pool

# ABMIL
python scripts/train_baselines.py \
    --config configs/model_baselines.yaml \
    --model abmil \
    --output_dir results/abmil

# CLAM-MB
python scripts/train_baselines.py \
    --config configs/model_baselines.yaml \
    --model clam_mb \
    --output_dir results/clam_mb
```

### Option B: Python API

```python
import torch
from src.models.mean_pool_baseline import MeanPoolBaseline
from src.models.mil_dataset import SplitsManager

# Load config
import yaml
with open("configs/model_baselines.yaml") as f:
    config = yaml.safe_load(f)

# Initialize baseline
baseline = MeanPoolBaseline(
    embedding_dim=2048,
    classifiers=["logistic_regression", "random_forest"]
)

# Load splits
splits_manager = SplitsManager("data/splits.csv")
train_ids, train_labels = splits_manager.get_split("train")
val_ids, val_labels = splits_manager.get_split("val")
test_ids, test_labels = splits_manager.get_split("test")

# Load embeddings
import numpy as np
train_emb, _ = baseline.load_embeddings_from_zarr(
    "data/embeddings.zarr",
    train_ids
)
val_emb, _ = baseline.load_embeddings_from_zarr(
    "data/embeddings.zarr",
    val_ids
)
test_emb, _ = baseline.load_embeddings_from_zarr(
    "data/embeddings.zarr",
    test_ids
)

# Train
train_results = baseline.fit(train_emb, np.array(train_labels))

# Evaluate
val_metrics = baseline.evaluate_all(val_emb, np.array(val_labels))
test_metrics = baseline.evaluate_all(test_emb, np.array(test_labels))

# Print results
for clf, metrics in val_metrics.items():
    print(f"\n{clf}:")
    print(f"  Val Acc: {metrics['accuracy']:.4f}")
    print(f"  Val F1:  {metrics['f1']:.4f}")
    print(f"  Val AUC: {metrics['auroc']:.4f}")
```

## 4. ABMIL Training Example

```python
import torch
from torch.utils.data import DataLoader
from src.models.abmil import ABMIL, ABMILTrainer
from src.models.mil_dataset import SplitsManager

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load splits manager
splits_manager = SplitsManager("data/splits.csv")
dataloaders = splits_manager.get_dataloaders(
    zarr_path="data/embeddings.zarr",
    batch_size=4,
    num_workers=0,
    max_patches=2000,
)

# Model
model = ABMIL(
    input_dim=2048,
    hidden_dim=256,
    attention_dim=128,
    num_classes=2,
    gated=True,
)

# Trainer
trainer = ABMILTrainer(
    model=model,
    device=device,
    learning_rate=2e-4,
    max_epochs=20,
    patience=5,
)

# Train
history = trainer.train(
    dataloaders["train"],
    dataloaders["val"],
    verbose=True
)

# Evaluate
test_loss, test_acc = trainer.validate(dataloaders["test"])
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
```

## 5. CLAM Training Example

```python
import torch
from src.models.clam import CLAM_MB, CLAMTrainer
from src.models.mil_dataset import SplitsManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataloaders
splits_manager = SplitsManager("data/splits.csv")
dataloaders = splits_manager.get_dataloaders(
    zarr_path="data/embeddings.zarr",
    batch_size=4,
)

# Model
model = CLAM_MB(
    input_dim=2048,
    hidden_dim=256,
    num_classes=2,
    num_heads=3,
    inst_cluster=True,
)

# Trainer
trainer = CLAMTrainer(
    model=model,
    device=device,
    learning_rate=1e-4,
    max_epochs=20,
    patience=5,
    inst_lambda=0.5,
)

# Train
history = trainer.train(
    dataloaders["train"],
    dataloaders["val"],
)

# Test
test_loss, test_acc = trainer.validate(dataloaders["test"])
print(f"Test Acc: {test_acc:.4f}")
```

## 6. Attention Visualization

```python
import torch
import numpy as np
from src.models.abmil import ABMIL
from src.models.mil_dataset import MILDataset

# Load model
model = ABMIL(num_classes=2)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()

# Get embeddings for one slide
dataset = MILDataset("data/embeddings.zarr", ["slide_001"], [0])
embeddings, coordinates, _, slide_id = dataset[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embeddings = embeddings.to(device)

# Get attention weights
with torch.no_grad():
    logits, attention_weights = model(embeddings, return_attention=True)

# Top-k tiles
top_k = 10
top_indices = torch.topk(attention_weights, k=top_k)[1]
top_coords = coordinates[top_indices]
top_weights = attention_weights[top_indices]

print(f"Top {top_k} tiles:")
for i, (coord, weight) in enumerate(zip(top_coords, top_weights)):
    print(f"  Tile {i+1}: ({coord[0]:.0f}, {coord[1]:.0f}) - weight {weight:.4f}")
```

## 7. Feature Importance (Mean Pool)

```python
from src.models.mean_pool_baseline import MeanPoolBaseline
import numpy as np

baseline = MeanPoolBaseline()

# Load and train
X_train = np.random.randn(50, 2048)
y_train = np.random.randint(0, 2, 50)
baseline.fit(X_train, y_train)

# Get feature importance for random forest
importances = baseline.get_feature_importance(
    classifier="random_forest",
    top_k=20
)

print("Top 20 important features:")
for i, imp in enumerate(importances):
    print(f"  Feature {i}: {imp:.4f}")
```

## 8. Radiomics + Survival

```python
import numpy as np
from src.models.radiomics_survival import (
    FeatureSelector,
    CoxProportionalHazards,
    RandomSurvivalForest,
)

# Synthetic radiomics features
X = np.random.randn(100, 200)  # 100 samples, 200 radiomics features
T = np.random.exponential(5, 100)  # Time to event
E = np.random.binomial(1, 0.7, 100)  # Event indicator (1=event, 0=censored)

# Feature selection
selector = FeatureSelector(method="lasso", n_features=50)
selected_features = selector.fit(X, E)
X_selected = selector.transform(X)

# Train Cox model
cox = CoxProportionalHazards(penalizer=0.1)
cox.fit(X_selected, T, E)

# Predict
risk_scores = cox.predict(X_selected)

# Evaluate C-index
c_index = cox.concordance_index(X_selected, T, E)
print(f"Cox C-index: {c_index:.4f}")

# Train Random Survival Forest
rsf = RandomSurvivalForest(n_estimators=100)
rsf.fit(X_selected, T, E)
c_index_rsf = rsf.concordance_index(X_selected, T, E)
print(f"RSF C-index: {c_index_rsf:.4f}")

# Feature importance
importances = rsf.get_feature_importance(top_k=10)
print("\nTop 10 important features (RSF):")
for feat, imp in importances.items():
    print(f"  {feat}: {imp:.4f}")
```

## 9. Run Tests

```bash
# All tests
pytest tests/test_models.py -v

# Specific test
pytest tests/test_models.py::TestABMIL::test_forward_pass -v

# With coverage
pytest tests/test_models.py --cov=src/models --cov-report=html
```

## 10. Troubleshooting

### Out of Memory (CUDA)
- Reduce batch size in config
- Reduce max_patches
- Use CPU: `device: "cpu"` in config

### Zarr Store Not Found
- Ensure `data/embeddings.zarr` exists
- Check path in config: `data.zarr_path`

### Splits CSV Invalid
- Verify columns: `slide_id,label,split`
- Check values in `split` column: only "train", "val", "test"

### Models Not Converging
- Reduce learning rate
- Increase patience for early stopping
- Check data balance (verify label distribution)

## Next Steps

1. **Start with mean pool baseline** to establish performance ceiling
2. **Verify ABMIL improves** over mean pool by 3-5%
3. **Compare CLAM variants** for interpretability gains
4. **Fine-tune hyperparameters** with grid search
5. **Add foundation model features** (if mean pool is insufficient)

For detailed documentation, see `BASELINES_README.md`.
