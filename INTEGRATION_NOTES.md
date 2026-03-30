# Integration Notes: Foundation Models with Data Pipeline

## Overview

This document provides integration guidance for connecting the foundation models module with the existing imaging pathology data pipeline.

## Architecture Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                   Data Pipeline Layer                           │
│  (WSI loading, tile extraction, preprocessing)                  │
└──────────────────────────┬──────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Foundation Models Layer (NEW)                   │
│  • Feature Extraction (UNI2-h, TITAN, ResNet50)                │
│  • MIL Heads (TransMIL, DTFD-MIL, HIPT)                        │
│  • Multimodal Fusion (Early, Late, Cross-Attn, Gated)          │
│  • Explainability & Reports                                     │
└──────────────────────────┬──────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│              Classical Models & Downstream Tasks                │
│  (Survival prediction, genomic integration, etc.)              │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Step 1: Tile Extraction (Existing Pipeline)
Input: WSI files (.svs, .tiff)
Output: Directory of tile images (256x256 at 20x magnification)

```
data/
├── raw_slides/
│   ├── MM-001.svs
│   ├── MM-002.svs
│   └── ...
└── tiles/
    ├── MM-001/
    │   ├── tile_0001.jpg
    │   ├── tile_0002.jpg
    │   └── ...
    └── MM-002/
        ├── tile_0001.jpg
        └── ...
```

### Step 2: Feature Extraction (Foundation Models - NEW)
Input: Tile directories
Output: Embedding vectors (cached in pickle/Zarr)

```
python scripts/extract_foundation_features.py \
    --input-dir data/tiles/ \
    --output-dir data/embeddings/ \
    --backbone uni2h \
    --batch-size 64
```

Output structure:
```
data/
└── embeddings/
    ├── MM-001/
    │   ├── embeddings.pkl  # Dict: {tile_name → embedding_array}
    │   └── embeddings.zarr # Zarr store for efficient access
    └── MM-002/
        ├── embeddings.pkl
        └── embeddings.zarr
```

### Step 3: Slide-Level Aggregation (MIL Head)
Input: Per-tile embeddings + coordinates
Output: Slide-level representation

```python
# Load embeddings
embeddings = joblib.load('data/embeddings/MM-001/embeddings.pkl')
# embeddings: Dict[tile_name → np.array (1536,)]

# Convert to batch tensor
tile_names = sorted(embeddings.keys())
embedding_vectors = np.array([embeddings[name] for name in tile_names])
embedding_tensor = torch.tensor(embedding_vectors).unsqueeze(0)  # (1, N, 1536)

# Load tile coordinates (from tile extraction metadata)
coords = np.array([extract_coords_from_name(name) for name in tile_names])
coords_tensor = torch.tensor(coords).unsqueeze(0)  # (1, N, 2)

# Apply MIL head
mil_head = TransMIL(input_dim=1536, hidden_dim=512, num_classes=2)
logits, aux = mil_head(embedding_tensor, coords=coords_tensor)

# Get slide-level representation
slide_repr = aux['cls_embedding']  # (1, 512)
```

### Step 4: Multimodal Fusion
Input: Imaging representation + radiomics features
Output: Fused prediction + explainability

```python
# Imaging features (from MIL head)
imaging_repr = slide_repr  # (1, 512)

# Radiomics features (from existing radiomics pipeline)
radiomics_features = extract_radiomics(tile_images)  # (1, 128)

# Fusion
fusion = CrossAttentionFusion(
    modality_dims={'imaging': 512, 'radiomics': 128},
    num_classes=2
)

features = {
    'imaging': imaging_repr,
    'radiomics': radiomics_features
}

logits, outputs = fusion(features)

# Interpret
modality_importance = outputs['modality_weights'][0]
print(f"Imaging: {modality_importance[0]:.3f}")
print(f"Radiomics: {modality_importance[1]:.3f}")
```

## Required Integration Points

### 1. Embedding Store (`src/data/embedding_store.py`)

Implement or import a Zarr-based embedding store for efficient access:

```python
class ZarrEmbeddingStore:
    """Zarr-based store for cached embeddings."""

    def __init__(self, storage_path, embedding_dim, compressor='blosc'):
        self.storage_path = Path(storage_path)
        self.embedding_dim = embedding_dim
        self.store = zarr.open(str(storage_path), mode='a')

    def __setitem__(self, key, value):
        """Store embedding for a slide."""
        self.store[key] = value

    def __getitem__(self, key):
        """Retrieve embedding for a slide."""
        return self.store[key][:]

    def __contains__(self, key):
        """Check if embedding exists."""
        return key in self.store
```

### 2. Tile Coordinate Tracking

Modify tile extraction to save coordinate metadata:

```json
// data/tiles/MM-001/metadata.json
{
    "tile_0001.jpg": {"x": 0, "y": 0, "mpp": 0.5},
    "tile_0002.jpg": {"x": 256, "y": 0, "mpp": 0.5},
    ...
}
```

Then load when aggregating:

```python
import json

with open('data/tiles/MM-001/metadata.json') as f:
    metadata = json.load(f)

tile_names = sorted(metadata.keys())
coords = np.array([metadata[name][['x', 'y']] for name in tile_names])
```

### 3. Data Pipeline Integration

Add to main training pipeline:

```python
from src.models.foundation import (
    get_extractor,
    TransMIL,
    CrossAttentionFusion,
    AttentionExplainer
)

class IntegratedPipeline:
    def __init__(self, config):
        self.config = config

        # Feature extraction
        self.extractor = get_extractor(
            config['backbone'],
            device=config['device']
        )

        # MIL aggregation
        self.mil_head = TransMIL(
            input_dim=self.extractor.embedding_dim,
            hidden_dim=config['mil']['hidden_dim'],
            num_classes=config['num_classes']
        )

        # Multimodal fusion
        self.fusion = get_fusion_module(
            config['fusion_type'],
            modality_dims=config['modality_dims'],
            num_classes=config['num_classes']
        )

        # Explainability
        self.explainer = AttentionExplainer()

    def process_slide(self, slide_path, save_report=True):
        """End-to-end slide processing."""

        # 1. Extract tiles (existing pipeline)
        tiles = extract_tiles(slide_path)
        tile_paths = [t.path for t in tiles]
        tile_coords = np.array([[t.x, t.y] for t in tiles])

        # 2. Extract features
        embeddings = [
            self.extractor.extract_single(path)
            for path in tile_paths
        ]
        embeddings = torch.tensor(np.array(embeddings)).unsqueeze(0)

        # 3. Aggregate to slide level
        coords = torch.tensor(tile_coords).unsqueeze(0).float()
        logits, mil_aux = self.mil_head(embeddings, coords=coords)
        imaging_repr = mil_aux['cls_embedding']

        # 4. Get radiomics features
        radiomics = extract_radiomics_features(tiles)

        # 5. Fuse modalities
        features = {
            'imaging': imaging_repr,
            'radiomics': radiomics
        }
        final_logits, fusion_aux = self.fusion(features)

        # 6. Generate report
        if save_report:
            # Get attention weights from MIL head
            attention_weights = compute_attention_weights(mil_aux)

            report = self.explainer.generate_report(
                slide_id=slide_path.stem,
                prediction=final_logits[0, 1].item(),  # Probability of positive class
                confidence=final_logits[0].softmax(-1).max().item(),
                attention_weights=attention_weights,
                coords=tile_coords,
                tile_paths=tile_paths,
                modality_weights=fusion_aux['modality_weights'][0].cpu().numpy(),
                radiomics_features=radiomics.squeeze().cpu().numpy()
            )

            self.explainer.export_html(
                report,
                Path(f'reports/{slide_path.stem}.html')
            )

        return {
            'slide_id': slide_path.stem,
            'prediction': final_logits.cpu().detach().numpy(),
            'imaging_repr': imaging_repr.cpu().detach().numpy(),
            'radiomics': radiomics.cpu().detach().numpy(),
            'modality_importance': fusion_aux['modality_weights'].cpu().detach().numpy(),
        }
```

## Configuration Integration

Merge foundation model config into main training config:

```yaml
# configs/training.yaml
model:
  type: multimodal_mm_classifier

  # Classical baseline
  baseline:
    backbone: resnet50

  # Foundation model stack
  foundation:
    feature_extraction:
      backbone: uni2h  # UNI2-h, TITAN, or resnet50
      batch_size: 64

    mil_head:
      type: transmil
      hidden_dim: 512
      use_coords: true

    fusion:
      type: cross_attention
      modalities: [imaging, radiomics]

  # Training
  training:
    num_epochs: 50
    learning_rate: 1.0e-4
    early_stopping_patience: 10

  # Output
  outputs:
    save_embeddings: true
    generate_reports: true
    export_heatmaps: true
```

## Testing & Validation

### Unit Tests
```bash
pytest tests/test_foundation.py -v
```

### Integration Tests
```python
def test_full_pipeline():
    """Test end-to-end processing."""
    # Create test slide
    test_slide = create_synthetic_slide()

    # Process
    pipeline = IntegratedPipeline(config)
    result = pipeline.process_slide(test_slide)

    # Validate outputs
    assert result['prediction'].shape == (2,)
    assert 0 <= result['prediction'].sum() <= 1.0  # Probabilities sum to 1
    assert 'modality_importance' in result
    assert abs(result['modality_importance'].sum() - 1.0) < 0.01
```

### Clinical Validation
- [ ] Compare UNI2-h predictions to ResNet50 baseline
- [ ] Validate MIL head attention weights against histopathologist annotations
- [ ] Check multimodal fusion improves over single-modality models
- [ ] Review generated HTML reports for clinical utility

## Performance Considerations

### Memory Usage
- UNI2-h: ~3 GB VRAM (inference only)
- Batch size 64: ~8 GB VRAM
- Consider gradient checkpointing if training (reduces by 30%)

### Computational Cost
- Feature extraction: ~2-5 minutes per 10k tiles on V100
- MIL aggregation: <1 second per slide
- Multimodal fusion: <100 ms per slide

### Caching Strategy
```
1. Cache tile-level embeddings (UNI2-h outputs)
2. Reuse for different MIL heads
3. Cache slide-level representations
4. Re-extract only on code changes
```

## Troubleshooting

### HuggingFace Authentication for UNI2-h
```bash
# Option 1: Set environment variable
export HF_TOKEN=hf_xxxxxxxxxxxx

# Option 2: Use huggingface-cli
huggingface-cli login

# Option 3: Pass to extractor
extractor = get_extractor('uni2h', hf_token='hf_xxx')
```

### Out of Memory
```python
# Reduce batch size
extractor = get_extractor('uni2h', batch_size=32)

# Enable gradient checkpointing in MIL head
mil_head = TransMIL(..., gradient_checkpoint=True)

# Process smaller chunks
for slide_batch in slide_batches:
    process_slide(slide_batch)
```

### Slow Inference
```python
# Use num_workers for data loading
extractor = get_extractor('uni2h', num_workers=8)

# Enable mixed precision
torch.cuda.amp.autocast()

# Pin memory for faster data transfer
torch.backends.cudnn.benchmark = True
```

## Future Extensions

### 1. Genomic Integration
```python
# Add genomic modality
fusion = get_fusion_module(
    'cross_attention',
    modality_dims={
        'imaging': 512,
        'radiomics': 128,
        'genomic': 256  # From genomic pipeline
    }
)
```

### 2. Survival Prediction
```python
# Replace classification head with Cox regression
survival_head = nn.Sequential(
    nn.Linear(fused_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 1)  # Log hazard
)
```

### 3. Uncertainty Quantification
```python
# Bayesian MIL head via dropout
class BayesianTransMIL(TransMIL):
    def forward(self, x, coords=None, n_samples=10):
        # Enable dropout during inference
        predictions = []
        for _ in range(n_samples):
            logits, _ = super().forward(x, coords)
            predictions.append(logits)

        # Compute mean and variance
        predictions = torch.stack(predictions)
        mean = predictions.mean(0)
        std = predictions.std(0)
        return mean, std
```

## References

See `FOUNDATION_MODELS_README.md` for detailed documentation and references.
