# Foundation Models Integration for Multiple Myeloma Imaging Pipeline

## Overview

This module implements the "foundation model second, multimodal fusion last" component of the Multiple Myeloma Imaging Pathology & Radiomics Surrogate-Genetics Pipeline. It provides production-grade integration of modern deep learning architectures for WSI analysis.

## Architecture

```
Whole-Slide Image (WSI)
        ↓
    [Tile Extraction]
        ↓
┌─────────────────────────────────────────┐
│     Feature Extraction Layer            │
│  ┌─────────────┬──────────┬───────────┐ │
│  │   UNI2-h    │  TITAN   │ ResNet50  │ │ (Foundation Models)
│  │ (1536-dim)  │(768-dim) │(2048-dim) │ │
│  └──────┬──────┴────┬─────┴─────┬─────┘ │
└─────────┼──────────┼───────────┼────────┘
          ↓          ↓           ↓
    ┌─────────────────────────────────────┐
    │    MIL Head Layer                   │
    │  ┌──────────┬─────────┬──────────┐  │
    │  │TransMIL  │ DTFD-MIL│HIPT-style│  │ (Slide-level aggregation)
    │  └────┬─────┴────┬────┴────┬─────┘  │
    └───────┼──────────┼─────────┼────────┘
            ↓          ↓         ↓
    ┌──────────────────────────────────────────────┐
    │  Multimodal Fusion Layer                     │
    │  ┌──────────┬────────┬──────────┬──────────┐│
    │  │  Early   │  Late  │ Cross-   │  Gated   ││
    │  │ Fusion   │Fusion  │Attention │ Fusion   ││
    │  └────┬─────┴───┬────┴────┬─────┴────┬─────┘│
    └───────┼─────────┼─────────┼──────────┼──────┘
            ↓         ↓         ↓          ↓
    ┌────────────────────────────────┐
    │  Classification & Prediction   │
    │  + Explainability Report       │
    └────────────────────────────────┘
```

## Components

### 1. Feature Extractors (`src/models/foundation/`)

#### UNI2H Encoder (`uni2h_encoder.py`)
- **Model**: Vision Transformer H/14 (DINOv2-based)
- **Output**: 1536-dimensional embeddings per tile
- **Training data**: 335k+ histopathology slides (Mass General Brigham)
- **License**: CC-BY-NC-ND 4.0 (academic non-commercial)
- **Access**: Requires HuggingFace authentication

```python
encoder = UNI2HEncoder(
    model_name="MahmoodLab/UNI2-h",
    embedding_dim=1536,
    batch_size=64,
    hf_token="hf_xxxx"  # Required
)

embeddings = encoder.extract_batch("path/to/tiles/")
```

**Key Features**:
- Automatic GPU/CPU device selection
- Batch inference with tqdm progress
- Embedding caching for resume support
- Memory-efficient data loading

#### TITAN Encoder (`titan_encoder.py`)
- **Model**: Multimodal WSI foundation model (ViT + language alignment)
- **Output**: 768-dimensional slide-level embeddings
- **Training data**: 335k+ WSIs with pathology reports
- **Published**: Nature Medicine, 2025

```python
encoder = TITANEncoder(
    model_name="MahmoodLab/TITAN",
    embedding_dim_slide=768,
    batch_size=16
)

features = encoder.extract_slide_features("slide.svs")
```

#### Feature Extractor Interface (`feature_extractor.py`)
- **Abstract base class** for all extractors
- **Factory pattern**: `get_extractor(name, config)`
- **Implementations**: ResNet50, UNI2H, TITAN
- **Standardized API**: All extractors expose `.extract_batch()` and `.extract_single()`

```python
# Factory pattern for easy model selection
extractor = get_extractor(
    'uni2h',
    batch_size=64,
    device='cuda',
    hf_token=os.environ['HF_TOKEN']
)

embeddings = extractor.extract_batch('tiles/')
```

### 2. Multiple Instance Learning Heads (`mil_heads.py`)

Modern MIL architectures that aggregate tile-level features to slide-level predictions.

#### TransMIL
- **Mechanism**: Transformer self-attention over patch embeddings
- **Key innovation**: Positional encoding from tile (x,y) coordinates
- **Learnable [CLS] token** for slide-level aggregation
- **Multi-head attention** (configurable heads/layers)

```python
mil_head = TransMIL(
    input_dim=1536,        # UNI2-h output
    hidden_dim=512,
    num_heads=8,
    num_layers=2,
    use_coords=True        # Use spatial coordinates
)

logits, aux = mil_head(embeddings, coords=coords)
# logits: (B, num_classes)
# aux: {'cls_embedding': (B, hidden_dim), 'patch_embeddings': ...}
```

#### DTFD-MIL (Double-Tier Feature Distillation)
- **Tier 1**: Attention aggregation within pseudo-bags
- **Tier 2**: Attention aggregation across pseudo-bags
- **Benefit**: Better capture of hierarchical slide structure

```python
mil_head = DTFDMIL(
    input_dim=1536,
    num_pseudo_bags=8,
    tier1_hidden=256,
    tier2_hidden=128
)

logits, aux = mil_head(embeddings)
# aux: {'tier1_weights': ..., 'tier2_weights': ..., 'assignments': ...}
```

#### HIPT-style Hierarchical Head
- **Region-level aggregation**: Group patches into spatial regions
- **Slide-level aggregation**: Aggregate regions to slide representation
- **Spatial awareness**: Enforces local structure learning

```python
mil_head = HIRCLSMILHead(
    input_dim=1536,
    num_regions=16,        # 4x4 grid
    hidden_dim=512
)

logits, aux = mil_head(embeddings, coords=coords)
# aux: {'region_features': (B, num_regions, hidden_dim), ...}
```

### 3. Multimodal Fusion (`multimodal_fusion.py`)

Combines imaging, radiomics, and genomic features for integrated predictions.

#### Early Fusion
- Simple concatenation of all modality features
- Baseline approach
- Treats all modalities equally

```python
fusion = EarlyFusion(
    modality_dims={'imaging': 1536, 'radiomics': 128},
    num_classes=2
)

features = {'imaging': emb_imaging, 'radiomics': emb_radiomics}
logits, outputs = fusion(features)
```

#### Late Fusion
- Independent classifier per modality
- Learned combination weights
- Better for handling modality-specific patterns

```python
fusion = LateFusion(
    modality_dims={'imaging': 1536, 'radiomics': 128},
    num_classes=2,
    learned_weights=True
)

logits, outputs = fusion(features)
# outputs: {'modality_weights': (B, 2), 'modality_logits': {...}}
```

#### Cross-Attention Fusion
- **Bidirectional attention** between modalities
- Each modality attends to all others
- Learns complex modality interactions

```python
fusion = CrossAttentionFusion(
    modality_dims={'imaging': 1536, 'radiomics': 128},
    num_classes=2,
    num_heads=4,
    hidden_dim=256
)

logits, outputs = fusion(features)
# outputs: {'attention_weights': {...}, 'modality_weights': ...}
```

#### Gated Fusion
- **Per-sample importance weights** learned via gating network
- Captures which modality to trust for each sample
- Interpretable modality contributions

```python
fusion = GatedFusion(
    modality_dims={'imaging': 1536, 'radiomics': 128},
    num_classes=2,
    hidden_dim=128
)

logits, outputs = fusion(features)
# outputs: {'gate_scores': (B, num_modalities), 'modality_weights': ...}
```

### 4. Explainability (`explainability.py`)

Clinically-grounded interpretability for model decisions.

#### Attention Heatmap Generation
Maps MIL attention weights back to slide coordinates.

```python
explainer = AttentionExplainer(tile_size=256, magnification=20)

heatmap = explainer.generate_heatmap(
    attention_weights=weights,  # (N,)
    coords=coords,              # (N, 2)
    output_size=(1024, 1024)
)

# Returns (1024, 1024) heatmap image
```

#### Top-K Tile Extraction
Identifies most important tiles.

```python
top_tiles = explainer.get_top_tiles(
    attention_weights=weights,
    tile_paths=tile_paths,
    k=10
)

# Returns [(idx, weight, path), ...] sorted by importance
```

#### Modality Importance Analysis
Quantifies contribution of each modality.

```python
importance = explainer.compute_modality_importance(
    modality_weights={'imaging': weights_img, 'radiomics': weights_rad}
)

# Returns {'imaging': 0.72, 'radiomics': 0.28}
```

#### HTML Report Generation
Creates clinical review reports.

```python
report = explainer.generate_report(
    slide_id='slide_001',
    prediction=0.85,
    confidence=0.92,
    attention_weights=weights,
    coords=coords,
    tile_paths=tile_paths,
    modality_weights=modality_weights,
    radiomics_features=radiomics_dict
)

explainer.export_html(report, Path('report.html'))
```

## Configuration (`configs/foundation_models.yaml`)

```yaml
feature_extraction:
  backbone: uni2h  # or resnet50, titan
  uni2h:
    batch_size: 64
    embedding_dim: 1536
    requires_auth: true

mil_head:
  type: transmil  # or dtfd, hipt, abmil
  transmil:
    hidden_dim: 512
    num_heads: 8
    use_coords: true

multimodal_fusion:
  type: cross_attention
  modalities: [imaging, radiomics]
  cross_attention:
    num_heads: 4
    hidden_dim: 256

explainability:
  attention_heatmap: true
  top_k_tiles: 10
  shap_radiomics: true
```

## Usage Examples

### Example 1: Extract Features and Train MIL Model

```python
from src.models.foundation import get_extractor, TransMIL
from torch.utils.data import DataLoader

# 1. Extract features
extractor = get_extractor('uni2h', batch_size=64, device='cuda')
embeddings = extractor.extract_batch('data/tiles/', output_path='embeddings.pkl')

# 2. Create MIL head
mil_head = TransMIL(
    input_dim=1536,
    hidden_dim=512,
    num_heads=8,
    num_classes=2
)

# 3. Training loop (simplified)
optimizer = torch.optim.AdamW(mil_head.parameters(), lr=1e-4)

for epoch in range(50):
    for batch in train_dataloader:
        embeddings_batch = batch['embeddings']  # (B, N, 1536)
        coords_batch = batch['coords']          # (B, N, 2)
        labels = batch['labels']                 # (B,)

        logits, aux = mil_head(embeddings_batch, coords=coords_batch)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Example 2: Multimodal Classification

```python
from src.models.foundation import TransMIL, CrossAttentionFusion

# 1. Extract imaging features (via MIL head)
mil_head = TransMIL(input_dim=1536, hidden_dim=256, num_classes=2)
imaging_logits, aux = mil_head(embeddings)
imaging_repr = aux['cls_embedding']  # (B, 256)

# 2. Get radiomics features (separate pipeline)
radiomics_repr = torch.randn(B, 128)  # From radiomics extractor

# 3. Fuse modalities
fusion = CrossAttentionFusion(
    modality_dims={'imaging': 256, 'radiomics': 128},
    num_classes=2
)

features = {'imaging': imaging_repr, 'radiomics': radiomics_repr}
final_logits, outputs = fusion(features)

# 4. Interpret results
modality_weights = outputs['modality_weights']
print(f"Imaging importance: {modality_weights[0, 0]:.3f}")
print(f"Radiomics importance: {modality_weights[0, 1]:.3f}")
```

### Example 3: Generate Clinical Report

```python
from src.models.foundation import AttentionExplainer

explainer = AttentionExplainer(tile_size=256, magnification=20, top_k=10)

report = explainer.generate_report(
    slide_id='MM-001',
    prediction=0.78,
    confidence=0.91,
    attention_weights=attention_weights,  # From MIL head
    coords=tile_coords,
    tile_paths=tile_paths,
    modality_weights=modality_weights,
    radiomics_features={
        'first_order_energy': 0.85,
        'glcm_contrast': -0.42,
        'shape_sphericity': 0.33,
    }
)

explainer.export_html(report, Path('reports/MM-001.html'))
```

## Running Tests

```bash
# Run all foundation model tests
pytest tests/test_foundation.py -v

# Run specific test class
pytest tests/test_foundation.py::TestTransMIL -v

# Run with coverage
pytest tests/test_foundation.py --cov=src/models/foundation --cov-report=html
```

## Feature Extraction Script

```bash
# Extract UNI2-h features
python scripts/extract_foundation_features.py \
    --input-dir data/tiles/ \
    --output-dir data/embeddings/ \
    --backbone uni2h \
    --batch-size 64 \
    --hf-token $HF_TOKEN

# Extract with resume support
python scripts/extract_foundation_features.py \
    --input-dir data/tiles/ \
    --output-dir data/embeddings/ \
    --backbone uni2h

# Use CPU
python scripts/extract_foundation_features.py \
    --input-dir data/tiles/ \
    --output-dir data/embeddings/ \
    --device cpu
```

## Key Design Principles

### 1. Classical Baseline First
Start with ResNet50 on ImageNet for sanity checks before moving to specialized models.

### 2. Foundation Models Second
Use pretrained models (UNI2-h, TITAN) rather than training from scratch.

### 3. Multimodal Fusion Last
Combine modalities systematically, testing fusion strategies in order:
- Early (simplest)
- Late (most flexible)
- Cross-attention (most complex)
- Gated (most interpretable)

### 4. Production-Grade Code
- Type hints and docstrings
- Error handling and logging
- Caching and resume support
- Configuration-driven
- Comprehensive tests

### 5. Clinical Interpretability
Every prediction comes with:
- Attention heatmap
- Top attended regions
- Per-modality contribution scores
- Feature importance (SHAP-like)
- HTML reports for clinicians

## Dependencies

```
torch>=2.0
torchvision
timm  # For ViT models
transformers  # For HuggingFace models
numpy
scipy
scikit-learn
pillow
pyyaml
tqdm
huggingface_hub
```

## References

1. **UNI2-h**: "UNI: A Universal Pathology Foundation Model" (Nature Medicine)
   - Model: https://huggingface.co/MahmoodLab/UNI2-h
   - Code: https://github.com/mahmoodlab/UNI

2. **TITAN**: "A multimodal whole-slide foundation model for pathology" (Nature Medicine, 2025)
   - Model: https://huggingface.co/MahmoodLab/TITAN
   - Code: https://github.com/mahmoodlab/TITAN

3. **TransMIL**: "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification" (NeurIPS 2021)

4. **DTFD-MIL**: "DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning for Histopathology Whole Slide Image Classification"

5. **HIPT**: "Vision Transformer based Multiple Instance Learning for Weakly-Supervised Histopathological Image Classification" (MICCAI 2021)

## Citation

If you use this module in research, please cite:

```bibtex
@article{mahmoodlab_uni2h_2024,
  title={UNI: A Universal Pathology Foundation Model},
  journal={Nature Medicine},
  year={2024},
  doi={10.1038/s41591-024-xxxx}
}

@article{mahmoodlab_titan_2025,
  title={A multimodal whole-slide foundation model for pathology},
  journal={Nature Medicine},
  year={2025},
  doi={10.1038/s41591-025-xxxxx}
}
```

## License

- UNI2-h: CC-BY-NC-ND 4.0 (non-commercial academic research only)
- TITAN: Check HuggingFace model card
- This codebase: Same as parent project

## Contact & Support

For issues with:
- **UNI2-h model access**: Contact MahmoodLab via GitHub
- **TITAN features/bugs**: See https://github.com/mahmoodlab/TITAN
- **This module**: Open issue in project repository
