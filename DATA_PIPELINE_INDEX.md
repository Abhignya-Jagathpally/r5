# Data Pipeline - Complete File Index & API Reference

## Quick Navigation

- **Getting Started:** See [DATA_PIPELINE_QUICKSTART.md](DATA_PIPELINE_QUICKSTART.md)
- **Full Documentation:** See [src/data/README.md](src/data/README.md)
- **Implementation Details:** See [PIPELINE_IMPLEMENTATION_SUMMARY.txt](PIPELINE_IMPLEMENTATION_SUMMARY.txt)

## Module Structure

### Core Modules (`src/data/`)

#### 1. **wsi_tiler.py** - WSI Tiling Pipeline
**Class:** `WSITiler`

| Method | Description |
|--------|-------------|
| `process_slide()` | Process single WSI slide (auto-detects format) |
| `process_svs_slide()` | Process OpenSlide-compatible formats |
| `process_standard_image()` | Process standard image formats (PNG, JPEG) |
| `process_slides()` | Batch process multiple slides in parallel |
| `_detect_tissue()` | HSV-based tissue detection |
| `_is_blurry()` | Laplacian variance-based blur detection |

**Key Parameters:**
- `tile_size`: Size of extracted tiles (default: 256)
- `magnification`: Target magnification level (default: 20x)
- `overlap`: Pixel overlap between tiles (default: 0)
- `min_tissue_fraction`: Minimum tissue required per tile (default: 0.5)

**Output:**
- PNG tiles in `output_dir/`
- CSV manifest: `tile_manifest.csv`

---

#### 2. **stain_normalizer.py** - Stain Normalization & Quality Filtering

**Classes:**
- `Macenko` - Reference-based stain normalization
- `Reinhard` - Distribution-matching color normalization
- `StainNormalizer` - High-level API with quality filtering

| Method | Class | Description |
|--------|-------|-------------|
| `fit()` | Macenko/Reinhard | Fit to reference image |
| `transform()` | Macenko/Reinhard | Normalize test image |
| `fit_to_reference()` | StainNormalizer | Fit to reference file path |
| `normalize()` | StainNormalizer | Normalize single tile |
| `process_tile_directory()` | StainNormalizer | Batch process directory |
| `is_high_quality()` | StainNormalizer | Check quality filters |

**Quality Filters:**
- Background detection (white/light regions > 80%)
- Blur detection (Laplacian variance < threshold)
- Pen mark detection (HSV-based color thresholding)

**Output:**
- Normalized PNG tiles
- Processing statistics (processed, filtered, normalized, errors)

---

#### 3. **deduplicator.py** - Tile Deduplication

**Class:** `TileDeduplicator`

| Method | Description |
|--------|-------------|
| `build_index()` | Compute perceptual hashes for all tiles |
| `find_clusters()` | Find near-duplicate clusters |
| `select_representatives()` | Choose one tile per cluster |
| `get_duplicates_to_remove()` | Get set of duplicate filenames |
| `deduplicate_directory()` | Full pipeline: index → cluster → remove |
| `get_cluster_report()` | Generate detailed cluster statistics |
| `save_report()` | Export report to CSV |

**Hash Algorithms:**
- `phash` - Perceptual hash (default, recommended)
- `ahash` - Average hash
- `dhash` - Difference hash
- `whash` - Wavelet hash

**Output:**
- Deduplicated tiles in `output_dir/`
- CSV report: cluster IDs, sizes, representatives

---

#### 4. **embedding_store.py** - Embedding Extraction & Storage

**Classes:**
- `EmbeddingExtractor` - Extract embeddings from images
- `EmbeddingStore` - Store embeddings in Zarr format
- `PatchDataset` - PyTorch Dataset for lazy loading
- `SimpleImageDataset` - Helper dataset for tile loading

**EmbeddingExtractor Methods:**
| Method | Description |
|--------|-------------|
| `extract_batch()` | Extract embeddings for batch of images |
| `extract_from_directory()` | Extract for all tiles in directory |

**Supported Backbones:**
- `resnet50` - Built-in ResNet50 (2048-dim)
- `uni2h` - Placeholder (use ResNet50 fallback)
- `titan` - Placeholder (use ResNet50 fallback)

**EmbeddingStore Methods:**
| Method | Description |
|--------|-------------|
| `add_slide_embeddings()` | Store embeddings for single slide |
| `get_slide_embeddings()` | Retrieve embeddings from Zarr |
| `list_slides()` | List all slide IDs |
| `save_metadata()` | Export slide metadata to Parquet |

**PatchDataset Methods:**
| Method | Description |
|--------|-------------|
| `__len__()` | Total number of patches |
| `__getitem__()` | Load single patch with metadata |

**Output:**
- Zarr store: `embeddings_dir/`
- Per-slide groups with datasets: embeddings (N×D), coordinates (N×2), attention_scores (N×1)
- Metadata parquet: `metadata.parquet`

---

#### 5. **radiomics_extractor.py** - Radiomics Feature Extraction

**Class:** `RadiomicsExtractor`

| Method | Description |
|--------|-------------|
| `extract_features()` | Extract features for single image-mask pair |
| `extract_batch()` | Extract for multiple pairs |
| `extract_from_directory()` | Extract for all images in directory |
| `save_config()` | Export extraction config to YAML |

**Feature Classes:**
- `firstorder` - First-order statistics
- `shape` - Shape descriptors
- `glcm` - Gray-Level Co-occurrence Matrix
- `glrlm` - Gray-Level Run-Length Matrix
- `glszm` - Gray-Level Size-Zone Matrix
- `gldm` - Gray-Level Dependence Matrix

**Output:**
- Dictionary of feature names → values
- Parquet format for batch extraction

---

### Configuration (`configs/`)

**data_pipeline.yaml** - Complete pipeline configuration

Sections:
- `tiling` - WSI extraction parameters
- `stain_norm` - Normalization method and reference
- `quality_filter` - Thresholds for filtering
- `dedup` - Deduplication settings
- `embeddings` - Model and batch processing
- `radiomics` - Feature extraction settings
- `storage` - Input/output directory paths
- `processing` - Pipeline steps and logging

---

### Scripts (`scripts/`)

**run_preprocessing.py** - End-to-end pipeline orchestration

```bash
python scripts/run_preprocessing.py \
    --config configs/data_pipeline.yaml \
    --slides data/raw/*.svs \
    --slide-ids s001 s002 s003 \
    --output-dir data/preprocessed \
    --steps tiling quality_filter stain_norm dedup embeddings \
    --resume \
    --skip-existing
```

**Features:**
- Sequential pipeline orchestration
- Progress tracking with tqdm
- Resume from checkpoint
- Skip already-processed slides
- Logging to file and console

---

### Tests (`tests/`)

**test_data_pipeline.py** - Comprehensive unit tests

Test Classes:
- `TestWSITiler` - Tiling functionality
- `TestStainNormalization` - Stain normalization and filtering
- `TestTileDeduplicator` - Deduplication accuracy
- `TestEmbeddingStore` - Zarr storage and metadata
- `TestEmbeddingExtractor` - Embedding extraction

Run tests:
```bash
pytest tests/test_data_pipeline.py -v
pytest tests/test_data_pipeline.py::TestWSITiler -v
pytest tests/test_data_pipeline.py --cov=src/data
```

---

## Class Hierarchy & Methods

### WSITiler
```
WSITiler
├── __init__(tile_size, magnification, overlap, min_tissue_fraction, output_dir)
├── process_slide(slide_path, slide_id, laplacian_threshold) → List[Dict]
├── process_slides(slide_paths, slide_ids, max_workers, laplacian_threshold) → DataFrame
├── process_svs_slide(slide_path, slide_id, laplacian_threshold) → List[Dict]
├── process_standard_image(image_path, slide_id, laplacian_threshold) → List[Dict]
├── _detect_tissue(tile) → float
└── _is_blurry(tile, laplacian_threshold) → bool
```

### StainNormalizer Hierarchy
```
Macenko
├── __init__(alpha, beta)
├── fit(image) → self
├── transform(image) → ndarray
└── _get_stain_matrix(image) → (ndarray, ndarray)

Reinhard
├── __init__()
├── fit(image) → self
└── transform(image) → ndarray

StainNormalizer
├── __init__(method, max_background_fraction, min_laplacian_variance, pen_mark_detection)
├── fit_to_reference(reference_path) → self
├── fit_to_image(image) → self
├── normalize(image) → ndarray
├── process_tile_directory(input_dir, output_dir, reference_image, file_extension) → dict
├── is_high_quality(image) → bool
├── _is_mostly_background(image) → bool
├── _is_blurry(image) → bool
└── _has_pen_marks(image) → bool
```

### TileDeduplicator
```
TileDeduplicator
├── __init__(hamming_threshold, hash_algorithm)
├── build_index(tile_directory) → Dict[str, Hash]
├── find_clusters() → List[Set[str]]
├── select_representatives() → Dict[int, str]
├── deduplicate_directory(tile_directory, output_directory, remove_duplicates) → dict
├── get_duplicates_to_remove() → Set[str]
├── get_cluster_report() → DataFrame
├── save_report(output_path)
└── _hamming_distance(hash1, hash2) → int
```

### Embedding Classes
```
EmbeddingExtractor
├── __init__(backbone, embedding_dim, device, pretrained)
├── extract_batch(images) → ndarray
├── extract_from_directory(tile_directory, batch_size, num_workers) → (ndarray, List[str])
├── _build_model(pretrained) → nn.Module
└── _get_transforms() → Compose

EmbeddingStore
├── __init__(store_path)
├── add_slide_embeddings(slide_id, embeddings, coordinates, attention_scores, label, split)
├── get_slide_embeddings(slide_id) → Dict[str, ndarray]
├── list_slides() → List[str]
└── save_metadata(output_path) → DataFrame

PatchDataset(Dataset)
├── __init__(zarr_store, metadata_path, split, tile_directory)
├── __len__() → int
└── __getitem__(idx) → Dict
```

### RadiomicsExtractor
```
RadiomicsExtractor
├── __init__(config_path, bin_width, resample_spacing, feature_classes)
├── extract_features(image_path, mask_path, sample_id) → Dict[str, float]
├── extract_batch(image_mask_pairs, sample_ids) → DataFrame
├── extract_from_directory(image_dir, mask_dir, output_path, image_suffix, mask_suffix) → DataFrame
├── load_config(config_path) → Dict
└── save_config(output_path)
```

---

## Data Flow Diagram

```
WSI Slides (SVS, TIFF, PNG)
    ↓ [WSITiler]
Extracted Tiles (PNG) + tile_manifest.csv
    ↓ [Quality Filtering + StainNormalizer]
Normalized Tiles (PNG)
    ↓ [TileDeduplicator]
Deduplicated Tiles (PNG) + dedup_report.csv
    ↓ [EmbeddingExtractor]
Embeddings (Float32, N×2048)
    ↓ [EmbeddingStore]
Zarr Store (hierarchical by slide) + metadata.parquet
    ↓ [PatchDataset → DataLoader]
Mini-batches for Training

(Parallel) DICOM Images + Segmentation Masks
    ↓ [RadiomicsExtractor]
Radiomics Features (Dict → Parquet)
```

---

## Configuration Examples

### Minimal Configuration
```yaml
tiling:
  tile_size: 256
  magnification: 20

stain_norm:
  method: macenko

dedup:
  hamming_threshold: 8

embeddings:
  backbone: resnet50
  batch_size: 64
```

### High-Performance Configuration
```yaml
tiling:
  tile_size: 512
  magnification: 40
  max_workers: 8

stain_norm:
  method: reinhard
  reference_slide: /path/to/reference.png

dedup:
  hamming_threshold: 10

embeddings:
  backbone: resnet50
  batch_size: 128
  num_workers: 8
  device: cuda
```

### Conservative Configuration (Memory-Limited)
```yaml
tiling:
  tile_size: 128
  magnification: 10

stain_norm:
  method: reinhard

quality_filter:
  min_laplacian_variance: 20.0

embeddings:
  batch_size: 16
  num_workers: 2
  device: cpu
```

---

## Common Usage Patterns

### Pattern 1: Process Single Slide
```python
from src.data import WSITiler

tiler = WSITiler()
tiler.process_slide("slide.svs", "slide_001")
```

### Pattern 2: Batch Processing with All Steps
```python
from src.data import (
    WSITiler, StainNormalizer, TileDeduplicator, 
    EmbeddingExtractor, EmbeddingStore
)

# Tiling
tiler = WSITiler(output_dir="data/tiles")
manifest = tiler.process_slides(slides, ids)

# Normalization
normalizer = StainNormalizer(method="macenko")
normalizer.process_tile_directory("data/tiles", "data/normalized")

# Deduplication
dedup = TileDeduplicator()
dedup.deduplicate_directory("data/normalized", "data/deduplicated")

# Embeddings
extractor = EmbeddingExtractor(device="cuda")
embeddings, files = extractor.extract_from_directory("data/deduplicated")

store = EmbeddingStore("data/embeddings")
store.add_slide_embeddings("slide_001", embeddings, coordinates)
store.save_metadata("data/embeddings/metadata.parquet")
```

### Pattern 3: Training with Lazy Loading
```python
from src.data import PatchDataset
from torch.utils.data import DataLoader

dataset = PatchDataset(
    "data/embeddings",
    "data/embeddings/metadata.parquet",
    split="train"
)

loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

for batch in loader:
    x = batch["embedding"]
    y = batch["label"]
    # Train model...
```

---

## File Paths (Absolute)

```
/sessions/blissful-amazing-cray/r5/
├── src/data/
│   ├── __init__.py
│   ├── wsi_tiler.py
│   ├── stain_normalizer.py
│   ├── deduplicator.py
│   ├── embedding_store.py
│   ├── radiomics_extractor.py
│   └── README.md
├── configs/
│   └── data_pipeline.yaml
├── scripts/
│   └── run_preprocessing.py
├── tests/
│   ├── __init__.py
│   └── test_data_pipeline.py
├── requirements.txt
├── DATA_PIPELINE_QUICKSTART.md
├── PIPELINE_IMPLEMENTATION_SUMMARY.txt
└── DATA_PIPELINE_INDEX.md (this file)
```

---

## Quick Reference: Import Everything

```python
from src.data import (
    # Tiling
    WSITiler,
    # Normalization
    StainNormalizer, Macenko, Reinhard,
    # Deduplication
    TileDeduplicator,
    # Embeddings
    EmbeddingExtractor, EmbeddingStore, PatchDataset,
    # Radiomics
    RadiomicsExtractor
)
```

---

## API Stability

All public classes and methods are production-ready. Future versions will maintain backward compatibility for:
- Class initialization parameters
- Public method signatures
- Data format specifications (Zarr, Parquet, CSV)

Internal implementation details may change without notice.

