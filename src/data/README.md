# Data Processing Pipeline

Comprehensive pipeline for processing Whole Slide Images (WSI) in the Multiple Myeloma imaging pathology study.

## Overview

The pipeline orchestrates the following steps:

1. **WSI Tiling** - Extract tiles from high-resolution WSI with automatic tissue detection
2. **Quality Filtering** - Remove low-quality tiles (background, blur, pen marks)
3. **Stain Normalization** - Normalize histological staining variations
4. **Deduplication** - Identify and remove near-duplicate tiles
5. **Embedding Extraction** - Generate patch embeddings using pretrained models
6. **Radiomics Extraction** (optional) - Extract quantitative features from CT/PET

## Architecture

### Module Structure

```
src/data/
├── __init__.py                  # Module exports
├── wsi_tiler.py                 # WSI tiling with tissue detection
├── stain_normalizer.py          # Stain normalization (Macenko, Reinhard)
├── deduplicator.py              # Perceptual hash-based deduplication
├── embedding_store.py           # Embedding storage and extraction
└── radiomics_extractor.py       # Radiomics feature extraction
```

## Components

### WSITiler

Extracts tiles from Whole Slide Images with automatic tissue detection.

**Features:**
- Supports OpenSlide formats (SVS, TIFF) and standard images (PNG, JPEG)
- HSV-based tissue detection with configurable thresholds
- Automatic blur detection via Laplacian variance
- Parallel processing with ThreadPoolExecutor
- CSV manifest generation with tile metadata

**Usage:**
```python
from src.data import WSITiler

tiler = WSITiler(
    tile_size=256,
    magnification=20,
    overlap=0,
    min_tissue_fraction=0.5,
    output_dir="data/tiles"
)

# Single slide
tiles_data = tiler.process_slide("slide.svs", "slide_001")

# Batch processing
manifest_df = tiler.process_slides(
    ["slide1.svs", "slide2.svs"],
    ["slide_001", "slide_002"],
    max_workers=4
)
```

**Output:**
- PNG tiles saved to `output_dir/`
- CSV manifest: `tile_manifest.csv` with columns:
  - `slide_id`, `tile_filename`, `x`, `y`, `magnification`, `tissue_fraction`

### StainNormalizer

Implements Macenko and Reinhard color normalization with quality filtering.

**Methods:**
- **Macenko** - Estimates stain color matrix; reference-based normalization
- **Reinhard** - Simpler baseline; matches color distribution in LAB space

**Quality Filters:**
- Background detection (tiles >80% white)
- Blur detection (Laplacian variance threshold)
- Pen mark detection (HSV color thresholding)

**Usage:**
```python
from src.data import StainNormalizer, Macenko

# Direct use
normalizer = StainNormalizer(method="macenko")
normalizer.fit_to_reference("reference_tile.png")
normalized = normalizer.normalize(tile_image)

# Batch processing
stats = normalizer.process_tile_directory(
    input_dir="data/tiles",
    output_dir="data/normalized",
    reference_image="reference.png"
)
```

**Output:**
- Normalized PNG tiles
- Processing statistics: `{processed, filtered, normalized, errors}`

### TileDeduplicator

Identifies and removes near-duplicate tiles using perceptual hashing.

**Algorithm:**
1. Compute perceptual hash (pHash) for all tiles
2. Compute Hamming distance matrix
3. Cluster tiles with distance < threshold
4. Select one representative per cluster

**Supported Hashes:**
- `phash` - Perceptual hash (default, recommended)
- `ahash` - Average hash
- `dhash` - Difference hash
- `whash` - Wavelet hash

**Usage:**
```python
from src.data import TileDeduplicator

dedup = TileDeduplicator(
    hamming_threshold=8,
    hash_algorithm="phash"
)

stats = dedup.deduplicate_directory(
    tile_directory="data/tiles",
    output_directory="data/deduplicated",
    remove_duplicates=True
)

# Get detailed report
report_df = dedup.get_cluster_report()
dedup.save_report("dedup_report.csv")
```

**Output:**
- Unique tiles copied/linked to `output_directory/`
- Dedup report: cluster IDs, sizes, representatives
- Statistics: `{total_tiles, num_clusters, num_kept, num_removed}`

### EmbeddingStore & EmbeddingExtractor

Store patch embeddings in Zarr format and extract embeddings from patches.

**Extractor Features:**
- ResNet50 backbone (built-in)
- Placeholders for UNI2-h and TITAN models
- Batch processing with DataLoader support
- GPU acceleration (CUDA)

**Storage Features:**
- Hierarchical Zarr organization (one group per slide)
- Per-slide datasets: embeddings (N×D), coordinates (N×2), attention_scores (N×1)
- Slide metadata in Parquet
- Lazy loading via `PatchDataset`

**Usage:**
```python
from src.data import EmbeddingExtractor, EmbeddingStore, PatchDataset
from torch.utils.data import DataLoader

# Extract embeddings
extractor = EmbeddingExtractor(
    backbone="resnet50",
    embedding_dim=2048,
    device="cuda",
    pretrained=True
)

embeddings, filenames = extractor.extract_from_directory(
    tile_directory="data/tiles",
    batch_size=64,
    num_workers=4
)

# Store embeddings
store = EmbeddingStore("data/embeddings")
store.add_slide_embeddings(
    slide_id="slide_001",
    embeddings=embeddings,
    coordinates=coordinates,
    label="MM",
    split="train"
)
store.save_metadata("data/embeddings/metadata.parquet")

# Load for training
dataset = PatchDataset(
    zarr_store="data/embeddings",
    metadata_path="data/embeddings/metadata.parquet",
    split="train"
)
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

for batch in dataloader:
    embeddings = batch["embedding"]  # (batch_size, 2048)
    coordinates = batch["coordinate"]  # (batch_size, 2)
    label = batch["label"]
```

**Output:**
- Zarr store at `embeddings_dir/`
- Metadata parquet: `embeddings_dir/metadata.parquet`

### RadiomicsExtractor

Extracts quantitative radiomics features from medical images (CT/PET).

**Feature Classes:**
- First-order statistics
- Shape descriptors
- GLCM (Gray-Level Co-occurrence Matrix)
- GLRLM (Gray-Level Run-Length Matrix)
- GLSZM (Gray-Level Size-Zone Matrix)
- GLDM (Gray-Level Dependence Matrix)

**Usage:**
```python
from src.data import RadiomicsExtractor

extractor = RadiomicsExtractor(
    bin_width=25,
    resample_spacing=(1.0, 1.0, 1.0),
    feature_classes=["firstorder", "shape", "glcm"]
)

# Single image
features = extractor.extract_features(
    image_path="ct_scan.dcm",
    mask_path="segmentation.dcm"
)

# Batch processing
features_df = extractor.extract_batch(
    image_mask_pairs=[
        ("ct1.dcm", "mask1.dcm"),
        ("ct2.dcm", "mask2.dcm")
    ],
    sample_ids=["patient_001", "patient_002"]
)

features_df.to_parquet("radiomics_features.parquet")
```

**Output:**
- Dictionary or DataFrame with feature values
- Parquet format for integration with downstream analysis

## Configuration

See `configs/data_pipeline.yaml` for full configuration. Key sections:

```yaml
tiling:
  tile_size: 256
  magnification: 20
  overlap: 0
  min_tissue_fraction: 0.5

stain_norm:
  method: macenko
  reference_slide: null

quality_filter:
  max_background_fraction: 0.8
  min_laplacian_variance: 15.0
  pen_mark_detection: true

dedup:
  hamming_threshold: 8

embeddings:
  backbone: resnet50
  batch_size: 64
  num_workers: 4
```

## End-to-End Pipeline

Use `scripts/run_preprocessing.py` to orchestrate the full pipeline:

```bash
python scripts/run_preprocessing.py \
    --config configs/data_pipeline.yaml \
    --slides data/raw/*.svs \
    --steps tiling stain_norm dedup embeddings \
    --resume \
    --skip-existing
```

**Features:**
- Automatic step orchestration
- Resume from last checkpoint
- Skip already-processed slides
- Progress tracking with tqdm
- Comprehensive logging

## Testing

Run tests with pytest:

```bash
pytest tests/test_data_pipeline.py -v

# Run specific test
pytest tests/test_data_pipeline.py::TestWSITiler::test_tissue_detection -v

# With coverage
pytest tests/test_data_pipeline.py --cov=src/data --cov-report=html
```

**Test Coverage:**
- Synthetic tile creation
- Tissue detection accuracy
- Stain normalization consistency
- Deduplication accuracy
- Embedding extraction and storage
- Batch processing and I/O

## Performance Notes

### Memory Usage
- WSI tiling: ~500MB per slide (typical SVS)
- Stain normalization: streaming, minimal memory
- Embedding extraction: batch processing, ~2GB per 100-tile batch (ResNet50)
- Zarr storage: ~2MB per patch (2048-dim embeddings)

### Speed
- WSI tiling: ~1-5 min per slide (10-20x magnification)
- Stain normalization: ~100 tiles/sec (single CPU)
- Deduplication: ~50 tiles/sec (hash computation)
- Embedding extraction: ~20 tiles/sec (single GPU, batch=64)

### Parallelization
- Tiling: `max_workers` threads for multi-slide processing
- Embedding extraction: DataLoader with `num_workers` processes
- Deduplication: Sequential (I/O bound, little benefit from parallelization)

## Dependencies

See `requirements.txt`. Key packages:

- **openslide-python** - WSI reading
- **scikit-image** - Image processing
- **torch/torchvision** - Deep learning backbone
- **imagehash** - Perceptual hashing
- **zarr** - Embedding storage
- **pyradiomics** - Radiomics features
- **pydicom** - DICOM reading

## Future Enhancements

1. **Advanced Models**
   - Integrate UNI2-h and TITAN foundations models
   - Fine-tuned domain-specific backbones

2. **Optimization**
   - Distributed processing (Dask, Ray)
   - Quantized embeddings for smaller storage
   - Progressive tile loading

3. **Validation**
   - Automatic quality metrics
   - Duplication detection visualization
   - Normalization effectiveness measures

4. **Integration**
   - Direct integration with downstream ML pipelines
   - Multi-omics feature fusion
   - Clinical outcome prediction models
