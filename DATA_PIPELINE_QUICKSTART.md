# Data Pipeline Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.data import WSITiler, StainNormalizer, TileDeduplicator, EmbeddingStore; print('All modules imported successfully')"
```

## Quick Usage Examples

### 1. Tiling WSI Slides

```python
from src.data import WSITiler

tiler = WSITiler(
    tile_size=256,
    magnification=20,
    min_tissue_fraction=0.5,
    output_dir="data/tiles"
)

# Process single slide
tiles_data = tiler.process_slide("myslide.svs", "slide_001")

# Process multiple slides in parallel
manifest_df = tiler.process_slides(
    slide_paths=["slide1.svs", "slide2.svs", "slide3.svs"],
    slide_ids=["s001", "s002", "s003"],
    max_workers=4
)

print(f"Extracted {len(manifest_df)} tiles")
print(manifest_df.head())
```

**Output:** Tiles saved to `data/tiles/` with manifest CSV

### 2. Stain Normalization

```python
from src.data import StainNormalizer

normalizer = StainNormalizer(
    method="macenko",  # or "reinhard"
    max_background_fraction=0.8,
    min_laplacian_variance=15.0
)

# Normalize directory of tiles
stats = normalizer.process_tile_directory(
    input_dir="data/tiles",
    output_dir="data/normalized",
    reference_image="data/tiles/reference_tile.png"  # Optional
)

print(f"Normalized: {stats['normalized']}, Filtered: {stats['filtered']}")
```

**Output:** Normalized tiles in `data/normalized/`

### 3. Deduplication

```python
from src.data import TileDeduplicator

dedup = TileDeduplicator(
    hamming_threshold=8,
    hash_algorithm="phash"
)

stats = dedup.deduplicate_directory(
    tile_directory="data/normalized",
    output_directory="data/deduplicated",
    remove_duplicates=False  # Set True to delete duplicates
)

print(f"Kept: {stats['num_kept']}, Removed: {stats['num_removed']}")

# Generate detailed report
dedup.save_report("dedup_report.csv")
```

**Output:** Unique tiles in `data/deduplicated/`, report in `dedup_report.csv`

### 4. Embedding Extraction

```python
from src.data import EmbeddingExtractor, EmbeddingStore
import numpy as np

# Extract embeddings
extractor = EmbeddingExtractor(
    backbone="resnet50",
    device="cuda",  # Use "cpu" if no GPU
    batch_size=64
)

embeddings, filenames = extractor.extract_from_directory(
    tile_directory="data/deduplicated",
    batch_size=64,
    num_workers=4
)

print(f"Extracted embeddings shape: {embeddings.shape}")

# Store in Zarr format
store = EmbeddingStore("data/embeddings")

# For each slide, add embeddings and coordinates
coordinates = np.random.randint(0, 1000, (len(embeddings), 2))
store.add_slide_embeddings(
    slide_id="slide_001",
    embeddings=embeddings,
    coordinates=coordinates,
    label="MM",  # Classification label
    split="train"
)

store.save_metadata("data/embeddings/metadata.parquet")

print("Embeddings stored in Zarr format")
```

**Output:** Zarr store at `data/embeddings/`, metadata in parquet

### 5. Load Embeddings for Training

```python
from src.data import PatchDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = PatchDataset(
    zarr_store="data/embeddings",
    metadata_path="data/embeddings/metadata.parquet",
    split="train"  # or "val", "test"
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True
)

# Training loop
for batch in dataloader:
    embeddings = batch["embedding"]      # (batch_size, 2048)
    coordinates = batch["coordinate"]    # (batch_size, 2)
    labels = batch["label"]              # (batch_size,)

    # Use in your model...
    pass
```

## End-to-End Pipeline

Run the complete pipeline with a single command:

```bash
python scripts/run_preprocessing.py \
    --config configs/data_pipeline.yaml \
    --slides data/raw/*.svs \
    --steps tiling quality_filter stain_norm dedup embeddings \
    --resume \
    --skip-existing
```

## Configuration

Edit `configs/data_pipeline.yaml` to customize:

```yaml
tiling:
  tile_size: 256
  magnification: 20
  min_tissue_fraction: 0.5

stain_norm:
  method: macenko

embeddings:
  backbone: resnet50
  batch_size: 64
```

## Testing

Run unit tests to verify installation:

```bash
# Run all tests
pytest tests/test_data_pipeline.py -v

# Run specific test class
pytest tests/test_data_pipeline.py::TestWSITiler -v

# Run with coverage
pytest tests/test_data_pipeline.py --cov=src/data --cov-report=html
```

## Common Issues

### ImportError: No module named 'openslide'

```bash
# Install openslide system library
# Ubuntu/Debian:
sudo apt-get install libopenslide0

# macOS:
brew install openslide

# Then install Python binding
pip install openslide-python
```

### CUDA/GPU issues

Use CPU mode if GPU unavailable:

```python
extractor = EmbeddingExtractor(device="cpu")
```

### Out of Memory

Reduce batch size:

```python
embeddings = extractor.extract_from_directory(
    tile_directory="data/deduplicated",
    batch_size=32  # Reduce from 64
)
```

## Performance Tuning

### Faster Processing
- Increase `max_workers` in tiling (4-8 threads)
- Increase `num_workers` in DataLoader (4-8 processes)
- Use GPU for embeddings (device="cuda")
- Reduce image resolution if acceptable

### Lower Memory Usage
- Reduce `batch_size` in embedding extraction
- Use quantized embeddings (future)
- Stream processing instead of batch

### Example: Small Dataset
```python
# For testing/development
tiler = WSITiler(tile_size=128, overlap=32)  # Smaller tiles
extractor = EmbeddingExtractor(device="cpu", batch_size=8)
```

## Output Structure

After complete pipeline:

```
data/
├── raw/                          # Original WSI slides
├── tiles/
│   ├── *.png                     # Extracted tiles
│   └── tile_manifest.csv         # Tile metadata
├── normalized/
│   └── *.png                     # Normalized tiles
├── deduplicated/
│   └── *.png                     # Unique tiles after dedup
└── embeddings/
    ├── .zgroup                   # Zarr group file
    ├── slide_001/                # Per-slide data
    │   ├── embeddings            # Zarr dataset (N×2048)
    │   ├── coordinates           # Zarr dataset (N×2)
    │   └── attention_scores      # Zarr dataset (N×1)
    └── metadata.parquet          # Slide-level metadata
```

## Next Steps

1. **Integrate with downstream models** - Use PatchDataset for training
2. **Add custom backbones** - Modify EmbeddingExtractor for UNI2-h, TITAN
3. **Multi-modal fusion** - Combine with radiomics features
4. **Clinical integration** - Link embeddings to outcomes, genomics

## Documentation

- **Full API documentation:** See `src/data/README.md`
- **Configuration reference:** See `configs/data_pipeline.yaml`
- **Code examples:** See `tests/test_data_pipeline.py`

## Support

For issues or questions:
1. Check logs in `logs/pipeline.log`
2. Review test cases in `tests/test_data_pipeline.py`
3. Consult docstrings in source code
4. See full documentation in `src/data/README.md`
