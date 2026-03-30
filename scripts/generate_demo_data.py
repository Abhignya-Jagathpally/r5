#!/usr/bin/env python3
"""Generate synthetic demo data for end-to-end pipeline testing.

Creates realistic synthetic tiles, embeddings, metadata, and labels
to allow the full pipeline to run without real WSI data.
"""

import logging
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_demo_data(
    output_dir: str = "data",
    n_patients: int = 60,
    n_slides_per_patient: int = 1,
    n_tiles_per_slide: int = 100,
    embedding_dim: int = 2048,
    seed: int = 42,
) -> Dict[str, str]:
    """Generate synthetic data for all pipeline stages.

    Args:
        output_dir: Root data directory.
        n_patients: Number of synthetic patients.
        n_slides_per_patient: Slides per patient.
        n_tiles_per_slide: Tiles per slide.
        embedding_dim: Embedding vector dimension.
        seed: Random seed.

    Returns:
        Dict of paths to generated files.
    """
    rng = np.random.RandomState(seed)
    out = Path(output_dir)

    # Create directories
    for d in ["raw", "tiles", "normalized", "deduplicated", "embeddings", "features", "splits"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    n_slides = n_patients * n_slides_per_patient
    n_tiles_total = n_slides * n_tiles_per_slide

    logger.info(f"Generating demo data: {n_patients} patients, {n_slides} slides, {n_tiles_total} tiles")

    # ── Patient metadata ──────────────────────────────────────────────

    patient_ids = [f"MM_{i:04d}" for i in range(n_patients)]
    labels = rng.binomial(1, 0.4, n_patients)  # ~40% positive (t(11;14))
    ages = rng.normal(65, 12, n_patients).clip(30, 90).astype(int)
    survival_times = rng.exponential(36, n_patients).clip(1, 120)
    events = rng.binomial(1, 0.6, n_patients)
    iss_stages = rng.choice(["I", "II", "III"], n_patients, p=[0.3, 0.4, 0.3])

    patient_df = pd.DataFrame({
        "patient_id": patient_ids,
        "label": labels,
        "age": ages,
        "sex": rng.choice(["M", "F"], n_patients),
        "iss_stage": iss_stages,
        "survival_time": survival_times.round(1),
        "event": events,
        "cytogenetic_risk": rng.choice(["standard", "high", "unknown"], n_patients, p=[0.5, 0.3, 0.2]),
    })
    patient_path = out / "patient_metadata.csv"
    patient_df.to_csv(patient_path, index=False)
    logger.info(f"Patient metadata: {patient_path} ({n_patients} patients)")

    # ── Slide metadata ────────────────────────────────────────────────

    slide_records = []
    for pid in patient_ids:
        for s in range(n_slides_per_patient):
            slide_id = f"{pid}_slide{s:02d}"
            slide_records.append({
                "patient_id": pid,
                "slide_id": slide_id,
                "label": patient_df[patient_df.patient_id == pid].label.values[0],
            })

    slide_df = pd.DataFrame(slide_records)
    slide_path = out / "slide_metadata.csv"
    slide_df.to_csv(slide_path, index=False)
    logger.info(f"Slide metadata: {slide_path} ({n_slides} slides)")

    # ── Tile manifest ─────────────────────────────────────────────────

    tile_records = []
    for _, slide_row in slide_df.iterrows():
        for t in range(n_tiles_per_slide):
            tile_records.append({
                "slide_id": slide_row.slide_id,
                "patient_id": slide_row.patient_id,
                "tile_id": f"{slide_row.slide_id}_tile{t:04d}",
                "x": rng.randint(0, 50000),
                "y": rng.randint(0, 50000),
                "tissue_fraction": rng.uniform(0.5, 1.0),
                "laplacian_var": rng.exponential(50),
                "label": slide_row.label,
            })

    tile_df = pd.DataFrame(tile_records)
    tile_manifest_path = out / "tiles" / "tile_manifest.csv"
    tile_df.to_csv(tile_manifest_path, index=False)
    logger.info(f"Tile manifest: {tile_manifest_path} ({n_tiles_total} tiles)")

    # ── Synthetic tile images (small numpy arrays) ────────────────────

    tiles_dir = out / "tiles" / "images"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Save a few representative tiles as .npy (not full images to save space)
    for slide_id in slide_df.slide_id.unique()[:5]:  # Just 5 slides worth
        slide_tile_dir = tiles_dir / slide_id
        slide_tile_dir.mkdir(exist_ok=True)
        for t in range(min(10, n_tiles_per_slide)):
            tile = rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            np.save(slide_tile_dir / f"tile_{t:04d}.npy", tile)

    logger.info(f"Sample tile images saved to {tiles_dir}")

    # ── Synthetic embeddings (main pipeline input) ────────────────────

    embeddings_path = out / "embeddings" / "embeddings.npz"

    all_embeddings = {}
    all_coords = {}

    for _, slide_row in slide_df.iterrows():
        slide_id = slide_row.slide_id
        label = slide_row.label

        # Generate embeddings with class-conditional signal
        base = rng.randn(n_tiles_per_slide, embedding_dim).astype(np.float32)
        if label == 1:
            # Add a subtle signal for positive class
            signal = rng.randn(embedding_dim).astype(np.float32) * 0.3
            base += signal[np.newaxis, :]

        all_embeddings[slide_id] = base
        all_coords[slide_id] = rng.randint(0, 50000, (n_tiles_per_slide, 2))

    np.savez_compressed(embeddings_path, **{f"emb_{k}": v for k, v in all_embeddings.items()})
    np.savez_compressed(
        out / "embeddings" / "coords.npz",
        **{f"coord_{k}": v for k, v in all_coords.items()},
    )
    logger.info(f"Embeddings: {embeddings_path} ({embedding_dim}-dim)")

    # ── Synthetic radiomics features ──────────────────────────────────

    n_radiomics_features = 93  # Typical pyradiomics count
    feature_names = [f"radiomics_{i:03d}" for i in range(n_radiomics_features)]

    radiomics_data = rng.randn(n_slides, n_radiomics_features).astype(np.float32)
    # Add class-conditional signal
    for i, row in enumerate(slide_df.itertuples()):
        if row.label == 1:
            radiomics_data[i, :10] += 0.5  # Signal in first 10 features

    radiomics_df = pd.DataFrame(radiomics_data, columns=feature_names)
    radiomics_df.insert(0, "slide_id", slide_df.slide_id.values)
    radiomics_df.insert(1, "patient_id", slide_df.patient_id.values)
    radiomics_df.insert(2, "label", slide_df.label.values)

    radiomics_path = out / "features" / "radiomics_features.csv"
    radiomics_df.to_csv(radiomics_path, index=False)
    logger.info(f"Radiomics features: {radiomics_path} ({n_radiomics_features} features)")

    # ── Train/Val/Test splits (patient-level) ─────────────────────────

    perm = rng.permutation(n_patients)
    n_train = int(0.7 * n_patients)
    n_val = int(0.15 * n_patients)

    train_patients = set(np.array(patient_ids)[perm[:n_train]])
    val_patients = set(np.array(patient_ids)[perm[n_train:n_train + n_val]])
    test_patients = set(np.array(patient_ids)[perm[n_train + n_val:]])

    splits = []
    for _, row in slide_df.iterrows():
        if row.patient_id in train_patients:
            split = "train"
        elif row.patient_id in val_patients:
            split = "val"
        else:
            split = "test"
        splits.append({"slide_id": row.slide_id, "patient_id": row.patient_id, "split": split, "label": row.label})

    splits_df = pd.DataFrame(splits)
    splits_path = out / "splits" / "splits.csv"
    splits_df.to_csv(splits_path, index=False)
    logger.info(
        f"Splits: {splits_path} "
        f"(train={len(train_patients)}, val={len(val_patients)}, test={len(test_patients)})"
    )

    # ── Summary ───────────────────────────────────────────────────────

    paths = {
        "patient_metadata": str(patient_path),
        "slide_metadata": str(slide_path),
        "tile_manifest": str(tile_manifest_path),
        "embeddings": str(embeddings_path),
        "radiomics_features": str(radiomics_path),
        "splits": str(splits_path),
        "data_dir": str(out),
    }

    summary_path = out / "demo_data_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_patients": n_patients,
            "n_slides": n_slides,
            "n_tiles_total": n_tiles_total,
            "embedding_dim": embedding_dim,
            "positive_rate": float(labels.mean()),
            "event_rate": float(events.mean()),
            "paths": paths,
            "seed": seed,
        }, f, indent=2)

    logger.info(f"Demo data summary: {summary_path}")
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    paths = generate_demo_data()
    print(f"\nDemo data generated. Run pipeline with:")
    print(f"  python3 main.py --config configs/pipeline.json --stages all --data-dir data")
