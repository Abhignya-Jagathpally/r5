"""
PyTorch Dataset for Multiple Instance Learning (MIL).

Reads from Zarr embedding store, handles variable-length bags,
supports sampling strategies (all, random, top-k).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import zarr
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class MILDataset(Dataset):
    """PyTorch Dataset for MIL from pre-extracted embeddings.

    Reads bag of embeddings from Zarr store and returns:
    (embeddings, coordinates, label, slide_id)

    Supports variable-length bags with padding/sampling.
    """

    def __init__(
        self,
        zarr_path: str,
        slide_ids: List[str],
        labels: List[int],
        max_patches: Optional[int] = None,
        sampling_strategy: str = "all",
        seed: int = 42,
    ):
        """Initialize MIL dataset.

        Args:
            zarr_path: Path to Zarr embedding store
            slide_ids: List of slide IDs
            labels: Corresponding labels
            max_patches: Maximum patches per slide (for padding/sampling)
            sampling_strategy: 'all', 'random', or 'topk'
            seed: Random seed
        """
        self.zarr_path = zarr_path
        self.slide_ids = slide_ids
        self.labels = labels
        self.max_patches = max_patches
        self.sampling_strategy = sampling_strategy
        self.seed = seed

        np.random.seed(seed)

        # Open Zarr store
        self.store = zarr.open(zarr_path, mode="r")

        # Verify all slides are in store and get stats
        self.valid_indices = []
        self.patch_counts = []

        for i, slide_id in enumerate(slide_ids):
            try:
                slide_group = self.store[slide_id]
                n_patches = len([k for k in slide_group.keys() if k.endswith(".npy")])

                if n_patches > 0:
                    self.valid_indices.append(i)
                    self.patch_counts.append(n_patches)
            except (KeyError, Exception):
                continue

        # Infer max_patches if not provided
        if self.max_patches is None and len(self.patch_counts) > 0:
            self.max_patches = max(self.patch_counts)

        if len(self.valid_indices) == 0:
            raise ValueError("No valid slides found in Zarr store")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """Get bag of embeddings and metadata.

        Args:
            idx: Index

        Returns:
            (embeddings_tensor, coordinates_tensor, label, slide_id)
            - embeddings_tensor: (num_patches, embedding_dim)
            - coordinates_tensor: (num_patches, 2) - (x, y) tile coordinates
        """
        real_idx = self.valid_indices[idx]
        slide_id = self.slide_ids[real_idx]
        label = self.labels[real_idx]

        # Load embeddings
        slide_group = self.store[slide_id]

        embeddings_list = []
        coordinates_list = []
        tile_ids = []

        for key in sorted(slide_group.keys()):
            if key.endswith(".npy"):
                emb = slide_group[key][:]
                embeddings_list.append(emb)

                # Extract coordinates from tile_id (format: tile_x_y.npy)
                tile_id = key.replace(".npy", "")
                tile_ids.append(tile_id)

                # Parse coordinates (assuming format: tile_X_Y or similar)
                parts = tile_id.split("_")
                if len(parts) >= 3:
                    try:
                        x, y = int(parts[1]), int(parts[2])
                        coordinates_list.append([x, y])
                    except (ValueError, IndexError):
                        coordinates_list.append([0, 0])
                else:
                    coordinates_list.append([0, 0])

        embeddings = np.array(embeddings_list)  # (num_patches, embedding_dim)
        coordinates = np.array(coordinates_list)  # (num_patches, 2)

        # Apply sampling strategy
        if self.sampling_strategy == "all":
            if self.max_patches is not None and len(embeddings) > self.max_patches:
                # Random sample if too many
                indices = np.random.choice(
                    len(embeddings), self.max_patches, replace=False
                )
                embeddings = embeddings[indices]
                coordinates = coordinates[indices]
            elif self.max_patches is not None and len(embeddings) < self.max_patches:
                # Pad with zeros if too few
                pad_size = self.max_patches - len(embeddings)
                embeddings = np.vstack([
                    embeddings,
                    np.zeros((pad_size, embeddings.shape[1]))
                ])
                coordinates = np.vstack([
                    coordinates,
                    np.zeros((pad_size, 2))
                ])

        elif self.sampling_strategy == "random":
            if self.max_patches is not None:
                n_samples = min(self.max_patches, len(embeddings))
                indices = np.random.choice(len(embeddings), n_samples, replace=False)
                embeddings = embeddings[indices]
                coordinates = coordinates[indices]

        elif self.sampling_strategy == "topk":
            # Keep top-k patches (would need attention weights, not implemented here)
            if self.max_patches is not None and len(embeddings) > self.max_patches:
                indices = np.random.choice(
                    len(embeddings), self.max_patches, replace=False
                )
                embeddings = embeddings[indices]
                coordinates = coordinates[indices]

        # Convert to tensors
        embeddings_tensor = torch.FloatTensor(embeddings)
        coordinates_tensor = torch.FloatTensor(coordinates)

        return embeddings_tensor, coordinates_tensor, label, slide_id


def collate_fn_mil(batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for MIL DataLoader.

    Handles variable-length bags.

    Args:
        batch: List of samples from MILDataset

    Returns:
        (embeddings_batch, labels_batch)
        - embeddings_batch: (batch_size, max_patches, embedding_dim) padded
        - labels_batch: (batch_size,)
    """
    embeddings_list, _, labels, _ = zip(*batch)

    # Pad to maximum length in batch
    max_len = max(e.shape[0] for e in embeddings_list)
    embedding_dim = embeddings_list[0].shape[1]

    embeddings_padded = []
    for e in embeddings_list:
        if e.shape[0] < max_len:
            pad_size = max_len - e.shape[0]
            e = torch.cat([e, torch.zeros(pad_size, embedding_dim)], dim=0)
        embeddings_padded.append(e)

    embeddings_batch = torch.stack(embeddings_padded)
    labels_batch = torch.LongTensor(labels)

    return embeddings_batch, labels_batch


def create_mil_dataloader(
    zarr_path: str,
    slide_ids: List[str],
    labels: List[int],
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    max_patches: Optional[int] = None,
    sampling_strategy: str = "all",
) -> DataLoader:
    """Create MIL DataLoader.

    Args:
        zarr_path: Path to Zarr embedding store
        slide_ids: List of slide IDs
        labels: Corresponding labels
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        max_patches: Maximum patches per slide
        sampling_strategy: Sampling strategy

    Returns:
        PyTorch DataLoader
    """
    dataset = MILDataset(
        zarr_path=zarr_path,
        slide_ids=slide_ids,
        labels=labels,
        max_patches=max_patches,
        sampling_strategy=sampling_strategy,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_mil,
    )


class SplitsManager:
    """Manage train/val/test splits from CSV file.

    CSV format:
    slide_id,label,split
    slide_1,0,train
    slide_2,1,val
    ...
    """

    def __init__(self, csv_path: str):
        """Initialize splits manager.

        Args:
            csv_path: Path to splits CSV file
        """
        self.df = pd.read_csv(csv_path)
        self.validate()

    def validate(self):
        """Validate CSV format."""
        required_cols = {"slide_id", "label", "split"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

    def get_split(
        self,
        split_name: str,
    ) -> Tuple[List[str], List[int]]:
        """Get slide IDs and labels for a split.

        Args:
            split_name: 'train', 'val', or 'test'

        Returns:
            (slide_ids, labels)
        """
        subset = self.df[self.df["split"] == split_name]
        return subset["slide_id"].tolist(), subset["label"].tolist()

    def get_dataloaders(
        self,
        zarr_path: str,
        batch_size: int = 4,
        num_workers: int = 0,
        max_patches: Optional[int] = None,
    ) -> Dict[str, DataLoader]:
        """Create DataLoaders for all splits.

        Args:
            zarr_path: Path to Zarr store
            batch_size: Batch size
            num_workers: Number of workers
            max_patches: Maximum patches

        Returns:
            Dictionary with dataloaders for each split
        """
        dataloaders = {}

        for split_name in ["train", "val", "test"]:
            if split_name not in self.df["split"].values:
                continue

            slide_ids, labels = self.get_split(split_name)

            dl = create_mil_dataloader(
                zarr_path=zarr_path,
                slide_ids=slide_ids,
                labels=labels,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                max_patches=max_patches,
                sampling_strategy="all",
            )

            dataloaders[split_name] = dl

        return dataloaders
