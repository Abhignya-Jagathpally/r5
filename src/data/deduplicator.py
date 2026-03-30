"""
Tile deduplication using perceptual hashing.

Identifies near-duplicate tiles using pHash and groups them into clusters,
keeping one representative tile per cluster.
"""

import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

import imagehash
from PIL import Image
import pandas as pd

logger = logging.getLogger(__name__)


class TileDeduplicator:
    """
    Detect and remove near-duplicate tiles using perceptual hashing.

    Uses imagehash library to compute perceptual hashes (pHash) and
    identifies clusters of near-duplicates based on Hamming distance.

    Parameters
    ----------
    hamming_threshold : int
        Maximum Hamming distance for considering tiles as near-duplicates (default: 8)
    hash_algorithm : str
        Hash algorithm to use ('phash', 'ahash', 'dhash', 'whash', default: 'phash')
    """

    def __init__(self, hamming_threshold: int = 8, hash_algorithm: str = "phash"):
        self.hamming_threshold = hamming_threshold
        self.hash_algorithm = hash_algorithm
        self.hashes = {}  # filename -> hash
        self.clusters = []  # List of clusters, each cluster is a set of filenames
        self.representative_tiles = {}  # cluster_id -> representative filename

        logger.info(
            f"TileDeduplicator initialized: threshold={hamming_threshold}, "
            f"algorithm={hash_algorithm}"
        )

    def _compute_hash(self, image_path: Path) -> imagehash.ImageHash:
        """Compute perceptual hash for an image."""
        try:
            image = Image.open(image_path)
            if self.hash_algorithm == "phash":
                return imagehash.phash(image)
            elif self.hash_algorithm == "ahash":
                return imagehash.ahash(image)
            elif self.hash_algorithm == "dhash":
                return imagehash.dhash(image)
            elif self.hash_algorithm == "whash":
                return imagehash.whash(image)
            else:
                raise ValueError(f"Unknown hash algorithm: {self.hash_algorithm}")
        except Exception as e:
            logger.error(f"Error computing hash for {image_path}: {e}")
            return None

    def build_index(self, tile_directory: str) -> Dict[str, imagehash.ImageHash]:
        """
        Compute and store hashes for all tiles in directory.

        Parameters
        ----------
        tile_directory : str
            Directory containing tile images

        Returns
        -------
        Dict[str, imagehash.ImageHash]
            Mapping of filename to hash
        """
        tile_path = Path(tile_directory)
        tile_files = list(tile_path.glob("*.png")) + list(tile_path.glob("*.jpg"))

        logger.info(f"Computing hashes for {len(tile_files)} tiles...")

        for i, tile_file in enumerate(tile_files):
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(tile_files)} tiles")

            hash_obj = self._compute_hash(tile_file)
            if hash_obj is not None:
                self.hashes[tile_file.name] = hash_obj

        logger.info(f"Hash index built: {len(self.hashes)} hashes computed")
        return self.hashes

    def _hamming_distance(
        self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash
    ) -> int:
        """Compute Hamming distance between two hashes."""
        return hash1 - hash2

    def find_clusters(self) -> List[Set[str]]:
        """
        Find clusters of near-duplicate tiles using union-find.

        Uses transitive clustering: if A~B and B~C, then A, B, and C are
        all in the same cluster, even if A and C are not directly similar.
        This is correct because near-duplicate is a transitive relation
        when the underlying images are truly duplicates with minor noise.

        Returns
        -------
        List[Set[str]]
            List of clusters, each containing filenames of similar tiles
        """
        if not self.hashes:
            logger.warning("No hashes in index; call build_index first")
            return []

        filenames = list(self.hashes.keys())

        # Union-Find data structure
        parent = {f: f for f in filenames}
        rank = {f: 0 for f in filenames}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        logger.info("Finding duplicate clusters (union-find)...")

        # Compare all pairs within threshold
        for i in range(len(filenames)):
            hash_i = self.hashes[filenames[i]]
            for j in range(i + 1, len(filenames)):
                hash_j = self.hashes[filenames[j]]
                distance = self._hamming_distance(hash_i, hash_j)
                if distance <= self.hamming_threshold:
                    union(filenames[i], filenames[j])

        # Collect clusters
        cluster_map = defaultdict(set)
        for f in filenames:
            cluster_map[find(f)].add(f)

        self.clusters = list(cluster_map.values())
        n_dup_clusters = sum(1 for c in self.clusters if len(c) > 1)
        logger.info(
            f"Found {len(self.clusters)} clusters "
            f"({n_dup_clusters} with duplicates, "
            f"{sum(len(c) - 1 for c in self.clusters if len(c) > 1)} total duplicates)"
        )
        return self.clusters

    def select_representatives(self) -> Dict[int, str]:
        """
        Select one representative tile from each cluster.

        Currently uses the first tile in each cluster (alphabetically).
        Can be extended to select by size, quality, or other criteria.

        Returns
        -------
        Dict[int, str]
            Mapping of cluster_id to representative filename
        """
        representatives = {}

        for cluster_id, cluster in enumerate(self.clusters):
            # Sort alphabetically and take first
            representative = sorted(list(cluster))[0]
            representatives[cluster_id] = representative

        self.representative_tiles = representatives
        logger.info(f"Selected {len(representatives)} representative tiles")
        return representatives

    def get_duplicates_to_remove(self) -> Set[str]:
        """
        Get set of filenames that should be removed (non-representatives).

        Returns
        -------
        Set[str]
            Set of filenames to remove
        """
        all_tiles = set()
        for cluster in self.clusters:
            all_tiles.update(cluster)

        representatives = set(self.representative_tiles.values())
        to_remove = all_tiles - representatives

        logger.info(f"Tiles to remove: {len(to_remove)} out of {len(all_tiles)}")
        return to_remove

    def deduplicate_directory(
        self, tile_directory: str, output_directory: str = None, remove_duplicates: bool = False
    ) -> Dict[str, any]:
        """
        Process directory to find and optionally remove duplicates.

        Parameters
        ----------
        tile_directory : str
            Directory containing tiles
        output_directory : str, optional
            Directory to copy unique tiles; if None, create in same location
        remove_duplicates : bool
            If True, remove duplicate files; if False, just identify them

        Returns
        -------
        Dict
            Statistics: {total_tiles, num_clusters, num_kept, num_removed}
        """
        # Build index
        self.build_index(tile_directory)

        # Find clusters
        self.find_clusters()

        # Select representatives
        self.select_representatives()

        # Get duplicates
        duplicates = self.get_duplicates_to_remove()

        stats = {
            "total_tiles": len(self.hashes),
            "num_clusters": len(self.clusters),
            "num_kept": len(self.representative_tiles),
            "num_removed": len(duplicates),
        }

        logger.info(
            f"Deduplication summary: {stats['total_tiles']} total, "
            f"{stats['num_clusters']} clusters, {stats['num_kept']} kept, "
            f"{stats['num_removed']} removed"
        )

        # Handle removal/copying
        tile_path = Path(tile_directory)

        if output_directory is not None:
            output_path = Path(output_directory)
            output_path.mkdir(parents=True, exist_ok=True)

            # Copy representatives
            for cluster_id, representative in self.representative_tiles.items():
                src = tile_path / representative
                dst = output_path / representative
                if src.exists():
                    import shutil
                    shutil.copy2(src, dst)

            logger.info(f"Unique tiles copied to {output_directory}")

        if remove_duplicates:
            for filename in duplicates:
                file_path = tile_path / filename
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed: {filename}")

            logger.info(f"Removed {len(duplicates)} duplicate files")

        return stats

    def get_cluster_report(self) -> pd.DataFrame:
        """
        Generate a detailed report of clusters.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: cluster_id, num_tiles, representative, tiles
        """
        report_data = []

        for cluster_id, cluster in enumerate(self.clusters):
            representative = self.representative_tiles.get(cluster_id)
            report_data.append(
                {
                    "cluster_id": cluster_id,
                    "num_tiles": len(cluster),
                    "representative": representative,
                    "tiles": "|".join(sorted(cluster)),
                }
            )

        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values("num_tiles", ascending=False)

        return report_df

    def save_report(self, output_path: str):
        """
        Save deduplication report to CSV.

        Parameters
        ----------
        output_path : str
            Path to save CSV report
        """
        report_df = self.get_cluster_report()
        report_df.to_csv(output_path, index=False)
        logger.info(f"Report saved to {output_path}")
