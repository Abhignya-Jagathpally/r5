"""
Mean pooling baseline for slide-level classification.

Simplest possible baseline: load pre-extracted embeddings, mean-pool by slide,
train logistic regression / SVM / random forest. This is the sanity check
baseline that all other MIL models must beat.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class MeanPoolBaseline:
    """Mean pooling baseline for slide-level classification.

    Workflow:
    1. Load pre-extracted tile embeddings from Zarr store
    2. Mean-pool all tile embeddings per slide
    3. Train simple classifiers (logistic regression, SVM, random forest)
    4. Report AUROC, accuracy, F1
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        classifiers: List[str] = None,
    ):
        """Initialize baseline.

        Args:
            embedding_dim: Dimension of pre-extracted embeddings
            classifiers: List of classifier names to train
                         ['logistic_regression', 'svm', 'random_forest']
        """
        self.embedding_dim = embedding_dim
        self.classifiers = classifiers or ["logistic_regression", "svm", "random_forest"]

        # Storage for fitted classifiers
        self.models = {}
        self.scaler = StandardScaler()
        self.fitted = False

    def load_embeddings_from_zarr(
        self,
        zarr_path: str,
        slide_ids: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Load pre-extracted embeddings from Zarr store.

        Expected structure:
        zarr_path/
            slide_id_1/
                tile_0.npy
                tile_1.npy
                ...
            slide_id_2/
                ...

        Args:
            zarr_path: Path to Zarr embedding store
            slide_ids: List of slide IDs to load

        Returns:
            (mean_pooled_embeddings, valid_slide_ids)
        """
        store = zarr.open(zarr_path, mode="r")
        embeddings_list = []
        valid_slide_ids = []

        for slide_id in slide_ids:
            try:
                slide_group = store[slide_id]

                # Collect all tile embeddings for this slide
                tile_embeddings = []
                for key in slide_group.keys():
                    if key.endswith(".npy"):
                        tile_emb = slide_group[key][:]  # Shape: (embedding_dim,)
                        tile_embeddings.append(tile_emb)

                if tile_embeddings:
                    # Mean pool
                    slide_embedding = np.mean(tile_embeddings, axis=0)
                    embeddings_list.append(slide_embedding)
                    valid_slide_ids.append(slide_id)
            except (KeyError, Exception):
                continue

        embeddings = np.array(embeddings_list)  # (num_slides, embedding_dim)
        return embeddings, valid_slide_ids

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Fit all classifiers.

        Args:
            X: Mean-pooled embeddings (num_samples, embedding_dim)
            y: Labels (num_samples,)
            verbose: Print training progress

        Returns:
            Dictionary with training metrics for each classifier
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        results = {}

        for clf_name in self.classifiers:
            if clf_name == "logistic_regression":
                clf = LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                    n_jobs=-1,
                )
            elif clf_name == "svm":
                clf = SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=42,
                )
            elif clf_name == "random_forest":
                clf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                raise ValueError(f"Unknown classifier: {clf_name}")

            clf.fit(X_scaled, y)
            self.models[clf_name] = clf

            # Training accuracy
            preds = clf.predict(X_scaled)
            acc = accuracy_score(y, preds)
            results[f"{clf_name}_train_acc"] = acc

            if verbose:
                print(f"{clf_name:25s}: Train Acc = {acc:.4f}")

        self.fitted = True
        return results

    def predict(
        self,
        X: np.ndarray,
        classifier: str = "logistic_regression",
    ) -> np.ndarray:
        """Predict on new data.

        Args:
            X: Mean-pooled embeddings (num_samples, embedding_dim)
            classifier: Which fitted classifier to use

        Returns:
            Predicted labels (num_samples,)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        if classifier not in self.models:
            raise ValueError(f"Classifier {classifier} not available")

        X_scaled = self.scaler.transform(X)
        return self.models[classifier].predict(X_scaled)

    def predict_proba(
        self,
        X: np.ndarray,
        classifier: str = "logistic_regression",
    ) -> np.ndarray:
        """Predict probabilities.

        Args:
            X: Mean-pooled embeddings (num_samples, embedding_dim)
            classifier: Which fitted classifier to use

        Returns:
            Predicted probabilities (num_samples, num_classes)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        if classifier not in self.models:
            raise ValueError(f"Classifier {classifier} not available")

        X_scaled = self.scaler.transform(X)
        return self.models[classifier].predict_proba(X_scaled)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifier: str = "logistic_regression",
    ) -> Dict[str, float]:
        """Evaluate classifier on test set.

        Args:
            X: Mean-pooled embeddings (num_samples, embedding_dim)
            y: True labels (num_samples,)
            classifier: Which fitted classifier to use

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        preds = self.predict(X, classifier=classifier)
        probs = self.predict_proba(X, classifier=classifier)

        # Handle binary vs multi-class
        y_unique = np.unique(y)
        is_binary = len(y_unique) == 2

        metrics = {
            "accuracy": accuracy_score(y, preds),
            "f1": f1_score(y, preds, average="weighted"),
        }

        # AUROC only for binary
        if is_binary:
            metrics["auroc"] = roc_auc_score(y, probs[:, 1])
        else:
            metrics["auroc"] = roc_auc_score(
                y, probs, multi_class="ovr", labels=y_unique
            )

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y, preds)

        return metrics

    def evaluate_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all fitted classifiers.

        Args:
            X: Mean-pooled embeddings (num_samples, embedding_dim)
            y: True labels (num_samples,)

        Returns:
            Dictionary with metrics for each classifier
        """
        all_results = {}

        for clf_name in self.classifiers:
            if clf_name in self.models:
                metrics = self.evaluate(X, y, classifier=clf_name)
                all_results[clf_name] = metrics

        return all_results

    def get_feature_importance(
        self,
        classifier: str = "random_forest",
        top_k: int = 20,
    ) -> Optional[np.ndarray]:
        """Get feature importance (only for tree-based models).

        Args:
            classifier: Which fitted classifier to use
            top_k: Return top-k important features

        Returns:
            Feature importances or None if not available
        """
        if classifier not in self.models:
            return None

        clf = self.models[classifier]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            top_indices = np.argsort(importances)[-top_k:][::-1]
            return importances[top_indices]

        return None
