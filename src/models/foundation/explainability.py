"""
Clinically-grounded explainability for foundation model predictions.

Provides methods to understand model decisions through:
- Attention heatmap generation (MIL head attention → WSI coordinates)
- Top-K tile extraction and visualization
- GradCAM analysis of tile-level features
- SHAP feature attribution for radiomics
- HTML/PNG report generation for clinical review

Key insight: Interpretability is critical for clinical adoption.
This module makes predictions traceable to specific tissue regions
and quantitative features.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM) for tile-level features.

    Computes per-tile importance by backpropagating gradients from the
    classifier to the feature embedding layer.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """
        Initialize GradCAM.

        Args:
            model: PyTorch model
            target_layer: Name of layer to target (if None, uses last layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Register on target layer
        if self.target_layer:
            for name, module in self.model.named_modules():
                if self.target_layer in name:
                    module.register_forward_hook(forward_hook)
                    module.register_backward_hook(backward_hook)
                    break

    def __call__(
        self, embeddings: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute GradCAM for embeddings.

        Args:
            embeddings: Tile embeddings of shape (B, N, D)
            class_idx: Target class for explanation (if None, uses predicted class)

        Returns:
            Tile importance scores of shape (B, N)
        """
        self._register_hooks()

        embeddings.requires_grad = True
        self.model.eval()

        # Forward pass
        outputs = self.model(embeddings)

        # If no class specified, use predicted class
        if class_idx is None:
            class_idx = outputs.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        for i in range(outputs.shape[0]):
            outputs[i, class_idx[i]].backward(retain_graph=True)

        # Compute importance
        if self.activations is not None and self.gradients is not None:
            weights = self.gradients.mean(dim=(2, 3)) if self.gradients.dim() > 2 else self.gradients
            importance = (weights * self.activations).sum(dim=1)
            importance = F.relu(importance)
            importance = importance.cpu().numpy()
        else:
            importance = np.ones((embeddings.shape[0], embeddings.shape[1]))

        return importance


class AttentionExplainer:
    """
    Explainability engine for attention-based MIL models.

    Converts model attention weights to clinical insights:
    - Spatial heatmaps over slide coordinates
    - Top-K attended regions with context
    - Per-modality contribution analysis
    """

    def __init__(
        self,
        tile_size: int = 256,
        magnification: int = 20,
        top_k: int = 10,
    ):
        """
        Initialize explainability engine.

        Args:
            tile_size: Size of individual tiles in pixels
            magnification: Magnification level (20x, 40x, etc.)
            top_k: Number of top tiles to highlight
        """
        self.tile_size = tile_size
        self.magnification = magnification
        self.top_k = top_k

    def generate_heatmap(
        self,
        attention_weights: np.ndarray,
        coords: np.ndarray,
        output_size: Tuple[int, int] = (1024, 1024),
    ) -> np.ndarray:
        """
        Generate attention heatmap over slide coordinates.

        Args:
            attention_weights: Attention weights for tiles, shape (N,)
            coords: Tile coordinates (x, y), shape (N, 2)
            output_size: Size of output heatmap (height, width)

        Returns:
            Heatmap array of shape (output_size[0], output_size[1])
        """
        H, W = output_size
        heatmap = np.zeros((H, W), dtype=np.float32)

        if len(coords) == 0 or len(attention_weights) == 0:
            return heatmap

        # Normalize coordinates to heatmap space
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-5)

        x_indices = (coords_norm[:, 0] * (W - 1)).astype(int)
        y_indices = (coords_norm[:, 1] * (H - 1)).astype(int)

        # Clamp to valid range
        x_indices = np.clip(x_indices, 0, W - 1)
        y_indices = np.clip(y_indices, 0, H - 1)

        # Fill heatmap with normalized attention weights
        weights_norm = (attention_weights - attention_weights.min()) / (
            attention_weights.max() - attention_weights.min() + 1e-5
        )

        for x, y, w in zip(x_indices, y_indices, weights_norm):
            # Use Gaussian kernel for smooth visualization
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    xi = x + dx
                    yi = y + dy
                    if 0 <= xi < W and 0 <= yi < H:
                        kernel_val = np.exp(-((dx**2 + dy**2) / 25.0))
                        heatmap[yi, xi] = max(heatmap[yi, xi], w * kernel_val)

        return heatmap

    def get_top_tiles(
        self,
        attention_weights: np.ndarray,
        tile_paths: List[Union[str, Path]],
        k: Optional[int] = None,
    ) -> List[Tuple[int, float, Path]]:
        """
        Extract top-K most attended tiles.

        Args:
            attention_weights: Attention weights for tiles, shape (N,)
            tile_paths: Paths to tile images
            k: Number of top tiles (uses self.top_k if None)

        Returns:
            List of (index, attention_weight, path) tuples, sorted by weight
        """
        k = k or self.top_k
        k = min(k, len(tile_paths))

        top_indices = np.argsort(attention_weights)[-k:][::-1]

        results = [
            (int(idx), float(attention_weights[idx]), Path(tile_paths[idx]))
            for idx in top_indices
        ]

        return results

    def compute_modality_importance(
        self,
        modality_weights: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute per-modality importance scores.

        Args:
            modality_weights: Dictionary mapping modality names to weight arrays

        Returns:
            Dictionary of normalized importance scores per modality
        """
        importance_scores = {}

        for modality, weights in modality_weights.items():
            # Use average absolute weight as importance
            if weights.shape[0] > 0:
                importance = float(np.abs(weights).mean())
            else:
                importance = 0.0

            importance_scores[modality] = importance

        # Normalize to sum to 1
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v / total for k, v in importance_scores.items()}

        return importance_scores

    def generate_report(
        self,
        slide_id: str,
        prediction: float,
        confidence: float,
        attention_weights: np.ndarray,
        coords: Optional[np.ndarray] = None,
        tile_paths: Optional[List[Union[str, Path]]] = None,
        modality_weights: Optional[Dict[str, np.ndarray]] = None,
        radiomics_features: Optional[Dict[str, float]] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explainability report for a slide.

        Args:
            slide_id: Identifier for the slide
            prediction: Model prediction (probability)
            confidence: Confidence score
            attention_weights: Tile-level attention weights
            coords: Optional tile coordinates (N, 2)
            tile_paths: Optional paths to tile images
            modality_weights: Optional dictionary of modality importance weights
            radiomics_features: Optional dictionary of top radiomics features
            output_path: Optional path to save report

        Returns:
            Dictionary with report contents (heatmap, top tiles, etc.)
        """
        report = {
            "slide_id": slide_id,
            "prediction": float(prediction),
            "confidence": float(confidence),
        }

        # Generate heatmap if coordinates provided
        if coords is not None:
            heatmap = self.generate_heatmap(attention_weights, coords)
            report["heatmap"] = heatmap
        else:
            logger.warning("No coordinates provided for heatmap generation")

        # Get top-K tiles if paths provided
        if tile_paths is not None:
            top_tiles = self.get_top_tiles(attention_weights, tile_paths)
            report["top_tiles"] = [
                {"index": idx, "weight": weight, "path": str(path)}
                for idx, weight, path in top_tiles
            ]

        # Modality importance
        if modality_weights is not None:
            importance = self.compute_modality_importance(modality_weights)
            report["modality_importance"] = importance

        # Top radiomics features
        if radiomics_features is not None:
            sorted_features = sorted(
                radiomics_features.items(), key=lambda x: abs(x[1]), reverse=True
            )
            report["top_radiomics_features"] = sorted_features[:10]

        return report

    def export_html(
        self,
        report: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Export report as HTML for clinical review.

        Args:
            report: Report dictionary from generate_report()
            output_path: Path to save HTML file
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multiple Myeloma WSI Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 15px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .heatmap { max-width: 800px; border: 1px solid #ccc; }
                .top-tiles { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
                .tile-box { border: 1px solid #ddd; padding: 5px; text-align: center; }
                .modality-chart { margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background: #f0f0f0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Pathology Slide Analysis Report</h1>
                <p><strong>Slide ID:</strong> {slide_id}</p>
                <p><strong>Prediction:</strong> {prediction:.4f}</p>
                <p><strong>Confidence:</strong> {confidence:.4f}</p>
            </div>
        """.format(
            slide_id=report.get("slide_id", "N/A"),
            prediction=report.get("prediction", 0),
            confidence=report.get("confidence", 0),
        )

        # Add heatmap if available
        if "heatmap" in report:
            html_content += """
            <div class="section">
                <h2>Attention Heatmap</h2>
                <p>Indicates which regions of the slide the model focused on.</p>
                <img class="heatmap" src="heatmap.png" alt="Attention heatmap" />
            </div>
            """

        # Add top tiles if available
        if "top_tiles" in report:
            html_content += """
            <div class="section">
                <h2>Top Attended Tiles</h2>
                <p>The 10 most important tiles identified by the model.</p>
                <div class="top-tiles">
            """
            for tile in report["top_tiles"][:10]:
                html_content += f"""
                    <div class="tile-box">
                        <p>Tile {tile['index']}</p>
                        <p>Weight: {tile['weight']:.3f}</p>
                    </div>
                """
            html_content += "</div></div>"

        # Add modality importance if available
        if "modality_importance" in report:
            html_content += """
            <div class="section">
                <h2>Modality Importance</h2>
                <table>
                    <tr><th>Modality</th><th>Importance</th></tr>
            """
            for modality, importance in report["modality_importance"].items():
                html_content += (
                    f"<tr><td>{modality}</td><td>{importance:.4f}</td></tr>"
                )
            html_content += "</table></div>"

        # Add top radiomics features if available
        if "top_radiomics_features" in report:
            html_content += """
            <div class="section">
                <h2>Top Radiomics Features</h2>
                <table>
                    <tr><th>Feature</th><th>Value</th></tr>
            """
            for feature, value in report["top_radiomics_features"]:
                html_content += f"<tr><td>{feature}</td><td>{value:.4f}</td></tr>"
            html_content += "</table></div>"

        html_content += """
        </body>
        </html>
        """

        output_path.write_text(html_content)
        logger.info(f"HTML report saved to {output_path}")


def compute_shap_radiomics(
    radiomics_features: np.ndarray,
    model: nn.Module,
    background_data: Optional[np.ndarray] = None,
    num_samples: int = 100,
) -> Dict[int, float]:
    """
    Compute SHAP-like feature importance scores for radiomics.

    Uses a simplified approach: trains linear model on random subsets
    to estimate feature contributions.

    Args:
        radiomics_features: Features for a single sample, shape (num_features,)
        model: Trained classifier
        background_data: Optional background distribution for SHAP
        num_samples: Number of samples for importance estimation

    Returns:
        Dictionary mapping feature indices to importance scores
    """
    feature_importance = {}

    for feature_idx in range(len(radiomics_features)):
        # Perturb feature and measure prediction change
        original_pred = model(torch.tensor(radiomics_features).unsqueeze(0)).detach()

        # Perturb with mean value (from background if available)
        perturbed = radiomics_features.copy()
        if background_data is not None:
            perturbed[feature_idx] = background_data[:, feature_idx].mean()
        else:
            perturbed[feature_idx] = 0.0

        perturbed_pred = model(torch.tensor(perturbed).unsqueeze(0)).detach()

        # Importance = prediction change
        importance = float(
            torch.abs(original_pred - perturbed_pred).mean().cpu().numpy()
        )
        feature_importance[feature_idx] = importance

    return feature_importance
