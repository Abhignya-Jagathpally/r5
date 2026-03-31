"""
Publication-quality visualization for clinical imaging evaluation.

Generates figures suitable for peer-reviewed papers with:
- Proper DPI (300), fonts, and no chartjunk
- Bootstrap confidence interval bands
- Consistent color palettes
- Both PNG and PDF export formats
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

logger = logging.getLogger(__name__)

# Consistent publication style for all plots
PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

import matplotlib
matplotlib.rcParams.update(PLOT_STYLE)

# Colorblind-friendly palette
COLORBLIND_PALETTE = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'purple': '#CA9161',
    'gray': '#949494',
}


class ROCCurveVisualizer:
    """ROC curve visualization with confidence intervals."""

    def __init__(self, figsize: Tuple[int, int] = (8, 6)):
        """
        Initialize visualizer.

        Args:
            figsize (Tuple[int, int]): Figure size (width, height)
        """
        self.figsize = figsize

    def plot(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        ci_lower: Optional[np.ndarray] = None,
        ci_upper: Optional[np.ndarray] = None,
        auroc: Optional[float] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot ROC curve with optional CI band.

        Args:
            y_true (np.ndarray): True binary labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Model name for legend
            ci_lower (Optional[np.ndarray]): Lower CI band
            ci_upper (Optional[np.ndarray]): Upper CI band
            auroc (Optional[float]): AUROC value to display
            output_path (Optional[Union[str, Path]]): Path to save figure

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auroc_value = auc(fpr, tpr) if auroc is None else auroc

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot main ROC curve
        ax.plot(
            fpr, tpr,
            color=COLORBLIND_PALETTE['blue'],
            lw=2.5,
            label=f'{model_name} (AUROC={auroc_value:.3f})'
        )

        # Plot CI band if provided
        if ci_lower is not None and ci_upper is not None:
            ax.fill_between(
                fpr, ci_lower, ci_upper,
                alpha=0.2,
                color=COLORBLIND_PALETTE['blue'],
                label='95% CI'
            )

        # Diagonal reference
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        if output_path:
            self._save_figure(fig, output_path)

        return fig, ax

    def plot_multiple(
        self,
        results: Dict[str, Dict[str, np.ndarray]],
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot multiple ROC curves on same axes.

        Args:
            results (Dict[str, Dict[str, np.ndarray]]): Dictionary mapping model names
                to {'y_true', 'y_pred_proba', 'auroc'} dicts
            output_path (Optional[Union[str, Path]]): Path to save figure

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = list(COLORBLIND_PALETTE.values())

        for idx, (model_name, data) in enumerate(results.items()):
            y_true = data['y_true']
            y_pred_proba = data['y_pred_proba']
            auroc = data.get('auroc')

            if len(y_pred_proba.shape) > 1:
                y_pred_proba = y_pred_proba[:, 1]

            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auroc_value = auc(fpr, tpr) if auroc is None else auroc

            color = colors[idx % len(colors)]
            ax.plot(
                fpr, tpr,
                color=color,
                lw=2.5,
                label=f'{model_name} (AUROC={auroc_value:.3f})'
            )

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        if output_path:
            self._save_figure(fig, output_path)

        return fig, ax

    @staticmethod
    def _save_figure(fig: plt.Figure, output_path: Union[str, Path]) -> None:
        """Save figure to PNG and PDF."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        base_path = str(output_path).rsplit('.', 1)[0]
        fig.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{base_path}.pdf', dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {base_path}.png and {base_path}.pdf")


class PrecisionRecallVisualizer:
    """Precision-Recall curve visualization."""

    def __init__(self, figsize: Tuple[int, int] = (8, 6)):
        """Initialize visualizer."""
        self.figsize = figsize

    def plot(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        auprc: Optional[float] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot Precision-Recall curve.

        Args:
            y_true (np.ndarray): True binary labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Model name for legend
            auprc (Optional[float]): AUPRC value to display
            output_path (Optional[Union[str, Path]]): Path to save figure

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        auprc_value = auc(recall, precision) if auprc is None else auprc

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(
            recall, precision,
            color=COLORBLIND_PALETTE['blue'],
            lw=2.5,
            label=f'{model_name} (AUPRC={auprc_value:.3f})'
        )

        # Baseline (prevalence)
        baseline = np.mean(y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', lw=1, alpha=0.5, label='Baseline')

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        if output_path:
            ROCCurveVisualizer._save_figure(fig, output_path)

        return fig, ax


class SurvivalCurveVisualizer:
    """Kaplan-Meier survival curves with risk stratification."""

    def __init__(self, figsize: Tuple[int, int] = (10, 7)):
        """Initialize visualizer."""
        self.figsize = figsize

    def plot(
        self,
        time_to_event: np.ndarray,
        event_indicator: np.ndarray,
        risk_groups: np.ndarray,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot Kaplan-Meier survival curves by risk group.

        Args:
            time_to_event (np.ndarray): Time to event or censoring
            event_indicator (np.ndarray): Binary event indicator
            risk_groups (np.ndarray): Risk group assignment (e.g., high/low)
            output_path (Optional[Union[str, Path]]): Path to save figure

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        unique_groups = np.unique(risk_groups)
        colors = list(COLORBLIND_PALETTE.values())

        for idx, group in enumerate(unique_groups):
            mask = risk_groups == group
            times = np.sort(time_to_event[mask])
            events = event_indicator[mask]

            # Simple KM curve calculation
            n_at_risk = []
            survival_prob = []
            time_points = []

            for t in times[events]:
                at_risk = np.sum(time_to_event[mask] >= t)
                n_events = np.sum((time_to_event[mask] == t) & events)

                if at_risk > 0:
                    surv = np.prod([1 - n_events / at_risk])
                    n_at_risk.append(at_risk)
                    survival_prob.append(surv)
                    time_points.append(t)

            if time_points:
                ax.step(
                    time_points, survival_prob,
                    where='post',
                    color=colors[idx % len(colors)],
                    lw=2.5,
                    label=f'Risk Group {group} (n={np.sum(mask)})'
                )

        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title('Kaplan-Meier Survival Curves', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        if output_path:
            ROCCurveVisualizer._save_figure(fig, output_path)

        return fig, ax


class ConfusionMatrixVisualizer:
    """Confusion matrix visualization."""

    def __init__(self, figsize: Tuple[int, int] = (8, 7)):
        """Initialize visualizer."""
        self.figsize = figsize

    def plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True,
        class_names: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot confusion matrix heatmap.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            normalize (bool): Normalize by true label frequency
            class_names (Optional[List[str]]): Class names
            output_path (Optional[Union[str, Path]]): Path to save figure

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            cmap_label = 'Normalized Count'
        else:
            fmt = 'd'
            cmap_label = 'Count'

        n_classes = cm.shape[0]
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]

        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(cm, cmap='Blues', aspect='auto')

        # Labels
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                 rotation_mode='anchor')

        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                value = cm[i, j]
                text = ax.text(j, i, f'{value:{fmt}}',
                              ha='center', va='center',
                              color='white' if value > cm.max() / 2 else 'black',
                              fontsize=11, fontweight='bold')

        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cmap_label, fontsize=11)

        if output_path:
            ROCCurveVisualizer._save_figure(fig, output_path)

        return fig, ax


class CalibrationVisualizer:
    """Calibration plot visualization."""

    def __init__(self, figsize: Tuple[int, int] = (8, 6)):
        """Initialize visualizer."""
        self.figsize = figsize

    def plot(
        self,
        mean_predicted_prob: np.ndarray,
        fraction_positive: np.ndarray,
        model_name: str = "Model",
        ece: Optional[float] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot calibration curve.

        Args:
            mean_predicted_prob (np.ndarray): Mean predicted probabilities per bin
            fraction_positive (np.ndarray): Fraction of positives per bin
            model_name (str): Model name for legend
            ece (Optional[float]): Expected calibration error
            output_path (Optional[Union[str, Path]]): Path to save figure

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Perfect Calibration')

        # Model calibration
        label = model_name
        if ece is not None:
            label += f' (ECE={ece:.3f})'

        ax.plot(
            mean_predicted_prob, fraction_positive,
            'o-',
            color=COLORBLIND_PALETTE['blue'],
            lw=2.5, markersize=8,
            label=label
        )

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Calibration Plot', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        if output_path:
            ROCCurveVisualizer._save_figure(fig, output_path)

        return fig, ax


class TrainingCurveVisualizer:
    """Training history visualization."""

    def __init__(self, figsize: Tuple[int, int] = (12, 5)):
        """Initialize visualizer."""
        self.figsize = figsize

    def plot(
        self,
        history: Dict[str, List[float]],
        metrics: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot training curves.

        Args:
            history (Dict[str, List[float]]): History dict with 'train_loss', 'val_loss', etc.
            metrics (Optional[List[str]]): Metrics to plot (e.g., ['loss', 'accuracy'])
            output_path (Optional[Union[str, Path]]): Path to save figure

        Returns:
            Tuple[plt.Figure, np.ndarray]: Figure and axes objects
        """
        if metrics is None:
            metrics = [k.replace('train_', '').replace('val_', '') for k in history.keys()]
            metrics = list(set(metrics))

        fig, axes = plt.subplots(1, len(metrics), figsize=self.figsize)

        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in enumerate(metrics):
            axis = axes[ax]

            train_key = f'train_{metric}'
            val_key = f'val_{metric}'

            if train_key in history:
                axis.plot(history[train_key], label='Train',
                         color=COLORBLIND_PALETTE['blue'], lw=2)
            if val_key in history:
                axis.plot(history[val_key], label='Validation',
                         color=COLORBLIND_PALETTE['orange'], lw=2)

            axis.set_xlabel('Epoch', fontsize=11)
            axis.set_ylabel(metric.capitalize(), fontsize=11)
            axis.set_title(f'{metric.capitalize()} vs Epoch', fontsize=12, fontweight='bold')
            axis.legend(loc='best', frameon=True)
            axis.grid(True, alpha=0.3, linestyle='--')

        if output_path:
            ROCCurveVisualizer._save_figure(fig, output_path)

        return fig, axes


class AttentionHeatmapVisualizer:
    """Attention heatmap overlay on WSI thumbnails."""

    def __init__(self):
        """Initialize visualizer."""
        pass

    def overlay_attention(
        self,
        wsi_thumbnail: np.ndarray,
        attention_map: np.ndarray,
        alpha: float = 0.4,
        cmap: str = 'hot',
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Overlay attention heatmap on WSI thumbnail.

        Args:
            wsi_thumbnail (np.ndarray): WSI thumbnail image (H x W x 3)
            attention_map (np.ndarray): Attention weights (H x W)
            alpha (float): Heatmap transparency (default: 0.4)
            cmap (str): Colormap name (default: 'hot')
            output_path (Optional[Union[str, Path]]): Path to save figure

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Display WSI
        ax.imshow(wsi_thumbnail)

        # Overlay attention
        if attention_map.shape != wsi_thumbnail.shape[:2]:
            # Resize attention map
            from scipy.ndimage import zoom
            scale_factors = (
                wsi_thumbnail.shape[0] / attention_map.shape[0],
                wsi_thumbnail.shape[1] / attention_map.shape[1]
            )
            attention_map = zoom(attention_map, scale_factors, order=1)

        im = ax.imshow(attention_map, cmap=cmap, alpha=alpha)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=11)

        ax.set_title('Attention Heatmap Overlay', fontsize=14, fontweight='bold')
        ax.axis('off')

        if output_path:
            ROCCurveVisualizer._save_figure(fig, output_path)

        return fig, ax


class FeatureImportanceVisualizer:
    """Feature importance bar plots."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """Initialize visualizer."""
        self.figsize = figsize

    def plot(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        top_n: int = 20,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot feature importance.

        Args:
            feature_names (List[str]): Feature names
            importance_scores (np.ndarray): Importance values
            top_n (int): Number of top features to show (default: 20)
            output_path (Optional[Union[str, Path]]): Path to save figure

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[::-1][:top_n]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.barh(sorted_names, sorted_scores, color=COLORBLIND_PALETTE['blue'])

        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')

        if output_path:
            ROCCurveVisualizer._save_figure(fig, output_path)

        return fig, ax
