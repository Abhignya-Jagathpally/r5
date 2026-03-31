"""
Comprehensive metrics suite for clinical imaging evaluation.

Computes classification, survival, and calibration metrics with bootstrap
confidence intervals and statistical testing.

All metrics return structured dicts with point estimates and 95% confidence intervals
via 1000 bootstrap iterations.
"""

from typing import Dict, Tuple, List, Optional, Any
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, auc, roc_curve, precision_recall_curve,
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, log_loss
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def stratified_bootstrap_indices(
    rng: np.random.RandomState,
    y: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Generate bootstrap indices that preserve class proportions.

    Resamples within each class independently, ensuring every bootstrap
    sample has the same class balance as the original data. This prevents
    degenerate bootstraps where a minority class is absent.

    Args:
        rng: Random state for reproducibility.
        y: Class labels array.
        n_samples: Total number of indices to return.

    Returns:
        Array of resampled indices.
    """
    classes, counts = np.unique(y, return_counts=True)
    fractions = counts / counts.sum()
    indices = []
    for cls, frac in zip(classes, fractions):
        cls_idx = np.where(y == cls)[0]
        n_cls = max(1, int(round(frac * n_samples)))
        indices.extend(rng.choice(cls_idx, size=n_cls, replace=True).tolist())
    return np.array(indices[:n_samples])


@dataclass
class MetricResult:
    """Container for a metric value with confidence interval."""
    value: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper
        }

    def __str__(self) -> str:
        return f"{self.value:.4f} [95% CI: {self.ci_lower:.4f}-{self.ci_upper:.4f}]"


class ClassificationMetrics:
    """
    Comprehensive classification metrics with bootstrap CIs.

    Computes AUROC, AUPRC, accuracy, balanced accuracy, F1 scores, sensitivity,
    specificity, PPV, NPV with 95% confidence intervals.
    """

    def __init__(self, bootstrap_iterations: int = 1000, random_seed: int = 42):
        """
        Initialize metrics calculator.

        Args:
            bootstrap_iterations (int): Number of bootstrap samples (default: 1000)
            random_seed (int): Random seed for reproducibility (default: 42)
        """
        self.bootstrap_iterations = bootstrap_iterations
        self.rng = np.random.RandomState(random_seed)

    def _patient_bootstrap_indices(
        self, patient_ids: np.ndarray, n_samples: int,
    ) -> np.ndarray:
        """Resample at the patient level, returning sample-level indices.

        This ensures bootstrap CIs respect patient-level structure and are
        not artificially tight due to within-patient correlation.
        """
        unique_patients = np.unique(patient_ids)
        resampled_patients = self.rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = []
        for pid in resampled_patients:
            patient_mask = np.where(patient_ids == pid)[0]
            indices.extend(patient_mask.tolist())
        return np.array(indices)

    @staticmethod
    def _validate_predictions(y_true, y_pred_proba, metric_name: str = "metric"):
        """Validate inputs before computing metrics."""
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            logger.warning(
                f"Cannot compute {metric_name}: only {len(unique_classes)} class(es) present. "
                f"Need at least 2. Returning NaN."
            )
            return False

        # FIXME: the minimum class size warning (5) is conservative for bootstrap
        # but 10 is probably more realistic for stable AUROC estimates
        min_class_size = min(np.sum(y_true == c) for c in unique_classes)
        if min_class_size < 5:
            logger.warning(
                f"{metric_name}: smallest class has only {min_class_size} samples. "
                f"Results may be unreliable (recommend >= 10 per class)."
            )

        # Check predicted probabilities are calibrated (not constant)
        if y_pred_proba is not None and len(y_pred_proba.shape) <= 1:
            if np.std(y_pred_proba) < 1e-8:
                logger.warning(
                    f"{metric_name}: predicted probabilities are near-constant "
                    f"(std={np.std(y_pred_proba):.2e}). Model may not be discriminative."
                )
            # Check probabilities are in [0, 1]
            if np.any(y_pred_proba < 0) or np.any(y_pred_proba > 1):
                logger.warning(
                    f"{metric_name}: predicted probabilities outside [0,1] range. "
                    f"Min={y_pred_proba.min():.4f}, Max={y_pred_proba.max():.4f}"
                )
        return True

    def compute_auroc(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        average: str = "macro",
        patient_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, MetricResult]:
        """
        Compute AUROC with bootstrap CI.

        Args:
            y_true (np.ndarray): True labels (shape: n_samples or n_samples x n_classes)
            y_pred_proba (np.ndarray): Predicted probabilities (shape: n_samples x n_classes)
            average (str): "macro", "micro", or None for per-class (default: "macro")

        Returns:
            Dict[str, MetricResult]: AUROC with CI and per-class if requested
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        if not self._validate_predictions(y_true, y_pred_proba, "AUROC"):
            return {"auroc": MetricResult(float("nan"), float("nan"), float("nan"))}

        # Binarize labels if multiclass
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 2:
            n_classes = y_pred_proba.shape[1]
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
        else:
            if len(y_pred_proba.shape) > 1:
                y_pred_proba = y_pred_proba[:, 1]
            y_true_bin = y_true

        # Compute main metric
        if average is None and len(y_pred_proba.shape) > 1:
            # Per-class AUROC
            auroc_values = []
            for i in range(y_pred_proba.shape[1]):
                try:
                    auroc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                    auroc_values.append(auroc)
                except ValueError:
                    auroc_values.append(np.nan)
            main_auroc = np.nanmean(auroc_values)
        else:
            main_auroc = roc_auc_score(y_true, y_pred_proba, average=average)

        # Bootstrap CI (patient-level > stratified > simple)
        bootstrap_aurocs = []
        for _ in range(self.bootstrap_iterations):
            if patient_ids is not None:
                idx = self._patient_bootstrap_indices(np.asarray(patient_ids), len(y_true))
            else:
                idx = stratified_bootstrap_indices(self.rng, y_true, len(y_true))
            try:
                if average is None and len(y_pred_proba.shape) > 1:
                    aurocs = []
                    for i in range(y_pred_proba.shape[1]):
                        aurocs.append(
                            roc_auc_score(y_true_bin[idx, i], y_pred_proba[idx, i])
                        )
                    bootstrap_aurocs.append(np.nanmean(aurocs))
                else:
                    bootstrap_aurocs.append(
                        roc_auc_score(y_true[idx], y_pred_proba[idx], average=average)
                    )
            except ValueError:
                continue

        bootstrap_aurocs = np.array(bootstrap_aurocs)
        ci_lower = np.percentile(bootstrap_aurocs, 2.5)
        ci_upper = np.percentile(bootstrap_aurocs, 97.5)

        result = {
            "auroc": MetricResult(main_auroc, ci_lower, ci_upper)
        }

        # Per-class if requested
        if average is None and len(y_pred_proba.shape) > 1:
            for i, auroc in enumerate(auroc_values):
                result[f"auroc_class_{i}"] = MetricResult(auroc, np.nan, np.nan)

        return result

    def compute_auprc(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        average: str = "macro"
    ) -> MetricResult:
        """
        Compute Area Under Precision-Recall Curve with bootstrap CI.

        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            average (str): "macro" or "micro" (default: "macro")

        Returns:
            MetricResult: AUPRC with CI
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_pred_proba = y_pred_proba[:, 1]

        # Compute main metric
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        main_auprc = auc(recall, precision)

        # Bootstrap CI
        bootstrap_auprcs = []
        for _ in range(self.bootstrap_iterations):
            idx = self.rng.choice(len(y_true), size=len(y_true), replace=True)
            precision, recall, _ = precision_recall_curve(
                y_true[idx], y_pred_proba[idx]
            )
            bootstrap_auprcs.append(auc(recall, precision))

        bootstrap_auprcs = np.array(bootstrap_auprcs)
        ci_lower = np.percentile(bootstrap_auprcs, 2.5)
        ci_upper = np.percentile(bootstrap_auprcs, 97.5)

        return MetricResult(main_auprc, ci_lower, ci_upper)

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
        """
        Compute accuracy with bootstrap CI.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            MetricResult: Accuracy with CI
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        main_acc = accuracy_score(y_true, y_pred)

        bootstrap_accs = []
        for _ in range(self.bootstrap_iterations):
            idx = self.rng.choice(len(y_true), size=len(y_true), replace=True)
            bootstrap_accs.append(accuracy_score(y_true[idx], y_pred[idx]))

        bootstrap_accs = np.array(bootstrap_accs)
        ci_lower = np.percentile(bootstrap_accs, 2.5)
        ci_upper = np.percentile(bootstrap_accs, 97.5)

        return MetricResult(main_acc, ci_lower, ci_upper)

    def compute_balanced_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> MetricResult:
        """
        Compute balanced accuracy (average of per-class recall) with bootstrap CI.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            MetricResult: Balanced accuracy with CI
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        main_bacc = balanced_accuracy_score(y_true, y_pred)

        bootstrap_baccs = []
        for _ in range(self.bootstrap_iterations):
            idx = self.rng.choice(len(y_true), size=len(y_true), replace=True)
            bootstrap_baccs.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))

        bootstrap_baccs = np.array(bootstrap_baccs)
        ci_lower = np.percentile(bootstrap_baccs, 2.5)
        ci_upper = np.percentile(bootstrap_baccs, 97.5)

        return MetricResult(main_bacc, ci_lower, ci_upper)

    def compute_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro"
    ) -> MetricResult:
        """
        Compute F1 score with bootstrap CI.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            average (str): "macro", "micro", or "weighted" (default: "macro")

        Returns:
            MetricResult: F1 score with CI
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        main_f1 = f1_score(y_true, y_pred, average=average)

        bootstrap_f1s = []
        for _ in range(self.bootstrap_iterations):
            idx = self.rng.choice(len(y_true), size=len(y_true), replace=True)
            bootstrap_f1s.append(f1_score(y_true[idx], y_pred[idx], average=average))

        bootstrap_f1s = np.array(bootstrap_f1s)
        ci_lower = np.percentile(bootstrap_f1s, 2.5)
        ci_upper = np.percentile(bootstrap_f1s, 97.5)

        return MetricResult(main_f1, ci_lower, ci_upper)

    def compute_sensitivity_specificity_ppv_npv(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, MetricResult]:
        """
        Compute sensitivity, specificity, PPV, NPV with bootstrap CIs.

        For binary classification only.

        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted binary labels

        Returns:
            Dict[str, MetricResult]: Dictionary with sensitivity, specificity, PPV, NPV
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(np.unique(y_true)) != 2:
            raise ValueError("This metric requires binary labels")

        # Minimum class size check
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        if n_pos < 5 or n_neg < 5:
            logger.warning(
                f"Sensitivity/specificity: small class sizes (pos={n_pos}, neg={n_neg}). "
                f"Results may be unreliable. Recommend >= 10 per class."
            )

        # Confusion matrix: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Bootstrap
        def compute_metrics(y_t, y_p):
            cm = confusion_matrix(y_t, y_p, labels=[0, 1])
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            return (
                tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                tn / (tn + fn) if (tn + fn) > 0 else 0.0
            )

        bootstrap_metrics = []
        for _ in range(self.bootstrap_iterations):
            idx = self.rng.choice(len(y_true), size=len(y_true), replace=True)
            bootstrap_metrics.append(compute_metrics(y_true[idx], y_pred[idx]))

        bootstrap_metrics = np.array(bootstrap_metrics)

        result = {}
        metrics_names = ["sensitivity", "specificity", "ppv", "npv"]
        main_values = [sensitivity, specificity, ppv, npv]

        for i, name in enumerate(metrics_names):
            ci_lower = np.percentile(bootstrap_metrics[:, i], 2.5)
            ci_upper = np.percentile(bootstrap_metrics[:, i], 97.5)
            result[name] = MetricResult(main_values[i], ci_lower, ci_upper)

        return result

    def compute_all_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, MetricResult]:
        """
        Compute all classification metrics.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities

        Returns:
            Dict[str, MetricResult]: Dictionary of all metrics with CIs
        """
        metrics = {}

        # Probability-based metrics
        metrics.update(self.compute_auroc(y_true, y_pred_proba))
        metrics["auprc"] = self.compute_auprc(y_true, y_pred_proba)

        # Hard prediction metrics
        metrics["accuracy"] = self.compute_accuracy(y_true, y_pred)
        metrics["balanced_accuracy"] = self.compute_balanced_accuracy(y_true, y_pred)
        metrics["f1_macro"] = self.compute_f1(y_true, y_pred, average="macro")
        metrics["f1_weighted"] = self.compute_f1(y_true, y_pred, average="weighted")

        # Binary-specific metrics
        if len(np.unique(y_true)) == 2:
            metrics.update(self.compute_sensitivity_specificity_ppv_npv(
                y_true, y_pred
            ))

        return metrics


class SurvivalMetrics:
    """
    Survival analysis metrics: concordance index, time-dependent AUROC, Brier score.

    Requires event indicators and time-to-event data.
    """

    def __init__(self, bootstrap_iterations: int = 1000, random_seed: int = 42):
        """
        Initialize survival metrics calculator.

        Args:
            bootstrap_iterations (int): Number of bootstrap samples (default: 1000)
            random_seed (int): Random seed for reproducibility (default: 42)
        """
        self.bootstrap_iterations = bootstrap_iterations
        self.rng = np.random.RandomState(random_seed)

    # TODO: concordance_index_censored from sksurv doesn't handle tied
    # predictions well — consider switching to lifelines' implementation
    # (we partially do this already via the try/except below)
    def concordance_index(
        self,
        event_indicator: np.ndarray,
        time_to_event: np.ndarray,
        predicted_risk: np.ndarray
    ) -> MetricResult:
        """
        Compute Harrell's concordance index (C-index).

        C-index measures the probability that predictions are concordant with
        actual outcomes (values 0.5-1.0, where 0.5 is random and 1.0 is perfect).

        Args:
            event_indicator (np.ndarray): Binary event indicator (1=event, 0=censored)
            time_to_event (np.ndarray): Time to event or censoring
            predicted_risk (np.ndarray): Predicted risk scores

        Returns:
            MetricResult: C-index with CI
        """
        event_indicator = np.asarray(event_indicator, dtype=bool)
        time_to_event = np.asarray(time_to_event, dtype=float)
        predicted_risk = np.asarray(predicted_risk, dtype=float)

        def compute_cindex(event, time, risk):
            """Harrell's C-index with proper censoring handling.

            A pair (i, j) is comparable only if:
            - time[i] < time[j] AND event[i] == True (i experienced the event)
            Censored subjects at the shorter time are NOT comparable because
            we don't know their true event time.

            Uses lifelines if available for O(n log n); falls back to
            corrected O(n^2) implementation.
            """
            try:
                from lifelines.utils import concordance_index as ci_fn
                # lifelines: higher predicted = longer survival
                return ci_fn(time, -risk, event)
            except ImportError:
                pass

            concordant = 0
            comparable = 0

            for i in range(len(time)):
                if not event[i]:
                    # Censored at time[i] — can only compare if j has event
                    # before censoring time, but that's handled by j's iteration
                    continue
                for j in range(len(time)):
                    if i == j:
                        continue
                    # Pair is comparable only if subject i had an event
                    # and time[i] < time[j] (or time[i] == time[j] and j is censored)
                    if time[i] < time[j]:
                        comparable += 1
                        if risk[i] > risk[j]:
                            concordant += 1
                        elif risk[i] == risk[j]:
                            concordant += 0.5
                    elif time[i] == time[j] and not event[j]:
                        # Tie in time: i had event, j censored — comparable
                        comparable += 1
                        if risk[i] > risk[j]:
                            concordant += 1
                        elif risk[i] == risk[j]:
                            concordant += 0.5

            return concordant / comparable if comparable > 0 else 0.5

        main_cindex = compute_cindex(event_indicator, time_to_event, predicted_risk)

        # Stratified bootstrap (preserve event/censored ratio)
        bootstrap_cindexes = []
        for _ in range(self.bootstrap_iterations):
            idx = stratified_bootstrap_indices(
                self.rng, event_indicator.astype(int), len(time_to_event)
            )
            cindex = compute_cindex(
                event_indicator[idx],
                time_to_event[idx],
                predicted_risk[idx]
            )
            bootstrap_cindexes.append(cindex)

        bootstrap_cindexes = np.array(bootstrap_cindexes)
        ci_lower = np.percentile(bootstrap_cindexes, 2.5)
        ci_upper = np.percentile(bootstrap_cindexes, 97.5)

        return MetricResult(main_cindex, ci_lower, ci_upper)

    def integrated_brier_score(
        self,
        event_indicator: np.ndarray,
        time_to_event: np.ndarray,
        predicted_risk: np.ndarray
    ) -> MetricResult:
        """
        Compute integrated Brier score (IBS).

        Measures prediction error, averaged over time. Values 0-0.25 (lower is better).

        Args:
            event_indicator (np.ndarray): Binary event indicator
            time_to_event (np.ndarray): Time to event or censoring
            predicted_risk (np.ndarray): Predicted probability of event

        Returns:
            MetricResult: IBS with CI
        """
        event_indicator = np.asarray(event_indicator, dtype=bool)
        time_to_event = np.asarray(time_to_event, dtype=float)
        predicted_risk = np.clip(np.asarray(predicted_risk, dtype=float), 0, 1)

        unique_times = np.unique(time_to_event[event_indicator])
        if len(unique_times) == 0:
            return MetricResult(0.0, np.nan, np.nan)

        # Simple IBS calculation
        brier_scores = []
        for t in unique_times:
            # At time t: is event observed?
            at_risk = time_to_event >= t
            if np.sum(at_risk) == 0:
                continue

            # Brier score at this time
            at_time = (time_to_event == t) & event_indicator
            surv_prob = 1 - predicted_risk[at_risk]
            brier = np.mean((surv_prob - at_time[at_risk]) ** 2)
            brier_scores.append(brier)

        main_ibs = np.mean(brier_scores) if brier_scores else 0.0

        # Bootstrap
        bootstrap_ibs = []
        for _ in range(self.bootstrap_iterations):
            idx = self.rng.choice(len(time_to_event), size=len(time_to_event), replace=True)
            brier_scores_boot = []
            for t in unique_times:
                at_risk = time_to_event[idx] >= t
                if np.sum(at_risk) == 0:
                    continue
                at_time = (time_to_event[idx] == t) & event_indicator[idx]
                surv_prob = 1 - predicted_risk[idx[at_risk]]
                brier = np.mean((surv_prob - at_time[at_risk]) ** 2)
                brier_scores_boot.append(brier)

            bootstrap_ibs.append(np.mean(brier_scores_boot) if brier_scores_boot else 0.0)

        bootstrap_ibs = np.array(bootstrap_ibs)
        ci_lower = np.percentile(bootstrap_ibs, 2.5)
        ci_upper = np.percentile(bootstrap_ibs, 97.5)

        return MetricResult(main_ibs, ci_lower, ci_upper)

    def compute_all_survival_metrics(
        self,
        event_indicator: np.ndarray,
        time_to_event: np.ndarray,
        predicted_risk: np.ndarray
    ) -> Dict[str, MetricResult]:
        """
        Compute all survival metrics.

        Args:
            event_indicator (np.ndarray): Binary event indicator
            time_to_event (np.ndarray): Time to event or censoring
            predicted_risk (np.ndarray): Predicted risk scores

        Returns:
            Dict[str, MetricResult]: Dictionary of all metrics with CIs
        """
        return {
            "c_index": self.concordance_index(
                event_indicator, time_to_event, predicted_risk
            ),
            "ibs": self.integrated_brier_score(
                event_indicator, time_to_event, predicted_risk
            )
        }


class CalibrationMetrics:
    """
    Calibration metrics: expected calibration error, calibration curves.

    Measures whether predicted probabilities match empirical frequencies.
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize calibration metrics.

        Args:
            n_bins (int): Number of bins for calibration curve (default: 10)
        """
        self.n_bins = n_bins

    def expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE = sum(|accuracy(bin) - confidence(bin)| * |bin|) / N

        Args:
            y_true (np.ndarray): True binary labels
            y_pred_proba (np.ndarray): Predicted probabilities (for positive class)

        Returns:
            float: ECE value (0-1, lower is better)
        """
        y_true = np.asarray(y_true, dtype=bool)
        y_pred_proba = np.clip(np.asarray(y_pred_proba, dtype=float), 0, 1)

        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
            if np.sum(mask) == 0:
                continue

            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_pred_proba[mask])
            ece += np.abs(bin_accuracy - bin_confidence) * np.sum(mask) / len(y_true)

        return ece

    def calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve data points.

        Returns mean predicted probability and fraction of positives in each bin.

        Args:
            y_true (np.ndarray): True binary labels
            y_pred_proba (np.ndarray): Predicted probabilities

        Returns:
            Tuple[np.ndarray, np.ndarray]: (mean_predicted_prob, fraction_positives)
        """
        y_true = np.asarray(y_true, dtype=bool)
        y_pred_proba = np.clip(np.asarray(y_pred_proba, dtype=float), 0, 1)

        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        mean_probs = []
        fractions = []

        for i in range(self.n_bins):
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
            if np.sum(mask) == 0:
                continue

            mean_probs.append(np.mean(y_pred_proba[mask]))
            fractions.append(np.mean(y_true[mask]))

        return np.array(mean_probs), np.array(fractions)

    def compute_all_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute all calibration metrics.

        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities

        Returns:
            Dict[str, Any]: Dictionary with ECE and calibration curve points
        """
        ece = self.expected_calibration_error(y_true, y_pred_proba)
        mean_probs, fractions = self.calibration_curve(y_true, y_pred_proba)

        return {
            "ece": ece,
            "mean_predicted_prob": mean_probs,
            "fraction_positive": fractions
        }


class BootstrapMetrics:
    """
    Utility class for computing bootstrap confidence intervals for any metric.

    Provides flexibility for custom metrics not in standard classes.
    """

    def __init__(self, bootstrap_iterations: int = 1000, random_seed: int = 42):
        """
        Initialize bootstrap metrics.

        Args:
            bootstrap_iterations (int): Number of bootstrap samples (default: 1000)
            random_seed (int): Random seed for reproducibility (default: 42)
        """
        self.bootstrap_iterations = bootstrap_iterations
        self.rng = np.random.RandomState(random_seed)

    def compute_ci(
        self,
        metric_func,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> MetricResult:
        """
        Compute metric with bootstrap CI.

        Args:
            metric_func: Function that computes metric(y_true, y_pred, **kwargs)
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predictions
            **kwargs: Additional arguments to pass to metric_func

        Returns:
            MetricResult: Metric value with CI
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        main_value = metric_func(y_true, y_pred, **kwargs)

        bootstrap_values = []
        for _ in range(self.bootstrap_iterations):
            idx = self.rng.choice(len(y_true), size=len(y_true), replace=True)
            value = metric_func(y_true[idx], y_pred[idx], **kwargs)
            bootstrap_values.append(value)

        bootstrap_values = np.array(bootstrap_values)
        ci_lower = np.percentile(bootstrap_values, 2.5)
        ci_upper = np.percentile(bootstrap_values, 97.5)

        return MetricResult(main_value, ci_lower, ci_upper)
