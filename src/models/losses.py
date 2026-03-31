"""
Loss functions for classical baselines and MIL models.

Includes:
- Standard cross-entropy
- Cox partial likelihood (survival)
- Instance-level clustering loss (CLAM)
- Focal loss (class imbalance)
- Smooth top-k loss (MIL)
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Thin wrapper around nn.CrossEntropyLoss for API consistency."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__(weight=weight, reduction=reduction)


class CoxPartialLikelihoodLoss(nn.Module):
    """Cox Partial Likelihood loss for survival analysis.

    Computes negative log-likelihood for Cox proportional hazards model.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        risk_scores: torch.Tensor,
        event_times: torch.Tensor,
        event_indicators: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Cox partial likelihood loss.

        Args:
            risk_scores: Predicted log hazard scores (batch_size,)
            event_times: Time to event (batch_size,)
            event_indicators: 1 if event, 0 if censored (batch_size,)

        Returns:
            Scalar loss value
        """
        # TODO: this O(n) loop is fine for typical cohort sizes (< 1000)
        # but should vectorize if we ever use this with large survival datasets
        # Sort by event time (descending: latest events first)
        sorted_indices = torch.argsort(event_times, descending=True)
        sorted_times = event_times[sorted_indices]
        sorted_scores = risk_scores[sorted_indices]
        sorted_events = event_indicators[sorted_indices]

        # Risk set: for each event at time t_i, the at-risk set includes all
        # subjects with event time >= t_i. With descending sort, the at-risk
        # set at position i is the REVERSE cumsum (from end to start).
        exp_scores = sorted_scores.exp()
        cumsum_scores = torch.cumsum(exp_scores.flip(0), dim=0).flip(0)

        # Loss for each event
        loss = 0.0
        n_events = 0

        for i in range(len(sorted_indices)):
            if sorted_events[i] == 1:  # Only for events, not censored
                # Risk at time t_i relative to all at-risk subjects
                risk_score = sorted_scores[i]
                risk_set_sum = cumsum_scores[i]

                # Negative log partial likelihood for this event
                loss += -risk_score + torch.log(risk_set_sum + 1e-8)
                n_events += 1

        # Normalize by number of events
        if n_events > 0:
            loss = loss / n_events
        else:
            loss = torch.tensor(0.0, device=risk_scores.device)

        return loss


class InstanceClusteringLoss(nn.Module):
    """Instance-level clustering constraint loss (for CLAM).

    Encourages instance-level predictions to match bag-level pseudo-labels.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        instance_logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute instance clustering loss.

        Args:
            instance_logits: Instance-level predictions (num_instances, num_classes)
            pseudo_labels: Pseudo-labels for instances (num_instances,)

        Returns:
            Scalar loss value
        """
        return self.criterion(instance_logits, pseudo_labels)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Reference: "Focal Loss for Dense Object Detection"
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """Initialize focal loss.

        Args:
            alpha: Weighting factor in range (0,1) to balance
                   positive vs negative examples
            gamma: Exponent of the modulating factor (1 - p_t) ^ gamma
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model outputs (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)

        Returns:
            Scalar loss value
        """
        # Get class probabilities
        p = F.softmax(logits, dim=1)

        # Get class log probabilities
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Get probability of the true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        if self.reduction == "none":
            return focal_loss
        elif self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class SmoothTopKLoss(nn.Module):
    """Smooth Top-K Loss for Multiple Instance Learning.

    Approximates top-k selection with smooth softmax to make it differentiable.
    """

    def __init__(
        self,
        k: int = 5,
        temperature: float = 1.0,
    ):
        """Initialize smooth top-k loss.

        Args:
            k: Number of top instances to select
            temperature: Temperature for softmax sharpening
        """
        super().__init__()
        # FIXME: SmoothTopK gradient can explode with temperature < 0.1
        # Clamping helps but may bias the optimization. Needs investigation.
        self.k = k
        self.temperature = temperature

    def forward(
        self,
        instance_scores: torch.Tensor,
        bag_label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute smooth top-k loss.

        Args:
            instance_scores: Instance predictions (num_instances, num_classes)
            bag_label: Bag-level label (scalar)

        Returns:
            Scalar loss value
        """
        # Get probabilities for the bag label class
        probs = F.softmax(instance_scores, dim=1)
        label_probs = probs[:, bag_label]  # (num_instances,)

        # Smooth approximation of top-k
        # Use softmax to create differentiable top-k selection
        smooth_weights = F.softmax(label_probs / self.temperature, dim=0)

        # Sum of top instances (soft selection)
        topk_prob = torch.sum(smooth_weights * label_probs)

        # Loss: negative log probability of bag label
        loss = -torch.log(topk_prob + 1e-8)

        return loss


class WeightedFocalLoss(nn.Module):
    """Weighted Focal Loss combining class weights and focal loss.

    Useful for imbalanced datasets with multiple classes.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        """Initialize weighted focal loss.

        Args:
            class_weights: Weight for each class
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.register_buffer("class_weights", class_weights) if class_weights is not None else None
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted focal loss.

        Args:
            logits: Model outputs
            targets: Ground truth labels

        Returns:
            Scalar loss value
        """
        # Standard focal loss
        p = F.softmax(logits, dim=1)
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")

        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation-like tasks.

    Can be used for multi-class classification as well.
    """

    def __init__(self, smooth: float = 1.0):
        """Initialize dice loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dice loss.

        Args:
            logits: Model outputs (batch_size, num_classes, ...)
            targets: Ground truth labels (batch_size, ...)

        Returns:
            Scalar loss value
        """
        probabilities = F.softmax(logits, dim=1)

        # Flatten
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])
        targets_one_hot = targets_one_hot.transpose(1, -1).float()

        probabilities_flat = probabilities.view(probabilities.shape[0], probabilities.shape[1], -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)

        # Compute Dice coefficient for each class
        intersection = (probabilities_flat * targets_flat).sum(dim=-1)
        union = probabilities_flat.sum(dim=-1) + targets_flat.sum(dim=-1)

        dice = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        return dice.mean()


def compute_class_weights(
    labels: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse class weights for imbalanced datasets.

    Args:
        labels: Array of class labels
        device: torch.device

    Returns:
        Tensor of class weights (num_classes,)
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    weights = {}
    for cls, count in zip(unique, counts):
        weights[cls] = total / (len(unique) * count)

    # Create weight tensor
    max_class = int(np.max(labels))
    weight_array = np.ones(max_class + 1)
    for cls, weight in weights.items():
        weight_array[int(cls)] = weight

    return torch.FloatTensor(weight_array).to(device)
