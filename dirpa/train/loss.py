from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel


class FocalLoss(nn.Module):
    """Multi-class focal Loss for imbalanced datasets.

    Args:
        gamma: Focusing parameter to adjust rate at which easy examples are down-weighted,
            gamma > 0 reduces relative loss for easy-to-classify samples,
            putting more focus on hard-to-classify ones.
        alpha: Class balancing factor to adjust loss contribution of each class.
            Defaults to None (which corresponds to 1.0).
            A dict of [leaf_idx: weight] (per-leaf weight).
        reduction: Way to accumulate final loss.
        eps: Small constant for numerical stability in logs/divisions
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: dict[int, float] | None = None,
        class_order: dict[int, int] | None = None,
        reduction: Literal["mean", "sum", None] = "mean",
        eps: float = 1e-8,
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

        # store alpha as a buffer for NLLLoss if it's a tensor
        w = None
        alpha_weights = None
        if alpha is not None:
            assert class_order is not None
            self.C = len(class_order)
            sorted_items = sorted(class_order.items())
            ordered_classes = dict(sorted_items)
            # allow keys as leaf indices (0..C-1) or raw leaf node ids
            w = torch.ones(self.C, dtype=torch.float32)
            if set(alpha.keys()) <= set(range(self.C)):
                for j in range(self.C):
                    w[j] = float(alpha.get(j, 1.0))
            else:
                # map raw leaf node ids via class_order
                for raw_nid, j in ordered_classes.items():
                    w[j] = float(alpha.get(int(raw_nid), 1.0))

            values_list = w.tolist()
            alpha_weights = torch.Tensor(values_list)
        self.alpha_weights: torch.Tensor
        self.register_buffer(
            "alpha_weights",
            alpha_weights if alpha_weights is not None else torch.tensor([]),
            persistent=False,
        )

    def forward(
        self,
        outputs: torch.Tensor,  # [B, C] leaf logits
        targets: torch.Tensor,  # [B] leaf ids
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for Focal Loss."""
        # compute weighted cross-entropy term: -alpha * log(pt)
        # log probabilities for all classes
        log_probs = torch.log_softmax(
            outputs, dim=-1
        )  # [B, C]; log probs of each class per sample in B
        # cross-entropy per-sample via negative log likelihood
        ce = nn.functional.nll_loss(log_probs, targets, reduction="none")  # [B]

        # gather true class: log of preditcted prob for true class t
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]

        # compute focal term: (1 - pt)^gamma
        # -torch.expm1(log_pt) = (1 - pt)
        oneminuspt = (-torch.expm1(log_pt)).clamp_min(self.eps)  # [B]
        focal_term = oneminuspt**self.gamma

        if weight is not None:
            # Use the dynamic weights from the DiPA logic
            alpha_weights = weight.to(outputs.device)
            alpha = alpha_weights.gather(0, targets)  # [B]
        elif self.alpha_weights.numel() != 0:
            # Gather the correct alpha value for each sample's target class
            alpha_weights = self.alpha_weights.to(outputs.device)
            alpha = alpha_weights.gather(0, targets)  # [B]
        else:
            # no weights applied
            alpha = outputs.new_ones(targets.shape[0])

        # full loss: alpha * ((1 - pt)^gamma) * (-log(pt))
        loss: torch.Tensor = alpha * focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class LossCEConfig(BaseModel):
    """Configuration for Cross-Entropy Loss."""

    name: Literal["CE"] = "CE"
    weighted: bool = False
    label_smoothing: float = 0.0
    reduction: Literal["mean", "sum", None] = "mean"

    def _build(self, weights: dict[int, float] | None = None, **_: object) -> nn.Module:
        return nn.CrossEntropyLoss(
            weight=(
                torch.Tensor(weights.values()) if self.weighted and weights is not None else None
            ),
            label_smoothing=self.label_smoothing,
            reduction="none" if self.reduction is None else self.reduction,
        )


class LossFocalConfig(BaseModel):
    """Configuration for Cross-Entropy Loss."""

    name: Literal["FCL"] = "FCL"
    gamma: float = 2.0
    alpha: bool = False
    reduction: Literal["mean", "sum", None] = "mean"
    eps: float = 1e-8

    def _build(
        self,
        *,
        class_order: dict[int, int],
        alpha_weights: dict[int, float] | None = None,
        **_: object,
    ) -> nn.Module:

        return FocalLoss(
            gamma=self.gamma,
            alpha=alpha_weights if self.alpha else None,
            class_order=class_order,
            reduction=self.reduction,
            eps=self.eps,
        )


LossConfig = LossCEConfig | LossFocalConfig
