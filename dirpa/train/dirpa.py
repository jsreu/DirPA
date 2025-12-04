from __future__ import annotations

import random
from typing import Literal, cast

import torch
import torch.nn as nn
from pydantic import BaseModel, field_validator


class DirichletAlphaLearner(nn.Module):
    """Asymmetric Dirichlet alpha learner using two distinct alphas.

    Args:
        alpha_focus: Start value for alpha focus (assigned to a single focus class every step).
        alpha_common: Start value for alpha common (assigned to all other classes).
        eps: Epsilon to prevent alpha from being zero.
    """

    def __init__(self, alpha_focus: float, alpha_common: float, eps: float = 1e-8):
        super().__init__()

        self.v_focus = nn.Parameter(torch.tensor(alpha_focus, dtype=torch.float32))
        self.v_common = nn.Parameter(torch.tensor(alpha_common, dtype=torch.float32))

        self.softplus = nn.Softplus()
        self.eps = eps

    def forward(self) -> dict[str, float]:
        """Forward pass for asymmetric Dirichlet alpha learner."""
        alpha_focus = (self.softplus(self.v_focus) + self.eps).item()
        alpha_common = (self.softplus(self.v_common) + self.eps).item()

        return {"alpha_focus": alpha_focus, "alpha_common": alpha_common}


class DirichletConfig(BaseModel):
    """Configuration for Dirichlet Distribution sampling."""

    enabled: bool = False
    alpha_mode: Literal["symmetric", "asymmetric"] = "symmetric"
    alpha: float | None = None  # spikiness of Dir(C, alpha)
    alpha_focus: float | None = None
    alpha_common: float | None = None
    alpha_lr: float | None = None
    tau: float = 1.0  # prior strength
    blend_with_uniform: bool = False
    beta: float | None = None  # if None and blend=True -> defaults to 0.8
    eps: float = 1e-8

    @field_validator("alpha", "alpha_focus", "alpha_common")
    @classmethod
    def _alpha_pos(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("alpha must be > 0")
        return v

    @field_validator("beta")
    @classmethod
    def _beta_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("beta must be in [0,1]")
        return v

    def _post_init(self) -> None:
        if self.alpha_mode == "symmetric" and self.alpha is None:
            raise AssertionError("For symmetric Dirichlet, please set alpha.")
        elif self.alpha_mode == "asymmetric" and (
            self.alpha_focus is None or self.alpha_common is None
        ):
            raise AssertionError(
                "For symmetric Dirichlet, please set alpha_focus and alpha_common."
            )


def _sample_dirichlet_prior(
    c: int,
    tau: float,
    alpha: float | None = None,
    alpha_focus: float | None = None,
    alpha_common: float | None = None,
    alpha_mode: Literal["symmetric", "asymmetric"] = "symmetric",
    eps: float = 1e-8,
    blend_with_uniform: bool = False,
    beta: float | None = 0.8,
) -> torch.Tensor:
    """Sample symmetric pseudo-prior from Dirichlet distribution: pi ~ Dir(alpha1).

    Args:
        c: Number of classes.
        tau: Prior strength.
        alpha: Dirichlet concentrations.
            Scalar (symmetric Dirichlet):
                - alpha = 1: uniform distribution on simplex
                - alpha > 1: favouring balanced vectors
                - alpha < 1: favoring sparse vectors
        alpha_focus: Start value for alpha focus (assigned to a single focus class every step)
        alpha_common: Start value for alpha common (assigned to all other classes).
        alpha_mode: Chooses symmetric or asymmetric Dirichlet sampling.
        eps: Small constant to avoid log(0) when clamping Dirichlet samples.
        blend_with_uniform: Whether to convex-mix sampling with uniform distribution
            to avoid extreme shifts.
        beta: Mixing factor in [0,1]; if None and blend=True, defaults to 0.8.
            beta=1: No smoothing (raw Dirichlet)
            beta=0: ignore sample (uniform prior)
            smaller beta: higher entropy, less extreme priors
    """

    if blend_with_uniform and beta is None:
        raise AssertionError("beta cannot be None when blend_with_uniform is True")

    # sample prior on (K-1)-simplex
    if alpha_mode == "asymmetric":
        alpha_focus = cast(float, alpha_focus)
        alpha_common = cast(float, alpha_common)
        focus_class_idx: int = random.randint(0, c - 1)
        alpha_tensor: torch.Tensor = torch.full((c,), alpha_common)
        alpha_tensor[focus_class_idx] = alpha_focus
        pi = torch.distributions.Dirichlet(alpha_tensor).sample()
    else:
        pi = torch.distributions.Dirichlet(torch.full((c,), cast(float, alpha))).sample()

    if blend_with_uniform:
        beta = cast(float, beta)
        pi = (1 - beta) * (torch.full((c,), 1.0 / c)) + beta * pi
        pi = pi / pi.sum()  # renormalize for numeric safety

    # numerical safety
    prior_adjustment = tau * torch.log(pi.clamp_min(eps))

    return prior_adjustment
