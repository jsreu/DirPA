from __future__ import annotations

import random
from abc import abstractmethod
from typing import Literal, cast

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, field_validator


def _inv_softplus(x: float) -> float:
    return cast(float, np.log(np.exp(x) - 1.0))

class DirichletAlphaLearner(nn.Module):
    """Dirichlet alpha learner."""
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self) -> dict[str, torch.Tensor]:
        """Forward pass for asymmetric Dirichlet alpha learner."""

class SymmetricDirichletAlphaLearner(DirichletAlphaLearner):
    """Symmetric Dirichlet alpha learner using a single alpha.

    Args:
        alpha: Start value for alpha.
            Default: 1.0 for uniform distribution.
        activation_fct:
        eps: Epsilon to prevent alpha from being zero.
    """

    def __init__(
            self,
            alpha: float = 1.0,
            activation_fct: Literal["softplus", "sigmoid"] = "softplus",
            eps: float = 1e-8
            ) -> None:
        super().__init__()

        if activation_fct == "softplus":
            val = _inv_softplus(alpha)
            init_alpha = torch.as_tensor(val, dtype=torch.float32)
            self.act_fct: nn.Softplus | nn.Sigmoid = nn.Softplus()
        else:
            alpha_clamped = max(min(alpha, 1.0 - 1e-6), 1e-6)
            init_alpha = torch.log(torch.tensor(alpha_clamped) / (1 - torch.tensor(alpha_clamped)))
            self.act_fct = nn.Sigmoid()

        self.value = nn.Parameter(init_alpha.clone().detach())

        self.eps = eps

    def forward(self) -> dict[str, torch.Tensor]:
        """Forward pass for asymmetric Dirichlet alpha learner."""
        alpha = self.act_fct(self.value) + self.eps

        return {"alpha": alpha}

class AsymmetricDirichletAlphaLearner(DirichletAlphaLearner):
    """Asymmetric Dirichlet alpha learner using two distinct alphas.

    Args:
        alpha_focus: Start value for alpha focus (assigned to a single focus class every step).
        alpha_common: Start value for alpha common (assigned to all other classes).
        eps: Epsilon to prevent alpha from being zero.
    """

    def __init__(self, alpha_focus: float, alpha_common: float, eps: float = 1e-8):
        super().__init__()

        init_alpha_focus = _inv_softplus(alpha_focus)
        init_alpha_common = _inv_softplus(alpha_common)
        self.v_focus = nn.Parameter(torch.tensor(init_alpha_focus, dtype=torch.float32))
        self.v_common = nn.Parameter(torch.tensor(init_alpha_common, dtype=torch.float32))

        self.softplus = nn.Softplus()
        self.eps = eps

    def forward(self) -> dict[str, torch.Tensor]:
        """Forward pass for asymmetric Dirichlet alpha learner."""
        alpha_focus = self.softplus(self.v_focus) + self.eps
        alpha_common = self.softplus(self.v_common) + self.eps

        return {"alpha_focus": alpha_focus, "alpha_common": alpha_common}

class DirichletConfig(BaseModel):
    """Configuration for Dirichlet Distribution sampling."""

    enabled: bool = False
    alpha_mode: Literal["symmetric", "asymmetric"] = "symmetric"
    warmup_epochs: int = 0
    alpha: float | None = None  # spikiness of Dir(C, alpha)
    activation_fct: Literal["sigmoid", "softplus"] = "softplus"
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
    alpha: float | torch.Tensor | None = None,
    alpha_focus: float | torch.Tensor | None = None,
    alpha_common: float | torch.Tensor | None = None,
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
        assert alpha_focus is not None
        assert alpha_common is not None
        # ensure we use the device of the tensors if they are learnable
        dev = alpha_common.device if isinstance(alpha_common, torch.Tensor) else torch.device("cpu")
        alpha_tensor = torch.ones(c, device=dev) * cast(float, alpha_common)
        focus_class_idx: int = random.randint(0, c - 1)
        alpha_tensor[focus_class_idx] = alpha_focus
        pi = torch.distributions.Dirichlet(alpha_tensor).rsample()
    else:
        assert alpha is not None
        # ensure we use the device of the tensors if they are learnable
        dev = alpha.device if isinstance(alpha, torch.Tensor) else torch.device("cpu")
        alpha_tensor = torch.ones(c, device=dev) * cast(float, alpha)
        pi = torch.distributions.Dirichlet(alpha_tensor).rsample()

    if blend_with_uniform:
        beta = cast(float, beta)
        uniform_prior = torch.full((c,), 1.0 / c, dtype=torch.float32, device=dev)
        pi = (1 - beta) * uniform_prior + beta * pi
        # renormalize for numerical safety after blending
        pi = pi / pi.sum()

    # numerical safety
    prior_adjustment = tau * torch.log(pi.clamp_min(eps))

    return prior_adjustment
