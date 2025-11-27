import torch
from torch import nn

import dirpa.train.loss as Losses


# ---------------- Focal basics ----------------
def test_focal_loss_reduces_to_ce_when_gamma0_no_alpha() -> None:
    torch.manual_seed(0)
    logits = torch.randn(5, 3, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 2, 1])
    ce = nn.CrossEntropyLoss(reduction="mean")
    focal = Losses.FocalLoss(gamma=0.0, alpha=None, reduction="mean")
    assert torch.allclose(focal(logits, targets), ce(logits, targets), atol=1e-10, rtol=1e-10)


def test_focal_loss_alpha_vector_matches_nll_weighting() -> None:
    logits = torch.tensor([[2.0, 0.1, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    targets = torch.tensor([0, 1, 2])
    alpha = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    focal = Losses.FocalLoss(gamma=0.0, alpha=alpha, reduction="mean")
    val_focal = focal(logits, targets)
    log_probs = torch.log_softmax(logits, dim=-1)
    val_manual = nn.NLLLoss(weight=alpha, reduction="none")(log_probs, targets).mean()
    assert torch.allclose(val_focal, val_manual, atol=1e-10, rtol=1e-10)
