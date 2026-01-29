import logging
from collections.abc import Sequence
from typing import Any, Literal, Type, cast

import optuna
import torch
import torch.nn as nn
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dirpa.dataset.task import Task
from dirpa.experiment.logger import MLFlowCallback
from dirpa.models.base import DataItem, Model
from dirpa.train.callback import TrainCallback
from dirpa.train.dirpa import (
    DirichletAlphaLearner,
    DirichletConfig,
    _sample_dirichlet_prior,
)
from dirpa.train.loss import (
    LossConfig,
    LossFocalConfig,
)
from dirpa.train.utils import EarlyStopping, TaskMetric
from dirpa.utils import BaseConfig

logger = logging.getLogger(__name__)


class TrainConfig(BaseConfig):
    """Configuration class for vanilla training.

    Args:
        batch_size: The number of samples per batch.
        epochs: The number of epochs.
        steps: The number of total training steps. Ignored if epochs is not 0.
        head_lr: Head learning rate for Training algorithm.
        backbone_lr: Backbone learning rate for Training algorithm.
        lr: Learning rate for Training algorithm.
        optimizer: PyTorch optimizer used for training
        loss_config:
        dirichlet_config:
        weight_decay: Weight decay (L2) penalty.
            If 0.0 (default), no weight decay is applied.
        cosine_annealing: Factor to define cycle length for CosineAnnealingWarmRestars.
            Default 0, no Cosine Annealing is applied. If 1, one single cosine cycle is applied
            over the entire training process, otherwise the training process is divided into
            multiple cycles.
        stop_early: Whether to use early stopping.
            Default: False
        patience:  Number of validation periods the validation loss is
            allowed to decrease. This corresponds to the number of validation periods
            which can either be batches or epochs.
    """

    batch_size: int
    epochs: int = 0
    steps: int = Field(0, validate_default=True)
    head_lr: float | None = Field(None, validate_default=True)
    backbone_lr: float | None = Field(None, validate_default=True)
    lr: float | None = Field(None, validate_default=True)
    optimizer: Literal["SGD", "Adam", "AdamW"] = "Adam"
    loss_config: LossConfig
    dirichlet_config: DirichletConfig | None = None
    weight_decay: float = 0.0
    cosine_annealing: int = 0
    warmup_epochs_scheduler: int = 0
    freeze_backbone_epoch: int = 0

    stop_early: bool = False
    patience: int | None = None

    def hyperparameters(self) -> dict[str, str | int | float | None]:
        """Return training hyperparameters."""
        params: dict[str, Any] = {
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "cosine_annealing": self.cosine_annealing,
            "warmup_epochs_scheduler": self.warmup_epochs_scheduler,
            "loss_config": self.loss_config,
            "dirichlet_config": self.dirichlet_config,
            "freeze_backbone_epoch": self.freeze_backbone_epoch,
        }
        if self.head_lr is not None and self.backbone_lr is not None:
            params["backbone_lr"] = self.backbone_lr
            params["head_lr"] = self.head_lr
        else:
            params["lr"] = self.lr
        return params

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v: int, info: ValidationInfo) -> int:
        """Ensure that either steps or epochs are non-zero."""
        if v == 0 and info.data["epochs"] == 0:
            raise ValueError("Either epochs or steps must be non-zero.")
        return v

    @field_validator("backbone_lr")
    @classmethod
    def validate_separate_lr(cls, v: float | None, info: ValidationInfo) -> float | None:
        """Ensure that backbone_lr is specified jointly with head_lr."""
        valid_head_lr = "head_lr" in info.data and info.data["head_lr"] is not None
        if (v is None and valid_head_lr) or (v is not None and not valid_head_lr):
            logger.warning(
                "Both head_lr and backbone_lr need to be jointly specified. "
                "Going to ignore separate head_lr and backbone_lr."
            )
            return None
        return v

    @field_validator("lr")
    @classmethod
    def validate_all_lr(cls, v: float | None, info: ValidationInfo) -> float | None:
        """Ensure that either head_lr and backbone_lr or lr are specified."""
        valid_head_and_backbone_lr = (
            "head_lr" in info.data
            and "backbone_lr" in info.data
            and info.data["head_lr"] is not None
            and info.data["backbone_lr"] is not None
        )
        if v is None and not valid_head_and_backbone_lr:
            raise ValueError("Either both head_lr and backbone_lr or lr must be specified.")
        elif v is not None and valid_head_and_backbone_lr:
            logger.warning(
                "Both head_lr and backbone_lr as well as lr are specified. "
                "Going to ignore lr and use separate head_lr and backbone_lr."
            )
            return None
        return v


class Trainer:
    """Class for running vanilla training algorithm.

    Args:
        config: Training config.
        callbacks: Callbacks to call during training
    """

    def __init__(self, config: TrainConfig, callbacks: list[TrainCallback] | None = None):
        self.config = config
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

    @staticmethod
    def _build_optimizer(
        model: Model,
        optimizer: Literal["SGD", "Adam", "AdamW"],
        head_lr: float | None,
        backbone_lr: float | None,
        lr: float | None,
        weight_decay: float,
        dirichlet_alpha_learner: nn.Module | None = None,
        alpha_lr: float | None = None,
    ) -> Optimizer:
        optimizer_class: Type[SGD | Adam | AdamW] = {
            "SGD": SGD,
            "Adam": Adam,
            "AdamW": AdamW,
        }[optimizer]
        if head_lr is not None and backbone_lr is not None:
            params: list[dict] = [
                {
                    "params": model.backbone.parameters(),
                    "lr": backbone_lr,
                    "weight_decay": weight_decay,
                    "name": "backbone",
                },
                {
                    "params": model.head.parameters(),
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                    "name": "head",
                },
            ]
            optimizer_lr = cast(float, backbone_lr)

        else:  # lr cannot be None as well
            params = [
                {
                    "params": model.parameters(),
                    "lr": cast(float, lr),
                    "weight_decay": weight_decay,
                }
            ]
            optimizer_lr = cast(float, lr)
        if dirichlet_alpha_learner is not None:
            params.append(
                {
                    "params": dirichlet_alpha_learner.parameters(),
                    "lr": alpha_lr if alpha_lr is not None else optimizer_lr,
                    "weight_decay": 0.0,
                    "name": "alpha_learner",
                }
            )

        return optimizer_class(params, lr=optimizer_lr)

    def _build_scheduler(
        self,
        optimizer: Optimizer,
        steps_per_epoch: int,
    ) -> tuple[SequentialLR | CosineAnnealingWarmRestarts | None, int | None]:
        # TODO: double-check logic
        # if annealing is off, return None and the standard patience
        if self.config.cosine_annealing == 0:
            return None, self.config.patience

        # total_epochs here refers to the active training horizon.
        # if called during unfreeze, self.config.epochs has been updated
        # to (original_total - freeze_backbone_epoch).
        total_epochs = self.config.epochs

        # calculate steps for the remaining horizon
        if self.config.warmup_epochs_scheduler > 0:
            warmup_steps = self.config.warmup_epochs_scheduler * steps_per_epoch

            # ensure we don't have more warmup epochs than total remaining epochs
            actual_warmup_steps = min(warmup_steps, total_epochs * steps_per_epoch)

            scheduler_epochs = max(1, total_epochs - self.config.warmup_epochs_scheduler)
            max_num_steps = steps_per_epoch * scheduler_epochs

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=actual_warmup_steps,
            )
        else:
            max_num_steps = steps_per_epoch * total_epochs
            warmup_steps = 0

        # calculate T_0 (first cycle length) based on the remaining steps
        if self.config.cosine_annealing == 1:
            t_0 = max_num_steps
        else:
            t_0 = max_num_steps // self.config.cosine_annealing

        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, t_0),  # avoid division by zero if steps are very low
            T_mult=2 if self.config.cosine_annealing > 1 else 1,
            eta_min=1e-7,
        )

        if warmup_steps > 0:
            return (
                SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps],
                ),
                total_epochs,
            )

        return cosine_scheduler, total_epochs

    def train(
        self,
        model: Model,
        task: Task,
        validate_every_step: int = 0,
        validate_every_epoch: int = 1,
        warmup_steps: int = 0,
        freeze_backbone_epoch: int = 0,
    ) -> Model:
        """Train model with configured optimizer and scheduling.

        Args:
            model: Model to adapt.
            task: Task that contains a train, validation and test dataset.
            validate_every_step: Number of training batches between validation runs (if epoch==0).
            validate_every_epoch: Number of epochs between validation runs (if epoch>0).
            warmup_steps: Number of initial warmup steps.
            freeze_backbone_epoch: Number of epochs during which the backbone is initially frozen.
                If 0, the backbone will not be frozen.

        Returns:
            Adapted model
        """
        if isinstance(self.config.loss_config, LossFocalConfig):
            loss_fn = self.config.loss_config._build(
                class_order=cast(dict[int, int], task.encoding),
                alpha_weights=task.class_weights,
            )
        else:
            loss_fn = self.config.loss_config._build(  # CE loss
                weights=task.class_weights,
            )

        if validate_every_step * validate_every_epoch != 0:
            raise AssertionError(
                "Either `validate_every_epoch` or `validate_every_step` " "must be zero."
            )

        train_dl = task.train_dl(batch_size=self.config.batch_size)
        val_dl = task.val_dl(batch_size=self.config.batch_size)

        dirichlet_alpha_learner = None
        alpha_lr: float | None = None
        if self.config.dirichlet_config is not None:
            dirichlet_config = cast(DirichletConfig, self.config.dirichlet_config)
            if (
                dirichlet_config.enabled
                and dirichlet_config.alpha_mode == "asymmetric"
                and dirichlet_config.alpha_lr != 0.0
            ):
                dirichlet_alpha_learner = cast(
                    DirichletAlphaLearner,
                    DirichletAlphaLearner(
                        cast(float, dirichlet_config.alpha_focus),
                        cast(float, dirichlet_config.alpha_common),
                    ),
                )
            alpha_lr = cast(float, dirichlet_config.alpha_lr)

        # initialize Optimizer
        # Note: If freeze_backbone_epoch > 0, we start with backbone_lr = 0.0
        optimizer = self._build_optimizer(
            model=model,
            optimizer=self.config.optimizer,
            head_lr=self.config.head_lr,
            backbone_lr=0.0 if freeze_backbone_epoch > 0 else self.config.backbone_lr,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            dirichlet_alpha_learner=dirichlet_alpha_learner,
            alpha_lr=alpha_lr,
        )

        # handle scheduler logic
        # If we are freezing, the scheduler is None initially.
        # We only build it when we unfreeze.
        scheduler: SequentialLR | CosineAnnealingWarmRestarts | None = None
        patience = self.config.patience
        if freeze_backbone_epoch == 0:
            scheduler, patience = self._build_scheduler(optimizer, steps_per_epoch=len(train_dl))

        early_stopping: EarlyStopping | None = None
        if self.config.stop_early:
            early_stopping = EarlyStopping(patience=patience, warmup_steps=warmup_steps)

        step = 0
        epochs_bar = trange(self.config.epochs, desc="Epochs", position=0, leave=True)

        for epoch in epochs_bar:
            # unfreeze backbone
            if freeze_backbone_epoch > 0 and epoch == freeze_backbone_epoch:
                logger.info(f"Unfreezing backbone. Initializing scheduler at epoch {epoch}")

                for param_group in optimizer.param_groups:
                    if param_group.get("name") == "backbone":
                        param_group["lr"] = self.config.backbone_lr

                # build scheduler now s.t. it starts its warmup/cosine from step 0
                # calculate remaining epochs so the scheduler fits the remaining time
                remaining_epochs = self.config.epochs - epoch

                # temporary override config to build scheduler for the remaining horizon
                original_epochs = self.config.epochs
                self.config.epochs = remaining_epochs
                scheduler, _ = self._build_scheduler(optimizer, steps_per_epoch=len(train_dl))
                self.config.epochs = original_epochs

            for batch in tqdm(train_dl, desc=f"Epoch {epoch}", leave=False):
                self._inner_step(
                    model=model,
                    train_batch=batch,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,  # This will be None until unfreeze
                    dirichlet_alpha_learner=dirichlet_alpha_learner,
                    step=step,
                )
                step += 1

                if validate_every_step != 0 and (step) % validate_every_step == 0:
                    stopped = self._validate(model, task, early_stopping, step, val_dl)
                    if stopped:
                        break

            if validate_every_epoch != 0 and (epoch + 1) % validate_every_epoch == 0:
                stopped = self._validate(model, task, early_stopping, step, val_dl)
                if stopped:
                    break

        # Epochs is zero, so count by steps.
        else:
            if validate_every_step == 0:
                raise AssertionError(
                    "When counting in steps, `validate_every_step` needs to be " " greater zero."
                )
            batches_bar = trange(
                min(len(train_dl), self.config.steps),
                desc="Batches",
                position=0,
                leave=True,
            )
            for step, batch in zip(batches_bar, train_dl):
                self._inner_step(
                    model=model,
                    train_batch=batch,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    dirichlet_alpha_learner=dirichlet_alpha_learner,
                    step=step,
                )
                if validate_every_step != 0 and (step) % validate_every_step == 0:
                    stopped = self._validate(model, task, early_stopping, step, val_dl)
                    if stopped:
                        break

        for callback in self.callbacks:
            callback.end_callback(model)
        return model

    def _inner_step(
        self,
        model: Model,
        train_batch: tuple[DataItem, torch.Tensor],
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: SequentialLR | CosineAnnealingWarmRestarts | None,
        step: int,
        dirichlet_alpha_learner: DirichletAlphaLearner | None = None,
    ) -> None:
        train_loss = _train_step(
            model=model,
            batch=train_batch,
            loss_fn=loss_fn,
            dirichlet_config=self.config.dirichlet_config,
            optimizer=optimizer,
            scheduler=scheduler,
            dirichlet_alpha_learner=dirichlet_alpha_learner,
            step=step,
        )
        train_metrics: Sequence[TaskMetric] = []
        for callback in self.callbacks:
            if step % 10 == 0:
                if isinstance(callback, MLFlowCallback) and scheduler:
                    cast(MLFlowCallback, callback).mlflow_logger.log(
                        "lr", float(scheduler.get_last_lr()[0]), global_step=step
                    )
                callback.train_callback(train_loss, train_metrics, model=model, step=step)

    def _validate(
        self,
        model: Model,
        task: Task,
        early_stopping: EarlyStopping | None,
        step: int,
        val_dl: DataLoader,
    ) -> bool:
        val_loss, val_metrics = self._evaluation_loop(val_dl, task, model)

        for callback in self.callbacks:
            try:
                callback.validation_callback(
                    val_loss,
                    val_metrics,
                    model,
                    step=step,
                )
            except optuna.TrialPruned as e:
                # even if we prune, make sure to properly end all callbacks
                for callback in self.callbacks:
                    callback.end_callback(model)
                raise e  # reraise the pruning exception after ending all callbacks

        stopped = False
        if early_stopping:
            stopped = early_stopping(val_loss, step, model)

        return stopped

    def test(
        self,
        model: Model,
        task: Task,
    ) -> tuple[float, Sequence[TaskMetric]]:
        """Test trained model on test set.

        Args:
            model: Trained Model.
            task: Task dataset to use.

        Returns:
            Average test loss per batch
            Test Metrics
        """

        test_dl = task.test_dl(batch_size=self.config.batch_size)

        test_loss, test_metrics = self._evaluation_loop(test_dl, task, model)

        for callback in self.callbacks:
            callback.test_callback(
                test_loss,
                test_metrics,
                model,
                step=None,
            )

        return test_loss, test_metrics

    def _evaluation_loop(
        self, dataloader: DataLoader, task: Task, model: Model
    ) -> tuple[float, Sequence[TaskMetric]]:
        """Returns loss, dict of scalar metrics, dict of additional (tensor) metrics."""

        if isinstance(self.config.loss_config, LossFocalConfig):
            loss_fn = self.config.loss_config._build(
                class_order=cast(dict[int, int], task.encoding),
                alpha_weights=task.class_weights,
            )
        else:
            loss_fn = self.config.loss_config._build(task.class_weights)

        for metric in task.metrics:
            metric.reset()

        eval_loss = 0.0
        for _, batch in enumerate(dataloader):
            with torch.no_grad():
                loss = _eval_step(
                    model,
                    batch,
                    loss_fn,
                    task.metrics,
                )

            eval_loss += loss
        eval_loss = eval_loss / len(dataloader)

        return eval_loss, task.metrics


def _train_step(
    model: Model,
    batch: tuple[DataItem, torch.Tensor],
    loss_fn: nn.Module,
    dirichlet_config: DirichletConfig | None,
    optimizer: Optimizer,
    step: int,
    scheduler: SequentialLR | CosineAnnealingWarmRestarts | None = None,
    dirichlet_alpha_learner: DirichletAlphaLearner | None = None,
) -> float:
    optimizer.zero_grad()
    model.train()

    data, target = batch[0].to(model.device), batch[1].to(model.device)

    out = model(data)
    if (
        dirichlet_config is not None
        and dirichlet_config.enabled
        and dirichlet_config.warmup_epochs <= step
    ):
        # Dirichlet prior
        dirichlet_cfg: dict[str, Any] = dirichlet_config.model_dump()
        del dirichlet_cfg["enabled"]
        del dirichlet_cfg["alpha_lr"]
        del dirichlet_cfg["warmup_epochs"]

        if dirichlet_alpha_learner is not None:
            current_alphas = dirichlet_alpha_learner()
            dirichlet_cfg["alpha_focus"] = current_alphas["alpha_focus"]
            dirichlet_cfg["alpha_common"] = current_alphas["alpha_common"]

        prior_adj = _sample_dirichlet_prior(
            out.size(-1),
            **dirichlet_cfg,
        )  # [C] tau·log(pi+eps)

        prior_adj = prior_adj.to(out)
        # Dirichlet dsitribution is closed under grouping
        # Hence, for hierarchical loss:
        # per-leaf prior is sufficient since sum is again Dirichlet-dsitributed
        out = out + prior_adj.view(1, -1)

    train_loss = loss_fn(out, target).to(model.device)

    train_loss.backward()

    # gradient clipping
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    loss = train_loss.detach().cpu().item()

    return cast(float, loss)


def _eval_step(
    model: Model,
    batch: tuple[torch.Tensor, torch.Tensor],
    loss_fn: nn.Module,
    metrics: Sequence[TaskMetric],
) -> float:
    model.eval()

    data, target = batch[0].to(model.device), batch[1].to(model.device)

    out = model(data)
    test_loss = loss_fn(out, target)

    for metric in metrics:
        metric.update(out.detach().cpu(), target.detach().cpu())

    return cast(float, test_loss.detach().cpu().item())
