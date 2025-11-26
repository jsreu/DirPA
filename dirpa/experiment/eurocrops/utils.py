import logging
from pathlib import Path
from typing import Literal, cast

from eurocropsml.dataset.config import (
    EuroCropsDatasetConfig,
    EuroCropsDatasetPreprocessConfig,
    EuroCropsSplit,
)
from eurocropsml.dataset.preprocess import get_class_ids_to_names

from dirpa.dataset.eurocrops.train import load_dataset_split
from dirpa.dataset.task import Task

logger = logging.getLogger(__name__)


def load_task(
    split_dir: Path,
    mode: Literal["pretraining", "finetuning"],
    split_config: EuroCropsSplit,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    dataset_config: EuroCropsDatasetConfig,
    max_samples: int | str = "all",
) -> Task:
    """Load pretraining or finetuning task from the EuroCrops dataset."""

    split = dataset_config.split

    classes = (
        set(split_config.pretrain_classes[split])
        if mode == "pretraining"
        else set(split_config.finetune_classes[split])
    )
    return cast(
        Task,
        load_dataset_split(
            mode=mode,
            classes=classes,
            split_dir=split_dir,
            preprocess_config=preprocess_config,
            dataset_config=dataset_config,
            max_samples=max_samples,
            class_ids_to_names=get_class_ids_to_names(preprocess_config.raw_data_dir),
        ),
    )
