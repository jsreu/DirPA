import logging
from functools import cache
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from eurocropsml.dataset.config import (
    EuroCropsDatasetConfig,
    EuroCropsDatasetPreprocessConfig,
    EuroCropsSplit,
)

# from eurocropsml.dataset.preprocess import get_class_ids_to_names
from eurocropsml.dataset.preprocess import read_metadata

from dirpa.dataset.eurocrops.train import load_dataset_split
from dirpa.dataset.task import Task

logger = logging.getLogger(__name__)


# TODO: remove once this is incorporated into eurocropsml
@cache
def get_class_ids_to_names(raw_data_dir: Path) -> dict[str, str]:
    """Get a dictionary mapping between class identifiers and readable names."""
    labels_df: pd.DataFrame = read_metadata(raw_data_dir.joinpath("labels"))
    unique_labels_df = labels_df.drop_duplicates()
    ids_to_names_dict = unique_labels_df.set_index("EC_hcat_c").to_dict()["EC_hcat_n"]
    return {str(k): v for k, v in ids_to_names_dict.items()}


def load_task(
    split_dir: Path,
    mode: Literal["pretraining", "finetuning"],
    split_config: EuroCropsSplit,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    dataset_config: EuroCropsDatasetConfig,
    max_samples: int | str = "all",
    downsample_classes: list[tuple[int, float]] | None = None,
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
            downsample_classes=dict(downsample_classes) if downsample_classes is not None else None,
        ),
    )
