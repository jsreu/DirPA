import json
import logging
from collections import Counter
from functools import reduce
from operator import add
from pathlib import Path
from typing import Literal

from eurocropsml.dataset.base import TransformDataset, custom_collate_fn
from eurocropsml.dataset.config import (
    EuroCropsDatasetConfig,
    EuroCropsDatasetPreprocessConfig,
)
from eurocropsml.dataset.utils import MMapStore

from dirpa.dataset.eurocrops.dataset import EuroCropsDataset
from dirpa.dataset.eurocrops.utils import _downsample
from dirpa.dataset.task import Task
from dirpa.train.utils import get_metrics

logger = logging.getLogger(__name__)


def load_dataset_split(
    mode: Literal["pretraining", "finetuning"],
    classes: set,
    split_dir: Path,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    dataset_config: EuroCropsDatasetConfig,
    max_samples: int | str,
    class_ids_to_names: dict[str, str] | None,
    downsample_classes: dict[int, float] | None = None,
) -> Task:
    """Load EuroCrops data.

    Args:
        mode: Whether to load pretrain or finetuning dataset.
        classes: The classes of the requested dataset split.
        split_dir: Directory where split is loaded from.
        preprocess_config: Config model of preprocessed data.
        dataset_config: Config model of dataset to be loaded.
        loss_fn: Loss function used to calculate the model's loss.
        max_samples: Maximum number of samples per class within finetuning dataset.
        class_ids_to_names: Optional mapping from class identifiers to readable class names.
        downsample_classes: Used for downsampling or fully removing classes from the training.

    Returns:
        Task containing train, validation, and optionally test dataset.

    Raises:
        FileNotFoundError: If dataset split file is not found.
        FileNotFoundError: If train and/or validation set do not exist.
        FileNotFoundError: If finetuning mode and test set does not exist.
    """

    if mode == "finetuning":
        split_file = split_dir.joinpath(
            "finetune", f"{dataset_config.split}_split_{max_samples}.json"
        )
    else:
        split_file = split_dir.joinpath("pretrain", f"{dataset_config.split}_split.json")
    if split_file.exists():
        with open(split_file) as outfile:
            data_split = json.load(outfile)

    else:
        raise FileNotFoundError(
            str(split_file) + " does not exist. Please first build the dataset split."
        )

    satellites = dataset_config.data_sources
    satellites.sort()

    data_satellite_split: dict[str, dict[str, list[Path]]] = {
        key: {s: [] for s in satellites} for key in data_split
    }
    data_satellite_split = {
        key: {
            s: [
                preprocess_config.preprocess_dir.joinpath(s, str(dataset_config.year), file)
                for file in file_list
            ]
            for s in satellites
        }
        for key, file_list in data_split.items()
    }

    try:
        train = data_satellite_split["train"]
        val = data_satellite_split["val"]

        if downsample_classes is not None:
            for drop_class, drop_prob in downsample_classes.items():
                train = _downsample(train, drop_class, drop_prob, satellites)
                val = _downsample(val, drop_class, drop_prob, satellites)
                if drop_prob == 1.0:
                    classes.discard(drop_class)
        train_list = reduce(add, train.values())
        val_list = reduce(add, val.values())
        if mode == "finetuning":
            test = data_satellite_split["test"]
            if downsample_classes is not None:
                for drop_class, drop_prob in downsample_classes.items():
                    test = _downsample(test, drop_class, drop_prob, satellites)

            test_list = [item for sublist in test.values() for item in sublist]
        else:
            test = None
            test_list = None
    except KeyError as err:
        raise FileNotFoundError() from err

    class_list = list(classes)  # ensure matching ordering between names and encoding
    if class_ids_to_names is not None:  # use readable class names if available
        class_names = [class_ids_to_names[str(c)] for c in class_list]
    else:  # use class identifiers as names otherwise
        class_names = [str(c) for c in class_list]
    encoding = {int(c): i for i, c in enumerate(class_list)}

    logger.info(f"Computing {mode} task.")
    mmap_store = MMapStore(train_list + val_list + (test_list if test_list is not None else []))
    metrics = get_metrics(
        dataset_config.metrics,
        num_classes=len(class_names),
        class_names=class_names,
    )

    # for class weights
    train_classes = [int(filepath.stem.split("_")[-1]) for filepath in train_list]
    train_class_counts = Counter(train_classes)
    total_samples_counts = len(train_classes)

    # inverse frequency
    class_frequencies = {
        encoding[class_id]: count / total_samples_counts
        for class_id, count in train_class_counts.items()
    }

    inverse_frequencies = {
        class_id: 1.0 / frequency for class_id, frequency in class_frequencies.items()
    }

    task = Task(
        task_id="eurocrops",
        encoding=encoding,
        class_weights=inverse_frequencies,
        train_set=TransformDataset(
            EuroCropsDataset(
                train,
                encode=encoding,
                mmap_store=mmap_store,
                config=dataset_config,
                preprocess_config=preprocess_config,
            ),
            collate_fn=custom_collate_fn,
        ),
        val_set=TransformDataset(
            EuroCropsDataset(
                val,
                encode=encoding,
                mmap_store=mmap_store,
                config=dataset_config,
                preprocess_config=preprocess_config,
            ),
            collate_fn=custom_collate_fn,
        ),
        test_set=(
            TransformDataset(
                EuroCropsDataset(
                    test,
                    encode=encoding,
                    mmap_store=mmap_store,
                    config=dataset_config,
                    preprocess_config=preprocess_config,
                ),
                collate_fn=custom_collate_fn,
            )
            if test
            else None
        ),
        num_classes=len(encoding.keys()),
        metrics=metrics,
    )

    return task
