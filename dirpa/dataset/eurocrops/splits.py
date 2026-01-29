"""Generating region based EuroCrops dataset splits."""

import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal, cast

import numpy as np
from eurocropsml.dataset.config import EuroCropsSplit
from eurocropsml.dataset.splits import split_dataset_by_class
from eurocropsml.dataset.utils import (
    _create_final_dict,
    _filter_regions,
    _sample_max_samples,
    _save_counts_to_csv,
    _save_to_dict,
    _save_to_json,
    _split_dataset,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from dirpa.dataset.eurocrops.utils import _downsample_class

logger = logging.getLogger(__name__)


def get_split_dir(split_dir: Path, split_name: str) -> Path:
    """Get directory where splits are saved.

    Args:
        split_dir: Path for the splits root directory.
        split_name: Subdirectory name for named split.

    Returns:
        The full path, consisting of root directory and subdirectory.
    """
    return split_dir.joinpath("split", split_name)


def create_splits(
    split_config: EuroCropsSplit,
    split_dir: Path,
) -> None:
    """Create EuroCrops dataset splits.

    Args:
        split_config: Configuration used for splitting dataset.
        split_dir: Data directory where split folder is saved.

    """

    split_dir = get_split_dir(split_dir, split_config.base_name)
    splits = split_config.pretrain_classes
    class_downsample = split_config.class_downsample
    for split in splits:
        _build_finetune_dataset_split(
            data_dir=split_config.data_dir,
            split=split,
            satellite=split_config.satellite,
            year=str(split_config.year),
            split_dir=split_dir,
            pretrain_classes=set(split_config.pretrain_classes[split]),
            finetune_classes=(
                finetune_classes
                if (finetune_classes := split_config.finetune_classes.get(split)) is None
                else set(finetune_classes)
            ),
            pretrain_regions=set(split_config.pretrain_regions),
            finetune_regions=set(split_config.finetune_regions),
            num_samples=split_config.num_samples,
            class_downsample=class_downsample,
            force_rebuild=False,
            seed=split_config.random_seed,
            finetune_split_mode=split_config.finetune_split_mode,
        )


def _build_finetune_dataset_split(
    data_dir: Path,
    split_dir: Path,
    satellite: list[Literal["S1", "S2"]],
    split: Literal["region", "class"],
    year: str,
    num_samples: dict[str, str | int | list[int | str]],
    seed: int,
    pretrain_classes: set,
    finetune_classes: set | None = None,
    pretrain_regions: set | None = None,
    finetune_regions: set | None = None,
    class_downsample: tuple[int | None, float | None] | None = None,
    force_rebuild: bool = False,
    finetune_split_mode: Literal["stratified", "random"] = "random",
) -> None:
    """Build data split for EuroCrops data.

    Args:
        data_dir: Directory where labels and data are stored.
        split_dir: Directory where split file is going to be saved to.
        split: Kind of data split to apply.
        satellite: Whether to build the splits using Sentinel-1 or Sentinel-2 or both.
        year: Year for which data are to be processed.
        num_samples: Number of samples to sample for finetuning.
        seed: Randoms seed,
        pretrain_classes: Classes of the requested dataset split for
            hyperparameter tuning and pretraining.
        finetune_classes: Classes of the requested dataset split for finetuning.
        pretrain_regions: Regions of the requested dataset split for
            hyperparameter tuning and pretraining.
        finetune_regions: Regions of the requested dataset split for finetuning.
            None if EuroCrops should only be used for pretraining.
        class_downsample: Class identifier. If specified, for the pre-training split,
            the given class will be downsampled to the given frequency/number or to the median
            frequency of all other classes, in case no downsample probability is given.
            If None, no downsampling is taking place. If the identifier is None, all classes will
            be downsampled.
        force_rebuild: Whether to rebuild split if split file already exists.
        finetune_split_mode: Whether to use stratified or random splitting for fine-tuning.
            stratified splitting, based on the number of samples for class c:
                - 1 sample for class c: assigned to train
                - 2 samples for class c: 1 to train and the other one randomly to val or test
                - >= 3 samples: randomly distributed between train, val, and test while making sure
                that the val and test ratios are being met.

    Raises:
        FileNotFoundError: If `data_dir` is not a directory.
        ValueError: If `pretrain_regions` is not specified but we want to split by regions.
    """

    if not data_dir.is_dir():
        raise FileNotFoundError(str(data_dir) + " is not a directory.")

    if not force_rebuild:
        split_files = [
            split_dir.joinpath("pretrain", f"{split}_split.json"),
            split_dir.joinpath("meta", f"{split}_split.json"),
        ]
        for num in num_samples["train"]:  # type: ignore[union-attr]
            split_files.append(split_dir.joinpath("finetune", f"{split}_split_{num}.json"))

        if all(file.is_file() for file in split_files):
            logger.info(
                "Files already exist and force_rebuild=False. "
                f"Skipping recreation of {split}-split.",
            )
            return
    logger.info(f"Creating {split}-split...")
    if split == "class":
        split_dataset_by_class(
            data_dir,
            split_dir,
            satellite,
            year=year,
            num_samples=num_samples,
            pretrain_classes=pretrain_classes,
            finetune_classes=finetune_classes,
            class_downsample=class_downsample,
            seed=seed,
        )
    else:
        if pretrain_regions is None:
            raise ValueError("Please specify the relevant pretrain regions to sample from.")
        split_dataset_by_region(
            data_dir,
            split_dir,
            split,
            satellite,
            year=year,
            num_samples=num_samples,
            pretrain_classes=pretrain_classes,
            finetune_classes=finetune_classes,
            pretrain_regions=pretrain_regions,
            finetune_regions=finetune_regions,
            class_downsample=class_downsample,
            seed=seed,
            finetune_split_mode=finetune_split_mode,
        )


def split_dataset_by_region(
    data_dir: Path,
    split_dir: Path,
    split: Literal["region"],
    satellite: list[Literal["S1", "S2"]],
    year: str,
    num_samples: dict[str, str | int | list[int | str]],
    seed: int,
    pretrain_classes: set[int],
    pretrain_regions: set[str],
    finetune_classes: set[int] | None = None,
    finetune_regions: set[str] | None = None,
    class_downsample: tuple[int | None, float | None] | None = None,
    test_size: float = 0.2,
    finetune_split_mode: Literal["stratified", "random"] = "random",
) -> None:
    """Split dataset by regions or regions and classes.

    Args:
        data_dir: Path that contains `.npy` files where labels and data are stored.
        split_dir: Directory where splits are going to be saved to.
        split: Kind of data split to apply.
        satellite: Whether to build the splits using Sentinel-1 or Sentinel-2 or both.
        year: Year for which data are to be processed.
        num_samples: Number of samples to sample for finetuning.
        seed: Random seed for data split.
        pretrain_classes: Classes of the requested dataset split for
            hyperparameter tuning and pretraining.
        finetune_classes: Classes of the requested dataset split for finetuning.
        pretrain_regions: Regions of the requested dataset split for
            hyperparameter tuning and pretraining.
        finetune_regions: Regions of the requested dataset split for finetuning.
            None if EuroCrops should only be used for pretraining.
        class_downsample: Class identifier. If specified, for the pre-training split,
            the given class will be downsampled to the given frequency/number or to the median
            frequency of all other classes, in case no downsample probability is given.
            If None, no downsampling is taking place. If the identifier is None, all classes will
            be downsampled.
        finetune_split_mode: Whether to use stratified or random splitting for fine-tuning.
            stratified splitting, based on the number of samples for class c:
                - 1 sample for class c: assigned to train
                - 2 samples for class c: 1 to train and the other one randomly to val or test
                - >= 3 samples: randomly distributed between train, val, and test while making sure
                that the val and test ratios are being met.
        test_size: Amount of data used for validation (test set).
            Defaults to 0.2.

    Raises:
        Exception: If there are similar samples within pretrain and finetune data-split.
    """

    regions = (
        pretrain_regions | finetune_regions if finetune_regions is not None else pretrain_regions
    )

    classes = (
        pretrain_classes | finetune_classes
        if finetune_classes is not None and split == "region"
        else pretrain_classes
    )

    # split into pretrain and finetune dataset
    pretrain_dataset, finetune_dataset = _split_dataset(
        data_dir=data_dir,
        satellite=satellite,
        year=year,
        pretrain_classes=classes,
        finetune_classes=None,
        regions=set(regions),
    )

    finetune_dataset = pretrain_dataset.copy()

    pretrain_dataset = _filter_regions(pretrain_dataset, pretrain_regions)

    if class_downsample is not None and (
        class_downsample[0] in pretrain_classes or class_downsample[0] is None
    ):
        pretrain_list: list[str] = _downsample_class(
            pretrain_dataset,
            seed=seed,
            class_key=class_downsample[0],
            no_sample_to=class_downsample[1],
        )
    else:
        pretrain_list = [file for files in pretrain_dataset.values() for file in files]

    if (
        finetune_dataset is not None and finetune_regions is not None
    ):  # otherwise EuroCrops is solely used for pretraining
        finetune_dataset = _filter_regions(finetune_dataset, finetune_regions)

        _create_finetune_set(
            finetune_dataset,
            split_dir.joinpath("finetune"),
            split,
            pretrain_list,
            num_samples,
            test_size,
            seed,
            finetune_split_mode,
        )

        # sorting list to make train_test_split deterministic
        pretrain_list.sort()
        # save pretraining split
        train, val = train_test_split(pretrain_list, test_size=test_size, random_state=seed)

        pretrain_dict = _save_to_dict(train, val)

        _save_to_json(split_dir.joinpath("pretrain", f"{split}_split.json"), pretrain_dict)

        _save_counts_to_csv(pretrain_list, split_dir.joinpath("counts", "pretrain"), split)

        meta_dict: dict = {
            "train": _create_final_dict(train, pretrain_regions),
            "val": _create_final_dict(val, pretrain_regions),
        }

        _save_to_json(split_dir.joinpath("meta", f"{split}_split.json"), meta_dict)


def _create_finetune_set(
    finetune_dataset: dict[int, list[str]],
    split_path: Path,
    split: str,
    pretrain_list: list[str],
    num_samples: dict[str, str | int | list[int | str]],
    test_size: float,
    seed: int,
    finetune_split_mode: Literal["random", "stratified"],
) -> None:
    finetune_list: list[str] = [
        value for values_list in finetune_dataset.values() for value in values_list
    ]

    finetune_list.sort()
    if set(pretrain_list) & set(finetune_list):
        raise Exception(
            f"There are {len((set(pretrain_list) & set(finetune_list)))} "
            "equal samples within upstream and downstream task."
        )

    if finetune_split_mode == "random":
        finetune_train, finetune_val = train_test_split(
            finetune_list, test_size=2 * test_size, random_state=seed
        )
        new_test_size = np.around(test_size / (2 * test_size), 2)
        finetune_val, finetune_test = train_test_split(
            finetune_val, test_size=new_test_size, random_state=seed
        )

        if num_samples["validation"] != "all":
            num_samples["validation"] = int(cast(int, num_samples["validation"]))
        if num_samples["test"] != "all":
            num_samples["test"] = int(cast(int, num_samples["test"]))
        if (
            isinstance(num_samples["validation"], int)
            and len(finetune_val) > num_samples["validation"]
        ):
            finetune_val = resample(
                finetune_val,
                replace=False,
                n_samples=num_samples["validation"],
                random_state=seed,
            )
        if isinstance(num_samples["test"], int) and len(finetune_test) > num_samples["test"]:
            finetune_test = resample(
                finetune_test,
                replace=False,
                n_samples=num_samples["test"],
                random_state=seed,
            )

    else:
        finetune_train = []
        finetune_val = []
        finetune_test = []

        # keep track of high-shot samples reserved for final stratified split
        high_shot_data_pool = []

        # handle N=1 and N=2 classes separately (few-shot protocol)
        n1_files = [files[0] for _, files in finetune_dataset.items() if len(files) == 1]
        n2_files = {c: files for c, files in finetune_dataset.items() if len(files) == 2}
        n_remain_files = {c: files for c, files in finetune_dataset.items() if len(files) > 2}
        n2_classes = list(n2_files.keys())

        finetune_train.extend(n1_files)
        finetune_train.extend([files[0] for _, files in n2_files.items()])

        random.shuffle(n2_classes)
        split_point = len(n2_classes) // 2
        c_val_query = set(n2_classes[:split_point])
        c_test_query = set(n2_classes[split_point:])

        finetune_test.extend([n2_files[c][1] for c in c_test_query])
        finetune_val.extend([n2_files[c][1] for c in c_val_query])
        # import pdb
        # pdb.set_trace()
        for c, files in n_remain_files.items():
            random.shuffle(files)
            # N>=3: all samples for this class go to the high_shot_data_pool
            # split them later to hit the target sizes
            for f in files:
                high_shot_data_pool.append((c, f))

        # Calculate remaining capacity for Val/Test splits
        # N=2 classes already added 1 sample to the Test set
        full_dataset_size = len(finetune_list)
        current_val_size = len(finetune_val)
        current_test_size = len(finetune_test)
        test_size_target = cast(
            int,
            (
                int(math.ceil(full_dataset_size * test_size))
                if num_samples["test"] == "all"
                else num_samples["test"]
            ),
        )
        val_size_target = cast(
            int,
            (
                int(math.ceil(full_dataset_size * test_size))
                if num_samples["validation"] == "all"
                else num_samples["validation"]
            ),
        )
        test_remaining_capacity = int(max(0, test_size_target - current_test_size))
        val_remaining_capacity = int(max(0, val_size_target - current_val_size))

        # calculate the total samples we need to draw from the pool for val and test
        total_to_draw = test_remaining_capacity + val_remaining_capacity

        # check if high-shot pool is large enough
        total_pool_size = len(high_shot_data_pool)

        if total_to_draw > total_pool_size:
            logger.warning(
                f"Warning: Pool size ({total_pool_size}) is smaller\
                than requested val+test ({total_to_draw})."
            )
            # adjusting the val/test capacities proportionally from the pool
            # this will result in a slightly smaller Val/Test set than requested, but guarantees
            # that the ratios are preserved and no data is double-counted.
            val_remaining_capacity = math.ceil(
                val_remaining_capacity * (total_pool_size / total_to_draw)
            )
            test_remaining_capacity = total_pool_size - val_remaining_capacity

        # stratified split for higher-shot data pool
        # calculate sampling ratios for val and test from the *needed* capacity
        # we use these ratios to guide stratified sampling on the pool
        val_ratio = val_remaining_capacity / total_pool_size if total_pool_size > 0 else 0
        test_ratio = test_remaining_capacity / total_pool_size if total_pool_size > 0 else 0

        # group samples by class for stratification
        class_pools = defaultdict(list)
        for c, f in high_shot_data_pool:
            class_pools[c].append(f)

        # perform the stratified sampling with a "Train-First" guarantee
        for _, class_samples in class_pools.items():
            random.shuffle(class_samples)
            class_count = len(class_samples)

            # Reserve 1 for train immediately
            reserved_for_train = 1
            available_for_splits = class_count - reserved_for_train

            # Calculate target counts based on the remaining available samples
            val_count = math.floor(class_count * val_ratio)
            test_count = math.floor(class_count * test_ratio)

            # Final safety check to ensure we don't exceed available_for_splits
            val_count = min(val_count, available_for_splits)
            test_count = min(test_count, available_for_splits - val_count)

            # Split the samples
            val_samples = class_samples[:val_count]
            test_samples = class_samples[val_count : val_count + test_count]
            # Everything else (including the reserved 1) goes to train
            train_samples = class_samples[val_count + test_count :]

            finetune_val.extend(val_samples)
            finetune_test.extend(test_samples)
            finetune_train.extend(train_samples)

    sample_list: list[str | int]
    if isinstance(num_samples["train"], list):
        sample_list = num_samples["train"]
    else:
        sample_list = [cast(int, num_samples["train"])]

    if "all" in sample_list:
        _save_to_json(
            split_path.joinpath(f"{split}_split_all.json"),
            _save_to_dict(finetune_train, finetune_val, finetune_test),
        )
        sample_list.remove("all")
    for max_samples in sample_list:
        train = _sample_max_samples(finetune_train, max_samples, seed)
        _save_to_json(
            split_path.joinpath(f"{split}_split_{max_samples}.json"),
            _save_to_dict(train, finetune_val, finetune_test),
        )

    _save_counts_to_csv(finetune_list, split_path.parents[0].joinpath("counts", "finetune"), split)
