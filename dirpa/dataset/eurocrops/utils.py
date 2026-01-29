import logging
import random
from itertools import chain
from pathlib import Path
from statistics import median
from typing import Literal

from sklearn.utils import resample

logger = logging.getLogger(__name__)


def _downsample(
    data: dict[str, list[Path]],
    drop_class: int,
    drop_prob: float,
    satellites: list[Literal["S2"]],
) -> dict[str, list[Path]]:
    filtered_data: dict[str, list[Path]] = {}
    for satellite in satellites:
        filtered_data[satellite] = []
        for path in data[satellite]:
            cls = int(path.stem.split("_")[-1])
            if cls != drop_class or random.random() >= drop_prob:
                filtered_data[satellite].append(path)
    return filtered_data


def _downsample_class(
    dataset_dict: dict[int, list[str]],
    seed: int,
    class_key: int | None = None,
    no_sample_to: float | int | None = None,
) -> list[str]:
    """Downsample class to n_samples.

    Args:
        dataset_dict: Dictionary with classes as keys and lists of file paths as values
        seed: Randoms seed
        class_key: Class to downsample. If None, all classes will be downsamples to n_samples.
        no_sample_to: Frequency or number of samples to which to downsample the class to.
            If not specified, median of remaining classes will be used.

    Returns:
        List of file paths.

    Raises:
        ValueError: If neither class_key nor n_samples is speficied.

    """

    if class_key is None:
        if no_sample_to is None or no_sample_to == 0.0:
            raise ValueError(
                "Please set the downsample proportion greater 0.0\
                             or specify the class_key."
            )
        else:
            for key, files in dataset_dict.items():
                no_files = len(files)
                if no_sample_to >= 1.0:
                    n_samples = int(no_sample_to)
                else:
                    n_samples = int(no_files * no_sample_to)
                if len(files) > n_samples:
                    dataset_dict[key] = resample(
                        files,
                        replace=False,
                        n_samples=n_samples,
                        random_state=seed,
                    )

    elif class_key in dataset_dict:
        if no_sample_to == 0.0 or no_sample_to == 0:
            dataset_dict.pop(class_key, None)
        else:
            if no_sample_to is None:
                n_samples = int(
                    median([len(val) for key, val in dataset_dict.items() if key != class_key])
                )
            else:
                if no_sample_to >= 1.0:
                    n_samples = int(no_sample_to)
                else:
                    no_files = len(dataset_dict[class_key])
                    n_samples = int(no_files * no_sample_to)
                no_files = len(dataset_dict[class_key])
            if len(dataset_dict[class_key]) > n_samples:
                dataset_dict[class_key] = resample(
                    dataset_dict[class_key],
                    replace=False,
                    n_samples=n_samples,
                    random_state=seed,
                )

    return list(chain.from_iterable(dataset_dict.values()))


def _format_band_name(band: str) -> str:
    if band.isdigit():
        # For numeric bands, remove leading zeros
        return f"B{band.lstrip('0')}"
    else:
        # For special bands like "8A"
        return f"B{band}"
