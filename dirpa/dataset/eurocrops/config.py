import logging
from pathlib import Path
from typing import Literal

from eurocropsml.dataset.config import EuroCropsDatasetConfig as DatasetConfig
from eurocropsml.dataset.config import EuroCropsDatasetPreprocessConfig
from eurocropsml.settings import Settings
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class EuroCropsDatasetConfig(DatasetConfig):
    """EuroCrops dataset config, adding possibility to downsample classes before training."""

    downsample_classes: list[tuple[int, float]] | None = None


class EuroCropsSplit(BaseModel):
    """Configuration for building EuroCrops splits."""

    base_name: str
    data_dir: Path
    random_seed: int
    num_samples: dict[str, str | int | list[int | str]]

    class_downsample: tuple[int | None, float | None] | None = None

    finetune_split_mode: Literal["stratified", "random"] = "random"

    satellite: list[Literal["S1", "S2"]] = ["S2"]

    year: int = 2021
    benchmark: bool = False

    pretrain_classes: dict[str, list[int]]
    finetune_classes: dict[str, list[int]] = {}

    pretrain_regions: list[str]
    finetune_regions: list[str] = []

    @field_validator("data_dir")
    @classmethod
    def relative_path(cls, v: Path) -> Path:
        """Interpret relative paths w.r.t. the project root."""
        if not v.is_absolute():
            v = Settings().data_dir.joinpath(v)
        return v


class EuroCropsConfig(BaseModel):
    """Main configuration for building EuroCrops splits."""

    preprocess: EuroCropsDatasetPreprocessConfig
    split: EuroCropsSplit
