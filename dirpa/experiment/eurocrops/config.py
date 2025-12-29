import logging
from typing import Any

from eurocropsml.dataset.config import EuroCropsDatasetPreprocessConfig
from pydantic import BaseModel

from dirpa.dataset.eurocrops.config import EuroCropsDatasetConfig, EuroCropsSplit
from dirpa.experiment.base import TrainExperimentConfig
from dirpa.models.transformer import TransformerConfig

logger = logging.getLogger(__name__)


class EuroCropsTransferConfig(BaseModel):
    """Configuration for the EuroCrops transfer experiment.

    Args:
        base_name: Name under which experiments are stored.
        model: Configuration for a (pre)-trained model.
        pretrain: Experiment configuration for pretraining.
        finetune: Experiment configuration for finetuning.
        eurocrops_dataset: Configuration for EuroCrops dataset.
        split: Configuration for EuroCrops splits.
        preprocess: Configuration for preprocessing EuroCrops dataset.
    """

    base_name: str
    model: TransformerConfig
    pretrain: TrainExperimentConfig
    finetune: TrainExperimentConfig

    eurocrops_dataset: EuroCropsDatasetConfig

    split: EuroCropsSplit
    preprocess: EuroCropsDatasetPreprocessConfig

    def __init__(self, **data: Any):
        super().__init__(**data)
