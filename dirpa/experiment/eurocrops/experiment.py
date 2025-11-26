from typing import Literal

import eurocropsml.settings
from eurocropsml.dataset.splits import get_split_dir

from dirpa.experiment.eurocrops.config import EuroCropsTransferConfig
from dirpa.experiment.eurocrops.utils import load_task
from dirpa.experiment.transfer import (
    TransferExperiment,
    TransferExperimentBuilder,
    build_pretrain_experiment,
)
from dirpa.settings import Settings


class EuroCropsExperimentBuilder(TransferExperimentBuilder[EuroCropsTransferConfig]):
    """Class for building EuroCrops related experiments."""

    def __init__(self, config: EuroCropsTransferConfig):
        super().__init__(config)
        self.dataset_config = config.eurocrops_dataset

    def build_experiment(self, mode: Literal["pretrain", "finetune"]) -> TransferExperiment:
        """Build transfer experiment."""
        model_channels = self.config.model.in_channels
        if model_channels != self.dataset_config.total_num_channels:
            raise AssertionError(
                "The number of channels in the model config "
                f"({self.config.model.in_channels}) does not match the number of actual channels "
                f"from the dataset config ({self.dataset_config.total_num_channels}). Please "
                "adjust the number of channels in the model config and make sure you are using the"
                " correct set of data sources as well as are not accidentally removing any S2 "
                "bands you were not planning to remove."
            )

        experiment_dir = Settings().experiment_dir
        experiment_dir.mkdir(exist_ok=True, parents=True)

        data_dir = eurocropsml.settings.Settings().data_dir

        split_dir = get_split_dir(data_dir, self.config.split.base_name)

        if mode == "finetune":
            finetuning_tasks = {
                f"eurocrops_finetuning_maxsamples_{num}": (
                    load_task(
                        split_dir,
                        mode="finetuning",
                        split_config=self.config.split,
                        preprocess_config=self.config.preprocess,
                        dataset_config=self.dataset_config,
                        max_samples=num,
                    ),
                    self.config.finetune,
                )
                for num in self.dataset_config.max_samples
            }

            return build_pretrain_experiment(
                pretrain_experiment_config=self.config.pretrain,
                pretrain_task=None,
                finetuning_tasks=finetuning_tasks,
                model_config=self.config.model,
                experiment_dir=experiment_dir,
            )

        elif mode == "pretrain":

            pretrain_task = load_task(
                split_dir,
                mode="pretraining",
                split_config=self.config.split,
                preprocess_config=self.config.preprocess,
                dataset_config=self.dataset_config,
            )
            return build_pretrain_experiment(
                pretrain_experiment_config=self.config.pretrain,
                pretrain_task=pretrain_task,
                finetuning_tasks=None,
                model_config=self.config.model,
                experiment_dir=experiment_dir,
            )

        else:
            raise ValueError(f"{mode} not implemented. Choose either 'pretrain' or 'finetune'.")
