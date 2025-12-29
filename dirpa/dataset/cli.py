import json
import logging
from pathlib import Path
from typing import Optional, Type, TypeVar, cast

import typer
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel

from dirpa.dataset.eurocrops.config import EuroCropsConfig
from dirpa.dataset.eurocrops.splits import create_splits
from dirpa.settings import Settings

logger = logging.getLogger(__name__)

datasets_app = typer.Typer(name="datasets")

ConfigT = TypeVar("ConfigT", bound=BaseModel)
OverridesT = Optional[list[str]]


def build_eurocrops_app(config_class: Type[EuroCropsConfig]) -> typer.Typer:
    """Build cli component for preparing EuroCrops dataset."""

    app = typer.Typer(name="eurocrops")

    def build_config(overrides: OverridesT, config_path: str | None = None) -> EuroCropsConfig:

        if config_path is not None:
            config_dir = Path(config_path)
        else:
            config_dir = Settings().experiment_dir.joinpath("common")
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            if overrides is None:
                overrides = []
            composed_config = compose(config_name="eurocrops", overrides=overrides)
        config = config_class(**OmegaConf.to_object(composed_config))  # type: ignore[arg-type]
        return config

    @app.command(name="config")
    def print_config(
        overrides: OverridesT = typer.Argument(None, help="Overrides to config"),
    ) -> None:
        """Print currently used config."""
        config = build_config(overrides)
        print(OmegaConf.to_yaml(json.loads(config.json())))

    @app.command()
    def build_splits(
        config_path: str = typer.Option(None, "--config-path", help="Path to config.yaml file."),
        overrides: OverridesT = typer.Argument(None, help="Overrides to split config"),
    ) -> None:
        config = cast(EuroCropsConfig, build_config(overrides, config_path))
        create_splits(config.split, config.preprocess.raw_data_dir.parent)

    return app


datasets_app.add_typer(build_eurocrops_app(EuroCropsConfig))
