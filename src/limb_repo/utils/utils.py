"""Utility functions."""

import os
from typing import Type, TypeVar

import omegaconf
from omegaconf import OmegaConf


def get_root_path() -> str:
    """Get the root path of the repository."""
    return os.path.abspath(os.path.join(__file__, "../../../.."))


def to_abs_path(input_path: str) -> str:
    """Get the absolute path of the repository."""
    return os.path.abspath(os.path.join(get_root_path(), input_path))


T = TypeVar("T")


def parse_config(path_to_yaml: str, config_class: Type[T]) -> omegaconf.DictConfig:
    """Parse config file with parametric dataclass."""
    config = OmegaConf.load(path_to_yaml)

    # Convert config into dictionary and initialize the specified dataclass
    config_dict = {str(key): value for key, value in dict(config).items()}

    config = OmegaConf.structured(config_class(**config_dict))
    assert isinstance(config, omegaconf.DictConfig)
    return config
