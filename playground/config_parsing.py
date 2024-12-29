"""Testing config parsing."""

from typing import Type, TypeVar

import omegaconf
from omegaconf import OmegaConf

from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletConfig
from limb_repo.environments.pybullet_env import PyBulletConfig


def parse_config(path_to_yaml: str) -> omegaconf.DictConfig:
    """Parse the configuration file."""
    config = OmegaConf.load(path_to_yaml)

    # to get around mypy "Keywords must be strings"
    # and "value after ** should be a mapping"
    config_dict = {str(key): value for key, value in dict(config).items()}

    config = OmegaConf.structured(LimbRepoPyBulletConfig(**config_dict))
    assert isinstance(config, omegaconf.DictConfig)

    return config


T = TypeVar("T")


def parametric_parse_config(
    path_to_yaml: str, config_class: Type[T]
) -> omegaconf.DictConfig:
    """Parse config with parametric dataclass."""
    config = OmegaConf.load(path_to_yaml)

    # Convert config into dictionary and initialize the specified dataclass
    config_dict = {str(key): value for key, value in dict(config).items()}

    config = OmegaConf.structured(config_class(**config_dict))
    assert isinstance(config, omegaconf.DictConfig)
    return config


def parametric_parse_config2(
    path_to_yaml: str, config_class: Type[T]
) -> omegaconf.DictConfig:
    """Parse config with parametric dataclass.

    (for PyBulletConfig)
    """
    config = OmegaConf.load(path_to_yaml)

    # Convert config into dictionary and initialize the specified dataclass
    config_dict = {str(key): value for key, value in dict(config).items()}[
        "pybullet_config"
    ]

    config = OmegaConf.structured(config_class(**config_dict))
    assert isinstance(config, omegaconf.DictConfig)
    return config


if __name__ == "__main__":
    parsed_config = parse_config("assets/configs/test_env_config.yaml")
    print(parsed_config)
    print(parsed_config.pybullet_config)

    parsed_config = parametric_parse_config(
        "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
    )
    print(parsed_config)
    print(parsed_config.pybullet_config)

    parsed_config = parametric_parse_config2(
        "assets/configs/test_env_config.yaml", PyBulletConfig
    )
    print(parsed_config)
