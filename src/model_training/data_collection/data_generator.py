"""Data generator for dynamics model."""

import numpy as np
import omegaconf

from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletEnv
from limb_repo.utils.utils import parse_config

class DataGenerator:
    """Data generator for dynamics model."""
    def __init__(self, config: omegaconf.DictConfig) -> None:
        self.env = LimbRepoPyBulletEnv(config=config)

    def generate_data(self) -> np.ndarray:
        """Generate data for dynamics model."""
        # Generate data
        data = np.random.rand(self.config.num_samples, self.config.num_features)
        return data

    def save_data(self, data: np.ndarray) -> None:
        """Save data to hdf5 file."""
