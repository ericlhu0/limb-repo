"""Generate data for dynamics model."""

import time

from limb_repo.dynamics.models.math_dynamics import MathDynamics
from limb_repo.environments.limb_repo_pybullet_env import (
    LimbRepoPyBulletConfig,
    LimbRepoPyBulletEnv,
)
from limb_repo.model_training.data_collection.dynamics_data_generator import (
    DynamicsDataGenerator,
)
from limb_repo.utils import file_utils, utils

if __name__ == "__main__":
    hdf5_saver = file_utils.HDF5Saver(
        file_utils.to_abs_path(f"_the_good_stuff/"),
        file_utils.to_abs_path("_out/temp/"),
    )

    hdf5_saver.combine_temp_hdf5s(
        data_dirs=["01-14_14-30-09", "01-14_14-30-27", "01-14_14-31-20"]
    )
