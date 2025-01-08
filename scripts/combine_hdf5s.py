"""Generate data for dynamics model."""

from limb_repo.dynamics.models.math_dynamics_no_n_vector import MathDynamicsNoNVector
from limb_repo.environments.limb_repo_pybullet_env import (
    LimbRepoPyBulletConfig,
    LimbRepoPyBulletEnv,
)
from limb_repo.model_training.data_collection.dynamics_data_generator import (
    DynamicsDataGenerator,
)
from limb_repo.utils import utils, file_utils

if __name__ == "__main__":
    hdf5_saver = file_utils.HDF5Saver(
        file_utils.to_abs_path("_the_good_stuff/7500000.hdf5"),
        file_utils.to_abs_path("_out/temp/"),
    )

    hdf5_saver.combine_temp_hdf5s()
