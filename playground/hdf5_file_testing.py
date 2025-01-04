from limb_repo.dynamics.models.math_dynamics_no_n_vector import MathDynamicsNoNVector
from limb_repo.environments.limb_repo_pybullet_env import (
    LimbRepoPyBulletConfig,
    LimbRepoPyBulletEnv,
)
from limb_repo.model_training.data_collection.dynamics_data_generator import (
    DynamicsDataGenerator,
)
from limb_repo.utils import utils
from limb_repo.utils.file_utils import HDF5Saver, to_abs_path

parsed_config = utils.parse_config(
    "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
)

environment = LimbRepoPyBulletEnv(parsed_config)
dynamics_model = MathDynamicsNoNVector(parsed_config)

data_generator = DynamicsDataGenerator(
    environment,
    dynamics_model,
    environment.active_joint_min,
    environment.active_joint_max,
)

data_generator.generate_data(
    10, to_abs_path("playground/out/test.hdf5"), to_abs_path("playground/out/temp/")
)
