"""Find active configurations where a grasp is possible."""

from itertools import product

import numpy as np

from limb_repo.dynamics.models.math_dynamics_no_n_vector import MathDynamicsNoNVector
from limb_repo.environments.limb_repo_pybullet_env import (
    LimbRepoPyBulletConfig,
    LimbRepoPyBulletEnv,
)
from limb_repo.model_training.data_collection.dynamics_data_generator import (
    DynamicsDataGenerator,
)
from limb_repo.structs import BodyState
from limb_repo.utils import utils

if __name__ == "__main__":
    parsed_config = utils.parse_config(
        "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
    )

    environment = LimbRepoPyBulletEnv(parsed_config)
    dynamics_model = MathDynamicsNoNVector(parsed_config)
    active_joint_min = environment.active_joint_min
    active_joint_max = environment.active_joint_max

    data_generator = DynamicsDataGenerator(
        environment,
        dynamics_model,
        active_joint_min,
        active_joint_max,
    )

    # Define the step size
    step_size = 0.5

    # Generate a range for each dimension of active_joint_min and active_joint_max
    ranges = [
        np.arange(active_joint_min[i], active_joint_max[i], step_size)
        for i in range(len(active_joint_min))
    ]

    valid_configs = []

    # Use itertools.product to iterate over all combinations
    for active_joint_tuple in product(*ranges):
        active_joint = np.array(active_joint_tuple)  # Convert tuple to numpy array
        init_active_state = BodyState(
            np.concatenate([active_joint, np.zeros_like(active_joint)])
        )
        if data_generator.find_passive_config(init_active_state) is not None:
            valid_configs.append(active_joint)

    print(len(valid_configs))

    np.save("out/valid_configs.npy", np.array(valid_configs))

    # data_generator.generate_data(
    #     10, to_abs_path("out/test.hdf5"), to_abs_path("out/temp/")
    # )
