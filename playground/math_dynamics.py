import time

import numpy as np

from limb_repo.dynamics.checks.check_ee_kinematics import check_ee_kinematics
from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.dynamics.models.math_dynamics_no_n_vector import MathDynamicsNoNVector
from limb_repo.dynamics.models.math_dynamics_with_n_vector import (
    MathDynamicsWithNVector,
)
from limb_repo.environments.lr_pybullet_env import LRPyBulletEnv

parsed_config = LRPyBulletEnv.parse_config("assets/configs/test_env_config.yaml")
# parsed_config.pybullet_config.use_gui = False
dynamics_model0: BaseDynamics = MathDynamicsNoNVector(parsed_config)
# dynamics_model: BaseDynamics = MathDynamicsWithNVector(parsed_config)


for i in range(1000000):
    action = np.array([1, 0, 1, 0, 1, 0])
    print(f"loop {i}")
    next_state0 = dynamics_model0.step(action)
    # next_state = dynamics_model.step(action)
    dynamics_model0.env.get_lr_ee_state()
    time.sleep(0.01)
