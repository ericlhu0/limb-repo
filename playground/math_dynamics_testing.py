"""Testing Limb Repo Pybullet Env and PyBullet Dynamics."""

import numpy as np

from limb_repo.dynamics.models.math_dynamics_no_n_vector import MathDynamicsNoNVector
from limb_repo.dynamics.models.math_dynamics_with_n_vector import MathDynamicsWithNVector
from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletConfig
from limb_repo.utils import utils

parsed_config = utils.parse_config("assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig)
# math_dynamics = MathDynamicsWithNVector(parsed_config)
math_dynamics = MathDynamicsNoNVector(parsed_config)

# test torque control
for i in range(5000):
    action = np.random.rand(6) * 10 - 5
    print(f"loop {i}")
    next_state = math_dynamics.step(action)
    print(next_state.active_q, next_state.passive_q)
