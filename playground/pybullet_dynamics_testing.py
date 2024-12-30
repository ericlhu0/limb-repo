"""Testing Limb Repo Pybullet Env and PyBullet Dynamics."""

import numpy as np
import time

from limb_repo.dynamics.models.pybullet_dynamics import PyBulletDynamics
from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletConfig
from limb_repo.structs import BodyState
from limb_repo.utils import utils

parsed_config = utils.parse_config("assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig)
pybullet_model = PyBulletDynamics(parsed_config)

# test torque control
for i in range(5000):
    action = np.random.rand(6) * 1
    print(f"loop {i}")
    next_state = pybullet_model.step(action)
    print(next_state.active_q, next_state.passive_q)

    time.sleep(0.01)