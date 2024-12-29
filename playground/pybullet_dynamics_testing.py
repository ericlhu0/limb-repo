"""Testing LR Pybullet Env and PyBullet Dynamics."""

import numpy as np
import time

from limb_repo.dynamics.models.pybullet_dynamics import PyBulletDynamics
from limb_repo.environments.lr_pybullet_env import LRPyBulletEnv
from limb_repo.structs import BodyState

parsed_config = LRPyBulletEnv.parse_config("assets/configs/test_env_config.yaml")
pybullet_model = PyBulletDynamics(parsed_config)

# test torque control
for i in range(5000):
    action = np.random.rand(6) * 0
    print(f"loop {i}")
    next_state = pybullet_model.step(action)
    print(next_state.active_q, next_state.passive_q)

    time.sleep(0.01)