"""Testing Limb Repo Pybullet Env."""

import time

import numpy as np

from limb_repo.environments.limb_repo_pybullet_env import (
    LimbRepoPyBulletConfig,
    LimbRepoPyBulletEnv,
)
from limb_repo.structs import BodyState
from limb_repo.utils import utils

parsed_config = utils.parse_config(
    "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
)

env = LimbRepoPyBulletEnv(config=parsed_config)

active_state = BodyState(np.concatenate([parsed_config.active_q, np.zeros(6)]))
passive_state = BodyState(np.concatenate([parsed_config.passive_q, np.zeros(6)]))

env.set_body_state(env.active_id, active_state)
env.set_body_state(env.passive_id, passive_state)

input("test if grasp constraint is stable (no drifting)")

# test if grasp constraint is stable (no drifting)
env.set_limb_repo_constraint()
for i in range(1000):
    env.step()
    print(env.get_limb_repo_state().active)
    print(env.get_limb_repo_state().passive)

input("test if torque control looks reasonable")

# test if torque control looks reasonable

for i in range(10000):
    env.send_torques(np.array([0, 0, 0, 0, 0, 1]))
    time.sleep(1 / 200)

input("done")
