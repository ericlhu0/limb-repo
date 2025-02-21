"""Testing Limb Repo Pybullet Env."""

import numpy as np

from limb_repo.environments.limb_repo_pybullet_env import (
    LimbRepoPyBulletConfig,
    LimbRepoPyBulletEnv,
)
from limb_repo.structs import BodyState, LimbRepoState
from limb_repo.utils import utils

parsed_config = utils.parse_config(
    "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
)
parsed_config.pybullet_config.use_gui = True

env = LimbRepoPyBulletEnv(config=parsed_config)

active_state = BodyState(np.concatenate([parsed_config.active_q, np.zeros(6)]))
passive_state = BodyState(np.concatenate([parsed_config.passive_q, np.zeros(6)]))

env.set_body_state(env.active_id, active_state)
env.set_body_state(env.passive_id, passive_state)

input("test if grasp constraint is stable (no drifting)")

# test if grasp constraint is stable (no drifting)
# env.set_limb_repo_constraint()
# for i in range(1000):
#     env.step()
#     print(env.get_limb_repo_state().active)
#     print(env.get_limb_repo_state().passive)

# input("test if torque control looks reasonable")

# # test if torque control looks reasonable

# for i in range(10000):
#     env.send_torques(np.array([0, 0, 0, 0, 0, 1]))
#     time.sleep(1 / 200)

# input("done")

env.set_limb_repo_state(
    LimbRepoState(
        np.array(
            [
                0.1851,
                0.0858,
                -1.7703,
                -1.2230,
                0.6085,
                0.4204,
                -0.9200,
                -0.4792,
                -0.5446,
                -0.7191,
                0.4741,
                0.0863,
                2.3607,
                -0.2981,
                -1.0999,
                1.2585,
                -2.5501,
                1.5502,
                1.1546,
                -0.4069,
                -0.2738,
                0.2241,
                -1.1725,
                -0.4323,
            ]
        )
    )
)

while True:
    env.step()
