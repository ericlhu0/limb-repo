"""Testing LR Pybullet Env."""

import numpy as np

from limb_repo.environments.lr_pybullet_env import LRPyBulletEnv
from limb_repo.structs import BodyState

parsed_config = LRPyBulletEnv.parse_config("assets/configs/test_env_config.yaml")

env = LRPyBulletEnv(config=parsed_config)

active_state = BodyState(np.concatenate([parsed_config.active_q, np.zeros(6 + 6)]))
passive_state = BodyState(np.concatenate([parsed_config.passive_q, np.zeros(6 + 6)]))

new_active_state = BodyState(
    np.concatenate(
        [active_state[active_state.pos_slice] + np.array([0.1] * 6), np.zeros(6 + 6)]
    )
)

print(env.get_body_state(env.active_id))

env.set_body_state(env.active_id, new_active_state, set_vel=True, zero_acc=True)
env.set_body_state(env.passive_id, passive_state, set_vel=True, zero_acc=True)

print(env.get_body_state(env.active_id))

while True:
    pass
