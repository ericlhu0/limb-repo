"""Testing Limb Repo Pybullet Env."""

import numpy as np

from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletEnv
from limb_repo.structs import BodyState
from limb_repo.utils import utils

parsed_config = utils.parse_config(
    "assets/configs/test_env_config.yaml", LimbRepoPyBulletEnv
)

env = LimbRepoPyBulletEnv(config=parsed_config)

active_state = BodyState(np.concatenate([parsed_config.active_q, np.zeros(6 + 6)]))
passive_state = BodyState(np.concatenate([parsed_config.passive_q, np.zeros(6 + 6)]))

env.set_body_state(env.active_id, active_state)
env.set_body_state(env.passive_id, passive_state)

input()
