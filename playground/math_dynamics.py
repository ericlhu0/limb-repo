import numpy as np

from limb_repo.dynamics.math_dynamics_with_n_vector import MathDynamicsWithNVector
from limb_repo.environments.lr_pybullet_env import LRPyBulletEnv

parsed_config = LRPyBulletEnv.parse_config("assets/configs/test_env_config.yaml")
dynamics_model = MathDynamicsWithNVector(parsed_config)

while True:
    next_state = dynamics_model.step(np.array([0, 0, 0, 0, 0, 0.1]))
    dynamics_model.env.set_lr_state(next_state)
