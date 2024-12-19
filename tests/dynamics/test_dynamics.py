"""Test dynamics formulation without n vector."""

import numpy as np

from limb_repo.dynamics.checks.check_ee_kinematics import check_ee_kinematics
from limb_repo.dynamics.models.math_dynamics_no_n_vector import MathDynamicsNoNVector
from limb_repo.environments.lr_pybullet_env import LRPyBulletEnv


def test_dynamics_no_n_vector():
    """Test dynamics formulation without n vector."""
    parsed_config = LRPyBulletEnv.parse_config("assets/configs/test_env_config.yaml")
    parsed_config.pybullet_config.use_gui = False
    dynamics_model = MathDynamicsNoNVector(parsed_config)

    for i in range(500):
        action = np.array([1, 0, 1, 0, 1, 0])
        print(f"loop {i}")
        dynamics_model.step(action)
        lr_ee_state = dynamics_model.env.get_lr_ee_state()
        assert check_ee_kinematics(
            lr_ee_state,
            dynamics_model.env.active_ee_to_passive_ee,
        )
