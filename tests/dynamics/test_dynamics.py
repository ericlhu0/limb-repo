"""Test dynamics formulation without n vector."""

import numpy as np

from limb_repo.dynamics.checks.check_ee_kinematics import check_ee_kinematics
from limb_repo.dynamics.models.math_dynamics import MathDynamics
from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletConfig
from limb_repo.utils import utils


def test_dynamics_no_n_vector():
    """Test dynamics formulation without n vector."""
    parsed_config = utils.parse_config(
        "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
    )
    parsed_config.pybullet_config.use_gui = False
    dynamics_model = MathDynamics(parsed_config)

    for _ in range(200):
        action = np.array([1, 0, 1, 0, 1, 0])
        dynamics_model.step(action)
        limb_repo_ee_state = dynamics_model.env.get_limb_repo_ee_state()
        assert check_ee_kinematics(
            limb_repo_ee_state,
            dynamics_model.env.active_ee_to_passive_ee,
        )
