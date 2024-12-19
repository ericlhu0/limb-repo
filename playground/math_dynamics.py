"""Testing math dynamics models."""

import numpy as np

from limb_repo.dynamics.checks.check_ee_kinematics import check_ee_kinematics
from limb_repo.dynamics.models.math_dynamics_no_n_vector import MathDynamicsNoNVector
from limb_repo.environments.lr_pybullet_env import LRPyBulletEnv

parsed_config = LRPyBulletEnv.parse_config("assets/configs/test_env_config.yaml")
# parsed_config.pybullet_config.use_gui = False
dynamics_model = MathDynamicsNoNVector(parsed_config)
# dynamics_model: BaseDynamics = MathDynamicsWithNVector(parsed_config)


for i in range(1000000):
    action = np.random.rand(6)
    print(f"loop {i}")
    next_state = dynamics_model.step(action)
    lr_ee_state = dynamics_model.env.get_lr_ee_state()
    print(
        check_ee_kinematics(
            **vars(lr_ee_state),
            active_ee_to_passive_ee=dynamics_model.env.active_ee_to_passive_ee,
        )
    )
    # time.sleep(0.05)
