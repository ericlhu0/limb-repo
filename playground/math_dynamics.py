"""Testing math dynamics models."""

import numpy as np

from limb_repo.dynamics.checks.check_ee_kinematics import check_ee_kinematics
from limb_repo.dynamics.models.math_dynamics_no_n_vector import MathDynamicsNoNVector
from limb_repo.dynamics.models.pybullet_dynamics import PyBulletDynamics
from limb_repo.environments.lr_pybullet_env import LRPyBulletEnv

parsed_config = LRPyBulletEnv.parse_config("assets/configs/test_env_config.yaml")
parsed_config.pybullet_config.use_gui = False
dynamics_model = MathDynamicsNoNVector(parsed_config)
# dynamics_model: BaseDynamics = MathDynamicsWithNVector(parsed_config)
pybullet_model = PyBulletDynamics(parsed_config)


for i in range(1000000):
    action = np.random.rand(6)
    print(f"loop {i}")
    next_state = dynamics_model.step(action)
    pybullet_next_state = pybullet_model.step(action)
    print('next_state', next_state.active_q, next_state.passive_q)
    print('pybullet_next_state', pybullet_next_state.active_q, pybullet_next_state.passive_q)
    print('diff', next_state.active_q - pybullet_next_state.active_q, next_state.passive_q - pybullet_next_state.passive_q)
    assert np.allclose(next_state.active_q, pybullet_next_state.active_q, atol=1e-2)
    lr_ee_state = dynamics_model.env.get_lr_ee_state()
    assert check_ee_kinematics(
        lr_ee_state,
        active_ee_to_passive_ee=dynamics_model.env.active_ee_to_passive_ee,
    )
    # time.sleep(0.05)
