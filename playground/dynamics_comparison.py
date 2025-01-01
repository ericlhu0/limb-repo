"""Testing math dynamics models."""

import matplotlib.pyplot as plt
import numpy as np

from limb_repo.dynamics.models.math_dynamics_no_n_vector import MathDynamicsNoNVector
from limb_repo.dynamics.models.math_dynamics_with_n_vector import (
    MathDynamicsWithNVector,
)
from limb_repo.dynamics.models.pybullet_dynamics import PyBulletDynamics
from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletConfig
from limb_repo.utils import utils

np.random.seed(0)

parsed_config = utils.parse_config(
    "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
)
parsed_config.pybullet_config.use_gui = False
dynamics_model_nvec = MathDynamicsWithNVector(parsed_config)
dynamics_model = MathDynamicsNoNVector(parsed_config)
pybullet_model = PyBulletDynamics(parsed_config)

time_steps = []
math_dynamics = []
math_dynamics_nvec = []
pybullet_dynamics = []

for i in range(1000):
    action = np.random.rand(6) * 0.5
    print(f"loop {i}")
    next_state = dynamics_model.step(action)
    next_state_nvec = dynamics_model_nvec.step(action)
    pybullet_next_state = pybullet_model.step(action)

    math_dynamics.append(next_state.active_q)
    math_dynamics_nvec.append(next_state_nvec.active_q)
    pybullet_dynamics.append(pybullet_next_state.active_q)
    time_steps.append(i)

plt.plot(time_steps, np.array(math_dynamics)[:], label="math_dynamics")
plt.plot(time_steps, np.array(math_dynamics_nvec)[:], label="math_dynamics_nvec")
plt.plot(time_steps, np.array(pybullet_dynamics)[:], label="pybullet_dynamics")
plt.xlabel("Time step")
plt.ylabel("robot second joint position")
plt.legend()
plt.show()

input()

# for i in range(1000000):
#     action = np.random.rand(6)
#     next_state = dynamics_model.step(action)
#     pybullet_next_state = pybullet_model.step(action)
#     assert np.allclose(next_state.active_q, pybullet_next_state.active_q, atol=1e-2)
#     limb_repo_ee_state = dynamics_model.env.get_limb_repo_ee_state()
#     assert check_ee_kinematics(
#         limb_repo_ee_state,
#         active_ee_to_passive_ee=dynamics_model.env.active_ee_to_passive_ee,
#     )
#     # time.sleep(0.05)
