"""Testing math dynamics models."""

import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.dynamics.models.learned_dynamics import LearnedDynamics
from limb_repo.dynamics.models.math_dynamics_no_n_vector import MathDynamicsNoNVector
from limb_repo.dynamics.models.math_dynamics_with_n_vector import (
    MathDynamicsWithNVector,
)
from limb_repo.dynamics.models.pybullet_dynamics import PyBulletDynamics
from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletConfig
from limb_repo.structs import LimbRepoState
from limb_repo.utils import utils

# np.random.seed(0)

parsed_config = utils.parse_config(
    "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
)
parsed_config.pybullet_config.use_gui = False
dynamics_model_nvec = MathDynamicsWithNVector(parsed_config)
dynamics_model = MathDynamicsNoNVector(parsed_config)
pybullet_model = PyBulletDynamics(parsed_config)
learned_dynamics_model_10e = LearnedDynamics(parsed_config, "weights-10-epochs.pth")
learned_dynamics_model_30e = LearnedDynamics(parsed_config, "weights-30-epochs.pth")
learned_dynamics_model_90e = LearnedDynamics(parsed_config, "weights-90-epochs.pth")
learned_dynamics_model_310e = LearnedDynamics(parsed_config, "weights-310-epochs.pth")
learned_dynamics_model_500e = LearnedDynamics(parsed_config, "weights-500-epochs.pth")

models: List[BaseDynamics] = [
    dynamics_model,
    dynamics_model_nvec,
    pybullet_model,
    learned_dynamics_model_10e,
    learned_dynamics_model_30e,
    learned_dynamics_model_90e,
    learned_dynamics_model_310e,
    learned_dynamics_model_500e,
]

tracked_robot_states = {}
tracked_human_states = {}
for model in models:
    tracked_robot_states[model] = []
    tracked_human_states[model] = []
time_steps = []

for i in range(500):
    action = np.random.rand(6) * 2 - 1
    print(f"loop {i}")
    for model in models:
        t1 = time.time()
        next_state = model.step(action)
        t2 = time.time()
        print(f"{model} took {t2-t1} seconds")
        print(model)
        print(next_state)
        tracked_robot_states[model].append(next_state.active)
        tracked_human_states[model].append(next_state.passive)

    time_steps.append(i)

for i in range(12):
    for model in models:
        plt.plot(time_steps, np.array(tracked_robot_states[model])[:, i], label=f"{model}")

    plt.xlabel("Time step")
    plt.ylabel(f"robot {i}th joint position")
    plt.legend()
    plt.show()

for i in range(12):
    for model in models:
        plt.plot(time_steps, np.array(tracked_human_states[model])[:, i], label=f"{model}")

    plt.xlabel("Time step")
    plt.ylabel(f"human {i}th joint position")
    plt.legend()
    plt.show()

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
