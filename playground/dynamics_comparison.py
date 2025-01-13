"""Testing math dynamics models."""

import time
from typing import List, Callable

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

np.random.seed(0)

max_features = np.array(
    [
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        2.8973,
        1.7606,
        -0.0698,
        2.8973,
        3.7525,
        2.8973,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        3.1416,
        3.1416,
        3.1416,
        3.1082,
        3.1416,
        3.1416,
        60.8486,
        17.8876,
        34.9453,
        6.5389,
        15.5955,
        12.0659,
    ]
)
min_features = np.array(
    [
        -1.0000e00,
        -1.0000e00,
        -1.0000e00,
        -1.0000e00,
        -1.0000e00,
        -1.0000e00,
        -2.8973e00,
        -1.7628e00,
        -3.0718e00,
        -2.8973e00,
        -1.7498e-02,
        -2.8973e00,
        -1.0000e00,
        -1.0000e00,
        -1.0000e00,
        -1.0000e00,
        -1.0000e00,
        -1.0000e00,
        -3.1416e00,
        -3.1416e00,
        -3.1416e00,
        -3.1162e00,
        -3.1416e00,
        -3.1416e00,
        -7.8739e01,
        -1.5045e01,
        -7.8826e01,
        -6.6380e00,
        -8.5202e00,
        -2.0322e01,
    ]
)
max_labels = np.array(
    [
        6.2598e00,
        1.0201e01,
        2.7169e01,
        2.4286e01,
        2.6831e01,
        4.8063e01,
        1.6045e04,
        6.3364e01,
        1.6037e04,
        2.7116e01,
        5.0295e01,
        4.8872e01,
    ]
)
min_labels = np.array(
    [
        -5.6922,
        -6.5697,
        -18.7674,
        -12.8434,
        -19.8408,
        -35.6991,
        -635.5103,
        -53.9756,
        -636.4579,
        -31.9322,
        -51.1465,
        -58.2604,
    ]
)

parsed_config = utils.parse_config(
    "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
)
parsed_config.pybullet_config.use_gui = False

def denormalize_fn_tanh(x: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    return torch.arctanh

def denormalize_fn_lin(min_values: torch.Tensor, max_values: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    range = max_values - min_values

    def _denormalize_fn_lin(x: torch.Tensor) -> torch.Tensor:
        return ((x + 1) / 2 * range) + min_values

    return _denormalize_fn_lin
    

dn_lin_in = denormalize_fn_lin(min_features, max_features)
dn_lin_out = denormalize_fn_lin(min_labels, max_labels)
dn_tanh = denormalize_fn_tanh()

models: List[BaseDynamics] = [
    MathDynamicsNoNVector(parsed_config),
    MathDynamicsWithNVector(parsed_config),
    PyBulletDynamics(parsed_config),
    LearnedDynamics(parsed_config, "weights-10-epochs.pth", dn_lin_in, dn_lin_out),
    # LearnedDynamics(parsed_config, "weights-30-epochs.pth"),
    LearnedDynamics(parsed_config, "weights-90-epochs.pth", dn_lin_in, dn_lin_out),
    # LearnedDynamics(parsed_config, "weights-310-epochs.pth"),
    LearnedDynamics(parsed_config, "weights-500-epochs.pth", dn_lin_in, dn_lin_out),
    LearnedDynamics(parsed_config, "weights-tanh-30.pth", dn_tanh, dn_tanh),
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
        plt.plot(
            time_steps, np.array(tracked_robot_states[model])[:, i], label=f"{model}"
        )

    plt.xlabel("Time step")
    plt.ylabel(f"robot {i}th joint position")
    plt.title(f"robot {i}th joint position")
    plt.legend()
    # plt.show()
    plt.savefig(f"_figs/robot_{i}_joint_position.png")

for i in range(12):
    for model in models:
        plt.plot(
            time_steps, np.array(tracked_human_states[model])[:, i], label=f"{model}"
        )

    plt.xlabel("Time step")
    plt.ylabel(f"human {i}th joint position")
    plt.legend()
    # plt.show()
    plt.savefig(f"_figs/human_{i}_joint_position.png")
