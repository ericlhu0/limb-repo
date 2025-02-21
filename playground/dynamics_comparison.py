"""Testing math dynamics models."""

import time
from typing import Any, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.dynamics.models.learned_dynamics import (
    LearnedDynamics,
    NeuralNetworkConfig,
)
from limb_repo.dynamics.models.math_dynamics import MathDynamics
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

max_features = torch.load("_weights/1024-2048-3std_2025-01-16_23-23-22/max_features.pt")
min_features = torch.load("_weights/1024-2048-3std_2025-01-16_23-23-22/min_features.pt")

print("max features", max_features)
print("min features", min_features)
input()


def normalize_fn_lin(
    min_values: torch.Tensor, max_values: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a linear normalization function."""
    range_value = max_values - min_values
    print("range value", range_value)

    def _normalize_fn_lin(x: torch.Tensor) -> torch.Tensor:
        return 2 * (x - min_values) / range_value - 1

    return _normalize_fn_lin


def denormalize_fn_lin(
    min_values: torch.Tensor, max_values: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a linear denormalization function."""
    range_value = max_values - min_values

    def _denormalize_fn_lin(x: torch.Tensor) -> torch.Tensor:
        return ((x + 1) / 2 * range_value) + min_values

    return _denormalize_fn_lin


def denormalize_fn_tanh(scaling: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a tanh denormalization function."""

    def _denormalize_fn_tanh(x: torch.Tensor) -> torch.Tensor:
        x[x > 1] = 0.9999
        x[x < -1] = -0.9999
        return scaling * torch.arctanh(x)

    return _denormalize_fn_tanh


n_lin_in = normalize_fn_lin(min_features, max_features)
dn_tanh = denormalize_fn_tanh(8)

nn_config_1024_2048 = utils.parse_config(
    "assets/configs/nn_configs/30-1024-2048-2048-1024-12.yaml", NeuralNetworkConfig
)

models: List[BaseDynamics] = [
    MathDynamics(parsed_config),
    MathDynamicsWithNVector(parsed_config),
    PyBulletDynamics(parsed_config),
    LearnedDynamics(
        parsed_config,
        nn_config_1024_2048,
        "_weights/1024-2048-3std_2025-01-16_23-23-22/model_weights_499.pth",
        n_lin_in,
        dn_tanh,
    ),
]

tracked_robot_states: dict[BaseDynamics, Any] = {}
tracked_human_states: dict[BaseDynamics, Any] = {}
for model in models:
    tracked_robot_states[model] = []
    tracked_human_states[model] = []
time_steps = []

for i in range(500):
    action = np.random.rand(6) * 2 - 1
    # action = np.zeros(6)
    print(f"loop {i}")
    for model in models:
        t1 = time.time()
        next_state = model.step(action)
        t2 = time.time()
        print(f"{model} took {t2-t1} seconds")
        print("model", model)
        print("next state", next_state)
        tracked_robot_states[model].append(next_state.active)
        tracked_human_states[model].append(next_state.passive)

    time_steps.append(i)

for i in range(12):
    color = "green"
    for model in models:
        plt.plot(
            time_steps,
            np.array(tracked_robot_states[model])[:, i],
            label=f"{model}"[20:40],
            # color=color,
        )
        color = "red"

    plt.xlabel("Time step")
    label = f"{'position' if i < 6 else 'velocity'} robot {i % 6}th joint"
    plt.ylabel(label)
    plt.title(label)
    plt.legend()
    plt.savefig(f"_figs/{label}.png")
    plt.close()

for i in range(12):
    color = "green"
    for model in models:
        plt.plot(
            time_steps,
            np.array(tracked_human_states[model])[:, i],
            label=f"{model}"[20:40],
            # color=color,
        )
        color = "red"

    plt.xlabel("Time step")
    label = f"{'position' if i < 6 else 'velocity'} human {i % 6}th joint"
    plt.ylabel(label)
    plt.title(label)
    plt.legend()
    plt.savefig(f"_figs/{label}.png")
    plt.close()
