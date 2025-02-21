"""Roll out a dynamics model with GUI."""

import time
from typing import Callable

import numpy as np
import torch

from limb_repo.dynamics.models.learned_dynamics import (
    LearnedDynamics,
    NeuralNetworkConfig,
)

# pylint: disable=unused-import
from limb_repo.dynamics.models.math_dynamics import MathDynamics

# pylint: disable=unused-import
from limb_repo.dynamics.models.math_dynamics_with_n_vector import (
    MathDynamicsWithNVector,
)
from limb_repo.environments.limb_repo_pybullet_env import (
    LimbRepoPyBulletConfig,
    LimbRepoPyBulletEnv,
)
from limb_repo.utils import utils

parsed_config = utils.parse_config(
    "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
)
parsed_config.pybullet_config.use_gui = False

nn_config = utils.parse_config(
    "assets/configs/nn_configs/30-1024-2048-2048-1024-12.yaml", NeuralNetworkConfig
)
max_features2 = torch.tensor(
    [
        0.9999996423721312,
        1,
        0.999999701976776,
        0.9999998807907104,
        0.9999998211860656,
        0.9999995231628418,
        2.8973000049591064,
        1.7625828981399536,
        -0.06980051100254059,
        2.8972997665405273,
        3.752498626708984,
        2.8972997665405273,
        0.999999701976776,
        0.9999998211860656,
        0.9999999403953552,
        0.9999998807907104,
        0.9999998807907104,
        0.9999995231628418,
        3.141592502593994,
        3.141592502593994,
        3.1415915489196777,
        3.1344685554504395,
        3.141592502593994,
        3.141590118408203,
        204.5392608642578,
        22.441301345825195,
        60.01301193237305,
        7.351491928100586,
        15.775938987731934,
        17.91392707824707,
    ]
)

min_features2 = torch.tensor(
    [
        -0.999999701976776,
        -0.9999992847442628,
        -0.9999995231628418,
        -0.9999994039535522,
        -0.9999999403953552,
        -0.9999997615814208,
        -2.897299528121948,
        -1.7627991437911987,
        -3.071798324584961,
        -2.8972997665405273,
        -0.017499864101409912,
        -2.897298574447632,
        -0.9999998211860656,
        -0.9999999403953552,
        -0.9999999403953552,
        -0.9999999403953552,
        -0.9999998807907104,
        -0.9999997615814208,
        -3.1415908336639404,
        -3.141592502593994,
        -3.141592264175415,
        -3.1203267574310303,
        -3.1415905952453613,
        -3.141591787338257,
        -60.76736068725586,
        -16.733293533325195,
        -205.0948944091797,
        -7.320549011230469,
        -23.103116989135746,
        -15.989590644836426,
    ]
)


def normalize_fn_lin(
    min_values: torch.Tensor, max_values: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function that normalizes input data between -1 and 1."""
    range_value = max_values - min_values
    print("range value", range_value)

    def _normalize_fn_lin(x: torch.Tensor) -> torch.Tensor:
        return 2 * (x - min_values) / range_value - 1

    return _normalize_fn_lin


def denormalize_fn_tanh(scaling: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function that denormalizes input data using tanh."""

    def _denormalize_fn_tanh(x: torch.Tensor) -> torch.Tensor:
        x[x > 1] = 0.9999
        x[x < -1] = -0.9999
        return scaling * torch.arctanh(x)

    return _denormalize_fn_tanh


dynamics = LearnedDynamics(
    parsed_config,
    nn_config,
    "_weights/1024-2048-3std_2025-01-16_23-23-22/model_weights_499.pth",
    normalize_fn_lin(min_features2, max_features2),
    denormalize_fn_tanh(8),
)

parsed_config.pybullet_config.use_gui = True
env = LimbRepoPyBulletEnv(parsed_config)

for i in range(350):
    env.step()
    time.sleep(0.002)
    print(i)

input("manual control")

# test torque control
for i in range(5000):
    # action = np.array([0, 0.33, 0, 0.5, -0.5, 0.75])
    action = np.random.random(6) * 2 - 1
    next_state = dynamics.step(action)
    print("robot velocity", next_state[6:12])
    print("human velocity", next_state[18:24])
    env.set_limb_repo_state(next_state)
    # time.sleep(parsed_config.pybullet_config.dt)
    print(i)
