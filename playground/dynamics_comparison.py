"""Testing math dynamics models."""

import time
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.dynamics.models.learned_dynamics import LearnedDynamics
from limb_repo.dynamics.models.math_dynamics import MathDynamics
from limb_repo.dynamics.models.math_dynamics_with_n_vector import (
    MathDynamicsWithNVector,
)
from limb_repo.dynamics.models.pybullet_dynamics import PyBulletDynamics
from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletConfig
from limb_repo.structs import LimbRepoState
from limb_repo.utils import utils

np.random.seed(0)

max_features = torch.tensor(
    [
        0.9999991655349731,
        0.9999998807907104,
        1,
        0.9999993443489076,
        0.9999991059303284,
        0.9999995827674866,
        2.8972959518432617,
        1.7606414556503296,
        -0.06980126351118088,
        2.8972980976104736,
        3.75249981880188,
        2.897298812866211,
        0.9999997615814208,
        1,
        0.999999463558197,
        0.9999999403953552,
        0.999999463558197,
        0.999999463558197,
        3.1415822505950928,
        3.141591787338257,
        3.141591310501098,
        3.108182191848755,
        3.141591787338257,
        3.1415791511535645,
        60.8486442565918,
        17.887603759765625,
        34.94528579711914,
        6.538883209228516,
        15.595539093017578,
        12.065893173217772,
    ]
)
min_features = torch.tensor(
    [
        -0.9999997615814208,
        -0.9999989867210388,
        -0.9999991059303284,
        -0.9999996423721312,
        -0.999997854232788,
        -0.9999972581863404,
        -2.897294521331787,
        -1.7627956867218018,
        -3.0717897415161133,
        -2.8972997665405273,
        -0.017497992143034935,
        -2.897289991378784,
        -0.9999989867210388,
        -0.9999977946281432,
        -0.9999982118606568,
        -0.9999988675117492,
        -0.9999999403953552,
        -0.999997854232788,
        -3.141592502593994,
        -3.141591310501098,
        -3.141589403152466,
        -3.1162242889404297,
        -3.141591310501098,
        -3.141590118408203,
        -78.73896026611328,
        -15.044720649719238,
        -78.82576751708984,
        -6.637977600097656,
        -8.520156860351562,
        -20.321762084960938,
    ]
)
max_labels = torch.tensor(
    [
        6.259791851043701,
        10.20071029663086,
        27.16858673095703,
        24.285869598388672,
        26.831071853637695,
        48.063228607177734,
        16045.322265625,
        63.36368942260742,
        16036.8857421875,
        27.11632919311523,
        50.295021057128906,
        48.87164688110352,
    ]
)
min_labels = torch.tensor(
    [
        -5.6922478675842285,
        -6.569695472717285,
        -18.767412185668945,
        -12.843420028686523,
        -19.84084129333496,
        -35.69911575317383,
        -635.5103149414062,
        -53.975643157958984,
        -636.4578857421875,
        -31.93218994140625,
        -51.14649963378906,
        -58.26044464111328,
    ]
)

parsed_config = utils.parse_config(
    "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
)
parsed_config.pybullet_config.use_gui = False


def normalize_fn_lin(
    min_values: torch.Tensor, max_values: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    range_value = max_values - min_values
    print("range value", range_value)

    def _normalize_fn_lin(x: torch.Tensor) -> torch.Tensor:
        return 2 * (x - min_values) / range_value - 1

    return _normalize_fn_lin


def denormalize_fn_lin(
    min_values: torch.Tensor, max_values: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    range_value = max_values - min_values

    def _denormalize_fn_lin(x: torch.Tensor) -> torch.Tensor:
        return ((x + 1) / 2 * range_value) + min_values

    return _denormalize_fn_lin


def denormalize_fn_tanh(scaling: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def _denormalize_fn_tanh(x: torch.Tensor) -> torch.Tensor:
        return scaling * torch.arctanh(x)

    return _denormalize_fn_tanh


n_lin_in = normalize_fn_lin(min_features, max_features)
dn_lin_out = denormalize_fn_lin(min_labels, max_labels)
dn_tanh = denormalize_fn_tanh(8)

models: List[BaseDynamics] = [
    MathDynamics(parsed_config),
    MathDynamicsWithNVector(parsed_config),
    PyBulletDynamics(parsed_config),
    LearnedDynamics(parsed_config, "weights-10-epochs.pth", n_lin_in, dn_lin_out),
    # # LearnedDynamics(parsed_config, "weights-30-epochs.pth"),
    LearnedDynamics(parsed_config, "weights-90-epochs.pth", n_lin_in, dn_lin_out),
    # # LearnedDynamics(parsed_config, "weights-310-epochs.pth"),
    LearnedDynamics(parsed_config, "weights-500-epochs.pth", n_lin_in, dn_lin_out),
    LearnedDynamics(parsed_config, "weights-tanh-30.pth", n_lin_in, dn_tanh),
    # LearnedDynamics(parsed_config, "weights-tanh-40.pth", n_lin_in, dn_tanh),
    # LearnedDynamics(parsed_config, "weights-tanh-50.pth", n_lin_in, dn_tanh),
    # LearnedDynamics(parsed_config, "weights-tanh-60.pth", n_lin_in, dn_tanh),
    LearnedDynamics(parsed_config, "weights-tanh-240.pth", n_lin_in, dn_tanh),
]

tracked_robot_states = {}
tracked_human_states = {}
for model in models:
    tracked_robot_states[model] = []
    tracked_human_states[model] = []
time_steps = []

for i in range(500):
    action = np.random.rand(6) * 0
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
    for model in models:
        plt.plot(
            time_steps,
            np.array(tracked_robot_states[model])[:, i],
            label=f"{model}"[20:40],
        )

    plt.xlabel("Time step")
    plt.ylabel(f"robot {i}th joint position")
    plt.title(f"robot {i}th joint position")
    plt.legend()
    plt.savefig(f"_figs/robot_{i}_joint_position.png")
    plt.close()

for i in range(12):
    for model in models:
        plt.plot(
            time_steps,
            np.array(tracked_human_states[model])[:, i],
            label=f"{model}"[20:40],
        )

    plt.xlabel("Time step")
    plt.ylabel(f"human {i}th joint position")
    plt.legend()
    plt.savefig(f"_figs/human_{i}_joint_position.png")
    plt.close()
