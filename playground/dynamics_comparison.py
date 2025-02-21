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
from limb_repo.dynamics.models.pybullet_dynamics import PyBulletDynamics
from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletConfig
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

max_features3 = torch.tensor(
    [
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        2.8973,
        1.7626,
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
        3.1217,
        3.1416,
        3.1416,
        4.5628,
        3.0144,
        3.6326,
        4.8081,
        4.0170,
        3.7522,
    ]
)

min_features3 = torch.tensor(
    [
        -1.0000,
        -1.0000,
        -1.0000,
        -1.0000,
        -1.0000,
        -1.0000,
        -2.8973,
        -1.7628,
        -3.0718,
        -2.8973,
        -0.0175,
        -2.8973,
        -1.0000,
        -1.0000,
        -1.0000,
        -1.0000,
        -1.0000,
        -1.0000,
        -3.1416,
        -3.1416,
        -3.1416,
        -3.1142,
        -3.1416,
        -3.1416,
        -4.5628,
        -3.0141,
        -3.6323,
        -4.8089,
        -4.0172,
        -3.7543,
    ]
)

parsed_config = utils.parse_config(
    "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
)
parsed_config.pybullet_config.use_gui = False


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
n_lin_in2 = normalize_fn_lin(min_features2, max_features2)
n_lin_in3 = normalize_fn_lin(min_features3, max_features3)
dn_lin_out = denormalize_fn_lin(min_labels, max_labels)
dn_tanh = denormalize_fn_tanh(8)

nn_config_64_128_64 = utils.parse_config(
    "assets/configs/nn_configs/30-64-128-64-12.yaml", NeuralNetworkConfig
)
nn_config_5123 = utils.parse_config(
    "assets/configs/nn_configs/30-512-512-512-12.yaml", NeuralNetworkConfig
)
nn_config_5124 = utils.parse_config(
    "assets/configs/nn_configs/30-512-512-512-512-12.yaml", NeuralNetworkConfig
)
nn_config_1024_2048 = utils.parse_config(
    "assets/configs/nn_configs/30-1024-2048-2048-1024-12.yaml", NeuralNetworkConfig
)

# pylint: disable=line-too-long
models: List[BaseDynamics] = [
    MathDynamics(parsed_config),
    # MathDynamicsWithNVector(parsed_config),
    PyBulletDynamics(parsed_config),
    # LearnedDynamics(parsed_config, nn_config_64_128_64, "weights-10-epochs.pth", n_lin_in, dn_lin_out),
    # LearnedDynamics(parsed_config, nn_config_64_128_64, "weights-30-epochs.pth", n_lin_in, dn_lin_out),
    # LearnedDynamics(parsed_config, nn_config_64_128_64, "weights-90-epochs.pth", n_lin_in, dn_lin_out),
    # LearnedDynamics(parsed_config, nn_config_64_128_64, "weights-310-epochs.pth", n_lin_in, dn_lin_out),
    # LearnedDynamics(parsed_config, nn_config_64_128_64, "weights-500-epochs.pth", n_lin_in, dn_lin_out),
    # LearnedDynamics(parsed_config, nn_config_64_128_64, "weights-tanh-30.pth", n_lin_in, dn_tanh),
    # LearnedDynamics(parsed_config, nn_config_64_128_64, "weights-tanh-40.pth", n_lin_in, dn_tanh),
    # LearnedDynamics(parsed_config, nn_config_64_128_64, "weights-tanh-50.pth", n_lin_in, dn_tanh),
    # LearnedDynamics(parsed_config, nn_config_64_128_64, "weights-tanh-60.pth", n_lin_in, dn_tanh),
    # LearnedDynamics(parsed_config, nn_config_64_128_64, "weights-tanh-240.pth", n_lin_in, dn_tanh),
    # LearnedDynamics(parsed_config, nn_config_5123, "_weights/10M_fullrun_2025-01-16_03-24-03/10M_fullrun_2025-01-16_03-24-03model_weights_499.pth", n_lin_in2, dn_tanh),
    # LearnedDynamics(parsed_config, nn_config_5124, "_weights/10M_512^4_fullrun_2025-01-16_12-13-56/model_weights_179.pth", n_lin_in2, dn_tanh),
    # LearnedDynamics(parsed_config, nn_config_5124, "_weights/10M_512^4_fullrun_2025-01-16_12-13-56/model_weights_269.pth", n_lin_in2, dn_tanh),
    LearnedDynamics(
        parsed_config,
        nn_config_1024_2048,
        "_weights/1024-2048-3std_2025-01-16_23-23-22/model_weights_499.pth",
        n_lin_in3,
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
            color=color,
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
            color=color,
        )
        color = "red"

    plt.xlabel("Time step")
    label = f"{'position' if i < 6 else 'velocity'} human {i % 6}th joint"
    plt.ylabel(label)
    plt.title(label)
    plt.legend()
    plt.savefig(f"_figs/{label}.png")
    plt.close()
