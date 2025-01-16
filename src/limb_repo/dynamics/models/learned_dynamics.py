"""Dynamics using a neural network."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import omegaconf
import torch
from torch import nn

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.structs import Action, JointState, LimbRepoEEState, LimbRepoState
from limb_repo.utils import utils


@dataclass
class NeuralNetworkConfig:
    input_size: int
    hidden_layers: list[int]
    output_size: int
    activation: str


class PyTorchLearnedDynamicsModel(nn.Module):
    def __init__(self, nn_config_path: str):
        super().__init__()

        self.nn_config = utils.parse_config(nn_config_path, NeuralNetworkConfig)

        # inputs: active torque, q sin & cos, q; passive q sin & cos, qd

        layers = []
        prev_size = self.nn_config.input_size

        for hidden_layer in self.nn_config.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_layer))
            layers.append(getattr(nn, self.nn_config.activation)())
            prev_size = hidden_layer

        layers.append(nn.Linear(prev_size, self.nn_config.output_size))

        self.layer_stack = nn.Sequential(*layers)

    def forward(self, inp):
        return self.layer_stack(inp)


class LearnedDynamics(BaseDynamics):
    """Dynamics using a neural network."""

    def __init__(
        self,
        env_config: omegaconf.DictConfig,
        nn_config: omegaconf.DictConfig,
        weights_path: str,
        normalize_features_fn: Callable[[torch.Tensor], torch.Tensor],
        denormalize_labels_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int = 1,
    ) -> None:
        super().__init__(env_config)

        self.batch_size = batch_size

        self.model = PyTorchLearnedDynamicsModel(nn_config)
        self.model = nn.DataParallel(self.model)

        self.normalize_features_fn = normalize_features_fn
        self.denormalize_labels_fn = denormalize_labels_fn

        self.weights_path = weights_path

        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(self.weights_path))
        else:
            self.model.load_state_dict(
                torch.load(self.weights_path, map_location="cpu")
            )

        self.model.eval()

        self.dt = self.config.pybullet_config.dt

        self.q_a = torch.tensor(
            np.repeat(env_config.active_q, batch_size), requires_grad=False
        )
        self.qd_a = torch.zeros_like(self.q_a, requires_grad=False)
        self.q_p = torch.tensor(
            np.repeat(env_config.passive_q, batch_size), requires_grad=False
        )
        self.qd_p = torch.zeros_like(self.q_p, requires_grad=False)

    def step(self, torques: Action) -> torch.Tensor:
        """Step the dynamics model."""

        # >>> a = np.concatenate([[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12], [13, 14], [15, 16]], [[17, 18], [19, 20], [21, 22], [23, 24]]], axis=-1)
        # >>> a
        # array([[ 1,  2,  9, 10, 17, 18],
        #     [ 3,  4, 11, 12, 19, 20],
        #     [ 5,  6, 13, 14, 21, 22],
        #     [ 7,  8, 15, 16, 23, 24]])

        torques = torch.tensor(torques, requires_grad=False).float()
        input_feature = self.normalize_features_fn(
            torch.concatenate(
                [
                    torques,
                    self.q_a,
                    self.qd_a,
                    self.q_p,
                    self.qd_p,
                ],
                axis=-1,
            ),
        ).float()

        print("input_feature", input_feature)

        qdd = self.model(input_feature)
        print("qdd", qdd)
        qdd = self.denormalize_labels_fn(qdd)
        print("denormalized qdd", qdd)

        qdd_a = qdd[: self.active_n_dofs]
        qdd_p = qdd[self.active_n_dofs :]

        self.qd_a += qdd_a * self.dt
        self.q_a += self.qd_a * self.dt
        self.qd_p += qdd_p * self.dt
        self.q_p += self.qd_p * self.dt

        return LimbRepoState(
            torch.concatenate([self.q_a, self.qd_a, self.q_p, self.qd_p])
            .detach()
            .numpy()
        )

    def get_state(self) -> LimbRepoState:
        """Get the state of the internal environment."""
        return LimbRepoState(
            torch.concatenate([self.q_a, self.qd_a, self.q_p, self.qd_p]).numpy()
        )

    def get_ee_state(self) -> LimbRepoEEState:
        """Get the state of the end effector."""
        raise NotImplementedError("not implemented for neural network")

    def set_state(self, state: LimbRepoState) -> None:
        """Set the state from which to step the dynamics model from."""
        self.q_a = state.active_q
        self.qd_a = state.active_qd
        self.q_p = state.passive_q
        self.qd_p = state.passive_qd
