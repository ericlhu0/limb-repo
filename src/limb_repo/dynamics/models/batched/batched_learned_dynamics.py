"""Dynamics using a neural network."""

from dataclasses import dataclass
from typing import Callable

import omegaconf
import torch
from torch import nn

from limb_repo.dynamics.models.batched.base_batched_dynamics import BaseBatchedDynamics


@dataclass
class NeuralNetworkConfig:
    """Configuration for a neural network."""
    input_size: int
    hidden_layers: list[int]
    output_size: int
    activation: str


class BatchedLearnedDynamicsModel(nn.Module):
    """Batched neural network model for learned dynamics."""
    def __init__(self, nn_config: omegaconf.DictConfig) -> None:
        super().__init__()

        self.nn_config = nn_config

        # inputs: active torque, q sin & cos, qd; passive q sin & cos, qd

        layers = []
        prev_size = self.nn_config.input_size

        for hidden_layer in self.nn_config.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_layer))
            layers.append(getattr(nn, self.nn_config.activation)())
            prev_size = hidden_layer

        layers.append(nn.Linear(prev_size, self.nn_config.output_size))

        self.layer_stack = nn.Sequential(*layers)

    def forward(self, inp):
        """Forward pass."""
        return self.layer_stack(inp)


class BatchedLearnedDynamics(BaseBatchedDynamics):
    """Dynamics using a neural network."""

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        env_config: omegaconf.DictConfig,
        nn_config: omegaconf.DictConfig,
        weights_path: str,
        normalize_features_fn: Callable[[torch.Tensor], torch.Tensor],
        denormalize_labels_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.config = env_config
        self.active_n_dofs = len(env_config.active_q)
        self.passive_n_dofs = len(env_config.passive_q)

        self.batch_size = batch_size

        self.model = BatchedLearnedDynamicsModel(nn_config)
        # self.model = nn.DataParallel(self.model)

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

        self.q_a_batch = torch.tile(
            torch.tensor(env_config.active_q, requires_grad=False), (batch_size, 1)
        )
        self.qd_a_batch = torch.zeros_like(self.q_a_batch, requires_grad=False)
        self.q_p_batch = torch.tile(
            torch.tensor(env_config.passive_q, requires_grad=False), (batch_size, 1)
        )
        self.qd_p_batch = torch.zeros_like(self.q_p_batch, requires_grad=False)

    def step_batched(self, torques: torch.Tensor) -> torch.Tensor:
        """Step the dynamics model."""

        # >>> a = np.concatenate([[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12],
        # [13, 14], [15, 16]], [[17, 18], [19, 20], [21, 22], [23, 24]]], axis=-1)
        # >>> a
        # array([[ 1,  2,  9, 10, 17, 18],
        #     [ 3,  4, 11, 12, 19, 20],
        #     [ 5,  6, 13, 14, 21, 22],
        #     [ 7,  8, 15, 16, 23, 24]])

        if len(torques.shape) == 1:
            torques = torques.unsqueeze(0)

        assert torques.shape[0] == self.batch_size

        torques = torques.float()

        input_feature = self.normalize_features_fn(
            torch.concatenate(
                [
                    torques,
                    self.q_a_batch,
                    self.qd_a_batch,
                    self.q_p_batch,
                    self.qd_p_batch,
                ],
                dim=-1,
            ),
        ).float()

        qdd = self.model(input_feature)
        qdd = self.denormalize_labels_fn(qdd)

        qdd_a = qdd[:, : self.active_n_dofs]
        qdd_p = qdd[:, self.active_n_dofs :]

        self.qd_a_batch += qdd_a * self.dt
        self.q_a_batch += self.qd_a_batch * self.dt
        self.qd_p_batch += qdd_p * self.dt
        self.q_p_batch += self.qd_p_batch * self.dt

        return torch.concatenate(
            [self.q_a_batch, self.qd_a_batch, self.q_p_batch, self.qd_p_batch], dim=-1
        )

    def get_state_batched(self) -> torch.Tensor:
        """Get the state of the internal environment."""
        return torch.concatenate(
            [self.q_a_batch, self.qd_a_batch, self.q_p_batch, self.qd_p_batch], dim=-1
        )

    def set_state_batched(self, state_batch: torch.Tensor) -> None:
        """Set the state from which to step the dynamics model from."""
        self.q_a_batch = state_batch[:, : self.active_n_dofs]
        self.qd_a_batch = state_batch[:, self.active_n_dofs : 2 * self.active_n_dofs]
        self.q_p_batch = state_batch[
            :, 2 * self.active_n_dofs : 2 * self.active_n_dofs + self.passive_n_dofs
        ]
        self.qd_p_batch = state_batch[
            :, 2 * self.active_n_dofs + self.passive_n_dofs :
        ]
