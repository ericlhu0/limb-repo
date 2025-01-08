"""Dynamics using a neural network."""

import numpy as np
import omegaconf
import torch
from torch import nn

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.structs import Action, JointState, LimbRepoEEState, LimbRepoState


class PyTorchLearnedDynamicsModel(nn.Module):
    def __init__(self, active_n_dofs: int, passive_n_dofs: int):
        super().__init__()

        # inputs: active torque, q sin & cos, q; passive q sin & cos, qd
        num_inputs = 3 * active_n_dofs + 2 * passive_n_dofs

        # outputs: next robot qdd, human qdd
        num_outputs = active_n_dofs + passive_n_dofs

        self.layer_stack = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
        )

    def forward(self, inp):
        return self.layer_stack(inp)


class LearnedDynamics(BaseDynamics):
    """Dynamics using a neural network."""

    def __init__(
        self, config: omegaconf.DictConfig, weights_path: str, batch_size: int = 1
    ) -> None:
        super().__init__(config)

        self.batch_size = batch_size

        self.active_n_dofs = len(config.active_q)
        self.passive_n_dofs = len(config.passive_q)

        self.model = PyTorchLearnedDynamicsModel(
            self.active_n_dofs, self.passive_n_dofs
        )
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

        self.dt = self.config.pybullet_config.dt

        self.q_a = torch.tensor(np.repeat(config.active_q, batch_size))
        self.qd_a = torch.zeros_like(self.q_a)
        self.q_p = torch.tensor(np.repeat(config.passive_q, batch_size))
        self.qd_p = torch.zeros_like(self.q_p)

    def step(self, torques: Action) -> torch.Tensor:
        """Step the dynamics model."""
        assert torques.shape[0] == self.batch_size

        # >>> a = np.concatenate([[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12], [13, 14], [15, 16]], [[17, 18], [19, 20], [21, 22], [23, 24]]], axis=-1)
        # >>> a
        # array([[ 1,  2,  9, 10, 17, 18],
        #     [ 3,  4, 11, 12, 19, 20],
        #     [ 5,  6, 13, 14, 21, 22],
        #     [ 7,  8, 15, 16, 23, 24]])

        input_feature = torch.tensor(
            np.concatenate(
                [
                    torques,
                    torch.sin(self.q_a),
                    torch.cos(self.q_a),
                    self.qd_a,
                    torch.sin(self.q_p),
                    torch.cos(self.q_p),
                    self.qd_p,
                ],
                axis=-1,
            )
        )

        qdd = self.model(input_feature)
        qdd_a = qdd[:, : self.active_n_dofs]
        qdd_p = qdd[:, self.active_n_dofs :]

        self.qd_a += qdd_a * self.dt
        self.q_a += self.qd_a * self.dt
        self.qd_p += qdd_p * self.dt
        self.q_p += self.qd_p * self.dt

        return torch.concatenate([self.q_a, self.qd_a, self.q_p, self.qd_p])

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
