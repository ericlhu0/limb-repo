"""Base dynamics model that uses a simulator."""

import abc

import omegaconf
import torch

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.structs import Action, JointState, LimbRepoEEState, LimbRepoState


class BaseDynamicsWithSim(BaseDynamics):
    """Base dynamics model that uses a simulator."""

    def __init__(self, config: omegaconf.DictConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def step(self, torques: Action) -> LimbRepoState | torch.Tensor:
        """Step the dynamics model."""

    @abc.abstractmethod
    def get_state(self) -> LimbRepoState | torch.Tensor:
        """Get the state of the internal environment."""

    @abc.abstractmethod
    def get_ee_state(self) -> LimbRepoEEState:
        """Get the state of the end effector."""

    @abc.abstractmethod
    def set_state(self, state: LimbRepoState) -> None:
        """Set the state from which to step the dynamics model from."""
