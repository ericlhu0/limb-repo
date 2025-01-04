"""Base Dynamics Model."""

import abc

import omegaconf

from limb_repo.structs import Action, LimbRepoEEState, LimbRepoState


class BaseDynamics(abc.ABC):
    """Abstract Base Dynamics Model."""

    def __init__(self, config: omegaconf.DictConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def step(self, torques: Action) -> LimbRepoState:
        """Step the dynamics model."""

    @abc.abstractmethod
    def get_state(self) -> LimbRepoState:
        """Get the state of the internal environment."""

    @abc.abstractmethod
    def set_state(self, state: LimbRepoState, set_vel: bool) -> None:
        """Set the state from which to step the dynamics model from."""
