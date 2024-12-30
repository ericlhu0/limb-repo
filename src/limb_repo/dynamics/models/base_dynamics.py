"""Base Dynamics Model."""

import abc

from limb_repo.structs import Action, LimbRepoState


class BaseDynamics(abc.ABC):
    """Abstract Base Dynamics Model."""

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def step(self, torques: Action) -> LimbRepoState:
        """Step the dynamics model."""

    @abc.abstractmethod
    def get_state(self) -> LimbRepoState:
        """Get the state of the dynamics model."""

    @abc.abstractmethod
    def set_state(self, state: LimbRepoState, set_vel: bool, zero_acc: bool) -> None:
        """Set the state of the dynamics model."""
