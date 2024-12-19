"""Base Dynamics Model."""

import abc

from limb_repo.structs import Action, LRState


class BaseDynamics(abc.ABC):
    """Abstract Base Dynamics Model."""

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def step(self, torques: Action) -> LRState:
        """Step the dynamics model."""

    @abc.abstractmethod
    def get_state(self) -> LRState:
        """Get the state of the dynamics model."""
