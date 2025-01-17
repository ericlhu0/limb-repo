"""Base Dynamics Model."""

import abc

import omegaconf
import torch


class BaseTorchDynamics(abc.ABC):
    """Abstract Base Dynamics Model."""

    @abc.abstractmethod
    def step_batched(self, torques: torch.Tensor) -> torch.Tensor:
        """Step the dynamics model."""

    @abc.abstractmethod
    def get_state(self) -> torch.Tensor:
        """Get the state of the internal environment."""

    @abc.abstractmethod
    def set_state(self, state: torch.Tensor) -> None:
        """Set the state from which to step the dynamics model from."""
