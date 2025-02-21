"""Base Dynamics Model."""

import abc

import torch


class BaseBatchedDynamics(abc.ABC):
    """Abstract Base Torch Dynamics Model."""

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def step_batched(self, torques: torch.Tensor) -> torch.Tensor:
        """Batch step the dynamics model."""

    @abc.abstractmethod
    def get_state_batched(self) -> torch.Tensor:
        """Get the batched state of the internal environment."""

    @abc.abstractmethod
    def set_state_batched(self, state_batch: torch.Tensor) -> None:
        """Set the batched state from which to step the dynamics model from."""
