"""Abstract Base Environment."""

import abc

from limb_repo.structs import LimbRepoEEState, LimbRepoState


class BaseEnv(abc.ABC):
    """An environment for the agent to interact with."""

    @abc.abstractmethod
    def get_limb_repo_state(self) -> LimbRepoState:
        """Get the state of the active and passive arm."""

    @abc.abstractmethod
    def get_limb_repo_ee_state(self) -> LimbRepoEEState:
        """Get the states of active and passive ee."""

    @abc.abstractmethod
    def set_limb_repo_state(self, state: LimbRepoState) -> None:
        """Teleports the active and passive arm to the desired state.

        *Does not step*.
        """

    @abc.abstractmethod
    def step(self) -> None:
        """Step the environment."""
