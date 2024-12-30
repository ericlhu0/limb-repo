"""Abstract Base Planner."""

import abc

from limb_repo.structs import Action


class BasePlanner(abc.ABC):
    """A base planner."""

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def get_next_action(self) -> Action:
        """Get the next action to execute."""
