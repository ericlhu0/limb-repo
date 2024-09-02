"""A base class defining the API for an approach."""

import abc
from typing import Callable

import numpy as np

from python_research_starter.structs import Action, Goal, State, Task


class Approach(abc.ABC):
    """A base class defining the API for an approach.

    In this example, an approach has access to the transition function,
    cost function, and goal function of a benchmark, but it does not
    have access to the task distribution.
    """

    def __init__(
        self,
        transition_fn: Callable[[State, Action], State],
        cost_fn: Callable[[State, Action, State], float],
        goal_fn: Callable[[State, Goal], bool],
    ) -> None:

        self._transition_fn = transition_fn
        self._cost_fn = cost_fn
        self._goal_fn = goal_fn

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """The name of the approach."""

    @abc.abstractmethod
    def generate_plan(
        self, task: Task, train_or_test: str, rng: np.random.Generator
    ) -> list[Action]:
        """Generate a plan to solve the given task."""