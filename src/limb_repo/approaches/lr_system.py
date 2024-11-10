"""A base class defining the API for a Limb Repositioning System."""

import abc

from limb_repo.environments.base_environment import BaseEnvironment
from limb_repo.structs import *


class LimbRepositioningSystem:
    """A base class defining the API for a Limb Repositioning System."""

    def __init__(
        self,
        environment: BaseEnvironment,
        controller: Controller,
    ) -> None:
        self.environment: BaseEnvironment = environment
        self.controller: Controller = controller

    @abc.abstractmethod
    def send_control(self, state: State) -> None:
        """Send control to the environment agent."""

    @abc.abstractmethod
    def get_state(self) -> State:
        """Get the state of the environment agent."""