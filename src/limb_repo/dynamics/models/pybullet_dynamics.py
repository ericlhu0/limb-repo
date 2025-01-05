"""Using PyBullet as a Dynamics Model."""

import omegaconf

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletEnv
from limb_repo.structs import Action, LimbRepoState


class PyBulletDynamics(BaseDynamics):
    """Using PyBullet as a Dynamics Model."""

    def __init__(self, config: omegaconf.DictConfig) -> None:
        """Initialize the dynamics model."""
        self.env = LimbRepoPyBulletEnv(config=config)

        # Set the grasp constraint in sim
        self.env.set_limb_repo_constraint()

    def step(self, torques: Action) -> LimbRepoState:
        """Step the dynamics model."""
        return self.env.send_torques(torques)

    def get_state(self) -> LimbRepoState:
        """Get the state of the dynamics model."""
        return self.env.get_limb_repo_state()

    def set_state(self, state, set_vel=True) -> None:
        """Set the state of the dynamics model."""
        self.env.set_limb_repo_state(state, set_vel)
