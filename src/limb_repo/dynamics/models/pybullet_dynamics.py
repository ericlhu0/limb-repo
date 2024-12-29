"""Using PyBullet as a Dynamics Model."""

import omegaconf

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.environments.lr_pybullet_env import LRPyBulletEnv
from limb_repo.structs import Action, LRState


class PyBulletDynamics(BaseDynamics):
    """Using PyBullet as a Dynamics Model."""

    def __init__(self, config: omegaconf.DictConfig) -> None:
        """Initialize the dynamics model."""
        self.env = LRPyBulletEnv(config=config)
        self.dt = self.env.dt
        self.current_state = self.env.get_lr_state()

        # Set the grasp constraint in sim
        self.env.set_grasp_constraint()

    def step(self, torques: Action) -> LRState:
        """Step the dynamics model."""
        # self.env.step()
        # return self.env.get_lr_state()/
        # return
        return self.env.send_torques(torques)

    def get_state(self) -> LRState:
        """Get the state of the dynamics model."""
        return self.env.get_lr_state()

    def set_state(self, state, set_vel=True, zero_acc=False):
        """Set the state of the dynamics model."""
        return self.env.set_lr_state(state, set_vel, zero_acc)
