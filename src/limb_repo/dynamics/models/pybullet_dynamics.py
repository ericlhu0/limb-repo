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

    def step(self, torques: Action) -> LRState:
        """Step the dynamics model."""
        current_state = self.current_state
        pos_a_i = current_state.active_q
        vel_a_i = current_state.active_qd
        pos_p_i = current_state.passive_q
        vel_p_i = current_state.passive_qd

        self.env.set_body_state(self.env.active_id, pos_a_i, vel_a_i, torques)
        self.env.set_body_state(self.env.passive_id, pos_p_i, vel_p_i, torques)

        next_state = self.env.get_lr_state()
        self.current_state = next_state

        return next_state