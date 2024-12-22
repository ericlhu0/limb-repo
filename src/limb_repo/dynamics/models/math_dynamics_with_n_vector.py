"""Dynamics Model Using Math Formulation With N Vector."""

import numpy as np
import omegaconf
import pybullet_utils.bullet_client as bc

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.environments.lr_pybullet_env import LRPyBulletEnv
from limb_repo.structs import Action, JointState, LRState


class MathDynamicsWithNVector(BaseDynamics):
    """Dynamics Model Using Math Formulation With N Vector."""

    def __init__(self, config: omegaconf.DictConfig) -> None:
        """Initialize the dynamics model."""
        # config.pybullet_config.use_gui = False
        self.env = LRPyBulletEnv(config=config)
        print("past init")
        self.dt = self.env.dt
        self.current_state = self.env.get_lr_state()

    def step(self, torques: Action) -> LRState:
        """Step the dynamics model."""
        current_state = self.current_state
        pos_a_i = current_state.active_q
        vel_a_i = current_state.active_qd
        pos_p_i = current_state.passive_q
        vel_p_i = current_state.passive_qd
        R = self.env.active_base_to_passive_base_twist

        Jr = self._calculate_jacobian(
            self.env.p, self.env.active_id, self.env.active_ee_link_id, pos_a_i
        )
        Jh = self._calculate_jacobian(
            self.env.p, self.env.passive_id, self.env.passive_ee_link_id, pos_p_i
        )
        Jhinv = np.linalg.pinv(Jh)

        Mr = self._calculate_mass_matrix(self.env.p, self.env.active_id, pos_a_i)
        Mh = self._calculate_mass_matrix(self.env.p, self.env.passive_id, pos_p_i)

        Nr = self._calculate_N_vector(self.env.p, self.env.active_id, pos_a_i, vel_a_i)
        Nh = self._calculate_N_vector(self.env.p, self.env.passive_id, pos_p_i, vel_p_i)

        acc_a = np.linalg.pinv((Jhinv @ R @ -Jr).T @ Mh @ (Jhinv @ R @ Jr) - Mr) @ (
            (Jhinv @ R @ Jr).T
            @ (
                Mh * (1 / self.dt) @ (Jhinv @ R @ Jr) @ vel_a_i
                - Mh * (1 / self.dt) @ vel_p_i
                + Nh
            )
            + Nr
            - np.array(torques)
        )

        vel_a = vel_a_i + acc_a * self.dt
        lin_vel_a = Jr @ vel_a
        lin_vel_p = R @ lin_vel_a
        vel_p = Jhinv @ lin_vel_p

        acc_p = (vel_p - vel_p_i) / self.dt

        pos_a = pos_a_i + vel_a * self.dt
        pos_p = pos_p_i + vel_p * self.dt

        resulting_state = LRState(
            np.concatenate([pos_a, vel_a, acc_a, pos_p, vel_p, acc_p])
        )

        print("with n vector")
        print("rpos", pos_a)
        print("rvel", vel_a)
        print("racc", acc_a)
        print("hpos", pos_p)
        print("hvel", vel_p)
        print("hacc", acc_p)
        print("")

        self.env.set_lr_state(resulting_state)

        self.current_state = LRState(resulting_state)

        return self.current_state

    def get_state(self) -> LRState:
        """Get the state of the dynamics model."""
        return self.env.get_lr_state()
    
    def set_state(self, state: LRState, set_vel: bool = True, set_acc: bool = False) -> None:
        """Set the state of the dynamics model."""
        self.env.set_lr_state(state, set_vel, set_acc)

    @staticmethod
    def _calculate_jacobian(
        p: bc.BulletClient, body_id: int, ee_link_id: int, joint_positions: JointState
    ) -> np.ndarray:
        jac_t, jac_r = p.calculateJacobian(
            body_id,
            ee_link_id,
            [0, 0, 0],
            joint_positions.tolist(),
            [0.0] * len(joint_positions),
            [0.0] * len(joint_positions),
        )
        return np.concatenate((np.array(jac_t), np.array(jac_r)), axis=0)

    @staticmethod
    def _calculate_mass_matrix(
        p: bc.BulletClient, body_id: int, joint_positions: JointState
    ) -> np.ndarray:
        mass_matrix = p.calculateMassMatrix(
            body_id,
            joint_positions.tolist(),
        )
        return np.array(mass_matrix)

    @staticmethod
    def _calculate_N_vector(
        p: bc.BulletClient,
        body_id: int,
        joint_positions: JointState,
        joint_velocities: JointState,
    ) -> np.ndarray:
        joint_accel = [0.0] * len(joint_positions)
        n_vector = p.calculateInverseDynamics(
            body_id,
            joint_positions.tolist(),
            joint_velocities.tolist(),
            joint_accel,
        )
        return np.array(n_vector)
