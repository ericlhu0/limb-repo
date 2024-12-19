# pylint: disable=no-member
"""Dynamics Model Using Math Formulation With N Vector."""

import numpy as np
import omegaconf
import pinocchio as pin
import pybullet_utils.bullet_client as bc

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.environments.lr_pybullet_env import LRPyBulletEnv
from limb_repo.structs import Action, JointState, LRState
from limb_repo.utils import pinocchio_utils


class MathDynamicsNoNVector(BaseDynamics):
    """Dynamics Model Using Math Formulation With N Vector."""

    def __init__(self, config: omegaconf.DictConfig) -> None:
        """Initialize the dynamics model."""
        # config.pybullet_config.use_gui = False
        self.env = LRPyBulletEnv(config=config)
        print("past init")
        self.dt = self.env.dt
        self.current_state = self.env.get_lr_state()

        # create pinnochio model for active
        self.active_model = pin.buildModelFromUrdf(self.env.active_urdf)
        self.active_data = self.active_model.createData()
        self.active_model.gravity.linear = np.array(config.pybullet_config.gravity)

        # create pinnochio model for passive (uses 6DoF while pybullet uses 4DoF)
        self.passive_model = pin.buildModelFromUrdf(self.env.passive_urdf)
        self.passive_data = self.passive_model.createData()
        self.passive_model.gravity.linear = np.array(config.pybullet_config.gravity)

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

        Mr = self._calculate_mass_matrix(self.active_model, self.active_data, pos_a_i)
        gr = self._calculate_gravity_vector(
            self.active_model, self.active_data, pos_a_i
        )
        Cr = self._calculate_coriolis_matrix(
            self.active_model, self.active_data, pos_a_i, vel_a_i
        )

        Mh = self._calculate_mass_matrix(self.passive_model, self.passive_data, pos_p_i)
        gh = self._calculate_gravity_vector(
            self.passive_model, self.passive_data, pos_p_i
        )
        Ch = self._calculate_coriolis_matrix(
            self.passive_model, self.passive_data, pos_p_i, vel_p_i
        )

        term1 = (
            (Jhinv @ R @ Jr).T @ (Mh + (Ch * self.dt)) @ Jhinv @ R @ Jr
            + Mr
            + Cr * self.dt
        )
        term1 = np.linalg.inv(term1)

        term2 = (
            torques
            - (Jhinv @ R @ Jr).T
            @ (
                (((Mh * (1 / self.dt)) - Ch) @ Jhinv @ R @ Jr @ vel_a_i)
                + ((Mh * (1 / self.dt)) @ vel_p_i - gh)
            )
            - Cr @ vel_a_i
            - gr
        )

        acc_a = term1 @ term2

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

        print("without n vector")
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
        body_model: pin.Model, body_data: pin.Data, joint_positions: JointState
    ) -> np.ndarray:
        joint_positions_pin = pinocchio_utils.joint_array_to_pinocchio(
            joint_positions, body_model
        )
        return pin.crba(body_model, body_data, joint_positions_pin)

    @staticmethod
    def _calculate_gravity_vector(
        body_model: pin.Model, body_data: pin.Data, joint_positions: JointState
    ) -> np.ndarray:
        joint_positions_pin = pinocchio_utils.joint_array_to_pinocchio(
            joint_positions, body_model
        )
        return pin.computeGeneralizedGravity(body_model, body_data, joint_positions_pin)

    @staticmethod
    def _calculate_coriolis_matrix(
        body_model: pin.Model,
        body_data: pin.Data,
        joint_positions: JointState,
        joint_velocities: JointState,
    ) -> np.ndarray:
        joint_positions_pin = pinocchio_utils.joint_array_to_pinocchio(
            joint_positions, body_model
        )
        return pin.computeCoriolisMatrix(
            body_model, body_data, joint_positions_pin, joint_velocities
        )
