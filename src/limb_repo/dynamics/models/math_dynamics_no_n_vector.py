"""Dynamics Model Using Math Formulation With N Vector."""

import numpy as np
import omegaconf
import pinocchio as pin
import pybullet_utils.bullet_client as bc

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletEnv
from limb_repo.structs import Action, JointState, LimbRepoState
from limb_repo.utils import pinocchio_utils


class MathDynamicsNoNVector(BaseDynamics):
    """Dynamics Model Using Math Formulation With N Vector."""

    def __init__(self, config: omegaconf.DictConfig) -> None:
        """Initialize the dynamics model."""
        # config.pybullet_config.use_gui = False
        self.env = LimbRepoPyBulletEnv(config=config)
        self.dt = self.env.dt
        self.current_state = self.env.get_limb_repo_state()

        # create pinnochio model for active
        self.active_model = pin.buildModelFromUrdf(self.env._active_urdf)
        self.active_data = self.active_model.createData()
        self.active_model.gravity.linear = np.array(config.pybullet_config.gravity)

        # create pinnochio model for passive
        self.passive_model = pin.buildModelFromUrdf(self.env._passive_urdf)
        self.passive_data = self.passive_model.createData()
        self.passive_model.gravity.linear = np.array(config.pybullet_config.gravity)

    def step(self, torques: Action) -> LimbRepoState:
        """Step the dynamics model."""
        current_state = self.current_state
        q_a_i = current_state.active_q
        qd_a_i = current_state.active_qd
        q_p_i = current_state.passive_q
        qd_p_i = current_state.passive_qd

        R = self.env.active_base_to_passive_base_twist

        assert np.allclose(R, R.T)
        assert np.allclose(R, np.linalg.pinv(R))

        Jr = self._calculate_jacobian(
            self.env.p, self.env.active_id, self.env.active_ee_link_id, q_a_i
        )
        Mr = self._calculate_mass_matrix(self.active_model, self.active_data, q_a_i)
        gr = self._calculate_gravity_vector(self.active_model, self.active_data, q_a_i)
        Cr = self._calculate_coriolis_matrix(
            self.active_model, self.active_data, q_a_i, qd_a_i
        )

        Jh = self._calculate_jacobian(
            self.env.p, self.env.passive_id, self.env.passive_ee_link_id, q_p_i
        )
        Jhinv = np.linalg.pinv(Jh)
        Mh = self._calculate_mass_matrix(self.passive_model, self.passive_data, q_p_i)
        gh = self._calculate_gravity_vector(
            self.passive_model, self.passive_data, q_p_i
        )
        Ch = self._calculate_coriolis_matrix(
            self.passive_model, self.passive_data, q_p_i, qd_p_i
        )

        ##### Using most simplified no n vector equation from the document

        term1 = np.linalg.pinv(
            Jr.T @ R @ np.linalg.pinv(Jh.T) @ (Mh + (Ch * self.dt)) @ Jhinv @ R @ Jr
            + Mr
            + Cr * self.dt
        )

        term2 = (
            torques
            - Jr.T
            @ R
            @ np.linalg.pinv(Jh.T)
            @ (
                (((Mh * (1 / self.dt)) - Ch) @ Jhinv @ R @ Jr @ qd_a_i)
                + ((Mh * (1 / self.dt)) @ qd_p_i - gh)
            )
            - Cr @ qd_a_i
            - gr
        )

        acc_a = term1 @ term2

        ###### Recreating equation using n vector with pinocchio values

        # Nh = Ch @ qd_p_i + gh
        # Nr = Cr @ qd_a_i + gr

        # acc_a = np.linalg.pinv((Jhinv @ R @ -Jr).T @ Mh @ (Jhinv @ R @ Jr) - Mr) @ (
        #     (Jhinv @ R @ Jr).T
        #     @ (
        #         Mh * (1 / self.dt) @ (Jhinv @ R @ Jr) @ qd_a_i
        #         - Mh * (1 / self.dt) @ qd_p_i
        #         + Nh
        #     )
        #     + Nr
        #     - np.array(torques)
        # )

        vel_a = qd_a_i + acc_a * self.dt
        lin_vel_a = Jr @ vel_a
        lin_vel_p = R @ lin_vel_a
        vel_p = Jhinv @ lin_vel_p

        # acc_p = (vel_p - qd_p_i) / self.dt

        pos_a = q_a_i + vel_a * self.dt
        pos_p = q_p_i + vel_p * self.dt

        resulting_state = LimbRepoState(np.concatenate([pos_a, vel_a, pos_p, vel_p]))

        self.env.set_limb_repo_state(resulting_state)

        self.current_state = LimbRepoState(resulting_state)

        return self.current_state

    def get_state(self) -> LimbRepoState:
        """Get the state of the dynamics model."""
        return self.env.get_limb_repo_state()

    def set_state(self, state: LimbRepoState, set_vel: bool = True) -> None:
        """Set the state of the dynamics model."""
        self.env.set_limb_repo_state(state, set_vel)

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
