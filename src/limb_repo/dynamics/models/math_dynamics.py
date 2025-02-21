"""Dynamics Model Using Math Formulation With N Vector."""

import numpy as np
import omegaconf
import pinocchio as pin
import pybullet_utils.bullet_client as bc

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletEnv
from limb_repo.structs import Action, JointState, LimbRepoEEState, LimbRepoState
from limb_repo.utils import pinocchio_utils


class MathDynamics(BaseDynamics):
    """Dynamics Model Using Math Formulation With N Vector."""

    def __init__(self, config: omegaconf.DictConfig) -> None:
        super().__init__(config)

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

        self.R = self.env.active_base_to_passive_base_twist

        self.q_a_i = self.current_state.active_q
        self.qd_a_i = self.current_state.active_qd
        self.q_p_i = self.current_state.passive_q
        self.qd_p_i = self.current_state.passive_qd

    def step(self, torques: Action) -> LimbRepoState:
        """Step the dynamics model."""
        current_state = self.current_state
        self.q_a_i = current_state.active_q
        self.qd_a_i = current_state.active_qd
        self.q_p_i = current_state.passive_q
        self.qd_p_i = current_state.passive_qd

        assert np.allclose(self.R, self.R.T)
        assert np.allclose(self.R, np.linalg.pinv(self.R))

        Jr = self.calculate_jacobian(
            self.env.p, self.env.active_id, self.env.active_ee_link_id, self.q_a_i
        )
        Mr = self.calculate_mass_matrix(self.active_model, self.active_data, self.q_a_i)
        gr = self.calculate_gravity_vector(
            self.active_model, self.active_data, self.q_a_i
        )
        Cr = self.calculate_coriolis_matrix(
            self.active_model, self.active_data, self.q_a_i, self.qd_a_i
        )

        Jh = self.calculate_jacobian(
            self.env.p, self.env.passive_id, self.env.passive_ee_link_id, self.q_p_i
        )
        Jhinv = np.linalg.pinv(Jh)
        Mh = self.calculate_mass_matrix(
            self.passive_model, self.passive_data, self.q_p_i
        )
        gh = self.calculate_gravity_vector(
            self.passive_model, self.passive_data, self.q_p_i
        )
        Ch = self.calculate_coriolis_matrix(
            self.passive_model, self.passive_data, self.q_p_i, self.qd_p_i
        )

        term1 = np.linalg.pinv(
            Jr.T
            @ self.R
            @ np.linalg.pinv(Jh.T)
            @ (Mh + (Ch * self.dt))
            @ Jhinv
            @ self.R
            @ Jr
            + Mr
            + Cr * self.dt
        )

        term2 = (
            torques
            - Jr.T
            @ self.R
            @ np.linalg.pinv(Jh.T)
            @ (
                (((Mh * (1 / self.dt)) + Ch) @ Jhinv @ self.R @ Jr @ self.qd_a_i)
                - ((Mh * (1 / self.dt)) @ self.qd_p_i + gh)
            )
            - Cr @ self.qd_a_i
            - gr
        )

        qdd_a = term1 @ term2

        qd_a = self.qd_a_i + qdd_a * self.dt
        qd_p = Jhinv @ self.R @ Jr @ qd_a

        q_a = self.q_a_i + qd_a * self.dt
        q_p = self.q_p_i + qd_p * self.dt

        resulting_state = LimbRepoState(np.concatenate([q_a, qd_a, q_p, qd_p]))

        self.current_state = resulting_state
        self.env.set_limb_repo_state(resulting_state)

        return self.current_state

    def get_state(self) -> LimbRepoState:
        """Get the state of the dynamics model."""
        return self.current_state

    def get_ee_state(self) -> LimbRepoEEState:
        """Get the state of the dynamics model."""
        return self.env.get_limb_repo_ee_state()

    def set_state(self, state: LimbRepoState) -> None:
        """Set the state of the dynamics model."""
        self.current_state = state
        self.env.set_limb_repo_state(state)

    @staticmethod
    def calculate_jacobian(
        p: bc.BulletClient, body_id: int, ee_link_id: int, joint_positions: JointState
    ) -> np.ndarray:
        """Calculate the jacobian of a body in a Pybullet sim."""
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
    def calculate_mass_matrix(
        body_model: pin.Model, body_data: pin.Data, joint_positions: JointState
    ) -> np.ndarray:
        """Calculate the mass matrix of a body in a Pinocchio sim."""
        joint_positions_pin = pinocchio_utils.joint_array_to_pinocchio(
            joint_positions, body_model
        )
        return pin.crba(body_model, body_data, joint_positions_pin)

    @staticmethod
    def calculate_gravity_vector(
        body_model: pin.Model, body_data: pin.Data, joint_positions: JointState
    ) -> np.ndarray:
        """Calculate the gravity vector of a body in a Pinocchio sim."""
        joint_positions_pin = pinocchio_utils.joint_array_to_pinocchio(
            joint_positions, body_model
        )
        return pin.computeGeneralizedGravity(body_model, body_data, joint_positions_pin)

    @staticmethod
    def calculate_coriolis_matrix(
        body_model: pin.Model,
        body_data: pin.Data,
        joint_positions: JointState,
        joint_velocities: JointState,
    ) -> np.ndarray:
        """Calculate the coriolis matrix of a body in a Pinocchio sim."""
        joint_positions_pin = pinocchio_utils.joint_array_to_pinocchio(
            joint_positions, body_model
        )
        return pin.computeCoriolisMatrix(
            body_model, body_data, joint_positions_pin, joint_velocities
        )
