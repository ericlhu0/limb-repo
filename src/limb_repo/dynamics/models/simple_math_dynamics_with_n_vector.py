"""Dynamics Model Using Math Formulation With N Vector."""

import numpy as np

from limb_repo.dynamics.models.base_math_dynamics import BaseMathDynamics
from limb_repo.structs import Action, JointState, LimbRepoState


class SimpleMathDynamicsWithNVector(BaseMathDynamics):
    """Dynamics Model Using Math Formulation With N Vector."""

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

        Jr = self.calculate_jacobian(
            self.env.p, self.env.active_id, self.env.active_ee_link_id, q_a_i
        )
        Mr = self.calculate_mass_matrix(self.active_model, self.active_data, q_a_i)
        gr = self.calculate_gravity_vector(self.active_model, self.active_data, q_a_i)
        Cr = self.calculate_coriolis_matrix(
            self.active_model, self.active_data, q_a_i, qd_a_i
        )
        Nr = Cr @ qd_a_i + gr

        Jh = self.calculate_jacobian(
            self.env.p, self.env.passive_id, self.env.passive_ee_link_id, q_p_i
        )
        Jhinv = np.linalg.pinv(Jh)
        Mh = self.calculate_mass_matrix(self.passive_model, self.passive_data, q_p_i)
        gh = self.calculate_gravity_vector(self.passive_model, self.passive_data, q_p_i)
        Ch = self.calculate_coriolis_matrix(
            self.passive_model, self.passive_data, q_p_i, qd_p_i
        )
        Nh = Ch @ qd_p_i + gh

        qdd_a = np.linalg.pinv((Jhinv @ R @ -Jr).T @ Mh @ (Jhinv @ R @ Jr) - Mr) @ (
            (Jhinv @ R @ Jr).T
            @ Nh + Nr - np.array(torques)
        )

        self.current_state = self.apply_active_acceleration(
            qdd_a, q_a_i, qd_a_i, q_p_i, Jr, Jhinv, R
        )

        return self.current_state

    def step_return_qdd(self, torques: Action) -> JointState:
        """Step the dynamics model."""
        current_state = self.current_state
        q_a_i = current_state.active_q
        qd_a_i = current_state.active_qd
        q_p_i = current_state.passive_q
        qd_p_i = current_state.passive_qd

        R = self.env.active_base_to_passive_base_twist

        assert np.allclose(R, R.T)
        assert np.allclose(R, np.linalg.pinv(R))

        Jr = self.calculate_jacobian(
            self.env.p, self.env.active_id, self.env.active_ee_link_id, q_a_i
        )
        Mr = self.calculate_mass_matrix(self.active_model, self.active_data, q_a_i)
        gr = self.calculate_gravity_vector(self.active_model, self.active_data, q_a_i)
        Cr = self.calculate_coriolis_matrix(
            self.active_model, self.active_data, q_a_i, qd_a_i
        )
        Nr = Cr @ qd_a_i + gr

        Jh = self.calculate_jacobian(
            self.env.p, self.env.passive_id, self.env.passive_ee_link_id, q_p_i
        )
        Jhinv = np.linalg.pinv(Jh)
        Mh = self.calculate_mass_matrix(self.passive_model, self.passive_data, q_p_i)
        gh = self.calculate_gravity_vector(self.passive_model, self.passive_data, q_p_i)
        Ch = self.calculate_coriolis_matrix(
            self.passive_model, self.passive_data, q_p_i, qd_p_i
        )
        Nh = Ch @ qd_p_i + gh

        qdd_a = np.linalg.pinv((Jhinv @ R @ -Jr).T @ Mh @ (Jhinv @ R @ Jr) - Mr) @ (
            (Jhinv @ R @ Jr).T
            @ (
                Mh * (1 / self.dt) @ (Jhinv @ R @ Jr) @ qd_a_i
                - Mh * (1 / self.dt) @ qd_p_i
                + Nh
            )
            + Nr
            - np.array(torques)
        )

        return qdd_a
