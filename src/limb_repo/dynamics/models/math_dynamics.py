"""Dynamics Model Using Math Formulation With N Vector."""

import numpy as np

from limb_repo.dynamics.models.base_math_dynamics import BaseMathDynamics
from limb_repo.structs import Action, JointState, LimbRepoState


class MathDynamics(BaseMathDynamics):
    """Dynamics Model Using Math Formulation With N Vector."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.R = self.env.active_base_to_passive_base_twist

        self.q_a_i = self.current_state.active_q
        self.qd_a_i = self.current_state.active_qd
        self.q_p_i = self.current_state.passive_q
        self.qd_p_i = self.current_state.passive_qd

        self.Jr = self.calculate_jacobian(
            self.env.p, self.env.active_id, self.env.active_ee_link_id, self.q_a_i
        )
        self.Mr = self.calculate_mass_matrix(
            self.active_model, self.active_data, self.q_a_i
        )
        self.gr = self.calculate_gravity_vector(
            self.active_model, self.active_data, self.q_a_i
        )
        self.Cr = self.calculate_coriolis_matrix(
            self.active_model, self.active_data, self.q_a_i, self.qd_a_i
        )

        self.Jh = self.calculate_jacobian(
            self.env.p, self.env.passive_id, self.env.passive_ee_link_id, self.q_p_i
        )
        self.Jhinv = np.linalg.pinv(self.Jh)
        self.Mh = self.calculate_mass_matrix(
            self.passive_model, self.passive_data, self.q_p_i
        )
        self.gh = self.calculate_gravity_vector(
            self.passive_model, self.passive_data, self.q_p_i
        )
        self.Ch = self.calculate_coriolis_matrix(
            self.passive_model, self.passive_data, self.q_p_i, self.qd_p_i
        )

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

        ##### Using most simplified no n vector equation from the document

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

        self.current_state = self.apply_active_acceleration(
            qdd_a, self.q_a_i, self.qd_a_i, self.q_p_i, Jr, Jhinv, self.R
        )

        return self.current_state
