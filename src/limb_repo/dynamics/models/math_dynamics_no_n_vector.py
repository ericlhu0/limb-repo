"""Dynamics Model Using Math Formulation With N Vector."""

import numpy as np

from limb_repo.dynamics.models.base_math_dynamics import BaseMathDynamics
from limb_repo.structs import Action, LimbRepoState


class MathDynamicsNoNVector(BaseMathDynamics):
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
                (((Mh * (1 / self.dt)) + Ch) @ Jhinv @ R @ Jr @ qd_a_i)
                - ((Mh * (1 / self.dt)) @ qd_p_i + gh)
            )
            - Cr @ qd_a_i
            - gr
        )

        qdd_a = term1 @ term2

        ###### Recreating equation using n vector with pinocchio values

        # Nr = Cr @ qd_a_i + gr
        # Nh = Ch @ qd_p_i + gh

        # qdd_a = np.linalg.pinv((Jhinv @ R @ -Jr).T @ Mh @ (Jhinv @ R @ Jr) - Mr) @ (
        #     (Jhinv @ R @ Jr).T
        #     @ (
        #         Mh * (1 / self.dt) @ (Jhinv @ R @ Jr) @ qd_a_i
        #         - Mh * (1 / self.dt) @ qd_p_i
        #         + Nh
        #     )
        #     + Nr
        #     - np.array(torques)
        # )

        self.current_state = self.apply_active_acceleration(
            qdd_a, q_a_i, qd_a_i, q_p_i, Jr, Jhinv, R
        )

        return self.current_state
