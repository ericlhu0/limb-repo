"""Batched Math Dynamics Model."""

import numpy as np
import omegaconf
import pinocchio as pin
import pybullet_utils.bullet_client as bc
import torch

from limb_repo.dynamics.models.batched.base_batched_dynamics import BaseBatchedDynamics
from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletEnv
from limb_repo.structs import JointState, LimbRepoState
from limb_repo.utils import pinocchio_utils


class BatchedMathDynamics(BaseBatchedDynamics):
    """Dynamics Model Using Math Formulation With N Vector (Batched Version).

    For each batch element the environment is updated with the
    corresponding state, non–batched helper functions are called, and
    then the batched calculation (using torch's linear algebra routines)
    computes the state update.
    """

    def __init__(self, config: omegaconf.DictConfig, batch_size: int = 1) -> None:
        super().__init__()
        self.batch_size = batch_size

        # Initialize environment and timestep.
        self.env = LimbRepoPyBulletEnv(config=config)
        self.dt = self.env.dt
        self.active_n_dofs = self.env.active_n_dofs
        self.passive_n_dofs = self.env.passive_n_dofs

        # Build Pinocchio models.
        self.active_model = pin.buildModelFromUrdf(self.env._active_urdf)
        self.active_data = self.active_model.createData()
        self.active_model.gravity.linear = np.array(config.pybullet_config.gravity)

        self.passive_model = pin.buildModelFromUrdf(self.env._passive_urdf)
        self.passive_data = self.passive_model.createData()
        self.passive_model.gravity.linear = np.array(config.pybullet_config.gravity)

        # Transformation matrix R (non-batched).
        self.R_np = self.env.active_base_to_passive_base_twist

        # Create initial batched state from config values.
        self.q_a_batch = torch.tensor(config.active_q, dtype=torch.float32).repeat(
            batch_size, 1
        )
        self.qd_a_batch = torch.zeros_like(self.q_a_batch)
        self.q_p_batch = torch.tensor(config.passive_q, dtype=torch.float32).repeat(
            batch_size, 1
        )
        self.qd_p_batch = torch.zeros_like(self.q_p_batch)

        self.current_state = torch.cat(
            [self.q_a_batch, self.qd_a_batch, self.q_p_batch, self.qd_p_batch], dim=1
        )

    def step_batched(self, torques: torch.Tensor) -> torch.Tensor:
        """Step the dynamics model for a batch of torques as fast as
        possible."""
        bs = torques.shape[0]
        dt = self.dt

        # Use the device of the current state.
        device = self.current_state.device

        # Get current batched state.
        state = self.get_state_batched()  # shape: [bs, state_dim]
        n_a = self.active_n_dofs
        n_p = self.passive_n_dofs

        # Split state into components.
        q_a_batch = state[:, :n_a]
        qd_a_batch = state[:, n_a : 2 * n_a]
        q_p_batch = state[:, 2 * n_a : 2 * n_a + n_p]
        qd_p_batch = state[:, 2 * n_a + n_p :]

        # Preallocate fixed-size lists.
        q_a_list = []
        qd_a_list = []
        q_p_list = []
        qd_p_list = []
        Jr_list = []
        Mr_list = []
        gr_list = []
        Cr_list = []
        Jh_list = []
        Mh_list = []
        gh_list = []
        Ch_list = []

        # Cache frequently used attributes.
        env = self.env
        act_model = self.active_model
        act_data = self.active_data
        pas_model = self.passive_model
        pas_data = self.passive_data

        # Loop over batch elements.
        for i in range(bs):
            # Concatenate and convert state_i to NumPy.
            state_i = torch.cat(
                [q_a_batch[i], qd_a_batch[i], q_p_batch[i], qd_p_batch[i]], dim=0
            )
            state_i_np = state_i.detach().cpu().numpy()
            state_obj = LimbRepoState(state_i_np)
            env.set_limb_repo_state(state_obj)

            # Extract state values.
            q_a_i = state_obj.active_q
            qd_a_i = state_obj.active_qd
            q_p_i = state_obj.passive_q
            qd_p_i = state_obj.passive_qd

            q_a_list.append(q_a_i)
            qd_a_list.append(qd_a_i)
            q_p_list.append(q_p_i)
            qd_p_list.append(qd_p_i)

            # Compute physics values (non-batched calls).
            Jr_list.append(
                self.calculate_jacobian(
                    env.p, env.active_id, env.active_ee_link_id, q_a_i
                )
            )
            Mr_list.append(self.calculate_mass_matrix(act_model, act_data, q_a_i))
            gr_list.append(self.calculate_gravity_vector(act_model, act_data, q_a_i))
            Cr_list.append(
                self.calculate_coriolis_matrix(act_model, act_data, q_a_i, qd_a_i)
            )

            Jh_list.append(
                self.calculate_jacobian(
                    env.p, env.passive_id, env.passive_ee_link_id, q_p_i
                )
            )
            Mh_list.append(self.calculate_mass_matrix(pas_model, pas_data, q_p_i))
            gh_list.append(self.calculate_gravity_vector(pas_model, pas_data, q_p_i))
            Ch_list.append(
                self.calculate_coriolis_matrix(pas_model, pas_data, q_p_i, qd_p_i)
            )

        # Convert lists to batched tensors (with proper device).
        Jr_batch = torch.as_tensor(
            np.stack(Jr_list), dtype=torch.float32, device=device
        )
        Mr_batch = torch.as_tensor(
            np.stack(Mr_list), dtype=torch.float32, device=device
        )
        gr_batch = torch.as_tensor(
            np.stack(gr_list), dtype=torch.float32, device=device
        )
        Cr_batch = torch.as_tensor(
            np.stack(Cr_list), dtype=torch.float32, device=device
        )
        Jh_batch = torch.as_tensor(
            np.stack(Jh_list), dtype=torch.float32, device=device
        )
        Mh_batch = torch.as_tensor(
            np.stack(Mh_list), dtype=torch.float32, device=device
        )
        gh_batch = torch.as_tensor(
            np.stack(gh_list), dtype=torch.float32, device=device
        )
        Ch_batch = torch.as_tensor(
            np.stack(Ch_list), dtype=torch.float32, device=device
        )

        q_a_tensor = torch.as_tensor(
            np.stack(q_a_list), dtype=torch.float32, device=device
        )
        qd_a_tensor = torch.as_tensor(
            np.stack(qd_a_list), dtype=torch.float32, device=device
        )
        q_p_tensor = torch.as_tensor(
            np.stack(q_p_list), dtype=torch.float32, device=device
        )
        qd_p_tensor = torch.as_tensor(
            np.stack(qd_p_list), dtype=torch.float32, device=device
        )

        # Convert R_np to a batched tensor on the proper device.
        R_batch = torch.as_tensor(self.R_np, dtype=torch.float32, device=device)
        if R_batch.ndim == 2:
            R_batch = R_batch.unsqueeze(0).repeat(bs, 1, 1)

        # Batched pseudo-inverse calculations.
        Jhinv_batch = torch.linalg.pinv(Jh_batch)
        Jh_T_pinv_batch = torch.linalg.pinv(Jh_batch.transpose(-1, -2))

        # Batched calculation of term1.
        A_batch = (
            Jr_batch.transpose(-1, -2)
            @ R_batch
            @ Jh_T_pinv_batch
            @ (Mh_batch + Ch_batch * dt)
            @ Jhinv_batch
            @ R_batch
            @ Jr_batch
            + Mr_batch
            + Cr_batch * dt
        )
        term1_batch = torch.linalg.pinv(A_batch)

        # Batched calculation of term2.
        B_batch = (
            (Mh_batch * (1 / dt)) + Ch_batch
        ) @ Jhinv_batch @ R_batch @ Jr_batch @ qd_a_tensor.unsqueeze(-1) - (
            (Mh_batch * (1 / dt)) @ qd_p_tensor.unsqueeze(-1) + gh_batch.unsqueeze(-1)
        )
        term2_batch = (
            torques.unsqueeze(-1)
            - Jr_batch.transpose(-1, -2) @ R_batch @ Jh_T_pinv_batch @ B_batch
            - Cr_batch @ qd_a_tensor.unsqueeze(-1)
            - gr_batch.unsqueeze(-1)
        )
        qdd_a_batch = (term1_batch @ term2_batch).squeeze(-1)

        # Update active state.
        qd_a_new = qd_a_tensor + qdd_a_batch * dt
        q_a_new = q_a_tensor + qd_a_new * dt
        # Compute passive update.
        qd_p_new = (Jhinv_batch @ R_batch @ Jr_batch @ qd_a_new.unsqueeze(-1)).squeeze(
            -1
        )
        q_p_new = q_p_tensor + qd_p_new * dt

        new_state_tensor = torch.cat([q_a_new, qd_a_new, q_p_new, qd_p_new], dim=1)
        self.set_state_batched(new_state_tensor)
        return new_state_tensor

    def get_state_batched(self) -> torch.Tensor:
        """Return the current batched state."""
        return self.current_state

    def set_state_batched(self, state_batch: torch.Tensor) -> None:
        """Set the batched state (using the first element for the simulator if
        needed)."""
        self.current_state = state_batch

    @staticmethod
    def calculate_jacobian(
        p: bc.BulletClient, body_id: int, ee_link_id: int, joint_positions: JointState
    ) -> np.ndarray:
        """Calculate the Jacobian of a body in a PyBullet simulation."""
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
        """Calculate the mass matrix using Pinocchio."""
        jp = pinocchio_utils.joint_array_to_pinocchio(joint_positions, body_model)
        return pin.crba(body_model, body_data, jp)

    @staticmethod
    def calculate_gravity_vector(
        body_model: pin.Model, body_data: pin.Data, joint_positions: JointState
    ) -> np.ndarray:
        """Calculate the gravity vector using Pinocchio."""
        jp = pinocchio_utils.joint_array_to_pinocchio(joint_positions, body_model)
        return pin.computeGeneralizedGravity(body_model, body_data, jp)

    @staticmethod
    def calculate_coriolis_matrix(
        body_model: pin.Model,
        body_data: pin.Data,
        joint_positions: JointState,
        joint_velocities: JointState,
    ) -> np.ndarray:
        """Calculate the Coriolis matrix using Pinocchio."""
        jp = pinocchio_utils.joint_array_to_pinocchio(joint_positions, body_model)
        return pin.computeCoriolisMatrix(body_model, body_data, jp, joint_velocities)
