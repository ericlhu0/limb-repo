"""Data generator for learning a dynamics model."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
from pybullet_helpers.robots.human import HumanArm6DoF
from pybullet_helpers.robots.panda import PandaPybulletRobotLimbRepo
from scipy.spatial.transform import Rotation as R

from limb_repo.dynamics.models.base_dynamics import BaseDynamics
from limb_repo.dynamics.models.base_math_dynamics import BaseMathDynamics
from limb_repo.environments.limb_repo_pybullet_env import LimbRepoPyBulletEnv
from limb_repo.structs import Action, BodyState, JointState, LimbRepoState
from limb_repo.utils import file_utils, utils


@dataclass
class DynamicsDataEntry:
    """Collected information for a dynamics step."""

    initial_state: LimbRepoState  # pos, vel for both active and passive
    action: Action  # torques for active arm
    resulting_qdd: Tuple[JointState, JointState]  # tuple of qdd for active and passive


class DynamicsDataGenerator:
    """Data generator for learning a dynamics model."""

    def __init__(
        self,
        env: LimbRepoPyBulletEnv,
        dynamics_model: BaseDynamics,
        active_joint_min: JointState = np.full(6, -np.inf),
        active_joint_max: JointState = np.full(6, np.inf),
    ) -> None:
        self._env = env
        self._dynamics_model = dynamics_model
        self._active_joint_max = active_joint_max
        self._active_joint_min = active_joint_min

        self._rng = np.random.default_rng(seed=42)

        self._pybullet_helpers_sim = bc.BulletClient(connection_mode=pybullet.DIRECT)
        # pylint: disable=protected-access
        self.pybullet_helpers_human = HumanArm6DoF(self._pybullet_helpers_sim._client)

        self._ACTIONS_PER_SAMPLED_JOINT_CONFIG = 100
        self._ACTION_MIN_TORQUE = -1
        self._ACTION_MAX_TORQUE = 1
        self._MIN_INIT_QD = -1
        self._MAX_INIT_QD = 1

    def generate_data(
        self,
        num_datapoints: int,
        final_file_path: str,
        tmp_dir: str = "/tmp/dynamics_data/",
    ) -> None:
        """Generate data for dynamics model."""
        collected_data = 0
        hdf5_saver = file_utils.HDF5Saver(final_file_path, tmp_dir)

        while collected_data < num_datapoints:
            print("col data, num data", collected_data, num_datapoints)
            # sample active initial q (to enforce joint limits earlier in this sequence)
            sampled_q_a = self._rng.uniform(
                self._active_joint_min, self._active_joint_max
            )
            sampled_qd_a = self._rng.uniform(
                self._MIN_INIT_QD, self._MAX_INIT_QD, self._env.active_n_dofs
            )

            init_active_state = BodyState(np.concatenate([sampled_q_a, sampled_qd_a]))
            self._env.set_body_state(self._env.active_id, init_active_state)

            # try to solve ik for human ee to be at the active ee
            sim_ee_state = self._env.get_limb_repo_ee_state()
            passive_ee_goal_pos = sim_ee_state.active_ee_pos
            passive_ee_goal_orn = (
                self._env.active_ee_to_passive_ee
                @ R.from_quat(sim_ee_state.active_ee_orn).as_matrix()
            )
            passive_ee_goal_pose = np.concatenate(
                [passive_ee_goal_pos, R.as_quat(R.from_matrix(passive_ee_goal_orn))]
            )
            solved_q_p = utils.inverse_kinematics(
                self.pybullet_helpers_human,
                passive_ee_goal_pose,
                self._env.passive_init_base_pose,
            )
            if solved_q_p is None:
                print("couldn't solve passive ik")
                continue

            print("do we ever pass")

            J_a = BaseMathDynamics.calculate_jacobian(
                self._env.p,
                self._env.active_id,
                self._env.active_ee_link_id,
                sampled_q_a,
            )
            J_p = BaseMathDynamics.calculate_jacobian(
                self._env.p,
                self._env.passive_id,
                self._env.passive_ee_link_id,
                solved_q_p,
            )

            solved_qd_p = (
                np.linalg.pinv(J_p)
                @ self._env.active_base_to_passive_base_twist
                @ J_a
                @ sampled_q_a
            )

            self._env.set_body_state(
                self._env.passive_id,
                BodyState(np.concatenate([solved_q_p, solved_qd_p])),
            )

            # sample a bunch of actions
            sampled_action_array = self._rng.uniform(
                self._ACTION_MIN_TORQUE,
                self._ACTION_MAX_TORQUE,
                (self._ACTIONS_PER_SAMPLED_JOINT_CONFIG, self._env.active_n_dofs),
            )

            # for each of them, step the dynamics model with a sampled action
            init_limb_repo_state = LimbRepoState(
                np.concatenate([sampled_q_a, sampled_qd_a, solved_q_p, solved_qd_p])
            )

            for sampled_torque in sampled_action_array:
                self._dynamics_model.set_state(init_limb_repo_state)
                result_state = self._dynamics_model.step(sampled_torque)
                hdf5_saver.save_demo(init_limb_repo_state, sampled_torque, result_state)

                collected_data += 1
                if collected_data >= num_datapoints:
                    break

        # merge all hdf5s into one
        hdf5_saver.combine_temp_hdf5s()