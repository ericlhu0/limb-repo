"""PyBullet environment for Limb Repositioning."""

from dataclasses import dataclass

import numpy as np
import omegaconf
import pybullet
from scipy.spatial.transform import Rotation as R

from limb_repo.environments.pybullet_env import (
    PyBulletConfig,
    PyBulletEnv,
)
from limb_repo.structs import BodyState, LimbRepoState, Pose
from limb_repo.utils import pybullet_utils


@dataclass
class LimbRepoPyBulletConfig:
    """Configuration for a Limb Repo PyBullet environment."""

    pybullet_config: PyBulletConfig
    active_base_pose: Pose
    active_q: np.ndarray
    active_urdf: str
    passive_base_pose: Pose
    passive_q: np.ndarray
    passive_urdf: str
    wheelchair_pose: Pose
    # wheelchair_config: np.ndarray
    wheelchair_urdf: str
    active_ee_to_passive_ee: np.ndarray


class LimbRepoPyBulletEnv(PyBulletEnv):
    """Pybullet environment for Limb Repositioning."""

    def __init__(self, config: omegaconf.DictConfig) -> None:
        self.config = config

        ## Initialize empty pybullet simulation
        super().__init__(config.pybullet_config)

        ## Set initial values
        print("config active urdf", self.config.keys())
        self._active_urdf: str = self.config.active_urdf
        self._active_init_base_pose = np.array(self.config.active_base_pose)
        self._active_init_base_pos = np.array(self._active_init_base_pose[:3])
        self._active_init_base_orn = R.from_euler(
            "xyz", self._active_init_base_pose[3:]
        )
        self._active_init_q = np.array(self.config.active_q)
        self._active_init_state = BodyState(
            np.concatenate([self._active_init_q, np.zeros(6)])
        )
        self._active_n_dofs = len(self._active_init_q)

        self._passive_urdf: str = self.config.passive_urdf
        self._passive_init_base_pose = np.array(self.config.passive_base_pose)
        self._passive_init_base_pos = np.array(self._passive_init_base_pose[:3])
        self._passive_init_base_orn = R.from_euler(
            "xyz", self._passive_init_base_pose[3:]
        )
        self._passive_init_q = np.array(self.config.passive_q)
        self._passive_init_state = BodyState(
            np.concatenate([self._passive_init_q, np.zeros(6)])
        )
        self._passive_n_dofs = len(self._passive_init_q)

        self._prev_active_q = self._active_init_q
        self._prev_passive_q = self._passive_init_q
        self._prev_active_qd = np.zeros(6)
        self._prev_passive_qd = np.zeros(6)

        ## Set useful rotations
        # rotates vector in active base frame to passive base frame: v_p = R @ v
        self._active_base_to_passive_base = (
            self._passive_init_base_orn.as_matrix().T
            @ self._active_init_base_orn.as_matrix()
        )
        self._active_base_to_passive_base_twist = np.block(
            [
                [self._active_base_to_passive_base, np.zeros((3, 3))],
                [np.zeros((3, 3)), self._active_base_to_passive_base],
            ]
        )
        # rotates active ee into passive ee, both in world frame: p_ee = R * a_ee
        self._active_ee_to_passive_ee = self.config.active_ee_to_passive_ee
        self._active_ee_to_passive_ee_twist = np.block(
            [
                [self._active_ee_to_passive_ee, np.zeros((3, 3))],
                [np.zeros((3, 3)), self._active_ee_to_passive_ee],
            ]
        )

        # set constraint id to none because it hasn't been created yet
        self._cid = None

        ## Load bodies into pybullet sim
        self.active_id = self.p.loadURDF(
            self._active_urdf,
            self._active_init_base_pos,
            self._active_init_base_orn.as_quat(),
            useFixedBase=True,
            flags=self.p.URDF_USE_INERTIA_FROM_FILE,
        )

        self.passive_id = self.p.loadURDF(
            self._passive_urdf,
            self._passive_init_base_pos,
            self._passive_init_base_orn.as_quat(),
            useFixedBase=True,
            flags=self.p.URDF_USE_INERTIA_FROM_FILE,
        )

        self._active_ee_link_id = self.p.getNumJoints(self.active_id) - 1
        self._passive_ee_link_id = self.p.getNumJoints(self.passive_id) - 1

        # Configure settings for sim bodies
        self.configure_body_settings()

        # Set initial states for active and passive
        for _ in range(2):  # doing it 2 times sets vel and acc to 0
            self.set_body_state(self.active_id, self._active_init_state)
            self.set_body_state(self.passive_id, self._passive_init_state)

    def step(self) -> None:
        """Step the environment."""
        self.p.stepSimulation()

    def send_torques(self, torques: np.ndarray) -> None:
        """Send joint torques to the active body."""
        # to use torque control, velocity control must be disabled at every time step
        prev_state = self.get_limb_repo_state()
        self._prev_active_q = prev_state.active_q
        self._prev_active_qd = prev_state.active_qd
        self._prev_passive_q = prev_state.active_q
        self._prev_passive_qd = prev_state.active_qd

        # disable velocity control
        for j in pybullet_utils.get_free_joints(self.p, self.active_id):
            self.p.setJointMotorControl2(
                bodyUniqueId=self.active_id,
                jointIndex=j,
                controlMode=self.p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,
            )

        for j in pybullet_utils.get_free_joints(self.p, self.passive_id):
            self.p.setJointMotorControl2(
                bodyUniqueId=self.passive_id,
                jointIndex=j,
                controlMode=self.p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,
            )

        # apply original torque command to robot
        self.p.setJointMotorControlArray(
            bodyUniqueId=self.active_id,
            jointIndices=pybullet_utils.get_free_joints(self.p, self.active_id),
            controlMode=pybullet.TORQUE_CONTROL,
            forces=torques,
        )

        print("state before stepping: ", self.get_limb_repo_state().active_qd)
        self.step()
        print("state after stepping: ", self.get_limb_repo_state().passive_qd)

    def set_body_state(
        self, body_id: int, state: BodyState, set_vel: bool = True
    ) -> None:
        """Set the states of active or passive using pos & vel from the state
        argument.

        If set_vel is False, only set pos, and vel is calculated using
        the last pos.
        """
        prev_state = self.get_body_state(body_id)
        if body_id == self.active_id:
            self._prev_active_q = prev_state.q
            self._prev_active_qd = prev_state.qd
        elif body_id == self.passive_id:
            self._prev_passive_q = prev_state.q
            self._prev_passive_qd = prev_state.qd
        else:
            raise ValueError("Invalid body id")

        if not set_vel:
            state[state.vel_slice] = (state.q - prev_state.q) / self.dt

        for i, joint_id in enumerate(pybullet_utils.get_free_joints(self.p, body_id)):
            self.p.resetJointState(
                body_id, joint_id, state.q[i], targetVelocity=state.qd[i]
            )

    def set_limb_repo_state(self, state: LimbRepoState, set_vel: bool = True) -> None:
        """Set the states of active and passive using pos & vel from the state
        argument.

        If set_vel is False, only set pos, and vel is calculated using
        the last pos.
        """
        self.set_body_state(state.active, self.active_id, set_vel)
        self.set_body_state(state.passive, self.passive_id, set_vel)

    def get_body_state(self, body_id: int) -> BodyState:
        """Get the states of active or passive."""
        pos = np.array(
            [
                self.p.getJointState(body_id, i)[0]
                for i in pybullet_utils.get_free_joints(self.p, body_id)
            ]
        )

        vel = np.array(
            [
                self.p.getJointState(body_id, i)[1]
                for i in pybullet_utils.get_free_joints(self.p, body_id)
            ]
        )

        return BodyState(np.concatenate([pos, vel]))

    def get_limb_repo_state(self) -> LimbRepoState:
        """Get the states of active and passive."""
        active_kinematics = self.get_body_state(self.active_id)
        passive_kinematics = self.get_body_state(self.passive_id)
        return LimbRepoState(np.concatenate([active_kinematics, passive_kinematics]))

    def set_limb_repo_constraint(self) -> None:
        """Create grasp constraint between active and passive ee."""
        self._cid = self.p.createConstraint(
            self.active_id,
            self._active_ee_link_id,
            self.passive_id,
            self._passive_ee_link_id,
            self.p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0, 1],
            R.from_matrix(self._active_ee_to_passive_ee).as_quat(),
        )

    def configure_body_settings(self) -> None:
        """Configure settings for sim bodies."""
        # remove friction terms as well contact stiffness and damping

        for body_id in [self.active_id, self.passive_id]:
            for i in range(self.p.getNumJoints(self.passive_id)):
                self.p.changeDynamics(
                    body_id,
                    i,
                    jointDamping=0.0,
                    anisotropicFriction=0.0,
                    maxJointVelocity=5000,
                    linearDamping=0.0,
                    angularDamping=0.0,
                    lateralFriction=0.0,
                    spinningFriction=0.0,
                    rollingFriction=0.0,
                    contactStiffness=0.0,
                    contactDamping=0.0,
                )  # , jointLowerLimit=-6.283185 * 500, jointUpperLimit=6.283185 * 500)

            # remove collision for both robot and human arms
            group = 0
            mask = 0
            for linkIndex in range(self.p.getNumJoints(self.passive_id)):
                self.p.setCollisionFilterGroupMask(body_id, linkIndex, group, mask)

            # # apply velocity control to panda arm to make it stationary
            # for i in range(self._active_n_dofs):
            #     self.p.setJointMotorControl2(
            #         body_id, i, self.p.VELOCITY_CONTROL, targetVelocity=0, force=50
            #     )

            # for i in range(1000):
            #     self.p.stepSimulation()

            # enable force torque
            for joint in range(self.p.getNumJoints(body_id)):
                self.p.enableJointForceTorqueSensor(body_id, joint, 1)


if __name__ == "__main__":
    pass
