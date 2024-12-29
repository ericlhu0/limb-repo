"""PyBullet environment for Limb Repositioning."""

from dataclasses import dataclass

import numpy as np
import omegaconf
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from limb_repo.environments.pybullet_env import (
    PyBulletConfig,
    PyBulletEnv,
)
from limb_repo.structs import BodyState, LREEState, LRState, Pose
from limb_repo.utils import utils, pybullet_utils

import time

@dataclass
class LRPyBulletConfig:
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


class LRPyBulletEnv(PyBulletEnv):
    """Pybullet environment for Limb Repositioning."""

    def __init__(self, config: omegaconf.DictConfig) -> None:
        self.config = config

        ## Initialize empty pybullet simulation
        super().__init__(config.pybullet_config)

        ## Set initial values
        print("config active urdf", self.config.keys())
        self.active_urdf = utils.to_abs_path(self.config.active_urdf)
        print("active urdf", self.active_urdf)
        self.active_init_base_pose = np.array(self.config.active_base_pose)
        self.active_init_base_pos = np.array(self.active_init_base_pose[:3])
        self.active_init_base_orn = R.from_euler("xyz", self.active_init_base_pose[3:])
        self.active_init_q = np.array(self.config.active_q)
        self.active_init_state = BodyState(
            np.concatenate([self.active_init_q, np.zeros(6 + 6)])
        )
        self.active_n_dofs = len(self.active_init_q)

        self.passive_urdf = utils.to_abs_path(self.config.passive_urdf)
        self.passive_init_base_pose = np.array(self.config.passive_base_pose)
        self.passive_init_base_pos = np.array(self.passive_init_base_pose[:3])
        self.passive_init_base_orn = R.from_euler(
            "xyz", self.passive_init_base_pose[3:]
        )
        self.passive_init_q = np.array(self.config.passive_q)
        self.passive_init_state = BodyState(
            np.concatenate([self.passive_init_q, np.zeros(6 + 6)])
        )
        self.passive_n_dofs = len(self.passive_init_q)

        self.prev_active_q = self.active_init_q
        self.prev_passive_q = self.passive_init_q
        self.prev_active_qd = np.zeros(6)
        self.prev_passive_qd = np.zeros(6)

        ## Set useful rotations
        # rotates vector in active base frame to passive base frame: v_p = R @ v
        self.active_base_to_passive_base = (
            self.passive_init_base_orn.as_matrix().T
            @ self.active_init_base_orn.as_matrix()
        )
        self.active_base_to_passive_base_twist = np.block(
            [
                [self.active_base_to_passive_base, np.zeros((3, 3))],
                [np.zeros((3, 3)), self.active_base_to_passive_base],
            ]
        )
        # rotates active ee into passive ee, both in world frame: p_ee = R * a_ee
        self.active_ee_to_passive_ee = self.config.active_ee_to_passive_ee
        self.active_ee_to_passive_ee_twist = np.block(
            [
                [self.active_ee_to_passive_ee, np.zeros((3, 3))],
                [np.zeros((3, 3)), self.active_ee_to_passive_ee],
            ]
        )

        # set constraint id to none because it hasn't been created yet
        self.cid = None

        ## Load bodies into pybullet sim
        self.active_id = self.p.loadURDF(
            self.active_urdf,
            self.active_init_base_pos,
            self.active_init_base_orn.as_quat(),
            useFixedBase=True,
            flags=self.p.URDF_USE_INERTIA_FROM_FILE,
        )

        self.passive_id = self.p.loadURDF(
            self.passive_urdf,
            self.passive_init_base_pos,
            self.passive_init_base_orn.as_quat(),
            useFixedBase=True,
            flags=self.p.URDF_USE_INERTIA_FROM_FILE,
        )

        self.active_ee_link_id = self.p.getNumJoints(self.active_id) - 1
        self.passive_ee_link_id = self.p.getNumJoints(self.passive_id) - 1

        # Configure settings for sim bodies
        self.configure_body_settings()

        # Set initial states for active and passive
        print("body state before setting init", self.get_body_state(self.active_id))

        ##### for some reason I need to do this like 50 times for the GUI to react and
        ##### set the bodies??? This is also substitutable with printing 1000000 times
        ##### Internal states are updated after one call
        for _ in range(50):  # doing it 3 times sets vel and acc to 0
            self.set_body_state(self.active_id, self.active_init_state)
            self.set_body_state(self.passive_id, self.passive_init_state)
        print("body state after setting init", self.get_body_state(self.active_id))
        print("straight from pybullet")
        for i in pybullet_utils.get_good_joints(self.p, self.active_id):
            print(self.p.getJointState(self.active_id, i))
        # input()

    def step(self) -> None:
        """Step the environment."""
        self.p.stepSimulation()
        print('active')
        for i in pybullet_utils.get_good_joints(self.p, self.active_id):
            print(self.p.getJointState(self.active_id, i))
        print('passive')
        for i in pybullet_utils.get_good_joints(self.p, self.passive_id):
            print(self.p.getJointState(self.passive_id, i))
        # input()

    def send_torques(self, torques: np.ndarray) -> LRState:
        """Send joint torques."""
        curr_state = self.get_lr_state()
        self.prev_active_q = curr_state.active_q
        self.prev_active_qd = curr_state.active_qd
        self.prev_passive_q = curr_state.passive_q
        self.prev_passive_qd = curr_state.passive_qd

        # apply original torque command to active arm
        self.p.setJointMotorControlArray(
            self.active_id,
            pybullet_utils.get_good_joints(self.p, self.active_id),
            self.p.TORQUE_CONTROL,
            forces=torques,
        )

        self.step()
        
        # for i in range(5000):
        #     print('torques', torques)
        #     self.step()
        # input("finished sending 5000 torques")

        return self.get_lr_state()

    def set_body_state(
        self,
        body_id: int,
        goal_state: BodyState,
        set_vel: bool = True,
        zero_acc: bool = False,
    ) -> None:
        """Set the states of active or passive using pos & vel from the goal_state
        argument.

        set_vel: if False, only set pos, and vel is calculated using the last pos

        zero_acc: if True, acc is set to 0 by setting previous vel = curr vel
        """
        curr_state = self.get_body_state(body_id)

        if not set_vel:
            goal_state[goal_state.vel_slice] = (goal_state.q - curr_state.q) / self.dt

        if body_id == self.active_id:
            self.prev_active_q = curr_state.q
            self.prev_active_qd = (
                curr_state.qd if not zero_acc else goal_state.qd
            )
        elif body_id == self.passive_id:
            self.prev_passive_q = curr_state.q
            self.prev_passive_qd = (
                curr_state.qd if not zero_acc else goal_state.qd
            )
        else:
            raise ValueError("Invalid body id")

        for i, joint_id in enumerate(pybullet_utils.get_good_joints(self.p, body_id)):
            self.p.resetJointState(
                body_id, joint_id, goal_state.q[i], targetVelocity=goal_state.qd[i]
            )

    def set_lr_state(
        self, state: LRState, set_vel: bool = True, zero_acc: bool = False
    ) -> None:
        """Set the states of active and passive using pos & vel from the state
        argument.

        If set_vel is False, only set pos, and vel is calculated using
        the last pos.
        """
        self.set_body_state(
            self.active_id, BodyState(state.active_kinematics), set_vel, zero_acc
        )
        self.set_body_state(
            self.passive_id, BodyState(state.passive_kinematics), set_vel, zero_acc
        )

    def get_body_state(self, body_id: int) -> BodyState:
        """Get the states of active or passive."""
        q = np.array(
            [
                self.p.getJointState(body_id, i)[0]
                for i in pybullet_utils.get_good_joints(self.p, body_id)
            ]
        )

        qd = np.array(
            [
                self.p.getJointState(body_id, i)[1]
                for i in pybullet_utils.get_good_joints(self.p, body_id)
            ]
        )

        if body_id == self.active_id:
            qdd = (qd - self.prev_active_qd) / self.dt
        elif body_id == self.passive_id:
            qdd = (qd - self.prev_passive_qd) / self.dt
        else:
            raise ValueError("Invalid body id")

        return BodyState(np.concatenate([q, qd, qdd]))

    def get_lr_state(self) -> LRState:
        """Get the states of active and passive."""
        active_kinematics = self.get_body_state(self.active_id)
        passive_kinematics = self.get_body_state(self.passive_id)
        return LRState(np.concatenate([active_kinematics, passive_kinematics]))

    def get_lr_ee_state(self) -> LREEState:
        """Get the states of active and passive ee.

        Returns:
        active_ee_pos, active_ee_vel, active_ee_orn,
        passive_ee_pos, passive_ee_vel, passive_ee_orn.
        """
        active_ee_state = self.p.getLinkState(
            self.active_id, self.active_ee_link_id, computeLinkVelocity=1
        )
        passive_ee_state = self.p.getLinkState(
            self.passive_id, self.passive_ee_link_id, computeLinkVelocity=1
        )
        active_ee_pos = active_ee_state[0]  # [0] and [4] are the same
        active_ee_vel = active_ee_state[6]
        active_ee_orn = R.from_quat(active_ee_state[1]).as_matrix()
        passive_ee_pos = passive_ee_state[0]
        passive_ee_vel = passive_ee_state[6]
        passive_ee_orn = R.from_quat(passive_ee_state[1]).as_matrix()

        return LREEState(
            active_ee_pos,
            active_ee_vel,
            active_ee_orn,
            passive_ee_pos,
            passive_ee_vel,
            passive_ee_orn,
        )

    def set_grasp_constraint(self) -> None:
        """Create grasp constraint between active and passive ee."""
        # self.stop_movement()

        self.cid = self.p.createConstraint(
            self.active_id,
            self.active_ee_link_id,
            self.passive_id,
            self.passive_ee_link_id,
            self.p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0, 1],
            R.from_matrix(self.active_ee_to_passive_ee).as_quat(),
        )

        self.p.changeConstraint(self.cid, erp=0.9)

        self.stop_movement()

    def configure_body_settings(self) -> None:
        """Configure settings for sim bodies."""
        for body_id in [self.active_id, self.passive_id]:
            # remove friction terms as well contact stiffness and damping
            for joint in range(self.p.getNumJoints(body_id)):
                self.p.changeDynamics(
                    body_id,
                    joint,
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

                # disable collisions
                self.p.setCollisionFilterGroupMask(body_id, joint, 0, 0)

                # # apply velocity control to panda arm to make it stationary
                # self.p.setJointMotorControl2(
                #     body_id, joint, self.p.VELOCITY_CONTROL, targetVelocity=0, force=50
                # )

                # self.p.setJointMotorControl2(body_id, joint, self.p.TORQUE_CONTROL, force=0)

                self.p.enableJointForceTorqueSensor(body_id, joint, 1)

    def stop_movement(self) -> None:
        """Stop all movement of the active and passive arms."""

        for body_id in [self.active_id, self.passive_id]:
            # remove friction terms as well contact stiffness and damping
            for joint in range(self.p.getNumJoints(body_id)):
                self.p.setJointMotorControl2(
                    body_id, joint, self.p.VELOCITY_CONTROL, targetVelocity=0, force=50
                )

        for _ in range(1000):
            self.p.stepSimulation()

        for body_id in [self.active_id, self.passive_id]:
            # remove friction terms as well contact stiffness and damping
            for joint in range(self.p.getNumJoints(body_id)):
                self.p.setJointMotorControl2(
                    body_id, joint, self.p.TORQUE_CONTROL, force=0
                )

        for _ in range(1000):
            self.p.stepSimulation()
            print("setting up2")

        for body_id in [self.active_id, self.passive_id]:
            # remove friction terms as well contact stiffness and damping
            for joint in range(self.p.getNumJoints(body_id)):
                self.p.setJointMotorControl2(
                    body_id, joint, self.p.VELOCITY_CONTROL, targetVelocity=0, force=50
                )

        for _ in range(1000):
            self.p.stepSimulation()
            print("setting up3")

        for body_id in [self.active_id, self.passive_id]:
            for joint in range(self.p.getNumJoints(body_id)):
                self.p.setJointMotorControl2(
                    body_id, joint, self.p.TORQUE_CONTROL, force=0
                )

        for _ in range(1000):
            self.p.stepSimulation()
            print("setting up4")

    @staticmethod
    def parse_config(path_to_yaml: str) -> omegaconf.DictConfig:
        """Parse a configuration file."""
        config = omegaconf.DictConfig(OmegaConf.load(path_to_yaml))

        # to get around mypy "Keywords must be strings"
        # and "value after ** should be a mapping"
        config_dict = {str(key): value for key, value in dict(config).items()}

        config = OmegaConf.structured(LRPyBulletConfig(**config_dict))

        return config


if __name__ == "__main__":
    pass
