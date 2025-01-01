"""Utility functions."""

import numpy as np
import os
import omegaconf
from omegaconf import OmegaConf
from scipy.spatial import transform as R
from typing import Type, TypeVar

from pybullet_helpers.inverse_kinematics import inverse_kinematics as pybullet_helpers_ik
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pybullet_helpers.robots.human import HumanArm6DoF
from pybullet_helpers.robots.panda import PandaPybulletRobotLimbRepo
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions

def get_root_path() -> str:
    """Get the root path of the repository."""
    return os.path.abspath(os.path.join(__file__, "../../../.."))


def to_abs_path(input_path: str) -> str:
    """Get the absolute path of the repository."""
    return os.path.abspath(os.path.join(get_root_path(), input_path))


T = TypeVar("T")


def parse_config(path_to_yaml: str, config_class: Type[T]) -> omegaconf.DictConfig:
    """Parse config file with parametric dataclass."""
    config = OmegaConf.load(path_to_yaml)

    # Convert config into dictionary and initialize the specified dataclass
    config_dict = {str(key): value for key, value in dict(config).items()}

    config = OmegaConf.structured(config_class(**config_dict))
    assert isinstance(config, omegaconf.DictConfig)
    return config


def inverse_kinematics(self, active_id: int, passive_id: int, robot: SingleArmPyBulletRobot, end_effector_pose: Pose, base_pose: Pose) -> JointPositions:
    """ IK using pybullet_helpers"""

    world_frame_ee_pos = np.array(end_effector_pose.position)
    world_frame_ee_orn = np.reshape(p.getMatrixFromQuaternion(end_effector_pose.orientation), (3, 3))
    world_frame_base_pos = np.array(base_pose.position)
    world_frame_base_orn = np.reshape(p.getMatrixFromQuaternion(base_pose.orientation), (3, 3))

    base_frame_ee_pos = tuple(np.linalg.inv(world_frame_base_orn) @ (world_frame_ee_pos - world_frame_base_pos))
    base_frame_ee_orn = tuple(R.as_quat(R.from_matrix(np.linalg.inv(world_frame_base_orn) @ world_frame_ee_orn)))
    base_frame_ee_pose = Pose(base_frame_ee_pos, base_frame_ee_orn)

    joint_positions = pybullet_helpers_ik(robot, base_frame_ee_pose) # raises InverseKinematics error
        
    return joint_positions