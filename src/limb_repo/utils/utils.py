"""Utility functions."""

from typing import Optional, Type, TypeVar

import numpy as np
import omegaconf
import pybullet_helpers
import pybullet_helpers.inverse_kinematics
from omegaconf import OmegaConf
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from scipy.spatial.transform import Rotation as R

from limb_repo.structs import JointState, Pose

T = TypeVar("T")


def parse_config(path_to_yaml: str, config_class: Type[T]) -> omegaconf.DictConfig:
    """Parse config file with parametric dataclass."""
    config = OmegaConf.load(path_to_yaml)

    # Convert config into dictionary and initialize the specified dataclass
    config_dict = {str(key): value for key, value in dict(config).items()}

    config = OmegaConf.structured(config_class(**config_dict))
    assert isinstance(config, omegaconf.DictConfig)
    return config


def inverse_kinematics(
    robot: SingleArmPyBulletRobot, end_effector_pose: Pose, base_pose: Pose
) -> Optional[JointState]:
    """IK using pybullet_helpers."""

    world_frame_ee_pos = end_effector_pose[:3]
    world_frame_ee_orn = R.from_quat(end_effector_pose[3:])
    world_frame_base_pos = base_pose[:3]
    world_frame_base_orn = R.from_quat(base_pose[3:])

    base_frame_ee_pos = tuple(
        world_frame_base_orn.inv().apply(world_frame_ee_pos - world_frame_base_pos)
    )
    base_frame_ee_orn = tuple(R.as_quat(world_frame_base_orn * world_frame_ee_orn))
    base_frame_ee_pose = pybullet_helpers.geometry.Pose(
        base_frame_ee_pos, base_frame_ee_orn
    )

    try:
        joint_positions = np.array(
            pybullet_helpers.inverse_kinematics.inverse_kinematics(
                robot, base_frame_ee_pose
            )
        )
    except InverseKinematicsError:
        return None  # Return None if IK fails

    return joint_positions  # Return joint positions if successful
