"""Utilities for working with PyBullet."""

import pybullet_utils.bullet_client as bc


def get_free_joints(p: bc.BulletClient, body_id: int):
    """Get indices of joints that are not locked."""
    free_joints = []
    for i in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, i)
        if joint_info[2] != 4:  # if not locked
            free_joints.append(i)
    return free_joints
