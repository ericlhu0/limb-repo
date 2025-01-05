"""Utilities for working with Pinocchio."""

import numpy as np


def joint_array_to_pinocchio(q: np.ndarray, model):
    """Convert a joint configuration array to one compatible with Pinocchio."""
    # Create an empty numpy array to store the Pinocchio joint configuration
    q_pin = np.zeros(model.nq)

    # Convert ROS joint config to Pinocchio config
    for i in range(model.njoints - 1):  # Iterating over the joints
        jidx = model.getJointId(model.names[i + 1])  # Get joint index from model name
        qidx = model.idx_qs[jidx]  # Get corresponding index in the configuration vector

        # Handle continuous joints (nqs[i] == 2 means sin/cos)
        if model.nqs[jidx] == 2:
            q_pin[qidx] = np.cos(q[i])  # cos for the first component
            q_pin[qidx + 1] = np.sin(q[i])  # sin for the second component
        else:
            q_pin[qidx] = q[i]  # For other joint types, just assign the position

    return q_pin
