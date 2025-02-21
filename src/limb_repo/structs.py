"""Data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

# Define some data structures for this example repository (to be changed).
State: TypeAlias = np.ndarray
Action: TypeAlias = np.ndarray
Goal: TypeAlias = State
Pose: TypeAlias = np.ndarray  # x, y, z, qx, qy, qz, qw
"""3D Position and Quaternion: x, y, z, qx, qy, qz, qw"""


class Task:
    """An initial state and goal."""

    init: State
    goal: Goal


class Controller:
    """A controller for the agent to interact with the environment."""


# Limb Repo Structs
JointState: TypeAlias = State


class BodyState(JointState):
    """Single Body State.

    This is a subclass of np.ndarray, and allows access to kinematic
    states as properties.
    """

    n_dofs: int
    pos_slice: slice
    vel_slice: slice

    # pylint: disable=attribute-defined-outside-init
    def __new__(cls, input_array: np.ndarray, n_dofs: int = 6) -> BodyState:
        assert len(input_array) == 2 * n_dofs

        obj = np.asarray(input_array).view(cls)
        obj.n_dofs = n_dofs

        obj.pos_slice = slice(0, obj.n_dofs)
        obj.vel_slice = slice(obj.n_dofs, 2 * obj.n_dofs)

        return obj

    # pylint: disable=attribute-defined-outside-init
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.n_dofs = getattr(obj, "n_dofs", None)
        self.pos_slice = getattr(obj, "pos_slice", None)
        self.vel_slice = getattr(obj, "vel_slice", None)

    @property
    def q(self):
        """Get position."""
        return self[self.pos_slice]

    @property
    def qd(self):
        """Get velocity."""
        return self[self.vel_slice]


class LimbRepoState(JointState):
    """Limb Repositioning State.

    This is a subclass of np.ndarray, and allows access to active and
    passive kinematic states as properties.
    """

    # pylint: disable=attribute-defined-outside-init
    def __new__(
        cls, input_array: np.ndarray, active_n_dofs: int = 6, passive_n_dofs: int = 6
    ) -> LimbRepoState:
        assert len(input_array) == 2 * (active_n_dofs + passive_n_dofs)

        obj = np.asarray(input_array).view(cls)
        obj.active_n_dofs = active_n_dofs
        obj.passive_n_dofs = passive_n_dofs

        obj.active_slice = slice(0, 2 * obj.active_n_dofs)
        obj.active_q_slice = slice(0, obj.active_n_dofs)
        obj.active_qd_slice = slice(obj.active_n_dofs, 2 * obj.active_n_dofs)

        obj.passive_slice = slice(
            2 * obj.active_n_dofs, 2 * obj.active_n_dofs + 2 * obj.passive_n_dofs
        )
        obj.passive_q_slice = slice(
            2 * obj.active_n_dofs, 2 * obj.active_n_dofs + obj.passive_n_dofs
        )
        obj.passive_qd_slice = slice(
            2 * obj.active_n_dofs + obj.passive_n_dofs,
            2 * obj.active_n_dofs + 2 * obj.passive_n_dofs,
        )

        return obj

    # pylint: disable=attribute-defined-outside-init
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.active_n_dofs = getattr(obj, "active_n_dofs", None)
        self.passive_n_dofs = getattr(obj, "passive_n_dofs", None)
        self.active_slice = getattr(obj, "active_slice", None)
        self.active_q_slice = getattr(obj, "active_q_slice", None)
        self.active_qd_slice = getattr(obj, "active_qd_slice", None)
        self.passive_slice = getattr(obj, "passive_slice", None)
        self.passive_q_slice = getattr(obj, "passive_q_slice", None)
        self.passive_qd_slice = getattr(obj, "passive_qd_slice", None)

    @property
    def active(self):
        """Get active pos and vel."""
        return BodyState(self[self.active_slice])

    @property
    def active_q(self):
        """Get active position."""
        return self[self.active_q_slice]

    @property
    def active_qd(self):
        """Get active velocity."""
        return self[self.active_qd_slice]

    @property
    def passive(self):
        """Get passive pos and vel."""
        return BodyState(self[self.passive_slice])

    @property
    def passive_q(self):
        """Get passive position."""
        return self[self.passive_q_slice]

    @property
    def passive_qd(self):
        """Get passive velocity."""
        return self[self.passive_qd_slice]

    # @property
    # def active_slice(self):
    #     return self.active_slice

    # @property
    # def active_q_slice(self):
    #     return self.active_q_slice

    # @property
    # def active_qd_slice(self):
    #     return self.active_qd_slice

    # @property
    # def passive_slice(self):
    #     return self.passive_slice

    # @property
    # def passive_q_slice(self):
    #     return self.passive_q_slice

    # @property
    # def passive_qd_slice(self):
    #     return self.passive_qd_slice


@dataclass
class LimbRepoEEState:
    """Limb Repositioning End Effector State."""

    active_ee_pos: np.ndarray
    active_ee_vel: np.ndarray
    active_ee_orn: np.ndarray
    passive_ee_pos: np.ndarray
    passive_ee_vel: np.ndarray
    passive_ee_orn: np.ndarray
