# pylint: disable=attribute-defined-outside-init
# mypy: disable-error-code="attr-defined"
"""Data structures."""

from typing import TypeAlias

import numpy as np

# Define some data structures for this example repository (to be changed).
State: TypeAlias = np.ndarray
Action: TypeAlias = np.ndarray
Goal: TypeAlias = State
Pose: TypeAlias = np.ndarray


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
    kinematics_slice: slice
    pos_slice: slice
    vel_slice: slice
    acc_slice: slice

    def __new__(cls, input_array: np.ndarray, n_dofs: int = 6) -> "BodyState":
        assert len(input_array) == 3 * n_dofs

        obj = np.asarray(input_array).view(cls)
        obj.n_dofs = n_dofs

        obj.kinematics_slice = slice(0, 3 * obj.n_dofs)
        obj.pos_slice = slice(0, obj.n_dofs)
        obj.vel_slice = slice(obj.n_dofs, 2 * obj.n_dofs)
        obj.acc_slice = slice(2 * obj.n_dofs, 3 * obj.n_dofs)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.n_dofs = getattr(obj, "n_dofs", None)
        self.kinematics_slice = getattr(obj, "kinematics_slice", None)
        self.pos_slice = getattr(obj, "pos_slice", None)
        self.vel_slice = getattr(obj, "vel_slice", None)
        self.acc_slice = getattr(obj, "acc_slice", None)

    @property
    def q(self):
        """Get position."""
        return self[self.pos_slice]

    @property
    def qd(self):
        """Get velocity."""
        return self[self.vel_slice]

    @property
    def qdd(self):
        """Get acceleration."""
        return self[self.acc_slice]


class LRState(JointState):
    """Limb Repositioning State.

    This is a subclass of np.ndarray, and allows access to active and
    passive kinematic states as properties.
    """

    def __new__(
        cls, input_array: np.ndarray, active_n_dofs: int = 6, passive_n_dofs: int = 6
    ) -> "LRState":
        assert len(input_array) == 3 * (active_n_dofs + passive_n_dofs)

        obj = np.asarray(input_array).view(cls)
        obj.active_n_dofs = active_n_dofs
        obj.passive_n_dofs = passive_n_dofs

        obj.active_kinematics_slice = slice(0, 3 * obj.active_n_dofs)
        obj.active_q_slice = slice(0, obj.active_n_dofs)
        obj.active_qd_slice = slice(obj.active_n_dofs, 2 * obj.active_n_dofs)
        obj.active_qdd_slice = slice(2 * obj.active_n_dofs, 3 * obj.active_n_dofs)

        obj.passive_kinematics_slice = slice(
            3 * obj.active_n_dofs, 3 * obj.active_n_dofs + 3 * obj.passive_n_dofs
        )
        obj.passive_q_slice = slice(
            3 * obj.active_n_dofs, 3 * obj.active_n_dofs + obj.passive_n_dofs
        )
        obj.passive_qd_slice = slice(
            3 * obj.active_n_dofs + obj.passive_n_dofs,
            3 * obj.active_n_dofs + 2 * obj.passive_n_dofs,
        )
        obj.passive_qdd_slice = slice(
            3 * obj.active_n_dofs + 2 * obj.passive_n_dofs,
            3 * obj.active_n_dofs + 3 * obj.passive_n_dofs,
        )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.active_n_dofs = getattr(obj, "active_n_dofs", None)
        self.passive_n_dofs = getattr(obj, "passive_n_dofs", None)
        self.active_kinematics_slice = getattr(obj, "active_kinematics_slice", None)
        self.active_q_slice = getattr(obj, "active_q_slice", None)
        self.active_qd_slice = getattr(obj, "active_qd_slice", None)
        self.active_qdd_slice = getattr(obj, "active_qdd_slice", None)
        self.passive_kinematics_slice = getattr(obj, "passive_kinematics_slice", None)
        self.passive_q_slice = getattr(obj, "passive_q_slice", None)
        self.passive_qd_slice = getattr(obj, "passive_qd_slice", None)
        self.passive_qdd_slice = getattr(obj, "passive_qdd_slice", None)

    @property
    def active_kinematics(self):
        """Get active kinematics."""
        return self[self.active_kinematics_slice]

    @property
    def active_q(self):
        """Get active position."""
        return self[self.active_q_slice]

    @property
    def active_qd(self):
        """Get active velocity."""
        return self[self.active_qd_slice]

    @property
    def active_qdd(self):
        """Get active acceleration."""
        return self[self.active_qdd_slice]

    @property
    def passive_kinematics(self):
        """Get passive kinematics."""
        return self[self.passive_kinematics_slice]

    @property
    def passive_q(self):
        """Get passive position."""
        return self[self.passive_q_slice]

    @property
    def passive_qd(self):
        """Get passive velocity."""
        return self[self.passive_qd_slice]

    @property
    def passive_qdd(self):
        """Get passive acceleration."""
        return self[self.passive_qdd_slice]
