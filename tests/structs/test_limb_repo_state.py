"""Testing LimbRepoState."""

import numpy as np

from limb_repo.structs import LimbRepoState


def test_():
    """Test if subclassing np arrays works as intended."""
    np_state = np.arange(36)
    state = LimbRepoState(np_state, active_n_dofs=6, passive_n_dofs=6)

    assert np.allclose(state.active_kinematics, np_state[: 3 * state.active_n_dofs])
    assert np.allclose(state.active_q * 10, np_state[: state.active_n_dofs] * 10)
    assert isinstance(state.active_q, LimbRepoState)
    assert isinstance(state.passive_qdd + 3, LimbRepoState)
    assert isinstance((state.passive_qdd + 3).passive_qdd, LimbRepoState)
    assert isinstance(state.passive_q + np.array([1, 2, 3, 4, 5, 6]), LimbRepoState)
