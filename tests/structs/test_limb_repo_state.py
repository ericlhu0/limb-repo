"""Testing LimbRepoState."""

import numpy as np

from limb_repo.structs import LimbRepoState


def test_limb_repo_state():
    """Test if subclassing np arrays works as intended."""
    np_state = np.arange(24)
    state = LimbRepoState(np_state, active_n_dofs=6, passive_n_dofs=6)

    nd_state = np.array([state, state, state])

    assert np.allclose(state.active, np_state[: 2 * state.active_n_dofs])
    assert np.allclose(state.active_q * 10, np_state[: state.active_n_dofs] * 10)
    assert np.allclose(nd_state[:, state.passive_qd], np.array([state.passive_qd] * 3))
    assert isinstance(state.active_q, LimbRepoState)
    assert isinstance(state.passive_qd + 3, LimbRepoState)
    assert isinstance((state.passive_qd + 3).passive_qd, LimbRepoState)
    assert isinstance(state.passive_q + np.array([1, 2, 3, 4, 5, 6]), LimbRepoState)
