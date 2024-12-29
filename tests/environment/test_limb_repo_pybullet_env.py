"""Test Limb Repo PyBullet Environment."""

import numpy as np
import omegaconf

from limb_repo.environments.limb_repo_pybullet_env import (
    LimbRepoPyBulletConfig,
    LimbRepoPyBulletEnv,
)
from limb_repo.utils import utils


def test_config_parsing():
    """Test if config parsing does not raise errors."""
    config_dict = utils.parse_config(
        "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
    )

    assert isinstance(config_dict.pybullet_config, omegaconf.DictConfig)


# pylint: disable=protected-access
def test_last_state_tracking():
    """Test if state updates keep track of last state correctly."""
    config_dict = utils.parse_config(
        "assets/configs/test_env_config.yaml", LimbRepoPyBulletConfig
    )
    config_dict.pybullet_config.use_gui = False

    env = LimbRepoPyBulletEnv(config=config_dict)

    active_state = env.get_body_state(env.active_id)
    passive_state = env.get_body_state(env.passive_id)

    assert np.allclose(active_state, env._active_init_state)
    assert np.allclose(passive_state, env._passive_init_state)

    pos_diff = 1

    new_active_state = env.get_body_state(env.active_id)
    new_active_state[new_active_state.pos_slice] += pos_diff
    new_active_state[new_active_state.vel_slice] = pos_diff / (env.dt)
    new_active_state[new_active_state.acc_slice] = pos_diff / ((env.dt) ** 2)

    new_passive_state = env.get_body_state(env.passive_id)
    new_passive_state[new_passive_state.pos_slice] += pos_diff
    new_passive_state[new_passive_state.vel_slice] = pos_diff / (env.dt)
    new_passive_state[new_passive_state.acc_slice] = pos_diff / ((env.dt) ** 2)

    env.set_body_state(env.active_id, new_active_state, set_vel=False)
    env.set_body_state(env.passive_id, new_passive_state, set_vel=False)

    assert np.allclose(env.get_body_state(env.active_id), new_active_state)
    assert np.allclose(env.get_body_state(env.passive_id), new_passive_state)

    assert np.allclose(env._prev_active_q, active_state.q)
    assert np.allclose(env._prev_active_qd, active_state.qd)
    assert np.allclose(env._prev_passive_q, passive_state.q)
    assert np.allclose(env._prev_passive_qd, passive_state.qd)
