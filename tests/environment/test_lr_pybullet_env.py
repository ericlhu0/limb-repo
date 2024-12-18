"""Test Limb Repo PyBullet Environment."""

import numpy as np
import omegaconf

from limb_repo.environments.lr_pybullet_env import LRPyBulletEnv


def test_config_parsing():
    """Test if config parsing does not raise errors."""
    config_dict = LRPyBulletEnv.parse_config("assets/configs/test_env_config.yaml")

    assert isinstance(config_dict.pybullet_config, omegaconf.DictConfig)


def test_last_state_tracking():
    """Test if state updates keep track of last state correctly."""
    config_dict = LRPyBulletEnv.parse_config("assets/configs/test_env_config.yaml")
    config_dict.pybullet_config.use_gui = False

    env = LRPyBulletEnv(config=config_dict)

    active_state = env.get_body_state(env.active_id)
    passive_state = env.get_body_state(env.passive_id)

    print("active_state", active_state)

    # assert np.allclose(active_state, env.active_init_state)
    # assert np.allclose(passive_state, env.passive_init_state)

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

    # print("env body state active", env.get_body_state(env.active_id))
    # print("new active state", new_active_state)
    # print("env body state passive", env.get_body_state(env.passive_id))
    # print("new passive state", new_passive_state)

    assert np.allclose(env.get_body_state(env.active_id), new_active_state)
    assert np.allclose(env.get_body_state(env.passive_id), new_passive_state)

    assert np.allclose(env.prev_active_q, active_state.pos)
    assert np.allclose(env.prev_active_qd, active_state.vel)
    assert np.allclose(env.prev_passive_q, passive_state.pos)
    assert np.allclose(env.prev_passive_qd, passive_state.vel)
