import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import rl_playground.envs  # noqa: F401
from rl_playground.envs import RacingEnv, Track


def test_passes_gymnasium_api_checks():
    check_env(RacingEnv(), skip_render_check=True)


def test_is_registered():
    env = gym.make("rl_playground/Racing-v0", track_mode="random")
    observation, _ = env.reset(seed=0)
    assert env.observation_space.contains(observation)


def test_rejects_unknown_track_mode():
    with pytest.raises(ValueError):
        RacingEnv(track_mode="oval")


def test_reset_is_deterministic_for_a_seed():
    a, _ = RacingEnv(track_mode="random").reset(seed=3)
    b, _ = RacingEnv(track_mode="random").reset(seed=3)
    np.testing.assert_array_equal(a, b)


def test_driving_forward_makes_progress():
    env = RacingEnv(random_start=False)
    env.reset(seed=0)
    total = 0.0
    for _ in range(10):
        _, reward, terminated, truncated, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
        total += reward
    assert env.laps > 0
    assert total > 0
    assert not (terminated or truncated)


def test_leaving_the_track_terminates_with_penalty():
    env = RacingEnv(random_start=False)
    env.reset(seed=0)
    for _ in range(500):
        _, reward, terminated, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        if terminated:
            break
    assert terminated
    assert info["outcome"] == "crashed"
    assert reward < -env.config.crash_penalty / 2


def test_reset_accepts_track_and_start_index():
    track = Track.random(np.random.default_rng(1))
    env = RacingEnv()
    env.reset(seed=0, options={"track": track, "start_index": 40})
    np.testing.assert_allclose(env.position, track.centerline[40])
