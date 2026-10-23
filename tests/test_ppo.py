import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_playground.algorithms.ppo import PPO, ActorCritic, PPOConfig, PPOPolicy, compute_gae
from rl_playground.envs import RacingEnv


def test_gae_matches_hand_computation():
    gamma, lam = 0.9, 0.8
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [1.0], [1.5]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    last_values = torch.tensor([2.0])

    delta_2 = 3.0 + gamma * 2.0 - 1.5
    delta_1 = 2.0 - 1.0
    delta_0 = 1.0 + gamma * 1.0 - 0.5
    expected = torch.tensor([[delta_0 + gamma * lam * delta_1], [delta_1], [delta_2]])

    advantages, returns = compute_gae(rewards, values, dones, last_values, gamma, lam)
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected + values)


def test_actor_critic_shapes_and_initial_std():
    network = ActorCritic(observation_dim=12, action_dim=2)
    observations = torch.zeros(5, 12)
    log_prob, entropy, value = network.evaluate(observations, torch.zeros(5, 2))
    assert log_prob.shape == entropy.shape == value.shape == (5,)
    torch.testing.assert_close(network.log_std.exp(), torch.ones(2))


@pytest.fixture
def small_envs():
    envs = gym.vector.SyncVectorEnv(
        [lambda: RacingEnv(max_episode_steps=40) for _ in range(2)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    yield envs
    envs.close()


def test_learn_logs_checkpoints_and_round_trips(small_envs, tmp_path):
    config = PPOConfig(n_steps=64, batch_size=32, n_epochs=2)
    model = PPO(small_envs, config, seed=0)
    model.learn(
        256, log_dir=tmp_path / "tb", checkpoint_dir=tmp_path / "ckpt", checkpoint_interval=128, progress_bar=False
    )

    assert model.num_timesteps == 256
    assert sorted(p.name for p in (tmp_path / "ckpt").iterdir()) == ["model_128_steps.pt", "model_256_steps.pt"]
    assert any((tmp_path / "tb").iterdir())

    model.save(tmp_path / "model.pt")
    restored = PPOPolicy.load(tmp_path / "model.pt")
    observation = np.random.default_rng(0).uniform(-1, 1, 12).astype(np.float32)
    np.testing.assert_allclose(restored.predict(observation), model.predict(observation))
    assert restored.num_timesteps == 256


def test_training_is_reproducible_for_a_seed(small_envs):
    def train_once():
        envs = gym.vector.SyncVectorEnv(
            [lambda: RacingEnv(max_episode_steps=40) for _ in range(2)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        model = PPO(envs, PPOConfig(n_steps=32, batch_size=16, n_epochs=1), seed=3)
        model.learn(64, progress_bar=False)
        envs.close()
        return [p.detach().clone() for p in model.policy.parameters()]

    for a, b in zip(train_once(), train_once(), strict=True):
        torch.testing.assert_close(a, b)
