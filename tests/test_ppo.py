import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Normal

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


@pytest.mark.parametrize(
    "ablation",
    [
        {"normalize_advantage": False},
        {"orthogonal_init": False},
        {"max_grad_norm": None},
        {"clip_objective": False},
        {"bootstrap_truncation": False},
        {"squash_actions": True},
    ],
    ids=lambda a: next(iter(a)),
)
def test_ablations_train(small_envs, ablation):
    model = PPO(small_envs, PPOConfig(n_steps=32, batch_size=16, n_epochs=1, **ablation), seed=0)
    model.learn(64, progress_bar=False)
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())


@pytest.mark.parametrize(
    "ablation",
    [
        {"normalize_advantage": False},
        {"orthogonal_init": False},
        {"max_grad_norm": None},
        {"clip_objective": False},
        {"bootstrap_truncation": False},
        {"squash_actions": True},
    ],
    ids=lambda a: next(iter(a)),
)
def test_each_ablation_changes_training(ablation):
    def trained_parameters(**overrides):
        envs = gym.vector.SyncVectorEnv(
            [lambda: RacingEnv(max_episode_steps=40) for _ in range(2)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        settings = {"n_steps": 64, "batch_size": 32, "n_epochs": 4, "max_grad_norm": 0.05, **overrides}
        model = PPO(envs, PPOConfig(**settings), seed=0)
        model.learn(256, progress_bar=False)
        envs.close()
        return torch.cat([p.detach().flatten() for p in model.policy.parameters()])

    assert not torch.allclose(trained_parameters(), trained_parameters(**ablation))


def test_squashed_log_prob_matches_change_of_variables():
    network = ActorCritic(observation_dim=12, action_dim=2, squash_actions=True)
    dist = network.distribution(torch.randn(64, 12))
    raw = dist.sample()

    reference_dist = Normal(dist.loc.double(), dist.scale.double())
    log_det = torch.log(1 - torch.tanh(raw.double()) ** 2)
    reference = reference_dist.log_prob(raw.double()).sum(-1) - log_det.sum(-1)
    torch.testing.assert_close(network.log_prob(dist, raw).double(), reference, atol=1e-4, rtol=1e-4)


def test_squashed_policy_round_trips_and_stays_in_bounds(small_envs, tmp_path):
    model = PPO(small_envs, PPOConfig(n_steps=32, batch_size=16, n_epochs=1, squash_actions=True), seed=0)
    model.learn(64, progress_bar=False)
    model.save(tmp_path / "model.pt")
    restored = PPOPolicy.load(tmp_path / "model.pt")
    assert restored.network.squash_actions

    observations = np.random.default_rng(0).uniform(-1, 1, (50, 12)).astype(np.float32)
    for observation in observations:
        np.testing.assert_allclose(restored.predict(observation), model.predict(observation))
        sampled = restored.predict(observation, deterministic=False)
        assert np.all(np.abs(sampled) < 1)
