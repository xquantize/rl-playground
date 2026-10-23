"""Proximal Policy Optimization (Schulman et al., 2017) for continuous action spaces.

A self-contained PyTorch implementation that mirrors the defaults of Stable-Baselines3's PPO so the two can be
compared like for like: separate actor and critic MLPs with tanh activations and orthogonal initialisation, a
diagonal Gaussian policy with a state-independent log standard deviation, generalised advantage estimation,
per-minibatch advantage normalisation, the clipped surrogate objective, and value bootstrapping on truncation.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

EPISODE_WINDOW = 100


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float | None = 0.5
    net_arch: tuple[int, ...] = (64, 64)
    normalize_advantage: bool = True
    orthogonal_init: bool = True
    clip_objective: bool = True
    bootstrap_truncation: bool = True
    squash_actions: bool = False


def _linear(input_dim: int, output_dim: int, gain: float, orthogonal: bool) -> nn.Linear:
    layer = nn.Linear(input_dim, output_dim)
    if orthogonal:
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
    return layer


def _mlp(
    input_dim: int, hidden: Iterable[int], output_dim: int, output_gain: float, orthogonal: bool = True
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers += [_linear(input_dim, width, np.sqrt(2), orthogonal), nn.Tanh()]
        input_dim = width
    return nn.Sequential(*layers, _linear(input_dim, output_dim, output_gain, orthogonal))


def _tanh_log_det(raw_actions: torch.Tensor) -> torch.Tensor:
    """``log(1 - tanh(u)^2)`` computed stably as ``2 * (log 2 - u - softplus(-2u))``."""
    return 2 * (np.log(2) - raw_actions - nn.functional.softplus(-2 * raw_actions))


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs with a diagonal Gaussian policy.

    With ``squash_actions`` the Gaussian sample ``u`` is passed through ``tanh`` so actions lie in (-1, 1) by
    construction, and log-probabilities include the change-of-variables correction. Without it, samples are
    clipped to the action bounds by the caller, as in Stable-Baselines3. Raw (pre-squash) samples are what the
    rollout buffer stores.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        net_arch: Iterable[int] = (64, 64),
        orthogonal_init: bool = True,
        squash_actions: bool = False,
    ):
        super().__init__()
        net_arch = tuple(net_arch)
        self.squash_actions = squash_actions
        self.actor = _mlp(observation_dim, net_arch, action_dim, output_gain=0.01, orthogonal=orthogonal_init)
        self.critic = _mlp(observation_dim, net_arch, 1, output_gain=1.0, orthogonal=orthogonal_init)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, observations: torch.Tensor) -> Normal:
        """Gaussian over raw, pre-squash actions."""
        return Normal(self.actor(observations), self.log_std.exp())

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)

    def log_prob(self, dist: Normal, raw_actions: torch.Tensor) -> torch.Tensor:
        log_prob = dist.log_prob(raw_actions).sum(-1)
        if self.squash_actions:
            log_prob = log_prob - _tanh_log_det(raw_actions).sum(-1)
        return log_prob

    def to_action(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_actions) if self.squash_actions else raw_actions

    def evaluate(self, observations: torch.Tensor, raw_actions: torch.Tensor):
        """Log-probability, entropy and value for a batch of observation-action pairs.

        The entropy of a squashed Gaussian has no closed form, so its single-sample estimate ``-log_prob`` is used.
        """
        dist = self.distribution(observations)
        log_prob = self.log_prob(dist, raw_actions)
        entropy = -log_prob if self.squash_actions else dist.entropy().sum(-1)
        return log_prob, entropy, self.value(observations)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over a rollout of shape ``(steps, envs)``.

    ``dones[t]`` marks that the episode ended at step ``t``, so no value is bootstrapped across it.
    Returns advantages and the corresponding value targets.
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(last_values)
    next_values = last_values
    for t in reversed(range(rewards.shape[0])):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * not_done - values[t]
        last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
        advantages[t] = last_advantage
        next_values = values[t]
    return advantages, advantages + values


class PPO:
    """Train an :class:`ActorCritic` on a Gymnasium vector environment using same-step autoreset."""

    def __init__(self, envs: gym.vector.VectorEnv, config: PPOConfig | None = None, seed: int = 0):
        self.envs = envs
        self.config = config or PPOConfig()
        self.seed = seed
        self.num_envs = envs.num_envs
        self.observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_low = envs.single_action_space.low
        self.action_high = envs.single_action_space.high

        torch.manual_seed(seed)
        if self.config.squash_actions and not (np.all(self.action_low == -1) and np.all(self.action_high == 1)):
            raise ValueError("squash_actions requires an action space bounded by [-1, 1]")
        self.policy = ActorCritic(
            self.observation_dim,
            self.action_dim,
            self.config.net_arch,
            self.config.orthogonal_init,
            self.config.squash_actions,
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate, eps=1e-5)
        self.num_timesteps = 0

    def learn(
        self,
        total_timesteps: int,
        log_dir: Path | None = None,
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 50_000,
        progress_bar: bool = True,
    ) -> PPO:
        cfg = self.config
        n_steps, n_envs = cfg.n_steps, self.num_envs
        shape = (n_steps, n_envs)
        observations = torch.zeros((*shape, self.observation_dim))
        actions = torch.zeros((*shape, self.action_dim))
        log_probs, rewards, dones, values = (torch.zeros(shape) for _ in range(4))

        writer = SummaryWriter(str(log_dir)) if log_dir is not None else None
        episode_returns: deque[float] = deque(maxlen=EPISODE_WINDOW)
        episode_lengths: deque[int] = deque(maxlen=EPISODE_WINDOW)
        running_return = np.zeros(n_envs)
        running_length = np.zeros(n_envs, dtype=int)
        finished_laps: list[float] = []
        finished_outcomes: list[str] = []
        next_checkpoint = checkpoint_interval

        observation, _ = self.envs.reset(seed=self.seed)
        observation = torch.as_tensor(observation, dtype=torch.float32)
        rollout_size = n_steps * n_envs
        planned_timesteps = -(-total_timesteps // rollout_size) * rollout_size
        progress = tqdm(total=planned_timesteps, unit="step", disable=not progress_bar, dynamic_ncols=True)
        start_time = time.perf_counter()

        while self.num_timesteps < total_timesteps:
            for step in range(n_steps):
                with torch.no_grad():
                    dist = self.policy.distribution(observation)
                    action = dist.sample()
                    log_probs[step] = self.policy.log_prob(dist, action)
                    values[step] = self.policy.value(observation)
                observations[step], actions[step] = observation, action

                env_action = np.clip(self.policy.to_action(action).numpy(), self.action_low, self.action_high)
                next_observation, reward, terminated, truncated, info = self.envs.step(env_action)
                running_return += reward
                running_length += 1

                reward = torch.as_tensor(reward, dtype=torch.float32)
                bootstrap = truncated & ~terminated
                if cfg.bootstrap_truncation and bootstrap.any():
                    final = np.stack(info["final_obs"][bootstrap])
                    with torch.no_grad():
                        reward[bootstrap] += cfg.gamma * self.policy.value(torch.as_tensor(final, dtype=torch.float32))
                rewards[step] = reward
                dones[step] = torch.as_tensor(terminated | truncated, dtype=torch.float32)

                for i in np.flatnonzero(terminated | truncated):
                    episode_returns.append(float(running_return[i]))
                    episode_lengths.append(int(running_length[i]))
                    finished_laps.append(float(info["final_info"]["laps"][i]))
                    finished_outcomes.append(str(info["final_info"]["outcome"][i]))
                    running_return[i], running_length[i] = 0.0, 0

                observation = torch.as_tensor(next_observation, dtype=torch.float32)
                self.num_timesteps += n_envs
                progress.update(n_envs)

            with torch.no_grad():
                last_values = self.policy.value(observation)
            advantages, returns = compute_gae(rewards, values, dones, last_values, cfg.gamma, cfg.gae_lambda)
            stats = self._update(
                observations.reshape(-1, self.observation_dim),
                actions.reshape(-1, self.action_dim),
                log_probs.reshape(-1),
                advantages.reshape(-1),
                returns.reshape(-1),
                values.reshape(-1),
            )

            if writer is not None:
                elapsed = time.perf_counter() - start_time
                scalars = {**stats, "time/fps": self.num_timesteps / elapsed}
                if episode_returns:
                    scalars["rollout/ep_rew_mean"] = float(np.mean(episode_returns))
                    scalars["rollout/ep_len_mean"] = float(np.mean(episode_lengths))
                if finished_laps:
                    outcomes = np.array(finished_outcomes)
                    scalars["racing/laps_mean"] = float(np.mean(finished_laps))
                    scalars["racing/crash_rate"] = float(np.mean(outcomes == "crashed"))
                    scalars["racing/completion_rate"] = float(np.mean(outcomes == "time_limit"))
                for tag, value in scalars.items():
                    writer.add_scalar(tag, value, self.num_timesteps)
            finished_laps.clear()
            finished_outcomes.clear()

            if checkpoint_dir is not None and self.num_timesteps >= next_checkpoint:
                self.save(checkpoint_dir / f"model_{self.num_timesteps}_steps.pt")
                next_checkpoint += checkpoint_interval

        progress.close()
        if writer is not None:
            writer.close()
        return self

    def _update(self, observations, actions, old_log_probs, advantages, returns, old_values) -> dict[str, float]:
        cfg = self.config
        batch_size = observations.shape[0]
        policy_losses, value_losses, entropy_losses, clip_fractions, approx_kls = [], [], [], [], []

        for _ in range(cfg.n_epochs):
            for indices in torch.randperm(batch_size).split(cfg.batch_size):
                log_prob, entropy, value = self.policy.evaluate(observations[indices], actions[indices])
                advantage = advantages[indices]
                if cfg.normalize_advantage and len(indices) > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                log_ratio = log_prob - old_log_probs[indices]
                ratio = log_ratio.exp()
                clipped = ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range)
                if cfg.clip_objective:
                    policy_loss = -torch.min(advantage * ratio, advantage * clipped).mean()
                else:
                    policy_loss = -(advantage * ratio).mean()
                value_loss = nn.functional.mse_loss(returns[indices], value)
                entropy_loss = -entropy.mean()
                loss = policy_loss + cfg.ent_coef * entropy_loss + cfg.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                if cfg.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    clip_fractions.append(((ratio - 1).abs() > cfg.clip_range).float().mean().item())
                    approx_kls.append(((ratio - 1) - log_ratio).mean().item())

        variance = returns.var()
        explained_variance = 1 - (returns - old_values).var() / variance if variance > 0 else torch.tensor(0.0)
        return {
            "train/policy_gradient_loss": float(np.mean(policy_losses)),
            "train/value_loss": float(np.mean(value_losses)),
            "train/entropy_loss": float(np.mean(entropy_losses)),
            "train/clip_fraction": float(np.mean(clip_fractions)),
            "train/approx_kl": float(np.mean(approx_kls)),
            "train/explained_variance": float(explained_variance),
            "train/std": self.policy.log_std.detach().exp().mean().item(),
        }

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return _predict(self.policy, observation, deterministic, self.action_low, self.action_high)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "num_timesteps": self.num_timesteps,
                "state_dict": self.policy.state_dict(),
            },
            path,
        )


@dataclass
class PPOPolicy:
    """Inference-only policy restored from a saved :class:`PPO` model."""

    network: ActorCritic
    action_low: np.ndarray
    action_high: np.ndarray
    num_timesteps: int = 0
    config: dict = field(default_factory=dict)
    deterministic: bool = True

    @classmethod
    def load(cls, path: Path, deterministic: bool = True) -> PPOPolicy:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        network = ActorCritic(
            checkpoint["observation_dim"],
            checkpoint["action_dim"],
            config["net_arch"],
            squash_actions=config.get("squash_actions", False),
        )
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()
        return cls(
            network,
            np.asarray(checkpoint["action_low"]),
            np.asarray(checkpoint["action_high"]),
            checkpoint["num_timesteps"],
            checkpoint["config"],
            deterministic,
        )

    def predict(self, observation: np.ndarray, deterministic: bool | None = None) -> np.ndarray:
        deterministic = self.deterministic if deterministic is None else deterministic
        return _predict(self.network, observation, deterministic, self.action_low, self.action_high)


def _predict(network: ActorCritic, observation, deterministic: bool, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Mean action when deterministic, otherwise a sample; squashed if the network squashes, then clipped."""
    with torch.no_grad():
        dist = network.distribution(torch.as_tensor(observation, dtype=torch.float32))
        raw = dist.mean if deterministic else dist.sample()
        return np.clip(network.to_action(raw).numpy(), low, high)
