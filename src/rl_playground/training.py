"""PPO training with checkpoints, TensorBoard logging and racing-specific metrics.

Two implementations share one configuration so they can be compared directly:
``sb3`` uses Stable-Baselines3, ``scratch`` uses :mod:`rl_playground.algorithms.ppo`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rl_playground.envs.racing import RacingEnv

ALGORITHMS = ("sb3", "scratch")
MODEL_FILENAMES = {"sb3": "model.zip", "scratch": "model.pt"}


@dataclass
class TrainConfig:
    run_name: str
    algorithm: str = "sb3"
    track_mode: str = "fixed"
    total_timesteps: int = 1_000_000
    num_envs: int = 8
    seed: int = 0
    checkpoint_interval: int = 50_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (64, 64)

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {self.algorithm!r}")


def train(config: TrainConfig, runs_dir: Path = Path("runs"), overwrite: bool = False) -> Path:
    """Train a PPO agent and return the run directory.

    Layout of the run directory::

        config.json         training configuration
        model.zip|.pt       final model (.zip for sb3, .pt for scratch)
        checkpoints/        intermediate models every ``checkpoint_interval`` steps
        tensorboard/        training curves
    """
    run_dir = runs_dir / config.run_name
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} already exists; use --overwrite to replace it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    if config.algorithm == "sb3":
        _train_sb3(config, run_dir)
    else:
        _train_scratch(config, run_dir)
    return run_dir


def _train_sb3(config: TrainConfig, run_dir: Path) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    class RacingMetricsCallback(BaseCallback):
        """Logs mean laps and outcome rates of completed episodes each rollout."""

        def __init__(self) -> None:
            super().__init__()
            self._laps: list[float] = []
            self._outcomes: list[str] = []

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self._laps.append(info["laps"])
                    self._outcomes.append(info["outcome"])
            return True

        def _on_rollout_end(self) -> None:
            if not self._laps:
                return
            outcomes = np.array(self._outcomes)
            self.logger.record("racing/laps_mean", float(np.mean(self._laps)))
            self.logger.record("racing/crash_rate", float(np.mean(outcomes == "crashed")))
            self.logger.record("racing/completion_rate", float(np.mean(outcomes == "time_limit")))
            self._laps.clear()
            self._outcomes.clear()

    env = make_vec_env(
        RacingEnv,
        n_envs=config.num_envs,
        seed=config.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"track_mode": config.track_mode},
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs={"net_arch": list(config.net_arch)},
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=config.seed,
        device="cpu",
        verbose=0,
    )
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(config.checkpoint_interval // config.num_envs, 1),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="model",
            ),
            RacingMetricsCallback(),
        ]
    )
    try:
        model.learn(total_timesteps=config.total_timesteps, callback=callbacks, tb_log_name="ppo", progress_bar=True)
        model.save(run_dir / MODEL_FILENAMES["sb3"])
    finally:
        env.close()


def _train_scratch(config: TrainConfig, run_dir: Path) -> None:
    import gymnasium as gym

    from rl_playground.algorithms.ppo import PPO, PPOConfig

    ppo_config = PPOConfig(
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        net_arch=tuple(config.net_arch),
    )
    track_mode = config.track_mode
    envs = gym.vector.AsyncVectorEnv(
        [lambda: RacingEnv(track_mode=track_mode) for _ in range(config.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        model = PPO(envs, ppo_config, seed=config.seed)
        model.learn(
            config.total_timesteps,
            log_dir=run_dir / "tensorboard" / "ppo_1",
            checkpoint_dir=run_dir / "checkpoints",
            checkpoint_interval=config.checkpoint_interval,
        )
        model.save(run_dir / MODEL_FILENAMES["scratch"])
    finally:
        envs.close()
