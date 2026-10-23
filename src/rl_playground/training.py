"""PPO training with checkpoints, TensorBoard logging and racing-specific metrics.

Two implementations share one configuration so they can be compared directly:
``sb3`` uses Stable-Baselines3, ``scratch`` uses :mod:`rl_playground.algorithms.ppo`.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

from rl_playground.envs.racing import RacingEnv
from rl_playground.runs import seed_dir

log = logging.getLogger(__name__)

ALGORITHMS = ("sb3", "scratch")
MODEL_FILENAMES = {"sb3": "model.zip", "scratch": "model.pt"}
SCRATCH_ONLY_DEFAULTS = {"orthogonal_init": True, "clip_objective": True, "bootstrap_truncation": True}


@dataclass(frozen=True)
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
    max_grad_norm: float | None = 0.5
    net_arch: tuple[int, ...] = (64, 64)
    normalize_advantage: bool = True
    orthogonal_init: bool = True
    clip_objective: bool = True
    bootstrap_truncation: bool = True

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {self.algorithm!r}")
        if self.algorithm == "sb3":
            changed = [k for k, default in SCRATCH_ONLY_DEFAULTS.items() if getattr(self, k) != default]
            if self.max_grad_norm is None:
                changed.append("max_grad_norm")
            if changed:
                raise ValueError(f"{', '.join(changed)} can only be changed for the scratch algorithm")

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}


def train(config: TrainConfig, seeds: list[int], runs_dir: Path = Path("runs"), overwrite: bool = False) -> Path:
    """Train one agent per seed under ``runs_dir / run_name / seed_<n>`` and return the run directory."""
    run_dir = runs_dir / config.run_name
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} already exists; use --overwrite to replace it")
        shutil.rmtree(run_dir)
    for index, seed in enumerate(seeds, start=1):
        log.info("Seed %d (%d of %d)", seed, index, len(seeds))
        train_run(replace(config, seed=seed), seed_dir(run_dir, seed))
    return run_dir


def train_run(config: TrainConfig, run_dir: Path) -> Path:
    """Train a single agent into ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
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
        normalize_advantage=config.normalize_advantage,
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

    shared = {f.name for f in fields(PPOConfig)} & TrainConfig.field_names()
    ppo_config = PPOConfig(**{name: getattr(config, name) for name in shared})
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
