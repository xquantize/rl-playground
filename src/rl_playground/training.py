"""PPO training with checkpoints, TensorBoard logging and racing-specific metrics."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl_playground.envs.racing import RacingEnv
from rl_playground.policies import MODEL_FILENAME


@dataclass
class TrainConfig:
    run_name: str
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
    net_arch: tuple[int, ...] = (64, 64)


class RacingMetricsCallback(BaseCallback):
    """Logs mean laps and outcome rates of completed episodes to TensorBoard each rollout."""

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


def train(config: TrainConfig, runs_dir: Path = Path("runs"), overwrite: bool = False) -> Path:
    """Train a PPO agent and return the run directory.

    Layout of the run directory::

        config.json         training configuration
        model.zip           final model
        checkpoints/        intermediate models every ``checkpoint_interval`` steps
        tensorboard/        training curves
    """
    run_dir = runs_dir / config.run_name
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run {run_dir} already exists; pass overwrite=True to replace it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

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
        model.save(run_dir / MODEL_FILENAME)
    finally:
        env.close()
    return run_dir
