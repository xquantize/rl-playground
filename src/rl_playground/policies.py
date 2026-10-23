"""Policies that map observations to actions: trained PPO models and keyboard control."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import numpy as np

MODEL_FILENAMES = ("model.zip", "model.pt")
_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)_steps")


class Policy(Protocol):
    def predict(self, observation: np.ndarray) -> np.ndarray: ...


class StableBaselinesPolicy:
    """Deterministic wrapper around a Stable-Baselines3 model."""

    def __init__(self, model):
        self.model = model

    def predict(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return action


class KeyboardPolicy:
    """Arrow keys: up/down for throttle and brake, left/right to steer."""

    def predict(self, observation: np.ndarray) -> np.ndarray:
        import pygame

        keys = pygame.key.get_pressed()
        steer = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        return np.array([steer, throttle], dtype=np.float32)


def format_steps(steps: int) -> str:
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:.1f}".removesuffix(".0") + "M"
    return f"{steps // 1000}k"


def resolve_model_path(path: str | Path) -> Path:
    """Accept a run directory, a model or checkpoint file, or a model path without its suffix."""
    path = Path(path)
    if path.is_dir():
        candidates = [path / name for name in MODEL_FILENAMES]
    elif path.suffix in (".zip", ".pt"):
        candidates = [path]
    else:
        candidates = [path.with_suffix(".zip"), path.with_suffix(".pt")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No model found at {path}")


def describe_model(path: Path) -> str:
    """Human-readable label, e.g. ``baseline`` or ``baseline @ 200k``."""
    match = _CHECKPOINT_PATTERN.fullmatch(path.stem)
    if match and path.parent.name == "checkpoints":
        return f"{path.parent.parent.name} @ {format_steps(int(match.group(1)))}"
    if path.name in MODEL_FILENAMES:
        return path.parent.name
    return path.stem


def load_policy(path: str | Path) -> tuple[Policy, str]:
    resolved = resolve_model_path(path)
    if resolved.suffix == ".pt":
        from rl_playground.algorithms.ppo import PPOPolicy

        return PPOPolicy.load(resolved), describe_model(resolved)

    from stable_baselines3 import PPO

    return StableBaselinesPolicy(PPO.load(resolved, device="cpu")), describe_model(resolved)
