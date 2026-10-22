"""Policies that map observations to actions: trained PPO models and keyboard control."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import numpy as np

MODEL_FILENAME = "model.zip"
_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)_steps")


class Policy(Protocol):
    def predict(self, observation: np.ndarray) -> np.ndarray: ...


class TrainedPolicy:
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
        return f"{steps / 1_000_000:.1f}M".replace(".0M", "M")
    return f"{steps // 1000}k"


def resolve_model_path(path: str | Path) -> Path:
    """Accept a run directory, a checkpoint file, or a model path without the ``.zip`` suffix."""
    path = Path(path)
    if path.is_dir():
        path = path / MODEL_FILENAME
    elif path.suffix != ".zip":
        path = path.with_suffix(".zip")
    if not path.is_file():
        raise FileNotFoundError(f"No model found at {path}")
    return path


def describe_model(path: Path) -> str:
    """Human-readable label, e.g. ``baseline`` or ``baseline @ 200k``."""
    match = _CHECKPOINT_PATTERN.fullmatch(path.stem)
    if match and path.parent.name == "checkpoints":
        return f"{path.parent.parent.name} @ {format_steps(int(match.group(1)))}"
    if path.name == MODEL_FILENAME:
        return path.parent.name
    return path.stem


def load_policy(path: str | Path) -> tuple[TrainedPolicy, str]:
    from stable_baselines3 import PPO

    resolved = resolve_model_path(path)
    return TrainedPolicy(PPO.load(resolved, device="cpu")), describe_model(resolved)
