"""Policies that map observations to actions: trained PPO models and keyboard control."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from rl_playground.runs import describe_model, resolve_models


class Policy(Protocol):
    def predict(self, observation: np.ndarray) -> np.ndarray: ...


class StableBaselinesPolicy:
    """Wrapper around a Stable-Baselines3 model that acts on single observations."""

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def predict(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return action


class KeyboardPolicy:
    """Arrow keys: up/down for throttle and brake, left/right to steer."""

    def predict(self, observation: np.ndarray) -> np.ndarray:
        import pygame

        keys = pygame.key.get_pressed()
        steer = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        return np.array([steer, throttle], dtype=np.float32)


def load_model(model_path: Path, deterministic: bool = True) -> Policy:
    """Load a trained policy. Deterministic policies act with the mean action, stochastic ones sample."""
    if model_path.suffix == ".pt":
        from rl_playground.algorithms.ppo import PPOPolicy

        return PPOPolicy.load(model_path, deterministic=deterministic)

    from stable_baselines3 import PPO

    return StableBaselinesPolicy(PPO.load(model_path, device="cpu"), deterministic=deterministic)


def load_policy(path: str | Path) -> tuple[Policy, str]:
    """Load a single policy. For a run with several seeds, the lowest seed is used."""
    model_path = resolve_models(path)[0]
    return load_model(model_path), describe_model(model_path)
