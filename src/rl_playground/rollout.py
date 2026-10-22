"""Step-by-step execution of a policy in the racing environment, with per-step history."""

from __future__ import annotations

from collections import deque
from enum import StrEnum

import numpy as np

from rl_playground.envs.racing import RacingEnv
from rl_playground.envs.track import Track
from rl_playground.policies import Policy

TRAIL_LENGTH = 90


class Status(StrEnum):
    RUNNING = "Running"
    COMPLETED = "Completed"
    CRASHED = "Crashed"
    STALLED = "Stalled"


_FINAL_STATUS = {"crashed": Status.CRASHED, "stalled": Status.STALLED, "time_limit": Status.COMPLETED}


class Rollout:
    def __init__(self, policy: Policy, label: str, color: tuple[int, int, int] = (255, 255, 255)):
        self.policy = policy
        self.label = label
        self.color = color
        self.env = RacingEnv(random_start=False)
        self.trail: deque[tuple[float, float]] = deque(maxlen=TRAIL_LENGTH)
        self.reset(self.env.track)

    @property
    def done(self) -> bool:
        return self.status is not Status.RUNNING

    def reset(self, track: Track, start_index: int = 0) -> None:
        self.observation, _ = self.env.reset(seed=0, options={"track": track, "start_index": start_index})
        self.status = Status.RUNNING
        self.action = np.zeros(2, dtype=np.float32)
        self.episode_return = 0.0
        self.total_speed = 0.0
        self.trail.clear()
        self.laps_history = [0.0]
        self.return_history = [0.0]

    def step(self) -> None:
        if not self.done:
            self.action = np.clip(np.asarray(self.policy.predict(self.observation), dtype=np.float32), -1, 1)
            self.observation, reward, terminated, truncated, info = self.env.step(self.action)
            self.episode_return += reward
            self.total_speed += self.env.speed
            self.trail.append((float(self.env.position[0]), float(self.env.position[1])))
            if terminated or truncated:
                self.status = _FINAL_STATUS[info["outcome"]]
        self.laps_history.append(self.env.laps)
        self.return_history.append(self.episode_return)

    def run(self, track: Track, start_index: int = 0) -> None:
        self.reset(track, start_index)
        while not self.done:
            self.step()

    def car_view(self):
        return self.env.car_view(self.color, self.trail, crashed=self.status is Status.CRASHED)
