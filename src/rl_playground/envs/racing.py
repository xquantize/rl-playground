"""Top-down racing environment in which a car learns to drive from distance sensors."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_playground.envs.track import Track

if TYPE_CHECKING:
    from rl_playground.rendering.scene import CarView

TRACK_MODES = ("fixed", "random")
PROGRESS_EPSILON = 1.0


@dataclass(frozen=True)
class VehicleConfig:
    max_speed: float = 320.0
    acceleration: float = 420.0
    drag: float = 0.6
    max_steer_rate: float = 3.2
    full_grip_speed: float = 60.0
    radius: float = 8.0


@dataclass(frozen=True)
class RacingConfig:
    track_mode: str = "fixed"
    random_start: bool = True
    num_sensors: int = 9
    sensor_fov: float = math.pi
    sensor_range: float = 260.0
    max_episode_steps: int = 1500
    stall_patience: int = 150
    progress_reward: float = 100.0
    time_penalty: float = 0.01
    crash_penalty: float = 10.0
    dt: float = 1 / 30
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)


class RacingEnv(gym.Env):
    """Drive a car around a closed track.

    Observation, shape ``(num_sensors + 3,)``:
        sensor distances scaled to [0, 1], speed scaled to [0, 1], and the sine and
        cosine of the car's heading relative to the local track direction.

    Action, shape ``(2,)``:
        steering and throttle in [-1, 1]. Negative throttle brakes.

    Reward:
        ``progress_reward`` per lap of forward progress, accrued continuously, minus
        ``time_penalty`` per step and ``crash_penalty`` on leaving the track.

    Episodes terminate on a crash and are truncated after ``max_episode_steps`` or
    ``stall_patience`` consecutive steps without progress.

    Reset options:
        ``track``: a :class:`Track` to use for this episode.
        ``start_index``: centerline index to start from.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: RacingConfig | None = None, render_mode: str | None = None, **overrides: Any):
        self.config = replace(config or RacingConfig(), **overrides)
        if self.config.track_mode not in TRACK_MODES:
            raise ValueError(f"track_mode must be one of {TRACK_MODES}, got {self.config.track_mode!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        num_sensors = self.config.num_sensors
        half_fov = self.config.sensor_fov / 2
        self.sensor_angles = np.linspace(-half_fov, half_fov, num_sensors)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(num_sensors + 3,), dtype=np.float32)

        self.track = Track.default()
        self._viewer = None
        self._place_car(0)
        self._observe()

    @property
    def laps(self) -> float:
        return self.progress / self.track.length

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "track" in options:
            self.track = options["track"]
        elif self.config.track_mode == "random":
            self.track = Track.random(self.np_random)

        if "start_index" in options:
            start = int(options["start_index"]) % len(self.track)
        elif self.config.random_start:
            start = int(self.np_random.integers(len(self.track)))
        else:
            start = 0
        self._place_car(start)

        observation = self._observe()
        if self.render_mode == "human":
            self.render()
        return observation, self._info("running")

    def step(self, action: np.ndarray):
        steer, throttle = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        vehicle, dt = self.config.vehicle, self.config.dt

        acceleration = throttle * vehicle.acceleration - vehicle.drag * self.speed
        self.speed = float(np.clip(self.speed + acceleration * dt, 0.0, vehicle.max_speed))
        grip = min(self.speed / vehicle.full_grip_speed, 1.0)
        self.heading += steer * vehicle.max_steer_rate * grip * dt
        self.position = self.position + self.speed * dt * np.array([math.cos(self.heading), math.sin(self.heading)])
        self.steps += 1

        projection = self.track.project(self.position)
        delta = self.track.wrap_arc_delta(projection.arc_length - self._arc_length)
        self._arc_length, self._segment = projection.arc_length, projection.segment
        self.progress += delta
        if self.progress > self._best_progress + PROGRESS_EPSILON:
            self._best_progress = self.progress
            self._steps_without_progress = 0
        else:
            self._steps_without_progress += 1

        crashed = projection.distance > self.track.half_width - vehicle.radius
        reward = self.config.progress_reward * delta / self.track.length - self.config.time_penalty
        if crashed:
            reward -= self.config.crash_penalty
            outcome = "crashed"
        elif self.steps >= self.config.max_episode_steps:
            outcome = "time_limit"
        elif self._steps_without_progress >= self.config.stall_patience:
            outcome = "stalled"
        else:
            outcome = "running"

        observation = self._observe()
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), crashed, outcome in ("time_limit", "stalled"), self._info(outcome)

    def sensor_endpoints(self) -> np.ndarray:
        return self.position + self._sensor_directions * (self.sensor_readings * self.config.sensor_range)[:, None]

    def car_view(self, color: tuple[int, int, int], trail=(), crashed: bool = False) -> CarView:
        from rl_playground.rendering.scene import CarView

        return CarView(self.position, self.heading, color, list(trail), self.sensor_endpoints(), crashed)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode")
            return None
        from rl_playground.rendering.theme import AGENT_COLORS
        from rl_playground.rendering.viewer import Viewer

        if self._viewer is None:
            self._viewer = Viewer(panel_width=0, window=self.render_mode == "human", fps=self.metadata["render_fps"])
        if "quit" in self._viewer.poll():
            self.close()
            raise SystemExit(0)
        self._viewer.draw(self.track, [self.car_view(AGENT_COLORS[0])])
        if self.render_mode == "human":
            self._viewer.present()
            return None
        return self._viewer.capture()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _place_car(self, index: int) -> None:
        self.position = self.track.centerline[index].copy()
        self.heading = self.track.heading_at(index)
        self.speed = 0.0
        projection = self.track.project(self.position)
        self._arc_length, self._segment = projection.arc_length, projection.segment
        self.progress = 0.0
        self._best_progress = 0.0
        self._steps_without_progress = 0
        self.steps = 0

    def _observe(self) -> np.ndarray:
        angles = self.heading + self.sensor_angles
        self._sensor_directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        distances = self.track.raycast(self.position, self._sensor_directions, self.config.sensor_range)
        self.sensor_readings = distances / self.config.sensor_range
        relative_heading = self.heading - self.track.heading_at(self._segment)
        extras = [
            self.speed / self.config.vehicle.max_speed,
            math.sin(relative_heading),
            math.cos(relative_heading),
        ]
        return np.concatenate([self.sensor_readings, extras]).astype(np.float32)

    def _info(self, outcome: str) -> dict[str, Any]:
        return {"laps": self.laps, "speed": self.speed, "outcome": outcome}
