"""Closed-loop race tracks and the geometric queries used by the racing environment."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

CANVAS_SIZE = 800
DEFAULT_HALF_WIDTH = 42.0
MAX_TRACK_RADIUS = 330.0
MIN_TURN_RADIUS_RATIO = 1.6


class Projection(NamedTuple):
    """Closest point on the centerline to a query position."""

    distance: float
    arc_length: float
    segment: int


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _min_turn_radius(points: np.ndarray) -> float:
    prev, nxt = np.roll(points, 1, axis=0), np.roll(points, -1, axis=0)
    a, b, c = points - prev, nxt - points, nxt - prev
    twice_area = np.abs(_cross(a, b))
    product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    with np.errstate(divide="ignore"):
        return float(np.min(product / (2 * twice_area)))


def _polar_loop(radius: np.ndarray, theta: np.ndarray) -> np.ndarray:
    radius = radius * (MAX_TRACK_RADIUS / radius.max())
    center = CANVAS_SIZE / 2
    return np.stack([center + radius * np.cos(theta), center + radius * np.sin(theta)], axis=1)


class Track:
    """A closed track described by its centerline and a constant half-width."""

    def __init__(self, centerline: np.ndarray, half_width: float = DEFAULT_HALF_WIDTH, name: str = "Track"):
        self.centerline = np.asarray(centerline, dtype=np.float64)
        self.half_width = float(half_width)
        self.name = name

        self.segment_vectors = np.roll(self.centerline, -1, axis=0) - self.centerline
        self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
        self.segment_offsets = np.concatenate([[0.0], np.cumsum(self.segment_lengths)[:-1]])
        self.length = float(self.segment_lengths.sum())
        directions = self.segment_vectors / self.segment_lengths[:, None]
        self.segment_headings = np.arctan2(directions[:, 1], directions[:, 0])

        tangents = np.roll(self.centerline, -1, axis=0) - np.roll(self.centerline, 1, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        left = self.centerline + normals * self.half_width
        right = self.centerline - normals * self.half_width
        center = self.centerline.mean(axis=0)
        if np.linalg.norm(left - center, axis=1).mean() > np.linalg.norm(right - center, axis=1).mean():
            self.outer_edge, self.inner_edge = left, right
        else:
            self.outer_edge, self.inner_edge = right, left

        edges = (self.outer_edge, self.inner_edge)
        self._wall_starts = np.concatenate(edges)
        self._wall_vectors = np.concatenate([np.roll(edge, -1, axis=0) for edge in edges]) - self._wall_starts

    def __len__(self) -> int:
        return len(self.centerline)

    @classmethod
    def default(cls, num_points: int = 240, half_width: float = DEFAULT_HALF_WIDTH) -> Track:
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        radius = 1 + 0.18 * np.cos(2 * theta) + 0.10 * np.cos(3 * theta + 1.0) + 0.05 * np.cos(5 * theta + 2.0)
        return cls(_polar_loop(radius, theta), half_width, name="Default track")

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        num_points: int = 240,
        half_width: float = DEFAULT_HALF_WIDTH,
        name: str = "Random track",
        max_attempts: int = 100,
    ) -> Track:
        """Sample a smooth loop as a circle perturbed by random low-frequency harmonics.

        Samples whose tightest corner is too sharp for the track width are rejected.
        """
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        for _ in range(max_attempts):
            radius = np.ones_like(theta)
            for harmonic in range(2, 6):
                amplitude = rng.uniform(0, 0.22 / (harmonic - 1))
                radius += amplitude * np.cos(harmonic * theta + rng.uniform(0, 2 * np.pi))
            points = _polar_loop(radius, theta)
            if _min_turn_radius(points) > MIN_TURN_RADIUS_RATIO * half_width:
                return cls(points, half_width, name=name)
        raise RuntimeError(f"No valid track found in {max_attempts} attempts")

    def heading_at(self, segment: int) -> float:
        return float(self.segment_headings[segment])

    def project(self, position: np.ndarray) -> Projection:
        relative = position - self.centerline
        t = np.clip((relative * self.segment_vectors).sum(axis=1) / self.segment_lengths**2, 0.0, 1.0)
        closest = self.centerline + t[:, None] * self.segment_vectors
        distances = np.linalg.norm(position - closest, axis=1)
        segment = int(np.argmin(distances))
        arc_length = self.segment_offsets[segment] + t[segment] * self.segment_lengths[segment]
        return Projection(float(distances[segment]), float(arc_length), segment)

    def wrap_arc_delta(self, delta: float) -> float:
        """Map a change in arc length into [-length / 2, length / 2] to handle the start line."""
        half = self.length / 2
        return (delta + half) % self.length - half

    def raycast(self, origin: np.ndarray, directions: np.ndarray, max_distance: float) -> np.ndarray:
        """Distance from ``origin`` along each unit direction to the nearest wall, capped at ``max_distance``."""
        to_wall = self._wall_starts - origin
        denom = _cross(directions[:, None, :], self._wall_vectors[None, :, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            along_ray = _cross(to_wall, self._wall_vectors)[None, :] / denom
            along_wall = _cross(to_wall[None, :, :], directions[:, None, :]) / denom
        hit = (np.abs(denom) > 1e-9) & (along_ray >= 0) & (along_wall >= 0) & (along_wall <= 1)
        return np.minimum(np.where(hit, along_ray, np.inf).min(axis=1), max_distance)
