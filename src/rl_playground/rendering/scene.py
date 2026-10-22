"""Drawing of the track, vehicles, trails and sensor rays."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pygame

from rl_playground.envs.track import Track
from rl_playground.rendering import theme

GRID_SPACING = 40
VEHICLE_HALF_LENGTH = 12.0
VEHICLE_HALF_WIDTH = 6.0


@dataclass
class CarView:
    position: np.ndarray
    heading: float
    color: theme.Color
    trail: Sequence[tuple[float, float]] = field(default_factory=list)
    sensor_endpoints: np.ndarray | None = None
    crashed: bool = False


def render_background(track: Track, size: tuple[int, int]) -> pygame.Surface:
    """Static layer: grid, asphalt, edges, centerline and start line."""
    surface = pygame.Surface(size)
    surface.fill(theme.BACKGROUND)
    width, height = size
    for x in range(0, width, GRID_SPACING):
        pygame.draw.line(surface, theme.GRID, (x, 0), (x, height))
    for y in range(0, height, GRID_SPACING):
        pygame.draw.line(surface, theme.GRID, (0, y), (width, y))

    asphalt = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.polygon(asphalt, theme.TRACK_SURFACE, track.outer_edge.tolist())
    pygame.draw.polygon(asphalt, (0, 0, 0, 0), track.inner_edge.tolist())
    surface.blit(asphalt, (0, 0))

    for edge in (track.outer_edge, track.inner_edge):
        pygame.draw.aalines(surface, theme.TRACK_EDGE, True, edge.tolist())

    centerline = track.centerline
    for i in range(0, len(centerline), 4):
        pygame.draw.aaline(surface, theme.TRACK_CENTERLINE, centerline[i], centerline[(i + 2) % len(centerline)])

    pygame.draw.line(surface, theme.TEXT_MUTED, track.outer_edge[0], track.inner_edge[0], 2)
    return surface


def draw_cars(surface: pygame.Surface, overlay: pygame.Surface, cars: Sequence[CarView], show_sensors: bool) -> None:
    overlay.fill((0, 0, 0, 0))
    for car in cars:
        trail = car.trail
        for i in range(1, len(trail)):
            alpha = round(140 * i / len(trail))
            pygame.draw.line(overlay, (*car.color, alpha), trail[i - 1], trail[i], 2)
        if show_sensors and car.sensor_endpoints is not None and not car.crashed:
            origin = (float(car.position[0]), float(car.position[1]))
            for end in car.sensor_endpoints:
                pygame.draw.line(overlay, (*car.color, 50), origin, (float(end[0]), float(end[1])), 1)
                pygame.draw.circle(overlay, (*car.color, 190), (round(end[0]), round(end[1])), 2)
    surface.blit(overlay, (0, 0))
    for car in cars:
        _draw_vehicle(surface, car)


def _draw_vehicle(surface: pygame.Surface, car: CarView) -> None:
    forward = np.array([math.cos(car.heading), math.sin(car.heading)])
    side = np.array([-forward[1], forward[0]])
    center = np.asarray(car.position, dtype=float)
    corners = [
        center + forward * VEHICLE_HALF_LENGTH * a + side * VEHICLE_HALF_WIDTH * b
        for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ]
    points = [(round(p[0]), round(p[1])) for p in corners]

    if car.crashed:
        color = theme.blend(car.color, theme.TEXT_FAINT, 0.6)
        pygame.draw.polygon(surface, color, points, 2)
        return

    pygame.draw.polygon(surface, car.color, points)
    pygame.draw.aalines(surface, car.color, True, points)
    nose = center + forward * (VEHICLE_HALF_LENGTH - 3)
    pygame.draw.line(
        surface,
        theme.blend(car.color, (255, 255, 255), 0.55),
        nose + side * (VEHICLE_HALF_WIDTH - 2),
        nose - side * (VEHICLE_HALF_WIDTH - 2),
        2,
    )
