"""Interactive window that composes the scene with an optional side panel, and GIF recording."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pygame

from rl_playground.envs.track import CANVAS_SIZE, Track
from rl_playground.rendering.scene import CarView, draw_cars, render_background
from rl_playground.rendering.theme import Fonts

PANEL_WIDTH = 360


class Panel(Protocol):
    def draw(self, surface: pygame.Surface, rect: pygame.Rect, fonts: Fonts) -> None: ...


class Viewer:
    """Draws frames to a window, or off-screen when ``window`` is False or ``headless`` is True."""

    def __init__(
        self,
        panel_width: int = PANEL_WIDTH,
        window: bool = True,
        headless: bool = False,
        fps: int = 30,
        title: str = "RL Playground",
    ):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.size = (CANVAS_SIZE + panel_width, CANVAS_SIZE)
        self.panel_rect = pygame.Rect(CANVAS_SIZE, 0, panel_width, CANVAS_SIZE)
        self.window = window
        self.headless = headless
        if window:
            self.surface = pygame.display.set_mode(self.size)
            pygame.display.set_caption(title)
        else:
            self.surface = pygame.Surface(self.size)
        self.fps = fps
        self.fonts = Fonts()
        self.paused = False
        self.fast_forward = headless
        self._clock = pygame.time.Clock()
        self._overlay = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE), pygame.SRCALPHA)
        self._track: Track | None = None
        self._background: pygame.Surface | None = None

    def poll(self) -> set[str]:
        """Process window events. Returns requested actions: ``quit`` and ``next``."""
        actions: set[str] = set()
        if not self.window:
            return actions
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions.add("quit")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    actions.add("quit")
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_f:
                    self.fast_forward = not self.fast_forward
                elif event.key == pygame.K_n:
                    actions.add("next")
        return actions

    def draw(self, track: Track, cars: Sequence[CarView], panel: Panel | None = None, show_sensors: bool = True):
        if track is not self._track:
            self._background = render_background(track, (CANVAS_SIZE, CANVAS_SIZE))
            self._track = track
        self.surface.blit(self._background, (0, 0))
        draw_cars(self.surface, self._overlay, cars, show_sensors)
        if panel is not None and self.panel_rect.width > 0:
            panel.draw(self.surface, self.panel_rect, self.fonts)

    def present(self) -> None:
        if self.window:
            pygame.display.flip()
        if self.paused or not self.fast_forward:
            self._clock.tick(self.fps)

    def capture(self) -> np.ndarray:
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def close(self) -> None:
        pygame.quit()


class Recorder:
    """Collects frames and writes them as an optimised looping GIF."""

    def __init__(self, path: str | Path, max_seconds: float = 20, fps: int = 30, stride: int = 2, scale: float = 0.6):
        self.path = Path(path)
        self.stride = stride
        self.scale = scale
        self.frame_duration_ms = round(1000 * stride / fps)
        self.max_frames = int(max_seconds * fps / stride)
        self._frames: list[np.ndarray] = []
        self._seen = 0

    @property
    def full(self) -> bool:
        return len(self._frames) >= self.max_frames

    def add(self, frame: np.ndarray) -> None:
        if not self.full and self._seen % self.stride == 0:
            self._frames.append(frame)
        self._seen += 1

    def save(self) -> Path | None:
        if not self._frames:
            return None
        from PIL import Image

        height, width = self._frames[0].shape[:2]
        size = (round(width * self.scale), round(height * self.scale))
        images = [Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS) for frame in self._frames]
        palette = _shared_palette(images)
        images = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            self.path,
            save_all=True,
            append_images=images[1:],
            duration=self.frame_duration_ms,
            loop=0,
            optimize=True,
        )
        return self.path


def _shared_palette(images, samples: int = 8):
    """Build one palette from frames spread across the recording so colors stay stable."""
    from PIL import Image

    picks = images[:: max(1, len(images) // samples)][:samples]
    width, height = picks[0].size
    montage = Image.new("RGB", (width, height * len(picks)))
    for i, image in enumerate(picks):
        montage.paste(image, (0, i * height))
    return montage.quantize(colors=192)
