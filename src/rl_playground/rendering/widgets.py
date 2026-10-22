"""Small drawing primitives used to compose the side panels."""

from __future__ import annotations

from collections.abc import Sequence

import pygame

from rl_playground.rendering import theme
from rl_playground.rendering.theme import Color, Fonts

PADDING = 28
RADIUS = 6


def text(surface: pygame.Surface, font: pygame.font.Font, value: str, color: Color, **anchor) -> pygame.Rect:
    image = font.render(value, True, color)
    rect = image.get_rect(**anchor)
    surface.blit(image, rect)
    return rect


def panel_background(surface: pygame.Surface, rect: pygame.Rect) -> None:
    pygame.draw.rect(surface, theme.PANEL, rect)
    pygame.draw.line(surface, theme.PANEL_BORDER, rect.topleft, rect.bottomleft)


def header(
    surface,
    rect: pygame.Rect,
    fonts: Fonts,
    title: str,
    subtitle: str,
    status: str | None = None,
    meta: str | None = None,
) -> int:
    """Panel title block. ``status`` renders as a colored indicator, ``meta`` as muted text, both top right."""
    x = rect.x + PADDING
    text(surface, fonts.label, "RL PLAYGROUND", theme.TEXT_FAINT, topleft=(x, 28))
    text(surface, fonts.title, title, theme.TEXT, topleft=(x, 46))
    text(surface, fonts.small, subtitle, theme.TEXT_MUTED, topleft=(x, 78))
    if status is not None:
        color = theme.STATUS_COLORS[status]
        label = text(surface, fonts.small, status, color, topright=(rect.right - PADDING, 54))
        pygame.draw.circle(surface, color, (label.left - 9, label.centery), 3)
    elif meta is not None:
        text(surface, fonts.mono, meta, theme.TEXT_MUTED, topright=(rect.right - PADDING, 54))
    return 114


def section(surface, fonts: Fonts, title: str, x: int, y: int, width: int) -> int:
    label = text(surface, fonts.label, title.upper(), theme.TEXT_MUTED, topleft=(x, y))
    pygame.draw.line(surface, theme.PANEL_BORDER, (label.right + 10, label.centery), (x + width, label.centery))
    return y + 24


def stat_card(surface, fonts: Fonts, rect: pygame.Rect, label: str, value: str, unit: str = "") -> None:
    pygame.draw.rect(surface, theme.CARD, rect, border_radius=RADIUS)
    text(surface, fonts.small, label, theme.TEXT_MUTED, topleft=(rect.x + 12, rect.y + 9))
    value_rect = text(surface, fonts.value, value, theme.TEXT, topleft=(rect.x + 12, rect.y + 25))
    if unit:
        text(
            surface,
            fonts.small,
            unit,
            theme.TEXT_FAINT,
            bottomleft=(value_rect.right + 5, value_rect.bottom - 2),
        )


def signed_bar(surface, fonts: Fonts, x: int, y: int, width: int, label: str, value: float, color: Color) -> int:
    """Horizontal bar for a value in [-1, 1], filled outward from the center."""
    text(surface, fonts.body, label, theme.TEXT_MUTED, topleft=(x, y))
    text(surface, fonts.mono, f"{value:+.2f}", theme.TEXT, topright=(x + width, y + 1))
    bar = pygame.Rect(x, y + 22, width, 6)
    pygame.draw.rect(surface, theme.CARD_INSET, bar, border_radius=3)
    center = bar.centerx
    extent = round(abs(value) * bar.width / 2)
    if extent > 0:
        fill = pygame.Rect(center if value > 0 else center - extent, bar.y, extent, bar.height)
        pygame.draw.rect(surface, color, fill, border_radius=3)
    pygame.draw.line(surface, theme.TEXT_FAINT, (center, bar.y - 2), (center, bar.bottom + 1))
    return y + 40


def sensor_bars(
    surface, fonts: Fonts, x: int, y: int, width: int, readings, angles_deg, max_range: float, color: Color
) -> int:
    label_width, value_width, row = 44, 36, 15
    bar_width = width - label_width - value_width
    for reading, angle in zip(readings, angles_deg, strict=True):
        text(surface, fonts.mono, f"{angle:+4.0f}°", theme.TEXT_FAINT, topleft=(x, y))
        bar = pygame.Rect(x + label_width, y + 5, bar_width, 4)
        pygame.draw.rect(surface, theme.CARD_INSET, bar, border_radius=2)
        filled = round(bar_width * float(reading))
        if filled > 0:
            fill_color = theme.blend(theme.NEGATIVE, color, float(reading) * 1.6)
            pygame.draw.rect(surface, fill_color, (bar.x, bar.y, filled, bar.height), border_radius=2)
        text(surface, fonts.mono, f"{reading * max_range:3.0f}", theme.TEXT_MUTED, topright=(x + width, y))
        y += row
    return y


def line_chart(
    surface, fonts: Fonts, rect: pygame.Rect, series: Sequence[tuple[Sequence[float], Color]], x_span: int
) -> None:
    pygame.draw.rect(surface, theme.CARD, rect, border_radius=RADIUS)
    plot = rect.inflate(-28, -28)
    values = [v for data, _ in series for v in data]
    low, high = min([0.0, *values]), max([1.0, *values])

    for k in range(5):
        y = plot.bottom - plot.height * k / 4
        pygame.draw.line(surface, theme.CARD_INSET, (plot.x, y), (plot.right, y))
    text(surface, fonts.mono, f"{high:.1f}", theme.TEXT_FAINT, bottomright=(plot.right, plot.y - 2))
    text(surface, fonts.mono, f"{low:.1f}", theme.TEXT_FAINT, topright=(plot.right, plot.bottom + 2))

    def to_y(v: float) -> float:
        return plot.bottom - (v - low) / (high - low) * plot.height

    if low < 0 < high:
        pygame.draw.line(surface, theme.TEXT_FAINT, (plot.x, to_y(0)), (plot.right, to_y(0)))

    for data, color in series:
        stride = max(1, len(data) // plot.width)
        points = [(plot.x + i * plot.width / x_span, to_y(v)) for i, v in enumerate(data) if i % stride == 0]
        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, 2)


def footer(surface, rect: pygame.Rect, fonts: Fonts, value: str) -> None:
    y = rect.bottom - 32
    pygame.draw.line(surface, theme.PANEL_BORDER, (rect.x + PADDING, y - 12), (rect.right - PADDING, y - 12))
    text(surface, fonts.small, value, theme.TEXT_FAINT, topleft=(rect.x + PADDING, y))
