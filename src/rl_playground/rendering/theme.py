"""Colors and typography shared by every view."""

from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

Color = tuple[int, int, int]

BACKGROUND: Color = (13, 15, 19)
GRID: Color = (20, 23, 28)
TRACK_SURFACE: Color = (31, 34, 41)
TRACK_EDGE: Color = (92, 98, 112)
TRACK_CENTERLINE: Color = (50, 54, 64)

PANEL: Color = (18, 20, 25)
PANEL_BORDER: Color = (36, 40, 49)
CARD: Color = (25, 28, 34)
CARD_INSET: Color = (37, 41, 50)

TEXT: Color = (232, 234, 238)
TEXT_MUTED: Color = (138, 144, 158)
TEXT_FAINT: Color = (88, 94, 106)

POSITIVE: Color = (110, 205, 150)
NEGATIVE: Color = (236, 98, 112)
WARNING: Color = (240, 190, 90)

AGENT_COLORS: tuple[Color, ...] = (
    (92, 158, 255),
    (255, 138, 92),
    (110, 205, 150),
    (196, 140, 255),
    (240, 190, 90),
    (236, 98, 112),
)

STATUS_COLORS: dict[str, Color] = {
    "Running": TEXT_MUTED,
    "Completed": POSITIVE,
    "Crashed": NEGATIVE,
    "Stalled": WARNING,
}

_SANS = "sfprotext,sfprodisplay,helveticaneue,helvetica,inter,segoeui,arial,dejavusans"
_MONO = "sfmono,menlo,jetbrainsmono,consolas,dejavusansmono"


def blend(a: Color, b: Color, t: float) -> Color:
    t = min(max(t, 0.0), 1.0)
    return (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t), round(a[2] + (b[2] - a[2]) * t))


class Fonts:
    def __init__(self) -> None:
        pygame.font.init()
        self.title = pygame.font.SysFont(_SANS, 22, bold=True)
        self.body = pygame.font.SysFont(_SANS, 14)
        self.body_bold = pygame.font.SysFont(_SANS, 14, bold=True)
        self.small = pygame.font.SysFont(_SANS, 12)
        self.label = pygame.font.SysFont(_SANS, 11, bold=True)
        self.value = pygame.font.SysFont(_MONO, 19, bold=True)
        self.mono = pygame.font.SysFont(_MONO, 12)
