"""Side panels: detailed telemetry for one agent, and a leaderboard for several."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pygame

from rl_playground.rendering import theme, widgets
from rl_playground.rendering.theme import Fonts
from rl_playground.rendering.widgets import PADDING

if TYPE_CHECKING:
    from rl_playground.rollout import Rollout

CONTROLS = "Space pause · F fast-forward · N next · Esc quit"
MIN_CHART_SPAN = 300


class AgentPanel:
    """Live telemetry for a single agent: episode stats, policy outputs, sensors and reward."""

    def __init__(self, rollout: Rollout, subtitle: str):
        self.rollout = rollout
        self.subtitle = subtitle

    def draw(self, surface: pygame.Surface, rect: pygame.Rect, fonts: Fonts) -> None:
        rollout, env = self.rollout, self.rollout.env
        widgets.panel_background(surface, rect)
        x, width = rect.x + PADDING, rect.width - 2 * PADDING
        y = widgets.header(surface, rect, fonts, rollout.label, self.subtitle, rollout.status.value)

        y = widgets.section(surface, fonts, "Episode", x, y, width)
        card_width, card_height, gap = (width - 8) // 2, 52, 8
        stats = [
            ("Laps", f"{env.laps:.2f}", ""),
            ("Speed", f"{env.speed:.0f}", "px/s"),
            ("Return", f"{rollout.episode_return:.1f}", ""),
            ("Step", f"{env.steps}", f"/ {env.config.max_episode_steps}"),
        ]
        for i, (label, value, unit) in enumerate(stats):
            card = pygame.Rect(
                x + (i % 2) * (card_width + gap), y + (i // 2) * (card_height + gap), card_width, card_height
            )
            widgets.stat_card(surface, fonts, card, label, value, unit)
        y += 2 * card_height + gap + 22

        y = widgets.section(surface, fonts, "Policy output", x, y, width)
        y = widgets.signed_bar(surface, fonts, x, y, width, "Steering", float(rollout.action[0]), rollout.color)
        y = widgets.signed_bar(surface, fonts, x, y, width, "Throttle", float(rollout.action[1]), rollout.color)
        y += 8

        y = widgets.section(surface, fonts, "Sensors", x, y, width)
        angles = [math.degrees(a) for a in env.sensor_angles]
        y = widgets.sensor_bars(
            surface, fonts, x, y, width, env.sensor_readings, angles, env.config.sensor_range, rollout.color
        )
        y += 18

        y = widgets.section(surface, fonts, "Cumulative reward", x, y, width)
        chart = pygame.Rect(x, y + 12, width, rect.bottom - 64 - (y + 12))
        span = max(len(rollout.return_history), MIN_CHART_SPAN)
        widgets.line_chart(surface, fonts, chart, [(rollout.return_history, rollout.color)], span)
        widgets.footer(surface, rect, fonts, CONTROLS)


class LeaderboardPanel:
    """Ranking and progress of several agents racing on the same track."""

    def __init__(self, rollouts: Sequence[Rollout], subtitle: str):
        self.rollouts = rollouts
        self.subtitle = subtitle

    def draw(self, surface: pygame.Surface, rect: pygame.Rect, fonts: Fonts) -> None:
        widgets.panel_background(surface, rect)
        x, width = rect.x + PADDING, rect.width - 2 * PADDING
        step = max(r.env.steps for r in self.rollouts)
        limit = self.rollouts[0].env.config.max_episode_steps
        y = widgets.header(surface, rect, fonts, "Comparison", self.subtitle, meta=f"Step {step} / {limit}")

        y = widgets.section(surface, fonts, "Leaderboard", x, y, width)
        ranked = sorted(self.rollouts, key=lambda r: r.env.laps, reverse=True)
        for rank, rollout in enumerate(ranked, start=1):
            row = pygame.Rect(x, y, width, 54)
            pygame.draw.rect(surface, theme.CARD, row, border_radius=widgets.RADIUS)
            widgets.text(surface, fonts.mono, f"{rank}", theme.TEXT_FAINT, midleft=(row.x + 14, row.centery))
            pygame.draw.circle(surface, rollout.color, (row.x + 38, row.centery), 4)
            widgets.text(surface, fonts.body_bold, rollout.label, theme.TEXT, topleft=(row.x + 52, row.y + 9))
            status = rollout.status.value
            widgets.text(surface, fonts.small, status, theme.STATUS_COLORS[status], topleft=(row.x + 52, row.y + 29))
            widgets.text(
                surface,
                fonts.value,
                f"{rollout.env.laps:.2f}",
                theme.TEXT,
                topright=(row.right - 14, row.y + 8),
            )
            widgets.text(surface, fonts.small, "laps", theme.TEXT_FAINT, topright=(row.right - 14, row.y + 31))
            if not rollout.done:
                fraction = rollout.env.speed / rollout.env.config.vehicle.max_speed
                track_width = row.width - 52 - 80
                pygame.draw.rect(surface, theme.CARD_INSET, (row.x + 52, row.bottom - 7, track_width, 2))
                pygame.draw.rect(surface, rollout.color, (row.x + 52, row.bottom - 7, round(track_width * fraction), 2))
            y += 62
        y += 12

        y = widgets.section(surface, fonts, "Laps completed", x, y, width)
        chart = pygame.Rect(x, y + 12, width, max(120, rect.bottom - 64 - (y + 12)))
        span = max(max(len(r.laps_history) for r in self.rollouts), MIN_CHART_SPAN)
        widgets.line_chart(surface, fonts, chart, [(r.laps_history, r.color) for r in self.rollouts], span)

        widgets.footer(surface, rect, fonts, CONTROLS)
