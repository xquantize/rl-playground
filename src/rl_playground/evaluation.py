"""Quantitative evaluation of policies over a fixed, reproducible set of episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from rl_playground.envs.track import Track
from rl_playground.policies import Policy
from rl_playground.rollout import Rollout, Status

EVALUATION_SEED = 10_000


@dataclass(frozen=True)
class Episode:
    track: Track
    start_index: int


@dataclass(frozen=True)
class EpisodeResult:
    track: str
    start_index: int
    laps: float
    outcome: str
    steps: int
    episode_return: float
    mean_speed: float


@dataclass
class EvaluationSummary:
    label: str
    episodes: list[EpisodeResult]

    @property
    def mean_laps(self) -> float:
        return float(np.mean([e.laps for e in self.episodes]))

    @property
    def crash_rate(self) -> float:
        return float(np.mean([e.outcome == Status.CRASHED for e in self.episodes]))

    @property
    def completion_rate(self) -> float:
        return float(np.mean([e.outcome == Status.COMPLETED for e in self.episodes]))

    @property
    def mean_return(self) -> float:
        return float(np.mean([e.episode_return for e in self.episodes]))

    @property
    def mean_speed(self) -> float:
        return float(np.mean([e.mean_speed for e in self.episodes]))

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "mean_laps": self.mean_laps,
            "crash_rate": self.crash_rate,
            "completion_rate": self.completion_rate,
            "mean_return": self.mean_return,
            "mean_speed": self.mean_speed,
            "episodes": [asdict(e) for e in self.episodes],
        }


def evaluation_episodes(track_mode: str, num_episodes: int, seed: int = EVALUATION_SEED) -> list[Episode]:
    """Fixed mode varies the start position around the default track; random mode samples new tracks."""
    if track_mode == "fixed":
        track = Track.default()
        return [Episode(track, i * len(track) // num_episodes) for i in range(num_episodes)]
    rng_seeds = range(seed, seed + num_episodes)
    return [Episode(Track.random(np.random.default_rng(s), name=f"Random track · seed {s}"), 0) for s in rng_seeds]


def evaluate(policy: Policy, label: str, episodes: list[Episode]) -> EvaluationSummary:
    torch.manual_seed(EVALUATION_SEED)
    rollout = Rollout(policy, label)
    results = []
    for episode in episodes:
        rollout.run(episode.track, episode.start_index)
        steps = rollout.env.steps
        results.append(
            EpisodeResult(
                track=episode.track.name,
                start_index=episode.start_index,
                laps=rollout.env.laps,
                outcome=rollout.status.value,
                steps=steps,
                episode_return=rollout.episode_return,
                mean_speed=rollout.total_speed / max(steps, 1),
            )
        )
    return EvaluationSummary(label, results)


METRICS = ("mean_laps", "crash_rate", "completion_rate", "mean_speed", "mean_return")


def mean_and_std(summaries: list[EvaluationSummary], metric: str) -> tuple[float, float]:
    values = [getattr(s, metric) for s in summaries]
    return float(np.mean(values)), float(np.std(values))


def format_metric(summaries: list[EvaluationSummary], metric: str) -> str:
    mean, std = mean_and_std(summaries, metric)
    spec = ".0%" if metric.endswith("_rate") else ".2f" if metric == "mean_laps" else ".1f"
    return f"{mean:{spec}}" if len(summaries) == 1 else f"{mean:{spec}} ± {std:{spec}}"


def format_table(groups: list[tuple[str, list[EvaluationSummary]]]) -> str:
    """Plain-text table with one row per agent; metrics show mean ± std across seeds."""
    headers = ("Agent", "Seeds", "Mean laps", "Crash rate", "Completed", "Mean speed", "Mean return")
    rows = [
        (label, str(len(summaries)), *(format_metric(summaries, m) for m in METRICS)) for label, summaries in groups
    ]
    widths = [max(len(cell) for cell in column) for column in zip(headers, *rows, strict=True)]

    def line(cells) -> str:
        first, *rest = cells
        return "  ".join([first.ljust(widths[0]), *(c.rjust(w) for c, w in zip(rest, widths[1:], strict=True))])

    return "\n".join([line(headers), line("-" * w for w in widths), *(line(r) for r in rows)])
