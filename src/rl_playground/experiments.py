"""Predefined multi-seed experiments: named groups of training variants."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rl_playground.runs import final_model, seed_dir
from rl_playground.training import TrainConfig, train_run

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    variants: tuple[Variant, ...]
    title: str = ""
    base: dict[str, Any] = field(default_factory=dict)
    seeds: tuple[int, ...] = (0, 1, 2)
    total_timesteps: int = 500_000

    def config(self, variant: Variant, seed: int, **overrides: Any) -> TrainConfig:
        settings = {"total_timesteps": self.total_timesteps, **self.base, **variant.overrides, **overrides}
        return TrainConfig(run_name=f"{self.name}/{variant.name}", seed=seed, **settings)

    def run_dir(self, runs_dir: Path, variant: Variant, seed: int) -> Path:
        return seed_dir(runs_dir / self.name / variant.name, seed)


EXPERIMENTS: dict[str, Experiment] = {
    experiment.name: experiment
    for experiment in (
        Experiment(
            name="sb3-vs-scratch",
            description="Stable-Baselines3 PPO against the from-scratch implementation, identical hyperparameters.",
            title="Stable-Baselines3 vs from-scratch PPO",
            variants=(
                Variant("sb3", "Stable-Baselines3 PPO", {"algorithm": "sb3"}),
                Variant("scratch", "From-scratch PPO", {"algorithm": "scratch"}),
            ),
        ),
        Experiment(
            name="ppo-ablations",
            description="From-scratch PPO with one implementation detail removed at a time.",
            title="PPO implementation-detail ablations",
            base={"algorithm": "scratch"},
            variants=(
                Variant("full", "All implementation details enabled"),
                Variant(
                    "no-advantage-norm",
                    "Raw advantages instead of per-minibatch standardised advantages",
                    {"normalize_advantage": False},
                ),
                Variant("no-orthogonal-init", "PyTorch default initialisation", {"orthogonal_init": False}),
                Variant("no-grad-clipping", "No global gradient-norm clipping", {"max_grad_norm": None}),
                Variant("no-clipped-objective", "Unclipped policy-gradient surrogate", {"clip_objective": False}),
                Variant(
                    "no-truncation-bootstrap",
                    "Time-limit truncation treated as a terminal state",
                    {"bootstrap_truncation": False},
                ),
            ),
        ),
        Experiment(
            name="action-squashing",
            description="Clipping out-of-range Gaussian samples against squashing them with tanh.",
            title="Clipped vs tanh-squashed actions",
            base={"algorithm": "scratch"},
            variants=(
                Variant("clipped", "Gaussian samples clipped to the action bounds"),
                Variant("tanh", "Gaussian samples squashed with tanh", {"squash_actions": True}),
            ),
        ),
    )
}


def run_experiment(
    experiment: Experiment,
    runs_dir: Path = Path("runs"),
    seeds: list[int] | None = None,
    **overrides: Any,
) -> None:
    """Train every variant for every seed. Completed runs are skipped, so an interrupted experiment resumes."""
    seeds = list(seeds or experiment.seeds)
    jobs = [(variant, seed) for variant in experiment.variants for seed in seeds]
    for index, (variant, seed) in enumerate(jobs, start=1):
        run_dir = experiment.run_dir(runs_dir, variant, seed)
        prefix = f"[{index}/{len(jobs)}] {variant.name}, seed {seed}"
        if final_model(run_dir):
            log.info("%s: already trained", prefix)
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)
        log.info("%s", prefix)
        train_run(experiment.config(variant, seed, **overrides), run_dir)
