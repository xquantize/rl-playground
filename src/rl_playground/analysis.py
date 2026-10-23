"""Analysis of multi-seed experiments: learning curves, sample efficiency and final performance.

For every variant and seed this reads the training curve of mean laps per episode from TensorBoard and
evaluates the final model on a fixed set of episodes. Results are aggregated as mean and standard
deviation across seeds.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl_playground.evaluation import EvaluationSummary, evaluate, evaluation_episodes, mean_and_std
from rl_playground.experiments import Experiment, Variant
from rl_playground.policies import load_model
from rl_playground.runs import final_model, format_steps, seed_dirs, seed_of

log = logging.getLogger(__name__)

CURVE_TAG = "racing/laps_mean"
SMOOTHING_WINDOW = 5
CURVE_POINTS = 100


def load_scalars(run_dir: Path, tag: str = CURVE_TAG) -> tuple[np.ndarray, np.ndarray]:
    """All values logged under ``tag`` in a run's TensorBoard event files, ordered by step."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    steps, values = [], []
    for event_file in sorted((run_dir / "tensorboard").rglob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag in accumulator.Tags()["scalars"]:
            for event in accumulator.Scalars(tag):
                steps.append(event.step)
                values.append(event.value)
    order = np.argsort(steps, kind="stable")
    return np.asarray(steps, dtype=np.int64)[order], np.asarray(values, dtype=np.float64)[order]


def smooth(values: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    """Trailing moving average, so a threshold crossing is never attributed to a single noisy rollout."""
    if len(values) == 0:
        return values
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    counts = np.minimum(np.arange(1, len(values) + 1), window)
    return (cumulative[1:] - cumulative[np.arange(len(values)) + 1 - counts]) / counts


def curve_mean(steps: np.ndarray, values: np.ndarray) -> float:
    """Average of the curve over training, i.e. the normalised area under it."""
    if len(steps) < 2:
        return float(values.mean()) if len(values) else float("nan")
    area = np.sum((values[1:] + values[:-1]) / 2 * np.diff(steps))
    return float(area / (steps[-1] - steps[0]))


def steps_to_threshold(steps: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    reached = np.flatnonzero(smooth(values) >= threshold)
    return int(steps[reached[0]]) if reached.size else None


@dataclass
class SeedResult:
    seed: int
    steps: np.ndarray
    laps: np.ndarray
    evaluation: EvaluationSummary

    @property
    def learning_mean(self) -> float:
        return curve_mean(self.steps, self.laps)


@dataclass
class VariantResult:
    variant: Variant
    seeds: list[SeedResult]

    @property
    def evaluations(self) -> list[EvaluationSummary]:
        return [s.evaluation for s in self.seeds]

    def learning_mean(self) -> tuple[float, float]:
        values = [s.learning_mean for s in self.seeds]
        return float(np.mean(values)), float(np.std(values))

    def steps_to(self, threshold: float) -> tuple[list[int], int]:
        """Steps at which each seed first reached ``threshold`` and the number of seeds that never did."""
        reached = [steps_to_threshold(s.steps, s.laps, threshold) for s in self.seeds]
        hits = [r for r in reached if r is not None]
        return hits, len(reached) - len(hits)

    def mean_curve(self, points: int = CURVE_POINTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Learning curves interpolated onto a shared step grid: steps, mean and standard deviation."""
        start = max(s.steps[0] for s in self.seeds)
        end = min(s.steps[-1] for s in self.seeds)
        grid = np.linspace(start, end, points)
        curves = np.stack([np.interp(grid, s.steps, smooth(s.laps)) for s in self.seeds])
        return grid, curves.mean(axis=0), curves.std(axis=0)


def analyse(
    experiment: Experiment,
    runs_dir: Path = Path("runs"),
    episodes: int = 10,
    track_mode: str = "fixed",
    deterministic: bool = True,
) -> list[VariantResult]:
    episode_set = evaluation_episodes(track_mode, episodes)
    results = []
    for variant in experiment.variants:
        variant_dir = runs_dir / experiment.name / variant.name
        if not variant_dir.is_dir() or not seed_dirs(variant_dir):
            log.warning("Skipping %s: not trained", variant.name)
            continue
        seeds = []
        for run_dir in seed_dirs(variant_dir):
            seed, model_path = seed_of(run_dir), final_model(run_dir)
            steps, laps = load_scalars(run_dir)
            if len(steps) == 0:
                log.warning("Skipping %s, seed %d: no %s data", variant.name, seed, CURVE_TAG)
                continue
            evaluation = evaluate(load_model(model_path, deterministic=deterministic), variant.name, episode_set)
            seeds.append(SeedResult(seed, steps, laps, evaluation))
        if seeds:
            results.append(VariantResult(variant, seeds))
    return results


def _format_steps_to(result: VariantResult, threshold: float) -> str:
    hits, misses = result.steps_to(threshold)
    if not hits:
        return "not reached"
    text = format_steps(int(np.mean(hits)))
    if len(hits) > 1:
        text += f" ± {format_steps(int(np.std(hits)))}"
    if misses:
        text += f" ({len(hits)}/{len(hits) + misses})"
    return text


def report_rows(results: list[VariantResult], threshold: float) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Variant",
        "Seeds",
        "Eval laps",
        "Crash rate",
        "Mean training laps",
        "vs first",
        f"Steps to {threshold:g} laps",
    ]
    reference = results[0].learning_mean()[0] if results else float("nan")
    rows = []
    for result in results:
        laps, laps_std = mean_and_std(result.evaluations, "mean_laps")
        crash, crash_std = mean_and_std(result.evaluations, "crash_rate")
        learning, learning_std = result.learning_mean()
        relative = f"{learning / reference - 1:+.0%}" if result is not results[0] else "–"
        rows.append(
            [
                result.variant.name,
                str(len(result.seeds)),
                f"{laps:.2f} ± {laps_std:.2f}",
                f"{crash:.0%} ± {crash_std:.0%}",
                f"{learning:.2f} ± {learning_std:.2f}",
                relative,
                _format_steps_to(result, threshold),
            ]
        )
    return headers, rows


def format_report(results: list[VariantResult], threshold: float, markdown: bool = False) -> str:
    headers, rows = report_rows(results, threshold)
    if markdown:
        lines = ["| " + " | ".join(headers) + " |", "| :-- | " + " | ".join("--:" for _ in headers[1:]) + " |"]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join(lines)
    widths = [max(len(cell) for cell in column) for column in zip(headers, *rows, strict=True)]

    def line(cells) -> str:
        first, *rest = cells
        return "  ".join([first.ljust(widths[0]), *(c.rjust(w) for c, w in zip(rest, widths[1:], strict=True))])

    return "\n".join([line(headers), line("-" * w for w in widths), *(line(row) for row in rows)])


def write_outputs(experiment: Experiment, results: list[VariantResult], output_dir: Path, threshold: float) -> None:
    """Write ``report.md``, ``summary.json`` and ``curves.csv`` (mean and std learning curves per variant)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(
        f"# {experiment.name}\n\n{experiment.description}\n\n{format_report(results, threshold, markdown=True)}\n"
    )

    summary = []
    for result in results:
        hits, misses = result.steps_to(threshold)
        summary.append(
            {
                "variant": result.variant.name,
                "description": result.variant.description,
                "overrides": result.variant.overrides,
                "seeds": [s.seed for s in result.seeds],
                "mean_training_laps": [s.learning_mean for s in result.seeds],
                "steps_to_threshold": {"threshold": threshold, "reached": hits, "not_reached": misses},
                "evaluation": [s.evaluation.to_dict() for s in result.seeds],
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with (output_dir / "curves.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["variant", "step", "laps_mean", "laps_std", "seeds"])
        for result in results:
            grid, mean, std = result.mean_curve()
            for step, m, s in zip(grid, mean, std, strict=True):
                writer.writerow([result.variant.name, int(step), f"{m:.4f}", f"{s:.4f}", len(result.seeds)])
