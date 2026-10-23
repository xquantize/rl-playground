"""Learning-curve figures from the ``curves.csv`` files written by ``rlp experiment report``."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

COLORS = ("#4c8dff", "#ff7a45", "#2fb67c", "#a371f7", "#e3b341", "#f05d6c")
AXIS_COLOR = "#8b949e"


def read_curves(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Per-variant arrays of ``step``, ``mean``, ``std`` and ``seeds``, in the order variants appear in the file."""
    columns: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"step": [], "mean": [], "std": [], "seeds": []})
    with path.open() as file:
        for row in csv.DictReader(file):
            curve = columns[row["variant"]]
            curve["step"].append(float(row["step"]))
            curve["mean"].append(float(row["laps_mean"]))
            curve["std"].append(float(row["laps_std"]))
            curve["seeds"].append(int(row["seeds"]))
    return {name: {key: np.asarray(values) for key, values in curve.items()} for name, curve in columns.items()}


def plot_curves(curves_path: Path, output_path: Path, title: str) -> Path:
    """Mean laps per episode during training, with a band of one standard deviation across seeds.

    The figure has a transparent background and neutral grey text so it reads well in light and dark themes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = read_curves(curves_path)
    figure, axes = plt.subplots(figsize=(8, 4.5), dpi=200)

    for color, (name, curve) in zip(COLORS, curves.items(), strict=False):
        steps = curve["step"] / 1000
        axes.plot(steps, curve["mean"], color=color, linewidth=2, label=name)
        axes.fill_between(steps, curve["mean"] - curve["std"], curve["mean"] + curve["std"], color=color, alpha=0.18)

    seed_counts = sorted({int(curve["seeds"].max()) for curve in curves.values()})
    seeds = seed_counts[0] if len(seed_counts) == 1 else f"{seed_counts[0]}–{seed_counts[-1]}"
    subtitle = f"Mean ± standard deviation across {seeds} seeds"
    axes.set_title(f"{title}\n", color=AXIS_COLOR, fontsize=13, loc="left", fontweight="bold")
    axes.text(0, 1.02, subtitle, transform=axes.transAxes, color=AXIS_COLOR, fontsize=9)
    axes.set_xlabel("Environment steps (thousands)", color=AXIS_COLOR)
    axes.set_ylabel("Laps per episode", color=AXIS_COLOR)
    axes.set_ylim(bottom=0)
    axes.set_xlim(left=0)
    axes.grid(color=AXIS_COLOR, alpha=0.15, linewidth=0.8)
    axes.tick_params(colors=AXIS_COLOR)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS_COLOR)
    legend = axes.legend(frameon=False, loc="lower right")
    for text in legend.get_texts():
        text.set_color(AXIS_COLOR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, transparent=True, bbox_inches="tight")
    plt.close(figure)
    return output_path
