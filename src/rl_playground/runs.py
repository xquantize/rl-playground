"""On-disk layout of training runs and resolution of model paths.

A training run with several seeds is stored as::

    runs/<name>/
    └── seed_<n>/
        ├── config.json
        ├── model.zip | model.pt
        ├── checkpoints/model_<steps>_steps.zip | .pt
        └── tensorboard/
"""

from __future__ import annotations

import re
from pathlib import Path

MODEL_FILENAMES = ("model.zip", "model.pt")
CHECKPOINTS_DIRNAME = "checkpoints"
_SEED_PATTERN = re.compile(r"seed_(\d+)")
_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)_steps")


def seed_dir(run_dir: Path, seed: int) -> Path:
    return run_dir / f"seed_{seed}"


def seed_of(directory: Path) -> int | None:
    match = _SEED_PATTERN.fullmatch(directory.name)
    return int(match.group(1)) if match else None


def final_model(directory: Path) -> Path | None:
    for name in MODEL_FILENAMES:
        if (directory / name).is_file():
            return directory / name
    return None


def seed_dirs(run_dir: Path) -> list[Path]:
    """Completed seed directories of a run, ordered by seed."""
    found = [d for d in run_dir.iterdir() if d.is_dir() and seed_of(d) is not None and final_model(d)]
    return sorted(found, key=seed_of)


def resolve_models(path: str | Path) -> list[Path]:
    """Model files for a checkpoint, a single-seed directory, or every seed of a run."""
    path = Path(path)
    if path.is_file():
        return [path]
    for suffix in (".zip", ".pt"):
        if path.with_suffix(suffix).is_file():
            return [path.with_suffix(suffix)]
    if path.is_dir():
        if model := final_model(path):
            return [model]
        if seeds := seed_dirs(path):
            return [final_model(d) for d in seeds]
    raise FileNotFoundError(f"No model found at {path}")


def format_steps(steps: int) -> str:
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:.1f}".removesuffix(".0") + "M"
    if steps >= 1000:
        return f"{steps // 1000}k"
    return str(steps)


def run_label(directory: Path) -> str:
    """Name of the run a directory belongs to, ignoring the seed subdirectory."""
    return directory.parent.name if seed_of(directory) is not None else directory.name


def describe_model(model_path: Path) -> str:
    """Human-readable label, e.g. ``baseline`` or ``baseline @ 200k``."""
    match = _CHECKPOINT_PATTERN.fullmatch(model_path.stem)
    if match and model_path.parent.name == CHECKPOINTS_DIRNAME:
        return f"{run_label(model_path.parent.parent)} @ {format_steps(int(match.group(1)))}"
    if model_path.name in MODEL_FILENAMES:
        return run_label(model_path.parent)
    return model_path.stem
