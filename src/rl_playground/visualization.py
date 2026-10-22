"""Interactive sessions that run rollouts on a sequence of tracks inside a viewer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import count

import numpy as np

from rl_playground.envs.track import Track
from rl_playground.rendering.viewer import Panel, Recorder, Viewer
from rl_playground.rollout import Rollout

LINGER_SECONDS = 2.0


def track_sequence(track_mode: str, seed: int, num_tracks: int | None = None) -> Iterator[Track]:
    """Yield the default track repeatedly, or random tracks with consecutive seeds."""
    indices = range(num_tracks) if num_tracks is not None else count()
    for i in indices:
        if track_mode == "random":
            yield Track.random(np.random.default_rng(seed + i), name=f"Random track · seed {seed + i}")
        else:
            yield Track.default()


def run_session(
    viewer: Viewer,
    rollouts: Sequence[Rollout],
    tracks: Iterable[Track],
    make_panel: Callable[[Track], Panel],
    recorder: Recorder | None = None,
    show_sensors: bool = True,
) -> None:
    """Race every rollout on each track in turn. Only the first track is recorded."""
    for track in tracks:
        for rollout in rollouts:
            rollout.reset(track)
        panel = make_panel(track)
        linger = int(LINGER_SECONDS * viewer.fps)

        while linger > 0:
            actions = viewer.poll()
            if "quit" in actions:
                return
            if "next" in actions:
                break
            if not viewer.paused:
                if all(r.done for r in rollouts):
                    linger -= 1
                else:
                    for rollout in rollouts:
                        rollout.step()

            cars = sorted(rollouts, key=lambda r: r.env.laps)
            viewer.draw(track, [r.car_view() for r in cars], panel, show_sensors)
            viewer.present()

            if recorder is not None and not viewer.paused:
                recorder.add(viewer.capture())
                if recorder.full and viewer.headless:
                    return
        if viewer.headless:
            return
        recorder = None
