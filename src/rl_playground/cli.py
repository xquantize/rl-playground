"""Command-line interface: ``rlp train | evaluate | watch | compare | drive``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rl_playground.envs.racing import TRACK_MODES
from rl_playground.training import ALGORITHMS

log = logging.getLogger("rl_playground")


def _train(args: argparse.Namespace) -> int:
    from rl_playground.training import TrainConfig, train

    config = TrainConfig(
        run_name=args.name,
        algorithm=args.algorithm,
        track_mode=args.track,
        total_timesteps=args.timesteps,
        num_envs=args.envs,
        seed=args.seed,
    )
    log.info(
        "Training %s with %s PPO on %s tracks for %s steps",
        config.run_name,
        config.algorithm,
        config.track_mode,
        f"{config.total_timesteps:,}",
    )
    run_dir = train(config, runs_dir=args.runs_dir, overwrite=args.overwrite)
    log.info("Run saved to %s", run_dir)
    log.info("Training curves: tensorboard --logdir %s", run_dir / "tensorboard")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from rl_playground.evaluation import evaluate, evaluation_episodes, format_table
    from rl_playground.policies import load_policy

    episodes = evaluation_episodes(args.track, args.episodes, args.seed)
    summaries = []
    for path in args.models:
        policy, label = load_policy(path)
        summaries.append(evaluate(policy, label, episodes))
    print(format_table(summaries))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps([s.to_dict() for s in summaries], indent=2))
        log.info("Results written to %s", args.output)
    return 0


def _visual_session(args: argparse.Namespace, rollouts, make_panel, num_tracks: int | None) -> int:
    from rl_playground.rendering import Recorder, Viewer
    from rl_playground.visualization import run_session, track_sequence

    if args.headless and not args.record:
        raise SystemExit("--headless requires --record")
    viewer = Viewer(headless=args.headless)
    recorder = Recorder(args.record, max_seconds=args.record_seconds) if args.record else None
    try:
        run_session(
            viewer,
            rollouts,
            track_sequence(args.track, args.seed, num_tracks),
            make_panel,
            recorder,
            show_sensors=not args.hide_sensors,
        )
    finally:
        viewer.close()
        if recorder is not None and (path := recorder.save()) is not None:
            log.info("Recording saved to %s", path)
    return 0


def _watch(args: argparse.Namespace) -> int:
    from rl_playground.policies import load_policy
    from rl_playground.rendering import AgentPanel
    from rl_playground.rendering.theme import AGENT_COLORS
    from rl_playground.rollout import Rollout

    policy, label = load_policy(args.model)
    rollout = Rollout(policy, label, AGENT_COLORS[0])
    return _visual_session(args, [rollout], lambda track: AgentPanel(rollout, track.name), args.episodes)


def _compare(args: argparse.Namespace) -> int:
    from rl_playground.policies import load_policy
    from rl_playground.rendering import LeaderboardPanel
    from rl_playground.rendering.theme import AGENT_COLORS
    from rl_playground.rollout import Rollout

    if len(args.models) > len(AGENT_COLORS):
        raise SystemExit(f"compare supports up to {len(AGENT_COLORS)} models")
    if args.labels and len(args.labels) != len(args.models):
        raise SystemExit("--labels must match the number of models")

    rollouts = []
    for i, path in enumerate(args.models):
        policy, label = load_policy(path)
        rollouts.append(Rollout(policy, args.labels[i] if args.labels else label, AGENT_COLORS[i]))
    num_tracks = args.rounds if args.track == "random" else 1
    return _visual_session(args, rollouts, lambda track: LeaderboardPanel(rollouts, track.name), num_tracks)


def _drive(args: argparse.Namespace) -> int:
    from rl_playground.policies import KeyboardPolicy
    from rl_playground.rendering import AgentPanel
    from rl_playground.rendering.theme import AGENT_COLORS
    from rl_playground.rollout import Rollout

    rollout = Rollout(KeyboardPolicy(), "Manual control", AGENT_COLORS[0])
    return _visual_session(args, [rollout], lambda track: AgentPanel(rollout, track.name), None)


def _add_track_options(parser: argparse.ArgumentParser, default_seed: int) -> None:
    parser.add_argument("--track", choices=TRACK_MODES, default="fixed", help="default track or random tracks")
    parser.add_argument("--seed", type=int, default=default_seed, help="seed of the first random track")


def _add_view_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--record", type=Path, help="save the first track as a GIF at this path")
    parser.add_argument("--record-seconds", type=float, default=20, help="maximum length of the recording")
    parser.add_argument("--headless", action="store_true", help="render off-screen; requires --record")
    parser.add_argument("--hide-sensors", action="store_true", help="do not draw sensor rays")


def build_parser() -> argparse.ArgumentParser:
    from rl_playground.evaluation import EVALUATION_SEED

    parser = argparse.ArgumentParser(prog="rlp", description="Train, evaluate and visualise racing agents.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    train = commands.add_parser("train", help="train a PPO agent")
    train.add_argument("--name", required=True, help="run name; outputs go to <runs-dir>/<name>")
    train.add_argument("--track", choices=TRACK_MODES, default="fixed", help="train on the default or random tracks")
    train.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        default="sb3",
        help="sb3: Stable-Baselines3 PPO; scratch: this repository's implementation",
    )
    train.add_argument("--timesteps", type=int, default=1_000_000)
    train.add_argument("--envs", type=int, default=8, help="parallel environments")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--runs-dir", type=Path, default=Path("runs"))
    train.add_argument("--overwrite", action="store_true", help="replace an existing run with the same name")
    train.set_defaults(handler=_train)

    evaluate = commands.add_parser("evaluate", help="measure performance over a fixed set of episodes")
    evaluate.add_argument("models", nargs="+", help="run directories or checkpoint files")
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--output", type=Path, help="write detailed results as JSON")
    _add_track_options(evaluate, EVALUATION_SEED)
    evaluate.set_defaults(handler=_evaluate)

    watch = commands.add_parser("watch", help="watch one agent with live telemetry")
    watch.add_argument("model", help="run directory or checkpoint file")
    watch.add_argument("--episodes", type=int, default=5)
    _add_track_options(watch, EVALUATION_SEED)
    _add_view_options(watch)
    watch.set_defaults(handler=_watch)

    compare = commands.add_parser("compare", help="race several agents on the same track")
    compare.add_argument("models", nargs="+", help="run directories or checkpoint files")
    compare.add_argument("--labels", nargs="+", help="display names, one per model")
    compare.add_argument("--rounds", type=int, default=3, help="number of random tracks")
    _add_track_options(compare, EVALUATION_SEED)
    _add_view_options(compare)
    compare.set_defaults(handler=_compare)

    drive = commands.add_parser("drive", help="drive the car yourself with the arrow keys")
    _add_track_options(drive, EVALUATION_SEED)
    _add_view_options(drive)
    drive.set_defaults(handler=_drive)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, FileExistsError) as error:
        log.error("Error: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
