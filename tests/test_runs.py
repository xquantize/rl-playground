import pytest

from rl_playground.runs import describe_model, format_steps, resolve_models


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_resolves_every_seed_in_numeric_order(tmp_path):
    for seed in (10, 2, 0):
        touch(tmp_path / "run" / f"seed_{seed}" / "model.pt")
    touch(tmp_path / "run" / "seed_5" / "config.json")
    models = resolve_models(tmp_path / "run")
    assert [m.parent.name for m in models] == ["seed_0", "seed_2", "seed_10"]


def test_resolves_single_run_checkpoint_and_missing_suffix(tmp_path):
    final = touch(tmp_path / "legacy" / "model.zip")
    checkpoint = touch(tmp_path / "run" / "seed_0" / "checkpoints" / "model_200000_steps.pt")
    assert resolve_models(tmp_path / "legacy") == [final]
    assert resolve_models(checkpoint) == [checkpoint]
    assert resolve_models(checkpoint.with_suffix("")) == [checkpoint]


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_models(tmp_path / "nothing")


def test_labels_ignore_seed_directories(tmp_path):
    assert describe_model(tmp_path / "baseline" / "seed_3" / "model.zip") == "baseline"
    checkpoint = tmp_path / "baseline" / "seed_3" / "checkpoints" / "model_1500000_steps.pt"
    assert describe_model(checkpoint) == "baseline @ 1.5M"
    assert describe_model(tmp_path / "legacy" / "model.zip") == "legacy"


@pytest.mark.parametrize(("steps", "label"), [(50_000, "50k"), (1_000_000, "1M"), (2_500_000, "2.5M")])
def test_format_steps(steps, label):
    assert format_steps(steps) == label
