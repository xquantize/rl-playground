import pytest

from rl_playground.experiments import EXPERIMENTS
from rl_playground.training import TrainConfig


@pytest.mark.parametrize("experiment", EXPERIMENTS.values(), ids=lambda e: e.name)
def test_every_variant_builds_a_valid_config(experiment, tmp_path):
    names = [variant.name for variant in experiment.variants]
    assert len(names) == len(set(names))
    for variant in experiment.variants:
        config = experiment.config(variant, seed=1)
        assert config.seed == 1
        assert config.run_name == f"{experiment.name}/{variant.name}"
        assert experiment.run_dir(tmp_path, variant, 1) == tmp_path / experiment.name / variant.name / "seed_1"


def test_ablations_change_exactly_one_setting():
    ablations = EXPERIMENTS["ppo-ablations"]
    reference = ablations.config(ablations.variants[0], seed=0)
    for variant in ablations.variants[1:]:
        config = ablations.config(variant, seed=0)
        changed = [name for name in TrainConfig.field_names() if getattr(config, name) != getattr(reference, name)]
        assert sorted(changed) == sorted([*variant.overrides, "run_name"])


def test_scratch_only_settings_are_rejected_for_sb3():
    with pytest.raises(ValueError, match="scratch"):
        TrainConfig(run_name="x", algorithm="sb3", clip_objective=False)
    with pytest.raises(ValueError, match="scratch"):
        TrainConfig(run_name="x", algorithm="sb3", max_grad_norm=None)
