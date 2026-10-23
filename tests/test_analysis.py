import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

from rl_playground.analysis import CURVE_TAG, curve_mean, load_scalars, smooth, steps_to_threshold


def test_smooth_is_a_trailing_moving_average():
    np.testing.assert_allclose(smooth(np.array([2.0, 4.0, 6.0, 8.0]), window=2), [2.0, 3.0, 5.0, 7.0])


def test_curve_mean_is_normalised_area():
    steps = np.array([0, 10, 20])
    assert curve_mean(steps, np.array([0.0, 10.0, 10.0])) == pytest.approx(7.5)


def test_steps_to_threshold_uses_smoothed_curve():
    steps = np.arange(1, 11) * 100
    values = np.array([0, 0, 0, 9, 0, 5, 5, 5, 5, 5], dtype=float)
    assert steps_to_threshold(steps, values, threshold=4.0) == 800
    assert steps_to_threshold(steps, values, threshold=50.0) is None


def test_load_scalars_reads_tensorboard_logs(tmp_path):
    writer = SummaryWriter(str(tmp_path / "tensorboard" / "ppo_1"))
    for step, value in [(200, 2.0), (100, 1.0), (300, 3.0)]:
        writer.add_scalar(CURVE_TAG, value, step)
    writer.close()
    steps, values = load_scalars(tmp_path)
    np.testing.assert_array_equal(steps, [100, 200, 300])
    np.testing.assert_allclose(values, [1.0, 2.0, 3.0])
