import numpy as np
import pytest

from rl_playground.envs.track import MIN_TURN_RADIUS_RATIO, Track, _min_turn_radius


@pytest.mark.parametrize("seed", range(5))
def test_random_tracks_respect_minimum_turn_radius(seed):
    track = Track.random(np.random.default_rng(seed))
    assert _min_turn_radius(track.centerline) > MIN_TURN_RADIUS_RATIO * track.half_width


def test_random_tracks_are_reproducible():
    a = Track.random(np.random.default_rng(7))
    b = Track.random(np.random.default_rng(7))
    np.testing.assert_allclose(a.centerline, b.centerline)


def test_projection_of_centerline_point_is_on_track():
    track = Track.default()
    projection = track.project(track.centerline[10])
    assert projection.distance == pytest.approx(0.0, abs=1e-9)
    assert projection.arc_length == pytest.approx(track.segment_offsets[10], abs=1e-6)


def test_raycast_to_the_side_hits_the_wall_at_half_width():
    track = Track.default()
    heading = track.heading_at(0)
    side = np.array([[np.cos(heading + np.pi / 2), np.sin(heading + np.pi / 2)]])
    distance = track.raycast(track.centerline[0], side, max_distance=500.0)[0]
    assert distance == pytest.approx(track.half_width, rel=0.05)


def test_wrap_arc_delta_handles_the_start_line():
    track = Track.default()
    assert track.wrap_arc_delta(-(track.length - 5.0)) == pytest.approx(5.0)
    assert track.wrap_arc_delta(track.length - 5.0) == pytest.approx(-5.0)
