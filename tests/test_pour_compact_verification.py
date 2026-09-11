"""A crop or refinement must not silently change the physical experiment."""
import copy

import numpy as np
import pytest

from experiments.pour.pour_compact_verification import crop_metrics


@pytest.fixture
def pair():
    original = dict(eta_pa_s=3.4, initial_volume_ml=300., source_wall="sticky",
                    receiver_wall="separable", particle_count=1000, grid=384,
                    receiver_ml=120., source_depletion_ml=121.)
    cropped = dict(original, dx_m=.7/384, phase_cells=0., sdf_resolution=160,
                   minimum_fluid_boundary_clearance_m=.04)
    times = np.array([0., 1., 2.])
    volumes = np.array([[300., 0., 0.], [200., 99., 1.], [179., 120., 1.]])
    curve = (times, volumes, 1000)
    return original, cropped, curve, copy.deepcopy(curve)


def test_identical_full_and_compact_solutions_pass(pair):
    result = crop_metrics(*pair)
    assert result["passed"]
    assert not result["measured_outcomes_used"]


def test_matching_endpoint_does_not_hide_curve_difference(pair):
    pair[3][1][1, 1] += 4.
    result = crop_metrics(*pair)
    assert not result["passed"]
    assert result["differences"]["final_receiver_ml"] == 0


@pytest.mark.parametrize("field,value", [("eta_pa_s", 3.5), ("particle_count", 999),
                                        ("dx_m", .002), ("source_wall", "separable"),
                                        ("phase_cells", .5), ("sdf_resolution", 384),
                                        ("minimum_fluid_boundary_clearance_m", .001)])
def test_crop_rejects_changed_physics_or_boundary_contact(pair, field, value):
    pair[1][field] = value
    with pytest.raises(ValueError):
        crop_metrics(*pair)


def test_source_depletion_must_agree_even_if_receiver_does(pair):
    pair[1]["source_depletion_ml"] += 1.01
    assert not crop_metrics(*pair)["passed"]
