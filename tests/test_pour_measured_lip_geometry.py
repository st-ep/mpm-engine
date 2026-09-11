import numpy as np

from warpmpm.geometry import measuring_cup as cup
from experiments.pour.pour_measured_lip_geometry import ORIGINAL_SOLID, measured_lip_solid, measured_lip_field


def test_measured_lip_preserves_real_solid_and_lower_padding():
    spec = cup.MeasuringCupSpec()
    rng = np.random.default_rng(3)
    x = rng.uniform([-.06, -.055, -.01], [.09, .055, .11], (20000, 3))
    ew, eb = .0043125, .0021725
    physical = ORIGINAL_SOLID(x, spec, 0., 0.)
    old = ORIGINAL_SOLID(x, spec, ew, eb)
    new = measured_lip_solid(x, spec, ew, eb)
    assert np.all(new[physical < 0] < 0)
    assert np.all(old[new < 0] < 0)
    lower = x[:, 2] <= spec.spout_z0
    top = x[:, 2] >= spec.rim_z - spec.rim_blend
    np.testing.assert_array_equal(new[lower], old[lower])
    np.testing.assert_array_equal(new[top], physical[top])


def test_field_override_restores_global_helper_after_failure():
    try:
        with measured_lip_field():
            assert cup.solid_sdf_local is measured_lip_solid
            raise RuntimeError("test")
    except RuntimeError:
        pass
    assert cup.solid_sdf_local is ORIGINAL_SOLID
