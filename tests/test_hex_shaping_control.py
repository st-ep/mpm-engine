"""The shaping objective must use the same independent geometry at every grid."""
import numpy as np
from experiments.robotics.hex_shaping_control import target_prism
from experiments.robotics.hex_shaping_control_report import area, clipped_to_hexagon


def test_target_has_common_physical_dimensions_and_volume_across_grids():
    coarse, fine = target_prism(48), target_prism(64)
    assert float(coarse["diameter"]) == float(fine["diameter"]) == .090
    assert float(coarse["height"]) == float(fine["height"])
    assert float(coarse["volume"]) == float(fine["volume"])
    np.testing.assert_allclose(coarse["vol0"].sum(dtype=float), coarse["volume"], rtol=1e-7)
    np.testing.assert_allclose(fine["vol0"].sum(dtype=float), fine["volume"], rtol=1e-7)


def test_target_samples_fill_a_regular_hexagon_in_the_fixed_floor_frame():
    d = target_prism()
    x = d["x"].astype(float)-np.array([.15, .15, float(d["floor"])])
    # Analytic planes of a flat-sided regular hexagon, independently written.
    bounds = np.column_stack((abs(x[:, 0]),
                              abs(.5*x[:, 0]+np.sqrt(3)/2*x[:, 1]),
                              abs(-.5*x[:, 0]+np.sqrt(3)/2*x[:, 1])))
    assert np.all(bounds <= .045+1e-7)
    assert np.all(bounds.max(0) > .043)
    assert x[:, 2].min() > 0
    assert x[:, 2].max() < float(d["height"])
    # No lateral registration or displacement to favor a simulated outcome.
    np.testing.assert_allclose(x[:, :2].mean(0), 0, atol=1e-8)


def test_hexagon_clipping_recovers_analytic_area_and_preserves_interior_polygon():
    square = np.array([[-2., -2.], [2., -2.], [2., 2.], [-2., 2.]])
    np.testing.assert_allclose(area(clipped_to_hexagon(square, 1.)), 2*np.sqrt(3), rtol=1e-12)
    inner = square*.1
    np.testing.assert_allclose(area(clipped_to_hexagon(inner, 1.)), .16, rtol=1e-12)
    assert len(clipped_to_hexagon(square+10, 1.)) == 0
