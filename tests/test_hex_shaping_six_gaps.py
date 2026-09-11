"""A hexagon-quality diagnostic must distinguish a circle of similar area."""
import numpy as np
from experiments.robotics.hex_shaping_six_report import corner_to_face_ratio


def test_hexagon_radial_ratio_matches_analytic_geometry_at_different_scales():
    angles = np.deg2rad(np.arange(30, 390, 60))
    vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    for radius in [.01, .05, .1]:
        np.testing.assert_allclose(corner_to_face_ratio(radius*vertices), 2/np.sqrt(3), rtol=1e-12)


def test_circle_has_no_six_corner_radial_contrast():
    angles = np.deg2rad(np.arange(360))
    vertices = .05*np.column_stack((np.cos(angles), np.sin(angles)))
    np.testing.assert_allclose(corner_to_face_ratio(vertices), 1., rtol=1e-12)
