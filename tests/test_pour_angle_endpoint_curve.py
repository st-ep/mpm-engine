"""The empirical inverse is well defined over the whole requested volume range."""
import numpy as np
import pytest
from experiments.pour.pour_angle_endpoint_curve import EndpointCurve, plan_targets


def test_measured_points_and_strict_monotonicity():
    curve = EndpointCurve([45, 50, 60], [48, 79, 159])
    np.testing.assert_allclose(curve.volume([45, 50, 60]), [48, 79, 159])
    a = np.linspace(45, curve.maximum_angle_deg, 10001)
    assert np.all(np.diff(curve.volume(a)) > 0)
    assert curve.end_slope == pytest.approx(9.2)
    assert float(curve.volume(curve.maximum_angle_deg)) == pytest.approx(160)


def test_inverse_at_intermediate_volumes_and_command_rounding():
    curve = EndpointCurve([45, 50, 60], [48, 79, 159])
    for v in np.linspace(60, 160, 301):
        assert float(curve.volume(curve.angle(v))) == pytest.approx(v, abs=1e-8)
    rows = plan_targets(curve, [60, 71, 80, 87, 100, 111, 120, 133, 140, 151, 159, 160])
    for row in rows:
        assert abs(row["expected_receiver_ml"] - row["target_ml"]) < 0.1
        assert row["pour_command_duration_s"] == pytest.approx(row["command_angle_deg"] / 10)
        assert row["dwell_s"] == row["return_command_duration_s"] == 2
    assert rows[-1]["command_angle_deg"] == 60.10


def test_no_silent_extrapolation_or_sorting():
    curve = EndpointCurve([45, 50, 60], [48, 79, 159])
    for volume in [0, 47, 161, 170, 200, float("nan")]:
        with pytest.raises(ValueError, match="outside"):
            curve.angle(volume)
    for angle in [44, 65, float("nan")]:
        with pytest.raises(ValueError, match="outside"):
            curve.volume(angle)
    with pytest.raises(ValueError, match="increase"):
        EndpointCurve([45, 50, 60], [48, 100, 90])
