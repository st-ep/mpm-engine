"""Independent endpoints must not silently calibrate the identification curve."""

import numpy as np
import pytest
from experiments.pour.pour_validate import (
    curve_volume_limit,
    final_volume_reference,
    numbers,
)


def sample_data():
    obs = {"t": np.array([0.0, 1.0, 2.0]),
           "rcv_vol": np.array([0.0, 60.0, 126.0]) * 1e-6}
    sim = {"t_pour": np.array([0.0, 1.0, 2.0]),
           "ml_rcv": np.array([0.0, 80.0, 159.0])}
    return obs, sim


def test_reported_endpoint_overrides_video_without_rescaling_curve():
    obs, sim = sample_data()
    before = obs["rcv_vol"].copy()
    measurement = {"final_receiver_volume_ml": 159.0, "final_volume_source": "cup reading",
                   "experiment_role": "identification"}
    result = numbers(obs, sim, measurement)
    assert result["v_final_real_mL"] == 159.0
    assert result["v_final_err_mL"] == 0.0
    assert result["v_final_video_mL"] == pytest.approx(126.0)
    assert result["v_final_video_minus_reference_mL"] == pytest.approx(-33.0)
    assert result["rms_overlap_mL"] == pytest.approx(np.sqrt((20**2 + 33**2) / 3))
    assert result["rms_overlap_reference"] == "video_receiver_curve"
    assert result["v_final_reference_kind"] == "reported_measurement"
    assert result["experiment_role"] == "identification"
    assert result["v_final_uncertainty_mL"] is None
    np.testing.assert_array_equal(obs["rcv_vol"], before)


def test_legacy_video_reference_is_explicit():
    obs, sim = sample_data()
    result = numbers(obs, sim)
    assert result["v_final_real_mL"] == pytest.approx(126.0)
    assert result["v_final_err_mL"] == pytest.approx(33.0)
    assert result["v_final_reference_kind"] == "video_estimate"


def test_independent_endpoint_works_without_readable_video():
    obs, sim = sample_data()
    obs["rcv_vol"][:] = np.nan
    result = numbers(obs, sim, {"final_receiver_volume_ml": 159.0})
    assert result["v_final_err_mL"] == 0.0
    assert np.isnan(result["v_final_video_mL"])
    assert np.isnan(result["rms_overlap_mL"])
    assert result["n_real_points"] == 0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_reported_endpoint_is_rejected(value):
    obs, _ = sample_data()
    with pytest.raises(ValueError, match="final_receiver_volume_ml"):
        final_volume_reference(obs, {"final_receiver_volume_ml": value})


def test_curve_axis_uses_millilitres_and_includes_independent_endpoint():
    obs, sim = sample_data()
    sim["ml_rcv"] *= 0.5
    reference = final_volume_reference(obs, {"final_receiver_volume_ml": 159.0})
    assert curve_volume_limit(obs, sim, reference) == pytest.approx(159.0 * 1.15)
    assert curve_volume_limit(obs, sim) == pytest.approx(126.0 * 1.15)
