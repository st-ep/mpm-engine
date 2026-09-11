"""Reject invalid forward-search brackets and retain calibration status safely."""
import json

import pytest

from experiments.pour import pour_navier_calibrate as calibration


@pytest.mark.parametrize("low,high,target", [
    ((1., 160.), (4., 140.), 159.),
    ((1., 120.), (1., 170.), 159.),
    ((1., 120.), (4., 140.), 159.),
])
def test_rejects_nonmonotone_or_unbracketed_target(low, high, target):
    with pytest.raises(ValueError, match="increasing simulation bracket"):
        calibration.interpolate_trial(low, high, target, 3)


def test_secant_trial_stays_inside_bracket():
    assert calibration.interpolate_trial((1., 120.), (4., 180.), 159., 3) == 2.95
    assert 1. < calibration.interpolate_trial((1., 120.), (4., 180.), 120., 3) < 4.


def test_status_accepts_frozen_parameter_record(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "STATUS", tmp_path / "status.json")
    calibration.status("calibrated", eta_pa_s=calibration.ETA, slip_length_mm=1.)
    saved = json.loads(calibration.STATUS.read_text())
    assert saved["stage"] == "calibrated"
    assert saved["eta_pa_s"] == calibration.ETA


def test_unjustified_parameter_range_blocks_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "OUT", tmp_path)
    with pytest.raises(ValueError, match="fitting disabled"):
        calibration.require_justified_range()
    (tmp_path / "parameter_range_review.json").write_text('{"calibration_fit_allowed": false}')
    with pytest.raises(ValueError, match="fitting disabled"):
        calibration.require_justified_range()
