"""Overnight planning must not hide reversals, reuse bad cases or release unchecked commands."""
import json

import pytest

from experiments.pour import pour_coulomb_overnight as run


def test_bracket_search_keeps_angle_order_and_rejects_missing_brackets():
    assert run.next_angle([(50, 80), (47, 60)], 70) == 48.5
    with pytest.raises(run.CaseFailure, match="No simulated increasing bracket"):
        run.next_angle([(47, 80), (50, 60)], 70)
    with pytest.raises(run.CaseFailure, match="No simulated increasing bracket"):
        run.next_angle([(47, 60), (50, 80)], 100)


def test_repeated_or_failed_candidate_is_not_automatically_retried():
    candidate = run.next_angle([(47, 60), (50, 80)], 70, attempted=[48.5])
    assert 47 < candidate < 50 and candidate != 48.5
    with pytest.raises(run.CaseFailure, match="precision"):
        run.next_angle([(48.36, 58), (48.37, 62)], 60)


@pytest.mark.parametrize("field,value", [
    ("eta_pa_s", 3.4), ("source_coulomb_friction", .1),
    ("validation_outcomes_used", [63.]), ("measured_endpoints_read_by_forward_runner", [159.]),
])
def test_result_cannot_change_model_or_read_validation_endpoints(field, value):
    result = dict(eta_pa_s=run.ETA, source_coulomb_friction=run.CONTACT,
        initial_volume_ml=300., source_wall="original-separable", n_grid=160,
        phase_xz_cells=0., dt_scale=1., identification_episode="09-04-60-2s",
        planned_angle_deg=48.37, validation_outcomes_used=[],
        measured_endpoints_read_by_forward_runner=[], input_sha256={})
    result[field] = value
    with pytest.raises(run.ProvenanceError):
        run.check_result_provenance(result, 48.37, 160, 0., 1.)


def test_missing_and_nonfinite_numerical_results_are_not_accepted():
    assert run.numerical_problem({})
    assert run.numerical_problem(dict(receiver_ml=float("nan"), outside_ml=0, tail_variation_ml=0))
    assert run.numerical_problem(dict(receiver_ml=60., outside_ml=3.1, tail_variation_ml=0))
    assert run.numerical_problem(dict(receiver_ml=60., outside_ml=0, tail_variation_ml=.6))
    assert run.numerical_problem(dict(receiver_ml=60., outside_ml=0, tail_variation_ml=.1)) is None


def dummy_search():
    search = object.__new__(run.Search)
    search.protocol = dict(half_dt_targets_ml=[60, 120, 160])
    search.state = dict(cases={}, targets={}, checks={})
    for angle in run.ANCHORS:
        search.state["cases"][str(angle)] = dict(status="complete", angle_deg=angle,
            receiver_ml=60.+(angle-48.)*10, grid=160, phase=0., dt_scale=1.)
    for target in run.TARGETS:
        angle = 48.+(target-60.)/10
        search.state["targets"][str(target)] = dict(target_ml=target, command_angle_deg=angle,
            predicted_ml=float(target), prediction_error_ml=0., tilt_duration_s=angle/10)
        for kind in ["grid_192", "half_cell_shift"] + (["half_time_step"] if target in [60,120,160] else []):
            search.state["checks"][f"{target}_{kind}"] = dict(passed=True)
    return search


def test_report_withholds_zip_when_numerical_check_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "WORK", tmp_path)
    search = dummy_search()
    search.state["checks"]["60_grid_192"]["passed"] = False
    assert search.report() is False
    assert not (tmp_path / "philip_handoff.zip").exists()
    assert json.loads((tmp_path / "predictions.json").read_text())["handoff_ready"] is False


def test_report_preserves_curve_reversal_and_does_not_release(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "WORK", tmp_path)
    search = dummy_search()
    search.state["cases"]["50.0"]["receiver_ml"] = 65.
    assert search.report() is False
    saved = json.loads((tmp_path / "predictions.json").read_text())
    assert saved["curve_anchors_strictly_increasing"] is False
    assert search.state["cases"]["50.0"]["receiver_ml"] == 65.
    assert not (tmp_path / "philip_handoff.zip").exists()


def test_completed_handoff_contains_only_three_requested_files(tmp_path, monkeypatch):
    import zipfile
    monkeypatch.setattr(run, "WORK", tmp_path)
    assert dummy_search().report() is True
    with zipfile.ZipFile(tmp_path / "philip_handoff.zip") as z:
        assert sorted(z.namelist()) == ["START_HERE.txt", "angle_table.csv", "angle_table.png"]


def test_numerical_checks_cannot_mask_a_missed_volume_target(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "WORK", tmp_path)
    search = dummy_search()
    search.state["targets"]["60"]["predicted_ml"] = 70.
    assert search.report() is False
    assert not (tmp_path / "philip_handoff.zip").exists()


def test_fatal_error_preserves_but_withdraws_previous_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "WORK", tmp_path)
    search = dummy_search()
    assert search.report() is True
    search.state["fatal_error"] = "Frozen input changed"
    assert search.report() is False
    assert not (tmp_path / "philip_handoff.zip").exists()
    assert len(list(tmp_path.glob("superseded_handoff_*.zip"))) == 1
