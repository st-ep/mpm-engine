"""Keep interpolation distinct from a forward check and guard data provenance."""
import hashlib
import json

import pytest
from experiments.pour import pour_weakform_table as table


def run(angle, volume):
    return {"provenance": {"angle_deg": angle}, "receiver_ml": volume}


def test_interpolated_commands_are_not_forward_checks():
    rows = table.proposed_targets([run(45, 40), run(50, 80), run(60, 160)])
    assert rows[0]["command_angle_deg"] == 47.5
    assert rows[0]["status"] == "needs_forward_run"
    assert rows[1]["status"] == "forward_checked"
    assert rows[-1]["status"] == "forward_checked"
    assert "result" not in rows[0]


def test_no_extrapolation_or_hidden_nonmonotonicity():
    rows = table.proposed_targets([run(50, 81), run(60, 150)])
    assert rows[0]["status"] == "unbracketed"
    assert rows[-1]["status"] == "unbracketed"
    with pytest.raises(ValueError, match="strictly increasing"):
        table.proposed_targets([run(45, 40), run(50, 80), run(51, 79), run(60, 160)])


@pytest.fixture
def identification(tmp_path, monkeypatch):
    monkeypatch.setattr(table, "ROOT", tmp_path)
    monkeypatch.setattr(table, "WORK", tmp_path / "work")
    location = tmp_path / "work/identified"
    location.mkdir(parents=True)
    source = tmp_path / "input.json"
    source.write_text("original input")
    record = dict(channel="brink-integrated", calibration_pour_count=1,
                  measured_endpoints_used=[], validation_measurements_used=[],
                  viscosity_fitted_by_simulator=False, eta=3.4,
                  input_sha256={"input.json": hashlib.sha256(source.read_bytes()).hexdigest()})
    path = location / "identify.json"
    path.write_text(json.dumps(record))
    return source, path, record


def test_input_change_invalidates_identification(identification):
    source, _, _ = identification
    table.read_results()
    source.write_text("modified input")
    with pytest.raises(ValueError, match="Identification input changed"):
        table.read_results()


@pytest.mark.parametrize("field,value", [
    ("measured_endpoints_used", [159]),
    ("validation_measurements_used", [79]),
    ("viscosity_fitted_by_simulator", True),
    ("calibration_pour_count", 3),
    ("channel", "empirical"),
])
def test_reject_calibration_substitutions(identification, field, value):
    _, path, record = identification
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="One-pour weak-form"):
        table.read_results()


@pytest.mark.parametrize("shifted_volume,passes", [(105, True), (111, False)])
def test_resolution_average_cannot_hide_origin_sensitivity(
        identification, shifted_volume, passes):
    source, _, record = identification
    for grid, shift, volume in [(320, 0, 100), (320, .5, 108),
                                (384, 0, 103), (384, .5, shifted_volume)]:
        location = table.WORK / f"mpm/n{grid}_shift{shift:g}/angle_055.00"
        location.mkdir(parents=True)
        result = dict(provenance=dict(eta_pa_s=3.4, n_grid=grid,
                                     grid_shift_fraction_xz=shift, angle_deg=55,
                                     hashes={source.name: record["input_sha256"][source.name]}),
                      receiver_ml=volume, particle_count_drift=0,
                      tail_variation_ml=0, outside_ml=0)
        (location / "result.json").write_text(json.dumps(result))
    if passes:
        checks = table.screen_numerics(384)
        assert checks[0]["difference_ml"] == 0
        assert checks[1]["difference_ml"] == 2
    else:
        with pytest.raises(ValueError, match="grid origin changes volume"):
            table.screen_numerics(384)


def test_existing_handoff_entrypoint_uses_checked_weakform_export(monkeypatch):
    from experiments.pour import pour_angle_handoff

    def empirical_must_not_run():
        pytest.fail("Default handoff called the superseded empirical model")

    monkeypatch.setattr(pour_angle_handoff, "export_model", empirical_must_not_run)
    marker = object()
    monkeypatch.setattr(table, "export", lambda: marker)
    assert pour_angle_handoff.export() is marker


def test_failed_reference_blocks_export_before_writing(monkeypatch,tmp_path):
    monkeypatch.setattr(table,"WORK",tmp_path)
    identification={"identification_episode":"09-04-60-2s"}
    monkeypatch.setattr(table,"read_results",lambda grid: (identification,[run(60,174.9)]))
    with pytest.raises(ValueError,match="Reference replay differs"):
        table.export()
    assert not (tmp_path/"philip").exists()


def test_reference_check_requires_matching_episode_and_replay():
    identification={"identification_episode":"09-04-60-2s"}
    with pytest.raises(ValueError,match="Exactly one replay"):
        table.screen_reference(identification,[run(55,130)])
    with pytest.raises(ValueError,match="does not match"):
        table.screen_reference({"identification_episode":"another"},[run(60,159)])
    check=table.screen_reference(identification,[run(60,162)])
    assert check["difference_ml"] == 3
    assert "not independent validation" in check["purpose"]


@pytest.mark.parametrize("error,within_goal", [(7., True), (-7., True), (8., False), (10., False), (-10., False)])
def test_reference_goal_and_hard_limit_are_distinct(error, within_goal):
    check = table.screen_reference({"identification_episode": "09-04-60-2s"},
                                   [run(60, 159 + error)])
    assert check["within_desired_tolerance"] is within_goal


@pytest.mark.parametrize("error", [10.01, -10.01])
def test_reference_rejects_either_side_of_hard_limit(error):
    with pytest.raises(ValueError, match="Reference replay differs"):
        table.screen_reference({"identification_episode": "09-04-60-2s"},
                               [run(60, 159 + error)])
