"""Package existing, frozen MPM commands for the user-requested robot pilot.

This is a separate pilot release after failed numerical checks. It does not change
the earlier acceptance criteria, mark them passed, fit data, or run simulations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import time
import zipfile

from experiments.pour.pour_coulomb_overnight import (
    ROOT, OUT, ETA, CONTACT, TARGETS, check_result_provenance, digest, numerical_problem,
)

SOURCE = OUT / "overnight_targets"
DESTINATION = OUT / "philip_pilot_20260907"

NOTES = """PILOT POURING TESTS — 60, 80, 100, 120, 140, 160 mL

Use angle_table.csv: target volume, commanded pour angle, tilt-command duration.
Before EACH run, fill the source cup to 300 mL and empty the receiver.
Use the same starting pose, cup mounting and robot planner as the 60-degree
identification recording. Keep the cup in the gripper throughout the session.

Tilt using the angle and duration in the table.
Hold 2.0 seconds AFTER the tilt acknowledgement.
Then return with a 2.0-second return command.

Record joint positions, action timestamps, side video and the final received
amount for each target. Preferably also record receiver weights before and after.
Keep the commands unchanged throughout the six trials.

Each command has a full MPM simulation. One 60-degree pour supplied the weak-form
viscosity and the contact calibration; the six earlier validation measurements
were not used to fit these commands. The graph connects simulated volumes.

This is a pilot: numerical sensitivity remains unresolved, and 7–10 mL accuracy
has not been established. Measure and retain the outcome of every trial.
"""


def main():
    predictions = json.loads((SOURCE / "predictions.json").read_text())
    model = predictions["model"]
    if (model["eta_pa_s"] != ETA or model["contact_coefficient"] != CONTACT
            or model["calibration_pour_count"] != 1
            or predictions["validation_outcomes_used"] != []
            or model["validation_outcomes_used"] != []):
        raise RuntimeError("Unexpected model or calibration inputs")
    with (SOURCE / "angle_table.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["target_ml", "command_angle_deg", "tilt_duration_s"]
        table = list(reader)
    assert [int(r["target_ml"]) for r in table] == TARGETS
    assert len(predictions["predictions"]) == len(table)
    for row, prediction in zip(table, predictions["predictions"]):
        assert int(row["target_ml"]) == prediction["target_ml"]
        assert float(row["command_angle_deg"]) == prediction["command_angle_deg"]
        assert float(row["tilt_duration_s"]) == prediction["tilt_duration_s"]
        path = ROOT / prediction["result_path"]
        assert digest(path) == prediction["result_sha256"]
        result = json.loads(path.read_text())
        check_result_provenance(result, prediction["command_angle_deg"], 160, 0., 1.)
        assert numerical_problem(result) is None
        assert result["receiver_ml"] == prediction["predicted_ml"]
        assert abs(result["receiver_ml"] - prediction["target_ml"]) <= 1.
    DESTINATION.mkdir(exist_ok=False)
    for name in ["angle_table.csv", "angle_table.png"]:
        shutil.copyfile(SOURCE / name, DESTINATION / name)
    (DESTINATION / "START_HERE.txt").write_text(NOTES)
    names = ["angle_table.csv", "angle_table.png", "START_HERE.txt"]
    archive = DESTINATION / "philip_pilot_handoff.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(DESTINATION / name, arcname=name)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == names and z.testzip() is None
        for name in names:
            assert z.read(name) == (DESTINATION / name).read_bytes()
    # Keep research provenance outside Philip's three-file package.
    provenance = dict(
        created_unix=time.time(), purpose="User-requested prospective robot pilot",
        release_basis="User requested completing the table with the existing calibrated settings after reviewing failed time-step checks",
        release_status="Pilot only; numerical checks remain failed and robot accuracy is unestablished",
        numerical_acceptance_passed=False, robot_accuracy_validated=False,
        previous_acceptance_criteria_changed=False, simulation_parameters_changed=False,
        commands_changed=False, extra_simulations=0, parameters_refitted=[],
        validation_outcomes_used=[],
        model=dict(eta_pa_s=ETA, source_contact=CONTACT, grid=160, dt_scale=1.,
                   calibration_episode="09-04-60-2s", calibration_pour_count=1),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [
            SOURCE / "predictions.json", SOURCE / "angle_table.csv", SOURCE / "angle_table.png",
            OUT / "recorded60_timestep_followup/comparison.json", Path(__file__)]},
        predictions=predictions["predictions"],
        package_files_sha256={name: digest(DESTINATION / name) for name in names},
        zip_sha256=digest(archive), zip_bytes=archive.stat().st_size,
    )
    (DESTINATION / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                         files=names, status=provenance["release_status"]), indent=2))


if __name__ == "__main__":
    main()
