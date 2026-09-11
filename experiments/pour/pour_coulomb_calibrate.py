"""One-pour calibration of the original engine contact, then one target prediction.

Viscosity comes from the unchanged weak spout fit. The same pour's 159 mL endpoint
calibrates only a bounded effective numerical contact coefficient. This does not
identify a microscopic liquid-wall friction law. No validation outcomes are read.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time

from experiments.pour.pour_navier_reference import ETA, OUT, ROOT
from experiments.pour.pour_navier_calibrate import interpolate_trial, status, write


def folder(mu, grid=160, angle=None):
    label = "recorded60" if angle is None else f"planned{angle:g}"
    name = f"{label}_bNA_n{grid}_phase0_dt1"
    if mu != .05:
        name += f"_mu{mu:.6f}"
    return OUT / "replays/original-separable" / name


def read(path):
    result = json.loads((path / "result.json").read_text())
    if (result["eta_pa_s"] != ETA or result["initial_volume_ml"] != 300.
            or result["source_wall"] != "original-separable"
            or result["validation_outcomes_used"]
            or result["measured_endpoints_read_by_forward_runner"]):
        raise ValueError("Invalid one-pour forward provenance")
    if result["outside_ml"] > 3 or result["tail_variation_ml"] > .5:
        raise ValueError("Capture or settled-tail check failed")
    for relative, expected in result["input_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual == expected:
            continue
        snapshot = OUT / "code_snapshots" / f"pour_navier_reference_{expected}.py"
        if (relative != "experiments/pour/pour_navier_reference.py" or not snapshot.exists()
                or hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected):
            raise ValueError(f"Unaccounted changed forward input: {relative}")
    return result


def replay(mu, grid=160, angle=None):
    if not .05 <= mu <= .2:
        raise ValueError("No extension beyond the predeclared original-contact interval")
    path = folder(mu, grid, angle)
    if (path / "result.json").exists():
        return read(path)
    if path.exists():
        if (mu, grid, angle) != (.2, 160, None):
            raise ValueError(f"Unfinished prior case: {path}")
        deadline = time.monotonic() + 1800
        status("waiting_for_original_contact_probe", source_coulomb_friction=mu)
        while not (path / "result.json").exists():
            active = subprocess.run(["systemctl", "--user", "is-active", "pour-original-friction-sensitivity.service"],
                                    capture_output=True, text=True).stdout.strip()
            if active != "active" or time.monotonic() > deadline:
                raise RuntimeError("Contact sensitivity probe stopped without a result")
            time.sleep(10)
        return read(path)
    log = OUT / "logs" / (path.name + ".log")
    command = [sys.executable, "-u", "experiments/pour/pour_navier_reference.py",
               "--wall", "original-separable", "--source-friction", f"{mu:g}",
               "--grid", str(grid), "--device", "cuda:1"]
    if angle is not None:
        command += ["--angle", f"{angle:g}"]
    status("running_original_contact_replay", source_coulomb_friction=mu, n_grid=grid,
           angle_deg=angle, log=str(log))
    with log.open("w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=2400)
    return read(path)


def main():
    protocol = json.loads((OUT / "coulomb_calibration_protocol.json").read_text())
    if protocol["eta_pa_s"] != ETA or protocol["validation_outcomes_used"]:
        raise ValueError("Invalid calibration protocol")
    target, tolerance = protocol["calibration_endpoint_ml"], protocol["endpoint_tolerance_ml"]
    samples = [replay(.05), replay(.2)]
    write(OUT / "coulomb_calibration_samples.json", samples)
    # More contact resistance is expected to lower volume. Reject reversals or an
    # absent bracket instead of forcing a fitted curve through unsuitable trials.
    low, high = (.05, -samples[0]["receiver_ml"]), (.2, -samples[1]["receiver_ml"])
    best = min(samples, key=lambda r:abs(r["receiver_ml"]-target))
    for _ in range(4):
        if abs(best["receiver_ml"]-target) <= tolerance:
            break
        mu = interpolate_trial(low, high, -target, 3)
        if any(r["source_coulomb_friction"] == mu for r in samples):
            raise RuntimeError("Contact search stalled at declared precision")
        value = replay(mu)
        samples.append(value)
        best = min(samples, key=lambda r:abs(r["receiver_ml"]-target))
        if value["receiver_ml"] > target:
            low = (mu, -value["receiver_ml"])
        else:
            high = (mu, -value["receiver_ml"])
        write(OUT / "coulomb_calibration_samples.json", samples)
    if abs(best["receiver_ml"]-target) > tolerance:
        raise RuntimeError("No endpoint match within the bounded contact search")
    mu = best["source_coulomb_friction"]
    status("coulomb_endpoint_matched_numerical_check_pending", receiver_ml=best["receiver_ml"],
           source_coulomb_friction=mu, measured_ml=target)
    fine = replay(mu, grid=192)
    difference = abs(fine["receiver_ml"]-best["receiver_ml"])
    comparison = dict(coarse=best, finer=fine, difference_ml=difference,
                      limit_ml=protocol["grid_difference_limit_ml"],
                      passed=difference <= protocol["grid_difference_limit_ml"])
    write(OUT / "coulomb_grid_check.json", comparison)
    if not comparison["passed"]:
        raise RuntimeError("Contact fit failed the 160/192 volume check; no finer runs or angle release")
    selected = dict(eta_pa_s=ETA, source_coulomb_friction=mu, n_grid=160,
                    calibration_episode="09-04-60-2s", calibration_pour_count=1,
                    calibration_endpoint_ml=target, simulated_endpoint_ml=best["receiver_ml"],
                    numerical_difference_ml=difference, validation_outcomes_used=[],
                    interpretation="Weak-form effective viscosity plus same-pour numerical contact calibration")
    write(OUT / "selected_coulomb_boundary.json", selected)
    status("coulomb_calibrated_starting_60ml_prediction", **selected)
    first = replay(mu, angle=45.)
    angles = [first]
    if first["receiver_ml"] < 60.:
        low, high = (45., first["receiver_ml"]), (60., best["receiver_ml"])
    else:
        lower = replay(mu, angle=35.)
        angles.append(lower)
        low, high = (35., lower["receiver_ml"]), (45., first["receiver_ml"])
    best_angle = min(angles, key=lambda r:abs(r["receiver_ml"]-60.))
    for _ in range(4):
        if abs(best_angle["receiver_ml"]-60.) <= 1.:
            break
        angle = interpolate_trial(low, high, 60., 2)
        if any(r["planned_angle_deg"] == angle for r in angles):
            raise RuntimeError("Angle search stalled")
        value = replay(mu, angle=angle)
        angles.append(value)
        best_angle = min(angles, key=lambda r:abs(r["receiver_ml"]-60.))
        if value["receiver_ml"] < 60.:
            low = (angle, value["receiver_ml"])
        else:
            high = (angle, value["receiver_ml"])
        write(OUT / "coulomb_60ml_samples.json", angles)
    write(OUT / "coulomb_60ml_samples.json", angles)
    if abs(best_angle["receiver_ml"]-60.) > 1.:
        raise RuntimeError("No 60 mL command within the bounded MPM search")
    prediction = dict(**selected, target_ml=60., predicted_ml=best_angle["receiver_ml"],
                      command_angle_deg=best_angle["planned_angle_deg"],
                      incline_duration_s=best_angle["planned_angle_deg"]/10., dwell_s=2., return_s=2.,
                      initial_volume_ml=300., physical_validation_done=False,
                      angle_selected_using="MPM outputs only")
    write(OUT / "prediction_60ml_coulomb.json", prediction)
    status("completed_60ml_prediction", prediction=prediction)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("needs_review", error_type=type(error).__name__, reason=str(error))
        raise
