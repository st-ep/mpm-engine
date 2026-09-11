"""Bounded one-pour boundary calibration, then an MPM prediction for 60 mL.

The only measured outcome read here is the 60-degree/159 mL calibration endpoint.
The angle search uses simulated volumes only. No robot commands are sent.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time

from experiments.pour.pour_navier_reference import ROOT, OUT, ETA

STATUS = OUT / "status.json"


def require_justified_range():
    review = OUT / "parameter_range_review.json"
    if not review.exists() or not json.loads(review.read_text()).get("calibration_fit_allowed", False):
        raise ValueError("Parameter range lacks an independent physical/numerical justification; fitting disabled")


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def status(stage, **details):
    value = dict(stage=stage, updated_unix=time.time(), eta_pa_s=ETA)
    value.update(details)
    write(STATUS, value)
    print(json.dumps(value), flush=True)


def case_dir(slip, grid=160, angle=None):
    label = "recorded60" if angle is None else f"planned{angle:g}"
    return OUT / "replays/wet-traction" / f"{label}_b{slip:.6f}_n{grid}_phase0_dt1"


def read_result(folder):
    result = json.loads((folder / "result.json").read_text())
    if result["eta_pa_s"] != ETA or result["validation_outcomes_used"]:
        raise ValueError("Invalid fixed-viscosity/provenance record")
    if result["source_wall"] != "wet-traction" or result["measured_endpoints_read_by_forward_runner"]:
        raise ValueError("Invalid wall model or measured inputs to forward simulation")
    for relative, recorded_hash in result["input_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != recorded_hash:
            raise ValueError(f"Forward inputs changed since simulation: {relative}")
    if result["tail_variation_ml"] > .5:
        raise ValueError("Receiver was not settled to the declared 0.5 mL tolerance")
    if result["outside_ml"] > 3.:
        raise ValueError("More than 3 mL outside both cups; review capture before fitting")
    return result


def replay(slip, grid=160, angle=None):
    folder = case_dir(slip, grid, angle)
    if (folder / "result.json").exists():
        return read_result(folder)
    if folder.exists():
        # The initial probe was started separately; retain its detached service.
        if slip != 1. or grid != 160 or angle is not None:
            raise ValueError(f"Incomplete prior case requires review: {folder}")
        status("waiting_for_initial_probe", folder=str(folder))
        deadline = time.monotonic() + 1800
        while not (folder / "result.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Initial replay exceeded the monitoring time budget")
            if (folder / "failure.json").exists():
                raise RuntimeError("Initial cup replay failed; inspect saved state")
            active = subprocess.run(["systemctl", "--user", "is-active", "pour-navier-traction-n160-b1"],
                                    text=True, capture_output=True).stdout.strip()
            if active != "active":
                raise RuntimeError("Initial replay service stopped without a result")
            time.sleep(10)
        return read_result(folder)
    log = OUT / "logs" / (folder.name + ".log")
    log.parent.mkdir(exist_ok=True)
    command = [sys.executable, "-u", "experiments/pour/pour_navier_reference.py",
               "--wall", "wet-traction", "--slip-mm", f"{slip:g}", "--grid", str(grid), "--device", "cuda:1"]
    if angle is not None:
        command += ["--angle", f"{angle:g}"]
    status("running_replay", grid=grid, slip_length_mm=slip, angle_deg=angle, log=str(log))
    with log.open("w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800)
    return read_result(folder)


def interpolate_trial(low, high, target, precision):
    x0, y0 = low
    x1, y1 = high
    if not (x1 > x0 and y1 > y0 and y0 <= target <= y1):
        raise ValueError("Need an increasing simulation bracket; do not force a nonmonotone fit")
    fraction = min(.85, max(.15, (target-y0)/(y1-y0)))
    return round(x0 + fraction*(x1-x0), precision)


def main():
    require_justified_range()
    protocol = json.loads((OUT / "calibration_protocol.json").read_text())
    if protocol["weak_eta_pa_s"] != ETA or protocol["independent_validation_outcomes_used"]:
        raise ValueError("Invalid one-pour calibration protocol")
    target = float(protocol["calibration_endpoint_ml"])
    tolerance = float(protocol["endpoint_fit_tolerance_ml"])
    first = replay(1.)
    samples = [first]
    best = first
    if abs(first["receiver_ml"] - target) > tolerance:
        other_b = 4. if first["receiver_ml"] < target else .25
        second = replay(other_b)
        samples.append(second)
        best = min(samples, key=lambda r: abs(r["receiver_ml"]-target))
        ordered = sorted(samples, key=lambda r:r["slip_length_mm"])
        low = (ordered[0]["slip_length_mm"], ordered[0]["receiver_ml"])
        high = (ordered[-1]["slip_length_mm"], ordered[-1]["receiver_ml"])
        for _ in range(4):
            require_justified_range()
            if abs(best["receiver_ml"] - target) <= tolerance:
                break
            b = interpolate_trial(low, high, target, 3)
            if any(r["slip_length_mm"] == b for r in samples):
                raise RuntimeError("Slip search stalled at numerical precision")
            value = replay(b)
            samples.append(value)
            best = min(samples, key=lambda r:abs(r["receiver_ml"]-target))
            if value["receiver_ml"] < target:
                low = (b, value["receiver_ml"])
            else:
                high = (b, value["receiver_ml"])
            write(OUT / "calibration_samples.json", samples)
    write(OUT / "calibration_samples.json", samples)
    if abs(best["receiver_ml"] - target) > tolerance:
        raise RuntimeError("No 159 mL match within the bounded calibration budget")
    status("endpoint_matched_numerical_check_pending", slip_length_mm=best["slip_length_mm"],
           receiver_ml=best["receiver_ml"], measurement_ml=target)
    fine = replay(best["slip_length_mm"], grid=192)
    difference = abs(fine["receiver_ml"] - best["receiver_ml"])
    comparison = dict(coarse=best, finer=fine, difference_ml=difference,
                      limit_ml=protocol["maximum_allowed_volume_difference_ml"],
                      passed=difference <= protocol["maximum_allowed_volume_difference_ml"])
    write(OUT / "calibration_grid_check.json", comparison)
    if not comparison["passed"]:
        raise RuntimeError("160/192 volume check failed; no automatic finer runs or angle release")
    selected = dict(eta_pa_s=ETA, slip_length_mm=best["slip_length_mm"], n_grid=160,
                    calibration_episode="09-04-60-2s", calibration_endpoint_ml=target,
                    simulated_endpoint_ml=best["receiver_ml"], numerical_difference_ml=difference,
                    calibration_pour_count=1, validation_outcomes_used=[],
                    status="Calibrated on one endpoint; new-angle physical accuracy remains unvalidated")
    write(OUT / "selected_boundary.json", selected)
    status("calibration_passed_starting_60ml_prediction", **selected)
    b = selected["slip_length_mm"]
    angle_sample = replay(b, angle=45.)
    angle_samples = [angle_sample]
    if angle_sample["receiver_ml"] < 60.:
        low, high = (45., angle_sample["receiver_ml"]), (60., best["receiver_ml"])
    else:
        lower = replay(b, angle=35.)
        angle_samples.append(lower)
        low, high = (35., lower["receiver_ml"]), (45., angle_sample["receiver_ml"])
    angle_best = min(angle_samples, key=lambda r:abs(r["receiver_ml"] - 60.))
    for _ in range(4):
        if abs(angle_best["receiver_ml"] - 60.) <= 1.:
            break
        angle = interpolate_trial(low, high, 60., 2)
        if any(r["planned_angle_deg"] == angle for r in angle_samples):
            raise RuntimeError("Angle search stalled at command precision")
        value = replay(b, angle=angle)
        angle_samples.append(value)
        angle_best = min(angle_samples, key=lambda r:abs(r["receiver_ml"] - 60.))
        if value["receiver_ml"] < 60.:
            low = (angle, value["receiver_ml"])
        else:
            high = (angle, value["receiver_ml"])
        write(OUT / "angle_60ml_samples.json", angle_samples)
    write(OUT / "angle_60ml_samples.json", angle_samples)
    if abs(angle_best["receiver_ml"] - 60.) > 1.:
        raise RuntimeError("No 60 mL angle within the bounded forward-search budget")
    prediction = dict(**selected, target_ml=60., predicted_ml=angle_best["receiver_ml"],
                      angle_deg=angle_best["planned_angle_deg"], dwell_s=2.,
                      incline_duration_s=angle_best["planned_angle_deg"]/10., return_duration_s=2.,
                      initial_volume_ml=300., physical_validation_done=False,
                      angle_optimized_against="MPM outputs only", simulation_case=angle_best["name"])
    write(OUT / "prediction_60ml.json", prediction)
    status("completed", prediction=prediction)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("needs_review", error_type=type(error).__name__, reason=str(error))
        raise
