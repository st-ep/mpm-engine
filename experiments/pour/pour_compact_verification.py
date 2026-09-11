"""Fixed numerical verification of full-scene MPM; never exports robot commands.

Freeze before inspecting the compact-domain endpoint. The worker has no optimizer
and runs every declared resolution/phase at the same one-video weak viscosity.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/pour_physics_audit/compact"
ORIGINAL = ROOT / "out/pour_physics_audit/replay/baseline_source_sticky_n384"
CROP_LIMITS = dict(final_receiver_ml=1., final_source_depletion_ml=1., receiver_curve_rms_ml=1.5)
NUMERICAL_LIMIT_ML = 3.
PHASES = [0., .5]
GRIDS = [320, 384]
SDF_RESOLUTION = 384


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def read_curve(directory):
    with (directory / "09-04-60-2s/metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    counts = np.array([[int(r[k]) for k in ["n_src", "n_rcv", "n_air_spill"]] for r in rows])
    if np.any(counts < 0) or np.ptp(counts.sum(1)):
        raise ValueError("Invalid particle ledger")
    n = int(counts[0].sum())
    t = np.array([float(r["t"]) for r in rows])
    return t, counts * 300. / n, n


def crop_metrics(original, cropped, original_curve, cropped_curve):
    """Compare identical physical discretizations, without experimental amounts."""
    if original["eta_pa_s"] != cropped["eta_pa_s"]:
        raise ValueError("Crop verification requires identical viscosity")
    for key in ["initial_volume_ml", "source_wall", "receiver_wall", "particle_count"]:
        if original[key] != cropped[key]:
            raise ValueError(f"Crop verification changed {key}")
    if not np.isclose(cropped["dx_m"], .7 / original["grid"], rtol=0, atol=1e-15):
        raise ValueError("Crop verification changed physical cell size")
    if cropped["phase_cells"] != 0 or cropped["sdf_resolution"] != 160:
        raise ValueError("Crop verification changed grid phase or SDF")
    if cropped["minimum_fluid_boundary_clearance_m"] <= 3 * cropped["dx_m"]:
        raise ValueError("Liquid reached artificial domain boundary")
    t0, c0, n0 = original_curve
    t1, c1, n1 = cropped_curve
    if n0 != n1 or n0 != original["particle_count"] or not np.array_equal(t0, t1):
        raise ValueError("Incompatible particle counts or motion timestamps")
    differences = dict(final_receiver_ml=abs(cropped["receiver_ml"] - original["receiver_ml"]),
                       final_source_depletion_ml=abs(cropped["source_depletion_ml"] - original["source_depletion_ml"]),
                       receiver_curve_rms_ml=float(np.sqrt(np.mean((c1[:, 1] - c0[:, 1]) ** 2))))
    passed = all(np.isfinite(v) and v <= CROP_LIMITS[k] for k, v in differences.items())
    return dict(passed=passed, differences=differences, limits=CROP_LIMITS,
                measured_outcomes_used=False, purpose="Numerical equivalence of full and compact domains")


def check_crop():
    original = json.loads((ORIGINAL / "result.json").read_text())
    cropped = json.loads((OUT / "crop_check/result.json").read_text())
    verify_hashes(original["input_sha256"])
    verify_hashes(cropped["input_sha256"])
    common = original["input_sha256"].keys() & cropped["input_sha256"].keys()
    if any(original["input_sha256"][k] != cropped["input_sha256"][k] for k in common):
        raise ValueError("Shared crop inputs differ from original replay")
    result = crop_metrics(original, cropped, read_curve(ORIGINAL), read_curve(OUT / "crop_check"))
    result["input_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in
                              [ORIGINAL / "result.json", OUT / "crop_check/result.json",
                               ORIGINAL / "09-04-60-2s/metrics.csv",
                               OUT / "crop_check/09-04-60-2s/metrics.csv"]}
    write(OUT / "crop_acceptance.json", result)
    if not result["passed"]:
        raise ValueError(f"Crop verification failed: {result['differences']}")
    return result


def freeze():
    destination = OUT / "refinement_protocol.json"
    if destination.exists():
        raise FileExistsError("Keep the original protocol; do not refreeze after results")
    if any((OUT / f"refined_n{n}_phase{p:g}_sdf{SDF_RESOLUTION}").exists() for n in GRIDS for p in PHASES):
        raise ValueError("Refinement already started; cannot claim a prior freeze")
    identification = ROOT / "out/pour_physics_audit/optical_full_rate/results.json"
    identity = json.loads(identification.read_text())
    p = identity["protocol"]
    if p["calibration_pour_count"] != 1 or p["measured_endpoints_used"] or p["other_recordings_used"]:
        raise ValueError("One-video, endpoint-free identification required")
    verify_hashes(p["input_sha256"])
    paths = [Path(__file__), ROOT / "experiments/pour/pour_compact_reference.py",
             ROOT / "examples/pour_recorded_twin.py", identification,
             ROOT / "experiments/pour/pour_transport_identify.py",
             ROOT / "experiments/pour/pour_weakform_transport.py",
             ROOT / "experiments/pour/pour_weakform_identify.py",
             *sorted((ROOT / "src/warpmpm").rglob("*.py"))]
    protocol = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                    episode="09-04-60-2s", initial_volume_ml=300., calibration_pour_count=1,
                    eta_pa_s=identity["cases"][0]["eta_pa_s"],
                    measured_endpoints_used_for_fitting=[], six_validation_outcomes_used=False,
                    simulator_inverse_fit=False, source_wall="sticky", receiver_wall="separable",
                    extent_m=.35, grids=GRIDS, phases_xz_cells=PHASES, sdf_resolution=SDF_RESOLUTION,
                    crop_limits=CROP_LIMITS, numerical_max_difference_ml=NUMERICAL_LIMIT_ML,
                    target_physical_error_ml=7., maximum_physical_error_ml=10.,
                    grid_selection="Use finer grid and phase 0 only if both phases pass; never choose by endpoint agreement",
                    execution="Both resolutions at both phases; fresh settling; exact recorded motion; stop on failures",
                    automatic_robot_export=False,
                    input_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths})
    protocol["input_sha256"].update(p["input_sha256"])
    write(destination, protocol)
    return protocol


def collect():
    protocol = json.loads((OUT / "refinement_protocol.json").read_text())
    verify_hashes(protocol["input_sha256"])
    paths = {(n, p): OUT / f"refined_n{n}_phase{p:g}_sdf{SDF_RESOLUTION}/result.json"
             for n in GRIDS for p in PHASES}
    missing = [str(p.relative_to(ROOT)) for p in paths.values() if not p.exists()]
    if missing:
        return dict(status="Refinement incomplete", missing=missing, robot_table_released=False)
    results = {key: json.loads(path.read_text()) for key, path in paths.items()}
    for (n, phase), r in results.items():
        verify_hashes(r["input_sha256"])
        if (r["eta_pa_s"] != protocol["eta_pa_s"] or r["source_wall"] != "sticky"
                or r["receiver_wall"] != "separable" or r["n_grid"] != n
                or r["phase_cells"] != phase or r["extent_m"] != .35
                or r["sdf_resolution"] != SDF_RESOLUTION):
            raise ValueError("Refinement deviated from the frozen protocol")
        read_curve(paths[(n, phase)].parent)
    checks = {f"refinement_phase{p:g}_ml": abs(results[(384, p)]["receiver_ml"] - results[(320, p)]["receiver_ml"])
              for p in PHASES}
    checks["fine_grid_phase_difference_ml"] = abs(results[(384, 0.)]["receiver_ml"] - results[(384, .5)]["receiver_ml"])
    tails_ok = all(r["tail_variation_ml"] <= .2 for r in results.values())
    boundaries_ok = all(r["minimum_fluid_boundary_clearance_m"] > 3 * r["dx_m"] for r in results.values())
    spill_ok = all(r["outside_ml"] <= 2 for r in results.values())
    passed = all(np.isfinite(v) and v <= NUMERICAL_LIMIT_ML for v in checks.values()) and tails_ok and boundaries_ok and spill_ok
    # This amount is consulted only after running the whole predeclared matrix.
    # It rejects inconsistent predictions; it cannot change any parameter.
    physical_difference = results[(384, 0.)]["receiver_ml"] - 159.
    summary = dict(status="Completed numerical diagnostics; table remains on hold",
                   numerical_screen_passed=passed, differences_ml=checks,
                   tails_ok=tails_ok, boundaries_ok=boundaries_ok, outside_volume_ok=spill_ok,
                   physical_reference_difference_ml=physical_difference,
                   physical_reference_within_goal=abs(physical_difference) <= 7.,
                   physical_reference_within_hard_limit=abs(physical_difference) <= 10.,
                   independent_validation=False, robot_table_released=False,
                   results=[{k: r[k] for k in ["n_grid", "phase_cells", "eta_pa_s", "dx_m", "receiver_ml",
                                              "source_depletion_ml", "outside_ml", "tail_variation_ml", "elapsed_s"]}
                            for r in results.values()],
                   input_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths.values()})
    write(OUT / "refinement_summary.json", summary)
    return summary


def worker(phase, device):
    protocol = json.loads((OUT / "refinement_protocol.json").read_text())
    verify_hashes(protocol["input_sha256"])
    check_crop()
    status_path = OUT / f"worker_phase{phase:g}.json"
    for n in GRIDS:
        verify_hashes(protocol["input_sha256"])
        name = f"refined_n{n}_phase{phase:g}_sdf{SDF_RESOLUTION}"
        destination = OUT / name
        write(status_path, dict(status="running", case=name, device=device,
                                updated_at_utc=datetime.now(timezone.utc).isoformat()))
        cmd = [sys.executable, "-u", "-m", "experiments.pour.pour_compact_reference", "--grid", str(n),
               "--phase", str(phase), "--sdf-res", str(SDF_RESOLUTION), "--refined-identification", "--device", device]
        with (OUT / f"{name}.log").open("x") as log:
            completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode or not (destination / "result.json").exists():
            write(status_path, dict(status="failed", case=name, returncode=completed.returncode))
            raise RuntimeError(f"Refinement failed; inspect {name}.log. Do not skip failed cases.")
        print(f"Completed {name}", flush=True)
    write(status_path, dict(status="completed", device=device, phase=phase,
                            updated_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(collect(), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--check-crop", action="store_true")
    actions.add_argument("--collect", action="store_true")
    actions.add_argument("--run-phase", type=float, choices=PHASES)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze(), indent=2))
    elif args.check_crop:
        print(json.dumps(check_crop(), indent=2))
    elif args.collect:
        print(json.dumps(collect(), indent=2))
    else:
        worker(args.run_phase, args.device)
