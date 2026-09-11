"""Plan/export target commands using forward MPM results at a frozen weak-form eta.

No measured pour endpoints enter angle interpolation or viscosity fitting. Candidate
angles interpolate numerical evaluations; exported commands require their own
forward evaluation and a passing same-pour reference check. That check is an
acceptance gate, not a fit or evidence of independent validation. Every numerical
sample remains in the research record.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import PchipInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "out/pour_weakform_recovery"
TARGETS = [60, 80, 100, 120, 140, 160]
TOLERANCE_ML = 2.0
MAX_NUMERICAL_DIFFERENCE_ML = 5.0
DESIRED_REFERENCE_DIFFERENCE_ML = 7.0
MAX_REFERENCE_DIFFERENCE_ML = 10.0
# User-confirmed observation of the identification pour. Used only to refuse an
# inconsistent handoff, never to select eta, geometry, wall settings or angles.
REFERENCE_OBSERVATION = dict(episode="09-04-60-2s", commanded_angle_deg=60.,
                             receiver_ml=159., source="User-reported settled measurement")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_results(grid=384):
    identification_path = WORK / "identified/identify.json"
    identification = json.loads(identification_path.read_text())
    if (identification["channel"] != "brink-integrated"
            or identification["calibration_pour_count"] != 1
            or identification["measured_endpoints_used"]
            or identification["validation_measurements_used"]
            or identification["viscosity_fitted_by_simulator"]):
        raise ValueError("One-pour weak-form identification is required")
    for name, expected in identification["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Identification input changed: {name}")
    results = []
    for path in sorted((WORK / f"mpm/n{grid}_shift0").glob("angle_*/result.json")):
        result = json.loads(path.read_text())
        provenance = result["provenance"]
        if (provenance["eta_pa_s"] != identification["eta"]
                or provenance["n_grid"] != grid
                or provenance["grid_shift_fraction_xz"] != 0):
            raise ValueError(f"Inconsistent forward run: {path}")
        for name, expected in provenance["hashes"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Simulation input changed: {name}")
        if result["particle_count_drift"] or result["tail_variation_ml"] > .2:
            raise ValueError(f"Unsettled/nonconserving forward run: {path}")
        if result["outside_ml"] > 2:
            raise ValueError(f"Significant spill: {path}")
        result["result_path"] = str(path.relative_to(ROOT))
        result["result_sha256"] = digest(path)
        results.append(result)
    return identification, sorted(results, key=lambda r: r["provenance"]["angle_deg"])


def proposed_targets(results):
    angles = np.array([r["provenance"]["angle_deg"] for r in results])
    volumes = np.array([r["receiver_ml"] for r in results])
    if len(angles) < 2 or np.any(np.diff(volumes) <= 0):
        raise ValueError("Need at least two strictly increasing MPM evaluations")
    forward = PchipInterpolator(angles, volumes, extrapolate=False)
    dense = np.linspace(angles[0], angles[-1], 100001)
    dense_volumes = forward(dense)
    rows = []
    for target in TARGETS:
        best = int(np.argmin(abs(volumes - target)))
        error = float(volumes[best] - target)
        if abs(error) <= TOLERANCE_ML:
            rows.append(dict(target_ml=target, command_angle_deg=float(angles[best]),
                             predicted_ml=float(volumes[best]), error_ml=error,
                             status="forward_checked", result=results[best]))
        elif target < volumes[0] or target > volumes[-1]:
            rows.append(dict(target_ml=target, status="unbracketed"))
        else:
            angle = round(float(np.interp(target, dense_volumes, dense)), 2)
            if angle in angles:
                raise ValueError(f"Angle resolution insufficient for {target} mL")
            rows.append(dict(target_ml=target, command_angle_deg=angle,
                             status="needs_forward_run"))
    return rows


def screen_numerics(grid):
    identification, _ = read_results(grid)
    checks = []
    if grid not in [256, 320, 384]:
        raise ValueError("Numerical screening protocol supports grids 256, 320 and 384")
    coarse = {256: 192, 320: 256, 384: 320}[grid]
    # At the failed middle angle, compare BOTH origin samples between resolutions:
    # an individual coarse-grid phase is an unreliable refinement reference.
    # The production origin is still fixed at zero, and the finer pair must pass
    # the same 5 mL origin screen. These are screens, not a convergence proof.
    if grid == 384:
        pairs = [("55-degree resolution, two-origin means",
                  [f"n{coarse}_shift{s:g}/angle_055.00" for s in [0, .5]],
                  [f"n{grid}_shift{s:g}/angle_055.00" for s in [0, .5]])]
    else:
        pairs = [("60-degree resolution", [f"n{coarse}_shift0/angle_060.00"],
                  [f"n{grid}_shift0/angle_060.00"])]
    pairs.append(("55-degree grid origin", [f"n{grid}_shift0/angle_055.00"],
                  [f"n{grid}_shift0.5/angle_055.00"]))
    for name, first, second in pairs:
        paths = [WORK / "mpm" / directory / "result.json" for directory in first + second]
        values = [json.loads(path.read_text()) for path in paths]
        for value in values:
            if value["provenance"]["eta_pa_s"] != identification["eta"]:
                raise ValueError("Numerical comparisons must use the frozen viscosity")
            if (value["particle_count_drift"] or value["tail_variation_ml"] > .2
                    or value["outside_ml"] > 2):
                raise ValueError("Numerical comparison has an invalid final particle ledger")
            for source, expected in value["provenance"]["hashes"].items():
                if digest(ROOT / source) != expected:
                    raise ValueError(f"Numerical check input changed: {source}")
        estimates = [float(np.mean([r["receiver_ml"] for r in values[:len(first)]])),
                     float(np.mean([r["receiver_ml"] for r in values[len(first):]]))]
        difference = abs(estimates[1] - estimates[0])
        checks.append(dict(name=name, difference_ml=difference,
                           estimates_ml=estimates,
                           result_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths}))
        if difference > MAX_NUMERICAL_DIFFERENCE_ML:
            raise ValueError(f"{name} changes volume by {difference:.2f} mL; resolve before export")
    return checks


def screen_reference(identification, results):
    if identification.get("identification_episode") != REFERENCE_OBSERVATION["episode"]:
        raise ValueError("Reference observation does not match the identification episode")
    matches = [r for r in results
               if r["provenance"]["angle_deg"] == REFERENCE_OBSERVATION["commanded_angle_deg"]]
    if len(matches) != 1:
        raise ValueError("Exactly one replay of the identification command is required before export")
    predicted = float(matches[0]["receiver_ml"])
    difference = predicted - REFERENCE_OBSERVATION["receiver_ml"]
    if not np.isfinite(predicted) or abs(difference) > MAX_REFERENCE_DIFFERENCE_ML:
        raise ValueError(f"Reference replay differs by {difference:+.2f} mL; "
                         "withhold the handoff and investigate without endpoint fitting")
    return dict(observation=REFERENCE_OBSERVATION.copy(), predicted_ml=predicted,
                difference_ml=difference, tolerance_ml=MAX_REFERENCE_DIFFERENCE_ML,
                desired_tolerance_ml=DESIRED_REFERENCE_DIFFERENCE_ML,
                within_desired_tolerance=abs(difference) <= DESIRED_REFERENCE_DIFFERENCE_ML,
                purpose="Same-pour consistency gate only; not independent validation")


def export(grid=384):
    identification, results = read_results(grid)
    reference_check = screen_reference(identification, results)
    rows = proposed_targets(results)
    if any(row["status"] != "forward_checked" for row in rows):
        raise ValueError("Every target needs its own successful forward run")
    checks = screen_numerics(grid)
    selected_angles = [row["command_angle_deg"] for row in rows]
    if np.any(np.diff(selected_angles) <= 0):
        raise ValueError("Target commands must be distinct and increasing")
    destination = WORK / "philip"
    destination.mkdir(exist_ok=True)
    table_text = io.StringIO()
    writer = csv.writer(table_text)
    writer.writerow(["target_ml", "command_angle_deg", "tilt_duration_s"])
    for row in rows:
        writer.writerow([row["target_ml"], f"{row['command_angle_deg']:.2f}",
                         f"{row['command_angle_deg']/10:.3f}"])
    (destination / "angle_table.csv").write_text(table_text.getvalue())
    instructions = f"""SIX POURING VALIDATION RUNS — WEAK-FORM + MPM

Start EVERY run with 300 mL in the source cup and an empty receiver.
Use angle_table.csv: commanded pour angle and tilt-command duration.
Hold 2.0 seconds AFTER the tilt acknowledgement, then send a 2.0-second
return command to the original upright pose. Use the existing linear planner.
Keep the liquid, temperature and cup placement in the gripper consistent with
the identification run.

Record exact joints, action timestamps and camera videos for each run.
Measure the settled receiver amount independently and save the measurement;
prefer weighing the empty and filled receiver when a balance is available.
Complete all six commands before changing the settings against their results.

These are prospective predictions from the 60-degree video's spout-edge
weak-form viscosity and full MPM runs. Experimental accuracy is to be measured.
The same-pour reference check predicts {reference_check['predicted_ml']:.1f} mL
against the reported {reference_check['observation']['receiver_ml']:.0f} mL.
This check does not establish accuracy at new angles; these runs test that.
The graph is a guide through simulated pours; the marked commands have each
been simulated directly. These files supersede the earlier empirical table.
"""
    (destination / "START_HERE.txt").write_text(instructions)
    angles = np.array([r["provenance"]["angle_deg"] for r in results])
    volumes = np.array([r["receiver_ml"] for r in results])
    curve = PchipInterpolator(angles, volumes)
    dense = np.linspace(angles[0], angles[-1], 500)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(dense, curve(dense), lw=2.1, color="#2563a6", label="MPM prediction")
    ax.scatter(angles, volumes, s=16, color="#2563a6", zorder=3)
    for i, row in enumerate(rows):
        angle, volume = row["command_angle_deg"], row["predicted_ml"]
        ax.scatter(angle, volume, s=48, color="#c7462d", zorder=4)
        left_half = angle < .5 * (angles[0] + angles[-1])
        ax.annotate(f"{row['target_ml']} mL · {angle:.2f}°", (angle, volume),
                    xytext=(8 if left_half else -8, -20 if i % 2 == 0 else 12),
                    textcoords="offset points", fontsize=9,
                    ha="left" if left_half else "right",
                    bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=2))
    ax.scatter([], [], s=48, color="#c7462d", label="Directly simulated target commands")
    ax.set(xlabel="Commanded final angle (degrees)", ylabel="Receiver volume (mL)",
           title="Pouring validation commands",
           xlim=(angles[0] - .6, angles[-1] + .7),
           ylim=(max(0, volumes[0] - 10), volumes[-1] + 20))
    ax.grid(alpha=.18)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(.5, .025, "300 mL initial fill · 2 s hold after acknowledgement · 2 s return",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(destination / "angle_table.png", dpi=160)
    plt.close(fig)
    manifest = dict(method="single-pour spout-edge weak-form fit followed by forward MPM",
                    eta_pa_s=identification["eta"], grid=grid,
                    measured_endpoints_used=[], validation_measurements_used=[],
                    numerical_target_tolerance_ml=TOLERANCE_ML,
                    numerical_screen_max_difference_ml=MAX_NUMERICAL_DIFFERENCE_ML,
                    numerical_checks=checks,
                    reference_acceptance_check=reference_check,
                    experimental_accuracy_claimed=False,
                    identification_sha256=digest(WORK / "identified/identify.json"),
                    targets=rows, all_forward_results=results,
                    files={p.name: digest(p) for p in destination.iterdir()})
    (WORK / "table_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = ROOT / "out/pour_weakform_handoff.zip"
    temporary = archive.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ["START_HERE.txt", "angle_table.csv", "angle_table.png"]:
            z.write(destination / name, name)
    temporary.replace(archive)
    print(table_text.getvalue())
    print(archive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=384)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--export", action="store_true")
    operation.add_argument("--check-grid", action="store_true")
    args = parser.parse_args()
    if args.export:
        export(args.grid)
    elif args.check_grid:
        print(json.dumps(screen_numerics(args.grid), indent=2))
    else:
        _, results = read_results(args.grid)
        print(json.dumps(proposed_targets(results), indent=2))
