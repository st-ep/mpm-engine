"""Export the current weak-form/MPM handoff; historical exports require --legacy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import PchipInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.pour.pour_angle_endpoint_curve import export_model
from experiments.pour.pour_angle_plan import (
    GEOMETRY,
    MAX_FINAL_OUTSIDE_ML,
    OUTPUT,
    ROOT,
    angle_name,
    fingerprint,
)

DESTINATION = ROOT / "out/pour_angle_handoff"
PHILIP_DESTINATION = ROOT / "out/pour_angle_philip"


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_research():
    status_path = OUTPUT / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    withdrawn = status.get("status") == "withdrawn"
    table = json.loads((OUTPUT / "table.json").read_text())
    sweep = json.loads((OUTPUT / "sweep.json").read_text())
    forward_results = json.loads((OUTPUT / "all_forward_results.json").read_text())
    if any(abs(r["target_error_ml"]) > 1.5
           or r["outside_both_cups_ml"] > MAX_FINAL_OUTSIDE_ML for r in table):
        raise ValueError("Resolve forward-check errors before exporting")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name in ["planned_joint_samples", "simulation_metrics", "provenance", "scripts"]:
        (DESTINATION / name).mkdir(exist_ok=True)
    write_csv(DESTINATION / "angle_table.csv", table)
    (DESTINATION / "angle_table.json").write_text(json.dumps(table, indent=2))
    actions = []
    for row in table:
        name = angle_name(row["command_angle_deg"])
        episode = OUTPUT / "episodes" / name
        records = [json.loads(line) for line in
                   (episode / "actions.jsonl").read_text().splitlines()]
        parameters = []
        for record in records[:2]:
            parameters.append({key: record[key] for key in
                               ["target_pos", "target_quat_xyzw", "duration_s"]})
        actions.append(dict(target_ml=row["target_ml"],
                            pour_angle_deg=row["command_angle_deg"],
                            pour=parameters[0], dwell_s=2.0, return_to_upright=parameters[1]))
        shutil.copy2(episode / "joint_trajectory.csv",
                     DESTINATION / "planned_joint_samples" / f"{name}.csv")
        shutil.copy2(OUTPUT / "simulations" / name / "metrics.csv",
                     DESTINATION / "simulation_metrics" / f"{name}.csv")
        shutil.copy2(OUTPUT / "simulations" / name / "result.json",
                     DESTINATION / "provenance" / f"{name}.json")
        result = json.loads((OUTPUT / "simulations" / name / "result.json").read_text())
        if (result["provenance"] != fingerprint(row["command_angle_deg"])
                or abs(result["receiver_ml"] - row["predicted_receiver_ml"]) > 1e-8):
            raise ValueError(f"Table and current simulation inputs disagree at {name}")
    recipe = dict(description="Parameters for the existing robot planner, not its API schema",
                  position_units="metres in robot base frame", quaternion_order="xyzw",
                  initial_target_pos=[0.5, 0.0, 0.1],
                  initial_target_quat_xyzw=[0.5, 0.5, -0.5, 0.5],
                  initial_source_volume_ml=300.0, motion="linear",
                  dwell_timing="wait after pour acknowledgement, then send return",
                  experiments=actions)
    (DESTINATION / "robot_parameters.json").write_text(json.dumps(recipe, indent=2))
    if withdrawn:
        shutil.copy2(status_path, DESTINATION / "provenance" / "mpm_plan_status.json")
        shutil.copy2(ROOT / "out/pour_angle_curve_audit/grid_sensitivity_report.json",
                     DESTINATION / "provenance" / "grid_sensitivity_report.json")
        shutil.copy2(ROOT / "out/pour_angle_curve_empirical/curve.json",
                     DESTINATION / "provenance" / "replacement_empirical_curve.json")
    validation = ROOT / "out/pour_wf/readout_workaround"
    for path in [OUTPUT / "sweep.json", OUTPUT / "all_forward_results.json",
                 OUTPUT / "refinement_diagnostics.json", OUTPUT / "motion_checks.json", GEOMETRY,
                 OUTPUT / "outside_110ml_inspection.json",
                 validation / "mpm_source_endpoint_frozen.json",
                 validation / "mpm_source_endpoint_result.json"]:
        shutil.copy2(path, DESTINATION / "provenance" / path.name)
    shutil.copy2(validation / "one_pour_endpoint_validation.png",
                 DESTINATION / "previous_recording_validation.png")
    for name in ["pour_angle_sweep.py", "pour_angle_plan.py", "pour_angle_handoff.py"]:
        shutil.copy2(Path(__file__).with_name(name), DESTINATION / "scripts" / name)

    angles = np.array([r["provenance"]["angle_deg"] for r in sweep])
    volumes = np.array([r["receiver_ml"] for r in sweep])
    all_angles = np.array([r["provenance"]["angle_deg"] for r in forward_results])
    all_volumes = np.array([r["receiver_ml"] for r in forward_results])
    interpolation = PchipInterpolator(all_angles, all_volumes)
    fig, (ax, error_ax) = plt.subplots(2, 1, figsize=(9, 8),
                                      gridspec_kw={"height_ratios": [3.1, 1.2]})
    dense = np.linspace(angles.min(), angles.max(), 400)
    ax.plot(dense, interpolation(dense), color="#2463a0", lw=2,
            label="Guide through all MPM evaluations")
    ax.scatter(all_angles, all_volumes, color="#2463a0", s=20, zorder=3,
                label="Forward simulations (including refinements)")
    ax.axvspan(60, angles.max(), color="#ebc68b", alpha=0.3,
               label="Beyond recorded 60° motion")
    for row in table:
        angle, volume = row["command_angle_deg"], row["predicted_receiver_ml"]
        ax.scatter(angle, volume, marker="s", color="#c54d32", s=38, zorder=4)
        near_left = angle < angles.min() + 2
        ax.annotate(f"{row['target_ml']:.0f} mL · {angle:.2f}°", (angle, volume),
                    xytext=(8 if near_left else -8, 11), textcoords="offset points",
                    ha="left" if near_left else "right", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
    ax.scatter([], [], marker="s", color="#c54d32", label="Forward-checked target angles")
    ax.set(xlabel="Commanded final angle (degrees)", ylabel="Receiver volume (mL)",
           xlim=(angles.min() - 0.6, angles.max() + 0.7), ylim=(25, max(volumes) + 18))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    errors = [r["target_error_ml"] for r in table]
    error_ax.bar([str(int(r["target_ml"])) for r in table], errors, color="#c54d32", width=0.55)
    error_ax.axhline(0, color="0.3", lw=0.7)
    error_ax.set(xlabel="Requested validation volume (mL)",
                 ylabel="Simulated - target\n(mL)", ylim=(-1.6, 1.6))
    error_ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Five prospective pouring targets\n"
                 "300 mL initial fill · 2 s dwell · η = 4.0 Pa·s · 192³ MPM grid", fontsize=13)
    fig.text(0.5, 0.01, "Numerical target matching is not experimental accuracy. "
             "Previous cross-angle errors: -5.35 and +2.30 mL.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(DESTINATION / "angle_table.png", dpi=180)
    fig.savefig(DESTINATION / "angle_table.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for row in table:
        name = angle_name(row["command_angle_deg"])
        metrics = np.genfromtxt(OUTPUT / "simulations" / name / "metrics.csv",
                                delimiter=",", names=True)
        ax.plot(metrics["t"] - 1, metrics["ml_rcv"],
                label=f"{row['target_ml']:.0f} mL / {row['command_angle_deg']:.2f}°")
    ax.set(xlabel="Time from pour command (s)", ylabel="Simulated receiver volume (mL)",
           title="Forward simulations of the five planned motions")
    ax.legend(fontsize=9); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(DESTINATION / "planned_volume_traces.png", dpi=180)
    plt.close(fig)

    # Leave all future measured results blank: this package freezes predictions.
    measurements = [dict(target_ml=r["target_ml"], command_angle_deg=r["command_angle_deg"],
                         predicted_receiver_ml=r["predicted_receiver_ml"],
                         initial_measured_ml="", final_measured_ml="", measurement_method="",
                         recording_file="", cup_removed_between_runs="", notes="") for r in table]
    write_csv(DESTINATION / "validation_results_blank.csv", measurements)
    lines = ["| Target (mL) | Command angle | Tilt command (s) | Hold (s) | Predicted (mL) |",
             "|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['target_ml']:.0f} | {row['command_angle_deg']:.2f}° | "
                     f"{row['pour_command_duration_s']:.3f} | 2.0 | "
                     f"{row['predicted_receiver_ml']:.2f} |")
    readme = """# Five angle-controlled robot validation pours

These are prospective predictions from one physical calibration pour: the replacement
`09-04-60-2s` recording, initial 300 mL and final receiver 159 mL. Effective viscosity
is fixed at 4.0 Pa·s. The 45°/50° final measurements do not enter this planning curve.
This endpoint-inverse MPM approach preserves one calibration pour but changes the
paper's original time-weak viscosity identification algorithm.

## Robot parameters

""" + "\n".join(lines) + """

The command angle is the robot's `pour_angle_deg`, not the physical cup-axis angle.
The fitted cup insertion adds about 1.9° at the tilted pose. Use the command column
directly; do not add this offset. All return commands have duration 2.0 s.

`robot_parameters.json` contains Cartesian waypoints (metres, robot base frame),
orientation quaternions (xyzw), and durations for the existing linear-motion planner.
It describes the parameters; it is not an assumed executable robot API format.
For angle A, the tilt command lasts A/10 seconds. Wait 2 s after its acknowledgement,
then send the 2 s return command to the original upright pose.

The simulator preserves the 60° recording's timing overhead: approximately 0.329 s
above the tilt command duration, 2.004 s between tilt acknowledgement and return
send, and 2.396 s between return send and acknowledgement. Actual controller timing
can differ, which is why exact joints should be recorded on every validation run.

## Run protocol

1. Start each run with 300 mL in the source and an empty receiver. Keep the same
   liquid, cup insertion, and receiver placement as the calibration session.
   Refill without removing the source cup from the gripper between runs.
2. Use the angle, tilt duration, 2 s dwell, and 2 s return specified above.
   Record joints, action timestamps and both camera videos. Save the actual executed
   trajectory file too, if the controller supports it.
3. Let the final receiver volume settle, measure it independently and record the
   measurement method. Retain a photo. Fill `validation_results_blank.csv` and
   preserve the original recordings.
4. Complete all five runs with these frozen settings before adjusting viscosity or
   geometry against their results. They can then test previously unseen targets.

## What is verified and what remains uncertain

Each table angle has its own full forward MPM simulation. The `Predicted` column is
that forward result, rather than only an inverse-curve interpolation. Particle
conservation and final fluid outside both cups are recorded in `angle_table.json`.
The 170 mL target extends beyond the measured 159 mL pour and recorded 60° motion.

Near the 110 mL target, two inspected simulations put 1.2-1.4 mL on the table outside
the cups. The receiver prediction excludes that spill. The plan permits up to 2 mL
of reported final catch loss after this inspection; the initial 1 mL review threshold
was not met. Complete catch is assumed only for the 60° calibration, where the model
predicts no final spill. Do not substitute source depletion for receiver volume in
the new experiments. Preserve any real spill in the measurement notes.

The coarse interpolation needed refinement, especially near the low-volume target.
The simulated response includes local decreases with increasing angle. All such
evaluations are retained in `provenance/all_forward_results.json` and listed in
`refinement_diagnostics.json`; no monotonic correction or volume sorting was applied.
The plotted curve is a visual guide through these simulations, not a separately
validated continuous control law. Target angles were refined within observed local
brackets and selected using direct simulated endpoints. The source of the response
irregularity is not resolved by this search; numerical and motion sensitivity remain.

The generated motion family uses only the 60° measured joints: recorded spatial
paths parameterized by angular progress, a rescaled incline duration, the original
dwell, and the original return duration. The 60° trajectory is reproduced to
floating-point precision. Other angles are predicted motions. The referenced robot
trajectory JSON files were absent from the archive, so this is interpolation of
measured joint samples, not recovery of the exact original continuous command.
`planned_joint_samples/*.csv` contains generated diagnostic samples in radians,
with time relative to pour send; it is not a controller-ready trajectory file.

Previously replaying the separate 45°/50° recordings gave errors of -5.35/+2.30 mL.
Those replay motions and receiver placements differ from this common-reference
planned family, so they are not points used to interpolate this curve. Their
endpoints were inspected during earlier diagnosis, although excluded from viscosity
fitting; they were not blind tests. New pours provide the prospective validation.

The strict ±5 mL experimental criterion has not yet passed. Grid convergence and
cup-insertion uncertainty remain unresolved. Angle decimal places express command
resolution, not a claim of volume accuracy. This bundle is ready for validation,
not certified precision dispensing. Geometry was fitted from cup contours, not
adjusted to the three measured final volumes. Effective viscosity may absorb
remaining model discrepancy and should not be read as a material-property assay.

## Files and reproduction

- `angle_table.csv/json`: targets, commands, direct simulated volumes and catch audit.
- `angle_table.png/pdf`: sweep, selected angles and numerical target residuals.
- `planned_volume_traces.png`: predicted time curves for the five commands.
- `robot_parameters.json`: waypoints and timing for the existing robot planner.
- `planned_joint_samples/`: clearly generated joint tracks, for comparison with logs.
- `simulation_metrics/`: per-frame results for the selected angles.
- `provenance/`: input hashes, frozen calibration, geometry and simulation results.
- `previous_recording_validation.png`: existing 45°/50° transfer checks.
- `validation_results_blank.csv`: unfilled future experimental measurements.

From the repository, with the original recording and geometry files available:

```bash
MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m experiments.pour.pour_angle_plan
PYTHONPATH=. .venv/bin/python -m experiments.pour.pour_angle_handoff --research
```

The planner uses two GPUs by default. A single angle can be run with
`--single-angle ANGLE --device cuda:0`. Scripts in this bundle are snapshots of the
repository modules; the rest of the simulator and original data remain dependencies.
"""
    if withdrawn:
        readme = ("# SUPERSEDED MPM RESEARCH PLAN\n\n"
                  "Withdrawn after the grid-translation audit; see provenance/mpm_plan_status.json "
                  "and grid_sensitivity_report.json. Current operator files use the three-pour "
                  "empirical curve and live in ../pour_angle_philip/ "
                  "and ../pour_angle_handoff.zip. "
                  "The table and report below are retained as historical research results.\n\n"
                  + readme)
    (DESTINATION / "README.md").write_text(readme)
    manifest = dict(created_at_utc=datetime.now(UTC).isoformat(),
                    status=("withdrawn MPM plan; historical research only" if withdrawn
                            else "frozen predictions for prospective robot validation"),
                    ready_for_prospective_validation=not withdrawn,
                    validated_to_plus_minus_5_ml=False,
                    training_episode="09-04-60-2s", training_receiver_ml=159.0,
                    initial_source_ml=300.0, eta_pa_s=4.0,
                    measured_endpoints_used_to_fit_viscosity=["09-04-60-2s"],
                    measured_endpoints_used_to_fit_geometry=[],
                    measured_future_validation_results_used=[],
                    targets_ml=[row["target_ml"] for row in table],
                    maximum_forward_target_residual_ml=max(abs(row["target_error_ml"])
                                                          for row in table),
                    maximum_final_outside_both_cups_ml=max(row["outside_both_cups_ml"]
                                                          for row in table),
                    prior_cross_angle_errors_ml={"45": -5.35, "50": 2.30})
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2))
    checksums = []
    for path in sorted(DESTINATION.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(hashlib.sha256(path.read_bytes()).hexdigest()
                             + "  " + str(path.relative_to(DESTINATION)))
    (DESTINATION / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    archive = ROOT / "out/pour_angle_research.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(DESTINATION.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(DESTINATION.parent))
    print(f"Wrote {archive}")


def operator_figure(curve, table, path):
    """Show only the operating curve and requested targets on Philip's figure."""
    minimum_volume = min(row["target_ml"] for row in table)
    maximum_volume = max(row["target_ml"] for row in table)
    minimum_angle = min(curve.angle(minimum_volume),
                        min(row["command_angle_deg"] for row in table))
    maximum_angle = max(curve.angle(maximum_volume),
                        max(row["command_angle_deg"] for row in table))
    a = np.linspace(minimum_angle, maximum_angle, 600)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(a, curve.volume(a), color="#2463a0", lw=2.4)
    for row in table:
        x, y = row["command_angle_deg"], row["expected_receiver_ml"]
        ax.scatter(x, y, color="#c54d32", s=35, zorder=3)
        near_left = x < minimum_angle + 1
        ax.annotate(f"{row['target_ml']:.0f} mL · {x:.2f}°", (x, y),
                    xytext=(9 if near_left else -9, 12), textcoords="offset points",
                    ha="left" if near_left else "right", fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2})
    ax.set(xlabel="Final angle (degrees)", ylabel="Expected receiver volume (mL)",
           xlim=(minimum_angle - 0.6, maximum_angle + 0.8),
           ylim=(minimum_volume - 10, maximum_volume + 15))
    ax.set_xticks(np.arange(np.floor(minimum_angle), np.ceil(maximum_angle) + 1, 2))
    ax.set_yticks(np.arange(minimum_volume - 10, maximum_volume + 1, 20))
    ax.grid(alpha=0.2)
    ax.set_title("Pouring angle and volume\n300 mL initial fill · 2 s hold", pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export_legacy_empirical():
    """Use the corrected continuous curve and an explicit three-file archive list."""
    curve, table = export_model()
    rows = []
    for row in table:
        angle = row["command_angle_deg"]
        rows.append(dict(target_ml=f"{row['target_ml']:.0f}",
                         pour_angle_deg=f"{angle:.2f}",
                         duration_s=f"{row['pour_command_duration_s']:.3f}",
                         dwell_s=f"{row['dwell_s']:.1f}",
                         return_duration_s=f"{row['return_command_duration_s']:.1f}"))
    PHILIP_DESTINATION.mkdir(parents=True, exist_ok=True)
    write_csv(PHILIP_DESTINATION / "angle_table.csv", rows)
    minimum_target = min(row["target_ml"] for row in table)
    maximum_target = max(row["target_ml"] for row in table)
    instructions = f"""PHILIP: POURING VALIDATION RUNS

Use angle_table.csv for the {len(table)} targets, angles and durations.
angle_table.png shows the angle-versus-volume curve for the same motion settings.
Angles are the robot's pour_angle_deg command values; use them directly.
For other targets between {minimum_target:g} and {maximum_target:g} mL,
read the angle from the curve and set
duration_s = angle / 10. Keep the hold and return durations at 2 s.

For each run:
1. Fill the source with 300 mL and empty the receiver.
2. Use the same recorded starting pose and linear motion setup.
   Set pour_angle_deg and duration_s from the table.
3. Wait 2 s AFTER the pour acknowledgement, then send the 2 s return to upright.
4. Let the liquid settle and measure the final receiver volume. Take a photo.

Keep the cup in the same gripper position between refills. Keep the liquid and
receiver placement unchanged. Complete all {len(table)} runs with the listed settings.

Save joints, action timestamps, the executed trajectory file, and both camera
videos for each run. Label each recording with its target and angle. Include
the measured final volume, how it was measured, and any spilled liquid.

These are validation targets: report the actual volumes, even if they differ.
"""
    (PHILIP_DESTINATION / "START_HERE.txt").write_text(instructions)
    operator_figure(curve, table, PHILIP_DESTINATION / "angle_table.png")
    archive = ROOT / "out/pour_angle_handoff.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name in ["START_HERE.txt", "angle_table.csv", "angle_table.png"]:
            handle.write(PHILIP_DESTINATION / name, name)
    print(f"Wrote {archive}: 3 files, {archive.stat().st_size} bytes")


def export():
    """The default must never silently substitute empirical endpoint calibration."""
    from experiments.pour.pour_weakform_table import export as export_weakform

    return export_weakform()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", action="store_true",
                        help="export the historical research archive")
    parser.add_argument("--legacy", action="store_true",
                        help="explicitly reproduce a superseded workflow")
    args = parser.parse_args()
    if args.research:
        if not args.legacy:
            parser.error("The old research export is withdrawn; reproducing it requires --legacy.")
        export_research()
    elif args.legacy:
        export_legacy_empirical()
    else:
        export()
