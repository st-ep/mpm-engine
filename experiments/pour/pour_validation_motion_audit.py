"""Audit supplied joint/action logs against frozen motion planning, without fluid data.

The cup-to-hand transform is held at the original 60-degree calibration. Inferred
cup angles therefore do not measure reinsertion or slipping of the physical cup.
No model parameter or robot command is fitted by this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from examples import pour_recorded_twin as twin
from experiments.pour.pour_angle_sweep import build_motion, write_planned_episode
from experiments.pour.pour_perception import rim_curve_local

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "pouring_real_data/09-04-60-2s"
GEOMETRY = ROOT / "out/pour_weakform_recovery/identified/geometry.json"
MESH = ROOT / "out/pour_wf/09-04-60-2s/cup_render.obj"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(quats):
    return Rotation.from_quat(np.roll(quats, -1, axis=-1))


def crossing_times(t, angle, threshold):
    changes = np.flatnonzero(np.diff((angle >= threshold).astype(int)))
    crossings = [float(t[i] + (threshold-angle[i]) * (t[i+1]-t[i]) /
                       (angle[i+1]-angle[i])) for i in changes]
    if len(crossings) != 2:
        return None
    return dict(up_s=crossings[0], down_s=crossings[1],
                above_s=crossings[1]-crossings[0])


def main(attachments, output):
    # Refuse to replace previous evidence or modify the production handoff.
    output.mkdir(parents=True, exist_ok=False)
    scene = json.loads(GEOMETRY.read_text())
    ref = twin.load_episode(REFERENCE, twin.PRE_ROLL, twin.HOLD_SECONDS)
    arm = twin.RecordedPanda(ref, MESH, height=64, width=64, max_geom=4000,
                            cup_reference_pos=scene["cup_reference_pos"],
                            cup_reference_quat=scene["cup_reference_quat"])
    hand_cup, grasp = arm._hand_cup.copy(), arm._grasp.copy()
    arm.close()
    motion, reference = build_motion(REFERENCE, MESH)
    cases, inputs = [], [Path(__file__), GEOMETRY, MESH,
                        ROOT / "examples/pour_recorded_twin.py",
                        ROOT / "experiments/pour/pour_angle_sweep.py"]
    inputs += [REFERENCE / name for name in ["actions.jsonl", "states.jsonl", "meta.json"]]
    for action_path in sorted(attachments.glob("actions*.jsonl")):
        state_path = action_path.with_name(action_path.name.replace("actions", "states", 1))
        actions, states = read_rows(action_path), read_rows(state_path)
        pours = [a for a in actions if (a.get("pour_angle_deg") or 0) > 0]
        if len(pours) != 1:
            raise ValueError(f"Expected exactly one pour in {action_path}")
        pour = pours[0]
        angle = float(pour["pour_angle_deg"])
        ts = np.array([r["t"] for r in states])
        if np.any(np.diff(ts) <= 0) or not ts[0] < pour["t_send"] < pour["t_ack"] < ts[-1]:
            raise ValueError("Action/state timestamps do not match or are not increasing")
        case = output / f"recorded_{angle:g}"
        case.mkdir()
        for src, name in [(action_path, "actions.jsonl"), (state_path, "states.jsonl")]:
            shutil.copyfile(src, case / name)
            assert sha(src) == sha(case / name)
            inputs.append(src)
        # load_episode needs a meta file; no camera metadata is invented or borrowed.
        (case / "meta.json").write_text(json.dumps({"logs_only": True,
            "camera_calibration_available": False, "source_actions": str(action_path),
            "source_states": str(state_path)}, indent=2) + "\n")
        cases.append((angle, case))
    cases.sort()
    if len(cases) != 2 or [v[0] for v in cases] != [47.06, 52.92]:
        raise ValueError("This audit expects the supplied 47.06 and 52.92 degree recordings")
    protocol = dict(purpose="Recorded versus planned robot motion only",
                    cup_transform="Frozen original 60-degree hand-to-cup transform",
                    physical_cup_pose_measured=False, parameters_fitted=[],
                    liquid_observations_used=[], commands_changed=False,
                    input_sha256={str(p): sha(p) for p in inputs})
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex="col")
    results = []
    rim = rim_curve_local()
    tip = rim[np.argmin(np.abs(rim[:, 1]))]
    for j, (angle, actual_path) in enumerate(cases):
        planned_path = output / f"planned_{angle:g}"
        write_planned_episode(REFERENCE, planned_path, motion, reference, angle)
        episodes = [twin.load_episode(p, twin.PRE_ROLL, twin.HOLD_SECONDS)
                    for p in [actual_path, planned_path]]
        timing = [dict(ack_s=e["t_hold"]-e["t_pour"],
                       return_send_s=e["t_return_start"]-e["t_pour"],
                       return_ack_s=e["t_return_done"]-e["t_pour"])
                  for e in episodes]
        end = max(v["return_ack_s"] for v in timing) + .25
        t = np.arange(-.2, end, 1/120)
        positions, quats, tilts = [], [], []
        for e in episodes:
            a = twin.RecordedPanda(e, MESH, height=64, width=64, max_geom=4000)
            a._hand_cup, a._grasp = hand_cup.copy(), grasp.copy()
            poses = [a.cup_pose_at(float(time+e["t_pour"])) for time in t]
            p, q = np.array([v[0] for v in poses]), np.array([v[1] for v in poses])
            positions.append(p + rotation(q).apply(np.tile(tip, (len(q), 1))))
            quats.append(q)
            tilts.append(np.array([a.tilt_degrees(v) for v in q]))
            a.close()
        held = ((t > max(v["ack_s"] for v in timing)+.2) &
                (t < min(v["return_send_s"] for v in timing)-.2))
        if not held.any():
            raise ValueError("No shared stable hold")
        difference = tilts[0]-tilts[1]
        orient_error = np.degrees((rotation(quats[0])*rotation(quats[1]).inv()).magnitude())
        tip_difference = (positions[0]-positions[1])*1000
        peak = int(np.argmax(np.abs(difference)))
        actions = read_rows(actual_path / "actions.jsonl")
        states = read_rows(actual_path / "states.jsonl")
        pour, ret, _ = twin.recorded_pour_actions(actual_path)
        raw_t = np.array([s["t"]-pour["t_send"] for s in states])
        active = (raw_t >= 0) & (raw_t <= timing[0]["return_ack_s"])
        stable = ((raw_t > timing[0]["ack_s"]+.2) &
                  (raw_t < timing[0]["return_send_s"]-.2))
        ee_rot = Rotation.from_quat([s["state"]["ee_quat_xyzw"] for s in states])
        upright = Rotation.from_quat([.5, .5, -.5, .5])
        target = Rotation.from_quat(pour["target_quat_xyzw"])
        axis = (target*upright.inv()).as_rotvec()
        axis /= np.linalg.norm(axis)
        ee_angle = np.degrees((ee_rot*upright.inv()).as_rotvec() @ axis)
        target_error = np.degrees((ee_rot*target.inv()).magnitude())
        # Independent consistency check: FK of raw joints against logger EE pose.
        a = twin.RecordedPanda(episodes[0], MESH, height=64, width=64, max_geom=4000)
        fk_rotation_errors, fk_position_errors = [], []
        for k in np.flatnonzero(active):
            a.set_time(float(raw_t[k]+episodes[0]["t_pour"]))
            r = rotation(a.data.xquat[a.ee])
            p = a.data.xpos[a.ee] + r.apply(a.TCP_LOCAL)
            fk_rotation_errors.append(np.degrees((r*ee_rot[k].inv()).magnitude()))
            fk_position_errors.append(1000*np.linalg.norm(p-states[k]["state"]["ee_pos"]))
        a.close()
        crossing = []
        for threshold in [40., 45., 50.]:
            pair = [crossing_times(t, tilt, threshold) for tilt in tilts]
            if all(p is not None for p in pair):
                crossing.append(dict(threshold_deg=threshold, recorded=pair[0], planned=pair[1],
                                     extra_time_above_s=pair[0]["above_s"]-pair[1]["above_s"]))
        result = dict(command_angle_deg=angle, recorded=timing[0], planned=timing[1],
            recorded_command_duration_s=pour["duration_s"],
            recorded_hold_after_ack_s=ret["t_send"]-pour["t_ack"],
            planned_hold_after_ack_s=timing[1]["return_send_s"]-timing[1]["ack_s"],
            recorded_return_duration_s=ret["t_ack"]-ret["t_send"],
            planned_return_duration_s=timing[1]["return_ack_s"]-timing[1]["return_send_s"],
            raw_state_interval_median_s=float(np.median(np.diff(raw_t))),
            raw_state_active_max_gap_s=float(np.max(np.diff(raw_t[active]))),
            held_robot_angle_from_reported_ee_deg=float(np.mean(ee_angle[stable])),
            held_robot_target_orientation_error_deg=float(np.mean(target_error[stable])),
            maximum_robot_angle_from_reported_ee_deg=float(np.max(ee_angle[active])),
            held_cup_tilt_fixed_grasp_recorded_deg=float(np.mean(tilts[0][held])),
            held_cup_tilt_fixed_grasp_planned_deg=float(np.mean(tilts[1][held])),
            held_recorded_minus_planned_tilt_deg=float(np.mean(difference[held])),
            held_orientation_difference_deg=float(np.mean(orient_error[held])),
            held_recorded_minus_planned_tip_height_mm=float(np.mean(tip_difference[held, 2])),
            max_abs_tilt_difference_deg=float(abs(difference[peak])),
            max_abs_tilt_difference_time_s=float(t[peak]),
            recorded_tilt_at_max_difference_deg=float(tilts[0][peak]),
            planned_tilt_at_max_difference_deg=float(tilts[1][peak]),
            fk_vs_logged_ee_max_orientation_error_deg=float(max(fk_rotation_errors)),
            fk_vs_logged_ee_max_position_error_mm=float(max(fk_position_errors)),
            threshold_crossings_fixed_grasp=crossing)
        for key, value in result.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"Non-finite {key}")
        results.append(result)
        np.savez_compressed(output / f"comparison_{angle:g}.npz", t=t,
            recorded_tilt_fixed_grasp=tilts[0], planned_tilt_fixed_grasp=tilts[1],
            recorded_minus_planned_tip_mm=tip_difference,
            orientation_difference_deg=orient_error)
        axes[0, j].plot(t, tilts[0], label="Recorded joints")
        axes[0, j].plot(t, tilts[1], "--", label="Predicted from 60°")
        axes[0, j].set_title(f"{angle:g}° command")
        axes[1, j].plot(t, difference)
        axes[1, j].axhline(0, color="k", lw=.5)
        axes[2, j].plot(t, tip_difference[:, 2])
        axes[2, j].set_xlabel("Seconds after inclination command")
        for ax in axes[:, j]:
            ax.grid(alpha=.2)
            ax.axvline(timing[0]["return_send_s"], color="C0", alpha=.3)
            ax.axvline(timing[1]["return_send_s"], color="C1", ls="--", alpha=.3)
    axes[0, 0].set_ylabel("Cup tilt assuming fixed grasp (°)")
    axes[1, 0].set_ylabel("Recorded − planned tilt (°)")
    axes[2, 0].set_ylabel("Recorded − planned tip height (mm)")
    axes[0, 0].legend()
    fig.suptitle("Robot motion audit: no liquid measurements or parameter fitting\n"
                 "Cup mounting held at the original 60° calibration")
    fig.tight_layout()
    fig.savefig(output / "motion_comparison.png", dpi=180)
    plt.close(fig)
    (output / "results.json").write_text(json.dumps(dict(protocol=protocol, comparisons=results), indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.attachments, args.output)
