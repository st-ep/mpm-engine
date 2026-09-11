"""Compare predicted and recorded robot motion without any liquid observations.

The fixed hand-to-cup calibration comes only from the 60-degree recording. This
checks the synthetic motion generator, not physical changes in cup insertion.
Nothing is fitted or corrected from the comparison.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples import pour_recorded_twin as twin
from experiments.pour.pour_angle_sweep import build_motion, write_planned_episode
from experiments.pour.pour_perception import rim_curve_local

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/pour_physics_audit/motion_transfer"
REFERENCE = ROOT / "pouring_real_data/09-04-60-2s"
MESH = ROOT / "out/pour_wf/09-04-60-2s/cup_render.obj"


def rotation(quat):
    return Rotation.from_quat(np.roll(quat, -1))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    scene_path = ROOT / "out/pour_weakform_recovery/identified/geometry.json"
    scene = json.loads(scene_path.read_text())
    ep = twin.load_episode(REFERENCE, twin.PRE_ROLL, twin.HOLD_SECONDS)
    arm = twin.RecordedPanda(ep, MESH, height=64, width=64, max_geom=4000,
                             cup_reference_pos=scene["cup_reference_pos"],
                             cup_reference_quat=scene["cup_reference_quat"])
    hand_cup, grasp = arm._hand_cup.copy(), arm._grasp.copy()
    arm.close()
    motion, reference = build_motion(REFERENCE, MESH)
    paths = [Path(__file__), ROOT / "experiments/pour/pour_angle_sweep.py",
             ROOT / "examples/pour_recorded_twin.py", scene_path]
    for angle in [60, 45, 50]:
        paths += [ROOT / f"pouring_real_data/09-04-{angle}-2s" / name
                  for name in ["actions.jsonl", "states.jsonl", "meta.json"]]
    protocol = dict(purpose="Robot motion transfer audit only; no fitting",
                    identification_episode=REFERENCE.name, liquid_measurements_used=[],
                    fixed_hand_to_cup_from_60=True, calibration_changed=False,
                    input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in paths})
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = []
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex="col")
    rim = rim_curve_local()
    tip = rim[np.argmin(np.abs(rim[:, 1]))]
    for j, angle in enumerate([60, 45, 50]):
        actual_path = ROOT / f"pouring_real_data/09-04-{angle}-2s"
        planned_path = OUT / f"planned_{angle}"
        write_planned_episode(REFERENCE, planned_path, motion, reference, angle)
        episodes = [twin.load_episode(path, twin.PRE_ROLL, twin.HOLD_SECONDS)
                    for path in [actual_path, planned_path]]
        arms = [twin.RecordedPanda(e, MESH, height=64, width=64, max_geom=4000) for e in episodes]
        for a in arms:
            a._hand_cup, a._grasp = hand_cup.copy(), grasp.copy()
        finish = max(e["duration"] - e["t_pour"] for e in episodes)
        times = np.arange(0., finish, 1/60.)
        positions, quaternions, tilts, tips = [], [], [], []
        for a, e in zip(arms, episodes):
            poses = [a.cup_pose_at(float(t + e["t_pour"])) for t in times]
            p, q = np.array([v[0] for v in poses]), np.array([v[1] for v in poses])
            positions.append(p); quaternions.append(q)
            tilts.append(np.array([a.tilt_degrees(v) for v in q]))
            tips.append(np.array([twin.quat_to_mat(v) @ tip for v in q]) + p)
            a.close()
        errors = np.degrees((Rotation.from_quat(np.roll(quaternions[1], -1, axis=1)) *
                             Rotation.from_quat(np.roll(quaternions[0], -1, axis=1)).inv()).magnitude())
        tilt_difference = tilts[1] - tilts[0]
        tip_error_mm = 1000 * (tips[1] - tips[0])
        actual_ack = episodes[0]["t_hold"] - episodes[0]["t_pour"]
        predicted_ack = episodes[1]["t_hold"] - episodes[1]["t_pour"]
        actual_return = episodes[0]["t_return_start"] - episodes[0]["t_pour"]
        predicted_return = episodes[1]["t_return_start"] - episodes[1]["t_pour"]
        held = (times >= max(actual_ack, predicted_ack) + .15) & (times <= min(actual_return, predicted_return) - .15)
        if not held.any():
            raise ValueError("No common dwell interval for motion comparison")
        r = dict(angle_deg=angle, actual_ack_s=actual_ack, predicted_ack_s=predicted_ack,
                 actual_return_send_s=actual_return, predicted_return_send_s=predicted_return,
                 max_orientation_difference_deg=float(errors.max()),
                 rms_orientation_difference_deg=float(np.sqrt(np.mean(errors**2))),
                 held_orientation_difference_deg=float(np.mean(errors[held])),
                 held_tilt_difference_deg=float(np.mean(tilt_difference[held])),
                 max_tilt_difference_deg=float(np.abs(tilt_difference).max()),
                 held_tip_height_difference_mm=float(np.mean(tip_error_mm[held, 2])),
                 max_tip_position_difference_mm=float(np.linalg.norm(tip_error_mm, axis=1).max()))
        results.append(r)
        np.savez_compressed(OUT / f"comparison_{angle}.npz", t=times, actual_tilt=tilts[0],
                            predicted_tilt=tilts[1], orientation_difference_deg=errors,
                            tip_position_difference_mm=tip_error_mm)
        axes[0, j].plot(times, tilts[0], label="Recorded joints")
        axes[0, j].plot(times, tilts[1], "--", label="Motion predicted from 60°")
        axes[0, j].set_title(f"{angle}° command")
        axes[1, j].plot(times, tilt_difference)
        axes[2, j].plot(times, tip_error_mm[:, 2])
        axes[2, j].set_xlabel("Seconds after tilt command")
        for a in axes[:, j]:
            a.axvspan(max(actual_ack, predicted_ack), min(actual_return, predicted_return), color="k", alpha=.05)
            a.grid(alpha=.2)
    axes[0, 0].set_ylabel("Cup tilt (degrees)")
    axes[1, 0].set_ylabel("Predicted − recorded tilt (degrees)")
    axes[2, 0].set_ylabel("Predicted − recorded tip height (mm)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Motion transfer only: fixed 60° grasp, no liquid data or parameter fitting")
    fig.tight_layout(); fig.savefig(OUT / "motion_comparison.png", dpi=160); plt.close(fig)
    (OUT / "results.json").write_text(json.dumps(dict(protocol=protocol, comparisons=results), indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
