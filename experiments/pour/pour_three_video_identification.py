"""Compare independent weak spout-edge fits to the three September recordings.

No measured endpoint, validation outcome, or simulator output enters these fits.
The original PDF two-pass estimator and the existing conservative-flight variant
are both evaluated; neither is selected according to agreement across recordings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from matplotlib.patches import Rectangle
import numpy as np

from examples import pour_recorded_twin as twin
from experiments.pour import pour_perception as perception
from experiments.pour.pour_transport_identify import geometry_cache, solve
from experiments.pour.pour_weakform_recovery import fit as pdf_fit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/pour_three_video_identification"
BASE = ROOT / "out/pour_weakform_recovery/identified"
MESH = ROOT / "out/pour_wf/09-04-60-2s/cup_render.obj"
ANGLES = (45, 50, 60)


def inputs(align_receiver=False):
    scene = json.loads((BASE / "geometry.json").read_text())
    runs = []
    paths = [Path(__file__), Path(perception.__file__), BASE / "geometry.json", MESH,
             ROOT / "experiments/pour/pour_weakform_identify.py",
             ROOT / "experiments/pour/pour_weakform_recovery.py",
             ROOT / "experiments/pour/pour_transport_identify.py",
             ROOT / "experiments/pour/pour_weakform_transport.py",
             ROOT / "examples/pour_recorded_twin.py",
             ROOT / "src/warpmpm/geometry/measuring_cup.py"]
    for angle in ANGLES:
        ep = ROOT / f"pouring_real_data/09-04-{angle}-2s"
        geometry_path = ROOT / f"out/pour_wf/{ep.name}/endpoint_geometry.json"
        geometry = json.loads(geometry_path.read_text())
        if geometry["source"] != "receiver outline only; endpoint not used":
            raise ValueError("Receiver geometry must be independent of liquid amount")
        pour, ret, _ = twin.recorded_pour_actions(ep)
        ack, back = pour["t_ack"] - pour["t_send"], ret["t_send"] - pour["t_send"]
        runs.append(dict(angle_deg=angle, episode=ep.name, receiver_geometry=geometry,
                         t_send=pour["t_send"], tilt_ack_s=ack, return_send_s=back,
                         fit_window_s=[ack + .15, back - .15]))
        paths += [geometry_path] + [ep / name for name in
                  ["actions.jsonl", "states.jsonl", "meta.json", "frames_side.jsonl", "side_rgb.mp4"]]
    centers = {}
    for run in runs:
        meta = json.loads((ROOT / "pouring_real_data" / run["episode"] / "meta.json").read_text())
        g = run["receiver_geometry"]
        centers[run["angle_deg"]] = perception.Camera(meta, "side").project(
            np.r_[g["receiver_xy"], g["table_z"]])[0][0, 1]
    for run in runs:
        shift = int(round(centers[run["angle_deg"]] - centers[60])) if align_receiver else 0
        run["roi_native_v"] = [238 + shift, 260 + shift]
        run["receiver_horizontal_shift_px"] = shift
    protocol = dict(
        purpose="Independent viscosity consistency diagnostic; no combined calibration",
        initial_volume_ml=300., independent_identification_pours=3, pours_per_fit=1,
        measured_endpoints_used=[], validation_measurements_used=[], simulator_outputs_used=[],
        volume_rescaling_applied=False, production_calibration_changed=False,
        source_geometry=scene,
        source_pose_rule="Exact recorded joints, fixed hand-to-cup transform from 60-degree exterior contours",
        source_pose_assumption="Cup-to-gripper transform is unchanged across recordings; not independently verified",
        receiver_geometry_rule="Per-recording exterior rim and base outline only",
        optical_rule=("Translate the fixed 60-degree strip with projected receiver base center, from exterior geometry only"
                      if align_receiver else "Existing fixed right-side strip, native v=238:260 inclusive"),
        hue_bounds=[.115, .965], saturation_min=.55, value_min=.12,
        ray_steps=4096, level_step_m=.0000078125, image_frame_stride=1,
        observation_stride=2,
        observation_grid_rule="Every second camera frame starting at tilt SEND minus 6 seconds, as in baseline",
        extraction_window_rule="Tilt ACK minus 2 seconds to return SEND plus 0.5 seconds",
        weak_fit_window_rule="Tilt ACK plus 0.15 seconds to return SEND minus 0.15 seconds",
        temporal_smoothing_s=.30, transport_prehistory_s=1.,
        estimators=["Original PDF two-pass weak brink fit", "Existing conservative-flight weak brink fit"],
        runs=runs,
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return protocol, scene


def extract(run, scene, hand_cup, grasp):
    episode_path = ROOT / "pouring_real_data" / run["episode"]
    folder = OUT / run["episode"]
    folder.mkdir()
    camera = perception.Camera(json.loads((episode_path / "meta.json").read_text()), "side")
    geom = run["receiver_geometry"]
    pos = np.r_[geom["receiver_xy"], geom["table_z"]]
    roi = list(map(int, perception.roi_from_pose(camera, pos, twin.R_CUP_REF)))
    roi[2:] = run.get("roi_native_v", [238, 260])
    u0, u1, v0, v1 = roi
    perception.RAY_STEPS = 4096
    perception.LEVEL_STEP = .0000078125
    z_on, _, chord = perception.z_on_map(camera, pos, twin.Q_RCV, roi, .001)
    rows = [json.loads(s) for s in (episode_path / "frames_side.jsonl").read_text().splitlines()]
    lo, hi = run["tilt_ack_s"] - 2., run["return_send_s"] + .5
    send = run["t_send"]
    times, levels, ious, frame_indices, previews = [], [], [], [], []
    preview_times = np.linspace(*run["fit_window_s"], 3)
    preview_rows = {min(rows, key=lambda r: abs(r["t_host"] - send - t))["frame_idx"]
                    for t in preview_times}
    for row, rgb in perception.frame_iter(episode_path, rows, send + lo, send + hi, 1):
        hsv = rgb_to_hsv(rgb[v0:v1+1, u0:u1+1] / 255.)
        mask = (((hsv[:, :, 0] < .115) | (hsv[:, :, 0] > .965))
                & (hsv[:, :, 1] > .55) & (hsv[:, :, 2] > .12))
        level, iou, predicted = perception.fit_level(z_on, chord, mask)
        times.append(row["t_host"] - send)
        levels.append(level); ious.append(iou); frame_indices.append(row["frame_idx"])
        if row["frame_idx"] in preview_rows:
            previews.append((rgb.copy(), predicted.copy(), times[-1], level, iou))
    times, levels, ious = np.array(times), np.array(levels), np.array(ious)
    volumes = np.array([twin.SPEC.cavity_volume(z - pos[2] - twin.SPEC.floor_z) * 1e6
                        if np.isfinite(z) else np.nan for z in levels])
    np.savez_compressed(folder / "optical_series.npz", t=times, level_m=levels,
                        receiver_ml=volumes, iou=ious, frame_idx=frame_indices, roi_native=roi)
    sampled = [r for r in rows if r["t_host"] >= send - 6.][::2]
    sampled = [r for r in sampled if send + lo <= r["t_host"] <= send + hi]
    obs_t = np.array([r["t_host"] - send for r in sampled])
    good = np.isfinite(volumes)
    if good.sum() < 12:
        raise ValueError(f"{run['episode']}: insufficient liquid visibility in fixed strip")
    obs_volume = np.interp(obs_t, times[good], volumes[good], left=np.nan, right=np.nan)
    depths = np.linspace(0, twin.SPEC.rim_z - twin.SPEC.floor_z, 2001)
    physical_volume = np.array([twin.SPEC.cavity_volume(d) * 1e6 for d in depths])
    episode = twin.load_episode(episode_path, twin.PRE_ROLL, twin.HOLD_SECONDS)
    arm = twin.RecordedPanda(episode, MESH, height=64, width=64, max_geom=4000)
    arm._hand_cup, arm._grasp = hand_cup.copy(), grasp.copy()
    poses = [arm.cup_pose_at(float(t + episode["t_pour"])) for t in obs_t]
    obs = dict(t=obs_t, frame_idx=np.array([r["frame_idx"] for r in sampled]),
               rcv_vol=obs_volume * 1e-6,
               rcv_level=pos[2] + twin.SPEC.floor_z + np.interp(obs_volume, physical_volume, depths),
               cup_pos=np.array([p for p, q in poses]), cup_quat=np.array([q for p, q in poses]),
               tilt_deg=np.array([arm.tilt_degrees(q) for p, q in poses]),
               lip=np.array([p + twin.quat_to_mat(q) @ np.array([twin.SPEC.tip_x, 0, twin.SPEC.rim_z])
                             for p, q in poses]),
               t_send=np.array(send), table_z=np.array(pos[2]), receiver_xy=pos[:2],
               cup_reference_pos=np.array(scene["cup_reference_pos"]),
               cup_reference_quat=np.array(scene["cup_reference_quat"]))
    arm.close()
    np.savez_compressed(folder / "observations.npz", **obs)
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    for ax, (rgb, pred, t, level, iou) in zip(axes, previews):
        whole = np.full(rgb.shape[:2], np.nan)
        whole[v0:v1+1, u0:u1+1] = pred.astype(float)
        ax.imshow(np.rot90(rgb))
        if pred.any():
            ax.contour(np.rot90(whole), levels=[.5], colors=["lime"], linewidths=1)
        ax.add_patch(Rectangle((v0, camera.w - 1 - u1), v1-v0, u1-u0,
                               fill=False, edgecolor="cyan", linewidth=1))
        ax.set(xlim=(110, 300), ylim=(590, 350), title=f"t={t:.2f} s; IoU={iou:.2f}")
        ax.axis("off")
    fig.suptitle(f"{run['angle_deg']}°: fixed optical strip (cyan), fitted liquid boundary (green)")
    fig.tight_layout(); fig.savefig(folder / "optical_check.png", dpi=150); plt.close(fig)
    in_fit = (times >= run["fit_window_s"][0]) & (times <= run["fit_window_s"][1])
    optical = dict(roi_native=roi, valid_fraction_fit=float(np.mean(good[in_fit])),
                   median_iou_fit=float(np.median(ious[in_fit])),
                   longest_valid_gap_s=float(np.max(np.diff(times[good]))),
                   number_of_optical_frames=len(times))
    return obs, optical


def main(align_receiver=False):
    global OUT
    if align_receiver:
        OUT = OUT.with_name(OUT.name + "_receiver_aligned")
    if (OUT / "results.json").exists():
        raise FileExistsError("Preserve completed comparison")
    if (OUT / "protocol.json").exists():
        protocol = json.loads((OUT / "protocol.json").read_text())
        scene = protocol["source_geometry"]
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        protocol, scene = inputs(align_receiver)
        (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (OUT / "completion_implementation_sha256.txt").write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + "\n")
    ep = twin.load_episode(ROOT / "pouring_real_data/09-04-60-2s", twin.PRE_ROLL, twin.HOLD_SECONDS)
    arm = twin.RecordedPanda(ep, MESH, height=64, width=64, max_geom=4000,
                            cup_reference_pos=scene["cup_reference_pos"],
                            cup_reference_quat=scene["cup_reference_quat"])
    hand_cup, grasp = arm._hand_cup.copy(), arm._grasp.copy()
    arm.close()
    results = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for run, ax in zip(protocol["runs"], axes):
        print(f"Extracting {run['episode']}", flush=True)
        folder = OUT / run["episode"]
        if (folder / "observations.npz").exists():
            obs = dict(np.load(folder / "observations.npz"))
            series = dict(np.load(folder / "optical_series.npz"))
            inside = ((series["t"] >= run["fit_window_s"][0])
                      & (series["t"] <= run["fit_window_s"][1]))
            valid = np.isfinite(series["receiver_ml"])
            optical = dict(roi_native=series["roi_native"].tolist(),
                           valid_fraction_fit=float(np.mean(valid[inside])),
                           median_iou_fit=float(np.median(series["iou"][inside])),
                           longest_valid_gap_s=float(np.max(np.diff(series["t"][valid]))),
                           number_of_optical_frames=len(series["t"]))
        else:
            obs, optical = extract(run, scene, hand_cup, grasp)
        window = run["fit_window_s"]
        eta, se, keep, dv, cf, forcing, rms = pdf_fit(obs, window)
        pdf = dict(eta_pa_s=eta, statistical_se_pa_s=se, rms_ml=rms,
                   n_kept=int(keep.sum()), n_frames=len(dv), receiver_gain_ml=float(dv[-1]))
        try:
            transport, arrays = solve(geometry_cache(obs, window, 1.))
        except ValueError as error:
            transport, arrays = dict(status="unavailable", reason=str(error)), None
        np.savez_compressed(folder / "pdf_fit.npz", **forcing, fit_keep=keep,
                            delta_receiver_ml=dv, cumulative_forcing=cf)
        if arrays is not None:
            np.savez_compressed(folder / "transport_fit.npz", **arrays)
        result = dict(angle_deg=run["angle_deg"], episode=run["episode"],
                      fit_window_s=window, optical=optical,
                      pdf_two_pass=pdf, conservative_transport=transport)
        if run["angle_deg"] == 60:
            expected = json.loads((ROOT / "out/pour_physics_audit/optical_full_rate/results.json").read_text())
            reference_eta = expected["cases"][0]["eta_pa_s"]
            delta = abs(transport["eta_pa_s"] - reference_eta)
            result["baseline_reproduction"] = dict(reference_eta_pa_s=reference_eta,
                                                   absolute_difference_pa_s=delta, passed=delta < 1e-8)
            if delta >= 1e-8:
                raise RuntimeError(f"60-degree reproduction mismatch: {delta}")
        results.append(result)
        (folder / "identify.json").write_text(json.dumps(result, indent=2) + "\n")
        (OUT / "results.json").write_text(json.dumps(dict(protocol=protocol, fits=results), indent=2) + "\n")
        print(json.dumps(result), flush=True)
        t = forcing["t"] - run["tilt_ack_s"]
        ax.plot(t, dv, ".", color="black", label="Video volume gain")
        ax.plot(t, cf / eta, label=f"PDF: {eta:.2f} Pa·s")
        if arrays is not None:
            ta = arrays["t"][arrays["fitmask"]] - run["tilt_ack_s"]
            ax.plot(ta, arrays["fit_cumulative_forcing"] / transport["eta_pa_s"], "--",
                    label=f"Flight correction: {transport['eta_pa_s']:.2f} Pa·s")
        ax.set(title=f"{run['angle_deg']}° recording", xlabel="Seconds after tilt acknowledgement",
               ylabel="Receiver gain in fit window (mL)")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Independent weak spout-edge fits; initial volume 300 mL; no endpoint calibration")
    fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--align-receiver", action="store_true",
                        help="Separate diagnostic: translate ROI using only external receiver position")
    main(parser.parse_args().align_receiver)
