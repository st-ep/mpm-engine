"""Capture a full, unchanged September 8 MPM replay for a private share video.

This exports display states at 30 Hz while preserving the original 60 Hz replay,
substeps, material/contact parameters, geometry, and recorded robot trajectory.
No parameter search or hardware outcome fitting is performed.
"""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import inspect
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "out/pour_physics_audit/aflip_blend_time_20260907"
FROZEN = BASE / "recalibration_mu0.272000_n160_dt1"
WORK = ROOT / "out/pour_share_video_20260911/capture"
SOURCE = BASE / "isolated_src"
sys.path.insert(0, str(SOURCE))
from warpmpm.kernels import mpm_utils
from experiments.pour import pour_transfer_recalibration_reference as reference


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.item()) + "\n")


def main():
    WORK.mkdir(parents=True, exist_ok=False)
    frames = WORK / "states"
    frames.mkdir()
    inputs = json.loads((FROZEN / "protocol.json").read_text())["input_sha256"]
    unavailable = []
    for rel, expected in inputs.items():
        path = ROOT / rel
        if rel == "out/pour_navier_calibration/philip_pilot_20260907_200/philip_pilot_handoff.zip" and not path.exists():
            unavailable.append(rel)
            continue
        assert sha(path) == expected, rel
    assert Path(mpm_utils.__file__).resolve().is_relative_to(SOURCE)
    twin = reference.twin
    assert twin.FPS == 60
    ep = twin.load_episode(reference.REFERENCE, twin.PRE_ROLL, twin.HOLD_SECONDS)
    write(WORK / "provenance.json", dict(
        purpose=__doc__, frozen_protocol=str((FROZEN / "protocol.json").relative_to(ROOT)),
        frozen_protocol_sha256=sha(FROZEN / "protocol.json"), input_sha256=inputs,
        unavailable_non_dynamics_artifacts=unavailable,
        actual_kernel=str(mpm_utils.__file__), kernel_sha256=sha(Path(mpm_utils.__file__)),
        script_sha256=sha(Path(__file__)), episode=reference.REFERENCE.name,
        host_start_s=ep["t0"], display_fps=30, replay_fps=twin.FPS,
        eta_pa_s=reference.ETA, source_contact=.272, grid=160, dt_scale=1.,
        calibration_not_held_out=True, parameters_changed=False,
        scope="Private video only; manuscript, paper figure, and original outputs are unchanged"))

    def capture(l):
        frame = l["frame"]
        if (frame + 1) % 2:
            return
        x_world = l["x"] - l["w2m"]
        assert len(x_world) == 229280 and np.isfinite(x_world).all()
        if not (WORK / "particles.npz").exists():
            np.savez(WORK / "particles.npz", vol=l["vol"], h=l["h"],
                     receiver_pos=l["receiver_pos"], receiver_quat=twin.Q_RCV,
                     world_to_grid_offset=l["w2m"])
        index = (frame + 1) // 2 - 1
        path = frames / f"state_{index:04d}.npz"
        temporary = path.with_suffix(".pending.npz")
        np.savez_compressed(temporary, x_world=x_world, source_pos=l["p_now"],
                            source_quat=l["q_now"], t=l["t_now"],
                            t_host=ep["t0"] + l["t_now"], replay_frame=frame)
        temporary.rename(path)

    original = inspect.getsource(twin.run)
    anchor = "        if video:\n            img = render_frame"
    assert original.count(anchor) == 1
    instrumented = original.replace(anchor, "        _share_video_capture(locals())\n\n" + anchor)
    (WORK / "original_run.py").write_text(original)
    (WORK / "instrumented_run.py").write_text(instrumented)
    twin.__dict__["_share_video_capture"] = capture
    exec(compile(instrumented, str(WORK / "instrumented_run.py"), "exec"), twin.__dict__)
    reference.OUT = WORK / "forward"
    old_project = twin.project_out_of_solid
    gate = {}

    def checked_project(x, v, *args, **kwargs):
        if not gate:
            speed = np.linalg.norm(v, axis=1)
            gate.update(mean_speed_m_s=float(speed.mean()),
                        passed=bool(np.isfinite(speed).all() and speed.mean() < twin.SETTLE_SPEED))
            write(WORK / "settling_gate.json", gate)
            assert gate["passed"], "Initial settling failed"
        return old_project(x, v, *args, **kwargs)

    twin.project_out_of_solid = checked_project
    try:
        reference.run(SimpleNamespace(source_friction=.272, grid=160, phase=0.,
            wall="original-separable", slip_mm=None, angle=None, max_angle=60.,
            dt_scale=1., device="cuda:0"))
    finally:
        twin.project_out_of_solid = old_project

    original_result, = (FROZEN / "forward").rglob("result.json")
    replay_result, = (WORK / "forward").rglob("result.json")
    original_csv, = original_result.parent.rglob("metrics.csv")
    replay_csv, = replay_result.parent.rglob("metrics.csv")
    old = list(csv.DictReader(original_csv.open()))
    new = list(csv.DictReader(replay_csv.open()))
    assert len(old) == len(new) == 773
    differences = {k: max(abs(int(a[k])-int(b[k]))*300/229280
                         for a, b in zip(old, new, strict=True))
                   for k in ("n_src", "n_rcv", "n_air_spill")}
    endpoint_difference = abs(json.loads(original_result.read_text())["receiver_ml"]
                              - json.loads(replay_result.read_text())["receiver_ml"])
    accepted = max(differences.values()) <= .1 and endpoint_difference <= .1
    write(WORK / "verification.json", dict(
        accepted=accepted, tolerance_ml=.1, compared_replay_frames=len(new),
        maximum_count_differences_ml=differences, endpoint_difference_ml=endpoint_difference,
        display_frames=len(list(frames.glob("state_*.npz"))),
        original_metrics_sha256=sha(original_csv), replay_metrics_sha256=sha(replay_csv),
        full_endpoint_reproduction_tested=True,
        scope="Replay reproducibility only; not held-out validation or numerical convergence"))
    assert accepted, "Video replay differs from frozen trajectory; inspect before rendering final video"
    write(WORK / "complete.json", dict(status="complete", display_frames=386))
    print("Private-video replay captured and verified", flush=True)


if __name__ == "__main__":
    main()
