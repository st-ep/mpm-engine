"""Capture a paper still from the frozen September 8 calibration replay.

No model changes or parameter search. Instrument the original runner with one
read-only snapshot call and stop after that frame; preserve the generated source
and verify frozen inputs. Original completed simulations supply all endpoints.
Run: MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m experiments.pour.pour_paper_replay
"""
from pathlib import Path
import hashlib
import inspect
import json
import sys
import csv
import re
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "out/pour_physics_audit/aflip_blend_time_20260907"
FROZEN = BASE / "recalibration_mu0.272000_n160_dt1"
WORK = ROOT / "out/pour_figure_final_20260911/replay"
SOURCE = BASE / "isolated_src"
sys.path.insert(0, str(SOURCE))
from warpmpm.kernels import mpm_utils
from experiments.pour import pour_transfer_recalibration_reference as reference


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda v: v.item()) + "\n")


class SnapshotCaptured(Exception):
    """Successful bounded stop after the selected frame has been saved."""


def verify_snapshot():
    """Validate the rendering state, without claiming a new endpoint replay."""
    twin = reference.twin
    data = np.load(WORK / "snapshot.npz")
    assert all(np.isfinite(data[k]).all() for k in data.files)
    n = len(data["x_world"])
    assert n == 229280
    x = (data["x_world"]+data["world_to_grid_offset"]).astype(np.float32)
    counts = []
    for name in ("source", "receiver"):
        audit = twin.cup_audit(x, data[name+"_pos"], data[name+"_quat"],
            float(data["h"]), twin.collision_extras(.35/160), data["world_to_grid_offset"])
        counts.append(int(audit[1].sum()))
    counts.append(n-sum(counts))
    assert min(counts) >= 0
    frozen_result, = (FROZEN / "forward").rglob("result.json")
    frozen_csv, = frozen_result.parent.rglob("metrics.csv")
    frozen_rows = list(csv.DictReader(frozen_csv.open()))
    frame = round(float(data["t"])*twin.FPS)-1
    original = frozen_rows[frame]
    keys = ("n_src", "n_rcv", "n_air_spill")
    differences = {k: (value-int(original[k]))*300/n for k, value in zip(keys, counts)}
    trajectory = WORK / "trajectory_through_snapshot.csv"
    if trajectory.exists():
        rows = list(csv.DictReader(trajectory.open()))
        samples = [(int(r["frame"]), int(r["n_rcv"])) for r in rows]
        coverage = "Every replay frame through the snapshot"
    else:
        # Recover the already saved state from the first attempt: NumPy scalar
        # serialization failed only after the NPZ write. Preserve that failure
        # and limit the trajectory claim to the samples actually logged.
        rows = (WORK / "replay.log").read_text().splitlines()
        samples = []
        for row in rows:
            match = re.match(r"frame\s+(\d+)/773.*rcv=\s*(\d+) spill=", row)
            if match:
                samples.append((int(match[1])-1, int(match[2])))
        coverage = "Logged 30-frame samples plus the saved snapshot; full per-frame trajectory was not exported"
    samples.append((frame, counts[1]))
    maximum_delta = max(abs(v-int(frozen_rows[i]["n_rcv"]))*300/n for i, v in samples)
    accepted = max(abs(v) for v in differences.values()) <= .1 and maximum_delta <= .1
    write(WORK / "snapshot.json", dict(frame=frame, t=float(data["t"]),
        t_host=float(data["t_host"]), n_src=counts[0], n_rcv=counts[1], n_air_spill=counts[2],
        receiver_ml=300*counts[1]/n, selection="Midpoint of the recorded hold"))
    write(WORK / "verification.json", dict(
        render_snapshot_accepted=accepted, tolerance_ml=.1,
        frozen_result=str(frozen_result.relative_to(ROOT)), frozen_result_sha256=sha(frozen_result),
        frozen_metrics_sha256=sha(frozen_csv), snapshot_sha256=sha(WORK / "snapshot.npz"),
        frame=frame, snapshot_count_differences_ml=differences,
        sampled_receiver_trajectory_maximum_difference_ml=maximum_delta,
        trajectory_samples=len(samples), trajectory_coverage=coverage,
        full_endpoint_reproduction_tested=False,
        endpoint_source="Original completed and independently reviewed September 8 outputs; unchanged",
        note="Snapshot agreement is not a convergence check. Documented failed numerical checks remain unchanged."))
    assert accepted, "Rendering snapshot differs from the frozen replay"


def main():
    WORK.mkdir(parents=True, exist_ok=False)
    parent = FROZEN / "protocol.json"
    inputs = json.loads(parent.read_text())["input_sha256"]
    # User explicitly deleted obsolete handoff ZIPs; none are dynamics inputs.
    deleted = "out/pour_navier_calibration/philip_pilot_20260907_200/philip_pilot_handoff.zip"
    unavailable = []
    for rel, expected in inputs.items():
        path = ROOT / rel
        if rel == deleted and not path.exists():
            unavailable.append(dict(path=rel, frozen_sha256=expected,
                reason="Superseded command ZIP deleted by user; not a dynamics input"))
            continue
        assert sha(path) == expected, rel
    assert Path(mpm_utils.__file__).resolve().is_relative_to(SOURCE)
    twin = reference.twin
    ep = twin.load_episode(reference.REFERENCE, twin.PRE_ROLL, twin.HOLD_SECONDS)
    # Fixed in advance: midpoint of the recorded hold, rounded to the replay tick.
    snapshot_time = round((ep["t_hold"] + ep["t_return_start"]) * .5 * twin.FPS) / twin.FPS
    write(WORK / "provenance.json", dict(
        purpose=__doc__, frozen_protocol=str(parent.relative_to(ROOT)),
        frozen_protocol_sha256=sha(parent), actual_kernel=str(mpm_utils.__file__),
        kernel_sha256=sha(Path(mpm_utils.__file__)), checked_input_count=len(inputs)-len(unavailable),
        unavailable_non_dynamics_artifacts=unavailable, snapshot_time_s=snapshot_time,
        selection="Midpoint of recorded hold, rounded to nearest 60 Hz simulation tick",
        calibration_not_held_out=True, identification_changed=False,
        device="cuda:0", input_sha256=inputs, script_sha256=sha(Path(__file__))))

    def capture(variables):
        if abs(variables["t_now"] - snapshot_time) > 1e-8:
            return
        l = variables
        np.savez_compressed(WORK / "snapshot.npz", x_world=l["x"]-l["w2m"],
            v=l["v"], vol=l["vol"], source_pos=l["p_now"], source_quat=l["q_now"],
            receiver_pos=l["receiver_pos"], receiver_quat=twin.Q_RCV,
            world_to_grid_offset=l["w2m"], h=l["h"], t=l["t_now"],
            t_host=ep["t0"]+l["t_now"], hand_cup=l["arm"]._hand_cup, grasp=l["arm"]._grasp)
        write(WORK / "snapshot.json", dict(row=l["rows_out"][-1],
            t_host=ep["t0"]+l["t_now"], selection="Midpoint of the recorded hold"))
        with (WORK / "trajectory_through_snapshot.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(l["rows_out"][0]))
            writer.writeheader()
            writer.writerows(l["rows_out"])
        l["arm"].close()
        print("Saved paper snapshot", snapshot_time, flush=True)
        raise SnapshotCaptured

    original = inspect.getsource(twin.run)
    anchor = "        if video:\n            img = render_frame"
    assert original.count(anchor) == 1
    instrumented = original.replace(anchor, "        _paper_capture(locals())\n\n" + anchor)
    (WORK / "original_run.py").write_text(original)
    (WORK / "instrumented_run.py").write_text(instrumented)
    twin.__dict__["_paper_capture"] = capture
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
    except SnapshotCaptured:
        pass
    finally:
        twin.project_out_of_solid = old_project
    verify_snapshot()


if __name__ == "__main__":
    if "--verify-snapshot" in sys.argv:
        verify_snapshot()
    else:
        main()
