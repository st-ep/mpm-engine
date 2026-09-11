"""Two forward MPM runs comparing measured and planned 52.92-degree motion.

The original weak viscosity, contact, geometry, grid, and timestep are frozen.
Only robot joint/action logs change. No measured liquid outcomes are read. This
includes measured differences in the starting hand pose and fresh settling; it
is a whole-motion diagnostic, not a pure return-delay intervention.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from experiments.pour import pour_navier_reference as reference
from experiments.pour.pour_angle_sweep import write_planned_episode
from experiments.pour.pour_coulomb_overnight import digest, numerical_problem, write
from warpmpm.geometry.measuring_cup import make_cup_mesh

ROOT = reference.ROOT
WORK = ROOT / "out/pour_physics_audit/recorded_motion_mpm_52p92"
LOGS = ROOT / "out/pour_physics_audit/validation_motion_47p06_52p92/recorded_52.92"
MODES = ["planned", "recorded"]
ANGLE = 52.92
CONTACT = .117


def prepare():
    WORK.mkdir(parents=True, exist_ok=False)
    geometry = json.loads(reference.GEOMETRY.read_text())
    offset, _, _, hc, grasp = reference.fixed_scene(geometry)
    motion, ep = reference.planned_motion()
    recorded = WORK / "inputs/recorded"
    recorded.mkdir(parents=True)
    for name in ["actions.jsonl", "states.jsonl", "meta.json"]:
        shutil.copyfile(LOGS / name, recorded / name)
        assert digest(LOGS / name) == digest(recorded / name)
    write_planned_episode(reference.REFERENCE, WORK / "inputs/planned", motion, ep, ANGLE)
    vertices, _ = make_cup_mesh(reference.twin.SPEC,
        *reference.twin.collision_extras(.35/160), n_theta=96, n_z=24)
    checks = {}
    for mode in MODES:
        path = WORK / "inputs" / mode
        e = reference.twin.load_episode(path, reference.twin.PRE_ROLL, reference.twin.HOLD_SECONDS)
        a = reference.twin.RecordedPanda(e, reference.MESH, height=64, width=64, max_geom=4000)
        a._hand_cup, a._grasp = hc.copy(), grasp.copy()
        clearance = .35
        for t in np.arange(0, a.duration+1e-9, 1/60):
            p, q = a.cup_pose_at(float(t))
            points = vertices @ reference.twin.quat_to_mat(q).T + p + offset
            clearance = min(clearance, float(min(points.min(), .35-points.max())))
        p, q = a.cup_pose_at(0.)
        checks[mode] = dict(minimum_cup_geometry_clearance_m=clearance,
            initial_cup_pos=p.tolist(), initial_cup_quat=q.tolist(),
            simulation_duration_s=a.duration, t_pour=e["t_pour"],
            return_ack_s=e["t_return_done"]-e["t_pour"])
        a.close()
        if clearance < .03:
            raise ValueError("Measured or planned motion exceeds the frozen crop")
    # Verify the same original weak fit; the new recordings supply motion only.
    from experiments.pour.pour_weakform_recovery import fit
    pour, ret, _ = reference.twin.recorded_pour_actions(reference.REFERENCE)
    window = [pour["t_ack"]-pour["t_send"]+.15, ret["t_send"]-pour["t_send"]-.15]
    eta = fit(dict(np.load(reference.OBSERVATIONS)), window)[0]
    if abs(eta-reference.ETA) > 1e-10:
        raise ValueError("Original one-pour weak viscosity does not reproduce")
    paths = [Path(__file__), Path(reference.__file__), Path(reference.twin.__file__),
        reference.GEOMETRY, reference.MESH, reference.OBSERVATIONS,
        ROOT / "experiments/pour/pour_angle_sweep.py",
        ROOT / "experiments/pour/pour_weakform_recovery.py",
        ROOT / "experiments/pour/pour_weakform_identify.py",
        ROOT / "experiments/pour/pour_coulomb_overnight.py",
        *sorted((ROOT / "src/warpmpm").rglob("*.py"))]
    paths += [reference.REFERENCE / n for n in ["actions.jsonl", "states.jsonl", "meta.json"]]
    paths += [WORK / "inputs" / m / n for m in MODES for n in ["actions.jsonl", "states.jsonl", "meta.json"]]
    protocol = dict(purpose=__doc__, angle_deg=ANGLE, number_of_runs=2,
        eta_pa_s=reference.ETA, source_coulomb_friction=CONTACT,
        source_wall="original-separable", receiver_friction=.05, initial_volume_ml=300.,
        grid=160, extent_m=.35, phase_xz_cells=0., dt_scale=1., particle_count=229280,
        identification_episode=reference.REFERENCE.name, weak_fit_window_s=window,
        parameters_refitted=[], measured_endpoints_used=[], validation_outcomes_used=[],
        physical_cup_mounting_measured=False, world_to_grid_offset=offset.tolist(),
        hand_cup=hc.tolist(), grasp=grasp.tolist(), preflight=checks,
        settling="Fresh standard settling at each recorded/planned initial hand pose",
        camera="Original camera view reused only to configure the disabled renderer; no new image calibration claimed",
        observation_window="Standard replay window and tail checks; original lift excluded",
        per_run_timeout_s=1800,
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths})
    write(WORK / "protocol.json", protocol)
    (WORK / "protocol.sha256").write_text(digest(WORK / "protocol.json")+"\n")
    write(WORK / "status.json", dict(stage="prepared", completed=[]))
    print(json.dumps(dict(eta_pa_s=eta, preflight=checks), indent=2))


def validate():
    path = WORK / "protocol.json"
    if digest(path) != (WORK / "protocol.sha256").read_text().strip():
        raise RuntimeError("Protocol changed")
    protocol = json.loads(path.read_text())
    for path, expected in protocol["input_sha256"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    return protocol


def run_case(mode, device):
    protocol = validate()
    output = WORK / mode
    output.mkdir(exist_ok=False)
    twin = reference.twin
    geometry = json.loads(reference.GEOMETRY.read_text())
    offset = np.array(protocol["world_to_grid_offset"])
    hand_cup, grasp = np.array(protocol["hand_cup"]), np.array(protocol["grasp"])
    old = twin.Solver, twin.RecordedPanda, twin.OUT_ROOT, twin.GRID_LIM, twin.SDF_RES, twin.world_to_mpm_offset, twin.side_camera_view
    clearances = []

    class PairSolver(old[0]):
        def add_sdf_collider(self, *args, **kwargs):
            index = getattr(self, "_pair_collider_count", 0)
            if index == 0:
                kwargs = dict(kwargs, surface="separable", friction=CONTACT)
            handle = super().add_sdf_collider(*args, **kwargs)
            self._pair_collider_count = index+1
            return handle

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            x = self.x()
            if not np.isfinite(x).all():
                raise RuntimeError("Nonfinite particle coordinates")
            clearance = float(min(x.min(), .35-x.max()))
            if clearance <= 3*(.35/160):
                raise RuntimeError("Liquid reached the artificial crop boundary")
            clearances.append(clearance)
            return result

    class PairPanda(old[1]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._hand_cup, self._grasp = hand_cup.copy(), grasp.copy()

    camera = old[-1](json.loads((reference.REFERENCE / "meta.json").read_text()))
    twin.Solver, twin.RecordedPanda, twin.OUT_ROOT = PairSolver, PairPanda, output
    twin.GRID_LIM, twin.SDF_RES = .35, 256
    twin.world_to_mpm_offset = lambda arm, receiver, dx: offset.copy()
    twin.side_camera_view = lambda meta: camera
    start = time.monotonic()
    try:
        sim = twin.run(WORK / "inputs" / mode, device=device, n_grid=160,
            video=False, side_by_side=False, rebake=True, eta=reference.ETA,
            volume_ml=300., **geometry)
    finally:
        (twin.Solver, twin.RecordedPanda, twin.OUT_ROOT, twin.GRID_LIM, twin.SDF_RES,
         twin.world_to_mpm_offset, twin.side_camera_view) = old
    rows = sim["rows"]
    counts = np.array([[r["n_src"], r["n_rcv"], r["n_air_spill"]] for r in rows])
    if np.any(counts < 0) or np.any(counts.sum(1) != 229280):
        raise RuntimeError("Particle ledger differs from frozen model")
    tail = [r["n_rcv"] for r in rows if r["t"] >= rows[-1]["t"]-.5]
    result = dict(mode=mode, receiver_ml=300.*counts[-1, 1]/229280,
        outside_ml=300.*counts[-1, 2]/229280, tail_variation_ml=300.*np.ptp(tail)/229280,
        eta_pa_s=reference.ETA, source_coulomb_friction=CONTACT, particle_count=229280,
        minimum_boundary_clearance_m=min(clearances), elapsed_s=time.monotonic()-start,
        protocol_sha256=digest(WORK / "protocol.json"), device=device,
        metrics_path=str((output / mode / "metrics.csv").relative_to(ROOT)))
    validate()
    result["numerical_problem"] = numerical_problem(result)
    write(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if result["numerical_problem"]:
        raise RuntimeError(result["numerical_problem"])


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    validate()
    results = {m: json.loads((WORK / m / "result.json").read_text()) for m in MODES}
    if any(r["numerical_problem"] for r in results.values()):
        raise RuntimeError("Pair failed capture/tail checks")
    delta = results["recorded"]["receiver_ml"]-results["planned"]["receiver_ml"]
    comparison = dict(results=results, recorded_minus_planned_ml=delta,
        parameters_refitted=[], measured_endpoints_used=[], validation_outcomes_used=[],
        interpretation="Motion sensitivity of the frozen numerical model, including initial pose/settling; not a fitted bias correction or proof of physical accuracy")
    write(WORK / "comparison.json", comparison)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for mode, result in results.items():
        with (ROOT / result["metrics_path"]).open() as f:
            rows = list(csv.DictReader(f))
        t = np.array([float(r["t"]) for r in rows])-reference.twin.PRE_ROLL
        axes[0].plot(t, [300*int(r["n_rcv"])/229280 for r in rows],
                     label=f"{mode}: {result['receiver_ml']:.2f} mL")
        axes[1].plot(t, [float(r["tilt_deg"]) for r in rows], label=mode)
    axes[0].set_ylabel("MPM receiver volume (mL)")
    axes[1].set_ylabel("Cup tilt with fixed grasp (°)")
    axes[1].set_xlabel("Seconds after inclination command")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle(f"52.92° measured versus planned motion · fixed weak viscosity/contact\nRecorded − planned: {delta:+.2f} mL")
    fig.savefig(WORK / "comparison.png", dpi=180)
    plt.close(fig)
    lines = ["# Recorded versus planned 52.92-degree motion", "",
        "Two full MPM forward runs, fixed weak viscosity 3.4392377844 Pa s and source contact 0.117; grid160, 229280 particles, phase0, standard timestep.", "",
        f"Planned motion: {results['planned']['receiver_ml']:.3f} mL. Recorded motion: {results['recorded']['receiver_ml']:.3f} mL. Recorded minus planned: {delta:+.3f} mL.", "",
        "The change measures sensitivity to the supplied joint/action history, including small starting-pose differences and fresh initial settling. It does not isolate return delay alone. Cup mounting remains the original 60-degree transform and was not measured for this grasp.", "",
        "No measured receiver volumes or validation outcomes were read. No viscosity, contact, geometry, or command was fitted. Existing numerical convergence concerns remain; this comparison is not a corrected handoff table.", ""]
    (WORK / "REPORT.md").write_text("\n".join(lines))
    return comparison


def launch():
    protocol = validate()
    # Do not overlap another diagnostic controller or start on occupied GPUs.
    import fcntl
    lock = (WORK / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if processes:
        raise RuntimeError("A GPU compute process is already active; pair not launched")
    children, handles = {}, []
    env = dict(os.environ, MUJOCO_GL="egl", PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    try:
        for mode, gpu in zip(MODES, [0, 1]):
            handle = (WORK / f"{mode}.log").open("w")
            handles.append(handle)
            children[mode] = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                "--case", mode, "--device", f"cuda:{gpu}"], cwd=ROOT, env=env,
                stdout=handle, stderr=subprocess.STDOUT)
        start = time.monotonic()
        while True:
            validate()
            exits = {m: p.poll() for m, p in children.items()}
            if any(v not in (None, 0) for v in exits.values()):
                raise RuntimeError(f"A replay failed: {exits}")
            write(WORK / "status.json", dict(stage="running", exits=exits,
                pids={m: p.pid for m, p in children.items()}, elapsed_s=time.monotonic()-start))
            if all(v == 0 for v in exits.values()):
                break
            if time.monotonic()-start > protocol["per_run_timeout_s"]:
                raise RuntimeError("Pair exceeded its 30-minute runtime limit")
            time.sleep(10)
        comparison = report()
        write(WORK / "status.json", dict(stage="completed_needs_review",
            recorded_minus_planned_ml=comparison["recorded_minus_planned_ml"],
            handoff_changed=False, parameters_refitted=[]))
    except BaseException as error:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        write(WORK / "status.json", dict(stage="failed", error=str(error)))
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--case", choices=MODES)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.case:
        run_case(args.case, args.device)
    else:
        launch()
