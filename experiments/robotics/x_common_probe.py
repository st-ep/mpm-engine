"""Identical cylindrical squeeze/release for supplied-state identification.

This protocol replaces the gentle, non-identifying force illustration. Positions,
velocities and elastic F are synthetic observations; force is held out of fitting.
Yield-activity diagnostics use ground truth ONLY after recording, never in fitting.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import time

import numpy as np

from experiments.robotics import x_shaping as core
from experiments.robotics.plastic_shaping_study import TRUTH, save_json, save_npz

PROBE = dict(gap_m=.032, speed_m_s=.020, settle_s=.20, hold_s=.10,
             frame_dt=.004, direction="y", observation_after_withdrawal_s=1.)
VALIDATION = dict(PROBE, gap_m=.024, speed_m_s=.030)


def dev_norm(F):
    eps = np.log(np.maximum(np.linalg.svd(F, compute_uv=False), .01))
    return np.linalg.norm(eps - eps.mean(axis=-1, keepdims=True), axis=-1)


def execute(law, *, protocol=PROBE, grid=64, dt=.00005, device="cuda:0", states=True):
    cfg = core.CONFIG
    tick = cfg["tick"]
    substeps = round(tick / dt)
    assert abs(substeps * dt - tick) < 1e-12
    x, vol = core.specimen(grid)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = core.Solver(core.GridConfig(n_grid=grid, grid_lim=cfg["domain"]),
                             device=device, inversion_policy="raise").load_particles(x, vol)
        solver.set_material(core.vonmises(**law, density=cfg["density"]))
    solver.add_plane(point=(0, 0, cfg["floor"]), normal=(0, 0, 1),
                     surface="separable", friction=cfg["floor_friction"])
    sdf = core.cylinder_sdf(cfg["radius"], cfg["finger_height"])
    pose = core.centers(np.pi / 2, cfg["opening"], cfg["hover_bottom"])
    handles = [solver.add_sdf_collider(sdf, center=cp, band=cfg["band"],
               surface="separable", friction=cfg["tool_friction"]) for cp in pose]
    data = dict(initial=x, vol0=vol, mass=vol * cfg["density"], floor=np.array(cfg["floor"]),
                n_grid=np.array(grid), grid_lim=np.array(cfg["domain"]), dt=np.array(dt),
                frame_dt=np.array(tick), gap_m=np.array(protocol["gap_m"]))
    fields = {k: [] for k in ["time", "tool_centers", "tool_velocity", "reaction_force", "phase_id"]}
    observations = {k: [] for k in ["x", "v", "F", "state_time", "audit_trial_dev_norm"]}
    phases = []
    elapsed = 0.

    def observe():
        if states:
            observations["x"].append(solver.x().copy())
            observations["v"].append(solver.v().copy())
            observations["F"].append(solver.F().copy())
            observations["state_time"].append(elapsed)
            # Outgoing trial is the input to the NEXT substep's return map.
            # It is an independent excitation audit, not an estimator channel.
            trial = solver._sim.mpm_state.particle_F_trial.numpy()
            observations["audit_trial_dev_norm"].append(dev_norm(trial))

    def move(end, duration, name, record=False):
        nonlocal pose, elapsed
        start = pose.copy()
        count = max(1, int(np.ceil(duration / tick)))
        phases.append(dict(name=name, start_s=elapsed, duration_s=count * tick))
        for j in range(count):
            nxt = start + (end - start) * (j + 1) / count
            velocity = (nxt - pose) / tick
            for h, cp, v in zip(handles, pose, velocity, strict=True):
                solver.set_sdf_pose(h, center=cp, velocity=v)
                solver.reset_sdf_force(h)
            solver.step(dt, substeps=substeps)
            force = np.stack([solver.sdf_wrench(h, tick)["force"] for h in handles])
            elapsed += tick
            pose = nxt
            row = dict(time=elapsed, tool_centers=pose.copy(), tool_velocity=velocity,
                       reaction_force=force, phase_id=len(phases) - 1)
            for key in fields:
                fields[key].append(row[key])
            if record:
                observe()

    move(core.centers(np.pi / 2, cfg["opening"], cfg["finger_bottom"]),
         (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"], "lower")
    move(pose.copy(), protocol["settle_s"], "settle")
    data["before_probe"] = solver.x().copy()
    data["probe_start_s"] = np.array(elapsed)
    observe()
    move(core.centers(np.pi / 2, protocol["gap_m"], cfg["finger_bottom"]),
         (cfg["opening"] - protocol["gap_m"]) / (2 * protocol["speed_m_s"]), "close", True)
    data.update(most_compressed=solver.x().copy(), compressed_centers=pose.copy())
    move(pose.copy(), protocol["hold_s"], "hold")
    move(core.centers(np.pi / 2, cfg["opening"], cfg["finger_bottom"]),
         (cfg["opening"] - protocol["gap_m"]) / (2 * protocol["speed_m_s"]), "open")
    move(core.centers(np.pi / 2, cfg["opening"], cfg["hover_bottom"]),
         (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"], "withdraw")
    data["x_after_withdrawal"] = solver.x().copy()
    move(pose.copy(), protocol["observation_after_withdrawal_s"], "wait")
    data.update(x_after_1s=solver.x().copy(), v_after_1s=solver.v().copy(),
                inverted_count=np.array(solver.inverted_count()))
    data.update({k: np.asarray(v) for k, v in fields.items()})
    if states:
        data.update({k: np.asarray(v) for k, v in observations.items()})
    assert not int(data["inverted_count"])
    assert all(np.isfinite(v).all() for v in data.values())
    return data, phases


def identify(data, *, stride=2, window_frames=26, margin=2.):
    from experiments.nclaw.suite import identify_yield
    from ident.weakform.elastic_grid import assemble_elastic_timeweak, solve_elastic_grid

    # A conservative contact-free slab: all cylindrical contacts lie outside
    # the deepest inner tangent planes. Remove floor-contact support as well.
    center = float(data["grid_lim"]) / 2
    half_gap = float(data["gap_m"]) / 2
    planes = [((0, 0, float(data["floor"])), (0, 0, 1)),
              ((0, center - half_gap, 0), (0, 1, 0)),
              ((0, center + half_gap, 0), (0, -1, 0))]
    system = assemble_elastic_timeweak(
        data["x"], data["F"], data["v"], data["vol0"], data["mass"],
        np.array([0., 0., -9.81]), float(data["frame_dt"]) * stride,
        int(data["n_grid"]), float(data["grid_lim"]),
        frames=list(range(0, len(data["x"]), stride)), window_frames=window_frames,
        collider_planes=planes, collider_margin_cells=margin, columns="hencky")
    fit = solve_elastic_grid(system)
    yfit = identify_yield({"F": data["F"]}, fit["mu"], log=lambda _: None)
    accepted = not yfit["refused"] and fit["E"] > 0 and 0 < fit["nu"] < .49
    return dict(accepted=accepted,
                law=dict(E=fit["E"], nu=fit["nu"], yield_stress=yfit["yield_stress"]),
                elastic=fit, plastic=yfit, frame_stride=stride, window_frames=window_frames,
                contact_margin_cells=margin, exclusion_planes=planes,
                channels=["x", "v", "elastic F", "mass", "reference volume"],
                force_used=False, ground_truth_used=False)


def excitation_audit(data, law):
    cap = law["yield_stress"] / (law["E"] / (1 + law["nu"]))
    eps = np.stack([dev_norm(F) for F in data["F"]])
    at_cap = eps >= .98 * cap
    trial_over = data["audit_trial_dev_norm"] > cap * 1.0001
    close = data["phase_id"] == 2
    return dict(true_cap=cap, at_true_cap_fraction=float(at_cap.mean()),
                ever_at_true_cap_fraction=float(at_cap.any(axis=0).mean()),
                outgoing_trial_over_true_cap_fraction=float(trial_over.mean()),
                ever_outgoing_trial_over_true_cap_fraction=float(trial_over.any(axis=0).mean()),
                peak_closing_force_per_finger_n=float(np.linalg.norm(data["reaction_force"][close], axis=-1).max()),
                released_rms_displacement_mm=float(np.sqrt(np.mean(np.sum(
                    (data["x_after_1s"] - data["before_probe"]) ** 2, axis=-1))) * 1000),
                diagnostic_only="Uses simulator truth; excluded from identification")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["record", "fit", "audit"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--material", choices=["A", "B"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gap-mm", type=float, default=32.)
    p.add_argument("--speed-mm-s", type=float, default=20.)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"probe_{args.material}.npz"
    start = time.monotonic()
    if args.stage == "record":
        assert not path.exists(), "Use a new directory"
        protocol = dict(PROBE, gap_m=args.gap_mm / 1000, speed_m_s=args.speed_mm_s / 1000)
        save_json(path.with_suffix(".config.json"), dict(protocol=protocol, law=TRUTH[args.material],
                  grid=64, dt=.00005, device=args.device, seed=0))
        data, phases = execute(TRUTH[args.material], protocol=protocol, device=args.device)
        save_npz(path, data)
        save_json(path.with_suffix(".phases.json"), phases)
        result = excitation_audit(data, TRUTH[args.material])
        save_json(args.out / f"excitation_{args.material}.json", result)
    elif args.stage == "fit":
        result = identify(dict(np.load(path)))
        save_json(args.out / f"identification_{args.material}.json", result)
    else:
        result = excitation_audit(dict(np.load(path)), TRUTH[args.material])
        save_json(args.out / f"excitation_{args.material}.json", result)
    print(json.dumps(dict(material=args.material, elapsed_s=time.monotonic() - start, **result)), flush=True)


if __name__ == "__main__":
    main()
