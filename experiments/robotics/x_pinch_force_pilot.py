"""Single cylindrical pinch with force feedback, separate from the paper study.

Run prepare, then sweep for A/B, then verify. A fresh directory freezes each
controller revision. Force is per finger, not the cancelling net wrist force.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from experiments.robotics import x_shaping as core
from experiments.robotics.plastic_shaping_study import save_json, save_npz

ROOT = core.ROOT
OUT = ROOT / "out/x_pinch_force_pilot_20260909"
BASE = ROOT / "out/x_letter_cross_20260909"
SETTINGS = {"baseline": (64, 0.00005), "half_dt": (64, 0.000025), "grid80": (80, 0.00005)}
CONTROL = dict(
    start_gap_m=0.064, touch_speed_m_s=0.010, pre_pulse_wait_s=0.20,
    rise_s=0.40, hold_s=1.20, fall_s=0.40,
    kp_m_s_n=0.050, ki_m_s2_n=0.040, filter_s=0.012,
    max_speed_m_s=0.050, max_acceleration_m_s2=0.50,
    min_gap_m=0.004, max_gap_m=0.072,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pulse(t, peak):
    c = CONTROL
    if t < c["rise_s"]:
        return peak * 0.5 * (1 - np.cos(np.pi * t / c["rise_s"]))
    if t < c["rise_s"] + c["hold_s"]:
        return peak
    u = (t - c["rise_s"] - c["hold_s"]) / c["fall_s"]
    return peak * 0.5 * (1 + np.cos(np.pi * np.clip(u, 0, 1)))


def prepare(folder, amplitudes):
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "protocol.json").exists(), "Use a fresh directory"
    source = folder / "source_snapshot"
    paths = list((ROOT / "src/warpmpm").rglob("*.py"))
    # Freeze the helpers and their transitive imports, including historical
    # experiment modules imported by the common geometry/serialization code.
    paths += [p for p in (ROOT / "experiments/robotics").glob("*.py")
              if p.name != "x_pinch_force_report.py"]
    hashes = {}
    for path in sorted(set(paths)):
        rel = path.relative_to(ROOT)
        dest = source / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[str(rel)] = digest(path)
    save_json(folder / "source_sha256.json", hashes)
    shutil.copy2(BASE / "inputs/models.json", folder / "models.json")
    save_json(folder / "protocol.json", dict(
        scene=core.CONFIG, control=CONTROL, settings=SETTINGS,
        amplitudes_n=amplitudes, seed=0, models_sha256=digest(folder / "models.json"),
        direction="One centered y pinch; unchanged X-study specimen and contacts",
        feedback="PI on mean compressive per-finger force, EMA, conditional integration and velocity/acceleration limits",
        observation="1 s after full opening and withdrawal, not assumed equilibrium",
        force="Signed outward y projection per finger; no clipping before filtering",
        scope="Controller feasibility; no identification, shape planning or hardware execution",
    ))
    (folder / "packages.txt").write_text("\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
    )))
    (folder / "gpu_before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    (folder / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True))
    (folder / "tracked_before.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"]))
    save_json(folder / "paper_before_sha256.json", {
        str(p.relative_to(ROOT)): digest(p)
        for p in [ROOT / "paper/icra2027/paper.tex", ROOT / "paper/icra2027/paper.pdf",
                  ROOT / "paper/icra2027/figs/identification_plastic_shaping.pdf"]
    })


def check(folder):
    p = json.loads((folder / "protocol.json").read_text())
    assert p["control"] == CONTROL and p["scene"] == core.CONFIG
    assert p["models_sha256"] == digest(folder / "models.json")
    for rel, sha in json.loads((folder / "source_sha256.json").read_text()).items():
        assert digest(ROOT / rel) == digest(folder / "source_snapshot" / rel) == sha, rel
    return p


def execute(law, peak, grid, dt, device):
    cfg, c = core.CONFIG, CONTROL
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
    data = dict(initial=x, vol0=vol, floor=np.array(cfg["floor"]),
                n_grid=np.array(grid), dt=np.array(dt), peak_force_n=np.array(peak))
    fields = {key: [] for key in ["time", "tool_centers", "tool_velocity", "reaction_force",
              "force_per_finger", "target_force", "filtered_force", "force_error",
              "integral_velocity", "command_velocity", "speed_limited", "acceleration_limited",
              "travel_limited", "phase_id"]}
    phases = []
    elapsed = 0.0
    filtered = 0.0
    alpha = 1 - np.exp(-tick / c["filter_s"])
    current_phase = -1

    def start_phase(name):
        nonlocal current_phase
        phases.append(dict(name=name, start_s=elapsed))
        current_phase = len(phases) - 1

    def advance(nxt, target=0., error=0., integral=0., command=0., speed=False,
                acceleration=False, travel=False):
        nonlocal pose, elapsed, filtered
        velocity = (nxt - pose) / tick
        for h, cp, v in zip(handles, pose, velocity, strict=True):
            solver.set_sdf_pose(h, center=cp, velocity=v)
            solver.reset_sdf_force(h)
        solver.step(dt, substeps=substeps)
        force = np.stack([solver.sdf_wrench(h, tick)["force"] for h in handles])
        compressive = force[:, 1] * np.array([-1., 1.])
        filtered += alpha * (compressive.mean() - filtered)
        elapsed += tick
        pose = nxt
        row = dict(time=elapsed, tool_centers=pose.copy(), tool_velocity=velocity,
                   reaction_force=force, force_per_finger=compressive,
                   target_force=target, filtered_force=filtered, force_error=error,
                   integral_velocity=integral, command_velocity=command,
                   speed_limited=speed, acceleration_limited=acceleration,
                   travel_limited=travel, phase_id=current_phase)
        for key in fields:
            fields[key].append(row[key])

    def move(end, duration, name):
        start_phase(name)
        start = pose.copy()
        count = max(1, int(np.ceil(duration / tick)))
        for j in range(count):
            advance(start + (end - start) * (j + 1) / count)

    move(core.centers(np.pi / 2, cfg["opening"], cfg["finger_bottom"]),
         (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"], "lower")
    move(core.centers(np.pi / 2, c["start_gap_m"], cfg["finger_bottom"]),
         (cfg["opening"] - c["start_gap_m"]) / (2 * c["touch_speed_m_s"]), "engage")
    move(pose.copy(), c["pre_pulse_wait_s"], "pre_pulse_wait")
    data["before_pulse"] = solver.x().copy()
    start_phase("force_pulse")
    data["pulse_start_s"] = np.array(elapsed)
    duration = c["rise_s"] + c["hold_s"] + c["fall_s"]
    integral, last_velocity = 0., 0.
    minimum_gap = c["max_gap_m"]
    pulse_frames, pulse_times = [], []
    for j in range(round(duration / tick)):
        target = pulse((j + 0.5) * tick, peak)
        error = target - filtered
        trial_integral = integral + c["ki_m_s2_n"] * error * tick
        raw = c["kp_m_s_n"] * error + trial_integral
        v = np.clip(raw, -c["max_speed_m_s"], c["max_speed_m_s"])
        speed_limited = abs(raw - v) > 1e-12
        va = np.clip(v, last_velocity - c["max_acceleration_m_s2"] * tick,
                     last_velocity + c["max_acceleration_m_s2"] * tick)
        acceleration_limited = abs(va - v) > 1e-12
        gap = pose[1, 1] - pose[0, 1] - 2 * cfg["radius"]
        next_gap = np.clip(gap - 2 * va * tick, c["min_gap_m"], c["max_gap_m"])
        actual_v = (gap - next_gap) / (2 * tick)
        travel_limited = abs(actual_v - va) > 1e-12
        # Integrate only when unsaturated, or when the error unwinds saturation.
        if (not (speed_limited or acceleration_limited or travel_limited)
                or error * (raw - actual_v) <= 0):
            integral = trial_integral
        advance(core.centers(np.pi / 2, next_gap, cfg["finger_bottom"]),
                target, error, integral, raw, speed_limited, acceleration_limited, travel_limited)
        last_velocity = actual_v
        if next_gap < minimum_gap:
            minimum_gap = next_gap
            data["most_compressed"] = solver.x().copy()
            data["compressed_centers"] = pose.copy()
            data["compressed_time_s"] = np.array(elapsed)
        if j % 25 == 24:
            pulse_frames.append(solver.x().copy())
            pulse_times.append(elapsed)
    data["pulse_frames"] = np.array(pulse_frames)
    data["pulse_frame_times"] = np.array(pulse_times)
    data["end_pulse"] = solver.x().copy()
    gap = pose[1, 1] - pose[0, 1] - 2 * cfg["radius"]
    move(core.centers(np.pi / 2, cfg["opening"], cfg["finger_bottom"]),
         abs(cfg["opening"] - gap) / (2 * cfg["finger_speed"]), "open")
    move(core.centers(np.pi / 2, cfg["opening"], cfg["hover_bottom"]),
         (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"], "withdraw")
    data["x_after_withdrawal"] = solver.x().copy()
    move(pose.copy(), cfg["release_s"], "wait")
    data.update(x_after_1s=solver.x().copy(), v_after_1s=solver.v().copy(),
                inverted_count=np.array(solver.inverted_count()))
    for key, values in fields.items():
        data[key] = np.array(values)
    for i, phase in enumerate(phases):
        phase["duration_s"] = (phases[i + 1]["start_s"] if i + 1 < len(phases) else elapsed) - phase["start_s"]
    assert int(data["inverted_count"]) == 0
    assert all(np.all(np.isfinite(v)) for v in data.values())
    return data, phases


def metrics(data):
    c = CONTROL
    t = data["time"] - data["pulse_start_s"]
    active = (t > 0) & (t <= c["rise_s"] + c["hold_s"] + c["fall_s"] + 1e-9)
    plateau = (t > c["rise_s"] + 0.2) & (t <= c["rise_s"] + c["hold_s"])
    f = data["force_per_finger"].mean(axis=1)
    target = data["target_force"]
    xyz = data["x_after_1s"]
    vol = data["vol0"].astype(float)
    gap = data["tool_centers"][:, 1, 1] - data["tool_centers"][:, 0, 1] - 2 * core.CONFIG["radius"]
    # A raw-particle measure, independent of rendered surface threshold.
    mid = np.abs(xyz[:, 0] - core.CONFIG["domain"] / 2) < 0.0025
    return dict(
        min_gap_mm=float(gap[active].min() * 1000),
        force_plateau_mean_n=float(f[plateau].mean()),
        force_plateau_rmse_n=float(np.sqrt(np.mean((f[plateau] - target[plateau]) ** 2))),
        force_peak_n=float(data["force_per_finger"][active].max()),
        signed_force_min_n=float(data["force_per_finger"][active].min()),
        force_left_right_rmse_n=float(np.sqrt(np.mean(np.diff(data["force_per_finger"][active], axis=1) ** 2))),
        force_tracking_mae_n=float(np.mean(abs(f[active] - target[active]))),
        speed_limited_fraction=float(data["speed_limited"][active].mean()),
        acceleration_limited_fraction=float(data["acceleration_limited"][active].mean()),
        travel_limited_fraction=float(data["travel_limited"][active].mean()),
        raw_waist_mm=float(np.ptp(xyz[mid, 1]) * 1000),
        raw_waist_98_mm=float(np.diff(np.quantile(xyz[mid, 1], [.01, .99]))[0] * 1000),
        released_height_mm=float((xyz[:, 2].max() - core.CONFIG["floor"]) * 1000),
        rms_displacement_mm=float(np.sqrt(np.average(np.sum((xyz - data["initial"]) ** 2, axis=1), weights=vol)) * 1000),
        rms_speed_mm_s=float(np.sqrt(np.average(np.sum(data["v_after_1s"] ** 2, axis=1), weights=vol)) * 1000),
        floor_max_penetration_mm=float(np.maximum(core.CONFIG["floor"] - xyz[:, 2], 0).max() * 1000),
        volume_ml=float(vol.sum() * 1e6),
    )


def sweep(folder, material, setting, device, amplitudes=None):
    protocol = check(folder)
    law = json.loads((folder / "models.json").read_text())[f"true_{material}"]
    grid, dt = SETTINGS[setting]
    for peak in protocol["amplitudes_n"] if amplitudes is None else amplitudes:
        assert peak in protocol["amplitudes_n"]
        # Decimal amplitudes must not be mistaken for a filename extension.
        path = folder / f"{setting}_{material}_{peak:g}N.npz"
        config = dict(material=material, peak_force_n=peak, law=law, n_grid=grid, dt=dt,
                      setting=setting, device=device, seed=0, protocol_sha256=digest(folder / "protocol.json"))
        config_path = path.with_suffix(".config.json")
        start = time.monotonic()
        if path.exists():
            assert json.loads(config_path.read_text()) == config
            data = np.load(path)
        else:
            save_json(config_path, config)
            print(json.dumps(dict(starting=path.name, device=device)), flush=True)
            data, phases = execute(law, peak, grid, dt, device)
            save_npz(path, data)
            save_json(path.with_suffix(".phases.json"), phases)
        record = dict(**config, **metrics(data), data_sha256=digest(path))
        save_json(path.with_suffix(".json"), record)
        print(json.dumps(dict(**record, elapsed_s=time.monotonic() - start)), flush=True)


def verify(folder):
    check(folder)
    rows = []
    laws = json.loads((folder / "models.json").read_text())
    for path in sorted(folder.glob("*.npz")):
        data = np.load(path)
        cfg = json.loads(path.with_suffix(".config.json").read_text())
        record = json.loads(path.with_suffix(".json").read_text())
        assert cfg["law"] == laws[f"true_{cfg['material']}"]
        assert cfg["protocol_sha256"] == digest(folder / "protocol.json")
        assert int(data["inverted_count"]) == 0
        assert all(np.all(np.isfinite(data[k])) for k in data.files)
        initial, vol = core.specimen(cfg["n_grid"])
        np.testing.assert_array_equal(data["initial"], initial)
        np.testing.assert_array_equal(data["vol0"], vol)
        np.testing.assert_allclose(vol.astype(float).sum(), .00009, rtol=1e-6)
        np.testing.assert_allclose(data["tool_centers"][:, :, 0], .08, atol=1e-10)
        np.testing.assert_allclose(data["tool_centers"][:, :, 1].mean(axis=1), .08, atol=1e-10)
        np.testing.assert_allclose(data["force_per_finger"], data["reaction_force"][:, :, 1] * [-1., 1.])
        assert record["data_sha256"] == digest(path)
        for key, value in metrics(data).items():
            np.testing.assert_allclose(record[key], value, rtol=1e-10, atol=1e-10)
        rows.append(record)
    assert rows
    for rel, sha in json.loads((folder / "paper_before_sha256.json").read_text()).items():
        assert digest(ROOT / rel) == sha, rel
    assert subprocess.check_output(["git", "diff", "--binary"]) == (folder / "tracked_before.diff").read_bytes()
    save_json(folder / "summary.json", rows)
    print(json.dumps(dict(verified_runs=len(rows), paper_unchanged=True)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "sweep", "verify"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--amplitudes", type=float, nargs="+")
    p.add_argument("--material", choices=["A", "B"])
    p.add_argument("--setting", choices=list(SETTINGS), default="baseline")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.out, args.amplitudes if args.amplitudes is not None else [0., .5, .75, 1., 1.25, 2.])
    elif args.stage == "sweep":
        assert args.material is not None
        sweep(args.out, args.material, args.setting, args.device, args.amplitudes)
    else:
        verify(args.out)
