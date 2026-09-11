"""Reproducible identified-model force planning and A/B cross-execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pyvista as pv
from scipy.optimize import minimize

from experiments.robotics.x_force_control import execute, diagnostics
from experiments.robotics.x_pinch_force_pilot import CONTROL, OUT as PILOT, check as check_pilot
from experiments.robotics.x_letter_study import OUT as POSITION, ROOT, SETTINGS, digest, score
from experiments.robotics.plastic_shaping_study import save_json, save_npz
from experiments.robotics.x_shaping import CONFIG

OUT = ROOT / "out/x_force_study_20260909_v2"
PROTOCOL = dict(scene=CONFIG, control=CONTROL, angles_deg=[90., 0., 90., 0.],
    settings=SETTINGS, max_evals=32, simplex_fraction=.10, force_bounds_n=[.05, 14.],
    initialization="0.8 times mean per-finger closing force over final 0.1 s of each pinch in the existing identified-model position plan; reduced after overcompression with multiplier 1.1",
    prior_planning="Each initialization trajectory came from 32 previous position-plan evaluations in its identified model",
    objective="Symmetric area-weighted mean closest-triangle surface distance at 1 s after withdrawal",
    feasibility="Add 100 mm times summed per-pulse travel-limited fractions, plus 100 mm times relative height excess above 45.5 mm; select only feasible trials",
    target="Unchanged rounded X from the position study, 64 x 80 mm, 90 mL",
    scope="Four force-command amplitudes; fixed low-level PI controller; no shape feedback or online replanning",
    seed=0)


def prepare(folder):
    check_pilot(PILOT)
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "protocol.json").exists(), "Use a fresh output directory"
    save_json(folder / "protocol.json", PROTOCOL)
    hashes = json.loads((PILOT / "source_sha256.json").read_text())
    for name in ["x_force_control.py", "x_force_study.py"]:
        path = Path(__file__).with_name(name)
        hashes[str(path.relative_to(ROOT))] = digest(path)
    for rel, sha in hashes.items():
        assert digest(ROOT / rel) == sha, rel
        dst = folder / "source_snapshot" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    save_json(folder / "source_sha256.json", hashes)
    inputs = folder / "inputs"
    inputs.mkdir()
    shutil.copy2(PILOT / "models.json", inputs / "models.json")
    shutil.copy2(PILOT / "packages.txt", folder / "packages.txt")
    for name in ["target.vtp", "target.json", "target_outline_m.npy"]:
        shutil.copy2(POSITION / name, folder / name)
    extracts = {}
    for path in sorted((POSITION / "inputs").glob("validation_*.npz")):
        data = np.load(path)
        save_npz(inputs / path.name, {key: data[key] for key in ["disp", "force"]})
        extracts[path.name] = dict(source=str(path), sha256=digest(path), fields=["disp", "force"])
    for name in "AB":
        old = PILOT / f"baseline_{name}_1.25N.npz"
        for suffix in [".npz", ".json", ".config.json", ".phases.json"]:
            shutil.copy2(old.with_suffix(suffix), inputs / f"same_force_{name}{suffix}")
        selected = json.loads((POSITION / f"plans/identified_{name}/selected.json").read_text())
        old = POSITION / selected["file"]
        data = np.load(old)
        phases = json.loads(old.with_suffix(".phases.json").read_text())
        seeds = []
        for i in range(4):
            idx = next(j for j, p in enumerate(phases) if p["name"] == f"{i}:close")
            force = data["reaction_force"][data["phase_id"] == idx][-25:]
            angle = np.deg2rad(data["angles_deg"][i])
            normal = np.array([np.cos(angle), np.sin(angle), 0.])
            seeds.append(float(.8 * ((force @ normal) * [-1., 1.]).mean()))
        save_json(inputs / f"initialization_{name}.json", dict(
            peaks_n=seeds, selected_position_plan=selected,
            source=str(old), source_sha256=digest(old), force_window_s=.1, multiplier=.8))
    save_json(inputs / "press_extracts.json", extracts)
    save_json(folder / "input_sha256.json", {
        str(p.relative_to(folder)): digest(p)
        for p in [*inputs.iterdir(), folder / "target.vtp", folder / "target.json", folder / "target_outline_m.npy"]
    })
    shutil.copytree(POSITION / "franka/panda_model_snapshot", folder / "franka/panda_model_snapshot")
    save_json(folder / "robot_input_sha256.json", {
        str(p.relative_to(folder)): digest(p) for p in (folder / "franka/panda_model_snapshot").rglob("*") if p.is_file()
    })
    for name in ["paper.tex", "paper.pdf", "figs/identification_plastic_shaping.pdf", "figs/identification_plastic_shaping.png"]:
        dst = folder / "manuscript_before" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "paper/icra2027" / name, dst)
    (folder / "tracked_before.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"]))
    (folder / "gpu_before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    (folder / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True))


def check(folder):
    p = json.loads((folder / "protocol.json").read_text())
    assert p == json.loads(json.dumps(PROTOCOL))
    for rel, sha in json.loads((folder / "source_sha256.json").read_text()).items():
        assert digest(ROOT / rel) == digest(folder / "source_snapshot" / rel) == sha, rel
    for manifest in ["input_sha256.json", "robot_input_sha256.json"]:
        for rel, sha in json.loads((folder / manifest).read_text()).items():
            assert digest(folder / rel) == sha, rel


def rollout(folder, path, law, peaks, grid, dt, device):
    config = dict(law=law, peaks_n=np.round(peaks, 6).tolist(), n_grid=grid, dt=dt,
                  device=device, seed=0, protocol_sha256=digest(folder / "protocol.json"))
    if path.exists():
        assert json.loads(path.with_suffix(".config.json").read_text()) == config
    else:
        save_json(path.with_suffix(".config.json"), config)
        data, phases = execute(law, config["peaks_n"], grid, dt, device)
        save_npz(path, data)
        save_json(path.with_suffix(".phases.json"), phases)
    return np.load(path), json.loads(path.with_suffix(".phases.json").read_text())


def plan(folder, name, device):
    check(folder)
    law = json.loads((folder / "inputs/models.json").read_text())[f"identified_{name}"]
    initial = np.array(json.loads((folder / f"inputs/initialization_{name}.json").read_text())["peaks_n"])
    target = pv.read(folder / "target.vtp")
    dest = folder / "plans" / name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "selected.json").exists():
        print((dest / "selected.json").read_text(), flush=True)
        return
    rows = []

    def objective(relative):
        peaks = np.round(relative * initial, 6)
        import hashlib
        path = dest / (hashlib.sha256(peaks.tobytes()).hexdigest()[:16] + ".npz")
        start = time.monotonic()
        print(json.dumps(dict(starting=name, evaluation=len(rows) + 1, peaks_n=peaks.tolist())), flush=True)
        try:
            data, phases = rollout(folder, path, law, peaks, 64, .00005, device)
            diagnostic = diagnostics(data, phases)
            error = score(data, target)
            feasible = not any(p["travel_limited_fraction"] > 0 for p in diagnostic["pulses"])
            feasible = feasible and diagnostic["max_recorded_height_mm"] <= 45.5
            penalty = 100 * sum(p["travel_limited_fraction"] for p in diagnostic["pulses"])
            penalty += 100 * max(0, diagnostic["max_recorded_height_mm"] / 45.5 - 1)
            row = dict(peaks_n=peaks.tolist(), file=str(path.relative_to(folder)),
                       surface_mm=error, feasible=feasible, objective_mm=error + penalty,
                       diagnostics=diagnostic)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            row = dict(peaks_n=peaks.tolist(), file=str(path.relative_to(folder)),
                       feasible=False, objective_mm=1000., failure=str(exc))
        rows.append(row)
        save_json(path.with_suffix(".json"), row)
        save_json(dest / "evaluations.json", rows)
        print(json.dumps(dict(model=name, evaluation=len(rows), elapsed_s=time.monotonic() - start,
                              peaks_n=row["peaks_n"], surface_mm=row.get("surface_mm"),
                              feasible=row["feasible"], min_gaps_mm=[p["min_gap_mm"] for p in row.get("diagnostics", {}).get("pulses", [])])), flush=True)
        return row["objective_mm"]

    simplex = np.ones((5, 4))
    simplex[1:] += PROTOCOL["simplex_fraction"] * np.eye(4)
    bounds = np.array(PROTOCOL["force_bounds_n"])[None, :] / initial[:, None]
    result = minimize(objective, np.ones(4), method="Nelder-Mead", bounds=bounds,
        options=dict(maxfev=PROTOCOL["max_evals"], xatol=0., fatol=0., initial_simplex=simplex))
    candidates = [r for r in rows if r["feasible"]]
    assert candidates, "No feasible force plan found"
    best = min(candidates, key=lambda r: r["surface_mm"])
    save_json(dest / "selected.json", dict(**best, model=f"identified_{name}", law=law,
        evaluations=len(rows), optimizer_success=bool(result.success), optimizer_message=str(result.message)))


def evaluate(folder, material, device, settings):
    check(folder)
    law = json.loads((folder / "inputs/models.json").read_text())[f"true_{material}"]
    target = pv.read(folder / "target.vtp")
    for setting in settings:
        grid, dt = SETTINGS[setting]
        for planned_for in "AB":
            selected = json.loads((folder / f"plans/{planned_for}/selected.json").read_text())
            path = folder / f"{setting}_{material}_plan_{planned_for}.npz"
            start = time.monotonic()
            print(json.dumps(dict(starting=path.name)), flush=True)
            data, phases = rollout(folder, path, law, selected["peaks_n"], grid, dt, device)
            record = dict(material=material, planned_for=planned_for, setting=setting,
                peaks_n=selected["peaks_n"], surface_mm=score(data, target),
                diagnostics=diagnostics(data, phases), file=path.name, data_sha256=digest(path))
            save_json(path.with_suffix(".json"), record)
            print(json.dumps(dict(**record, elapsed_s=time.monotonic() - start)), flush=True)


def verify(folder):
    check(folder)
    target = pv.read(folder / "target.vtp")
    laws = json.loads((folder / "inputs/models.json").read_text())
    rows = []
    for setting, (grid, dt) in SETTINGS.items():
        for material in "AB":
            for planned_for in "AB":
                path = folder / f"{setting}_{material}_plan_{planned_for}.npz"
                data = np.load(path)
                record = json.loads(path.with_suffix(".json").read_text())
                cfg = json.loads(path.with_suffix(".config.json").read_text())
                selected = json.loads((folder / f"plans/{planned_for}/selected.json").read_text())
                assert cfg["law"] == laws[f"true_{material}"]
                assert cfg["peaks_n"] == selected["peaks_n"]
                assert cfg["n_grid"] == grid and cfg["dt"] == dt
                assert record["data_sha256"] == digest(path)
                assert int(data["inverted_count"]) == 0
                assert all(np.all(np.isfinite(data[k])) for k in data.files)
                np.testing.assert_allclose(data["vol0"].astype(float).sum(), .00009, rtol=1e-6)
                np.testing.assert_allclose(np.diff(data["time"]), .004, atol=1e-10)
                np.testing.assert_allclose(score(data, target), record["surface_mm"], rtol=1e-10)
                if material == planned_for:
                    assert not any(p["travel_limited_fraction"] > 0 for p in record["diagnostics"]["pulses"])
                rows.append(record)
    save_json(folder / "summary.json", dict(results=rows, executions=len(rows),
        plan_evaluations={n: json.loads((folder / f"plans/{n}/selected.json").read_text())["evaluations"] for n in "AB"}))
    print(json.dumps([dict(material=r["material"], planned_for=r["planned_for"], setting=r["setting"], surface_mm=r["surface_mm"]) for r in rows], indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "plan", "evaluate", "verify"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--material", choices=["A", "B"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--settings", choices=list(SETTINGS), nargs="+", default=list(SETTINGS))
    a = p.parse_args()
    if a.stage == "prepare":
        prepare(a.out)
    elif a.stage == "plan":
        plan(a.out, a.material, a.device)
    elif a.stage == "evaluate":
        evaluate(a.out, a.material, a.device, a.settings)
    else:
        verify(a.out)
