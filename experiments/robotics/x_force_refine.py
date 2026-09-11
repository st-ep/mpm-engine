"""Broaden a stalled force search without using true-material outcomes.

Nine coarse trials vary the first two peak commands; sixteen Nelder-Mead
evaluations then refine the best feasible candidate. Physics and target are
unchanged. The original 32-evaluation selection remains archived.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pyvista as pv
from scipy.optimize import minimize

from experiments.robotics.x_force_study import OUT, check, digest, rollout, score
from experiments.robotics.x_force_control import diagnostics
from experiments.robotics.plastic_shaping_study import save_json


def prepare(folder, name):
    check(folder)
    dest = folder / "refinement" / name
    assert not dest.exists(), "Use a new refinement archive"
    dest.mkdir(parents=True)
    original = folder / f"plans/{name}/selected.json"
    selected = json.loads(original.read_text())
    assert selected["evaluations"] == 32
    shutil.copy2(original, dest / "selected_initial32.json")
    shutil.copy2(__file__, dest / "source.py")
    seed = np.array(json.loads((folder / f"inputs/initialization_{name}.json").read_text())["peaks_n"]) / .8
    grid = []
    for first, second in itertools.product([.95, 1., 1.03], [.95, 1.05, 1.15]):
        peaks = np.array(selected["peaks_n"])
        peaks[:2] = seed[:2] * [first, second]
        grid.append(np.round(peaks, 6).tolist())
    save_json(dest / "protocol.json", dict(material=name, coarse_peaks_n=grid,
        source_sha256=digest(dest / "source.py"), selected_initial_sha256=digest(original),
        refinement_evaluations=16, simplex_relative_steps=[.02, .04, .08, .08],
        scope="Identified-model trials only; fixed physics, controller, target, metric, bounds, and feasibility rules"))


def state(folder, name):
    check(folder)
    dest = folder / "refinement" / name
    protocol = json.loads((dest / "protocol.json").read_text())
    assert digest(dest / "source.py") == digest(Path(__file__)) == protocol["source_sha256"]
    assert digest(dest / "selected_initial32.json") == protocol["selected_initial_sha256"]
    return dest, protocol


def trial(folder, name, peaks, device):
    dest, _ = state(folder, name)
    peaks = np.round(peaks, 6)
    path = dest / (hashlib.sha256(peaks.tobytes()).hexdigest()[:16] + ".npz")
    if path.with_suffix(".json").exists():
        return json.loads(path.with_suffix(".json").read_text())
    law = json.loads((folder / "inputs/models.json").read_text())[f"identified_{name}"]
    target = pv.read(folder / "target.vtp")
    print(json.dumps(dict(starting=path.name, peaks_n=peaks.tolist())), flush=True)
    start = time.monotonic()
    try:
        data, phases = rollout(folder, path, law, peaks, 64, .00005, device)
        diagnostic = diagnostics(data, phases)
        error = score(data, target)
        travel = sum(p["travel_limited_fraction"] for p in diagnostic["pulses"])
        excess = max(0, diagnostic["max_recorded_height_mm"] / 45.5 - 1)
        row = dict(peaks_n=peaks.tolist(), file=str(path.relative_to(folder)),
            surface_mm=error, feasible=travel == 0 and excess == 0,
            objective_mm=error + 100 * (travel + excess), diagnostics=diagnostic)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        row = dict(peaks_n=peaks.tolist(), file=str(path.relative_to(folder)),
            feasible=False, objective_mm=1000., failure=str(exc))
    save_json(path.with_suffix(".json"), row)
    print(json.dumps(dict(**row, elapsed_s=time.monotonic() - start)), flush=True)
    return row


def finish(folder, name, device):
    dest, protocol = state(folder, name)
    assert not (dest / "complete.json").exists()
    original = json.loads((dest / "selected_initial32.json").read_text())
    coarse = []
    for peaks in protocol["coarse_peaks_n"]:
        key = hashlib.sha256(np.array(peaks).tobytes()).hexdigest()[:16]
        coarse.append(json.loads((dest / f"{key}.json").read_text()))
    initial = np.array(min((r for r in [original, *coarse] if r["feasible"]),
                           key=lambda r: r["surface_mm"])["peaks_n"])
    rows = []

    def objective(relative):
        row = trial(folder, name, relative * initial, device)
        rows.append(row)
        save_json(dest / "evaluations.json", rows)
        return row["objective_mm"]

    simplex = np.ones((5, 4))
    simplex[1:] += np.diag(protocol["simplex_relative_steps"])
    result = minimize(objective, np.ones(4), method="Nelder-Mead",
        bounds=np.array([.05, 14.])[None, :] / initial[:, None],
        options=dict(maxfev=protocol["refinement_evaluations"], xatol=0., fatol=0., initial_simplex=simplex))
    best = min((r for r in [original, *coarse, *rows] if r["feasible"]), key=lambda r: r["surface_mm"])
    best = {key: best[key] for key in ["peaks_n", "file", "surface_mm", "feasible", "objective_mm", "diagnostics"]}
    selected = dict(**best, model=original["model"], law=original["law"],
        evaluations=32 + len(coarse) + len(rows),
        optimizer_success=bool(result.success), optimizer_message=str(result.message),
        refinement_protocol_sha256=digest(dest / "protocol.json"))
    save_json(dest / "complete.json", dict(selected=selected,
        original_error_mm=original["surface_mm"], coarse_trials=len(coarse), refinement_calls=len(rows)))
    save_json(folder / f"plans/{name}/selected.json", selected)
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "probe", "finish"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--material", choices=["A", "B"], required=True)
    p.add_argument("--index", type=int, choices=range(9))
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.stage == "prepare":
        prepare(a.out, a.material)
    elif a.stage == "probe":
        _, config = state(a.out, a.material)
        trial(a.out, a.material, config["coarse_peaks_n"][a.index], a.device)
    else:
        finish(a.out, a.material, a.device)
