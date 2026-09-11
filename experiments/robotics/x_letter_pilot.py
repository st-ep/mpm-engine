"""Material-A-only exploratory letter shaping, with retained targets and rollouts.

Position commands; true A parameters are used for feasibility and optimization.
This is not an identification comparison and does not replace the paper assets.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.optimize import minimize

from experiments.robotics import x_shaping as core
from experiments.robotics.x_letter_target import TARGETS, mesh
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.plastic_shaping_study import ROOT, save_json, save_npz

OUT = ROOT/"out/x_letter_A_20260909"
LAW = dict(E=80000., nu=.30, yield_stress=1000.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder/"protocol.json").exists(), "Prepare a fresh directory"
    save_json(folder/"protocol.json", dict(material="true_A", law=LAW, targets=TARGETS,
              scene=dict(core.CONFIG, rotation_speed=np.pi/2),
              scope="Exploratory true-parameter A-only shaping; targets compared explicitly, no identification benefit claim",
              observation="1 s after full opening and withdrawal", objective="Symmetric area-weighted mean surface distance, mm"))
    sources = [Path(__file__), Path(__file__).with_name("x_letter_target.py"),
               *[ROOT/r for r in json.loads((ROOT/"out/x_shaping_study_20260909/source_sha256.json").read_text())]]
    hashes = {}
    for p in set(sources):
        rel = p.relative_to(ROOT); dest = folder/"source_snapshot"/rel
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
        hashes[str(rel)] = digest(p)
    save_json(folder/"source_sha256.json", hashes)
    target_hashes = {}
    for name in TARGETS:
        path = folder/f"target_{name}.vtp"; mesh(name).save(path); target_hashes[path.name] = digest(path)
    save_json(folder/"target_sha256.json", target_hashes)
    (folder/"packages.txt").write_text("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
    (folder/"gpu.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    (folder/"git.txt").write_text(subprocess.check_output(["git", "status", "--short"], text=True))


def check(folder):
    for rel, expected in json.loads((folder/"source_sha256.json").read_text()).items():
        assert digest(ROOT/rel) == digest(folder/"source_snapshot"/rel) == expected, rel
    for rel, expected in json.loads((folder/"target_sha256.json").read_text()).items():
        assert digest(folder/rel) == expected
    protocol = json.loads((folder/"protocol.json").read_text())
    assert protocol["targets"] == TARGETS and protocol["law"] == LAW


@contextmanager
def scene(radius_mm):
    before = core.CONFIG.copy()
    try:
        core.CONFIG.update(radius=radius_mm/1000, rotation_speed=np.pi/2)
        yield
    finally:
        core.CONFIG.clear(); core.CONFIG.update(before)


def scores(data, voxel=.00125):
    actual = surface(data["x_after_1s"], data["vol0"], h=voxel)
    return {name: mesh_distance_mm(actual, mesh(name)) for name in TARGETS}


def run_case(folder, gaps, radius_mm=14., grid=64, dt=.00005, device="cuda:0"):
    check(folder)
    gaps = np.round(gaps, 6).tolist()
    config = dict(law=LAW, gaps_mm=gaps, radius_mm=radius_mm, n_grid=grid, dt=dt,
                  rotation_speed_rad_s=np.pi/2, seed=0, device=device)
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    dest = folder/"rollouts"; dest.mkdir(exist_ok=True); path = dest/f"{key}.npz"
    start = time.monotonic()
    if not path.exists():
        save_json(path.with_suffix(".config.json"), config)
        with scene(radius_mm):
            data, phases = core.execute(LAW, gaps, device=device, n_grid=grid, dt=dt)
        save_npz(path, data); save_json(path.with_suffix(".phases.json"), phases)
    else:
        assert json.loads(path.with_suffix(".config.json").read_text()) == config
    data = np.load(path)
    result = dict(config=config, file=str(path.relative_to(folder)), errors_mm=scores(data),
                  rms_speed_m_s=float(np.sqrt(np.mean(np.sum(data["v_after_1s"]**2, axis=1)))))
    save_json(path.with_suffix(".json"), result)
    print(json.dumps(dict(seconds=time.monotonic()-start, **result)), flush=True)
    return result


def search(folder, tag, target, initial, radius_mm, budget, device):
    dest = folder/"searches"/tag; dest.mkdir(parents=True, exist_ok=True)
    config = dict(target=target, initial_mm=initial, radius_mm=radius_mm, budget=budget,
                  bounds_mm=[18., 54.], simplex_offset_mm=-3., objective="surface distance only",
                  model="true_A", source_sha256=digest(__file__))
    if (dest/"search.json").exists():
        assert json.loads((dest/"search.json").read_text()) == config
    else:
        save_json(dest/"search.json", config)
    rows = []

    def objective(gaps):
        record = run_case(folder, gaps, radius_mm=radius_mm, device=device)
        rows.append(record); save_json(dest/"evaluations.json", rows)
        print(json.dumps(dict(search=tag, evaluation=len(rows), error_mm=record["errors_mm"][target])), flush=True)
        return record["errors_mm"][target]

    x = np.array(initial, float); simplex = np.tile(x, (len(x)+1, 1))
    simplex[1:] += -3.*np.eye(len(x))
    minimize(objective, x, method="Nelder-Mead", bounds=[(18., 54.)]*len(x),
             options=dict(maxfev=budget, initial_simplex=simplex, xatol=0., fatol=0.))
    save_json(dest/"selected.json", min(rows, key=lambda r: r["errors_mm"][target]))


def verify(folder):
    check(folder); results = []
    for path in sorted((folder/"rollouts").glob("*.npz")):
        data = np.load(path); r = json.loads(path.with_suffix(".json").read_text())
        assert int(data["inverted_count"]) == 0 and all(np.all(np.isfinite(data[k])) for k in data.files)
        np.testing.assert_array_equal(data["gaps_mm"], r["config"]["gaps_mm"])
        np.testing.assert_allclose(data["vol0"].astype(float).sum(), .00009, rtol=1e-6)
        recalculated = scores(data)
        for target in TARGETS:
            np.testing.assert_allclose(recalculated[target], r["errors_mm"][target], rtol=1e-10)
        phases = json.loads(path.with_suffix(".phases.json").read_text())
        for i, phase in enumerate(phases):
            np.testing.assert_allclose(data["tool_centers"][data["phase_id"] == i][-1], phase["end_centers"], atol=1e-10)
            if phase["name"].endswith("rotate"):
                assert phase["duration_s"] == 1.
                assert np.max(np.abs(data["reaction_force"][data["phase_id"] == i])) == 0.
        assert phases[-1]["name"] == "final:wait" and phases[-1]["duration_s"] == 1.
        results.append(r)
    for search_dir in (folder/"searches").glob("*"):
        if not (search_dir/"selected.json").exists():
            continue
        protocol = json.loads((search_dir/"search.json").read_text())
        rows = json.loads((search_dir/"evaluations.json").read_text())
        assert len(rows) == protocol["budget"]
        assert json.loads((search_dir/"selected.json").read_text()) == min(rows, key=lambda r: r["errors_mm"][protocol["target"]])
    save_json(folder/"summary.json", dict(rollouts=len(results), results=results))
    print(json.dumps(dict(verified_rollouts=len(results))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "run", "search", "verify"])
    p.add_argument("--out", type=Path, default=OUT); p.add_argument("--gaps", nargs="+", type=float, default=[29., 47.])
    p.add_argument("--radius", type=float, default=14.); p.add_argument("--grid", type=int, default=64)
    p.add_argument("--dt", type=float, default=.00005); p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", default="two_gap"); p.add_argument("--target", choices=TARGETS, default="balanced")
    p.add_argument("--budget", type=int, default=24)
    a = p.parse_args(); folder = a.out.resolve()
    if a.stage == "prepare":
        prepare(folder)
    elif a.stage == "run":
        run_case(folder, a.gaps, a.radius, a.grid, a.dt, a.device)
    elif a.stage == "search":
        search(folder, a.tag, a.target, a.gaps, a.radius, a.budget, a.device)
    else:
        verify(folder)


if __name__ == "__main__":
    main()
