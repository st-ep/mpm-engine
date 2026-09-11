"""Separate forming/finishing gaps with original material B; exploratory study.

The active manuscript and previous study sources/data are not modified.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import minimize

from experiments.robotics.hex_shaping_control import snapshot_sources, target_prism
from experiments.robotics.hex_shaping_check import execute as repeated_execute
from experiments.robotics.hex_shaping_pilot import ANGLES, BAND, HALF, box_sdf, jaw_pose
from experiments.robotics.plastic_shaping_study import (
    ROOT, TRUTH, NOMINAL, press, identify, chamfer_mm, save_json, save_npz,
)
from warpmpm import GridConfig, Solver
from warpmpm.materials import vonmises
from warpmpm.scenes import block

OUT = ROOT/"out/hex_shaping_six_gaps_20260909"
MODELS = ("nominal", "identified_A", "identified_B", "true_A", "true_B")
PROTOCOL = dict(truth=TRUTH, nominal=NOMINAL,
    angles_deg=list(ANGLES)*2, initial_gaps_mm=[80., 85., 87.]*2,
    bounds_mm=[[40., 115.]]*6, simplex_coordinate_step_mm=-10., max_evals=64,
    optimizer="Bounded Nelder-Mead; equal initial simplex and budget for all models",
    objective="Symmetric mean unsquared full-cloud NN distance, mm, 0.30 s after release",
    selected_action="Best observed objective; no convergence/global optimality claim",
    target_diameter_m=.09, target_volume="Baseline initial particle volume",
    n_grid=48, domain_m=.30, dt=1e-4, control_dt=.0008, ppc=2, seed=0,
    initial_size_m=[.12,.08,.06], density=1000., opening_mm=160., jaw_speed_mm_s=100.,
    inter_squeeze_release_s=.30, reported_final_release_s=1.,
    contact="Sticky floor and SDF jaws, instantaneous pair activation/removal",
    fit_observations=["x","v","elastic F","mass","reference volume"],
    scope="Test of six independent gaps, original E=80 kPa for both materials",
    selection_policy="Do not change the nominal model or objective after seeing outcomes",
    adoption="Need recognizable hexagons for both and clearer nominal/identified separation, with numerical checks")


def execute(law, gaps_mm, *, device, n_grid=48, dt=1e-4, final_wait_s=.30):
    """Same jaw motion as the verified repeated-gap driver, with six explicit gaps."""
    gaps_mm = np.asarray(gaps_mm, dtype=float)
    if gaps_mm.shape != (6,) or not np.all(np.isfinite(gaps_mm)) or np.any((gaps_mm < 40) | (gaps_mm > 115)):
        raise ValueError("Expected six finite gaps in [40, 115] mm")
    if final_wait_s not in (.30, 1.):
        raise ValueError("Unsupported final release duration")
    grid = GridConfig(n_grid=n_grid, grid_lim=.30)
    pos, vol0, floor = block(grid, size=(.12, .08, .06), ppc=2, seed=0)
    tick, release, opening, speed = .0008, .30, .160, .10
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(grid, device=device, inversion_policy="raise").load_particles(pos, vol0)
        solver.set_material(vonmises(**law, density=1000.))
    solver.add_plane(point=(0, 0, floor), normal=(0, 0, 1), surface="sticky")
    sdf = box_sdf()
    data = dict(initial=pos, vol0=vol0, floor=np.array(floor), half=HALF,
                angles=np.tile(ANGLES, 2), gaps=gaps_mm/1000,
                dt=np.array(dt), n_grid=np.array(n_grid), final_wait_s=np.array(final_wait_s))
    elapsed = 0.
    for stage, (angle, gap) in enumerate(zip(data["angles"], data["gaps"], strict=True)):
        centers, normal, quat = jaw_pose(angle, opening, floor)
        count = int(np.ceil((opening-gap)/(2*speed*tick)))
        stop = elapsed+count*tick
        velocities = np.array([1, -1])[:, None]*normal*(opening-gap)/(2*count*tick)
        handles = [solver.add_sdf_collider(sdf, center=c, quat=quat, band=BAND,
                   surface="sticky", friction=0., start_time=elapsed-.5*dt,
                   end_time=stop-.5*dt) for c in centers]
        for step in range(count):
            for h, center, velocity in zip(handles, centers+step*tick*velocities, velocities, strict=True):
                solver.set_sdf_pose(h, center=center, velocity=velocity)
            solver.step(dt, substeps=round(tick/dt))
        data[f"stage_{stage}_pressed"] = solver.x().copy()
        data[f"stage_{stage}_centers"] = jaw_pose(angle, gap, floor)[0]
        data[f"stage_{stage}_quat"] = quat
        for _ in range(round(release/tick)):
            solver.step(dt, substeps=round(tick/dt))
        data[f"stage_{stage}_released"] = solver.x().copy()
        elapsed = stop+release
    data.update(x=solver.x().copy(), v=solver.v().copy(), elapsed=np.array(elapsed))
    if final_wait_s == 1.:
        for _ in range(round(.70/tick)):
            solver.step(dt, substeps=round(tick/dt))
        data.update(x_after_1s=solver.x().copy(), v_after_1s=solver.v().copy())
    data["inverted_count"] = np.array(solver.inverted_count())
    if data["inverted_count"] or not all(np.all(np.isfinite(x)) for x in data.values()):
        raise RuntimeError("Invalid six-gap execution")
    return data


def freeze(folder, device):
    snapshot_sources(folder, PROTOCOL)
    files = [Path(__file__), ROOT/"experiments/robotics/hex_shaping_check.py",
             ROOT/"experiments/nclaw/suite.py"]
    files += list((ROOT/"src/ident").rglob("*.py"))+list((ROOT/"src/common").rglob("*.py"))
    hashes = json.loads((folder/"source_sha256.json").read_text())
    for p in files:
        rel = str(p.relative_to(ROOT)); dest = folder/"source_snapshot"/rel
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed: {rel}; use a fresh output folder")
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest); hashes[rel] = digest
    save_json(folder/"source_sha256.json", hashes)
    save_json(folder/"device.json", dict(device=device))
    before = folder/"manuscript_before.json"
    if not before.exists():
        paths = ["paper/icra2027/paper.tex", "paper/icra2027/paper.pdf",
                 "paper/icra2027/figs/identification_plastic_shaping.pdf",
                 "paper/icra2027/figs/identification_plastic_shaping.png"]
        save_json(before, {rel: hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in paths})


def check_sources(folder):
    assert json.loads((folder/"protocol.json").read_text()) == PROTOCOL
    for rel, digest in json.loads((folder/"source_sha256.json").read_text()).items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
        assert hashlib.sha256((folder/"source_snapshot"/rel).read_bytes()).hexdigest() == digest, rel


def prepare(folder, device):
    freeze(folder, device); save_npz(folder/"target.npz", target_prism())
    fits = {}
    for name, law in TRUTH.items():
        path = folder/f"probe_{name}.npz"
        if not path.exists(): save_npz(path, press(law, device=device))
        fits[name] = identify(dict(np.load(path)))
        save_json(folder/f"identification_{name}.json", fits[name])
        print("Identified", name, fits[name]["law"], flush=True)
    laws = {"nominal": NOMINAL, **{f"true_{n}": v for n, v in TRUTH.items()},
            **{f"identified_{n}": v["law"] for n, v in fits.items()}}
    save_json(folder/"models.json", laws)
    # Independent compatibility check against the old command implementation.
    triplet = [67., 73., 75.]
    for tag in ["explicit", "repeated"]:
        path = folder/f"compatibility_{tag}.npz"
        if not path.exists():
            d = (execute(TRUTH["B"], triplet*2, device=device, final_wait_s=1.) if tag == "explicit"
                 else repeated_execute(TRUTH["B"], triplet, 2, device=device))
            save_npz(path, d)
    a = np.load(folder/"compatibility_explicit.npz"); b = np.load(folder/"compatibility_repeated.npz")
    diff = {key: chamfer_mm(a[key], b[key]) for key in
            [*[f"stage_{i}_released" for i in range(6)], "x_after_1s"]}
    assert max(diff.values()) < .05, diff
    save_json(folder/"compatibility.json", diff)
    print("Compatibility replay", diff, flush=True)


def plan(folder, model, device):
    check_sources(folder)
    assert (folder/"compatibility.json").exists()
    law = json.loads((folder/"models.json").read_text())[model]
    dest = folder/"plans"/model; dest.mkdir(parents=True, exist_ok=True)
    target = np.load(folder/"target.npz")["x"]; rows = []

    def objective(gaps):
        gaps = np.round(gaps, 6)
        path = dest/(hashlib.sha256(gaps.tobytes()).hexdigest()[:16]+".npz")
        start = time.monotonic()
        if not path.exists(): save_npz(path, execute(law, gaps, device=device))
        value = chamfer_mm(np.load(path)["x"], target)
        rows.append(dict(gaps_mm=gaps.tolist(), predicted_error_mm=value,
                         file=str(path.relative_to(folder)), seconds=time.monotonic()-start))
        save_json(dest/"evaluations.json", rows)
        best = min(rows, key=lambda r: r["predicted_error_mm"])
        save_json(dest/"best.json", best)
        print(json.dumps(dict(model=model, evaluation=len(rows), error_mm=value,
                              best_mm=best["predicted_error_mm"], gaps_mm=gaps.tolist())), flush=True)
        return value

    initial = np.array(PROTOCOL["initial_gaps_mm"])
    simplex = np.tile(initial, (7, 1)); simplex[1:] -= 10*np.eye(6)
    result = minimize(objective, initial, method="Nelder-Mead", bounds=PROTOCOL["bounds_mm"],
        options=dict(maxfev=PROTOCOL["max_evals"], xatol=.25, fatol=.005, initial_simplex=simplex))
    best = min(rows, key=lambda r: r["predicted_error_mm"])
    save_json(dest/"selected.json", dict(**best, model=model, law=law,
        optimizer_success=bool(result.success), optimizer_message=str(result.message), evaluations=len(rows)))


def run_selected(folder, model, device, numerics=False):
    check_sources(folder)
    selected = json.loads((folder/"plans"/model/"selected.json").read_text())
    names = "AB" if model == "nominal" else model[-1]
    method = "nominal" if model == "nominal" else ("oracle" if model.startswith("true_") else "identified")
    settings = [("baseline", 48, 1e-4)] if not numerics else [("half_dt", 48, 5e-5), ("grid64", 64, 1e-4)]
    for tag, n_grid, dt in settings:
        target = target_prism(n_grid)
        save_npz(folder/f"target_{tag}.npz", target)
        for name in names:
            path = folder/f"{tag}_{name}_{method}.npz"
            if not path.exists(): save_npz(path, execute(TRUTH[name], selected["gaps_mm"],
                device=device, n_grid=n_grid, dt=dt, final_wait_s=1.))
            data = np.load(path)
            record = dict(material=name, method=method, model=model, setting=tag,
                gaps_mm=selected["gaps_mm"], error_03s_mm=chamfer_mm(data["x"], target["x"]),
                executed_error_mm=chamfer_mm(data["x_after_1s"], target["x"]),
                shape_change_03_to_1s_mm=chamfer_mm(data["x"], data["x_after_1s"]))
            if method == "oracle" and tag == "baseline":
                record["independent_replay_mm"] = chamfer_mm(data["x"], np.load(folder/selected["file"])["x"])
                assert record["independent_replay_mm"] < .05
            save_json(path.with_suffix(".json"), record); print(json.dumps(record), flush=True)


def verify(folder):
    from experiments.robotics.hex_shaping_control_report import section_iou
    check_sources(folder)
    target = np.load(folder/"target.npz")["x"]; counts = {}
    for selected_path in sorted(folder.glob("plans/*/selected.json")):
        selected = json.loads(selected_path.read_text())
        rows = json.loads((selected_path.parent/"evaluations.json").read_text())
        counts[selected["model"]] = len(rows)
        for row in rows:
            data = np.load(folder/row["file"])
            assert int(data["inverted_count"]) == 0
            assert all(np.all(np.isfinite(data[k])) for k in data.files)
            np.testing.assert_allclose(chamfer_mm(data["x"], target), row["predicted_error_mm"], rtol=1e-12)
        assert selected["gaps_mm"] == min(rows, key=lambda r: r["predicted_error_mm"])["gaps_mm"]
    results = []
    for tag in ["baseline", "half_dt", "grid64"]:
        for path in sorted(folder.glob(f"{tag}_?_*.json")):
            row = json.loads(path.read_text()); d = np.load(path.with_suffix(".npz"))
            t = np.load(folder/f"target_{tag}.npz")
            np.testing.assert_allclose(chamfer_mm(d["x_after_1s"], t["x"]), row["executed_error_mm"], rtol=1e-12)
            assert int(d["inverted_count"]) == 0
            assert all(np.all(np.isfinite(d[k])) for k in d.files)
            # Audit the actual saved jaw centers against each independent gap.
            for i, (angle, gap) in enumerate(zip(d["angles"], d["gaps"], strict=True)):
                normal = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.])
                centers = d[f"stage_{i}_centers"]
                np.testing.assert_allclose((centers[1]-centers[0])@normal-2*(HALF[0]+BAND), gap, atol=1e-12)
            row["slice_hull_iou_at_25_50_75pct_height"] = section_iou(d, t, "x_after_1s")
            results.append(row)
    before = json.loads((folder/"manuscript_before.json").read_text())
    summary = dict(search_evaluations=counts, results=results,
        coverage="Complete five-model comparison" if len(counts) == 5 and sum(r["setting"] == "baseline" for r in results) == 6 else "Partial feasibility check",
        manuscript_unchanged=all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == h for rel, h in before.items()))
    save_json(folder/"summary.json", summary)
    hashes = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.rglob("*") if p.is_file() and p.name != "checksums.json" and p.suffix != ".log"}
    save_json(folder/"checksums.json", hashes); print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "plan", "execute", "numerics", "verify"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--model", choices=MODELS, default="true_B")
    args = parser.parse_args(); folder = args.out.resolve()
    if args.stage == "prepare": prepare(folder, args.device)
    elif args.stage == "plan": plan(folder, args.model, args.device)
    elif args.stage in ["execute", "numerics"]:
        run_selected(folder, args.model, args.device, numerics=args.stage == "numerics")
    else: verify(folder)


if __name__ == "__main__":
    main()
