"""Equal-budget boundary-objective planning and work-controlled execution.

Run prepare, plan for each of five models, calibrate, selfcheck, execute,
numerics, and verify. Imported identification data and historical studies stay
unchanged. Every stage checks frozen simulation sources and imported inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import minimize

from experiments.robotics.hex_shaping_check import execute as gap_execute
from experiments.robotics.hex_shaping_control import snapshot_sources, target_prism
from experiments.robotics.hex_shaping_study import MODELS, TRUTH
from experiments.robotics.hex_shaping_surface import METRIC, surface_error_mm, target_surface
from experiments.robotics.hex_shaping_work import PROTOCOL as WORK_PROTOCOL, execute as work_execute
from experiments.robotics.plastic_shaping_study import ROOT, NOMINAL, chamfer_mm, save_json, save_npz

OUT = ROOT / "out/hex_shaping_surface_20260909"
SOURCE = ROOT / "out/hex_shaping_study_20260908"
PROTOCOL = dict(WORK_PROTOCOL, planning="Three model-planned gaps repeated twice, then model-only work calibration",
    objective=METRIC, objective_release_s=1., initial_gaps_mm=[80., 85., 87.],
    simplex_offset_mm=10., bounds_mm=[[65., 115.]] * 3, max_evals=64,
    search="Bounded Nelder-Mead, xatol=fatol=0, select best observed of 64 evaluations",
    gap_rounding_decimals=6, particle_metric="Mean of two directed mean unsquared full-cloud NN distances, mm",
    identification="Reuse the current paper's supplied-state fits without modification",
    numerics="Execute the same work budgets at dt/2 and grid64 without retuning")
SETTINGS = dict(baseline=(48, 1e-4), half_dt=(48, 5e-5), grid64=(64, 1e-4))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(folder, source, device):
    snapshot_sources(folder, PROTOCOL)
    hashes = json.loads((folder / "source_sha256.json").read_text())
    files = [Path(__file__), *[ROOT / "experiments/robotics" / f for f in [
        "hex_shaping_surface.py", "hex_shaping_work.py", "hex_shaping_study.py",
        "hex_shaping_check.py", "hex_shaping_figure.py", "plastic_shaping_figure.py",
        "hex_shaping_six_report.py", "hex_shaping_six_gaps.py", "hex_shaping_control_report.py"]]]
    for p in files:
        rel = str(p.relative_to(ROOT)); dest = folder / "source_snapshot" / rel
        if dest.exists() and digest(dest) != digest(p):
            raise RuntimeError(f"Changed source {rel}; use a fresh output folder")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest); hashes[rel] = digest(p)
    save_json(folder / "source_sha256.json", hashes)
    # The laws and identification evidence are imported, not refitted for the target.
    models = json.loads((source / "models.json").read_text())
    assert models["nominal"] == NOMINAL
    assert all(models[f"true_{n}"] == law for n, law in TRUTH.items())
    imports = {}
    names = ["models.json", *[f"probe_{n}.npz" for n in TRUTH],
             *[f"identification_{n}.json" for n in TRUTH],
             *[f"validation_{m}.npz" for m in MODELS]]
    for rel in names:
        dest = folder / "inputs" / rel; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and digest(dest) != digest(source / rel):
            raise RuntimeError(f"Changed imported input {rel}")
        shutil.copy2(source / rel, dest)
        imports[rel] = dict(source=str(source / rel), sha256=digest(dest))
    save_json(folder / "imported_inputs.json", imports)
    save_json(folder / "device.json", dict(device=device))
    # Shared targets are written once here; parallel per-model jobs only read them.
    save_npz(folder / "target.npz", target_prism())
    for tag, (grid, _) in SETTINGS.items():
        save_npz(folder / f"target_{tag}.npz", target_prism(grid))
    save_json(folder / "target_sha256.json", {p.name: digest(p) for p in folder.glob("target*.npz")})
    paths = ["paper/icra2027/paper.tex", "paper/icra2027/paper.pdf",
             "paper/icra2027/figs/identification_plastic_shaping.pdf",
             "paper/icra2027/figs/identification_plastic_shaping.png"]
    if not (folder / "manuscript_before.json").exists():
        save_json(folder / "manuscript_before.json", {rel: digest(ROOT / rel) for rel in paths})
        for rel in paths:
            dest = folder / "manuscript_before" / rel
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, dest)


def check_sources(folder):
    assert json.loads((folder / "protocol.json").read_text()) == PROTOCOL
    for rel, h in json.loads((folder / "source_sha256.json").read_text()).items():
        assert digest(ROOT / rel) == h, rel
        assert digest(folder / "source_snapshot" / rel) == h, rel
    for rel, record in json.loads((folder / "imported_inputs.json").read_text()).items():
        assert digest(folder / "inputs" / rel) == record["sha256"], rel
    for rel, h in json.loads((folder / "target_sha256.json").read_text()).items():
        assert digest(folder / rel) == h, rel


def metrics(data, target, mesh):
    return dict(surface_error_1s_mm=surface_error_mm(data, mesh),
                particle_error_1s_mm=chamfer_mm(data["x_after_1s"], target["x"]))


def plan(folder, model, device):
    check_sources(folder)
    law = json.loads((folder / "inputs/models.json").read_text())[model]
    dest = folder / "plans" / model; dest.mkdir(parents=True, exist_ok=True)
    target = np.load(folder / "target.npz"); mesh = target_surface(target); rows = []

    def objective(gaps):
        gaps = np.round(gaps, PROTOCOL["gap_rounding_decimals"])
        path = dest / (hashlib.sha256(gaps.tobytes()).hexdigest()[:16] + ".npz")
        started = time.monotonic()
        if not path.exists():
            save_npz(path, gap_execute(law, gaps, 2, device=device))
        data = np.load(path)
        row = dict(gaps_mm=gaps.tolist(), file=str(path.relative_to(folder)),
                   **metrics(data, target, mesh))
        rows.append(row); save_json(dest / "evaluations.json", rows)
        print(json.dumps(dict(model=model, evaluation=len(rows), seconds=time.monotonic()-started, **row)), flush=True)
        return row["surface_error_1s_mm"]

    initial = np.array(PROTOCOL["initial_gaps_mm"])
    simplex = np.tile(initial, (4, 1)); simplex[1:] -= 10 * np.eye(3)
    result = minimize(objective, initial, method="Nelder-Mead", bounds=PROTOCOL["bounds_mm"],
        options=dict(maxfev=64, xatol=0., fatol=0., initial_simplex=simplex))
    best = min(rows, key=lambda r: r["surface_error_1s_mm"])
    save_json(dest / "selected.json", dict(**best, model=model, law=law, evaluations=len(rows),
        optimizer_success=bool(result.success), optimizer_message=str(result.message)))


def calibrate(folder, model, device):
    check_sources(folder)
    selected = json.loads((folder / "plans" / model / "selected.json").read_text())
    path = folder / f"calibration_{model}.npz"
    if not path.exists():
        data, _ = work_execute(selected["law"], gaps_mm=selected["gaps_mm"], device=device)
        save_npz(path, data)
    data = np.load(path); old = np.load(folder / selected["file"])
    difference = chamfer_mm(data["x_after_1s"], old["x_after_1s"])
    assert difference < .05, (model, difference)
    target = np.load(folder / "target.npz")
    record = dict(model=model, law=selected["law"], planned_gaps_mm=selected["gaps_mm"],
        work_limits_j=data["work_j"].tolist(), gap_replay_difference_mm=difference,
        **metrics(data, target, target_surface(target)))
    save_json(path.with_suffix(".json"), record); print(json.dumps(record), flush=True)


def run_selected(folder, model, device, stage):
    check_sources(folder)
    calibration = json.loads((folder / f"calibration_{model}.json").read_text())
    method = "nominal" if model == "nominal" else ("oracle" if model.startswith("true_") else "identified")
    names = list(TRUTH) if model == "nominal" else [model[-1]]
    if stage == "selfcheck":
        names = [model]
    settings = ["half_dt", "grid64"] if stage == "numerics" else ["baseline"]
    for tag in settings:
        grid, dt = SETTINGS[tag]
        target = np.load(folder / f"target_{tag}.npz"); mesh = target_surface(target)
        for name in names:
            stem = f"selfcheck_{model}" if stage == "selfcheck" else f"{tag}_{name}_{method}"
            path = folder / f"{stem}.npz"
            if not path.exists():
                law = calibration["law"] if stage == "selfcheck" else TRUTH[name]
                data, stops = work_execute(law, budgets_j=calibration["work_limits_j"], device=device, n_grid=grid, dt=dt)
                save_json(folder / f"{stem}_stops.json", stops); save_npz(path, data)
            data = np.load(path)
            record = dict(model=model, material=name, method=method, setting=tag,
                work_limits_j=calibration["work_limits_j"], actual_work_j=data["work_j"].tolist(),
                actual_gaps_mm=(data["gaps"]*1000).tolist(),
                stops=json.loads((folder / f"{stem}_stops.json").read_text()),
                **metrics(data, target, mesh))
            if stage == "selfcheck":
                ref = np.load(folder / f"calibration_{model}.npz")
                record["gap_difference_mm"] = (1000*(data["gaps"]-ref["gaps"])).tolist()
                record["cloud_difference_1s_mm"] = chamfer_mm(data["x_after_1s"], ref["x_after_1s"])
            save_json(folder / f"{stem}.json", record); print(json.dumps(record), flush=True)


def verify(folder):
    from experiments.robotics.hex_shaping_control_report import section_iou
    check_sources(folder)
    target = np.load(folder / "target.npz"); mesh = target_surface(target)
    counts = {}; records = []; raw_count = 0
    for model in MODELS:
        dest = folder / "plans" / model
        rows = json.loads((dest / "evaluations.json").read_text())
        counts[model] = len(rows); assert len(rows) == 64
        for row in rows:
            data = np.load(folder / row["file"])
            assert int(data["inverted_count"]) == 0
            assert all(np.all(np.isfinite(data[k])) for k in data.files)
            assert all(65 <= gap <= 115 for gap in row["gaps_mm"])
            for key, value in metrics(data, target, mesh).items():
                np.testing.assert_allclose(value, row[key], rtol=1e-10)
            raw_count += 1
        selected = json.loads((dest / "selected.json").read_text())
        assert selected["file"] == min(rows, key=lambda r: r["surface_error_1s_mm"])["file"]
        replay = json.loads((folder / f"calibration_{model}.json").read_text())
        assert replay["gap_replay_difference_mm"] < .05
        selfcheck = json.loads((folder / f"selfcheck_{model}.json").read_text())
        assert max(abs(x) for x in selfcheck["gap_difference_mm"]) <= .20
        assert selfcheck["cloud_difference_1s_mm"] < .15
    for path in sorted(folder.glob("*.npz")):
        if path.name.startswith("target"):
            continue
        data = np.load(path); raw_count += 1
        assert int(data["inverted_count"]) == 0
        assert all(np.all(np.isfinite(data[k])) for k in data.files)
        for i in range(6):
            steps = np.maximum(-np.sum(data[f"stage_{i}_force"]*data[f"stage_{i}_velocity"], axis=2), 0.).sum(axis=1)*.0008
            np.testing.assert_allclose(steps, data[f"stage_{i}_increment"], rtol=1e-12)
            np.testing.assert_allclose(steps.cumsum(), data[f"stage_{i}_work"], rtol=1e-12)
            np.testing.assert_allclose(steps.sum(), data["work_j"][i], rtol=1e-12)
            assert data["gaps"][i] >= .065-1e-12
            if "requested_work_j" in data:
                budget = data["requested_work_j"][i]
                if data["work_j"][i] >= budget:
                    assert data["work_j"][i]-steps[-1] < budget+1e-12
                else:
                    np.testing.assert_allclose(data["gaps"][i], .065, atol=1e-12)
        if path.stem.startswith(tuple(SETTINGS)):
            record = json.loads(path.with_suffix(".json").read_text())
            t = np.load(folder / f"target_{record['setting']}.npz")
            for key, value in metrics(data, t, target_surface(t)).items():
                np.testing.assert_allclose(value, record[key], rtol=1e-10)
            calibration = json.loads((folder / f"calibration_{record['model']}.json").read_text())
            np.testing.assert_array_equal(data["requested_work_j"], calibration["work_limits_j"])
            expected_stops = ["work" if w >= b else "stroke_limit" for w, b in zip(data["work_j"], data["requested_work_j"], strict=True)]
            assert record["stops"] == expected_stops
            record["slice_iou"] = section_iou(data, t, "x_after_1s")
            records.append(record)
    assert len(records) == 18
    means = {tag: {method: {metric: float(np.mean([r[metric] for r in records if r["setting"] == tag and r["method"] == method]))
              for metric in ["surface_error_1s_mm", "particle_error_1s_mm"]}
              for method in ["nominal", "identified", "oracle"]} for tag in SETTINGS}
    summary = dict(evaluations=counts, raw_rollout_checks=raw_count, results=records, means=means)
    save_json(folder / "summary.json", summary)
    save_json(folder / "checksums.json", {str(p.relative_to(folder)): digest(p)
        for p in folder.rglob("*") if p.is_file() and p.name != "checksums.json" and p.suffix != ".log"})
    print(json.dumps(dict(evaluations=counts, raw_rollout_checks=raw_count, means=means), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "plan", "calibrate", "selfcheck", "execute", "numerics", "verify"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--model", choices=MODELS, default="nominal")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args(); folder = args.out.resolve()
    if args.stage == "prepare":
        prepare(folder, args.source.resolve(), args.device)
    elif args.stage == "plan":
        plan(folder, args.model, args.device)
    elif args.stage == "calibrate":
        calibrate(folder, args.model, args.device)
    elif args.stage == "verify":
        verify(folder)
    else:
        run_selected(folder, args.model, args.device, args.stage)


if __name__ == "__main__":
    main()
