"""Frozen equal-budget planning, true-material execution, and numerical checks."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.optimize import minimize

from experiments.robotics.x_shaping import CONFIG, ROOT, SOURCE, execute, score, target_mesh
from experiments.robotics.plastic_shaping_study import save_json, save_npz

OUT = ROOT/"out/x_shaping_study_20260909"
MODELS = ["nominal", "identified_A", "identified_B", "true_A", "true_B"]
PROTOCOL = dict(scene=CONFIG, gaps="Two physical cylinder-surface gaps, 0 then 90 degrees",
                bounds_mm=[[24., 52.]]*2, initial_gaps_mm=[36., 36.], simplex_offset_mm=-8.,
                max_evals=32, search="Bounded Nelder-Mead; fixed evaluation budget; best observed point",
                n_grid=64, dt=.00005, objective="Symmetric area-weighted mean reconstructed surface distance in mm",
                observation="1 s after final full withdrawal", geometry="Fixed 70 mm X, volume 90 mL",
                feedback="Position commands only, no work or force stopping and no shape replanning",
                identification="Separate supplied-state press calibration; known Hencky/von-Mises family",
                numerics="Repeat fixed selected gaps at dt/2 and grid80, with unchanged physical volume and floor")
SETTINGS = dict(baseline=(64, .00005), half_dt=(64, .000025), grid80=(80, .00005))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(folder, source):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/"protocol.json").exists():
        raise RuntimeError("Use a fresh output directory for preparation")
    save_json(folder/"protocol.json", PROTOCOL)
    sources = [Path(__file__), ROOT/"experiments/robotics/x_shaping.py",
               ROOT/"experiments/robotics/hex_shaping_surface.py",
               ROOT/"experiments/robotics/hex_shaping_figure.py",
               ROOT/"experiments/robotics/hex_shaping_study.py",
               ROOT/"experiments/robotics/hex_shaping_check.py",
               ROOT/"experiments/robotics/hex_shaping_control.py",
               ROOT/"experiments/robotics/hex_shaping_pilot.py",
               ROOT/"experiments/robotics/plastic_shaping_figure.py",
               ROOT/"experiments/robotics/plastic_shaping_study.py",
               *sorted((ROOT/"src/warpmpm").rglob("*.py"))]
    hashes = {}
    for p in sources:
        rel = str(p.relative_to(ROOT)); dest = folder/"source_snapshot"/rel
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest); hashes[rel] = digest(p)
    save_json(folder/"source_sha256.json", hashes)
    imports = {}
    names = ["models.json", *[f"probe_{n}.npz" for n in "AB"],
             *[f"identification_{n}.json" for n in "AB"],
             *[f"validation_{m}.npz" for m in MODELS]]
    for rel in names:
        p = source/rel; dest = folder/"inputs"/rel; dest.parent.mkdir(exist_ok=True)
        shutil.copy2(p, dest); imports[rel] = dict(source=str(p.resolve()), sha256=digest(p))
    save_json(folder/"imported_inputs.json", imports)
    target_mesh().save(folder/"target.vtp")
    save_json(folder/"target_sha256.json", {"target.vtp": digest(folder/"target.vtp")})
    (folder/"packages.txt").write_text("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
    for name, cmd in [("gpu.txt", ["nvidia-smi"]), ("git.txt", ["git", "status", "--short"]), ("commit.txt", ["git", "rev-parse", "HEAD"])]:
        (folder/name).write_text(subprocess.check_output(cmd, text=True))
    for rel in ["paper/icra2027/paper.tex", "paper/icra2027/paper.pdf",
                "paper/icra2027/figs/identification_plastic_shaping.pdf",
                "paper/icra2027/figs/identification_plastic_shaping.png"]:
        dest = folder/"manuscript_before"/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dest)


def check_sources(folder):
    assert json.loads((folder/"protocol.json").read_text()) == PROTOCOL
    for rel, h in json.loads((folder/"source_sha256.json").read_text()).items():
        assert digest(ROOT/rel) == h, rel
        assert digest(folder/"source_snapshot"/rel) == h, rel
    for rel, record in json.loads((folder/"imported_inputs.json").read_text()).items():
        assert digest(folder/"inputs"/rel) == record["sha256"], rel
    for rel, h in json.loads((folder/"target_sha256.json").read_text()).items():
        assert digest(folder/rel) == h


def rollout(folder, path, law, gaps, device, n_grid=64, dt=.00005):
    if not path.exists():
        data, phases = execute(law, gaps, device=device, n_grid=n_grid, dt=dt)
        save_npz(path, data); save_json(path.with_suffix(".phases.json"), phases)
    return np.load(path)


def plan(folder, model, device):
    check_sources(folder)
    law = json.loads((folder/"inputs/models.json").read_text())[model]
    dest = folder/"plans"/model; dest.mkdir(parents=True, exist_ok=True); rows = []

    def objective(gaps):
        gaps = np.round(gaps, 6)
        path = dest/(hashlib.sha256(gaps.tobytes()).hexdigest()[:16]+".npz")
        start = time.monotonic(); data = rollout(folder, path, law, gaps, device)
        row = dict(gaps_mm=gaps.tolist(), file=str(path.relative_to(folder)), surface_mm=score(data))
        rows.append(row); save_json(dest/"evaluations.json", rows)
        print(json.dumps(dict(model=model, evaluation=len(rows), seconds=time.monotonic()-start, **row)), flush=True)
        return row["surface_mm"]

    initial = np.array(PROTOCOL["initial_gaps_mm"])
    simplex = np.tile(initial, (3, 1)); simplex[1:] += PROTOCOL["simplex_offset_mm"]*np.eye(2)
    result = minimize(objective, initial, method="Nelder-Mead", bounds=PROTOCOL["bounds_mm"],
                      options=dict(maxfev=PROTOCOL["max_evals"], xatol=0., fatol=0., initial_simplex=simplex))
    best = min(rows, key=lambda r: r["surface_mm"])
    save_json(dest/"selected.json", dict(**best, model=model, law=law, evaluations=len(rows),
              optimizer_success=bool(result.success), optimizer_message=str(result.message)))


def evaluate(folder, model, device, numerics=False):
    check_sources(folder)
    selected = json.loads((folder/"plans"/model/"selected.json").read_text())
    laws = json.loads((folder/"inputs/models.json").read_text())
    names = "AB" if model == "nominal" else model[-1]
    method = "nominal" if model == "nominal" else ("oracle" if model.startswith("true_") else "identified")
    for setting in (["half_dt", "grid80"] if numerics else ["baseline"]):
        for name in names:
            path = folder/f"{setting}_{name}_{method}.npz"; grid, dt = SETTINGS[setting]
            data = rollout(folder, path, laws[f"true_{name}"], selected["gaps_mm"], device, grid, dt)
            result = dict(material=name, model=model, method=method, setting=setting,
                          gaps_mm=selected["gaps_mm"], surface_mm=score(data),
                          peak_force_per_finger_n=np.linalg.norm(data["reaction_force"], axis=-1).max(axis=0).tolist(),
                          initial_volume_m3=float(data["vol0"].astype(float).sum()),
                          rms_speed_m_s=float(np.sqrt(np.mean(np.sum(data["v_after_1s"]**2, axis=1)))))
            save_json(path.with_suffix(".json"), result); print(json.dumps(result), flush=True)


def verify(folder):
    check_sources(folder); paths = []; summary = []
    for model in MODELS:
        dest = folder/"plans"/model; rows = json.loads((dest/"evaluations.json").read_text())
        assert len(rows) == PROTOCOL["max_evals"]
        for row in rows:
            path = folder/row["file"]; data = np.load(path)
            np.testing.assert_allclose(score(data), row["surface_mm"], rtol=1e-10)
            np.testing.assert_array_equal(data["gaps_mm"], row["gaps_mm"])
            paths.append(path)
        selected = json.loads((dest/"selected.json").read_text())
        assert selected["file"] == min(rows, key=lambda r: r["surface_mm"])["file"]
    for path in folder.glob("*.npz"):
        if path.stem.startswith(tuple(SETTINGS)):
            record = json.loads(path.with_suffix(".json").read_text()); data = np.load(path)
            np.testing.assert_allclose(score(data), record["surface_mm"], rtol=1e-10)
            plan_record = json.loads((folder/"plans"/record["model"]/"selected.json").read_text())
            np.testing.assert_array_equal(data["gaps_mm"], plan_record["gaps_mm"])
            summary.append(record); paths.append(path)
    assert len(summary) == 18
    for path in set(paths):
        data = np.load(path); assert int(data["inverted_count"]) == 0
        assert all(np.all(np.isfinite(data[k])) for k in data.files)
        np.testing.assert_allclose(data["vol0"].astype(float).sum(), np.prod(CONFIG["size"]), rtol=1e-6)
        phases = json.loads(path.with_suffix(".phases.json").read_text())
        assert [p["name"] for p in phases] == ["0:approach", "0:close", "0:open", "0:withdraw", "1:rotate", "1:approach", "1:close", "1:open", "1:withdraw", "final:wait"]
        np.testing.assert_allclose(np.diff(data["time"]), CONFIG["tick"], atol=1e-10)
        for i, phase in enumerate(phases):
            pose = data["tool_centers"][data["phase_id"] == i]
            np.testing.assert_allclose(pose[-1], phase["end_centers"], atol=1e-10)
            if phase["name"].endswith("close"):
                gap = np.linalg.norm(pose[:, 1]-pose[:, 0], axis=1)-2*CONFIG["radius"]
                assert np.all(np.diff(gap) <= 1e-12)
                stage = int(phase["name"].split(":")[0])
                np.testing.assert_allclose(gap[-1]*1000, data["gaps_mm"][stage])
        for key in ["initial", "x_after_1s", "stage_0_pressed", "stage_1_pressed"]:
            assert data[key].min() > 0 and data[key].max() < CONFIG["domain"]
    save_json(folder/"summary.json", dict(rollouts=len(set(paths)), evaluations=PROTOCOL["max_evals"]*len(MODELS), results=summary))
    save_json(folder/"checksums.json", {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "checksums.json"})
    print(json.dumps(dict(rollouts=len(set(paths)), results=summary)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "plan", "evaluate", "numerics", "verify"])
    p.add_argument("--out", type=Path, default=OUT); p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--model", choices=MODELS, default="nominal"); p.add_argument("--device", default="cuda:0")
    args = p.parse_args(); folder = args.out.resolve()
    if args.stage == "prepare":
        prepare(folder, args.source.resolve())
    elif args.stage == "plan":
        plan(folder, args.model, args.device)
    elif args.stage in ("evaluate", "numerics"):
        evaluate(folder, args.model, args.device, args.stage == "numerics")
    else:
        verify(folder)


if __name__ == "__main__":
    main()
