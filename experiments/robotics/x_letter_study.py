"""Frozen rounded-X comparison, using previously identified A/B materials."""

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
import pyvista as pv
from scipy.optimize import minimize

from experiments.robotics.x_shaping import CONFIG, ROOT
from experiments.robotics.x_letter_control import execute
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.plastic_shaping_study import save_json, save_npz

OUT = ROOT / "out/x_letter_study_20260909"
OLD = ROOT / "out/x_shaping_study_20260909"
TARGET = ROOT / "out/x_letter_target_redesign_20260909"
MODELS = ["identified_B", "nominal", "identified_A"]
SETTINGS = dict(baseline=(64, 0.00005), half_dt=(64, 0.000025), grid80=(80, 0.00005))
PROTOCOL = dict(
    scene=dict(CONFIG, rotation_speed=np.pi / 2),
    angles_deg=[90.0, 0.0, 90.0, 0.0],
    initial_gaps_mm=[20.0, 31.0, 35.0, 20.0],
    simplex_offset_mm=-3.0,
    bounds_mm=[[16.0, 54.0]] * 4,
    max_evals=32,
    search="Bounded Nelder-Mead; fixed evaluations; best observed objective",
    models=MODELS,
    n_grid=64,
    dt=0.00005,
    seed=0,
    objective="Symmetric area-weighted mean reconstructed surface distance in mm",
    observation="1 s after final full opening and withdrawal",
    target="Frozen rounded_capital_x_v2, 64 x 80 mm, exactly 90 mL",
    design_provenance="Target and common rounded initialization informed by prior true-A feasibility trials; frozen before all three model searches",
    execution="Selected gaps replayed in true A/B without force or shape feedback or replanning",
    identification="Separate existing supplied-state press calibration; known Hencky/von-Mises family",
    robot="Only cylinder contacts simulated; 90/0 deg alternating orientation; proposed Panda adapters",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "protocol.json").exists(), "Use a fresh output directory"
    save_json(folder / "protocol.json", PROTOCOL)
    hashes = json.loads((OLD / "source_sha256.json").read_text())
    for name in ["x_letter_control.py", "x_letter_study.py", "x_letter_soft_target.py"]:
        path = Path(__file__).with_name(name)
        hashes[str(path.relative_to(ROOT))] = digest(path)
    for rel, sha in hashes.items():
        assert digest(ROOT / rel) == sha, rel
        dest = folder / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    save_json(folder / "source_sha256.json", hashes)
    shutil.copytree(OLD / "inputs", folder / "inputs")
    for name in ["target.json", "target.vtp", "target_outline_m.npy"]:
        shutil.copy2(TARGET / name, folder / name)
    save_json(
        folder / "input_sha256.json",
        {
            str(p.relative_to(folder)): digest(p)
            for p in [
                *sorted((folder / "inputs").rglob("*")),
                folder / "target.json",
                folder / "target.vtp",
                folder / "target_outline_m.npy",
            ]
            if p.is_file()
        },
    )
    assert (
        digest(folder / "target.vtp")
        == json.loads((folder / "target.json").read_text())["target_sha256"]
    )
    (folder / "packages.txt").write_text(
        "\n".join(
            sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
        )
    )
    for name, cmd in [
        ("gpu.txt", ["nvidia-smi"]),
        ("git.txt", ["git", "status", "--short"]),
        ("commit.txt", ["git", "rev-parse", "HEAD"]),
    ]:
        (folder / name).write_text(subprocess.check_output(cmd, text=True))
    for rel in [
        "paper.tex",
        "paper.pdf",
        "figs/identification_plastic_shaping.pdf",
        "figs/identification_plastic_shaping.png",
    ]:
        dest = folder / "manuscript_before" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "paper/icra2027" / rel, dest)


def check(folder):
    assert json.loads((folder / "protocol.json").read_text()) == PROTOCOL
    for rel, sha in json.loads((folder / "source_sha256.json").read_text()).items():
        assert digest(ROOT / rel) == digest(folder / "source_snapshot" / rel) == sha, rel
    for rel, sha in json.loads((folder / "input_sha256.json").read_text()).items():
        assert digest(folder / rel) == sha, rel


def score(data, target, h=0.00125):
    return mesh_distance_mm(surface(data["x_after_1s"], data["vol0"], h=h), target)


def rollout(path, law, gaps, grid, dt, device):
    config = dict(
        law=law,
        gaps_mm=np.round(gaps, 6).tolist(),
        n_grid=grid,
        dt=dt,
        device=device,
        seed=0,
        angles_deg=PROTOCOL["angles_deg"],
    )
    if path.exists():
        assert json.loads(path.with_suffix(".config.json").read_text()) == config
    else:
        save_json(path.with_suffix(".config.json"), config)
        data, phases = execute(law, config["gaps_mm"], grid, dt, device)
        save_npz(path, data)
        save_json(path.with_suffix(".phases.json"), phases)
    return np.load(path)


def plan(folder, model, device):
    check(folder)
    law = json.loads((folder / "inputs/models.json").read_text())[model]
    target = pv.read(folder / "target.vtp")
    dest = folder / "plans" / model
    dest.mkdir(parents=True, exist_ok=True)
    rows = []

    def objective(gaps):
        gaps = np.round(gaps, 6)
        path = dest / (hashlib.sha256(gaps.tobytes()).hexdigest()[:16] + ".npz")
        start = time.monotonic()
        data = rollout(path, law, gaps, PROTOCOL["n_grid"], PROTOCOL["dt"], device)
        row = dict(
            gaps_mm=gaps.tolist(),
            file=str(path.relative_to(folder)),
            surface_mm=score(data, target),
        )
        rows.append(row)
        save_json(dest / "evaluations.json", rows)
        print(
            json.dumps(
                dict(model=model, evaluation=len(rows), seconds=time.monotonic() - start, **row)
            ),
            flush=True,
        )
        return row["surface_mm"]

    x = np.array(PROTOCOL["initial_gaps_mm"])
    simplex = np.tile(x, (len(x) + 1, 1))
    simplex[1:] += PROTOCOL["simplex_offset_mm"] * np.eye(len(x))
    result = minimize(
        objective,
        x,
        method="Nelder-Mead",
        bounds=PROTOCOL["bounds_mm"],
        options=dict(maxfev=PROTOCOL["max_evals"], xatol=0.0, fatol=0.0, initial_simplex=simplex),
    )
    best = min(rows, key=lambda r: r["surface_mm"])
    save_json(
        dest / "selected.json",
        dict(
            **best,
            model=model,
            law=law,
            evaluations=len(rows),
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
        ),
    )


def evaluate(folder, model, device, numerics=False):
    check(folder)
    selected = json.loads((folder / "plans" / model / "selected.json").read_text())
    laws = json.loads((folder / "inputs/models.json").read_text())
    target = pv.read(folder / "target.vtp")
    for setting in ["half_dt", "grid80"] if numerics else ["baseline"]:
        grid, dt = SETTINGS[setting]
        for name in "AB" if model == "nominal" else model[-1]:
            method = "nominal" if model == "nominal" else "identified"
            path = folder / f"{setting}_{name}_{method}.npz"
            data = rollout(path, laws[f"true_{name}"], selected["gaps_mm"], grid, dt, device)
            row = dict(
                material=name,
                model=model,
                method=method,
                setting=setting,
                gaps_mm=selected["gaps_mm"],
                surface_mm=score(data, target),
                rms_speed_m_s=float(np.sqrt(np.mean(np.sum(data["v_after_1s"] ** 2, axis=1)))),
            )
            save_json(path.with_suffix(".json"), row)
            print(json.dumps(row), flush=True)


def verify(folder):
    check(folder)
    target = pv.read(folder / "target.vtp")
    paths = set()
    for model in MODELS:
        dest = folder / "plans" / model
        rows = json.loads((dest / "evaluations.json").read_text())
        assert len(rows) == PROTOCOL["max_evals"]
        selected = json.loads((dest / "selected.json").read_text())
        assert selected["file"] == min(rows, key=lambda r: r["surface_mm"])["file"]
        for row in rows:
            path = folder / row["file"]
            data = np.load(path)
            np.testing.assert_allclose(score(data, target), row["surface_mm"], rtol=1e-10)
            np.testing.assert_allclose(data["gaps_mm"], row["gaps_mm"], atol=1e-10)
            paths.add(path)
    results = []
    for setting in SETTINGS:
        for name in "AB":
            for method in ["nominal", "identified"]:
                path = folder / f"{setting}_{name}_{method}.npz"
                row = json.loads(path.with_suffix(".json").read_text())
                data = np.load(path)
                selected = json.loads(
                    (folder / "plans" / row["model"] / "selected.json").read_text()
                )
                np.testing.assert_allclose(data["gaps_mm"], selected["gaps_mm"], atol=1e-10)
                np.testing.assert_allclose(score(data, target), row["surface_mm"], rtol=1e-10)
                config = json.loads(path.with_suffix(".config.json").read_text())
                assert (
                    config["law"]
                    == json.loads((folder / "inputs/models.json").read_text())[f"true_{name}"]
                )
                results.append(row)
                paths.add(path)
    for path in paths:
        data = np.load(path)
        assert int(data["inverted_count"]) == 0
        assert all(np.all(np.isfinite(data[k])) for k in data.files)
        np.testing.assert_allclose(data["vol0"].astype(float).sum(), 0.00009, rtol=1e-6)
        np.testing.assert_allclose(data["angles_deg"], PROTOCOL["angles_deg"], atol=1e-10)
        np.testing.assert_allclose(np.diff(data["time"]), CONFIG["tick"], atol=1e-10)
        phases = json.loads(path.with_suffix(".phases.json").read_text())
        expected = [
            f"{j}:{name}"
            for j in range(4)
            for name in (["rotate"] if j else []) + ["approach", "close", "open", "withdraw"]
        ] + ["final:wait"]
        assert [p["name"] for p in phases] == expected
        for i, p in enumerate(phases):
            mask = data["phase_id"] == i
            np.testing.assert_allclose(data["tool_centers"][mask][-1], p["end_centers"], atol=1e-10)
            if p["name"].endswith("rotate"):
                assert (
                    p["duration_s"] == 1.0 and np.max(np.abs(data["reaction_force"][mask])) == 0.0
                )
            if p["name"].endswith("close"):
                gap = (
                    np.linalg.norm(
                        data["tool_centers"][mask, 1] - data["tool_centers"][mask, 0], axis=1
                    )
                    - 2 * CONFIG["radius"]
                )
                assert np.all(np.diff(gap) <= 1e-12)
                np.testing.assert_allclose(gap[-1] * 1000, data["gaps_mm"][int(p["name"][0])])
        assert phases[-1]["duration_s"] == 1.0
        for key in ["initial", "x_after_1s", *[f"stage_{i}_pressed" for i in range(4)]]:
            assert data[key].min() > 0 and data[key].max() < CONFIG["domain"]
    save_json(folder / "summary.json", dict(rollouts=len(paths), evaluations=96, results=results))
    save_json(
        folder / "checksums.json",
        {
            str(p.relative_to(folder)): digest(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != "checksums.json"
        },
    )
    print(json.dumps(dict(verified_rollouts=len(paths), results=results)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["prepare", "plan", "evaluate", "numerics", "verify", "complete"]
    )
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--model", choices=MODELS, default="identified_B")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    folder = a.out.resolve()
    if a.stage == "prepare":
        prepare(folder)
    elif a.stage == "plan":
        plan(folder, a.model, a.device)
    elif a.stage in ["evaluate", "numerics"]:
        evaluate(folder, a.model, a.device, a.stage == "numerics")
    elif a.stage == "complete":
        for model in MODELS:
            plan(folder, model, a.device)
            evaluate(folder, model, a.device)
        for model in MODELS:
            evaluate(folder, model, a.device, True)
        verify(folder)
    else:
        verify(folder)
