"""Model-predicted compression work and A/B cross-execution for the fixed X."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pyvista as pv

from experiments.robotics.x_work_control import execute, MIN_GAP
from experiments.robotics.x_force_study import OUT as FORCE, check as check_force
from experiments.robotics.x_letter_study import OUT as POSITION, ROOT, SETTINGS, digest, score
from experiments.robotics.x_shaping import CONFIG
from experiments.robotics.plastic_shaping_study import save_json, save_npz

OUT = ROOT / "out/x_work_study_20260909"
PROTOCOL = dict(scene=CONFIG, settings=SETTINGS, seed=0, minimum_gap_m=MIN_GAP,
    command="Four positive compression-work targets, summed over both fingers, in joules",
    feedback="Close at 50 mm/s per finger; stop at the first 4 ms tick reaching commanded work, or at the common 4 mm travel guard",
    work="Sum max(-reaction_force dot finger_velocity, 0) over fingers and closing ticks, multiplied by 4 ms",
    planning="Replay each existing identified-model four-gap plan at fixed closing speed to predict its work; no new shape optimization or tuning on true-material outcomes",
    inherited_search="32 gap objective evaluations per identified model, as archived in the position study",
    target="Unchanged rounded X from prior A feasibility work, 64 x 80 mm, 90 mL",
    motion="Actual opening, withdrawal, 1 s raised quarter turns, and 1 s observation after final withdrawal",
    scope="Measured-work stopping, no force regulation, no shape feedback, no online replanning, and no arm or gripper actuator dynamics")


def prepare(folder):
    check_force(FORCE)
    assert not folder.exists(), "Use a fresh archive"
    folder.mkdir(parents=True)
    save_json(folder / "protocol.json", PROTOCOL)
    hashes = json.loads((FORCE / "source_sha256.json").read_text())
    for name in ["x_work_control.py", "x_work_study.py"]:
        path = Path(__file__).with_name(name)
        hashes[str(path.relative_to(ROOT))] = digest(path)
    for rel, sha in hashes.items():
        assert digest(ROOT / rel) == sha
        dest = folder / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    save_json(folder / "source_sha256.json", hashes)
    inputs = folder / "inputs"
    inputs.mkdir()
    for path in (FORCE / "inputs").iterdir():
        if path.name.startswith(("models", "validation_", "same_force_", "press_extracts")):
            shutil.copy2(path, inputs / path.name)
    references = {}
    for name in "AB":
        old = POSITION / f"plans/identified_{name}/selected.json"
        shutil.copy2(old, inputs / f"position_{name}.json")
        selected = json.loads(old.read_text())
        references[name] = dict(selection_source=str(old), selection_sha256=digest(old),
            data_source=str(POSITION / selected["file"]), data_sha256=digest(POSITION / selected["file"]))
    save_json(inputs / "position_sources.json", references)
    for name in ["target.vtp", "target.json", "target_outline_m.npy"]:
        shutil.copy2(FORCE / name, folder / name)
    save_json(folder / "input_sha256.json", {str(p.relative_to(folder)): digest(p)
        for p in [*inputs.iterdir(), folder / "target.vtp", folder / "target.json", folder / "target_outline_m.npy"]})
    shutil.copytree(FORCE / "franka/panda_model_snapshot", folder / "franka/panda_model_snapshot")
    save_json(folder / "robot_input_sha256.json", {str(p.relative_to(folder)): digest(p)
        for p in (folder / "franka/panda_model_snapshot").rglob("*") if p.is_file()})
    shutil.copy2(FORCE / "packages.txt", folder / "packages.txt")
    for name in ["paper.tex", "paper.pdf", "figs/identification_plastic_shaping.pdf", "figs/identification_plastic_shaping.png"]:
        dest = folder / "manuscript_before" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "paper/icra2027" / name, dest)
    (folder / "tracked_before.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"]))
    (folder / "commit.txt").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"]))
    (folder / "gpu_before.txt").write_bytes(subprocess.check_output(["nvidia-smi"]))


def check(folder):
    assert json.loads((folder / "protocol.json").read_text()) == json.loads(json.dumps(PROTOCOL))
    for rel, sha in json.loads((folder / "source_sha256.json").read_text()).items():
        assert digest(ROOT / rel) == digest(folder / "source_snapshot" / rel) == sha, rel
    for manifest in ["input_sha256.json", "robot_input_sha256.json"]:
        for rel, sha in json.loads((folder / manifest).read_text()).items():
            assert digest(folder / rel) == sha, rel


def rollout(folder, path, law, *, budgets=None, gaps=None, grid=64, dt=.00005, device):
    config = dict(law=law, budgets_j=budgets, gaps_mm=gaps, n_grid=grid, dt=dt,
                  device=device, protocol_sha256=digest(folder / "protocol.json"))
    if path.exists():
        assert json.loads(path.with_suffix(".config.json").read_text()) == config
    else:
        save_json(path.with_suffix(".config.json"), config)
        data, phases, reasons = execute(law, budgets=budgets, gaps=gaps, grid=grid, dt=dt, device=device)
        save_npz(path, data)
        save_json(path.with_suffix(".phases.json"), phases)
        save_json(path.with_suffix(".stops.json"), reasons)
    return np.load(path), json.loads(path.with_suffix(".stops.json").read_text())


def predict(folder, name, device):
    check(folder)
    selected = json.loads((folder / f"inputs/position_{name}.json").read_text())
    path = folder / f"prediction_{name}.npz"
    data, _ = rollout(folder, path, selected["law"], gaps=selected["gaps_mm"], device=device)
    reference = json.loads((folder / "inputs/position_sources.json").read_text())[name]
    old = np.load(reference["data_source"])
    assert digest(reference["data_source"]) == reference["data_sha256"]
    rms = float(np.sqrt(np.mean(np.sum((data["x_after_1s"] - old["x_after_1s"]) ** 2, axis=-1))) * 1000)
    assert rms < .5, (name, "Fixed-speed replay differs materially from the inherited plan", rms)
    record = dict(model=f"identified_{name}", law=selected["law"], budgets_j=data["actual_work_j"].tolist(),
        inherited_gaps_mm=selected["gaps_mm"], inherited_search_evaluations=32,
        fixed_speed_replay_rms_difference_mm=rms, surface_mm=score(data, pv.read(folder / "target.vtp")),
        file=path.name, data_sha256=digest(path))
    save_json(folder / f"plans/{name}/selected.json", record)
    print(json.dumps(record), flush=True)


def selfcheck(folder, name, device):
    check(folder)
    selected = json.loads((folder / f"plans/{name}/selected.json").read_text())
    path = folder / f"selfcheck_{name}.npz"
    data, stops = rollout(folder, path, selected["law"], budgets=selected["budgets_j"], device=device)
    predicted = np.load(folder / f"prediction_{name}.npz")
    rms = float(np.sqrt(np.mean(np.sum((data["x_after_1s"] - predicted["x_after_1s"]) ** 2, axis=-1))) * 1000)
    assert all(s == "work" for s in stops) and rms < .5, (name, stops, rms)
    record = dict(model=name, surface_mm=score(data, pv.read(folder / "target.vtp")),
        work_replay_rms_difference_mm=rms, gaps_mm=data["gaps_mm"].tolist(),
        gap_difference_mm=(data["gaps_mm"] - predicted["gaps_mm"]).tolist(), stops=stops,
        data_sha256=digest(path))
    save_json(path.with_suffix(".json"), record)
    print(json.dumps(record), flush=True)


def evaluate(folder, material, device, settings):
    check(folder)
    law = json.loads((folder / "inputs/models.json").read_text())[f"true_{material}"]
    target = pv.read(folder / "target.vtp")
    for setting in settings:
        grid, dt = SETTINGS[setting]
        for planned_for in "AB":
            selected = json.loads((folder / f"plans/{planned_for}/selected.json").read_text())
            path = folder / f"{setting}_{material}_plan_{planned_for}.npz"
            print(json.dumps(dict(starting=path.name)), flush=True)
            start = time.monotonic()
            data, stops = rollout(folder, path, law, budgets=selected["budgets_j"], grid=grid, dt=dt, device=device)
            record = dict(material=material, planned_for=planned_for, setting=setting,
                budgets_j=selected["budgets_j"], actual_work_j=data["actual_work_j"].tolist(),
                gaps_mm=data["gaps_mm"].tolist(), stops=stops, surface_mm=score(data, target),
                max_sampled_height_mm=float((data["extent_samples"][:, 6].max() - data["floor"]) * 1000),
                max_sampled_floor_penetration_mm=float(max(0, data["floor"] - data["extent_samples"][:, 3].min()) * 1000),
                rms_speed_mm_s=float(np.sqrt(np.mean(np.sum(data["v_after_1s"] ** 2, axis=-1))) * 1000),
                file=path.name, data_sha256=digest(path))
            save_json(path.with_suffix(".json"), record)
            print(json.dumps(dict(**record, elapsed_s=time.monotonic() - start)), flush=True)


def verify(folder):
    check(folder)
    target = pv.read(folder / "target.vtp")
    laws = json.loads((folder / "inputs/models.json").read_text())
    rows = []
    for path in sorted(folder.glob("*.npz")):
        data = np.load(path)
        cfg = json.loads(path.with_suffix(".config.json").read_text())
        assert cfg["protocol_sha256"] == digest(folder / "protocol.json")
        assert int(data["inverted_count"]) == 0 and all(np.all(np.isfinite(data[k])) for k in data.files)
        np.testing.assert_allclose(data["vol0"].astype(float).sum(), .00009, rtol=1e-6)
        np.testing.assert_allclose(np.diff(data["time"]), .004, atol=1e-10)
        increments = np.maximum(-(data["reaction_force"] * data["tool_velocity"]).sum(axis=-1), 0).sum(axis=-1) * .004
        np.testing.assert_allclose(increments, data["work_increment_j"], rtol=1e-12, atol=1e-14)
        stops = json.loads(path.with_suffix(".stops.json").read_text())
        for i in range(4):
            active = data["pinch_id"] == i
            work = increments[active]
            np.testing.assert_allclose(work.sum(), data["actual_work_j"][i], rtol=1e-12)
            if "requested_work_j" in data:
                budget = data["requested_work_j"][i]
                if stops[i] == "work":
                    assert work.sum() >= budget - 1e-12 and work[:-1].sum() < budget + 1e-12
                else:
                    assert stops[i] == "travel_guard" and work.sum() < budget
                    np.testing.assert_allclose(data["gaps_mm"][i], MIN_GAP * 1000, atol=1e-8)
        if path.stem.startswith(tuple(SETTINGS)):
            row = json.loads(path.with_suffix(".json").read_text())
            selected = json.loads((folder / f"plans/{row['planned_for']}/selected.json").read_text())
            assert cfg["law"] == laws[f"true_{row['material']}"]
            assert cfg["budgets_j"] == selected["budgets_j"] == row["budgets_j"]
            assert [cfg["n_grid"], cfg["dt"]] == list(SETTINGS[row["setting"]])
            assert digest(path) == row["data_sha256"]
            np.testing.assert_allclose(score(data, target), row["surface_mm"], rtol=1e-10)
            if row["material"] == row["planned_for"]:
                assert all(s == "work" for s in stops)
            rows.append(row)
    assert len(rows) == 12
    save_json(folder / "summary.json", dict(results=rows, executions=len(rows), model_predictions=2, model_replays=2))
    print(json.dumps([dict(material=r["material"], planned_for=r["planned_for"], setting=r["setting"], surface_mm=r["surface_mm"], stops=r["stops"]) for r in rows], indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "predict", "selfcheck", "evaluate", "verify"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--material", choices=["A", "B"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--settings", choices=list(SETTINGS), nargs="+", default=list(SETTINGS))
    a = p.parse_args()
    if a.stage == "prepare":
        prepare(a.out)
    elif a.stage == "predict":
        predict(a.out, a.material, a.device)
    elif a.stage == "selfcheck":
        selfcheck(a.out, a.material, a.device)
    elif a.stage == "evaluate":
        evaluate(a.out, a.material, a.device, a.settings)
    else:
        verify(a.out)
