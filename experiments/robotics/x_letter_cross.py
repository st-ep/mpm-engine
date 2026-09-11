"""Execute the frozen identified A/B plans on both true materials.

Only the two off-diagonal executions are new at each numerical setting.
The matched executions are copied verbatim from the original rounded-X study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pyvista as pv

from experiments.robotics.x_letter_study import (
    OUT as BASE, ROOT, SETTINGS, check as check_base, digest, rollout, score,
)
from experiments.robotics.plastic_shaping_study import save_json

OUT = ROOT / "out/x_letter_cross_20260909"


def prepare(folder, base):
    check_base(base)
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "protocol.json").exists(), "Use a fresh directory"
    inputs = folder / "inputs"
    inputs.mkdir()
    source = folder / "source_snapshot"
    shutil.copytree(base / "source_snapshot", source)
    own = Path(__file__).resolve()
    dst = source / own.relative_to(ROOT)
    shutil.copy2(own, dst)
    save_json(folder / "source_sha256.json", {
        **json.loads((base / "source_sha256.json").read_text()),
        str(own.relative_to(ROOT)): digest(own),
    })
    for rel in ["target.vtp", "target.json", "target_outline_m.npy"]:
        shutil.copy2(base / rel, folder / rel)
    for rel in ["inputs/models.json", "protocol.json", "packages.txt", "franka/audit.json"]:
        shutil.copy2(base / rel, inputs / Path(rel).name)
    for name in "AB":
        shutil.copy2(base / f"plans/identified_{name}/selected.json", inputs / f"plan_{name}.json")
    reused = {}
    for setting in SETTINGS:
        for name in "AB":
            old = base / f"{setting}_{name}_identified.npz"
            new = folder / f"{setting}_{name}_plan_{name}.npz"
            for suffix in [".npz", ".json", ".config.json", ".phases.json"]:
                shutil.copy2(old.with_suffix(suffix), new.with_suffix(suffix))
                reused[str(new.with_suffix(suffix).relative_to(folder))] = digest(old.with_suffix(suffix))
    save_json(folder / "reused_sha256.json", reused)
    save_json(folder / "input_sha256.json", {
        str(p.relative_to(folder)): digest(p)
        for p in [*inputs.iterdir(), folder / "target.vtp", folder / "target.json", folder / "target_outline_m.npy"]
    })
    save_json(folder / "protocol.json", dict(
        base=str(base), settings=SETTINGS, seed=0,
        task="Frozen rounded X; existing identified A/B plans executed on both true materials",
        controls="Four unchanged y/x/y/x openings; no new optimization, feedback, or retuning",
        metric="Symmetric area-weighted mean closest-triangle surface distance, no alignment",
        observation="1 s after full opening and withdrawal",
        reused="Six matched executions copied verbatim; nominal results remain in the base archive",
    ))
    for rel in ["paper.tex", "paper.pdf", "figs/identification_plastic_shaping.pdf", "figs/identification_plastic_shaping.png"]:
        dst = folder / "manuscript_before" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "paper/icra2027" / rel, dst)
    (folder / "tracked_before.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
    (folder / "gpu_before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))


def check(folder):
    protocol = json.loads((folder / "protocol.json").read_text())
    check_base(Path(protocol["base"]))
    assert protocol["settings"] == {k: list(v) for k, v in SETTINGS.items()}
    for rel, sha in json.loads((folder / "source_sha256.json").read_text()).items():
        assert digest(ROOT / rel) == digest(folder / "source_snapshot" / rel) == sha, rel
    for manifest in ["input_sha256.json", "reused_sha256.json"]:
        for rel, sha in json.loads((folder / manifest).read_text()).items():
            assert digest(folder / rel) == sha, rel


def evaluate(folder, material, device):
    check(folder)
    planned_for = "B" if material == "A" else "A"
    laws = json.loads((folder / "inputs/models.json").read_text())
    selected = json.loads((folder / f"inputs/plan_{planned_for}.json").read_text())
    target = pv.read(folder / "target.vtp")
    for setting, (grid, dt) in SETTINGS.items():
        path = folder / f"{setting}_{material}_plan_{planned_for}.npz"
        started = time.monotonic()
        print(json.dumps(dict(starting=path.name, device=device)), flush=True)
        data = rollout(path, laws[f"true_{material}"], selected["gaps_mm"], grid, dt, device)
        record = dict(
            material=material, planned_for=planned_for, model=f"identified_{planned_for}",
            method="cross", setting=setting, gaps_mm=selected["gaps_mm"],
            surface_mm=score(data, target),
            rms_speed_m_s=float(np.sqrt(np.mean(np.sum(data["v_after_1s"] ** 2, axis=1)))),
        )
        save_json(path.with_suffix(".json"), record)
        print(json.dumps(dict(**record, elapsed_s=time.monotonic() - started)), flush=True)


def verify(folder):
    check(folder)
    target = pv.read(folder / "target.vtp")
    laws = json.loads((folder / "inputs/models.json").read_text())
    base = Path(json.loads((folder / "protocol.json").read_text())["base"])
    results = []
    for setting, (grid, dt) in SETTINGS.items():
        for material in "AB":
            reference = np.load(base / f"{setting}_{material}_identified.npz")
            for planned_for in "AB":
                path = folder / f"{setting}_{material}_plan_{planned_for}.npz"
                data = np.load(path)
                record = json.loads(path.with_suffix(".json").read_text())
                config = json.loads(path.with_suffix(".config.json").read_text())
                selected = json.loads((folder / f"inputs/plan_{planned_for}.json").read_text())
                command_reference = folder / f"{setting}_{planned_for}_plan_{planned_for}.npz"
                commands = np.load(command_reference)
                assert config["law"] == laws[f"true_{material}"]
                assert config["gaps_mm"] == selected["gaps_mm"]
                assert config["n_grid"] == grid and config["dt"] == dt and config["seed"] == 0
                assert record["model"] == f"identified_{planned_for}"
                assert record["material"] == material and record["setting"] == setting
                np.testing.assert_array_equal(data["initial"], reference["initial"])
                np.testing.assert_array_equal(data["vol0"], reference["vol0"])
                for key in ["time", "tool_centers", "phase_id", "angles_deg", "gaps_mm"]:
                    np.testing.assert_array_equal(data[key], commands[key])
                assert path.with_suffix(".phases.json").read_bytes() == command_reference.with_suffix(".phases.json").read_bytes()
                assert int(data["inverted_count"]) == 0
                assert all(np.all(np.isfinite(data[k])) for k in data.files)
                np.testing.assert_allclose(data["vol0"].astype(float).sum(), 0.00009, rtol=1e-6)
                np.testing.assert_allclose(score(data, target), record["surface_mm"], rtol=1e-10)
                results.append(dict(
                    material=material, planned_for=planned_for, setting=setting,
                    matched=material == planned_for, surface_mm=record["surface_mm"],
                    rms_speed_m_s=record["rms_speed_m_s"], file=path.name, sha256=digest(path),
                ))
    save_json(folder / "summary.json", dict(
        new_executions=6, reused_executions=6, results=results,
        commands="Each plan has byte-identical tool trajectories and phases in both materials",
        kinematics="The two frozen command trajectories reuse the original sampled Panda position/speed checks",
    ))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "evaluate", "verify"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--material", choices=["A", "B"])
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.stage == "prepare":
        prepare(a.out.resolve(), a.base.resolve())
    elif a.stage == "evaluate":
        assert a.material, "Select the true material with --material"
        evaluate(a.out.resolve(), a.material, a.device)
    else:
        verify(a.out.resolve())
