"""Bounded material-design pilot; never changes the adopted paper experiment.

Two predeclared, one-at-a-time yield perturbations of at most 20 percent.
Screening reuses the existing gap sequence but recomputes work in the newly
identified material. These are feasibility checks, not fresh optimized results.
Only a promising screen justifies a fresh search with the original 32-evaluation
budget. Every tested candidate and failed screen is retained.
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

from experiments.robotics import x_common_study as study
from experiments.robotics import x_common_probe as probe
from experiments.robotics import x_letter_study as position
from experiments.robotics import x_work_study as work
from experiments.robotics.plastic_shaping_study import save_json, save_npz

BASE = study.OUT
OUT = position.ROOT / "out/x_material_sensitivity_20260910"
CANDIDATES = {
    "B_y12000": dict(material="B", law=dict(E=80000., nu=.30, yield_stress=12000.)),
    "A_y0800": dict(material="A", law=dict(E=80000., nu=.30, yield_stress=800.)),
}
PROTOCOL = dict(
    candidates=CANDIDATES,
    baseline=str(BASE),
    maximum_change_percent=20,
    fixed=["elastic modulus", "Poisson ratio", "constitutive family", "density",
           "initial particles", "contact", "identification trajectory", "target",
           "four pinches", "12 mm opening guard", "shape metric", "render scale"],
    screening="Inherited matching gap plan replayed under the new identified law; recompute work; execute on true material",
    continuation="Only if the release contrast visibly improves and matching shape remains promising; fresh original 32-evaluation search before adoption",
    adoption="No increase in matched shape error for either material, increased response contrast, shared probe yields both; repeat numerical checks before adoption",
    scope="Synthetic material design sensitivity, not new hardware calibration or proof that a particular physical formulation has these parameters",
)


def prepare(root):
    assert not root.exists(), "Use a fresh output directory"
    root.mkdir(parents=True)
    save_json(root / "protocol.json", PROTOCOL)
    sources = [*position.ROOT.joinpath("src").rglob("*.py"),
               *position.ROOT.joinpath("experiments/robotics").glob("*.py"),
               position.ROOT / "experiments/nclaw/suite.py"]
    manifest = {}
    for src in sources:
        rel = src.relative_to(position.ROOT)
        dest = root / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        manifest[str(rel)] = position.digest(src)
    save_json(root / "source_sha256.json", manifest)
    for rel in ["target.vtp", "target.json", "target_outline_m.npy", "inputs/models.json"]:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / rel, dest)
    for rel in ["packages.txt", "commit.txt"]:
        shutil.copy2(BASE / rel, root / ("baseline_" + rel))
    save_json(root / "baseline_sha256.json", {
        str(p.relative_to(BASE)): position.digest(p)
        for p in BASE.rglob("*") if p.is_file() and
        (p.name in ["checksums.json", "selected.json"] or
         p.name.startswith(("probe_", "identification_", "baseline_")))})
    (root / "tracked_before.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"]))
    (root / "gpu_before.txt").write_bytes(subprocess.check_output(["nvidia-smi"]))
    paper = position.ROOT / "paper/icra2027"
    save_json(root / "paper_before_sha256.json", {
        rel: position.digest(paper / rel) for rel in ["paper.tex", "paper.pdf",
            "figs/identification_plastic_shaping.pdf", "figs/identification_plastic_shaping.png"]})
    for name, definition in CANDIDATES.items():
        folder = root / name
        save_json(folder / "protocol.json", dict(PROTOCOL, candidate=definition,
                                                parent_source_sha256=position.digest(root / "source_sha256.json")))
        (folder / "inputs").mkdir()
        models = json.loads((BASE / "inputs/models.json").read_text())
        models["true_" + definition["material"]] = definition["law"]
        save_json(folder / "inputs/models.json", models)
        for rel in ["target.vtp", "target.json", "target_outline_m.npy"]:
            shutil.copy2(root / rel, folder / rel)


def check(folder):
    root = folder.parent
    assert json.loads((root / "protocol.json").read_text()) == PROTOCOL
    for rel, sha in json.loads((root / "source_sha256.json").read_text()).items():
        assert position.digest(position.ROOT / rel) == position.digest(root / "source_snapshot" / rel) == sha, rel
    for rel, sha in json.loads((root / "baseline_sha256.json").read_text()).items():
        assert position.digest(BASE / rel) == sha, rel
    assert position.digest(folder / "target.vtp") == position.digest(BASE / "target.vtp")


def identify(folder, name, device):
    check(folder)
    definition = CANDIDATES[name]
    material, law = definition["material"], definition["law"]
    path = folder / "inputs" / f"probe_{material}.npz"
    assert not path.exists(), "Use a new directory"
    start = time.monotonic()
    data, phases = probe.execute(law, device=device)
    save_npz(path, data)
    save_json(path.with_suffix(".phases.json"), phases)
    save_json(path.with_suffix(".config.json"), dict(law=law, protocol=probe.PROBE,
              grid=64, dt=.00005, device=device, seed=0))
    audit = probe.excitation_audit(data, law)
    save_json(folder / f"excitation_{material}.json", audit)
    # Pass only declared observation channels, never the audit/force/truth arrays.
    keys = ["x", "v", "F", "vol0", "mass", "floor", "frame_dt", "n_grid", "grid_lim", "gap_m"]
    fit = probe.identify({k: data[k] for k in keys})
    save_json(folder / f"identification_{material}.json", fit)
    assert fit["accepted"] and audit["outgoing_trial_over_true_cap_fraction"] > .01
    baseline = np.load(BASE / f"inputs/probe_{material}.npz")
    for key in ["initial", "vol0", "time", "tool_centers", "tool_velocity", "phase_id", "state_time"]:
        np.testing.assert_array_equal(data[key], baseline[key])
    models = json.loads((folder / "inputs/models.json").read_text())
    models[f"identified_{material}"] = fit["law"]
    save_json(folder / "inputs/models.json", models)
    print(json.dumps(dict(candidate=name, elapsed_s=time.monotonic() - start,
                          law=fit["law"], excitation=audit)), flush=True)


def screen(folder, name, device):
    check(folder)
    material = CANDIDATES[name]["material"]
    other = "A" if material == "B" else "B"
    models = json.loads((folder / "inputs/models.json").read_text())
    selection = json.loads((BASE / f"plans/identified_{material}/selected.json").read_text())
    target = pv.read(folder / "target.vtp")
    prediction, _ = work.rollout(folder, folder / "screen_prediction.npz",
        models[f"identified_{material}"], gaps=selection["gaps_mm"], device=device)
    budgets = prediction["actual_work_j"].tolist()
    old_budgets = json.loads((BASE / f"plans/{other}/selected.json").read_text())["budgets_j"]
    rows = []
    for true_name, plan_name, commands in [(material, material, budgets),
            (material, other, old_budgets), (other, material, budgets)]:
        data, stops = work.rollout(folder, folder / f"screen_{true_name}_plan_{plan_name}.npz",
            models[f"true_{true_name}"], budgets=commands, device=device)
        row = dict(material=true_name, planned_for=plan_name, budgets_j=commands,
                   surface_mm=position.score(data, target), stops=stops,
                   gaps_mm=data["gaps_mm"].tolist())
        rows.append(row)
        save_json(folder / "screen_results.json", dict(
            inherited_gaps_mm=selection["gaps_mm"], matching_work_j=budgets,
            optimized_for_candidate=False, results=rows))
        print(json.dumps(dict(candidate=name, **row)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "identify", "screen"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--candidate", choices=list(CANDIDATES))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.out)
    else:
        assert args.candidate
        folder = args.out / args.candidate
        {"identify": identify, "screen": screen}[args.stage](folder, args.candidate, args.device)


if __name__ == "__main__":
    main()
